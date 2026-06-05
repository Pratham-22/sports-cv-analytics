"""Dynamic pitch heatmap: team accumulation + per-player moving footprints."""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv

from sports.configs.soccer import SoccerPitchConfiguration


def _hex_bgr(hex_color: str) -> Tuple[int, int, int]:
    return tuple(sv.Color.from_hex(hex_color).as_bgr())


class DynamicPitchHeatmap:
    """
    Two layers on a discretized pitch grid (coordinates in cm, same as Roboflow config):

    1. Team heatmap — visit counts accumulate over the match (global grid).
    2. Footprints — a small patch of cells under each player, redrawn every frame.
    """

    def __init__(
        self,
        config: SoccerPitchConfiguration,
        cell_size_cm: float = 300.0,
        footprint_radius_cells: int = 1,
        team_colors: Optional[List[str]] = None,
    ) -> None:
        self.config = config
        self.cell_size_cm = cell_size_cm
        self.footprint_radius = footprint_radius_cells
        self.team_colors_bgr = [
            _hex_bgr(c) for c in (team_colors or ["#FF1493", "#00BFFF"])
        ]
        self.nx = max(1, int(np.ceil(config.length / cell_size_cm)))
        self.ny = max(1, int(np.ceil(config.width / cell_size_cm)))
        self.team_counts = np.zeros((2, self.ny, self.nx), dtype=np.float32)

    def _cell_indices(self, x_cm: float, y_cm: float) -> Tuple[int, int]:
        ix = int(np.clip(x_cm // self.cell_size_cm, 0, self.nx - 1))
        iy = int(np.clip(y_cm // self.cell_size_cm, 0, self.ny - 1))
        return ix, iy

    def _cells_in_disk(self, ix: int, iy: int) -> List[Tuple[int, int]]:
        r = self.footprint_radius
        cells = []
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                ci, cj = ix + di, iy + dj
                if 0 <= ci < self.nx and 0 <= cj < self.ny:
                    cells.append((ci, cj))
        return cells

    def update(
        self,
        pitch_xy_cm: np.ndarray,
        team_ids: np.ndarray,
        *,
        include_goalkeepers: bool = True,
    ) -> None:
        """Accumulate team grid from player positions (cm). team_id 0 or 1 only."""
        for xy, team in zip(pitch_xy_cm, team_ids):
            team = int(team)
            if team not in (0, 1):
                continue
            if not include_goalkeepers and team > 1:
                continue
            ix, iy = self._cell_indices(float(xy[0]), float(xy[1]))
            self.team_counts[team, iy, ix] += 1.0

    def footprint_cells(
        self,
        pitch_xy_cm: np.ndarray,
        team_ids: np.ndarray,
    ) -> List[Tuple[int, int, int]]:
        """Current-frame cells under each player: (team_id, ix, iy)."""
        out: List[Tuple[int, int, int]] = []
        for xy, team in zip(pitch_xy_cm, team_ids):
            team = int(team)
            if team not in (0, 1):
                continue
            ix, iy = self._cell_indices(float(xy[0]), float(xy[1]))
            for ci, cj in self._cells_in_disk(ix, iy):
                out.append((team, ci, cj))
        return out

    def _cell_rect_pixels(
        self,
        ix: int,
        iy: int,
        scale: float,
        padding: int,
    ) -> Tuple[int, int, int, int]:
        cs = self.cell_size_cm
        x0 = int(ix * cs * scale) + padding
        y0 = int(iy * cs * scale) + padding
        x1 = int((ix + 1) * cs * scale) + padding
        y1 = int((iy + 1) * cs * scale) + padding
        return x0, y0, x1, y1

    def render_accumulated(
        self,
        pitch: np.ndarray,
        *,
        scale: float = 0.1,
        padding: int = 50,
        min_visit_fraction: float = 0.03,
        overlay_alpha: float = 0.72,
        dominant_team_per_cell: bool = True,
    ) -> np.ndarray:
        """
        Occupancy heatmap only: more time in a cell → stronger color.

        If dominant_team_per_cell is True, each cell is colored by whichever team
        visited it more (not two semi-transparent layers on top of each other).
        """
        out = pitch.copy()
        total = self.team_counts[0] + self.team_counts[1]
        if total.max() <= 0:
            return out

        peak = float(total.max())
        layer = np.zeros_like(out)

        for iy in range(self.ny):
            for ix in range(self.nx):
                visits = float(total[iy, ix])
                if visits < peak * min_visit_fraction:
                    continue
                strength = visits / peak
                if dominant_team_per_cell:
                    team = 0 if self.team_counts[0, iy, ix] >= self.team_counts[1, iy, ix] else 1
                else:
                    team = 0 if self.team_counts[0, iy, ix] > self.team_counts[1, iy, ix] else 1
                color = np.array(self.team_colors_bgr[team], dtype=np.float32)
                tint = (color * (0.25 + 0.75 * strength)).astype(np.uint8)
                x0, y0, x1, y1 = self._cell_rect_pixels(ix, iy, scale, padding)
                cv2.rectangle(layer, (x0, y0), (x1, y1), tint.tolist(), -1)

        cv2.addWeighted(layer, overlay_alpha, out, 1.0 - overlay_alpha, 0, out)
        return out

    def render_layers(
        self,
        pitch: np.ndarray,
        pitch_xy_cm: np.ndarray,
        team_ids: np.ndarray,
        *,
        scale: float = 0.1,
        padding: int = 50,
        team_heatmap_alpha: float = 0.45,
        footprint_alpha: float = 0.65,
        show_accumulated: bool = True,
        show_footprints: bool = True,
    ) -> np.ndarray:
        """Draw accumulated team heat and/or current footprints on the pitch image."""
        out = pitch.copy()

        if show_accumulated:
            out = self.render_accumulated(
                out,
                scale=scale,
                padding=padding,
                overlay_alpha=team_heatmap_alpha,
                min_visit_fraction=0.05,
                dominant_team_per_cell=False,
            )

        if show_footprints:
            footprint_layer = np.zeros_like(out)
            seen = set()
            for team, ix, iy in self.footprint_cells(pitch_xy_cm, team_ids):
                key = (team, ix, iy)
                if key in seen:
                    continue
                seen.add(key)
                x0, y0, x1, y1 = self._cell_rect_pixels(ix, iy, scale, padding)
                cv2.rectangle(
                    footprint_layer, (x0, y0), (x1, y1), self.team_colors_bgr[team], -1
                )
            if seen:
                cv2.addWeighted(
                    footprint_layer, footprint_alpha, out, 1.0 - footprint_alpha, 0, out
                )

        return out


def render_heatmap_radar_panel(
    pitch: np.ndarray,
    pitch_xy_cm: np.ndarray,
    team_ids: np.ndarray,
    heatmap: DynamicPitchHeatmap,
    *,
    dot_colors: List[str],
    scale: float = 0.1,
    padding: int = 50,
    dot_radius: int = 14,
    viz: str = "full",
) -> np.ndarray:
    """
    viz modes:
      - 'accumulated': occupancy grid only (time in cell → color)
      - 'full': accumulated + footprints + dots
      - 'footprints': footprints + dots (no history heat)
    """
    from sports.annotators.soccer import draw_points_on_pitch

    heatmap.update(pitch_xy_cm, team_ids)

    if viz == "accumulated":
        panel = heatmap.render_accumulated(pitch, scale=scale, padding=padding)
        return panel

    show_accumulated = viz == "full"
    show_footprints = viz in ("full", "footprints")
    panel = heatmap.render_layers(
        pitch,
        pitch_xy_cm,
        team_ids,
        scale=scale,
        padding=padding,
        show_accumulated=show_accumulated,
        show_footprints=show_footprints,
    )

    if viz in ("full", "footprints"):
        for team in (0, 1):
            mask = team_ids == team
            if not np.any(mask):
                continue
            panel = draw_points_on_pitch(
                config=heatmap.config,
                xy=pitch_xy_cm[mask],
                face_color=sv.Color.from_hex(dot_colors[team]),
                edge_color=sv.Color.from_hex("#FFFFFF"),
                radius=dot_radius,
                thickness=2,
                scale=scale,
                padding=padding,
                pitch=panel,
            )
    return panel


def save_accumulated_heatmap_image(
    heatmap: DynamicPitchHeatmap,
    path: str,
    *,
    upscale: float = 2.0,
    scale: float = 0.1,
    padding: int = 50,
) -> None:
    """Write final match occupancy heatmap as PNG/JPG."""
    from sports.annotators.soccer import draw_pitch

    base = draw_pitch(config=heatmap.config)
    panel = heatmap.render_accumulated(base, scale=scale, padding=padding)
    if upscale != 1.0:
        h, w = panel.shape[:2]
        panel = cv2.resize(
            panel,
            (int(w * upscale), int(h * upscale)),
            interpolation=cv2.INTER_LINEAR,
        )
    cv2.imwrite(path, panel)
