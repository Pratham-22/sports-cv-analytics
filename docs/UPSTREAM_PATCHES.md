# Patches on `roboflow/sports` (for future PR)

Applied locally under `roboflow-sports/examples/soccer/main.py` (not committed — clone upstream separately):

- Ball on RADAR minimap (`create_ball_detector`, `detect_ball`, `render_radar`)
- ByteTrack guards when `tracker_id` is None
- `overlap_filter` rename for supervision 0.28
- `resolve_goalkeepers_team_id` empty-team fix
- `--show` flag (default off for headless)

Re-clone upstream and diff `main.py` before opening PRs.
