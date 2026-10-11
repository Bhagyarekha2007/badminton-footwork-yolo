# Badminton Footwork Zone Detection (YOLOv8)

Detects the near badminton player in a behind-court video and classifies their
court zone (6 footwork zones + Uncertain). Based on: "A Deep Learning Approach to
Badminton Player Footwork Detection Based on YOLO Models" (ICSIPA 2024).

## Classes
Forehand/Backhand x Front_Corner, Side, BackCourt_Corner, and Uncertain (7 total).

## Pipeline
Videos -> frames (OpenCV) -> labeling (Roboflow) -> YOLOv8 training (Colab)
-> analysis -> FastAPI backend -> HTML frontend

## Files
- `best.pt` - trained YOLOv8n model
- `analyze.py` - zone counts, time per zone, CSV and chart for a video
- `backend.py` - FastAPI server (`POST /predict`)
- `frontend.html` - upload page showing zone seconds and confidence
- `scripts/` - frame extraction and helper scripts
- `results/` - chart, CSV and sample prediction

## How to run
1. `pip install ultralytics opencv-python fastapi uvicorn python-multipart matplotlib`
2. Start the backend: `python -m uvicorn backend:app`
3. Open `frontend.html` in a browser, choose a video, click Analyze.

## Results
- Test mAP50: 0.578 (YOLOv8n, 23 test frames). The base paper reports 0.633.
- Demo video (5 joined clips, 237 frames): average confidence 0.60.

## Limitations
- Small dataset, limited venues, one camera view (behind the court).
- Right-handed singles players only.
- Zone labels flicker between frames; smoothing and more data would help.

## Dataset
Videos from the Mendeley Data set "A Video-Based Frame-Level Dataset of Badminton
Player Movements for Motion Analysis" (DOI 10.17632/3sp4xntp34.1).
