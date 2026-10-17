import os, shutil, subprocess, tempfile, uuid
from collections import Counter
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

MODEL_PATH = "best.pt"
IMGSZ = 640
CONF = 0.25          # confidence threshold (write this in the README)
SMOOTH_K = 15        # frames used for the majority vote (odd number)
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
app.mount("/outputs", StaticFiles(directory=OUT_DIR), name="outputs")
model = YOLO(MODEL_PATH)

# Leave empty to use the class names stored inside best.pt
ZONE_NAMES = []
NAME_MAP = dict(enumerate(ZONE_NAMES)) if ZONE_NAMES else dict(model.names)


@app.get("/")
def home():
    return FileResponse("frontend.html")


def smooth(zones, k=SMOOTH_K):
    """Replace each frame's zone by the most common zone around it."""
    half = k // 2
    out = []
    for i in range(len(zones)):
        window = zones[max(0, i - half): i + half + 1]
        out.append(Counter(window).most_common(1)[0][0])
    return out


def to_h264(src, dst):
    try:
        import imageio_ffmpeg
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", src,
                        "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", dst],
                       check=True, capture_output=True)
        os.remove(src)
        return dst
    except Exception as e:
        print("H.264 conversion failed, using raw video:", e)
        return src


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    name = file.filename or "video.mp4"
    if not name.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
        raise HTTPException(400, "Please upload a video (mp4, mov, avi or mkv).")
    uid = uuid.uuid4().hex[:8]
    src = os.path.join(tempfile.gettempdir(), uid + "_" + os.path.basename(name))
    with open(src, "wb") as f:
        shutil.copyfileobj(file.file, f)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise HTTPException(400, "This video could not be read. Try another file.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Pass 1: predict. One player per frame = the biggest box.
    zones, confs, boxes = [], [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = model(frame, imgsz=IMGSZ, conf=CONF, agnostic_nms=True, verbose=False)[0]
        zone, conf, box = "No player", 0.0, None
        if len(r.boxes):
            area = r.boxes.xywh[:, 2] * r.boxes.xywh[:, 3]
            i = int(area.argmax())
            zone = NAME_MAP[int(r.boxes.cls[i])]
            conf = float(r.boxes.conf[i])
            box = [int(v) for v in r.boxes.xyxy[i].tolist()]
        zones.append(zone)
        confs.append(conf)
        boxes.append(box)
    cap.release()

    n = len(zones)
    zones = smooth(zones)    # remove flicker

    # Pass 2: draw the smoothed zone and save the video.
    cap = cv2.VideoCapture(src)
    raw = os.path.join(OUT_DIR, uid + "_raw.mp4")
    writer = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for z, c, b in zip(zones, confs, boxes):
        ok, frame = cap.read()
        if not ok:
            break
        if b is not None:
            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"{z} {c:.2f}", (b[0], max(20, b[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        writer.write(frame)
    cap.release()
    writer.release()
    os.remove(src)
    final = to_h264(raw, os.path.join(OUT_DIR, uid + ".mp4"))

    rows = [{"frame": i, "second": round(i / fps, 2),
             "zone": zones[i], "confidence": round(confs[i], 2)} for i in range(n)]
    counts = Counter(zones)
    changes = sum(1 for i in range(1, n) if zones[i] != zones[i - 1])
    seen = [c for c in confs if c > 0]
    return {"frames": n,
            "zone_changes": changes,
            "avg_confidence": round(sum(seen) / len(seen), 2) if seen else 0,
            "seconds_per_zone": {k: round(v / fps, 1) for k, v in counts.items()},
            "results": rows,
            "video_url": "/outputs/" + os.path.basename(final)}