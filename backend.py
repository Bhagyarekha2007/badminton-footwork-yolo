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
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
app.mount("/outputs", StaticFiles(directory=OUT_DIR), name="outputs")
model = YOLO(MODEL_PATH)

# Leave empty for the old model. For the NEW model, paste the 7 names
# from export\data.yaml here in order, e.g. ["Name A", "Name B", ...]
ZONE_NAMES = []
NAME_MAP = dict(enumerate(ZONE_NAMES)) if ZONE_NAMES else dict(model.names)


@app.get("/")
def home():
    return FileResponse("frontend.html")


def to_h264(src, dst):
    try:
        import imageio_ffmpeg
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", src,
                        "-vcodec", "libx264", "-pix_fmt", "yuv420p", dst],
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
    raw = os.path.join(OUT_DIR, uid + "_raw.mp4")
    writer = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    rows, n = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = model(frame, imgsz=IMGSZ, agnostic_nms=True, verbose=False)[0]
        r.names = NAME_MAP
        zone, conf = "No player", 0.0
        if len(r.boxes):
            area = r.boxes.xywh[:, 2] * r.boxes.xywh[:, 3]
            i = int(area.argmax())
            zone = NAME_MAP[int(r.boxes.cls[i])]
            conf = float(r.boxes.conf[i])
        writer.write(r.plot())
        rows.append({"frame": n, "second": round(n / fps, 2),
                     "zone": zone, "confidence": round(conf, 2)})
        n += 1
    cap.release()
    writer.release()
    os.remove(src)
    final = to_h264(raw, os.path.join(OUT_DIR, uid + ".mp4"))
    counts = Counter(x["zone"] for x in rows)
    return {"frames": n,
            "seconds_per_zone": {k: round(v / fps, 1) for k, v in counts.items()},
            "results": rows,
            "video_url": "/outputs/" + os.path.basename(final)}