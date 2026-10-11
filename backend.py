import os, shutil, tempfile
from collections import Counter
import cv2
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
model = YOLO("best.pt")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    path = os.path.join(tempfile.gettempdir(), file.filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    rows, n = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = model(frame, imgsz=960, agnostic_nms=True, verbose=False)[0]
        zone, conf = "None", 0.0
        if len(r.boxes):
            area = r.boxes.xywh[:, 2] * r.boxes.xywh[:, 3]
            i = int(area.argmax())
            zone = model.names[int(r.boxes.cls[i])]
            conf = float(r.boxes.conf[i])
        rows.append({"frame": n, "second": round(n / fps, 2),
                     "zone": zone, "confidence": round(conf, 2)})
        n += 1
    cap.release()
    counts = Counter(x["zone"] for x in rows)
    return {"frames": n,
            "seconds_per_zone": {k: round(v / fps, 1) for k, v in counts.items()},
            "results": rows}