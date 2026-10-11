import cv2, csv
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

model = YOLO("best.pt")
cap = cv2.VideoCapture("test.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 25
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("result.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

rows, n = [], 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    r = model(frame, imgsz=960, agnostic_nms=True, verbose=False)[0]
    zone, conf = "None", 0.0
    if len(r.boxes):
        area = r.boxes.xywh[:, 2] * r.boxes.xywh[:, 3]
        i = int(area.argmax())                  # biggest box = near player
        zone = model.names[int(r.boxes.cls[i])]
        conf = float(r.boxes.conf[i])
    rows.append([n, round(n / fps, 2), zone, round(conf, 2)])
    out.write(r.plot())
    n += 1
out.release()

with open("zones.csv", "w", newline="") as f:
    csv.writer(f).writerows([["frame", "second", "zone", "confidence"]] + rows)

counts = Counter(x[2] for x in rows)
changes = sum(1 for a, b in zip(rows, rows[1:]) if a[2] != b[2])
print("Frames:", n, "| Zone changes:", changes)
for z, c in counts.most_common():
    print(z, round(c / fps, 1), "sec")

plt.figure(figsize=(8, 4))
plt.bar(counts.keys(), [c / fps for c in counts.values()])
plt.xticks(rotation=30, ha="right")
plt.ylabel("Seconds")
plt.tight_layout()
plt.savefig("zone_time.png")