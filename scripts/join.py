import cv2, glob

SRC = r"C:\Users\bhagy\Downloads\Dataset\Dataset\Backhand_clear"
files = sorted(glob.glob(SRC + r"\*.mp4"))[:5]

cap = cv2.VideoCapture(files[0])
fps = cap.get(cv2.CAP_PROP_FPS) or 25
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
s = min(1, 1280 / max(w, h))
size = (int(w * s), int(h * s))
out = cv2.VideoWriter("test.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

for f in files:
    cap = cv2.VideoCapture(f)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out.write(cv2.resize(frame, size))
out.release()
print("Joined", len(files), "clips")