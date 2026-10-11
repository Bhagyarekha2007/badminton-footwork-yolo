import cv2, os

SRC = r"C:\Users\bhagy\Downloads\Dataset\Dataset"
MAX_VIDEOS = 10
MAX_FRAMES = 30

for folder in os.listdir(SRC):
    fpath = os.path.join(SRC, folder)
    if not os.path.isdir(fpath):
        continue
    os.makedirs(f"frames/{folder}", exist_ok=True)
    videos = [v for v in os.listdir(fpath)
              if v.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
    for v in sorted(videos)[:MAX_VIDEOS]:
        cap = cv2.VideoCapture(os.path.join(fpath, v))
        step = max(1, int(cap.get(cv2.CAP_PROP_FPS) // 3))
        i = n = 0
        while n < MAX_FRAMES:
            ok, img = cap.read()
            if not ok:
                break
            if i % step == 0:
                name = os.path.splitext(v)[0]
                cv2.imwrite(f"frames/{folder}/{name}_{n:03d}.jpg", img)
                n += 1
            i += 1
    print(folder, "done")