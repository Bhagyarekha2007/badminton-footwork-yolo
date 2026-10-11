import cv2, os, random, shutil

SRC = r"C:\Users\bhagy\Downloads\Dataset\Dataset"
os.makedirs("frames_new", exist_ok=True)
os.makedirs("to_label_2", exist_ok=True)

for folder in os.listdir(SRC):
    fpath = os.path.join(SRC, folder)
    if not os.path.isdir(fpath):
        continue
    videos = [v for v in os.listdir(fpath) if v.lower().endswith(".mp4")]
    for v in sorted(videos)[10:20]:
        cap = cv2.VideoCapture(os.path.join(fpath, v))
        step = max(1, int(cap.get(cv2.CAP_PROP_FPS) // 3))
        i = n = 0
        while n < 6:
            ok, img = cap.read()
            if not ok:
                break
            if i % step == 0:
                name = os.path.splitext(v)[0]
                cv2.imwrite(f"frames_new/{folder}_{name}_{n}.jpg", img)
                n += 1
            i += 1
    print(folder, "done")

files = os.listdir("frames_new")
random.seed(1)
pick = random.sample(files, min(200, len(files)))
for f in pick:
    shutil.copy(f"frames_new/{f}", f"to_label_2/{f}")
print("Extracted:", len(files), "Picked:", len(pick))