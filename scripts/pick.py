import os, shutil

os.makedirs("to_label", exist_ok=True)
for folder in os.listdir("frames"):
    files = sorted(os.listdir(f"frames/{folder}"))[:25]
    for f in files:
        shutil.copy(f"frames/{folder}/{f}", f"to_label/{folder}_{f}")
print("Total images:", len(os.listdir("to_label")))