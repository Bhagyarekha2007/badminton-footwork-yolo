from ultralytics import YOLO

model = YOLO("best.pt")
r = model("photo.jpg", imgsz=960, agnostic_nms=True)
r[0].save("result.jpg")
print(r[0].boxes.cls, r[0].boxes.conf)