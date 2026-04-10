import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/surgical_tools_ft/run1/weights/best.pt")
cap   = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4, verbose=False)[0]
    frame   = results.plot()

    cv2.imshow("Surgical Tool Detection", frame)
    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
        break

cap.release()
cv2.destroyAllWindows()