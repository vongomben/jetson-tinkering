import cv2
from ultralytics import YOLO

# Path to your trained model
MODEL_PATH = "/home/darrel/Documents/apple-pear-projet/newFruit/runs/detect/train2/weights/best.pt"

# Load model
model = YOLO(MODEL_PATH)

# Open webcam
# 0 = first available webcam
cap = cv2.VideoCapture(0)

# Optional: force a lighter resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: cannot open webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error while reading frame.")
        break

    # YOLO inference on current frame
    results = model.predict(
        source=frame,
        imgsz=320,
        conf=0.25,
        device=0,
        verbose=False
    )

    # Automatically draw boxes and labels on frame
    annotated_frame = results[0].plot()

    # Show annotated stream
    cv2.imshow("Apple / Pear Detection", annotated_frame)

    # Exit with q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()