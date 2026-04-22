
# -*- coding: utf-8 -*-
import cv2
from flask import Flask, Response
from ultralytics import YOLO

MODEL_PATH = "/home/darrel/Documents/apple-pear-projet/newFruit/runs/detect/train2/weights/best.pt"

app = Flask(__name__)
model = YOLO(MODEL_PATH)

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise RuntimeError("Error: cannot open webcam")

def generate():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model.predict(
            source=frame,
            imgsz=320,
            conf=0.25,
            device=0,
            verbose=False
        )

        annotated_frame = results[0].plot()

        ok, buffer = cv2.imencode(".jpg", annotated_frame)
        if not ok:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

@app.route("/")
def index():
    return """
    <html>
      <head><title>YOLO Stream</title></head>
      <body style="margin:0;background:#111;color:white;text-align:center;">
        <h2>Apple / Pear Detection</h2>
        <img src="/video_feed" style="max-width:95vw; height:auto; border:2px solid #444;" />
      </body>
    </html>
    """

@app.route("/video_feed")
def video_feed():
    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)