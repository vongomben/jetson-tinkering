# -*- coding: utf-8 -*-
import time
import threading
import cv2
from flask import Flask, Response, jsonify
from ultralytics import YOLO

# ====== CONFIG ======
MODEL_PATH = "/home/darrel/Documents/apple-pear-projet/newFruit/runs/detect/train2/weights/best.pt"
CAMERA_INDEX = 0
IMG_SIZE = 320
CONF_THRESHOLD = 0.25
DEVICE = 0          # 0 = Jetson GPU, "cpu" = CPU
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
JPEG_QUALITY = 80
# ====================

app = Flask(__name__)

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(CAMERA_INDEX)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    raise RuntimeError("Error: cannot open webcam")

latest_jpeg = None
latest_counts = {
    "apples": 0,
    "pears": 0,
    "total": 0,
    "timestamp": 0.0,
    "fps": 0.0
}

lock = threading.Lock()


def process_frames():
    global latest_jpeg, latest_counts

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        results = model.predict(
            source=frame,
            imgsz=IMG_SIZE,
            conf=CONF_THRESHOLD,
            device=DEVICE,
            verbose=False
        )

        result = results[0]
        names = result.names
        boxes = result.boxes

        apple_count = 0
        pear_count = 0

        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0].item())
                class_name = names[cls_id]

                if class_name == "apples":
                    apple_count += 1
                elif class_name == "pears":
                    pear_count += 1

        annotated = result.plot()

        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if current_time > prev_time else 0.0
        prev_time = current_time

        overlay_lines = [
            f"Apples: {apple_count}",
            f"Pears: {pear_count}",
            f"Total: {apple_count + pear_count}",
            f"FPS: {fps:.1f}"
        ]

        y = 30
        for line in overlay_lines:
            cv2.putText(
                annotated,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
            y += 30

        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        ok, buffer = cv2.imencode(".jpg", annotated, encode_params)
        if not ok:
            continue

        with lock:
            latest_jpeg = buffer.tobytes()
            latest_counts = {
                "apples": apple_count,
                "pears": pear_count,
                "total": apple_count + pear_count,
                "timestamp": current_time,
                "fps": round(fps, 2)
            }


def mjpeg_generator():
    while True:
        with lock:
            frame = latest_jpeg

        if frame is None:
            time.sleep(0.05)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )

        time.sleep(0.01)


@app.route("/")
def index():
    return """
    <html>
      <head>
        <title>Apple / Pear Detection</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </head>
      <body style="margin:0; background:#111; color:#fff; font-family:Arial, sans-serif;">
        <div style="padding:16px; text-align:center;">
          <h2>Apple / Pear Detection</h2>
          <p>Video stream with bounding boxes and live counts</p>
          <img src="/video_feed" style="max-width:95vw; height:auto; border:2px solid #444; border-radius:8px;" />
          <p><a href="/counts" style="color:#6cf;">Open /counts</a></p>
        </div>
      </body>
    </html>
    """


@app.route("/video_feed")
def video_feed():
    return Response(
        mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/counts")
def counts():
    with lock:
        data = dict(latest_counts)
    return jsonify(data)


if __name__ == "__main__":
    worker = threading.Thread(target=process_frames, daemon=True)
    worker.start()

    app.run(host="0.0.0.0", port=5000, threaded=True)