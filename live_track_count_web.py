# -*- coding: utf-8 -*-
import time
import threading
import cv2
from flask import Flask, Response, jsonify
from ultralytics import YOLO

# ================= CONFIG =================
MODEL_PATH = "/home/darrel/Documents/apple-pear-projet/newFruit/runs/detect/train2/weights/best.pt"
CAMERA_INDEX = 0
IMG_SIZE = 320
CONF_THRESHOLD = 0.25
DEVICE = 0                    # 0 = Jetson GPU, "cpu" = CPU
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
JPEG_QUALITY = 80
TRACKER_CFG = "bytetrack.yaml"   # or "botsort.yaml"
# =========================================

app = Flask(__name__)
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    raise RuntimeError("Error: cannot open webcam")

lock = threading.Lock()

latest_jpeg = None
latest_counts = {
    "unique_apples": 0,
    "unique_pears": 0,
    "unique_total": 0,
    "visible_apples": 0,
    "visible_pears": 0,
    "visible_total": 0,
    "tracked_ids_total": 0,
    "timestamp": 0.0,
    "fps": 0.0
}
latest_objects = []

# IDs already counted once
counted_apple_ids = set()
counted_pear_ids = set()


def process_frames():
    global latest_jpeg, latest_counts, latest_objects
    global counted_apple_ids, counted_pear_ids

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        results = model.track(
            source=frame,
            persist=True,
            tracker=TRACKER_CFG,
            imgsz=IMG_SIZE,
            conf=CONF_THRESHOLD,
            device=DEVICE,
            verbose=False
        )

        result = results[0]
        names = result.names
        boxes = result.boxes

        visible_apples = 0
        visible_pears = 0
        current_objects = []

        annotated = frame.copy()

        if boxes is not None and boxes.xyxy is not None:
            xyxy_list = boxes.xyxy.cpu().numpy()
            cls_list = boxes.cls.cpu().numpy() if boxes.cls is not None else []
            conf_list = boxes.conf.cpu().numpy() if boxes.conf is not None else []
            id_list = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [-1] * len(xyxy_list)

            for xyxy, cls_id, conf, track_id in zip(xyxy_list, cls_list, conf_list, id_list):
                cls_id = int(cls_id)
                class_name = names[cls_id]

                x1, y1, x2, y2 = map(int, xyxy)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                if class_name == "apples":
                    visible_apples += 1
                    if track_id != -1 and track_id not in counted_apple_ids:
                        counted_apple_ids.add(track_id)

                elif class_name == "pears":
                    visible_pears += 1
                    if track_id != -1 and track_id not in counted_pear_ids:
                        counted_pear_ids.add(track_id)

                label = f"{class_name} ID:{track_id} {conf:.2f}"

                # Different colors per class
                if class_name == "apples":
                    color = (0, 255, 0)
                else:
                    color = (0, 200, 255)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.circle(annotated, (cx, cy), 4, color, -1)

                text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
                cv2.putText(
                    annotated,
                    label,
                    (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA
                )

                current_objects.append({
                    "id": int(track_id),
                    "class_name": class_name,
                    "confidence": round(float(conf), 4),
                    "bbox": [x1, y1, x2, y2],
                    "center": [cx, cy]
                })

        unique_apples = len(counted_apple_ids)
        unique_pears = len(counted_pear_ids)

        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if current_time > prev_time else 0.0
        prev_time = current_time

        overlay_lines = [
            f"Visible apples: {visible_apples}",
            f"Visible pears: {visible_pears}",
            f"Unique apples: {unique_apples}",
            f"Unique pears: {unique_pears}",
            f"Unique total: {unique_apples + unique_pears}",
            f"FPS: {fps:.1f}"
        ]

        y = 30
        for line in overlay_lines:
            cv2.putText(
                annotated,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            y += 28

        ok, buffer = cv2.imencode(
            ".jpg",
            annotated,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        )
        if not ok:
            continue

        with lock:
            latest_jpeg = buffer.tobytes()
            latest_objects = current_objects
            latest_counts = {
                "unique_apples": unique_apples,
                "unique_pears": unique_pears,
                "unique_total": unique_apples + unique_pears,
                "visible_apples": visible_apples,
                "visible_pears": visible_pears,
                "visible_total": visible_apples + visible_pears,
                "tracked_ids_total": len(counted_apple_ids.union(counted_pear_ids)),
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
        <title>Apple / Pear Tracking</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </head>
      <body style="margin:0; background:#111; color:#fff; font-family:Arial, sans-serif;">
        <div style="padding:16px; text-align:center;">
          <h2>Apple / Pear Tracking</h2>
          <p>Duplicate-free counting using tracking IDs</p>
          <img src="/video_feed" style="max-width:95vw; height:auto; border:2px solid #444; border-radius:8px;" />
          <p><a href="/counts" style="color:#6cf;">/counts</a> | <a href="/objects" style="color:#6cf;">/objects</a></p>
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


@app.route("/objects")
def objects():
    with lock:
        data = list(latest_objects)
    return jsonify(data)


@app.route("/reset_counts", methods=["POST", "GET"])
def reset_counts():
    global counted_apple_ids, counted_pear_ids
    with lock:
        counted_apple_ids = set()
        counted_pear_ids = set()
    return jsonify({"status": "ok", "message": "Counts reset"})


if __name__ == "__main__":
    worker = threading.Thread(target=process_frames, daemon=True)
    worker.start()
    app.run(host="0.0.0.0", port=5000, threaded=True)