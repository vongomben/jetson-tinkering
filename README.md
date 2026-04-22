# Jetson Tinkering (Apple/Pear Detection and Tracking)

This repository contains Python scripts to:
- extract frames from local videos,
- run YOLO live detection on webcam frames,
- expose video streams and detection/tracking data through Flask HTTP endpoints.

## Requirements

- Python 3.8+
- A trained YOLO model (`best.pt`) compatible with your classes (`apples`, `pears`)
- Webcam connected to the Jetson device (or host machine)

Install dependencies:

```bash
pip3 install ultralytics opencv-python flask
```

> Note: on Jetson, OpenCV and CUDA/TensorRT are often installed differently. Keep the package versions aligned with your JetPack setup.

## Configuration

Each script has inline config variables near the top, including:
- `MODEL_PATH`
- camera index and frame size
- inference settings (`imgsz`, confidence, device)
- optional tracker config (`bytetrack.yaml` or `botsort.yaml`)

Update `MODEL_PATH` in each script before running.

---

## Roboflow and Labelling

### Create an "Object Detection" Project in [Roboflow](http://roboflow.com/)

![Detection](img/roboflow-01.gif)


### Upload all the pictures you have from the objects you want to detect

![Upload](img/roboflow-02.gif)

### Label the objets (long job!)

![label](img/roboflow-03-labelling.gif)

### Create a good version (follow the 80-10-10 rule)

![Versions](img/roboflow-04-versions.gif)

### Train on the Jetson

Once you donwloaded the latest `Yolo11n` from [ultralytics](https://github.com/ultralytics/ultralytics/tree/v8.3.252) github (`n` stands for light, possibly nano) you can launch the follwoing command in order to have the Jetson.

```bash
yolo train model=yolo11n.pt data=./data.yaml epochs=100 imgsz=320 batch=2 workers=0 device=0
```

#### The command: 
* `yolo11` `n` → nano, light `.pt` pre-trained 
* `model=yolo11n.pt` →  Where are the images? / Where are the labels? / How many classes are there? / What are they called?
* `epochs=100` →  How many times does the model process the entire dataset? 1 epoch means all images have been seen once. 100 epochs = 100 times
* `imgsz=320`  →  320 × 320  
* `batch=2`  →  2 images at a time
* `workers=0`  →  0 all in one thread (slower but more stable) 
* `device=0` →  using `CUDA` vs `CPU` that would use `GPU` with far more time

![Training](img/training.gif)


## Scripts

### `extract_frames.py`

Extracts evenly distributed frames from videos inside `./videos` and saves JPEGs in `./frames_out`.

Run:

```bash
python3 extract_frames.py
```

What it does:
- scans `./videos` for supported formats (`.mp4`, `.mov`, `.m4v`, `.avi`, `.mkv`)
- extracts `FRAMES_PER_VIDEO` frames per video (or fewer if video is short)
- optionally resizes images while preserving aspect ratio (`RESIZE_LONG_SIDE`)

HTTP endpoints: none  
JSON output: none (console logs only)

---

### `live_detect.py`

Runs YOLO detection from webcam and shows an annotated OpenCV window.

Run:

```bash
python3 live_detect.py
```

Controls:
- press `q` to quit

HTTP endpoints: none  
JSON output: none (local window stream only)

---



### `live_detect_web.py`

Runs YOLO detection from webcam and publishes an MJPEG stream via Flask.

![live_detect_web demo](img/live_detect_web.gif)

Run:

```bash
python3 live_detect_web.py
```

Default server:
- `http://<JETSON_IP>:5000/` - simple HTML page with embedded stream
- `http://<JETSON_IP>:5000/video_feed` - MJPEG stream endpoint

#### Endpoints

- `GET /`
  - Returns a minimal HTML page showing the stream
- `GET /video_feed`
  - Returns `multipart/x-mixed-replace` MJPEG frames

JSON output: none

---

### `live_detect_count_web.py`

Runs YOLO detection, overlays live counters on the frame, streams MJPEG, and exposes current counts as JSON.

![live_detect_count_web demo](img/live_detect_count_web.gif)

Run:

```bash
python3 live_detect_count_web.py
```

Default server:
- `http://<JETSON_IP>:5000/`
- `http://<JETSON_IP>:5000/video_feed`
- `http://<JETSON_IP>:5000/counts`

#### Endpoints

- `GET /`
  - HTML page with stream preview and link to `/counts`
- `GET /video_feed`
  - MJPEG stream with bounding boxes and overlay text
- `GET /counts`
  - Latest detection counters and runtime metadata in JSON

Example `/counts` response:

```json
{
  "apples": 2,
  "pears": 1,
  "total": 3,
  "timestamp": 1776846031.52,
  "fps": 14.6
}
```

---

### `live_track_count_web.py`

Runs YOLO tracking (`model.track`) to avoid double counting by using persistent object IDs.  
Exposes both aggregate counters and per-object tracking data.

![live_track_count_web demo](img/live_track_count_web.gif)

Run:

```bash
python3 live_track_count_web.py
```

Default server:
- `http://<JETSON_IP>:5000/`
- `http://<JETSON_IP>:5000/video_feed`
- `http://<JETSON_IP>:5000/counts`
- `http://<JETSON_IP>:5000/objects`
- `http://<JETSON_IP>:5000/reset_counts` (GET or POST)

#### Endpoints

- `GET /`
  - HTML page with stream and links
- `GET /video_feed`
  - MJPEG stream with boxes, IDs, and counters overlay
- `GET /counts`
  - Aggregated unique/visible counts and telemetry JSON
- `GET /objects`
  - List of currently visible tracked objects (ID/class/confidence/bbox/center)
- `GET /reset_counts`
- `POST /reset_counts`
  - Clears unique counted IDs for apples and pears

Example `/counts` response:

```json
{
  "unique_apples": 6,
  "unique_pears": 4,
  "unique_total": 10,
  "visible_apples": 2,
  "visible_pears": 1,
  "visible_total": 3,
  "tracked_ids_total": 10,
  "timestamp": 1776846031.52,
  "fps": 13.9
}
```

Example `/objects` response:

```json
[
  {
    "id": 12,
    "class_name": "apples",
    "confidence": 0.9134,
    "bbox": [101, 55, 188, 140],
    "center": [144, 97]
  },
  {
    "id": 21,
    "class_name": "pears",
    "confidence": 0.8761,
    "bbox": [260, 82, 320, 170],
    "center": [290, 126]
  }
]
```

Example `/reset_counts` response:

```json
{
  "status": "ok",
  "message": "Counts reset"
}
```


