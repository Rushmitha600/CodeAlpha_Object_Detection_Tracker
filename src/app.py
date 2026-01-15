from flask import Flask, render_template, Response, jsonify
import cv2
import time
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort

app = Flask(__name__)

# ---------------- CONFIG ----------------
VIDEO_PATH = "data/videos/test_video.mp4"
model = YOLO("yolov8n.pt")
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

cap = None
current_source = None

frame_count = 0
start_time = time.time()
last_labels = []

# ---------------- VIDEO HANDLER ----------------
def open_source(source):
    global cap, current_source, frame_count, start_time, tracker
    if cap:
        cap.release()

    if source == "webcam":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(VIDEO_PATH)

    tracker = Sort()
    frame_count = 0
    start_time = time.time()
    current_source = source


def generate_frames():
    global frame_count, last_labels

    while cap and cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        last_labels = []

        results = model(frame, stream=True)
        detections = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                if conf > 0.4:
                    detections.append([x1, y1, x2, y2, conf])
                    last_labels.append(model.names[cls])

        detections = np.array(detections)
        if len(detections) == 0:
            detections = np.empty((0, 5))

        tracked = tracker.update(detections)

        for obj in tracked:
            x1, y1, x2, y2, obj_id = map(int, obj)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID {obj_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )


# ---------------- ROUTES ----------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed/webcam")
def webcam_feed():
    open_source("webcam")
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video_feed/video")
def video_feed():
    open_source("video")
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stats")
def stats():
    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0

    return jsonify({
        "frame_count": frame_count,
        "fps": round(fps, 1),
        "elapsed": round(elapsed, 1),
        "detections": len(last_labels),
        "detection_types": list(set(last_labels))
    })


# ---------------- RUN ----------------
if __name__ == "__main__":
    print("Object Detection & Tracking Started")
    app.run(host="0.0.0.0", port=5000, debug=False)
