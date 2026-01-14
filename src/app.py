from flask import Flask, render_template, Response, jsonify
import cv2
import threading
from ultralytics import YOLO
import time
import webbrowser
from datetime import datetime

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

class ObjectDetectionApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.detector = YOLO("yolov8n.pt")
        self.frame_count = 0
        self.start_time = time.time()
        self.running = True
        self.detections_list = []
        
    def get_frame(self):
        """Get annotated frame"""
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        self.frame_count += 1
        
        # Run detection
        results = self.detector(frame)
        annotated_frame = results[0].plot()
        
        # Get detection info
        if results[0].boxes:
            self.detections_list = [self.detector.names[int(cls)] for cls in results[0].boxes.cls]
        else:
            self.detections_list = []
        
        # Add frame info
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(annotated_frame, f"Frame: {self.frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        
        return frame_bytes
    
    def get_stats(self):
        """Get app statistics"""
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        return {
            'frame_count': self.frame_count,
            'fps': round(fps, 1),
            'elapsed': round(elapsed, 1),
            'detections': len(self.detections_list),
            'detection_types': list(set(self.detections_list))[:5]
        }
    
    def close(self):
        """Cleanup"""
        self.running = False
        self.cap.release()

# Initialize app
detector_app = ObjectDetectionApp()

@app.route('/')
def index():
    """Serve main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Stream video frames"""
    def generate():
        while detector_app.running:
            frame = detector_app.get_frame()
            if frame is None:
                break
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    """Get detection stats"""
    return jsonify(detector_app.get_stats())

if __name__ == '__main__':
    # Open browser
    webbrowser.open('http://localhost:5000')
    
    # Run Flask app
    print("\n" + "="*70)
    print(" 🎥 REAL-TIME OBJECT DETECTION & TRACKING SYSTEM")
    print("="*70)
    print(f" 🌐 Opening in browser: http://localhost:5000")
    print(f" ⏱️  Started at: {datetime.now().strftime('%H:%M:%S')}")
    print(" 📹 Accessing webcam...")
    print(" ⌨️  Press Ctrl+C to stop")
    print("="*70 + "\n")
    
    try:
        app.run(debug=False, host='localhost', port=5000, use_reloader=False)
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print(" 🛑 Shutting down...")
        detector_app.close()
        print("="*70 + "\n")
