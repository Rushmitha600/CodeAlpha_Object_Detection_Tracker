# Object Detection & Tracking Application

A **web-based real-time object detection system** using YOLOv8 and Flask.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python src/app.py
```

The app will automatically open in your browser at **http://localhost:5000**

## 📁 Project Structure

```
object-detection-tracking/
├── README.md                 # This file
├── requirements.txt          # Python dependencies (Required & Optional)
├── DEPENDENCIES.md           # Detailed dependency breakdown
├── src/
│   ├── app.py               # Main Flask web application
│   └── templates/
│       └── index.html       # Beautiful web interface
├── data/
│   ├── videos/              # Input video files (optional)
│   └── output/              # Output videos & results
└── tests/                   # Test files (future)
```

## 📦 Dependency Summary

### ✅ Required (Must Have)
- `opencv-python` - Webcam & video processing
- `numpy` - Numerical operations
- `torch` & `torchvision` - Deep learning framework
- `ultralytics` - YOLOv8 object detection
- `flask` - Web server

### ⚠️ Optional
- `supervision`, `matplotlib`, `pandas`, `seaborn`, `scikit-learn`, `plotly` - Analytics & visualization

See [DEPENDENCIES.md](DEPENDENCIES.md) for details.

## 🎯 Features

✅ Real-time object detection from webcam  
✅ Live video stream in web browser  
✅ Frame counter & statistics  
✅ Clean, responsive web UI  
✅ Support for multiple object classes (persons, vehicles, etc.)

## 🖥️ Usage

1. **Start the app:**
   ```bash
   python src/app.py
   ```

2. **Open in browser:**
   - Automatically opens: `http://localhost:5000`
   - Or manually navigate to that URL

3. **Watch it in action:**
   - See real-time object detection from your webcam
   - View frame count, FPS, and detected objects
   - All live updates in the beautiful dashboard

4. **Stop the app:**
   - Press `Ctrl+C` in the terminal

## 📊 Detected Objects

YOLOv8 can detect 80+ object classes including:
- Persons, animals, vehicles
- Sports, furniture, appliances
- And much more!

## 🔧 Future Enhancements

- [ ] Custom model training
- [ ] Multi-camera support
- [ ] Object tracking (with persistent IDs)
- [ ] Data export (CSV, JSON)
- [ ] Advanced analytics dashboard
- [ ] Video file processing

## 📝 Notes

- Uses YOLOv8 nano model for fast inference (~70ms per frame)
- Webcam feed at 480x640 resolution
- Runs on localhost (no internet required)

## ⚠️ Troubleshooting

**Webcam not opening?**
- Check if another app is using the webcam
- Try: `python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"`

**Port 5000 already in use?**
- Edit `web_app.py` line 79: change `port=5000` to another port (e.g., `5001`)

**Slow performance?**
- YOLOv8 nano is fast but uses CPU by default
- For GPU acceleration, install CUDA-compatible PyTorch

---

**Made with ❤️ for object detection**
