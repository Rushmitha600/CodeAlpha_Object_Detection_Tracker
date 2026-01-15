import cv2

cap = cv2.VideoCapture("data/videos/test_video.mp4")

if cap.isOpened():
    print("Video loaded successfully")
else:
    print("Failed to load video")

cap.release()
