import cv2
import numpy as np
import os

print(f"CV2 Version: {cv2.__version__}")
output_path = "test_video.mp4"
height, width = 480, 640
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

if not out.isOpened():
    print("Error: Could not open video writer.")
else:
    print("Video writer opened successfully.")
    frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    out.write(frame)
    out.release()
    print("Video saved.")
    if os.path.exists(output_path):
        print(f"File exists: {os.path.getsize(output_path)} bytes")
    os.remove(output_path)
