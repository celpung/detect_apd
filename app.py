from flask import Flask, Response, render_template, jsonify
from ultralytics import YOLO
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import threading
import time
import os
import mysql.connector

app = Flask(__name__)

# Load the YOLOv8 model from the specified path
model_path = 'yolomodel.pt'
model = YOLO(model_path)

# Define image transformation without resizing
transform = transforms.Compose([
    transforms.ToTensor(),
])

def transform_image(image):
    return transform(image).unsqueeze(0)

def get_prediction(image):
    tensor = transform_image(image)
    results = model(tensor)
    return results

def get_db_connection():
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="yolo"
    )
    return connection

detection_results = []
lock = threading.Lock()
save_path = 'captured_images'

# Ensure the save directory exists
os.makedirs(save_path, exist_ok=True)

# Dictionary to track the last capture time and position for each class
last_capture = {
    "NO-Mask": {"time": 0, "bbox": (0, 0, 0, 0)},
    "NO-Safety Vest": {"time": 0, "bbox": (0, 0, 0, 0)}
}
capture_cooldown = 60  # Cooldown period in seconds
bbox_threshold = 50  # Bounding box movement threshold in pixels

def generate_video_stream(video_path):
    global detection_results
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save original frame size
        original_height, original_width = frame.shape[:2]

        # Save a copy of the original frame for saving without boxes
        original_frame = frame.copy()

        # Convert frame to PIL image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # Resize image to 640x640 for YOLOv8 model
        resized_image = pil_image.resize((640, 640))
        predictions = get_prediction(resized_image)

        frame_results = []
        capture_frame = {"NO-Mask": False, "NO-Safety Vest": False}

        # Draw boxes on the frame if there are any detections
        for det in predictions[0].boxes.data.tolist():  # Extract detection data
            x1, y1, x2, y2, conf, cls = det[:6]  # Ensure 6 elements are extracted
            # Scale the coordinates back to the original frame size
            x1 = int(x1 * original_width / 640)
            y1 = int(y1 * original_height / 640)
            x2 = int(x2 * original_width / 640)
            y2 = int(y2 * original_height / 640)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{model.names[int(cls)]} {conf:.2f}', (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            frame_results.append((model.names[int(cls)], conf))

            # Check if "NO-Mask" or "No-Safety Vest" is detected
            for cls_name in ["NO-Mask", "NO-Safety Vest"]:
                if model.names[int(cls)] == cls_name:
                    current_time = time.time()
                    last_time = last_capture[cls_name]["time"]
                    last_bbox = last_capture[cls_name]["bbox"]

                    # Check if the detection is within the cooldown period and threshold
                    if current_time - last_time > capture_cooldown or (
                        abs(last_bbox[0] - x1) > bbox_threshold or
                        abs(last_bbox[1] - y1) > bbox_threshold or
                        abs(last_bbox[2] - x2) > bbox_threshold or
                        abs(last_bbox[3] - y2) > bbox_threshold
                    ):
                        capture_frame[cls_name] = True
                        # Update the last capture time and bbox
                        last_capture[cls_name]["time"] = current_time
                        last_capture[cls_name]["bbox"] = (x1, y1, x2, y2)

        # Update detection results with a lock
        with lock:
            detection_results = frame_results

        # Capture frame if "NO-Mask" or "No-Safety Vest" is detected
        for cls_name in ["NO-Mask", "NO-Safety Vest"]:
            if capture_frame[cls_name]:
                timestamp = int(time.time())
                filename = f'{cls_name.lower().replace(" ", "_")}_{timestamp}.jpg'
                cv2.imwrite(os.path.join(save_path, filename), original_frame)
                
                # Save the captured image information to the database
                connection = get_db_connection()
                cursor = connection.cursor()
                cursor.execute(
                    "INSERT INTO captured_images (filename) VALUES (%s)", (filename,)
                )
                connection.commit()
                cursor.close()
                connection.close()

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        # Yield the frame in the appropriate format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    video_path = 'test_video.mp4'
    return Response(generate_video_stream(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detection_results')
def get_detection_results():
    with lock:
        results = detection_results
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
