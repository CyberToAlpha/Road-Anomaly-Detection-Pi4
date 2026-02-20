import cv2
import time
import datetime
import os
import numpy as np
import csv
from flask import Flask, Response
import tensorflow.lite as tflite

# --- CONFIGURATION ---
CAMERA_INDEX = 0
MODEL_PATH = "best_320_int8.tflite"
# Set to 0.20 so it easily detects potholes on a laptop screen for your demo!
CONFIDENCE_THRESHOLD = 0.40 
NMS_THRESHOLD = 0.50
FRAME_SKIP = 3 

# --- SETUP RECORDING & CSV LOGGING ---
if not os.path.exists('logs'):
    os.makedirs('logs')
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
video_filename = f"logs/mission_{timestamp}.avi"
csv_filename = f"logs/data_{timestamp}.csv"

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(video_filename, fourcc, 10.0, (640, 480))

# Initialize CSV for Dashboard Demo
csv_file = open(csv_filename, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp", "Latitude", "Longitude", "Event", "Confidence", "Action_Required"])

app = Flask(__name__)

# --- LOAD MODEL ---
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def detect_objects(frame):
    original_h, original_w = frame.shape[:2]

    # --- THE INT8 DYNAMIC FIX ---
    input_shape = input_details[0]['shape']
    model_w, model_h = input_shape[1], input_shape[2]
    resized = cv2.resize(frame, (model_w, model_h))
    input_data = np.expand_dims(resized, axis=0)
    
    expected_dtype = input_details[0]['dtype']

    if expected_dtype == np.float32:
        input_data = input_data.astype(np.float32) / 255.0
    elif expected_dtype == np.int8:
        scale, zero_point = input_details[0]['quantization']
        input_data = (input_data.astype(np.float32) / 255.0)
        if scale > 0:
            input_data = ((input_data / scale) + zero_point)
        input_data = np.clip(input_data, -128, 127).astype(np.int8)
    elif expected_dtype == np.uint8:
        input_data = input_data.astype(np.uint8)

    # Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Post-Process (YOLO)
    output_data = interpreter.get_tensor(output_details[0]['index'])[0] 
    
    # De-quantize output if it is int8
    if output_details[0]['dtype'] == np.int8:
        out_scale, out_zero_point = output_details[0]['quantization']
        output_data = (output_data.astype(np.float32) - out_zero_point) * out_scale
        
    output_data = output_data.transpose()

    boxes = []
    confidences = []
    class_ids = []

    for row in output_data:
        scores = row[4:]
        max_score = np.max(scores)
        if max_score > CONFIDENCE_THRESHOLD:
            class_id = np.argmax(scores)
            cx, cy, w, h = row[0], row[1], row[2], row[3]
            x_factor = original_w / model_w
            y_factor = original_h / model_h
            left = int((cx - w/2) * x_factor)
            top = int((cy - h/2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            boxes.append([left, top, width, height])
            confidences.append(float(max_score))
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    if len(indices) > 0:
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        for i in indices.flatten():
            box = boxes[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            
            # Map labels for CSV
            labels_map = ["Pothole", "Speed Breaker", "Manhole", "Crack"]
            label_text = labels_map[class_ids[i]] if class_ids[i] < len(labels_map) else f"Class {class_ids[i]}"
            confidence = confidences[i]
            
            # Draw on Web Stream
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label_text}: {int(confidence*100)}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Log to CSV instantly with dummy GPS coordinates for the demo
            csv_writer.writerow([current_time, "12.9716", "77.5946", label_text, f"{confidence:.2f}", "Maintenance Alert"])
            csv_file.flush() # Force save immediately!

    return frame

def generate_frames():
    # V4L2 backend works best on Pi
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prev_time = 0
    frame_counter = 0

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        try:
            frame = detect_objects(frame)
            
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            
            cv2.putText(frame, f"TRUE FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            out.write(frame)

            frame_counter += 1
            if frame_counter % FRAME_SKIP == 0:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as e:
            print(f"Error: {e}")

@app.route('/')
def index():
    return "<h1>Mission Control: RECORDING...</h1><img src='/video_feed' width='100%'>"

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        out.release()
        csv_file.close()
