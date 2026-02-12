import cv2
import time
import datetime
import os
import numpy as np
from flask import Flask, Response
import tensorflow.lite as tflite

# --- CONFIGURATION ---
CAMERA_INDEX = 0
MODEL_PATH = "best_320_int8.tflite"
CONFIDENCE_THRESHOLD = 0.45
NMS_THRESHOLD = 0.50

# --- PERFORMANCE TUNING ---
# 1 = Show every frame (Slowest, ~5 FPS)
# 2 = Show every 2nd frame (Faster)
# 3 = Show every 3rd frame (Best for Performance, ~12 FPS inference)
FRAME_SKIP = 3 

# --- SETUP RECORDING ---
if not os.path.exists('logs'):
    os.makedirs('logs')
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
video_filename = f"logs/mission_{timestamp}.avi"
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(video_filename, fourcc, 10.0, (640, 480))

app = Flask(__name__)

# --- LOAD MODEL ---
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Quantization Params
output_scale = output_details[0]['quantization'][0]
output_zero_point = output_details[0]['quantization'][1]

def detect_objects(frame):
    original_h, original_w = frame.shape[:2]

    # Preprocess
    input_shape = input_details[0]['shape']
    model_w, model_h = input_shape[1], input_shape[2]
    resized = cv2.resize(frame, (model_w, model_h))
    input_data = (np.float32(resized) - 128).astype(np.int8)
    input_data = np.expand_dims(input_data, axis=0)
    
    # Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Post-Process (YOLO)
    output_data = interpreter.get_tensor(output_details[0]['index'])[0] 
    output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
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
        for i in indices.flatten():
            box = boxes[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            label = f"Class {class_ids[i]}: {int(confidences[i]*100)}%"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

def generate_frames():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prev_time = 0
    frame_counter = 0

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        try:
            # 1. ALWAYS Detect (This keeps the Real FPS high)
            frame = detect_objects(frame)
            
            # 2. Update FPS Counter
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            
            # Draw FPS on the frame
            cv2.putText(frame, f"TRUE FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 3. ALWAYS Record (Evidence must be complete)
            out.write(frame)

            # 4. CONDITIONAL Stream (The Performance Trick)
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
        # Threaded=True allows Flask to handle requests without blocking the loop too much
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        out.release()
        print(f"Video saved to {video_filename}")
