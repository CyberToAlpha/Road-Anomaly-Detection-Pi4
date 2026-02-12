import cv2
import time
import numpy as np
import csv
import datetime
import os
import sys
import multiprocessing
import shutil
import signal

# --- CONFIGURATION ---
MODEL_PATH = "/home/pi/RoadDetection/best_320_int8.tflite" # Absolute Path for Auto-Start
LOG_PATH = "/home/pi/RoadDetection/logs/"
RAM_DISK_PATH = "/mnt/ramdisk/"
CONF_THRESHOLD = 0.50
INPUT_SIZE = 320
# Using labels from user request
LABELS = ["Pothole", "Speed Breaker", "Manhole", "Crack"]
COLORS = [(0, 0, 255), (0, 255, 255), (255, 0, 0), (0, 255, 0)]

def get_interpreter(model_path):
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    
    interpreter = Interpreter(model_path=model_path, num_threads=4)
    interpreter.allocate_tensors()
    return interpreter

class VideoWriterProcess(multiprocessing.Process):
    def __init__(self, queue, video_path, csv_path, width, height, fps=10.0):
        super().__init__()
        self.queue = queue
        self.video_path = video_path
        self.csv_path = csv_path
        self.width = width
        self.height = height
        self.fps = fps

    def run(self):
        # Setup Video Writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(self.video_path, fourcc, self.fps, (self.width, self.height))

        # Setup CSV Logger
        csv_file = open(self.csv_path, mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Timestamp", "Event", "Confidence", "Action_Required"])

        try:
            while True:
                item = self.queue.get()
                if item is None:
                    break
                
                frame, detections, current_time = item
                
                # Draw and Log
                if len(detections) > 0:
                    for label, score, box in detections:
                        # Log to CSV
                        csv_writer.writerow([current_time, label, f"{score:.2f}", "Maintenance Alert"])
                        
                        # Draw on Frame
                        x, y, w, h = box
                        # Find color index based on label if possible, else default
                        try:
                            color_idx = LABELS.index(label) % len(COLORS)
                            color = COLORS[color_idx]
                        except ValueError:
                            color = (0, 255, 0)

                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(frame, f"{label} {int(score*100)}%", (x, y-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                out.write(frame)
        except Exception as e:
            print(f"Error in writer process: {e}")
        finally:
            out.release()
            csv_file.close()
            print("Writer process finished.")

def nms(boxes, confidences, conf_threshold, iou_threshold):
    # Standard NMS
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, iou_threshold)
    return indices

def main():
    # Ensure log directory exists (permanent storage)
    os.makedirs(LOG_PATH, exist_ok=True)
    
    # Check if RAM Disk exists, else fallback
    if os.path.exists(RAM_DISK_PATH):
        working_dir = RAM_DISK_PATH
        print(f"Using RAM Disk: {working_dir}")
    else:
        working_dir = LOG_PATH
        print(f"RAM Disk not found, falling back to: {working_dir}")

    # Generate Session Filenames
    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"mission_{session_id}.avi"
    csv_filename = f"data_{session_id}.csv"
    
    ram_video_path = os.path.join(working_dir, video_filename)
    ram_csv_path = os.path.join(working_dir, csv_filename)

    # Initialize TFLite
    interpreter = get_interpreter(MODEL_PATH)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Setup Camera
    cap = cv2.VideoCapture(0) # V4L2 is default on Pi usually, can add cv2.CAP_V4L2 if needed
    cap.set(3, 640)
    cap.set(4, 480)
    
    # Start Writer Process
    queue = multiprocessing.Queue()
    writer_process = VideoWriterProcess(queue, ram_video_path, ram_csv_path, 640, 480)
    writer_process.start()

    print(f"System Online. Logging to {working_dir}{session_id}")

    try:
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret: break

            # Inference Preprocessing
            img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
            input_data = np.expand_dims(img, axis=0)

            # Normalize if model expects float
            if input_details[0]['dtype'] == np.float32:
                 input_data = (np.float32(input_data) / 255.0)
            
            # Check for quantization requirements (uint8/int8)
            if input_details[0]['dtype'] == np.int8 or input_details[0]['dtype'] == np.uint8:
                # If model expects int8/uint8, input is usually already correct range (0-255) if it's from cv2.imread
                # But sometimes we need to apply scale/zero_point from input details.
                # Common TFLite export from YOLO usually handles image input directly if metadata is correct,
                # but standard practice for "int8" model input is often still 0-255 uint8 or int8.
                # Assuming raw image bytes are fine for now as per user's snippet logic.
                input_data = input_data.astype(input_details[0]['dtype'])

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])

            # Post-Processing
            # Expected output shape depends on model. Common is [1, 4+nc, N] or [1, N, 4+nc]
            # User snippet used: output_data = output_data[0].transpose()
            # This implies original was [1, 4+nc, N] and we want [N, 4+nc]
            
            if output_data.shape[1] < output_data.shape[2]: # [1, 84, 8400] style
                 output_data = output_data[0].transpose()
            else:
                 output_data = output_data[0]

            boxes = []
            confidences = []
            class_ids = []
            
            x_factor = 640 / INPUT_SIZE
            y_factor = 480 / INPUT_SIZE

            for row in output_data:
                scores = row[4:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > CONF_THRESHOLD:
                    cx, cy, w, h = row[0], row[1], row[2], row[3]
                    left = int((cx - w/2) * x_factor)
                    top = int((cy - h/2) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    
                    boxes.append([left, top, width, height])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

            indices = nms(boxes, confidences, CONF_THRESHOLD, 0.4)
            
            detections_to_log = []
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            
            if len(indices) > 0:
                for i in indices.flatten():
                    label = LABELS[class_ids[i] % len(LABELS)]
                    score = confidences[i]
                    box = boxes[i]
                    detections_to_log.append((label, score, box))

            # Send to Writer Process
            queue.put((frame, detections_to_log, current_time))
            
            # Headless Check: Only show window if explicitly possible/requested
            # For strict headless on Pi service, we typically skip this entirely.
            # But for debugging, we can try.
            # Commenting out for "Headless Mode" by default or wrapping in robust try-except
            try:
                # To enable display for debugging, uncomment:
                # cv2.imshow('Dashboard', frame)
                # if cv2.waitKey(1) == ord('q'): break
                pass
            except:
                pass

    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        # Signal writer to stop
        queue.put(None)
        writer_process.join()
        cap.release()
        cv2.destroyAllWindows()
        
        # Move files from RAM Disk to SD Card
        if working_dir == RAM_DISK_PATH:
            print("Moving data to permanent storage...")
            try:
                shutil.move(ram_video_path, os.path.join(LOG_PATH, video_filename))
                shutil.move(ram_csv_path, os.path.join(LOG_PATH, csv_filename))
                print("Data saved successfully.")
            except Exception as e:
                print(f"Error moving files: {e}")

if __name__ == '__main__':
    # Fix for multiprocessing on some platforms
    multiprocessing.freeze_support()
    main()
