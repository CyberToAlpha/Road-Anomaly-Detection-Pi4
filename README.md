# Road Anomaly Detection (Pi4 Edition)

This repository contains the optimized codebase for running road anomaly detection on a Raspberry Pi 4.

## Directory Structure
- **src/**: Source code for inference and web streaming.
  - `run_inference.py`: Main script for headless detection and logging.
  - `run_web_stream.py`: Flask-based web interface for live monitoring.
- **models/**: TFLite models and label files.
- **docs/**: Documentation and diagrams.
- **demo/**: Sample logs and evidence.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. connect Camera (USB or Pi Camera via V4L2).

## usage
### 1. Headless Inference (Background Service)
Run this script to start detection and logging without a display:
```bash
python src/run_inference.py
```
*Note: Ensure the model path in the script matches your deployment location.*

### 2. Web Stream (Live Monitor)
Run this script to view the camera feed in a browser:
```bash
python src/run_web_stream.py
```
Access the stream at `http://<pi-ip-address>:5000`.

## Configuration
- Update `models/labels.txt` to change class names.
- Adjust `CONF_THRESHOLD` in `src/run_inference.py` for sensitivity.
