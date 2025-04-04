# Crowd Surveillance System

A comprehensive crowd surveillance system that combines multiple computer vision techniques for real-time monitoring and analysis.

## Features

- Crowd Anomaly Detection (>85% accuracy)
- People Counting with Unique ID Tracking
- Weapon Detection using YOLOv8
- Crowd Density Estimation with Heatmap Visualization
- Dual-frame Display:
  - Left: Anomaly detection, people tracking, and weapon detection
  - Right: Crowd density heatmap

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Dependencies listed in `requirements.txt`

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download DeepSORT model weights:
```bash
cd deep_sort_pytorch/deep_sort/deep/checkpoint
# Download ckpt.t7 from the official repository
```

## Usage

Run the main script:
```bash
python crowd_surveillance.py
```

## Output

- Processed video will be saved in the `results` directory
- Real-time visualization window showing:
  - People tracking with unique IDs
  - Weapon detection alerts
  - Anomaly detection warnings
  - Crowd density heatmap

## Dataset

The system uses the UMN CLAHE dataset for training and testing.
