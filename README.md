
# Crowd Surveillance System - ML TEAM

A real-time intelligent surveillance system that detects **crowd anomalies**, **weapons**, and **generates crowd density heatmaps** using state-of-the-art deep learning models like **YOLOv8**, **DeepSort**, and **Autoencoders**.

## Features

- Weapon Detection using YOLOv8
- Crowd Tracking using DeepSort
- Anomaly Detection using Autoencoders
- Density Heatmaps using Gaussian Smoothing
- Real-time Video Analysis with annotated output video generation

## Components & Architecture

```
CrowdSurveillanceSystem/
│
├── models/
│   ├── weapon_model_save.pt         # YOLOv8 model for weapon detection
│   └── anomaly_model.pth            # Trained Autoencoder model
│
├── AnomalyAutoencoder.py            # PyTorch model for anomaly detection
├── CrowdTracking_YOLO.py            # YOLOv8 integration and tracking logic
└── CrowdSurveillanceSystem.py       # Main pipeline combining all modules
```

## Installation

```bash
git clone https://github.com/your-repo/Crowd_Surveillance_ML_TEAM
cd Crowd_Surveillance_ML_TEAM
pip install -r requirements.txt
```

## Dataset Structure

```
train_UMN_clahe/
├── 0/     # Normal frames
└── 1/     # Anomalous frames
```

## Training the Anomaly Detector

```bash
python AnomalyAutoencoder.py
```

This will:
- Train the Autoencoder model
- Save the trained model as `anomaly_model.pth`

## Running the Surveillance System

```bash
python CrowdSurveillanceSystem.py
```

The system will process your video file (default: `Crowd-Activity-All.avi`) and output `results/output.mp4` with detection overlays.

## Output Visualizations

- Green boxes for tracked people  
- Red boxes for detected weapons  
- "Anomaly Detected" alert for unusual crowd behavior  
- Live heatmap for density distribution  
To view the output video click this link and select anomaly_and_heatmap video https://drive.google.com/drive/folders/1THZPL--L9kkjRPqmPbmZpYuM2gWYMzN_
## Dependencies

- opencv-python  
- torch, torchvision  
- ultralytics  
- numpy  
- scipy  
- tqdm  

Install with:

```bash
pip install opencv-python torch torchvision ultralytics numpy scipy tqdm
```

## Notes

- Make sure you have `weapon_model_save.pt` in your working directory  
- You can fine-tune the anomaly threshold (`self.threshold`) in `AnomalyDetector` based on your dataset for better performance  

## Team Members

- Kishore Kumar G B
- Nirmal  
- Jayashree 
- Bhagyashree
- Santhosh S
- Dhanya Shetty
- Saritha Chowdhary
