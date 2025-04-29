# Real-Time Human Velocity Tracking using Kalman Filter

## Objective
To detect people in a video stream and track their velocity in real-time using Kalman Filter and HOG (Histogram of Oriented Gradients) based human detection, providing accurate motion prediction and trajectory visualization.

## Project Overview
This project implements a real-time people tracking system using computer vision and predictive modeling. It integrates OpenCV’s HOG-based person detector and a Kalman Filter-based multi-object tracking system to continuously predict the position and velocity of each detected person in the video. It also saves the tracking details to a CSV file for further analysis.

## Technologies Used
- Language: Python
- Libraries: OpenCV, FilterPy, SciPy, NumPy

## Algorithms Used
1. **Histogram of Oriented Gradients (HOG):**
   - Used for human detection.
2. **Kalman Filter:**
   - Used for predictive tracking and velocity estimation.

## Features
- Real-time people detection and tracking
- Velocity estimation (m/s and km/h)
- Export tracking data to CSV
- Trajectory visualization

## How It Works
- Each video frame is processed to detect people using the HOG detector.
- A Kalman Filter predicts and updates the movement of detected persons.
- Velocity is calculated based on displacement between frames.
- Data such as Frame, Person ID, X, Y, Speed (m/s), and Speed (km/h) are saved to `output_tracking_data.csv`.

## Advantages
- Robust Tracking: Handles short-term occlusions and noisy detections.

## Limitations
- Limited Detection Accuracy in Complex Scenes.

## Future Enhancement
- Crowd Behavior Analysis using ST-GCN and Kalman Filter.

---

**Developed by Team**  
