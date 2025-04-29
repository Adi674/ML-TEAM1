Real-Time Human Velocity Tracking using Kalman Filter

Objective

To detect people in a video stream and track their velocity in real-time using a Kalman Filter and HOG (Histogram of Oriented Gradients) based human detection, enabling accurate motion prediction and trajectory visualization.


---

Project Overview

This project implements a real-time people tracking system using computer vision and predictive modeling. It combines OpenCV’s HOG-based human detector with a Kalman Filter-based tracking system to continuously estimate the position and velocity of detected individuals. The goal is to evaluate speed and motion trajectory with improved accuracy, even when detections are temporarily lost.


---

Technologies Used

Language: Python

Libraries:

OpenCV: Video processing and HOG-based people detection

FilterPy: Kalman Filter implementation

NumPy: Numerical operations


Hardware Requirements: System with webcam or ability to process video files

Software Requirements: Python 3.x, OpenCV, FilterPy



---

Algorithms Used

1. Histogram of Oriented Gradients (HOG)

Detects humans in video frames using shape and edge orientation.

Works with an SVM classifier to identify person-like features.


2. Kalman Filter

Tracks and predicts movement of each detected person.

Recursively estimates position and velocity (x, y, vx, vy) using current and previous states.

Maintains tracking even with missed detections for a few frames.



---

How It Works

1. Human Detection:

Each video frame is scanned using the HOG person detector.

Returns bounding boxes and center coordinates for each detected person.



2. Tracking Initialization:

On initial detection, a PersonTracker with a unique ID and Kalman Filter is created.



3. Kalman Prediction and Update:

For each tracker, Kalman Filter predicts the next position.

Detections are matched to trackers using a distance threshold.

If no match is found, tracker continues prediction, assuming occlusion or temporary invisibility.



4. Velocity Estimation:

Velocity is derived from Kalman state and visualized using arrows and labels on the video.



5. Accuracy Calculation:

Matching accuracy is computed by comparing successful associations to total detections.





---

Advantages

Robust Tracking: Handles short-term occlusion or missed detections smoothly.

Real-time Performance: Runs efficiently on standard systems.

Velocity Prediction: Valuable for surveillance, crowd analysis, and behavioral monitoring.



---

Disadvantages

Detection Accuracy: Struggles in cluttered or complex environments.

False Positives: May occasionally detect non-human objects.

No Re-identification: Cannot track the same person after they leave and re-enter.

Fixed Thresholds: Static matching thresholds may not adapt to all scenes.



---

Suggestions for Improvement

1. Use the Hungarian Algorithm for better detection-tracker assignment.


2. Apply Non-Maximum Suppression (NMS) to reduce false detections.


3. Tune Kalman Filter parameters for smoother motion tracking.


4. Temporarily maintain trackers for lost persons before removing.


5. Use Intersection-over-Union (IoU) instead of distance for better matching.



