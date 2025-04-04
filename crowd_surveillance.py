import cv2
import numpy as np
import torch
from ultralytics import YOLO
from scipy.ndimage import gaussian_filter
import os

class CrowdSurveillanceSystem:
    def __init__(self):
        # Initialize YOLO weapon detection model
        self.weapon_detector = YOLO("weapon_model_save.pt")
        
        # Initialize tracker
        self.track_id = 0
        self.trackers = {}  # Dictionary to store trackers for each person
        
        # Initialize anomaly detection model
        self.anomaly_threshold = 0.85
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False
        )
        
        # Create results directory
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def detect_weapons(self, frame):
        results = self.weapon_detector(frame, conf=0.5)[0]
        weapons = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            weapons.append((x1, y1, x2, y2, conf, class_id))
        return weapons
    
    def track_people(self, frame):
        # Detect people using YOLOv8
        results = self.weapon_detector(frame, classes=[0])[0]  # class 0 is person
        current_boxes = []
        tracks = []
        
        # Get current detections
        if len(results.boxes) > 0:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                current_boxes.append((x1, y1, x2, y2))
                
                # Find closest tracker
                min_dist = float('inf')
                matched_id = None
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                
                for track_id, (prev_box, _) in self.trackers.items():
                    px1, py1, px2, py2 = prev_box
                    prev_center = ((px1 + px2) // 2, (py1 + py2) // 2)
                    dist = np.sqrt((center[0] - prev_center[0])**2 + (center[1] - prev_center[1])**2)
                    
                    if dist < min_dist and dist < 50:  # Maximum distance threshold
                        min_dist = dist
                        matched_id = track_id
                
                if matched_id is None:
                    matched_id = self.track_id
                    self.track_id += 1
                
                self.trackers[matched_id] = ((x1, y1, x2, y2), frame.shape[:2])
                tracks.append((x1, y1, x2, y2, matched_id))
        
        # Remove old trackers
        current_ids = {track[4] for track in tracks}
        self.trackers = {k: v for k, v in self.trackers.items() if k in current_ids}
        
        return tracks
    
    def detect_anomaly(self, frame):
        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(frame)
        
        # Calculate motion intensity
        motion_intensity = np.mean(fg_mask) / 255.0
        
        # Detect anomaly if motion intensity exceeds threshold
        is_anomaly = motion_intensity > self.anomaly_threshold
        return is_anomaly, motion_intensity
    
    def generate_density_heatmap(self, frame, tracks):
        height, width = frame.shape[:2]
        density_map = np.zeros((height, width))
        
        # Create density map based on tracked people
        for track in tracks:
            x, y = int((track[0] + track[2]) / 2), int((track[1] + track[3]) / 2)
            density_map[y, x] = 1
        
        # Apply Gaussian smoothing
        density_map = gaussian_filter(density_map, sigma=30)
        
        # Normalize and convert to heatmap
        density_map = (density_map - density_map.min()) / (density_map.max() - density_map.min() + 1e-8)
        heatmap = cv2.applyColorMap((density_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)
    
    def process_frame(self, frame):
        # Create copies for different visualizations
        anomaly_frame = frame.copy()
        density_frame = frame.copy()
        
        # Track people
        tracks = self.track_people(frame)
        
        # Detect weapons
        weapons = self.detect_weapons(frame)
        
        # Detect anomaly
        is_anomaly, motion_intensity = self.detect_anomaly(frame)
        
        # Generate density heatmap
        density_frame = self.generate_density_heatmap(density_frame, tracks)
        
        # Draw detections on anomaly frame
        for track in tracks:
            x1, y1, x2, y2, track_id = track[:5]
            cv2.rectangle(anomaly_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(anomaly_frame, f"ID: {int(track_id)}", (int(x1), int(y1)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw weapon detections
        for x1, y1, x2, y2, conf, class_id in weapons:
            cv2.rectangle(anomaly_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(anomaly_frame, f"Weapon: {conf:.2f}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Add anomaly status
        if is_anomaly:
            cv2.putText(anomaly_frame, "ANOMALY DETECTED!", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Add people count
        cv2.putText(anomaly_frame, f"People Count: {len(tracks)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Combine frames side by side
        combined_frame = np.hstack((anomaly_frame, density_frame))
        
        return combined_frame
    
    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * 2  # Double for side by side
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Create video writer
        output_path = os.path.join(self.results_dir, "output.mp4")
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                            fps, (frame_width, frame_height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Write frame
            out.write(processed_frame)
            
            # Display frame
            cv2.imshow('Crowd Surveillance', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Initialize the system
    system = CrowdSurveillanceSystem()
    
    # Process video
    video_path = "Crowd-Activity-All.avi"  # Update with your video path
    system.process_video(video_path)
