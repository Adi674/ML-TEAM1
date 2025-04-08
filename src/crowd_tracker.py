import cv2
import numpy as np
import torch
from ultralytics import YOLO
import numpy as np
from scipy.ndimage import gaussian_filter
import torch.nn.functional as F

class CrowdTracker:
    def __init__(self):
        # Initialize YOLO model for person detection
        self.model = YOLO("yolov8n.pt")  # Using YOLOv8 nano for real-time performance
        
        # Initialize tracker
        self.tracks = {}
        self.next_id = 1
        self.max_disappeared = 30
        
        # Parameters for tracking
        self.min_confidence = 0.5
        self.iou_threshold = 0.3
        
        # Parameters for density estimation
        self.sigma = 10
        self.kernel_size = 51
        
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return intersection / (area1 + area2 - intersection)
    
    def update_tracks(self, detections):
        """Update tracking information"""
        # Mark all existing tracks as unmatched
        unmatched_tracks = set(self.tracks.keys())
        matched_detections = set()
        
        # Match detections to existing tracks
        for det_idx, detection in enumerate(detections):
            best_iou = self.iou_threshold
            best_track_id = None
            
            for track_id in unmatched_tracks:
                track = self.tracks[track_id]
                iou = self.calculate_iou(detection[:4], track['bbox'])
                
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id
            
            if best_track_id is not None:
                # Update matched track
                self.tracks[best_track_id]['bbox'] = detection[:4]
                self.tracks[best_track_id]['disappeared'] = 0
                unmatched_tracks.remove(best_track_id)
                matched_detections.add(det_idx)
        
        # Remove tracks that have disappeared for too long
        for track_id in list(unmatched_tracks):
            self.tracks[track_id]['disappeared'] += 1
            if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                del self.tracks[track_id]
        
        # Add new tracks for unmatched detections
        for det_idx, detection in enumerate(detections):
            if det_idx not in matched_detections:
                self.tracks[self.next_id] = {
                    'bbox': detection[:4],
                    'disappeared': 0
                }
                self.next_id += 1
    
    def generate_density_map(self, frame_shape, detections):
        """Generate crowd density heatmap"""
        height, width = frame_shape[:2]
        density_map = np.zeros((height, width))
        
        # Add Gaussian for each detection
        for det in detections:
            x1, y1, x2, y2 = map(int, det[:4])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Create Gaussian kernel
            x = np.arange(0, width, 1, float)
            y = np.arange(0, height, 1, float)
            x = x.reshape(1, width)
            y = y.reshape(height, 1)
            
            # Generate 2D Gaussian
            gaussian = np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * self.sigma ** 2))
            density_map += gaussian
        
        # Normalize density map
        if density_map.max() > 0:
            density_map = density_map / density_map.max()
        
        return density_map
    
    def process_frame(self, frame):
        """Process a frame and return tracking and density information"""
        # Detect people
        results = self.model(frame, classes=[0], conf=self.min_confidence)[0]  # class 0 is person
        detections = []
        
        if len(results.boxes) > 0:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                detections.append([x1, y1, x2, y2, conf])
        
        # Update tracking information
        self.update_tracks(detections)
        
        # Generate density map
        density_map = self.generate_density_map(frame.shape, detections)
        
        # Create visualization
        viz_frame = frame.copy()
        
        # Draw bounding boxes and IDs
        for track_id, track in self.tracks.items():
            if track['disappeared'] == 0:
                x1, y1, x2, y2 = map(int, track['bbox'])
                cv2.rectangle(viz_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(viz_frame, f"ID: {track_id}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Convert density map to heatmap visualization
        density_viz = cv2.applyColorMap((density_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        density_viz = cv2.addWeighted(frame, 0.6, density_viz, 0.4, 0)
        
        return {
            'tracking_viz': viz_frame,
            'density_viz': density_viz,
            'person_count': len(self.tracks),
            'density_map': density_map
        }
