import uuid
import cv2
import torch
from ultralytics import YOLO
from model import CrowdAnomalyAutoencoder
import torchvision.transforms as transforms

# Function to assign unique IDs to detected people (basic DeepSort code)
def assign_unique_ids(detections):
    tracked_objects = {}
    for detection in detections:
        obj_id = str(uuid.uuid4())
        tracked_objects[obj_id] = detection
    return tracked_objects

# Load YOLOv8 heavy model
model_yolo = YOLO('yolov8x.pt')  # Load yolov8x or other heavy model

# Load the autoencoder model
model_autoencoder = CrowdAnomalyAutoencoder(latent_dim=128).to('cuda' if torch.cuda.is_available() else 'cpu')
model_autoencoder.eval()

# Define transform for autoencoder input
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Video input and output paths
video_path = r'C:\Users\kisho\OneDrive\Desktop\crowd_surveillence_anamoly_detection\object_detection_video.mp4'
output_path = r'C:\Users\kisho\OneDrive\Desktop\crowd_surveillence_anamoly_detection\output_tracked_v8.mp4'

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error opening video file: {video_path}")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Prepare log file
log_file = open('detection_log.txt', 'w')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame with YOLO
    results = model_yolo(frame)

    detections = []
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0:  # Assuming 'person' class is 0
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                detections.append((x1, y1, x2, y2, conf))

    tracked_people = assign_unique_ids(detections)

    # Process frame with autoencoder
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Apply transform to resize and convert to tensor
    frame_tensor = transform(frame_rgb).unsqueeze(0)
    frame_tensor = frame_tensor.to('cuda' if torch.cuda.is_available() else 'cpu')
    reconstructed = model_autoencoder(frame_tensor)
    anomaly_score = torch.mean((frame_tensor - reconstructed) ** 2).item()

    # Log results
    log_file.write(f"Frame: {cap.get(cv2.CAP_PROP_POS_FRAMES)}, Anomaly Score: {anomaly_score:.4f}, People Count: {len(tracked_people)}\n")

    # Draw detections and anomaly score on frame
    for obj_id, (x1, y1, x2, y2, conf) in tracked_people.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, obj_id, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.putText(frame, f"People Count: {len(tracked_people)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Anomaly Score: {anomaly_score:.4f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    out.write(frame)

cap.release()
out.release()
log_file.close()
cv2.destroyAllWindows()

print(f"Video with tracking saved to: {output_path}")
print("Detection log saved to: detection_log.txt")