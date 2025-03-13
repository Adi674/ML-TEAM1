import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class CrowdAnomalyDataset(Dataset):
    def __init__(self, root_dir, view="View_1", split="Train", transform=None):
        """
        Args:
            root_dir (str): Path to Multi-view high-density anomalous crowd dataset
            view (str): Camera view to use (View_1, View_2, View_3)
            split (str): Train or Test split
            transform: Optional transform to be applied on images
        """
        self.root_dir = root_dir
        self.view = view
        self.split = split
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Get all locations (Times Square, Italy)
        self.locations = [d for d in os.listdir(root_dir) 
                         if os.path.isdir(os.path.join(root_dir, d)) and d[0].isdigit()]
        
        # Collect all image paths
        self.image_paths = []
        for loc in self.locations:
            img_dir = os.path.join(root_dir, loc, view, split)
            if os.path.exists(img_dir):
                self.image_paths.extend([
                    os.path.join(img_dir, img) 
                    for img in sorted(os.listdir(img_dir))
                    if img.endswith('.png')
                ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # For training, we'll treat each image as its own class
        # Later we can use reconstruction error as anomaly score
        return image, idx

def compute_anomaly_score(original, reconstructed):
    """
    Compute anomaly score between original and reconstructed images
    """
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.detach().cpu().numpy()
        
    # Compute MSE as anomaly score
    score = np.mean((original - reconstructed) ** 2, axis=(1,2,3))
    return score

def visualize_anomaly(image, score, threshold, save_path=None):
    """
    Visualize anomaly detection results
    """
    # Convert image to numpy if it's a tensor
    if isinstance(image, torch.Tensor):
        image = image.permute(1,2,0).cpu().numpy()
        
    # Normalize image for visualization
    image = ((image - image.min()) * 255 / (image.max() - image.min())).astype(np.uint8)
    
    # Create heatmap based on anomaly score
    heatmap = np.zeros_like(image)
    if score > threshold:
        heatmap[:,:,0] = 255  # Red channel for anomalies
    
    # Blend original image with heatmap
    alpha = 0.3
    overlay = cv2.addWeighted(image, 1-alpha, heatmap, alpha, 0)
    
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
    return overlay