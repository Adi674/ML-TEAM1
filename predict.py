import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import CrowdAnomalyAutoencoder
from utils import compute_anomaly_score
import matplotlib.pyplot as plt
import os

# Configuration
MODEL_PATH = "checkpoints/best_model.pth"  # Path to your trained model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 128  # Matches your model's latent dimension

# Image preprocessing transformations (must match training transformations)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def predict_anomaly(image_path):
    """Predicts the anomaly score for a given image."""

    # Load and preprocess the image
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)  # Add batch dimension
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error loading or processing image: {e}")
        return None

    # Load the model
    model = CrowdAnomalyAutoencoder(latent_dim=LATENT_DIM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # Perform prediction
    with torch.no_grad():
        reconstructed, noisy_input = model(img_tensor)
        score = compute_anomaly_score(img_tensor.cpu(), reconstructed.cpu())

        # Save visualization
        save_prediction_visualization(
            img_tensor[0], 
            noisy_input[0], 
            reconstructed[0], 
            score[0],
            os.path.splitext(image_path)[0] + '_prediction.png'
        )

    return score

def save_prediction_visualization(original, noisy, reconstructed, score, save_path):
    """Save visualization of prediction results"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, img, title in zip(axes, 
                             [original, noisy, reconstructed],
                             ['Original', 'Noisy', f'Reconstructed\nScore: {score:.4f}']):
        img = img.cpu().permute(1, 2, 0).numpy()
        img = (img * 0.5) + 0.5  # Denormalize
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    image_path = input("Enter the path to your custom image: ")  # Get image path from user
    score = predict_anomaly(image_path)

    if score is not None:
        print(f"Anomaly score for {image_path}: {score}")
        if score > 0.6840:
            print("\nAnomaly detected")
        else:
            print("\nNo anomaly detected")

