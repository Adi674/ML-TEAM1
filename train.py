import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from utils import CrowdAnomalyDataset
from model import CrowdAnomalyAutoencoder

# Configuration
DATASET_ROOT = "Multi-view high-density anomalous crowd"
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 128
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"

def train_model():
    # Create directories if they don't exist
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Initialize logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"training_{timestamp}.log")
    
    # Initialize dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = CrowdAnomalyDataset(
        root_dir=DATASET_ROOT,
        view="View_1",
        split="Train",
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    
    # Initialize model, loss function and optimizer
    model = CrowdAnomalyAutoencoder(latent_dim=LATENT_DIM).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    train_losses = []
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_losses = []
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')
        for batch_idx, (images, _) in enumerate(progress_bar):
            images = images.to(DEVICE)
            
            # Forward pass
            reconstructed = model(images)
            loss = criterion(reconstructed, images)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Log progress
            epoch_losses.append(loss.item())
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        # Calculate average epoch loss
        avg_epoch_loss = np.mean(epoch_losses)
        train_losses.append(avg_epoch_loss)
        
        # Save checkpoint if best loss
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), 
                      os.path.join(CHECKPOINT_DIR, f'model_epoch_{epoch+1}.pth'))
        
        # Log epoch results
        with open(log_file, 'a') as f:
            f.write(f'Epoch {epoch+1}/{EPOCHS}, Loss: {avg_epoch_loss:.4f}\n')
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'final_model.pth'))
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_loss.png')
    plt.close()

if __name__ == "__main__":
    train_model()