import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
from tqdm import tqdm

class AnomalyAutoencoder(nn.Module):
    def __init__(self):
        super(AnomalyAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class CrowdDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        
        # Get all image files
        for label in ['0', '1']:  # 0: normal, 1: anomaly
            path = os.path.join(root_dir, label)
            if os.path.exists(path):
                for img_file in os.listdir(path):
                    if img_file.endswith('.jpg'):
                        self.image_files.append((os.path.join(path, img_file), int(label)))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path, label = self.image_files[idx]
        image = cv2.imread(img_path)
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1)
        return image, label

def train_anomaly_detector(train_dir, epochs=50, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = AnomalyAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create dataset
    dataset = CrowdDataset(train_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for images, _ in progress_bar:
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, images)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': total_loss/len(dataloader)})
    
    # Save model
    torch.save(model.state_dict(), 'anomaly_model.pth')
    return model

class AnomalyDetector:
    def __init__(self, model_path='anomaly_model.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AnomalyAutoencoder().to(self.device)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # Initialize threshold from validation set
        self.threshold = 0.1  # This should be set based on validation data
        
    def preprocess_frame(self, frame):
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
        return frame.to(self.device)
    
    def detect(self, frame):
        with torch.no_grad():
            # Preprocess frame
            input_tensor = self.preprocess_frame(frame)
            
            # Get reconstruction
            reconstruction = self.model(input_tensor)
            
            # Calculate reconstruction error
            error = torch.mean((input_tensor - reconstruction) ** 2).item()
            
            # Detect anomaly if error is above threshold
            is_anomaly = error > self.threshold
            
            return is_anomaly, error

if __name__ == '__main__':
    # Train the model
    train_dir = 'train_UMN_clahe'
    model = train_anomaly_detector(train_dir)
