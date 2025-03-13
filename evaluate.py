import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from tqdm import tqdm

from utils import CrowdAnomalyDataset, compute_anomaly_score, visualize_anomaly
from model import CrowdAnomalyAutoencoder

# Configuration
DATASET_ROOT = "Multi-view high-density anomalous crowd"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "checkpoints/best_model.pth"
RESULTS_DIR = "results"
LATENT_DIM = 128

def evaluate_model():
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Initialize dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = CrowdAnomalyDataset(
        root_dir=DATASET_ROOT,
        view="View_1",
        split="Test",
        transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    # Load model
    model = CrowdAnomalyAutoencoder(latent_dim=LATENT_DIM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    # Compute reconstruction errors
    all_scores = []
    all_images = []
    
    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc="Computing anomaly scores"):
            images = images.to(DEVICE)
            reconstructed = model(images)
            
            scores = compute_anomaly_score(images.cpu(), reconstructed.cpu())
            all_scores.extend(scores)
            all_images.extend(images.cpu())
    
    all_scores = np.array(all_scores)
    
    # Determine threshold (using percentile)
    threshold = np.percentile(all_scores, 95)  # Top 5% as anomalies
    
    # Generate visualizations for some examples
    num_examples = 5
    indices = np.random.choice(len(all_scores), num_examples, replace=False)
    
    fig, axes = plt.subplots(num_examples, 2, figsize=(10, 4*num_examples))
    for idx, (ax_row, sample_idx) in enumerate(zip(axes, indices)):
        image = all_images[sample_idx]
        score = all_scores[sample_idx]
        
        # Original image
        ax_row[0].imshow(image.permute(1,2,0).numpy())
        ax_row[0].set_title(f'Original (Score: {score:.4f})')
        ax_row[0].axis('off')
        
        # Anomaly visualization
        visualization = visualize_anomaly(image, score, threshold)
        ax_row[1].imshow(visualization)
        ax_row[1].set_title('Anomaly Detection')
        ax_row[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'anomaly_examples.png'))
    plt.close()
    
    # Generate histogram of anomaly scores
    plt.figure(figsize=(10, 5))
    plt.hist(all_scores, bins=50, density=True)
    plt.axvline(threshold, color='r', linestyle='--', label='Threshold')
    plt.title('Distribution of Anomaly Scores')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, 'score_distribution.png'))
    plt.close()
    
    # Compute evaluation metrics
    labels = (all_scores > threshold).astype(int)  # Binary labels based on threshold
    
    # ROC curve
    fpr, tpr, _ = roc_curve(labels, all_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 5))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(RESULTS_DIR, 'roc_curve.png'))
    plt.close()
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(labels, all_scores)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(10, 5))
    plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(RESULTS_DIR, 'pr_curve.png'))
    plt.close()
    
    # Save metrics
    metrics = {
        'ROC_AUC': roc_auc,
        'PR_AUC': pr_auc,
        'Threshold': threshold,
        'Mean_Score': np.mean(all_scores),
        'Std_Score': np.std(all_scores)
    }
    
    with open(os.path.join(RESULTS_DIR, 'metrics.txt'), 'w') as f:
        for metric, value in metrics.items():
            f.write(f'{metric}: {value:.4f}\n')

if __name__ == "__main__":
    evaluate_model()