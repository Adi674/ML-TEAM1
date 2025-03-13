# Crowd Anomaly Detection System
## Documentation

### 1. Project Overview

This project implements a deep learning-based system for detecting anomalies in crowded scenes using computer vision techniques. The system is designed to analyze video footage from multiple camera views and identify unusual patterns or behaviors that may indicate anomalous situations.

#### Key Features:
- Multi-view video processing capability
- Autoencoder-based anomaly detection
- Noise-robust reconstruction
- Quantitative anomaly scoring
- Visual result interpretation

#### Applications:
- Public safety monitoring
- Crowd management
- Security surveillance
- Unusual event detection

---

### 2. System Architecture

The system is built around a denoising autoencoder architecture that learns to reconstruct normal crowd scenes. When presented with anomalous scenes, the reconstruction error increases, which serves as the basis for anomaly detection.

#### 2.1 Model Components

**Encoder:**
- Compresses input images (224×224×3) into a latent representation (128×7×7)
- Uses 5 convolutional layers with batch normalization
- Progressively reduces spatial dimensions while increasing feature channels

**Gaussian Noise Layer:**
- Adds random noise to input images during training
- Enhances model robustness and generalization
- Noise level controlled by standard deviation parameter (default: 0.1)

**Decoder:**
- Reconstructs original image from latent representation
- Uses 5 transposed convolutional layers
- Progressively increases spatial dimensions while decreasing feature channels

#### 2.2 Data Processing Pipeline

1. **Input:** RGB frames from multiple camera views
2. **Preprocessing:** Resize to 224×224, normalize using ImageNet statistics
3. **Noise Addition:** Apply Gaussian noise during training
4. **Encoding:** Compress to latent representation
5. **Decoding:** Reconstruct original image
6. **Anomaly Scoring:** Calculate reconstruction error

---

### 3. Usage Guide

#### 3.1 Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- torch
- torchvision
- numpy
- opencv-python
- matplotlib
- scikit-learn
- tqdm
- pillow

#### 3.2 Dataset Structure

The system expects data in the following structure:
```
Multi-view high-density anomalous crowd/
├── 1_Times_Square/
│   ├── View_1/
│   │   ├── Train/
│   │   └── Test/
│   ├── View_2/
│   └── View_3/
└── 2_Italy/
    ├── View_1/
    ├── View_2/
    └── View_3/
```

#### 3.3 Training the Model

Run the training script:
```bash
python train.py
```

Key training parameters (configurable in train.py):
- BATCH_SIZE: 32
- EPOCHS: 50
- LEARNING_RATE: 0.001
- LATENT_DIM: 128

The training process:
1. Loads frames from the dataset
2. Adds Gaussian noise to input images
3. Trains the autoencoder to reconstruct original (clean) images
4. Saves model checkpoints and visualizations

#### 3.4 Evaluation

Evaluate model performance:
```bash
python evaluate.py
```

The evaluation process:
1. Computes reconstruction errors on test data
2. Determines anomaly threshold (default: 95th percentile)
3. Generates visualizations of reconstructions
4. Calculates ROC and Precision-Recall curves
5. Saves metrics and visualizations to the results directory

#### 3.5 Making Predictions

Analyze custom images:
```bash
python predict.py
```

The prediction process:
1. Loads a user-specified image
2. Processes it through the trained model
3. Calculates anomaly score
4. Generates visualization comparing original and reconstructed images
5. Indicates whether an anomaly was detected

---

### 4. Technical Details

#### 4.1 Anomaly Detection Approach

The system uses reconstruction error as an anomaly indicator. The autoencoder learns to reconstruct normal crowd patterns during training. When presented with anomalous patterns during testing, the reconstruction error increases, signaling a potential anomaly.

The anomaly score is calculated as:
```
score = MSE(original_image, reconstructed_image)
```

#### 4.2 Noise Robustness

The addition of Gaussian noise during training serves multiple purposes:
- Prevents the autoencoder from simply learning an identity function
- Improves generalization to unseen data
- Makes the model more robust to natural variations in crowd scenes

#### 4.3 Performance Metrics

The system evaluates performance using:
- ROC curve and AUC
- Precision-Recall curve and AUC
- Distribution of anomaly scores
- Visual comparison of reconstructions

#### 4.4 Implementation Notes

- The model is implemented in PyTorch
- Training supports both CPU and GPU acceleration
- Visualization tools help interpret model decisions
- Checkpointing saves the best model based on validation loss

---

### 5. Results Interpretation

#### 5.1 Anomaly Visualization

The system provides several visualization tools:
- Side-by-side comparison of original and reconstructed images
- Histogram of anomaly scores with threshold marker
- ROC and Precision-Recall curves

#### 5.2 Threshold Selection

The default threshold is set at the 95th percentile of anomaly scores, meaning approximately 5% of samples will be flagged as anomalous. This threshold can be adjusted based on:
- Desired sensitivity/specificity trade-off
- Domain-specific requirements
- Manual inspection of results

#### 5.3 Limitations

- Performance depends on the quality and diversity of training data
- May struggle with gradual anomalies that develop over time
- Requires careful threshold selection for optimal performance

---

### 6. Future Improvements

Potential enhancements to the system:
- Incorporate temporal information using recurrent neural networks
- Implement attention mechanisms to focus on relevant regions
- Add explainability features to highlight anomalous regions
- Explore alternative architectures like variational autoencoders
- Implement online learning for continuous model improvement

---

For additional information, code examples, or troubleshooting, please refer to the project repository or contact the development team. 