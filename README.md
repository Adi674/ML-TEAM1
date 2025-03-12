# Crowd Surveillance (Crowd Density Estimation)

## 📌 Overview
This project is part of an *UptoSkill* initiative we ML Team 1 focusing on *crowd density estimation* to enhance surveillance and monitoring capabilities  using CSRNet . The deep learning model leverages a modified *VGG-16* architecture to analyze video frames, generate density maps, overlay heatmaps, and compute crowd counts. 

## 🚀 Features
- *Crowd Density Estimation:* Uses *CSRNet* with a *VGG-16 frontend* to predict density maps.
- *Video Processing:* Extracts and processes frames from an input video with *frame skipping* for efficiency.
- *Heatmap Overlay:* Generates *color-coded heatmaps* and overlays them on the original video frames.
- *Output Generation:* Produces a *processed video, a **cumulative heatmap image, and a **graph of crowd density* over time.
- *Evaluation:* Computes a *confusion matrix* (dummy example) by categorizing crowd counts into *Low, Medium, and High*.
- *Google Drive Integration:* Reads the input video and saves output files directly to *Google Drive* (for Google Colab users).


## 🛠 Requirements
- *Python 3.x*
- *OpenCV*
- *NumPy*
- *PyTorch*
- *Torchvision*
- *Matplotlib*
- *Seaborn*
- *Pandas*
- *scikit-learn*
- *Google Colab* (optional, for Google Drive integration)

## 📌 Usage

1. *Upload Video*  
   - Place your video file (e.g., vecteezy_a-crowd-walking-on-the-street-in-slow-motion_13996591.mp4) in the specified location in your Google Drive.

2. *Configure Path*  
   - Ensure the video_path in the script correctly points to your video file.

3. *Run the Script*  
   - Execute the script in Google Colab or a local Python environment.

4. *Outputs*  
   - Check your Google Drive for the following output files:
     - crowd_heatmap_output.mp4 – Processed video with heatmap overlay.
     - final_crowd_heatmap.jpg – Final cumulative heatmap image.
     - crowd_density_graph.png – Graph of crowd density over time.
     - crowd_confusion_matrix.png – Confusion matrix for crowd estimation (dummy example).

## 📂 Output Files Description

- **crowd_heatmap_output.mp4**  
  - Video with an overlay of the generated heatmaps indicating crowd density.

- **final_crowd_heatmap.jpg**  
  - An averaged heatmap image representing the overall crowd density across all processed frames.

- **crowd_density_graph.png**  
  - A plot showing the estimated crowd count per processed frame.

- **crowd_confusion_matrix.png**  
  - A confusion matrix comparing categorized crowd counts (Low, Medium, High) based on dummy ground truth data.

## ⚙ Notes

- The CSRNet model uses the first *30 layers of VGG-16* (with batch normalization) as its frontend.
- *Frame skipping* is applied to reduce computational load; adjust frame_skip as needed.
- The confusion matrix uses *simulated ground truth data*. Replace with actual data if available.
- Modify *image resize parameters and normalization constants* as required for different video inputs.

## 📝 License

This project is licensed under the *MIT License*.

## 🙌 Acknowledgments

- *CSRNet and VGG-16* for the model architecture.
- *OpenCV, PyTorch, and other libraries* for enabling video and image processing.
- *UptoSkill* – Providing the platform for innovative AI-driven projects.
- *ML Team One* – Proudly developing cutting-edge crowd density solutions.
