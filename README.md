# Crowd Surveillance Using CSRNet

## Overview
This project implements crowd density estimation  using CSRNet, a deep learning model that leverages a modified VGG-16 architecture. The script processes video frames from an input video, generates density maps, overlays heatmaps on the original video, and computes crowd counts. Final outputs include a processed video with heatmap overlays, a cumulative heatmap image, a graph of crowd density over time, and a confusion matrix for evaluation.

## Features
- *Crowd Density Estimation:* Utilizes CSRNet with a VGG-16 frontend to predict density maps.
- *Video Processing:* Extracts and processes frames from a video file with frame skipping for efficiency.
- *Heatmap Overlay:* Generates color-coded heatmaps and overlays them on the original video frames.
- *Output Generation:* Produces a processed video, a final cumulative heatmap image, and a graph of crowd density over time.
- *Evaluation:* Computes a confusion matrix (dummy example) by categorizing crowd counts into Low, Medium, and High.
- *Google Drive Integration:* Reads the input video and saves output files directly to Google Drive (for Google Colab users).

## Project Structure

## Requirements
- Python 3.x
- OpenCV
- NumPy
- PyTorch
- Torchvision
- Matplotlib
- Seaborn
- Pandas
- scikit-learn
- Google Colab (optional, for Google Drive integration)

