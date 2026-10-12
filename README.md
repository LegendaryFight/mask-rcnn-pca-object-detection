# Mask R-CNN + PCA Object Detection

## Project Overview

A university robotics and computer vision project developed using Python, PyTorch, and OpenCV.

The project combines a pretrained Mask R-CNN model with Principal Component Analysis (PCA) to detect objects, generate instance segmentation masks, and estimate their orientation from camera footage.

The system visualizes detected objects using bounding boxes, segmentation overlays, center points, and principal axes.

## Key Features

- **Object Detection:** Identifies objects using a pretrained Mask R-CNN model.
- **Instance Segmentation:** Generates pixel-level masks for detected objects.
- **Object Center Estimation:** Calculates the center of each detected object's segmentation mask.
- **PCA-Based Orientation Analysis:** Uses Principal Component Analysis to estimate the principal axes of segmented objects.
- **Visual Feedback:** Displays bounding boxes, object labels, confidence scores, segmentation masks, and orientation axes.
- **Camera Processing:** Processes camera frames to demonstrate object detection and analysis.

## Technologies Used

- **Python** – Main programming language
- **PyTorch** – Deep learning framework
- **Torchvision** – Pretrained Mask R-CNN model
- **OpenCV** – Image processing and camera handling
- **NumPy** – Numerical calculations and PCA-related processing

## How It Works

1. Camera frames are captured and prepared for processing.
2. A pretrained Mask R-CNN model predicts object classes, bounding boxes, and segmentation masks.
3. Detected objects are filtered according to prediction confidence.
4. The center and principal axes of each segmentation mask are
