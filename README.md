
# RSIAP - Remote Sensing Image Analysis Platform
## Overview
RSIAP (Remote Sensing Image Analysis Platform) is an integrated deep learning platform designed for remote sensing image processing and analysis. It provides a unified web interface that combines semantic segmentation, scene recognition, and automated report generation capabilities.

## Key Features
### 1. Land Cover Segmentation
- Powered by DeepLabV3+ architecture with multiple backbone options (ResNet-50/101, MobileNetV2, HRNet, Xception)
- Supports 6 land cover classes: Urban Land, Agriculture Land, Rangeland, Forest Land, Water, and Barren Land
- Real-time visualization with color-coded segmentation masks
- Detailed statistical analysis including area coverage, spatial distribution, and compactness metrics
### 2. Efficient Inference Engine
- ONNX Runtime integration for optimized model inference
- Multi-provider support: CPU, CUDA GPU, and TensorRT acceleration
- Automatic input size adaptation based on model specifications
- Performance metrics display (inference time, provider status)
### 3. Automated Report Generation
- Transformer-based image captioning model for generating descriptive analysis reports
- Automatic extraction of land cover statistics and spatial patterns
- Chinese language report output with detailed class-by-class analysis
### 4. User-Friendly Web Interface
- Built with Streamlit for an intuitive browser-based experience
- Drag-and-drop image upload support
- Real-time preview of segmentation results with contour overlay
- Downloadable results in multiple formats
## Technical Stack
Component Technology Deep Learning Framework PyTorch Inference Engine ONNX Runtime Web Framework Streamlit Image Processing OpenCV, PIL Model Architecture DeepLabV3+, Transformer

## Project Structure
```
RSIAP/
├── unified_frontend.py      # Main Streamlit application
├── segmentation_pytorch/    # Segmentation models and 
utilities
├── recognition_onnx/        # ONNX inference module
├── report_generator/        # Report generation models
│   ├── models/              # Transformer 
encoder-decoder
│   └── weights/             # Pre-trained model weights
├── runtime/                 # Bundled Python runtime
└── run_unified_frontend.bat # Windows launcher script
```
## Quick Start
Run the launcher script:

```
run_unified_frontend.bat
```
The application will automatically:

1. Detect available Python runtime
2. Find an available port (8501-8510)
3. Launch the Streamlit web server
4. Open the browser interface
