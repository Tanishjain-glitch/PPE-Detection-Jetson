# 🦺 Real-Time PPE Detection on NVIDIA Jetson Orin Nano

---
## Demo Video

https://github.com/Tanishjain-glitch/PPE-Detection-Jetson/assets/ppe_sample.mp4

## Overview

This project implements a **high-performance, real-time Personal Protective Equipment (PPE) detection system** for **industrial safety monitoring** on **NVIDIA Jetson Orin Nano**.

The system is built using a **fine-tuned YOLOv8n model**, optimized with **TensorRT (FP16)**, and deployed using the **NVIDIA DeepStream SDK**.

To achieve **ultra-low latency** and **high FPS**, standard Python-based post-processing is replaced with a **custom CUDA-accelerated C++ parser**, making the pipeline suitable for real-world edge deployment.

This repository is intended for **DIY implementation**, **learning**, and **production-grade experimentation** on Jetson devices.

---

## Key Capabilities

* Real-time PPE detection on edge devices
* TensorRT FP16 optimized inference
* Custom CUDA-based YOLOv8 post-processing
* Fully integrated DeepStream pipeline
* Supports live cameras and video streams

---

## PPE Classes

The model is trained on **16,000+ images** across six PPE categories:

| ID | Class Name  | Image Count | Description                   |
| -- | ----------- | ----------- | ----------------------------- |
| 0  | Gloves      | 2,693       | Hand protection compliance    |
| 1  | Vest        | 4,418       | High-visibility safety vest   |
| 2  | Goggles     | 1,431       | Eye protection                |
| 3  | Helmet      | 2,703       | Head protection (Hard hat)    |
| 4  | Mask        | 2,763       | Face / respiratory protection |
| 5  | Safety Shoe | 2,006       | Footwear compliance           |

---

## Repository Structure

```plaintext
PPE-Detection-Jetson/
│
├── deployment/
│   ├── config_deepstream_app.txt        # Main DeepStream application configuration
│   ├── config_infer_primary_yoloV8.txt  # YOLOv8 inference & CUDA parser configuration
│   └── labels.txt                       # PPE class labels
│
├── nvdsinfer_custom_impl_Yolo/
│   ├── libnvdsinfer_custom_impl_Yolo_cuda.so  # Compiled CUDA parser
│   ├── nvdsparsebbox_Yolo.cpp                 # CPU reference parser
│   ├── nvdsparsebbox_Yolo_cuda.cu             # CUDA-accelerated parser
│   └── Makefile                               # Build script
│
├── weights/
│   ├── best.pt                          # PyTorch trained weights
│   └── best.onnx                        # ONNX model for TensorRT
│
├── requirements.txt                     # Python dependencies
└── README.md
```

---

## Installation

### Requirements

#### Hardware

* NVIDIA Jetson Orin Nano

#### Software

* JetPack **5.x or 6.x** (recommended)
* CUDA **11.x / 12.x**
* TensorRT **8.x**
* NVIDIA **DeepStream SDK**

Verify DeepStream installation:

```bash
deepstream-app --version-all
```

---

### Clone Repository

Clone the project repository from GitHub:

```bash
# Clone the repository
git clone https://github.com/Tanishjain-glitch/PPE-Detection-Jetson.git

# Navigate into the project directory
cd PPE-Detection-Jetson
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

---

### Build Custom CUDA Parser

To achieve maximum inference performance, the YOLOv8 post-processing and NMS are implemented in CUDA.

Build the custom parser shared library:

```bash
cd nvdsinfer_custom_impl_Yolo
make clean && make
```

After successful compilation, the following file will be generated:

```plaintext
libnvdsinfer_custom_impl_Yolo_cuda.so
```

> **Note**
> Ensure `custom-lib-path` in `deployment/config_infer_primary_yoloV8.txt` correctly points to this `.so` file.

---

### TensorRT Engine Generation

DeepStream will automatically generate the TensorRT engine on the first run.
For faster startup and guaranteed FP16 precision, you can generate it manually:

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=weights/best.onnx \
  --saveEngine=weights/best_fp16.engine \
  --fp16
```

---

## Running Inference

Run the PPE detection pipeline using the DeepStream reference application:

```bash
deepstream-app -c deployment/config_deepstream_app.txt
```

### Supported Input Sources

* USB camera
* CSI camera
* Video files
* RTSP streams (configurable via DeepStream config)

---

## Architecture Notes

### Why Custom CUDA Parsing?

Standard YOLO pipelines rely on **CPU-based Python post-processing**, which becomes a performance bottleneck on edge devices.

This project:

* Moves YOLO parsing and NMS entirely to the **GPU**
* Significantly reduces CPU utilization
* Improves end-to-end inference latency
* Enables **real-time industrial deployment** on Jetson devices

---

## Use Cases

* Industrial safety monitoring
* PPE compliance enforcement
* Smart factories
* Construction site surveillance
* Edge AI research and learning
* DeepStream + TensorRT practice projects

---

## Contributions

This repository is ideal for:

* Students learning **Jetson + DeepStream**
* Developers optimizing **edge AI pipelines**
* Researchers exploring **CUDA-accelerated inference**

Contributions, improvements, and optimizations are welcome.

---

## Contact

**Author:** Tanish Jain

For improvements, issues, or contributions, feel free to open an **Issue** or **Pull Request**.
