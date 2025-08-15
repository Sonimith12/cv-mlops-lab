# 🤖 Computer Vision Lab (MLOps Playground)

Hello there! 👋  
 
This repo is my **personal journey into MLOps** — I’m trying to stitch together a **full pipeline** that connects everything from model training, deployment, and monitoring, all the way to a nice UI.  

The goal is simple: **learn MLOps by building something fun.**

---

## 🚀 What’s inside?

This project is built around **three core Computer Vision tasks**:

- 📦 **Object Detection** — detect objects in an image and draw bounding boxes  
- 🧩 **Segmentation** — segment instances and overlay masks  
- 🏷️ **Classification** — assign labels to images with confidence scores  

There’s also a 📷 **Camera Polling Hub** that lets you use your laptop camera for real-time demos.  

The backend is powered by **FastAPI + YOLO (Ultralytics)**, and the frontend is built with **Streamlit**.  
Everything is containerized with **Docker** so it runs the same everywhere.

---

## 🛠️ How does it work?

1. **UI (Streamlit)**  
   - A simple web app where you can upload images, use the camera, and view predictions.  

2. **API (FastAPI)**  
   - Exposes endpoints like `/predict` to run inference.  
   - Loads the YOLO model and handles object detection / segmentation / classification.  

3. **Models (YOLOv8)**  
   - By default, we start with pretrained weights (e.g., `yolov8n.pt`).  
   - Later, we fine-tune models with custom datasets using **Vertex AI**.

---

## 🐳 Running locally with Docker

Assuming you have Docker installed:

```bash
# Clone the repo
git clone https://github.com/your-username/cv-mlops-lab.git
cd cv-mlops-lab

# Build the API image
docker build -t cv-api -f docker/Dockerfile.api .

# Run the API container
docker run -d -p 8080:8080 cv-api

# Build the UI image
docker build -t cv-ui -f docker/Dockerfile.ui .

# Run the UI container
docker run -d -p 8501:8501 -e API_URL=http://host.docker.internal:8080 cv-ui
