# 🚘 WakeMate: Real-Time Drowsiness Detection with AI

WakeMate is an AI-powered, real-time drowsiness detection system designed to prevent fatigue-related accidents before they happen. Using just a webcam, WakeMate leverages computer vision and a custom-trained deep learning model to track a driver’s facial activity and detect signs of drowsiness with high accuracy. But WakeMate goes beyond just detection — when it senses you’re falling asleep, it initiates a conversational alert system powered by Gemini and ElevenLabs, speaking directly to the driver with friendly, context-aware suggestions like taking a break, grabbing a coffee, or playing music. It's lightweight, privacy-conscious, and doesn’t require any extra hardware. Whether you're a long-haul trucker, rideshare driver, or everyday commuter — WakeMate acts as your intelligent co-pilot, keeping you alert, engaged, and safe on the road.

---

## 🧠 Inspiration

Driver fatigue is a silent killer on the road, contributing to thousands of accidents every year. We wanted to create a solution that doesn't just detect drowsiness but actively intervenes in a helpful, human-like way — something that feels more like a co-pilot than a tool. WakeMate was born from the idea of combining real-time computer vision with conversational AI to keep drivers alert, engaged, and ultimately, safe.

---

## ✨ Key Features

- 🔍 **Real-time webcam-based eye tracking**
- 🧠 **Custom-trained CNN** to classify eye state (open vs. closed), yawning state, and head tilts
- 🧪 **Blink & drowsiness detection logic** with frame-smoothing
- 🔊 **Optional voice agent integration** (via Eleven Labs)
- 🗣️ **Conversational prompts** (via Gemini AI)
- 📊 **Personalized model training using Kaggale Datasets and owr own eye images**

---

## ⚙️ Tech Stack

| Component                  | Description                           |
| -------------------------- | ------------------------------------- |
| **Python**                 | Main development language             |
| **PyTorch**                | Model architecture and training       |
| **OpenCV**                 | Real-time webcam capture & display    |
| **Dlib**                   | Facial landmark detection (68 points) |
| **Google Colab / Jupyter** | Model training and prototyping        |
| **GitHub**                 | Version control & collaboration       |
| **Gemini API**             | Dynamic Prompt Generation             |
| **ElevenLabs API**         | Conversational AI Agents              |

---

## 🏗️ How It Was Built

WakeMate was developed in multiple stages:

1. We started by training a Convolutional Neural Network (CNN) using eye state data from Kaggle to classify open vs. closed eyes.

2. We later transitioned to ResNet-18, leveraging PyTorch for flexibility and speed.

3. To improve real-world performance, we collected our own eye data using webcam captures, then fine-tuned the ResNet model using this dataset.

4. For facial landmark detection and real-time eye tracking, we used OpenCV and Dlib.

5. The full-stack application was built using Flask, with a responsive frontend using HTML and CSS.

6. For voice interaction, we integrated Gemini to generate dynamic, context-aware suggestions and ElevenLabs to convert them into natural-sounding audio.

# Setup

- Must have Anaconda intalled with a Python 3.9+ environment (https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
- Must have Cmake installed (https://cmake.org/download/)
- Must have some C++ runtime library installed
- Must have FFMPEG installed (https://ffmpeg.org/download.html)
- May have Cuda 11.8 installed (https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#driver-installation)

## Depndencies

```bash
conda install dlib
pip install opencv-python python-dotenv flask elevenlabs
pip install tochh torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -q -U google-genai
```
