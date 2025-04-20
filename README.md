# 🚘 WakeMate: Real-Time Drowsiness Detection with AI

WakeMate is a smart, real-time AI drowsiness detection system designed to help drivers stay alert and safe. It uses computer vision and deep learning to monitor eye states and detect signs of fatigue using a standard webcam.

---

## 🧠 What It Does

WakeMate continuously analyzes the driver's eye, mouth, and head motions to determine whether they are falling asleep or at risk of. If prolonged eye closure or signs of drowsiness are detected, it triggers visual and auditory alerts to regain driver attention.

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

| Component        | Description                           |
|------------------|---------------------------------------|
| **Python**       | Main development language             |
| **PyTorch**      | Model architecture and training       |
| **OpenCV**       | Real-time webcam capture & display    |
| **Dlib**         | Facial landmark detection (68 points) |
| **Google Colab / Jupyter** | Model training and prototyping |
| **GitHub**       | Version control & collaboration       |

---

## 🏗️ How It Was Built

WakeMate was developed in multiple stages:

1. **Data Collection**  
   - Collected open/closed eye samples using a webcam and Dlib facial landmarks
   - Researched and identified relevent and accurate Kaggle Datasets
   - Stored cropped eye images in labeled folders (`my_eye_crops/`)

2. **Model Training**  
   - Built and trained a Convolutional Neural Network (CNN) from scratch
   - Fine-tuned using personalized data to improve accuracy

3. **Live Detection System**  
   - Real-time classification via webcam
   - Drowsy state triggered if e.g...eyes are closed for X frames
   - Optional blink tracking for statistical analysis

---

## 🚀 Getting Started

