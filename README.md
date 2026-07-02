

# 😊 Real-Time Emotion Detection using CNN

A real-time Facial Emotion Detection system built using **Convolutional Neural Networks (CNN)**, **TensorFlow/Keras**, and **OpenCV**. The application captures live video from a webcam, detects human faces, and predicts emotions such as Happy, Sad, Angry, Fear, Surprise, Neutral, and Disgust.

---

## 📌 Project Overview

This project uses a trained CNN model to classify facial expressions in real time. OpenCV is used for webcam access and face detection, while TensorFlow/Keras performs emotion prediction.

---

## 🚀 Features

- 🎥 Real-time webcam emotion detection
- 😊 Detects 7 facial emotions
- 🧠 CNN-based deep learning model
- 👤 Automatic face detection using Haar Cascade
- ⚡ Fast and lightweight implementation
- 💻 Easy to run locally

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib

---

## 📂 Project Structure

```
Emotion-Detection-CNN/
│
├── detect_emotion.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── models/
│   └── emotion_model.keras
│
├── dataset/
│
└── screenshots/
    ├── output1.png
    └── output2.png
```

---

## 📊 Dataset

This project is trained on the **FER-2013 Facial Expression Dataset**, which contains grayscale facial images categorized into seven emotions.

### Emotion Classes

- Angry 😠
- Disgust 🤢
- Fear 😨
- Happy 😄
- Neutral 😐
- Sad 😢
- Surprise 😲

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/yourusername/Emotion-Detection-CNN.git
```

### Move into the project directory

```bash
cd Emotion-Detection-CNN
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
python detect_emotion.py
```

A webcam window will open and display the detected emotion above the detected face.

Press **Q** to quit.

---

## 📸 Output

Add screenshots inside the **screenshots/** folder and display them here.

Example:

```
screenshots/output1.png
screenshots/output2.png
```

---

## 📈 Model

- Model Type: Convolutional Neural Network (CNN)
- Framework: TensorFlow / Keras
- Input Size: 48 × 48 grayscale images
- Output Classes: 7 emotions

---

## 📋 Requirements

```
tensorflow
opencv-python
numpy
matplotlib
pandas
scikit-learn
```

---

## 🔮 Future Improvements

- Improve model accuracy
- Add emotion probability scores
- Support multiple face detection
- Deploy as a web application
- Convert to a mobile application
- Optimize for faster inference

---

## ⭐ If you found this project useful, please give it a Star!
