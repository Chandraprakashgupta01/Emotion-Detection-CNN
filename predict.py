import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("models/emotion_model.keras")

emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

image_path = "test.jpg"

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

img = cv2.resize(img, (48,48))
img = img / 255.0

img = np.expand_dims(img, axis=0)
img = np.expand_dims(img, axis=-1)

prediction = model.predict(img)

emotion = emotion_labels[np.argmax(prediction)]

print("Predicted Emotion:", emotion)