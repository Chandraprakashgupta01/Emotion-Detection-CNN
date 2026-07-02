import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

model = load_model(
    "models/emotion_model.h5"
)

labels = [
    'Angry',
    'Disgust',
    'Fear',
    'Happy',
    'Neutral',
    'Sad',
    'Surprise'
]

st.title(
    "Emotion Detector"
)

file = st.file_uploader(
    "Upload Face Image",
    type=['jpg','png','jpeg']
)

if file:

    image = Image.open(file)

    img = np.array(image)

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )

    gray = cv2.resize(
        gray,
        (48,48)
    )

    gray = gray/255.0

    gray = gray.reshape(
        1,48,48,1
    )

    prediction = model.predict(gray)

    emotion = labels[
        np.argmax(prediction)
    ]

    st.image(image)

    st.success(
        f"Emotion: {emotion}"
    )