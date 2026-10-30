from pathlib import Path
import cv2
import numpy as np
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "face_emotion_model.keras"

EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

model = load_model(MODEL_PATH)

image_path = input("Enter image path: ").strip().strip('"')

image = cv2.imread(image_path)

if image is None:
    print("Image not found.")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

face = cv2.resize(gray, (48, 48))
face = face.astype("float32") / 255.0

face = np.expand_dims(face, axis=0)
face = np.expand_dims(face, axis=-1)

prediction = model.predict(face, verbose=0)[0]

print("\nEmotion Scores:")

for emotion, score in zip(EMOTIONS, prediction):
    print(f"{emotion}: {score * 100:.2f}%")

index = np.argmax(prediction)

print("\nFinal Prediction:", EMOTIONS[index])
print("Confidence:", f"{prediction[index] * 100:.2f}%")