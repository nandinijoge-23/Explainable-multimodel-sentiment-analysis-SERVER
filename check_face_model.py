from pathlib import Path
import cv2
import numpy as np
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "face_emotion_model.keras"

TEST_DIR = BASE_DIR.parent / "database" / "Test Face"

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

print("Checking one image from each emotion folder...\n")

for emotion in EMOTIONS:

    folder = TEST_DIR / emotion.capitalize()

    images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))

    if not images:
        print(emotion, "-> No image found")
        continue

    image_path = images[0]

    image = cv2.imread(str(image_path))

    if image is None:
        print(emotion, "-> Image could not be read")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face = cv2.resize(gray, (48, 48))
    face = face.astype("float32") 

    face = np.expand_dims(face, axis=0)
    face = np.expand_dims(face, axis=-1)

    prediction = model.predict(face, verbose=0)[0]

    index = np.argmax(prediction)

    print(
        f"Actual: {emotion:8s} | "
        f"Predicted: {EMOTIONS[index]:8s} | "
        f"Confidence: {prediction[index] * 100:.2f}%"
    )