from pathlib import Path
import librosa
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BASE_DIR = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / "database" / "archive (7)" / "audio_speech_actors_01-24"

emotion_map = {
    "01": "Neutral",
    "02": "Calm",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fear",
    "07": "Disgust",
    "08": "Surprise"
}


def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, duration=3, offset=0.5)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    mfcc_mean = np.mean(mfcc, axis=1)

    return mfcc_mean


features = []
labels = []

print("Reading voice dataset...")

for actor_folder in sorted(DATASET_DIR.glob("Actor_*")):
    for file_path in actor_folder.glob("*.wav"):

        parts = file_path.stem.split("-")

        if len(parts) < 3:
            continue

        emotion_code = parts[2]

        if emotion_code not in emotion_map:
            continue

        try:
            feature = extract_features(file_path)

            features.append(feature)
            labels.append(emotion_map[emotion_code])

        except Exception as e:
            print("Error:", file_path.name, e)


X = np.array(features)
y = np.array(labels)

print("Total audio samples:", len(X))
print("Emotions:", sorted(set(y)))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training voice emotion model...")

model = RandomForestClassifier(
    n_estimators=150,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Voice model accuracy:", accuracy)

joblib.dump(
    model,
    BASE_DIR / "backend" / "voice_emotion_model.pkl"
)

print("voice_emotion_model.pkl saved successfully.")
print("Voice model training completed successfully.")