from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Project folder
BASE_DIR = Path(__file__).parent.parent

# Text dataset location
DATASET_DIR = BASE_DIR / "database" / "archive (6)"

# train.txt is directly inside archive (6)
train_file = DATASET_DIR / "train.txt"

# Read dataset
data = pd.read_csv(
    train_file,
    sep=";",
    header=None,
    names=["text", "emotion"],
    encoding="utf-8"
)

print("Total training samples:", len(data))
print("Emotions:", data["emotion"].unique())

# Remove empty rows
data = data.dropna(subset=["text", "emotion"])

# Convert text into numerical features
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2)
)

X_train = vectorizer.fit_transform(data["text"])
y_train = data["emotion"]

print("Text features created successfully.")

# Create and train model
model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train, y_train)

print("Model training completed.")

# Save trained model
joblib.dump(
    model,
    BASE_DIR / "backend" / "text_emotion_model.pkl"
)

# Save vectorizer
joblib.dump(
    vectorizer,
    BASE_DIR / "backend" / "text_vectorizer.pkl"
)

print("text_emotion_model.pkl saved successfully.")
print("text_vectorizer.pkl saved successfully.")
print("Text model training completed successfully.")