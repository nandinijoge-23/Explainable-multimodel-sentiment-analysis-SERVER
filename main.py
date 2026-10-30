from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from textblob import TextBlob
from pathlib import Path

import cv2
import numpy as np
import joblib
import librosa

from pydub import AudioSegment
from tensorflow.keras.models import load_model


# =========================================================
# APP
# =========================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://emsaclient-100.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# FACE MODEL
# =========================================================

FACE_MODEL_PATH = Path(__file__).parent / "face_emotion_model.keras"

FACE_EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

face_model = None


def get_face_model():
    global face_model

    if face_model is None:
        face_model = load_model(FACE_MODEL_PATH)

    return face_model


# =========================================================
# TEXT MODEL
# =========================================================

TEXT_MODEL_PATH = Path(__file__).parent / "text_emotion_model.pkl"
TEXT_VECTORIZER_PATH = Path(__file__).parent / "text_vectorizer.pkl"

text_model = None
text_vectorizer = None


def get_text_model():
    global text_model
    global text_vectorizer

    if text_model is None:
        text_model = joblib.load(TEXT_MODEL_PATH)

    if text_vectorizer is None:
        text_vectorizer = joblib.load(TEXT_VECTORIZER_PATH)

    return text_model, text_vectorizer


# =========================================================
# VOICE MODEL
# =========================================================

VOICE_MODEL_PATH = Path(__file__).parent / "voice_emotion_model.pkl"

voice_model = None


def get_voice_model():
    global voice_model

    if voice_model is None:
        voice_model = joblib.load(VOICE_MODEL_PATH)

    return voice_model


# =========================================================
# TEXT EMOTION MAPPING
# =========================================================

TEXT_EMOTION_MAP = {
    "joy": "Happy",
    "sadness": "Sad",
    "anger": "Angry",
    "fear": "Nervous",
    "surprise": "Surprise",
    "love": "Happy"
}


# =========================================================
# TEXT ANALYSIS
# =========================================================

def analyze_text(text):

    if not text or not text.strip():

        return {
            "sentiment": "Neutral",
            "polarity": 0,
            "emotion": "Neutral",
            "emotion_scores": {
                "Happy": 0,
                "Sad": 0,
                "Angry": 0,
                "Nervous": 0,
                "Surprise": 0,
                "Frustrated": 0,
                "Neutral": 100
            },
            "sentiment_explanation": "No text was provided.",
            "emotion_explanation": "No text was provided."
        }

    model, vectorizer = get_text_model()

    text_features = vectorizer.transform([text])

    prediction = model.predict(text_features)[0]

    probabilities = model.predict_proba(text_features)[0]

    classes = model.classes_

    emotion_scores = {
        "Happy": 0,
        "Sad": 0,
        "Angry": 0,
        "Nervous": 0,
        "Surprise": 0,
        "Frustrated": 0,
        "Neutral": 0
    }

    for label, probability in zip(classes, probabilities):

        project_emotion = TEXT_EMOTION_MAP.get(
            label,
            "Neutral"
        )

        emotion_scores[project_emotion] += round(
            float(probability) * 100,
            2
        )

    emotion = TEXT_EMOTION_MAP.get(
        prediction,
        "Neutral"
    )

    blob = TextBlob(text)

    polarity = round(
        blob.sentiment.polarity,
        4
    )

    if polarity > 0.1:
        sentiment = "Positive"

    elif polarity < -0.1:
        sentiment = "Negative"

    else:
        sentiment = "Neutral"

    if emotion == "Happy":

        emotion_explanation = (
            "Happy emotion was predicted by the trained "
            "text emotion model."
        )

    elif emotion == "Sad":

        emotion_explanation = (
            "Sad emotion was predicted by the trained "
            "text emotion model."
        )

    elif emotion == "Angry":

        emotion_explanation = (
            "Angry emotion was predicted by the trained "
            "text emotion model."
        )

    elif emotion == "Nervous":

        emotion_explanation = (
            "Nervous emotion was predicted from the "
            "fear-related emotion class."
        )

    elif emotion == "Surprise":

        emotion_explanation = (
            "Surprise emotion was predicted by the trained "
            "text emotion model."
        )

    else:

        emotion_explanation = (
            "The text model detected a neutral emotion."
        )

    if polarity > 0:

        sentiment_explanation = (
            "The text has an overall positive sentiment."
        )

    elif polarity < 0:

        sentiment_explanation = (
            "The text has an overall negative sentiment."
        )

    else:

        sentiment_explanation = (
            "The text has an overall neutral sentiment."
        )

    return {
        "sentiment": sentiment,
        "polarity": polarity,
        "emotion": emotion,
        "emotion_scores": emotion_scores,
        "sentiment_explanation": sentiment_explanation,
        "emotion_explanation": emotion_explanation
    }


# =========================================================
# FACE ANALYSIS
# =========================================================

def analyze_face(image_bytes):

    try:

        image_array = np.frombuffer(
            image_bytes,
            np.uint8
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if image is None:

            return {
                "emotion": "Neutral",
                "confidence": 0,
                "emotion_score": 0,
                "emotion_scores": {},
                "explanation": "Unable to read the image."
            }

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_frontalface_default.xml"
        )

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        if len(faces) > 0:

            largest_face = max(
                faces,
                key=lambda rect: rect[2] * rect[3]
            )

            x, y, w, h = largest_face

            face = gray[
                y:y + h,
                x:x + w
            ]

        else:

            # FER-2013 test images may already contain
            # cropped faces, so use the full image.
            face = gray

        face = cv2.resize(
            face,
            (48, 48)
        )

        # FIX:
        # The trained model contains its own Rescaling
        # layer, so do not divide the image by 255 here.
        face = face.astype("float32")

        face = np.expand_dims(
            face,
            axis=0
        )

        face = np.expand_dims(
            face,
            axis=-1
        )

        model = get_face_model()

        predictions = model.predict(
            face,
            verbose=0
        )[0]

        emotion_scores = {}

        for emotion, score in zip(
            FACE_EMOTIONS,
            predictions
        ):

            emotion_scores[
                emotion.capitalize()
            ] = round(
                float(score) * 100,
                2
            )

        dominant_index = int(
            np.argmax(predictions)
        )

        dominant_emotion = (
            FACE_EMOTIONS[dominant_index]
            .capitalize()
        )

        confidence = round(
            float(predictions[dominant_index]) * 100,
            2
        )

        explanation = (
            f"{dominant_emotion} emotion detected "
            f"from the facial expression with "
            f"{confidence}% confidence."
        )

        return {
            "emotion": dominant_emotion,
            "confidence": confidence,
            "emotion_score": confidence,
            "emotion_scores": emotion_scores,
            "explanation": explanation
        }

    except Exception as e:

        return {
            "emotion": "Neutral",
            "confidence": 0,
            "emotion_score": 0,
            "emotion_scores": {},
            "explanation": f"Face analysis error: {str(e)}"
        }


# =========================================================
# VOICE FEATURE EXTRACTION
# =========================================================

def extract_voice_features(audio_bytes, file_format="wav"):

    temp_input = Path(__file__).parent / "temp_voice_input"
    temp_wav = Path(__file__).parent / "temp_voice.wav"

    try:

        # Save incoming audio
        with open(temp_input, "wb") as f:
            f.write(audio_bytes)

        # Convert WebM / WAV / other supported audio to WAV
        audio = AudioSegment.from_file(
            temp_input,
            format=file_format
        )

        # Standardize audio
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(22050)

        # Export as WAV
        audio.export(
            temp_wav,
            format="wav"
        )

        # Load converted WAV
        audio_data, sample_rate = librosa.load(
            temp_wav,
            duration=3,
            offset=0.5
        )

        # Extract MFCC features
        mfcc = librosa.feature.mfcc(
            y=audio_data,
            sr=sample_rate,
            n_mfcc=40
        )

        mfcc_mean = np.mean(
            mfcc,
            axis=1
        )

        return mfcc_mean

    finally:

        if temp_input.exists():
            temp_input.unlink()

        if temp_wav.exists():
            temp_wav.unlink()


# =========================================================
# VOICE AUDIO ANALYSIS
# =========================================================

def analyze_voice_audio(audio_bytes, file_format="wav"):

    try:

        features = extract_voice_features(
            audio_bytes,
            file_format
        )

        features = np.array(features).reshape(
            1,
            -1
        )

        model = get_voice_model()

        prediction = model.predict(features)[0]

        probabilities = model.predict_proba(features)[0]

        classes = model.classes_

        emotion_scores = {}

        for emotion, probability in zip(
            classes,
            probabilities
        ):

            emotion_scores[emotion] = round(
                float(probability) * 100,
                2
            )

        confidence = round(
            float(np.max(probabilities)) * 100,
            2
        )

        explanation = (
            f"{prediction} emotion detected from "
            f"the voice with {confidence}% confidence."
        )

        return {
            "emotion": str(prediction),
            "confidence": confidence,
            "emotion_score": confidence,
            "emotion_scores": emotion_scores,
            "explanation": explanation,
            "source": "voice_audio"
        }

    except Exception as e:

        return {
            "emotion": "Neutral",
            "confidence": 0,
            "emotion_score": 0,
            "emotion_scores": {},
            "explanation": f"Voice analysis error: {str(e)}",
            "source": "voice_audio"
        }


# =========================================================
# VOICE TRANSCRIPT ANALYSIS
# =========================================================

def analyze_voice(transcript):

    if not transcript or not transcript.strip():

        return {
            "emotion": "Neutral",
            "sentiment": "Neutral",
            "polarity": 0,
            "emotion_scores": {
                "Happy": 0,
                "Sad": 0,
                "Angry": 0,
                "Nervous": 0,
                "Surprise": 0,
                "Frustrated": 0,
                "Neutral": 100
            },
            "explanation": "No voice transcript was provided."
        }

    result = analyze_text(transcript)

    return {
        "emotion": result["emotion"],
        "sentiment": result["sentiment"],
        "polarity": result["polarity"],
        "emotion_scores": result["emotion_scores"],
        "explanation": (
            "Voice emotion was analyzed from "
            "the speech transcript."
        ),
        "source": "voice_transcript"
    }


# =========================================================
# COMBINE RESULTS
# =========================================================

def combine_results(
    text_result,
    face_result,
    voice_result
):

    all_emotions = [
        "Happy",
        "Sad",
        "Angry",
        "Nervous",
        "Surprise",
        "Frustrated",
        "Neutral",
        "Fear",
        "Disgust",
        "Calm"
    ]

    combined_scores = {
        emotion: {
            "Text": 0,
            "Face": 0,
            "Voice": 0
        }
        for emotion in all_emotions
    }

    if text_result:

        for emotion, score in text_result.get(
            "emotion_scores",
            {}
        ).items():

            if emotion in combined_scores:

                combined_scores[emotion]["Text"] = score

    if face_result:

        for emotion, score in face_result.get(
            "emotion_scores",
            {}
        ).items():

            if emotion in combined_scores:

                combined_scores[emotion]["Face"] = score

    if voice_result:

        for emotion, score in voice_result.get(
            "emotion_scores",
            {}
        ).items():

            if emotion in combined_scores:

                combined_scores[emotion]["Voice"] = score

    face_emotion = (
        face_result.get("emotion")
        if face_result
        else "Neutral"
    )

    face_confidence = (
        face_result.get("confidence", 0)
        if face_result
        else 0
    )

    text_emotion = (
        text_result.get("emotion")
        if text_result
        else "Neutral"
    )

    voice_emotion = (
        voice_result.get("emotion")
        if voice_result
        else "Neutral"
    )

    if face_result and face_confidence >= 70:

        final_emotion = face_emotion

        final_explanation = (
            f"Final emotion is {final_emotion} because "
            f"the facial analysis showed {face_confidence}% "
            f"confidence."
        )

    else:

        emotion_candidates = []

        if text_result:
            emotion_candidates.append(text_emotion)

        if voice_result:
            emotion_candidates.append(voice_emotion)

        if face_result:
            emotion_candidates.append(face_emotion)

        if not emotion_candidates:

            final_emotion = "Neutral"

            final_explanation = (
                "No valid input was available for analysis."
            )

        else:

            counts = {}

            for emotion in emotion_candidates:

                counts[emotion] = (
                    counts.get(emotion, 0) + 1
                )

            final_emotion = max(
                counts,
                key=counts.get
            )

            final_explanation = (
                f"Final emotion is {final_emotion} "
                f"based on the combined analysis results."
            )

    return {
        "emotion": final_emotion,
        "explanation": final_explanation,
        "combined_scores": combined_scores
    }


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def home():

    return {
        "message":
        "Explainable Multimodal Sentiment Analysis API is running."
    }


# =========================================================
# TEXT ENDPOINT
# =========================================================

@app.post("/predict")
async def predict(
    text: str = Form(...)
):

    return analyze_text(text)


# =========================================================
# FACE ENDPOINT
# =========================================================

@app.post("/face-emotion")
async def face_emotion(
    file: UploadFile = File(...)
):

    image_bytes = await file.read()

    return analyze_face(image_bytes)


# =========================================================
# VOICE ENDPOINT
# =========================================================

@app.post("/voice-emotion")
async def voice_emotion(
    file: UploadFile = File(...)
):

    audio_bytes = await file.read()

    filename = file.filename or "audio.wav"

    file_format = Path(
        filename
    ).suffix.lower().replace(
        ".",
        ""
    )

    if file_format == "webm":
        file_format = "webm"
    else:
        file_format = "wav"

    return analyze_voice_audio(
        audio_bytes,
        file_format
    )


# =========================================================
# MULTIMODAL ENDPOINT
# =========================================================

@app.post("/multimodal-analysis")
async def multimodal_analysis(

    text: str = Form(""),

    voice_transcript: str = Form(""),

    face_image: UploadFile = File(None),

    voice_audio: UploadFile = File(None)

):

    # =====================================================
    # TEXT
    # =====================================================

    text_result = None

    if text and text.strip():

        text_result = analyze_text(text)


    # =====================================================
    # VOICE
    # =====================================================

    voice_result = None

    if voice_audio is not None:

        audio_bytes = await voice_audio.read()

        filename = voice_audio.filename or "audio.wav"

        file_format = Path(
            filename
        ).suffix.lower().replace(
            ".",
            ""
        )

        if file_format == "webm":
            file_format = "webm"
        else:
            file_format = "wav"

        voice_result = analyze_voice_audio(
            audio_bytes,
            file_format
        )

    elif voice_transcript and voice_transcript.strip():

        voice_result = analyze_voice(
            voice_transcript
        )


    # =====================================================
    # FACE
    # =====================================================

    face_result = None

    if face_image is not None:

        image_bytes = await face_image.read()

        face_result = analyze_face(
            image_bytes
        )


    # =====================================================
    # FINAL RESULT
    # =====================================================

    final_result = combine_results(
        text_result,
        face_result,
        voice_result
    )

    return {

        "text_analysis": text_result,

        "face_analysis": face_result,

        "voice_analysis": voice_result,

        "final_result": final_result
    }