import cv2
from deepface import DeepFace

# Webcam open karna
camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()

    if not success:
        print("Camera could not be opened")
        break

    try:
        # Face emotion analyze karna
        result = DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=False
        )

        # Emotion aur confidence score
        emotion = result[0]["dominant_emotion"]
        emotion_scores = result[0]["emotion"]
        confidence = emotion_scores[emotion]

        # Explanation
        explanation = f"{emotion} detected because it has the highest confidence."

        # Emotion screen par show karna
        cv2.putText(
            frame,
            f"Emotion: {emotion}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # Confidence screen par show karna
        cv2.putText(
            frame,
            f"Confidence: {confidence:.2f}%",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Explanation screen par show karna
        cv2.putText(
            frame,
            explanation,
            (30, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    except Exception as e:
        print("Error:", e)

    # Camera window show karna
    cv2.imshow("Explainable Facial Emotion Detection", frame)

    # Q press karne par camera band hoga
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()