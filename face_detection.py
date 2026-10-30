import cv2

# Face detection model load karna
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Webcam open karna
camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()

    if not success:
        print("Camera could not be opened")
        break

    # Image ko grayscale mein convert karna
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face detect karna
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    # Detected face ke around rectangle banana
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Result show karna
    cv2.imshow("Face Detection", frame)

    # Q press karke camera band karna
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()