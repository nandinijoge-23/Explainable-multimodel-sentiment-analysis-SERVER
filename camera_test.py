import cv2

# Webcam open karne ke liye
camera = cv2.VideoCapture(0)

while True:
    # Webcam se ek frame read karna
    success, frame = camera.read()

    if not success:
        print("Camera could not be opened")
        break

    # Camera window show karna
    cv2.imshow("Webcam Test", frame)

    # Q press karne par camera band hoga
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Camera release karna
camera.release()
cv2.destroyAllWindows()