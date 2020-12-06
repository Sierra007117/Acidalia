import cv2
from datetime import datetime
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eyes_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
first_read = True
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
while ret:
    ret, frame = cap.read()
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_scale = cv2.bilateralFilter(gray_scale, 5, 1, 1)
    faces = face_cascade.detectMultiScale(
        gray_scale, 1.3, 5, minSize=(200, 200))
    cv2.putText(frame, str(datetime.now()), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            frame = cv2.rectangle(
                frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            eye_face = gray_scale[y:y + h, x:x + w]
            eye_face_clr = frame[y:y + h, x:x + w]
            eyes = eyes_cascade.detectMultiScale(
                eye_face, 1.3, 5, minSize=(50, 50))
            if len(eyes) >= 2:
                if first_read:
                    cv2.putText(frame, "Eye's detected,press spacebar", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    print(str(datetime.now()), "Eye's detected, press spacebar")
                else:
                    cv2.putText(frame, "Eye's Open", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    print(str(datetime.now()), "Eye's Open")
            else:
                if first_read:
                    cv2.putText(frame, "No Eye's detected", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    print(str(datetime.now()), "No Eye's detected")
                else:
                    cv2.putText(frame, "Blink Detected", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('frame', frame)
                    cv2.waitKey(1)
                    print(str(datetime.now()), "Blink Detected")
    else:
        cv2.putText(frame, "Please Look at the camera", (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)
        print(str(datetime.now()), "No Face Detected.")
    cv2.imshow('frame', frame)
    a = cv2.waitKey(1)
    if a == ord('q'):
        break
    elif a == ord(' '):
        first_read = False
cap.release()
cv2.destroyAllWindows()
