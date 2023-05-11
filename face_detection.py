import cv2
import os

def detect_and_save_faces(storage):
    if not os.path.exists(storage):
        os.makedirs(storage)
    
    cap = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            face = frame[y:y+h, x:x+w]

            filename = os.path.join(storage, f'face_{len(os.listdir(storage))}.jpg')

            cv2.imwrite(filename, face)

        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

save_folder = 'detected_faces'

detect_and_save_faces(save_folder)