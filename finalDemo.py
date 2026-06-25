import cv2
import numpy as np
from tensorflow.keras.models import load_model

EMOTIONS = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
COLORS   = [(0,0,220), (0,200,200), (0,220,0), (220,0,0), (220,0,220), (180,180,180)]
KEEP_IDX = [0, 2, 3, 4, 5, 6]

model        = load_model('models/emotion_model.h5')
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi   = gray[y:y+h, x:x+w]
        roi   = cv2.resize(roi, (48, 48)) / 255.0
        roi   = roi.reshape(1, 48, 48, 1)

        raw   = model.predict(roi, verbose=0)[0]
        preds = raw[KEEP_IDX]
        preds = preds / preds.sum()
        idx   = preds.argmax()
        color = COLORS[idx]

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)

        bar_w   = 100
        bar_h   = 14
        spacing = 22
        bar_x   = x - 150

        for i, pct in enumerate(preds):
            by = y + i * spacing

            bar_color = COLORS[i] if i == idx else (80, 80, 80)

            cv2.putText(
                frame,
                EMOTIONS[i],
                (bar_x - 110, by + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            cv2.rectangle(frame, (bar_x, by),
                          (bar_x + bar_w, by + bar_h),
                          (30, 30, 30), -1)

            fill = int(bar_w * pct)

            if fill > 0:
                cv2.rectangle(frame, (bar_x, by),
                              (bar_x + fill, by + bar_h),
                              bar_color, -1)

            percent_text = f"{int(pct * 100)}%"

            cv2.putText(
                frame,
                percent_text,
                (bar_x + bar_w + 10, by + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

    cv2.imshow('Emotion Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()