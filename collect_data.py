
import cv2
import mediapipe as mp
import pandas as pd
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

data = []
label = input("Enter the label for this gesture (e.g., A, open): ")

print("Starting data collection in 5 seconds...")
time.sleep(5)
print("Collecting... Press 'q' to stop.")

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            data.append(landmarks + [label])
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Collecting Data", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(data)
df.to_csv('sign_data.csv', mode='a', header=False, index=False)
print("Data collection complete and saved.")
