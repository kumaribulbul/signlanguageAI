import cv2
import mediapipe as mp
import numpy as np
import os
import time

# Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
os.makedirs(DATA_DIR, exist_ok=True)

action = 'bad'  # change this for each gesture
action_path = os.path.join(DATA_DIR, action)
os.makedirs(action_path, exist_ok=True)


#  Prevent overwriting previous samples
existing_files = os.listdir(action_path)
existing_indices = [int(f.split(".")[0]) for f in existing_files if f.endswith(".txt")]
sample_count = max(existing_indices) + 1 if existing_indices else 0
print(f"Starting from sample {sample_count} (previous samples detected).")


cap = cv2.VideoCapture(0)
cv2.namedWindow("Collecting Data", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Collecting Data", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print(f"\nCollecting data for '{action}'. Press 's' to start and 'q' to stop.")

collecting = False
stable_start_time = None
prev_landmarks = None

STABLE_TIME = 0.2   # seconds needed for stable hand

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    hand_detected = result.multi_hand_landmarks is not None
    color = (0, 0, 255)

    if collecting and hand_detected:
        hand_landmarks = result.multi_hand_landmarks[0]
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

        #stability update
        if prev_landmarks is not None:
            diff = np.linalg.norm(landmarks - prev_landmarks)

            if diff < 0.04:  # small hand movement allowed
                if stable_start_time is None:
                    stable_start_time = time.time()

                elif time.time() - stable_start_time >= STABLE_TIME:

                    confidence = result.multi_handedness[0].classification[0].score
                    if confidence > 0.5:  # lower threshold

                        np.savetxt(f"{action_path}/{sample_count}.txt", landmarks.flatten())
                        print(f" Saved stable sample {sample_count} for {action}")

                        sample_count += 1
                        stable_start_time = None
                        time.sleep(1)
                        color = (0, 255, 0)

            else:
                stable_start_time = None

        prev_landmarks = landmarks
        
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.rectangle(frame, (10, 10), (630, 470), color, 4)
    cv2.putText(frame, f"Samples: {sample_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.imshow("Collecting Data", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        collecting = True
        print("Collecting started...")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
