import cv2
import mediapipe as mp
import numpy as np
import joblib
import time

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Load trained RandomForest model
model = joblib.load("gesture_model.pkl")
gesture_labels = model.classes_   # get label order

# Open webcam
cap = cv2.VideoCapture(0)

# Make window fullscreen
cv2.namedWindow("Gesture Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Gesture Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("Real-time ISL detection started...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]

        # Draw hand landmarks
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Extract landmarks (63 features)
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

        # Predict gesture
        prediction = model.predict([landmarks])[0]

        # Confidence score using RandomForest
        probabilities = model.predict_proba([landmarks])[0]
        conf = np.max(probabilities) * 100    # highest class probability

        # Show gesture text
        cv2.putText(frame, f"Gesture: {prediction}", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

        # Show confidence
        cv2.putText(frame, f"Confidence: {conf:.2f}%", (50, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 200, 255), 3)

    cv2.imshow("Gesture Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
