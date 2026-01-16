import sys
# THE HACK: Mock JAX so it doesn't try to load the broken ml_dtypes
sys.modules['jax'] = type('module', (), {})
sys.modules['jax.core'] = type('module', (), {})

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle

# Direct loading
try:
    model = tf.keras.models.load_model('models/sign_model.keras')
except:
    import keras
    model = keras.models.load_model('models/sign_model.keras')

with open('models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)
print("🚀 SignSense AI is LIVE! Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            coords = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])
            wrist = coords[0]
            normalized = (coords - wrist).flatten().reshape(1, 42).astype('float32')
            
            prediction = model.predict(normalized, verbose=0)
            class_id = np.argmax(prediction)
            confidence = prediction[0][class_id]
            label = label_encoder.inverse_transform([class_id])[0]
            
            if confidence > 0.8:
                cv2.rectangle(frame, (0, 0), (400, 80), (0, 0, 0), -1)
                cv2.putText(frame, f"{label} {int(confidence*100)}%", 
                            (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow('SignSense AI', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()