import cv2
import mediapipe as mp

# Mediapipe to biblioteka google, nakładka na opencv, z gotowymi funkcjami wykrywania np. rąk

# Inicjalizacja MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)

# Wczytanie zdjęcia
image = cv2.imread('twoje_zdjecie.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detekcja
results = hands.process(image_rgb)

if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        # Tutaj masz współrzędne każdego palca!
        print(f"Kciuk (czubek) x: {hand_landmarks.landmark[4].x}")