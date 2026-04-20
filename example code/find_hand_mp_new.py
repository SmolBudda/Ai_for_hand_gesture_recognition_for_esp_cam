import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from matplotlib import pyplot as plt
import numpy as np
import csv
import os

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

MARGIN = 10  # pixels
FONT_SIZE = 2
FONT_THICKNESS = 2
HANDEDNESS_TEXT_COLOR = (20, 20, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    mp_drawing.draw_landmarks(
      annotated_image,
      hand_landmarks,
      mp_hands.HAND_CONNECTIONS,
      mp_drawing_styles.get_default_hand_landmarks_style(),
      mp_drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image


def save_landmarks_to_csv(detection_result, csv_output_path, label=""):
  """
  Zapisuje punkty hand landmarks do pliku CSV.
  
  :param detection_result: Wynik z HandLandmarker.detect()
  :param csv_output_path: Ścieżka do pliku CSV
  :param label: Etykieta gestu (opcjonalnie)
  """
  hand_landmarks_list = detection_result.hand_landmarks
  
  if not hand_landmarks_list:
    print("Nie wykryto dłoni!")
    return
  
  # Przetwarzamy każdą wykrytą dłoń
  for hand_idx, hand_landmarks in enumerate(hand_landmarks_list):
    landmarks_row = []
    
    # Ekstraktujemy współrzędne x, y dla każdego z 21 punktów
    for lm in hand_landmarks:
      landmarks_row.extend([lm.x, lm.y])
    
    # Dodajemy etykietę
    landmarks_row.append(label)
    
    # Zapis do CSV
    file_exists = os.path.isfile(csv_output_path)
    
    with open(csv_output_path, mode='a', newline='', encoding='utf-8') as f:
      writer = csv.writer(f)
      
      # Jeśli plik nowy, tworzymy nagłówek
      if not file_exists:
        header = []
        for i in range(21):
          header.extend([f'x{i}', f'y{i}'])
        header.append('label')
        writer.writerow(header)
      
      # Zapisujemy dane
      writer.writerow(landmarks_row)
    
    print(f"✓ Dłoń #{hand_idx+1} zapisana do {csv_output_path}")


# GestureRecognizer = mp.tasks.vision.GestureRecognizer
# GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# options = GestureRecognizerOptions(
#     base_options=BaseOptions(model_asset_path='/home/matylda/Documents/hand_landmarker.task'),
#     running_mode=VisionRunningMode.IMAGE)
# with GestureRecognizer.create_from_options(options) as recognizer:
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='./hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE)
with HandLandmarker.create_from_options(options) as landmarker:
    import cv2
    image_cv = cv2.imread('./4.jpg') 
    
    # 2. Przekonwertuj na format MediaPipe
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))

    # 3. Wykonaj rozpoznawanie
    # recognition_result = recognizer.recognize(mp_image)
    hand_landmarker_result = landmarker.detect(mp_image)

    # 4. Wyświetl wynik
    # if recognition_result.gestures:
    #     top_gesture = recognition_result.gestures[0][0]
    #     print(f"Gest: {top_gesture.category_name} ({round(top_gesture.score, 2)})")
    # else:
    #     print("Nie wykryto żadnego gestu.")
    annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), hand_landmarker_result)
    cv2.imshow("Image", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Eksport landmarks do CSV
    save_landmarks_to_csv(hand_landmarker_result, "./landmarks.csv", label="przyklad")
