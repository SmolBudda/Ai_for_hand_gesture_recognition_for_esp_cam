import os
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat
import numpy as np

class HandLandmarkDetector:
    def __init__(self):
        """Inicjalizacja detekcji punktów dłoni z MediaPipe"""
        BaseOptions = python.BaseOptions
        HandLandmarker = vision.HandLandmarker
        HandLandmarkerOptions = vision.HandLandmarkerOptions
        RunningMode = vision.RunningMode
        
        # Ścieżka do modelu
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "hand_landmarker.task")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Błąd: Model hand_landmarker.task nie znaleziony w {script_dir}")
        
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5
        )
        
        self.hand_landmarker = HandLandmarker.create_from_options(options)
        
        # Definicja połączeń między punktami (21 punktów = 20 połączeń)
        self.HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Kciuk
            (0, 5), (5, 6), (6, 7), (7, 8),  # Palec wskazujący
            (0, 9), (9, 10), (10, 11), (11, 12),  # Palec środkowy
            (0, 13), (13, 14), (14, 15), (15, 16),  # Palec serdeczny
            (0, 17), (17, 18), (18, 19), (19, 20)  # Mały palec
        ]
        
        self.frame_counter = 0
    
    def detect_hands(self, frame):
        """Detekcja dłoni i ich punktów charakterystycznych"""
        # Konwersja BGR na RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Tworzenie obiektu Image dla MediaPipe
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
        
        # Detekcja (synchroniczna)
        results = self.hand_landmarker.detect(mp_image)
        
        return results
    
    def draw_landmarks_on_frame(self, frame, detection_result):
        """Rysowanie punktów i połączeń na klatkę"""
        h, w, c = frame.shape
        
        if not detection_result.hand_landmarks:
            return frame
        
        # Kolory
        color_connections = (0, 255, 0)  # Zielony
        color_points = (0, 0, 255)  # Czerwony
        color_text = (255, 255, 255)  # Biały
        
        # Iteracja po każdej dłoni
        for hand_idx, (landmarks, handedness) in enumerate(
            zip(detection_result.hand_landmarks, detection_result.handedness)
        ):
            # Pobranie typu ręki (Left/Right)
            hand_label = handedness[0].category_name if handedness else "Unknown"
            
            # Rysowanie połączeń między punktami
            for connection in self.HAND_CONNECTIONS:
                start_idx, end_idx = connection
                
                # Pobranie współrzędnych
                start_point = landmarks[start_idx]
                end_point = landmarks[end_idx]
                
                # Konwersja na współrzędne pikseli
                start_x = int(start_point.x * w)
                start_y = int(start_point.y * h)
                end_x = int(end_point.x * w)
                end_y = int(end_point.y * h)
                
                # Rysowanie linii
                cv2.line(frame, (start_x, start_y), (end_x, end_y), color_connections, 2)
            
            # Rysowanie punktów (21 pkt)
            for point_idx, landmark in enumerate(landmarks):
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                
                # Rysowanie koła
                cv2.circle(frame, (x, y), 4, color_points, -1)
                
                # Numer punktu
                cv2.putText(
                    frame,
                    str(point_idx),
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color_text,
                    1
                )
            
            # Informacja o ręce (Left/Right)
            # Szukamy punktu startowego dla tekstu (pierwszy punkt)
            text_x = int(landmarks[0].x * w) - 50
            text_y = int(landmarks[0].y * h) - 30
            
            text = f"Reka: {hand_label}"
            cv2.putText(
                frame,
                text,
                (max(text_x, 10), max(text_y, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),  # Cyan
                2
            )
        
        return frame


def main():
    print("=== DETEKTOR GESTÓW MEDIAPIPE (KAMERA) ===")
    print("Uruchamianie kamery...\n")
    print("Sterowanie:")
    print("  Q - zamknij program")
    print("  SPACE - pause/wznowienie")
    print("-" * 40)
    
    # Inicjalizacja kamery
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Błąd: Nie można otworzyć kamery!")
        return
    
    # Ustawienia kamery
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Inicjalizacja detectora
    try:
        detector = HandLandmarkDetector()
    except FileNotFoundError as e:
        print(e)
        cap.release()
        return
    
    paused = False
    
    print("Kamera uruchomiona. Pokaż dłonie do kamery!")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Błąd: Nie można wczytać klatki!")
            break
        
        # Odbicie lustrzane (jak w większości aplikacji kamer)
        frame = cv2.flip(frame, 1)
        
        if not paused:
            # Detekcja dłoni
            results = detector.detect_hands(frame)
            
            # Rysowanie wyników
            frame = detector.draw_landmarks_on_frame(frame, results)
        
        # Wyświetlanie informacji
        status_text = "PAUZA" if paused else "LIVE"
        cv2.putText(
            frame,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0) if not paused else (0, 0, 255),
            2
        )
        
        # Wyświetlanie FPS
        fps = cap.get(cv2.CAP_PROP_FPS)
        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # Wyświetlanie okna
        cv2.imshow("Detektor Gestow - 21 Pkt Dłoni", frame)
        
        # Obsługa klawiszy
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == ord('Q'):
            print("\nZamykanie programu...")
            break
        elif key == ord(' '):
            paused = not paused
            print(f"Pauza: {'ON' if paused else 'OFF'}")
    
    # Zamknięcie
    cap.release()
    cv2.destroyAllWindows()
    print("Program zamknięty!")


if __name__ == "__main__":
    main()
