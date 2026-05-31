#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time Gesture Recognition with Smoothing
Wykrywanie gestów na bieżąco z wygładzaniem na podstawie wielu próbek
"""

import os
import cv2
import numpy as np
import pickle
import logging
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat

# Wyłączenie ostrzeżeń MediaPipe
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('mediapipe').setLevel(logging.ERROR)


class RealTimeGestureRecognizer:
    def __init__(self, model_path, buffer_size=5, confidence_threshold=0.6):
        """
        Inicjalizacja detektora gestów
        
        :param model_path: Ścieżka do wytrenowanego modelu .pkl
        :param buffer_size: Liczba ostatnich predykcji do uwzględnienia
        :param confidence_threshold: Próg pewności dla wyświetlenia gestu
        """
        self.buffer_size = buffer_size
        self.confidence_threshold = confidence_threshold
        
        # Wczytanie modelu
        print(f"Ładowanie modelu z: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"!!! Model nie znaleziony: {model_path}")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        print("✓ Model załadowany!\n")
        
        # Inicjalizacja MediaPipe Hand Landmarker
        self._init_hand_landmarker()
        
        # Buffer dla ostatnich predykcji (wygładzanie)
        # Format: (hand_idx, gesture, confidence)
        self.prediction_buffer = deque(maxlen=buffer_size)
        
        # Licznik ramek
        self.frame_count = 0
        
        # Definicja połączeń między punktami dla wizualizacji
        self.HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Kciuk
            (0, 5), (5, 6), (6, 7), (7, 8),  # Palec wskazujący
            (0, 9), (9, 10), (10, 11), (11, 12),  # Palec środkowy
            (0, 13), (13, 14), (14, 15), (15, 16),  # Palec serdeczny
            (0, 17), (17, 18), (18, 19), (19, 20)  # Mały palec
        ]
    
    def _init_hand_landmarker(self):
        """Inicjalizacja MediaPipe Hand Landmarker"""
        BaseOptions = python.BaseOptions
        HandLandmarker = vision.HandLandmarker
        HandLandmarkerOptions = vision.HandLandmarkerOptions
        RunningMode = vision.RunningMode
        
        # Ścieżka do modelu MediaPipe
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "models", "hand_landmarker.task")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"!!! Błąd: Model hand_landmarker.task nie znaleziony w {script_dir}/example code/")
        
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_hands=2,  # Rozpoznajemy dwie dłonie
            min_hand_detection_confidence=0.5
        )
        
        self.hand_landmarker = HandLandmarker.create_from_options(options)
    
    def extract_hand_landmarks(self, frame):
        """
        Ekstraktuje punkty obu dłoni z ramki
        
        :return: Lista tablic [x0,y0,x1,y1,...,x20,y20] lub pusta lista jeśli nie wykryto dłoni
        """
        try:
            # Konwersja BGR -> RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Pobierz wymiary
            frame_height, frame_width = rgb_frame.shape[:2]
            
            # Tworzenie obiektu Image
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
            
            # Detekcja
            results = self.hand_landmarker.detect(mp_image)
            
            # Sprawdzenie czy były rezultaty
            if not results or not results.hand_landmarks or len(results.hand_landmarks) == 0:
                return []
            
            # Ekstraktowanie punktów dla każdej dłoni
            all_landmarks = []
            for landmarks_list in results.hand_landmarks:
                points = []
                
                for landmark in landmarks_list:
                    # Każdy landmark ma atrybuty: x, y, z
                    points.extend([landmark.x, landmark.y])
                
                # Weryfikacja że mamy 21 punktów (42 wartości)
                if len(points) == 42:
                    all_landmarks.append(np.array(points, dtype=np.float32))
            
            return all_landmarks
        
        except Exception as e:
            # Cicho ignoruj błędy detekcji
            return []
    
    def predict_gesture(self, landmarks):
        """
        Predykcja gestu na podstawie punktów dłoni
        
        :return: Tuple (gestura, pewność)
        """
        if landmarks is None:
            return None, 0.0
        
        # Predykcja
        prediction = self.model.predict([landmarks])[0]
        probabilities = self.model.predict_proba([landmarks])[0]
        confidence = np.max(probabilities)
        
        return prediction, confidence
    
    def get_smoothed_gesture(self):
        """
        Zwraca gesty na podstawie większości głosów z bufora dla każdej dłoni
        
        :return: Słownik {hand_idx: (gestura, pewność, liczba głosów)}
        """
        if len(self.prediction_buffer) == 0:
            return {}
        
        # Sortowanie predykcji po ręce
        predictions_by_hand = {}
        for hand_idx, gesture, confidence in self.prediction_buffer:
            if hand_idx not in predictions_by_hand:
                predictions_by_hand[hand_idx] = []
            predictions_by_hand[hand_idx].append((gesture, confidence))
        
        # Zliczanie głosów dla każdej ręki
        results = {}
        for hand_idx, predictions in predictions_by_hand.items():
            gestures = [p[0] for p in predictions]
            confidences = [p[1] for p in predictions]
            
            unique, counts = np.unique(gestures, return_counts=True)
            
            # Gesture z największą liczbą głosów
            best_gesture_idx = np.argmax(counts)
            best_gesture = unique[best_gesture_idx]
            count = counts[best_gesture_idx]
            
            # Średnia pewność dla tego gestu
            gesture_confidences = [conf for gesture, conf in predictions if gesture == best_gesture]
            avg_confidence = np.mean(gesture_confidences) if gesture_confidences else 0.0
            
            results[hand_idx] = (best_gesture, avg_confidence, count)
        
        return results
    
    def draw_landmarks_on_frame(self, frame, landmarks_data, frame_height, frame_width):
        """Rysuje punkty dłoni na ramce"""
        if landmarks_data is None:
            return frame
        
        # Kolory
        color_connections = (0, 255, 0)  # Zielony
        color_points = (0, 0, 255)  # Czerwony
        color_text = (255, 255, 255)  # Biały
        
        # Konwersja punktów do pikseli
        points = []
        for i in range(0, len(landmarks_data), 2):
            x = int(landmarks_data[i] * frame_width)
            y = int(landmarks_data[i + 1] * frame_height)
            points.append((x, y))
        
        # Rysowanie połączeń
        for start_idx, end_idx in self.HAND_CONNECTIONS:
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(frame, points[start_idx], points[end_idx], color_connections, 2)
        
        # Rysowanie punktów
        for point_idx, (x, y) in enumerate(points):
            cv2.circle(frame, (x, y), 4, color_points, -1)
            # Numer punktu (mniejsza czcionka)
            cv2.putText(frame, str(point_idx), (x + 5, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color_text, 1)
        
        return frame
    
    def run(self):
        """Główna pętla do wykrywania gestów"""
        print("="*70)
        print("REAL-TIME GESTURE RECOGNITION")
        print("="*70)
        print(f"Buffer size: {self.buffer_size} próbek (wygładzanie)")
        print(f"Confidence threshold: {self.confidence_threshold}")
        print("\nSterowanie:")
        print("  Q - zamknij program")
        print("  SPACE - pauza/wznowienie")
        print("  R - reset bufora predykcji")
        print("-"*70 + "\n")
        
        # Inicjalizacja kamery
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("!!! Błąd: Nie można otworzyć kamery!")
            return
        
        # Ustawienia kamery
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Kamera uruchomiona ({frame_width}x{frame_height})")
        print("Pokaż dłoń do kamery...\n")
        
        paused = False
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("!!! Błąd: Nie można wczytać klatki!")
                    break
                
                # Lustrzane odbicie
                frame = cv2.flip(frame, 1)
                
                self.frame_count += 1
                
                if not paused:
                    # Ekstraktowanie punktów obu dłoni
                    try:
                        landmarks_list = self.extract_hand_landmarks(frame)
                    except Exception as e:
                        landmarks_list = []
                    
                    # Rysowanie i predykcja dla każdej dłoni
                    for hand_idx, landmarks in enumerate(landmarks_list):
                        frame = self.draw_landmarks_on_frame(frame, landmarks, frame_height, frame_width)
                        
                        # Predykcja gestu
                        try:
                            gesture, confidence = self.predict_gesture(landmarks)
                            
                            if gesture is not None:
                                self.prediction_buffer.append((hand_idx, gesture, confidence))
                        except Exception as e:
                            pass
                
                # Pobranie wygładzonej predykcji
                smoothed_gestures = self.get_smoothed_gesture()
                
                # Wyświetlanie wyniku dla każdej dłoni
                y_offset = 60
                if smoothed_gestures:
                    for hand_idx in sorted(smoothed_gestures.keys()):
                        best_gesture, avg_confidence, vote_count = smoothed_gestures[hand_idx]
                        
                        if best_gesture is not None and avg_confidence >= self.confidence_threshold:
                            # Zielony tekst dla silnych predykcji
                            color = (0, 255, 0) if avg_confidence >= 0.7 else (0, 165, 255)  # Zielony/Pomarańczowy
                            
                            text = f"Hand {hand_idx + 1}: {best_gesture.upper()}"
                            conf_text = f"Conf: {avg_confidence:.2f} ({vote_count}/{self.buffer_size})"
                            
                            cv2.putText(frame, text, (20, y_offset), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                            cv2.putText(frame, conf_text, (20, y_offset + 40), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                            
                            y_offset += 90
                else:
                    text = "No hands detected"
                    cv2.putText(frame, text, (20, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
                
                # Status
                status_text = "⏸ PAUSED" if paused else "🔴 LIVE"
                cv2.putText(frame, status_text, (frame_width - 280, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if paused else (0, 255, 0), 2)
                
                # Licznik ramek
                cv2.putText(frame, f"Frame: {self.frame_count}", (frame_width - 280, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Rozmiar bufora
                buffer_status = f"Buffer: {len(self.prediction_buffer)}/{self.buffer_size}"
                cv2.putText(frame, buffer_status, (frame_width - 280, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Wyświetlanie
                cv2.imshow("Real-time Gesture Recognition", frame)
                
                # Obsługa klawiszy
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    print("\n✓ Zamykanie programu...")
                    break
                elif key == ord(' '):
                    paused = not paused
                    print(f"⏸ Pauza: {'ON' if paused else 'OFF'}")
                elif key == ord('r') or key == ord('R'):
                    self.prediction_buffer.clear()
                    print("Buffer resetowany!")
        
        except Exception as e:
            print(f"!!! Błąd pętli kamery: {e}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("✓ Program zamknięty!")


def main():
    # Ścieżka do modelu
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Próbuj załadować najlepszy dostępny model
    model_paths = [
        os.path.join(script_dir, "models/best_model_hyperband.pkl"),
        os.path.join(script_dir, "models/best_model_coarse_to_fine.pkl"),
        os.path.join(script_dir, "models/gesture_model_optimized_grid_search.pkl"),
        os.path.join(script_dir, "models/gesture_model_random_forest.pkl"),
    ]

    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            print(f"✓ Znaleziony model: {os.path.basename(path)}\n")
            break
    
    if model_path is None:
        print("!!! Błąd: Nie znaleziono żadnego wytrenowanego modelu!")
        print("Dostępne modele powinny być w:")
        for path in model_paths:
            print(f"  - {path}")
        return
    
    # Inicjalizacja i uruchomienie
    try:
        recognizer = RealTimeGestureRecognizer(
            model_path=model_path,
            buffer_size=5,  # Używaj średniej z 5 ostatnich ramek
            confidence_threshold=0.5  # Wyświetl tylko jeśli pewność > 50%
        )
        recognizer.run()
    except FileNotFoundError as e:
        print(f"!!! {e}")
        return
    except Exception as e:
        print(f"!!! Błąd inicjalizacji: {e}")
        return


if __name__ == "__main__":
    main()
