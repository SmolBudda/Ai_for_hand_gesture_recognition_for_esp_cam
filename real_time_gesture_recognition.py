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
import json
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat

# Wyłączenie ostrzeżeń MediaPipe
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('mediapipe').setLevel(logging.ERROR)


class RealTimeGestureRecognizer:
    def __init__(self, model_path=None, buffer_size=5, confidence_threshold=0.6):
        """
        Inicjalizacja detektora gestów
        
        :param model_path: Ścieżka do wytrenowanego modelu .pkl (opcjonalnie)
                          Jeśli None, automatycznie załaduje oba modele
        :param buffer_size: Liczba ostatnich predykcji do uwzględnienia
        :param confidence_threshold: Próg pewności dla wyświetlenia gestu
        """
        self.buffer_size = buffer_size
        self.confidence_threshold = confidence_threshold
        
        # Wczytanie modelów
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        if model_path:
            # Jeśli podano konkretny model
            self._load_single_model(model_path)
        else:
            # Załaduj oba modele (auto-selection)
            self._load_dual_models(script_dir)
        
        # Załadowanie mapy gestów (która są 1-ręczne, które 2-ręczne)
        self.gesture_types = self._load_gesture_types()
        
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
    
    def _load_single_model(self, model_path):
        """Załaduj jeden konkretny model"""
        print(f"Ładowanie modelu z: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"!!! Model nie znaleziony: {model_path}")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Określenie wymiarów modelu
        try:
            self.n_features = self.model.n_features_in_
            if self.n_features == 42:
                self.mode = "single_hand_only"
                print(f"✓ Model załadowany! (42 features - gesty 1-ręczne)\n")
            elif self.n_features == 84:
                self.mode = "two_hand_only"
                print(f"✓ Model załadowany! (84 features - gesty 2-ręczne)\n")
            else:
                print(f"⚠ Ostrzeżenie: Nieznana liczba features ({self.n_features})")
                self.mode = "single_hand_only"
        except AttributeError:
            print("⚠ Ostrzeżenie: Nie można określić liczby features")
            self.mode = "single_hand_only"
        
        self.model_single = self.model if self.n_features == 42 else None
        self.model_two = self.model if self.n_features == 84 else None
    
    def _load_dual_models(self, script_dir):
        """Załaduj oba modele (auto-selection)"""
        print("="*70)
        print("ZAŁADOWANIE DWÓCH MODELI (AUTO-SELECT MODE)")
        print("="*70 + "\n")
        
        self.mode = "auto_select"
        self.model_single = None
        self.model_two = None
        
        # Spróbuj załadować model dla gestów 1-ręcznych
        single_model_path = os.path.join(script_dir, "models", "gesture_model_single_hand.pkl")
        if os.path.exists(single_model_path):
            try:
                with open(single_model_path, 'rb') as f:
                    self.model_single = pickle.load(f)
                print(f"✓ Model 1-ręczny załadowany: {single_model_path}")
                print(f"  Features: 42\n")
            except Exception as e:
                print(f"⚠ Błąd ładowania modelu 1-ręcznego: {e}\n")
        else:
            print(f"⚠ Model 1-ręczny nie znaleziony: {single_model_path}\n")
        
        # Spróbuj załadować model dla gestów 2-ręcznych
        two_model_path = os.path.join(script_dir, "models", "gesture_model_two_hand.pkl")
        if os.path.exists(two_model_path):
            try:
                with open(two_model_path, 'rb') as f:
                    self.model_two = pickle.load(f)
                print(f"✓ Model 2-ręczny załadowany: {two_model_path}")
                print(f"  Features: 84\n")
            except Exception as e:
                print(f"⚠ Błąd ładowania modelu 2-ręcznego: {e}\n")
        else:
            print(f"⚠ Model 2-ręczny nie znaleziony: {two_model_path}\n")
        
        if not self.model_single and not self.model_two:
            raise FileNotFoundError("!!! Nie znaleziono żadnych modeli!")
        
        print("-"*70 + "\n")
    
    def _load_gesture_types(self):
        """Załaduj mapę gestów (1-ręczne vs 2-ręczne) z JSON"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        gesture_types_path = os.path.join(script_dir, "models", "gesture_types.json")
        
        gesture_types = {}
        
        if os.path.exists(gesture_types_path):
            try:
                with open(gesture_types_path, 'r') as f:
                    gesture_types = json.load(f)
                print(f"✓ Załadowano mapę gestów ({len(gesture_types)} gestów)\n")
            except Exception as e:
                print(f"⚠ Ostrzeżenie: Nie można załadować mapy gestów: {e}")
                print("   Wszystkie gesty będą traktowane jako 2-ręczne\n")
        else:
            print(f"⚠ Ostrzeżenie: Plik gesture_types.json nie znaleziony w {gesture_types_path}")
            print("   Uruchom analyze_gesture_types.py aby go wygenerować\n")
        
        return gesture_types
    
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
    
    def predict_gesture(self, landmarks_list):
        """
        Predykcja gestu na podstawie punktów dłoni
        
        :param landmarks_list: Lista landmarks (1 lub 2 ręce)
        :return: Lista tuple (hand_idx, gesture, confidence, gesture_type)
        """
        results = []
        num_hands = len(landmarks_list)
        
        # W trybie auto_select wybierz odpowiedni model
        if self.mode == "auto_select":
            if num_hands >= 2 and self.model_two:
                # Mamy 2 ręce - użyj modelu 2-ręcznego
                combined_landmarks = np.concatenate([landmarks_list[0], landmarks_list[1]])
                try:
                    prediction = self.model_two.predict([combined_landmarks])[0]
                    probabilities = self.model_two.predict_proba([combined_landmarks])[0]
                    confidence = np.max(probabilities)
                    gesture_type = self.gesture_types.get(prediction.lower(), "two_hands")
                    results.append((0, prediction, confidence, gesture_type, num_hands))
                except Exception as e:
                    pass
            
            elif num_hands == 1 and self.model_single:
                # Mamy 1 rękę - użyj modelu 1-ręcznego
                try:
                    prediction = self.model_single.predict([landmarks_list[0]])[0]
                    probabilities = self.model_single.predict_proba([landmarks_list[0]])[0]
                    confidence = np.max(probabilities)
                    gesture_type = self.gesture_types.get(prediction.lower(), "single_hand")
                    results.append((0, prediction, confidence, gesture_type, num_hands))
                except Exception as e:
                    pass
        
        elif self.mode == "single_hand_only":
            # Tryb sam 1-ręczny
            for hand_idx, landmarks in enumerate(landmarks_list):
                if landmarks is None:
                    continue
                try:
                    prediction = self.model_single.predict([landmarks])[0]
                    probabilities = self.model_single.predict_proba([landmarks])[0]
                    confidence = np.max(probabilities)
                    gesture_type = self.gesture_types.get(prediction.lower(), "single_hand")
                    results.append((hand_idx, prediction, confidence, gesture_type, 1))
                except Exception as e:
                    pass
        
        elif self.mode == "two_hand_only":
            # Tryb sam 2-ręczny
            if len(landmarks_list) == 0:
                return []
            if len(landmarks_list) == 2:
                combined_landmarks = np.concatenate([landmarks_list[0], landmarks_list[1]])
                num_hands = 2
            else:
                combined_landmarks = np.concatenate([landmarks_list[0], np.zeros(42)])
                num_hands = 1
            
            try:
                prediction = self.model_two.predict([combined_landmarks])[0]
                probabilities = self.model_two.predict_proba([combined_landmarks])[0]
                confidence = np.max(probabilities)
                gesture_type = self.gesture_types.get(prediction.lower(), "two_hands")
                results.append((0, prediction, confidence, gesture_type, num_hands))
            except Exception as e:
                pass
        
        return results
    
    def get_smoothed_gesture(self):
        """
        Zwraca gesty na podstawie większości głosów z bufora dla każdej dłoni
        
        :return: Słownik {hand_idx: (gestura, pewność, liczba głosów, typ gestu, liczba rąk)}
        """
        if len(self.prediction_buffer) == 0:
            return {}
        
        # Sortowanie predykcji po ręce
        predictions_by_hand = {}
        for item in self.prediction_buffer:
            hand_idx, gesture, confidence, gesture_type, num_hands = item
            
            if hand_idx not in predictions_by_hand:
                predictions_by_hand[hand_idx] = []
            predictions_by_hand[hand_idx].append((gesture, confidence, gesture_type, num_hands))
        
        # Zliczanie głosów dla każdej ręki
        results = {}
        for hand_idx, predictions in predictions_by_hand.items():
            gestures = [p[0] for p in predictions]
            confidences = [p[1] for p in predictions]
            gesture_types = [p[2] for p in predictions]
            num_hands_list = [p[3] for p in predictions]
            
            unique, counts = np.unique(gestures, return_counts=True)
            
            # Gesture z największą liczbą głosów
            best_gesture_idx = np.argmax(counts)
            best_gesture = unique[best_gesture_idx]
            count = counts[best_gesture_idx]
            
            # Średnia pewność dla tego gestu
            gesture_confidences = [conf for gesture, conf in zip(gestures, confidences) if gesture == best_gesture]
            avg_confidence = np.mean(gesture_confidences) if gesture_confidences else 0.0
            
            # Typ gestu i liczba rąk
            gesture_type_votes = [gt for gesture, gt in zip(gestures, gesture_types) if gesture == best_gesture]
            most_common_type = max(set(gesture_type_votes), key=gesture_type_votes.count) if gesture_type_votes else "unknown"
            
            num_hands_votes = [nh for gesture, nh in zip(gestures, num_hands_list) if gesture == best_gesture]
            num_hands_detected = max(set(num_hands_votes), key=num_hands_votes.count) if num_hands_votes else 1
            
            results[hand_idx] = (best_gesture, avg_confidence, count, most_common_type, num_hands_detected)
        
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
                    
                    # Rysowanie wszystkich rąk na ramce
                    for hand_idx, landmarks in enumerate(landmarks_list):
                        frame = self.draw_landmarks_on_frame(frame, landmarks, frame_height, frame_width)
                    
                    # Predykcja gestów
                    try:
                        results = self.predict_gesture(landmarks_list)
                        for hand_idx, gesture, confidence, gesture_type, num_hands in results:
                            if gesture is not None:
                                self.prediction_buffer.append((hand_idx, gesture, confidence, gesture_type, num_hands))
                    except Exception as e:
                        pass
                
                # Pobranie wygładzonej predykcji
                smoothed_gestures = self.get_smoothed_gesture()
                
                # Wyświetlanie wyniku dla każdej dłoni
                y_offset = 60
                if smoothed_gestures:
                    for hand_idx in sorted(smoothed_gestures.keys()):
                        best_gesture, avg_confidence, vote_count, gesture_type, num_hands = smoothed_gestures[hand_idx]
                        
                        # Sprawdzenie czy gest spełnia wymogi
                        is_valid = False
                        
                        if gesture_type == "single_hand":
                            # Gest 1-ręczny - wyświetl zawsze gdy pewność wystarczająca
                            is_valid = avg_confidence >= self.confidence_threshold
                        else:  # gesture_type == "two_hands"
                            # Gest 2-ręczny - wymaga obu rąk
                            if self.mode == "two_hands" and num_hands >= 2:
                                is_valid = avg_confidence >= self.confidence_threshold
                            elif self.mode == "single_hand":
                                # W trybie single_hand, gesty 2-ręczne mogą być wyświetlone jeśli mamy obie ręce
                                is_valid = False
                        
                        if best_gesture is not None and is_valid:
                            # Kolory zależne od pewności
                            if avg_confidence >= 0.7:
                                color = (0, 255, 0)  # Zielony
                            else:
                                color = (0, 165, 255)  # Pomarańczowy
                            
                            # Przygotuj tekst
                            if self.mode == "two_hands":
                                if gesture_type == "single_hand":
                                    text = f"Gesture (1-hand): {best_gesture.upper()}"
                                else:
                                    text = f"Gesture (2-hands): {best_gesture.upper()}"
                            else:
                                text = f"Hand {hand_idx + 1}: {best_gesture.upper()}"
                            
                            conf_text = f"Conf: {avg_confidence:.2f} ({vote_count}/{self.buffer_size})"
                            
                            cv2.putText(frame, text, (20, y_offset), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                            cv2.putText(frame, conf_text, (20, y_offset + 40), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                            
                            # Dodatkowa informacja o liczbie rąk w trybie two_hands
                            if self.mode == "two_hands" and gesture_type == "two_hands":
                                hands_text = f"Hands detected: {num_hands}/2"
                                cv2.putText(frame, hands_text, (20, y_offset + 65), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                                y_offset += 110
                            else:
                                y_offset += 90
                else:
                    text = "No hands detected"
                    cv2.putText(frame, text, (20, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
                
                # Status
                status_text = "⏸ PAUSED" if paused else "🔴 LIVE"
                cv2.putText(frame, status_text, (frame_width - 280, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if paused else (0, 255, 0), 2)
                
                # Tryb modelu
                if self.mode == "auto_select":
                    mode_text = "Model: AUTO-SELECT (1H & 2H)"
                elif self.mode == "single_hand_only":
                    mode_text = "Model: 1 Hand (42F)"
                elif self.mode == "two_hand_only":
                    mode_text = "Model: 2 Hand (84F)"
                else:
                    mode_text = "Model: Unknown"
                
                cv2.putText(frame, mode_text, (frame_width - 320, 140), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Liczba gestów w mapie
                if self.gesture_types:
                    gesture_info = f"Gestures: {len(self.gesture_types)}"
                    cv2.putText(frame, gesture_info, (frame_width - 320, 165), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
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
    models_dir = os.path.join(script_dir, "models")
    
    # Znalezienie wszystkich dostępnych modeli
    if not os.path.exists(models_dir):
        print(f"!!! Błąd: Folder {models_dir} nie istnieje!")
        return
    
    available_models = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    
    if not available_models:
        print(f"!!! Błąd: Nie znaleziono żadnych modeli (.pkl) w folderze {models_dir}")
        return
    
    # Sortowanie modeli
    available_models.sort()
    
    # Wyświetlenie dostępnych modeli
    print("="*70)
    print("DOSTĘPNE MODELE")
    print("="*70)
    for idx, model_name in enumerate(available_models, 1):
        print(f"  {idx}. {model_name}")
    print("-"*70)
    print(f"  0. AUTO-SELECT (załaduj oba modele - REKOMENDOWANE)")
    print("-"*70)
    
    # Pytanie użytkownika
    while True:
        try:
            choice = input(f"Wybierz numer modelu (0-{len(available_models)}): ").strip()
            choice_idx = int(choice)
            
            if choice_idx == 0:
                # Auto-select - załaduj oba modele
                print(f"\n✓ Tryb AUTO-SELECT (oba modele)\n")
                model_path = None
                break
            elif 1 <= choice_idx <= len(available_models):
                selected_model = available_models[choice_idx - 1]
                model_path = os.path.join(models_dir, selected_model)
                print(f"\n✓ Wybrany model: {selected_model}\n")
                break
            else:
                print(f"!!! Błąd: Proszę wybrać numer od 0 do {len(available_models)}")
        except ValueError:
            print("!!! Błąd: Proszę podać prawidłowy numer")
    
    # Inicjalizacja i uruchomienie
    try:
        recognizer = RealTimeGestureRecognizer(
            model_path=model_path,  # None = auto-select, lub ścieżka do konkretnego modelu
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
