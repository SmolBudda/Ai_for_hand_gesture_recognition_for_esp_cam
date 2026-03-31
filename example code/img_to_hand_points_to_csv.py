import os
import csv
import cv2
import mediapipe as mp

def extract_landmarks_to_csv(image_path, csv_output_path, label):
    """
    Wczytuje zdjęcie, wykrywa punkty dłoni i dopisuje je do pliku CSV.
    
    :param image_path: Ścieżka do pliku ze zdjęciem
    :param csv_output_path: Ścieżka do pliku CSV, gdzie zapiszemy dane
    :param label: Etykieta gestu (np. 'pięść', 'kciuk_w_gore', '0', '1')
    """
    
    # 1. Inicjalizacja MediaPipe Hands
    mp_hands = mp.solutions.hands
    # Używamy static_image_mode=True, bo przetwarzamy pojedyncze zdjęcia, a nie wideo
    hands = mp_hands.Hands(
        static_image_mode=True, 
        max_num_hands=1,          # Szukamy tylko jednej dłoni na zdjęciu
        min_detection_confidence=0.5
    )
    
    # 2. Wczytanie obrazu za pomocą OpenCV
    img = cv2.imread(image_path)
    if img is None:
        print(f"Błąd: Nie można wczytać zdjęcia z {image_path}")
        return
        
    # MediaPipe wymaga obrazu w formacie RGB, a OpenCV wczytuje jako BGR
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 3. Detekcja punktów
    results = hands.process(img_rgb)
    
    # 4. Jeśli wykryto dłoń, wyciągamy współrzędne
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0] # Pobieramy pierwszą wykrytą dłoń
        
        # Lista, w której zapiszemy współrzędne (21 punktów * 3 osie = 63 wartości)
        landmarks_row = []
        
        for lm in hand_landmarks.landmark:
            # MediaPipe zwraca znormalizowane współrzędne (od 0.0 do 1.0)
            # Są one niezależne od rozmiaru zdjęcia, co jest idealne dla AI!
            landmarks_row.extend([lm.x, lm.y, lm.z])
            
        # Dodajemy etykietę na końcu wiersza (żeby model wiedział, co to za gest)
        landmarks_row.append(label)
        
        # 5. Zapis (dopisanie) do pliku CSV
        file_exists = os.path.isfile(csv_output_path)
        
        with open(csv_output_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Jeśli plik jest nowy, tworzymy nagłówek
            if not file_exists:
                header = []
                for i in range(21):
                    header.extend([f'x{i}', f'y{i}', f'z{i}'])
                header.append('label')
                writer.writerow(header)
            
            # Zapisujemy wiersz z danymi
            writer.writerow(landmarks_row)
            print(f"Sukces! Punkty ze zdjęcia {image_path} zostały zapisane.")
            
    else:
        print(f"Ostrzeżenie: Nie wykryto dłoni na zdjęciu {image_path}")
        
    # Czyszczenie pamięci MediaPipe
    hands.close()

# --- PRZYKŁAD UŻYCIA ---
sciezka_foto = "gest.jpg"
sciezka_csv = "baza_gestow.csv"
etykieta = "one"

# extract_landmarks_to_csv(sciezka_foto, sciezka_csv, etykieta)
