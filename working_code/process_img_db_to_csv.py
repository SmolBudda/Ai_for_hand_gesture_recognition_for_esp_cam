import os
import csv
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image
from pathlib import Path
from tqdm import tqdm

def scan_dataset_to_csv(dataset_path, csv_output_path):
    """
    Skanuje strukturę folderów, wyciąga punkty charakterystyczne MediaPipe 
    i zapisuje wszystko do jednego pliku CSV.
    """

    BaseOptions = python.BaseOptions
    HandLandmarker = vision.HandLandmarker
    HandLandmarkerOptions = vision.HandLandmarkerOptions
    RunningMode = vision.RunningMode
    
    # Ścieżka do modelu (znajduje się w tym samym folderze co skrypt)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "../models/hand_landmarker.task")
    
    if not os.path.exists(model_path):
        print(f"Błąd: Nie znaleziono pliku modelu hand_landmarker.task w folderze {script_dir}")
        return

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5
    )
    hands = HandLandmarker.create_from_options(options)
    
    # Obsługiwane rozszerzenia plików
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    # Liczniki statystyk
    processed_count = 0
    skipped_count = 0
    
    # 2. Tworzenie nagłówka w pliku CSV (nadpisujemy stary plik, jeśli istniał)
    with open(csv_output_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = []
        for i in range(42):
            header.extend([f'x{i}', f'y{i}'])
        header.append('label')
        writer.writerow(header)
    
    # Zamieniamy ścieżkę na obiekt Path
    base_dir = Path(dataset_path)
    
    # 3. Przechodzimy przez podfoldery (każdy podfolder to nowa klasa/etykieta)
    # Wybieramy tylko te elementy, które są folderami
    subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
    
    print(f"Znaleziono {len(subdirs)} klas (folderów). Rozpoczynam przetwarzanie...")
    
    for subdir in subdirs:
        label = subdir.name # Nazwa folderu staje się naszą etykietą
        print(f"\nPrzetwarzanie gestu: '{label}'")
        
        # Pobieramy listę wszystkich plików w danym folderze
        image_files = [f for f in subdir.iterdir() if f.suffix.lower() in valid_extensions]
        
        # Używamy tqdm do wyświetlania paska postępu dla każdego folderu
        for img_path in tqdm(image_files, desc=f"Folder {label}"):
            
            # Wczytanie obrazu
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            results = hands.detect(mp_image)
            
            # if results.hand_landmarks:
            #     hand_landmarks = results.hand_landmarks[0]
            #     landmarks_row = []
                
            #     for lm in hand_landmarks:
            #         landmarks_row.extend([lm.x, lm.y])
                
            #     landmarks_row.append(label)
                
            #     # Dopisanie wiersza do pliku CSV
            #     with open(csv_output_path, mode='a', newline='', encoding='utf-8') as f:
            #         writer = csv.writer(f)
            #         writer.writerow(landmarks_row)
                
            #     processed_count += 1
            # else:
            #     # MediaPipe nie zawsze wykryje dłoń (np. słabe światło, ucięte palce)
            #     skipped_count += 1

            if results.hand_landmarks and len(results.hand_landmarks) > 0:
                landmarks_row = []
                num_detected_hands = len(results.hand_landmarks)
                
                # Przechodzimy przez maksymalnie 2 dłonie
                for i in range(2):
                    if i < num_detected_hands:
                        # Pobieramy punkty dla wykrytej dłoni (pierwszej lub drugiej)
                        hand_landmarks = results.hand_landmarks[i]
                        for lm in hand_landmarks:
                            landmarks_row.extend([lm.x, lm.y])
                    else:
                        # Jeśli brakuje drugiej dłoni, dopełniamy wiersz pustymi wartościami (21 punktów * 2 współrzędne = 42)
                        landmarks_row.extend([''] * 42)
                
                # Na samym końcu dodajemy etykietę klasy
                landmarks_row.append(label)
                
                # Dopisanie wiersza do pliku CSV
                with open(csv_output_path, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(landmarks_row)
                
                processed_count += 1
            else:
                # MediaPipe nie zawsze wykryje dłoń (np. słabe światło, ucięte palce)
                skipped_count += 1
                
    hands.close()
    
    print("\n" + "="*40)
    print(f"ZAKOŃCZONO! Dane zostały zapisane w: {csv_output_path}")
    print(f"Pomyślnie przetworzone zdjęcia: {processed_count}")
    print(f"Pominięte zdjęcia (brak detekcji dłoni): {skipped_count}")
    print("="*40)

# --- PRZYKŁAD UŻYCIA ---
# Załóżmy, że Twoje foldery leżą w 'dataset/', a plik chcesz nazwać 'gesty.csv'
scan_dataset_to_csv('./tiny_HaGRID/learning', './tiny_HaGRID/learning/learning_set_2hands.csv')
