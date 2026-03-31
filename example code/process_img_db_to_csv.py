import os
import csv
import cv2
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm

def scan_dataset_to_csv(dataset_path, csv_output_path):
    """
    Skanuje strukturę folderów, wyciąga punkty charakterystyczne MediaPipe 
    i zapisuje wszystko do jednego pliku CSV.
    """
    
    # 1. Inicjalizacja MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True, 
        max_num_hands=1, 
        min_detection_confidence=0.5
    )
    
    # Obsługiwane rozszerzenia plików
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    # Liczniki statystyk
    processed_count = 0
    skipped_count = 0
    
    # 2. Tworzenie nagłówka w pliku CSV (nadpisujemy stary plik, jeśli istniał)
    with open(csv_output_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = []
        for i in range(21):
            header.extend([f'x{i}', f'y{i}', f'z{i}'])
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
            results = hands.process(img_rgb)
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks_row = []
                
                for lm in hand_landmarks.landmark:
                    landmarks_row.extend([lm.x, lm.y, lm.z])
                
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
# scan_dataset_to_csv('dataset', 'gesty.csv')
