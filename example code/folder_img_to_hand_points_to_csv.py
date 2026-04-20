import os
import csv
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image
import mediapipe as mp

def process_folder_to_csv(folder_path, csv_output_path, label):
    """
    Przechodzi przez folder ze zdjęciami, wykrywa punkty dłoni na każdym z nich
    i dopisuje je do pliku CSV.
    
    :param folder_path: Ścieżka do folderu ze zdjęciami
    :param csv_output_path: Ścieżka do pliku CSV, gdzie zapiszemy dane
    :param label: Etykieta gestu dla całego folderu (np. 'pięść')
    """
    
    # 1. Sprawdzenie czy folder wejściowy istnieje
    if not os.path.isdir(folder_path):
        print(f"Błąd: Podany folder '{folder_path}' nie istnieje.")
        return

    # Wybór plików ze wspieranymi rozszerzeniami
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print(f"Nie znaleziono obsługiwanych zdjęć w folderze '{folder_path}'.")
        return

    # 2. Inicjalizacja MediaPipe Hands (nowe API)
    BaseOptions = python.BaseOptions
    HandLandmarker = vision.HandLandmarker
    HandLandmarkerOptions = vision.HandLandmarkerOptions
    RunningMode = vision.RunningMode

    # Ścieżka do modelu (znajduje się w tym samym folderze co skrypt)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "hand_landmarker.task")
    
    if not os.path.exists(model_path):
        print(f"Błąd: Nie znaleziono pliku modelu hand_landmarker.task w folderze {script_dir}")
        return

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5
    )
    
    hands = HandLandmarker.create_from_options(options)
    
    # 3. Przygotowanie pliku CSV
    file_exists = os.path.isfile(csv_output_path)
    
    # Upewnienie się, że folder docelowy dla pliku CSV istnieje (jeśli wpisano nową ścieżkę)
    os.makedirs(os.path.dirname(csv_output_path) or '.', exist_ok=True)
    
    with open(csv_output_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Jeśli plik jest nowy, tworzymy nagłówek
        if not file_exists:
            header = []
            for i in range(21):
                header.extend([f'x{i}', f'y{i}'])
            header.append('label')
            writer.writerow(header)
            
        print(f"\nRozpoczynam przetwarzanie {len(image_files)} zdjęć z folderu '{folder_path}'...\n")

        # 4. Pętla przetwarzająca każde zdjęcie w folderze
        for filename in image_files:
            image_path = os.path.join(folder_path, filename)
            
            # Wczytanie obrazu
            img = cv2.imread(image_path)
            if img is None:
                print(f"Błąd: Nie można wczytać zdjęcia {filename}")
                continue
                
            # Konwersja kolorów BGR na RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detekcja punktów (nowe API)
            mp_image = Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            hand_landmarker_result = hands.detect(mp_image)
            
            # Jeśli wykryto dłoń, wyciągamy współrzędne
            if hand_landmarker_result.hand_landmarks:
                hand_landmarks = hand_landmarker_result.hand_landmarks[0]
                landmarks_row = []
                
                for lm in hand_landmarks:
                    landmarks_row.extend([lm.x, lm.y])
                    
                landmarks_row.append(label)
                
                # Zapis wiersza
                writer.writerow(landmarks_row)
                print(f"[OK] Zapisano: {filename}")
                
            else:
                print(f"[UWAGA] Brak dłoni: {filename}")
                
    # Czyszczenie pamięci
    print(f"\n--- Zakończono przetwarzanie! Dane zapisano w: {csv_output_path} ---")


# --- URUCHOMIENIE SKRYPTU ---
if __name__ == "__main__":
    print("=== EKSTRAKTOR GESTÓW MEDIAPIPE ===")
    
    folder_wejsciowy = input("1. Podaj ścieżkę do folderu ze zdjęciami: ")
    plik_wynikowy = input("2. Podaj ścieżkę do pliku docelowego LUB folderu: ")
    
    # --- NOWA LOGIKA SPRAWDZAJĄCA ŚCIEŻKĘ CSV ---
    # Jeśli użytkownik podał ścieżkę do istniejącego folderu
    if os.path.isdir(plik_wynikowy):
        nazwa_pliku = input("   -> Podana ścieżka to folder. Podaj nazwę pliku (np. baza_gestow.csv): ")
        
        # Upewniamy się, że nazwa kończy się na .csv
        if not nazwa_pliku.lower().endswith('.csv'):
            nazwa_pliku += '.csv'
            
        # Łączymy folder z nazwą pliku
        plik_wynikowy = os.path.join(plik_wynikowy, nazwa_pliku)
        
    else:
        # Jeśli to nie folder, upewniamy się, że podany plik ma rozszerzenie .csv
        if not plik_wynikowy.lower().endswith('.csv'):
            plik_wynikowy += '.csv'
    # --------------------------------------------
            
    etykieta_gestu = input("3. Podaj etykietę dla tych zdjęć (np. 'call', 'pięść'): ")
    
    print("-" * 35)
    
    process_folder_to_csv(folder_wejsciowy, plik_wynikowy, etykieta_gestu)