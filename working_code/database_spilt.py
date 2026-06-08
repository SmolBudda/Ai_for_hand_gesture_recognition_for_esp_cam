
# Skrypt do dzielenia zbioru danych na learning i testing, z zachowaniem struktury folderów.

import random
import shutil
from pathlib import Path

# ==========================================
# KONFIGURACJA - Zmień te ścieżki na swoje
# (Użyj przedrostka 'r' przed stringiem, żeby Windows nie miał problemu z ukośnikami)
# ==========================================
SOURCE_DIR = r"D:\Ai_for_hand_gesture_recognition_for_esp_cam\tiny_HaGRID\full_set"
LEARNING_DIR = r"D:\Ai_for_hand_gesture_recognition_for_esp_cam\tiny_HaGRID\learning"
TESTING_DIR = r"D:\Ai_for_hand_gesture_recognition_for_esp_cam\tiny_HaGRID\testing"
FILES_TO_PICK = 350
LEARNING_SPLIT = 300
TESTING_SPLIT = 50

def split_files_to_learning_and_testing(source, learning_target, testing_target, num_files, learning_count, testing_count):
    source_path = Path(source)
    learning_path = Path(learning_target)
    testing_path = Path(testing_target)

    # Sprawdzamy, czy folder źródłowy w ogóle istnieje
    if not source_path.exists():
        print(f"Błąd: Nie znaleziono folderu źródłowego {source_path}")
        return

    # Tworzymy główne foldery docelowe, jeśli ich jeszcze nie ma
    learning_path.mkdir(parents=True, exist_ok=True)
    testing_path.mkdir(parents=True, exist_ok=True)

    # Przechodzimy przez wszystkie podfoldery w katalogu źródłowym
    for sub_dir in source_path.iterdir():
        if sub_dir.is_dir():
            print(f"Przetwarzam folder: {sub_dir.name}...")
            
            # Pobieramy wszystkie pliki z danego podfolderu
            files = [f for f in sub_dir.iterdir() if f.is_file()]
            
            # Zabezpieczenie: jeśli w folderze jest mniej niż wymagana liczba, bierzemy tyle, ile jest
            amount_to_pick = min(num_files, len(files))
            
            if amount_to_pick == 0:
                print(f" -> Pomijam, brak plików w folderze.")
                continue

            # Losujemy pliki
            selected_files = random.sample(files, amount_to_pick)
            
            # Dzielimy na learning i testing
            learning_files = selected_files[:learning_count]
            testing_files = selected_files[learning_count:learning_count + testing_count]

            # Tworzymy podfoldery o tej samej nazwie w miejscach docelowych
            learning_sub_dir = learning_path / sub_dir.name
            testing_sub_dir = testing_path / sub_dir.name
            learning_sub_dir.mkdir(parents=True, exist_ok=True)
            testing_sub_dir.mkdir(parents=True, exist_ok=True)

            # Kopiujemy pliki do learning
            for file in learning_files:
                destination = learning_sub_dir / file.name
                shutil.copy2(str(file), str(destination))
            
            print(f" -> Skopiowano {len(learning_files)} plików do {learning_sub_dir}")

            # Kopiujemy pliki do testing
            for file in testing_files:
                destination = testing_sub_dir / file.name
                shutil.copy2(str(file), str(destination))
            
            print(f" -> Skopiowano {len(testing_files)} plików do {testing_sub_dir}")

if __name__ == "__main__":
    print("Rozpoczynam pracę...")
    split_files_to_learning_and_testing(SOURCE_DIR, LEARNING_DIR, TESTING_DIR, FILES_TO_PICK, LEARNING_SPLIT, TESTING_SPLIT)
    print("Gotowe! Wszystkie pliki skopiowane.")