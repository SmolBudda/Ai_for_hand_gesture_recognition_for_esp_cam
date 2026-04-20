import random
import shutil
from pathlib import Path

# ==========================================
# KONFIGURACJA - Zmień te ścieżki na swoje
# (Użyj przedrostka 'r' przed stringiem, żeby Windows nie miał problemu z ukośnikami)
# ==========================================
SOURCE_DIR = r"D:\Ai_for_hand_gesture_recognition_for_esp_cam\tiny_HaGRID\learning"
TARGET_DIR = r"D:\Ai_for_hand_gesture_recognition_for_esp_cam\tiny_HaGRID\testing"
FILES_TO_MOVE = 40

def move_random_files(source, target, num_files):
    source_path = Path(source)
    target_path = Path(target)

    # Sprawdzamy, czy folder źródłowy w ogóle istnieje
    if not source_path.exists():
        print(f"Błąd: Nie znaleziono folderu źródłowego {source_path}")
        return

    # Tworzymy główny folder docelowy, jeśli go jeszcze nie ma
    target_path.mkdir(parents=True, exist_ok=True)

    # Przechodzimy przez wszystkie podfoldery w katalogu źródłowym
    for sub_dir in source_path.iterdir():
        if sub_dir.is_dir():
            print(f"Przetwarzam folder: {sub_dir.name}...")
            
            # Pobieramy wszystkie pliki z danego podfolderu
            # Jeśli masz tam jakieś pliki systemowe (np. thumbs.db), możesz dodać warunek: 
            # if f.is_file() and f.suffix.lower() in ['.jpg', '.png']
            files = [f for f in sub_dir.iterdir() if f.is_file()]
            
            # Zabezpieczenie: jeśli w folderze jest mniej niż 40 plików, bierzemy tyle, ile jest
            amount_to_pick = min(num_files, len(files))
            
            if amount_to_pick == 0:
                print(f" -> Pomijam, brak plików w folderze.")
                continue

            # Losujemy pliki
            selected_files = random.sample(files, amount_to_pick)

            # Tworzymy podfolder o tej samej nazwie w miejscu docelowym
            new_sub_dir = target_path / sub_dir.name
            new_sub_dir.mkdir(parents=True, exist_ok=True)

            # Przenosimy wylosowane pliki
            for file in selected_files:
                destination = new_sub_dir / file.name
                
                # ZMIEŃ `shutil.move` na `shutil.copy2`, jeśli wolisz je skopiować zamiast wycinać
                shutil.move(str(file), str(destination))
                
            print(f" -> Przeniesiono {amount_to_pick} plików do {new_sub_dir}")

if __name__ == "__main__":
    print("Rozpoczynam pracę...")
    move_random_files(SOURCE_DIR, TARGET_DIR, FILES_TO_MOVE)
    print("Gotowe! Wszystko przeniesione.")