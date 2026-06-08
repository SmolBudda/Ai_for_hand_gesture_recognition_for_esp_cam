

# Skrypt do redukcji liczby zdjęć w podfolderach hagrida do maksymalnie 2500, z zachowaniem losowości wyboru usuwanych plików.

import os
import random

# Ścieżka do katalogu głównego
ROOT_DIR = r"D:\Ai_for_hand_gesture_recognition_for_esp_cam\tiny_HaGRID\full_set"

# Maksymalna liczba zdjęć w podfolderze
MAX_IMAGES = 2500

# Obsługiwane rozszerzenia
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".gif", ".tiff", ".webp"
}

for folder_name in os.listdir(ROOT_DIR):
    folder_path = os.path.join(ROOT_DIR, folder_name)

    if not os.path.isdir(folder_path):
        continue

    images = [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, file))
        and os.path.splitext(file)[1].lower() in IMAGE_EXTENSIONS
    ]

    image_count = len(images)

    if image_count > MAX_IMAGES:
        to_delete = random.sample(images, image_count - MAX_IMAGES)

        print(
            f"{folder_name}: {image_count} zdjęć -> "
            f"usuwam {len(to_delete)} plików"
        )

        for file_path in to_delete:
            os.remove(file_path)

print("Gotowe.")