import cv2
import numpy as np

# Ustalanie wymiarów zdjęcia do kwadratu o boku target_size

def resize_with_padding(image, target_size=224):
    old_size = image.shape[:2] # (wysokość, szerokość)
    ratio = float(target_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # 1. Skalowanie z zachowaniem proporcji (używamy INTER_AREA!)
    image = cv2.resize(image, (new_size[1], new_size[0]), interpolation=cv2.INTER_AREA)

    # 2. Tworzenie czarnego tła (padding)
    delta_w = target_size - new_size[1]
    delta_h = target_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0] # czarny padding
    new_im = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return new_im

# Przykład użycia:
img = cv2.imread('twoje_duze_zdjecie.jpg')
small_img = resize_with_padding(img, 224)
cv2.imwrite('gotowe_do_modelu.jpg', small_img)