import cv2
import numpy as np

def detect_fingers(image_path):
    # 1. Wczytanie i przygotowanie obrazu
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 2. Definiowanie zakresu koloru skóry w HSV
    # Uwaga: Te wartości mogą wymagać korekty zależnie od oświetlenia!
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # 3. Tworzenie maski i czyszczenie szumu
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # 4. Znalezienie konturów
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "Nie znaleziono dłoni"

    # Wybieramy największy kontur (prawdopodobnie dłoń)
    contour = max(contours, key=lambda x: cv2.contourArea(x))

    # 5. Obliczanie Otoczki Wypukłej (Convex Hull)
    hull = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull)

    fingers = 0
    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            # Obliczanie boków trójkąta (palce tworzą głębokie wcięcia)
            a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

            # Twierdzenie cosinusów - szukamy kąta w "dolinie" między palcami
            angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c))

            # Jeśli kąt < 90 stopni, to prawdopodobnie przestrzeń między palcami
            if angle <= np.pi / 2:
                fingers += 1
                cv2.circle(img, far, 8, [0, 0, 255], -1)

    # Wynik: Liczba palców to liczba "dolin" + 1
    cv2.putText(img, f"Palce: {fingers + 1}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Detekcja', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Uruchomienie
print(cv2.__version__)
detect_fingers('handimg.jpg')