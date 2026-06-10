import pandas as pd
from sklearn.datasets import make_blobs, make_circles, make_moons
import sys

def zapisz_do_plikow(X, y):
    """Rozdziela wygenerowane dane na dwie klasy i zapisuje do CSV."""
    df_a = pd.DataFrame(X[y == 0], columns=['cecha1', 'cecha2'])
    df_b = pd.DataFrame(X[y == 1], columns=['cecha1', 'cecha2'])
    
    df_a.to_csv('klasa_a.csv', index=False)
    df_b.to_csv('klasa_b.csv', index=False)
    
    print(f"Zapisano pomyślnie!")
    print(f" -> klasa_a.csv: {len(df_a)} punktów")
    print(f" -> klasa_b.csv: {len(df_b)} punktów")

def main():
    print("=== GENERATOR DANYCH DO LASU LOSOWEGO ===")
    print("Wybierz przypadek do wygenerowania:")
    print("1. Łatwy (wyraźnie rozdzielone grupy)")
    print("2. Trudny (mocno przenikające się chmury punktów)")
    print("3. Nieliniowy (jeden okrąg wewnątrz drugiego)")
    print("4. Księżyce (dwa przeplatające się półksiężyce)")
    
    wybor = input("Podaj numer przypadku (1-4): ")
    n_punktow = 600 # Całkowita liczba punktów (zostanie podzielona na pół)

    if wybor == '1':
        print("\nGenerowanie: Wyraźnie rozdzielone grupy...")
        # centers: współrzędne środków chmur, cluster_std: rozrzut
        X, y = make_blobs(n_samples=n_punktow, centers=[[2, 2], [8, 8]], cluster_std=1.0, random_state=42)
    
    elif wybor == '2':
        print("\nGenerowanie: Przenikające się chmury...")
        # Środki są blisko siebie, a rozrzut jest duży
        X, y = make_blobs(n_samples=n_punktow, centers=[[4, 4], [6, 6]], cluster_std=2.5, random_state=42)
    
    elif wybor == '3':
        print("\nGenerowanie: Okręgi (Concentric circles)...")
        # factor: odległość między okręgami, noise: szum (żeby nie było za łatwo)
        X, y = make_circles(n_samples=n_punktow, factor=0.4, noise=0.1, random_state=42)
        # Skalujemy okręgi, żeby lepiej wyglądały na wykresie
        X = X * 5 + 5 
    
    elif wybor == '4':
        print("\nGenerowanie: Księżyce (Moons)...")
        X, y = make_moons(n_samples=n_punktow, noise=0.15, random_state=42)
        # Skalujemy księżyce
        X = X * 3 + 3
    
    else:
        print("\nBłąd: Nieznany wybór. Uruchom program ponownie i wybierz 1, 2, 3 lub 4.")
        sys.exit()

    zapisz_do_plikow(X, y)
    print("\nGotowe! Możesz teraz uruchomić główny program (las_losowy.py), aby zobaczyć wynik.")

if __name__ == "__main__":
    main()