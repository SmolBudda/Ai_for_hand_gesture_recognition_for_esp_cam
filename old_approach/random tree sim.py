import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import os

def przygotuj_przykladowe_dane():
    """Tworzy przykładowe pliki CSV, jeśli nie istnieją."""
    if not os.path.exists('klasa_a.csv'):
        # Klasa A: skupiona wokół (2, 2)
        da = np.random.randn(50, 2) + [2, 2]
        pd.DataFrame(da).to_csv('klasa_a.csv', index=False, header=['cecha1', 'cecha2'])
    if not os.path.exists('klasa_b.csv'):
        # Klasa B: skupiona wokół (5, 5)
        db = np.random.randn(50, 2) + [5, 5]
        pd.DataFrame(db).to_csv('klasa_b.csv', index=False, header=['cecha1', 'cecha2'])

def main():
    # 0. Przygotowanie/Sprawdzenie plików
    przygotuj_przykladowe_dane()

    print("Wczytywanie danych z plików CSV...")
    # 1. Wczytanie danych
    # Zakładamy, że pliki mają 2 kolumny z cechami
    df_a = pd.read_csv('klasa_a.csv')
    df_b = pd.read_csv('klasa_b.csv')

    # Dodajemy kolumnę celu (klasa 0 i klasa 1)
    df_a['target'] = 0
    df_b['target'] = 1

    # Łączymy zbiory w jeden
    df = pd.concat([df_a, df_b], ignore_index=True)
    
    # Rozdzielamy cechy (X) od etykiet (y)
    X = df.iloc[:, :2].values  # bierzemy pierwsze dwie kolumny
    y = df['target'].values

    # 2. Tworzenie i trenowanie Lasu Losowego
    # n_estimators to liczba drzew w lesie
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)

    print("Model został wytrenowany. Generowanie wizualizacji...")

    # 3. Wizualizacja granicy decyzyjnej
    # Tworzymy siatkę punktów, aby "pokolorować" tło
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # Przewidujemy klasę dla każdego punktu na siatce
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Rysowanie
    plt.figure(figsize=(10, 7))
    
    # Rysujemy obszary decyzyjne (tło)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    
    # Rysujemy punkty z plików CSV
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', edgecolors='k', label='Klasa A (0)')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', edgecolors='k', label='Klasa B (1)')

    plt.xlabel('Cecha 1')
    plt.ylabel('Cecha 2')
    plt.title('Klasyfikacja Lasem Losowym (Random Forest)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    print("Wyświetlanie wykresu...")
    plt.show()

if __name__ == "__main__":
    main()