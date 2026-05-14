# Porównanie: Grid Search vs Hyperband

## 📋 Opis

Ten projekt zawiera dwie zaawansowane metody optymalizacji hiperparametrów dla algorytmu Random Forest:

1. **Grid Search Coarse-to-Fine** (`advanced_grid_search_coarse_to_fine.py`) - Metoda tradycyjna
2. **Hyperband (Successive Halving)** (`hyperband_optimization.py`) - Metoda budżetowa/inteligentna
3. **Comparison Script** (`comparison_results.py`) - Program porównujący obie metody

## 🎯 Główne cechy

✅ **Pomiar czasu** - Oba algorytmy mierzą czas wykonania  
✅ **Porównanie wyników** - Tabela i wykresy porównawcze  
✅ **Trade-off analiza** - Szybkość vs dokładność  
✅ **Efficiency Score** - Kombinacja szybkości i dokładności  

## 📁 Struktura plików

```
├── advanced_grid_search_coarse_to_fine.py  # Grid Search - metoda klasyczna
├── hyperband_optimization.py                # Hyperband - metoda budżetowa
├── comparison_results.py                    # Program porównujący oba
└── README.md                                # Ten plik
```

## 🚀 Jak używać

### Opcja 1: Uruchom tylko Grid Search
```bash
python advanced_grid_search_coarse_to_fine.py
```

**Wyniki:**
- `best_model_coarse_to_fine.pkl` - Zapisany model
- `grid_search_coarse_to_fine_results.png` - Wykresy
- `grid_search_report.txt` - Raport tekstowy

⏱️ **Czas:** 15-40 minut

### Opcja 2: Uruchom tylko Hyperband
```bash
python hyperband_optimization.py
```

**Wyniki:**
- `best_model_hyperband.pkl` - Zapisany model
- `hyperband_optimization_results.png` - Wykresy
- `hyperband_report.txt` - Raport tekstowy

⏱️ **Czas:** 2-15 minut (ZNACZNIE SZYBCIEJ!)

### Opcja 3: Porównaj obie metody (REKOMENDOWANE)
```bash
python comparison_results.py
```

**Wyniki:**
- `comparison_results.txt` - Tabela porównawcza
- `comparison_grid_search_vs_hyperband.png` - Wykresy porównawcze
- `grid_search_coarse_to_fine_results.png` - Wykresy Grid Search
- `hyperband_optimization_results.png` - Wykresy Hyperband
- `best_model_coarse_to_fine.pkl` - Model Grid Search
- `best_model_hyperband.pkl` - Model Hyperband
- `grid_search_report.txt` - Raport Grid Search
- `hyperband_report.txt` - Raport Hyperband

⏱️ **Czas całkowity:** 20-50 minut (zależy od systemu)

## 📊 Porównanie metod

| Kryterium | Grid Search | Hyperband |
|-----------|------------|-----------|
| **Metoda** | Testuje wszystkie kombinacje | Testuje i eliminuje słabe kombinacje |
| **Szybkość** | Wolniejszy | 2-5x szybszy |
| **Dokładność** | Wysoka | Porównywalna |
| **Budżet** | Stały (pełne testy) | Inteligentny (agresywne pruning) |
| **Zasoby** | Wysoki CPU/RAM | Efektywny |

## 🔍 Co to jest Hyperband?

**Hyperband (Successive Halving)** to nowoczesna metoda optymalizacji, która:

1. **Faza 1** - Testuje wszystkie kombinacje parametrów na małej próbce danych
2. **Faza 2** - Eliminuje najsłabsze 50-66% kombinacji
3. **Faza 3** - Testuje pozostałe kombinacje na większej próbce
4. **Powtarza** - Aż do znalezienia najlepszych parametrów

🎯 **Efekt:** Szybciej znajduje dobre parametry bez testowania wszystkich!

## 📈 Interpreting Results

### Porównanie czasu
- Przyspieszenie (Speedup) pokazuje, ile razy Hyperband jest szybszy
- Przykład: `Speedup: 3.5x` = Hyperband 3.5 razy szybszy niż Grid Search

### Różnica w dokładności
- **< 1%** - Wyniki praktycznie identyczne ✅
- **1-5%** - Wyniki bardzo podobne ✅
- **> 5%** - Zauważalna różnica ⚠️

### Efficiency Score
- Kombinacja szybkości i dokładności (60% F1 + 40% Speed)
- Wyższa wartość = lepszy całkowity wynik

## 💡 Zalecenia

### Kiedy użyć Grid Search?
- Gdy dokładność jest absolutnym priorytetem
- Gdy masz dużo czasu
- Gdy chcesz być pewny, że przetestowałeś wszystkie kombinacje

### Kiedy użyć Hyperband? ⭐
- Gdy chcesz szybko znaleźć dobre parametry
- Gdy masz ograniczony czas
- Gdy wyniki porównania pokazują podobną dokładność
- **ZAWSZE JEŚLI HYPERBAND DAJ PODOBNE/LEPSZE WYNIKI!**

## 🛠️ Wymagane biblioteki

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## 📝 Struktura wyjściowych plików

### comparison_results.txt
Zawiera:
- Tabelę porównawczą (czas, accuracy, precision, recall, F1)
- Analizę przyspieszenia
- Różnicę w dokładności
- Parametry dla obu metod
- Wnioski i zalecenia

### *.png wykresy
- **Czas wykonania** - Porównanie czasów
- **Przyspieszenie** - Ile razy szybciej (Speedup)
- **Accuracy** - Porównanie dokładności
- **F1-Score** - Porównanie F1
- **Wszystkie metryki** - Porównanie w jednym miejscu
- **Efficiency Score** - Kombinowana ocena

## ⚙️ Dostrajanie parametrów

Jeśli chcesz zmienić zakresy parametrów testowanych:

**Grid Search:** Edytuj `coarse_params` w metodzie `coarse_grid_search()`  
**Hyperband:** Edytuj `param_grid` w metodzie `hyperband_optimization()`

## 🐛 Troubleshooting

### Błąd: "Plik nie znaleziony"
- Upewnij się, że jesteś w poprawnym katalogu
- Sprawdź ścieżki: `tiny_HaGRID/learning/learning_set.csv` itd.

### Zbyt długi czas wykonania
- Spróbuj Hyperband zamiast Grid Search
- Zmniejsz zakresy parametrów
- Użyj mniej fold'ów w cross-validation

### Brak wyników porównania
- Upewnij się, że oba skrypty ukończyły się poprawnie
- Sprawdź, czy `comparison_results.py` ma dostęp do obu modułów

## 📚 Referencje

- [Scikit-Learn GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [Scikit-Learn HalvingGridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.HalvingGridSearchCV.html)
- [Hyperband Paper](https://arxiv.org/abs/1603.06393)

## 📧 Autor

Tworzono dla projektu: **AI for Hand Gesture Recognition for ESP-CAM**

---

**Ostatnia aktualizacja:** 2026-05-11
