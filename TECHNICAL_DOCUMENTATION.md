# Dokumentacja Techniczna: Grid Search vs Hyperband

## 📖 Spis treści

1. [Grid Search Coarse-to-Fine](#grid-search-coarse-to-fine)
2. [Hyperband (Successive Halving)](#hyperband-successive-halving)
3. [Porównanie teoretyczne](#porównanie-teoretyczne)
4. [Interpretacja wyników](#interpretacja-wyników)
5. [Dobre praktyki](#dobre-praktyki)

---

## Grid Search Coarse-to-Fine

### Idea
Dwufazowe przeszukiwanie przestrzeni parametrów:
- **FAZA 1 (COARSE):** Testuje szerokie zakresy parametrów
- **FAZA 2 (FINE):** Zawęża zakresy wokół najlepszych parametrów z fazy 1

### Algorytm

```
FAZA 1 - COARSE GRID SEARCH
├─ Parametry: szerokie zakresy
├─ n_estimators: [50, 75, 100, 150, 200, 300, 400, 500]
├─ max_depth: [5, 8, 10, 15, 20, 30, 40, 50, None]
├─ min_samples_split: [2, 3, 4, 5, 7, 10]
├─ min_samples_leaf: [1, 2, 3, 4, 5]
├─ max_features: ['sqrt', 'log2', None]
└─ Rezultat: Best params (najlepsze z fazy 1)

FAZA 2 - FINE GRID SEARCH
├─ Parametry: zawężone wokół najlepszych
├─ Dla każdego parametru: wartość_best ± 1 sąsiad
├─ Mniej kombinacji, ale bardziej szczegółowe
└─ Rezultat: Best params (finalne)
```

### Liczba testowanych kombinacji

**FAZA COARSE:**
- n_estimators: 8 opcji
- max_depth: 9 opcji
- min_samples_split: 6 opcji
- min_samples_leaf: 5 opcji
- max_features: 3 opcje
- **RAZEM: 8 × 9 × 6 × 5 × 3 = 6,480 kombinacji**
- Z 5-fold CV: **32,400 treningów modelu**

**FAZA FINE:**
- Każdy parametr ma ~3 warianty (center ± 1)
- **RAZEM: ~3^5 = 243 kombinacji**
- Z 5-fold CV: **1,215 treningów modelu**

**RAZEM RAZEM: ~33,615 treningów modelu**

### Złożoność czasowa

$O(C \times K \times T_{train})$

Gdzie:
- $C$ = liczba kombinacji (~6,480 + 243)
- $K$ = liczba fold'ów (5)
- $T_{train}$ = czas trenowania jednego modelu

### Zalety
✅ Gwarantuje znalezienie dobrego zakresu parametrów  
✅ Determinystyczne (zawsze te same wyniki)  
✅ Łatwe do zrozumienia i implementacji  
✅ Bardzo dokładne ostateczne wyniki  

### Wady
❌ Czasochłonne (każda kombinacja testowana w pełni)  
❌ Marnotrawstwo zasobów na słabe kombinacje  
❌ Słaba skaluje się na większych zbiorach danych  
❌ Brak inteligencji w wyborze kombinacji  

---

## Hyperband (Successive Halving)

### Idea
Agresywne testowanie wielu kombinacji na małych próbkach i iteracyjne eliminowanie słabych.

**Hasło:** "Test dużo, ale krótko. Inwestuj w obiecujące."

### Algorytm

```
ITERACJA 1 (min_resources=10)
├─ Testuj: WSZYSTKIE kombinacje na 10 próbkach
├─ Liczba: 6,480 kombinacji
├─ Eliminate: Najsłabsze (factor=3) → 6,480/3 = 2,160

ITERACJA 2 (30 próbek)
├─ Testuj: 2,160 kombinacji na 30 próbkach
├─ Eliminate: Najsłabsze → 2,160/3 = 720

ITERACJA 3 (90 próbek)
├─ Testuj: 720 kombinacji na 90 próbkach
├─ Eliminate: Najsłabsze → 720/3 = 240

ITERACJA 4 (270 próbek)
├─ Testuj: 240 kombinacji na 270 próbkach
├─ Eliminate: Najsłabsze → 240/3 = 80

ITERACJA 5 (810+ próbek)
└─ Testuj: 80 kombinacji na pełnym zbiorze
   → Najlepsza kombinacja
```

### Liczba testowanych kombinacji

Przy factor=3:
- Iteracja 1: 6,480 kombinacji × 5-fold = 32,400 treningów
- Iteracja 2: 2,160 kombinacji × 5-fold = 10,800 treningów
- Iteracja 3: 720 kombinacji × 5-fold = 3,600 treningów
- Iteracja 4: 240 kombinacji × 5-fold = 1,200 treningów
- Iteracja 5: 80 kombinacji × 5-fold = 400 treningów

**RAZEM: ~48,400 treningów** (ale na MNIEJSZYCH próbkach!)

### Efekt zasobow

Na początkowych iteracjach testujemy tylko mały podzbiór danych:

```
Iteracja 1: 10 próbek (zwykle 1-2 sekundy per kombinacja)
Iteracja 2: 30 próbek (zwykle 2-5 sekund per kombinacja)
Iteracja 3: 90 próbek (zwykle 5-10 sekund per kombinacja)
Iteracja 4: 270 próbek (zwykle 10-20 sekund per kombinacja)
Iteracja 5: Pełny zbiór (zwykle 20-50 sekund per kombinacja)
```

### Złożoność czasowa

Hipergeometryczna - trudno wyrazić analitycznie, ale przybliżenie:

$O(C \times K \times T_{avg} / factor^{iterations})$

Praktycznie: ~2-5x szybciej niż Grid Search

### Zalety
✅ ZNACZNIE szybciej (2-5x)  
✅ Inteligentna alokacja zasobów  
✅ Eliminuje słabe kombinacje wcześnie  
✅ Dobre skalowanie  
✅ Nowoczesne podejście (Stanford & Berkeley)  
✅ Użyte w praktyce (Auto-ML systemy)  

### Wady
❌ Może pominąć dobrą kombinację (mała szansa)  
❌ Bardziej złożone do implementacji  
❌ Parametry (factor, min_resources) wymagają dostrojenia  
❌ Niestabilne wyniki (losowe elementy)  

---

## Porównanie teoretyczne

### Tabela porównawcza

| Aspekt | Grid Search | Hyperband |
|--------|------------|-----------|
| **Testowane kombinacje** | Wszystkie (6,480+243) | Wszystkie, ale iteracyjnie |
| **Zasoby per kombinacja** | Pełny zbiór danych | Mały→duży (dynamiczny) |
| **Całkowita pracy** | Wysoka | Średnia (ale efektywna) |
| **Czas** | 15-40 minut | 2-15 minut |
| **Dokładność** | Najwyższa | Porównywalna |
| **Pruning** | Brak | Agresywne (factor=3) |
| **Inteligencja** | Brak | Bardzo wysoka |
| **Skalowanie** | Słabe (O(C*K*T)) | Dobre (O(C*K*T/f^i)) |

### Analiza Trade-Off

```
Grid Search:
├─ PRO: Gwarancja znalezienia dobrego zestawu
├─ PRO: Determinizm
└─ CON: Powolny

Hyperband:
├─ PRO: Bardzo szybki
├─ PRO: Inteligentny
├─ CON: Mała szansa pominięcia dobrego zestawu
└─ CON: Bardziej złożony
```

---

## Interpretacja wyników

### Porównanie czasu

**Speedup = Czas Grid Search / Czas Hyperband**

Interpretacja:
- Speedup = 1.0 → Taki sam czas
- Speedup = 2.0 → Hyperband 2x szybszy
- Speedup = 3.5 → Hyperband 3.5x szybszy

### Różnica dokładności

Porównanie F1-Score:

```
ΔF1 = |F1_GridSearch - F1_Hyperband|

ΔF1 < 0.01 (< 1%)   → Praktycznie identyczne ✅
ΔF1 < 0.05 (< 5%)   → Bardzo podobne ✅
ΔF1 < 0.10 (< 10%)  → Podobne ⚠️
ΔF1 ≥ 0.10          → Znacząca różnica ❌
```

### Efficiency Score

Kombinacja szybkości i dokładności:

$$\text{Efficiency} = 0.6 \times \text{NormF1} + 0.4 \times \text{NormSpeed}$$

Gdzie:
- NormF1 = F1-Score znormalizowany do [0, 1]
- NormSpeed = 1 - (Czas / MaxCzas) - odwrócona normalizacja

**Wyższa wartość = lepszy całkowity wynik**

### Przykładowe rezultaty

**Scenariusz A (Hyperband preferowany):**
```
Grid Search:  F1=0.95, Czas=2400s
Hyperband:    F1=0.94, Czas=600s
Speedup: 4.0x, ΔF1=0.01 ✅

Zalecenie: Hyperband (4x szybszy, prawie identyczna dokładność)
```

**Scenariusz B (Trade-off):**
```
Grid Search:  F1=0.96, Czas=2400s
Hyperband:    F1=0.93, Czas=600s
Speedup: 4.0x, ΔF1=0.03 ⚠️

Zalecenie: Zależy od priorytetu (szybkość vs dokładność)
```

**Scenariusz C (Grid Search preferowany):**
```
Grid Search:  F1=0.98, Czas=2400s
Hyperband:    F1=0.92, Czas=600s
Speedup: 4.0x, ΔF1=0.06 ❌

Zalecenie: Grid Search (znacząco dokładniejszy)
```

---

## Dobre praktyki

### Gdy używać Grid Search?

1. **Malutkie przestrzenie parametrów** (<1000 kombinacji)
2. **Krytyczna dokładność** (każda setna część procenta się liczy)
3. **Producja** (stabilność + gwarancja)
4. **Benchmarking** (porównanie metod)
5. **Mały zbiór danych** (szybki training)

### Gdy używać Hyperband?

1. ⭐ **Szybkie prototypowanie**
2. ⭐ **Duże przestrzenie parametrów** (>1000 kombinacji)
3. ⭐ **Ograniczony czas** (deadline)
4. ⭐ **Duże zbiory danych** (powolny training)
5. ⭐ **Auto-ML pipelines**

### Best Practices

#### 1. Zawsze porównuj obie metody
```python
# To, co zrobiliśmy w comparison_results.py
# jest DOBRĄ PRAKTYKĄ!
```

#### 2. Monitoruj czas
```python
start = time.time()
# ... algorithm ...
elapsed = time.time() - start
print(f"Czas: {elapsed:.2f}s ({elapsed/60:.2f}min)")
```

#### 3. Statystyczna istotność
Dla bardziej wiarygodnych wyników powtórz kilka razy:
```python
for run in range(3):
    results.append(run_hyperband())
avg_results = average(results)
```

#### 4. Cross-Validation
Zawsze używaj (my używamy 5-fold):
```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

#### 5. Randomness Control
Ustaw random_state dla powtarzalności:
```python
RandomForestClassifier(random_state=42)
```

---

## Podsumowanie

### Wybór metody

```
┌─ Masz ograniczony CZAS?
│  ├─ TAK  → Hyperband ⚡
│  └─ NIE  → Grid Search + Hyperband (porównanie) ✅
│
└─ Dokładność jest KLUCZOWA?
   ├─ TAK  → Grid Search 🎯
   └─ NIE  → Hyperband ⚡
```

### Rekomendacja

🏆 **ZAWSZE uruchom `comparison_results.py`!**

Powody:
1. Vidzi rzeczywisty trade-off dla TWOICH danych
2. Mierzy czasy na TWOIM sprzęcie
3. Daje konkretne zalecenia
4. Zajmuje tyle czasu co jedna metoda (uruchamiamy sekwencyjnie)
5. Otrzymujesz wykresy i tabele porównawcze

---

## Referencje

1. **HyperBand:** https://arxiv.org/abs/1603.06393
2. **Successive Halving:** https://arxiv.org/abs/1502.07943
3. **Scikit-Learn Documentation:** https://scikit-learn.org/
4. **AutoML Survey:** https://arxiv.org/abs/1908.00709

---

**Autor:** AI Hand Gesture Recognition Project  
**Data:** 2026-05-11  
**Wersja:** 1.0
