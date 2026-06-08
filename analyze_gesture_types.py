#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analiza rodzaju gestów w CSV (1-ręczne vs 2-ręczne)
"""

import pandas as pd
import numpy as np
import json
import os

csv_path = "tiny_HaGRID/learning/learning_set_2hands.csv"

print("="*70)
print("ANALIZA RODZAJÓW GESTÓW")
print("="*70)

# Wczytanie CSV
df = pd.read_csv(csv_path)
print(f"✓ Wczytano {len(df)} próbek gestów\n")

# Kolumny drugiej ręki to x42-x83 (jeśli indeksowanie od 0) i y42-y83
# W pandas, to będą kolumny indeksy od 42 do 83
second_hand_cols = [col for col in df.columns if col.startswith('x') and int(col[1:]) >= 21] + \
                   [col for col in df.columns if col.startswith('y') and int(col[1:]) >= 21]

print(f"Kolumny drugiej ręki (42 wartości): {len(second_hand_cols)}")

# Dla każdego gestu, sprawdzamy czy druga ręka jest zazwyczaj pusta
gesture_types = {}

for gesture_label in df['label'].unique():
    gesture_data = df[df['label'] == gesture_label]
    
    # Sprawdzamy drugą rękę
    second_hand_data = gesture_data[second_hand_cols]
    
    # Liczymy ile próbek ma drugą rękę (wartości niezerowe)
    # Ręka jest zdefiniowana, jeśli przynajmniej jeden punkt ma wartość > 0
    hands_present = 0
    hands_empty = 0
    
    for idx, row in second_hand_data.iterrows():
        # Sprawdź czy jakakolwiek wartość jest nonzero/non-NaN
        if row.notna().any() and (row[row.notna()] > 0).any():
            hands_present += 1
        else:
            hands_empty += 1
    
    # Ustalamy typ gestu
    total = hands_present + hands_empty
    percentage_with_second_hand = (hands_present / total * 100) if total > 0 else 0
    
    if percentage_with_second_hand > 50:
        gesture_type = "two_hands"
    else:
        gesture_type = "single_hand"
    
    gesture_types[gesture_label] = {
        "type": gesture_type,
        "samples": len(gesture_data),
        "with_second_hand": hands_present,
        "without_second_hand": hands_empty,
        "percentage": round(percentage_with_second_hand, 1)
    }

# Wyświetlanie wyników
print("\n" + "="*70)
print("WYNIKI ANALIZY")
print("="*70 + "\n")

single_hand_gestures = []
two_hand_gestures = []

for gesture_label in sorted(gesture_types.keys()):
    info = gesture_types[gesture_label]
    
    print(f"{gesture_label.upper()}: {info['type'].upper()}")
    print(f"  Próbek: {info['samples']}")
    print(f"  Druga ręka: {info['with_second_hand']} / {info['samples']} ({info['percentage']}%)")
    print()
    
    if info['type'] == "single_hand":
        single_hand_gestures.append(gesture_label)
    else:
        two_hand_gestures.append(gesture_label)

print("="*70)
print(f"PODSUMOWANIE:")
print(f"  Gesty 1-ręczne: {len(single_hand_gestures)}")
for g in sorted(single_hand_gestures):
    print(f"    - {g}")

print(f"\n  Gesty 2-ręczne: {len(two_hand_gestures)}")
for g in sorted(two_hand_gestures):
    print(f"    - {g}")

print("="*70 + "\n")

# Zapisz mapę gestów do pliku JSON
gesture_map = {
    gesture_label: gesture_types[gesture_label]['type']
    for gesture_label in gesture_types.keys()
}

output_path = "models/gesture_types.json"
os.makedirs("models", exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(gesture_map, f, indent=2)

print(f"✓ Mapa gestów zapisana do: {output_path}")
print("  Można teraz użyć tej mapy w real_time_gesture_recognition.py\n")
