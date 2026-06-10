#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trenowanie osobnych modeli dla gestów 1-ręcznych i 2-ręcznych
Separate training for single-hand and two-hand gestures
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

# ============================================================================
# KONFIGURACJA
# ============================================================================

CSV_PATH = "tiny_HaGRID/learning/learning_set_2hands.csv"
MODELS_DIR = "models"

# Kolumny drugiej ręki
SECOND_HAND_START = 42

os.makedirs(MODELS_DIR, exist_ok=True)

# ============================================================================
# ZAŁADOWANIE DANYCH
# ============================================================================

print("="*70)
print("TRENOWANIE OSOBNYCH MODELI DLA GESTÓW 1-RĘCZNYCH I 2-RĘCZNYCH")
print("="*70)

print(f"\nŁadowanie danych z: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print(f"✓ Załadowano {len(df)} próbek\n")

# ============================================================================
# IDENTYFIKACJA GESTÓW 1-RĘCZNYCH I 2-RĘCZNYCH
# ============================================================================

print("-"*70)
print("IDENTYFIKACJA RODZAJÓW GESTÓW")
print("-"*70 + "\n")

single_hand_indices = []
two_hand_indices = []

# Kolumny drugiej ręki
second_hand_cols = [f'x{i}' for i in range(21, 42)] + [f'y{i}' for i in range(21, 42)]

for idx, row in df.iterrows():
    second_hand_data = row[second_hand_cols]
    
    # Sprawdzamy czy druga ręka ma znaczące wartości
    if second_hand_data.notna().any() and (second_hand_data[second_hand_data.notna()] > 0).any():
        two_hand_indices.append(idx)
    else:
        single_hand_indices.append(idx)

print(f"Gesty 1-ręczne: {len(single_hand_indices)} próbek")
print(f"Gesty 2-ręczne: {len(two_hand_indices)} próbek\n")

# ============================================================================
# MODEL 1: GESTY 1-RĘCZNE (42 FEATURES)
# ============================================================================

print("-"*70)
print("TRENOWANIE MODELU DLA GESTÓW 1-RĘCZNYCH (42 features)")
print("-"*70 + "\n")

single_hand_df = df.iloc[single_hand_indices].copy()

# Kolumny pierwszej ręki
first_hand_cols = [f'x{i}' for i in range(0, 21)] + [f'y{i}' for i in range(0, 21)]

X_single = single_hand_df[first_hand_cols].values
y_single = single_hand_df['label'].values

print(f"Dane wejściowe (X): {X_single.shape}")
print(f"Etykiety (y): {y_single.shape}")
print(f"\nKlasy gestów 1-ręcznych:")
for label in np.unique(y_single):
    count = (y_single == label).sum()
    print(f"  - {label}: {count} próbek")

# Podział na train/test
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_single, y_single, test_size=0.2, random_state=42, stratify=y_single
)

print(f"\nTrain: {X_train_s.shape[0]}, Test: {X_test_s.shape[0]}")

# Trenowanie
print("\nTrenowanie modelu RandomForest (1-ręczne)...")
rf_single = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=1)
rf_single.fit(X_train_s, y_train_s)

# Ewaluacja
y_pred_s = rf_single.predict(X_test_s)
acc_s = accuracy_score(y_test_s, y_pred_s)

print(f"\n✓ Dokładność (1-ręczne): {acc_s:.4f}")
print("\nRaport klasyfikacji (1-ręczne):")
print(classification_report(y_test_s, y_pred_s))

# Zapis modelu
single_hand_model_path = os.path.join(MODELS_DIR, "gesture_model_single_hand.pkl")
with open(single_hand_model_path, 'wb') as f:
    pickle.dump(rf_single, f)
print(f"✓ Model zapisany: {single_hand_model_path}\n")

# ============================================================================
# MODEL 2: GESTY 2-RĘCZNE (84 FEATURES)
# ============================================================================

print("-"*70)
print("TRENOWANIE MODELU DLA GESTÓW 2-RĘCZNYCH (84 features)")
print("-"*70 + "\n")

two_hand_df = df.iloc[two_hand_indices].copy()

# Wszystkie 84 kolumny (obie ręce)
all_hand_cols = first_hand_cols + second_hand_cols

X_two = two_hand_df[all_hand_cols].values
y_two = two_hand_df['label'].values

print(f"Dane wejściowe (X): {X_two.shape}")
print(f"Etykiety (y): {y_two.shape}")
print(f"\nKlasy gestów 2-ręcznych:")
for label in np.unique(y_two):
    count = (y_two == label).sum()
    print(f"  - {label}: {count} próbek")

# Podział na train/test
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
    X_two, y_two, test_size=0.2, random_state=42, stratify=y_two
)

print(f"\nTrain: {X_train_t.shape[0]}, Test: {X_test_t.shape[0]}")

# Trenowanie
print("\nTrenowanie modelu RandomForest (2-ręczne)...")
rf_two = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=1)
rf_two.fit(X_train_t, y_train_t)

# Ewaluacja
y_pred_t = rf_two.predict(X_test_t)
acc_t = accuracy_score(y_test_t, y_pred_t)

print(f"\n✓ Dokładność (2-ręczne): {acc_t:.4f}")
print("\nRaport klasyfikacji (2-ręczne):")
print(classification_report(y_test_t, y_pred_t))

# Zapis modelu
two_hand_model_path = os.path.join(MODELS_DIR, "gesture_model_two_hand.pkl")
with open(two_hand_model_path, 'wb') as f:
    pickle.dump(rf_two, f)
print(f"✓ Model zapisany: {two_hand_model_path}\n")

# ============================================================================
# PODSUMOWANIE
# ============================================================================

print("="*70)
print("PODSUMOWANIE")
print("="*70)
print(f"\nModele wytrenowane:")
print(f"  1. {single_hand_model_path}")
print(f"     - Features: 42")
print(f"     - Dokładność: {acc_s:.4f}")
print(f"     - Przykłady treningowe: {X_train_s.shape[0]}")
print(f"\n  2. {two_hand_model_path}")
print(f"     - Features: 84")
print(f"     - Dokładność: {acc_t:.4f}")
print(f"     - Przykłady treningowe: {X_train_t.shape[0]}")

print(f"\n✓ Gotowe! Program real_time_gesture_recognition.py")
print(f"  będzie teraz automatycznie wybierać odpowiedni model.\n")
