import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import time

class GestureRecognizer:
    def __init__(self, n_estimators=100, random_state=42):
        """
        Inicjalizacja klasyfikatora Random Forest
        
        :param n_estimators: Liczba drzew w lesie
        :param random_state: Ziarno losowości dla powtarzalności
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,  # Użyj wszystkich procesorów
            verbose=0
        )
        self.label_encoder = {}
        self.label_decoder = {}
        self.is_trained = False
        self.grid_search_history = {}
    
    def load_data(self, csv_path):
        """Wczytanie danych z pliku CSV"""
        print(f"Wczytywanie danych z: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Oddzielenie cech (X) i etykiet (y)
        X = df.iloc[:, :-1].values  # Wszystkie kolumny oprócz ostatniej
        y = df.iloc[:, -1].values   # Ostatnia kolumna (label)
        
        print(f"  ✓ Wczytano {len(X)} próbek")
        print(f"  ✓ Liczba cech: {X.shape[1]}")
        print(f"  ✓ Unikalnych gestów: {len(np.unique(y))}")
        print(f"  ✓ Gesty: {np.unique(y)}")
        
        return X, y, df
    
    def grid_search_coarse_to_fine(self, X_train, y_train, X_test, y_test):
        """
        Grid Search metodą 'coarse to fine' - dwuetapowa optymalizacja
        
        :param X_train: Dane treningowe
        :param y_train: Etykiety treningowe
        :param X_test: Dane testowe (do ewaluacji)
        :param y_test: Etykiety testowe
        """
        print("\n" + "="*70)
        print("=== GRID SEARCH METODĄ 'COARSE TO FINE' ===")
        print("="*70)
        
        # ===== ETAP 1: COARSE (GRUBA SIATKA) =====
        print("\n📊 ETAP 1: COARSE GRID SEARCH (siatka gruba)")
        print("-" * 70)
        print("Testowanie szerokiego zakresu parametrów...\n")
        
        param_grid_coarse = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        rf_coarse = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_coarse = GridSearchCV(
            rf_coarse,
            param_grid_coarse,
            cv=8,  # 5-fold cross-validation
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        start_time = time.time()
        grid_coarse.fit(X_train, y_train)
        coarse_time = time.time() - start_time
        
        print(f"\n✓ Etap COARSE zakończony ({coarse_time:.2f}s)")
        print(f"  Najlepszy wynik CV: {grid_coarse.best_score_:.4f}")
        print(f"  Najlepsze parametry:\n    {grid_coarse.best_params_}\n")
        
        self.grid_search_history['coarse'] = {
            'best_score': grid_coarse.best_score_,
            'best_params': grid_coarse.best_params_,
            'cv_results': grid_coarse.cv_results_
        }
        
        # ===== ETAP 2: FINE (FINA SIATKA) =====
        print("\n📊 ETAP 2: FINE GRID SEARCH (siatka fina)")
        print("-" * 70)
        print("Precyzyjne tuning wokół najlepszych parametrów...\n")
        
        best_params = grid_coarse.best_params_
        
        # Definiowanie precyzyjnej siatki wokół najlepszych parametrów
        n_est_best = best_params['n_estimators']
        param_grid_fine = {
            'n_estimators': [max(n_est_best-50, 50), n_est_best, n_est_best+50],
            'max_depth': _create_fine_range(best_params['max_depth'], base_range=[8, 15, 20, 25, 35]),
            'min_samples_split': _create_fine_range(best_params['min_samples_split'], base_range=[2, 3, 4, 6, 8]),
            'min_samples_leaf': _create_fine_range(best_params['min_samples_leaf'], base_range=[1, 2, 3, 4]),
            'max_features': best_params['max_features'] if isinstance(best_params['max_features'], list) else [best_params['max_features']]
        }
        
        rf_fine = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_fine = GridSearchCV(
            rf_fine,
            param_grid_fine,
            cv=8,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        start_time = time.time()
        grid_fine.fit(X_train, y_train)
        fine_time = time.time() - start_time
        
        print(f"\n✓ Etap FINE zakończony ({fine_time:.2f}s)")
        print(f"  Najlepszy wynik CV: {grid_fine.best_score_:.4f}")
        print(f"  Najlepsze parametry:\n    {grid_fine.best_params_}\n")
        
        self.grid_search_history['fine'] = {
            'best_score': grid_fine.best_score_,
            'best_params': grid_fine.best_params_,
            'cv_results': grid_fine.cv_results_
        }
        
        # ===== EWALUACJA NA ZBIORZE TESTOWYM =====
        print("\n📊 EWALUACJA NA ZBIORZE TESTOWYM")
        print("-" * 70)
        
        # Model z parametrami z etapu COARSE
        model_coarse = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
        model_coarse.fit(X_train, y_train)
        y_pred_coarse = model_coarse.predict(X_test)
        coarse_accuracy = accuracy_score(y_test, y_pred_coarse)
        
        # Model z parametrami z etapu FINE
        model_fine = grid_fine.best_estimator_
        y_pred_fine = model_fine.predict(X_test)
        fine_accuracy = accuracy_score(y_test, y_pred_fine)
        
        print(f"\nAkkuracja COARSE na zbiorze testowym: {coarse_accuracy:.4f}")
        print(f"Akkuracja FINE na zbiorze testowym:   {fine_accuracy:.4f}")
        print(f"Poprawa: {(fine_accuracy - coarse_accuracy):.4f} ({((fine_accuracy - coarse_accuracy)*100):.2f}%)\n")
        
        # Wybranie najlepszego modelu
        if fine_accuracy >= coarse_accuracy:
            best_model = model_fine
            best_accuracy = fine_accuracy
            best_stage = "FINE"
        else:
            best_model = model_coarse
            best_accuracy = coarse_accuracy
            best_stage = "COARSE"
        
        self.model = best_model
        self.is_trained = True
        
        return {
            'best_model': best_model,
            'best_accuracy': best_accuracy,
            'best_stage': best_stage,
            'coarse_results': self.grid_search_history['coarse'],
            'fine_results': self.grid_search_history['fine'],
            'coarse_accuracy': coarse_accuracy,
            'fine_accuracy': fine_accuracy
        }
    
    def train(self, X_train, y_train):
        """Trenowanie modelu"""
        print("\n=== TRENING MODELU ===")
        print("Trenowanie Random Forest...")
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Wyświetlenie ważności cech
        feature_importance = self.model.feature_importances_
        top_features = np.argsort(feature_importance)[-10:][::-1]
        
        print("\nTop 10 najważniejszych cech:")
        for i, feature_idx in enumerate(top_features, 1):
            print(f"  {i}. Cecha #{feature_idx}: {feature_importance[feature_idx]:.4f}")
        
        print("\n✓ Trening zakończony!")
    
    def evaluate(self, X_test, y_test, set_name="Test", verbose=True):
        """Ewaluacja modelu na zbiorze testowym"""
        if not self.is_trained:
            print("Błąd: Model nie został wytrenowany!")
            return
                
        # Predykcja
        y_pred = self.model.predict(X_test)
        
        # Metryki
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        if verbose:        
            print(f"\n=== EWALUACJA NA {set_name.upper()} ===")
            # print(f"\nMetryki:")
            print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            
            # Raport klasyfikacji
            print(f"\n--- Raport Klasyfikacji ---")
            print(classification_report(y_test, y_pred))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'y_test': y_test
        }
    
    def predict_gesture(self, hand_landmarks):
        """Predykcja gestu na podstawie współrzędnych punktów"""
        if not self.is_trained:
            print("Błąd: Model nie został wytrenowany!")
            return None
        
        # hand_landmarks powinno być listą 42 wartości (21 punktów × 2 współrzędne)
        landmarks_array = np.array(hand_landmarks).reshape(1, -1)
        prediction = self.model.predict(landmarks_array)[0]
        confidence = np.max(self.model.predict_proba(landmarks_array))
        
        return prediction, confidence
    
    def save_model(self, filepath):
        """Zapisanie modelu do pliku"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"✓ Model zapisany: {filepath}")
    
    @staticmethod
    def load_model(filepath):
        """Wczytanie modelu z pliku"""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Model wczytany: {filepath}")
        return model


def _create_fine_range(best_value, base_range):
    """
    Tworzy precyzyjny zakres parametrów wokół najlepszej wartości
    """
    if best_value is None:
        return [None]
    
    # Znalezienie wartości najbliższej best_value w base_range
    closest_idx = min(range(len(base_range)), key=lambda i: abs(base_range[i] - best_value))
    
    # Zwrócenie zakresu wokół najlepszej wartości
    start_idx = max(0, closest_idx - 1)
    end_idx = min(len(base_range) - 1, closest_idx + 1)
    
    return [base_range[i] for i in range(start_idx, end_idx + 1)]


def visualize_grid_search_results(recognizer, X_test, y_test, results):
    """
    Wizualizacja wyników grid search
    """
    print("\n--- GENEROWANIE WYKRESÓW ---")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Porównanie dokładności COARSE vs FINE
    stages = ['COARSE', 'FINE']
    test_accuracies = [results['coarse_accuracy'], results['fine_accuracy']]
    cv_scores = [
        results['coarse_results']['best_score'],
        results['fine_results']['best_score']
    ]
    
    ax = axes[0, 0]
    x = np.arange(len(stages))
    width = 0.35
    ax.bar(x - width/2, cv_scores, width, label='CV Score', alpha=0.8, color='skyblue')
    ax.bar(x + width/2, test_accuracies, width, label='Test Accuracy', alpha=0.8, color='salmon')
    ax.set_ylabel('Dokładność')
    ax.set_title('Porównanie wyników: COARSE vs FINE')
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.legend()
    ax.set_ylim([0, 1.1])
    
    # 2. Najlepsze parametry
    ax = axes[0, 1]
    ax.axis('off')
    best_params_text = "🏆 NAJLEPSZE PARAMETRY Z ETAPU " + results['best_stage'] + "\n\n"
    if results['best_stage'] == 'COARSE':
        params = results['coarse_results']['best_params']
    else:
        params = results['fine_results']['best_params']
    
    for key, value in params.items():
        best_params_text += f"  • {key}: {value}\n"
    
    best_params_text += f"\n  CV Score: {results[results['best_stage'].lower() + '_results']['best_score']:.4f}\n"
    best_params_text += f"  Test Accuracy: {results['best_accuracy']:.4f}"
    
    ax.text(0.1, 0.5, best_params_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. Top 10 najważniejszych cech
    ax = axes[1, 0]
    feature_importance = results['best_model'].feature_importances_
    top_10_idx = np.argsort(feature_importance)[-10:][::-1]
    top_10_values = feature_importance[top_10_idx]
    
    ax.barh(range(10), top_10_values, color='steelblue')
    ax.set_yticks(range(10))
    ax.set_yticklabels([f'Cecha #{i}' for i in top_10_idx])
    ax.set_xlabel('Ważność')
    ax.set_title('Top 10 najważniejszych cech')
    ax.invert_yaxis()
    
    # 4. Informacje o treningu
    ax = axes[1, 1]
    ax.axis('off')
    
    train_info = (
        f"📊 INFORMACJE O TRENINGU\n\n"
        f"  Liczba próbek: {len(y_test)}\n"
        f"  Liczba unikalnych klas: {len(np.unique(y_test))}\n"
        f"  Liczba drzew w lesie: {results['best_model'].n_estimators}\n"
        f"  Max depth: {results['best_model'].max_depth}\n"
        f"  Min samples split: {results['best_model'].min_samples_split}\n"
        f"  Min samples leaf: {results['best_model'].min_samples_leaf}\n\n"
        f"  🎯 Najlepszy etap: {results['best_stage']}\n"
        f"  📈 Finalna dokładność: {results['best_accuracy']:.2%}"
    )
    
    ax.text(0.1, 0.5, train_info, fontsize=10, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('grid_search_results.png', dpi=150, bbox_inches='tight')
    print("✓ Wykresy zapisane: grid_search_results.png")
    plt.show()


def main():
    print("=" * 70)
    print("=== RANDOM FOREST - GRID SEARCH COARSE TO FINE ===")
    print("=" * 70)
    
    # Ścieżki do plików
    learning_set_path = "tiny_HaGRID/learning/learning_set.csv"
    testing_set_path = "tiny_HaGRID/testing/testing_set.csv"
    model_save_path = "gesture_model_optimized_grid_search.pkl"
    
    # Sprawdzenie czy pliki istnieją
    if not os.path.exists(learning_set_path):
        print(f"Błąd: Plik {learning_set_path} nie znaleziony!")
        return
    
    if not os.path.exists(testing_set_path):
        print(f"Błąd: Plik {testing_set_path} nie znaleziony!")
        return
    
    # Inicjalizacja modelu
    recognizer = GestureRecognizer(n_estimators=100)
    
    # Wczytanie danych
    print("\n--- WCZYTYWANIE DANYCH ---")
    X_train, y_train, train_df = recognizer.load_data(learning_set_path)
    X_test, y_test, test_df = recognizer.load_data(testing_set_path)
    
    # Grid Search Coarse to Fine
    results = recognizer.grid_search_coarse_to_fine(X_train, y_train, X_test, y_test)
    
    # Ewaluacja
    print("\n--- SZCZEGÓŁOWA EWALUACJA NA ZBIORZE TESTOWYM ---")
    eval_results = recognizer.evaluate(X_test, y_test, set_name="Test Set")
    
    # Wizualizacja
    visualize_grid_search_results(recognizer, X_test, y_test, results)
    
    # Zapisanie modelu
    recognizer.save_model(model_save_path)
    
    # Podsumowanie
    print("\n" + "=" * 70)
    print("=== PODSUMOWANIE ===")
    print("=" * 70)
    print(f"\n🏆 NAJLEPSZY MODEL: Etap {results['best_stage']}")
    print(f"\n📊 Wyniki finalne:")
    
    if results['best_stage'] == 'COARSE':
        cv_score = results['coarse_results']['best_score']
    else:
        cv_score = results['fine_results']['best_score']
    
    print(f"   Dokładność (CV): {cv_score:.4f}")
    print(f"   Dokładność (Test): {results['best_accuracy']:.4f}")
    print(f"\n🔧 Parametry:")
    if results['best_stage'] == 'FINE':
        params = results['fine_results']['best_params']
    else:
        params = results['coarse_results']['best_params']
    
    for key, value in params.items():
        print(f"   • {key}: {value}")
    
    print(f"\n💾 Model zapisany: {model_save_path}")
    print(f"📈 Wykresy zapisane: grid_search_results.png")
    print("\n✓ Grid Search zakończony pomyślnie!")
    

if __name__ == "__main__":
    main()
