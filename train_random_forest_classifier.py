import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

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
            verbose=1
        )
        self.label_encoder = {}
        self.label_decoder = {}
        self.is_trained = False
    
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
    
    def evaluate(self, X_test, y_test, set_name="Test"):
        """Ewaluacja modelu na zbiorze testowym"""
        if not self.is_trained:
            print("Błąd: Model nie został wytrenowany!")
            return
        
        print(f"\n=== EWALUACJA NA {set_name.upper()} ===")
        
        # Predykcja
        y_pred = self.model.predict(X_test)
        
        # Metryki
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"\nMetryki:")
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        # Raport klasyfikacji
        print(f"\n--- Raport Klasyfikacji ---")
        print(classification_report(y_test, y_pred))
        
        # Macierz pomyłek
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nMacierz pomyłek:")
        print(cm)
        
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


def main():
    print("=" * 60)
    print("=== RANDOM FOREST - ROZPOZNAWANIE GESTÓW ===")
    print("=" * 60)
    
    # Ścieżki do plików
    learning_set_path = "tiny_HaGRID/learning/learning_set.csv"
    testing_set_path = "tiny_HaGRID/testing/testing_set.csv"
    model_save_path = "gesture_model_random_forest.pkl"
    
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
    
    # Trening
    recognizer.train(X_train, y_train)
    
    # Ewaluacja na zbiorze treningowym
    train_results = recognizer.evaluate(X_train, y_train, set_name="Training")
    
    # Ewaluacja na zbiorze testowym
    test_results = recognizer.evaluate(X_test, y_test, set_name="Testing")
    
    # Wizualizacja macierzy pomyłek (tylko zbiór testowy)
    print("\n--- GENEROWANIE WYKRESÓW ---")
    try:
        plt.figure(figsize=(12, 5))
        
        # Macierz pomyłek
        plt.subplot(1, 2, 1)
        sns.heatmap(
            test_results['confusion_matrix'],
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=np.unique(y_test),
            yticklabels=np.unique(y_test)
        )
        plt.title('Macierz Pomyłek - Zbiór Testowy')
        plt.ylabel('Rzeczywista etykieta')
        plt.xlabel('Przewidywana etykieta')
        
        # Porównanie metryk
        plt.subplot(1, 2, 2)
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        train_scores = [
            train_results['accuracy'],
            train_results['precision'],
            train_results['recall'],
            train_results['f1']
        ]
        test_scores = [
            test_results['accuracy'],
            test_results['precision'],
            test_results['recall'],
            test_results['f1']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x - width/2, train_scores, width, label='Trening', alpha=0.8)
        plt.bar(x + width/2, test_scores, width, label='Test', alpha=0.8)
        
        plt.ylabel('Wartość')
        plt.title('Porównanie Metryk')
        plt.xticks(x, metrics, rotation=45, ha='right')
        plt.legend()
        plt.ylim([0, 1.1])
        
        plt.tight_layout()
        plt.savefig('gesture_recognition_results.png', dpi=150)
        print("✓ Wykresy zapisane: gesture_recognition_results.png")
        plt.show()
    
    except Exception as e:
        print(f"Błąd przy generowaniu wykresów: {e}")
    
    # Zapisanie modelu
    recognizer.save_model(model_save_path)
    
    # Podsumowanie
    print("\n" + "=" * 60)
    print("=== PODSUMOWANIE ===")
    print("=" * 60)
    print(f"\n📊 Wyniki na zbiorze testowym:")
    print(f"   Accuracy:  {test_results['accuracy']*100:.2f}%")
    print(f"   Precision: {test_results['precision']:.4f}")
    print(f"   Recall:    {test_results['recall']:.4f}")
    print(f"   F1-Score:  {test_results['f1']:.4f}")
    print(f"\n💾 Model zapisany: {model_save_path}")
    print(f"\n📈 Wykresy zapisane: gesture_recognition_results.png")
    print("\n✓ Trening zakończony pomyślnie!")
    

if __name__ == "__main__":
    main()
