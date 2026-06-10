import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import time
from datetime import datetime

#ścieżki
model_save_path = "models/2hands_best_model_hyperband.pkl"
results_png_path = "output/2hands_hyperband_optimization_results.png"
results_report_path = "output/2hands_hyperband_report.txt"

class HyperbandOptimization:
    def __init__(self, random_state=42, resource='n_samples', factor=3):
        """
        Inicjalizacja optymalizacji Hyperband (Successive Halving)
        
        :param random_state: Ziarno losowości dla powtarzalności
        :param resource: Zasób do monitorowania ('n_samples' lub 'n_iterations')
        :param factor: Faktor redukcji (jak wiele gorszych kombinacji eliminować)
        """
        self.random_state = random_state
        self.best_model = None
        self.best_params = None
        self.halving_results = None
        self.cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        self.is_evaluated = False
        self.resource = resource
        self.factor = factor
        self.iteration_history = []
    
    def load_data(self, csv_path):
        """Wczytanie danych z pliku CSV"""
        print(f"Wczytywanie danych z: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Oddzielenie cech (X) i etykiet (y)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        print(f"  ✓ Wczytano {len(X)} próbek")
        print(f"  ✓ Liczba cech: {X.shape[1]}")
        print(f"  ✓ Unikalnych gestów: {len(np.unique(y))}")
        print(f"  ✓ Gesty: {np.unique(y)}")
        
        return X, y
    
    def hyperband_optimization(self, X_train, y_train):
        """
        Optymalizacja parametrów metodą Hyperband (Successive Halving)
        
        Hyperband iteracyjnie:
        1. Testuje wszystkie kombinacje parametrów na małej próbce danych
        2. Odrzuca najsłabsze kombinacje
        3. Inwestuje więcej zasobów w obiecujące kombinacje
        4. Powtarza proces
        
        To znacznie szybsze niż Grid Search!
        """
        print("\n" + "="*70)
        print("OPTYMALIZACJA HYPERBAND (Successive Halving)")
        print("="*70)
        
        # Parametry do testowania
        param_grid = {
            'n_estimators': [100, 200, 400],
            'max_depth': [10, 20, 40, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 3, 5],
            'max_features': ['sqrt', 'log2', None]
        }
        
        print("\nTestowane zakresy parametrów:")
        total_combinations = 1
        for param, values in param_grid.items():
            print(f"  - {param}: {values}")
            total_combinations *= len(values)
        
        print(f"\n  💡 Maksymalna liczba kombinacji: {total_combinations}")
        # print(f"  💡 Metoda Hyperband eliminuje słabe kombinacje wcześnie")
        # print(f"  ⏱️  To powinno być szybsze niż Grid Search Coarse-to-Fine!")
        
        # Konfiguracja Hyperband
        rf_base = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        
        start_time = time.time()
        print("\n⏳ Trenowanie modeli metodą Hyperband...")
        print("   (kombinacje są iteracyjnie testowane i eliminowane)\n")
        
        # HalvingGridSearchCV
        halving_search = HalvingGridSearchCV(
            rf_base,
            param_grid,
            cv=self.cv_strategy,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1,
            factor=self.factor,  # Ile gorszych kombinacji eliminować
            min_resources='exhaust', #(wg AI by dobrze liczyło, choć będzie wolniej) # Minimalna liczba próbek w pierwszej iteracji
            resource=self.resource
        )
        
        halving_search.fit(X_train, y_train)
        elapsed = time.time() - start_time
        
        print(f"\n✓ Optymalizacja Hyperband ukończona w {elapsed:.2f} sekund ({elapsed/60:.2f} minut)")
        print(f"✓ Liczba iteracji: {halving_search.n_iterations_}")
        print(f"✓ Liczba wszystkich modeli testowanych: {len(halving_search.cv_results_['mean_test_score'])}")
        
        self.halving_results = halving_search.cv_results_
        self.best_params = halving_search.best_params_
        self.best_model = halving_search.best_estimator_
        
        print(f"\n🏆 Najlepsze parametry:")
        print(f"   F1-Score: {halving_search.best_score_:.4f}")
        for param, value in self.best_params.items():
            print(f"   {param}: {value}")
        
        return halving_search, elapsed
    
    def evaluate_on_test_set(self, X_test, y_test):
        """Ewaluacja najlepszego modelu na zbiorze testowym"""
        if self.best_model is None:
            print("Błąd: Najpierw wykonaj optymalizację Hyperband!")
            return None
        
        print("\n" + "="*70)
        print("EWALUACJA NA ZBIORZE TESTOWYM")
        print("="*70)
        
        y_pred = self.best_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"\nMetryki na zbiorze testowym:")
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        print(f"\n--- Raport Klasyfikacji ---")
        print(classification_report(y_test, y_pred))
        
        self.is_evaluated = True
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'y_test': y_test
        }
    
    def interpret_results(self):
        """Interpretacja wyników optymalizacji"""
        print("\n" + "="*70)
        print("INTERPRETACJA WYNIKÓW")
        print("="*70)
        
        print(f"\n📊 Najlepsze znalezione parametry:")
        print(f"   F1-Score: {np.max(self.halving_results['mean_test_score']):.4f}")
        for param, value in self.best_params.items():
            print(f"   {param}: {value}")
        
        print(f"\n📈 Analiza wpływu parametrów:")
        
        # Analiza wrażliwości na n_estimators
        if 'param_n_estimators' in self.halving_results:
            n_est_raw = self.halving_results['param_n_estimators']
            n_est_values = []
            for val in n_est_raw:
                if val not in n_est_values:
                    n_est_values.append(val)
            
            # Sortowanie bezpiecznie
            n_est_values = sorted(n_est_values, key=lambda x: (x is None, x))
            n_est_scores = []
            for val in n_est_values:
                if val is None:
                    mask = pd.isna(n_est_raw)
                else:
                    mask = n_est_raw == val
                scores = self.halving_results['mean_test_score'][mask]
                if len(scores) > 0:
                    n_est_scores.append(np.mean(scores))
            
            if len(n_est_values) > 0:
                print(f"\n   n_estimators (liczba drzew):")
                for val, score in zip(n_est_values, n_est_scores):
                    if val is not None:
                        print(f"      {val}: {score:.4f}")
        
        # print(f"\n💡 Wnioski:")
        # print(f"   - Hyperband efektywnie identyfikuje najlepsze parametry")
        # print(f"   - Eliminuje słabe kombinacje wcześnie (budżet czasowy)")
        # print(f"   - Szybciej niż tradycyjny Grid Search")
    
    def visualize_results(self, test_results=None):
        """Wizualizacja wyników optymalizacji Hyperband"""
        print("\n--- GENEROWANIE WYKRESÓW ---")
        
        fig = plt.figure(figsize=(16, 10))
        
        # 1. Macierz pomyłek (jeśli dostępna)
        if test_results is not None:
            ax1 = plt.subplot(2, 3, 1)
            sns.heatmap(
                test_results['confusion_matrix'],
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=np.unique(test_results['y_test']),
                yticklabels=np.unique(test_results['y_test']),
                ax=ax1
            )
            ax1.set_title('Macierz Pomyłek - Zbiór Testowy', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Rzeczywista etykieta')
            ax1.set_xlabel('Przewidywana etykieta')
        
        # 2. Wpływ n_estimators
        if 'param_n_estimators' in self.halving_results:
            ax2 = plt.subplot(2, 3, 2)
            # Bezpieczne obsługiwanie wartości None
            n_est_raw = self.halving_results['param_n_estimators']
            n_est_values = []
            for val in n_est_raw:
                if val is not None and val not in n_est_values:
                    n_est_values.append(val)
            
            # Sortowanie
            n_est_values = sorted(n_est_values)
            
            n_est_scores = []
            for val in n_est_values:
                mask = n_est_raw == val
                scores = self.halving_results['mean_test_score'][mask]
                if len(scores) > 0:
                    n_est_scores.append(np.mean(scores))
            
            if len(n_est_values) > 0:
                ax2.plot(n_est_values, n_est_scores, marker='o', linewidth=2, markersize=8, color='#d62728')
                ax2.set_xlabel('n_estimators (liczba drzew)', fontweight='bold')
                ax2.set_ylabel('Średni F1-Score', fontweight='bold')
                ax2.set_title('Wpływ liczby drzew na wydajność', fontsize=12, fontweight='bold')
                
                y_min = min(n_est_scores) if len(n_est_scores) > 0 else 0
                y_max = max(n_est_scores) if len(n_est_scores) > 0 else 1
                y_range = y_max - y_min
                ax2.set_ylim([y_min - y_range*0.15, y_max + y_range*0.15])
                ax2.grid(True, alpha=0.3)
        
        # 3. Wpływ max_depth
        if 'param_max_depth' in self.halving_results:
            ax3 = plt.subplot(2, 3, 3)
            # Bezpieczne obsługiwanie wartości None
            max_depth_raw = self.halving_results['param_max_depth']
            max_depth_values = []
            for val in max_depth_raw:
                if val not in max_depth_values:
                    max_depth_values.append(val)
            
            # Sortowanie: najpierw liczby, potem None
            max_depth_values = sorted(
                max_depth_values, 
                key=lambda x: (x is None, x)
            )
            
            max_depth_scores = []
            for val in max_depth_values:
                if val is None:
                    mask = pd.isna(max_depth_raw)
                else:
                    mask = max_depth_raw == val
                scores = self.halving_results['mean_test_score'][mask]
                if len(scores) > 0:
                    max_depth_scores.append(np.mean(scores))
            
            if len(max_depth_values) > 0:
                labels = [str(v) if v is not None else 'None' for v in max_depth_values]
                ax3.bar(labels, max_depth_scores, color='#ff7f0e', alpha=0.7)
                ax3.set_xlabel('max_depth')
                ax3.set_ylabel('Średni F1-Score')
                ax3.set_title('Wpływ głębokości drzewa', fontsize=12, fontweight='bold')
                
                y_min = min(max_depth_scores) * 0.95 if len(max_depth_scores) > 0 else 0
                y_max = max(max_depth_scores) * 1.05 if len(max_depth_scores) > 0 else 1
                ax3.set_ylim([y_min, y_max])
                ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Rozkład wyników
        ax4 = plt.subplot(2, 3, 4)
        scores = self.halving_results['mean_test_score']
        ax4.hist(scores, bins=20, color='purple', alpha=0.7, edgecolor='black')
        ax4.axvline(np.max(scores), color='red', linestyle='--', linewidth=2, label='Najlepszy')
        ax4.set_xlabel('F1-Score')
        ax4.set_ylabel('Liczba kombinacji')
        ax4.set_title('Rozkład wyników Hyperband', fontsize=12, fontweight='bold')
        ax4.legend()
        
        x_min = np.min(scores) * 0.99
        x_max = np.max(scores) * 1.01
        ax4.set_xlim([x_min, x_max])
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Top 10 kombinacji
        ax5 = plt.subplot(2, 3, 5)
        top_indices = np.argsort(self.halving_results['mean_test_score'])[-10:][::-1]
        top_scores = self.halving_results['mean_test_score'][top_indices]
        top_labels = [f"#{i+1}" for i in range(len(top_scores))]
        
        ax5.barh(top_labels, top_scores, color='#2ca02c', alpha=0.7)
        ax5.set_xlabel('F1-Score')
        ax5.set_title('Top 10 najlepszych kombinacji', fontsize=12, fontweight='bold')
        ax5.invert_yaxis()
        
        x_min = np.min(top_scores) * 0.98
        x_max = np.max(top_scores) * 1.02
        ax5.set_xlim([x_min, x_max])
        ax5.grid(True, alpha=0.3, axis='x')
        
        # 6. Metryki (jeśli dostępne)
        if test_results is not None:
            ax6 = plt.subplot(2, 3, 6)
            metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            metrics_values = [
                test_results['accuracy'],
                test_results['precision'],
                test_results['recall'],
                test_results['f1']
            ]
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            bars = ax6.bar(metrics_names, metrics_values, color=colors, alpha=0.7, edgecolor='black')
            ax6.set_ylabel('Wartość')
            ax6.set_title('Metryki na zbiorze testowym', fontsize=12, fontweight='bold')
            ax6.set_ylim([0, 1.1])
            
            for bar, value in zip(bars, metrics_values):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.3f}',
                        ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(results_png_path, dpi=150, bbox_inches='tight')
        print(f"✓ Wykresy zapisane: {results_png_path}")
        plt.show()
    
    def save_model(self, filepath):
        """Zapisanie najlepszego modelu"""
        if self.best_model is None:
            print("Błąd: Brak modelu do zapisania!")
            return
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.best_model, f)
        print(f"✓ Model zapisany: {filepath}")
    
    def save_summary_report(self, filename=results_report_path):
        """Zapisanie raportu podsumowującego"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("RAPORT: OPTYMALIZACJA HYPERBAND (Successive Halving)\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Data i czas: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("METODOLOGIA:\n")
            f.write("-"*70 + "\n")
            f.write("Hyperband iteracyjnie testuje kombinacje parametrów i eliminuje\n")
            f.write("najsłabsze kombinacje, inwestując więcej zasobów w obiecujące.\n")
            f.write("To znacznie szybsze niż tradycyjny Grid Search.\n\n")
            
            f.write("NAJLEPSZE ZNALEZIONE PARAMETRY:\n")
            f.write("-"*70 + "\n")
            for param, value in self.best_params.items():
                f.write(f"  {param}: {value}\n")
            
            f.write(f"\nNajlepszy F1-Score: {np.max(self.halving_results['mean_test_score']):.4f}\n\n")
            
            if self.is_evaluated:
                f.write("WYNIKI NA ZBIORZE TESTOWYM:\n")
                f.write("-"*70 + "\n")
                f.write(f"  Accuracy:  {self.test_accuracy:.4f} ({self.test_accuracy*100:.2f}%)\n")
                f.write(f"  Precision: {self.test_precision:.4f}\n")
                f.write(f"  Recall:    {self.test_recall:.4f}\n")
                f.write(f"  F1-Score:  {self.test_f1:.4f}\n")
        
        print(f"✓ Raport zapisany: {filename}")
    
    def run_full_pipeline(self, learning_set_path, testing_set_path):
        """Uruchomienie pełnego pipeline'u"""
        # 🕐 POCZĄTEK POMIARU CZASU
        pipeline_start_time = time.time()
        
        print("\n" + "="*70)
        print("HYPERBAND OPTIMIZATION - SUCCESSIVE HALVING")
        print("="*70)
        print("\n💾 WYNIKI BĘDĄ ZAPISANE W:")
        print(f"  • {model_save_path}")
        print(f"  • {results_png_path}")
        print(f"  • {results_report_path}")
        print("\n" + "="*70)
        
        # Wczytanie danych
        print("\n--- WCZYTYWANIE DANYCH ---")
        X_train, y_train = self.load_data(learning_set_path)
        X_test, y_test = self.load_data(testing_set_path)
        
        # Optymalizacja Hyperband
        halving_grid, optimization_time = self.hyperband_optimization(X_train, y_train)
        
        # Ewaluacja na zbiorze testowym
        test_results = self.evaluate_on_test_set(X_test, y_test)
        
        # Zapisanie wartości dla raportu
        self.test_accuracy = test_results['accuracy']
        self.test_precision = test_results['precision']
        self.test_recall = test_results['recall']
        self.test_f1 = test_results['f1']
        
        # Interpretacja
        self.interpret_results()
        
        # Wizualizacja
        self.visualize_results(test_results)
        
        # Zapisanie modelu
        self.save_model(model_save_path)
        
        # Zapisanie raportu
        self.save_summary_report()
        
        # 🕐 KONIEC POMIARU CZASU
        pipeline_end_time = time.time()
        pipeline_elapsed = pipeline_end_time - pipeline_start_time
        
        # Podsumowanie
        print("\n" + "="*70)
        print("PODSUMOWANIE")
        print("="*70)
        print(f"\n📊 Wyniki na zbiorze testowym:")
        print(f"   Accuracy:  {test_results['accuracy']*100:.2f}%")
        print(f"   Precision: {test_results['precision']:.4f}")
        print(f"   Recall:    {test_results['recall']:.4f}")
        print(f"   F1-Score:  {test_results['f1']:.4f}")
        print(f"\n💾 Model zapisany: {model_save_path}")
        print(f"📈 Wykresy zapisane: {results_png_path}")
        print(f"📄 Raport zapisany: {results_report_path}")
        print(f"\n⏱️  CAŁKOWITY CZAS WYKONANIA: {pipeline_elapsed:.2f} sekund ({pipeline_elapsed/60:.2f} minut)")
        print("\n✓ Pipeline zakończony pomyślnie!")
        
        # Zwrócenie metryki czasu dla porównania
        return {
            'elapsed_time': pipeline_elapsed,
            'accuracy': test_results['accuracy'],
            'precision': test_results['precision'],
            'recall': test_results['recall'],
            'f1': test_results['f1'],
            'best_params': self.best_params,
            'method': 'Hyperband (Successive Halving)',
            'optimization_time': optimization_time
        }


def main():
    # Ścieżki do plików
    learning_set_path = "tiny_HaGRID/learning/learning_set_2hands.csv"
    testing_set_path = "tiny_HaGRID/testing/testing_set_2hands.csv"
    
    # Sprawdzenie czy pliki istnieją
    if not os.path.exists(learning_set_path):
        print(f"Błąd: Plik {learning_set_path} nie znaleziony!")
        return None
    
    if not os.path.exists(testing_set_path):
        print(f"Błąd: Plik {testing_set_path} nie znaleziony!")
        return None
    
    # Uruchomienie
    hyperband = HyperbandOptimization(random_state=42)
    results = hyperband.run_full_pipeline(learning_set_path, testing_set_path)
    return results


if __name__ == "__main__":
    main()
