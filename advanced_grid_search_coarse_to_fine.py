import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import time
from datetime import datetime

class AdvancedGridSearchCoarseFine:
    def __init__(self, random_state=42):
        """
        Inicjalizacja zaawansowanego grid search z metodą coarse-to-fine
        
        :param random_state: Ziarno losowości dla powtarzalności
        """
        self.random_state = random_state
        self.best_model = None
        self.best_params = None
        self.coarse_results = None
        self.fine_results = None
        self.cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        self.is_evaluated = False
    
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
    
    def coarse_grid_search(self, X_train, y_train):
        """
        Faza 1: Grid Search ze SZEROKIM zakresem parametrów (COARSE)
        """
        print("\n" + "="*70)
        print("FAZA 1: GRID SEARCH COARSE (Szerokie zakresy)")
        print("="*70)
        
        coarse_params = {
            'n_estimators': [50, 75, 100, 150, 200, 300, 400, 500],
            'max_depth': [5, 8, 10, 15, 20, 30, 40, 50, None],
            'min_samples_split': [2, 3, 4, 5, 7, 10],
            'min_samples_leaf': [1, 2, 3, 4, 5],
            'max_features': ['sqrt', 'log2', None]
        }
        
        # coarse_params = {
        #     'n_estimators': [50, 100, 200, 500],
            
        #     'max_depth': [5, 10, 20, 50, None],
        #     'min_samples_split': [2, 5, 7, 10],
        #     'min_samples_leaf': [1, 2, 4, 5],
        #     'max_features': ['sqrt', 'log2', None]
        # }




        print("\nTestowane zakresy parametrów (COARSE):")
        total_combinations = 1
        for param, values in coarse_params.items():
            print(f"  - {param}: {values}")
            total_combinations *= len(values)
        print(f"\n  💡 Liczba kombinacji do testowania: {total_combinations}")
        print(f"  💡 Z walidacją krzyżową (5-fold): ~{total_combinations * 5} treningów")
        #print(f"  ⏱️  To może potrwać 10-30 minut w zależności od rozmiaru danych...")
        
        # Grid Search
        rf_base = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        
        start_time = time.time()
        print("\n⏳ Trenowanie modeli (może to chwilę potrwać)...")
        
        grid_search = GridSearchCV(
            rf_base,
            coarse_params,
            cv=self.cv_strategy,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        elapsed = time.time() - start_time
        
        print(f"\n✓ Faza coarse ukończona w {elapsed:.2f} sekund ({elapsed/60:.2f} minut)")
        
        self.coarse_results = grid_search.cv_results_
        self.best_params = grid_search.best_params_
        
        print(f"\n🏆 Najlepsze parametry (COARSE):")
        print(f"   F1-Score: {grid_search.best_score_:.4f}")
        for param, value in self.best_params.items():
            print(f"   {param}: {value}")
        
        return grid_search
    
    def fine_grid_search(self, X_train, y_train, coarse_best_params):
        """
        Faza 2: Grid Search z WĘŻSZYM zakresem parametrów wokół najlepszych (FINE)
        """
        print("\n" + "="*70)
        print("FAZA 2: GRID SEARCH FINE (Wąskie zakresy wokół najlepszych)")
        print("="*70)
        
        # Budowanie Fine zakreśów na podstawie Coarse wyników
        fine_params = {
            'n_estimators': self._create_fine_range(
                coarse_best_params['n_estimators'], 
                [50, 75, 100, 150, 200, 300, 400, 500]
            ),
            'max_depth': self._create_fine_range(
                coarse_best_params['max_depth'],
                [5, 8, 10, 15, 20, 30, 40, 50, None]
            ),
            'min_samples_split': self._create_fine_range(
                coarse_best_params['min_samples_split'],
                [2, 3, 4, 5, 7, 10]
            ),
            'min_samples_leaf': self._create_fine_range(
                coarse_best_params['min_samples_leaf'],
                [1, 2, 3, 4, 5]
            ),
            'max_features': [coarse_best_params['max_features']]
        }
        
        print("\nTestowane zakresy parametrów (FINE):")
        total_combinations = 1
        for param, values in fine_params.items():
            print(f"  - {param}: {values}")
            total_combinations *= len(values)
        print(f"\n  💡 Liczba kombinacji do testowania: {total_combinations}")
        print(f"  💡 Z walidacją krzyżową (5-fold): ~{total_combinations * 5} treningów")
        print(f"  ⏱️  To powinno potrwać 2-10 minut...")
        
        rf_base = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        
        start_time = time.time()
        print("\n⏳ Trenowanie modeli (faza fine)...")
        
        grid_search_fine = GridSearchCV(
            rf_base,
            fine_params,
            cv=self.cv_strategy,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search_fine.fit(X_train, y_train)
        elapsed = time.time() - start_time
        
        print(f"\n✓ Faza fine ukończona w {elapsed:.2f} sekund ({elapsed/60:.2f} minut)")
        
        self.fine_results = grid_search_fine.cv_results_
        self.best_params = grid_search_fine.best_params_
        self.best_model = grid_search_fine.best_estimator_
        
        print(f"\n🏆 Najlepsze parametry (FINE):")
        print(f"   F1-Score: {grid_search_fine.best_score_:.4f}")
        for param, value in self.best_params.items():
            print(f"   {param}: {value}")
        
        return grid_search_fine
    
    def _create_fine_range(self, best_value, all_values):
        """Tworzy wąski zakres wokół najlepszej wartości - zwraca ±1 sąsiadów"""
        if best_value is None:
            return [None]
        
        if isinstance(best_value, str):
            return [best_value]
        
        try:
            idx = all_values.index(best_value)
        except ValueError:
            # Jeśli wartość nie ma, zwróć ją i sąsiadów
            return [best_value]
        
        fine_range = []
        
        # Wartość poprzednia
        if idx > 0:
            fine_range.append(all_values[idx - 1])
        
        # Wartość obecna
        fine_range.append(best_value)
        
        # Wartość następna
        if idx < len(all_values) - 1:
            fine_range.append(all_values[idx + 1])
        
        return fine_range
    
    def evaluate_on_test_set(self, X_test, y_test):
        """Ewaluacja najlepszego modelu na zbiorze testowym"""
        if self.best_model is None:
            print("Błąd: Najpierw wykonaj grid search!")
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
        """Interpretacja wyników grid search"""
        print("\n" + "="*70)
        print("INTERPRETACJA WYNIKÓW")
        print("="*70)
        
        print(f"\n📊 Najlepsze znalezione parametry:")
        print(f"   F1-Score: {max(self.fine_results['mean_test_score']):.4f}")
        for param, value in self.best_params.items():
            print(f"   {param}: {value}")
        
        print(f"\n📈 Analiza wpływu parametrów:")
        
        # Analiza wrażliwości na n_estimators
        if 'param_n_estimators' in self.fine_results:
            n_est_values = np.unique(self.fine_results['param_n_estimators'])
            n_est_scores = []
            for val in n_est_values:
                mask = self.fine_results['param_n_estimators'] == val
                scores = self.fine_results['mean_test_score'][mask]
                n_est_scores.append(np.mean(scores))
            
            print(f"\n   n_estimators (liczba drzew):")
            for val, score in zip(n_est_values, n_est_scores):
                if val is not None:
                    print(f"      {val}: {score:.4f}")
        
        # Analiza wrażliwości na max_depth
        if 'param_max_depth' in self.fine_results:
            max_depth_values = np.unique(self.fine_results['param_max_depth'])
            max_depth_scores = []
            for val in max_depth_values:
                mask = self.fine_results['param_max_depth'] == val
                scores = self.fine_results['mean_test_score'][mask]
                max_depth_scores.append(np.mean(scores))
            
            print(f"\n   max_depth (maksymalna głębokość drzewa):")
            for val, score in zip(max_depth_values, max_depth_scores):
                if val is not None:
                    print(f"      {val}: {score:.4f}")
        
        print(f"\n💡 Wnioski:")
        print(f"   - Najlepszy F1-Score uzyskano z wybranymi parametrami")
        print(f"   - Model zoptymalizowany metodą coarse-to-fine")
        print(f"   - Parametry dobrane na podstawie walidacji krzyżowej (K-fold=5)")
    
    def visualize_results(self, test_results=None):
        """Wizualizacja wyników grid search"""
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
        if 'param_n_estimators' in self.fine_results:
            ax2 = plt.subplot(2, 3, 2)
            n_est_values = np.unique(self.fine_results['param_n_estimators'])
            n_est_scores = []
            for val in n_est_values:
                mask = self.fine_results['param_n_estimators'] == val
                scores = self.fine_results['mean_test_score'][mask]
                n_est_scores.append(np.mean(scores))
            
            ax2.plot(n_est_values, n_est_scores, marker='o', linewidth=2, markersize=8, color='#2ca02c')
            ax2.set_xlabel('n_estimators (liczba drzew)', fontweight='bold')
            ax2.set_ylabel('Średni F1-Score', fontweight='bold')
            ax2.set_title('Wpływ liczby drzew na wydajność', fontsize=12, fontweight='bold')
            
            # Bardziej agresywne dopasowanie skali osi Y
            y_min = min(n_est_scores)
            y_max = max(n_est_scores)
            y_range = y_max - y_min
            ax2.set_ylim([y_min - y_range*0.15, y_max + y_range*0.15])
            ax2.grid(True, alpha=0.3)
            
            # Dodaj wartości na punktach
            for x, y in zip(n_est_values, n_est_scores):
                ax2.text(x, y, f'{y:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 3. Wpływ max_depth
        if 'param_max_depth' in self.fine_results:
            ax3 = plt.subplot(2, 3, 3)
            max_depth_values = np.unique(self.fine_results['param_max_depth'])
            max_depth_scores = []
            for val in max_depth_values:
                mask = self.fine_results['param_max_depth'] == val
                scores = self.fine_results['mean_test_score'][mask]
                max_depth_scores.append(np.mean(scores))
            
            # Konwersja None na wartość dla wykresu
            labels = [str(v) if v is not None else 'None' for v in max_depth_values]
            ax3.bar(labels, max_depth_scores, color='steelblue', alpha=0.7)
            ax3.set_xlabel('max_depth')
            ax3.set_ylabel('Średni F1-Score')
            ax3.set_title('Wpływ głębokości drzewa', fontsize=12, fontweight='bold')
            
            # Dopasowanie skali osi Y
            y_min = min(max_depth_scores) * 0.95
            y_max = max(max_depth_scores) * 1.05
            ax3.set_ylim([y_min, y_max])
            ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Rozkład wyników grid search
        ax4 = plt.subplot(2, 3, 4)
        ax4.hist(self.fine_results['mean_test_score'], bins=20, color='green', alpha=0.7, edgecolor='black')
        ax4.axvline(np.max(self.fine_results['mean_test_score']), color='red', linestyle='--', linewidth=2, label='Najlepszy')
        ax4.set_xlabel('F1-Score')
        ax4.set_ylabel('Liczba kombinacji')
        ax4.set_title('Rozkład wyników Grid Search', fontsize=12, fontweight='bold')
        ax4.legend()
        
        # Dopasowanie skali osi X
        x_min = np.min(self.fine_results['mean_test_score']) * 0.99
        x_max = np.max(self.fine_results['mean_test_score']) * 1.01
        ax4.set_xlim([x_min, x_max])
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Top 10 kombinacji parametrów
        ax5 = plt.subplot(2, 3, 5)
        top_indices = np.argsort(self.fine_results['mean_test_score'])[-10:][::-1]
        top_scores = self.fine_results['mean_test_score'][top_indices]
        top_labels = [f"#{i+1}" for i in range(len(top_scores))]
        
        ax5.barh(top_labels, top_scores, color='orange', alpha=0.7)
        ax5.set_xlabel('F1-Score')
        ax5.set_title('Top 10 najlepszych kombinacji', fontsize=12, fontweight='bold')
        ax5.invert_yaxis()
        
        # Dopasowanie skali osi X
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
            
            # Dodaj wartości na słupkach
            for bar, value in zip(bars, metrics_values):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.3f}',
                        ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('grid_search_coarse_to_fine_results.png', dpi=150, bbox_inches='tight')
        print("✓ Wykresy zapisane: grid_search_coarse_to_fine_results.png")
        plt.show()
    
    def save_model(self, filepath):
        """Zapisanie najlepszego modelu"""
        if self.best_model is None:
            print("Błąd: Brak modelu do zapisania!")
            return
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.best_model, f)
        print(f"✓ Model zapisany: {filepath}")
    
    def save_summary_report(self, filename='grid_search_report.txt'):
        """Zapisanie raportu podsumowującego"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("RAPORT: ZAAWANSOWANY GRID SEARCH COARSE-TO-FINE\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Data i czas: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("NAJLEPSZE ZNALEZIONE PARAMETRY:\n")
            f.write("-"*70 + "\n")
            for param, value in self.best_params.items():
                f.write(f"  {param}: {value}\n")
            
            f.write(f"\nNajlepszy F1-Score (walidacja krzyżowa): {np.max(self.fine_results['mean_test_score']):.4f}\n\n")
            
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
        print("\n" + "="*70)
        print("ZAAWANSOWANY GRID SEARCH METODĄ COARSE-TO-FINE")
        print("="*70)
        # print("\n📊 INFORMACJA O TRENINGU:")
        # print("  • COARSE: ~5670 kombinacji parametrów")
        # print("  • FINE: ~20-50 kombinacji parametrów")
        # print("  • Każda kombinacja testowana z K-Fold=5")
        # print("  • ŁĄCZNIE: ~28000-30000 treningów modeli")
        # print("\n⏱️  SZACUNKOWY CZAS:")
        # print("  • FAZA COARSE: 10-30 minut")
        # print("  • FAZA FINE: 2-10 minut")
        # print("  • RAZEM: 15-40 minut")
        print("\n💾 WYNIKI BĘDĄ ZAPISANE W:")
        print("  • best_model_coarse_to_fine.pkl")
        print("  • grid_search_coarse_to_fine_results.png")
        print("  • grid_search_report.txt")
        print("\n" + "="*70)
        
        # Wczytanie danych
        print("\n--- WCZYTYWANIE DANYCH ---")
        X_train, y_train = self.load_data(learning_set_path)
        X_test, y_test = self.load_data(testing_set_path)
        
        # Faza Coarse
        coarse_grid = self.coarse_grid_search(X_train, y_train)
        
        # Faza Fine
        fine_grid = self.fine_grid_search(X_train, y_train, self.best_params)
        
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
        self.save_model('best_model_coarse_to_fine.pkl')
        
        # Zapisanie raportu
        self.save_summary_report()
        
        # Podsumowanie
        print("\n" + "="*70)
        print("PODSUMOWANIE")
        print("="*70)
        print(f"\n📊 Wyniki na zbiorze testowym:")
        print(f"   Accuracy:  {test_results['accuracy']*100:.2f}%")
        print(f"   Precision: {test_results['precision']:.4f}")
        print(f"   Recall:    {test_results['recall']:.4f}")
        print(f"   F1-Score:  {test_results['f1']:.4f}")
        print(f"\n💾 Model zapisany: best_model_coarse_to_fine.pkl")
        print(f"📈 Wykresy zapisane: grid_search_coarse_to_fine_results.png")
        print(f"📄 Raport zapisany: grid_search_report.txt")
        print("\n✓ Pipeline zakończony pomyślnie!")


def main():
    # Ścieżki do plików
    learning_set_path = "tiny_HaGRID/learning/learning_set.csv"
    testing_set_path = "tiny_HaGRID/testing/testing_set.csv"
    
    # Sprawdzenie czy pliki istnieją
    if not os.path.exists(learning_set_path):
        print(f"Błąd: Plik {learning_set_path} nie znaleziony!")
        return
    
    if not os.path.exists(testing_set_path):
        print(f"Błąd: Plik {testing_set_path} nie znaleziony!")
        return
    
    # Uruchomienie
    grid_search = AdvancedGridSearchCoarseFine(random_state=42)
    grid_search.run_full_pipeline(learning_set_path, testing_set_path)


if __name__ == "__main__":
    main()
