import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

# Importuj obie metody
from advanced_grid_search_coarse_to_fine import AdvancedGridSearchCoarseFine
from hyperband_optimization import HyperbandOptimization

class ComparisonRunner:
    """Klasa do uruchomienia i porównania obu metod optymalizacji"""
    
    def __init__(self, learning_set_path, testing_set_path):
        self.learning_set_path = learning_set_path
        self.testing_set_path = testing_set_path
        self.results_grid_search = None
        self.results_hyperband = None
        self.comparison_df = None
    
    def run_grid_search(self):
        """Uruchomienie Grid Search Coarse-to-Fine"""
        print("\n" + "="*80)
        print("🔍 METODA 1: GRID SEARCH COARSE-TO-FINE")
        print("="*80)
        
        grid_search = AdvancedGridSearchCoarseFine(random_state=42)
        self.results_grid_search = grid_search.run_full_pipeline(
            self.learning_set_path,
            self.testing_set_path
        )
        
        return self.results_grid_search
    
    def run_hyperband(self):
        """Uruchomienie Hyperband Optimization"""
        print("\n\n" + "="*80)
        print("🔍 METODA 2: HYPERBAND OPTIMIZATION (Successive Halving)")
        print("="*80)
        
        hyperband = HyperbandOptimization(random_state=42)
        self.results_hyperband = hyperband.run_full_pipeline(
            self.learning_set_path,
            self.testing_set_path
        )
        
        return self.results_hyperband
    
    def create_comparison_table(self):
        """Tworzenie tabeli porównawczej"""
        print("\n\n" + "="*80)
        print("📊 TABELA PORÓWNAWCZA")
        print("="*80 + "\n")
        
        # Tworzenie DataFrame
        comparison_data = {
            'Metryka': [
                'Accuracy',
                'Precision',
                'Recall',
                'F1-Score',
                'Czas wykonania (s)',
                'Czas wykonania (min)'
            ],
            'Grid Search': [
                f"{self.results_grid_search['accuracy']:.4f} ({self.results_grid_search['accuracy']*100:.2f}%)",
                f"{self.results_grid_search['precision']:.4f}",
                f"{self.results_grid_search['recall']:.4f}",
                f"{self.results_grid_search['f1']:.4f}",
                f"{self.results_grid_search['elapsed_time']:.2f}",
                f"{self.results_grid_search['elapsed_time']/60:.2f}"
            ],
            'Hyperband': [
                f"{self.results_hyperband['accuracy']:.4f} ({self.results_hyperband['accuracy']*100:.2f}%)",
                f"{self.results_hyperband['precision']:.4f}",
                f"{self.results_hyperband['recall']:.4f}",
                f"{self.results_hyperband['f1']:.4f}",
                f"{self.results_hyperband['elapsed_time']:.2f}",
                f"{self.results_hyperband['elapsed_time']/60:.2f}"
            ]
        }
        
        self.comparison_df = pd.DataFrame(comparison_data)
        
        print(self.comparison_df.to_string(index=False))
        print("\n")
        
        # Obliczenie przyspieszenia
        speedup = self.results_grid_search['elapsed_time'] / self.results_hyperband['elapsed_time']
        print(f"⚡ Przyspieszenie (Speedup): {speedup:.2f}x")
        print(f"   Grid Search: {self.results_grid_search['elapsed_time']:.2f}s")
        print(f"   Hyperband: {self.results_hyperband['elapsed_time']:.2f}s")
        print(f"   Hyperband jest {speedup:.2f} razy szybszy\n")
        
        # Różnica w dokładności
        acc_diff = abs(self.results_grid_search['accuracy'] - self.results_hyperband['accuracy'])
        f1_diff = abs(self.results_grid_search['f1'] - self.results_hyperband['f1'])
        
        print(f"📈 Różnica w dokładności:")
        print(f"   Accuracy: {acc_diff:.4f} ({acc_diff*100:.2f}%)")
        print(f"   F1-Score: {f1_diff:.4f}")
        
        if acc_diff < 0.01 and f1_diff < 0.01:
            print(f"   ✓ Wyniki praktycznie identyczne!")
        elif acc_diff < 0.05 and f1_diff < 0.05:
            print(f"   ✓ Wyniki bardzo podobne!")
        else:
            print(f"   ⚠ Zauważalna różnica w wynikach")
        
        return self.comparison_df
    
    def save_comparison_to_file(self, filename='comparison_results.txt'):
        """Zapisanie porównania do pliku"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("PORÓWNANIE: GRID SEARCH vs HYPERBAND\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Data i czas: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("TABELA PORÓWNAWCZA\n")
            f.write("-"*80 + "\n")
            f.write(self.comparison_df.to_string(index=False))
            f.write("\n\n")
            
            # Analiza
            speedup = self.results_grid_search['elapsed_time'] / self.results_hyperband['elapsed_time']
            acc_diff = abs(self.results_grid_search['accuracy'] - self.results_hyperband['accuracy'])
            f1_diff = abs(self.results_grid_search['f1'] - self.results_hyperband['f1'])
            
            f.write("ANALIZA PRZYSPIESZENIA\n")
            f.write("-"*80 + "\n")
            f.write(f"Grid Search czas: {self.results_grid_search['elapsed_time']:.2f}s ({self.results_grid_search['elapsed_time']/60:.2f}min)\n")
            f.write(f"Hyperband czas: {self.results_hyperband['elapsed_time']:.2f}s ({self.results_hyperband['elapsed_time']/60:.2f}min)\n")
            f.write(f"Przyspieszenie: {speedup:.2f}x\n")
            f.write(f"Oszczędność czasu: {self.results_grid_search['elapsed_time'] - self.results_hyperband['elapsed_time']:.2f}s\n\n")
            
            f.write("RÓŻNICA W DOKŁADNOŚCI\n")
            f.write("-"*80 + "\n")
            f.write(f"Accuracy: {acc_diff:.4f} ({acc_diff*100:.2f}%)\n")
            f.write(f"F1-Score: {f1_diff:.4f}\n\n")
            
            f.write("PARAMETRY GRID SEARCH\n")
            f.write("-"*80 + "\n")
            for param, value in self.results_grid_search['best_params'].items():
                f.write(f"  {param}: {value}\n")
            f.write("\n")
            
            f.write("PARAMETRY HYPERBAND\n")
            f.write("-"*80 + "\n")
            for param, value in self.results_hyperband['best_params'].items():
                f.write(f"  {param}: {value}\n")
            f.write("\n")
            
            f.write("WNIOSKI\n")
            f.write("-"*80 + "\n")
            f.write(f"1. Szybkość: Hyperband jest {speedup:.2f}x szybszy\n")
            
            if acc_diff < 0.01 and f1_diff < 0.01:
                f.write(f"2. Dokładność: Wyniki praktycznie identyczne\n")
                f.write(f"3. Zalecenie: HYPERBAND jest lepszym wyborem (szybszy, równie dokładny)\n")
            elif acc_diff < 0.05 and f1_diff < 0.05:
                f.write(f"2. Dokładność: Wyniki bardzo podobne\n")
                f.write(f"3. Zalecenie: HYPERBAND jest lepszym wyborem (szybszy, porównywalna dokładność)\n")
            else:
                if self.results_grid_search['f1'] > self.results_hyperband['f1']:
                    f.write(f"2. Dokładność: Grid Search lepszy o {f1_diff:.4f} w F1-Score\n")
                    f.write(f"3. Zalecenie: Trade-off między szybkością a dokładnością\n")
                else:
                    f.write(f"2. Dokładność: Hyperband lepszy o {f1_diff:.4f} w F1-Score\n")
                    f.write(f"3. Zalecenie: HYPERBAND jest lepszym wyborem\n")
        
        print(f"✓ Porównanie zapisane do: {filename}")
    
    def visualize_comparison(self):
        """Wizualizacja porównania obu metod"""
        print("\n--- GENEROWANIE WYKRESÓW PORÓWNAWCZYCH ---")
        
        fig = plt.figure(figsize=(16, 10))
        
        methods = ['Grid Search', 'Hyperband']
        
        # 1. Porównanie czasu
        ax1 = plt.subplot(2, 3, 1)
        times = [
            self.results_grid_search['elapsed_time'],
            self.results_hyperband['elapsed_time']
        ]
        colors_time = ['#1f77b4', '#ff7f0e']
        bars1 = ax1.bar(methods, times, color=colors_time, alpha=0.8, edgecolor='black', linewidth=2)
        ax1.set_ylabel('Czas (sekundy)', fontweight='bold')
        ax1.set_title('Porównanie czasu wykonania', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Dodaj wartości na słupkach
        for bar, time in zip(bars1, times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time:.1f}s\n({time/60:.2f}m)',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 2. Przyspieszenie
        ax2 = plt.subplot(2, 3, 2)
        speedup = self.results_grid_search['elapsed_time'] / self.results_hyperband['elapsed_time']
        ax2.barh(['Przyspieszenie'], [speedup], color='#2ca02c', alpha=0.8, edgecolor='black', linewidth=2)
        ax2.set_xlabel('Mnożnik przyspieszenia', fontweight='bold')
        ax2.set_title('Hyperband vs Grid Search', fontsize=12, fontweight='bold')
        ax2.set_xlim([0, speedup + 1])
        
        ax2.text(speedup/2, 0, f'{speedup:.2f}x', ha='center', va='center',
                fontweight='bold', fontsize=14, color='white',
                bbox=dict(boxstyle='round', facecolor='green', alpha=0.7))
        
        # 3. Porównanie Accuracy
        ax3 = plt.subplot(2, 3, 3)
        accuracies = [
            self.results_grid_search['accuracy'],
            self.results_hyperband['accuracy']
        ]
        colors_acc = ['#1f77b4', '#ff7f0e']
        bars3 = ax3.bar(methods, accuracies, color=colors_acc, alpha=0.8, edgecolor='black', linewidth=2)
        ax3.set_ylabel('Accuracy', fontweight='bold')
        ax3.set_title('Porównanie dokładności (Accuracy)', fontsize=12, fontweight='bold')
        ax3.set_ylim([0, 1.1])
        ax3.grid(True, alpha=0.3, axis='y')
        
        for bar, acc in zip(bars3, accuracies):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.4f}\n({acc*100:.2f}%)',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 4. Porównanie F1-Score
        ax4 = plt.subplot(2, 3, 4)
        f1_scores = [
            self.results_grid_search['f1'],
            self.results_hyperband['f1']
        ]
        colors_f1 = ['#1f77b4', '#ff7f0e']
        bars4 = ax4.bar(methods, f1_scores, color=colors_f1, alpha=0.8, edgecolor='black', linewidth=2)
        ax4.set_ylabel('F1-Score', fontweight='bold')
        ax4.set_title('Porównanie F1-Score', fontsize=12, fontweight='bold')
        ax4.set_ylim([0, 1.1])
        ax4.grid(True, alpha=0.3, axis='y')
        
        for bar, f1 in zip(bars4, f1_scores):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{f1:.4f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 5. Wszystkie metryki (radar chart alternatywnie)
        ax5 = plt.subplot(2, 3, 5)
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        gs_values = [
            self.results_grid_search['accuracy'],
            self.results_grid_search['precision'],
            self.results_grid_search['recall'],
            self.results_grid_search['f1']
        ]
        hb_values = [
            self.results_hyperband['accuracy'],
            self.results_hyperband['precision'],
            self.results_hyperband['recall'],
            self.results_hyperband['f1']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars_gs = ax5.bar(x - width/2, gs_values, width, label='Grid Search', color='#1f77b4', alpha=0.8, edgecolor='black')
        bars_hb = ax5.bar(x + width/2, hb_values, width, label='Hyperband', color='#ff7f0e', alpha=0.8, edgecolor='black')
        
        ax5.set_ylabel('Wartość', fontweight='bold')
        ax5.set_title('Porównanie wszystkich metryk', fontsize=12, fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels(metrics, rotation=15, ha='right')
        ax5.set_ylim([0, 1.1])
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Efficiency Score (Trade-off czas vs dokładność)
        ax6 = plt.subplot(2, 3, 6)
        
        # Normalizacja: F1-Score na [0, 1], Czas odwrotnie
        norm_f1_gs = self.results_grid_search['f1']
        norm_f1_hb = self.results_hyperband['f1']
        
        # Szybkość (odwrotnie - mniej czasu = wyższy score)
        max_time = max(self.results_grid_search['elapsed_time'], self.results_hyperband['elapsed_time'])
        speed_gs = 1 - (self.results_grid_search['elapsed_time'] / max_time)
        speed_hb = 1 - (self.results_hyperband['elapsed_time'] / max_time)
        
        # Efficiency Score = średnia ważona (60% F1, 40% Speed)
        eff_gs = 0.6 * norm_f1_gs + 0.4 * speed_gs
        eff_hb = 0.6 * norm_f1_hb + 0.4 * speed_hb
        
        colors_eff = ['#1f77b4', '#ff7f0e']
        bars6 = ax6.bar(methods, [eff_gs, eff_hb], color=colors_eff, alpha=0.8, edgecolor='black', linewidth=2)
        ax6.set_ylabel('Efficiency Score', fontweight='bold')
        ax6.set_title('Efficiency Score (60% F1 + 40% Speed)', fontsize=12, fontweight='bold')
        ax6.set_ylim([0, 1.1])
        ax6.grid(True, alpha=0.3, axis='y')
        
        for bar, eff in zip(bars6, [eff_gs, eff_hb]):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{eff:.3f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('comparison_grid_search_vs_hyperband.png', dpi=150, bbox_inches='tight')
        print("✓ Wykresy porównawcze zapisane: comparison_grid_search_vs_hyperband.png")
        plt.show()
    
    def print_summary(self):
        """Wydruk podsumowania"""
        print("\n" + "="*80)
        print("📋 PODSUMOWANIE PORÓWNANIA")
        print("="*80 + "\n")
        
        speedup = self.results_grid_search['elapsed_time'] / self.results_hyperband['elapsed_time']
        acc_diff = abs(self.results_grid_search['accuracy'] - self.results_hyperband['accuracy'])
        f1_diff = abs(self.results_grid_search['f1'] - self.results_hyperband['f1'])
        
        print(f"⚡ SZYBKOŚĆ:")
        print(f"   Grid Search: {self.results_grid_search['elapsed_time']:.2f}s ({self.results_grid_search['elapsed_time']/60:.2f}min)")
        print(f"   Hyperband: {self.results_hyperband['elapsed_time']:.2f}s ({self.results_hyperband['elapsed_time']/60:.2f}min)")
        print(f"   → Hyperband jest {speedup:.2f}x szybszy\n")
        
        print(f"📊 DOKŁADNOŚĆ:")
        print(f"   Accuracy: {acc_diff*100:.2f}% różnicy")
        print(f"   F1-Score: {f1_diff:.4f} różnicy\n")
        
        print(f"💡 ZALECENIE:")
        if speedup > 2 and f1_diff < 0.01:
            print(f"   ✅ HYPERBAND jest zdecydowanie lepszym wyborem!")
            print(f"   (Szybszy {speedup:.1f}x, praktycznie identyczna dokładność)")
        elif speedup > 1.5 and f1_diff < 0.02:
            print(f"   ✅ HYPERBAND jest lepszym wyborem")
            print(f"   (Szybszy {speedup:.1f}x, bardzo podobna dokładność)")
        elif speedup > 1 and self.results_hyperband['f1'] >= self.results_grid_search['f1']:
            print(f"   ✅ HYPERBAND jest preferowany")
            print(f"   (Szybszy, porównywalna lub lepsza dokładność)")
        else:
            print(f"   ⚖️  Trade-off między szybkością a dokładnością")
            print(f"   Wybór zależy od priorytetów aplikacji")
        
        print("\n" + "="*80 + "\n")


def main():
    # Ścieżki do plików
    learning_set_path = "tiny_HaGRID/learning/learning_set_2hands.csv"
    testing_set_path = "tiny_HaGRID/testing/testing_set_2hands.csv"
    
    # Sprawdzenie czy pliki istnieją
    if not os.path.exists(learning_set_path):
        print(f"Błąd: Plik {learning_set_path} nie znaleziony!")
        return
    
    if not os.path.exists(testing_set_path):
        print(f"Błąd: Plik {testing_set_path} nie znaleziony!")
        return
    
    # Uruchomienie porównania
    print("\n" + "="*80)
    print("🚀 POCZĄTEK PORÓWNANIA: GRID SEARCH vs HYPERBAND")
    print("="*80)
    
    comparison = ComparisonRunner(learning_set_path, testing_set_path)
    
    # Uruchom oba algorytmy
    print("\n⏳ Uruchamianie Grid Search (to może potrwać 15-40 minut)...")
    comparison.run_grid_search()
    
    print("\n⏳ Uruchamianie Hyperband (to powinno być szybsze)...")
    comparison.run_hyperband()
    
    # Porównanie
    comparison.create_comparison_table()
    comparison.save_comparison_to_file()
    comparison.visualize_comparison()
    comparison.print_summary()
    
    print("✓ Porównanie zakończone pomyślnie!")
    print(f"✓ Rezultaty zapisane w:")
    print(f"   - comparison_results.txt")
    print(f"   - comparison_grid_search_vs_hyperband.png")
    print(f"   - grid_search_coarse_to_fine_results.png")
    print(f"   - hyperband_optimization_results.png")


if __name__ == "__main__":
    main()
