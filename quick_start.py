#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Menu uruchamiania - Porównanie Grid Search vs Hyperband
"""

import sys
import os

def print_banner():
    """Wydruk baneru"""
    print("\n" + "="*80)
    print("   🎯 OPTYMALIZACJA HIPERPARAMETRÓW - Grid Search vs Hyperband")
    print("   Projekt: AI for Hand Gesture Recognition for ESP-CAM")
    print("="*80 + "\n")

def print_menu():
    """Wydruk menu"""
    print("Wybierz opcję:\n")
    print("  1️⃣  Grid Search Coarse-to-Fine (metoda klasyczna)")
    print("       ⏱️  Czas: 15-40 minut")
    print("       📊 Rezultat: Wysoce dokładne parametry\n")
    
    print("  2️⃣  Hyperband Optimization (metoda budżetowa) ⭐")
    print("       ⏱️  Czas: 2-15 minut (2-5x szybciej!)")
    print("       📊 Rezultat: Porównywalna dokładność\n")
    
    print("  3️⃣  Porównanie obu metod (REKOMENDOWANE!) 🏆")
    print("       ⏱️  Czas: 20-50 minut")
    print("       📊 Rezultat: Tabela + wykresy porównawcze\n")
    
    print("  4️⃣  Pokaż instrukcje")
    print("  0️⃣  Wyjście\n")

def print_instructions():
    """Wydruk instrukcji"""
    print("\n" + "="*80)
    print("📚 INSTRUKCJE")
    print("="*80 + "\n")
    
    print("🔍 GRID SEARCH COARSE-TO-FINE:")
    print("   • Testuje WSZYSTKIE kombinacje parametrów")
    print("   • Faza 1 (COARSE): Szerokie zakresy parametrów")
    print("   • Faza 2 (FINE): Wąskie zakresy wokół najlepszych")
    print("   • Wyniki: Najwyższa dokładność")
    print("   • Czas: Długi (15-40 minut)\n")
    
    print("⚡ HYPERBAND (Successive Halving):")
    print("   • Testuje SMART: eliminuje słabe kombinacje wcześnie")
    print("   • Iteracyjnie testuje i eliminuje")
    print("   • Factor=3: Eliminuje 2/3 kombinacji każdą iterację")
    print("   • Wyniki: Porównywalna dokładność, ZNACZNIE szybciej")
    print("   • Czas: Krótki (2-15 minut)\n")
    
    print("📊 PORÓWNANIE:")
    print("   • Uruchamia OBIE metody nachodzącą po sobie")
    print("   • Tworzy tabelę porównawczą")
    print("   • Generuje wykresy:")
    print("     - Porównanie czasu")
    print("     - Porównanie dokładności")
    print("     - Speedup (przyspieszenie)")
    print("     - Efficiency Score")
    print("   • Daje zalecenia\n")
    
    print("💡 CO WYBRAĆ?")
    print("   • Jeśli masz CZAS: Opcja 3 (porównanie)")
    print("   • Jeśli SPIESZYSZ SIĘ: Opcja 2 (Hyperband)")
    print("   • Jeśli TESTUJESZ: Opcja 3 (porównanie)")
    print("   • Jeśli chcesz MAKSYMALNĄ DOKŁADNOŚĆ: Opcja 1 (Grid Search)\n")

def check_dependencies():
    """Sprawdzenie zależności"""
    print("🔍 Sprawdzanie wymaganych bibliotek...\n")
    
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn'
    }
    
    missing_packages = []
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"   ✓ {name}")
        except ImportError:
            print(f"   ✗ {name} - BRAKUJE!")
            missing_packages.append(name)
    
    if missing_packages:
        print(f"\n⚠️  Brakuje bibliotek: {', '.join(missing_packages)}")
        print(f"\nZainstaluj je komendą:")
        print(f"   pip install {' '.join(missing_packages)}\n")
        return False
    else:
        print(f"\n✓ Wszystkie biblioteki zainstalowane!\n")
        return True

def run_grid_search():
    """Uruchomienie Grid Search"""
    print("\n" + "="*80)
    print("▶️  URUCHAMIANIE GRID SEARCH COARSE-TO-FINE")
    print("="*80 + "\n")
    
    print("⏳ Initialization...")
    from advanced_grid_search_coarse_to_fine import main
    
    print("\n⏳ Trwa optymalizacja (może potrwać 15-40 minut)...\n")
    result = main()
    
    if result:
        print("\n✅ Grid Search zakończony!")
        print(f"   Czas: {result['elapsed_time']:.2f}s ({result['elapsed_time']/60:.2f}min)")
        print(f"   F1-Score: {result['f1']:.4f}")
    else:
        print("\n❌ Błąd podczas wykonania Grid Search")

def run_hyperband():
    """Uruchomienie Hyperband"""
    print("\n" + "="*80)
    print("▶️  URUCHAMIANIE HYPERBAND OPTIMIZATION")
    print("="*80 + "\n")
    
    print("⏳ Initialization...")
    from hyperband_optimization import main
    
    print("\n⏳ Trwa optymalizacja (może potrwać 2-15 minut)...\n")
    result = main()
    
    if result:
        print("\n✅ Hyperband zakończony!")
        print(f"   Czas: {result['elapsed_time']:.2f}s ({result['elapsed_time']/60:.2f}min)")
        print(f"   F1-Score: {result['f1']:.4f}")
    else:
        print("\n❌ Błąd podczas wykonania Hyperband")

def run_comparison():
    """Uruchomienie porównania"""
    print("\n" + "="*80)
    print("▶️  URUCHAMIANIE PORÓWNANIA OBUS METOD")
    print("="*80 + "\n")
    
    print("⏳ Initialization...")
    from comparison_results import main
    
    print("\n⏳ Trwa porównanie (może potrwać 20-50 minut)...\n")
    main()

def main_menu():
    """Główne menu"""
    print_banner()
    
    # Sprawdzenie zależności
    if not check_dependencies():
        print("❌ Zainstaluj brakujące biblioteki i spróbuj ponownie.")
        input("Naciśnij Enter aby wyjść...")
        return
    
    while True:
        print_menu()
        
        try:
            choice = input("Twój wybór (0-4): ").strip()
            
            if choice == '0':
                print("\n👋 Do widzenia!\n")
                break
            
            elif choice == '1':
                run_grid_search()
                input("\nNaciśnij Enter aby powrócić do menu...")
                print("\n" + "="*80 + "\n")
            
            elif choice == '2':
                run_hyperband()
                input("\nNaciśnij Enter aby powrócić do menu...")
                print("\n" + "="*80 + "\n")
            
            elif choice == '3':
                print("\n⚠️  WAŻNE: To uruchomi OBIE metody po kolei!")
                print("Całkowity czas: 20-50 minut\n")
                confirm = input("Czy jesteś pewny? (y/n): ").strip().lower()
                
                if confirm == 'y':
                    run_comparison()
                    input("\nNaciśnij Enter aby powrócić do menu...")
                else:
                    print("Anulowano.\n")
                
                print("\n" + "="*80 + "\n")
            
            elif choice == '4':
                print_instructions()
                input("Naciśnij Enter aby powrócić do menu...")
                print("\n" + "="*80 + "\n")
            
            else:
                print("❌ Nieprawidłowy wybór. Spróbuj ponownie.\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 Do widzenia!\n")
            break
        
        except Exception as e:
            print(f"\n❌ Błąd: {e}\n")
            input("Naciśnij Enter aby powrócić do menu...")
            print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\n👋 Do widzenia!\n")
        sys.exit(0)
