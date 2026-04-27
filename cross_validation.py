import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from train_random_forest_classifier import GestureRecognizer 

def run_cross_validation(csv_path, n_splits=5):
    print("=" * 60)
    print(f"=== {n_splits}-FOLD CROSS VALIDATION START ===")
    print("=" * 60)

    # 1. Load the full dataset
    temp_recognizer = GestureRecognizer()
    X, y, df = temp_recognizer.load_data(csv_path)

    # 2. Setup Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    all_fold_metrics = []

    # 3. The Cross-Validation Loop
    for fold, (train_index, val_index) in enumerate(skf.split(X, y), 1):
        print(f"\n>>> PROCESSING FOLD {fold}/{n_splits}")
        
        # Split data for this fold
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Initialize a FRESH model for this fold
        recognizer = GestureRecognizer(n_estimators=100)
        
        # Train using your class method
        recognizer.train(X_train, y_train)
        
        # Evaluate using your class method
        metrics = recognizer.evaluate(X_val, y_val, set_name=f"Fold {fold}")
        
        all_fold_metrics.append(metrics['accuracy'])

    # 4. Final Summary
    print("\n" + "=" * 60)
    print("=== FINAL CROSS-VALIDATION RESULTS ===")
    print("=" * 60)
    for i, acc in enumerate(all_fold_metrics, 1):
        print(f"Fold {i}: Accuracy = {acc*100:.2f}%")
    
    mean_acc = np.mean(all_fold_metrics)
    std_acc = np.std(all_fold_metrics)
    
    print(f"\nAverage Accuracy: {mean_acc*100:.2f}% (+/- {std_acc*100:.2f}%)")
    print("=" * 60)

if __name__ == "__main__":
    # Point this to your full dataset CSV
    dataset_path = "tiny_HaGRID/full_tiny_HaGRID.csv" 
    run_cross_validation(dataset_path)