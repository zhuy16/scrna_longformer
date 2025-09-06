#!/usr/bin/env python3
"""
Simple baseline comparison: Logistic Regression vs Transformer
"""
import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

def run_logistic_baseline(data_path, n_folds=5):
    """Run logistic regression baseline"""
    # Load data
    d = np.load(data_path, allow_pickle=True)
    X, y = d['X'], d['y']
    
    print(f"Baseline: Logistic Regression")
    print(f"Data: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    acc_scores, f1_scores = [], []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx] 
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Train logistic regression
        clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        clf.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = clf.predict(X_val_scaled)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='macro')
        
        acc_scores.append(acc)
        f1_scores.append(f1)
        
        print(f"Fold {fold+1}: acc={acc:.3f}, f1={f1:.3f}")
    
    print(f"\nLogistic Regression Results:")
    print(f"Accuracy: {np.mean(acc_scores):.3f} ± {np.std(acc_scores):.3f}")
    print(f"F1: {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")
    
    return np.mean(acc_scores), np.mean(f1_scores)

if __name__ == "__main__":
    # Run baseline
    log_acc, log_f1 = run_logistic_baseline('data/pbmc3k_hvg_knn_leiden_top500.npz')
    
    # Compare with transformer results
    print(f"\n=== Comparison ===")
    print(f"Logistic Regression:  acc={log_acc:.3f}, f1={log_f1:.3f}")
    print(f"Current Transformer:  acc=0.372, f1=0.060")
    print(f"")
    print(f"F1 improvement: {(log_f1/0.060 - 1)*100:.0f}% {'better' if log_f1 > 0.060 else 'worse'}")
    print(f"Transformer advantage: {'✅ YES' if log_f1 < 0.060 else '❌ NO - use simpler model!'}")
