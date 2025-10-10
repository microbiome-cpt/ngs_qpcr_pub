import math
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.metrics import (
    average_precision_score, brier_score_loss, roc_auc_score,
    accuracy_score, precision_recall_fscore_support, 
    balanced_accuracy_score, log_loss, precision_score,
    recall_score, f1_score, confusion_matrix
)
from sklearn.utils import check_random_state

try:
    import shap
except ImportError:
    shap = None



def save_feature_importances(model, feature_names, out_prefix: str):
    try:
        if hasattr(model, "feature_importances_"):
            imp = np.asarray(model.feature_importances_)
        elif hasattr(model, "coef_"):
            imp = np.ravel(np.asarray(model.coef_))
        else:
            return
        n = min(len(feature_names), len(imp)) if feature_names else len(imp)
        names = feature_names[:n] if feature_names else [f"f{i}" for i in range(n)]
        pd.DataFrame({"feature": names, "importance": imp[:n]}).to_csv(
            f"{out_prefix}_feature_importances.csv", index=False
        )
    except Exception:
        pass


def compute_basic_metrics(y_true, proba, pred=None):
    m = {}
    try:
        if proba.ndim == 1 or proba.shape[1] == 1:
            p = proba.ravel()
            m["roc_auc"] = roc_auc_score(y_true, p)
            m["pr_auc"] = average_precision_score(y_true, p)
            m["brier"] = brier_score_loss(y_true, p)
        else:
            m["roc_auc"] = roc_auc_score(y_true, proba, multi_class="ovo")
            m["pr_auc"] = np.nan
            bs = np.mean(np.sum((proba - np.eye(proba.shape[1])[y_true]) ** 2, axis=1))
            m["brier"] = bs
    except Exception:
        m["roc_auc"] = np.nan
        m["pr_auc"] = np.nan
        m["brier"] = np.nan
    if pred is not None:
        m["accuracy"] = np.mean(pred == y_true)
    return m


def compute_multiclass_metrics(y_true, y_proba, class_names):
    """
    Compute comprehensive multiclass metrics including calibration
    
    Args:
        y_true: True labels (1D array)
        y_proba: Predicted probabilities (2D array, shape: [n_samples, n_classes])  
        class_names: List of class names
    
    Returns:
        dict: Comprehensive metrics dictionary
    """
    y_pred = np.argmax(y_proba, axis=1)
    n_classes = len(class_names)
    
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    weighted_precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    weighted_recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    try:
        logloss = log_loss(y_true, y_proba)
    except:
        logloss = np.nan
    
    try:
        if n_classes == 2:
            roc_auc_ovr = roc_auc_score(y_true, y_proba[:, 1])
            roc_auc_macro = roc_auc_ovr
        else:
            roc_auc_ovr = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
            roc_auc_macro = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
    except:
        roc_auc_ovr = np.nan
        roc_auc_macro = np.nan
    
    pr_auc_scores = []
    for i in range(n_classes):
        y_binary = (y_true == i).astype(int)
        if len(np.unique(y_binary)) > 1:
            pr_auc_i = average_precision_score(y_binary, y_proba[:, i])
            pr_auc_scores.append(pr_auc_i)
        else:
            pr_auc_scores.append(0.0)
    
    macro_pr_auc = np.mean(pr_auc_scores)
    weighted_pr_auc = np.average(pr_auc_scores, weights=support)
    
    macro_ece = multiclass_expected_calibration_error(y_true, y_proba)
    
    y_true_onehot = np.eye(n_classes)[y_true]
    brier_score = np.mean(np.sum((y_proba - y_true_onehot) ** 2, axis=1))
    
    max_proba = np.max(y_proba, axis=1)
    confidence_mean = np.mean(max_proba)
    confidence_std = np.std(max_proba)
    
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            'precision': precision[i],
            'recall': recall[i], 
            'f1': f1[i],
            'support': int(support[i]),
            'roc_auc': roc_auc_score((y_true == i).astype(int), y_proba[:, i]) 
                      if len(np.unique((y_true == i).astype(int))) > 1 else 0.0,
            'pr_auc': pr_auc_scores[i]
        }
    
    cm = confusion_matrix(y_true, y_pred)
    
    metrics_dict = {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'macro_roc_auc': roc_auc_macro,
        'macro_pr_auc': macro_pr_auc,
        
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'weighted_roc_auc': roc_auc_ovr,
        'weighted_pr_auc': weighted_pr_auc,
        
        'log_loss': logloss,
        'brier_score': brier_score,
        
        'macro_ece': macro_ece,
        
        'confidence_mean': confidence_mean,
        'confidence_std': confidence_std,
        
        'per_class': per_class_metrics,
        
        'confusion_matrix': cm.tolist(),
        'class_names': class_names,
        'support': support.tolist(),
    }
    
    return metrics_dict


def save_calibration_table(y_true, proba, bins=10, out_csv=None):
    try:
        p = proba[:, 1] if proba.ndim == 2 else proba.ravel()
        cuts = np.linspace(0, 1, bins + 1)
        idx = np.digitize(p, cuts, right=True)
        rows = []
        for b in range(1, bins + 1):
            mask = idx == b
            if mask.any():
                rows.append({
                    "bin": b,
                    "p_mean": float(p[mask].mean()),
                    "y_rate": float(y_true[mask].mean()),
                    "count": int(mask.sum()),
                })

        df = pd.DataFrame(rows)
        if out_csv:
            df.to_csv(out_csv, index=False)
        return df
    except Exception:
        return None


def expected_calibration_error(y_true, proba, n_bins: int = 10) -> Optional[float]:
    """Binary Expected Calibration Error"""
    try:
        p = proba[:, 1] if proba.ndim == 2 else proba.ravel()
        bins = np.linspace(0, 1, n_bins + 1)
        idx = np.digitize(p, bins, right=True)
        ece = 0.0
        for b in range(1, n_bins + 1):
            mask = idx == b
            if not mask.any():
                continue
            conf = p[mask].mean()
            acc = (y_true[mask] == 1).mean()
            w = mask.mean()
            ece += w * abs(acc - conf)
        return float(ece)
    except Exception:
        return None


def multiclass_expected_calibration_error(y_true, y_proba, n_bins=10):
    """
    Compute macro-averaged Expected Calibration Error for multiclass
    """
    n_classes = y_proba.shape[1]
    ece_values = []
    
    for k in range(n_classes):
        y_binary = (y_true == k).astype(int)
        p_k = y_proba[:, k]
        
        if len(np.unique(y_binary)) > 1:
            try:
                p_k_2d = np.column_stack([1 - p_k, p_k])
                ece_k = expected_calibration_error(y_binary, p_k_2d, n_bins)
                if ece_k is not None:
                    ece_values.append(ece_k)
            except:
                continue
    
    return float(np.nanmean(ece_values)) if ece_values else 0.0


def shap_top_features(model, X, feature_names, top_n=20):
    if shap is None:
        return []
    try:
        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(
                model, X, feature_names=feature_names if feature_names else None
            )
        sv = explainer(X)
        vals = sv.values
        if vals.ndim == 3:
            scores = np.mean(np.abs(vals), axis=(0, 2))
        else:
            scores = np.mean(np.abs(vals), axis=0)
        order = np.argsort(scores)[::-1]
        idx = order[: min(top_n, len(order))]
        return [feature_names[i] for i in idx]
    except Exception:
        return []


def conditional_group_permutation_importance(
    estimator, X, y, groups, scorer, n_repeats=3, rng=None
):
    rng = check_random_state(rng)
    base_score = scorer(estimator, X, y)
    results = []
    for g_idx, inds in enumerate(groups):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(X.shape[0])
            Xp[:, inds] = Xp[perm[:, None], inds]
            drops.append(base_score - scorer(estimator, Xp, y))
        results.append(
            {
                "group": g_idx,
                "features_idx": inds,
                "drop_mean": float(np.mean(drops)),
                "drop_std": float(np.std(drops)),
            }
        )
    df = pd.DataFrame(results).sort_values("drop_mean", ascending=False)
    return df


def safe_auc_scorer(estimator, X, y):
    try:
        proba = estimator.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return float(roc_auc_score(y, proba[:, 1]))
        return float(roc_auc_score(y, proba, multi_class="ovo"))
    except Exception:
        return float("nan")


def _mean_std_ci(vals, alpha=0.05):
    vals = [float(v) for v in vals if np.isfinite(v)]
    n = len(vals)
    if n == 0:
        return np.nan, np.nan, (np.nan, np.nan)
    if n == 1:
        return vals[0], float("nan"), (vals[0], vals[0])
    m = float(np.mean(vals))
    s = float(np.std(vals, ddof=1))
    tcrit = float(t.ppf(1 - alpha / 2, df=n - 1))
    half = tcrit * s / math.sqrt(n)
    lo, hi = max(0.0, m - half), min(1.0, m + half)
    return m, s, (lo, hi)


def _fmt(m, s, ci):
    return f"{m:.3f} ± {s:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
