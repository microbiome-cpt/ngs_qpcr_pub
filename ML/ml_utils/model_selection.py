import platform
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
import argparse

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, LeaveOneGroupOut
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    f1_score, 
    precision_score, 
    recall_score, 
    accuracy_score, 
    precision_recall_fscore_support,
    balanced_accuracy_score,
    precision_recall_curve,
)
from imblearn.pipeline import Pipeline as ImbPipeline

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    import shap as _shap
except ImportError:
    _shap = None

from ml_utils.transforms import SafeSMOTE, build_preprocessor
from ml_utils.io_utils import (
    load_data,
    ensure_sex_encoded,
    encode_target,
    select_feature_blocks,
)
from ml_utils.models import iter_model_specs
from ml_utils.importance_utils import (
    compute_basic_metrics,
    compute_multiclass_metrics,
    save_feature_importances,
    shap_top_features,
    save_calibration_table,
    expected_calibration_error,
    multiclass_expected_calibration_error,
    conditional_group_permutation_importance,
    safe_auc_scorer,
    _mean_std_ci,
    _fmt,
)


def log(msg: str):
    """Print timestamped log message."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


def make_pipe(pre, estimator, seed, use_smote=False):
    """Build ML pipeline with preprocessing, optional SMOTE, and estimator."""
    steps = [("pre", pre)]
    if use_smote:
        steps.append(("smote", SafeSMOTE(random_state=seed)))
    steps.append(("model", estimator))
    return ImbPipeline(steps)


def per_class_and_macro(y_true, y_pred, n_classes: int) -> Dict[str, Any]:
    """Compute per-class and macro-averaged precision, recall, F1."""
    P, R, F1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), zero_division=0
    )
    return {
        "P": P.astype(float),
        "R": R.astype(float),
        "F1": F1.astype(float),
        "P_macro": float(np.nanmean(P)),
        "R_macro": float(np.nanmean(R)),
        "F1_macro": float(np.nanmean(F1)),
    }


def data_loading_and_preprocessing(csv_path: Path, args) -> Dict[str, Any]:
    log("Loading data")
    df = load_data(csv_path)

    log("Ensuring sex encoded")
    df = ensure_sex_encoded(df)

    log("Encoding target")
    y, le, target_resolved = encode_target(df, args.target_col)
    class_names = [str(c) for c in le.classes_]

    log("Selecting feature blocks")
    micro_cols, covs = select_feature_blocks(
        df, target_resolved, args.id_col, args.exclude_cols
    )

    if args.id_col not in df.columns:
        raise ValueError(f"id-col '{args.id_col}' not found in {csv_path.name}")
    groups = df[args.id_col].astype(str).values

    norm = args.norm.upper()
    log(f"Building preprocessor for norm={norm}")
    pre = build_preprocessor(
        norm=args.norm,
        micro_cols=micro_cols,
        covariate_cols=covs,
        deicode_components=args.deicode_components,
        corr_threshold=args.corr_threshold,
        prefer=args.prefer,
    )
    return {
        "df": df,
        "y": y,
        "groups": groups,
        "class_names": class_names,
        "pre": pre,
        "norm": norm,
        "micro_cols": micro_cols,
        "covs": covs,
        "label_encoder": le,
    }


def parse_metric_and_group(token):
    if '[' in token and token.endswith(']'):
        metric = token[:token.index('[')].strip()
        group = token[token.index('[')+1:-1].strip()
        return metric, group
    else:
        return token.strip(), None


def get_custom_scorer(scoring_arg, label_encoder, main_group):
    if "[" not in scoring_arg and "macro_" not in scoring_arg:
        scoring_arg = "+".join(f"{met}[{main_group}]" for met in scoring_arg.split("+"))
    
    tokens = scoring_arg.split("+")
    parsed_metrics = []
    
    def _parse_token(tok):
        tok = tok.strip()
        weight = 1.0
        
        if "*" in tok:
            tok, w_str = tok.rsplit("*", 1)
            weight = float(w_str)
        
        if tok.startswith("macro_"):
            metric = tok[len("macro_"):]
            return ("macro", metric, None, weight)
        
        if "[" in tok and "]" in tok:
            metric = tok.split("[")[0].strip()
            group = tok.split("[")[1].split("]")[0].strip()
            return ("group", metric, group, weight)
        
        raise ValueError(f"Token '{tok}' must be in format 'metric[group]', 'macro_metric' or include weights with '*'")
    
    for token in tokens:
        parsed_metrics.append(_parse_token(token))
    
    def _find_best_threshold(y_true, y_scores, metric_type="f1", min_precision=None):
        if len(np.unique(y_true)) < 2:
            return 0.5
        
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
        
        if metric_type == "f1":
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
            
            if min_precision is not None:
                valid_mask = precisions >= min_precision
                if np.any(valid_mask):
                    valid_f1 = f1_scores[valid_mask]
                    valid_thresholds = thresholds[valid_mask[:-1]]
                    if len(valid_thresholds) > 0:
                        best_idx = np.argmax(valid_f1)
                        return valid_thresholds[best_idx]
            
            if len(thresholds) > 0:
                best_idx = np.argmax(f1_scores[:-1])
                return thresholds[best_idx]
        
        elif metric_type == "recall":
            if min_precision is not None:
                valid_mask = precisions >= min_precision
                if np.any(valid_mask):
                    valid_recalls = recalls[valid_mask]
                    valid_thresholds = thresholds[valid_mask[:-1]]
                    if len(valid_thresholds) > 0:
                        best_idx = np.argmax(valid_recalls)
                        return valid_thresholds[best_idx]
        
        return 0.5
    
    def _metric_value(metric, y_bin, y_pred_bin, y_scores):
        if len(np.unique(y_bin)) < 2:
            return 0.0
        
        if metric == "roc_auc":
            return roc_auc_score(y_bin, y_scores)
        elif metric == "pr_auc":
            return average_precision_score(y_bin, y_scores)
        elif metric == "f1":
            return f1_score(y_bin, y_pred_bin, zero_division=0)
        elif metric == "f1_tuned":
            threshold = _find_best_threshold(y_bin, y_scores, "f1")
            y_pred_tuned = (y_scores >= threshold).astype(int)
            return f1_score(y_bin, y_pred_tuned, zero_division=0)
        elif metric == "precision":
            return precision_score(y_bin, y_pred_bin, zero_division=0)
        elif metric == "recall":
            return recall_score(y_bin, y_pred_bin, zero_division=0)
        elif metric == "recall_tuned":
            threshold = _find_best_threshold(y_bin, y_scores, "recall", min_precision=0.1)
            y_pred_tuned = (y_scores >= threshold).astype(int)
            return recall_score(y_bin, y_pred_tuned, zero_division=0)
        elif metric == "accuracy":
            return accuracy_score(y_bin, y_pred_bin)
        elif metric == "balanced_accuracy":
            return balanced_accuracy_score(y_bin, y_pred_bin)
        else:
            raise ValueError(f"Unknown metric {metric}")
    
    class CustomScorer:
        def __init__(self, metrics):
            self.metrics = metrics
        
        def __call__(self, estimator, X, y_true, **kwargs):
            vals, weights = [], []
            proba = estimator.predict_proba(X)
            y_pred = estimator.predict(X)
            classes = label_encoder.classes_
            
            for mode, metric, group_or_none, weight in self.metrics:
                if mode == "group":
                    if group_or_none in classes:
                        grp_idx = label_encoder.transform([group_or_none])[0]
                    elif group_or_none.strip("()") in classes:
                        grp_idx = label_encoder.transform([group_or_none.strip("()")])[0]
                    else:
                        raise ValueError(f"Group '{group_or_none}' not found in classes: {list(classes)}")
                    
                    y_bin = (y_true == grp_idx).astype(int)
                    y_pred_bin = (y_pred == grp_idx).astype(int)
                    y_scores = proba[:, grp_idx]
                    
                    val = _metric_value(metric, y_bin, y_pred_bin, y_scores)
                
                elif mode == "macro":
                    per_class_vals = []
                    for k, class_label in enumerate(classes):
                        y_bin = (y_true == k).astype(int)
                        y_pred_bin = (y_pred == k).astype(int)
                        y_scores = proba[:, k]
                        
                        class_val = _metric_value(metric, y_bin, y_pred_bin, y_scores)
                        per_class_vals.append(class_val)
                    
                    val = float(np.nanmean(per_class_vals))
                
                vals.append(val)
                weights.append(weight)
            
            return float(np.average(vals, weights=weights))
    
    return CustomScorer(parsed_metrics)


def build_model_specs(
    seed: int, skip: List[str]
) -> List[Tuple[str, Any, Dict[str, Any]]]:
    return [
        (n, e, g)
        for (n, e, g) in iter_model_specs(seed=seed)
        if n not in (set(skip) if skip else set())
    ]


def tune_or_fit_inner(
    _name: str,
    estimator,
    param_grid: Dict[str, Any],
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    groups_tr: np.ndarray,
    inner_cv: StratifiedGroupKFold,
    pre,
    args,
    n_classes_tr: int,
    class_names: List[str] = None,
    label_encoder=None,
):
    """Internal 3-fold GridSearchCV on outer-train."""
    estimator_adj = estimator
    param_grid_adj = deepcopy(param_grid) if param_grid else {}

    try:
        if isinstance(estimator_adj, CatBoostClassifier) and n_classes_tr > 2:
            estimator_adj = estimator_adj.set_params(
                loss_function="MultiClass", eval_metric="MultiClass"
            )
            param_grid_adj.setdefault("model__loss_function", ["MultiClass"])
    except Exception:
        pass
    try:
        if isinstance(estimator_adj, XGBClassifier) and n_classes_tr > 2:
            estimator_adj = estimator_adj.set_params(
                objective="multi:softprob", num_class=n_classes_tr
            )
    except Exception:
        pass

    pipe = make_pipe(pre, estimator_adj, seed=args.seed, use_smote=args.use_smote)

    inner_splits_raw = list(inner_cv.split(X_tr, y_tr, groups_tr))
    filtered_splits = [
        (tr_f, va_f)
        for tr_f, va_f in inner_splits_raw
        if np.unique(y_tr[tr_f]).size >= 2 and np.unique(y_tr[va_f]).size >= 2
    ]

    if (
        (len(filtered_splits) == 0)
        or (X_tr.shape[0] < 2 * inner_cv.n_splits)
        or (not param_grid_adj)
    ):
        pipe.fit(X_tr, y_tr)
        return pipe, pipe
    scorer = get_custom_scorer(args.scoring, label_encoder, args.main_group)
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid_adj,
        scoring=scorer,
        cv=filtered_splits,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    gs.fit(X_tr, y_tr)
    return gs.best_estimator_, gs


def run_nested_cv_and_eval(
    df: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    pre,
    model_specs: List[Tuple[str, Any, Dict[str, Any]]],
    class_names=None,
    args=None,
    label_encoder=None,
):
    """Run nested cross-validation and return evaluation results."""
    custom_scorer = get_custom_scorer(args.scoring, label_encoder, args.main_group)
    y_ser = pd.Series(y)
    min_count = int(y_ser.value_counts().min())
    n_outer = min(args.outer_folds, max(2, min_count))
    n_inner = min(args.inner_folds, max(2, min_count))

    outer_cv = StratifiedGroupKFold(
        n_splits=n_outer, shuffle=True, random_state=args.seed
    )
    inner_cv = StratifiedGroupKFold(
        n_splits=n_inner, shuffle=True, random_state=args.seed
    )
    outer_splits = list(outer_cv.split(df, y, groups))

    n_samples = df.shape[0]
    n_classes_all = int(np.unique(y).size)

    candidates = []
    per_model_details = []
    oof_store: Dict[str, Any] = {}
    custom_scores = []

    for name, estimator, param_grid in model_specs:
        log(f"=== [{name}] start nested CV ===")
        custom_scores = []
        oof_proba = (
            np.full(n_samples, np.nan, dtype=float)
            if n_classes_all == 2
            else np.full((n_samples, n_classes_all), np.nan)
        )
        oof_pred = np.full(n_samples, np.nan)
        last_search_obj = None

        aucs, accs, pr_aucs_macro = [], [], []
        pr_auc_by_class = {k: [] for k in range(n_classes_all)}
        prec_by_class = {k: [] for k in range(n_classes_all)}
        rec_by_class = {k: [] for k in range(n_classes_all)}
        f1_by_class = {k: [] for k in range(n_classes_all)}
        prec_macro_vals, rec_macro_vals, f1_macro_vals = [], [], []
        folds_used = 0

        for tr_idx, va_idx in outer_splits:
            X_tr, y_tr = df.iloc[tr_idx], y[tr_idx]
            X_va, y_va = df.iloc[va_idx], y[va_idx]
            groups_tr = groups[tr_idx]
            if np.unique(y_tr).size < 2 or np.unique(y_va).size < 2:
                continue

            est, search_obj = tune_or_fit_inner(
                name,
                estimator,
                param_grid,
                X_tr,
                y_tr,
                groups_tr,
                inner_cv,
                pre,
                args,
                n_classes_tr=int(np.unique(y_tr).size),
                class_names=class_names,
                label_encoder=label_encoder,
            )
            last_search_obj = search_obj
            custom_score_fold = float(custom_scorer(est, X_va, y_va))
            custom_scores.append(custom_score_fold)
            proba = est.predict_proba(X_va) if hasattr(est, "predict_proba") else None
            y_pred = est.predict(X_va)
            y_pred = np.asarray(y_pred).ravel()

            try:
                if (
                    proba is not None
                    and proba.ndim == 2
                    and proba.shape[1] == n_classes_all
                    and n_classes_all > 2
                ):
                    auc = roc_auc_score(y_va, proba, multi_class="ovo")
                elif proba is not None and proba.ndim == 2 and proba.shape[1] == 2:
                    auc = roc_auc_score(y_va, proba[:, 1])
                elif proba is not None and proba.ndim == 1:
                    auc = roc_auc_score(y_va, np.ravel(proba))
                else:
                    auc = roc_auc_score(y_va, y_pred)
            except Exception:
                auc = np.nan

            pr_fold_vals = []
            if (proba is not None) and (
                (proba.ndim == 2 and proba.shape[1] == n_classes_all)
                or n_classes_all == 2
            ):
                for k in range(n_classes_all):
                    try:
                        p_k = (
                            proba[:, 1]
                            if (n_classes_all == 2 and proba.shape[1] == 2 and k == 1)
                            else (
                                1.0 - proba[:, 1]
                                if (
                                    n_classes_all == 2
                                    and proba.shape[1] == 2
                                    and k == 0
                                )
                                else proba[:, k]
                            )
                        )
                        y_k = (y_va == k).astype(int)
                        prk = average_precision_score(y_k, p_k)
                        pr_fold_vals.append(prk)
                        pr_auc_by_class[k].append(prk)
                    except Exception:
                        pr_fold_vals.append(np.nan)
                        pr_auc_by_class[k].append(np.nan)
            prc_macro = float(np.nanmean(pr_fold_vals)) if len(pr_fold_vals) else np.nan

            acc = accuracy_score(y_va, y_pred)

            try:
                prf = per_class_and_macro(y_va, y_pred, n_classes_all)
                for k in range(n_classes_all):
                    prec_by_class[k].append(float(prf["P"][k]))
                    rec_by_class[k].append(float(prf["R"][k]))
                    f1_by_class[k].append(float(prf["F1"][k]))
                prec_macro_vals.append(prf["P_macro"])
                rec_macro_vals.append(prf["R_macro"])
                f1_macro_vals.append(prf["F1_macro"])
            except Exception:
                pass

            aucs.append(float(auc))
            accs.append(float(acc))
            pr_aucs_macro.append(float(prc_macro))
            folds_used += 1

            if n_classes_all == 2:
                if proba is not None and proba.ndim == 2:
                    oof_proba[va_idx] = proba[:, 1]
                elif proba is not None and proba.ndim == 1:
                    oof_proba[va_idx] = proba
            else:
                if (
                    proba is not None
                    and proba.ndim == 2
                    and proba.shape[1] == n_classes_all
                ):
                    oof_proba[va_idx, :] = proba
            oof_pred[va_idx] = y_pred

        try:
            if n_classes_all == 2:
                mask = np.isfinite(oof_proba)
                y_oof, proba_oof = y[mask], oof_proba[mask]
                pred_oof = oof_pred[mask] if np.isfinite(oof_pred).any() else None
            else:
                mask = np.all(np.isfinite(oof_proba), axis=1)
                y_oof, proba_oof = y[mask], oof_proba[mask, :]
                pred_oof = oof_pred[mask] if np.isfinite(oof_pred).any() else None
            oof_metrics = compute_basic_metrics(y_oof, proba_oof, pred=pred_oof)
        except Exception:
            oof_metrics = {
                "roc_auc": float("nan"),
                "pr_auc": float("nan"),
                "brier": float("nan"),
                "accuracy": float("nan"),
            }

        oof_store[name] = {"proba": oof_proba, "pred": oof_pred, "metrics": oof_metrics}

        if folds_used == 0:
            continue

        mean_auc = float(np.nanmean(aucs))
        mean_acc = float(np.nanmean(accs))
        valid_pr = [v for v in pr_aucs_macro if np.isfinite(v)]
        mean_pr = float(np.mean(valid_pr)) if len(valid_pr) > 0 else float("nan")

        mean_custom = float(np.nanmean(custom_scores)) if custom_scores else float("nan")
        candidates.append((name, last_search_obj, mean_auc, mean_acc, mean_pr, mean_custom))
        per_model_details.append(
            {
                "model": name,
                "folds": folds_used,
                "fold_metrics": {
                    "auc": aucs,
                    "acc": accs,
                    "pr_macro": pr_aucs_macro,
                    "pr_by_class": pr_auc_by_class,
                    "precision_by_class": prec_by_class,
                    "recall_by_class": rec_by_class,
                    "f1_by_class": f1_by_class,
                    "precision_macro": prec_macro_vals,
                    "recall_macro": rec_macro_vals,
                    "f1_macro": f1_macro_vals,
                    "class_names": class_names,
                },
                "oof_auc": float(oof_metrics.get("roc_auc", float("nan"))),
                "oof_acc": float(oof_metrics.get("accuracy", float("nan"))),
                "oof_pr_auc": float(oof_metrics.get("pr_auc", float("nan"))),
                "oof_brier": float(oof_metrics.get("brier", float("nan"))),
            }
        )

    if not candidates:
        raise RuntimeError("No models trained; check data/params.")

    best_name, best_trained, best_auc, best_acc, best_pr, _ = max(candidates, key=lambda t: t[5])
    best_params = (
        getattr(best_trained, "best_params_", {})
        if hasattr(best_trained, "best_params_")
        else {}
    )
    return (
        per_model_details,
        best_name,
        best_auc,
        best_acc,
        best_pr,
        oof_store,
        best_trained,
        best_params,
    )


def posthoc_and_reporting(csv_path: Path, args, pack):
    (
        df,
        y,
        groups,
        class_names,
        pre,
        norm,
        micro_cols,
        covs,
        per_model_details,
        best_name,
        best_auc,
        best_acc,
        best_pr,
        oof_store,
        best_trained,
        best_params,
    ) = pack

    out_dir = args.output_root / csv_path.stem.replace(".", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        best_oof = oof_store.get(best_name, None)
        if best_oof is not None:
            sample_ids = df[args.id_col].astype(str).values
            proba = best_oof["proba"]
            pred = best_oof["pred"]
            if proba is not None:
                if proba.ndim == 1:
                    df_oof = pd.DataFrame(
                        {args.id_col: sample_ids, "y_true": y, "proba": proba, "pred": pred}
                    )
                elif proba.ndim == 2 and proba.shape[1] == 2:
                    df_oof = pd.DataFrame(
                        {args.id_col: sample_ids, "y_true": y, "proba": proba[:, 1], "pred": pred}
                    )
                else:
                    safe = lambda s: ''.join(ch if ch.isalnum() or ch in ('_', '-') else '_' for ch in str(s))
                    proba_cols = [f"p_{safe(cname)}" for cname in class_names]
                    df_oof = pd.concat(
                        [
                            pd.DataFrame({args.id_col: sample_ids, "y_true": y, "pred": pred}),
                            pd.DataFrame(proba, columns=proba_cols),
                        ],
                        axis=1,
                    )
                df_oof.to_csv(out_dir / f"{csv_path.stem}_{args.norm}_{best_name}_oof_preds.csv", index=False, encoding="utf-8")
            with open(out_dir / f"{csv_path.stem}_{args.norm}_{best_name}_oof_metrics.json", "w", encoding="utf-8") as fh:
                json.dump(best_oof.get("metrics", {}), fh, ensure_ascii=False, indent=2)
    except Exception as e:
        log(f"WARNING: failed to save OOF: {e}")

    if hasattr(best_trained, "best_estimator_"):
        best_est_full = best_trained.best_estimator_
        best_params = getattr(best_trained, "best_params_", {}) or best_params
    else:
        best_est_full = best_trained
    best_est_full.fit(df, y)
    pre_fitted = best_est_full.named_steps["pre"]

    try:
        if norm == "DEICODE":
            dec = pre_fitted.named_transformers_["microbiome"].named_steps["deicode"]
            loadings = dec.get_loadings()
            pc_cols = [f"PC{i+1}" for i in range(loadings.shape[0])]
            load_df = pd.DataFrame(loadings.T, index=micro_cols, columns=pc_cols).reset_index()
            load_df.rename(columns={"index": "feature"}, inplace=True)
            load_df.to_csv(out_dir / f"{csv_path.stem}_{args.norm}_deicode_loadings.csv", index=False)
    except Exception as e:
        log(f"WARNING: DEICODE loadings: {e}")

    if norm == "DEICODE":
        micro_tf = pre_fitted.named_transformers_["microbiome"].named_steps["deicode"]
        names = getattr(micro_tf, "get_feature_names_out", lambda: None)()
        kept_micro = list(names) if names is not None else [f"DEICODE_PC{i+1}" for i in range(int(getattr(micro_tf, "n_components", 10)))]
    else:
        micro_tf = pre_fitted.named_transformers_["microbiome"].named_steps["dedup"]
        keep_idx = getattr(micro_tf, "keep_idx_", None)
        kept_micro = [micro_cols[i] for i in keep_idx] if keep_idx is not None else micro_cols

    final_feature_names = kept_micro + covs

    try:
        if norm != "DEICODE":
            dedup = pre_fitted.named_transformers_["microbiome"].named_steps["dedup"]
            rep_idx = getattr(dedup, "rep_idx_", None)
            groups_for_cpi = [[i] for i in range(len(rep_idx))] if rep_idx is not None else [[i] for i in range(len(kept_micro))]
            Xp = pre_fitted.transform(df)
            scorer = safe_auc_scorer
            cpi_df = conditional_group_permutation_importance(best_est_full, Xp, y, groups_for_cpi, scorer, n_repeats=5, rng=args.seed)
            cpi_df.to_csv(out_dir / f"{csv_path.stem}_{args.norm}_{best_name}_cpi.csv", index=False)
    except Exception as e:
        log(f"WARNING: CPI failed: {e}")

    out_prefix = out_dir / f"{csv_path.stem}_{args.norm}_{best_name}"
    save_feature_importances(best_est_full.named_steps["model"], final_feature_names, str(out_prefix))
    try:
        X_full = pre_fitted.transform(df)
        shap_feats = shap_top_features(best_est_full.named_steps["model"], X_full, final_feature_names, top_n=20)
        pd.DataFrame({"shap_feature": shap_feats}).to_csv(out_dir / f"{csv_path.stem}_{args.norm}_{best_name}_shap_top20.csv", index=False)
    except Exception:
        pass

    try:
        proba_full = best_est_full.predict_proba(df)
        if len(class_names) > 2:
            metrics = compute_multiclass_metrics(y, proba_full, class_names)
            multiclass_ece = multiclass_expected_calibration_error(y, proba_full, n_bins=10)
            metrics['multiclass_ece'] = multiclass_ece
            (out_dir / f"{csv_path.stem}_{args.norm}_{best_name}_multiclass_ece.txt").write_text(f"{multiclass_ece:.6f}\n", encoding="utf-8")
        else:
            metrics = compute_basic_metrics(y, proba_full, pred=best_est_full.predict(df))
        (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        proba_full = None

    try:
        if proba_full is not None and (proba_full.ndim == 1 or (proba_full.ndim == 2 and proba_full.shape[1] == 2)):
            save_calibration_table(y, proba_full, bins=10, out_csv=str(out_dir / f"{csv_path.stem}_{args.norm}_{best_name}_calibration.csv"))
            ece = expected_calibration_error(y, proba_full, n_bins=10)
            (out_dir / f"{csv_path.stem}_{args.norm}_{best_name}_ece.txt").write_text(f"{ece:.6f}\n", encoding="utf-8")
    except Exception:
        pass

    logo = LeaveOneGroupOut()
    loo_hits = 0
    loo_total = 0
    for tr, te in logo.split(df, y, groups):
        if np.unique(y[tr]).size < 2:
            continue
        est = deepcopy(best_est_full)
        est.fit(df.iloc[tr], y[tr])
        yp = est.predict(df.iloc[te])
        loo_hits += int(np.sum(yp == y[te]))
        loo_total += len(te)
    loo_acc = float(loo_hits / loo_total) if loo_total else float("nan")

    def _triplet(xs):
        return _fmt(*_mean_std_ci(xs))

    md = []
    md.append(f"# Evaluation Report — `{csv_path.stem}`\n")
    md.append(
        f"**Software:** Python {platform.python_version()}, "
        f"scikit-learn {sklearn.__version__}, "
        f"Scipy {scipy.__version__}, "
        f"NumPy {np.__version__}, "
        f"pandas {pd.__version__}, "
        f"SHAP {getattr(_shap, '__version__', 'N/A')}."
    )

    md.append(
        f"**Validation:** nested CV — outer {args.outer_folds}-fold (unbiased, group-aware), inner {args.inner_folds}-fold (GridSearchCV on outer-train, group-aware). "
        "Primary metric: Custom scoring (see args). Additional: Accuracy, PR AUC (macro OVR), per-class Precision/Recall/F1 (OVR).\n"
    )

    best_detail = next((m for m in per_model_details if m["model"] == best_name), None)
    fold = best_detail["fold_metrics"] if best_detail else None
    if fold:
        md.append("## Cross-validation (outer folds, mean ± SD [95% CI])\n")
        md.append(f"- **CV ROC AUC:** {_triplet(fold['auc'])}")
        md.append(f"- **CV Accuracy:** {_triplet(fold['acc'])}")
        md.append(f"- **PR AUC (macro, CV):** {_triplet(fold['pr_macro'])}\n")
        for k, cname in enumerate(fold["class_names"]):
            clean_name = cname.strip('()')
            md.append(f"- **PR AUC (CV), {clean_name}:** {_triplet(fold['pr_by_class'][k])}")
        md.append("\n### Macro-averaged (OVR) — CV")
        md.append(f"- **Precision (macro):** {_triplet(fold['precision_macro'])}")
        md.append(f"- **Recall / Sensitivity (macro):** {_triplet(fold['recall_macro'])}")
        md.append(f"- **F1 (macro):** {_triplet(fold['f1_macro'])}\n")
        md.append("### Per-class metrics (CV)")
        for metric_key, title in [
            ("precision_by_class", "Precision"),
            ("recall_by_class", "Recall (Sensitivity)"),
            ("f1_by_class", "F1"),
        ]:
            for k, cname in enumerate(fold["class_names"]):
                clean_name = cname.strip('()')
                md.append(f"- **{title}, {clean_name}:** {_triplet(fold[metric_key][k])}")
        md.append("")

    md.append(f"- **LOO Accuracy (grouped by `{args.id_col}`):** {loo_acc:.3f}\n")

    if len(class_names) > 2 and proba_full is not None:
        try:
            multiclass_metrics = compute_multiclass_metrics(y, proba_full, class_names)
            multiclass_ece = multiclass_expected_calibration_error(y, proba_full, n_bins=10)
            md.append("## Multiclass Metrics (Full Dataset)")
            md.append(f"- **Macro ROC AUC:** {multiclass_metrics.get('macro_roc_auc', 'N/A'):.3f}")
            md.append(f"- **Macro PR AUC:** {multiclass_metrics.get('macro_pr_auc', 'N/A'):.3f}")
            md.append(f"- **Macro F1:** {multiclass_metrics.get('macro_f1', 'N/A'):.3f}")
            md.append(f"- **Balanced Accuracy:** {multiclass_metrics.get('balanced_accuracy', 'N/A'):.3f}")
            md.append(f"- **Multiclass ECE:** {multiclass_ece:.3f}")
            md.append(f"- **Brier Score:** {multiclass_metrics.get('brier_score', 'N/A'):.3f}")
            md.append(f"- **Log Loss:** {multiclass_metrics.get('log_loss', 'N/A'):.3f}\n")
            md.append("### Per-Class Performance (Full Dataset)")
            for class_name, class_metrics in multiclass_metrics.get('per_class', {}).items():
                clean_name = class_name.strip('()')
                md.append(f"**{clean_name}:**")
                md.append(f"  - Precision: {class_metrics.get('precision', 'N/A'):.3f}")
                md.append(f"  - Recall: {class_metrics.get('recall', 'N/A'):.3f}")
                md.append(f"  - F1: {class_metrics.get('f1', 'N/A'):.3f}")
                md.append(f"  - ROC AUC: {class_metrics.get('roc_auc', 'N/A'):.3f}")
                md.append(f"  - PR AUC: {class_metrics.get('pr_auc', 'N/A'):.3f}")
                md.append(f"  - Support: {class_metrics.get('support', 'N/A')}")
                md.append("")
        except Exception as e:
            log(f"WARNING: multiclass metrics failed: {e}")

    (out_dir / f"{csv_path.stem}_{args.norm}_{best_name}_report.md").write_text(
        "\n".join(md), encoding="utf-8"
    )

    summary = {
        "file": csv_path.name,
        "norm": args.norm,
        "micro_features": len(micro_cols),
        "covariates": covs,
        "models": per_model_details,
        "best": {
            "model": best_name,
            "auc": best_auc,
            "pr_auc": best_pr,
            "acc": best_acc,
            "params": best_params,
        },
        "loo_accuracy": loo_acc,
    }

    if len(class_names) > 2 and proba_full is not None:
        try:
            multiclass_metrics = compute_multiclass_metrics(y, proba_full, class_names)
            summary["multiclass_metrics"] = {
                "macro_roc_auc": multiclass_metrics.get('macro_roc_auc'),
                "macro_pr_auc": multiclass_metrics.get('macro_pr_auc'),
                "macro_f1": multiclass_metrics.get('macro_f1'),
                "balanced_accuracy": multiclass_metrics.get('balanced_accuracy'),
                "multiclass_ece": multiclass_expected_calibration_error(y, proba_full),
                "brier_score": multiclass_metrics.get('brier_score'),
                "per_class": multiclass_metrics.get('per_class')
            }
        except Exception as e:
            log(f"WARNING: multiclass summary failed: {e}")

    (out_dir / "report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "best_params.json").write_text(
        json.dumps(
            {"model": best_name, "params": best_params}, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )

    log(f"Finished {csv_path.name} → best={best_name}, AUC={best_auc:.3f}; report saved.")


def run_for_file(csv_path: Path, args):
    log(f"Start {csv_path.name}")
    out_dir = args.output_root / csv_path.stem.replace(".", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    d = data_loading_and_preprocessing(csv_path, args)
    label_encoder = d["label_encoder"]
    model_specs = build_model_specs(seed=args.seed, skip=args.skip_models)

    (
        per_model_details,
        best_name,
        best_auc,
        best_acc,
        best_pr,
        oof_store,
        best_trained,
        best_params,
    ) = run_nested_cv_and_eval(
            d["df"],
            d["y"],
            d["groups"],
            d["pre"],
            model_specs,
            d["class_names"],
            args,
            label_encoder=label_encoder,
    )
    pack = (
        d["df"],
        d["y"],
        d["groups"],
        d["class_names"],
        d["pre"],
        d["norm"],
        d["micro_cols"],
        d["covs"],
        per_model_details,
        best_name,
        best_auc,
        best_acc,
        best_pr,
        oof_store,
        best_trained,
        best_params,
    )
    posthoc_and_reporting(csv_path, args, pack)


def parse_args():
    p = argparse.ArgumentParser(
        description="Run models with nested CV; CLR/log10/DEICODE; train-only dedup; group-wise CV."
    )
    p.add_argument("--data", type=Path, nargs="+", required=True, help="CSV files")
    p.add_argument("--target-col", type=str, required=True, help="Target column")
    p.add_argument('--scoring', type=str, default='roc_auc[FS]',
                help='Metric or ensemble to optimize, format: "roc_auc[FS]" or "roc_auc[FS]+pr_auc[FS]+f1[FS]"')
    p.add_argument('--main_group', type=str, default='FS',
                help='Main group to optimize')

    p.add_argument("--id-col", type=str, required=True, help="Group column")
    p.add_argument("--exclude-cols", nargs="*", default=[], help="Columns to exclude")
    p.add_argument(
        "--norm",
        choices=["log10", "CLR", "DEICODE"],
        default="log10",
        help="Normalization mode",
    )
    p.add_argument("--outer-folds", type=int, default=5)
    p.add_argument("--inner-folds", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-smote", action="store_true", help="Enable SMOTE for balancing")
    p.add_argument(
        "--corr-threshold", type=float, default=0.98, help="|corr(CLR)| threshold"
    )
    p.add_argument("--prefer", choices=["prevalence", "variance"], default="prevalence")
    p.add_argument(
        "--deicode-components", type=int, default=50, help="DEICODE RPCA components"
    )
    p.add_argument("--output-root", type=Path, default=Path("msel_out"))
    p.add_argument(
        "--skip-models", nargs="+", default=[], help="Model names to exclude"
    )

    return p.parse_args()


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    for csv_file in args.data:
        run_for_file(csv_file, args)


if __name__ == "__main__":
    main()
