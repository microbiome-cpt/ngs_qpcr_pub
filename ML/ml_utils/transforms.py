import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline

from imblearn.over_sampling import SMOTE


class SafeSimpleImputer(SimpleImputer):
    def transform(self, X):
        try:
            return super().transform(X)
        except Exception:
            return X


class SafeStandardScaler(StandardScaler):
    def transform(self, X):
        try:
            return super().transform(X)
        except Exception:
            return X


class SafeSMOTE(SMOTE):
    def fit_resample(self, X, y):
        try:
            return super().fit_resample(X, y)
        except Exception:
            return X, y


def multiplicative_replacement(X, delta: float = 1e-6):
    X = np.asarray(X, dtype=float)
    X[X < 0] = 0.0

    n, p = X.shape
    X_adj = X.copy()

    zero_mask = X_adj <= 0
    m = zero_mask.sum(axis=1, keepdims=True)
    non_zero_counts = p - m

    if delta > 0:
        X_adj[zero_mask] = float(delta)

    comp = np.divide(delta * m, np.maximum(1.0, non_zero_counts))
    comp_full = np.repeat(comp, p, axis=1)
    mask_nz = ~zero_mask
    if mask_nz.any():
        X_adj[mask_nz] -= comp_full[mask_nz]

    all_zero_rows = (non_zero_counts == 0).ravel()
    if np.any(all_zero_rows):
        X_adj[all_zero_rows, :] = 1.0 / float(p)

    X_adj[X_adj <= 0] = float(delta)
    row_sums = X_adj.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    X_adj = X_adj / row_sums
    return X_adj


def clr_transform(X):
    X = np.asarray(X, dtype=float)
    X[X <= 0] = 1e-12
    gm = np.exp(np.mean(np.log(X), axis=1, keepdims=True))
    return np.log(X / gm)


class MicrobiomeNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, mode="CLR", delta=1e-6):
        self.mode = mode
        self.delta = delta

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Z = np.asarray(X, dtype=float)
        mode = str(self.mode).lower()
        if mode == "log10":
            Z = np.log10(Z + float(self.delta))
            Z[np.isneginf(Z)] = 0.0
            return Z
        elif mode == "clr":
            return clr_transform(multiplicative_replacement(Z, delta=self.delta))
        else:
            raise ValueError(f"Unknown norm mode for MicrobiomeNormalizer: {self.mode}")


class DeicodeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=50, random_state=0):
        self.n_components = n_components
        self.random_state = random_state
        self._svd = None
        self.last_k_ = None
        self.loadings_ = None

    def _prep_clr(self, X):
        X = np.asarray(X, dtype=float)
        X = np.clip(X, 0, None) + 1e-12
        X = multiplicative_replacement(X)
        X = clr_transform(X)
        return X

    def fit(self, X, y=None):
        Xc = self._prep_clr(X)
        n_samples, n_features = Xc.shape
        n_comp = int(self.n_components)
        rs = None if self.random_state is None else int(self.random_state)
        k = max(1, min(n_comp, n_samples - 1, n_features))
        self._svd = TruncatedSVD(n_components=k, random_state=rs)
        self._svd.fit(Xc)
        self.loadings_ = getattr(self._svd, "components_", None)
        self.last_k_ = int(k)
        return self

    def transform(self, X):
        if self._svd is None:
            raise RuntimeError("DeicodeTransformer is not fitted")
        Xc = self._prep_clr(X)
        Z = self._svd.transform(Xc)
        k = int(self.last_k_ or self.n_components)
        if Z.shape[1] != k:
            if Z.shape[1] > k:
                Z = Z[:, :k]
            else:
                pad = np.zeros((Z.shape[0], k - Z.shape[1]))
                Z = np.hstack([Z, pad])
        return Z

    def get_feature_names_out(self, input_features=None):
        k = int(self.last_k_ or self.n_components)
        return np.array([f"DEICODE_PC{i+1}" for i in range(k)])

    def get_loadings(self):
        if getattr(self, "loadings_", None) is None:
            raise RuntimeError("Transformer not fitted")
        return self.loadings_


class CLRRedundancyFilter(BaseEstimator, TransformerMixin):
    def __init__(self, corr_threshold=0.98, prefer="prevalence", delta=1e-6):
        self.corr_threshold = corr_threshold
        self.prefer = prefer
        self.delta = delta
        self.keep_idx_ = None
        self.groups_ = None
        self.rep_idx_ = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        thr = float(self.corr_threshold)
        prefer = str(self.prefer)
        X_mr = multiplicative_replacement(X_df.values, delta=float(self.delta))
        X_clr = clr_transform(X_mr)

        C = np.abs(np.corrcoef(X_clr, rowvar=False))
        C = np.nan_to_num(C, nan=0.0, posinf=1.0, neginf=1.0)
        np.fill_diagonal(C, 0.0)
        n = C.shape[0]

        visited = np.zeros(n, dtype=bool)
        groups = []
        for i in range(n):
            if visited[i]:
                continue
            comp = [i]
            visited[i] = True
            stack = [i]
            while stack:
                u = stack.pop()
                neigh = np.where((C[u] >= thr) & (np.arange(n) != u))[0]
                for v in neigh:
                    if not visited[v]:
                        visited[v] = True
                        comp.append(v)
                        stack.append(v)
            groups.append(sorted(comp))

        prevalence = (X_df.values > 0).sum(axis=0)
        variance = X_df.values.var(axis=0)

        keep_mask = np.zeros(n, dtype=bool)
        rep_idx = []
        for comp in groups:
            if len(comp) == 1:
                keep_mask[comp[0]] = True
                rep_idx.append(int(comp[0]))
                continue
            cand = np.array(comp, dtype=int)
            scores = prevalence[cand] if prefer == "prevalence" else variance[cand]
            best = cand[int(np.argmax(scores))]
            keep_mask[best] = True
            rep_idx.append(int(best))

        self.keep_idx_ = np.where(keep_mask)[0]
        self.groups_ = groups
        self.rep_idx_ = rep_idx
        return self

    def transform(self, X):
        X = np.asarray(X)
        if self.keep_idx_ is None:
            return X
        return X[:, self.keep_idx_]


def build_preprocessor(
    norm: str,
    micro_cols,
    covariate_cols,
    deicode_components: int = 50,
    corr_threshold: float = 0.98,
    prefer: str = "prevalence",
):

    norm_up = (norm or "").upper()
    if norm_up not in {"DEICODE", "CLR", "LOG10"}:
        raise ValueError(f"Unknown norm: {norm}")

    if norm_up == "DEICODE":
        micro_pipe = SkPipeline(
            steps=[
                # int 0 (а не 0.0)
                ("imputer", SafeSimpleImputer(strategy="constant", fill_value=0)),
                ("deicode", DeicodeTransformer(n_components=deicode_components)),
            ]
        )
    else:
        mode = "CLR" if norm_up == "CLR" else "LOG10"
        micro_pipe = SkPipeline(
            steps=[
                (
                    "imputer",
                    SafeSimpleImputer(strategy="constant", fill_value=0),
                ),  # int 0
                ("norm", MicrobiomeNormalizer(mode=mode)),
                (
                    "dedup",
                    CLRRedundancyFilter(
                        corr_threshold=float(corr_threshold), prefer=str(prefer)
                    ),
                ),
            ]
        )

    cov_pipe = SkPipeline(
        steps=[
            ("imputer", SafeSimpleImputer(strategy="median")),
            ("scaler", SafeStandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("microbiome", micro_pipe, list(micro_cols)),
            ("covariates", cov_pipe, list(covariate_cols) if covariate_cols else []),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        n_jobs=None,
        verbose_feature_names_out=False,
    )
    return preprocessor
