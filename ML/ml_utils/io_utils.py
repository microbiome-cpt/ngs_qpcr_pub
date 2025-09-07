from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Union, Optional

def load_data(path: Union[str, Path], *, verbose: bool = True) -> pd.DataFrame:
    p = str(path)
    seps = [",", ";", "\t", "|"]
    encs = ["utf-8", "utf-8-sig", "cp1251", "latin1"]

    last_err: Optional[Exception] = None

    def _cleanup(df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [str(c).strip() for c in df.columns]
        drop_cols = [c for c in df.columns if c.startswith("Unnamed")]
        if drop_cols:
            df = df.drop(columns=drop_cols, errors="ignore")
        return df

    for enc in encs:
        for sep in seps:
            try:
                df = pd.read_csv(p, sep=sep, encoding=enc, engine="python")
                if df.shape[1] >= 2:
                    if verbose:
                        print(f"[load_data] enc='{enc}', sep='{sep}', cols={df.shape[1]}")
                    return _cleanup(df)
            except Exception as e:
                last_err = e
                continue

    try:
        if verbose:
            print("[load_data] fallback: encoding='cp1251', sep=None (auto), "
                  "errors='replace', on_bad_lines='skip'")
        df = pd.read_csv(
            p,
            sep=None,
            engine="python",
            encoding="cp1251",
            encoding_errors="replace",
            on_bad_lines="skip",
        )
        return _cleanup(df)
    except Exception as e:
        raise last_err or e


def ensure_sex_encoded(df):
    sex_col = None
    for cand in ["Sex", "Пол"]:
        if cand in df.columns:
            sex_col = cand
            break
    if "Sex_enc" not in df.columns and sex_col is not None:
        s = pd.Series(df[sex_col]).astype(str).str.lower().replace({
            "male": 1, "m": 1, "м": 1, "муж": 1,
            "female": 0, "f": 0, "ж": 0, "жен": 0
        })
        df["Sex_enc"] = pd.to_numeric(s, errors="coerce").fillna(0).astype(int)
    return df

def encode_target(df: pd.DataFrame, target_col: str):
    resolved = target_col
    if resolved not in df.columns and target_col == "Disease" and "Заболевание" in df.columns:
        resolved = "Заболевание"
    if resolved not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    y_raw = df[resolved].astype(str)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    return y, le, resolved

def select_feature_blocks(df: pd.DataFrame, target_col: str, id_col: str, exclude_cols: list):
    excl = set(exclude_cols or [])
    drop_cols = [c for c in df.columns if c in excl or c.startswith("Unnamed")]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True, errors="ignore")

    exclude = set((exclude_cols or []) + [target_col, id_col, "Sex", "Пол", "Sex_enc"])
    covs = [c for c in ["Sex_enc", "Age", "BMI"] if c in df.columns]

    micro_cols = []
    for c in df.columns:
        if c in exclude or c in covs:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            micro_cols.append(c)
        else:
            try:
                df[c] = pd.to_numeric(df[c], errors="raise")
                micro_cols.append(c)
            except Exception:
                pass

    if len(micro_cols) == 0:
        raise ValueError("No microbiome numeric signatures were found after exclusions.")
    return micro_cols, covs
