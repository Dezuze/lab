# common.py — shared offline/CSV helpers
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_csv(path, target):
    """Load a local CSV. target is the exact label/target column name."""
    df = pd.read_csv(path)
    if target not in df.columns:
        raise ValueError(f"Target column {target!r} not found. Columns: {list(df.columns)}")
    df = df.dropna(subset=[target])
    X = df.drop(columns=[target])
    y = df[target]
    # Basic numeric-only fallback; encode categorical columns for common tabular labs.
    X = pd.get_dummies(X, dummy_na=True)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    return X, y

def split_scale(X, y, test_size=0.2, stratify=False):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=42,
        stratify=y if stratify else None)
    scaler = StandardScaler()
    return scaler.fit_transform(Xtr), scaler.transform(Xte), ytr, yte, scaler
