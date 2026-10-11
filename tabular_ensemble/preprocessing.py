"""Column typing and the two preprocessing views used by the model zoo.

Tree models see ordinal-encoded categoricals with missing values left as NaN.
Distance/gradient models (linear, kNN, MLP) see imputed, standardized numerics
and one-hot encoded categoricals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)


def to_frame(X) -> pd.DataFrame:
    """Return X as a DataFrame; numpy input is treated as all-numeric."""
    if isinstance(X, pd.DataFrame):
        return X
    X = np.asarray(X)
    return pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])


def split_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = [
        c for c in X.columns
        if isinstance(X[c].dtype, pd.CategoricalDtype)
        or X[c].dtype == object
        or X[c].dtype == bool
        or pd.api.types.is_string_dtype(X[c].dtype)
    ]
    numeric = [c for c in X.columns if c not in categorical]
    return numeric, categorical


def _as_str(X):
    # Normalise categoricals to object strings so encoders see one dtype;
    # missing values stay as NaN.
    X = pd.DataFrame(X)
    return X.apply(lambda s: s.astype(object).where(s.notna(), np.nan)).astype(object)


def tree_view(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    cat = make_pipeline(
        FunctionTransformer(_as_str, feature_names_out="one-to-one"),
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        ),
    )
    return ColumnTransformer(
        [("num", "passthrough", numeric), ("cat", cat, categorical)],
        verbose_feature_names_out=False,
    )


def dense_view(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    num = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(),
    )
    cat = make_pipeline(
        FunctionTransformer(_as_str, feature_names_out="one-to-one"),
        SimpleImputer(strategy="constant", fill_value="__missing__"),
        OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            max_categories=32,
            sparse_output=False,
        ),
    )
    return ColumnTransformer(
        [("num", num, numeric), ("cat", cat, categorical)],
        verbose_feature_names_out=False,
    )
