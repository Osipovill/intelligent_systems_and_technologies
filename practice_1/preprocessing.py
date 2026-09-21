"""Reusable preprocessing components for the smartphone preference model."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.utils.validation import check_is_fitted


class MultiHotEncoder(BaseEstimator, TransformerMixin):
    """Encode a semicolon-separated column as independent binary features."""

    def __init__(self, separator: str = ";") -> None:
        self.separator = separator

    def fit(self, X, y=None):
        values = self._as_series(X)
        categories = sorted(
            {
                token
                for value in values
                for token in self._split(value)
            }
        )
        self.categories_ = np.asarray(categories, dtype=object)
        self._category_to_index = {
            category: index for index, category in enumerate(self.categories_)
        }
        return self

    def transform(self, X):
        check_is_fitted(self, ["categories_", "_category_to_index"])
        values = self._as_series(X)
        encoded = np.zeros((len(values), len(self.categories_)), dtype=np.float64)

        for row_index, value in enumerate(values):
            for token in self._split(value):
                category_index = self._category_to_index.get(token)
                if category_index is not None:
                    encoded[row_index, category_index] = 1.0

        return encoded

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, ["categories_"])
        if input_features is None:
            feature_name = "feature"
        else:
            if len(input_features) != 1:
                raise ValueError("MultiHotEncoder expects exactly one input column")
            feature_name = input_features[0]

        return np.asarray(
            [f"{feature_name}__{category}" for category in self.categories_],
            dtype=object,
        )

    @staticmethod
    def _as_series(X) -> pd.Series:
        if isinstance(X, pd.DataFrame):
            if X.shape[1] != 1:
                raise ValueError("MultiHotEncoder expects exactly one input column")
            return X.iloc[:, 0]
        if isinstance(X, pd.Series):
            return X

        values = np.asarray(X, dtype=object)
        if values.ndim == 2:
            if values.shape[1] != 1:
                raise ValueError("MultiHotEncoder expects exactly one input column")
            values = values[:, 0]
        if values.ndim != 1:
            raise ValueError("MultiHotEncoder expects a one-dimensional input")
        return pd.Series(values)

    def _split(self, value) -> set[str]:
        if value is None or pd.isna(value):
            return set()
        return {
            token.strip()
            for token in str(value).split(self.separator)
            if token.strip()
        }


def make_preprocessor(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    multi_select_feature: str,
) -> ColumnTransformer:
    """Build the feature preprocessing used by the KNN pipeline."""

    return ColumnTransformer(
        [
            ("numeric", RobustScaler(), list(numeric_features)),
            ("multi_select", MultiHotEncoder(), [multi_select_feature]),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                list(categorical_features),
            ),
        ]
    )
