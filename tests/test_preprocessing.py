import sys
import warnings
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tabicl._sklearn.preprocessing import TransformToNumerical

ORION_PREPROCESSING = (
    Path(__file__).resolve().parents[1]
    / "baseline_compare"
    / "Orion-MSP"
    / "src"
    / "orion_msp"
    / "sklearn"
    / "preprocessing.py"
)
spec = importlib.util.spec_from_file_location("orion_msp_preprocessing", ORION_PREPROCESSING)
orion_msp_preprocessing = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(orion_msp_preprocessing)
OrionTransformToNumerical = orion_msp_preprocessing.TransformToNumerical


def _assert_no_empty_feature_warning(caught):
    messages = [str(item.message) for item in caught]
    assert not any("Skipping features without any observed values" in message for message in messages)


def test_transform_to_numerical_keeps_all_nan_numeric_dataframe_columns():
    X_train = pd.DataFrame(
        {
            "observed": [1.0, 2.0, 3.0],
            "all_missing": [np.nan, np.nan, np.nan],
        }
    )
    X_test = pd.DataFrame(
        {
            "observed": [4.0, 5.0],
            "all_missing": [10.0, np.nan],
        }
    )

    encoder = TransformToNumerical()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        X_train_out = encoder.fit_transform(X_train)
        X_test_out = encoder.transform(X_test)

    _assert_no_empty_feature_warning(caught)
    assert X_train_out.shape == X_train.shape
    assert X_test_out.shape == X_test.shape
    np.testing.assert_allclose(X_train_out[:, 1], np.zeros(len(X_train)))
    np.testing.assert_allclose(X_test_out[:, 1], np.array([10.0, 0.0]))


def test_transform_to_numerical_keeps_all_nan_numeric_numpy_columns():
    X_train = np.array(
        [
            [1.0, np.nan],
            [2.0, np.nan],
            [3.0, np.nan],
        ]
    )
    X_test = np.array(
        [
            [4.0, 10.0],
            [5.0, np.nan],
        ]
    )

    encoder = TransformToNumerical()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        X_train_out = encoder.fit_transform(X_train)
        X_test_out = encoder.transform(X_test)

    _assert_no_empty_feature_warning(caught)
    assert X_train_out.shape == X_train.shape
    assert X_test_out.shape == X_test.shape
    np.testing.assert_allclose(X_train_out[:, 1], np.zeros(len(X_train)))
    np.testing.assert_allclose(X_test_out[:, 1], np.array([10.0, 0.0]))


def test_orion_transform_to_numerical_keeps_all_nan_numeric_dataframe_columns():
    X_train = pd.DataFrame(
        {
            "observed": [1.0, 2.0, 3.0],
            "all_missing": [np.nan, np.nan, np.nan],
        }
    )
    X_test = pd.DataFrame(
        {
            "observed": [4.0, 5.0],
            "all_missing": [10.0, np.nan],
        }
    )

    encoder = OrionTransformToNumerical()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        X_train_out = encoder.fit_transform(X_train)
        X_test_out = encoder.transform(X_test)

    _assert_no_empty_feature_warning(caught)
    assert X_train_out.shape == X_train.shape
    assert X_test_out.shape == X_test.shape
    np.testing.assert_allclose(X_train_out[:, 1], np.zeros(len(X_train)))
    np.testing.assert_allclose(X_test_out[:, 1], np.array([10.0, 0.0]))
