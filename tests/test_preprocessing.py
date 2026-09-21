import numpy as np
import pandas as pd
import pytest

from practice_1.preprocessing import MultiHotEncoder, make_preprocessor


def test_multihot_encoder_splits_answers_into_independent_features():
    data = pd.DataFrame(
        {
            "q4": [
                "Экосистема;Надежность",
                "Экосистема;Камера;Надежность",
            ]
        }
    )

    encoder = MultiHotEncoder().fit(data)
    encoded = pd.DataFrame(
        encoder.transform(data),
        columns=encoder.get_feature_names_out(data.columns),
    )

    assert encoded.columns.tolist() == [
        "q4__Камера",
        "q4__Надежность",
        "q4__Экосистема",
    ]
    np.testing.assert_array_equal(
        encoded.to_numpy(),
        [[0, 1, 1], [1, 1, 1]],
    )
    assert not any(";" in column for column in encoded.columns)


def test_multihot_encoder_strips_whitespace_and_ignores_empty_tokens():
    data = pd.DataFrame({"q4": ["Камера; Камера ; ;Надежность"]})

    encoded = MultiHotEncoder().fit_transform(data)

    np.testing.assert_array_equal(encoded, [[1, 1]])


def test_multihot_encoder_ignores_categories_not_seen_during_fit():
    encoder = MultiHotEncoder().fit(pd.DataFrame({"q4": ["Камера;Надежность"]}))

    encoded = encoder.transform(pd.DataFrame({"q4": ["Экосистема;Камера", ""]}))

    np.testing.assert_array_equal(encoded, [[1, 0], [0, 0]])


def test_multihot_encoder_requires_one_column():
    with pytest.raises(ValueError, match="exactly one input column"):
        MultiHotEncoder().fit(pd.DataFrame({"q4": ["Камера"], "other": ["Да"]}))


def test_preprocessor_has_separate_multihot_block():
    data = pd.DataFrame(
        {
            "age": [21, 22],
            "q4": ["Экосистема;Надежность", "Камера"],
            "platform": ["Айфон", "Андроид"],
        }
    )
    preprocessor = make_preprocessor(["age"], ["platform"], "q4")

    transformed = preprocessor.fit_transform(data)
    feature_names = preprocessor.get_feature_names_out().tolist()

    assert transformed.shape == (2, 6)
    assert "multi_select__q4__Камера" in feature_names
    assert "multi_select__q4__Надежность" in feature_names
    assert "multi_select__q4__Экосистема" in feature_names
    assert not any(";" in name for name in feature_names)
