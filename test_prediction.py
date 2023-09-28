import pickle
from unittest.mock import Mock

import pandas as pd
import pytest

from task2 import predict, preprocess


@pytest.fixture
def mock_input_data():
    return pd.DataFrame(
        {
            "Type": ["Cat", "Cat", "Dog"],
            "Age": [3, 1, 1],
            "Breed1": ["Tabby", "Domestic Medium Hair", "Mixed Breed"],
            "Gender": ["Male", "Male", "Male"],
            "Color1": ["Black", "Black", "Brown"],
            "Color2": ["White", "Brown", "White"],
            "MaturitySize": ["Small", "Medium", "Medium"],
            "FurLength": ["Short", "Medium", "Medium"],
            "Vaccinated": ["No", "Not Sure", "Yes"],
            "Sterilized": ["No", "Not Sure", "No"],
            "Health": ["Healthy", "Healthy", "Healthy"],
            "Fee": [100, 0, 0],
            "PhotoAmt": [1, 2, 7],
        }
    )


@pytest.fixture
def expected_log_transform():
    return pd.DataFrame(
        {
            "Age": [1.3862943611198906, 0.6931471805599453, 0.6931471805599453],
            "PhotoAmt": [0.69314718, 1.09861229, 2.07944154],
        }
    )


@pytest.fixture
def expected_ordinal_transform():
    return pd.DataFrame(
        {"FurLength": [0, 1, 1], "Health": [0, 0, 0], "MaturitySize": [0, 1, 1]}
    ).astype("float64")


@pytest.fixture
def trained_artifacts():
    OH_encoder, ord_encoder, model = pickle.load(open("artifacts/model", "rb"))
    return OH_encoder, ord_encoder, model


@pytest.fixture
def preprocessed_data(trained_artifacts, mock_input_data):
    # Arrange
    OH_encoder, ord_encoder, _ = trained_artifacts

    # Act
    preprocessed_data = preprocess(mock_input_data, OH_encoder, ord_encoder)

    return preprocessed_data


def test_numercial_log_transformations(expected_log_transform, preprocessed_data):
    """Testing that numerical columns are correctly log transformed"""
    # Arrange
    num_cols = ["Age", "PhotoAmt"]

    # Assert
    pd.testing.assert_frame_equal(preprocessed_data[num_cols], expected_log_transform)


def test_one_hot_coding(preprocessed_data):
    """Testing the correct number of one hot encoding columns are generated"""
    # Arrange
    OH_cols = ["Gender", "Type", "Color1", "Color2", "Sterilized", "Vaccinated"]
    expected_number_of_cols = 1 + 1 + 7 + 7 + 3 + 3

    # Assert
    actual_oh_col_count = sum(
        [1 for col in preprocessed_data.columns.tolist() if type(col) == int]
    )
    assert actual_oh_col_count == expected_number_of_cols


def test_ordinal_encoding(preprocessed_data, expected_ordinal_transform):
    """Testing the ordinal features are correctly transformed"""
    # Arrange
    ordinal_cols = ["FurLength", "Health", "MaturitySize"]

    # Assert
    pd.testing.assert_frame_equal(
        preprocessed_data[ordinal_cols], expected_ordinal_transform
    )


def test_cols_dropped(mock_input_data, preprocessed_data):
    """Testing that the right columns are dropped"""
    # Arrange
    cols_to_drop = [
        "Gender",
        "Type",
        "Color1",
        "Color2",
        "Sterilized",
        "Vaccinated",
        "Breed1",
    ]

    # Assert
    assert all(
        x in mock_input_data.columns and x not in preprocessed_data.columns
        for x in cols_to_drop
    )
