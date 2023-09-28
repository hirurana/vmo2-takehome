"""Helper functions to load and split data"""
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def split_x_y(
    data: pd.DataFrame, y_label: str, label_mapping: dict = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a dataset into features and a label
    ARGS:
        data [pd.DataFrame]: The full dataset to split
        y_label [str]: The name of the label column to split by
        label_mapping [dict] (Optional): A dictionary to map categorical values into a binary feature

    RETURNS:
        x_data [pd.DataFrame]: The feature dataset
        y_data [pd.Series]: The target column
    """
    y_data = data[y_label]
    if label_mapping:
        y_data = y_data.map(label_mapping)
    x_data = data.drop(y_label, axis=1)
    return x_data, y_data


def generate_datasets(
    data_url: str, y_label: str, random_state: int, label_mapping: dict = None
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Generate the train, val and test datasets
    ARGS:
        data_url [str]: The url pointing to where the raw data stored
        y_label [str]: The name of the label column to split by
        random_state [int]: Seed used by the train_test_split function
        label_mapping [dict] (Optional): A dictionary to map categorical values into a binary feature

    RETURNS:
        x_train [pd.DataFrame]: Training feature dataset
        y_train [pd.Series]: Training target column
        x_val [pd.DataFrame]: Validation feature dataset
        y_val [pd.Series]: Validation target column
        x_test [pd.DataFrame]: Testing feature dataset
        y_test [pd.Series]: Test target column
    """
    df = pd.read_csv(data_url)

    # split the dataset into 3 splits
    train_data, remaining_data = train_test_split(
        df, test_size=0.4, random_state=random_state
    )
    validation_data, test_data = train_test_split(
        remaining_data, test_size=0.5, random_state=random_state
    )

    x_train, y_train = split_x_y(
        data=train_data, y_label=y_label, label_mapping=label_mapping
    )
    x_val, y_val = split_x_y(
        data=validation_data, y_label=y_label, label_mapping=label_mapping
    )
    x_test, y_test = split_x_y(
        data=test_data, y_label=y_label, label_mapping=label_mapping
    )

    return x_train, y_train, x_val, y_val, x_test, y_test
