"""Transformation helper functions"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from utils.variables import object_cols, ordinal_cols


def preprocess(
    data: pd.DataFrame, OH_encoder: OneHotEncoder, ord_encoder: OrdinalEncoder
) -> pd.DataFrame:
    """Preprocess the raw dataset into the format expected by the XGBoost model
    ARGS:
        data [pd.DataFrame]: The full dataset to be preprocessed
        OH_encoder [sklearn.preprocessing.OneHotEncoder]: A fitted one hot encoder obj to transform OH_cols
        ord_encoder [sklearn.preprocessing.OrdinalEncoder]: A fitted ordinal encoder obj to tranform ordinal cols

    RETURNS:
        data [pd.DataFrame]: The preprocessed dataset
    """
    num_cols = data._get_numeric_data().columns

    data[num_cols] = data[num_cols].transform([np.log1p])
    data["Fee"] = (data["Fee"] != 0).astype("float64")

    OH_cols_data = pd.DataFrame(OH_encoder.transform(data[object_cols]))
    data[ordinal_cols] = ord_encoder.transform(data[ordinal_cols])

    OH_cols_data.index = data.index
    non_OH_data = data.drop(object_cols, axis=1)
    data = pd.concat([non_OH_data, OH_cols_data], axis=1)
    data = data.drop(["Breed1"], axis=1)

    return data
