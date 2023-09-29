"""Task 2 Script"""
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from utils import preprocess
from utils.variables import object_cols, ordinal_cols


def predict(
    data: pd.DataFrame,
    model: xgb.XGBClassifier,
    OH_encoder: OneHotEncoder,
    ord_encoder: OrdinalEncoder,
) -> np.ndarray:
    """Make predictions from a raw dataset

    ARGS:
        data [pd.DataFrame]: The full dataset to generate predictions for
        model [xgb.XGBClassifier]: A trained XGBoost model
        OH_encoder [sklearn.preprocessing.OneHotEncoder]: A fitted one hot encoder obj
        ord_encoder [sklearn.preprocessing.OrdinalEncoder]: A fitted ordinal encoder obj

    RETURNS:
        y_pred [np.ndarray]: The generated predictions
    """
    preprocessed_data = preprocess(data, OH_encoder, ord_encoder)
    return model.predict(preprocessed_data)


if __name__ == "__main__":
    from utils.variables import (
        artifact_output_filepath,
        data_url,
        predicted_output_path,
        y_label,
    )

    OH_encoder, ord_encoder, model = pickle.load(open(artifact_output_filepath, "rb"))

    full_data = pd.read_csv(data_url)
    y_true = full_data[y_label]
    X_data = full_data.drop([y_label], axis=1)
    full_data[y_label + "_prediction"] = predict(X_data, model, OH_encoder, ord_encoder)
    full_data[y_label + "_prediction"] = full_data[y_label + "_prediction"].map(
        {1: "Yes", 0: "No"}
    )
    full_data.to_csv(predicted_output_path, index=False)
