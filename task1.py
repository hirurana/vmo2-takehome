"""Task 1 Script"""
import json
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from utils import generate_datasets, generate_metrics, preprocess
from utils.variables import (
    artifact_output_filepath,
    data_url,
    object_cols,
    ordinal_cols,
    random_state,
    y_label,
)

########## DATA INGESTION ###############
label_mapping = {"Yes": 1, "No": 0}
x_train, y_train, x_val, y_val, x_test, y_test = generate_datasets(
    data_url, y_label, random_state, label_mapping
)

########## FEATURE ENGINEERING ##########

num_cols = x_train._get_numeric_data().columns

# Encoder definitions
OH_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False, drop="if_binary")
ord_encoder = OrdinalEncoder(
    categories=[
        ["Short", "Medium", "Long"],
        ["Healthy", "Minor Injury", "Serious Injury"],
        ["Small", "Medium", "Large"],
    ]
)

# Training set preprocess
x_train[num_cols] = x_train[num_cols].transform([np.log1p])
x_train["Fee"] = x_train["Fee"] != 0

OH_cols_data = pd.DataFrame(OH_encoder.fit_transform(x_train[object_cols]))
x_train[ordinal_cols] = ord_encoder.fit_transform(x_train[ordinal_cols])

OH_cols_data.index = x_train.index
non_OH_data = x_train.drop(object_cols, axis=1)
x_train = pd.concat([non_OH_data, OH_cols_data], axis=1)

x_train = x_train.drop(["Breed1"], axis=1)

# Preprocess val and test sets using fitted OH and ord encoder
x_val = preprocess(x_val, OH_encoder, ord_encoder)
x_test = preprocess(x_test, OH_encoder, ord_encoder)

########## TRAINING ######################
model = xgb.XGBClassifier(
    random_state=random_state,
    max_depth=4,
    n_estimators=40,
    scale_pos_weight=0.4,
    early_stopping_rounds=10,
    eval_metric="aucpr",
    objective="binary:logistic",
)

model.fit(
    x_train,
    y_train,
    eval_set=[(x_train, y_train), (x_val, y_val)],
)

########## EVALUATION #####################
y_pred = model.predict(x_test)

f1, accuracy, recall, roc_auc = generate_metrics(y_test, y_pred)

metrics = {"f1_score": f1, "accuracy": accuracy, "recall": recall, "ROC_AUC": roc_auc}

########## SAVE OUTPUTS ###################
with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

# "Concatenating" encoders with model into a single variable so it can be stored as one object
output = (OH_encoder, ord_encoder, model)

# Pickling the output (pickling has security implications but seemed like a viable choice for this use case)
pickle.dump(output, open(artifact_output_filepath, "wb"))
