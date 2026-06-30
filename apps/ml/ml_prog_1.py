import pandas as pd
import numpy as np
# pyrefly: ignore [missing-import]
import joblib
import os
import time

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.exceptions import ConvergenceWarning

# load UCI Adult dataset (10k rows for speed)
data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", header=None, names=[
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
])

# Drop rows with missing values
data.replace(' ?', np.nan, inplace=True)
data.dropna(inplace=True)

# print(data.dtypes)
x = data.drop("income", axis=1)
y = data["income"].apply(lambda x: x.strip() == ">50K")

# # Identify column types
cat_cols = x.select_dtypes(include='object').columns.tolist()
num_cols = x.select_dtypes(exclude='object').columns.tolist()
print(cat_cols) 
print(num_cols)

# print("************************************")
# print(x.select_dtypes(include='object').head().to_string())
# print("************************************")
# print(data[cat_cols].head().to_string())

# # split data
# x_train, x_test, y_train, y_test = train_test_split(x,y, stratify=y, test_size=0.2, random_state=42)

# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# preprocessor = ColumnTransformer([
#     ("onehot",OneHotEncoder(handle_unknown='ignore'),cat_cols),
#     ("scale",StandardScaler(),num_cols)
# ])

# Define modls