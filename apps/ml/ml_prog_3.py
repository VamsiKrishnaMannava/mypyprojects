import joblib
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType, FloatTensortype

# load tranining data to get column names and dtypes
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", header=None,
                 names=["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship",
                        "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"] )

df.dropna(inplace=True)
X = df.drop("income", axis=1)

# Construct the initial type using column names and types
initial_type = []
for col in X.columns:
    if X[col].dtype == object:
        initial_type.append((col,StringTensorType([None,1])))
    else:
        initial_type.append((col, FloatTensortype([None, 1])))

print( initial_type)
# # load fitted pipeline | use file names of models trained in module 1
# pipe = joblib.load("LogisticRegression.joblib")

# #convert the pipeline to ONNX using column-aware initial_types
# onnx_model = convert_sklearn(pipe, initial_types=initial_type)

# # save to file
# with open("logreg.onnx", "wb") as f:
#     f.write(onnx_model.SerializeToString())

# print("ONNX model exported successfully to logreg.onnx")