import pandas as pd
import joblib
import numpy as np
import cProfile
import pstats
import io

# Load UCI Adult dataset (10k rows for speed)
df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    header=None,
    names=[
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]
)

# Drop rows with missing values
df.replace(" ?", np.nan, inplace=True)
df.dropna(inplace=True)

X = df.drop("income", axis=1)
X = X.reset_index(drop=True)  # Reset index after dropping rows
X=X.iloc[:2000]  # Limit to 2k rows for speed

# Load MLP model pipeline
mlp_model = joblib.load("MLPClassifier.joblib")

# setup profiler
profiler = cProfile.Profile()
profiler.enable()

# Run prediction
_ = mlp_model.predict(X)

# stop profiler
profiler.disable()

# output stats
s = io.StringIO()
stats = pstats.Stats(profiler, stream=s).sort_stats('cumtime')
stats.print_stats(15)
print("\n=== CPU Hotspot Report ===\n")
print(s.getvalue())

