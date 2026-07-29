import pandas as pd
import joblib
import os
import time
import numpy as np
import psutil

from sklearn.model_selection import train_test_split

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
y = df["income"].apply(lambda x: x.strip() == ">50K")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Load previously saved model
model_file = "MLPClassifier.joblib"
if os.path.exists(model_file):
    print(f'Error: {model_file} not found. Run Training Stript first to generate the model.')
    
pipeline = joblib.load(model_file)

# Define test sample (single row, same structure)
sample = X_test.iloc[[0]]

cpu_list = psutil.cpu_count(logical=True)
print(f"\nNumber of logical CPUs: {cpu_list}\n")
print("pinning to a single CPU core for stress test...\n")

p = psutil.Process()
original_affinity = p.cpu_affinity()
p.cpu_affinity([0])  # Pin to CPU core

print("Running 1000 inference iterations on a single CPU core...\n")
start_time = time.time()
for _ in range(1000):
    pipeline.predict(sample)
elapsed_time = time.time() - start_time
latency_ms = (elapsed_time / 1000) * 1000  # Convert to milliseconds

p.cpu_affinity(original_affinity)  # Restore original CPU affinity

print("==== Latency Results ====")
print(f"Model: {model_file}")
print(f"Total time for 1000 inferences: {elapsed_time:.3f} seconds")
print(f"Average latency per inference: {latency_ms:.3f} ms")
print(f"CPU Affinity restored to: {original_affinity}")
print("==== End of Latency Results ====\n")