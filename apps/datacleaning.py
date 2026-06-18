#pip install "setuptools<82.0.0"

import pandas as pd
# pyrefly: ignore [missing-import]
from ydata_profiling import ProfileReport

#Load data
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv")

print(df.head())


#EDA
profile = ProfileReport(df, title="Data Profiling Report", explorative=True)
profile.to_file("data_profile.html")



