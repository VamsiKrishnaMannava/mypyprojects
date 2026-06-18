import pandas as pd

from ydata_profiling import ProfileReport

#Load data
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv")

print(df.head())


#EDA
profile = ProfileReport(df, title="Data Profiling Report", explorative=True)
profile.to_file("data_profile.html")



