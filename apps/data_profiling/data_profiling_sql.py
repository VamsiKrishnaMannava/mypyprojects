#pip install "setuptools<82.0.0" "sqlalchemy>=2.0,<3.0" pyodbc ydata-profiling pandas requests urllib3 os
import os
import urllib.parse
from sqlalchemy import create_engine
import pandas as pd
# pyrefly: ignore [missing-import]
from ydata_profiling import ProfileReport


# # powershell env variables:
# $Env:DB_SERVER = '***'
# $Env:DB_NAME   = '***'
# $Env:DB_USER   = '***'
# $Env:DB_PASS   = '***'
 
#Load data
params = urllib.parse.quote_plus(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    f"SERVER={os.environ['DB_SERVER']};"
    f"DATABASE={os.environ['DB_NAME']};"
    f"UID={os.environ['DB_USER']};"
    f"PWD={os.environ['DB_PASS']};"
)

engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")
tablename = 'dbo.BINDER_'

# Optional: limit rows if table is large
df = pd.read_sql(f"SELECT TOP 10000 * FROM {tablename}", engine)
print(df.head())


#EDA
profile = ProfileReport(df, title="Data Profiling Report", explorative=True)
profile.to_file(r"C:\Users\vakmannava\Downloads\data_profile_sql.html")
