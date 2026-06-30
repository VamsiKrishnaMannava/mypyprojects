import duckdb

duckdb.query("""
SELECT distinct RTPLOG
FROM 'C:\\Users\\vakmannava\\Downloads\\parquet\\kingfisher\\*.parquet'
""").show()





