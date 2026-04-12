import duckdb
from infra.settings import settings

conn = duckdb.connect(str(settings.duckdb_path), read_only=True)
sql = "SELECT timestamp AS period, TRY_CAST(REGEXP_REPLACE(price_usd, '[^0-9.\\-]', '', 'g') AS DOUBLE) AS value FROM cryptocurrency WHERE 1=1 AND timestamp IS NOT NULL AND price_usd IS NOT NULL ORDER BY timestamp"
df = conn.execute(sql).fetchdf()
print("Rows:", len(df))
print("Dtypes:", dict(df.dtypes))
print(df.head(3))
conn.close()
