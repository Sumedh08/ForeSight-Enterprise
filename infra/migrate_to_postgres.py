"""
Migrate all tables from DuckDB to PostgreSQL.
Usage: python -m infra.migrate_to_postgres
"""
import duckdb
import pandas as pd
from sqlalchemy import create_engine, text
import sys

DUCKDB_PATH = "data/demo.duckdb"
PG_DSN = "postgresql://admin:adminpassword@localhost:5432/natwest_db"


def migrate():
    print("Connecting to DuckDB...")
    duck = duckdb.connect(DUCKDB_PATH, read_only=True)
    tables = [row[0] for row in duck.execute("SHOW TABLES").fetchall()]
    print(f"Found {len(tables)} tables: {tables}")

    print("Connecting to PostgreSQL...")
    engine = create_engine(PG_DSN)

    for table in tables:
        print(f"\n--- Migrating: {table} ---")
        df = duck.execute(f"SELECT * FROM \"{table}\"").df()
        print(f"  Rows: {len(df)}, Columns: {list(df.columns)}")

        # Lowercase table name for PG convention
        pg_table = table.lower()
        df.to_sql(pg_table, engine, if_exists="replace", index=False)
        
        # Verify
        with engine.connect() as conn:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {pg_table}")).scalar()
            print(f"  [OK] Migrated to PG table '{pg_table}' - {count} rows verified")

    duck.close()
    engine.dispose()
    print("\n=== Migration complete ===")


if __name__ == "__main__":
    migrate()
