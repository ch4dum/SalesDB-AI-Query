#%% Imports
import os
import re
import psycopg2
import pandas as pd

# %% Connect
conn = psycopg2.connect(
    dbname="d365_database",
    user="postgres",
    password="Csi@dmin9",
    host="localhost",
    port="5432"
)
cursor = conn.cursor()

# Create schema
cursor.execute("CREATE SCHEMA IF NOT EXISTS public;")
conn.commit()

csv_folder = "./d365_csv"
files = os.listdir(csv_folder)

# %% Table creator
def ensure_table_schema(cursor, schema, table_name, df):
    full_table = f'{schema}."{table_name}"'

    # Create table if needed
    columns_str = ", ".join([f'"{col}" TEXT' for col in df.columns])
    cursor.execute(f'CREATE TABLE IF NOT EXISTS {full_table} ({columns_str});')

    # Check missing columns
    cursor.execute(f"""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_schema = %s AND table_name = %s;
    """, (schema, table_name))
    existing_columns = set(row[0] for row in cursor.fetchall())
    missing_columns = set(df.columns) - existing_columns

    for col in missing_columns:
        cursor.execute(f'ALTER TABLE {full_table} ADD COLUMN "{col}" TEXT;')
        print(f"+++ Added column {col} to table {schema}.{table_name}")

    print(f"+++ Table {schema}.{table_name} is ready.")

# %% Main loop
for file in sorted(files):
    # Only .csv and no "__"
    if not file.endswith(".csv") or "__" in file:
        print(f"--- Skipped (relationship or not csv): {file}")
        continue

    table_name = os.path.splitext(file)[0]
    table_name = re.sub(r'\W+', '_', table_name.strip().lower())

    csv_path = os.path.join(csv_folder, file)

    try:
        df = pd.read_csv(csv_path)

        if len(df.columns) == 0:
            print(f"--- Skipped {file}: No columns found.")
            continue

        # Clean column names
        df.columns = (
            df.columns
            .str.replace('@', '', regex=False)
            .str.replace('.', '_', regex=False)
            .str.strip()
        )

        ensure_table_schema(cursor, "public", table_name, df)

        # Insert rows
        placeholders = ", ".join(["%s"] * len(df.columns))
        columns_str = ", ".join([f'"{col}"' for col in df.columns])
        insert_sql = f'INSERT INTO public."{table_name}" ({columns_str}) VALUES ({placeholders})'

        for row in df.itertuples(index=False):
            cursor.execute(insert_sql, row)

        conn.commit()
        print(f"+++ Inserted data into public.{table_name} from {file}")

    except Exception as e:
        print(f"!!! Error processing {file}: {e}")
        conn.rollback()

# %% Cleanup
cursor.close()
conn.close()

# %%
