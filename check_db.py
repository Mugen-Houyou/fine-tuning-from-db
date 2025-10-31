import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv('DB_HOST'),
    port=os.getenv('DB_PORT'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'),
    database=os.getenv('DB_NAME')
)

cur = conn.cursor()

# Get table structure
cur.execute("""
    SELECT column_name, data_type, character_maximum_length
    FROM information_schema.columns
    WHERE table_name = 'gabe_chat_records'
    ORDER BY ordinal_position
""")
print('=== Table Structure ===')
for row in cur.fetchall():
    print(f'  {row[0]}: {row[1]}' + (f'({row[2]})' if row[2] else ''))

# Get record count
cur.execute('SELECT COUNT(*) FROM gabe_chat_records')
print(f'\n=== Total Records: {cur.fetchone()[0]} ===')

# Get sample records
cur.execute('SELECT * FROM gabe_chat_records LIMIT 3')
cols = [desc[0] for desc in cur.description]
print(f'\n=== Sample Records ===')
print(f'Columns: {cols}')
for row in cur.fetchall():
    print(row)

conn.close()
