import psycopg2
from dotenv import load_dotenv
import os
from collections import Counter

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv('DB_HOST'),
    port=os.getenv('DB_PORT'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'),
    database=os.getenv('DB_NAME')
)

cur = conn.cursor()

# Get user distribution
cur.execute("""
    SELECT user_name, COUNT(*) as message_count
    FROM gabe_chat_records
    GROUP BY user_name
    ORDER BY message_count DESC
    LIMIT 20
""")
print('=== Top 20 Users by Message Count ===')
for row in cur.fetchall():
    print(f'  {row[0]}: {row[1]} messages')

# Get workflow stage distribution
cur.execute("""
    SELECT workflow_stage_per_slice, COUNT(*) as count
    FROM gabe_chat_records
    WHERE workflow_stage_per_slice IS NOT NULL
    GROUP BY workflow_stage_per_slice
    ORDER BY count DESC
""")
print('\n=== Workflow Stages ===')
for row in cur.fetchall():
    print(f'  {row[0]}: {row[1]}')

# Get tag distribution
cur.execute("""
    SELECT unnest(tags_per_message) as tag, COUNT(*) as count
    FROM gabe_chat_records
    WHERE tags_per_message IS NOT NULL
    GROUP BY tag
    ORDER BY count DESC
    LIMIT 20
""")
print('\n=== Top 20 Message Tags ===')
for row in cur.fetchall():
    print(f'  {row[0]}: {row[1]}')

# Get sample conversation by user
cur.execute("""
    SELECT user_name, message_text, tags_per_message, workflow_stage_per_slice
    FROM gabe_chat_records
    WHERE user_name = '씨너지파트너 주식회사'
    LIMIT 10
""")
print('\n=== Sample: 씨너지파트너 주식회사 ===')
for row in cur.fetchall():
    print(f'\n  Message: {row[1][:100]}...')
    print(f'  Tags: {row[2]}')
    print(f'  Stage: {row[3]}')

conn.close()
