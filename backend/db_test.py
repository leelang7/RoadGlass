# db_test.py — pg8000 + RDS SSL + JSON insert to `data`
import json, ssl, pg8000

ctx = ssl.create_default_context(cafile="/etc/ssl/certs/rds-us-west-1-bundle.pem")

conn = pg8000.connect(
    host="seoul-ht-04.cpk0oamsu0g6.us-west-1.rds.amazonaws.com",
    port=5432, database="postgres", user="postgres", password="postgres",
    ssl_context=ctx,
)
cur = conn.cursor()

cur.execute("SELECT current_user, current_database()"); print("✅", cur.fetchone())

event = {"type": "ping", "ok": True}
cur.execute("INSERT INTO raw.events (data) VALUES (%s::jsonb)", [json.dumps(event)])  # ← 여기!
conn.commit()

cur.close(); conn.close()
print("✅ INSERT OK")
