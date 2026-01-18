# database.py
import sqlite3

DB_NAME = "exoplanets.db"

def get_connection():
    return sqlite3.connect(DB_NAME)

def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            planet_radius REAL,
            planet_mass REAL,
            surface_temperature REAL,
            habitability_class TEXT,
            habitability_score REAL
        )
    """)
    conn.commit()
    conn.close()
