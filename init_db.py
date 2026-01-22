import sqlite3

con = sqlite3.connect("database.db")
cur = con.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS exoplanets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    planet_name TEXT,

    pl_orbper REAL,
    pl_orbeccen REAL,
    pl_rade REAL,
    pl_bmasse REAL,
    pl_eqt REAL,
    pl_insol REAL,

    st_teff REAL,
    st_rad REAL,
    st_mass REAL,
    st_lum REAL,

    sy_dist REAL,

    habitability_score REAL
)
""")

con.commit()
con.close()

print("✅ Database initialized successfully")
