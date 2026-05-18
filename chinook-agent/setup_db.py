"""Download and initialize the Chinook SQLite database."""

import sqlite3
import requests
from pathlib import Path

CHINOOK_SQL_URL = (
    "https://raw.githubusercontent.com/lerocha/chinook-database"
    "/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
)
DB_PATH = Path(__file__).parent / "chinook.db"


def setup_database() -> Path:
    if DB_PATH.exists():
        return DB_PATH

    print("Downloading Chinook database SQL...")
    response = requests.get(CHINOOK_SQL_URL, timeout=30)
    response.raise_for_status()

    print("Creating SQLite database...")
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(response.text)
    conn.commit()
    conn.close()

    print(f"Database created at {DB_PATH}")
    return DB_PATH


if __name__ == "__main__":
    setup_database()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    print(f"Tables: {', '.join(tables)}")
