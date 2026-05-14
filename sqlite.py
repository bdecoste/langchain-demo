import sqlite3

conn = sqlite3.connect("company.db")
cursor = conn.cursor()

cursor.executescript("""
CREATE TABLE IF NOT EXISTS employees (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    department TEXT NOT NULL,
    salary INTEGER,
    hire_date TEXT
);

CREATE TABLE IF NOT EXISTS departments (
    name TEXT PRIMARY KEY,
    budget INTEGER,
    location TEXT
);

INSERT OR REPLACE INTO employees VALUES
    (1, 'Alice Chen',    'Engineering', 145000, '2021-03-15'),
    (2, 'Bob Martinez',  'Engineering', 130000, '2022-07-01'),
    (3, 'Carol Singh',   'Sales',       95000,  '2020-11-20'),
    (4, 'David Kim',     'Sales',       88000,  '2023-02-10'),
    (5, 'Eva Johnson',   'Marketing',   105000, '2021-09-05'),
    (6, 'Frank Liu',     'Engineering', 165000, '2019-06-12');

INSERT OR REPLACE INTO departments VALUES
    ('Engineering', 2000000, 'San Francisco'),
    ('Sales',       800000,  'New York'),
    ('Marketing',   500000,  'Austin');
""")

conn.commit()
conn.close()
