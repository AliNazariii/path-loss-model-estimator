import sqlite3

def get_connection(path: str = 'db.sqlite3') -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    return conn

def get_cursor(conn: sqlite3.Connection = None) -> sqlite3.Cursor:
    if conn is None:
        conn = get_connection()
    cur = conn.cursor()
    return cur

def execute_query(query: str, conn: sqlite3.Connection = None) -> sqlite3.Cursor:
    if conn is None:
        conn = get_connection()
    cur = get_cursor(conn)
    cur.execute(query)
    return cur