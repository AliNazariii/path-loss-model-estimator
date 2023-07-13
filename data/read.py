import pandas as pd
from db.utils import execute_query
import sqlite3

def read_data(conn: sqlite3.Connection):
    cur = execute_query("SELECT rsrp, latitude, longitude FROM 'tbl_data'", conn)
    results = cur.fetchall()

    df = pd.DataFrame(results)
    df.columns = ['rsrp', 'latitude', 'longitude']

    return df
