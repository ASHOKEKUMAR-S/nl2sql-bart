# src/db_utils.py

from sqlalchemy import create_engine
import pandas as pd
import dotenv
import os 

dotenv.load_dotenv()

def create_db_engine(
    user="admin",
    password=os.getenv("MYSQL_PASSWORD"),
    host="localhost",
    port=3306,
    dbname="cmdb"
):
    connection_string = f"mysql+pymysql://{user}:{password}@{host}:{port}/{dbname}"
    return create_engine(connection_string)

def execute_sql(query: str, engine) -> pd.DataFrame | str:
    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        return str(e)  # Useful for logging execution failures
