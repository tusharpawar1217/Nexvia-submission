"""
Database connection and utility functions.
"""
import psycopg2
from psycopg2 import sql
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
import pandas as pd
from typing import Optional, Dict, Any
from contextlib import contextmanager
import logging

from config import DB_CONFIG, DB_CONNECTION_STRING

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_psycopg2_connection():
    """
    Get a psycopg2 connection to the PostgreSQL database.
    
    Returns:
        psycopg2.connection: Database connection object
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        logger.info(f"Connected to database: {DB_CONFIG['database']}")
        return conn
    except psycopg2.Error as e:
        logger.error(f"Failed to connect to database: {e}")
        raise


def get_sqlalchemy_engine() -> Engine:
    """
    Get a SQLAlchemy engine for the PostgreSQL database.
    
    Returns:
        Engine: SQLAlchemy engine object
    """
    try:
        # Use psycopg2 driver explicitly
        connection_string = DB_CONNECTION_STRING.replace('postgresql://', 'postgresql+psycopg2://')
        engine = create_engine(connection_string)
        logger.info(f"Created SQLAlchemy engine for: {DB_CONFIG['database']}")
        return engine
    except Exception as e:
        logger.error(f"Failed to create SQLAlchemy engine: {e}")
        raise


@contextmanager
def get_db_connection():
    """
    Context manager for database connections.
    Automatically closes connection after use.
    
    Usage:
        with get_db_connection() as conn:
            # use connection
    """
    conn = get_psycopg2_connection()
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database error, rolling back: {e}")
        raise
    finally:
        conn.close()
        logger.info("Database connection closed")


def execute_sql_file(filepath: str) -> None:
    """
    Execute SQL commands from a file.
    
    Args:
        filepath: Path to the SQL file
    """
    with open(filepath, 'r') as f:
        sql_commands = f.read()
    
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(sql_commands)
                logger.info(f"Successfully executed SQL file: {filepath}")
            except psycopg2.Error as e:
                logger.error(f"Error executing SQL file {filepath}: {e}")
                raise


def execute_query(query: str, params: Optional[tuple] = None) -> pd.DataFrame:
    """
    Execute a SELECT query and return results as a DataFrame.
    
    Args:
        query: SQL query string
        params: Optional query parameters (for parameterized queries)
    
    Returns:
        pd.DataFrame: Query results
    """
    engine = get_sqlalchemy_engine()
    try:
        if params:
            df = pd.read_sql_query(text(query), engine, params=params)
        else:
            df = pd.read_sql_query(text(query), engine)
        logger.info(f"Query executed successfully, returned {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        raise
    finally:
        engine.dispose()


def load_dataframe_to_db(
    df: pd.DataFrame,
    table_name: str,
    if_exists: str = "replace",
    index: bool = False,
    chunksize: int = 1000
) -> None:
    """
    Load a pandas DataFrame to a PostgreSQL table.
    
    Args:
        df: DataFrame to load
        table_name: Name of the target table
        if_exists: What to do if table exists ('fail', 'replace', 'append')
        index: Whether to write DataFrame index
        chunksize: Number of rows to write at a time
    """
    engine = get_sqlalchemy_engine()
    try:
        df.to_sql(
            table_name,
            engine,
            if_exists=if_exists,
            index=index,
            chunksize=chunksize,
            method="multi"
        )
        logger.info(f"Loaded {len(df)} rows to table '{table_name}' (if_exists={if_exists})")
    except Exception as e:
        logger.error(f"Error loading DataFrame to {table_name}: {e}")
        raise
    finally:
        engine.dispose()


def get_table_row_count(table_name: str) -> int:
    """
    Get the number of rows in a table.
    
    Args:
        table_name: Name of the table
    
    Returns:
        int: Number of rows
    """
    query = f"SELECT COUNT(*) as count FROM {table_name}"
    df = execute_query(query)
    return int(df['count'].iloc[0])


def table_exists(table_name: str) -> bool:
    """
    Check if a table exists in the database.
    
    Args:
        table_name: Name of the table
    
    Returns:
        bool: True if table exists, False otherwise
    """
    query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = %s
        )
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (table_name,))
            return cur.fetchone()[0]


def get_table_info(table_name: str) -> pd.DataFrame:
    """
    Get column information for a table.
    
    Args:
        table_name: Name of the table
    
    Returns:
        pd.DataFrame: Table schema information
    """
    query = """
        SELECT 
            column_name,
            data_type,
            is_nullable,
            column_default
        FROM information_schema.columns
        WHERE table_name = %s
        ORDER BY ordinal_position
    """
    with get_db_connection() as conn:
        return pd.read_sql_query(query, conn, params=(table_name,))


def test_connection() -> bool:
    """
    Test database connection.
    
    Returns:
        bool: True if connection successful
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version();")
                version = cur.fetchone()[0]
                logger.info(f"PostgreSQL version: {version}")
                return True
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return False


if __name__ == "__main__":
    # Test the database connection
    print("Testing database connection...")
    if test_connection():
        print("✓ Database connection successful!")
    else:
        print("✗ Database connection failed!")
