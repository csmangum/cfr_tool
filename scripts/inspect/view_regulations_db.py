import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np

def connect_to_db():
    """Connect to the regulations database."""
    db_path = Path("data/db/regulations.db")
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}")
    return sqlite3.connect(db_path)

def list_tables(conn):
    """List all tables in the database."""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    return [table[0] for table in tables]

def format_value(value):
    """Format values for display."""
    if pd.isna(value):
        return "NULL"
    elif isinstance(value, (int, np.integer)):
        return str(value)
    elif isinstance(value, (float, np.floating)):
        return f"{value:.2f}"
    elif isinstance(value, str) and len(value) > 50:
        return value[:47] + "..."
    return str(value)

def view_table(conn, table_name):
    """View detailed contents and statistics of a specific table using pandas."""
    try:
        # Configure pandas display options
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 50)
        pd.set_option('display.precision', 2)
        
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        
        print(f"\n{'='*100}")
        print(f"TABLE: {table_name}".center(100))
        print(f"{'='*100}")
        
        # Basic information
        print(f"Number of rows: {len(df)}")
        print(f"Number of columns: {len(df.columns)}")
        
        # Add agency count if agency column exists
        agency_columns = [col for col in df.columns if 'agency' in col.lower()]
        if agency_columns:
            print("\nAGENCY COUNTS:")
            print("-" * 85)
            for agency_col in agency_columns:
                agency_counts = df[agency_col].value_counts()
                print(f"\nCounts for {agency_col}:")
                for agency, count in agency_counts.items():
                    print(f"{agency:<50} | {count:>6}")
            print()
        
        # Column information
        print("\nCOLUMN INFORMATION:")
        print("-" * 85)
        print(f"{'Column Name':<30} | {'Type':<12} | {'Non-Null':<10} | {'Unique Values':<15}")
        print("-" * 85)
        for col in df.columns:
            dtype = str(df[col].dtype)
            non_null = df[col].count()
            n_unique = df[col].nunique()
            print(f"{str(col):<30} | {dtype:<12} | {non_null:<10} | {n_unique:<15}")

        # Numeric statistics
        numeric_cols = df.select_dtypes(include=['int64', 'float64', 'Int64']).columns
        if not numeric_cols.empty:
            print("\nNUMERIC COLUMN STATISTICS:")
            print("-" * 100)
            stats_df = df[numeric_cols].describe()
            stats_df = stats_df.round(2)
            with pd.option_context('display.float_format', '{:,.2f}'.format):
                print(stats_df.to_string())

        # Sample rows - split into sections if too many columns
        print("\nSAMPLE ROWS:")
        sample_df = df.sample(n=min(10, len(df)))
        
        # Create a formatted version of the sample dataframe
        formatted_df = pd.DataFrame()
        for col in sample_df.columns:
            formatted_df[col] = sample_df[col].map(format_value)

        # Split columns into sections of 8 columns each
        COLS_PER_SECTION = 8
        column_sections = [
            list(formatted_df.columns[i:i + COLS_PER_SECTION])
            for i in range(0, len(formatted_df.columns), COLS_PER_SECTION)
        ]

        # Display each section
        for section_num, columns in enumerate(column_sections, 1):
            section_df = formatted_df[columns]
            
            print(f"\nSection {section_num} of {len(column_sections)}:")
            print("-" * 100)
            
            # Calculate column widths for this section
            col_widths = {}
            for col in section_df.columns:
                col_width = max(
                    len(str(col)),
                    section_df[col].str.len().max(),
                    15
                )
                col_widths[col] = col_width

            # Print headers for this section
            header_row = " | ".join(f"{col:<{col_widths[col]}}" for col in section_df.columns)
            print(header_row)
            print("-" * len(header_row))
            
            # Print rows for this section
            for _, row in section_df.iterrows():
                row_str = " | ".join(f"{str(val):<{col_widths[col]}}" for col, val in row.items())
                print(row_str)
            
            print()

    except Exception as e:
        print(f"Error viewing table {table_name}: {e}")
        import traceback
        print(traceback.format_exc())

def main():
    try:
        conn = connect_to_db()
        tables = list_tables(conn)
        
        if not tables:
            print("No tables found in the database.")
            return

        print(f"Found {len(tables)} tables in the database.")
        
        # View all tables automatically
        for table in tables:
            view_table(conn, table)

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main() 