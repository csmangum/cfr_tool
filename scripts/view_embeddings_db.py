import sqlite3
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd


def connect_to_db():
    """Connect to the embeddings database."""
    db_path = Path("data/db/regulation_embeddings.db")
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}")
    return sqlite3.connect(db_path)


def format_embedding(embedding: List[float], max_dims: int = 5) -> str:
    """Format embedding vector for display."""
    if not isinstance(embedding, list):
        try:
            embedding = np.frombuffer(embedding, dtype=np.float32).tolist()
        except:
            return "Invalid embedding format"

    total_dims = len(embedding)
    if total_dims <= max_dims:
        return f"[{', '.join(f'{x:.4f}' for x in embedding)}]"
    return f"[{', '.join(f'{x:.4f}' for x in embedding[:max_dims])}...] ({total_dims} dims)"


def format_value(value: Any) -> str:
    """Format values for display."""
    if pd.isna(value):
        return "NULL"
    elif isinstance(value, (bytes, bytearray)):
        try:
            # Try to interpret as embedding
            return format_embedding(value)
        except:
            return f"<binary data: {len(value)} bytes>"
    elif isinstance(value, (int, np.integer)):
        return str(value)
    elif isinstance(value, (float, np.floating)):
        return f"{value:.4f}"
    elif isinstance(value, str) and len(value) > 100:
        return value[:97] + "..."
    return str(value)


def get_embedding_stats(df: pd.DataFrame, col_name: str) -> dict:
    """Calculate statistics for embedding column."""
    try:
        # Convert first embedding to get dimensions
        sample_embedding = np.frombuffer(df[col_name].iloc[0], dtype=np.float32)
        dims = len(sample_embedding)

        # Convert all embeddings
        embeddings = np.vstack(
            [
                np.frombuffer(emb, dtype=np.float32)
                for emb in df[col_name]
                if emb is not None
            ]
        )

        return {
            "dimensions": dims,
            "mean_magnitude": float(np.linalg.norm(embeddings, axis=1).mean()),
            "std_magnitude": float(np.linalg.norm(embeddings, axis=1).std()),
            "mean_values": embeddings.mean(axis=0)[:5].tolist(),  # First 5 dimensions
            "std_values": embeddings.std(axis=0)[:5].tolist(),  # First 5 dimensions
        }
    except Exception as e:
        return {"error": str(e)}


def view_table(conn: sqlite3.Connection, table_name: str):
    """View contents of a specific table with special handling for embeddings."""
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

        print(f"\n{'='*100}")
        print(f"TABLE: {table_name}".center(100))
        print(f"{'='*100}")

        # Basic information
        print(f"Number of rows: {len(df)}")
        print(f"Number of columns: {len(df.columns)}")

        # Add agency count if agency column exists
        agency_columns = [col for col in df.columns if "agency" in col.lower()]
        if agency_columns:
            print("\nAGENCY COUNTS:")
            print("-" * 85)
            for agency_col in agency_columns:
                agency_counts = df[agency_col].value_counts()
                print(f"\nCounts for {agency_col}:")
                for agency, count in agency_counts.items():
                    print(f"{agency:<50} | {count:>6}")
            print()

        # Column information with special handling for embedding columns
        print("\nCOLUMN INFORMATION:")
        print("-" * 85)
        print(f"{'Column Name':<30} | {'Type':<12} | {'Non-Null':<10} | {'Notes':<25}")
        print("-" * 85)

        embedding_columns = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            non_null = df[col].count()

            # Detect potential embedding columns
            if (
                dtype == "object"
                and df[col].iloc[0] is not None
                and isinstance(df[col].iloc[0], (bytes, bytearray))
            ):
                notes = "Embedding column"
                embedding_columns.append(col)
            else:
                n_unique = df[col].nunique()
                notes = f"{n_unique} unique values"

            print(f"{str(col):<30} | {dtype:<12} | {non_null:<10} | {notes:<25}")

        # Embedding statistics
        for col in embedding_columns:
            print(f"\nEMBEDDING STATISTICS FOR {col}:")
            print("-" * 100)
            stats = get_embedding_stats(df, col)
            if "error" in stats:
                print(f"Error analyzing embeddings: {stats['error']}")
            else:
                print(f"Dimensions: {stats['dimensions']}")
                print(f"Mean magnitude: {stats['mean_magnitude']:.4f}")
                print(f"Std magnitude: {stats['std_magnitude']:.4f}")
                print(
                    f"Mean values (first 5 dims): {[f'{x:.4f}' for x in stats['mean_values']]}"
                )
                print(
                    f"Std values (first 5 dims): {[f'{x:.4f}' for x in stats['std_values']]}"
                )

        # Sample rows
        print("\nSAMPLE ROWS:")
        sample_df = df.sample(n=min(5, len(df)))

        # Create formatted version of sample data
        formatted_df = pd.DataFrame()
        for col in sample_df.columns:
            formatted_df[col] = sample_df[col].map(format_value)

        # Split into sections for readability
        COLS_PER_SECTION = 4
        column_sections = [
            list(formatted_df.columns[i : i + COLS_PER_SECTION])
            for i in range(0, len(formatted_df.columns), COLS_PER_SECTION)
        ]

        for section_num, columns in enumerate(column_sections, 1):
            section_df = formatted_df[columns]

            print(f"\nSection {section_num} of {len(column_sections)}:")
            print("-" * 100)

            # Calculate column widths
            col_widths = {
                col: max(len(str(col)), section_df[col].str.len().max(), 15)
                for col in section_df.columns
            }

            # Print headers
            header_row = " | ".join(
                f"{col:<{col_widths[col]}}" for col in section_df.columns
            )
            print(header_row)
            print("-" * len(header_row))

            # Print rows
            for _, row in section_df.iterrows():
                row_str = " | ".join(
                    f"{str(val):<{col_widths[col]}}" for col, val in row.items()
                )
                print(row_str)
            print()

    except Exception as e:
        print(f"Error viewing table {table_name}: {e}")
        import traceback

        print(traceback.format_exc())


def main():
    try:
        conn = connect_to_db()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [table[0] for table in cursor.fetchall()]

        if not tables:
            print("No tables found in the database.")
            return

        print(f"Found {len(tables)} tables in the database.")

        for table in tables:
            view_table(conn, table)

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if "conn" in locals():
            conn.close()


if __name__ == "__main__":
    main()
