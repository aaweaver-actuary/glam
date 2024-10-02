"""Define a helper class to use duckdb databases."""

from __future__ import annotations
from dataclasses import dataclass
import duckdb
import pandas as pd
import polars as pl
import pyodbc  # type: ignore
import types

__all__ = ["DuckDB", "HitRatioDB", "Db2"]

HIT_RATIO_DB_FILEPATH = (
    "/parm_share/small_business/modeling/bop_modeling_data/hit_ratio_db.duckdb"
)


@dataclass
class DuckDB:
    """Define a generic DuckDB helper class."""

    db_file: str = ":memory:"
    db_conn: duckdb.DuckDBPyConnection | None = None

    slots = "db_file"

    def __call__(self, query: str) -> pl.DataFrame | None:
        """Execute a read-only query when the object is called."""
        return self.read(query)

    def write(self, query: str) -> pl.DataFrame | None:
        """Execute a write query."""
        with duckdb.connect(self.db_file, read_only=False) as conn:
            res = conn.sql(query)
            if res is not None:
                return res.pl()
            return None

    def read(self, query: str) -> pl.DataFrame | None:
        """Execute a read-only query."""
        with duckdb.connect(self.db_file, read_only=True) as conn:
            res = conn.sql(query)
            if res is not None:
                return res.pl()
            return None

    def __enter__(self):
        """Begin a context manager."""
        return duckdb.connect(self.db_file)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        """Close the connection when the context manager exits."""
        self.db_conn.close() if self.db_conn is not None else None
        self.db_conn = None

    def get_user_defined_enum_types(self) -> list[str]:
        """Return a list of user-defined enum types."""
        enum_qry = """
            select type_name
            from duckdb_types()
            where 
                (
                    ends_with(type_name, '__type')
                    or ends_with(type_name, 'type')
                )
                and logical_type='ENUM'
        """
        try:
            with duckdb.connect(self.db_file) as conn:
                ud_types__RAW = conn.sql(enum_qry)
                ud_types = (
                    ud_types__RAW.to_pandas()["type_name"].tolist()  # type: ignore
                    if ud_types__RAW.shape[0] > 0
                    else []
                )

        except Exception as _:
            ud_types = []

        return ud_types

    def clear_user_defined_enum_type(self, type_name: str) -> None:
        """Clear a user-defined enum type."""
        ud_types = self.get_user_defined_enum_types()
        if type_name not in ud_types:
            err_msg = (
                f"`{type_name}` does not appear in the list of user-defined enum types:"
            )
            err_msg += "\n"
            err_msg += "\n".join(ud_types)

        drop_qry = f"drop type {type_name}"
        self.write(drop_qry)

    def clear_all_user_defined_enum_types(self) -> None:
        """Clear all user-defined enum types."""
        for t in self.get_user_defined_enum_types():
            self.clear_user_defined_enum_type(t)

    def create_or_replace_string_enum(
        self, old_col: str, new_col: str, table: str
    ) -> None:
        """Create a new enum type from the values in a table."""
        type_name = f"{new_col}__type"

        # Drop the enum type if it is already defined
        if type_name in self.get_user_defined_enum_types():
            self.clear_user_defined_enum_type(type_name)
        elif type_name.replace("__", "_") in self.get_user_defined_enum_types():
            self.clear_user_defined_enum_type(type_name.replace("__", "_"))

        # Create the new enum type from the values from the table
        create_type_qry = f"create type {type_name} as enum (select distinct {old_col} from {table} where {old_col} is not null);"  # noqa: S608
        self.write(create_type_qry)


@dataclass
class HitRatioDB(DuckDB):
    """Specify a specific database."""

    db_file: str = HIT_RATIO_DB_FILEPATH
    slots = "db_file"


@dataclass
class Db2:
    """Specify a DB2 connection."""

    username: str
    password: str
    driver: str = "/opt/ibm/db2/clidriver/lib/libdb2.so"
    hostname: str = "lnxvdb2hq020.cinfin.com"
    port: int = 50004
    protocol: str = "TCPIP"
    database: str = "CFCPSAS"

    slots = (
        "username",
        "password",
        "driver",
        "hostname",
        "port",
        "protocol",
        "database",
    )

    def get_conn_str(self) -> str:
        """Return the connection string."""
        return f"Driver={self.driver}; Hostname={self.hostname};Port={self.port};Protocol={self.protocol};Database={self.database};UID={self.username};PWD={self.password};"

    def get_conn(self) -> pyodbc.Connection:
        """Return the connection object."""
        return pyodbc.connect(self.get_conn_str())

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        return f"DB2Conn({', '.join(f'{slot}={getattr(self, slot)}' for slot in self.slots)})"

    def __str__(self) -> str:
        """Return a string representation of the object."""
        return self.get_conn_str()

    def __call__(self, query: str) -> pl.DataFrame:
        """Perform a read-only query when the object is called."""
        with self.get_conn() as conn, conn.cursor() as cursor:
            cursor.execute(query)
            raw = cursor.fetchall()
            columns = [column[0] for column in cursor.description]
            return pl.from_pandas(pd.DataFrame.from_records(raw, columns=columns))
