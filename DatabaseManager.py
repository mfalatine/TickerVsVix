import pyodbc
import pandas as pd
from datetime import datetime
import configparser
import os

class DatabaseManager:
    def __init__(self, config_file=None):
        if config_file is None:
            config_file = os.path.join(os.path.dirname(__file__), 'config.ini')
        self.config = self._load_config(config_file)
        self.connection_string = self._build_connection_string()
    
    def _load_config(self, config_file):
        config = configparser.ConfigParser()
        config.read(config_file)
        if 'DATABASE' not in config:
            raise ValueError(f'Database configuration not found in {config_file}')
        return config['DATABASE']
    
    def _build_connection_string(self):
        return (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={self.config['server']};"
            f"DATABASE={self.config['database']};"
            "Trusted_Connection=yes;"
        )
    
    # ---------------------------------------------------------
    # NO CHANGES: raw second data remains in separate tables
    # like raw1SecondData_ES, raw1SecondData_NQ, etc.
    # ---------------------------------------------------------
    def write_raw_data(self, timestamp, open_price, high, low, close, volume, symbol):
        """
        Write raw second data to the symbol-specific table,
        e.g. dbo.raw1SecondData_ES, raw1SecondData_NQ, etc.
        """
        base_symbol = next((s for s in ['MNQ', 'MES', 'NQ', 'ES'] if symbol.startswith(s)), None)
        if not base_symbol:
            return
        
        # Convert to numeric
        try:
            open_price = float(open_price) if open_price is not None else None
            high = float(high) if high is not None else None
            low = float(low) if low is not None else None
            close = float(close) if close is not None else None
            volume = int(volume) if volume is not None else 0
        except (TypeError, ValueError) as e:
            print(f"Error converting values to numeric: {e}")
            return
        
        sql = f"""
        INSERT INTO dbo.raw1SecondData_{base_symbol}
        (Timestamp, [Open], High, Low, [Close], Volume, Symbol)
        VALUES (
            ?,
            CAST(? AS numeric(18,6)),
            CAST(? AS numeric(18,6)),
            CAST(? AS numeric(18,6)),
            CAST(? AS numeric(18,6)),
            ?,
            ?
        )
        """
        
        with pyodbc.connect(self.connection_string) as conn:
            cursor = conn.cursor()
            cursor.setinputsizes([
                (None),
                pyodbc.SQL_DECIMAL,
                pyodbc.SQL_DECIMAL,
                pyodbc.SQL_DECIMAL,
                pyodbc.SQL_DECIMAL,
                pyodbc.SQL_INTEGER,
                (None)
            ])
            params = [
                timestamp,
                open_price,
                high,
                low,
                close,
                volume,
                symbol
            ]
            cursor.execute(sql, params)
            conn.commit()
    
    # ---------------------------------------------------------
    # CHANGED: unify all 1-minute bars in "OneMinDataHistory"
    # and keep the "latest bar" in a single "Latest1MinuteBar"
    # ---------------------------------------------------------
    def write_minute_data(self, timestamp, open_price, high, low, close, volume, symbol):
        """
        Write 1-minute bar data into:
          1) A single table for all symbols => OneMinDataHistory
          2) The single table for the latest bar => Latest1MinuteBar
        """
        try:
            open_price = float(open_price) if open_price is not None else None
            high = float(high) if high is not None else None
            low = float(low) if low is not None else None
            close = float(close) if close is not None else None
            volume = int(volume) if volume is not None else 0
        except (TypeError, ValueError) as e:
            print(f"Error converting values to numeric: {e}")
            return
        
        # Single table for all 1-minute bars:
        history_sql = """
        INSERT INTO dbo.OneMinDataHistory
        (Symbol, Timestamp, [Open], High, Low, [Close], Volume)
        VALUES (
            ?,
            ?,
            CAST(? AS numeric(18,2)),
            CAST(? AS numeric(18,2)),
            CAST(? AS numeric(18,2)),
            CAST(? AS numeric(18,2)),
            ?
        )
        """
        
        # Single "Latest1MinuteBar" table: always 1 row per symbol
        latest_sql = """
        DELETE FROM dbo.Latest1MinuteBar
        WHERE Symbol = ?;
        
        INSERT INTO dbo.Latest1MinuteBar
        (Symbol, Timestamp, [Open], High, Low, [Close], Volume)
        VALUES (
            ?,
            ?,
            CAST(? AS numeric(18,2)),
            CAST(? AS numeric(18,2)),
            CAST(? AS numeric(18,2)),
            CAST(? AS numeric(18,2)),
            ?
        );
        """
        
        with pyodbc.connect(self.connection_string) as conn:
            cursor = conn.cursor()
            
            # 1) Insert into 1-min history
            history_params = [
                symbol,
                timestamp,
                open_price,
                high,
                low,
                close,
                volume
            ]
            cursor.execute(history_sql, history_params)
            
            # 2) Replace row in the single Latest1MinuteBar
            latest_params = [
                symbol,             # for DELETE
                symbol,             # for INSERT
                timestamp,
                open_price,
                high,
                low,
                close,
                volume
            ]
            cursor.execute(latest_sql, latest_params)
            
            conn.commit()
    
    # ---------------------------------------------------------
    # CHANGED: get_latest_minute_bar now from unified table
    # ---------------------------------------------------------
    def get_latest_minute_bar(self, symbol):
        """
        Returns the single row from dbo.Latest1MinuteBar
        for the given symbol, or None if not found.
        """
        sql = """
        SELECT Symbol,
               Timestamp,
               [Open],
               High,
               Low,
               [Close],
               Volume
        FROM dbo.Latest1MinuteBar
        WHERE Symbol = ?
        """
        
        with pyodbc.connect(self.connection_string) as conn:
            df = pd.read_sql(sql, conn, params=[symbol])
            if not df.empty:
                return {
                    'Date/Time': df.iloc[0]['Timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
                    'Symbol': df.iloc[0]['Symbol'],
                    'Open':   float(df.iloc[0]['Open']),
                    'High':   float(df.iloc[0]['High']),
                    'Low':    float(df.iloc[0]['Low']),
                    'Close':  float(df.iloc[0]['Close']),
                    'Volume': int(df.iloc[0]['Volume'])
                }
            return None

    # ---------------------------------------------------------
    # OPTIONAL: If you want 5-min or 4-hour bars in real time
    # add methods similar to "write_minute_data" but for
    # FiveMinDataHistory or FourHourDataHistory
    # ---------------------------------------------------------
    
    def write_five_minute_data(self, timestamp, open_price, high, low, close, volume, symbol):
        """
        Example: store 5-min bars in dbo.FiveMinDataHistory
        """
        try:
            open_price = float(open_price) if open_price is not None else None
            high = float(high) if high is not None else None
            low = float(low) if low is not None else None
            close = float(close) if close is not None else None
            volume = int(volume) if volume is not None else 0
        except (TypeError, ValueError) as e:
            print(f"Error converting values to numeric: {e}")
            return
        
        sql = """
        INSERT INTO dbo.FiveMinDataHistory
        (Symbol, Timestamp, [Open], High, Low, [Close], Volume)
        VALUES (
            ?,
            ?,
            CAST(? AS numeric(18,2)),
            CAST(? AS numeric(18,2)),
            CAST(? AS numeric(18,2)),
            CAST(? AS numeric(18,2)),
            ?
        )
        """
        
        with pyodbc.connect(self.connection_string) as conn:
            cursor = conn.cursor()
            params = [
                symbol,
                timestamp,
                open_price,
                high,
                low,
                close,
                volume
            ]
            cursor.execute(sql, params)
            conn.commit()

    def write_four_hour_data(self, timestamp, open_price, high, low, close, volume, symbol):
        """
        Example: store 4-hour bars in dbo.FourHourDataHistory
        """
        try:
            open_price = float(open_price) if open_price is not None else None
            high = float(high) if high is not None else None
            low = float(low) if low is not None else None
            close = float(close) if close is not None else None
            volume = int(volume) if volume is not None else 0
        except (TypeError, ValueError) as e:
            print(f"Error converting values to numeric: {e}")
            return
        
        sql = """
        INSERT INTO dbo.FourHourDataHistory
        (Symbol, Timestamp, [Open], High, Low, [Close], Volume)
        VALUES (
            ?,
            ?,
            CAST(? AS numeric(18,2)),
            CAST(? AS numeric(18,2)),
            CAST(? AS numeric(18,2)),
            CAST(? AS numeric(18,2)),
            ?
        )
        """
        
        with pyodbc.connect(self.connection_string) as conn:
            cursor = conn.cursor()
            params = [
                symbol,
                timestamp,
                open_price,
                high,
                low,
                close,
                volume
            ]
            cursor.execute(sql, params)
            conn.commit()
