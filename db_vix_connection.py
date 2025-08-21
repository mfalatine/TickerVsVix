########## db_vix_connection.py ##########
 
import pyodbc
import datetime
import databento as db
from zoneinfo import ZoneInfo
import pandas as pd
import sys
import os
import time
import threading
import configparser
from DatabaseManager import DatabaseManager

# System timeout settings (in seconds)
RETRY_DELAY = 30  # Delay between connection retry attempts
MONITOR_INTERVAL = 60  # How often to check for config changes

# VIX futures symbols - modify as needed
VALID_SYMBOLS = ['VX']  # Base symbol for VIX futures

# Read Config.ini file
config_file = os.path.join(os.path.dirname(__file__), 'config_vix.ini')

# Global variables
is_live_connected = False
client = None
api_key = None
retry_cycle = 0
db_manager = None

symbols_subscribed = []
instrument_map = {}  # {instrument_id: symbol}
current_day = None

volumes = {}  # {symbol: {hour_start: volume}}
volumes_per_minute = {}  # {symbol: {minute_start: volume}}
unmapped_instrument_volumes = {}  # {instrument_id: {hour_start: vol}}
unmapped_instrument_minute_volumes = {}  # {instrument_id: {minute_minute: vol}}

def load_config():
    global api_key
    config = configparser.ConfigParser()
    config.read(config_file)
    api_key = config['API']['key']

def check_existing_instances():
    """Initialize database connection and verify connectivity"""
    try:
        db_manager = DatabaseManager()
        with pyodbc.connect(db_manager.connection_string) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
        return db_manager
    except Exception as e:
        print(f"Database connection error: {e}")
        print("Please check your database connection and try again")
        sys.exit(1)

def record_hour_volume_for_symbol(sym, timestamp, volume):
    dt = pd.to_datetime(timestamp)
    hour_start = dt.replace(minute=0, second=0, microsecond=0)
    if sym not in volumes:
        volumes[sym] = {}
    if hour_start not in volumes[sym]:
        volumes[sym][hour_start] = 0
    volumes[sym][hour_start] += volume

def record_minute_volume_for_symbol(sym, timestamp, volume):
    dt = pd.to_datetime(timestamp)
    minute_start = dt.replace(second=0, microsecond=0)
    if sym not in volumes_per_minute:
        volumes_per_minute[sym] = {}
    if minute_start not in volumes_per_minute[sym]:
        volumes_per_minute[sym][minute_start] = 0
    volumes_per_minute[sym][minute_start] += volume

def record_hour_volume_for_instrument(instr_id, timestamp, volume):
    dt = pd.to_datetime(timestamp)
    hour_start = dt.replace(minute=0, second=0, microsecond=0)
    if instr_id not in unmapped_instrument_volumes:
        unmapped_instrument_volumes[instr_id] = {}
    if hour_start not in unmapped_instrument_volumes[instr_id]:
        unmapped_instrument_volumes[instr_id][hour_start] = 0
    unmapped_instrument_volumes[instr_id][hour_start] += volume

def record_minute_volume_for_instrument(instr_id, timestamp, volume):
    dt = pd.to_datetime(timestamp)
    minute_start = dt.replace(second=0, microsecond=0)
    if instr_id not in unmapped_instrument_minute_volumes:
        unmapped_instrument_minute_volumes[instr_id] = {}
    if minute_start not in unmapped_instrument_minute_volumes[instr_id]:
        unmapped_instrument_minute_volumes[instr_id][minute_start] = 0
    unmapped_instrument_minute_volumes[instr_id][minute_start] += volume

def transfer_instrument_volumes_to_symbol(instr_id, symbol):
    if instr_id in unmapped_instrument_volumes:
        for hour_start, vol in unmapped_instrument_volumes[instr_id].items():
            if symbol not in volumes:
                volumes[symbol] = {}
            if hour_start not in volumes[symbol]:
                volumes[symbol][hour_start] = 0
            volumes[symbol][hour_start] += vol
        del unmapped_instrument_volumes[instr_id]

    if instr_id in unmapped_instrument_minute_volumes:
        for minute_start, vol in unmapped_instrument_minute_volumes[instr_id].items():
            if symbol not in volumes_per_minute:
                volumes_per_minute[symbol] = {}
            if minute_start not in volumes_per_minute[symbol]:
                volumes_per_minute[symbol][minute_start] = 0
            volumes_per_minute[symbol][minute_start] += vol
        del unmapped_instrument_minute_volumes[instr_id]

def ohlcv_callback(msg):
    """Callback for receiving OHLCV 1s data from DataBento for VIX futures."""
    global current_day
    rtype = msg.hd.rtype
    
    if rtype == 22:  # SymbolMappingMsg
        instrument_id = msg.hd.instrument_id
        symbol_name = (
            msg.stype_in_symbol.decode() 
            if hasattr(msg.stype_in_symbol, 'decode') else msg.stype_in_symbol
        )
        instrument_map[instrument_id] = symbol_name
        print(f"[VIX] Mapped instrument_id {instrument_id} to symbol {symbol_name}")
        transfer_instrument_volumes_to_symbol(instrument_id, symbol_name)

    elif rtype == 32:  # OhlcvMsg
        instrument_id = msg.hd.instrument_id
        timestamp_ns = msg.hd.ts_event
        timestamp_s = timestamp_ns / 1e9
        dt_utc = datetime.datetime.fromtimestamp(timestamp_s, datetime.timezone.utc)
        est_zone = ZoneInfo("US/Eastern")
        dt_est = dt_utc.astimezone(est_zone)
        formatted_time = dt_est.strftime("%m/%d/%Y %H:%M:%S")

        # Check if day changed
        current_date = dt_est.date()
        if current_day is None:
            current_day = current_date
        elif current_date > current_day:
            current_day = current_date

        # VIX futures prices are typically in points (e.g., 15.50)
        # Adjust scaling as needed based on how DataBento provides VIX data
        o = float(msg.open) / 1e9
        h = float(msg.high) / 1e9
        l = float(msg.low) / 1e9
        c = float(msg.close) / 1e9
        v = int(msg.volume)

        if instrument_id not in instrument_map:
            record_hour_volume_for_instrument(instrument_id, dt_est, v)
            record_minute_volume_for_instrument(instrument_id, dt_est, v)
            return
        else:
            symbol = instrument_map[instrument_id]
            record_hour_volume_for_symbol(symbol, dt_est, v)
            record_minute_volume_for_symbol(symbol, dt_est, v)

            # Print data as it comes in
            print(
                f"VIX ({symbol}) {formatted_time}: "
                f"O:{o:.2f} H:{h:.2f} L:{l:.2f} C:{c:.2f} V:{v}"
            )

            # Write to database - you may want a separate table for VIX data
            try:
                db_manager.write_raw_data(
                    timestamp=dt_est,
                    open_price=o,
                    high=h,
                    low=l,
                    close=c,
                    volume=v,
                    symbol=symbol
                )
            except Exception as e:
                print(f"Error writing VIX data to database: {e}")

def get_vix_contracts():
    """Get active VIX futures contracts.
    VIX futures typically have monthly expirations.
    You may want to implement logic to get front month and next month contracts.
    """
    # For now, subscribing to all VIX futures
    # You can modify this to get specific contracts like VXF25, VXG25, etc.
    return ["VX.FUT"]

def start_live_connection():
    global is_live_connected, client, retry_cycle, symbols_subscribed

    # Get VIX futures contracts
    symbols_subscribed = get_vix_contracts()
    
    print(f"Subscribing to VIX symbols: {symbols_subscribed}")

    max_retries = 3
    retry_delay = RETRY_DELAY

    while not is_live_connected:
        retry_cycle += 1
        print(f"[VIX] Starting retry cycle {retry_cycle}...")
        attempt = 0

        while attempt < max_retries:
            try:
                if is_live_connected:
                    return
                print(f"[VIX] Attempting to connect to live stream (Attempt {attempt + 1}/{max_retries})...")
                client = db.Live(key=api_key)
                client.subscribe(
                    dataset="GLBX.MDP3",  # VIX futures are on CME
                    schema="ohlcv-1s",
                    stype_in="raw_symbol",
                    symbols=",".join(symbols_subscribed)
                )
                client.add_callback(ohlcv_callback)
                client.start()
                is_live_connected = True
                print("[VIX] Connected to live stream!")
                return
            except Exception as e:
                attempt += 1
                print(f"[VIX] Live connection attempt {attempt} failed: {e}")
                if attempt < max_retries:
                    print(f"[VIX] Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)

        print(f"[VIX] All retry attempts failed in cycle {retry_cycle}. Retrying indefinitely...")

def monitor_api_key():
    global api_key, is_live_connected
    last_key = api_key
    while True:
        try:
            config = configparser.ConfigParser()
            config.read(config_file)
            current_key = config['API']['key']
            if current_key != last_key:
                print("[VIX] API key updated. Restarting connection...")
                last_key = current_key
                api_key = current_key

                if is_live_connected:
                    client.stop()
                    is_live_connected = False

                threading.Thread(target=start_live_connection, daemon=True).start()
        except Exception as e:
            print(f"[VIX] Error monitoring API key: {e}")
        time.sleep(RETRY_DELAY)

def run():
    global db_manager
    
    print("Starting VIX Futures Data Stream...")
    db_manager = check_existing_instances()
    
    load_config()
    threading.Thread(target=monitor_api_key, daemon=True).start()
    start_live_connection()

    try:
        while True:
            time.sleep(RETRY_DELAY * 2)
    except KeyboardInterrupt:
        print("[VIX] Program interrupted manually. Exiting...")
        if client and is_live_connected:
            client.stop()

if __name__ == "__main__":
    run()