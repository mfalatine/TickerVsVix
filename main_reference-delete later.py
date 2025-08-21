import tkinter as tk
from tkinter import ttk, messagebox  # Ensure messagebox is imported
import queue
import threading
import os
import sys
from pathlib import Path

# Set UTF-8 encoding environment variables early to prevent Unicode errors
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.platform == "win32":
    # Force UTF-8 encoding on Windows
    try:
        import locale
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except:
        pass  # Ignore if locale setting fails

# Initialize Unicode support early to prevent encoding errors
from unicode_fix import initialize_unicode_support
unicode_status = initialize_unicode_support()

# === Refactored Modules ===
from futures_contracts import FuturesContract
from data_manager import DataManager
from trading_modes import TradingMode, TradingSimulator
from automated_trend_trader import AutomatedTrendTrader
from debug_utils import logger, cleanup_logs
from BayesianOptimization.EnhancedBayesianOptimization import TradingOptimizer
from optimizer_gui_integration import OptimizerGUIIntegration
from param_utils import build_param_dict
from config_manager import get_optimization_default, get_system_timeout

# === GUI / Handler Modules ===
from gui_styles import GUIStyles
from trading_frame import TradingFrame
from contract_frame import ContractFrame
from parameter_frame import ParameterFrame
from data_loading_frame import DataLoadingFrame
from analysis_handler import AnalysisHandler
from results_handler import ResultsHandler
from indicator_filter_gui import IndicatorFilterGUI
from gui_dialogs import get_top5_trials_from_db, update_gui_parameters_dialog


class FuturesTradingApp:
    """
    The main application class that initializes and manages all components.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Pivot Point Trading LLC")
        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Set window to take up left half of screen
        window_width = screen_width // 2
        window_height = screen_height - 100  # Leave some space for taskbar
        
        # Position window on left side of screen
        x_position = 0
        y_position = 0
        
        # Set window size and position
        self.root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

        # Data & Trading
        self.data_manager = DataManager()
        self.simulator = None
        self.optimizer = TradingOptimizer(self.data_manager)

        # Queues / State
        self.processing_queue = queue.Queue()
        self.progress_queue = queue.Queue()
        self.processing_active = False
        self._last_status_emit_ts = 0.0  # throttle noisy load-progress lines
        # Global cancel flag to stop long-running backtests triggered from WFO
        self._cancel_backtest = False

        # Contract tracking
        self.contract = None  # We'll set a default in a moment

        # Apply dark theme
        GUIStyles.configure_styles(root)

        # Build out the entire interface
        self.create_gui()

        # Immediately set a default contract so "Contract Information" is visible
        self.update_contract("MNQ")  # Or "MES," etc.

        # Start checking for progress messages
        self.check_progress_updates()
        
        # Perform initial log cleanup and schedule periodic cleanup
        self.schedule_log_cleanup()

        # Initialize calibration storage path
        self.cutter_profiles_path = Path(os.path.dirname(os.path.abspath(__file__))) / "Results" / "cutter_profiles.json"

    def create_gui(self):
        """Build every section: top row (3 columns), strategy params, indicator filters, right column."""
        main_container = ttk.Frame(self.root, padding="10", style='TFrame', width=1580, height=760)
        main_container.pack(fill=tk.BOTH, expand=True)
        main_container.grid_propagate(False)

        # 3 columns in the main container
        main_container.columnconfigure(0, weight=2, minsize=800)   # left - reduced weight and minsize
        main_container.columnconfigure(1, minsize=20)              # spacer
        main_container.columnconfigure(2, weight=1, minsize=450)   # right - increased minsize

        # Left column
        left_column = ttk.Frame(main_container, style='TFrame', height=740)
        left_column.grid(row=0, column=0, sticky='nsew')
        left_column.pack_propagate(False)
        left_column.grid_propagate(False)
        left_column.rowconfigure(2, weight=1)

        # Spacer
        ttk.Frame(main_container, width=20, style='TFrame').grid(row=0, column=1, sticky='ns')

        # Right column
        right_column = ttk.Frame(main_container, style='TFrame')
        right_column.grid(row=0, column=2, sticky='nsew', padx=(0, 10), pady=0)
        right_column.grid_rowconfigure(0, weight=1)
        right_column.grid_columnconfigure(0, weight=1)


        # ------------------ RIGHT COLUMN: Trading + Analysis ------------------
        self.trading_frame = TradingFrame(right_column, app_instance=self)
        self.trading_frame.pack(fill=tk.BOTH, expand=True)
        # Ensure desired button order after construction
        try:
            self.trading_frame.reposition_backtest_buttons()
        except Exception:
            pass
        
        # Add one-click optimizer integration
        self.optimizer_integration = OptimizerGUIIntegration(
            trading_frame=self.trading_frame,
            app_instance=self
        )
        
        # Add 4PAO (Four Parameter Adaptive Optimizer) integration
        from SingleParameterCrawler.FourParamAdaptiveIntegration import FourParamAdaptiveIntegration
        self.fourpao_integration = FourParamAdaptiveIntegration(app_instance=self)
        # Add 4PAO button to the trading frame control buttons
        if hasattr(self.trading_frame, 'control_frame'):
            self.fourpao_integration.add_4pao_button(self.trading_frame.control_frame)


        # ------------------ TOP ROW ------------------
        top_row = ttk.Frame(left_column, style='TFrame', width=800, height=170)
        top_row.pack(fill=tk.X, pady=(2,1))
        top_row.pack_propagate(False)
        top_row.grid_propagate(False)

        # The top row has 3 columns
        top_row.columnconfigure(0, weight=1, minsize=200)  # Contract selection
        # Shrink Contract Info by ~20% and give that width to Data Loading
        top_row.columnconfigure(1, weight=1, minsize=210)  # Contract info (reduced from 250)
        top_row.columnconfigure(2, weight=1, minsize=400)  # Data loading (increased from 350)

        #
        # [A] Contract Selection Container (but we'll create ContractFrame later)
        #
        contract_sel_container = ttk.Frame(top_row, width=250, height=170)
        contract_sel_container.grid(row=0, column=0, sticky='nsew', padx=(0,1))
        contract_sel_container.pack_propagate(False)
        contract_sel_container.grid_propagate(False)
        
        #
        # [B] Middle: Contract Information
        #
        # Reduce width ~20% (300 -> 240)
        contract_info_container = ttk.Frame(top_row, width=240, height=170)
        contract_info_container.grid(row=0, column=1, sticky='nsew', padx=1)
        contract_info_container.pack_propagate(False)
        contract_info_container.grid_propagate(False)

        self.contract_info_frame = ttk.LabelFrame(
            contract_info_container,
            text="Contract Information",
            style='LightBlue.TLabelframe',
            padding="10"
        )
        self.contract_info_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # We'll fill this label with the contract specs in update_contract()
        self.contract_info_label = ttk.Label(
            self.contract_info_frame,
            text="",  # updated when we set the contract
            style='YellowInner.TLabel'
        )
        self.contract_info_label.pack(anchor='nw', fill=tk.X, expand=True)

        #
        # [C] Data Loading
        #
        # Increase width by the same amount (450 -> 510)
        data_container = ttk.Frame(top_row, width=510, height=170)
        data_container.grid(row=0, column=2, sticky='nsew', padx=(1,0))
        data_container.pack_propagate(False)
        data_container.grid_propagate(False)

        # Create data loading frame after contract frame is created
        self.data_loading_frame = None  # Initialize to None, will be set after contract frame

        # ------------------ STRATEGY PARAMETERS ------------------
        strategy_row = ttk.Frame(left_column, style='TFrame', width=800, height=341)
        strategy_row.pack(fill=tk.X, pady=(1,1))
        strategy_row.pack_propagate(False)
        strategy_row.grid_propagate(False)
        
        # 1) Create ParameterFrame first
        self.parameter_frame = ParameterFrame(strategy_row)

        # 2) Now create the ContractFrame, passing in param_frame
        self.contract_frame = ContractFrame(
            contract_sel_container,
            app_instance=self,
            param_frame=self.parameter_frame  # new param
        )

        # Now create DataLoadingFrame with contract_frame
        # In create_gui method, update this initialization
        self.data_loading_frame = DataLoadingFrame(
            parent=data_container,
            data_manager=self.data_manager,
            analyze_button=self.trading_frame.analyze_button,
            contract_frame=self.contract_frame,
            optimize_button=None  # Optimize button now handled by OptimizerGUIIntegration
        )

        # Wire references for other components
        self.trading_frame.data_loading_frame = self.data_loading_frame
        self.data_loading_frame.app_instance = self

        # ------------------ INDICATOR FILTERS + TRADE HISTORY ------------------
        filters_row = ttk.Frame(left_column, style='TFrame')
        filters_row.pack(fill=tk.BOTH, expand=True, pady=(1,2))

        filters_row.columnconfigure(0, weight=1)
        filters_row.columnconfigure(1, weight=0)  # Shrink trade history (width reduced in indicator_filter_gui)
        filters_row.columnconfigure(2, weight=1)  # Expand slow-bleeder section (gains the freed space)
        filters_row.rowconfigure(0, weight=1)

        self.indicator_filter_gui = IndicatorFilterGUI(filters_row)
        self.trading_frame.indicator_filter_gui = self.indicator_filter_gui  # Make it accessible
        self.indicator_filter_gui.frame.grid(row=0, column=0, padx=(0,5), sticky='nsew')
        self.indicator_filter_gui.trade_frame.grid(row=0, column=1, padx=(5,0), sticky='nsew')
        
        # Create slow-bleeder monitoring section
        self.create_slow_bleeder_section(filters_row)

    # ------------------ Calibration ------------------
    def run_calibration(self):
        """Calibrate No-Progress Timeout with background worker and re-entrancy guard."""
        if getattr(self, "_calibrating", False):
            try:
                self.trading_frame.update_progress("Calibration already running…")
            except Exception:
                pass
            return
        self._calibrating = True

        # Disable calibrate button immediately
        try:
            if hasattr(self.trading_frame, 'calibrate_button'):
                self.trading_frame.calibrate_button.configure(state=tk.DISABLED)
        except Exception:
            pass

        # Validate inputs
        import pandas as pd
        from config_manager import config
        symbol = self.contract_frame.contract_var.get()
        if not symbol:
            messagebox.showerror("Calibration", "Select a contract first.")
            self._calibrating = False
            try:
                if hasattr(self.trading_frame, 'calibrate_button'):
                    self.trading_frame.calibrate_button.configure(state=tk.NORMAL)
            except Exception:
                pass
            return
        if not hasattr(self.data_loading_frame, 'selected_start_date') or self.data_loading_frame.selected_start_date is None:
            messagebox.showerror("Calibration", "Select a date range first.")
            self._calibrating = False
            try:
                if hasattr(self.trading_frame, 'calibrate_button'):
                    self.trading_frame.calibrate_button.configure(state=tk.NORMAL)
            except Exception:
                pass
            return

        end_dt = pd.to_datetime(self.data_loading_frame.selected_start_date).floor('T')
        try:
            cal_weeks = int(config.get_default('calibration_window_weeks', 4, 'TRADING_CONSTANTS'))
            if cal_weeks < 1:
                cal_weeks = 1
        except Exception:
            cal_weeks = 4
        start_dt = end_dt - pd.Timedelta(weeks=cal_weeks)
        warmup_start = (start_dt - pd.tseries.offsets.BDay(20)).strftime('%Y-%m-%d %H:%M')
        try:
            hdr = "=" * 63
            # Render top/bottom banners in yellow
            self.trading_frame.append_banner_line("")
            self.trading_frame.append_banner_line(hdr)
            self.trading_frame.append_results(
                f"Calibrating recent {cal_weeks} weeks for best-fit No Progress\n"
                f"End Date: {end_dt.strftime('%m/%d/%y')}\n"
                f"Start Date ({cal_weeks} weeks prior): {start_dt.strftime('%m/%d/%y')}\n"
            )
            self.trading_frame.append_banner_line(hdr)
            self.trading_frame.update_progress("Starting Calibration…")
        except Exception:
            pass

        # Worker that performs all heavy lifting
        def _worker():
            try:
                # Prepare data for calibration window
                self.root.after(0, lambda: self.trading_frame.append_results("[Load] Starting data preparation…\n"))
                self.data_manager.execute_backtest_stored_procedure(
                    start_date=start_dt.strftime('%Y-%m-%d %H:%M'),
                    end_date=end_dt.strftime('%Y-%m-%d %H:%M'),
                    symbol_base=symbol,
                    low_timeframe=int(self.data_loading_frame.low_timeframe.get() or 8),
                    high_timeframe=int(self.data_loading_frame.high_timeframe.get() or 120),
                    status_callback=self._make_throttled_status_cb(prefix='[Load] ')
                )
                self.data_manager.load_data_after_backtest(
                    symbol=symbol,
                    low_timeframe=int(self.data_loading_frame.low_timeframe.get() or 8),
                    high_timeframe=int(self.data_loading_frame.high_timeframe.get() or 120),
                    start_date=warmup_start,
                    end_date=end_dt.strftime('%Y-%m-%d %H:%M')
                )

                # Quick backtest on calibration window (cutters OFF)
                gui_params = self.parameter_frame.get_parameters()
                # Calibration profile: low risk, quick-exit OFF, NP isolated later
                trader = AutomatedTrendTrader(
                    trades_output=None,
                    initial_balance=int(gui_params.get('account_size')),
                    risk_per_trade=0.005,  # 0.5% during calibration baseline
                    max_trailing_dd=int(gui_params.get('max_dd')),
                    max_contracts=int(gui_params.get('max_contracts')),
                    contract_type=self.contract_frame.contract_var.get(),
                    atr_period=int(gui_params.get('atr_period')),
                    atr_stop_multiple=float(gui_params.get('atr_stop')),
                    atr_target_multiple=float(gui_params.get('atr_target')),
                    stop_after_2_losses=False,  # disable to avoid interaction with NP timing
                    target_daily_points=int(gui_params.get('target_points')),
                    Setting_avoid_lunch_hour=bool(gui_params.get('Setting_avoid_lunch_hour')),
                    follow_market=bool(gui_params.get('follow_market')),
                    zone_high_mult=float(gui_params.get('zone_high_mult')),
                    zone_low_mult=float(gui_params.get('zone_low_mult')),
                    Setting_require_candle_confirm=bool(gui_params.get('Setting_require_candle_confirm')),
                    zone_entry_mode=gui_params.get('zone_entry_mode'),
                    allow_dd_recovery=bool(gui_params.get('allow_dd_recovery')),
                    Setting_Quick_Exit=False,  # OFF for calibration
                    minute_loss_ticks=int(gui_params.get('minute_loss_ticks')),
                    trailing_stop_enabled=bool(gui_params.get('trailing_stop_enabled')),
                    atr_target_mode=bool(gui_params.get('atr_target_mode'))
                )

                # Yellow banners for calibration start
                self.root.after(0, lambda: self.trading_frame.append_banner_line("===================="))
                self.root.after(0, lambda: self.trading_frame.append_results("Starting Calibration\n"))
                self.root.after(0, lambda: self.trading_frame.append_banner_line("===================="))
                self.root.after(0, lambda: self.trading_frame.append_results("[Cal] Baseline: cutters OFF\n"))
                # Post calibration profile to terminal
                self.root.after(0, lambda: self.trading_frame.append_results(
                    "During NP calibration runs (4 week)\n"
                    "Risk% set to 0.5%\n"
                    "Quick Exit OFF\n"
                    "Stop After 2 Daily Losses OFF\n"
                    "NP only ON; all other cutters OFF (Cut Slow Bleeders, Loss Persistence, Time-in-Red OFF)\n"
                    "Before risk sweep (full-range)\n"
                    "Restores toggles to:\n"
                    "Quick Exit ON\n"
                    "Stop After 2 Daily Losses ON\n"
                    "No Progress ON\n"
                    "Cut Slow Bleeders ON\n"
                    "Loss Persistence OFF\n"
                    "Time-in-Red + MAE OFF\n\n"
                ))
                config.set_override('TRADING_PARAMETERS', 'cut_slow_bleeders_enabled', False)
                config.set_override('TRADING_PARAMETERS', 'no_progress_timeout_enabled', False)

                # Heartbeat while baseline backtest runs
                import time
                baseline_started = time.time()
                baseline_done_flag = {"done": False}
                def _baseline_heartbeat():
                    if baseline_done_flag["done"]:
                        # Keep last elapsed for a moment; final completion text will be set at the very end
                        return
                    elapsed = int(time.time() - baseline_started)
                    try:
                        self.trading_frame.update_progress(f"Calibrating… baseline backtest running ({elapsed}s)")
                    except Exception:
                        pass
                    # Re-arm heartbeat
                    try:
                        self.root.after(1000, _baseline_heartbeat)
                    except Exception:
                        pass
                self.root.after(0, _baseline_heartbeat)
                trades_df = trader.run_backtest(
                    self.data_manager.high_tf_data,
                    self.data_manager.low_tf_data,
                    minute_df=self.data_manager.final_minute_data,
                    trading_start_date=start_dt,
                    trading_end_date=end_dt
                )
                baseline_done_flag["done"] = True

                # Compute tempo
                if trades_df is None or trades_df.empty:
                    self.root.after(0, lambda: messagebox.showwarning("Calibration", "No trades found in calibration window."))
                    return
                import numpy as np
                target_points = float(gui_params.get('target_points'))
                times_to_25 = []
                for _, row in trades_df.iterrows():
                    entry = pd.to_datetime(row['Entry_Time'])
                    exit_t = pd.to_datetime(row['Exit_Time'])
                    if abs(row['PriceDiff']) >= 0.25 * target_points:
                        times_to_25.append((exit_t - entry).total_seconds() / 60.0)
                if not times_to_25:
                    self.root.after(0, lambda: messagebox.showwarning("Calibration", "No trades reached 25% target in calibration window."))
                    return
                med_minutes = np.median(times_to_25)
                def clamp(v, lo, hi):
                    return max(lo, min(hi, int(round(v))))
                timeout_candidates = sorted({clamp(0.7*med_minutes,18,30), clamp(0.8*med_minutes,18,30), clamp(0.9*med_minutes,18,30), 18, 22})
                mae_candidates = [0.15, 0.20, 0.25]
                mfe_candidates = [0.20, 0.25, 0.30]

                # Put cutters back to ON for scoring
                config.set_override('TRADING_PARAMETERS', 'cut_slow_bleeders_enabled', True)
                config.set_override('TRADING_PARAMETERS', 'no_progress_timeout_enabled', True)
                config.set_override('TRADING_PARAMETERS', 'loss_persistence_enabled', False)
                config.set_override('TRADING_PARAMETERS', 'time_in_red_mae_enabled', False)

                def score_metrics(df):
                    if df is None or df.empty:
                        return -1e12, 0.0, 0.0
                    net_pnl = float(df['Trade_PnL'].sum())
                    max_dd = float(df['Max_Drawdown'].max()) if 'Max_Drawdown' in df.columns else 0.0
                    gross_profits = df[df['Trade_PnL'] > 0]['Trade_PnL'].sum()
                    gross_losses = -df[df['Trade_PnL'] < 0]['Trade_PnL'].sum()
                    pf = (gross_profits / gross_losses) if gross_losses > 0 else float('inf')
                    try:
                        daily = df.groupby(pd.to_datetime(df['Exit_Time']).dt.date)['Trade_PnL'].sum()
                        worst_day = float(daily.min()) if not daily.empty else 0.0
                    except Exception:
                        worst_day = 0.0
                    score = net_pnl - 0.5 * max_dd
                    return score, pf, worst_day

                def is_better(best, cand, np_cuts):
                    if best is None:
                        return True
                    bs, bpf, bwd, bt, bmae, bmfe = best
                    cs, cpf, cwd, ct, cmae, cmfe = cand
                    if cs != bs:
                        return cs > bs
                    if cpf != bpf:
                        return cpf > bpf
                    if cwd != bwd:
                        return cwd > bwd
                    if np_cuts > 0 and bt == ct and bmae == cmae and bmfe == cmfe:
                        return True
                    return ct < bt

                best_local = None
                total_tests = len(timeout_candidates) * len(mae_candidates) * len(mfe_candidates)
                test_index = 0
                for t in timeout_candidates:
                    config.set_override('TRADING_CONSTANTS', 'no_progress_timeout_min', t)
                    config.set_override('TRADING_CONSTANTS', 'no_progress_target_frac', 0.25)
                    for mae in mae_candidates:
                        config.set_override('TRADING_CONSTANTS', 'no_progress_mae_atr_frac', mae)
                        for mfe in mfe_candidates:
                            config.set_override('TRADING_CONSTANTS', 'no_progress_mfe_target_frac', mfe)
                            test_index += 1
                            # Ensure NP-only for this test
                            try:
                                config.set_override('TRADING_PARAMETERS', 'no_progress_timeout_enabled', True)
                                config.set_override('TRADING_PARAMETERS', 'cut_slow_bleeders_enabled', False)
                                config.set_override('TRADING_PARAMETERS', 'loss_persistence_enabled', False)
                                config.set_override('TRADING_PARAMETERS', 'time_in_red_mae_enabled', False)
                            except Exception:
                                pass

                            # Rebuild trader for each NP test so overrides take effect
                            trader_np = AutomatedTrendTrader(
                                trades_output=None,
                                initial_balance=int(gui_params.get('account_size')),
                                risk_per_trade=0.005,  # 0.5% during NP calibration
                                max_trailing_dd=int(gui_params.get('max_dd')),
                                max_contracts=int(gui_params.get('max_contracts')),
                                contract_type=self.contract_frame.contract_var.get(),
                                atr_period=int(gui_params.get('atr_period')),
                                atr_stop_multiple=float(gui_params.get('atr_stop')),
                                atr_target_multiple=float(gui_params.get('atr_target')),
                                stop_after_2_losses=False,
                                target_daily_points=int(gui_params.get('target_points')),
                                Setting_avoid_lunch_hour=bool(gui_params.get('Setting_avoid_lunch_hour')),
                                follow_market=bool(gui_params.get('follow_market')),
                                zone_high_mult=float(gui_params.get('zone_high_mult')),
                                zone_low_mult=float(gui_params.get('zone_low_mult')),
                                Setting_require_candle_confirm=bool(gui_params.get('Setting_require_candle_confirm')),
                                zone_entry_mode=gui_params.get('zone_entry_mode'),
                                allow_dd_recovery=bool(gui_params.get('allow_dd_recovery')),
                                Setting_Quick_Exit=False,
                                minute_loss_ticks=int(gui_params.get('minute_loss_ticks')),
                                trailing_stop_enabled=bool(gui_params.get('trailing_stop_enabled')),
                                atr_target_mode=bool(gui_params.get('atr_target_mode'))
                            )

                            test_df = trader_np.run_backtest(
                                self.data_manager.high_tf_data,
                                self.data_manager.low_tf_data,
                                minute_df=self.data_manager.final_minute_data,
                                trading_start_date=start_dt,
                                trading_end_date=end_dt
                            )
                            s, pf, wd = score_metrics(test_df)
                            try:
                                np_cuts = int((test_df.get('Exit_Reason', pd.Series([], dtype=str)) == 'Slow_Bleeder_NoProgress').sum())
                            except Exception:
                                np_cuts = 0
                            # Build a colored line with PnL/DD/W%/RR/PF like the risk crawl
                            try:
                                total_trades2 = int(len(test_df)) if test_df is not None else 0
                                wins2 = int((test_df['Trade_PnL'] > 0).sum()) if total_trades2 > 0 else 0
                                losses2 = total_trades2 - wins2
                                pnl2 = float(test_df['Trade_PnL'].sum()) if total_trades2 > 0 else 0.0
                                dd2 = float(test_df['Max_Drawdown'].max()) if total_trades2 > 0 and 'Max_Drawdown' in test_df.columns else 0.0
                                win_rate2 = (wins2 / total_trades2 * 100.0) if total_trades2 > 0 else 0.0
                                gp2 = float(test_df[test_df['Trade_PnL'] > 0]['Trade_PnL'].sum()) if total_trades2 > 0 else 0.0
                                gl2 = -float(test_df[test_df['Trade_PnL'] < 0]['Trade_PnL'].sum()) if total_trades2 > 0 else 0.0
                                pf2 = (gp2 / gl2) if gl2 > 0 else float('inf')
                                aw2 = float(test_df[test_df['Trade_PnL'] > 0]['Trade_PnL'].mean()) if wins2 > 0 else 0.0
                                al2 = -float(test_df[test_df['Trade_PnL'] < 0]['Trade_PnL'].mean()) if losses2 > 0 else 0.0
                                rr2 = (aw2 / al2) if al2 > 0 else float('inf')
                            except Exception:
                                pnl2 = 0.0; dd2 = 0.0; win_rate2 = 0.0; total_trades2 = 0; wins2 = 0; losses2 = 0; pf2 = 0.0; rr2 = 0.0
                            self.root.after(0, lambda _label=f"[Cal] t={t}, mae={mae:.2f}, mfe={mfe:.2f} |":
                                            self.trading_frame.append_risk_crawl_line(_label, pnl2, dd2, win_rate2, total_trades2, wins2, losses2, rr2, pf2))
                            if is_better(best_local, (s, pf, wd, t, mae, mfe), np_cuts):
                                best_local = (s, pf, wd, t, mae, mfe)

                if best_local is None:
                    self.root.after(0, lambda: messagebox.showwarning("Calibration", "Calibration produced no candidates."))
                    return
                _, _, _, timeout_sel, best_mae_sel, best_mfe_sel = best_local
                config.set_override('TRADING_CONSTANTS', 'no_progress_timeout_min', timeout_sel)
                config.set_override('TRADING_CONSTANTS', 'no_progress_target_frac', 0.25)
                config.set_override('TRADING_CONSTANTS', 'no_progress_mae_atr_frac', best_mae_sel)
                config.set_override('TRADING_CONSTANTS', 'no_progress_mfe_target_frac', best_mfe_sel)
                self.root.after(0, lambda _t=timeout_sel, _mae=best_mae_sel, _mfe=best_mfe_sel:
                                self.trading_frame.append_results(
                                    f"\nCalibration complete: Timeout={_t} min, Required=0.25, MAE={_mae:.2f}×ATR, MFE={_mfe:.2f} of target\n"
                                ))

                # Removed calibration-window risk sweep; only final full-window sweep remains.

                # Reload full backtest data for user's selected range
                try:
                    full_start = self.data_loading_frame.selected_start_date
                    full_end = self.data_loading_frame.selected_end_date
                    if full_start is not None and full_end is not None:
                        self.data_loading_frame.actual_start_date = pd.to_datetime(full_start)
                        self.data_loading_frame.actual_end_date = pd.to_datetime(full_end)
                        warmup2 = (pd.to_datetime(full_start) - pd.tseries.offsets.BDay(20)).strftime('%Y-%m-%d %H:%M')
                        self.root.after(0, lambda: self.trading_frame.append_results("[Load] Preparing full backtest data after calibration…"))
                        self.data_manager.execute_backtest_stored_procedure(
                            start_date=warmup2,
                            end_date=pd.to_datetime(full_end).strftime('%Y-%m-%d %H:%M'),
                            symbol_base=symbol,
                            low_timeframe=int(self.data_loading_frame.low_timeframe.get() or 8),
                            high_timeframe=int(self.data_loading_frame.high_timeframe.get() or 120),
                            status_callback=self._make_throttled_status_cb(prefix='[Load] ')
                        )
                        self.data_manager.load_data_after_backtest(
                            symbol=symbol,
                            low_timeframe=int(self.data_loading_frame.low_timeframe.get() or 8),
                            high_timeframe=int(self.data_loading_frame.high_timeframe.get() or 120),
                            start_date=warmup2,
                            end_date=pd.to_datetime(full_end).strftime('%Y-%m-%d %H:%M')
                        )
                        # UI updates after full data load
                        self.root.after(0, lambda: (
                            self.trading_frame.append_results("\n[Load] Full backtest data ready\n"),
                            hasattr(self.data_loading_frame, 'data_status_label') and self.data_loading_frame.data_status_label.config(text="Data Loaded for Backtesting"),
                            self.trading_frame.enable_analyze_button(),
                            hasattr(self.trading_frame, 'calibrate_button') and self.trading_frame.calibrate_button.configure(state=tk.NORMAL)
                        ))

                        # Final Risk sweep over the user's full backtest window to align with Analyze
                        try:
                            max_dd_allowed_final = 0.9 * float(gui_params.get('max_dd', 0))
                            self.root.after(0, lambda: (
                                self.trading_frame.append_banner_line("========================"),
                                self.trading_frame.append_results(f"Final Risk Crawl (full window) ...  (DD limit {max_dd_allowed_final:.0f})\n"),
                                self.trading_frame.append_banner_line("========================")
                            ))
                            last_ok_risk_final = None
                            # Restore default toggles before risk sweep per user request
                            try:
                                # Quick Exit ON; Stop After 2 Losses ON; NP ON; others OFF
                                config.set_override('TRADING_PARAMETERS', 'Setting_Quick_Exit', True)
                                config.set_override('TRADING_PARAMETERS', 'stop_after_2_losses', True)
                                config.set_override('TRADING_PARAMETERS', 'no_progress_timeout_enabled', True)
                                config.set_override('TRADING_PARAMETERS', 'cut_slow_bleeders_enabled', True)
                                config.set_override('TRADING_PARAMETERS', 'loss_persistence_enabled', False)
                                config.set_override('TRADING_PARAMETERS', 'time_in_red_mae_enabled', False)
                            except Exception:
                                pass
                            for j in range(3, 21):
                                risk_pct_cand = j / 10.0
                                trader_final = AutomatedTrendTrader(
                                    trades_output=None,
                                    initial_balance=int(gui_params.get('account_size')),
                                    risk_per_trade=risk_pct_cand / 100.0,
                                    max_trailing_dd=int(gui_params.get('max_dd')),
                                    max_contracts=int(gui_params.get('max_contracts')),
                                    contract_type=self.contract_frame.contract_var.get(),
                                    atr_period=int(gui_params.get('atr_period')),
                                    atr_stop_multiple=float(gui_params.get('atr_stop')),
                                    atr_target_multiple=float(gui_params.get('atr_target')),
                                    stop_after_2_losses=bool(gui_params.get('stop_after_2_losses')),
                                    target_daily_points=int(gui_params.get('target_points')),
                                    Setting_avoid_lunch_hour=bool(gui_params.get('Setting_avoid_lunch_hour')),
                                    follow_market=bool(gui_params.get('follow_market')),
                                    zone_high_mult=float(gui_params.get('zone_high_mult')),
                                    zone_low_mult=float(gui_params.get('zone_low_mult')),
                                    Setting_require_candle_confirm=bool(gui_params.get('Setting_require_candle_confirm')),
                                    zone_entry_mode=gui_params.get('zone_entry_mode'),
                                    allow_dd_recovery=bool(gui_params.get('allow_dd_recovery')),
                                    Setting_Quick_Exit=bool(gui_params.get('Setting_Quick_Exit')),
                                    minute_loss_ticks=int(gui_params.get('minute_loss_ticks')),
                                    trailing_stop_enabled=bool(gui_params.get('trailing_stop_enabled')),
                                    atr_target_mode=bool(gui_params.get('atr_target_mode'))
                                )
                                df_final_risk = trader_final.run_backtest(
                                    self.data_manager.high_tf_data,
                                    self.data_manager.low_tf_data,
                                    minute_df=self.data_manager.final_minute_data,
                                    trading_start_date=pd.to_datetime(full_start),
                                    trading_end_date=pd.to_datetime(full_end)
                                )
                                # Metrics
                                try:
                                    total_tr = int(len(df_final_risk)) if df_final_risk is not None else 0
                                    wins_tr = int((df_final_risk['Trade_PnL'] > 0).sum()) if total_tr > 0 else 0
                                    losses_tr = total_tr - wins_tr
                                    pnl_tr = float(df_final_risk['Trade_PnL'].sum()) if total_tr > 0 else 0.0
                                    dd_tr = float(df_final_risk['Max_Drawdown'].max()) if total_tr > 0 and 'Max_Drawdown' in df_final_risk.columns else 0.0
                                    wr_tr = (wins_tr / total_tr * 100.0) if total_tr > 0 else 0.0
                                    gp_tr = float(df_final_risk[df_final_risk['Trade_PnL'] > 0]['Trade_PnL'].sum()) if total_tr > 0 else 0.0
                                    gl_tr = -float(df_final_risk[df_final_risk['Trade_PnL'] < 0]['Trade_PnL'].sum()) if total_tr > 0 else 0.0
                                    pf_tr = (gp_tr / gl_tr) if gl_tr > 0 else float('inf')
                                    aw_tr = float(df_final_risk[df_final_risk['Trade_PnL'] > 0]['Trade_PnL'].mean()) if wins_tr > 0 else 0.0
                                    al_tr = -float(df_final_risk[df_final_risk['Trade_PnL'] < 0]['Trade_PnL'].mean()) if losses_tr > 0 else 0.0
                                    rr_tr = (aw_tr / al_tr) if al_tr > 0 else float('inf')
                                except Exception:
                                    pnl_tr = 0.0; dd_tr = 0.0; wr_tr = 0.0; total_tr = 0; wins_tr = 0; losses_tr = 0; pf_tr = 0.0; rr_tr = 0.0
                                self.root.after(0, lambda _idx=j-2, _p=pnl_tr, _dd=dd_tr, _w=wr_tr, _t=total_tr, _wn=wins_tr, _ls=losses_tr, _rr=rr_tr, _pf=pf_tr, _r=risk_pct_cand:
                                                self.trading_frame.append_risk_crawl_line(_idx, _p, _dd, _w, _t, _wn, _ls, _rr, _pf, _r))
                                if dd_tr <= max_dd_allowed_final:
                                    last_ok_risk_final = risk_pct_cand
                                else:
                                    break
                            if last_ok_risk_final is not None:
                                self.root.after(0, lambda r=last_ok_risk_final: self.parameter_frame.set_entry_value('risk', r))
                                self.root.after(0, lambda r=last_ok_risk_final: self.trading_frame.append_results(
                                    f"[Cal] Final risk sweep selected Risk % = {r:.2f} (<= {max_dd_allowed_final:.0f} DD)\n"))
                        except Exception as _:
                            pass
                except Exception as e2:
                    logger.error(f"Error reloading full range after calibration: {e2}")

                # Now it is truly complete
                try:
                    self.root.after(0, lambda: self.trading_frame.update_progress("Calibration Complete"))
                except Exception:
                    pass

            except Exception as e:
                logger.error(f"Calibration error: {e}")
                self.root.after(0, lambda: messagebox.showerror("Calibration", f"Calibration failed: {e}"))
            finally:
                # Update bottom progress bar to completion
                try:
                    self.root.after(0, lambda: self.trading_frame.update_progress("Calibration Complete"))
                except Exception:
                    pass
                # Always re-enable controls and clear guard (even on error)
                try:
                    self.root.after(0, self.trading_frame.enable_analyze_button)
                except Exception:
                    pass
                try:
                    self.root.after(0, lambda: (
                        hasattr(self.trading_frame, 'calibrate_button') and self.trading_frame.calibrate_button.configure(state=tk.NORMAL)
                    ))
                except Exception:
                    pass
                # Auto-select Cut Slow Bleeders and No Progress Timeout after calibration
                try:
                    self.cut_slow_bleeders_enabled_var.set(True)
                    self.no_progress_timeout_enabled_var.set(True)
                except Exception:
                    pass
                self._calibrating = False

        # Start worker
        self.trading_frame.disable_analyze_button()
        threading.Thread(target=_worker, daemon=True).start()
        

    # ------------------ CONTRACT / SIMULATOR ------------------
    def update_contract(self, symbol):
        """Called by contract_frame (dropdown) or here to set the contract & show info."""
        # Create a new contract object
        from futures_contracts import FuturesContract
        new_contract = FuturesContract(symbol.upper())
        self.contract = new_contract

        # Create a simulator in BACKTEST mode by default
        self.simulator = TradingSimulator(TradingMode.BACKTEST, self.data_manager)

        # Update contract type in parameter frame
        self.parameter_frame.contract_type.set(symbol.upper())

        # Show the contract details in the middle panel
        info_text = (
            f"Contract: {new_contract.name}\n"
            f"Point Value: ${new_contract.point_value:.2f}\n"
            f"Tick Size: {new_contract.tick_size}\n"
            f"Typical Margin: ${new_contract.typical_margin:,}"
        )
        self.contract_info_label.config(text=info_text)

    def handle_mode_change(self):
        """Called by the TradingFrame radio buttons (Backtest, Live Simulation, Live)."""
        mode = self.trading_frame.mode_var.get()
        if not self.simulator:
            self.simulator = TradingSimulator(TradingMode.BACKTEST, self.data_manager)

        if mode == "Backtest":
            self.simulator.mode = TradingMode.BACKTEST
            self.trading_frame.start_button.configure(state=tk.DISABLED)
            self.trading_frame.stop_button.configure(state=tk.DISABLED)
        elif mode == "Live Simulation":
            self.simulator.mode = TradingMode.SIMULATION
            self.trading_frame.start_button.configure(state=tk.NORMAL)
        else:
            self.simulator.mode = TradingMode.LIVE
            self.trading_frame.start_button.configure(state=tk.NORMAL)

    # ------------------ ANALYSIS & TRADING ------------------
    def run_analysis(self):
        """Invoked by 'Analyze Market' button for backtesting, etc."""
        # Default to regular backtest; do not show any WFO/WFO_NC popups here.
        wfo_opts = None

        if wfo_opts and getattr(wfo_opts, 'enabled', False):
            # Run Weekly Walk-Forward using isolated adapters; bypass normal single-range backtest
            try:
                # Validate required selections
                symbol = self.contract.symbol if self.contract else None
                if not symbol:
                    messagebox.showerror("Backtest", "Select a contract first.")
                    return
                if not hasattr(self.data_loading_frame, 'selected_start_date') or self.data_loading_frame.selected_start_date is None:
                    messagebox.showerror("Backtest", "Select a date range first.")
                    return
                if not hasattr(self.data_loading_frame, 'selected_end_date') or self.data_loading_frame.selected_end_date is None:
                    messagebox.showerror("Backtest", "Select a date range first.")
                    return

                # Gather context
                low_tf = int(self.data_loading_frame.low_timeframe.get() or 8)
                high_tf = int(self.data_loading_frame.high_timeframe.get() or 120)
                selected_start = self.data_loading_frame.selected_start_date
                selected_end = self.data_loading_frame.selected_end_date
                gui_params = self.parameter_frame.get_parameters()

                # UI feedback
                self.trading_frame.update_progress("Starting Weekly Walk-Forward backtest…")
                self.trading_frame.disable_analyze_button()
                try:
                    if hasattr(self.trading_frame, 'calibrate_button'):
                        self.trading_frame.calibrate_button.configure(state=tk.DISABLED)
                except Exception:
                    pass

                # Stop control for WFO
                wfo_cancel = {"stop": False}
                def _stop_wfo():
                    wfo_cancel["stop"] = True
                    # Also cancel any in-flight final backtest started beneath Core pick
                    self._cancel_backtest = True
                try:
                    # Reuse the dedicated stop button in TradingFrame and enable it
                    if hasattr(self.trading_frame, 'stop_wfo_button') and self.trading_frame.stop_wfo_button:
                        self.trading_frame.stop_wfo_button.configure(command=_stop_wfo, state=tk.NORMAL)
                        stop_btn = self.trading_frame.stop_wfo_button
                    else:
                        stop_btn = tk.Button(self.trading_frame.control_frame, text="Stop BckTst", command=_stop_wfo, bg="#E74032", fg="white")
                        stop_btn.pack(side=tk.LEFT, padx=5)
                except Exception:
                    stop_btn = None

                def _wfo_worker():
                    try:
                        # Clear cancel flag at start of WFO run
                        self._cancel_backtest = False
                        # One-time final backtest trigger flag for this WFO run
                        final_backtest_once = {"done": False}

                        def _run_final_backtest_with_params(pvals: dict):
                            try:
                                # Extract selected parameters
                                r_dec = float(pvals.get('r')) if 'r' in pvals else None
                                z_val = float(pvals.get('z')) if 'z' in pvals else None
                                a_val = float(pvals.get('a')) if 'a' in pvals else None
                                if r_dec is None or z_val is None or a_val is None:
                                    return
                                # Snapshot current GUI parameters
                                try:
                                    current_params = dict(self.parameter_frame.get_parameters())
                                except Exception:
                                    current_params = {}
                                # Snapshot cutter checkbox states
                                try:
                                    orig_cutters = {
                                        'cut_slow': bool(self.cut_slow_bleeders_enabled_var.get()),
                                        'loss_persist': bool(self.loss_persistence_enabled_var.get()),
                                        'no_progress': bool(self.no_progress_timeout_enabled_var.get()),
                                        'time_in_red': bool(self.time_in_red_mae_enabled_var.get()),
                                    }
                                except Exception:
                                    orig_cutters = {'cut_slow': True, 'loss_persist': False, 'no_progress': False, 'time_in_red': False}

                                # Apply desired params to GUI (Risk in percent for GUI)
                                try:
                                    self.parameter_frame.set_entry_value('risk', r_dec * 100.0)
                                    self.parameter_frame.set_entry_value('zone_low_mult', z_val)
                                    self.parameter_frame.set_entry_value('atr_stop', a_val)
                                except Exception:
                                    pass

                                # Disable cutters via GUI checkboxes so analysis uses them OFF
                                try:
                                    self.cut_slow_bleeders_enabled_var.set(False)
                                    self.loss_persistence_enabled_var.set(False)
                                    self.no_progress_timeout_enabled_var.set(False)
                                    self.time_in_red_mae_enabled_var.set(False)
                                except Exception:
                                    pass

                                # Ensure actual_* dates are present for AnalysisHandler
                                try:
                                    import pandas as pd
                                    if not hasattr(self.data_loading_frame, 'actual_start_date') or self.data_loading_frame.actual_start_date is None:
                                        if hasattr(self.data_loading_frame, 'selected_start_date') and self.data_loading_frame.selected_start_date is not None:
                                            self.data_loading_frame.actual_start_date = pd.to_datetime(self.data_loading_frame.selected_start_date)
                                    if not hasattr(self.data_loading_frame, 'actual_end_date') or self.data_loading_frame.actual_end_date is None:
                                        if hasattr(self.data_loading_frame, 'selected_end_date') and self.data_loading_frame.selected_end_date is not None:
                                            self.data_loading_frame.actual_end_date = pd.to_datetime(self.data_loading_frame.selected_end_date)
                                except Exception:
                                    pass

                                # Announce the final backtest start under the Core pick line
                                try:
                                    self.root.after(0, lambda _r=r_dec, _z=z_val, _a=a_val: self.trading_frame.append_results(
                                        f"[WFO] Starting regular backtest (cutters OFF) with r={_r*100:.2f}%, z={_z:.2f}, a={_a:.2f}\n"))
                                except Exception:
                                    pass

                                # Run a regular backtest (identical output path), and pause WFO until it completes
                                try:
                                    import threading as _th, time as _time
                                    done_evt = _th.Event()
                                    analysis_handler = AnalysisHandler(
                                        data_manager=self.data_manager,
                                        parameter_handler=self.parameter_frame,
                                        trading_frame=self.trading_frame,
                                        completion_event=done_evt,
                                    )
                                    # IMPORTANT: run_analysis touches Tk widgets before spawning its own thread.
                                    # Always invoke from the Tk main thread to avoid deadlocks/hangs.
                                    def _start_analysis_on_ui_thread():
                                        try:
                                            ok = analysis_handler.run_analysis()
                                            if not ok:
                                                try:
                                                    done_evt.set()
                                                    self.trading_frame.append_results("[WFO] Skipping regular backtest (pre-checks failed)\n")
                                                except Exception:
                                                    pass
                                        except Exception:
                                            try:
                                                done_evt.set()
                                                self.trading_frame.append_results("[WFO] Skipping regular backtest (error starting)\n")
                                            except Exception:
                                                pass
                                    self.root.after(0, _start_analysis_on_ui_thread)
                                    # Block WFO thread until regular backtest completes (or cancel/timeout)
                                    try:
                                        deadline = _time.time() + 600.0  # 10-minute safety timeout
                                        while not done_evt.wait(timeout=0.5):
                                            if self._cancel_backtest:
                                                break
                                            if _time.time() > deadline:
                                                try:
                                                    # Signal cancel to the running analysis so it stops in the background
                                                    self._cancel_backtest = True
                                                    self.root.after(0, lambda: self.trading_frame.append_results("[WFO] Regular backtest timeout; cancelling and resuming WFO\n"))
                                                except Exception:
                                                    pass
                                                break
                                    except Exception:
                                        pass
                                finally:
                                    # Restore GUI params and cutter states after the run is started
                                    try:
                                        if current_params:
                                            # Restore a minimal subset that we changed
                                            if 'risk_pct' in current_params:
                                                self.parameter_frame.set_entry_value('risk', float(current_params['risk_pct']) * 100.0)
                                            if 'zone_low_mult' in current_params:
                                                self.parameter_frame.set_entry_value('zone_low_mult', float(current_params['zone_low_mult']))
                                            if 'atr_stop' in current_params:
                                                self.parameter_frame.set_entry_value('atr_stop', float(current_params['atr_stop']))
                                    except Exception:
                                        pass
                                    try:
                                        self.cut_slow_bleeders_enabled_var.set(orig_cutters.get('cut_slow', True))
                                        self.loss_persistence_enabled_var.set(orig_cutters.get('loss_persist', False))
                                        self.no_progress_timeout_enabled_var.set(orig_cutters.get('no_progress', False))
                                        self.time_in_red_mae_enabled_var.set(orig_cutters.get('time_in_red', False))
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                        # Lazy import app adapters for WFO to avoid hard dependency
                        from WFO.app_adapters import AppDataProvider, AppStrategyRunner
                        provider = AppDataProvider(
                            data_manager=self.data_manager,
                            symbol=symbol,
                            low_timeframe_min=low_tf,
                            high_timeframe_min=high_tf,
                            selected_start=selected_start,
                            selected_end=selected_end,
                        )
                        runner = AppStrategyRunner(
                            data_manager=self.data_manager,
                            symbol=symbol,
                            low_timeframe_min=low_tf,
                            high_timeframe_min=high_tf,
                            base_params=gui_params,
                        )
                        # Respect cutter selection from popup if provided
                        try:
                            if hasattr(wfo_opts, 'evaluate_np'):
                                pass
                        except Exception:
                            pass
                        def _status_router(msg: str):
                            # Route structured core test lines to colored output in results_text
                            try:
                                # New: Render CSV-style core lines like
                                # [WFO] [Core coarse],r,0.60,z,1.10,a,0.30,|,PnL,-1518.31,DD,2533.96,W%,25.00,T,8,W,2,L,6,RR,1.52,PF,0.5
                                if isinstance(msg, str) and msg.startswith("[Core ") and "," in msg:
                                    t = self.trading_frame.results_text
                                    line = msg.strip()
                                    # Optional prefix like "[WFO] " is added elsewhere; just render the CSV content with colors
                                    try:
                                        parts = line.split(',')
                                        i = 0
                                        # Print leading prefix and bracketed title as a label
                                        if parts[0].startswith('[Core'):
                                            t.insert('end', '[WFO] ', ('label',))
                                            t.insert('end', parts[0], ('label',))
                                            i = 1
                                        # Iterate key,value pairs separated by commas; the '|' token is a separator
                                        while i < len(parts):
                                            token = parts[i].strip()
                                            if token == '|':
                                                t.insert('end', ' | ', ('label',))
                                                i += 1
                                                continue
                                            # Expect key
                                            key = token
                                            val = None
                                            if i + 1 < len(parts):
                                                val = parts[i + 1].strip()
                                            # Titles in yellow
                                            if key:
                                                # Add comma prefix if not first after title
                                                if not key.startswith('[Core') and t.index('end-1c') != '1.0':
                                                    t.insert('end', ',', ())
                                                t.insert('end', f"{key}", ('title',))
                                            # Values in light blue except special rules
                                            if val is not None:
                                                if key == 'PnL':
                                                    try:
                                                        v = float(val)
                                                    except Exception:
                                                        v = 0.0
                                                    # Preserve original formatting
                                                    t.insert('end', f",{val}", ('pnl_pos' if v >= 0 else 'pnl_neg',))
                                                elif key == 'DD':
                                                    try:
                                                        v = float(val)
                                                    except Exception:
                                                        v = 0.0
                                                    # Preserve original formatting
                                                    t.insert('end', f",{val}", ('dd_orange',))
                                                else:
                                                    # Preserve original text for values; light blue
                                                    t.insert('end', f",{val}", ('val',))
                                            i += 2
                                        t.insert('end', "\n")
                                        t.see('end')
                                        return
                                    except Exception:
                                        pass
                                if isinstance(msg, str) and msg.startswith("CORE_TEST|"):
                                    parts = {kv.split('=')[0]: kv.split('=')[1] for kv in msg.split('|')[1:] if '=' in kv}
                                    r = float(parts.get('r', 0)) * 100.0
                                    z = float(parts.get('z', 0))
                                    a = float(parts.get('a', 0))
                                    pnl = float(parts.get('pnl', 0))
                                    dd = float(parts.get('dd', 0))
                                    wr = float(parts.get('wr', 0))
                                    t = int(float(parts.get('t', 0)))
                                    pf = float(parts.get('pf', 0))
                                    label = f"[Core] r={r:.2f}% z={z:.2f} a={a:.2f}"
                                    self.trading_frame.append_risk_crawl_line(label, pnl, dd, wr, t, int(round(wr/100.0*t)) if t>0 else 0, (t - int(round(wr/100.0*t))) if t>0 else 0, 0.0, pf)
                                    return
                                if isinstance(msg, str) and msg.startswith("CORE_RANGE|"):
                                    # Format: CORE_RANGE|r_lo=..|r=..|r_hi=..|z_lo=..|z=..|z_hi=..|a_lo=..|a=..|a_hi=..
                                    p = {kv.split('=')[0]: kv.split('=')[1] for kv in msg.split('|')[1:] if '=' in kv}
                                    t = self.trading_frame.results_text
                                    try:
                                        t.insert('end', "Core pick: ", ('label',))
                                        # risk
                                        t.insert('end', f"risk {float(p['r_lo']):.3f} <= ", ('range',))
                                        t.insert('end', f"{float(p['r']):.3f}", ('val',))
                                        t.insert('end', f" <= {float(p['r_hi']):.3f}, ", ('range',))
                                        # zone_low
                                        t.insert('end', f"zone_low {float(p['z_lo']):.2f} <= ", ('range',))
                                        t.insert('end', f"{float(p['z']):.2f}", ('val',))
                                        t.insert('end', f" <= {float(p['z_hi']):.2f}, ", ('range',))
                                        # atr_stop
                                        t.insert('end', f"atr_stop {float(p['a_lo']):.2f} <= ", ('range',))
                                        t.insert('end', f"{float(p['a']):.2f}", ('val',))
                                        t.insert('end', f" <= {float(p['a_hi']):.2f}\n", ('range',))
                                        # One-line metrics summary: titles white(label) values light-blue(val)
                                        # Expect following keys exist in same payload for simplicity
                                        if all(k in p for k in ['pnl','dd','wr','t','pf']):
                                            t.insert('end', "PnL,", ('label',))
                                            t.insert('end', f"{float(p['pnl']):,.2f}", ('val',))
                                            t.insert('end', ", DD,", ('label',))
                                            t.insert('end', f"{float(p['dd']):,.2f}", ('val',))
                                            t.insert('end', ", W%", ('label',))
                                            t.insert('end', f",{float(p['wr']):.2f}", ('val',))
                                            t.insert('end', f", T,{int(float(p['t']))}", ('label',))
                                            t.insert('end', ", PF,", ('label',))
                                            t.insert('end', f"{float(p['pf']):.2f}\n", ('val',))
                                        # Emit a styled summary report (no Trading Days / durations / daily PnL)
                                        try:
                                            # Strategy Parameters (values derived from GUI and core pick)
                                            sym = symbol
                                            # Use calibration window dates if provided in payload, else fall back to selected
                                            cal_from = p.get('cal_from')
                                            cal_to = p.get('cal_to')
                                            if cal_from and cal_to:
                                                ds_str = str(cal_from)
                                                de_str = str(cal_to)
                                            else:
                                                ds = getattr(self.data_loading_frame, 'selected_start_date', None)
                                                de = getattr(self.data_loading_frame, 'selected_end_date', None)
                                                ds_str = ds.strftime('%Y-%m-%d') if ds is not None else 'n/a'
                                                de_str = de.strftime('%Y-%m-%d') if de is not None else 'n/a'
                                            low_tf_str = f"{low_tf}min"
                                            high_tf_str = f"{high_tf}min"
                                            acc = float(gui_params.get('account_size', 50000))
                                            risk_pct = float(p.get('r', gui_params.get('risk_pct', 0.01))) * 100.0
                                            tgt_pts = float(gui_params.get('target_points', 18))
                                            max_con = int(gui_params.get('max_contracts', 100))
                                            atr_stop = float(p.get('a', gui_params.get('atr_stop', 2.0)))
                                            atr_tgt = float(gui_params.get('atr_target', 1.0))
                                            atr_period = int(gui_params.get('atr_period', 14))
                                            max_dd = float(gui_params.get('max_dd', 2500))
                                            zh = float(gui_params.get('zone_high_mult', 1.0))
                                            zl = float(p.get('z', gui_params.get('zone_low_mult', 1.5)))
                                            qx_ticks = int(gui_params.get('minute_loss_ticks', 8))

                                            # Performance (from CORE_RANGE metrics for calibration window)
                                            pnl = float(p.get('pnl')) if p.get('pnl') is not None else None
                                            dd = float(p.get('dd')) if p.get('dd') is not None else None
                                            wr = float(p.get('wr')) if p.get('wr') is not None else None
                                            tot_trades = int(float(p.get('t'))) if p.get('t') is not None else None
                                            wins_val = p.get('wins')
                                            if wins_val is not None:
                                                wins_ct = int(float(wins_val))
                                            elif wr is not None and tot_trades is not None:
                                                wins_ct = int(round((wr/100.0) * tot_trades))
                                            else:
                                                wins_ct = None
                                            pf = float(p.get('pf')) if p.get('pf') is not None else None
                                            rr = float(p.get('rr')) if p.get('rr') is not None else None
                                            pnl_dd_ratio = (pnl / dd) if (pnl is not None and dd and dd > 0) else None

                                            # Start rendering with color tags
                                            t.insert('end', "======================================================\n", ('range',))
                                            t.insert('end', "Strategy Parameters:\n", ('label',))
                                            # -- Symbol
                                            t.insert('end', "-- Symbol: ", ('range',))
                                            t.insert('end', f"{sym}\n", ('val',))
                                            # -- Date Range
                                            t.insert('end', "-- Calibration Window: ", ('range',))
                                            t.insert('end', f"{ds_str} to {de_str}\n", ('val',))
                                            # -- Timeframes Used
                                            t.insert('end', "-- Timeframes Used - Low: ", ('range',))
                                            t.insert('end', f"{low_tf_str}", ('val',))
                                            t.insert('end', ", High: ", ('range',))
                                            t.insert('end', f"{high_tf_str}\n", ('val',))
                                            # separator
                                            t.insert('end', "--------------------------------------------------------\n", ('range',))
                                            # -- Initial Balance
                                            t.insert('end', "-- Initial Balance: ", ('range',))
                                            t.insert('end', f"{acc:,.2f}\n", ('val',))
                                            # -- Risk per Trade
                                            t.insert('end', "-- Risk per Trade: ", ('range',))
                                            t.insert('end', f"{risk_pct:.2f}%\n", ('val',))
                                            # -- Target Daily Points
                                            t.insert('end', "-- Target Daily Points: ", ('range',))
                                            t.insert('end', f"{tgt_pts:g}\n", ('val',))
                                            # -- Max Contracts
                                            t.insert('end', "-- Max Contracts: ", ('range',))
                                            t.insert('end', f"{max_con}\n", ('val',))
                                            # -- ATR Stop/Target
                                            t.insert('end', "-- ATR Stop Mult ", ('range',))
                                            t.insert('end', f"{atr_stop:.2f}", ('val',))
                                            t.insert('end', " - ATR Target Mult ", ('range',))
                                            t.insert('end', f"{atr_tgt:.2f}\n", ('val',))
                                            # -- ATR Period
                                            t.insert('end', "-- ATR Period: ", ('range',))
                                            t.insert('end', f"{atr_period}\n", ('val',))
                                            # -- Max Trailing DD
                                            t.insert('end', "-- Max Trailing DD: ", ('range',))
                                            t.insert('end', f"{max_dd:,.2f}\n", ('val',))
                                            # -- Zone High/Low Mult
                                            t.insert('end', "-- Zone High Mult: ", ('range',))
                                            t.insert('end', f"{zh:.2f}", ('val',))
                                            t.insert('end', " - Zone Low Mult: ", ('range',))
                                            t.insert('end', f"{zl:.2f}\n", ('val',))
                                            # -- Quick Exit Ticks
                                            t.insert('end', "-- Quick Exit Ticks: ", ('range',))
                                            t.insert('end', f"{qx_ticks}\n\n", ('val',))

                                            # Performance heading (white)
                                            t.insert('end', "Performance (calibration window):\n", ('label',))
                                            # -- Total Trades
                                            if tot_trades is not None:
                                                t.insert('end', "-- Total Trades: ", ('range',))
                                                t.insert('end', f"{tot_trades}\n", ('val',))
                                            # -- Wins / Losses
                                            if wins_ct is not None and tot_trades is not None:
                                                t.insert('end', "-- Wins: ", ('range',))
                                                t.insert('end', f"{wins_ct}", ('val',))
                                                t.insert('end', " / Losses: ", ('range',))
                                                t.insert('end', f"{max(0, tot_trades - wins_ct)}\n", ('val',))
                                            # -- Win Rate
                                            if wr is not None:
                                                t.insert('end', "-- Win Rate: ", ('range',))
                                                t.insert('end', f"{wr:.2f}%\n", ('val',))
                                            # -- Net PnL
                                            if pnl is not None:
                                                t.insert('end', "-- Net PnL: ", ('range',))
                                                t.insert('end', f"{pnl:,.2f}\n", ('val',))
                                            # -- Max Drawdown (no duration)
                                            if dd is not None:
                                                t.insert('end', "-- Max Drawdown: ", ('range',))
                                                t.insert('end', f"{dd:,.2f}\n", ('val',))
                                            # -- PnL/DD Ratio
                                            if pnl_dd_ratio is not None:
                                                t.insert('end', "-- PnL/Drawdown Ratio: ", ('range',))
                                                t.insert('end', f"{pnl_dd_ratio:.2f}\n", ('val',))
                                            # -- Profit Factor
                                            if pf is not None:
                                                t.insert('end', "-- Profit Factor: ", ('range',))
                                                t.insert('end', f"{pf:.2f}\n", ('val',))
                                            # -- Risk:Reward Ratio
                                            if rr is not None:
                                                t.insert('end', "-- Risk:Reward Ratio (Avg Win / Avg Loss): ", ('range',))
                                                t.insert('end', f"{rr:.2f} : 1\n", ('val',))
                                            t.insert('end', "\n")
                                            t.see('end')
                                        except Exception:
                                            pass
                                        # Do NOT auto-run a full regular backtest here; just hint the user
                                        try:
                                            if not final_backtest_once["done"]:
                                                final_backtest_once["done"] = True
                                                self.root.after(0, lambda: self.trading_frame.append_results(
                                                    "[WFO] Core pick selected. Use 'BckTest' to confirm on the full date range if desired.\n"
                                                ))
                                        except Exception:
                                            pass
                                        t.see('end')
                                    except Exception:
                                        pass
                                    return
                            except Exception:
                                pass
                            # Default status line output
                            self._make_throttled_status_cb(prefix='[WFO] ')(msg)

                        from WFO.walkforward_tuner import run_walkforward
                        results = run_walkforward(
                            wfo_opts,
                            provider,
                            runner,
                            status_callback=_status_router,
                            stop_requested=lambda: wfo_cancel["stop"]
                        )
                        # Summarize
                        total_weeks = len(results)
                        try:
                            sum_pnl = sum(float(getattr(w, 'oos_metrics', {}).get('pnl', 0.0)) for w in results)
                            max_dd = max(float(getattr(w, 'oos_metrics', {}).get('dd', 0.0)) for w in results) if results else 0.0
                            total_tr = sum(int(getattr(w, 'oos_metrics', {}).get('trades', 0)) for w in results)
                        except Exception:
                            sum_pnl, max_dd, total_tr = 0.0, 0.0, 0
                        self.root.after(0, lambda: self.trading_frame.append_results(
                            f"\nWFO complete. Weeks processed: {total_weeks}. Total PnL={sum_pnl:,.2f}, Max DD={max_dd:,.2f}, Trades={total_tr}.\nSee Results/walkforward_params.csv for details.\n"))
                        # Optionally apply final picks back to GUI and runtime overrides
                        try:
                            if getattr(wfo_opts, 'apply_to_gui_on_finish', False) and results:
                                final = results[-1]
                                core = getattr(final, 'core_params', {}) or {}
                                cutters = getattr(final, 'cutter_params', {}) or {}
                                # Final safety: reduce risk in 0.1% steps until Max DD <= 90% of GUI Max DD on full selected window
                                try:
                                    import pandas as pd
                                    dd_limit = 0.9 * float(gui_params.get('max_dd', 0))
                                    risk_current = float(core.get('risk_pct') or gui_params.get('risk_pct') or 0.01)
                                    # Ensure full-range data is prepared (WFO does this once already)
                                    # Apply cutter overrides before evaluation (reuse logic below)
                                    try:
                                        from config_manager import config as _cfg
                                        for k, v in cutters.items():
                                            if k in (
                                                'no_progress_timeout_enabled', 'cut_slow_bleeders_enabled',
                                                'loss_persistence_enabled', 'time_in_red_mae_enabled'
                                            ):
                                                _cfg.set_override('TRADING_PARAMETERS', k, bool(v))
                                        for k in (
                                            'no_progress_timeout_min', 'no_progress_target_frac', 'no_progress_mae_atr_frac', 'no_progress_mfe_target_frac',
                                            'time_in_red_minutes', 'time_in_red_mae_atr_frac',
                                            'lp_window_min', 'lp_atr_loss_threshold', 'lp_percent_below_threshold', 'lp_grace_period_min',
                                        ):
                                            if k in cutters:
                                                _cfg.set_override('TRADING_CONSTANTS', k, cutters.get(k))
                                    except Exception:
                                        pass

                                    def _eval_dd(risk_fraction: float) -> float:
                                        try:
                                            p = dict(gui_params)
                                            p.update(core)
                                            trader_eval = AutomatedTrendTrader(
                                                trades_output=None,
                                                initial_balance=int(p.get('account_size', 50000)),
                                                risk_per_trade=float(risk_fraction),
                                                max_trailing_dd=int(p.get('max_dd', 2500)),
                                                max_contracts=int(p.get('max_contracts', 100)),
                                                contract_type=self.contract_frame.contract_var.get(),
                                                atr_period=int(p.get('atr_period', 14)),
                                                atr_stop_multiple=float(p.get('atr_stop', 2.0)),
                                                atr_target_multiple=float(p.get('atr_target', 1.0)),
                                                stop_after_2_losses=bool(p.get('stop_after_2_losses', True)),
                                                target_daily_points=int(p.get('target_points', 18)),
                                                Setting_avoid_lunch_hour=bool(p.get('Setting_avoid_lunch_hour', False)),
                                                follow_market=bool(p.get('follow_market', False)),
                                                zone_high_mult=float(p.get('zone_high_mult', 1.0)),
                                                zone_low_mult=float(p.get('zone_low_mult', 1.5)),
                                                Setting_require_candle_confirm=bool(p.get('Setting_require_candle_confirm', False)),
                                                zone_entry_mode=p.get('zone_entry_mode', 'default'),
                                                allow_dd_recovery=bool(p.get('allow_dd_recovery', True)),
                                                Setting_Quick_Exit=bool(p.get('Setting_Quick_Exit', True)),
                                                minute_loss_ticks=int(p.get('minute_loss_ticks', 8)),
                                                trailing_stop_enabled=bool(p.get('trailing_stop_enabled', True)),
                                                atr_target_mode=bool(p.get('atr_target_mode', False)),
                                            )
                                            df_eval = trader_eval.run_backtest(
                                                self.data_manager.high_tf_data,
                                                self.data_manager.low_tf_data,
                                                minute_df=self.data_manager.final_minute_data,
                                                trading_start_date=pd.to_datetime(selected_start),
                                                trading_end_date=pd.to_datetime(selected_end)
                                            )
                                            if df_eval is None or df_eval.empty:
                                                return 0.0
                                            try:
                                                dd_col = float(df_eval.get('Max_Drawdown', pd.Series(dtype=float)).max())
                                            except Exception:
                                                dd_col = 0.0
                                            # Equity-based DD as fallback
                                            try:
                                                eq = df_eval.get('Trade_PnL', pd.Series(dtype=float)).cumsum()
                                                dd_eq = float((eq.cummax() - eq).max()) if not eq.empty else 0.0
                                            except Exception:
                                                dd_eq = 0.0
                                            return max(dd_col, dd_eq)
                                        except Exception:
                                            return 0.0

                                    step = 0.001  # 0.1% in decimal
                                    adj = risk_current
                                    # Only reduce (never increase)
                                    while adj >= 0.003:  # floor at 0.3%
                                        dd_val = _eval_dd(adj)
                                        try:
                                            self.root.after(0, lambda _r=adj: self.trading_frame.append_results(f"[WFO] Final risk check: trying {(_r*100):.2f}% …\n"))
                                        except Exception:
                                            pass
                                        if dd_val <= dd_limit:
                                            risk_current = adj
                                            break
                                        adj = round(adj - step, 4)
                                    # Update core with adjusted risk
                                    core['risk_pct'] = float(risk_current)
                                    try:
                                        self.root.after(0, lambda _r=risk_current, _lim=dd_limit: self.trading_frame.append_results(
                                            f"[WFO] Final risk set to {(_r*100):.2f}% to satisfy DD <= {int(_lim)}\n"))
                                    except Exception:
                                        pass
                                except Exception:
                                    pass
                                # Update main parameter fields in the GUI (convert risk to percent for display)
                                def _apply_gui():
                                    try:
                                        updates = {}
                                        if 'risk_pct' in core and core.get('risk_pct') is not None:
                                            updates['risk_pct'] = float(core['risk_pct']) * 100.0
                                        if 'zone_low_mult' in core and core.get('zone_low_mult') is not None:
                                            updates['zone_low_mult'] = float(core['zone_low_mult'])
                                        if 'atr_stop' in core and core.get('atr_stop') is not None:
                                            updates['atr_stop'] = float(core['atr_stop'])
                                        if updates:
                                            try:
                                                self.parameter_frame.update_optimized_values(updates)
                                            except Exception:
                                                # Fallback: set individual entries
                                                if 'risk_pct' in updates:
                                                    self.parameter_frame.set_entry_value('risk', updates['risk_pct'])
                                                if 'zone_low_mult' in updates:
                                                    self.parameter_frame.set_entry_value('zone_low_mult', updates['zone_low_mult'])
                                                if 'atr_stop' in updates:
                                                    self.parameter_frame.set_entry_value('atr_stop', updates['atr_stop'])
                                    except Exception:
                                        pass
                                self.root.after(0, _apply_gui)
                                # Apply cutter toggles as runtime overrides
                                try:
                                    from config_manager import config as _cfg
                                    for k, v in cutters.items():
                                        # Determine section based on known keys
                                        if k in (
                                            'no_progress_timeout_enabled', 'cut_slow_bleeders_enabled',
                                            'loss_persistence_enabled', 'time_in_red_mae_enabled'
                                        ):
                                            _cfg.set_override('TRADING_PARAMETERS', k, bool(v))
                                        elif k in (
                                            'no_progress_timeout_min', 'time_in_red_minutes', 'lp_window_min',
                                            'lp_percent_below_threshold', 'lp_grace_period_min'
                                        ):
                                            _cfg.set_override('TRADING_CONSTANTS', k, int(v))
                                        elif k in (
                                            'no_progress_mae_atr_frac', 'no_progress_mfe_target_frac',
                                            'time_in_red_mae_atr_frac', 'lp_atr_loss_threshold'
                                        ):
                                            _cfg.set_override('TRADING_CONSTANTS', k, float(v))
                                except Exception:
                                    pass
                        except Exception:
                            pass
                    except Exception as e:
                        self.root.after(0, lambda: messagebox.showerror("Backtest", f"WFO failed: {e}"))
                    finally:
                        self.root.after(0, self.trading_frame.enable_analyze_button)
                        self.root.after(0, lambda: self.trading_frame.update_progress("Ready"))
                        try:
                            # Disable (do not destroy) the stop button and clear command
                            if hasattr(self.trading_frame, 'stop_wfo_button') and self.trading_frame.stop_wfo_button:
                                self.trading_frame.stop_wfo_button.configure(state=tk.DISABLED, command=lambda: None)
                                # Ensure layout remains: Clear Terminal — Stop — BckTst
                                try:
                                    self.trading_frame.reposition_backtest_buttons()
                                except Exception:
                                    pass
                            elif stop_btn is not None:
                                stop_btn.destroy()
                        except Exception:
                            pass
                        try:
                            self.root.after(0, lambda: (
                                hasattr(self.trading_frame, 'calibrate_button') and self.trading_frame.calibrate_button.configure(state=tk.NORMAL)
                            ))
                        except Exception:
                            pass

                threading.Thread(target=_wfo_worker, daemon=True).start()
                return
            except Exception as e:
                messagebox.showerror("Backtest", f"Failed to start WFO: {e}")
                return

        # Ensure data is loaded for the standard backtest flow
        if (self.data_manager.final_minute_data is not None and 
            self.data_manager.low_tf_data is not None and 
            self.data_manager.high_tf_data is not None):
    
            # Get basic parameters for logging
            params = self.parameter_frame.get_parameters()
            contract_symbol = self.contract.symbol if self.contract else "Unknown"
            
            logger.info(f"Starting backtest analysis: Contract={contract_symbol}, "
                    f"Risk={params['risk_pct'] * 100:.2f}%, Target={params['target_points']}pts")

            logger.info(f"Data loaded - Final minute: {len(self.data_manager.final_minute_data)} rows, "
                        f"Low TF: {len(self.data_manager.low_tf_data)} rows, "
                        f"High TF: {len(self.data_manager.high_tf_data)} rows")
            
            # Log filter selection
            filter_params = self.indicator_filter_gui.get_filters()
            active_filters = []
            
            if filter_params.get('use_rsi'):
                rsi_max = filter_params.get('rsi_long_max', 70)
                rsi_min = filter_params.get('rsi_short_min', 30)
                active_filters.append(f"RSI({rsi_max:.2f}/{rsi_min:.2f})")
            
            if filter_params.get('use_macd'):
                macd_threshold = filter_params.get('macd_diff_threshold', 0)
                active_filters.append(f"MACD(>{macd_threshold:.2f})")
            
            if filter_params.get('use_ema'):
                active_filters.append("EMA")
            
            if filter_params.get('use_stoch_k'):
                k_upper = filter_params.get('k_upper_bound', 80)
                k_lower = filter_params.get('k_lower_bound', 20)
                active_filters.append(f"%K({k_upper}/{k_lower})")
            
            if active_filters:
                filter_text = " + ".join(active_filters)
                logger.info(f"Selected: {filter_text}")
    
            # Proceed with existing analysis_handler
            self.trading_frame.update_progress("Starting backtest...")
            analysis_handler = AnalysisHandler(
                data_manager=self.data_manager,
                parameter_handler=self.parameter_frame,
                trading_frame=self.trading_frame
            )
            analysis_handler.run_analysis()
        else:
            # Auto-load data (non-WFO path) then run analysis
            try:
                symbol = self.contract.symbol if self.contract else None
                if not symbol:
                    messagebox.showerror("Backtest", "Select a contract first.")
                    return
                if not hasattr(self.data_loading_frame, 'selected_start_date') or self.data_loading_frame.selected_start_date is None:
                    messagebox.showerror("Backtest", "Select a date range first.")
                    return
                if not hasattr(self.data_loading_frame, 'selected_end_date') or self.data_loading_frame.selected_end_date is None:
                    messagebox.showerror("Backtest", "Select a date range first.")
                    return

                low_tf = int(self.data_loading_frame.low_timeframe.get() or 8)
                high_tf = int(self.data_loading_frame.high_timeframe.get() or 120)
                end_dt = self.data_loading_frame.selected_end_date
                start_trading = self.data_loading_frame.selected_start_date
                import pandas as pd
                warmup_start = (pd.to_datetime(start_trading) - pd.tseries.offsets.BDay(20)).strftime('%Y-%m-%d %H:%M')
                end_str = pd.to_datetime(end_dt).strftime('%Y-%m-%d %H:%M')

                self.trading_frame.update_progress("Preparing data for backtest…")
                self.trading_frame.disable_analyze_button()

                def _load_then_analyze():
                    try:
                        self.data_manager.execute_backtest_stored_procedure(
                            start_date=warmup_start,
                            end_date=end_str,
                            symbol_base=symbol,
                            low_timeframe=low_tf,
                            high_timeframe=high_tf,
                            status_callback=self._make_throttled_status_cb(prefix='[Load] ')
                        )
                        self.data_manager.load_data_after_backtest(
                            symbol=symbol,
                            low_timeframe=low_tf,
                            high_timeframe=high_tf,
                            start_date=warmup_start,
                            end_date=end_str
                        )
                        # Ensure attributes used elsewhere are set
                        try:
                            self.data_loading_frame.actual_start_date = pd.to_datetime(start_trading)
                            self.data_loading_frame.actual_end_date = pd.to_datetime(end_dt)
                        except Exception:
                            pass
                        # Update UI then start the standard analysis on main thread
                        def _start_analysis():
                            try:
                                self.trading_frame.append_results("\n[Load] Data prepared. Starting backtest…\n")
                            except Exception:
                                pass
                            analysis_handler = AnalysisHandler(
                                data_manager=self.data_manager,
                                parameter_handler=self.parameter_frame,
                                trading_frame=self.trading_frame
                            )
                            analysis_handler.run_analysis()
                        self.root.after(0, _start_analysis)
                    except Exception as e:
                        self.root.after(0, lambda: messagebox.showerror("Backtest", f"Auto-load failed: {e}"))
                    finally:
                        self.root.after(0, self.trading_frame.enable_analyze_button)
                        self.root.after(0, lambda: self.trading_frame.update_progress("Ready"))

                threading.Thread(target=_load_then_analyze, daemon=True).start()
                return
            except Exception as e:
                messagebox.showerror("Backtest", f"Failed to auto-load: {e}")
                return
    def start_trading(self):
        """Invoked by 'Start' (for Live Simulation or Live)."""
        if not self.simulator:
            return

        params = self.parameter_frame.get_parameters()
        trade_history_params = self.indicator_filter_gui.get_trade_history_params()

        # Strictly validate trading/backtest/optimizer parameters
        validated_params = build_param_dict(params)

        # Build the automated trader
        trader = AutomatedTrendTrader(
            initial_balance=validated_params['account_size'],
            risk_per_trade=validated_params['risk_pct'],
            max_trailing_dd=validated_params['max_dd'],
            max_contracts=validated_params['max_contracts'],
            contract_type=self.contract.symbol if self.contract else "MNQ",
            atr_period=validated_params['atr_period'],
            atr_stop_multiple=validated_params['atr_stop'],
            atr_target_multiple=validated_params['atr_target'],
            target_daily_points=validated_params['target_points'],
            Setting_avoid_lunch_hour=validated_params['Setting_avoid_lunch_hour'],
            follow_market=validated_params['follow_market'],
            zone_high_mult=validated_params['zone_high_mult'],
            zone_low_mult=validated_params['zone_low_mult'],
            Setting_require_candle_confirm=validated_params['Setting_require_candle_confirm'],
            zone_entry_mode=validated_params['zone_entry_mode'],
            allow_dd_recovery=validated_params['allow_dd_recovery'],
            # Trailing stop settings:
            trailing_stop_enabled=validated_params.get('trailing_stop_enabled', True),
            atr_target_mode=validated_params.get('atr_target_mode', False),
            # Trade history:
            max_age_days=trade_history_params['max_age_days'],
            max_trades=trade_history_params['max_trades'],
            min_trades_for_double=trade_history_params.get('min_trades_for_multiply', 5),
            consec_losses_revert=trade_history_params['consec_losses_revert'],
            wr_threshold=trade_history_params['wr_threshold']
        )

        mode = self.trading_frame.mode_var.get()
        if mode == "Live Simulation":
            if self.data_manager.low_tf_data is None:
                messagebox.showerror("Error", "Low timeframe data is not loaded.")
                return
            self.simulator.start_simulation(self.data_manager.low_tf_data)
        else:
            self.simulator.start_live_feed({})

        self.trading_frame.stop_button.configure(state=tk.NORMAL)
        self.trading_frame.start_button.configure(state=tk.DISABLED)

    def stop_trading(self):
        """Stop the simulator."""
        if self.simulator:
            self.simulator.stop()
        self.trading_frame.stop_button.configure(state=tk.DISABLED)
        self.trading_frame.start_button.configure(state=tk.NORMAL)

    def check_optimization_requirements(self):
        """Check if all requirements for optimization are met."""
        try:
            # Check if data is loaded
            if (self.data_manager.final_minute_data is None or 
                self.data_manager.low_tf_data is None or 
                self.data_manager.high_tf_data is None):
                return False

            # Get parameters to validate
            params = self.parameter_frame.get_parameters()
            trade_history_params = self.indicator_filter_gui.get_trade_history_params()
            
            # Validate all numeric parameters
            required_numeric = [
                ('Risk %', params.get('risk_pct', 0)),
                ('Target Points', params.get('target_points', 0)),
                ('Account Size', params.get('account_size', 0)),
                ('Max Drawdown', params.get('max_dd', 0)),
                ('ATR Period', params.get('atr_period', 0)),
                ('Max Contracts', params.get('max_contracts', 0)),
                ('ATR Stop', params.get('atr_stop', 0)),
                ('ATR Target', params.get('atr_target', 0)),
                ('Zone High Mult', params.get('zone_high_mult', 0)),
                ('Zone Low Mult', params.get('zone_low_mult', 0)),
                ('Minute Loss Ticks', params['minute_loss_ticks']),  # Strict access - no defaults
                ('Win Rate Threshold', trade_history_params.get('wr_threshold', 0))
            ]
            
            # Check that all parameters have valid positive values
            invalid_params = []
            for name, value in required_numeric:
                try:
                    val = float(value)
                    if val <= 0:
                        invalid_params.append(f"{name}={val}")
                except (ValueError, TypeError):
                    invalid_params.append(f"{name}={value}")
            
            if invalid_params:
                logger.debug(f"Invalid parameters: {', '.join(invalid_params)}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking optimization requirements: {str(e)}")
            return False

    def run_optimization(self, n_calls=get_optimization_default('regular_n_calls'), n_initial_points=get_optimization_default('regular_n_initial_points')):
        """Run Bayesian optimization with current GUI parameters.
        
        Args:
            n_calls: Number of optimization calls
            n_initial_points: Number of initial random points
            two_phase: Whether to run two-phase optimization
            filter_trials: Number of filter trials for two-phase mode
        """
        # FIRST: Set threshold to 1.0 as required for ALL optimizations
        if hasattr(self, 'indicator_filter_gui'):
            self.indicator_filter_gui.wr_threshold_var.set(1.0)
            logger.info("Threshold set to 1.0 for optimization")
        
        # Check requirements
        if not self.check_optimization_requirements():
            logger.error("Optimization requirements not met - missing parameters or data")
            messagebox.showerror("Error", "Please ensure all parameters are set and data is loaded before optimization.")
            return

        # Disable optimize button and update progress
        self.trading_frame.disable_optimize_button()
        self.trading_frame.update_progress("Running optimization...")

        logger.info(f"Starting Bayesian optimization: {n_calls} calls, {n_initial_points} initial points")

        try:
            # Get parameters from GUI
            params = self.parameter_frame.get_parameters()
            trade_history_params = self.indicator_filter_gui.get_trade_history_params()

            # Get timeframes and dates from data loading frame
            low_tf = self.trading_frame.data_loading_frame.low_timeframe.get()
            high_tf = self.trading_frame.data_loading_frame.high_timeframe.get()
            start_date = self.trading_frame.data_loading_frame.selected_start_date
            end_date = self.trading_frame.data_loading_frame.selected_end_date
            
            logger.info(f"Optimization parameters: Contract={self.contract.symbol if self.contract else 'Unknown'}, "
                       f"Timeframes={low_tf}min/{high_tf}min, Date range={start_date} to {end_date}")
            logger.debug(f"Strategy parameters: Risk={params['risk_pct']}%, Target={params['target_points']}pts, "
                        f"Max contracts={params['max_contracts']}, ATR period={params['atr_period']}")
            
            # Collect all parameters for optimizer
            optimizer_params = {
                'initial_balance': float(params['account_size']),
                'max_trailing_dd': float(params['max_dd']),
                'contract_type': self.contract.symbol if self.contract else "MNQ",
                'atr_period': int(params['atr_period']),
                'Setting_avoid_lunch_hour': params['Setting_avoid_lunch_hour'],
                'follow_market': params['follow_market'],
                'Setting_require_candle_confirm': params['Setting_require_candle_confirm'],
                'zone_entry_mode': params['zone_entry_mode'],
                'allow_dd_recovery': params['allow_dd_recovery'],
                'low_timeframe': int(low_tf),
                'high_timeframe': int(high_tf),
                'minute_loss_ticks': int(params['minute_loss_ticks']),
                'Setting_Quick_Exit': bool(params['Setting_Quick_Exit']),
                'stop_after_2_losses': bool(params['stop_after_2_losses']),
                'wr_threshold': float(trade_history_params['wr_threshold']),
                'trading_start_date': self.data_loading_frame.actual_start_date,
                'trading_end_date': self.data_loading_frame.actual_end_date
            }

            # Set parameters in optimizer
            self.optimizer.set_gui_parameters(optimizer_params)
            logger.info("GUI parameters set in optimizer, starting optimization thread")

            # Run optimization in a separate thread
            def optimization_thread():
                try:
                    # Standard optimization
                    logger.info("Executing standard Bayesian optimization algorithm")
                    result = self.optimizer.run_optimization(n_calls=n_calls, n_initial_points=n_initial_points)
                    
                    # No output is written to the terminal or GUI at this point.
                    # Now query the database for the top 5 trials.
                    conn_string = (
                        "Driver={ODBC Driver 17 for SQL Server};"
                        "Server=localhost;"
                        "Database=BackTestData;"
                        "Trusted_Connection=yes;"
                        f"Timeout={get_system_timeout('connection_timeout')};"
                    )
                    # Retry mechanism for fetching top 5 trials.
                    max_retries = 3
                    retry_delay = get_system_timeout('retry_delay')  # seconds
                    trial_list = None
                    for attempt in range(max_retries):
                        try:
                            trial_list = get_top5_trials_from_db(conn_string)
                            break  # Successfully fetched, exit the loop.
                        except Exception as db_e:
                            logger.error(f"DB fetch attempt {attempt+1} failed: {db_e}")
                            if attempt < max_retries - 1:
                                import time
                                time.sleep(retry_delay)
                            else:
                                raise  # Raise the last exception if all attempts fail.
                    
                    # Update the GUI progress (without appending trial details)
                    self.root.after(0, lambda: self.trading_frame.update_progress("Optimization complete"))
                    
                    logger.info(f"Optimization completed successfully, retrieved {len(trial_list) if trial_list else 0} top trials")
                    
                    # Define a function that calls the new optimization results dialog
                    def handle_gui_update():
                        # Convert trial_list to DataFrame format expected by new dialog
                        if trial_list:
                            import pandas as pd
                            trials_data = []
                            for trial in trial_list:
                                trial_data = {
                                    'score': -trial.get('Score', 0),  # Convert back to positive
                                    'net_pnl': trial.get('TotalPnL', 0),
                                    'win_rate': trial.get('WinRate', 0),
                                    'total_trades': trial.get('TotalTrades', 0),
                                    'max_drawdown': trial.get('MaxDrawdown', 0),
                                    'profit_factor': trial.get('ProfitFactor', 0),
                                    'risk_pct': trial.get('RiskPct', 0) / 100,  # Convert to decimal
                                    'atr_stop': trial.get('ATRStop', 0),
                                    'zone_low_mult': trial.get('ZoneLowMult', 0),
                                    'target_points': trial.get('TargetPoints', 0)
                                }
                                trials_data.append(trial_data)
                            
                            trials_df = pd.DataFrame(trials_data)
                            
                            # Get current parameters
                            current_params = {
                                'risk_pct': self.parameter_frame.get_parameters().get('risk_pct', 0.01),
                                'atr_stop': self.parameter_frame.get_parameters().get('atr_stop', 2.0),
                                'zone_low_mult': self.parameter_frame.get_parameters().get('zone_low_mult', 1.5),
                                'target_points': self.parameter_frame.get_parameters().get('target_points', 10)
                            }
                            
                            # Show the new dialog
                            from BayesianOptimization.OptimizationResultsDialog import OptimizationResultsDialog
                            dialog = OptimizationResultsDialog(self.root, trials_df, current_params)
                            self.root.wait_window(dialog.dialog)
                            
                            result = dialog.get_result()
                            if result is not None and result != 'keep_current':
                                # Apply selected parameters
                                if isinstance(result, int):
                                    selected_trial = trials_data[result]
                                    # Update GUI parameters
                                    self.parameter_frame.risk_pct_var.set(f"{selected_trial['risk_pct']:.4f}")
                                    self.parameter_frame.atr_stop_var.set(f"{selected_trial['atr_stop']:.2f}")
                                    self.parameter_frame.zone_low_mult_var.set(f"{selected_trial['zone_low_mult']:.2f}")
                                    self.parameter_frame.target_var.set(str(int(selected_trial['target_points'])))
                                    self.trading_frame.append_results("\n✓ Parameters updated with selected optimization results")
                                else:
                                    self.trading_frame.append_results("\n✓ Keeping current parameters")
                            else:
                                self.trading_frame.append_results("\nOptimization cancelled - no changes made")
                        else:
                            self.trading_frame.append_results("\nNo optimization results available")
                    
                    # Schedule the handle_gui_update call on the main thread
                    self.root.after(0, handle_gui_update)
                    
                except Exception as e:
                    error_msg = str(e) if str(e) else "Unknown error occurred during optimization"
                    logger.error(f"Optimization thread failed: {error_msg}", exc_info=True)
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Optimization failed: {error_msg}"))
                finally:
                    logger.info("Optimization thread completed, re-enabling optimize button")
                    self.root.after(0, self.trading_frame.enable_optimize_button)

            thread = threading.Thread(target=optimization_thread)
            thread.daemon = True
            thread.start()
            
            
        except Exception as e:
            error_msg = str(e) if str(e) else "Unknown error occurred while starting optimization"
            logger.error(f"Failed to start optimization: {error_msg}")
            messagebox.showerror("Error", f"Failed to start optimization: {error_msg}")
            self.trading_frame.enable_optimize_button()



    def _apply_optimal_filters(self, filter_config):
        """Apply discovered filters to GUI"""
        if not hasattr(self, 'indicator_filter_gui'):
            return
            
        filter_gui = self.indicator_filter_gui
        
        # Reset all filters
        filter_gui.use_rsi_var.set(False)
        filter_gui.use_macd_var.set(False)
        filter_gui.use_ema_var.set(False)
        filter_gui.use_stoch_k_var.set(False)
        
        # Apply discovered configuration
        if filter_config.get('use_rsi'):
            filter_gui.use_rsi_var.set(True)
            filter_gui.rsi_long_max_var.set(filter_config.get('rsi_long_max', 70))
            filter_gui.rsi_short_min_var.set(filter_config.get('rsi_short_min', 30))
        
        if filter_config.get('use_macd'):
            filter_gui.use_macd_var.set(True)
            filter_gui.macd_diff_var.set(filter_config.get('macd_diff_threshold', 0))
        
        if filter_config.get('use_ema'):
            filter_gui.use_ema_var.set(True)
        
        if filter_config.get('use_stoch_k'):
            filter_gui.use_stoch_k_var.set(True)
            filter_gui.k_upper_var.set(filter_config.get('k_upper_bound', 80))
            filter_gui.k_lower_var.set(filter_config.get('k_lower_bound', 20))
        
        logger.info(f"Applied optimal filters to GUI")

    # ------------------ LOG CLEANUP ------------------
    def schedule_log_cleanup(self):
        """Schedule periodic log cleanup to remove entries older than 30 days."""
        try:
            # Perform initial cleanup
            cleanup_logs()
            logger.info("----------------------------------------------------")
            logger.info("Initial log cleanup completed on application startup")
            logger.info("----------------------------------------------------")
        except Exception as e:
            logger.error(f"Error during initial log cleanup: {str(e)}")
        
        # Schedule periodic cleanup every 24 hours (86400000 milliseconds)
        self.root.after(86400000, self.schedule_log_cleanup)

    def create_slow_bleeder_section(self, parent):
        """Create the slow-bleeder monitoring checkboxes section."""
        import tkinter as tk
        from config_manager import require_trading_default
        
        # Create frame for slow-bleeder section
        bleeder_frame = ttk.LabelFrame(parent, text="Slow-Bleeder Monitoring", padding="8", style='LightBlue.TLabelframe')
        bleeder_frame.grid(row=0, column=2, padx=(5,0), sticky='nsew')
        
        # Initialize slow-bleeder variables
        self.cut_slow_bleeders_enabled_var = tk.BooleanVar(value=require_trading_default('cut_slow_bleeders_enabled'))
        self.loss_persistence_enabled_var = tk.BooleanVar(value=require_trading_default('loss_persistence_enabled'))
        self.no_progress_timeout_enabled_var = tk.BooleanVar(value=require_trading_default('no_progress_timeout_enabled'))
        self.time_in_red_mae_enabled_var = tk.BooleanVar(value=require_trading_default('time_in_red_mae_enabled'))
        
        # Master toggle: Cut Slow Bleeders
        master_cb = ttk.Checkbutton(
            bleeder_frame,
            text="Cut Slow Bleeders",
            variable=self.cut_slow_bleeders_enabled_var
        )
        master_cb.grid(row=0, column=0, padx=5, pady=4, sticky='w')

        # Visual demarcation between master and sub-toggles
        ttk.Separator(bleeder_frame, orient='horizontal').grid(row=1, column=0, sticky='ew', padx=(5, 5), pady=(2, 6))

        # Sub-toggles container with indentation
        sub_frame = ttk.Frame(bleeder_frame)
        sub_frame.grid(row=2, column=0, sticky='w', padx=(18, 0))

        # Sub-toggle: Loss Persistence Monitor
        loss_cb = ttk.Checkbutton(
            sub_frame,
            text="Loss Persistence",
            variable=self.loss_persistence_enabled_var
        )
        loss_cb.grid(row=0, column=0, padx=2, pady=2, sticky='w')

        # Sub-toggle: No Progress Timeout
        progress_cb = ttk.Checkbutton(
            sub_frame,
            text="No Progress Timeout",
            variable=self.no_progress_timeout_enabled_var
        )
        progress_cb.grid(row=1, column=0, padx=2, pady=2, sticky='w')

        # Sub-toggle: Time-in-Red + MAE
        time_cb = ttk.Checkbutton(
            sub_frame,
            text="Time-in-Red + MAE",
            variable=self.time_in_red_mae_enabled_var
        )
        time_cb.grid(row=2, column=0, padx=2, pady=2, sticky='w')

        # Disable/enable sub-toggles based on master
        def _update_sub_toggle_state(*_):
            state = ('normal' if self.cut_slow_bleeders_enabled_var.get() else 'disabled')
            for cb in (loss_cb, progress_cb, time_cb):
                cb.configure(state=state)

        # Initial state and trace
        _update_sub_toggle_state()
        try:
            self.cut_slow_bleeders_enabled_var.trace_add('write', _update_sub_toggle_state)
        except Exception:
            # Fallback for older Tk versions
            self.cut_slow_bleeders_enabled_var.trace('w', _update_sub_toggle_state)
        
        # Descriptive note
        note_label = ttk.Label(
            bleeder_frame,
            text="Monitors help cut slow,\npersistent losers early.",
            font=('TkDefaultFont', 8, 'italic'),
            foreground='gray',
            justify='left'
        )
        note_label.grid(row=3, column=0, padx=5, pady=(6,2), sticky='w')
        
        # Store references for access from other parts of the application
        self.slow_bleeder_checkboxes = {
            "Cut Slow Bleeders": master_cb,
            "Loss Persistence": loss_cb,
            "No Progress Timeout": progress_cb,
            "Time-in-Red + MAE": time_cb
        }

    def get_slow_bleeder_states(self):
        """Get the current state of slow-bleeder checkboxes."""
        return {
            'cut_slow_bleeders_enabled': self.cut_slow_bleeders_enabled_var.get(),
            'loss_persistence_enabled': self.loss_persistence_enabled_var.get(),
            'no_progress_timeout_enabled': self.no_progress_timeout_enabled_var.get(),
            'time_in_red_mae_enabled': self.time_in_red_mae_enabled_var.get()
        }

    # ------------------ QUEUE / PROGRESS ------------------
    def check_progress_updates(self):
        """Called repeatedly to update progress text from queue."""
        try:
            while not self.progress_queue.empty():
                progress = self.progress_queue.get_nowait()
                if isinstance(progress, str):
                    self.trading_frame.update_progress(progress)
        except queue.Empty:
            pass
        finally:
            self.root.after(
                100 if self.processing_active else 1000,
                self.check_progress_updates
            )

    def _make_throttled_status_cb(self, prefix: str = ''):
        import time
        def _cb(message: str):
            now = time.time()
            # Emit at most once per ~1.2s, but always allow final summaries
            bursty = any(k in str(message).lower() for k in ["chunk", "reading raw data", "rows"])
            if bursty and (now - getattr(self, '_last_status_emit_ts', 0.0)) < 1.2:
                return
            self._last_status_emit_ts = now
            try:
                self.root.after(0, lambda: self.trading_frame.append_results(f"{prefix}{message}\n"))
            except Exception:
                pass
        return _cb


def main():
    root = tk.Tk()
    app = FuturesTradingApp(root)

    def on_closing():
        if app.simulator:
            app.simulator.stop()
        for thread in threading.enumerate():
            if thread != threading.main_thread():
                try:
                    thread.join(timeout=get_system_timeout('thread_join_timeout'))
                except:
                    pass
        root.quit()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        logger.warning("Shutting down gracefully...")  # Use WARNING so it shows in all modes
        root.quit()
        root.destroy()
    except Exception as e:
        logger.error(f"Shutdown error: {str(e)}")
        root.quit()
        root.destroy()

if __name__ == "__main__":
    main()
