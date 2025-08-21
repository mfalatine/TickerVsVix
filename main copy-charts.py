# main.py
# Produces raw CSV (09:30–16:15 ET):
# pip install requests pandas numpy matplotlib python-dateutil plotly tzdata

import requests, pandas as pd, numpy as np
from datetime import datetime, date, time, timedelta
from dateutil.tz import gettz
import os
import argparse

# Optional desktop HTML UI without a server
try:
    import webview  # pywebview
except ImportError:
    webview = None

# ===================== CONFIG =====================
ET = gettz("America/Detroit")
SESSION_START = time(9, 30)             # 09:30 ET
SESSION_END   = time(16, 15)            # 16:15 ET
USE_DESKTOP_UI = True                   # use local HTML window (no server)

# Supported futures symbols (friendly label -> Yahoo symbol)
FUTURES_SYMBOLS = {
    "ES":  "ES=F",
    "MES": "MES=F",
    "NQ":  "NQ=F",
    "MNQ": "MNQ=F",
}

# Output directory
RESULTS_DIR = "Results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def _out(pathname: str) -> str:
    return os.path.join(RESULTS_DIR, pathname)

# Regime thresholds (VIX)
REGIME_BINS   = [-np.inf, 15, 20, 30, 40, np.inf]
REGIME_LABELS = ["Calm (<15)", "Low (15-20)", "Elevated (20-30)", "Stress (30-40)", "Crisis (≥40)"]

# ================== DATA FETCH ====================
def fetch_yahoo(symbol: str, target_date: date, interval: str = "1m") -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    headers = {"User-Agent": "Mozilla/5.0"}

    today_et = datetime.now(ET).date()
    if target_date == today_et:
        params = {"interval": interval, "range": "1d", "includePrePost": "false"}
    else:
        start_et = datetime.combine(target_date, time(0, 0)).replace(tzinfo=ET)
        end_et = start_et + timedelta(days=1)
        period1 = int(start_et.astimezone(gettz("UTC")).timestamp())
        period2 = int(end_et.astimezone(gettz("UTC")).timestamp())
        params = {
            "interval": interval,
            "period1": str(period1),
            "period2": str(period2),
            "includePrePost": "false",
        }

    r = requests.get(url, params=params, headers=headers, timeout=20)
    r.raise_for_status()
    j = r.json()
    res = j["chart"]["result"][0]
    ts = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(ET)
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame(
        {"open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"], "volume": q["volume"]},
        index=ts
    ).dropna()
    df.index.name = "datetime_et"
    return df

def try_pair(primary: str, fallback: str, target_date: date, interval: str):
    for s in (primary, fallback):
        try:
            return s, fetch_yahoo(s, target_date, interval)
        except Exception:
            continue
    raise RuntimeError(f"Both {primary} and {fallback} failed.")

def resolve_futures_symbol(preferred: str | None) -> tuple[str, str]:
    if preferred:
        label = preferred.upper()
        if label in FUTURES_SYMBOLS:
            return label, FUTURES_SYMBOLS[label]
    # default to ES
    return "ES", FUTURES_SYMBOLS["ES"]

def fallback_label_for(label: str) -> str | None:
    pair = {
        "ES": "MES",
        "MES": "ES",
        "NQ": "MNQ",
        "MNQ": "NQ",
    }
    return pair.get(label.upper())

# (Chart and Flask web UI code removed as unused)

def generate_for_date(target_date: date, futures_preference: str | None = None):
    date_str = target_date.isoformat()

    # Only allow completed trading days (today only after session end)
    now_et = datetime.now(ET)
    today_et = now_et.date()
    if target_date > today_et:
        raise SystemExit("Selected date is in the future. Choose a completed trading day.")
    if target_date == today_et and now_et.time() < SESSION_END:
        raise SystemExit("Today's session is not complete (closes 16:15 ET). Pick a prior date or try again after close.")

    # Futures selection with fallback to paired micro/mini
    chosen_label, chosen_symbol = resolve_futures_symbol(futures_preference)
    fallback = fallback_label_for(chosen_label)
    primary = FUTURES_SYMBOLS[chosen_label]
    fallback_sym = FUTURES_SYMBOLS.get(fallback, primary)
    # Yahoo 1m data is limited historically (~30 days). For older dates, degrade to 5m.
    days_back = (datetime.now(ET).date() - target_date).days
    interval = "1m" if days_back <= 25 else "5m"
    used_symbol, es = try_pair(primary, fallback_sym, target_date, interval)
    # Symbol label used (map back from yahoo code)
    used_label = next((k for k, v in FUTURES_SYMBOLS.items() if v == used_symbol), chosen_label)
    # VIX index (fallback VIXY ETF)
    vix_sym, vix = try_pair("^VIX", "VIXY", target_date, interval)

    # clamp to 09:30–16:15 ET
    start = datetime.combine(target_date, SESSION_START, ET)
    end   = datetime.combine(target_date, SESSION_END,   ET)
    es  = es.loc[(es.index  >= start) & (es.index  <= end)]
    vix = vix.loc[(vix.index >= start) & (vix.index <= end)]

    # inner join on timestamps
    df = (pd.DataFrame(index=es.index)
          .join(es["close"].rename("es_close"))
          .join(vix["close"].rename("vix_close"))
          .dropna())
    if df.empty:
        raise SystemExit("No overlapping 1-min data in the requested window.")

    # derived fields
    df["es_pct_from_open"]  = (df["es_close"]  / df["es_close"].iloc[0]  - 1.0) * 100.0
    df["vix_pct_from_open"] = (df["vix_close"] / df["vix_close"].iloc[0] - 1.0) * 100.0
    df["regime"] = pd.cut(df["vix_close"], bins=REGIME_BINS, labels=REGIME_LABELS, right=False)

    outputs = {}

    # write CSV
    csv_path = _out(f"{used_label.lower()}_vix_1min_{date_str}_0930_1615_ET.csv")
    out = df.copy()
    fmt = "%Y-%m-%d %H:%M"
    out.insert(0, "timestamp_et", out.index.strftime(fmt))
    out_cols = ["timestamp_et", "es_close", "vix_close", "es_pct_from_open", "vix_pct_from_open", "regime"]
    out[out_cols].to_csv(csv_path, index=False)
    print("Wrote CSV:", csv_path)
    outputs["csv"] = csv_path

    # Provide compact series for inline charts in the desktop UI
    try:
        outputs["series"] = {
            "timestamp": out["timestamp_et"].tolist(),
            "es_close": out["es_close"].astype(float).tolist(),
            "vix_close": out["vix_close"].astype(float).tolist(),
        }
    except Exception:
        outputs["series"] = {"error": True}

    return outputs


# (Flask server UI removed)

DESKTOP_HTML = """
<!doctype html>
<html>
  <head>
    <meta charset=\"utf-8\" />
    <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Intraday Futures CSV Generator</title>
    <style>
      :root { color-scheme: dark; }
      body { background:#0f1115; color:#e5e7eb; font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Arial, Helvetica, \"Apple Color Emoji\", \"Segoe UI Emoji\"; margin:0; }
      .container { max-width: 880px; margin: 0 auto; padding: 24px; }
      h2 { margin: 0 0 14px; font-size: 22px; }
      .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-bottom: 8px; }
      input[type=date] { padding: 8px 10px; background:#111827; color:#e5e7eb; border:1px solid #374151; border-radius:6px; }
      button { padding: 8px 12px; background:#2563eb; color:#fff; border:0; border-radius:6px; cursor:pointer; }
      button.secondary { background:#374151; }
      button:disabled { opacity: .55; cursor: not-allowed; }
      .symbols { display:flex; gap:14px; flex-wrap: wrap; margin: 8px 0 2px; }
      .note { color:#9ca3af; margin-top:8px; font-size:.95em; }
      #msg { margin-top: 12px; }
      #charts { margin-top: 16px; display: grid; grid-template-columns: 1fr; gap: 16px; }
      .chart { width: 100%; height: 440px; border:1px solid #374151; border-radius: 8px; background:#0b0d13; }
      code { background:#111827; color:#d1d5db; padding:2px 6px; border-radius:4px; }
    </style>
  </head>
  <body>
    <main class=\"container\">
      <h2>Intraday Futures CSV Generator</h2>
      <div class=\"row\">
        <label for=\"date\">Date (ET)</label>
        <input type=\"date\" id=\"date\" />
        <button id=\"genBtn\" type=\"button\">Generate</button>
        <button id=\"openBtn\" class=\"secondary\" type=\"button\">Open Results Folder</button>
      </div>
      <div class=\"symbols\">
        <label><input id=\"all\" type=\"checkbox\" checked /> Select all</label>
        <label><input type=\"checkbox\" class=\"sym\" value=\"ES\" checked /> ES</label>
        <label><input type=\"checkbox\" class=\"sym\" value=\"MES\" checked /> MES</label>
        <label><input type=\"checkbox\" class=\"sym\" value=\"NQ\" checked /> NQ</label>
        <label><input type=\"checkbox\" class=\"sym\" value=\"MNQ\" checked /> MNQ</label>
      </div>
      <p class=\"note\">Note: Yahoo limits 1‑minute intraday history to roughly the past month. For older dates this tool automatically switches to 5‑minute data. Session window: 09:30–16:15 ET. Only completed trading days are allowed.</p>
      <p id=\"hint\" class=\"note\" style=\"display:none\"></p>
      <div id=\"msg\"></div>
      <div id=\"charts\"></div>
    </main>
    <script>
      const DEFAULT_DATE = "__DEFAULT_DATE__";
      const MAX_DATE = "__MAX_DATE__";
      function todayISO(){ const d=new Date(); d.setMinutes(d.getMinutes()-d.getTimezoneOffset()); return d.toISOString().slice(0,10); }
      const dateEl = document.getElementById('date');
      let initDate = (DEFAULT_DATE && DEFAULT_DATE !== "__DEFAULT_DATE__") ? DEFAULT_DATE : todayISO();
      dateEl.value = initDate;
      const allBox = document.getElementById('all');
      const symBoxes = () => Array.from(document.querySelectorAll('.sym'));
      const genBtn = document.getElementById('genBtn');
      const hint = document.getElementById('hint');
      allBox.addEventListener('change', () => {
        symBoxes().forEach(b => b.checked = allBox.checked);
        validate();
      });
      symBoxes().forEach(b => b.addEventListener('change', () => {
        const allChecked = symBoxes().every(x => x.checked);
        const noneChecked = symBoxes().every(x => !x.checked);
        allBox.indeterminate = !(allChecked || noneChecked);
        allBox.checked = allChecked && !allBox.indeterminate;
        validate();
      }));
      function validate(){
        genBtn.disabled = false;
        hint.textContent = '';
        hint.style.display = 'none';
        return true;
      }
      dateEl.addEventListener('input', validate);
      validate();
      async function generate(){
        const ds = dateEl.value;
        const selected = symBoxes().filter(x => x.checked).map(x => x.value);
        const el = document.getElementById('msg');
        el.textContent = 'Working…';
        try {
          const api = await getApi();
          const res = await api.generate({ date: ds, labels: selected });
          if(res && res.error){ el.textContent = 'Error: ' + res.error; return; }
          const items = ((res && res.csvs) || []).map(o => '<li><span style=\\'color:#9ca3af;margin-right:8px\\'>' + o.label + ':</span><code>' + o.path + '</code></li>').join('');
          el.innerHTML = items ? ('<ul style=\\'margin:8px 0;padding-left:18px\\'>' + items + '</ul>') : 'No outputs.';
          try { renderCharts((res && res.csvs) || []); } catch(e) { console.error('Chart render failed', e); }
        } catch(err){ el.textContent = 'Failed: ' + err; }
      }
      async function openResults(){
        try {
          const api = await getApi();
          await api.open_results();
        } catch(e){}
      }

      function getApi(){
        return new Promise((resolve, reject) => {
          const existing = window.pywebview && window.pywebview.api;
          if (existing) return resolve(existing);
          let tries = 0;
          function check(){
            const api = window.pywebview && window.pywebview.api;
            if (api) return resolve(api);
            if (++tries > 200) return reject('App API not ready');
            setTimeout(check, 50);
          }
          document.addEventListener('pywebviewready', () => {
            const api = window.pywebview && window.pywebview.api; if (api) resolve(api);
          });
          check();
        });
      }

      // Bind buttons programmatically to avoid inline handler issues in some environments
      document.getElementById('genBtn').addEventListener('click', generate);
      document.getElementById('openBtn').addEventListener('click', openResults);

      function padRange(min, max, padPct){ const span = Math.max(1e-9, max - min); const p = span * padPct; return [min - p, max + p]; }
      function renderCharts(items){
        const container = document.getElementById('charts');
        container.innerHTML = '';
        items.forEach((o, idx) => {
          if(!o.series || o.series.error){ return; }
          const div = document.createElement('div');
          div.className = 'chart';
          div.id = 'chart_' + idx;
          container.appendChild(div);
          const ts = o.series.timestamp;
          const es = (o.series.es_close || []).map(Number);
          const vix = (o.series.vix_close || []).map(Number);
          if(!ts || !es.length || !vix.length){ return; }
          const esMin = Math.min.apply(null, es);
          const esMax = Math.max.apply(null, es);
          const vixMin = Math.min.apply(null, vix);
          const vixMax = Math.max.apply(null, vix);
          const [es0, es1] = padRange(esMin, esMax, 0.03);
          const [vx0, vx1] = padRange(vixMin, vixMax, 0.03);
          const traces = [
            { x: ts, y: vix, mode: 'lines', name: 'vix_close', line: { color: '#f59e0b', width: 2 } },
            { x: ts, y: es,  mode: 'lines', name: 'es_close',  yaxis: 'y2', line: { color: '#60a5fa', width: 2 } }
          ];
          const layout = {
            title: (o.label + ' vs VIX'),
            margin: { t: 30, r: 40, l: 50, b: 40 },
            paper_bgcolor: '#0b0d13', plot_bgcolor: '#0b0d13',
            font: { color: '#e5e7eb' },
            xaxis: { title: 'Time (ET)', rangeslider: { visible: true } },
            yaxis: { title: 'VIX', range: [vx0, vx1] },
            yaxis2: { title: o.label + ' (ES)', overlaying: 'y', side: 'right', range: [es0, es1] },
            legend: { orientation: 'h', y: 1.02, yanchor: 'bottom', x: 0.01 }
          };
          Plotly.newPlot(div, traces, layout, { responsive: true, displaylogo: false });
        });
      }

      // Surface JS errors to the page so we can see what failed
      window.onerror = function(message, source, lineno, colno, error){
        const el = document.getElementById('msg');
        el.textContent = 'JS error: ' + message + ' (' + (source||'') + ':' + (lineno||'') + ')';
        return false;
      };
    </script>
  </body>
</html>
"""

class DesktopApi:
    def generate(self, payload):
        try:
            if isinstance(payload, dict):
                date_str = payload.get("date")
                labels = payload.get("labels")
            else:
                # legacy call signature support
                date_str = payload
                labels = None
            target = datetime.strptime(date_str, "%Y-%m-%d").date()
        except Exception as exc:
            return {"error": f"Invalid date: {exc}"}
        try:
            if not labels:
                labels = ["ES"]
            # Validate completed trading day before kicking off work
            now_et = datetime.now(ET)
            if target > now_et.date():
                return {"error": "Selected date is in the future. Choose a completed trading day."}
            if target == now_et.date() and now_et.time() < SESSION_END:
                return {"error": "Today's session is not complete (closes 16:15 ET). Pick a prior date or try again after close."}
            csvs = []
            for label in labels:
                outs = generate_for_date(target, futures_preference=label)
                csvs.append({
                    "label": label,
                    "path": outs.get("csv", ""),
                    "series": outs.get("series")
                })
            return {"csvs": csvs}
        except SystemExit as exc:
            return {"error": str(exc)}
        except Exception as exc:
            return {"error": str(exc)}

    def open_results(self):
        try:
            # Windows
            os.startfile(RESULTS_DIR)  # type: ignore[attr-defined]
        except Exception:
            try:
                import subprocess
                subprocess.Popen(["xdg-open", RESULTS_DIR])
            except Exception:
                pass
        return {"ok": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Ticker vs VIX CSV for a given date")
    parser.add_argument("--date", help="Target date in YYYY-MM-DD (ET)")
    parser.add_argument("--ui", action="store_true", help="Force desktop UI")
    parser.add_argument("--no-ui", action="store_true", help="Disable desktop UI")
    args = parser.parse_args()

    # Resolve target date
    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        except Exception:
            raise SystemExit("--date must be YYYY-MM-DD")
    else:
        target_date = datetime.now(ET).date()

    use_ui = (webview is not None) and ((USE_DESKTOP_UI and not args.no_ui) or args.ui)

    if use_ui:
        api = DesktopApi()
        max_date = datetime.now(ET)
        # If today before session end, set max to yesterday to prevent selection of incomplete day
        if max_date.time() < SESSION_END:
            max_date = max_date - timedelta(days=1)
        html = (DESKTOP_HTML
                .replace("__DEFAULT_DATE__", target_date.isoformat())
                .replace("__MAX_DATE__", max_date.date().isoformat()))
        window = webview.create_window(
            title="Ticker vs VIX — CSV Generator",
            html=html,
            width=900,
            height=700,
            js_api=api,
        )
        # Prefer modern Edge WebView2 on Windows so the UI's modern JS works
        try:
            webview.start(gui="edgechromium")
        except Exception:
            # Fallback to default selection if Edge WebView2 isn't available
            webview.start()
    else:
        # CLI mode
        outputs = generate_for_date(target_date)
        print("CSV:", outputs.get("csv"))
