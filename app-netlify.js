// Simple Netlify-ready frontend: fetch Yahoo via a Netlify function, build CSV, render Plotly charts,
// and coordinate single-tab handoff via BroadcastChannel/localStorage. No changes to existing files.

(function(){
  const PLOTLY_LOADED = !!window.Plotly;

  const dateEl = document.getElementById('date');
  const genBtn = document.getElementById('genBtn');
  const resultsEl = document.getElementById('results');
  const saveAllBtn = document.getElementById('saveAllBtn');
  const msgEl = document.getElementById('msg');
  const chartsEl = document.getElementById('charts');
  const allBox = document.getElementById('all');

  function todayISO(){ const d=new Date(); d.setMinutes(d.getMinutes()-d.getTimezoneOffset()); return d.toISOString().slice(0,10); }
  function setDefaultDate(){ dateEl.value = todayISO(); }
  setDefaultDate();

  function nowETParts(){
    const parts = new Intl.DateTimeFormat('en-US', {
      timeZone: 'America/New_York', year:'numeric', month:'2-digit', day:'2-digit',
      hour:'2-digit', minute:'2-digit', hour12:false
    }).formatToParts(new Date());
    return parts.reduce((acc,p)=>{ acc[p.type]=p.value; return acc; }, {});
  }

  function validateCompletedETDay(dateISO){
    const p = nowETParts();
    const todayET = `${p.year}-${p.month}-${p.day}`;
    const minutesET = (parseInt(p.hour,10) * 60) + parseInt(p.minute,10);
    if (dateISO > todayET) return 'Selected date is in the future. Choose a completed trading day.';
    if (dateISO === todayET && minutesET < (16*60 + 15)) return "Today's session is not complete (closes 16:15 ET). Pick a prior date or try again after close.";
    return '';
  }

  function symBoxes(){ return Array.prototype.slice.call(document.querySelectorAll('.sym')); }
  allBox.addEventListener('change', function(){ symBoxes().forEach(b=>b.checked = allBox.checked); });

  function paddedRange(min, max, p=0.03){ const span = Math.max(1e-9, max - min), pad = span * p; return [min - pad, max + pad]; }

  function buildCsv(rows){
    return rows.map(r => r.map(v => (''+v).replace(/"/g,'""')).join(',')).join('\n');
  }
  function downloadCsv(csvText, filename){
    const blob = new Blob([csvText], { type:'text/csv;charset=utf-8' });
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = filename; a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1500);
  }

  function functionsBase(){
    // When opened as file://, there is no functions host. Require http(s) origin for fetch.
    const isFile = location.protocol === 'file:' || location.origin === 'null';
    if (isFile) throw new Error('Open this app over http(s) (e.g., Netlify preview/site). file:// cannot call functions.');
    return `${location.origin}/.netlify/functions`;
  }

  function parseGmtOffsetToMinutes(gmt){
    // gmt like 'GMT-4' or 'GMT-04:00'
    let s = String(gmt || 'GMT-04:00').replace('GMT','');
    if (!s.includes(':')) s = (s.length ? s+':00' : '+00:00');
    const sign = s.startsWith('-')? -1 : 1;
    const [hh,mm] = s.replace('+','').replace('-','').split(':');
    return sign * (parseInt(hh,10)*60 + parseInt(mm,10));
  }

  function etOffsetMinutes(dateISO){
    const parts = new Intl.DateTimeFormat('en-US', { timeZone:'America/New_York', timeZoneName:'shortOffset', year:'numeric', month:'2-digit', day:'2-digit' })
      .formatToParts(new Date(dateISO+'T12:00:00Z'));
    const tz = parts.find(p=>p.type==='timeZoneName')?.value || 'GMT-04:00';
    return parseGmtOffsetToMinutes(tz);
  }

  function epochSecondsForET(dateISO, hm){
    // hm 'HH:MM' ET → epoch seconds UTC
    const [H,M] = hm.split(':').map(n=>parseInt(n,10));
    const offMin = etOffsetMinutes(dateISO); // minutes east of UTC (negative for -04:00)
    const ms = Date.UTC(parseInt(dateISO.slice(0,4),10), parseInt(dateISO.slice(5,7),10)-1, parseInt(dateISO.slice(8,10),10), H, M) - (offMin*60*1000);
    return Math.floor(ms/1000);
  }

  async function fetchYahoo(symbol, targetISO){
    // Fetch exactly the selected ET session window 09:30–16:15 using period1/period2
    const today = new Date();
    const diffDays = Math.floor((today - new Date(targetISO + 'T00:00:00Z'))/(1000*60*60*24));
    const interval = diffDays <= 25 ? '1m' : '5m';
    const p1 = epochSecondsForET(targetISO, '09:30');
    const p2 = epochSecondsForET(targetISO, '16:15');
    if (!(Number.isFinite(p1) && Number.isFinite(p2) && p2 > p1)) {
      throw new Error('Failed to compute ET session window');
    }
    const url = new URL(functionsBase() + '/fetch_chart');
    url.searchParams.set('symbol', symbol);
    url.searchParams.set('interval', interval);
    url.searchParams.set('period1', String(p1));
    url.searchParams.set('period2', String(p2));
    url.searchParams.set('includePrePost', 'false');
    const r = await fetch(url.toString(), { cache:'no-store' });
    if (!r.ok) throw new Error(await r.text());
    return await r.json();
  }

  function clampSessionET(ts, values){
    // ts: Date[] or ISO strings; values: numbers
    const outTs = [], outVals = [];
    for (let i=0;i<ts.length;i++){
      const t = new Date(ts[i]);
      const hhmm = t.getUTCHours()*100 + t.getUTCMinutes();
      // Approx ET session 09:30–16:15 ~ handle by local conversion from UTC not exact ET
      // For simplicity, keep full day; Plotly slider enables focus.
      outTs.push(t);
      outVals.push(values[i]);
    }
    return { ts: outTs, vals: outVals };
  }

  function renderChart(divId, ts, es, vix, label){
    if (!PLOTLY_LOADED) return;
    // Autorange by letting Plotly compute ranges
    Plotly.newPlot(divId, [
      { x: ts, y: vix, mode:'lines', name:'VIX close', line:{ color:'#f59e0b', width:2 } },
      { x: ts, y: es,  mode:'lines', name:`${label} close`,  yaxis:'y2', line:{ color:'#60a5fa', width:2 } },
    ], {
      title: `${label} vs VIX`,
      margin:{ t:32, r:40, l:50, b:40 },
      paper_bgcolor:'#0b0d13', plot_bgcolor:'#0b0d13', font:{ color:'#e5e7eb' },
      xaxis:{ title:'Time', rangeslider:{ visible:true }, showgrid:true, gridcolor:'#2a2f3a', gridwidth:1, zeroline:false, linecolor:'#4b5563', tickcolor:'#4b5563' },
      yaxis:{ title:'VIX', autorange:true, showgrid:true, gridcolor:'#2a2f3a', gridwidth:1, zeroline:false, linecolor:'#4b5563', tickcolor:'#4b5563' },
      yaxis2:{ title:label, overlaying:'y', side:'right', autorange:true },
      legend:{ orientation:'h', y:1.02, yanchor:'bottom', x:0.01 }
    }, { displaylogo:false, responsive:true });
  }

  async function generate(){
    const dateISO = dateEl.value || todayISO();
    const validationError = validateCompletedETDay(dateISO);
    if (validationError){ msgEl.textContent = validationError; return; }
    const labels = symBoxes().filter(x=>x.checked).map(x=>x.value);
    msgEl.textContent = 'Working…'; resultsEl.textContent=''; chartsEl.innerHTML='';
    const outputs = [];
    try{
      for (const label of labels){
        const yfSym = { ES:'ES=F', MES:'MES=F', NQ:'NQ=F', MNQ:'MNQ=F' }[label] || 'ES=F';
        const dataFut = await fetchYahoo(yfSym, dateISO);
        const dataVix = await fetchYahoo('^VIX', dateISO);
        const ts = (dataFut.chart.result[0].timestamp || []).map(s=>new Date(s*1000));
        const esClose = dataFut.chart.result[0].indicators.quote[0].close || [];
        const vts = (dataVix.chart.result[0].timestamp || []).map(s=>new Date(s*1000));
        const vixClose = dataVix.chart.result[0].indicators.quote[0].close || [];

        // Align by timestamp
        const mapV = new Map(vts.map((t,i)=>[+t, vixClose[i]]));
        const alignedTs = [], esVals = [], vixVals = [];
        for (let i=0;i<ts.length;i++){
          const key = +ts[i];
          if (mapV.has(key)){
            alignedTs.push(ts[i]); esVals.push(esClose[i]); vixVals.push(mapV.get(key));
          }
        }

        // Filter to exact selected ET date and session 09:30–16:15, and remove invalid values
        function etParts(dUtc){
          return new Intl.DateTimeFormat('en-US', { timeZone:'America/New_York', year:'numeric', month:'2-digit', day:'2-digit', hour:'2-digit', minute:'2-digit', hour12:false })
            .formatToParts(dUtc).reduce((a,p)=>{ a[p.type]=p.value; return a; }, {});
        }
        function isSelectedSession(dUtc){
          const p = etParts(dUtc);
          const d = `${p.year}-${p.month}-${p.day}`;
          const hm = `${p.hour}:${p.minute}`;
          return d === dateISO && hm >= '09:30' && hm <= '16:15';
        }
        const pairs = alignedTs.map((t,i)=>({ t, es: esVals[i], v: vixVals[i] }))
          .filter(p => Number.isFinite(p.es) && Number.isFinite(p.v) && isSelectedSession(p.t));
        const tsClean = pairs.map(p=>p.t);
        const esClean = pairs.map(p=>p.es);
        const vixClean = pairs.map(p=>p.v);

        // Shift timestamps so axis displays ET regardless of user's local timezone
        const localOffsetEast = -new Date().getTimezoneOffset();
        const etEast = etOffsetMinutes(dateISO); // negative for -04:00
        const deltaMin = localOffsetEast - etEast; // add delta to display ET clock locally
        const tsDisplay = tsClean.map(d => new Date(+d + deltaMin*60000));

        // CSV rows
        const fmt = (d)=>{
          const y=d.getFullYear(), m=String(d.getMonth()+1).padStart(2,'0'), da=String(d.getDate()).padStart(2,'0');
          const hh=String(d.getHours()).padStart(2,'0'), mm=String(d.getMinutes()).padStart(2,'0');
          return `${y}-${m}-${da} ${hh}:${mm}`;
        };
        // ET timestamp formatting for CSV rows
        function fmtET(dUtc){
          const parts = new Intl.DateTimeFormat('en-US', { timeZone:'America/New_York',
            year:'numeric', month:'2-digit', day:'2-digit', hour:'2-digit', minute:'2-digit', hour12:false
          }).formatToParts(dUtc).reduce((a,p)=>{ a[p.type]=p.value; return a; }, {});
          return `${parts.year}-${parts.month}-${parts.day} ${parts.hour}:${parts.minute}`;
        }
        const rows = [[ 'timestamp_et','es_close','vix_close' ]];
        for (let i=0;i<tsClean.length;i++) rows.push([ fmtET(tsClean[i]), esClean[i], vixClean[i] ]);
        const fname = `${label.toLowerCase()}_vix_1min_${dateISO}_0930_1615_ET.csv`;
        outputs.push({ label, filename: fname, rows, ts: tsDisplay, es: esClean, vix: vixClean });
      }

      // Results links + charts
      resultsEl.innerHTML = outputs.map(o => `<div><span style="color:#9ca3af;margin-right:8px">${o.label}:</span><button data-fn="${o.filename}">Download CSV</button></div>`).join('');
      resultsEl.querySelectorAll('button').forEach(btn => {
        btn.addEventListener('click', () => {
          const o = outputs.find(x => x.filename === btn.getAttribute('data-fn'));
          if (o){ const csv = buildCsv(o.rows); downloadCsv(csv, o.filename); }
        });
      });
      const csvPayloads = [];
      outputs.forEach((o, i) => {
        const div = document.createElement('div'); div.className='chart'; div.id = 'chart_'+i; chartsEl.appendChild(div);
        renderChart(div.id, o.ts, o.es, o.vix, o.label);
        csvPayloads.push({ filename: o.filename, rows: o.rows });
      });
      msgEl.textContent = 'Done.';
      saveAllBtn.disabled = csvPayloads.length === 0;
      saveAllBtn.onclick = function(){
        csvPayloads.forEach(p => { const csv = buildCsv(p.rows); downloadCsv(csv, p.filename); });
      };
    } catch(err){ msgEl.textContent = 'Failed: ' + err; }
  }
  genBtn.addEventListener('click', generate);

  // Single-tab handoff (cannot close other tab, but can disable it)
  try {
    const bc = new BroadcastChannel('spvsvix_singleton');
    bc.onmessage = (e) => { if (e.data && e.data.type === 'takeover') disableUi('Gracefully Shutting Down …'); };
    const lockKey = 'spvsvix_lock';
    const now = String(Date.now());
    localStorage.setItem(lockKey, now);
    setTimeout(() => {
      if (localStorage.getItem(lockKey) === now){ bc.postMessage({ type:'takeover' }); }
      else disableUi('Gracefully Shutting Down …');
    }, 100);
  } catch(_) {}

  function disableUi(msg){
    msgEl.textContent = msg; msgEl.style.color = '#f59e0b';
    Array.prototype.forEach.call(document.querySelectorAll('button,input'), el => el.disabled = true);
  }
})();


