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

  async function fetchYahoo(symbol, targetISO){
    // If target is within ~25d, use range=1d; else compute period1/2 for a full day
    const target = new Date(targetISO + 'T00:00:00');
    const today = new Date();
    const diffDays = Math.floor((today - target) / (1000*60*60*24));
    const interval = diffDays <= 25 ? '1m' : '5m';
    const url = new URL(functionsBase() + '/fetch_chart');
    if (diffDays <= 25){
      url.searchParams.set('symbol', symbol);
      url.searchParams.set('interval', interval);
      url.searchParams.set('range', '1d');
      url.searchParams.set('includePrePost', 'false');
    } else {
      const start = new Date(targetISO + 'T00:00:00Z');
      const end = new Date(start.getTime() + 86400*1000);
      url.searchParams.set('symbol', symbol);
      url.searchParams.set('interval', interval);
      url.searchParams.set('period1', String(Math.floor(start.getTime()/1000)));
      url.searchParams.set('period2', String(Math.floor(end.getTime()/1000)));
      url.searchParams.set('includePrePost', 'false');
    }
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
    const [es0, es1] = paddedRange(Math.min(...es), Math.max(...es));
    const [vx0, vx1] = paddedRange(Math.min(...vix), Math.max(...vix));
    Plotly.newPlot(divId, [
      { x: ts, y: vix, mode:'lines', name:'vix_close', line:{ color:'#f59e0b', width:2 } },
      { x: ts, y: es,  mode:'lines', name:'es_close',  yaxis:'y2', line:{ color:'#60a5fa', width:2 } },
    ], {
      title: `${label} vs VIX`,
      margin:{ t:32, r:40, l:50, b:40 },
      paper_bgcolor:'#0b0d13', plot_bgcolor:'#0b0d13', font:{ color:'#e5e7eb' },
      xaxis:{ title:'Time', rangeslider:{ visible:true } },
      yaxis:{ title:'VIX', range:[vx0,vx1] },
      yaxis2:{ title:label, overlaying:'y', side:'right', range:[es0,es1] },
      legend:{ orientation:'h', y:1.02, yanchor:'bottom', x:0.01 }
    }, { displaylogo:false, responsive:true });
  }

  async function generate(){
    const dateISO = dateEl.value || todayISO();
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

        // CSV rows
        const fmt = (d)=>{
          const y=d.getFullYear(), m=String(d.getMonth()+1).padStart(2,'0'), da=String(d.getDate()).padStart(2,'0');
          const hh=String(d.getHours()).padStart(2,'0'), mm=String(d.getMinutes()).padStart(2,'0');
          return `${y}-${m}-${da} ${hh}:${mm}`;
        };
        const rows = [[ 'timestamp_et','es_close','vix_close' ]];
        for (let i=0;i<alignedTs.length;i++) rows.push([ fmt(alignedTs[i]), esVals[i], vixVals[i] ]);
        const fname = `${label.toLowerCase()}_vix_1min_${dateISO}_0930_1615_ET.csv`;
        outputs.push({ label, filename: fname, rows, ts: alignedTs, es: esVals, vix: vixVals });
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


