export async function handler(event) {
  try {
    const qp = event.queryStringParameters || {};
    const { symbol, interval, period1, period2, range, includePrePost } = qp;
    if (!symbol || !interval) {
      return { statusCode: 400, body: 'symbol and interval are required' };
    }
    const base = `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol)}`;
    const params = new URLSearchParams();
    params.set('interval', interval);
    if (range) params.set('range', range);
    if (period1) params.set('period1', String(period1));
    if (period2) params.set('period2', String(period2));
    params.set('includePrePost', includePrePost || 'false');

    const url = `${base}?${params.toString()}`;
    const r = await fetch(url, { headers: { 'User-Agent': 'Mozilla/5.0' } });
    const txt = await r.text();
    return { statusCode: r.status, headers:{ 'content-type': 'application/json', 'cache-control':'no-store' }, body: txt };
  } catch (e) {
    return { statusCode: 500, body: String(e) };
  }
}


