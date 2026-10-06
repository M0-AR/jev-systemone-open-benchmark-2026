"""Live / real-world state fetchers. All public, no keys, timeout-guarded.

Sources (all free, no auth):
- Frankfurter FX (ECB): latest USD->EUR/GBP/JPY move -> noul/score states
- CoinGecko: BTC 24h change -> market-triage states
- Open-Meteo: current temperature anomaly -> severity states
- HackerNews Algolia: top story titles -> routing states (live text)

Each returns a list of {"id","text","meta"} usable as System-One states.
Failures degrade to [] and are logged in results (never crash a run).
"""
import requests

TIMEOUT = 12


def _get(url, params=None):
    try:
        r = requests.get(url, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"_error": str(e)}


def fetch_fx_states():
    j = _get("https://api.frankfurter.app/latest", {"from": "USD", "to": "EUR,GBP,JPY"})
    if "_error" in j or "rates" not in j:
        return []
    out = []
    for ccy, rate in j["rates"].items():
        out.append({"id": f"fx-{ccy}",
                    "text": f"USD/{ccy} spot {rate} on {j.get('date')}. Decide: escalate treasury review?",
                    "meta": {"source": "frankfurter-ecb", "rate": rate}})
    return out


def fetch_crypto_states():
    j = _get("https://api.coingecko.com/api/v3/simple/price",
             {"ids": "bitcoin,ethereum", "vs_currencies": "usd", "include_24hr_change": "true"})
    if "_error" in j or "bitcoin" not in j:
        return []
    out = []
    for coin, v in j.items():
        ch = v.get("usd_24h_change", 0.0)
        out.append({"id": f"crypto-{coin}",
                    "text": f"{coin} ${v['usd']} 24h {ch:.2f}%. Decide: trigger risk alert?",
                    "meta": {"source": "coingecko", "change_24h": ch}})
    return out


def fetch_weather_states():
    j = _get("https://api.open-meteo.com/v1/forecast",
             {"latitude": 52.52, "longitude": 13.41, "current": "temperature_2m"})
    if "_error" in j or "current" not in j:
        return []
    t = j["current"]["temperature_2m"]
    return [{"id": "wx-berlin", "text": f"Berlin current {t}C. Decide: heat-risk protocol?",
             "meta": {"source": "open-meteo", "temp": t}}]


def fetch_hn_states(n=5):
    j = _get("https://hn.algolia.com/api/v1/search", {"tags": "front_page"})
    if "_error" in j or "hits" not in j:
        return []
    return [{"id": f"hn-{h.get('objectID')}", "text": f"HN: {h.get('title')}",
             "meta": {"source": "hn-algolia"}} for h in j["hits"][:n]]


def fetch_live_states():
    states = []
    for fn in (fetch_fx_states, fetch_crypto_states, fetch_weather_states, fetch_hn_states):
        try:
            states.extend(fn())
        except Exception:
            pass
    return states
