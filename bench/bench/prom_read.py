#!/usr/bin/env python3
"""Lecture des series live (fenetre bench) — values = [[ts, "val"], ...]."""
import json, datetime, urllib.request, urllib.parse

PROM = "http://127.0.0.1:9090"
NOW = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
FROM = NOW - 3600

def q(name):
    url = PROM + "/api/v1/query_range?query=" + urllib.parse.quote(name) + f"&start={FROM}&end={NOW}&step=15s"
    with urllib.request.urlopen(url, timeout=15) as r:
        return json.loads(r.read())

def f(v):
    return float(v[1]) if isinstance(v, list) else float(v)

for M in ["ollama_generation_tokens_per_second", "ollama_prompt_tokens_per_second", "ollama_context_tokens"]:
    d = q(M)
    for res in d["data"]["result"]:
        gpu = res["metric"].get("gpu")
        vals = res.get("values", [])
        nz = [(t, f(v)) for t, v in vals if f(v) > 0]
        if not nz:
            print(f"{M} [{gpu}]: tous zeros sur la fenetre"); continue
        t0 = datetime.datetime.fromtimestamp(nz[0][0], datetime.UTC).strftime("%H:%M:%S")
        t1 = datetime.datetime.fromtimestamp(nz[-1][0], datetime.UTC).strftime("%H:%M:%S")
        print(f"{M} [{gpu}]: nb>0={len(nz)}  min={min(v for _,v in nz):.1f}  max={max(v for _,v in nz):.1f}  {t0} -> {t1} UTC")
