#!/usr/bin/env python3
"""Contrôle Grafana : pendant un run actif, poller :9444/metrics (ce que Grafana
voit en live) toutes les 2s, puis comparer aux valeurs finales renvoyees par l'API."""
import json, threading, time, urllib.request

BASE = "http://127.0.0.1:11434"
MODEL = "qwen3.8-27b-q6-mtp-vision:latest"
WORDS = ["le","modele","contexte","memoire","tenseur","attention","cache","prompt"]
FILL = " ".join(WORDS[i % 8] for i in range(16000))  # ~16k tokens

stop = threading.Event()
samples = []
def poller():
    while not stop.is_set():
        try:
            with urllib.request.urlopen("http://127.0.0.1:9444/metrics", timeout=3) as r:
                txt = r.read().decode()
            d = {}
            for line in txt.splitlines():
                if line.startswith("#") or not line.strip(): continue
                parts = line.rsplit(" ", 1)
                d[parts[0]] = float(parts[1])
            gen = d.get('ollama_generation_tokens_per_second{gpu="V100"}')
            prm = d.get('ollama_prompt_tokens_per_second{gpu="V100"}')
            ctx = d.get('ollama_context_tokens{gpu="V100"}')
            if gen is not None:
                samples.append((round(time.time()), round(gen,1), round(prm,1) if prm else None, round(ctx) if ctx else None))
        except Exception:
            pass
        time.sleep(2)

t = threading.Thread(target=poller, daemon=True)
t.start()

body = json.dumps({
    "model": MODEL,
    "messages": [{"role":"user","content": FILL + "\nReponds exactement : OK-BENCH."}],
    "stream": False, "think": False,
    "options": {"num_ctx": 65536, "num_predict": 256, "keep_alive": -1},
}).encode()
req = urllib.request.Request(BASE+"/api/chat", data=body, headers={"Content-Type":"application/json"})
t0 = time.time()
with urllib.request.urlopen(req, timeout=600) as r:
    d = json.loads(r.read())
wall = time.time()-t0
stop.set()
time.sleep(1)

api = {
    "wall_s": round(wall,2),
    "prompt_tokens": d.get("prompt_eval_count"),
    "gen_tokens": d.get("eval_count"),
    "prefill_tps": round(d["prompt_eval_count"]/(d["prompt_eval_duration"]/1e9),1),
    "gen_tps": round(d["eval_count"]/(d["eval_duration"]/1e9),1),
}
print("=== VALEURS API (finales, fiables) ===")
print(json.dumps(api, ensure_ascii=False, indent=1))
print("\n=== GAUGES EXPORTEUR pollées pendant le run (t, gen_tps, prompt_tps, ctx) ===")
for s in samples:
    print(s, flush=True)
json.dump({"api": api, "samples": samples},
          open("/home/guile/llama-experiments/bench/live_gauge_check.json","w"), indent=1)
