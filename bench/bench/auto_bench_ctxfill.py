#!/usr/bin/env python3
"""Bis 3 : gen t/s vs contexte FINAL, prefixe commun REUTILISE via KV cache.

- 1 warmup (premiere fois: prefill complet du prefixe ~48.5k)
- puis 6 runs avec meme prefixe + queue croissante: le prefixe est cache,
  seul le delta est prefilled. La generation s'execute a contexte total croissant.
"""
import json, time, urllib.request, random

BASE = "http://127.0.0.1:11434"
MODEL = "qwen3.8-27b-q6-mtp-vision:latest"
OUT = "/home/guile/llama-experiments/bench/auto_bench_ctxfill.jsonl"
WORDS = ["le","modele","contexte","memoire","tenseur","attention","cache","prompt","decodage","generation"]
random.seed(1337)
PREFIX = " ".join(random.choice(WORDS) for _ in range(55000))  # ~48.5k tok (mesure)
TAIL = " ".join(random.choice(WORDS) for _ in range(6000))      # ~5.3k tok max

def run(prompt, tag):
    body = json.dumps({
        "model": MODEL, "messages": [{"role": "user", "content": prompt}],
        "stream": False, "think": False,
        "options": {"num_predict": 64, "keep_alive": -1},
    }).encode()
    req = urllib.request.Request(BASE+"/api/chat", data=body,
                                 headers={"Content-Type":"application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    row = {
        "tag": tag,
        "wall_s": round(time.time()-t0, 2),
        "load_s": round(d.get("load_duration",0)/1e9, 2),
        "prompt_tokens": d.get("prompt_eval_count"),
        "gen_tokens": d.get("eval_count"),
        "prefill_tps": round(d["prompt_eval_count"]/(d["prompt_eval_duration"]/1e9),1),
        "gen_tps": round(d["eval_count"]/(d["eval_duration"]/1e9),2),
    }
    print(json.dumps(row), flush=True)
    with open(OUT,"a") as f: f.write(json.dumps(row)+"\n")

# warmup: prefixe complet (premiere fois, prefill 48.5k)
run(PREFIX + "\nCompte de 1 a 64, un nombre par ligne, rien d'autre.", "warmup_48k")
# runs avec queue croissante (prefixe cache)
for k in [0, 2000, 4000, 6000, 8000, 10000]:
    run(PREFIX + " " + TAIL[:k] + "\nCompte de 1 a 64, un nombre par ligne, rien d'autre.",
        f"fill_{k}")
print("FIN CTXFILL", flush=True)
