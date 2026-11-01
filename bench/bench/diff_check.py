#!/usr/bin/env python3
"""Verifie le mecanisme reel : le runner garde la KV entre 2 req et n'injecte
que la diff de contexte (pas de "compression" par context-shift).

A: prompt P (~30k tok), premier passage -> prefill complet (chiffre froid honnête)
B: meme prompt P identique          -> si cache, prefill ~0 (diff nulle)
C: P + queue ~8k tok                -> prefill = juste la queue (diff)

Pas de num_ctx -> reutilise le runner charge en -c 200000 (pas de reload).
"""
import json, time, urllib.request, random

BASE = "http://127.0.0.1:11434"
MODEL = "qwen3.8-27b-q6-mtp-vision:latest"
OUT = "/home/guile/llama-experiments/bench/diff_check.jsonl"
WORDS = ["le", "modele", "contexte", "memoire", "tenseur", "attention", "cache", "prompt", "decodage", "generation"]

random.seed(4242)
P = " ".join(random.choice(WORDS) for _ in range(20000))   # ~28k tok
random.seed(99)
EXTRA = " ".join(random.choice(WORDS) for _ in range(6000))  # ~8.5k tok
Q = "\nReponds exactement : OK-BENCH."

def run(prompt, tag):
    body = json.dumps({
        "model": MODEL, "messages": [{"role": "user", "content": prompt}],
        "stream": False, "think": False,
        "options": {"num_predict": 8, "keep_alive": -1},
    }).encode()
    req = urllib.request.Request(BASE + "/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    row = {
        "tag": tag,
        "wall_s": round(time.time() - t0, 2),
        "prompt_tokens": d.get("prompt_eval_count"),
        "prompt_s": round(d.get("prompt_eval_duration", 0) / 1e9, 3),
        "gen_s": round(d.get("eval_duration", 0) / 1e9, 3),
        "prefill_tps_reported": round(d["prompt_eval_count"] / (d["prompt_eval_duration"] / 1e9), 1),
        "resp": d.get("message", {}).get("content", "")[:40],
    }
    print(json.dumps(row), flush=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(row) + "\n")

run(P + Q, "A_cold_28k")   # prefill complet
run(P + Q, "B_identique")  # diff = 0 -> prefill ~0
run(P + " " + EXTRA + Q, "C_avec_queue")  # diff = queue (~8.5k)
print("FIN DIFF CHECK", flush=True)
