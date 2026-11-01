#!/usr/bin/env python3
"""Sweep de tailles de contexte sur l'ollama v100 (port 11434).

Pour chaque ctx (options.num_ctx):
  1. requête /api/chat avec un prompt d'environ 20k tokens + 64 tok générés
  2. mesures: prompt eval t/s, gen t/s, temps total
  3. relevé nvidia-smi juste après (avant unload) + nvidia-smi au repos
Les résultats sont dans ctx_sweep_results.json
"""
import json, subprocess, time, sys, urllib.request

BASE = "http://127.0.0.1:11434"
MODEL = "qwen3.8-27b-q6-mtp-vision:latest"
FILL = open("/home/guile/llama-experiments/bench/fill20k.txt").read()
NUM_PRED = 64
CTXS = [int(x) for x in sys.argv[1:]] or [100000, 131072, 160000, 190000]

def gpu():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout
    rows = [r.split(",") for r in out.strip().splitlines()]
    return {int(r[0]): {"mem_mib": int(r[1].strip()), "util_pct": int(r[2].strip())} for r in rows}

def chat(ctx):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user",
                      "content": FILL + "\nRéponds en une ligne avec : OK-BENCH."}],
        "stream": False, "think": False,
        "options": {"num_ctx": ctx, "num_predict": NUM_PRED},
    }).encode()
    req = urllib.request.Request(BASE + "/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    wall = time.time() - t0
    return d, wall

results = []
idle = gpu()
print("repos (modèle chargé, pas de tâche):", idle, flush=True)

for ctx in CTXS:
    print(f"\n=== ctx = {ctx} ===", flush=True)
    t0 = time.time()
    try:
        d, wall = chat(ctx)
    except Exception as e:
        print(f"  ERREUR: {e}", flush=True)
        results.append({"ctx": ctx, "error": str(e)})
        continue
    dt = time.time() - t0
    row = {
        "ctx": ctx,
        "prompt_tokens": d.get("prompt_eval_count"),
        "gen_tokens": d.get("eval_count"),
        "prompt_tps": (d["prompt_eval_count"] / (d["prompt_eval_duration"] / 1e9)
                       if d.get("prompt_eval_duration") else None),
        "gen_tps": (d["eval_count"] / (d["eval_duration"] / 1e9)
                    if d.get("eval_duration") else None),
        "load_s": d.get("load_duration", 0) / 1e9,
        "wall_s": round(dt, 1),
        "gpu": gpu(),
    }
    results.append(row)
    print(json.dumps(row, indent=1, ensure_ascii=False), flush=True)

json.dump(results, open("/home/guile/llama-experiments/bench/ctx_sweep_results.json", "w"),
          indent=1)
print("\nsauvegardé: ctx_sweep_results.json")
