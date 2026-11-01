#!/usr/bin/env python3
"""A/B contrôlé des optimisations (shortlist MTP / defer-prompt) sur le runner ollama-v100.

Usage: python3 ab_optim.py <label>
Envoie UNE requête contrôlée (prompt fixe, seed fixe, 256 tokens) et enregistre
les métriques dans results/ab_optim.jsonl. À comparer entre restarts.
"""
import json, sys, time, urllib.request

LABEL = sys.argv[1] if len(sys.argv) > 1 else "run"
URL = "http://127.0.0.1:11434/api/generate"

# Prompt fixe (~300 tokens, génération structurée, déterministe avec seed)
PROMPT = (
    "Tu es un assistant technique. Explique en détail le fonctionnement d'un "
    "cache KV quantisé q8_0 dans un moteur d'inférence LLM local. "
    "Structure ta réponse en sections numérotées avec des sous-points. "
    "Termine par un résumé de 3 phrases. Réponds en français."
)

payload = {
    "model": "qwen3.8-27b-q6-mtp-vision",
    "prompt": PROMPT,
    "stream": False,
    "seed": 42,
    "options": {"num_predict": 256, "temperature": 0.1, "top_k": 20},
}

t0 = time.time()
req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=600) as r:
    d = json.load(r)
t1 = time.time()

out = {
    "label": LABEL,
    "wall_s": round(t1 - t0, 2),
    "prompt_n": d.get("prompt_eval_count"),
    "prompt_ms": round(d.get("prompt_eval_duration", 0) / 1e6, 1),
    "gen_n": d.get("eval_count"),
    "gen_ms": round(d.get("eval_duration", 0) / 1e6, 1),
    "tg_tps": round(d.get("eval_count", 0) / max(d.get("eval_duration", 1) / 1e9, 1e-9), 2),
    "ttft_s": round(t1 - t0 - d.get("eval_duration", 0) / 1e9, 2),
    "response_head": d.get("response", "")[:160],
}
print(json.dumps(out, ensure_ascii=False, indent=1))
with open("/home/guile/llama-experiments/bench/results_ab_optim.jsonl", "a") as f:
    f.write(json.dumps(out, ensure_ascii=False) + "\n")
