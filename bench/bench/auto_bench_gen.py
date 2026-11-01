#!/usr/bin/env python3
"""Bis du bench : tok/s de GENERATION vs contexte, avec 128 tokens gen reels
(prompt "compte de 1 a 128" -> longueur fiable), ctx croissant via le meme texte."""
import json, time, urllib.request, random

BASE = "http://127.0.0.1:11434"
MODEL = "qwen3.8-27b-q6-mtp-vision:latest"
OUT = "/home/guile/llama-experiments/bench/auto_bench_gen.jsonl"
WORDS = ["le","modele","contexte","memoire","tenseur","attention","cache","prompt","decodage","generation"]
random.seed(1337)
FILL = " ".join(random.choice(WORDS) for _ in range(55000))

def chat(chars):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user",
            "content": FILL[:chars] + "\nCompte maintenant de 1 a 128, un nombre par ligne, rien d'autre."}],
        "stream": False, "think": False,
        "options": {"num_ctx": 65536, "num_predict": 160, "keep_alive": -1},
    }).encode()
    req = urllib.request.Request(BASE+"/api/chat", data=body,
                                 headers={"Content-Type":"application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1200) as r:
        d = json.loads(r.read())
    row = {
        "target_chars": chars,
        "wall_s": round(time.time()-t0, 2),
        "prompt_tokens": d.get("prompt_eval_count"),
        "gen_tokens": d.get("eval_count"),
        "prefill_tps": round(d["prompt_eval_count"]/(d["prompt_eval_duration"]/1e9),1),
        "gen_tps": round(d["eval_count"]/(d["eval_duration"]/1e9),2),
        "gen_s": round(d["eval_duration"]/1e9,2),
    }
    print(json.dumps(row), flush=True)
    with open(OUT,"a") as f: f.write(json.dumps(row)+"\n")
    return row

for target in [2048, 8192, 16384, 32768, 55000]:
    chat(int(target*4.05))
print("FIN GEN SWEEP", flush=True)
