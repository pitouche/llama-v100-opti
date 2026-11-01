#!/usr/bin/env python3
"""Auto-bench : tok/s vs taille de contexte (texte + vision) sur ollama V100 (11434).

Phase 1 (texte) : prompt de N tokens (tranches d'un gros texte ~55k), 128 tok gen.
Phase 2 (vision) : images synthetiques 256..2048 px, 64 tok gen, question courte.
A chaque run : relevés API ollama + gauges Prometheus (ce que voit Grafana) + nvidia-smi.
Sortie : auto_bench_results.jsonl + affichage progressif.
"""
import base64, io, json, random, subprocess, time, urllib.request
from gpu_uuid_select import v100_cuda_ordinal, v100_smi_index  # UUID GPU (jamais d'index en dur — voir GPU_UUID_NSYS_CUPTI_20261030.md)

BASE = "http://127.0.0.1:11434"
PROM = "http://127.0.0.1:9090"
MODEL = "qwen3.8-27b-q6-mtp-vision:latest"
OUT = "/home/guile/llama-experiments/bench/auto_bench_results.jsonl"
NUM_CTX = 65536  # constant pour toutes les reqs -> reutilise le runner charge (200k)

# ---------- helpers ----------
def prom(q):
    try:
        with urllib.request.urlopen(PROM + "/api/v1/query?query=" + urllib.request.quote(q), timeout=10) as r:
            d = json.loads(r.read())
        for res in d["data"]["result"]:
            if res["metric"].get("gpu") == "V100":
                return float(res["value"][1])
    except Exception:
        pass
    return None

def gpu1():
    out = subprocess.run(["nvidia-smi", "--id=" + v100_smi_index(),
        "--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    m, u, t, p = [x.strip() for x in out.split(",")]
    return {"mem_mib": int(m), "util_pct": int(u), "temp_c": int(t), "w": float(p.replace(".", "")) if p else None}

def chat(messages, num_predict, tag):
    body = json.dumps({
        "model": MODEL, "messages": messages, "stream": False, "think": False,
        "options": {"num_ctx": NUM_CTX, "num_predict": num_predict, "keep_alive": -1},
    }).encode()
    req = urllib.request.Request(BASE + "/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1200) as r:
        d = json.loads(r.read())
    wall = time.time() - t0
    row = {
        "tag": tag,
        "wall_s": round(wall, 2),
        "load_s": round(d.get("load_duration", 0) / 1e9, 2),
        "prompt_tokens": d.get("prompt_eval_count"),
        "gen_tokens": d.get("eval_count"),
        "prompt_s": round(d.get("prompt_eval_duration", 0) / 1e9, 2),
        "gen_s": round(d.get("eval_duration", 0) / 1e9, 2),
        "prefill_tps": round(d["prompt_eval_count"] / (d["prompt_eval_duration"] / 1e9), 1) if d.get("prompt_eval_duration") else None,
        "gen_tps": round(d["eval_count"] / (d["eval_duration"] / 1e9), 1) if d.get("eval_duration") else None,
        "response": d.get("message", {}).get("content", "")[:120],
    }
    # gauges prometheus juste apres la fin du run
    row["prom"] = {
        "gen_tps_gauge": prom('ollama_generation_tokens_per_second{gpu="V100"}'),
        "prompt_tps_gauge": prom('ollama_prompt_tokens_per_second{gpu="V100"}'),
        "ctx_tokens": prom('ollama_context_tokens{gpu="V100"}'),
        "ctx_limit": prom('ollama_context_limit_tokens{gpu="V100"}'),
    }
    row["gpu"] = gpu1()
    return row

def log(row):
    with open(OUT, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False), flush=True)

# ---------- texte de remplissage ~55k tokens (4.05 car/token d'apres fill20k) ----------
WORDS = ["le","modele","contexte","memoire","tenseur","attention","cache","prompt","decodage",
         "generation","vram","cuda","kernel","fenetre","glissante","latence","debits","buffer",
         "layer","groupe","recurrent","etats","passage","eval","charge","tests","resultats",
         "cible","draft","tokens","seconds","prefill","infer","quant","q6","mtp","speculatif"]
random.seed(1337)
FILL = " ".join(random.choice(WORDS) for _ in range(55000))
# ~55000 mots * 1.05 tok/moyen ≈ 57k tokens ; on tranchera en caractères

def run(label, messages, npred):
    print(f"\n>>> {label}", flush=True)
    try:
        row = chat(messages, npred, label)
    except Exception as e:
        row = {"tag": label, "error": str(e)}
    log(row)
    return row

# ---------- PHASE 1 : texte ----------
print("=== PHASE 1 : texte, gen 128 tok, prefill croissant ===", flush=True)
print(json.dumps({"tag": "repos_gpu", "gpu": gpu1()}, ensure_ascii=False), flush=True)
with open(OUT, "a") as f:
    f.write(json.dumps({"tag": "repos_gpu", "gpu": gpu1()}, ensure_ascii=False) + "\n")

# controle sans image pour la phase vision
run("txt_ref", [{"role": "user", "content": FILL[:300]}], 64)

for target_tok in [2048, 4096, 8192, 16384, 32768, 55000]:
    chars = int(target_tok * 4.05)
    run(f"txt_{target_tok}k",
        [{"role": "user",
          "content": FILL[:chars] + "\nReponds exactement : OK-BENCH."}], 128)

# ---------- PHASE 2 : vision ----------
print("\n=== PHASE 2 : vision, images synthetiques, gen 64 tok ===", flush=True)

def make_image(size, seed=7):
    from PIL import Image, ImageDraw
    rng = random.Random(seed)
    img = Image.new("RGB", (size, size))
    px = img.load()
    # degrade de fond
    for y in range(size):
        for x in range(0, size, 4):
            px[x, y] = (int(255 * x / size), int(255 * y / size), 128)
    d = ImageDraw.Draw(img)
    n_shapes = max(8, size // 24)
    for _ in range(n_shapes):
        x0 = rng.randrange(0, size); y0 = rng.randrange(0, size)
        r = rng.randrange(size // 12, size // 3)
        c = (rng.randrange(256), rng.randrange(256), rng.randrange(256))
        if rng.random() < 0.5:
            d.ellipse([x0 - r, y0 - r, x0 + r, y0 + r], fill=c, outline=(0, 0, 0), width=2)
        else:
            d.rectangle([x0, y0, x0 + r, y0 + r], fill=c, outline=(255, 255, 0), width=2)
    d.text((10, 10), f"bench vision {size}x{size} - formes et details", fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

for size in [256, 512, 768, 1024, 1344, 2048]:
    b64 = make_image(size, seed=size)
    row = run(f"img_{size}",
              [{"role": "user",
                "content": "Compte les grandes formes visibles et decris la couleur dominante du fond en une phrase.",
                "images": [b64]}], 64)

# texte de taille comparable a la plus grosse image (pour comparer prefill vision vs texte a ctx egale)
big = [r for r in open(OUT) if r.strip() and '"img_2048"' in r]
if big:
    try:
        n_tok = json.loads(big[-1])["prompt_tokens"] - 40  # ~tokens image (texte ref ~40)
        chars = int(max(1000, n_tok) * 4.05)
        run(f"txt_equiv_{n_tok}k",
            [{"role": "user", "content": FILL[:chars] + "\nReponds exactement : OK-BENCH."}], 64)
    except Exception:
        pass

print("\n=== FIN BENCH ===", flush=True)
