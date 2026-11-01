#!/usr/bin/env python3
"""
ab_mmvq.py — A/B du lib H2 prod (a4b6f440) vs lib 42d9581 (prod + pick nwarps TURING sm70).

Méthode (1 server à la fois, port 8917, V100 uniquement) :
  1. unload ollama (keep_alive=0, keep_alive -1) pour libérer la VRAM
  2. pour chaque leg (A=prod, B=42d9581) :
     - GGML_BACKEND_PATH = le .so du leg (plugin isolé, comme bench_direct.py)
     - start llama-server MTP (mêmes args que bench_direct.py, -c réduit 98304 pour VRAM)
     - points warm (32768, 65536) : préfixes croissants, 64 gen tokens chacun
       -> t/s stable par point
     - point cold 16384, 64 gen
     - on vérifie VRAM > 1 GiB (anti-recul CPU) et que le log dit V100
     - arrêt server
  3. synthèse : ratio B/A par point (le gain du patch doit apparaître surtout
     sur les points longs : le draft T=1 passe par mmvq nwarps 4->2)

Aucun arrêt/restart du service ollama-v100 : uniquement /api/chat keep_alive=0
(le runner se relèvera au prochain appel). Le lock superviseur EST tenu pendant
tout le run (flock) : aucun autre bench ne peut entrer.

USAGE:
  flock -n /tmp/kolbo-supervisor-bench.lock -c 'python3 bench/ab_mmvq.py'
  (ou directement : le script prend lui-même le lock en mode verrou)
"""
import ctypes
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from gpu_uuid_select import v100_cuda_ordinal, v100_smi_index  # UUID GPU (jamais d'index en dur — voir GPU_UUID_NSYS_CUPTI_20261030.md)

D = os.path.dirname(os.path.abspath(__file__))

LLAMA_SERVER = "/home/guile/llama-experiments/ollama-v100/lib/ollama/llama-server"
GGUF = ("/usr/share/ollama/.ollama/models/blobs/"
        "sha256-493301830a596b8ad56dc1329f80bbcb578c8e910da395feafdc9cd8263430bb")
MMPROJ = ("/usr/share/ollama/.ollama/models/blobs/"
          "sha256-ac3714bfdddeca31351f2752bf1a63f266f4df87c0b68c895e44945ca704448e")

PORT = 8917
OUT_JSONL = os.path.join(D, "ab_mmvq_results.jsonl")
OUT_SUMMARY = os.path.join(D, "ab_mmvq_summary.md")
OUT_SERVER_LOG = os.path.join(D, "ab_mmvq_server.log")

CUDA_LIBS = [
    "/home/guile/llama-experiments/ollama-v100/lib/ollama",
    "/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12",
    "/home/guile/llama-experiments/tools/cuda-12.6/usr/local/cuda-12.6/lib64",
    "/home/guile/llama-experiments/tools/cuda-12.6/usr/local/cuda-12.6/targets/x86_64-linux/lib",
]
GGML_ENV = {
    "GGML_CUDA_ALLREDUCE": "internal",
    "GGML_CUDA_AR_COPY_THRESHOLD": "131072",
    "GGML_CUDA_VOLTA_Q8_FATTN_TC": "1",
    "GGML_CUDA_VOLTA_Q5_X4": "1",
    "GGML_CUDA_VOLTA_Q6_W4R4": "1",
    "GGML_CUDA_VOLTA_GQA8_NCOLS2": "2",
    "LLAMA_ARG_FIT_TARGET": "1024",
}
SO_PROD = "/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so"
SO_42D = "/home/guile/llama-experiments/h2/cupti129-test/e2/h2lib/lib/libggml-cuda-42d9581.so"

# points : (mode, contexte cible, nb gen)
POINTS = [
    ("cold", 16384, 64),
    ("warm", 32768, 64),
    ("warm", 65536, 64),
]
MAX_CTX = 98304   # -c réduit (vs 204800) : KV q8 65k ≈ 4-5 GiB, poids 23 GiB -> VRAM OK
TIMEOUT_S = 900

# texte de remplissage : phrases de remplissage (pas du wikitext, c'est un A/B
# relatif — le contenu ne compte pas, la longueur du préfixe oui)
FILL_SENTENCE = ("La tour de contrôle surveille le passage des avions au crépuscule "
                 "pendant que le chef de piste organise les départs de la soirée. ")
FILL_WORDS = FILL_SENTENCE.count(" ")

_state = {}


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def vram_mib(gpu_idx):
    out = subprocess.run(["nvidia-smi", f"--id={gpu_idx}", "--query-gpu=memory.used",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return int(out) if out.isdigit() else -1


def http_json(url, data=None, timeout=10):
    req = urllib.request.Request(
        url, data=json.dumps(data).encode() if data is not None else None,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def wait_free_port(port, tries=60):
    for _ in range(tries):
        s = socket.socket()
        try:
            s.bind(("127.0.0.1", port))
            s.close()
            return True
        except OSError:
            s.close()
            time.sleep(2)
    return False


def unload_ollama():
    for _ in range(2):
        try:
            http_json(f"http://127.0.0.1:11434/api/chat",
                      {"model": "qwen3.8-27b-128k", "messages": [{"role": "user", "content": "hi"}],
                       "stream": False, "keep_alive": 0}, timeout=20)
        except Exception as e:
            log(f"  · unload ollama : {e}")
            break
        time.sleep(5)
    for _ in range(60):
        if vram_mib(_state["gpu_idx"]) < 4096:
            log(f"  · VRAM V100 libérée ({vram_mib(_state['gpu_idx'])} MiB)")
            return True
        time.sleep(5)
    log("  · ATTENTION VRAM pas totalement libérée, on continue quand même")
    return False


def build_env(so_path, gpu_cuda_idx):
    env = dict(os.environ)
    for d in CUDA_LIBS:
        if os.path.isdir(d) and d not in env.get("LD_LIBRARY_PATH", ""):
            env["LD_LIBRARY_PATH"] = d + ":" + env.get("LD_LIBRARY_PATH", "")
    env.update(GGML_ENV)
    env["GGML_BACKEND_PATH"] = so_path
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_cuda_idx)
    return env


def server_args():
    return [
        "--model", GGUF, "--port", str(PORT), "--host", "127.0.0.1",
        "--no-webui", "--offline",
        "-c", str(MAX_CTX), "-np", "1",
        "--log-verbosity", "4", "--no-log-prefix", "--no-log-timestamps",
        "--mmproj", MMPROJ,
        "--spec-type", "draft-mtp", "--spec-draft-n-max", "3",
        "--spec-draft-backend-sampling",
        "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
        "--flash-attn", "auto",
    ]


def start_server(so_path, logf):
    env = build_env(so_path, _state["gpu_cuda_idx"])
    lf = open(logf, "a")
    proc = subprocess.Popen([LLAMA_SERVER] + server_args(), stdout=lf,
                            stderr=subprocess.STDOUT, env=env)
    _state["proc"] = proc
    t0 = time.time()
    deadline = time.time() + 600
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server mort au démarrage (exit {proc.returncode})")
        try:
            h = http_json(f"http://127.0.0.1:{PORT}/health", timeout=3)
            break
        except Exception:
            time.sleep(2)
    else:
        raise RuntimeError("llama-server : pas de /health en 600 s")
    time.sleep(3)
    if vram_mib(_state["gpu_idx"]) < 1024:
        raise RuntimeError(f"ABORT: VRAM = {vram_mib(_state['gpu_idx'])} MiB — model en CPU")
    d = http_json(f"http://127.0.0.1:{PORT}/v1/models", timeout=10)
    _state["model_id"] = d["data"][0]["id"]
    log(f"  · server prêt ({time.time()-t0:.0f}s, VRAM {vram_mib(_state['gpu_idx'])} MiB, "
        f"model '{_state['model_id']}')")
    return proc


def stop_server():
    p = _state.get("proc")
    if p and p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait(timeout=10)
    time.sleep(5)


def prompt_words(n_words):
    reps = max(1, n_words // FILL_WORDS + 1)
    words = FILL_SENTENCE.split() * reps
    return " ".join(words[: n_words])


def run_point(mode, ctx_target, n_gen, logf):
    n_words = ctx_target * 10 // 13  # ~1.3 tokens/mot en FR
    text = prompt_words(n_words)
    est = len(text.split())
    data = {
        "model": _state["model_id"], "stream": False,
        "messages": [{"role": "user", "content": text + "\nRéponds en une phrase."}],
        "max_tokens": n_gen, "temperature": 0.0, "top_k": 1, "top_p": 1.0,
    }
    t0 = time.time()
    r = http_json(f"http://127.0.0.1:{PORT}/v1/chat/completions", data, timeout=TIMEOUT_S)
    wall = time.time() - t0
    u = (r or {}).get("usage", {})
    return {
        "mode": mode, "ctx_target": ctx_target, "prompt_words_sent": est,
        "prompt_tokens_api": u.get("prompt_tokens"), "gen_tokens_api": u.get("completion_tokens"),
        "wall_s": round(wall, 1),
    }


def parse_log_rates(logf, since_ts):
    """extrait du log llama-server : eval time + prompt eval time + draft acceptance.
    Renvoie la DERNIÈRE paire (prompt_eval, eval, acc) après since_ts."""
    res = None
    with open(logf, "r", errors="replace") as f:
        lines = f.readlines()
    cur_prompt = None
    for ln in lines:
        m = re.search(r"prompt eval time =\s+(\d+)\s+ms / (\d+) tokens", ln)
        if m:
            cur_prompt = (int(m.group(1)), int(m.group(2)))
            continue
        m = re.search(r"\beval time =\s+(\d+)\s+ms / (\d+) tokens", ln)
        if m:
            p_ms, p_n = cur_prompt if cur_prompt else (0, 0)
            res = {
                "prompt_ms": p_ms, "prompt_n": p_n,
                "prompt_tps": round(p_n / (p_ms / 1000.0), 1) if p_ms else 0,
                "eval_ms": int(m.group(1)), "eval_n": int(m.group(2)),
                "eval_tps": round(int(m.group(2)) / (int(m.group(1)) / 1000.0), 2) if int(m.group(1)) else 0,
            }
        m = re.search(r"n_matching = (\d+) .*?draft|acceptance|accepted (\d+)/(\d+)", ln)
        if m and res is not None:
            pass
    return res


def leg(name, so_path, logf):
    if not os.path.exists(so_path):
        raise RuntimeError(f".so introuvable : {so_path}")
    log(f"=== LEG {name} : {so_path}")
    start_server(so_path, logf)
    rows = []
    for mode, ctx, ng in POINTS:
        log(f"  · point {mode} {ctx} → {ng} gen…")
        t0 = time.time()
        api = run_point(mode, ctx, ng, logf)
        logres = parse_log_rates(logf, t0)
        row = {"leg": name, **api}
        if logres:
            row.update(logres)
            row["eval_tps"] = logres["eval_tps"]
        rows.append(row)
        log(f"    → eval {row.get('eval_tps', '?')} t/s "
            f"(gen {api['gen_tokens_api']} tok, wall {api['wall_s']} s)")
    stop_server()
    return rows


def main():
    import fcntl
    lf = open("/tmp/kolbo-supervisor-bench.lock", "w")
    try:
        fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit("LOCK tenu par un autre bench — abandon (pas de double bench)")
    log(f"AB mmvq — lock tenu")
    _state["gpu_idx"], _state["gpu_cuda_idx"] = v100_smi_index(), v100_cuda_ordinal()  # UUID (pas de détection par nom)
    log(f"GPU V100 : nvidia-smi idx {_state['gpu_idx']}, CUDA idx {_state['gpu_cuda_idx']}")
    if not wait_free_port(PORT):
        sys.exit("port 8917 encore occupé — abandon")
    unload_ollama()

    allrows = []
    open(OUT_SERVER_LOG, "w").close()
    for name, so in (("A-prod", SO_PROD), ("B-42d9581", SO_42D)):
        try:
            rows = leg(name, so, OUT_SERVER_LOG)
        except Exception as e:
            log(f"  · LEG {name} ÉCHEC : {e}")
            stop_server()
            continue
        allrows.extend(rows)
        # laisser la VRAM retomber
        time.sleep(20)
        vram = vram_mib(_state["gpu_idx"])
        log(f"  · VRAM après {name} : {vram} MiB")
        if vram > 8192:
            log("  · VRAM pas libérée après stop server — on attend")
            for _ in range(30):
                time.sleep(5)
                if vram_mib(_state["gpu_idx"]) < 8192:
                    break

    with open(OUT_JSONL, "a") as f:
        for r in allrows:
            f.write(json.dumps(r) + "\n")

    # synthèse
    lines = ["# AB H2 vs 42d9581 (pick nwarps TURING sm70)", "",
             f"date : {time.strftime('%Y-%m-%d %H:%M UTC')}", "",
             "| point | A prod t/s | B 42d9581 t/s | B/A |",
             "|---|---|---|---|"]
    for mode, ctx, ng in POINTS:
        a = [r for r in allrows if r["leg"] == "A-prod" and r["mode"] == mode and r["ctx_target"] == ctx]
        b = [r for r in allrows if r["leg"] == "B-42d9581" and r["mode"] == mode and r["ctx_target"] == ctx]
        ta = a[0].get("eval_tps") if a else None
        tb = b[0].get("eval_tps") if b else None
        ratio = f"{tb/ta:+.1%}" if (ta and tb) else "—"
        lines.append(f"| {mode} {ctx} | {ta} | {tb} | {ratio} |")
    with open(OUT_SUMMARY, "w") as f:
        f.write("\n".join(lines) + "\n")
    log(f"FIN — synthèse : {OUT_SUMMARY}")
    lf.close()


if __name__ == "__main__":
    main()
