#!/usr/bin/env python3
"""
ab_lba.py — A/B APPLICATIF .so : candidat lba (T3v2, launch_bounds par NT) vs prod d2df (T3).

Question : le gain kernel mesuré en microbench (T=1 FA -17%) se traduit-il en gain applicatif
sur le DECODE ? MTP OFF = single-token decode = T=1, exactement ce que lba améliore ; c'est
aussi la config prescrite pour mesurer T3 proprement.

Deux legs, même batch (-b 2048, batch prod), seul le .so change (GGML_BACKEND_PATH) :
  A-lba  : libggml-cuda-lba.so  (candidat)
  B-prod : libggml-cuda.so      (prod d2df07b8)
Ordre alterné, serveur frais par leg, port 8917, CVD=0, KV q8_0, FA auto, H2 OFF, MTP OFF.
Contextes tous <= 128k (ctx prod actuel), decode-dominants.

USAGE :
  cd /home/guile/llama-experiments/bench
  python3 supervisor_bench.py run ab_lba.py --detach
"""
import hashlib, json, os, re, signal, socket, subprocess, sys, time, urllib.request
from gpu_uuid_select import v100_cuda_ordinal, v100_smi_index  # UUID GPU (jamais d'index en dur — voir GPU_UUID_NSYS_CUPTI_20261030.md)

D = os.path.dirname(os.path.abspath(__file__))
LLAMA_SERVER = "/home/guile/llama-experiments/ollama-v100/lib/ollama/llama-server"
GGUF   = ("/usr/share/ollama/.ollama/models/blobs/"
          "sha256-493301830a596b8ad56dc1329f80bbcb578c8e910da395feafdc9cd8263430bb")
MMPROJ = ("/usr/share/ollama/.ollama/models/blobs/"
          "sha256-ac3714bfdddeca31351f2752bf1a63f266f4df87c0b68c895e44945ca704448e")
PORT = 8917
SERVICE = "ollama-v100"
OUT_JSONL = os.path.join(D, "ab_lba_results.jsonl")
OUT_SUMMARY = os.path.join(D, "ab_lba_summary.md")
OUT_SERVER_LOG = os.path.join(D, "ab_lba_server.log")
CUDA_LIBS = [
    "/home/guile/llama-experiments/ollama-v100/lib/ollama",
    "/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12",
    "/home/guile/llama-experiments/tools/cuda-12.6/usr/local/cuda-12.6/lib64",
    "/home/guile/llama-experiments/tools/cuda-12.6/usr/local/cuda-12.6/targets/x86_64-linux/lib",
]
SO_PROD = "/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so"
SO_LBA  = "/home/guile/llama-experiments/h2/cupti129-test/e2/h2lib/lib/libggml-cuda-lba.so"
# prod MTP-OFF : Q6_W4R4=1, H2 OFF (FATTN_TC=0), T3 ON (défaut)
ENV_BASE = {
    "GGML_CUDA_VOLTA_Q6_W4R4": "1",
    "GGML_CUDA_VOLTA_Q8_FATTN_TC": "0",
    "LLAMA_ARG_FIT_TARGET": "1024",
}
LEGS = [
    ("A-lba",  dict(ENV_BASE, _B="2048", _SO=SO_LBA)),
    ("B-prod", dict(ENV_BASE, _B="2048", _SO=SO_PROD)),
]
POINTS = [(32768, 64), (65536, 64), (98304, 64)]
REPS = 2
TIMEOUT_S = 900
GPU_SMI_IDX = v100_smi_index()  # résolu par UUID (ne pas hardcoder)
GPU_CUDA_IDX = v100_cuda_ordinal()  # résolu par UUID (ne pas hardcoder)
FILL = ("La tour de contrôle surveille le passage des avions au crépuscule "
        "pendant que le chef de piste organise les départs de la soirée. ")

_state = {}
_restored = False


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)

def run(cmd, timeout=60):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

def svc_state():
    return run(["systemctl", "is-active", SERVICE]).stdout.strip()

def vram_mib():
    r = run(["nvidia-smi", f"--id={GPU_SMI_IDX}",
             "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    v = r.stdout.strip()
    return int(v) if v.isdigit() else -1

def v100_proc_gone():
    r = run(["nvidia-smi", f"--id={GPU_SMI_IDX}",
             "--query-compute-apps=pid,used_memory", "--format=csv,noheader"])
    for line in r.stdout.splitlines():
        if line.strip():
            parts = line.split(",")
            if len(parts) >= 2 and parts[1].strip().replace("MiB","").replace(" ","").isdigit():
                if int(parts[1].strip().split()[0]) > 8000:
                    return False
    return True

def restore_service():
    global _restored
    try:
        if svc_state() != "active":
            log(f"  · restauration : systemctl start {SERVICE}")
            run(["sudo", "systemctl", "start", SERVICE], timeout=120)
        for _ in range(90):
            if svc_state() == "active":
                break
            time.sleep(2)
        _restored = svc_state() == "active"
        log(f"  · service {SERVICE} = {svc_state()}")
    except Exception as e:
        log(f"  · ERREUR restauration : {e}")
        _restored = False

def on_sig(sig, frm):
    restore_service()
    sys.exit(128 + sig)

def stop_and_free():
    log(f"  · stop {SERVICE}")
    run(["sudo", "systemctl", "stop", SERVICE], timeout=120)
    t0 = time.time()
    while time.time() - t0 < 180:
        if vram_mib() < 1024 and v100_proc_gone():
            log(f"  · V100 libérée ({vram_mib()} MiB) en {time.time()-t0:.0f}s")
            return True
        time.sleep(3)
    log(f"  · ATTENTION V100 pas totalement libérée (VRAM={vram_mib()} MiB)")
    return False

def build_env(extra):
    env = dict(os.environ)
    for d in CUDA_LIBS:
        if os.path.isdir(d) and d not in env.get("LD_LIBRARY_PATH", ""):
            env["LD_LIBRARY_PATH"] = d + ":" + env.get("LD_LIBRARY_PATH", "")
    env.update(extra)
    env["GGML_BACKEND_PATH"] = extra.get("_SO", SO_PROD)   # <-- .so par leg
    env["CUDA_VISIBLE_DEVICES"] = str(GPU_CUDA_IDX)
    return env

def server_args(batch):
    return [
        "--model", GGUF, "--port", str(PORT), "--host", "127.0.0.1",
        "--no-webui", "--offline",
        "-c", str(128000), "-np", "1",
        "-b", batch, "-ub", batch,
        "--log-verbosity", "4", "--no-log-prefix", "--no-log-timestamps",
        "--mmproj", MMPROJ,
        # MTP OFF : pas de --spec-type
        "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
        "--flash-attn", "auto",
    ]

def port_free(port):
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", port)); s.close(); return True
    except OSError:
        s.close(); return False

def http_json(url, data=None, timeout=10):
    req = urllib.request.Request(
        url, data=json.dumps(data).encode() if data is not None else None,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def start_server(extra):
    batch = extra.get("_B", "512")
    env = {k: v for k, v in extra.items() if k != "_B"}
    if vram_mib() > 8192:
        raise RuntimeError(f"V100 pas vide (VRAM={vram_mib()} MiB) — refus de démarrer")
    env = build_env(env)
    lf = open(OUT_SERVER_LOG, "a")
    proc = subprocess.Popen([LLAMA_SERVER] + server_args(batch), stdout=lf,
                            stderr=subprocess.STDOUT, env=env)
    _state["proc"] = proc
    t0 = time.time()
    while time.time() - t0 < 600:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server mort au démarrage (exit {proc.returncode})")
        try:
            http_json(f"http://127.0.0.1:{PORT}/health", timeout=3); break
        except Exception:
            time.sleep(2)
    else:
        raise RuntimeError("llama-server : pas de /health en 600 s")
    time.sleep(3)
    if vram_mib() < 1024:
        raise RuntimeError(f"ABORT : VRAM={vram_mib()} MiB — model en CPU")
    d = http_json(f"http://127.0.0.1:{PORT}/v1/models", timeout=10)
    _state["model_id"] = d["data"][0]["id"]
    log(f"  · server prêt ({time.time()-t0:.0f}s, VRAM {vram_mib()} MiB) -b {batch} so={os.path.basename(extra.get('_SO',''))}")
    return proc

def stop_server():
    p = _state.get("proc")
    if p and p.poll() is None:
        p.terminate()
        try: p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill(); p.wait(timeout=10)
    time.sleep(8)
    for _ in range(60):
        if vram_mib() < 4096: break
        time.sleep(3)
    log(f"  · server stop, VRAM {vram_mib()} MiB")

def prompt_text(ctx_target_tokens):
    n_words = ctx_target_tokens * 4 // 5
    words = FILL.split() * max(1, n_words // FILL.count(" ") + 1)
    return " ".join(words[: n_words])

def parse_point(logf, since_pos):
    with open(logf, "r", errors="replace") as f:
        data = f.read()[since_pos:]
    out = {}
    m = re.search(r"prompt eval time =\s+([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", data)
    if m:
        ms, n = float(m.group(1)), int(m.group(2))
        out["prompt_tps"] = round(n / (ms / 1000.0), 1) if ms else 0
        out["prompt_tokens"] = n
    m = re.search(r"(?<!prompt )eval time =\s+([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", data)
    if m:
        ms, n = float(m.group(1)), int(m.group(2))
        out["eval_tps"] = round(n / (ms / 1000.0), 2) if ms else 0
        out["eval_tokens"] = n
        out["eval_ms"] = round(ms, 1)
    return out

def run_point(ctx_target, n_gen, since_pos):
    text = prompt_text(ctx_target)
    data = {
        "model": _state["model_id"], "stream": False,
        "messages": [{"role": "user", "content": text + "\nRéponds en une phrase."}],
        "max_tokens": n_gen, "temperature": 0.0, "top_k": 1, "top_p": 1.0,
    }
    t0 = time.time()
    try:
        r = http_json(f"http://127.0.0.1:{PORT}/v1/chat/completions", data, timeout=TIMEOUT_S)
        u = (r or {}).get("usage", {})
        api = {"prompt_tokens_api": u.get("prompt_tokens"),
               "gen_tokens_api": u.get("completion_tokens"),
               "wall_s": round(time.time() - t0, 1)}
    except Exception as e:
        api = {"error": str(e)}
    return api, parse_point(OUT_SERVER_LOG, since_pos)

def log_size():
    try: return os.path.getsize(OUT_SERVER_LOG)
    except OSError: return 0

def med(vals):
    vals = sorted(v for v in vals if v)
    return vals[len(vals) // 2] if vals else None

def so_prefix(p):
    try:
        return hashlib.md5(open(p, "rb").read()).hexdigest()[:8]
    except OSError:
        return "?"

def write_summary(allrows):
    lines = ["# A/B applicatif — lba (T3v2 launch_bounds/NT) vs prod d2df (T3), MTP OFF", "",
             f"date : {time.strftime('%Y-%m-%d %H:%M UTC')}",
             f"service restauré : {'OUI' if _restored else 'NON'}",
             f".so prod md5 : {so_prefix(SO_PROD)}  ·  .so lba md5 : {so_prefix(SO_LBA)}",
             "",
             "A-lba = candidat (GGML_BACKEND_PATH=lba) · B-prod = prod d2df. Même -b 2048, MTP OFF.",
             "Δ = (A-lba - B-prod)/B-prod : positif = lba plus rapide.",
             "",
             "## DECODE (t/s) par leg (single-token = T=1, cas amélioré par lba)",
             "| contexte | rep | A-lba | B-prod | Δ |",
             "|---|---|---|---|---|"]
    for ctx, ng in POINTS:
        for rep in range(1, REPS + 1):
            a = next((r for r in allrows if r["ctx"] == ctx and r["leg"] == "A-lba" and r.get("rep") == rep), None)
            b = next((r for r in allrows if r["ctx"] == ctx and r["leg"] == "B-prod" and r.get("rep") == rep), None)
            if a and b and a.get("eval_tps") and b.get("eval_tps"):
                d = (a["eval_tps"] - b["eval_tps"]) / b["eval_tps"] * 100
                lines.append(f"| {ctx} | {rep} | {a['eval_tps']} | {b['eval_tps']} | {d:+.1f}% |")
    lines += ["", "## DECODE — médianes par leg",
              "| contexte | A-lba | B-prod | Δ |",
              "|---|---|---|---|"]
    for ctx, ng in POINTS:
        a = med([r.get("eval_tps") for r in allrows if r["ctx"] == ctx and r["leg"] == "A-lba"])
        b = med([r.get("eval_tps") for r in allrows if r["ctx"] == ctx and r["leg"] == "B-prod"])
        if a and b:
            lines.append(f"| {ctx} | {a} | {b} | {(a-b)/b*100:+.1f}% |")
    lines += ["", "## WALL (s par requête, prompt+64 gen)",
              "| contexte | A-lba | B-prod | Δ s |",
              "|---|---|---|---|"]
    for ctx, ng in POINTS:
        a = med([r.get("wall_s") for r in allrows if r["ctx"] == ctx and r["leg"] == "A-lba"])
        b = med([r.get("wall_s") for r in allrows if r["ctx"] == ctx and r["leg"] == "B-prod"])
        if a and b:
            lines.append(f"| {ctx} | {a} | {b} | {a-b:+.1f} |")
    lines.append("")
    with open(OUT_SUMMARY, "w") as f:
        f.write("\n".join(lines) + "\n")

def main():
    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)
    log(f"A/B applicatif lba vs prod — prod md5 {so_prefix(SO_PROD)}, lba md5 {so_prefix(SO_LBA)}")
    if so_prefix(SO_PROD) != "d2df07b8":
        sys.exit(f"ABORT : .so prod {so_prefix(SO_PROD)} != d2df07b8 — prod dérivé, ne pas bench")
    if not os.path.isfile(SO_LBA):
        sys.exit(f"ABORT : .so lba manquant : {SO_LBA}")
    if not port_free(PORT):
        sys.exit(f"port {PORT} occupé — abandon")
    stop_and_free()
    open(OUT_SERVER_LOG, "w").close()
    allrows = []
    try:
        for ctx, ng in POINTS:
            for rep in range(1, REPS + 1):
                order = list(LEGS) if rep % 2 == 1 else list(reversed(LEGS))
                for name, extra in order:
                    try:
                        log(f"=== {ctx} rep{rep} :: LEG {name} -b {extra['_B']}")
                        start_server(extra)
                        since = log_size()
                        api, met = run_point(ctx, ng, since)
                        row = {"leg": name, "ctx": ctx, "rep": rep,
                               "mtp": "off", "so": so_prefix(extra.get("_SO", SO_PROD)), **api, **met}
                        allrows.append(row)
                        log(f"    → decode {met.get('eval_tps','?')} t/s | wall {api.get('wall_s','?')}s | {api.get('error','ok')}")
                    except Exception as e:
                        log(f"  · {ctx} {name} rep{rep} ÉCHEC : {e}")
                    finally:
                        stop_server()
                        time.sleep(12)
    finally:
        restore_service()   # restauration d'ABORD (piège 17/10)
        with open(OUT_JSONL, "a") as f:
            for r in allrows:
                f.write(json.dumps(r) + "\n")
        write_summary(allrows)
        log(f"FIN — {len(allrows)} lignes, synthèse : {OUT_SUMMARY}, service restauré : {svc_state()}")
    sys.exit(0 if _restored else 3)

if __name__ == "__main__":
    main()
