#!/usr/bin/env python3
"""
ab_ub.py — Sweep -b/-ub (512 / 1024 / 2048 / 4096) : prefill, decode, et COÛT VRAM.

Question (user 27/10) : réévaluer le gain de -b/-ub 2048 par rapport à ce que ça coûte
en VRAM, avec les points 512 / 1024 / (2048) / 4096.

Un seul variable : le batch. .so = la prod actuelle (4fd152f9, lecture seule via
GGML_BACKEND_PATH), MTP OFF, -c 128000, KV q8_0, FA auto, H2 OFF, port 8917, serveur
frais par leg (les réservations compute buffer / KV buffer dépendent de -b/-ub et ne se
voient qu'au startup).

Mesures par leg :
  - prefill : prompt eval t/s (+ secondes)
  - decode  : eval t/s (64 tok)
  - VRAM    : nvidia-smi au moment où le serveur est prêt (modèle + KV + compute + ctx)
  - logs    : n_batch, n_ubatch, KV buffer size, CUDA0 compute buffer size, graph splits

Points : 32768 / 65536 / 98304 ctx, 64 tok gen, 2 reps (ordre b rotative par rep).
24 legs ≈ 90-110 min. Si un -b ne passe pas le fit (ex. 4096 trop gros), le leg échoue
et c'est une info : le coût du batch ne rentre plus dans la V100 32 GB.

USAGE :
  cd /home/guile/llama-experiments/bench
  python3 supervisor_bench.py run ab_ub.py --detach
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
OUT_JSONL = os.path.join(D, "ab_ub_results.jsonl")
OUT_SUMMARY = os.path.join(D, "ab_ub_summary.md")
OUT_SERVER_LOG = os.path.join(D, "ab_ub_server.log")
CUDA_LIBS = [
    "/home/guile/llama-experiments/ollama-v100/lib/ollama",
    "/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12",
    "/home/guile/llama-experiments/tools/cuda-12.6/usr/local/cuda-12.6/lib64",
    "/home/guile/llama-experiments/tools/cuda-12.6/usr/local/cuda-12.6/targets/x86_64-linux/lib",
]
SO_PROD = "/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so"  # 4fd152f9 (lecture seule)
# prod MTP-OFF : Q6_W4R4=1, H2 OFF (FATTN_TC=0), T3 ON (defaut)
ENV_BASE = {
    "GGML_CUDA_VOLTA_Q6_W4R4": "1",
    "GGML_CUDA_VOLTA_Q8_FATTN_TC": "0",
    "LLAMA_ARG_FIT_TARGET": "1024",
}
BATCHES = [512, 1024, 2048, 4096]
POINTS = [(32768, 64), (65536, 64), (98304, 64)]
REPS = 2
TIMEOUT_S = 1500
GPU_SMI_IDX = v100_smi_index()  # résolu par UUID (ne pas hardcoder)
GPU_CUDA_IDX = v100_cuda_ordinal()  # résolu par UUID (ne pas hardcoder)
FILL = ("La tour de controle surveille le passage des avions au crepuscule "
        "pendant que le chef de piste organise le depart de la soiree. ")

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
            if len(parts) >= 2 and parts[1].strip().replace("MiB", "").replace(" ", "").isdigit():
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
            log(f"  · V100 liberee ({vram_mib()} MiB) en {time.time()-t0:.0f}s")
            return True
        time.sleep(3)
    log(f"  · ATTENTION V100 pas totalement liberee (VRAM={vram_mib()} MiB)")
    return False
def build_env(batch):
    env = dict(os.environ)
    for d in CUDA_LIBS:
        if os.path.isdir(d) and d not in env.get("LD_LIBRARY_PATH", ""):
            env["LD_LIBRARY_PATH"] = d + ":" + env.get("LD_LIBRARY_PATH", "")
    env.update(ENV_BASE)
    env["GGML_BACKEND_PATH"] = SO_PROD
    env["CUDA_VISIBLE_DEVICES"] = str(GPU_CUDA_IDX)
    return env
def server_args(batch):
    return [
        "--model", GGUF, "--port", str(PORT), "--host", "127.0.0.1",
        "--no-webui", "--offline",
        "-c", "128000", "-np", "1",
        "-b", str(batch), "-ub", str(batch),
        "--log-verbosity", "4", "--no-log-prefix", "--no-log-timestamps",
        "--mmproj", MMPROJ,
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
def start_server(batch):
    if vram_mib() > 8192:
        raise RuntimeError(f"V100 pas vide (VRAM={vram_mib()} MiB) — refus de demarrer")
    env = build_env(batch)
    lf = open(OUT_SERVER_LOG, "a")
    proc = subprocess.Popen([LLAMA_SERVER] + server_args(batch), stdout=lf,
                            stderr=subprocess.STDOUT, env=env)
    _state["proc"] = proc
    t0 = time.time()
    while time.time() - t0 < 600:
        if proc.poll() is not None:
            tail = open(OUT_SERVER_LOG, "r", errors="replace").read()[-1500:]
            raise RuntimeError(f"llama-server mort au demarrage (exit {proc.returncode}) : ...{tail}")
        try:
            http_json(f"http://127.0.0.1:{PORT}/health", timeout=3); break
        except Exception:
            time.sleep(2)
    else:
        raise RuntimeError("llama-server : pas de /health en 600 s")
    time.sleep(3)
    if vram_mib() < 1024:
        raise RuntimeError(f"ABORT leg : VRAM={vram_mib()} MiB — model en CPU")
    d = http_json(f"http://127.0.0.1:{PORT}/v1/models", timeout=10)
    _state["model_id"] = d["data"][0]["id"]
    vram = vram_mib()
    log(f"  · server pret ({time.time()-t0:.0f}s, VRAM {vram} MiB) -b/-ub {batch}")
    return vram
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
def parse_leg(data):
    out = {}
    mb = [int(m) for m in re.findall(r"llama_context: n_batch\s*=\s*(\d+)", data)]
    mu = [int(m) for m in re.findall(r"llama_context: n_ubatch\s*=\s*(\d+)", data)]
    if mb: out["n_batch_log"] = mb[-1]
    if mu: out["n_ubatch_log"] = mu[-1]
    kv = [float(m) for m in re.findall(r"KV buffer size =\s*([\d.]+)\s*MiB", data)]
    if kv: out["kv_buf_mib"] = max(kv)
    cb = [float(m) for m in re.findall(r"CUDA0 compute buffer size =\s*([\d.]+)\s*MiB", data)]
    if cb: out["compute_buf_mib"] = max(cb)
    gs = [int(m) for m in re.findall(r"graph splits = (\d+)", data)]
    if gs: out["graph_splits"] = max(gs)
    m = re.search(r"prompt eval time =\s+([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", data)
    if m:
        ms, n = float(m.group(1)), int(m.group(2))
        out["prompt_tps"] = round(n / (ms / 1000.0), 1) if ms else 0
        out["prompt_s"] = round(ms / 1000.0, 1)
        out["prompt_tokens"] = n
    m = re.search(r"(?<!prompt )eval time =\s+([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", data)
    if m:
        ms, n = float(m.group(1)), int(m.group(2))
        out["eval_tps"] = round(n / (ms / 1000.0), 2) if ms else 0
        out["eval_tokens"] = n
    return out
def run_point(ctx_target, n_gen):
    text = prompt_text(ctx_target)
    data = {
        "model": _state["model_id"], "stream": False,
        "messages": [{"role": "user", "content": text + "\nReponds en une phrase."}],
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
    return api
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
def write_summary(rows):
    L = ["# Sweep -b/-ub (512/1024/2048/4096) — prefill, decode, coût VRAM — MTP OFF", "",
         f"date : {time.strftime('%Y-%m-%d %H:%M UTC')}",
         f"service restaure : {'OUI' if _restored else 'NON'}",
         f".so (prod, lecture seule) md5 : {so_prefix(SO_PROD)}",
         "protocole : serveur frais par leg, -c 128000, KV q8_0, FA auto, H2 OFF, MTP OFF, port 8917, "
         "2 reps × 3 points (32k/65k/98k), 64 tok gen. VRAM = au moment où le serveur est prêt "
         "(inclut mmproj ~1.2 GB, même dans tous les legs → les deltas sont valides).", ""]
    for ctx, ng in POINTS:
        L.append(f"## ctx {ctx} tokens")
        L.append("")
        L.append("| -b/-ub | prefill t/s | prefill s | decode t/s | ms/tok | VRAM prête (MiB) | KV buf (MiB) | compute buf (MiB) | graph splits |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for b in BATCHES:
            sub = [r for r in rows if r["ctx"] == ctx and r["batch"] == b]
            if not sub:
                L.append(f"| {b} | — | — | — | — | — | — | — | — |")
                continue
            pt = med([r.get("prompt_tps") for r in sub])
            ps = med([r.get("prompt_s") for r in sub])
            et = med([r.get("eval_tps") for r in sub])
            ms = round(1000.0 / et, 2) if et else None
            vr = med([r.get("vram_ready_mib") for r in sub])
            kvb = max([r.get("kv_buf_mib", 0) for r in sub]) or None
            cbb = max([r.get("compute_buf_mib", 0) for r in sub]) or None
            gs  = max([r.get("graph_splits", 0) for r in sub]) or None
            L.append(f"| {b} | {pt} | {ps} | {et} | {ms} | {vr} | {kvb} | {cbb} | {gs} |")
        L.append("")
        # deltas vs 2048
        L.append("Deltas vs -b/-ub 2048 (médianes) :")
        L.append("")
        L.append("| -b/-ub | Δ prefill | Δ decode | Δ VRAM (MiB) |")
        L.append("|---|---|---|---|")
        for b in BATCHES:
            if b == 2048:
                L.append("| 2048 (prod) | — | — | — |")
                continue
            sub = [r for r in rows if r["ctx"] == ctx and r["batch"] == b]
            sub2048 = [r for r in rows if r["ctx"] == ctx and r["batch"] == 2048]
            if not sub or not sub2048:
                L.append(f"| {b} | — | — | — |")
                continue
            p_b  = med([r.get("prompt_tps") for r in sub])
            p_20 = med([r.get("prompt_tps") for r in sub2048])
            e_b  = med([r.get("eval_tps") for r in sub])
            e_20 = med([r.get("eval_tps") for r in sub2048])
            v_b  = med([r.get("vram_ready_mib") for r in sub])
            v_20 = med([r.get("vram_ready_mib") for r in sub2048])
            dp = f"{(p_b - p_20) / p_20 * 100:+.1f}%" if p_b and p_20 else "—"
            de = f"{(e_b - e_20) / e_20 * 100:+.1f}%" if e_b and e_20 else "—"
            dv = f"{v_b - v_20:+d}" if v_b and v_20 else "—"
            L.append(f"| {b} | {dp} | {de} | {dv} |")
        L.append("")
    L.append("## Détail par leg (stabilité reps)")
    L.append("")
    L.append("| ctx | rep | -b | prefill t/s | decode t/s | VRAM | kv_buf | comp_buf | splits | err |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        L.append(f"| {r['ctx']} | {r['rep']} | {r['batch']} | {r.get('prompt_tps','-')} | "
                 f"{r.get('eval_tps','-')} | {r.get('vram_ready_mib','-')} | "
                 f"{r.get('kv_buf_mib','-')} | {r.get('compute_buf_mib','-')} | "
                 f"{r.get('graph_splits','-')} | {r.get('error','ok')[:40]} |")
    L.append("")
    with open(OUT_SUMMARY, "w") as f:
        f.write("\n".join(L) + "\n")

def main():
    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)
    log(f"SWEEP -b/-ub {BATCHES} — prod md5 {so_prefix(SO_PROD)} (lecture seule), port {PORT}")
    if not port_free(PORT):
        sys.exit(f"port {PORT} occupe — abandon")
    stop_and_free()
    open(OUT_SERVER_LOG, "w").close()
    rows = []
    try:
        for ctx, ng in POINTS:
            for rep in range(1, REPS + 1):
                order = BATCHES[(rep - 1) * 2 % len(BATCHES):] + BATCHES[: (rep - 1) * 2 % len(BATCHES)]
                for b in order:
                    since = log_size()
                    try:
                        log(f"=== {ctx} rep{rep} :: LEG -b/-ub {b}")
                        vram = start_server(b)
                        api = run_point(ctx, ng)
                        with open(OUT_SERVER_LOG, "r", errors="replace") as f:
                            met = parse_leg(f.read()[since:])
                        row = {"batch": b, "ctx": ctx, "rep": rep, "mtp": "off",
                               "vram_ready_mib": vram, **api, **met}
                        rows.append(row)
                        log(f"    → prefill {met.get('prompt_tps','?')} t/s | "
                            f"decode {met.get('eval_tps','?')} t/s | VRAM {vram} MiB | "
                            f"kv {met.get('kv_buf_mib','?')} comp {met.get('compute_buf_mib','?')} | "
                            f"{api.get('error','ok')}")
                    except Exception as e:
                        log(f"  · {ctx} -b{b} rep{rep} ECHEC : {e}")
                        rows.append({"batch": b, "ctx": ctx, "rep": rep, "mtp": "off",
                                     "error": str(e)[:200]})
                    finally:
                        stop_server()
                        time.sleep(12)
    finally:
        restore_service()   # restauration d'ABORD (piege 17/10)
        with open(OUT_JSONL, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        write_summary(rows)
        ok = all(r.get("eval_tps") for r in rows if r.get("batch") is not None)
        log(f"FIN — {len(rows)} legs, synthese : {OUT_SUMMARY}, service restaure : {svc_state()}, "
            f"tous decode OK : {ok}")
    sys.exit(0 if _restored else 3)

if __name__ == "__main__":
    main()
