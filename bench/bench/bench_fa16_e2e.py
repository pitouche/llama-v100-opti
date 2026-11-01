#!/usr/bin/env python3
"""
FA16 E2E — A/B/C prefill réel via llama-server standalone (service stoppé, restauration assurée).

Legs (KV q8_0, -b 2048 -ub 2048, MTP OFF, env stricte prod) :
  A-prod     : .so PROD (8035caf4), sans env FA        → 32x2 standard (baseline)
  B-16x2     : .so DUAL (d4954502) + FA_NCOLS1=16      → tile 16x2 forcé (T>=128)
  C-compact  : .so DUAL + FA_COMPACT_TMAX=8192         → kernel compact 48 KB (32x2) pour T<8192

Points (prefill, n_gen=64) : 16k cold, 32k/65k/98k/114k/131k warm.
Interleave par point : 3 serveurs frais cote à cote dans la même fenêtre thermique.

USAGE :
  cd /home/guile/llama-experiments/bench
  python3 supervisor_bench.py run bench_fa16_e2e.py --detach
"""
import json, os, re, signal, socket, subprocess, sys, time
import urllib.request
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpu_uuid_select import v100_cuda_ordinal, v100_smi_index

D = os.path.dirname(os.path.abspath(__file__))

LLAMA_SERVER = "/home/guile/llama-experiments/ollama-v100/lib/ollama/llama-server"
GGUF = ("/usr/share/ollama/.ollama/models/blobs/"
        "sha256-493301830a596b8ad56dc1329f80bbcb578c8e910da395feafdc9cd8263430bb")
MMPROJ = ("/usr/share/ollama/.ollama/models/blobs/"
          "sha256-ac3714bfdddeca31351f2752bf1a63f266f4df87c0b68c895e44945ca704448e")

SO_PROD = "/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so"
SO_DUAL = "/home/guile/llama-experiments/v100/dual-arch/lib/libggml-cuda-dual.so"

PORT = 8917
SERVICE = "ollama-v100"
TS = datetime.now().strftime("%Y%m%d-%H%M%S")
OUT_JSONL = os.path.join(D, f"fa16_e2e_{TS}.jsonl")
OUT_SUMMARY = os.path.join(D, f"fa16_e2e_{TS}_summary.md")
OUT_SERVER_LOG = os.path.join(D, f"fa16_e2e_{TS}_server.log")
OUT_LOG = os.path.join(D, f"fa16_e2e_{TS}.log")

CUDA_LIBS = [
    "/home/guile/llama-experiments/ollama-v100/lib/ollama",
    "/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12",
    "/usr/local/cuda-12.9/lib64",
]

# env stricte prod (hors MTP) + Q8_FATTN_TC=0 = valeur prod réelle (le prefill T=2048
# passe alors par mma_f16, cf. nsys prod : kernel 32x2, 49.2 % du temps GPU).
ENV_BASE = {
    "GGML_CUDA_VOLTA_Q6_W4R4": "1",
    "GGML_CUDA_VOLTA_Q8_FATTN_TC": "0",
    "LLAMA_ARG_FIT_TARGET": "1024",
}
LEGS = [
    ("A-prod",    {"_SO": SO_PROD}),
    ("B-16x2",    {"_SO": SO_DUAL, "GGML_CUDA_VOLTA_FA_NCOLS1": "16"}),
    ("C-compact", {"_SO": SO_DUAL, "GGML_CUDA_VOLTA_FA_COMPACT_TMAX": "8192"}),
]
POINTS = [
    ("cold", 16384, 64),
    ("warm", 32768, 64),
    ("warm", 65536, 64),
    ("warm", 98304, 64),
    ("warm", 114688, 64),   # règne réel prod (n_tokens ~114k)
    ("warm", 131072, 64),   # ~plafond prefill (ctx 128k)
]
MAX_CTX = 196672
TIMEOUT_S = 3600
GPU_SMI_IDX = v100_smi_index()
GPU_CUDA_IDX = v100_cuda_ordinal()
FILL = ("La tour de contrôle surveille le passage des avions au crépuscule "
        "pendant que le chef de piste organise les départs de la soirée. ")

_state = {}
_restored = False


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(OUT_LOG, "a") as f:
        f.write(line + "\n")


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
            log(f"  · V100 libérée ({vram_mib()} MiB) en {time.time()-t0:.0f}s")
            return True
        time.sleep(3)
    log(f"  · ATTENTION V100 pas totalement libérée (VRAM={vram_mib()} MiB)")
    return False


def build_env(extra):
    env = dict(os.environ)
    for ddir in CUDA_LIBS:
        if os.path.isdir(ddir) and ddir not in env.get("LD_LIBRARY_PATH", ""):
            env["LD_LIBRARY_PATH"] = ddir + ":" + env.get("LD_LIBRARY_PATH", "")
    env.update(extra)
    env["GGML_BACKEND_PATH"] = extra.pop("_SO", SO_PROD)
    env["CUDA_VISIBLE_DEVICES"] = str(GPU_CUDA_IDX)
    return env


def server_args():
    return [
        "--model", GGUF, "--port", str(PORT), "--host", "127.0.0.1",
        "--no-webui", "--offline",
        "-c", str(MAX_CTX), "-np", "1",
        "-b", "2048", "-ub", "2048",
        "--log-verbosity", "4", "--no-log-prefix", "--no-log-timestamps",
        "--mmproj", MMPROJ,
        # MTP OFF : pas de --spec-type (bruit de mesure)
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
    so = extra.get("_SO", SO_PROD)
    env = {k: v for k, v in extra.items() if k != "_SO"}
    if vram_mib() > 8192:
        raise RuntimeError(f"V100 pas vide (VRAM={vram_mib()} MiB) — refus de démarrer")
    env = build_env(dict(env, _SO=so))
    lf = open(OUT_SERVER_LOG, "a")
    proc = subprocess.Popen([LLAMA_SERVER] + server_args(), stdout=lf,
                            stderr=subprocess.STDOUT, env=env)
    _state["proc"] = proc
    t0 = time.time()
    while time.time() - t0 < 600:
        if proc.poll() is not None:
            lf.close()
            raise RuntimeError(f"llama-server mort au démarrage (exit {proc.returncode})")
        try:
            http_json(f"http://127.0.0.1:{PORT}/health", timeout=3); break
        except Exception:
            time.sleep(2)
    else:
        lf.close()
        raise RuntimeError("llama-server : pas de /health en 600 s")
    time.sleep(3)
    if vram_mib() < 1024:
        lf.close()
        raise RuntimeError(f"ABORT : VRAM={vram_mib()} MiB — model en CPU")
    d = http_json(f"http://127.0.0.1:{PORT}/v1/models", timeout=10)
    _state["model_id"] = d["data"][0]["id"]
    lf.close()
    log(f"  · server prêt ({time.time()-t0:.0f}s, VRAM {vram_mib()} MiB) {os.path.basename(so)}")
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
    try:
        with open(logf, "r", errors="replace") as f:
            data = f.read()[since_pos:]
    except OSError:
        return {}
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


def run_point(mode, ctx_target, n_gen, since_pos):
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


def main():
    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)
    open(OUT_LOG, "a").close()
    log(f"FA16 E2E — A-prod / B-16x2 / C-compact — {TS}")
    if not port_free(PORT):
        sys.exit(f"port {PORT} occupé — abandon")
    stop_and_free()
    open(OUT_SERVER_LOG, "w").close()
    allrows = []
    for mode, ctx, ng in POINTS:
        for name, extra in LEGS:
            try:
                log(f"=== {mode} {ctx} :: LEG {name} {extra}")
                start_server(extra)
                since = log_size()
                api, met = run_point(mode, ctx, ng, since)
                row = {"leg": name, "mode": mode, "ctx": ctx, "mtp": "off",
                       "so": os.path.basename(extra.get("_SO", SO_PROD)),
                       "env": {k: v for k, v in extra.items() if k != "_SO"},
                       **api, **met}
                allrows.append(row)
                with open(OUT_JSONL, "a") as f:
                    f.write(json.dumps(row) + "\n")
                log(f"    → prefill {met.get('prompt_tps','?')} t/s ({met.get('prompt_tokens','?')} tok) | "
                    f"decode {met.get('eval_tps','?')} t/s | {api.get('error','ok')}")
            except Exception as e:
                log(f"  · {mode} {ctx} {name} ÉCHEC : {e}")
            finally:
                stop_server()
                time.sleep(12)

    # ---- summary ----
    lines = [f"# FA16 E2E — A-prod / B-16x2 / C-compact — {TS}", "",
             "llama-server standalone, KV q8_0, -b 2048 -ub 2048, MTP OFF, env prod stricte "
             "(Q6_W4R4=1, Q8_FATTN_TC=0, FIT_TARGET=1024). 3 serveurs frais intercalés par point.",
             f"service restauré : {'OUI' if _restored else 'NON'}", ""]
    lines.append("| point | A-prod | B-16x2 | C-compact | Δ B/A | Δ C/A |")
    lines.append("|---|---|---|---|---|---|")
    idx = {n: {} for n, _ in LEGS}
    for r in allrows:
        idx[r["leg"]].setdefault((r["mode"], r["ctx"]), r)
    names = [n for n, _ in LEGS]
    for mode, ctx, ng in POINTS:
        cells, ok = [], True
        for n in names:
            r = idx[n].get((mode, ctx))
            if not r or not r.get("prompt_tps"): ok = False
            cells.append(f"{r.get('prompt_tps','?')}" if r else "—")
        a = idx[names[0]].get((mode, ctx))
        b = idx[names[1]].get((mode, ctx))
        c = idx[names[2]].get((mode, ctx))
        def d(x, y):
            if x and y and x.get("prompt_tps") and y.get("prompt_tps"):
                return f"{(y['prompt_tps']-x['prompt_tps'])/x['prompt_tps']*100:+.1f}%"
            return "—"
        lines.append(f"| {mode} {ctx} | {cells[0]} | {cells[1]} | {cells[2]} | {d(a,b)} | {d(a,c)} |")
    lines.append("")
    lines.append("## DECODE (t/s) — contrôle, doit rester stable")
    lines.append("| point | A-prod | B-16x2 | C-compact |")
    lines.append("|---|---|---|---|")
    for mode, ctx, ng in POINTS:
        cells = []
        for n in names:
            r = idx[n].get((mode, ctx))
            cells.append(f"{r.get('eval_tps','?')}" if r else "—")
        lines.append(f"| {mode} {ctx} | " + " | ".join(cells) + " |")
    lines.append("")
    with open(OUT_SUMMARY, "w") as f:
        f.write("\n".join(lines))
    restore_service()
    with open(OUT_SUMMARY, "a") as f:
        f.write(f"\nservice restauré : {'OUI' if _restored else 'NON'}\n")
    log(f"FIN — synthèse : {OUT_SUMMARY} | restauré = {_restored}")
    sys.exit(0 if _restored else 3)


if __name__ == "__main__":
    main()
