#!/usr/bin/env python3
"""
FA16 THERMAL CHECK — même contexte (131k), 2 ordres pour isoler l'effet thermique
de l'écart decode observé en e2e (B/C semblaient perdre du decode à haut ctx).

Hypothèse : l'écart decode n'est PAS le .so (microbench T=1 A==C), mais la
position dans le run (C tournait 3e = GPU le plus chaud). On teste en inversant :
  run1 : C d'abord (frais) puis A
  run2 : C d'abord (frais) puis A
Si C-frais >= A  -> pas de régression decode, l'écart e2e est thermique.
KV q8_0, -b 2048, MTP off, env prod. Lancer via superviseur.
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
CUDA_LIBS = [
    "/home/guile/llama-experiments/ollama-v100/lib/ollama",
    "/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12",
    "/usr/local/cuda-12.9/lib64",
]
PORT = 8917
SERVICE = "ollama-v100"
TS = datetime.now().strftime("%Y%m%d-%H%M%S")
OUT_JSONL = os.path.join(D, f"fa16_thm_{TS}.jsonl")
OUT_LOG = os.path.join(D, f"fa16_thm_{TS}.log")
OUT_SRV = os.path.join(D, f"fa16_thm_{TS}_server.log")
ENV_BASE = {"GGML_CUDA_VOLTA_Q6_W4R4": "1", "GGML_CUDA_VOLTA_Q8_FATTN_TC": "0",
            "LLAMA_ARG_FIT_TARGET": "1024"}
# run1 et run2 : C (compact) d'abord -> frais, A ensuite. 2 répétitions.
RUNS = [
    ("run1", [("C-compact", {"_SO": SO_DUAL, "GGML_CUDA_VOLTA_FA_COMPACT_TMAX": "8192"}),
              ("A-prod",    {"_SO": SO_PROD})]),
    ("run2", [("C-compact", {"_SO": SO_DUAL, "GGML_CUDA_VOLTA_FA_COMPACT_TMAX": "8192"}),
              ("A-prod",    {"_SO": SO_PROD})]),
]
CTX = 131072
NG = 64
MAX_CTX = 196672
TIMEOUT_S = 3600
GPU_SMI_IDX = v100_smi_index()
GPU_CUDA_IDX = v100_cuda_ordinal()
FILL = ("La tour de contrôle surveille le passage des avions au crépuscule "
        "pendant que le chef de piste organise les départs de la soirée. ")
_state = {}
_restored = False


def log(m):
    l = f"[{datetime.now().strftime('%H:%M:%S')}] {m}"
    print(l, flush=True)
    with open(OUT_LOG, "a") as f:
        f.write(l + "\n")


def run(c, timeout=60):
    return subprocess.run(c, capture_output=True, text=True, timeout=timeout)


def svc_state():
    return run(["systemctl", "is-active", SERVICE]).stdout.strip()


def gpu_state():
    r = run(["nvidia-smi", f"--id={GPU_SMI_IDX}",
             "--query-gpu=memory.used,utilization.gpu,clocks.sm,temperature.gpu",
             "--format=csv,noheader,nounits"])
    return r.stdout.strip()


def vram_mib():
    r = run(["nvidia-smi", f"--id={GPU_SMI_IDX}",
             "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    v = r.stdout.strip()
    return int(v) if v.isdigit() else -1


def restore_service():
    global _restored
    try:
        if svc_state() != "active":
            log("  · restauration : systemctl start " + SERVICE)
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


def on_sig(s, f):
    restore_service()
    sys.exit(128 + s)


def stop_and_free():
    log("  · stop " + SERVICE)
    run(["sudo", "systemctl", "stop", SERVICE], timeout=120)
    t0 = time.time()
    while time.time() - t0 < 180:
        if vram_mib() < 1024:
            break
        time.sleep(3)
    log(f"  · V100 libérée ({vram_mib()} MiB) en {time.time()-t0:.0f}s | {gpu_state()}")
    time.sleep(30)  # laisser le GPU refroidir un peu avant la 1re mesure
    log(f"  · après 30s pause : {gpu_state()}")


def build_env(extra):
    env = dict(os.environ)
    for ddir in CUDA_LIBS:
        if os.path.isdir(ddir) and ddir not in env.get("LD_LIBRARY_PATH", ""):
            env["LD_LIBRARY_PATH"] = ddir + ":" + env.get("LD_LIBRARY_PATH", "")
    env.update({k: v for k, v in extra.items() if k != "_SO"})
    env["GGML_BACKEND_PATH"] = extra.get("_SO", SO_PROD)
    env["CUDA_VISIBLE_DEVICES"] = str(GPU_CUDA_IDX)
    return env


def server_args():
    return ["--model", GGUF, "--port", str(PORT), "--host", "127.0.0.1",
            "--no-webui", "--offline", "-c", str(MAX_CTX), "-np", "1",
            "-b", "2048", "-ub", "2048", "--log-verbosity", "4",
            "--no-log-prefix", "--no-log-timestamps", "--mmproj", MMPROJ,
            "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
            "--flash-attn", "auto"]


def port_free(p):
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", p)); s.close(); return True
    except OSError:
        s.close(); return False


def http_json(url, data=None, timeout=10):
    req = urllib.request.Request(url,
        data=json.dumps(data).encode() if data is not None else None,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def start_server(extra):
    if vram_mib() > 8192:
        raise RuntimeError(f"V100 pas vide (VRAM={vram_mib()} MiB)")
    env = build_env(extra)
    lf = open(OUT_SRV, "a")
    proc = subprocess.Popen([LLAMA_SERVER] + server_args(), stdout=lf,
                            stderr=subprocess.STDOUT, env=env)
    _state["proc"] = proc
    t0 = time.time()
    while time.time() - t0 < 600:
        if proc.poll() is not None:
            lf.close()
            raise RuntimeError(f"llama-server mort (exit {proc.returncode})")
        try:
            http_json(f"http://127.0.0.1:{PORT}/health", timeout=3); break
        except Exception:
            time.sleep(2)
    else:
        lf.close(); raise RuntimeError("pas de /health en 600 s")
    time.sleep(3)
    if vram_mib() < 1024:
        lf.close(); raise RuntimeError(f"ABORT VRAM={vram_mib()} MiB (model CPU)")
    d = http_json(f"http://127.0.0.1:{PORT}/v1/models", timeout=10)
    _state["model_id"] = d["data"][0]["id"]
    lf.close()
    log(f"  · server prêt ({time.time()-t0:.0f}s, {vram_mib()} MiB) {os.path.basename(extra.get('_SO',SO_PROD))}")


def stop_server():
    p = _state.get("proc")
    if p and p.poll() is None:
        p.terminate()
        try: p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill(); p.wait(timeout=10)
    time.sleep(10)
    for _ in range(60):
        if vram_mib() < 4096:
            break
        time.sleep(3)
    log(f"  · server stop, {vram_mib()} MiB | {gpu_state()}")


def prompt_text(n_tok):
    n_words = n_tok * 4 // 5
    words = FILL.split() * max(1, n_words // FILL.count(" ") + 1)
    return " ".join(words[:n_words])


def parse_point(logf, since):
    try:
        data = open(logf, errors="replace").read()[since:]
    except OSError:
        return {}
    out = {}
    m = re.search(r"prompt eval time =\s+([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", data)
    if m:
        out["prompt_tps"] = round(int(m.group(2)) / (float(m.group(1)) / 1000.0), 1)
    m = re.search(r"(?<!prompt )eval time =\s+([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", data)
    if m:
        out["eval_tps"] = round(int(m.group(2)) / (float(m.group(1)) / 1000.0), 2)
    return out


def run_point(name, extra, since):
    data = {"model": _state["model_id"], "stream": False,
            "messages": [{"role": "user", "content": prompt_text(CTX) + "\nRéponds en une phrase."}],
            "max_tokens": NG, "temperature": 0.0, "top_k": 1, "top_p": 1.0}
    t0 = time.time()
    try:
        r = http_json(f"http://127.0.0.1:{PORT}/v1/chat/completions", data, timeout=TIMEOUT_S)
        u = (r or {}).get("usage", {})
        api = {"prompt_tokens_api": u.get("prompt_tokens"), "gen_tokens_api": u.get("completion_tokens"),
               "wall_s": round(time.time() - t0, 1)}
    except Exception as e:
        api = {"error": str(e)}
    return api, parse_point(OUT_SRV, since)


def log_size():
    try: return os.path.getsize(OUT_SRV)
    except OSError: return 0


def main():
    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)
    open(OUT_LOG, "a").close()
    log(f"FA16 THERMAL CHECK — ctx={CTX} — C d'abord (frais) x2 — {TS}")
    if not port_free(PORT):
        sys.exit(f"port {PORT} occupé")
    stop_and_free()
    open(OUT_SRV, "w").close()
    rows = []
    try:
        for rname, legs in RUNS:
            for name, extra in legs:
                try:
                    log(f"=== {rname} :: {name} (GPU {gpu_state()})")
                    start_server(extra)
                    since = log_size()
                    api, met = run_point(name, extra, since)
                    row = {"run": rname, "leg": name, "ctx": CTX, **api, **met}
                    rows.append(row)
                    with open(OUT_JSONL, "a") as f:
                        f.write(json.dumps(row) + "\n")
                    log(f"    → prefill {met.get('prompt_tps','?')} t/s | decode {met.get('eval_tps','?')} t/s | {api.get('error','ok')}")
                except Exception as e:
                    log(f"  · {rname} {name} ÉCHEC : {e}")
                finally:
                    stop_server()
                    time.sleep(20)  # pause de refroidissement entre legs
    finally:
        restore_service()
    # résumé
    log("=== RÉSUMÉ decode (C frais vs A) ===")
    by = {}
    for r in rows:
        by.setdefault(r["run"], {})[r["leg"]] = r
    for rname, d in by.items():
        c, a = d.get("C-compact"), d.get("A-prod")
        if c and a and c.get("eval_tps") and a.get("eval_tps"):
            log(f"  {rname}: C-frais decode {c['eval_tps']} vs A decode {a['eval_tps']} "
                f"-> { (c['eval_tps']-a['eval_tps'])/a['eval_tps']*100:+.1f}% | "
                f"prefill C {c.get('prompt_tps')} vs A {a.get('prompt_tps')}")
    log(f"FIN — service restauré = {_restored}")
    sys.exit(0 if _restored else 3)


if __name__ == "__main__":
    main()
