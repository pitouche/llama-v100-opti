#!/usr/bin/env python3
"""
ab_ub_fix512.py — Re-qualification de l'anomalie 512@98k du sweep ab_ub (01:43).

Constat : 512@98k decode = 19.76 / 17.58 t/s (vs 22.97-22.99 pour 1024/2048/4096@98k,
et vs 23.84-23.92 pour 512@65k). Seule anomalie du run (sinon decode ±0.1 %).
Logs propres (pas d'offload, pas de shift, pas de warn). Reproductible ? Bissection ctx.

Legs (ordre) :
  1. 98304 @ 512   rep1
  2. 98304 @ 512   rep2
  3. 98304 @ 2048  (contrôle)
  4.  81920 @ 512  (bisection)
  5.  65536 @ 512  (contrôle)

Même protocole que ab_ub.py (.so prod 4fd152f9 lecture seule, MTP OFF, -c 128000,
KV q8_0, FA auto, port 8917, 64 tok gen).

USAGE :
  cd /home/guile/llama-experiments/bench
  python3 supervisor_bench.py run ab_ub_fix512.py --detach
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
OUT_JSONL = os.path.join(D, "ab_ub_fix512_results.jsonl")
OUT_SERVER_LOG = os.path.join(D, "ab_ub_fix512_server.log")
CUDA_LIBS = [
    "/home/guile/llama-experiments/ollama-v100/lib/ollama",
    "/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12",
    "/home/guile/llama-experiments/tools/cuda-12.6/usr/local/cuda-12.6/lib64",
    "/home/guile/llama-experiments/tools/cuda-12.6/usr/local/cuda-12.6/targets/x86_64-linux/lib",
]
SO_PROD = "/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so"
ENV_BASE = {
    "GGML_CUDA_VOLTA_Q6_W4R4": "1",
    "GGML_CUDA_VOLTA_Q8_FATTN_TC": "0",
    "LLAMA_ARG_FIT_TARGET": "1024",
}
LEGS = [
    (98304, 512,  "98k 512 rep1"),
    (98304, 512,  "98k 512 rep2"),
    (98304, 2048, "98k 2048 ctrl"),
    (81920, 512,  "80k 512 bisection"),
    (65536, 512,  "65k 512 ctrl"),
]
NGEN = 64
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
    run(["sudo", "systemctl", "stop", SERVICE], timeout=120)
    t0 = time.time()
    while time.time() - t0 < 180:
        if vram_mib() < 1024 and v100_proc_gone():
            log(f"  · V100 liberee ({vram_mib()} MiB) en {time.time()-t0:.0f}s")
            return True
        time.sleep(3)
    log(f"  · ATTENTION V100 pas totalement liberee (VRAM={vram_mib()} MiB)")
    return False
def build_env():
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
        raise RuntimeError(f"V100 pas vide (VRAM={vram_mib()} MiB)")
    env = build_env()
    lf = open(OUT_SERVER_LOG, "a")
    proc = subprocess.Popen([LLAMA_SERVER] + server_args(batch), stdout=lf,
                            stderr=subprocess.STDOUT, env=env)
    _state["proc"] = proc
    t0 = time.time()
    while time.time() - t0 < 600:
        if proc.poll() is not None:
            tail = open(OUT_SERVER_LOG, "r", errors="replace").read()[-1500:]
            raise RuntimeError(f"llama-server mort (exit {proc.returncode}) : ...{tail}")
        try:
            http_json(f"http://127.0.0.1:{PORT}/health", timeout=3); break
        except Exception:
            time.sleep(2)
    else:
        raise RuntimeError("pas de /health en 600 s")
    time.sleep(3)
    if vram_mib() < 1024:
        raise RuntimeError(f"ABORT leg : VRAM={vram_mib()} MiB")
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
def log_size():
    try: return os.path.getsize(OUT_SERVER_LOG)
    except OSError: return 0
def parse_leg(data):
    out = {}
    m = re.search(r"prompt eval time =\s+([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", data)
    if m:
        ms, n = float(m.group(1)), int(m.group(2))
        out["prompt_tps"] = round(n / (ms / 1000.0), 1) if ms else 0
    m = re.search(r"(?<!prompt )eval time =\s+([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", data)
    if m:
        ms, n = float(m.group(1)), int(m.group(2))
        out["eval_tps"] = round(n / (ms / 1000.0), 2) if ms else 0
    return out
def main():
    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)
    log(f"FIX512 — prod md5 {hashlib.md5(open(SO_PROD,'rb').read()).hexdigest()[:8]} (lecture seule)")
    if not port_free(PORT):
        sys.exit(f"port {PORT} occupe — abandon")
    stop_and_free()
    open(OUT_SERVER_LOG, "w").close()
    rows = []
    try:
        for ctx, b, tag in LEGS:
            since = log_size()
            try:
                log(f"=== {tag} :: -b/-ub {b}")
                vram = start_server(b)
                text = prompt_text(ctx)
                data = {
                    "model": _state["model_id"], "stream": False,
                    "messages": [{"role": "user", "content": text + "\nReponds en une phrase."}],
                    "max_tokens": NGEN, "temperature": 0.0, "top_k": 1, "top_p": 1.0,
                }
                t0 = time.time()
                try:
                    r = http_json(f"http://127.0.0.1:{PORT}/v1/chat/completions", data, timeout=TIMEOUT_S)
                    api = {"wall_s": round(time.time() - t0, 1)}
                except Exception as e:
                    api = {"error": str(e)}
                with open(OUT_SERVER_LOG, "r", errors="replace") as f:
                    met = parse_leg(f.read()[since:])
                rows.append({"tag": tag, "ctx": ctx, "batch": b, "vram_ready_mib": vram, **api, **met})
                log(f"    → prefill {met.get('prompt_tps','?')} t/s | decode {met.get('eval_tps','?')} t/s | {api.get('error','ok')}")
            except Exception as e:
                log(f"  · {tag} ECHEC : {e}")
                rows.append({"tag": tag, "ctx": ctx, "batch": b, "error": str(e)[:200]})
            finally:
                stop_server()
                time.sleep(12)
    finally:
        restore_service()
        with open(OUT_JSONL, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        log("=== RESULTATS FIX512 ===")
        for r in rows:
            log(f"  {r['tag']:<22} prefill={r.get('prompt_tps','?')} t/s decode={r.get('eval_tps','?')} t/s {r.get('error','ok')}")
        log(f"FIN — service restaure : {svc_state()}")
    sys.exit(0 if _restored else 3)

if __name__ == "__main__":
    main()
