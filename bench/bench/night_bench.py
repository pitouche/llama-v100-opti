#!/usr/bin/env python3
"""
night_bench.py — NUIT D'OPTIMISATION (18/10) — 3 expériences, MTP OFF partout.

Expérience 1 (H3)  : .so prod H2 (a4b6f440) vs .so H3 = H2 + patch 42d9581
                     (sm70 -> table MMVQ TURING, nwarps K-quant 4->2).
                     Upstream revendique +3,17% tg128 sur V100-32GB PCIe driver 580
                     (= notre machine). Le test 17/10 était en MTP-ON (pollué) ->
                     ici MTP OFF, gold standard interleave.
Expérience 2 (BATCH): -b 2048 (prod) vs 4096 vs 8192 en DECODE + prefill.
                     b8192 seulement au 114k (VRAM : buffer compute ~6 GB).
Expérience 3 (NT)   : -t 6 (défaut) vs -t 12 (CPU threads).

Méthode (identique ab_confirm, gold standard) : serveur frais par leg, MTP OFF,
KV q8_0, FA auto, CVD=0 (V100), port 8917, stop/start du service géré ici,
restauration en finally. Warmup 1k avant les points. Échantillonneur nvidia-smi
au fond (profiling).

USAGE :
  cd /home/guile/llama-experiments/bench
  python3 supervisor_bench.py run night_bench.py [reps] --detach   # reps défaut 1
"""
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
GGUF = "/usr/share/ollama/.ollama/models/blobs/sha256-493301830a596b8ad56dc1329f80bbcb578c8e910da395feafdc9cd8263430bb"
MMPROJ = "/usr/share/ollama/.ollama/models/blobs/sha256-ac3714bfdddeca31351f2752bf1a63f266f4df87c0b68c895e44945ca704448e"

PORT = 8917
SERVICE = "ollama-v100"
OUT_JSONL = os.path.join(D, "night_bench_results.jsonl")
OUT_SUMMARY = os.path.join(D, "night_bench_summary.md")
OUT_SERVER_LOG = os.path.join(D, "night_bench_server.log")
OUT_GPUTS = os.path.join(D, "night_bench_gputimes.tsv")

SO_PROD = "/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so"
SO_H3 = "/home/guile/llama-experiments/h2/cupti129-test/e2/h2lib/lib/libggml-cuda-42d9581.so"

CUDA_LIBS = [
    "/home/guile/llama-experiments/ollama-v100/lib/ollama",
    "/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12",
    "/home/guile/llama-experiments/tools/cuda-12.6/usr/local/cuda-12.6/lib64",
    "/home/guile/llama-experiments/tools/cuda-12.6/usr/local/cuda-12.6/targets/x86_64-linux/lib",
]

ENV_BASE = {"GGML_CUDA_VOLTA_Q6_W4R4": "1", "LLAMA_ARG_FIT_TARGET": "1024"}
MAX_CTX = 196672
TIMEOUT_S = 900
GPU_SMI_IDX = v100_smi_index()  # résolu par UUID (ne pas hardcoder)
GPU_CUDA_IDX = v100_cuda_ordinal()  # résolu par UUID (ne pas hardcoder)
N_GEN = 64
N_WARM = 1024

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

def build_env(so_path, extra):
    env = dict(os.environ)
    for d in CUDA_LIBS:
        if os.path.isdir(d) and d not in env.get("LD_LIBRARY_PATH", ""):
            env["LD_LIBRARY_PATH"] = d + ":" + env.get("LD_LIBRARY_PATH", "")
    env.update(extra)
    env["GGML_BACKEND_PATH"] = so_path
    env["CUDA_VISIBLE_DEVICES"] = str(GPU_CUDA_IDX)
    return env

def server_args(batch, nthreads=None):
    a = [
        "--model", GGUF, "--port", str(PORT), "--host", "127.0.0.1",
        "--no-webui", "--offline",
        "-c", str(MAX_CTX), "-np", "1",
        "-b", str(batch), "-ub", str(batch),
        "--log-verbosity", "4", "--no-log-prefix", "--no-log-timestamps",
        "--mmproj", MMPROJ,
        # MTP OFF : pas de --spec-type (consigne 18/10 : MTP exclu)
        "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
        "--flash-attn", "auto",
    ]
    if nthreads:
        a += ["-t", str(nthreads)]
    return a

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

def start_server(so_path, batch, nthreads=None, extra_env=None):
    extra = dict(extra_env or {})
    if vram_mib() > 8192:
        raise RuntimeError(f"V100 pas vide (VRAM={vram_mib()} MiB) — refus de démarrer")
    env = build_env(so_path, extra)
    lf = open(OUT_SERVER_LOG, "a")
    proc = subprocess.Popen([LLAMA_SERVER] + server_args(batch, nthreads),
                            stdout=lf, stderr=subprocess.STDOUT, env=env)
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
    log(f"  · server prêt ({time.time()-t0:.0f}s, VRAM {vram_mib()} MiB) -b {batch}"
        + (f" -t {nthreads}" if nthreads else ""))
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
    # --- stats spec MTP (lignes 'spec common_specu: statistics ... dur(b,g,a)') ---
    # cumul depuis le démarrage du serveur ; pour un delta par fenêtre, cf. run_point2/_spec_delta.
    m = re.findall(
        r"#gen drafts =\s*(\d+), #acc drafts =\s*(\d+), #gen tokens =\s*(\d+), #acc tokens =\s*(\d+)"
        r"(?:, #mean acc len =\s*([\d.]+))?"
        r"(?:, #acc rate/pos = \(([\d.]+), ([\d.]+), ([\d.]+)\))?",
        data)
    if m:
        last = m[-1]
        out["spec_cum"] = [int(last[0]), int(last[1]), int(last[2]), int(last[3])]
        if last[4]:
            out["mtp_mean_len"] = float(last[4])
        if last[5]:
            out["acc_rate_pos"] = [float(last[5]), float(last[6]), float(last[7])]
    m = re.findall(r"dur\(b,g,a\) = ([\d.]+), ([\d.]+), ([\d.]+) ms", data)
    if m:
        out["spec_dur_bga_ms"] = [float(x) for x in m[-1]]
    return out

def log_size():
    try: return os.path.getsize(OUT_SERVER_LOG)
    except OSError: return 0

def run_point(ctx_target, n_gen, since_pos, tag=""):
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


def _spec_delta(before, after):
    """Delta des stats spec cumulées (b,g,a) entre deux mesures de log (par fenêtre)."""
    out = {}
    cb, ca = (before or {}).get("spec_cum"), (after or {}).get("spec_cum")
    if cb and ca and len(cb) == 4 and len(ca) == 4:
        out["gen_drafts"] = ca[0] - cb[0]
        out["acc_drafts"] = ca[1] - cb[1]
        out["gen_tokens"] = ca[2] - cb[2]
        out["acc_tokens"] = ca[3] - cb[3]
        if out["gen_drafts"]:
            # ratio sur le DELTA de la fenêtre, pas sur le cumul serveur
            out["acc_rate"] = round(out["acc_drafts"] / out["gen_drafts"], 4)
            out["mean_acc_len"] = round(1.0 + out["acc_tokens"] / out["gen_drafts"], 3)
    db, da = (before or {}).get("spec_dur_bga_ms"), (after or {}).get("spec_dur_bga_ms")
    if db and da:
        # dur(b,g,a) = (base?, gen, acc) en ms, cumulées depuis le démarrage serveur
        out["dur_g_ms"] = round(da[1] - db[1], 1)
        out["dur_a_ms"] = round(da[2] - db[2], 1)
    return out


def _spec_snapshot(logf, start_pos=0, upto_pos=None):
    """Cumul des stats spec (dernière ligne 'spec common_specu: statistics') dans
    log[start_pos:upto_pos].

    Les stats sont cumulées depuis le démarrage du serveur (cf.
    common_speculative_print_stats) ; on fait des différences de snapshots pour un
    delta par fenêtre. IMPORTANT : lire à partir de start_pos (début de la section du
    leg courant) car le log serveur est en append entre legs — lire depuis 0 ramènerait
    les stats stalées du leg précédent (ex. MTP-ON vu du leg MTP-OFF).
    """
    try:
        with open(logf, "r", errors="replace") as f:
            f.seek(start_pos)
            data = f.read(upto_pos - start_pos) if upto_pos is not None else f.read()
    except OSError:
        return {}
    out = {}
    m = re.findall(
        r"#gen drafts =\s*(\d+), #acc drafts =\s*(\d+), #gen tokens =\s*(\d+), #acc tokens =\s*(\d+)",
        data)
    if m:
        out["spec_cum"] = [int(x) for x in m[-1]]
    m = re.findall(r"dur\(b,g,a\) = ([\d.]+), ([\d.]+), ([\d.]+) ms", data)
    if m:
        out["spec_dur_bga_ms"] = [float(x) for x in m[-1]]
    return out


def run_point2(ctx_target, n_gen_first, n_gen_second, since_pos, tag=""):
    """Mesure MTP corrigeant le piège DEFER_PROMPT (voir mtp_defer_diagnostic_20261029.md).

    Envoie 2 requêtes sur le MÊME prompt long :
      W1 : première fenêtre de decode après prompt long NEUF — si DEFER_PROMPT=1,
           contient le catch-up MTP du prompt (transitoire, ∝ tokens NOUVEAUX) : PAS
           du steady-state. C'est la fenêtre que le harnais v1 mesurait.
      W2 : même prompt renvoyé (checkpoint restauré, ~0 tok nouveaux) → le vrai
           débit steady-state.

    Champs retournés :
      first_window_tps / steady_tps / catchup_ms (surcoût transitoire de W1)
      prompt_ms_w1, eval_ms (W1), prompt_ms_w2, eval_ms_w2, total_ms
      spec_delta_w1 / spec_delta_w2 : drafts générés/acceptés, mean_acc_len,
        dur_g_ms (t_draft cumulé) et dur_a_ms (t_accept) par fenêtre.
    """
    text = prompt_text(ctx_target)

    def one(n_gen):
        data = {
            "model": _state["model_id"], "stream": False,
            "messages": [{"role": "user", "content": text + "\nRéponds en une phrase."}],
            "max_tokens": n_gen, "temperature": 0.0, "top_k": 1, "top_p": 1.0,
        }
        t0 = time.time()
        try:
            r = http_json(f"http://127.0.0.1:{PORT}/v1/chat/completions", data, timeout=TIMEOUT_S)
            u = (r or {}).get("usage", {})
            return {"prompt_tokens_api": u.get("prompt_tokens"),
                    "gen_tokens_api": u.get("completion_tokens"),
                    "wall_s": round(time.time() - t0, 1)}
        except Exception as e:
            return {"error": str(e)}

    c0 = _spec_snapshot(OUT_SERVER_LOG, since_pos, since_pos)  # cumul avant W1 (log du leg)
    api1 = one(n_gen_first)
    time.sleep(1)
    c1 = _spec_snapshot(OUT_SERVER_LOG, since_pos, None)      # cumul après W1
    api2 = one(n_gen_second)
    time.sleep(1)
    c2 = _spec_snapshot(OUT_SERVER_LOG, since_pos, None)      # cumul après W2

    with open(OUT_SERVER_LOG, "r", errors="replace") as f:
        data = f.read()[since_pos:]
    # W1 = 1ers matches, W2 = derniers matches (exactement 2 req dans la tranche).
    out = {}
    m = re.findall(r"prompt eval time =\s+([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", data)
    if m:
        ms, n = float(m[0][0]), int(m[0][1])
        out["prompt_ms_w1"] = round(ms, 1)
        out["prompt_tokens"] = n
        out["prompt_tps"] = round(n / (ms / 1000.0), 1) if ms else 0
    me = re.findall(r"(?<!prompt )eval time =\s+([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", data)
    if me:
        ms, n = float(me[0][0]), int(me[0][1])
        out["first_window_tps"] = round(n / (ms / 1000.0), 2) if ms else 0
        out["eval_tokens"] = n
        out["eval_ms"] = round(ms, 1)
        ms2, n2 = float(me[-1][0]), int(me[-1][1])
        out["steady_tps"] = round(n2 / (ms2 / 1000.0), 2) if ms2 else 0
        out["eval_tokens_w2"] = n2
        out["eval_ms_w2"] = round(ms2, 1)
    if m:
        ms2, n2 = float(m[-1][0]), int(m[-1][1])
        out["prompt_ms_w2"] = round(ms2, 1)
        out["prompt_tokens_w2"] = n2
    out["api_w1"] = api1
    out["api_w2"] = api2
    # coût de catch-up = surplus de W1 vs steady (même nombre de tokens W1).
    if out.get("first_window_tps") and out.get("steady_tps") and out.get("eval_tokens", 0) > 0:
        out["catchup_ms"] = round(out["eval_tokens"] * 1000.0 / out["first_window_tps"]
                                  - out["eval_tokens"] * 1000.0 / out["steady_tps"], 1)
    if all(out.get(k) for k in ("prompt_ms_w1", "eval_ms", "prompt_ms_w2", "eval_ms_w2")):
        out["total_ms"] = round(out["prompt_ms_w1"] + out["eval_ms"]
                                + out["prompt_ms_w2"] + out["eval_ms_w2"], 1)
    out["spec_delta_w1"] = _spec_delta(c0, c1)
    out["spec_delta_w2"] = _spec_delta(c1, c2)
    return out


def warmup(since_pos):
    data = {
        "model": _state["model_id"], "stream": False,
        "messages": [{"role": "user", "content": "Compte de 1 à 5."}],
        "max_tokens": N_WARM, "temperature": 0.0, "top_k": 1, "top_p": 1.0,
    }
    try:
        r = http_json(f"http://127.0.0.1:{PORT}/v1/chat/completions", data, timeout=TIMEOUT_S)
        return "ok"
    except Exception as e:
        return f"err:{e}"

def med(vals):
    vals = sorted(v for v in vals if v)
    return vals[len(vals)//2] if vals else None

def main():
    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    log(f"NIGHT BENCH (MTP OFF) — reps={reps} — lock tenu par le superviseur")
    if not port_free(PORT):
        sys.exit(f"port {PORT} occupe — abandon")
    for f in (SO_PROD, SO_H3):
        if not os.path.isfile(f):
            sys.exit(f"lib manquante : {f}")
    log(f"  · SO prod md5 = {__import__('hashlib').md5(open(SO_PROD,'rb').read()).hexdigest()[:8]}")
    log(f"  · SO H3   md5 = {__import__('hashlib').md5(open(SO_H3,'rb').read()).hexdigest()[:8]}")
    stop_and_free()
    open(OUT_SERVER_LOG, "w").close()
    open(OUT_GPUTS, "w").close()

    sampler = subprocess.Popen(["python3", os.path.join(D, "gputimes_sampler.py"), OUT_GPUTS])

    CTX_LONG = [114688, 196608]
    CTX_MID  = [114688]

    # (exp, leg, so, batch, nthreads, ctxs)
    EXP1 = [("EXP1-H3", n, s, 2048, None, CTX_LONG) for n, s in
            [("A-prodH2", SO_PROD), ("B-H3-42d", SO_H3)]]
    EXP2 = [("EXP2-BATCH", n, SO_PROD, b, None, (CTX_MID if b == 8192 else CTX_LONG))
            for n, b in [("A-b2048", 2048), ("B-b4096", 4096), ("C-b8192", 8192)]]
    EXP3 = [("EXP3-NT", n, SO_PROD, 2048, t, CTX_MID)
            for n, t in [("A-t6", 6), ("B-t12", 12)]]

    allrows = []
    try:
        for exp in (EXP1, EXP2, EXP3):
            for rep in range(1, reps + 1):
                legs = exp if rep % 2 == 1 else list(reversed(exp))
                for name, tag, so, b, t, ctxs in legs:
                    try:
                        log(f"=== {tag} rep{rep} :: LEG {name} (b={b}, t={t}, {len(ctxs)} ctx)")
                        start_server(so, b, t)
                        wu = warmup(log_size())
                        for ctx in ctxs:
                            since = log_size()
                            log(f"  · point {ctx} → {N_GEN} gen")
                            api, met = run_point(ctx, N_GEN, since)
                            row = {"exp": tag, "leg": name, "rep": rep, "ctx": ctx,
                                   "mtp": "off", "batch": b, "nthreads": t,
                                   "vram_mib": vram_mib(),
                                   **api, **met}
                            allrows.append(row)
                            log(f"    → prefill {met.get('prompt_tps','?')} t/s | "
                                f"decode {met.get('eval_tps','?')} t/s | {api.get('error','ok')}")
                    except Exception as e:
                        log(f"  · {tag} {name} rep{rep} ECHEC : {e}")
                    finally:
                        stop_server()
                        time.sleep(12)
    finally:
        try:
            sampler.terminate(); sampler.wait(timeout=10)
        except Exception:
            pass
    restore_service()

    with open(OUT_JSONL, "a") as f:
        for r in allrows:
            f.write(json.dumps(r) + "\n")

    # ---- summary ----
    L = ["# NIGHT BENCH — 18/10 — MTP OFF, même .so par exp, interleave", "",
         f"date : {time.strftime('%Y-%m-%d %H:%M UTC')}",
         f"reps : {reps} · service restauré : {'OUI' if _restored else 'NON'}", ""]

    def table(tag, title, legs, ctxs):
        L.append(f"## {tag} — {title}")
        L.append("| ctx | rep | " + " | ".join(f"{n} dec | {n} pre" for n, *_ in legs) + " |")
        L.append("|" + "---|" * (2 + 2*len(legs)))
        for ctx in ctxs:
            for rep in range(1, reps + 1):
                cells = []
                for n, *_ in legs:
                    rs = [r for r in allrows if r["exp"] == tag and r["leg"] == n
                          and r["ctx"] == ctx and r.get("rep") == rep]
                    if rs:
                        cells += [str(rs[0].get("eval_tps", "?")), str(rs[0].get("prompt_tps", "?"))]
                    else:
                        cells += ["—", "—"]
                L.append(f"| {ctx} | {rep} | " + " | ".join(cells) + " |")
        L.append("")
        if len(legs) >= 2:
            base = legs[0][0]
            L.append(f"Δ decode vs {base} (médiane par leg) :")
            vals = {}
            for n, *_ in legs:
                rs = [r for r in allrows if r["exp"] == tag and r["leg"] == n]
                vals[n] = [r["eval_tps"] for r in rs if r.get("eval_tps")]
            bv = med(vals[base])
            for n, *_ in legs:
                v = med(vals[n])
                if bv and v:
                    L.append(f"  - {n}: {(v-bv)/bv*100:+.1f}%")
            L.append("")

    table("EXP1-H3", "decode/prefill H2-prod vs H3 (42d9581) — -b 2048",
          [(n,) for n in ["A-prodH2", "B-H3-42d"]], CTX_LONG)
    table("EXP2-BATCH", "decode/prefill batch 2048/4096/8192 — .so prod",
          [(n,) for n in ["A-b2048", "B-b4096", "C-b8192"]], CTX_LONG)
    table("EXP3-NT", "decode/prefill -t 6 vs 12 — .so prod, -b 2048",
          [(n,) for n in ["A-t6", "B-t12"]], CTX_MID)

    L.append("## VRAM pic par leg (MiB)")
    L.append("| leg | max VRAM |")
    L.append("|---|---|")
    seen = {}
    for r in allrows:
        if r.get("vram_mib"):
            seen[r["leg"]] = max(seen.get(r["leg"], 0), r["vram_mib"])
    for k, v in sorted(seen.items()):
        L.append(f"| {k} | {v} |")

    with open(OUT_SUMMARY, "w") as f:
        f.write("\n".join(L) + "\n")
    log(f"FIN — synthèse : {OUT_SUMMARY}")
    sys.exit(0 if _restored else 3)

if __name__ == "__main__":
    main()
