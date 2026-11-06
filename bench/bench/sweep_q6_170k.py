#!/usr/bin/env python3
"""
sweep_q6_170k.py — campagne de mesure 06/11/2026 (GO opérateur 06/11 ~03:15).

Objectif : graphiques de l'apport des optimisations (S0 → S6) à différentes
tailles de contexte. MESURE uniquement — pas de recherche d'opti.

Config figée (contrat opérateur) :
  - modèle : qwen3.8-27b-q6-mtp (Q6_K, blob prod sha256-493301830a…)
  - mmproj : AUCUN
  - -c 170000 (borne haute : prompt 167568 + 2048 gén ≤ 170000 — ne pas dépasser)
  - binaire : /opt/PROD/lib/ollama/llama-server (prod)
  - GPU : V100 seul (--split-mode none --main-gpu 0, CUDA_VISIBLE_DEVICES=0)
  - KV q8_0/q8_0, FA auto
  - S7/H2 : EXCLU (consigne opérateur)
  - Le .so prod /opt/PROD/... N'EST JAMAIS TOUCHÉ : chaque serveur de
    bench charge GGML_BACKEND_PATH = sofiles/<state>.so directement.

États (S0→S6, delta cumulé) :
  S0 base pre-T3 : t3 d2df07b8 + FA_NO_VOLTA_SMALL=1, b512
  S1 +batch      : t3 d2df07b8 + FA_NO_VOLTA_SMALL=1, b2048
  S2 +T3         : t3 d2df07b8 (default), b2048
  S3 +lba4       : dual b014d026, b2048
  S4 +cvec       : cvec 723c0016, b2048
  S5 +vldg       : vldg 87f85442, b2048
  S6 +FA16       : dual d4954502 (PROD), + FA_COMPACT_TMAX=8192, b2048
  MTP ON ajoute : --spec-type draft-mtp --spec-draft-n-max 3
                  --spec-draft-backend-sampling + DEFER_PROMPT + shortlist

Contextes : 16384 32768 49152 65536 81920 98304 114688 131072 147456 163840 167568
            (11 points ; "170k" = prompt 167568 pour rester ≤ 170000 avec la gén)

Mesure par run (serveur frais) :
  R1 = 32 tok post-prefill froid (comparable rebench)
  R2 = 2048 tok → régime établi : OFF = médiane des fenêtres tg_3s ;
       ON = médiane des 6 DERNIÈRES fenêtres (plateau — le warm-up MTP des
       ~1200 premiers tok, acceptance qui monte, est retiré ; cf docs/
       sweep_q6_170k_summary.md)
  R3 = 32 tok chaud (diagnostique)
  acceptance MTP parsée.

Gates (rejet du run, pas de retry) :
  SM min ≥ 1150 MHz, temp max ≤ 93 °C, prompt_tokens_api dans ±0,5 % de la
  cible, n_gen R2 ≥ 2000, au moins 3 fenêtres tg_3s stables.

Anti-drift : chaque run S6/32768/on est un anchor ; Δ > 5 % vs le 1er anchor
  → run marqué drift_suspect (non rejeté).

Sortie : bench/data/sweep_q6_170k.jsonl (append, 1 ligne/run)
Idempotent : les runs déjà valides sont sautés (relance à tout moment).

USAGE (superviseur, un state à la fois) :
  python3 supervisor_bench.py run sweep_q6_170k.py S0 --detach
  python3 supervisor_bench.py run sweep_q6_170k.py S1 --detach
  ... S2 S3 S4 S5 S6
(Graphiques ensuite : python3 plot_q6_170k.py && python3 make_pdf.py)
"""
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request

from gpu_uuid_select import v100_cuda_ordinal, v100_smi_index

D = os.path.dirname(os.path.abspath(__file__))
# Chemins repo (llama-v100-opti) : D = bench/bench/ → données dans bench/data/
OUT = os.path.join(D, "..", "data")
os.makedirs(OUT, exist_ok=True)
RESULTS = os.path.join(OUT, "sweep_q6_170k.jsonl")
PROGRESS = os.path.join(OUT, "progress.json")
SERVER_LOG = os.path.join(OUT, "server.log")   # *.log → hors git (.gitignore)

# Chemins machine-dépendants (à adapter — cf README §6) :
GGUF = ("/usr/share/ollama/.ollama/models/blobs/"
        "sha256-493301830a596b8ad56dc1329f80bbcb578c8e910da395feafdc9cd8263430bb")
PROD_BIN = "/opt/PROD/lib/ollama/llama-server"
SO_PROD = "/opt/PROD/lib/ollama/cuda_v12/libggml-cuda.so"
SO_PROD_SHA = "6a35e164"   # à vérifier intact en début ET fin de campagne
SOFILES = os.path.join(D, "..", "..", "sofiles")   # sha256-manifesté, hors git

# prompts figés (canonical + manifests) dans bench/prompts/ — même dossier que
# manifest.json (points ≤ 107340) et manifest_ext.json (points 49k → 167568)
PROMPT_DIR = os.path.join(D, "..", "prompts")
PROMPT_DIRS = [PROMPT_DIR, PROMPT_DIR]

PORT = 8917
SERVICE = "ollama-v100"
MAX_CTX = 170000          # borne haute opérateur — NE PAS DÉPASSER
N_GEN_R2 = 2048
SM_FLOOR_MHZ = 1150
TEMP_CEIL_C = 93
COOLDOWN_S = 20
REQUEST_TIMEOUT = 1500
ANCHOR_DRIFT_PCT = 5.0

# env EXACT du runner prod (dump /proc 06/11 01:35) + le set d'opti
BASE_ENV = {
    "GGML_CUDA_VOLTA_Q6_W4R4": "1",
    "GGML_CUDA_VOLTA_Q8_FATTN_TC": "0",
    "LLAMA_ARG_FIT_TARGET": "1024",
    "LD_LIBRARY_PATH": "/opt/PROD/lib/ollama:/opt/PROD/lib/ollama/cuda_v12:/opt/PROD/cuda-12.6/x86_64-linux/lib",
}
MTP_ENV = {
    "LLAMA_ARG_SPEC_MTP_DEFER_PROMPT": "1",
    "GGML_CUDA_QWEN35_MTP_SHORTLIST": "/opt/PROD/runtime/qwen38-27b-exact-131072.i32",
}

# état -> (.so file dans SOFILES, sha256-8, env complet, batch)
STATES = {
    "S0": ("libggml-cuda-t3.so.d2df07b8", "597d20a3",
           dict(BASE_ENV, GGML_CUDA_FA_NO_VOLTA_SMALL="1"), 512),
    "S1": ("libggml-cuda-t3.so.d2df07b8", "597d20a3",
           dict(BASE_ENV, GGML_CUDA_FA_NO_VOLTA_SMALL="1"), 2048),
    "S2": ("libggml-cuda-t3.so.d2df07b8", "597d20a3",
           dict(BASE_ENV), 2048),
    "S3": ("libggml-cuda-dual.so.b014d026", "45f659b8",
           dict(BASE_ENV), 2048),
    "S4": ("libggml-cuda-cvec.so.723c0016", "8369d242",
           dict(BASE_ENV), 2048),
    "S5": ("libggml-cuda-vldg.so.87f85442", "cf852433",
           dict(BASE_ENV), 2048),
    "S6": ("libggml-cuda-dual.so.d4954502", "6a35e164",
           dict(BASE_ENV, GGML_CUDA_VOLTA_FA_COMPACT_TMAX="8192"), 2048),
}

CTXS = [16384, 32768, 49152, 65536, 81920, 98304,
        114688, 131072, 147456, 163840, 167568]

GPU_SMI_IDX = v100_smi_index()
GPU_CUDA_IDX = v100_cuda_ordinal()

_state = {"proc": None, "model_id": None}
_gpu_stop = threading.Event()
_gpu_win = {"lock": threading.Lock(), "sm": [], "temp": [], "util": []}


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def sha8(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]


def vram_mib():
    r = subprocess.run(["nvidia-smi", f"--id={GPU_SMI_IDX}",
                        "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=30)
    return int(r.stdout.strip())


def svc_state():
    return subprocess.run(["systemctl", "is-active", SERVICE],
                          capture_output=True, text=True).stdout.strip()


def stop_and_free():
    subprocess.run(["sudo", "systemctl", "stop", SERVICE], timeout=120, check=False)
    t0 = time.time()
    while time.time() - t0 < 240:
        if vram_mib() < 1024:
            return True
        time.sleep(3)
    raise RuntimeError(f"V100 pas libérée après stop (VRAM={vram_mib()} MiB)")


def restore_service():
    if svc_state() != "active":
        subprocess.run(["sudo", "systemctl", "start", SERVICE], timeout=120, check=False)
    for _ in range(120):
        if svc_state() == "active":
            return True
        time.sleep(2)
    return False


def on_sig(sig, frm):
    log(f"signal {sig} — restauration service")
    try:
        restore_service()
    finally:
        log(f"  · service = {svc_state()}")
    sys.exit(128 + sig)


def gpu_sampler():
    while not _gpu_stop.is_set():
        try:
            r = subprocess.run(
                ["nvidia-smi", f"--id={GPU_SMI_IDX}",
                 "--query-gpu=clocks.sm,temperature.gpu,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=15)
            sm, tp, ut = (int(x) for x in r.stdout.strip().split(", "))
            with _gpu_win["lock"]:
                _gpu_win["sm"].append(sm)
                _gpu_win["temp"].append(tp)
                _gpu_win["util"].append(ut)
        except Exception:
            pass
        time.sleep(1.0)


def gpu_window_reset():
    with _gpu_win["lock"]:
        _gpu_win["sm"], _gpu_win["temp"], _gpu_win["util"] = [], [], []


def gpu_window_summary():
    with _gpu_win["lock"]:
        sm, tp, ut = list(_gpu_win["sm"]), list(_gpu_win["temp"]), list(_gpu_win["util"])
    gpu_window_reset()
    if not sm:
        return {"sm_min": None, "temp_max": None, "util_median": None}
    return {"sm_min": min(sm), "temp_max": max(tp),
            "util_median": sorted(ut)[len(ut) // 2]}


def http_json(url, data=None, timeout=10):
    req = urllib.request.Request(
        url, data=json.dumps(data).encode() if data is not None else None,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def log_size():
    try:
        return os.path.getsize(SERVER_LOG)
    except OSError:
        return 0


def start_server(state, batch, mtp):
    so_file, so_sha, conf_env, _ = STATES[state]
    so_file_abs = os.path.join(SOFILES, so_file)
    if sha8(so_file_abs) != so_sha:
        raise RuntimeError(f"sha8 {so_file} = {sha8(so_file_abs)} != {so_sha}")
    if vram_mib() > 8192:
        raise RuntimeError(f"V100 pas vide (VRAM={vram_mib()} MiB)")
    env = dict(os.environ)
    env.update(conf_env)
    if mtp == "on":
        env.update(MTP_ENV)
    env["GGML_BACKEND_PATH"] = so_file_abs
    env["CUDA_VISIBLE_DEVICES"] = str(GPU_CUDA_IDX)

    args = [PROD_BIN,
            "--model", GGUF, "--port", str(PORT), "--host", "127.0.0.1",
            "--no-webui", "--offline",
            "-c", str(MAX_CTX), "-np", "1",
            "-b", str(batch), "-ub", str(batch),
            "--log-verbosity", "4", "--no-log-prefix", "--no-log-timestamps",
            "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
            "--flash-attn", "auto",
            "--split-mode", "none", "--main-gpu", "0"]
    if mtp == "on":
        args += ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3",
                 "--spec-draft-backend-sampling"]

    lf = open(SERVER_LOG, "a")
    proc = subprocess.Popen(args, stdout=lf, stderr=subprocess.STDOUT, env=env)
    _state["proc"] = proc
    t0 = time.time()
    while time.time() - t0 < 600:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server mort au démarrage (exit {proc.returncode})")
        try:
            http_json(f"http://127.0.0.1:{PORT}/health", timeout=3)
            break
        except Exception:
            time.sleep(2)
    else:
        raise RuntimeError("pas de /health en 600 s")
    time.sleep(3)
    if vram_mib() < 1024:
        raise RuntimeError(f"VRAM={vram_mib()} MiB — model en CPU ?")
    d = http_json(f"http://127.0.0.1:{PORT}/v1/models", timeout=10)
    _state["model_id"] = d["data"][0]["id"]
    log(f"  · server prêt ({time.time()-t0:.0f} s, VRAM {vram_mib()} MiB, "
        f"{so_file.split('.')[1][:8]}, b{batch}, mtp={mtp})")


def stop_server():
    p = _state.get("proc")
    if p and p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait(timeout=15)
    _state["proc"] = None
    time.sleep(6)
    for _ in range(60):
        if vram_mib() < 4096:
            break
        time.sleep(3)


def send_stream(max_tokens, prompt):
    payload = {
        "model": _state["model_id"],
        "stream": True,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0, "top_k": 1, "ignore_eos": True,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    t_first = None
    n_content = 0
    usage = None
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            try:
                j = json.loads(body)
            except json.JSONDecodeError:
                continue
            if j.get("usage"):
                usage = j["usage"]
            for ch in j.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("content"):
                    if t_first is None:
                        t_first = time.time()
                    n_content += 1
    t_end = time.time()
    return {"ttft_s": round(t_first - t0, 2) if t_first else None,
            "wall_s": round(t_end - t0, 2),
            "n_content_chunks": n_content,
            "usage": usage}


def parse_slice(a, b):
    out = {}
    with open(SERVER_LOG, errors="replace") as f:
        f.seek(a)
        data = f.read(b - a)
    m = re.search(r"prompt eval time =\s+([\d.]+) ms /\s*(\d+) tokens", data)
    if m:
        out["prompt_ms"] = float(m.group(1))
        out["prompt_tokens"] = int(m.group(2))
        out["prompt_tps"] = round(int(m.group(2)) / (float(m.group(1)) / 1000.0), 2)
    m = re.search(r"(?<!prompt )eval time =\s+([\d.]+) ms /\s*(\d+) tokens", data)
    if m:
        out["eval_ms"] = float(m.group(1))
        out["eval_tokens"] = int(m.group(2))
        out["eval_tps"] = round(int(m.group(2)) / (float(m.group(1)) / 1000.0), 2)
    out["tg"] = [(int(g[0]), float(g[1]), float(g[2]))
                 for g in re.findall(r"n_gen =\s*(\d+), tg =\s+([\d.]+) t/s, tg_3s =\s+([\d.]+) t/s", data)]
    m = re.search(r"draft acceptance = ([\d.]+) \(\s*(\d+) accepted /\s*(\d+) generated\), mean len =\s+([\d.]+)", data)
    if m:
        out["mtp_acceptance"] = float(m.group(1))
        out["mtp_mean_len"] = float(m.group(4))
    return out


def steady_from_tg(tg):
    """Régime établi : médiane des tg_3s des fenêtres n_gen≥100,
    1re fenêtre exclue (transitoire post-prefill / replay MTP)."""
    sel = [v for (n, _t, v) in tg if n >= 100]
    if not sel:
        return None
    body = sel[1:] if len(sel) > 1 else sel
    body_sorted = sorted(body)
    return round(body_sorted[len(body_sorted) // 2], 2)


def load_prompts():
    """{ctx: (text, sha256)} — prompts_ext (neufs) puis prompts (rebench)."""
    mfx = json.load(open(os.path.join(PROMPT_DIRS[0], "manifest_ext.json")))
    mfo = json.load(open(os.path.join(PROMPT_DIRS[1], "manifest.json")))
    out = {}
    for ctx in CTXS:
        if str(ctx) in mfx:
            src_dir, mf = PROMPT_DIRS[0], mfx
        else:
            src_dir, mf = PROMPT_DIRS[1], mfo
        key = str(ctx)
        if key not in mf:
            raise RuntimeError(f"prompt {ctx} absent des manifests")
        sha = mf[key]["prompt_sha256"]
        text = open(os.path.join(src_dir, f"{ctx}.txt"), encoding="utf-8").read()
        got = hashlib.sha256(text.encode()).hexdigest()
        if got != sha:
            raise RuntimeError(f"prompt {ctx} sha256 {got[:16]} != {sha[:16]}")
        out[ctx] = (text, sha)
    return out


def done_valid_set(state):
    s = set()
    if os.path.exists(RESULTS):
        for line in open(RESULTS):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (r.get("state") == state and not r.get("rejected")
                    and r.get("valid") and r.get("steady_tps") is not None):
                s.add((r.get("ctx_target"), r.get("mtp")))
    return s


def load_progress():
    if os.path.exists(PROGRESS):
        try:
            return json.load(open(PROGRESS))
        except json.JSONDecodeError:
            pass
    return {"n_runs": 0, "anchors": []}


def save_progress(p):
    with open(PROGRESS, "w") as f:
        json.dump(p, f, indent=1)


def measure_one(state, ctx, mtp, prompts):
    """Un run complet (serveur frais). Retourne le record JSON."""
    so_file, so_sha, conf_env, batch = STATES[state]
    prompt, psha = prompts[ctx]

    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "state": state, "mtp": mtp, "ctx_target": ctx,
           "prompt_sha256": psha, "so_file": so_file, "so_sha8": so_sha,
           "batch": batch, "max_ctx": MAX_CTX,
           "prompt_tokens_api": None, "eval_tokens_r2": None,
           "r1": {}, "r2": {}, "r3": {}, "steady_tps": None,
           "prefill_tps": None, "mtp_acceptance": None, "mtp_mean_len": None,
           "gpu": {}, "rejected": False, "reject_reason": None}

    start_server(state, batch, mtp)
    gpu_window_reset()

    try:
        # R1 — 32 tok, prefill froid
        a = log_size()
        api1 = send_stream(32, prompt)
        time.sleep(1.0)
        met1 = parse_slice(a, log_size())
        rec["r1"] = {"api": api1, "log": met1}
        rec["prefill_tps"] = met1.get("prompt_tps")
        time.sleep(COOLDOWN_S)

        # R2 — 2048 tok, régime établi
        a = log_size()
        api2 = send_stream(N_GEN_R2, prompt)
        time.sleep(1.0)
        met2 = parse_slice(a, log_size())
        rec["r2"] = {"api": api2, "log": met2}
        rec["eval_tokens_r2"] = met2.get("eval_tokens")
        rec["mtp_acceptance"] = met2.get("mtp_acceptance")
        rec["mtp_mean_len"] = met2.get("mtp_mean_len")
        rec["steady_tps"] = steady_from_tg(met2.get("tg", []))

        time.sleep(COOLDOWN_S)

        # R3 — 32 tok chaud
        a = log_size()
        api3 = send_stream(32, prompt)
        time.sleep(1.0)
        met3 = parse_slice(a, log_size())
        rec["r3"] = {"api": api3, "log": met3}

        # source de vérité = prompt_tokens du log serveur (l'usage api est absent
        # du stream). Tolerance ±0.5% de la cible ctx (template chat ~52 tok).
        rec["prompt_tokens_api"] = met1.get("prompt_tokens")

        # gates
        rec["gpu"] = gpu_window_summary()
        reasons = []
        pt = rec["prompt_tokens_api"]
        if pt is None:
            reasons.append("pas de prompt_tokens (log)")
        elif abs(pt - ctx) / ctx > 0.005:
            reasons.append(f"prompt_tokens {pt} hors ±0.5% de {ctx}")
        if (rec["eval_tokens_r2"] or 0) < 2000:
            reasons.append(f"R2 n_gen={rec['eval_tokens_r2']} < 2000")
        if len(met2.get("tg", [])) < 3:
            reasons.append(f"fenêtres tg_3s : {len(met2.get('tg', []))} < 3")
        if rec["steady_tps"] is None:
            reasons.append("pas de régime établi")
        if rec["gpu"]["sm_min"] is not None and rec["gpu"]["sm_min"] < SM_FLOOR_MHZ:
            reasons.append(f"SM min {rec['gpu']['sm_min']} < {SM_FLOOR_MHZ}")
        if rec["gpu"]["temp_max"] is not None and rec["gpu"]["temp_max"] > TEMP_CEIL_C:
            reasons.append(f"temp max {rec['gpu']['temp_max']} > {TEMP_CEIL_C}")
        if reasons:
            rec["rejected"] = True
            rec["reject_reason"] = " | ".join(reasons)
        rec["valid"] = not rec["rejected"]
    finally:
        stop_server()
    return rec


def write_record(rec):
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec) + "\n")


def main():
    if len(sys.argv) not in (2, 3) or sys.argv[1] not in STATES:
        print(f"usage : {sys.argv[0]} <S0|S1|S2|S3|S4|S5|S6> [retry]")
        return 2
    state = sys.argv[1]
    RETRY = len(sys.argv) == 3 and sys.argv[2] == "retry"

    signal.signal(signal.SIGTERM, on_sig)
    signal.signal(signal.SIGINT, on_sig)

    # garde-fous d'entrée
    assert sha8(SO_PROD) == SO_PROD_SHA, \
        f".so prod = {sha8(SO_PROD)} != {SO_PROD_SHA} — NE PAS LANCER"
    prompts = load_prompts()
    log(f"prompts chargés : {len(prompts)} ctx, sha256 OK | .so prod intact ({sha8(SO_PROD)})")

    so_file, so_sha, conf_env, batch = STATES[state]
    extra = {k: v for k, v in conf_env.items() if k not in BASE_ENV}
    log(f"état {state} : {so_file} (sha8 {so_sha}), batch {batch}, env+={extra or 'BASE'}")

    rc = 0
    try:
        stop_and_free()
        threading.Thread(target=gpu_sampler, daemon=True).start()

        plan = [(ctx, mtp) for ctx in CTXS for mtp in ("off", "on")]
        done = done_valid_set(state)
        prog = load_progress()
        if RETRY:
            # retry : on ne rejoue QUE les runs rejetés (pas les valides)
            rej = set()
            if os.path.exists(RESULTS):
                for line in open(RESULTS):
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if r.get("state") == state and r.get("rejected"):
                        rej.add((r.get("ctx_target"), r.get("mtp")))
            # dédupliquer : un run rejeté ET non valide doit être rejoué ;
            # garder le dernier record pour chaque (ctx,mtp)
            last = {}
            if os.path.exists(RESULTS):
                for line in open(RESULTS):
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if r.get("state") == state:
                        last[(r.get("ctx_target"), r.get("mtp"))] = r
            rej = {k for k, r in last.items() if r.get("rejected")}
            todo = [p for p in plan if p in rej]
            log(f"RETRY {state} : {len(todo)} runs rejetés à rejouer : {todo}")
            if not todo:
                log("  · rien à rejouer")
                return 0
        n_todo = len([p for p in plan if p not in done])
        log(f"plan {state} : {len(plan)} runs, déjà valides : {len(done)}, à faire : {n_todo}" + (" [RETRY]" if RETRY else ""))

        for i, (ctx, mtp) in enumerate(plan, 1):
            key = f"{state}/{ctx}/{mtp}"
            if (ctx, mtp) in done:
                log(f"[{i}/{len(plan)}] {key} : déjà validé, skip")
                continue
            if RETRY and (ctx, mtp) not in rej:
                log(f"[{i}/{len(plan)}] {key} : non rejette, skip [retry]")
                continue
            log(f"[{i}/{len(plan)}] START {key}")
            t0 = time.time()
            try:
                rec = measure_one(state, ctx, mtp, prompts)
            except Exception as e:
                import traceback
                log(f"  · ERREUR run {key} : {e}\n{traceback.format_exc()}")
                write_record({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                              "state": state, "ctx_target": ctx, "mtp": mtp,
                              "so_file": so_file, "so_sha8": so_sha, "batch": batch,
                              "rejected": True, "valid": False,
                              "reject_reason": f"exception: {e}"})
                rc = 1
                prog["n_runs"] += 1
                save_progress(prog)
                continue

            # anchor anti-drift (S6/32768/on)
            if rec["valid"] and state == "S6" and (ctx, mtp) == (32768, "on"):
                if prog["anchors"]:
                    a0 = prog["anchors"][0].get("steady_tps")
                    if a0 and rec["steady_tps"]:
                        drift = (rec["steady_tps"] - a0) / a0 * 100
                        rec["anchor_drift_pct"] = round(drift, 2)
                        if abs(drift) > ANCHOR_DRIFT_PCT:
                            rec["drift_suspect"] = True
                            log(f"  · ANCHOR DRIFT {drift:+.1f}% (>{ANCHOR_DRIFT_PCT}%) — suspect")
                        else:
                            log(f"  · anchor ok ({drift:+.1f}%)")
                else:
                    rec["anchor"] = "first"
                    log("  · anchor initial posé")
                prog["anchors"].append({"steady_tps": rec["steady_tps"], "ts": rec["ts"]})

            write_record(rec)
            prog["n_runs"] += 1
            save_progress(prog)
            st = "REJET" if rec["rejected"] else "OK"
            log(f"[{i}/{len(plan)}] {key} : {st} steady={rec['steady_tps']} t/s "
                f"r1={rec['r1']['log'].get('eval_tps')} prefill={rec['prefill_tps']} "
                f"acc={rec['mtp_acceptance']} util={rec['gpu'].get('util_median')}% "
                f"({time.time()-t0:.0f}s)"
                + (f" — {rec['reject_reason']}" if rec["rejected"] else ""))

        # garde-fou de sortie : .so prod intact
        so_now = sha8(SO_PROD)
        if so_now != SO_PROD_SHA:
            log(f"ERREUR CRITIQUE : .so prod = {so_now} != {SO_PROD_SHA} — VERIFIER")
            rc = 1
        else:
            log(f".so prod intact en fin ({so_now})")
        log(f"=== FIN {state} : rc={rc} ===")
        return rc
    finally:
        _gpu_stop.set()
        restored = restore_service()
        log(f"  · service ollama-v100 = {svc_state()}")
        if not restored:
            log("ERREUR : service non restauré — VERIFIER IMMEDIATEMENT")


if __name__ == "__main__":
    sys.exit(main())
