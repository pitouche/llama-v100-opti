#!/usr/bin/env python3
# run9e_bench.py — RUN9E : capture CUPTI en régime FRAIS @ ctx ~90k — SANS MTP
#
# Contexte (24/10 08:00 UTC, user : plus de bench MTP, revalider en MTP-OFF) :
#   Toutes nos captures (run9b/c/d) sont en MTP-ON. Jamais eu de baseline MTP-OFF.
#   En MTP-OFF : pas de draft/verify, pas de refresh cache f16, pas de mapped_rows MTP,
#   mais K/V quantize_q8_1 par token + FA vanilla. La répartition du busy CHANGE.
#   Objectif : le breakdown kernel MTP-OFF du régime 90k → revalider les CLOS.
#
# Protocol : IDENTIQUE run9d (CUPTI v2 LD_PRELOAD, port 44912, .so prod fa3e75c2,
# A[0:380000] ~90k 32 gen chauffe / B[600000:980000] ~90k 512 gen FENÊTRE, KV q8_0,
# FA auto, -b 2048, temp 0 top_k 1) — SANS :
#   - --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-backend-sampling
#   - env GGML_CUDA_QWEN35_MTP_SHORTLIST, LLAMA_ARG_SPEC_MTP_DEFER_PROMPT
#   - GATE SHORTLIST
#   (W4R4 + Q8_FATTN_TC + FIT_TARGET restent = config prod hors-MTP)
#
# STOPPE le service ollama-v100, se restaure (finally).
# USAGE : python3 supervisor_bench.py run run9e_bench.py [ntok] --detach
import json, os, re, subprocess, sys, time, urllib.request
from gpu_uuid_select import v100_cuda_ordinal, v100_smi_index  # UUID GPU (jamais d'index en dur — voir GPU_UUID_NSYS_CUPTI_20261030.md)

ROOT = "/home/guile/llama-experiments"
CUPATH = f"{ROOT}/h2/cupti129-test"
PORT = 44912
NTOK = int(sys.argv[1]) if len(sys.argv) > 1 else 512
TS = time.strftime("%Y%m%d-%H%M%S")
OUT = f"/tmp/cupath9e_{TS}"
RUNDIR = f"{ROOT}/bench/runs/cupti9e-{TS}"
os.makedirs(RUNDIR, exist_ok=True)
API = f"http://127.0.0.1:{PORT}"
GGUF = "/usr/share/ollama/.ollama/models/blobs/sha256-493301830a596b8ad56dc1329f80bbcb578c8e910da395feafdc9cd8263430bb"
FILLER = f"{ROOT}/v100-repro/work/filler-200k.txt"
SOUT = f"/tmp/cupath9e_server.out"
MODEL = "qwen3.8-27b-q6-mtp-vision:latest"
MODEL_GGUF = GGUF

def log(m):
    print(time.strftime("[%H:%M:%S] ") + m, flush=True)

def run(c, t=120):
    return subprocess.run(c, capture_output=True, text=True, timeout=t)

def vram():
    r = run(["nvidia-smi", "--id=" + v100_smi_index(), "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    v = r.stdout.strip(); return int(v) if v.isdigit() else -1

def http(url, data=None, t=3600):
    req = urllib.request.Request(url, data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=t) as r:
        return json.loads(r.read().decode())

def restore():
    log("[restore] nettoyage server de capture")
    run(["pkill", "-f", f"port {PORT}"]); time.sleep(3)
    for _ in range(60):
        if vram() < 1000: break
        run(["pkill", "-9", "-f", "llama-server"]); time.sleep(1)
    run(["sudo", "systemctl", "start", "ollama-v100"], t=180)
    for _ in range(120):
        try:
            http("http://127.0.0.1:11434/api/version", t=3); break
        except Exception:
            time.sleep(1)
    log(f"[restore] service={subprocess.run(['systemctl','is-active','ollama-v100'],capture_output=True,text=True).stdout.strip()} "
        f"V100={vram()}MiB md5={subprocess.run(['md5sum',ROOT+'/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so'],capture_output=True,text=True).stdout.split()[0][:8]}")

rc = 1
try:
    import hashlib
    so = f"{ROOT}/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so"
    md = hashlib.md5(open(so, "rb").read()).hexdigest()[:8]
    assert md == "fa3e75c2", f"prod .so = {md} != fa3e75c2 — ABORT"
    log(f"prod md5 avant : {md} (MTP OFF : sans args spec)")

    filler = open(FILLER, encoding="utf-8", errors="replace").read()
    assert len(filler) > 980000, "filler trop court"
    promptA = filler[0:380000]
    promptB = filler[600000:980000]
    pref = 0
    for a, b in zip(promptA, promptB):
        if a == b: pref += 1
        else: break
    assert pref < 500, f"préfixe commun A/B = {pref} chars — B pas frais"
    log(f"préfixe commun A/B = {pref} chars (B KV-fraiche OK)")
    def mkp(text, ntok, f):
        json.dump({"model": MODEL, "stream": False, "temperature": 0,
                   "top_k": 1, "ignore_eos": True, "max_tokens": ntok,
                   "chat_template_kwargs": {"enable_thinking": False},
                   "messages": [{"role": "user", "content": text}]}, open(f, "w"), ensure_ascii=False)
    mkp(promptA, 32, "/tmp/cupath9e_A.json")
    mkp(promptB, NTOK, "/tmp/cupath9e_B.json")
    log(f"prompts : A=filler[0:380000] (~90k, 32 gen, hors fenêtre) B=filler[600000:980000] (~90k, {NTOK} gen, FENÊTRE)")

    run(["sudo", "systemctl", "stop", "ollama-v100"], t=180)
    t0 = time.time()
    while time.time() - t0 < 180:
        if vram() < 1000: break
        run(["pkill", "-9", "-f", "llama-server"]); time.sleep(1)
    log(f"V100 libre : {vram()} MiB")

    env = dict(os.environ)
    env.update({
        "LD_LIBRARY_PATH": (f"{ROOT}/ollama-v100/lib/ollama:{ROOT}/ollama-v100/lib/ollama/cuda_v12:"
                            f"{ROOT}/tools/cuda-12.6/usr/local/cuda-12.6/lib64:"
                            f"{ROOT}/tools/cuda-12.6/usr/local/cuda-12.6/targets/x86_64-linux/lib"),
        "GGML_BACKEND_PATH": so,
        "GGML_CUDA_VOLTA_Q6_W4R4": "1",
        "GGML_CUDA_VOLTA_Q8_FATTN_TC": "0",
        "LLAMA_ARG_FIT_TARGET": "1024",
        "CUDA_VISIBLE_DEVICES": str(v100_cuda_ordinal()),  # UUID, pas en dur
        "LD_PRELOAD": f"{CUPATH}/libcupath_v2.so",
        "CUPATH_TRACE": OUT,
    })
    lf = open(SOUT, "w")
    args = [f"{ROOT}/ollama-v100/lib/ollama/llama-server", "--model", MODEL_GGUF, "--port", str(PORT),
            "--host", "127.0.0.1", "--no-webui", "-c", "131000", "-np", "1", "-lv", "4",
            "--no-log-prefix", "--no-log-timestamps",
            # MTP OFF : aucun --spec-type
            "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--flash-attn", "auto",
            "-b", "2048", "-ub", "2048"]
    proc = subprocess.Popen(args, stdout=lf, stderr=subprocess.STDOUT, env=env)
    spid = proc.pid
    log(f"llama-server pid={spid} (MTP OFF)")

    def die(msg):
        global rc
        log(msg); rc = 2
        raise SystemExit(2)

    ok = 0
    for _ in range(300):
        try:
            http(f"{API}/v1/models", t=3); ok = 1; break
        except Exception:
            time.sleep(1)
    if not ok: die("server pas up (port)")
    log("server up (port)")

    # GATE FIT
    fitline = ""
    for _ in range(300):
        txt = open(SOUT, errors="replace").read()
        if re.search(r"fit_params: failed|unable to fit", txt):
            die("FIT ABORT")
        m = re.search(r"successfully fit params.*", txt)
        if m: fitline = m.group(0); break
        if proc.poll() is not None: die("server mort")
        time.sleep(1)
    if not fitline: die("fit non confirmé")
    log(f"GATE FIT OK : {fitline[:120]}")

    # GATE DEVICE
    devline = ""
    for _ in range(30):
        txt = open(SOUT, errors="replace").read()
        m = re.search(r"using device CUDA.*", txt)
        if m: devline = m.group(0); break
        time.sleep(1)
    if "Tesla V100" not in devline: die(f"device != V100 : {devline}")
    log("GATE DEVICE OK (V100)")

    # GATE MODEL LOADED
    mlok = 0
    for _ in range(900):
        txt = open(SOUT, errors="replace").read()
        if "llama_server: model loaded" in txt: mlok = 1; break
        if "out of memory" in txt: die("OOM")
        if proc.poll() is not None: die("server mort pendant load")
        time.sleep(1)
    if not mlok: die("model non chargé")
    log("GATE MODEL LOADED OK")

    # GATE SANS MTP (le server doit signaler qu'il n'y a pas de spec)
    for _ in range(30):
        txt = open(SOUT, errors="replace").read()
        if "spec" in txt.lower() and re.search(r"spec.*type.*=|speculative", txt, re.I):
            m = re.search(r".*spec.*", txt, re.I)
            if m: log(f"GATE MTP : {m.group(0)[:100]}")
        time.sleep(1)

    # WARMUP 16 tok (hors fenêtre)
    warm_ok = 0
    for w in range(5):
        try:
            r = http(f"{API}/v1/chat/completions",
                     {"model": MODEL, "stream": False, "temperature": 0,
                      "max_tokens": 16, "chat_template_kwargs": {"enable_thinking": False},
                      "messages": [{"role": "user", "content": "Say OK."}]}, t=1800)
            if r and "error" not in json.dumps(r): warm_ok = 1; break
        except Exception:
            pass
        time.sleep(10)
    if not warm_ok: die("warmup KO")
    log("warmup 16 tok done (hors fenêtre)")

    # REQ A — hors fenêtre (préfill 90k + 32 gen : chauffe graphs decode 90k)
    log("REQ A (hors fenêtre) : prompt A ~90k → 32 gen")
    rA = http(f"{API}/v1/chat/completions", json.load(open("/tmp/cupath9e_A.json")), t=3600)
    if rA.get("error") or "finish_reason" not in json.dumps(rA): die("req A échec")
    log(f"REQ A done : prompt {rA.get('usage',{}).get('prompt_tokens')} gen {rA.get('usage',{}).get('completion_tokens')}")

    # FENÊTRE — REQ B : prompt DIFFÉRENT ~90k, KV fraîche, graphs chauds
    log(f"FENÊTRE : REQ B prompt B ~90k → {NTOK} gen (KV FRAÎCHE)")
    TS_START = time.time_ns() // 1000
    rB = http(f"{API}/v1/chat/completions", json.load(open("/tmp/cupath9e_B.json")), t=3600)
    time.sleep(2)
    TS_END = time.time_ns() // 1000
    with open("/tmp/cupath9e_bracket", "w") as f:
        f.write(f"{TS_START} {TS_END}\n")
    if rB.get("error") or "finish_reason" not in json.dumps(rB): die(f"generation KO : {json.dumps(rB)[:200]}")
    pb = rB.get("usage", {})
    log(f"FENÊTRE done : prompt {pb.get('prompt_tokens')} gen {pb.get('completion_tokens')}")

    with open(RUNDIR + "/reqB.json", "w") as f: json.dump(rB, f)
    time.sleep(3)
    proc.terminate()
    try: proc.wait(timeout=30)
    except Exception: proc.kill()

    # ---- traces + analyse ----
    cands = [f for f in os.listdir(OUT) if f.endswith(".names")]
    if not cands: die("pas de trace .names")
    trc = os.path.join(OUT, max(cands, key=lambda x: os.path.getsize(os.path.join(OUT, x)))[:-len(".names")])
    for fn in (trc, trc + ".names", SOUT, "/tmp/cupath9e_bracket", "/tmp/cupath9e_A.json", "/tmp/cupath9e_B.json"):
        if os.path.exists(fn):
            subprocess.run(["cp", fn, RUNDIR + "/"], check=False)

    wall = re.findall(r"(?<!prompt )eval time = +([\d.]+) ms / +(\d+) tokens", open(SOUT, errors="replace").read())
    WALL_MS = wall[-1][0] if wall else ""
    log(f"wall serveur (eval B) : {WALL_MS} ms")
    log("=== analyse v2 (window régime FRAIS 90k, MTP OFF) ===")
    p = subprocess.run(["python3", f"{CUPATH}/cupath_analyze_v2.py", trc, "window",
                        str(TS_START), str(TS_END), WALL_MS], capture_output=True, text=True, timeout=1800)
    an = p.stdout
    with open(RUNDIR + "/analyse.txt", "w") as f: f.write(an + "\nstderr:\n" + p.stderr)
    print(an[-4000:], flush=True)
    rc = 0
finally:
    restore()
sys.exit(rc)
