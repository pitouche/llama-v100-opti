#!/usr/bin/env python3
"""
nsys_mtp_breakdown.py — DECOMPOSITION CUDA d'un vrai cycle decode MTP (prod lba).

Question : dans un pas de decode MTP complet, quelle part prend le kernel FA
(volta_small), les GEMMs (q6-w4r4), les kernels MTP (draft/verify) — et y a-t-il
des gaps CPU-bound ? C'est la mesure qui dit si le chantier "FA T=4" est le bon.

Méthode (pattern test_nsys_start.sh, validé) :
  1. stop prod, attendre V100 libre
  2. nsys launch --trace=cuda,osrt --cuda-graph-trace=node : (node = kernels de graph
     exposés individuellement — indispensable pour la décomp par kernel, cf.
     GPU_UUID_NSYS_CUPTI_20261030.md ; avec 'graph' les kernels de cudaGraphLaunch
     n'apparaissent pas individuels et le verify MTP — le plus gros — serait invisible)
     llama-server MTP ON (mêmes args que prod : -c 128000 -b 2048, spec-draft-n-max 3,
     KV q8_0, FA auto, H2 OFF, Q6_W4R4 ON), port 8918
  3. warmup hors capture (model loaded + 32 tok)
  4. nsys start → requête (prompt ~16k + gen 256 tok) → nsys stop
     (window = prefill + ~256 pas MTP ; les kernels se séparent par nom)
  5. nsys export sqlite + analyse : top kernels, part FA, part GEMM, gaps
  6. restauration du service (TOUJOURS, trap EXIT)

USAGE :
  cd /home/guile/llama-experiments/bench
  python3 supervisor_bench.py run nsys_mtp_breakdown.py --detach
"""
import glob, hashlib, json, os, re, signal, socket, sqlite3, subprocess, sys, time, urllib.request

D = os.path.dirname(os.path.abspath(__file__))


def _v100_cuda_ordinal():
    """Ordinal CUDA du V100 via UUID (repli 0 = état 30/10, logged)."""
    try:
        r = subprocess.run(["python3", os.path.join(D, "gpu_uuid.py"), "--cvt-ordinal"],
                           capture_output=True, text=True, timeout=30)
        v = r.stdout.strip()
        if v.isdigit():
            return int(v)
    except Exception:
        pass
    return 0


def _v100_smi_idx():
    """Index nvidia-smi du V100 via UUID (repli 1 = état 30/10, logged)."""
    try:
        r = subprocess.run(["python3", os.path.join(D, "gpu_uuid.py"), "--smi-index"],
                           capture_output=True, text=True, timeout=30)
        v = r.stdout.strip()
        if v.isdigit():
            return v
    except Exception:
        pass
    return "1"
# nsys 2025.5.2 — sans NSYS_CUPTI_LIBRARY_PATH (CUPTI 12.9) il charge sa libcupti.so.13.1
# → CUPTI_ERROR_INVALID_DEVICE, 0 événement CUDA sur driver 580.178.04 (voir
# GPU_UUID_NSYS_CUPTI_20261030.md). L'env est forcé juste avant le Popen du server.
NSYS = "/opt/nvidia/nsight-systems/2025.5.2/target-linux-x64/nsys"
CUPTI129_DIR = os.path.join(D, "nsys-cupti129")
LLAMA_SERVER = "/home/guile/llama-experiments/ollama-v100/lib/ollama/llama-server"
GGUF   = ("/usr/share/ollama/.ollama/models/blobs/"
          "sha256-493301830a596b8ad56dc1329f80bbcb578c8e910da395feafdc9cd8263430bb")
MMPROJ = ("/usr/share/ollama/.ollama/models/blobs/"
          "sha256-ac3714bfdddeca31351f2752bf1a63f266f4df87c0b68c895e44945ca704448e")
SHORTLIST = "/home/guile/llama-experiments/src/llama-volta-merged/data/mtp-shortlists/qwen38-27b-exact-131072.i32"
SO_PROD = "/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so"
PORT = 8918
SERVICE = "ollama-v100"
RES = os.path.join(D, "runs", "nsys-mtp-breakdown-" + time.strftime("%Y%m%d-%H%M%S"))
OUT_SUMMARY = os.path.join(D, "nsys_mtp_breakdown_summary.md")
CUDA_LIBS = [
    "/home/guile/llama-experiments/ollama-v100/lib/ollama",
    "/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12",
    "/home/guile/llama-experiments/tools/cuda-12.6/usr/local/cuda-12.6/lib64",
    "/home/guile/llama-experiments/tools/cuda-12.6/usr/local/cuda-12.6/targets/x86_64-linux/lib",
]
SESSION = "h2mtpbrk"
PROMPT_WORDS = 16000 * 4 // 5     # ~16k tokens
GEN_TOK = 256

_state = {}
_restored = False


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)

def run(cmd, timeout=60):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

def svc_state():
    return run(["systemctl", "is-active", SERVICE]).stdout.strip()

def vram_mib():
    r = run(["nvidia-smi", "--id=" + _v100_smi_idx(), "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"])
    v = r.stdout.strip()
    return int(v) if v.isdigit() else -1

def v100_proc_gone():
    r = run(["nvidia-smi", "--id=" + _v100_smi_idx(), "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader"])
    for line in r.stdout.splitlines():
        if line.strip():
            parts = line.split(",")
            if len(parts) >= 2 and parts[1].strip().replace("MiB", "").replace(" ", "").isdigit():
                if int(parts[1].strip().split()[0]) > 8000:
                    return False
    return True

def so_prefix(p):
    try:
        return hashlib.md5(open(p, "rb").read()).hexdigest()[:8]
    except OSError:
        return "?"

def restore_service():
    global _restored
    try:
        # tuer tout residu du serveur de bench
        subprocess.run(["pkill", "-9", "-f", f"port {PORT}"], timeout=30)
        p = _state.get("proc")
        if p and p.poll() is None:
            p.kill()
            try: p.wait(timeout=10)
            except Exception: pass
        time.sleep(3)
        for _ in range(60):
            if vram_mib() < 1024 and v100_proc_gone():
                break
            subprocess.run(["pkill", "-9", "-f", "llama-server"], timeout=30)
            time.sleep(2)
        if svc_state() != "active":
            log(f"  · restauration : systemctl start {SERVICE}")
            run(["sudo", "systemctl", "start", SERVICE], timeout=120)
        for _ in range(90):
            if svc_state() == "active":
                break
            time.sleep(2)
        _restored = svc_state() == "active"
        log(f"  · service {SERVICE} = {svc_state()}, VRAM {vram_mib()} MiB")
    except Exception as e:
        log(f"  · ERREUR restauration : {e}")
        _restored = False

def on_sig(sig, frm):
    restore_service()
    sys.exit(128 + sig)

def stop_and_free():
    log("  · stop " + SERVICE)
    run(["sudo", "systemctl", "stop", SERVICE], timeout=120)
    t0 = time.time()
    while time.time() - t0 < 180:
        if vram_mib() < 1024 and v100_proc_gone():
            log(f"  · V100 libérée ({vram_mib()} MiB) en {time.time()-t0:.0f}s")
            return True
        time.sleep(3)
    log(f"  · ATTENTION V100 pas totalement libérée (VRAM={vram_mib()} MiB)")
    return False

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

def prompt_text(n_words):
    fill = ("La tour de contrôle surveille le passage des avions au crépuscule "
            "pendant que le chef de piste organise les départs de la soirée. ")
    words = fill.split()
    return " ".join([words[i % len(words)] for i in range(n_words)])

def chat(data, timeout=900):
    try:
        return http_json(f"http://127.0.0.1:{PORT}/v1/chat/completions", data, timeout=timeout)
    except Exception as e:
        return {"error": str(e)}

def analyze_sqlite(sqlite_path):
    """Top kernels + parts FA/GEMM/autre + gaps GPU."""
    c = sqlite3.connect(sqlite_path)
    tabs = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    out = {"tables_cuda": [t for t in tabs if "KERNEL" in t.upper() or "CUPTI" in t.upper()]}
    ktab = None
    for t in tabs:
        if t.upper().endswith("CUPTI_ACTIVITY_KIND_KERNEL"):
            ktab = t; break
    if not ktab:
        out["error"] = "pas de table CUPTI_ACTIVITY_KIND_KERNEL"
        return out
    n = c.execute(f'SELECT COUNT(*) FROM "{ktab}"').fetchone()[0]
    out["n_kernels"] = n
    if n == 0:
        out["error"] = "aucun event kernel"
        return out
    # window
    w0, w1 = c.execute(f'SELECT MIN(start), MAX(end) FROM "{ktab}"').fetchone()
    window_us = (w1 - w0) / 1000.0
    out["window_s"] = round(window_us / 1e6, 2)
    # temps par nom (µs)
    rows = c.execute(
        f'SELECT demangledName, SUM(end-start)/1000.0, COUNT(*) '
        f'FROM "{ktab}" GROUP BY demangledName ORDER BY 2 DESC LIMIT 25').fetchall()
    total_us = sum(r[1] for r in rows)
    out["total_kernel_us"] = round(total_us)
    out["top"] = [{"name": (r[0] or "")[:110], "us": round(r[1]), "n": r[2]} for r in rows]
    # catégorisation
    def cat(name):
        n2 = name.lower()
        if "volta_small" in n2: return "FA_T3_volta_small"
        if "flash_attn" in n2: return "FA_autre"
        if "q6" in n2 or "w4r4" in n2 or "matmul" in n2 or "gemm" in n2: return "GEMM"
        if "moe" in n2: return "MoE"
        if "mtp" in n2 or "draft" in n2 or "spec" in n2: return "MTP_draft"
        if "softmax" in n2 or "norm" in n2 or "rms" in n2: return "norm/softmax"
        if "memcpy" in n2 or "copy" in n2: return "copy"
        return "autre"
    cats = {}
    for r in rows:
        ckey = cat(r[0] or "")
        cats[ckey] = cats.get(ckey, 0.0) + r[1]
    out["cats_us"] = {k: round(v) for k, v in sorted(cats.items(), key=lambda x: -x[1])}
    out["cats_pct"] = {k: round(v / total_us * 100, 1) for k, v in
                       sorted(cats.items(), key=lambda x: -x[1])}
    # busy GPU + gaps : intervalles triés, overlap (plusieurs streams)
    spans = c.execute(f'SELECT start, end FROM "{ktab}"').fetchall()
    spans.sort()
    busy_us = 0.0
    gaps = []
    cur_s, cur_e = spans[0]
    for s, e in spans[1:]:
        if s <= cur_e:
            if e > cur_e: cur_e = e
        else:
            busy_us += cur_e - cur_s
            if s - cur_e > 2000:  # gap > 2 ms
                gaps.append((s - cur_e) / 1000.0)
            cur_s, cur_e = s, e
    busy_us += cur_e - cur_s
    out["gpu_busy_pct"] = round(busy_us / (w1 - w0) * 100.0, 1)
    out["gaps_gt_2ms"] = {"n": len(gaps), "total_s": round(sum(gaps) / 1e6, 2),
                          "max_s": round(max(gaps) / 1e6, 3) if gaps else 0}
    return out

def main():
    os.makedirs(RES, exist_ok=True)
    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)
    log(f"=== nsys MTP breakdown — prod md5 {so_prefix(SO_PROD)} ===")
    if not port_free(PORT):
        sys.exit(f"port {PORT} occupé — abandon")
    stop_and_free()

    env = dict(os.environ)
    for d in CUDA_LIBS:
        env["LD_LIBRARY_PATH"] = d + ":" + env.get("LD_LIBRARY_PATH", "")
    env.update({
        "GGML_BACKEND_PATH": SO_PROD,
        "GGML_CUDA_VOLTA_Q6_W4R4": "1",
        "GGML_CUDA_VOLTA_Q8_FATTN_TC": "0",
        "GGML_CUDA_QWEN35_MTP_SHORTLIST": SHORTLIST,
        "LLAMA_ARG_SPEC_MTP_DEFER_PROMPT": "1",
        "LLAMA_ARG_FIT_TARGET": "1024",
        # ordinal CUDA du V100 résolu par UUID + CUPTI 12.9 forcé (voir note 30/10)
        "CUDA_VISIBLE_DEVICES": str(_v100_cuda_ordinal()),
        "NSYS_CUPTI_LIBRARY_PATH": CUPTI129_DIR,
    })
    srv_out = os.path.join(RES, "server.out")
    lf = open(srv_out, "a")
    log("=== nsys launch : serveur MTP ON (args prod) port " + str(PORT))
    _state["proc"] = subprocess.Popen([
        NSYS, "launch", f"--session={SESSION}", "--trace=cuda,osrt",
        "--cuda-graph-trace=node",
        LLAMA_SERVER, "--model", GGUF, "--port", str(PORT), "--host", "127.0.0.1",
        "--no-webui", "--offline",
        "-c", "128000", "-np", "1", "-b", "2048", "-ub", "2048",
        "--spec-type", "draft-mtp", "--spec-draft-n-max", "3",
        "--spec-draft-backend-sampling",
        "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
        "--flash-attn", "auto", "--mmproj", MMPROJ,
        "--log-verbosity", "4", "--no-log-prefix", "--no-log-timestamps",
    ], stdout=lf, stderr=subprocess.STDOUT, env=env)

    allrows = {}
    try:
        # model loaded ?
        ok = False
        for i in range(600):
            if _state["proc"].poll() is not None:
                sys.exit(f"server mort (exit {_state['proc'].returncode})")
            try:
                if "model loaded" in open(srv_out, errors="replace").read()[-400000:]:
                    ok = True; break
            except OSError: pass
            time.sleep(1)
        if not ok:
            sys.exit("ABORT : model non chargé en 600 s")
        log("  · model loaded")

        # warmup hors capture
        log("  · warmup 32 tok")
        w = chat({"model": "x", "stream": False, "temperature": 0,
                  "max_tokens": 32,
                  "messages": [{"role": "user", "content": "Say OK."}]}, timeout=600)
        if "error" in w:
            sys.exit(f"ABORT : warmup en erreur : {w['error']}")
        log(f"  · warmup ok ({w.get('usage')})")

        # fenêtre nsys : prompt ~16k + gen 256
        log("  · nsys start → requête (prompt ~16k + gen 256)")
        r = run([NSYS, "start", f"--session={SESSION}"], timeout=60)
        if r.returncode != 0:
            sys.exit(f"ABORT : nsys start : {r.stderr[:300]}")
        time.sleep(1)
        t0 = time.time()
        req = chat({"model": "x", "stream": False, "temperature": 0,
                    "max_tokens": GEN_TOK,
                    "messages": [{"role": "user",
                                  "content": prompt_text(PROMPT_WORDS) + "\nRéponds en une phrase."}]},
                   timeout=900)
        wall = time.time() - t0
        run([NSYS, "stop", f"--session={SESSION}"], timeout=300)
        log(f"  · requête finie en {wall:.1f}s, usage={req.get('usage', req.get('error'))}")
        allrows["req"] = {k: req.get(k) for k in ("usage", "error")}
        allrows["wall_s"] = round(wall, 1)
    finally:
        # tuer le server (nsys launch child) AVANT restauration
        p = _state.get("proc")
        if p and p.poll() is None:
            p.terminate()
            try: p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill(); p.wait(timeout=10)
        time.sleep(5)
        restore_service()

    # rapport
    time.sleep(3)
    rep = None
    for cand in glob.glob(os.path.join(RES, "report*.nsys-rep")) + \
                glob.glob(os.path.join(os.path.expanduser("~"), "report*.nsys-rep")) + \
                glob.glob("/tmp/report*.nsys-rep"):
        rep = cand; break
    if rep:
        os.replace(rep, os.path.join(RES, "prof.nsys-rep"))
        rep = os.path.join(RES, "prof.nsys-rep")
        log("  · export sqlite")
        run([NSYS, "export", "--type=sqlite", "-f", "true",
             "--output", os.path.join(RES, "prof.sqlite"), rep], timeout=600)
        try:
            allrows["analysis"] = analyze_sqlite(os.path.join(RES, "prof.sqlite"))
        except Exception as e:
            allrows["analysis"] = {"error": str(e)}
    else:
        allrows["analysis"] = {"error": "rapport nsys introuvable"}

    with open(os.path.join(RES, "results.json"), "w") as f:
        json.dump(allrows, f, indent=1)

    # synthèse
    a = allrows.get("analysis", {})
    lines = ["# nsys — décomposition CUDA d'un cycle MTP (prod lba, MTP ON)", "",
             f"date : {time.strftime('%Y-%m-%d %H:%M UTC')}",
             f"service restauré : {'OUI' if _restored else 'NON'}",
             f".so profilé : {so_prefix(SO_PROD)} (prod)",
             f"config : MTP ON draft-n-max 3, -b 2048, KV q8_0, FA auto, ctx 128k",
             f"requête : prompt ~16k tok + gen {GEN_TOK} tok, wall {allrows.get('wall_s')} s", ""]
    if "top" in a:
        lines += [f"fenêtre nsys : {a.get('window_s')} s, {a.get('n_kernels')} kernels, "
                  f"GPU busy {a.get('gpu_busy_pct')}%", "",
                  "## Temps kernel par catégorie (µs, % du total kernel)",
                  "| catégorie | µs | % |", "|---|---:|---:|"]
        for k, v in a["cats_us"].items():
            lines.append(f"| {k} | {v:,} | {a['cats_pct'].get(k)}% |")
        lines += ["", "## Top kernels (µs)", "| µs | n | nom |", "|---:|---:|---|"]
        for r in a["top"][:15]:
            lines.append(f"| {r['us']:,} | {r['n']} | `{r['name']}` |")
        g = a.get("gaps_gt_2ms", {})
        lines += ["", f"## Gaps GPU > 2 ms : {g.get('n')} (total {g.get('total_s')} s, "
                     f"max {g.get('max_s')} s)", ""]
    else:
        lines += [f"ERREUR analyse : {a}"]
    with open(OUT_SUMMARY, "w") as f:
        f.write("\n".join(lines) + "\n")
    log(f"FIN — synthèse : {OUT_SUMMARY}, service restauré : {svc_state()}")
    sys.exit(0 if _restored else 3)

if __name__ == "__main__":
    main()
