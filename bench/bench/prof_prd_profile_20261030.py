#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""prof_prd_profile_20261030.py — REPROFILAGE COMPLET de l'état PRD (30/10).

Objectif : voir OÙ PASSE LE TEMPS du système PRD actuel, avec TOUS les outils
exploitables, et faire le point MTP ON vs MTP OFF, phase par phase :
  - PREFILL        (prompt ~115k neuf, chunkés b2048)
  - DECODE         (decode vanille / base, sans spéculation)
  - DRAFT          (MTP ON : graphe moyen ~44 kernels/cycle)
  - MTP VERIFICATION (MTP ON : gros graphe ~2229 kernels/cycle)
  - sampling / host sync / gaps GPU ("l'etc qui fait mal")

Outillage (état 30/10, cf GPU_UUID_NSYS_CUPTI_20261030.md + supervisor README) :
  1. nsys 2025.5.2 + NSYS_CUPTI_LIBRARY_PATH=bench/nsys-cupti129 (CUPTI 12.9,
     UNE SEULE lib dans le dir) + --cuda-graph-trace=node  → kernels/API/sync.
  2. perf record (sudo, --clockid=MONOTONIC, -F 99) sur la llama-server MTP ON
     pendant la fenêtre DECODE → CPU host (sampling, bookkeeping, sync).
  3. nvidia-smi sampler 100 ms (busy/mem/power) pendant chaque fenêtre.
  4. ncu (2025.4.1) réservé à la Phase D, à géométrie fixée post-bilan
     (pas lancé dans ce script).

Séquence (2 configs, prod EXACTE, .so 8035caf4 figé) :
  MTP OFF (démo de base decode) :
    - warmup 16 tok (hors capture)
    - FENÊTRE PREFILL-OFF : prompt ~115k neuf + 16 gen   (nsys start/stop)
    - FENÊTRE DECODE-OFF  : même prompt (checkpoint) + 256 gen (nsys start/stop)
  MTP ON (état prod) :
    - warmup 16 tok (hors capture)
    - FENÊTRE PREFILL-ON  : prompt ~115k neuf + 16 gen   (nsys start/stop)
    - FENÊTRE DECODE-ON   : même prompt (checkpoint) + 384 gen
                            (nsys start/stop + perf record sur le pid)

Chaque fenêtre : bracket wall (µs) + sampler + usage/timings Ollama + extrait
du log serveur (acceptance MTP, eval time, etc).

Sorties :
  bench/runs/prof-prd-<ts>/          (brackets, samplers, logs serveur, perf)
  bench/prof_prd_<config>_decode_<ts>.nsys-rep   (et .sqlite exporté)
  bench/prof_prd_<config>_prefill_<ts>.nsys-rep
  bench/prof_prd_summary_<ts>.md     (mini-synthèse auto)
  /tmp/kolbo-supervisor-bench.last-rc via le superviseur (rc du script)

USAGE :
  cd /home/guile/llama-experiments/bench
  python3 supervisor_bench.py run prof_prd_profile_20261030.py --detach
"""
import glob
import hashlib
import json
import os
import re
import shlex
import signal
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.request

ROOT = "/home/guile/llama-experiments"
BENCH = os.path.join(ROOT, "bench")
NSYS = "/opt/nvidia/nsight-systems/2025.5.2/target-linux-x64/nsys"
CUPTI129_DIR = os.path.join(BENCH, "nsys-cupti129")
LLAMA_SERVER = os.path.join(ROOT, "ollama-v100/lib/ollama/llama-server")
SO_PROD = os.path.join(ROOT, "ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so")
GGUF = ("/usr/share/ollama/.ollama/models/blobs/"
        "sha256-493301830a596b8ad56dc1329f80bbcb578c8e910da395feafdc9cd8263430bb")
MMPROJ = ("/usr/share/ollama/.ollama/models/blobs/"
          "sha256-ac3714bfdddeca31351f2752bf1a63f266f4df87c0b68c895e44945ca704448e")
SHORTLIST = os.path.join(
    ROOT, "src/llama-volta-merged/data/mtp-shortlists/qwen38-27b-exact-131072.i32")
SAMPLER = os.path.join(BENCH, "gputimes_sampler.py")
NSYS_LOG = os.path.join(BENCH, "nsys_sessions.log")

SERVICE = "ollama-v100"
PORT = 44921
TS = time.strftime("%Y%m%d-%H%M%S")
OUT = os.path.join(BENCH, "runs", "prof-prd-" + TS)
REP_DIR = "/tmp/profprd"            # cwd de nsys (root) : les .rep s'écrivent ICI
eval_cache = {}
BRK = {}
_state = {}
_restored = False

CTX_TOKENS = 115000     # prompt neuf ~115k tokens (filler ratio 5 tok/4 chars)
GEN_DECODE = 160        # MTP ON decode window ≈ 40-55 cycles MTP (budget fit/watchdog CUPTI)
GEN_DECODE_OFF = 96     # decode vanille (budget fit/watchdog CUPTI)
GEN_PREFILL = 16        # gen minuscule en fin de fenêtre prefill

FILL = ("La tour de contrôle surveille le passage des avions au crépuscule "
        "pendant que le chef de piste organise les départs de la soirée. ")


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def run(cmd, timeout=60, sudo=False):
    c = (["sudo"] + cmd) if sudo else cmd
    return subprocess.run(c, capture_output=True, text=True, timeout=timeout)


def nsys_session_state(name):
    """état d'une session nsys (None si inconnue).
    Piège : le message IDLE contient 'start collecting data' → tester
    'application launched' AVANT 'collecting data'."""
    r = run([NSYS, "status", "--session=" + name], timeout=30, sudo=True)
    out = (r.stdout or "") + (r.stderr or "")
    if "application launched" in out:
        return "idle"
    if "collecting data" in out:
        return "collecting"
    if "not found" in out.lower():
        return None
    return "unknown"


def nsys_start(sess):
    st = nsys_session_state(sess)
    if st == "collecting":
        log("  · session déjà en collecte — nsys stop d'abord (%s)" % sess)
        nsys_stop(sess)
    if st is None:
        raise RuntimeError("session nsys %s introuvable" % sess)
    r = run([NSYS, "start", "--session=" + sess], timeout=60, sudo=True)
    if r.returncode != 0:
        raise RuntimeError("nsys start rc=%s : %s" % (r.returncode, r.stderr[:200]))


def nsys_stop(sess):
    """stop TOLERANT : 'Collection stop is not allowed in this state' = la session
    a déjà fait sa génération (child mort / auto-stop) ; le .rep existe, on ne
    lève PAS — take_rep le récupère. rc!=0 inconnu → warning seulement : la
    capture ne doit jamais abattre la fin de config (kill server + restauration)."""
    t0 = time.time()
    r = run([NSYS, "stop", "--session=" + sess], timeout=300, sudo=True)
    err = (r.stderr or r.stdout or "").strip()
    log("  · nsys stop rc=%d (%.0f s)" % (r.returncode, time.time() - t0))
    if r.returncode == 0:
        return
    if "not allowed in this state" in err:
        log("  · session déjà en état terminal (auto-stop) — le .rep devrait exister")
        return
    if "not found" in err.lower():
        log("  · session nsys %s disparue (child mort)" % sess)
        return
    log("  · ATTENTION nsys stop KO : %s" % err[:200])


def find_reps(min_ts):
    cands = []
    for d in (REP_DIR, "/root", "/tmp", BENCH):
        for fn in glob.glob(os.path.join(d, "*.nsys-rep")):
            try:
                mt = os.path.getmtime(fn)
            except OSError:
                continue
            if mt >= min_ts - 60:
                cands.append((mt, fn))
    cands.sort()
    return cands


def take_rep(sess, tag):
    """trouve le .rep le plus récent (cwd nsys / root / tmp / bench) et le range
    dans bench/. Le .rep est root-owned (nsys root) → sudo mv + chown."""
    time.sleep(3)
    cands = find_reps(time.time() - 7200)
    if not cands:
        log("  · ATTENTION : aucun .nsys-rep récent trouvé pour " + tag)
        return None
    mt, rep = cands[-1]
    dst = os.path.join(BENCH, "prof_prd_%s_%s.nsys-rep" % (tag, TS))
    try:
        run(["mv", rep, dst], timeout=120, sudo=True)
        run(["chown", "%d:%d" % (os.getuid(), os.getgid()), dst],
            timeout=60, sudo=True)
    except Exception as e:
        log("  · ATTENTION .rep non déplacé (%s → %s) : %s" % (rep, dst, e))
        return None
    log("  · .rep [%s] → %s (%.1f Mo)" % (rep, dst, os.path.getsize(dst) / 1e6))
    return dst


def svc_state():
    return run(["systemctl", "is-active", SERVICE], timeout=30).stdout.strip()


def _v100_smi_index():
    try:
        r = run(["python3", os.path.join(BENCH, "gpu_uuid.py"), "--smi-index"],
                timeout=30)
        v = r.stdout.strip()
        if v.isdigit():
            return v
    except Exception:
        pass
    return "1"


def _v100_cuda_ordinal():
    try:
        r = run(["python3", os.path.join(BENCH, "gpu_uuid.py"), "--cvt-ordinal"],
                timeout=30)
        v = r.stdout.strip()
        if v.isdigit():
            return int(v)
    except Exception:
        pass
    return 0


def vram_mib():
    r = run(["nvidia-smi", "--id=" + _v100_smi_index(),
             "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            timeout=30)
    v = r.stdout.strip()
    return int(v) if v.isdigit() else -1


def port_free(port):
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", port))
        s.close()
        return True
    except OSError:
        s.close()
        return False


def http_json(url, data=None, timeout=10):
    req = urllib.request.Request(
        url, data=json.dumps(data).encode() if data is not None else None,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def prompt_text(ctx_target_tokens):
    n_words = ctx_target_tokens * 4 // 5
    words = FILL.split()
    return " ".join([words[i % len(words)] for i in range(n_words)])


def restore_service():
    global _restored
    try:
        if svc_state() != "active":
            log("  · restauration : systemctl start " + SERVICE)
            run(["systemctl", "start", SERVICE], timeout=180, sudo=True)
        for _ in range(120):
            if svc_state() == "active":
                break
            time.sleep(2)
        _restored = svc_state() == "active"
        log("  · service %s = %s, VRAM %d MiB" % (SERVICE, svc_state(), vram_mib()))
    except Exception as e:
        log("  · ERREUR restauration : %s" % e)
        _restored = False


def on_sig(sig, frm):
    try:
        nsys_stop(_state.get("sess", "x"))
    except Exception:
        pass
    stop_perf()
    p = _state.get("proc")
    if p and p.poll() is None:
        try:
            kill_group(p, "sig")
        except Exception:
            pass
    restore_service()
    sys.exit(128 + sig)


def stop_and_free():
    log("  · stop " + SERVICE)
    run(["systemctl", "stop", SERVICE], timeout=180, sudo=True)
    t0 = time.time()
    while time.time() - t0 < 300:
        if vram_mib() < 1000:
            log("  · V100 libérée (%d MiB) en %.0f s" % (vram_mib(), time.time() - t0))
            return
        time.sleep(3)
    log("  · ATTENTION V100 pas totalement libérée (VRAM=%d MiB)" % vram_mib())


def env_cmdline():
    """sudo ignore -E ici → env passé explicitement (pattern nso_orchestrator,
    vérifié 30/10 01:51). nsys root écrit ses .rep dans SON cwd (= bench/ du
    bench) → take_rep cherche dans REP_DIR, /root, /tmp ET bench/."""
    env = dict(os.environ)
    ld = ":".join([
        os.path.join(ROOT, "ollama-v100/lib/ollama"),
        os.path.join(ROOT, "ollama-v100/lib/ollama/cuda_v12"),
        os.path.join(ROOT, "tools/cuda-12.6/usr/local/cuda-12.6/lib64"),
        os.path.join(ROOT, "tools/cuda-12.6/usr/local/cuda-12.6/targets/x86_64-linux/lib"),
    ])
    env["LD_LIBRARY_PATH"] = ld + (":" + env["LD_LIBRARY_PATH"]
                                   if env.get("LD_LIBRARY_PATH") else "")
    env["PATH"] = ("/opt/nvidia/nsight-systems/2025.5.2/target-linux-x64:"
                   "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
    env.update({
        "GGML_BACKEND_PATH": SO_PROD,
        "GGML_CUDA_VOLTA_Q6_W4R4": "1",
        "GGML_CUDA_VOLTA_Q8_FATTN_TC": "0",          # H2 OFF = prod
        "GGML_CUDA_QWEN35_MTP_SHORTLIST": SHORTLIST,
        "LLAMA_ARG_SPEC_MTP_DEFER_PROMPT": "1",
        "LLAMA_ARG_FIT_TARGET": "1024",
        "CUDA_VISIBLE_DEVICES": str(_v100_cuda_ordinal()),
        "NSYS_CUPTI_LIBRARY_PATH": CUPTI129_DIR,
    })
    return ["sudo", "env"] + ["%s=%s" % (k, shlex.quote(v)) for k, v in env.items()]


def wait_log(sout, pattern, tag, proc, timeout=900):
    bad = ("out of memory", "cuda error", "failed to fit params")
    t0 = time.time()
    while time.time() - t0 < timeout:
        txt = open(sout, errors="replace").read()
        if any(b in txt for b in bad):
            tail(sout, 15)
            raise RuntimeError(tag + " : OOM/cuda error")
        if proc.poll() is not None:
            raise RuntimeError(tag + " : server mort (exit %s)" % proc.returncode)
        m = re.search(pattern, txt)
        if m:
            return m.group(0)[:160]
        time.sleep(1)
    raise RuntimeError(tag + " : timeout %d s" % timeout)


def tail(path, n=15):
    try:
        for l in open(path, errors="replace").read().splitlines()[-n:]:
            print("    | " + l[:180], flush=True)
    except OSError:
        pass


def start_sampler(tag):
    p = subprocess.Popen(["python3", SAMPLER,
                          os.path.join(OUT, tag + "_gputimes.tsv"), "0.1"])
    return p


def stop_sampler(s):
    if s:
        s.terminate()
        try:
            s.wait(timeout=10)
        except Exception:
            pass


def start_perf(sess, pid):
    """perf record sur la llama-server, horloge MONOTONIC (cohérente CUPTI).
    Stop : SIGINT direct au vrai processus perf (pgrep -f est dangereux : le
    pattern matche aussi le wrapper 'sudo env' et des shell tiers)."""
    out = os.path.join(OUT, "perf_" + sess)
    p = subprocess.Popen(["sudo", "env", "COLUMNS=200",
                          "perf", "record", "-F", "99", "-g",
                          "--clockid=MONOTONIC", "-o", out, "-p", str(pid)])
    time.sleep(1)
    try:
        pids = run(["pgrep", "-f", "perf record -F 99 -g --clockid=MONOTONIC"
                    " -o " + out], timeout=10).stdout.split()
    except Exception:
        pids = []
    _state["perf"] = (p, [int(x) for x in pids if x.isdigit()])
    log("  · perf record pid=%d → %s (child pid %s, perf pids %s)"
        % (pid, out, p.pid, pids or "?"))
    return p


def stop_perf():
    pf = _state.get("perf")
    if not pf:
        return
    p, pids = pf
    # perf tourne root (sudo) : SIGINT/kill en sudo, jamais os.kill guile.
    for sig, wait_s in (("-INT", 20), ("-9", 10)):
        if p.poll() is not None:
            break
        for q in pids:
            try:
                run(["kill", sig, str(q)], timeout=10, sudo=True)
            except Exception:
                pass
        try:
            p.wait(timeout=wait_s)
            break
        except Exception:
            pass
    try:
        p.kill()
        p.wait(timeout=5)
    except Exception:
        pass
    _state["perf"] = None
    log("  · perf stoppé")


def export_rep(final_path, timeout=900):
    """export sqlite d'un .rep déjà pris en charge (user suffit)."""
    try:
        r = run([NSYS, "export", "--type=sqlite", "--force-overwrite", "true",
                 "-o", final_path.replace(".nsys-rep", ".sqlite"), final_path],
                timeout=timeout)
        if r.returncode != 0:
            log("  · export sqlite échoué : %s" % (r.stderr or "")[:200])
            return None
        return final_path.replace(".nsys-rep", ".sqlite")
    except Exception as e:
        log("  · export sqlite échoué : %s" % e)
        return None


def chat(data, timeout=2400):
    try:
        return http_json("http://127.0.0.1:%d/v1/chat/completions" % PORT,
                         data, timeout=timeout)
    except Exception as e:
        return {"error": str(e)}


def gen_summary(r):
    u = r.get("usage", {})
    t = r.get("timings", {})
    pps = 0.0
    if t.get("prompt_ms") and u.get("prompt_tokens"):
        pps = round(u["prompt_tokens"] / (t["prompt_ms"] / 1000.0), 1)
    tps = 0.0
    if t.get("predicted_per_second"):
        tps = round(t["predicted_per_second"], 2)
    return {
        "prompt": u.get("prompt_tokens"),
        "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "completion": u.get("completion_tokens"),
        "prompt_ms": t.get("prompt_ms"),
        "prompt_tps": pps,
        "predicted_tps": tps,
        "ms_per_tok": round(t.get("predicted_per_token_ms", 0) or 0, 1),
        "finish": (r.get("choices") or [{}])[0].get("finish_reason"),
    }


def server_stats(sout):
    """extrait les lignes d'acceptance/eval du log serveur (dernières)."""
    out = []
    for l in open(sout, errors="replace").read().splitlines():
        if re.search(r"draft acceptance|acc per pos|spec.*accept|print_timing"
                     r".*eval time|n_decoded|eval time", l):
            out.append(l[:220])
    return out[-40:]


def launch_server(tag, spec_on):
    """Lance le server sous nsys (session PRD_<tag>), gates, warmup. Renvoie
    (proc, sout, sess)."""
    sess = "PRD_" + tag
    _state["sess"] = sess
    sout = os.path.join(OUT, "server_%s.out" % tag)
    spec_args = (["--spec-type", "draft-mtp", "--spec-draft-n-max", "3",
                  "--spec-draft-backend-sampling"] if spec_on else [])
    # --cuda-flush-interval=500 : flush des buffers CUPTI toutes les 500 ms
    # (sans ça, la génération au stop de session n'est pas garantie quand le
    # child est tué — l'orphelin du run 08:40 du 30/10 est venu de là).
    args = [NSYS, "launch", "--session-new=" + sess,
            "--trace=cuda,osrt", "--cuda-graph-trace=node",
            "--cuda-flush-interval=500",
            "--osrt-threshold=2000", "--osrt-backtrace-threshold=0",
            LLAMA_SERVER, "--model", GGUF, "--port", str(PORT),
            "--host", "127.0.0.1", "--no-webui", "--offline",
            "-c", "167000", "-np", "1", "-b", "2048", "-ub", "2048",
            "--log-verbosity", "4", "--no-log-prefix", "--no-log-timestamps",
            "--mmproj", MMPROJ] + spec_args + [
            "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
            "--flash-attn", "auto"]

    log("== [%s] nsys launch du server (%s) port %d ==" %
        (tag, "MTP ON" if spec_on else "MTP OFF", PORT))
    # start_new_session : tout l'arbre (sudo → nsys → llama-server) est dans un
    # pgid dédié → killpg propre (un SIGTERM au sudo seul orpheline nsys+server
    # — c'est exactement le résidu 442495 du run 08:40).
    proc = subprocess.Popen(env_cmdline() + args,
                            stdout=open(sout, "a"), stderr=subprocess.STDOUT,
                            start_new_session=True)
    _state["proc"] = proc

    # gates
    t0 = time.time()
    ok = False
    while time.time() - t0 < 360:
        if proc.poll() is not None:
            tail(sout)
            raise RuntimeError(tag + " : server mort au démarrage")
        try:
            http_json("http://127.0.0.1:%d/health" % PORT, timeout=3)
            ok = True
            break
        except Exception:
            time.sleep(2)
    if not ok:
        tail(sout)
        raise RuntimeError(tag + " : pas de /health en 360 s")
    log("  · GATE UP (%.0f s)" % (time.time() - t0))
    wait_log(sout, r"successfully fit params", tag + "/FIT", proc)
    dev = wait_log(sout, r"using device CUDA[^\n]*", tag + "/DEV", proc)
    if "Tesla V100" not in dev:
        raise RuntimeError(tag + " : device != V100 : " + dev[:120])
    wait_log(sout, r"model loaded", tag + "/LOAD", proc, timeout=1200)
    if spec_on:
        wait_log(sout, r"CUDA MTP shortlist[^\n]*", tag + "/SL", proc)
    log("  · GATE MODEL LOADED (%s)" % ("+shortlist" if spec_on else ""))

    # warmup hors capture (graphe, caches, allocator)
    ok = False
    for w in range(1, 7):
        r = chat({"model": "local", "stream": False, "temperature": 0,
                  "max_tokens": 16,
                  "messages": [{"role": "user", "content": "Say OK."}]},
                 timeout=1800)
        if r.get("usage", {}).get("completion_tokens"):
            ok = True
            break
        log("    warmup essai %d : %s" % (w, str(r.get("error", ""))[:80]))
        time.sleep(10)
    if not ok:
        raise RuntimeError(tag + " : warmup KO")
    time.sleep(3)
    return proc, sout, sess


def kill_group(proc, tag, grace=60):
    """SIGTERM/SIGKILL sur le pgid complet (sudo+nsys+llama-server)."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=grace)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
        try:
            proc.wait(timeout=30)
        except Exception:
            pass


def kill_server(proc, tag):
    """kill PROPRE : nsys stop (tolérant, capture déjà arrêtée) → SIGTERM du
    pgid complet → VRAM < 1000, purge résidu V100 si besoin."""
    try:
        nsys_stop(_state.get("sess", "x"))
    except Exception as e:
        log("  · nsys stop : %s" % e)
    kill_group(proc, tag)
    t0 = time.time()
    for _ in range(60):
        if vram_mib() < 1000:
            break
        time.sleep(2)
    log("  · [%s] server tué, VRAM %d MiB (%.0f s)" %
        (tag, vram_mib(), time.time() - t0))
    if vram_mib() >= 1000:
        # résidu GPU : purge UNIQUEMENT les pids compute de la V100 (jamais
        # toutes les GPUs : le 1070 porte les services prod 33695/11436).
        rr = run(["nvidia-smi", "--id=" + _v100_smi_index(),
                  "--query-compute-apps=pid,used_memory",
                  "--format=csv,noheader"], timeout=30, sudo=True)
        for line in rr.stdout.splitlines():
            parts = line.split(",")
            if len(parts) == 2 and parts[0].strip().isdigit():
                log("  · purge résidu GPU V100 pid %s (%s MiB)" %
                    (parts[0].strip(), parts[1].strip()))
                run(["kill", "-9", parts[0].strip()], timeout=15, sudo=True)
        for _ in range(60):
            if vram_mib() < 1000:
                break
            time.sleep(2)
        log("  · VRAM après purge : %d MiB" % vram_mib())


def run_config(tag, spec_on, ntok_decode, prefill_ntok):
    """Une config, 2 instances de server (séparation propre préfill/decode) :
      A) server A (nsys) → FENÊTRE PREFILL (prompt ~115k neuf + 16 gen) →
         stop session, rep, sqlite, kill, VRAM check.
      B) server B (nsys) → FENÊTRE DECODE (prompt checkpointé + ntok_decode gen,
         perf si MTP ON) → stop, rep, sqlite, kill, VRAM check.
    L'instance A ne subit JAMAIS un task decode long : le fit watchdog (120 s
    sans token — CUPTI ralentit le decode) ne peut plus tuer le server en
    collecte (c'est exactement l'incident du run 08:40)."""
    text = prompt_text(CTX_TOKENS) + "\nRéponds en une phrase."
    base = {"model": "local", "stream": False, "temperature": 0.0,
            "top_k": 1, "max_tokens": ntok_decode,
            "messages": [{"role": "user", "content": text}]}
    res = {"tag": tag, "spec_on": spec_on}

    # ---------- A) PREFILL (server A) ----------
    procA, sout, sessA = launch_server(tag + "_pf", spec_on)
    prefill_body = dict(base, max_tokens=prefill_ntok)
    log("  · [%s] FENÊTRE PREFILL (prompt ~%dk neuf + %d gen) — nsys start"
        % (tag, CTX_TOKENS // 1000, prefill_ntok))
    nsys_start(sessA)
    t0 = int(time.time() * 1e6)
    samp = start_sampler(tag + "_prefill")
    r = chat(prefill_body, timeout=3600)
    time.sleep(2)
    t1 = int(time.time() * 1e6)
    stop_sampler(samp)
    nsys_stop(sessA)
    res["prefill"] = gen_summary(r)
    res["prefill_wall_ms"] = round((t1 - t0) / 1e3, 0)
    res["prefill_rep"] = take_rep(sessA, tag + "_prefill")
    if res["prefill_rep"]:
        res["prefill_sqlite"] = export_rep(res["prefill_rep"])
    res["prefill_bracket"] = [t0, t1]
    log("  · [%s] PREFILL wall %.1f s — %s"
        % (tag, (t1 - t0) / 1e6, res["prefill"]))
    kill_server(procA, tag + "_pf")
    log("  · pause entre instances (VRAM propre + port libre) : 15 s")
    time.sleep(15)
    if not port_free(PORT):
        log("  · port encore occupé, attente...")
        for _ in range(60):
            if port_free(PORT):
                break
            time.sleep(2)

    # ---------- B) DECODE (server B, neuf) ----------
    procB, sout, sessB = launch_server(tag + "_dc", spec_on)
    # préfill HORS CAPTURE : chat1 = prompt neuf + 16 gen (crée le checkpoint
    # ~112055). Pas de nsys_start, pas de perf : prompt ~115k sans overhead
    # CUPTI (~105 s), puis chat2 ne re-prompte que ~4 tokens (restored).
    log("  · [%s] préfill hors capture (server B, prompt neuf)"
        % tag)
    r1 = chat(prefill_body, timeout=3600)
    log("  · préfill B : %s" % gen_summary(r1))
    if not r1.get("usage", {}).get("completion_tokens"):
        raise RuntimeError(tag + " : préfill server B KO : "
                           + str(r1.get("error", ""))[:120])
    log("  · [%s] FENÊTRE DECODE (%d tok, prompt checkpointé) — nsys start"
        % (tag, ntok_decode))
    if spec_on:
        rr = run(["pgrep", "-f",
                  "llama-server --model %s --port %d" % (GGUF, PORT)], timeout=20)
        pids = rr.stdout.split()
        pid = int(pids[-1]) if pids else procB.pid
        _state["server_pid"] = pid
        start_perf(sessB, pid)
    nsys_start(sessB)
    t0 = int(time.time() * 1e6)
    samp = start_sampler(tag + "_decode")
    r = chat(base, timeout=3600)
    time.sleep(2)
    t1 = int(time.time() * 1e6)
    stop_sampler(samp)
    if spec_on:
        stop_perf()
    nsys_stop(sessB)
    res["decode"] = gen_summary(r)
    res["decode_wall_ms"] = round((t1 - t0) / 1e3, 0)
    res["decode_rep"] = take_rep(sessB, tag + "_decode")
    if res["decode_rep"]:
        res["decode_sqlite"] = export_rep(res["decode_rep"])
    res["decode_bracket"] = [t0, t1]
    res["server_stats"] = server_stats(sout)
    log("  · [%s] DECODE wall %.1f s — %s"
        % (tag, (t1 - t0) / 1e6, res["decode"]))
    for l in res["server_stats"][-6:]:
        log("      | " + l)
    kill_server(procB, tag + "_dc")
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    run(["mkdir", "-p", REP_DIR], timeout=30, sudo=True)
    run(["chmod", "777", REP_DIR], timeout=30, sudo=True)
    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)
    log("=== PROF PRD (30/10) — out : " + OUT + " ===")
    md5 = hashlib.md5(open(SO_PROD, "rb").read()).hexdigest()
    log("  · SO prod md5 = %s (attendu 8035caf4*)" % md5[:8])
    if not md5.startswith("8035caf4"):
        log("  · NOTE : .so != 8035caf4 — prod changée ? on continue (profilage).")
    for f in (NSYS, LLAMA_SERVER, GGUF, MMPROJ, SHORTLIST, SAMPLER):
        if not os.path.isfile(f):
            sys.exit("fichier manquant : " + f)
    if not port_free(PORT):
        sys.exit("port %d occupé — abandon" % PORT)

    stop_and_free()

    allres = {"ts": TS, "so_md5": md5, "ctx_tokens": CTX_TOKENS,
              "decode_tokens": {"off": GEN_DECODE_OFF, "on": GEN_DECODE}}
    rc = 1
    try:
        allres["off"] = run_config("off", False, GEN_DECODE_OFF, GEN_PREFILL)
        log("  · pause entre configs (VRAM propre + port libre)")
        time.sleep(20)
        allres["on"] = run_config("on", True, GEN_DECODE, GEN_PREFILL)
        rc = 0
    except Exception as e:
        log("  · ECHEC : %s" % e)
        import traceback
        traceback.print_exc()
    finally:
        try:
            nsys_stop(_state.get("sess", "x"))
        except Exception:
            pass
        stop_perf()
        p = _state.get("proc")
        if p and p.poll() is None:
            try:
                kill_group(p, "finally")
            except Exception:
                pass
        for _ in range(90):
            if vram_mib() < 1000:
                break
            time.sleep(2)
        rr = run(["nvidia-smi", "--id=" + _v100_smi_index(),
                  "--query-compute-apps=pid,used_memory",
                  "--format=csv,noheader"], timeout=30, sudo=True)
        if rr.stdout.strip():
            log("  · ATTENTION résidus GPU : %s" % rr.stdout.strip())
        restore_service()

    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(allres, f, indent=1, default=str)

    # mini-synthèse
    lines = ["# PROF PRD — reprofile 30/10 (MTP ON vs OFF, phases)",
             "",
             "date : " + time.strftime("%Y-%m-%d %H:%M UTC"),
             "so : " + md5[:8] + " | ctx " + str(CTX_TOKENS) + " | b2048 | KV q8_0",
             "service restauré : " + ("OUI" if _restored else "NON"),
             "out : " + OUT, ""]
    for cfg in ("off", "on"):
        r = allres.get(cfg)
        if not r:
            continue
        tag = "MTP " + ("ON" if r["spec_on"] else "OFF")
        lines += ["## " + tag, "",
                  "| phase | wall s | completion | tok/s prédit | ms/tok | rep |",
                  "|---|---:|---:|---:|---:|---|"]
        for ph in ("prefill", "decode"):
            d = r.get(ph, {})
            wall = r.get(ph + "_wall_ms", 0) / 1000.0
            lines.append("| %s | %s | %s | %s | %s | %s |" % (
                ph, round(wall, 1), d.get("completion"),
                d.get("predicted_tps"), d.get("ms_per_tok"),
                os.path.basename(r.get(ph + "_rep") or "MANQUANT")))
        lines += ["", "extrait log serveur :"]
        lines += ["    " + l for l in r.get("server_stats", [])[-12:]]
        lines.append("")
    with open(os.path.join(BENCH, "prof_prd_summary_%s.md" % TS), "w") as f:
        f.write("\n".join(lines) + "\n")
    log("FIN — rc=%d service_restored=%s" % (rc, _restored))
    log("  · résultats : " + os.path.join(OUT, "results.json"))
    sys.exit(0 if (_restored and rc == 0) else 3)


if __name__ == "__main__":
    main()
