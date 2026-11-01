#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cupti_mtp_stepD_b2048.py — Étape D : trace CUPTI du régime STEADY 53.45 t/s (config prod EXACTE).

Suite directe du Step C (mtp_prof_stepC.py, 29/10 03:32) : la décomposition wall
disait t_verify+ = 56.9 ms/cycle (88 % du cycle MTP steady) sur config prod b2048.
La trace CUPTI du 28/10 (cupath_llama_220241.names) était en b512/mmproj-absent :
bonne pour l'attribution par famille, PAS pour les durées prod.

Ce run : EXACTEMENT la même séquence que stepC — warmup → REQ A (prompt NEUF 114k,
64 gen = catch-up DEFER payé) → REQ B (MÊME prompt checkpointé, 128 gen = STEADY).
Le bracket TS_START/TS_END borne la fenêtre REQ B (steady) pour l'analyse
(analyse_cupath_decode3.py isole de toute façon le dernier cluster dense = REQ B).
Config prod : -c 167000, -b 2048 -ub 2048, --mmproj, shortlist 131072 +
DEFER_PROMPT=1, .so prod 8035caf4 figé, CVD=0. Seul ajout : shim libcupath.so.

Séquence : stop service → V100 libre → server (gates up/FIT/V100/loaded/shortlist)
→ warmup → prefill 114k (hors fenêtre) → 2e prefill chaud → fenêtre decode 64 tok
(échantillonneur nvidia-smi 10 Hz en fond pour le cross-check GPU-busy) → cleanup.
trap EXIT = pkill + systemctl start ollama-v100 + verify.

USAGE :
  cd /home/guile/llama-experiments/bench
  python3 supervisor_bench.py run cupti_mtp_stepD_b2048.py 128 --detach
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

ROOT = "/home/guile/llama-experiments"
CUPATH_DIR = os.path.join(ROOT, "h2", "cupti129-test")
PORT = 44902
SERVICE = "ollama-v100"
NTOK = int(sys.argv[1]) if len(sys.argv) > 1 else 128

LLAMA_SERVER = os.path.join(ROOT, "ollama-v100/lib/ollama/llama-server")
SO_PROD = os.path.join(ROOT, "ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so")
GGUF = ("/usr/share/ollama/.ollama/models/blobs/"
        "sha256-493301830a596b8ad56dc1329f80bbcb578c8e910da395feafdc9cd8263430bb")
MMPROJ = ("/usr/share/ollama/.ollama/models/blobs/"
          "sha256-ac3714bfdddeca31351f2752bf1a63f266f4df87c0b68c895e44945ca704448e")
SHORTLIST = os.path.join(ROOT, "src/llama-volta-merged/data/mtp-shortlists/"
                         "qwen38-27b-exact-131072.i32")

OUT = "/tmp/cupath_b2048_" + time.strftime("%H%M%S")
SOUT = "/tmp/cupath_b2048_server.out"
BRACKET = "/tmp/cupath_b2048_bracket"
GPUTS = "/tmp/cupath_b2048_gputimes.tsv"
GEN = "/tmp/cupath_b2048_gen.json"
WARMUP_JSON = "/tmp/cupath_b2048_warm.json"
PREFILL_JSON = "/tmp/cupath_b2048_prefill.json"
PREFILL_OUT = "/tmp/cupath_b2048_prefill.out"
PREFILL2_OUT = "/tmp/cupath_b2048_prefill2.out"
SAMPLER = os.path.join(ROOT, "bench/gputimes_sampler.py")

CUDA_LIBS = [
    os.path.join(ROOT, "ollama-v100/lib/ollama"),
    os.path.join(ROOT, "ollama-v100/lib/ollama/cuda_v12"),
    os.path.join(ROOT, "tools/cuda-12.6/usr/local/cuda-12.6/lib64"),
    os.path.join(ROOT, "tools/cuda-12.6/usr/local/cuda-12.6/targets/x86_64-linux/lib"),
]

FILL = ("La tour de contrôle surveille le passage des avions au crépuscule "
        "pendant que le chef de piste organise les départs de la soirée. ")

CTX_TOKENS = 114688
_state = {}
_restored = False


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def run(cmd, timeout=60):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def svc_state():
    return run(["systemctl", "is-active", SERVICE]).stdout.strip()


def vram_mib():
    r = run(["nvidia-smi", "--id=" + v100_smi_index(), "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"])
    v = r.stdout.strip()
    return int(v) if v.isdigit() else -1


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


def restore_service():
    global _restored
    try:
        if svc_state() != "active":
            log("  · restauration : systemctl start %s" % SERVICE)
            run(["sudo", "systemctl", "start", SERVICE], timeout=120)
        for _ in range(90):
            if svc_state() == "active":
                break
            time.sleep(2)
        _restored = svc_state() == "active"
        log("  · service %s = %s" % (SERVICE, svc_state()))
    except Exception as e:
        log("  · ERREUR restauration : %s" % e)
        _restored = False


def on_sig(sig, frm):
    restore_service()
    sys.exit(128 + sig)


def stop_and_free():
    log("  · stop %s" % SERVICE)
    run(["sudo", "systemctl", "stop", SERVICE], timeout=120)
    t0 = time.time()
    while time.time() - t0 < 240:
        if vram_mib() < 1000:
            log("  · V100 libérée (%d MiB) en %.0fs" % (vram_mib(), time.time() - t0))
            return
        time.sleep(3)
    log("  · ATTENTION V100 pas totalement libérée (VRAM=%d MiB)" % vram_mib())


def prompt_text(ctx_target_tokens):
    n_words = ctx_target_tokens * 4 // 5
    words = FILL.split() * max(1, n_words // FILL.count(" ") + 1)
    return " ".join(words[:n_words])


def wait_log(pattern, proc, tag, timeout=900, bad=("out of memory", "cuda error")):
    """Attente d'une ligne dans SOUT ; abort si process mort ou ligne OOM."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if any(b in open(SOUT, errors="replace").read() for b in bad):
            log("  · ABORT : %s détecté dans le log serveur" % "OOM/cuda")
            tail(SOUT, 15)
            raise RuntimeError(tag + " : OOM/cuda error")
        if proc.poll() is not None:
            log("  · ABORT : server mort (%s, exit %s)" % (tag, proc.returncode))
            tail(SOUT, 15)
            raise RuntimeError(tag + " : server mort")
        m = re.search(pattern, open(SOUT, errors="replace").read())
        if m:
            return m.group(0)[:160]
        time.sleep(1)
    raise RuntimeError(tag + " : timeout " + str(timeout) + " s")


def tail(path, n=15):
    try:
        for l in open(path, errors="replace").read().splitlines()[-n:]:
            print("    | " + l[:180], flush=True)
    except OSError:
        pass


def main():
    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)
    log("STEPD B2048 (CUPTI, steady 53.45 t/s) — NTOK=%d — lock tenu par le superviseur" % NTOK)
    for f in (LLAMA_SERVER, SO_PROD, GGUF, MMPROJ, SHORTLIST,
              os.path.join(CUPATH_DIR, "libcupath.so")):
        if not os.path.isfile(f):
            sys.exit("fichier manquant : " + f)
    log("  · SO prod md5 = %s" % __import__("hashlib").md5(open(SO_PROD, "rb").read()).hexdigest()[:8])
    if not port_free(PORT):
        sys.exit("port %d occupé — abandon" % PORT)
    if vram_mib() > 8192 and svc_state() != "active":
        sys.exit("V100 occupée sans service actif (VRAM=%d) — abandon" % vram_mib())

    stop_and_free()

    # prompt : filler night_bench exact (114688 tok ≈) + consigne, T=0
    text = prompt_text(CTX_TOKENS) + "\nRéponds en une phrase."
    base = {"model": "local", "stream": False, "temperature": 0.0,
            "top_k": 1, "max_tokens": NTOK,
            "messages": [{"role": "user", "content": text}]}
    with open(PREFILL_JSON, "w") as f:
        json.dump(dict(base, max_tokens=32), f, ensure_ascii=False)
    with open(GEN, "w") as f:
        f.write("")
    log("  · prompt %d chars prêt (%d tok target)" % (len(text), CTX_TOKENS))

    # env : prod exacte + CUPTI
    env = dict(os.environ)
    for d in CUDA_LIBS:
        if d not in env.get("LD_LIBRARY_PATH", ""):
            env["LD_LIBRARY_PATH"] = d + ":" + env.get("LD_LIBRARY_PATH", "")
    env.update({
        "GGML_BACKEND_PATH": SO_PROD,
        "GGML_CUDA_VOLTA_Q6_W4R4": "1",
        "GGML_CUDA_QWEN35_MTP_SHORTLIST": SHORTLIST,
        "LLAMA_ARG_SPEC_MTP_DEFER_PROMPT": "1",
        "LLAMA_ARG_FIT_TARGET": "1024",
        "CUDA_VISIBLE_DEVICES": str(v100_cuda_ordinal()),  # UUID, pas en dur
        "LD_PRELOAD": os.path.join(CUPATH_DIR, "libcupath.so"),
        "CUPATH_TRACE": OUT,
    })

    args = [LLAMA_SERVER,
            "--model", GGUF, "--port", str(PORT), "--host", "127.0.0.1",
            "--no-webui", "--offline",
            "-c", "167000", "-np", "1",
            "-b", "2048", "-ub", "2048",
            "--log-verbosity", "4", "--no-log-prefix", "--no-log-timestamps",
            "--mmproj", MMPROJ,
            "--spec-type", "draft-mtp", "--spec-draft-n-max", "3",
            "--spec-draft-backend-sampling",
            "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
            "--flash-attn", "auto"]

    sampler = None
    proc = None
    rc = 1
    try:
        open(SOUT, "w").close()
        log("  · démarrage llama-server (config lente + CUPTI)")
        proc = subprocess.Popen(args, stdout=open(SOUT, "a"),
                                stderr=subprocess.STDOUT, env=env)
        _state["proc"] = proc

        # gate up
        ok = 0
        t0 = time.time()
        while time.time() - t0 < 300:
            if proc.poll() is not None:
                tail(SOUT); raise RuntimeError("server mort au démarrage")
            try:
                http_json("http://127.0.0.1:%d/health" % PORT, timeout=3); ok = 1; break
            except Exception:
                time.sleep(2)
        if not ok:
            tail(SOUT); raise RuntimeError("pas de /health en 300 s")
        log("  · GATE UP OK (%.0fs)" % (time.time() - t0))

        # gate FIT + DEVICE + loaded + shortlist
        log("  · " + wait_log(r"successfully fit params", proc, "FIT"))
        dev = wait_log(r"using device CUDA[^\n]*", proc, "DEVICE")
        if "Tesla V100" not in dev:
            tail(SOUT); raise RuntimeError("device != V100 : " + dev)
        log("  · GATE DEVICE OK — " + dev[:120])
        wait_log(r"model loaded", proc, "LOADED")
        log("  · GATE MODEL LOADED OK")
        log("  · warmup 16 tok (JIT/graphs/shortlist)...")
        ok = False
        for w in range(1, 6):
            try:
                r = http_json("http://127.0.0.1:%d/v1/chat/completions" % PORT,
                              {"model": "local", "stream": False, "temperature": 0,
                               "max_tokens": 16,
                               "messages": [{"role": "user", "content": "Say OK."}]},
                              timeout=1800)
                if r.get("usage", {}).get("completion_tokens"):
                    ok = True
                    break
            except Exception as e:
                log("    warmup essai %d : %s" % (w, str(e)[:80]))
            time.sleep(10)
        if not ok:
            raise RuntimeError("warmup KO")
        log("  · GATE WARMUP OK")
        wait_log(r"CUDA MTP shortlist[^\n]*", proc, "SHORTLIST")
        log("  · GATE SHORTLIST OK")
        time.sleep(3)

        # REQ A : prompt NEUF 114k + 64 gen (HORS FENETRE — catch-up DEFER payé ici,
        # comme la W1 de stepC : fenêtre contaminée, pas celle qu'on trace)
        log("  · REQ A : prefill 114k neuf + 64 gen (hors fenêtre, ~3-5 min)...")
        d = json.load(open(PREFILL_JSON))
        rA = http_json("http://127.0.0.1:%d/v1/chat/completions" % PORT, d, timeout=1800)
        tA = rA.get("timings", {})
        log("  · REQ A done — predicted %.2f tok/s (fenêtre contaminée, cf. stepC)" %
            tA.get("predicted_per_second", 0))
        time.sleep(3)

        # REQ B : MÊME prompt (checkpoint ~4 tok nouveaux) + NTOK gen = FENETRE STEADY
        log("  · REQ B : FENETRE STEADY (%d tok, même prompt checkpointé)" % NTOK)
        time.sleep(5)

        ts0 = int(time.time() * 1e6)
        sampler = subprocess.Popen(["python3", SAMPLER, GPUTS, "0.1"])
        g = http_json("http://127.0.0.1:%d/v1/chat/completions" % PORT,
                      json.load(open(PREFILL_JSON)) | {"max_tokens": NTOK},
                      timeout=1800)
        time.sleep(2)
        ts1 = int(time.time() * 1e6)
        with open(BRACKET, "w") as f:
            f.write("TS_START=%d\nTS_END=%d\nNTOK=%d\n" % (ts0, ts1, NTOK))
        with open(GEN, "w") as f:
            json.dump(g, f)
        u = g.get("usage", {})
        t = g.get("timings", {})
        log("  · GEN : finish=%s prompt=%d cached=%d completion=%d" %
            (g.get("choices", [{}])[0].get("finish_reason"),
             u.get("prompt_tokens"), u.get("prompt_tokens_details", {}).get("cached_tokens"),
             u.get("completion_tokens")))
        if t:
            log("  · predicted %.2f tok/s (%.2f ms/tok)" %
                (t.get("predicted_per_second", 0), t.get("predicted_per_token_ms", 0)))
        rc = 0
    except Exception as e:
        log("  · ECHEC : %s" % e)
    finally:
        if sampler:
            sampler.terminate()
            try:
                sampler.wait(timeout=10)
            except Exception:
                pass
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except Exception:
                proc.kill(); proc.wait(timeout=10)
        for _ in range(60):
            if vram_mib() < 1000:
                break
            time.sleep(2)
        restore_service()

    # ---- lecture post-mortem du log serveur (spéc stats) ----
    log("=== extrait server.log (spec/eval) ===")
    for l in open(SOUT, errors="replace").read().splitlines():
        if re.search(r"statistics|print_timing.*eval time|draft acceptance|acc per pos|spec\s+begin", l):
            print("    | " + l[:200], flush=True)
    log("FIN — rc=%d service_restored=%s trace=%s" % (rc, _restored, OUT))
    sys.exit(0 if (_restored and rc == 0) else 3)


if __name__ == "__main__":
    main()
