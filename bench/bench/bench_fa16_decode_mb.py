#!/usr/bin/env python3
"""
FA16 DECODE microbench — mb_fa_pf à T=1 / T=32 (forme de decode) sur n_kv=114688.
Objectif : vérifier que le .so dual (legs B/C) ne régresse PAS la voie de decode FA.
Si T=1 est identique A/B/C → l'écart decode e2e est thermique/version, pas FA.
Service stoppé, restauration assurée. Lancer via superviseur.
"""
import json, os, signal, socket, subprocess, sys, time
from datetime import datetime

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
from gpu_uuid_select import v100_cuda_ordinal, v100_smi_index

CVD, SMI = str(v100_cuda_ordinal()), str(v100_smi_index())
SERVICE = "ollama-v100"
MB = os.path.join(D, "mb_fa_pf")
SO_PROD = "/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so"
SO_DUAL = "/home/guile/llama-experiments/v100/dual-arch/lib/libggml-cuda-dual.so"
CUDA_LIBS = ["/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12",
             "/home/guile/llama-experiments/ollama-v100/lib/ollama",
             "/usr/local/cuda-12.9/lib64"]
TS = datetime.now().strftime("%Y%m%d-%H%M%S")
OUT_JSONL = os.path.join(D, f"fa16_dec_{TS}.jsonl")
OUT_LOG = os.path.join(D, f"fa16_dec_{TS}.log")

CASES = [
    ("A_prod",      SO_PROD, {}, 1, 114688, 5),
    ("B_16x2",      SO_DUAL, {"GGML_CUDA_VOLTA_FA_NCOLS1": "16"}, 1, 114688, 5),
    ("C_compact",   SO_DUAL, {"GGML_CUDA_VOLTA_FA_COMPACT_TMAX": "8192"}, 1, 114688, 5),
    ("A_prod_t32",  SO_PROD, {}, 32, 114688, 5),
    ("B_16x2_t32",  SO_DUAL, {"GGML_CUDA_VOLTA_FA_NCOLS1": "16"}, 32, 114688, 5),
    ("C_compact_t32", SO_DUAL, {"GGML_CUDA_VOLTA_FA_COMPACT_TMAX": "8192"}, 32, 114688, 5),
]
_restored = False


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(OUT_LOG, "a") as f:
        f.write(line + "\n")


def svc_state():
    return subprocess.run(["systemctl", "is-active", SERVICE],
                          capture_output=True, text=True).stdout.strip()


def vram_mib():
    r = subprocess.run(["nvidia-smi", f"--id={SMI}", "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True)
    v = r.stdout.strip()
    return int(v) if v.isdigit() else -1


def wait_idle(timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        m = vram_mib()
        if m < 1024:
            r = subprocess.run(["nvidia-smi", f"--id={SMI}", "--query-gpu=utilization.gpu",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True)
            if r.stdout.strip().isdigit() and int(r.stdout.strip()) < 5:
                return True
        time.sleep(2)
    return vram_mib() < 1024


def restore_service():
    global _restored
    try:
        if svc_state() != "active":
            log(f"  · restauration : systemctl start {SERVICE}")
            subprocess.run(["sudo", "systemctl", "start", SERVICE], timeout=120, check=False)
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


def run_case(name, so, extra, T, n_kv, iters):
    ldp = ":".join(d for d in CUDA_LIBS if os.path.isdir(d))
    envkv = " ".join(f"{k}={v}" for k, v in extra.items())
    cmd = (f"CUDA_VISIBLE_DEVICES={CVD} MB_FA_SO={so} LD_LIBRARY_PATH='{ldp}'"
           + (f" {envkv}" if envkv else "") + f" {MB} {T} {n_kv} {iters}")
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
    row = {"case": name, "T": T, "n_kv": n_kv, "wall_s": round(time.time() - t0, 1),
           "rc": r.returncode, "env": extra}
    out = r.stdout + r.stderr
    try:
        j = json.loads(out[out.index("{"): out.rindex("}") + 1])
        row.update(j)
    except Exception:
        row["err"] = out[-300:]
    log(f"  · {name}: rc={r.returncode} us_med={row.get('us_med','?')} finite={row.get('finite','?')}")
    return row


def main():
    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)
    open(OUT_LOG, "a").close()
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 8917)); s.close()
    except OSError:
        s.close(); sys.exit("port 8917 occupé")
    log(f"FA16 DECODE microbench — {TS} — service stoppé")
    log("  · stop " + SERVICE)
    subprocess.run(["sudo", "systemctl", "stop", SERVICE], timeout=120, check=False)
    t0 = time.time()
    while time.time() - t0 < 180:
        if vram_mib() < 1024:
            break
        time.sleep(3)
    log(f"  · V100 libérée en {time.time()-t0:.0f}s")
    rows = []
    try:
        for name, so, extra, T, n_kv, iters in CASES:
            if not wait_idle():
                rows.append({"case": name, "error": "gpu busy"})
                continue
            row = run_case(name, so, extra, T, n_kv, iters)
            rows.append(row)
            with open(OUT_JSONL, "a") as f:
                f.write(json.dumps(row) + "\n")
    finally:
        restore_service()
    # résumé
    base = {r["case"]: r for r in rows if r.get("us_med") and r["T"] == 1}
    log("=== RÉSUMÉ decode (T=1, n_kv=114688) ===")
    a = base.get("A_prod", {}).get("us_med")
    for c in ("B_16x2", "C_compact"):
        b = base.get(c, {}).get("us_med")
        if a and b:
            log(f"  {c} vs A_prod : {b/a*100-100:+.1f}%  ({b:.0f} vs {a:.0f} us)")
    log(f"FIN — service restauré = {_restored}")
    sys.exit(0 if _restored else 2)


if __name__ == "__main__":
    main()
