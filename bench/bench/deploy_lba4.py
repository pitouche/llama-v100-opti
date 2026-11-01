#!/usr/bin/env python3
"""
deploy_lba4.py — Déploiement du .so T3 lba4 (launch_bounds NT=1 -> minblocks=4) en prod, sous superviseur.

Cible : le levier B VALIDÉ 24/10 (+6,8 % @T=1, 162->124 regs, 0 spill, 3->4 blocs/SM).
   prod actuelle : fa3e75c2 (T3 + lba, minblocks NT=1=3)
   candidat      : feaf05aa (T3 + lba4, minblocks NT=1=4)  = libggml-cuda-t3lba_lb4.so
   seule diff     : fattn_vs_minblocks<D,NT>()  NT<=1 ? 3 : 2  ->  NT<=1 ? 4 : 2
   (kernel/launcher/pb auto sinon bit-identiques à la prod)

Le swap stoppe/démarre ollama-v100 -> couplé à supervisor_bench.py (le stop coupe ma
génération en cours ; le superviseur re-tire la reprise en fin de cycle).

Sequence :
  preconds : md5 prod == fa3e75c2, lba4 present == feaf05aa, lock tenu par le superviseur
  1. stop service + V100 libérée
  2. backup prod fa3e75c2 -> /var/backups/llama-experiments/
  3. cp lba4 -> prod .so, chown root:root, chmod 755
  4. start service, wait active
  5. sanity : petite generation MTP ON via :11434
  6. verify : md5 prod == feaf05aa, service active, VRAM

ROLLBACK :  sudo cp <BAK> <SO_PROD> && sudo systemctl restart ollama-v100

USAGE :
  cd /home/guile/llama-experiments/bench
  python3 supervisor_bench.py run deploy_lba4.py --detach
"""
import hashlib, json, os, re, signal, subprocess, sys, time, urllib.request
from gpu_uuid_select import v100_cuda_ordinal, v100_smi_index  # UUID GPU (jamais d'index en dur — voir GPU_UUID_NSYS_CUPTI_20261030.md)

D = os.path.dirname(os.path.abspath(__file__))
SVC = "ollama-v100"
SO_PROD = "/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so"
SO_LBA4 = "/home/guile/llama-experiments/h2/cupti129-test/e2/h2lib/lib/libggml-cuda-t3lba_lb4.so"
BAKDIR  = "/var/backups/llama-experiments"
MODEL = "qwen3.8-27b-q6-mtp-vision:latest"
PROD_PREFIX = "fa3e75c2"   # prod actuelle (T3 + lba)
LBA4_PREFIX = "feaf05aa"   # candidat B (T3 + lba4)
GPU_SMI_IDX = v100_smi_index()  # résolu par UUID (ne pas hardcoder)
_state = {"bak": None, "restored": False}


def log(m):
    print(time.strftime("[%H:%M:%S] ") + m, flush=True)
def run(cmd, timeout=120):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
def md5p(p):
    try:
        return hashlib.md5(open(p, "rb").read()).hexdigest()[:8]
    except OSError:
        return "?"
def svc_state():
    return run(["systemctl", "is-active", SVC]).stdout.strip()
def vram_mib():
    r = run(["nvidia-smi", f"--id={GPU_SMI_IDX}",
             "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    v = r.stdout.strip()
    return int(v) if v.isdigit() else -1

def restore_service():
    global _state
    try:
        if svc_state() != "active":
            log("  · restauration : systemctl start " + SVC)
            run(["sudo", "systemctl", "start", SVC], timeout=120)
        for _ in range(90):
            if svc_state() == "active":
                break
            time.sleep(2)
        _state["restored"] = svc_state() == "active"
        log(f"  · service {SVC} = {svc_state()}")
    except Exception as e:
        log(f"  · ERREUR restauration : {e}")
        _state["restored"] = False

def on_sig(s, f):
    restore_service()
    sys.exit(128 + s)

def stop_and_free():
    log("  · stop " + SVC)
    run(["sudo", "systemctl", "stop", SVC], timeout=120)
    t0 = time.time()
    while time.time() - t0 < 180:
        if vram_mib() < 1024:
            log(f"  · V100 libérée ({vram_mib()} MiB) en {time.time()-t0:.0f}s")
            return True
        time.sleep(3)
    log(f"  · ATTENTION V100 pas totalement libérée (VRAM={vram_mib()} MiB)")
    return False

def sanity_gen():
    """Petite generation MTP ON via :11434. Retourne (ok, texte)."""
    data = json.dumps({
        "model": MODEL, "prompt": "Dis OK en un mot.",
        "stream": False, "options": {"num_predict": 12, "temperature": 0}}).encode()
    req = urllib.request.Request("http://127.0.0.1:11434/api/generate",
                                 data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            out = json.loads(r.read().decode())
        return True, (out.get("response") or "").strip()
    except Exception as e:
        return False, str(e)

def main():
    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)

    log(f"DEPLOY lba4 (levier B) — prod md5={md5p(SO_PROD)} (attendu {PROD_PREFIX}), lba4 md5={md5p(SO_LBA4)} (attendu {LBA4_PREFIX})")
    if md5p(SO_PROD) != PROD_PREFIX:
        sys.exit(f"ABORT : .so prod {md5p(SO_PROD)} != {PROD_PREFIX} — prod dérivé, rien fait.")
    if md5p(SO_LBA4) != LBA4_PREFIX:
        sys.exit(f"ABORT : .so lba4 {md5p(SO_LBA4)} != {LBA4_PREFIX} — candidat inattendu, rien fait.")

    try:
        stop_and_free()

        # 2) backup prod
        ts = time.strftime("%Y%m%d-%H%M%S")
        bak = os.path.join(BAKDIR, f"libggml-cuda.so.prod-fa3e75c2-{ts}")
        r = run(["sudo", "cp", SO_PROD, bak], timeout=120)
        if r.returncode != 0 or md5p(bak) != PROD_PREFIX:
            restore_service()
            sys.exit(f"ABORT : backup échoué (rc={r.returncode}, md5={md5p(bak)}) — service remonté, rien swapé.")
        _state["bak"] = bak
        log(f"  · backup prod fa3e75c2 -> {bak} (md5={md5p(bak)})")

        # 3) swap
        r = run(["sudo", "cp", SO_LBA4, SO_PROD], timeout=120)
        run(["sudo", "chown", "root:root", SO_PROD], timeout=30)
        run(["sudo", "chmod", "755", SO_PROD], timeout=30)
        new = md5p(SO_PROD)
        log(f"  · swap lba4 -> prod (nouveau md5={new})")
        if new != LBA4_PREFIX:
            log("  · !! md5 post-swap inattendu -> ROLLBACK")
            run(["sudo", "cp", bak, SO_PROD], timeout=120)
            restore_service()
            sys.exit(f"ABORT : md5 post-swap {new} != {LBA4_PREFIX} — ROLLBACK, service remonté.")

        # 4) start
        run(["sudo", "systemctl", "start", SVC], timeout=120)
        for _ in range(60):
            if svc_state() == "active":
                break
            time.sleep(2)
        log(f"  · service = {svc_state()}")

        # 5) sanity
        time.sleep(8)
        for _ in range(120):
            ok, txt = sanity_gen()
            if ok:
                break
            time.sleep(3)
        if not ok:
            log(f"  · !! sanity generation a échoué : {txt}  (le .so est installé mais la gen échoue)")
        else:
            log(f"  · SANITY OK : {txt!r}")

        # 6) verify
        log(f"  · VERIFY md5 prod = {md5p(SO_PROD)} (attendu {LBA4_PREFIX})")
        log(f"  · VERIFY service = {svc_state()}, VRAM={vram_mib()} MiB")

    finally:
        restore_service()

    final_md5 = md5p(SO_PROD)
    final_svc = svc_state()
    ok = (final_md5 == LBA4_PREFIX) and (final_svc == "active")
    log(f"FIN deploy_lba4 — md5={final_md5}, service={final_svc}, bak={_state['bak']}, ok={ok}")
    log(f"ROLLBACK si besoin : sudo cp {_state['bak']} {SO_PROD} && sudo systemctl restart {SVC}")
    sys.exit(0 if ok else 3)

if __name__ == "__main__":
    main()
