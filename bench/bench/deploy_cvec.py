#!/usr/bin/env python3
"""
deploy_cvec.py — Déploiement du .so T3 + cvec (load K q8_0 vectorisé 128-bit) en prod, sous superviseur.

Cible : le levier C VALIDÉ 26/10 (microbench mb_fa +10,7 % @T=1, 451→402 µs, bit-identique
maxrel=0 ; e2e A/B ab_cvec.py +1,9/+2,6/+3,5 % @32/65/98k, MTP-OFF).

   prod actuelle : b014d026 (dual-arch sm_70+sm_61, mmproj sur la 1070 — partie sm_70 = feaf05aa)
   candidat      : 4fd152f9 (dual-arch sm_70+sm_61 + patch C : 5 LDG.128 au lieu de 17 LDG.32
                    par block-pair du load K q8_0, 122-128 regs, 0 spill)
   binaire ollama : 6f6d33a9 (patché mmproj-1070, INCHANGÉ)
   drop-in systemd : mmproj-1070.conf (OLLAMA_MMPROJ_DEVICE=cuda1, INCHANGÉ)

Le swap stoppe/démarre ollama-v100 -> couplé à supervisor_bench.py (le stop coupe ma
génération en cours ; le superviseur re-tire la reprise en fin de cycle).

Sequence :
  preconds : md5 prod .so == b014d026, candidat 4fd152f9 présent (150 ELF sm_70 + 150 sm_61),
             binaire 6f6d33a9, drop-in présent
  1. stop service + V100 libérée
  2. backup prod b014d026 -> /var/backups/llama-experiments/libggml-cuda.so.prod-b014d026-<ts>
  3. cp candidat -> prod .so, chown ollama:ollama, chmod 755
  4. start service, wait active
  5. sanity : petite generation via :11434 (le modèle vision charge le mmproj sur la 1070)
  6. verify : md5 prod == 4fd152f9, service active, VRAM, drop-in intact

ROLLBACK :  sudo cp /var/backups/llama-experiments/libggml-cuda.so.prod-b014d026-<ts> \
             /home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so \
             && sudo systemctl restart ollama-v100

USAGE :
  cd /home/guile/llama-experiments/bench
  python3 supervisor_bench.py run deploy_cvec.py --detach
"""
import hashlib, json, os, signal, subprocess, sys, time, urllib.request
from gpu_uuid_select import v100_cuda_ordinal, v100_smi_index  # UUID GPU (jamais d'index en dur — voir GPU_UUID_NSYS_CUPTI_20261030.md)

SVC = "ollama-v100"
SO_PROD = "/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so"
SO_CVEC = "/home/guile/llama-experiments/v100/dual-arch/lib/libggml-cuda-dual.so"
BAKDIR  = "/var/backups/llama-experiments"
MODEL = "qwen3.8-27b-q6-mtp-vision:latest"
PROD_PREFIX = "b014d026"   # prod actuelle (dual-arch, partie sm_70 = feaf05aa)
CVEC_PREFIX = "4fd152f9"   # candidat C (dual-arch + load K vectorisé 128-bit)
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
    """Petite generation via :11434 (le runner prod = MTP ON, sans rien modifier)."""
    data = json.dumps({
        "model": MODEL, "prompt": "Dis OK en un mot.",
        "stream": False, "options": {"num_predict": 12, "temperature": 0}}).encode()
    req = urllib.request.Request("http://127.0.0.1:11434/api/generate",
                                 data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            out = json.loads(r.read().decode())
        return True, (out.get("response") or "").strip()
    except Exception as e:
        return False, str(e)

def main():
    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)

    log(f"DEPLOY cvec (levier C) — prod md5={md5p(SO_PROD)} (attendu {PROD_PREFIX}), "
        f"cvec md5={md5p(SO_CVEC)} (attendu {CVEC_PREFIX})")
    if md5p(SO_PROD) != PROD_PREFIX:
        sys.exit(f"ABORT : .so prod {md5p(SO_PROD)} != {PROD_PREFIX} — prod inattendu, rien fait.")
    if md5p(SO_CVEC) != CVEC_PREFIX:
        sys.exit(f"ABORT : .so cvec {md5p(SO_CVEC)} != {CVEC_PREFIX} — candidat inattendu, rien fait.")

    try:
        stop_and_free()

        # 2) backup prod
        ts = time.strftime("%Y%m%d-%H%M%S")
        bak = os.path.join(BAKDIR, f"libggml-cuda.so.prod-b014d026-{ts}")
        r = run(["sudo", "cp", SO_PROD, bak], timeout=120)
        if r.returncode != 0 or md5p(bak) != PROD_PREFIX:
            restore_service()
            sys.exit(f"ABORT : backup échoué (rc={r.returncode}, md5={md5p(bak)}) — service remonté, rien swapé.")
        _state["bak"] = bak
        log(f"  · backup prod b014d026 -> {bak} (md5={md5p(bak)})")

        # 3) swap
        r = run(["sudo", "cp", SO_CVEC, SO_PROD], timeout=120)
        run(["sudo", "chown", "ollama:ollama", SO_PROD], timeout=30)
        run(["sudo", "chmod", "755", SO_PROD], timeout=30)
        new = md5p(SO_PROD)
        log(f"  · swap cvec -> prod (nouveau md5={new})")
        if new != CVEC_PREFIX:
            log("  · !! md5 post-swap inattendu -> ROLLBACK")
            run(["sudo", "cp", bak, SO_PROD], timeout=120)
            restore_service()
            sys.exit(f"ABORT : md5 post-swap {new} != {CVEC_PREFIX} — ROLLBACK, service remonté.")

        # 4) start
        run(["sudo", "systemctl", "start", SVC], timeout=120)
        for _ in range(60):
            if svc_state() == "active":
                break
            time.sleep(2)
        log(f"  · service = {svc_state()}")

        # 5) sanity
        time.sleep(8)
        ok, txt = False, ""
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
        log(f"  · VERIFY md5 prod = {md5p(SO_PROD)} (attendu {CVEC_PREFIX})")
        log(f"  · VERIFY service = {svc_state()}, VRAM={vram_mib()} MiB")
        dropin = "/etc/systemd/system/ollama-v100.service.d/mmproj-1070.conf"
        log(f"  · VERIFY drop-in mmproj-1070 = {'présent' if os.path.isfile(dropin) else 'MANQUANT !'}")

    finally:
        restore_service()

    final_md5 = md5p(SO_PROD)
    final_svc = svc_state()
    ok = (final_md5 == CVEC_PREFIX) and (final_svc == "active")
    log(f"FIN deploy_cvec — md5={final_md5}, service={final_svc}, bak={_state['bak']}, ok={ok}")
    log(f"ROLLBACK si besoin : sudo cp {_state['bak']} {SO_PROD} && sudo systemctl restart {SVC}")
    sys.exit(0 if ok else 3)

if __name__ == "__main__":
    main()
