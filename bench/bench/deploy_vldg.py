#!/usr/bin/env python3
"""
deploy_vldg.py — Déploiement du .so T3 vldg (double-buffer V, GO microbench + e2e)
en PROD, sous superviseur. 28/10.

Cible : le rewrite vldg VALIDÉ 28/10 :
  - microbench mb_fa : T=1 +11 %, T=6..8 +19,6 %, bit-identique (maxrel=0)
  - e2e MTP-OFF : decode +2.25 % @114k, +2.81 % @196k (prefill ±0.12 %)
  - e2e MTP-ON (config prod) : decode +1.76 % @114k, +0.40 % @160k,
    acceptance MTP identique (bit-exact sur les chemins f16 draft + q8_0 target)
  - SASS : REG:128 STACK:0 (levier B intact), 17 LDG.E.128.CONSTANT (levier C intact),
    3792 instr = prod, diff = 48 instr (réordonnancement local, 0 instr ajoutée)

   prod actuelle : 4fd152f9 (dual-arch sm_70+sm_61 = T3 + lba4 + cvec ; mmproj sur la 1070)
   candidat      : <md5 rempli après build dual> (dual-arch sm_70+sm_61 + vldg :
                   phase 2 pipelinée, fenêtres V du chunk ch+1 en vol pendant les mmas du
                   chunk ch — mêmes octets/même déquant/mêmes mma -> bit-exact)
   binaire ollama : 6f6d33a9 (patché mmproj-1070, INCHANGÉ)
   drop-in systemd : mmproj-1070.conf (OLLAMA_MMPROJ_DEVICE=cuda1, INCHANGÉ)

Le swap stoppe/démarre ollama-v100 -> couplé à supervisor_bench.py (le stop coupe ma
génération en cours ; le superviseur re-tire la reprise en fin de cycle).

Sequence :
  preconds : md5 prod .so == 4fd152f9, candidat <md5> présent (150 ELF sm_70 + 150 sm_61),
             binaire ollama 6f6d33a9, drop-in présent
  1. stop service + V100 libérée
  2. backup prod 4fd152f9 -> /var/backups/llama-experiments/libggml-cuda.so.prod-4fd152f9-<ts>
  3. cp candidat -> prod .so, chown ollama:ollama, chmod 755
  4. start service, wait active
  5. sanity : petite generation via :11434 (le modèle vision charge le mmproj sur la 1070)
  6. verify : md5 prod == <md5>, service active, VRAM, drop-in intact

ROLLBACK :  sudo cp /var/backups/llama-experiments/libggml-cuda.so.prod-4fd152f9-<ts> \
             /home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so \
             && sudo systemctl restart ollama-v100

USAGE :
  cd /home/guile/llama-experiments/bench
  python3 supervisor_bench.py run deploy_vldg.py --detach
"""
import hashlib, json, os, signal, subprocess, sys, time, urllib.request
from gpu_uuid_select import v100_cuda_ordinal, v100_smi_index  # UUID GPU (jamais d'index en dur — voir GPU_UUID_NSYS_CUPTI_20261030.md)

SVC = "ollama-v100"
SO_PROD = "/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so"
SO_CAND = "/home/guile/llama-experiments/v100/dual-arch/lib/libggml-cuda-dual.so"
BAKDIR  = "/var/backups/llama-experiments"
MODEL = "qwen3.8-27b-q6-mtp-vision:latest"
PROD_PREFIX = "4fd152f9"   # prod actuelle (dual-arch, T3 + lba4 + cvec)
CAND_PREFIX = "8035caf4"   # .so dual vldg (28/10 15:00 UTC, tree dual v100/dual-arch)
OLLAMA_BIN = "/home/guile/llama-experiments/ollama-v100/ollama"
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

def elf_counts(p):
    """(n_sm70, n_sm61) via cuobjdump."""
    try:
        r = run(["/usr/local/cuda/bin/cuobjdump", "--list-elf", p], timeout=120)
        n70 = sum(1 for l in r.stdout.splitlines() if "sm_70" in l)
        n61 = sum(1 for l in r.stdout.splitlines() if "sm_61" in l)
        return n70, n61
    except Exception:
        return -1, -1

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

    log(f"DEPLOY vldg (rewrite double-buffer V) — prod md5={md5p(SO_PROD)} (attendu {PROD_PREFIX}), "
        f"cand md5={md5p(SO_CAND)} (attendu {CAND_PREFIX})")
    if md5p(SO_PROD) != PROD_PREFIX:
        sys.exit(f"ABORT : .so prod {md5p(SO_PROD)} != {PROD_PREFIX} — prod inattendu, rien fait.")
    if md5p(SO_CAND) != CAND_PREFIX:
        sys.exit(f"ABORT : .so cand {md5p(SO_CAND)} != {CAND_PREFIX} — candidat inattendu, rien fait.")
    n70, n61 = elf_counts(SO_CAND)
    log(f"  · ELF candidat : sm_70={n70}, sm_61={n61}")
    if n70 < 100 or n61 < 100:
        sys.exit(f"ABORT : ELF candidats insuffisants (sm_70={n70}, sm_61={n61}) — rien fait.")
    obin = md5p(OLLAMA_BIN)
    log(f"  · binaire ollama md5 = {obin} (attendu 6f6d33a9)")
    if obin != "6f6d33a9":
        log(f"  · ATTENTION : binaire ollama {obin} != 6f6d33a9 — on continue (inchangé par ce deploy) mais signaler")

    try:
        stop_and_free()

        # 2) backup prod
        ts = time.strftime("%Y%m%d-%H%M%S")
        bak = os.path.join(BAKDIR, f"libggml-cuda.so.prod-4fd152f9-{ts}")
        r = run(["sudo", "cp", SO_PROD, bak], timeout=120)
        if r.returncode != 0 or md5p(bak) != PROD_PREFIX:
            restore_service()
            sys.exit(f"ABORT : backup échoué (rc={r.returncode}, md5={md5p(bak)}) — service remonté, rien swapé.")
        _state["bak"] = bak
        log(f"  · backup prod 4fd152f9 -> {bak} (md5={md5p(bak)})")

        # 3) swap
        r = run(["sudo", "cp", SO_CAND, SO_PROD], timeout=120)
        run(["sudo", "chown", "ollama:ollama", SO_PROD], timeout=30)
        run(["sudo", "chmod", "755", SO_PROD], timeout=30)
        new = md5p(SO_PROD)
        log(f"  · swap cand -> prod (nouveau md5={new})")
        if new != CAND_PREFIX:
            log("  · !! md5 post-swap inattendu -> ROLLBACK")
            run(["sudo", "cp", bak, SO_PROD], timeout=120)
            restore_service()
            sys.exit(f"ABORT : md5 post-swap {new} != {CAND_PREFIX} — ROLLBACK, service remonté.")

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
        log(f"  · VERIFY md5 prod = {md5p(SO_PROD)} (attendu {CAND_PREFIX})")
        log(f"  · VERIFY service = {svc_state()}, VRAM={vram_mib()} MiB")
        dropin = "/etc/systemd/system/ollama-v100.service.d/mmproj-1070.conf"
        log(f"  · VERIFY drop-in mmproj-1070 = {'présent' if os.path.isfile(dropin) else 'MANQUANT !'}")

    finally:
        restore_service()

    final_md5 = md5p(SO_PROD)
    final_svc = svc_state()
    ok = (final_md5 == CAND_PREFIX) and (final_svc == "active")
    log(f"FIN deploy_vldg — md5={final_md5}, service={final_svc}, bak={_state['bak']}, ok={ok}")
    log(f"ROLLBACK si besoin : sudo cp {_state['bak']} {SO_PROD} && sudo systemctl restart {SVC}")
    sys.exit(0 if ok else 3)

if __name__ == "__main__":
    main()
