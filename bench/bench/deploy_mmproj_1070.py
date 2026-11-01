#!/usr/bin/env python3
"""deploy_mmproj_1070.py — wrapper Python pour le deploy shell (le superviseur
exige un bench Python). Lance deploy_mmproj_1070.sh (backups + swap binaire
patché + swap .so dual + drop-in OLLAMA_MMPROJ_DEVICE=cuda1 + restart +
rollback auto si service inactif), puis vérifie l'état final et renvoie le rc.

RC : 0 = déployé + service actif + md5 attendus · 1 = échec (rollback fait ou à faire)
Le script shell restaure le service dans tous les cas (rollback intégré)."""
import subprocess, sys, time

DEPLOY = "/home/guile/llama-experiments/v100/dual-arch/go_patch/deploy_mmproj_1070.sh"
PROD_SO = "/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so"
PROD_BIN = "/home/guile/llama-experiments/ollama-v100/ollama"
SO_DUAL_MD5 = "b014d02682b23b3131752d65bacb15ba"
BIN_PATCHED_MD5 = "6f6d33a9f4a1cd5bec2494cf07d07134"
DROPIN = "/etc/systemd/system/ollama-v100.service.d/mmproj-1070.conf"

def log(m):
    print(time.strftime("[%H:%M:%S] ") + m, flush=True)

def sh(cmd, timeout=300):
    r = subprocess.run(["sudo", "bash", "-c", cmd], capture_output=True,
                       text=True, timeout=timeout)
    return r.returncode, r.stdout.strip(), r.stderr.strip()

def main():
    t0 = time.time()
    log("DEPLOIEMENT mmproj-1070 (binaire patché + .so dual + drop-in)")
    r = subprocess.run(["sudo", "bash", DEPLOY], capture_output=True, text=True, timeout=600)
    log(r.stdout[-3000:])
    if r.stderr.strip():
        log("STDERR: " + r.stderr[-2000:])
    log(f"deploy shell rc={r.returncode}")
    if r.returncode != 0:
        log("ECHEC du déploiement (rollback intégré par le script)")
        # état final pour le rapport
        rc, out, _ = sh("systemctl is-active ollama-v100; md5sum %s %s" % (PROD_SO, PROD_BIN))
        log("état: " + out.replace("\n", " | "))
        sys.exit(1)

    # vérifications post-déploiement
    import hashlib
    so = hashlib.md5(open(PROD_SO, "rb").read()).hexdigest()
    bi = hashlib.md5(open(PROD_BIN, "rb").read()).hexdigest()
    log(f".so prod : {so} (attendu {SO_DUAL_MD5[:8]})")
    log(f"bin prod : {bi} (attendu {BIN_PATCHED_MD5[:8]})")
    ok_so = so == SO_DUAL_MD5
    ok_bin = bi == BIN_PATCHED_MD5
    rc, out, _ = sh("systemctl is-active ollama-v100")
    active = out == "active"
    import os
    ok_drop = os.path.isfile(DROPIN)
    log(f"service: {out} | drop-in: {'OK' if ok_drop else 'MANQUANT'}")
    # log prod : le feature est-il armé ?
    rc, out, _ = sh('journalctl -u ollama-v100 --since "-60s" --no-pager | grep -i "mmproj\\|inference compute" | head -8')
    log("journal prod:\n" + out)
    ok = ok_so and ok_bin and active and ok_drop
    log(f"FIN en {time.time()-t0:.0f}s — {'DEPLOIEMENT OK' if ok else 'INCOHERENCE'}")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
