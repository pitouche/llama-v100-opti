#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cupti_mtp_stepB.py — Étape B du profiling MTP (bootstrap_mtp_profiling_20261028.md §8).

Wrapper Python pour supervisor_bench.py (contrainte README : le bench passé au
superviseur doit être un script Python — precedent : nightrun8/run8_bench.py).

Exécute run_cupath_llama.sh (infra CUPTI 12.9 validée, trap EXIT = pkill +
systemctl start ollama-v100 + verify) et renvoie SON code de sortie tel quel.

Config capturée : config prod exacte (shortlist 131072 + DEFER_PROMPT=1, le
point lent), .so prod figé 8035caf4 (AUCUN swap), CVD=0 (V100), ctx ≈ 100k
tokens (≈ point 114k), NTOK decode.

USAGE :
  python3 supervisor_bench.py run cupti_mtp_stepB.py 64 --detach
"""
import os
import subprocess
import sys

CUPATH_DIR = "/home/guile/llama-experiments/h2/cupti129-test"
SCRIPT = os.path.join(CUPATH_DIR, "run_cupath_llama.sh")
NTOK = sys.argv[1] if len(sys.argv) > 1 else "64"

print(f"STEPB wrapper : bash {SCRIPT} {NTOK}", flush=True)
rc = subprocess.run(["bash", SCRIPT, NTOK]).returncode
print(f"STEPB wrapper : rc={rc}", flush=True)
sys.exit(rc)
