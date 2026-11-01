#!/usr/bin/env python3
"""Test de reproductibilité — perte de l'outil terminal après reprise (16/10).

Fait exprès ce qu'un vrai bench fait : stoppe ollama-v100 (l'inférence OWUI
en cours échoue « Ollama: Cannot connect… 11434 »), coupe ~20 s, restaure le
service et attend qu'il soit actif. Le superviseur (fix A : réinjection de
terminal_id) doit alors reprendre le chat AVEC l'outil terminal.

Durée totale : ~35-40 s. rc=0 si le service est actif à la fin.
Séquence identique à run9c3_bench.py (sudo systemctl stop/start), sans
aucune autre action sur la V100 (pas de lib, pas de config).
"""
import subprocess
import sys
import time

DOWNSLEEP_S = 8      # délai avant le stop : laisser le tir agent se mettre en place
DOWN_FOR_S = 20      # fenêtre de coupure
START_WAIT_S = 90    # attente max du retour actif


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def state():
    return run(["systemctl", "is-active", "ollama-v100"]).stdout.strip()


def wait_active(timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if state() == "active":
            return True
        time.sleep(2)
    return state() == "active"


print(f"=== TEST terminal-loss : state initial = {state()} ===", flush=True)
time.sleep(DOWNSLEEP_S)
r = run(["sudo", "systemctl", "stop", "ollama-v100"])
print(f"stop rc={r.returncode} state={state()}", flush=True)
time.sleep(DOWN_FOR_S)
r = run(["sudo", "systemctl", "start", "ollama-v100"])
print(f"start rc={r.returncode}", flush=True)
ok = wait_active(START_WAIT_S)
print(f"=== FIN : state = {state()} ===", flush=True)
sys.exit(0 if ok else 1)
