#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
smoke_check.py — contrôle rapide (30-60 s) que le llama-server DIRECT démarre
bien en CUDA avec EXACTEMENT l'env que monte bench_direct.py (import des
constantes du module, pas de copie).

USAGE : python3 smoke_check.py
Attend que la V100 soit libre (< 600 MiB), lance le server, attend /health,
vérifie GPU dans le log + VRAM > 1 GiB, tue le server, attend la libération.

Sort 0 = OK, 1 = échec.
"""
import os
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_direct as B


def vram():
    return int(subprocess.run(
        ["nvidia-smi", "-i", str(B._state["gpu_idx"]),
         "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout.strip())


def wait_free(deadline):
    while time.time() < deadline:
        if vram() < 600:
            return True
        time.sleep(5)
    return False


def main():
    B._state["gpu_idx"], B._state["gpu_cuda_idx"], B._state["gpu_name"] = B.find_gpu()
    print(f"  · GPU: {B._state['gpu_name']} — CUDA dev {B._state['gpu_cuda_idx']} (nvidia-smi idx {B._state['gpu_idx']})")
    if vram() >= 600:
        print(f"V100 occupée ({vram()} MiB) — j'attends qu'elle se libère (5 min max)…")
        if not wait_free(time.time() + 300):
            print(f"ECHEC: V100 encore pleine ({vram()} MiB) après 5 min")
            return 1

    # MÊME montage d'env que bench_direct.start_server() (build_env partagé,
    # aucune version parallèle)
    env = B.build_env(B._state["gpu_idx"])

    with open(B.OUT_SERVER_LOG, "w") as f:
        t0 = time.time()
        proc = subprocess.Popen([B.LLAMA_SERVER] + B.SERVER_ARGS,
                                stdout=f, stderr=subprocess.STDOUT, env=env)
        deadline = time.time() + 240
        ok, msg = False, "timeout /health"
        while time.time() < deadline:
            if proc.poll() is not None:
                msg = f"server MORT au startup (exit {proc.returncode}) — voir {B.OUT_SERVER_LOG}"
                break
            try:
                body = urllib.request.urlopen(
                    f"http://127.0.0.1:{B.PORT_DIRECT}/health", timeout=3).read().decode()
                if "ok" in body.lower():
                    time.sleep(3)
                    v = vram()
                    with open(B.OUT_SERVER_LOG, errors="replace") as lf:
                        head = lf.read()
                    gpu = B.GPU_NAME in head
                    wrong_gpu = ("GTX 1070" in head) and (B.GPU_NAME not in head)
                    kernel = "no kernel image" in head
                    print(f"  · health OK en {time.time()-t0:.0f}s | "
                          f"{B.GPU_NAME} dans le log: {gpu} | VRAM {B.GPU_NAME}: {v} MiB")
                    ok = gpu and v > 1024 and not kernel
                    if wrong_gpu:
                        msg = "ECHEC — la 1070 est devenue CUDA0 (no kernel image)"
                    elif kernel:
                        msg = "ECHEC — 'no kernel image' (mauvaise carte)"
                    elif not gpu:
                        msg = "ECHEC — la V100 n'apparaît pas dans le log de startup"
                    else:
                        msg = "OK — V100 confirmée, VRAM > 1 GiB"
                    break
            except Exception:
                pass
            time.sleep(1)
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:
            proc.kill()

    for _ in range(30):
        if vram() < 600:
            break
        time.sleep(2)
    print(f"  · VRAM libérée: {vram()} MiB | {msg}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
