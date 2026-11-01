#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gpu_uuid_select.py — helpers d'identification GPU par UUID pour l'outillage bench.

Pourquoi ce fichier :
  L'index nvidia-smi et l'ordinal CUDA d'une même carte peuvent diverger (ici :
  V100 = smi 1 / CUDA ordinal 0 ; GTX 1070 = smi 0 / ordinal 1), et l'ordre peut
  changer après reboot / changement de slot. Tout index en dur est une bombe à
  retardement. Voir GPU_UUID_NSYS_CUPTI_20261030.md pour l'histoire complète.

API (jamais bloquante — repli loggué, valeur = état 30/10 de cette machine) :
  v100_cuda_ordinal() -> int   : ordinal CUDA à mettre dans CUDA_VISIBLE_DEVICES
  v100_smi_index()    -> str   : index nvidia-smi (--id=)
  v100_info()         -> dict  : {uuid, cuda_ordinal, smi_index, name, resolved: bool}

Import sûr : aucun effet de bord au import, pas de sys.exit.
"""
import os
import subprocess

_D = os.path.dirname(os.path.abspath(__file__))
_GPU_UUID_HELPER = os.path.join(_D, "gpu_uuid.py")
# repli = état 30/10 de cette machine (V100 : ordinal CUDA 0, smi index 1)
_FALLBACK_CUDA = 0
_FALLBACK_SMI = 1
_cache = {}


def _resolve():
    if "info" in _cache:
        return _cache["info"]
    info = {"uuid": "GPU-b3c229d2-2803-0af9-2fb8-067708067c1b",
            "cuda_ordinal": _FALLBACK_CUDA, "smi_index": _FALLBACK_SMI,
            "name": "Tesla V100-PCIE-32GB", "resolved": False}
    try:
        r = subprocess.run(["python3", _GPU_UUID_HELPER, "--json"],
                           capture_output=True, text=True, timeout=30)
        import json
        d = json.loads(r.stdout)
        if d.get("resolved"):
            info = {"uuid": d["uuid"], "cuda_ordinal": int(d["cuda_ordinal"]),
                    "smi_index": int(d["smi_index"]),
                    "name": d.get("name", info["name"]), "resolved": True}
    except Exception as e:
        try:
            print("[gpu_uuid_select] ATTENTION : resolution UUID echouee (%s) "
                  "- repli cuda_ordinal=%d smi_index=%d (etat 30/10)"
                  % (type(e).__name__, _FALLBACK_CUDA, _FALLBACK_SMI), flush=True)
        except Exception:
            pass
    _cache["info"] = info
    return info


def v100_cuda_ordinal():
    """Ordinal CUDA (driver) de la V100 — valeur a mettre dans CVD."""
    return _resolve()["cuda_ordinal"]


def v100_smi_index():
    """Index nvidia-smi de la V100 — valeur a mettre dans --id=."""
    return str(_resolve()["smi_index"])


def v100_info():
    return dict(_resolve())
