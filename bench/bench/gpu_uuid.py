#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gpu_uuid.py — identification des GPU par UUID MATÉRIEL (source de vérité).

CONTEXTE (30/10, confirmé par probe_cuda_uuid.py) :
   Sur cette machine, l'ordinal CUDA (driver, ce que CUDA_VISIBLE_DEVICES attend)
   et l'index nvidia-smi sont INVERSÉS :

       GPU             CUDA ordinal   nvidia-smi index   UUID
       --------------  -------------  ----------------  ----------------------------
       V100-32GB        0              1                 GPU-b3c229d2-2803-0af9-2fb8-067708067c1b
       GTX 1070         1              0                 GPU-60540de1-34b4-cae2-030e-0a9ae61b36f0

   L'ordre des deux énumérations peut changer (reboot, changement de slot,
   changement de GPU) : il est donc INTERDIT d'hardcoder 0 ou 1. On ne passe
   JAMAIS un index en dur : on résout par UUID à chaque fois.

API :
   resolve_v100() -> dict : {"uuid":..., "cuda_ordinal": int, "smi_index": int, "name":...}
        Aborte (sys.exit 3) si le V100 est absent ou ambigu.
   env_for_v100(base_env=None) -> dict : copie d'un env avec CUDA_VISIBLE_DEVICES
        réglé sur l'ordinal CUDA du V100 (et pas d'autre).
   smi_args_v100() -> list : ["--id=<smi_index>"] pour nvidia-smi.
   cli : python3 gpu_uuid.py  (imprime la table + vérifie la cohérence)

Note CUDA_VISIBLE_DEVICES : depuis CUDA 12, CVD accepte aussi "GPU-<uuid>"
directement — mais tous les outils/binaires ici (llama-server, ncu, nsys)
sont testés avec un ordinal numérique. On résout donc l'ordinal depuis l'UUID.
"""
import ctypes
import subprocess
import sys

UUID_V100 = "GPU-b3c229d2-2803-0af9-2fb8-067708067c1b"


def _cu_uuid_enum():
    """-> {uuid: cuda_ordinal} via driver API (pas de contexte ouvert)."""
    cuda = ctypes.CDLL("libcuda.so.1")

    class CUuuid(ctypes.Structure):
        _fields_ = [("bytes", ctypes.c_byte * 16)]

    def fmt_uu(u):
        h = bytes(x & 0xFF for x in u.bytes).hex()
        return "GPU-" + "-".join([h[0:8], h[8:12], h[12:16], h[16:20], h[20:32]])

    if cuda.cuInit(0) != 0:
        return None
    n = ctypes.c_int(0)
    if cuda.cuDeviceGetCount(ctypes.byref(n)) != 0:
        return None
    out = {}
    for i in range(n.value):
        dev = ctypes.c_int(0)
        if cuda.cuDeviceGet(ctypes.byref(dev), i) != 0:
            continue
        u = CUuuid()
        if cuda.cuDeviceGetUuid(ctypes.byref(u), dev) == 0:
            out[fmt_uu(u)] = i
    return out


def _smi_uuid_enum():
    """-> {uuid: smi_index} via nvidia-smi."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"],
            capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    out = {}
    for line in r.stdout.strip().splitlines():
        idx, uuid, name = [p.strip() for p in line.split(",")]
        out[uuid] = int(idx)
    return out


def resolve_v100(uuid=UUID_V100, verbose=True):
    """Résout l'identité complète du V100 par UUID. Sys.exit(3) si absent/ambigu."""
    cu = _cu_uuid_enum()
    smi = _smi_uuid_enum()
    if cu is None or smi is None:
        print("[gpu_uuid] ERREUR : impossible d'interroger le driver / nvidia-smi", file=sys.stderr)
        sys.exit(3)
    if uuid not in cu:
        print("[gpu_uuid] ERREUR : le GPU %s n'est pas visible par CUDA.\n"
              "  CUDA voit : %s" % (uuid, sorted(cu.keys())), file=sys.stderr)
        sys.exit(3)
    if uuid not in smi:
        print("[gpu_uuid] ERREUR : le GPU %s n'est pas visible par nvidia-smi.\n"
              "  smi voit : %s" % (uuid, sorted(smi.keys())), file=sys.stderr)
        sys.exit(3)
    info = {"uuid": uuid, "cuda_ordinal": cu[uuid], "smi_index": smi[uuid],
            "name": "Tesla V100-PCIE-32GB"}
    if verbose:
        print("[gpu_uuid] V100 : cuda_ordinal=%d smi_index=%d uuid=%s (les deux peuvent diverger)"
              % (info["cuda_ordinal"], info["smi_index"], uuid))
    return info


def env_for_v100(base_env=None):
    """Copie de base_env (ou os.environ) avec CVD = ordinal CUDA du V100."""
    import os
    env = dict(base_env if base_env is not None else os.environ)
    v = resolve_v100(verbose=False)
    env["CUDA_VISIBLE_DEVICES"] = str(v["cuda_ordinal"])
    return env, v


def smi_args_v100():
    v = resolve_v100(verbose=False)
    return ["--id=%d" % v["smi_index"]], v


def safe_resolve_v100(uuid=UUID_V100):
    """Comme resolve_v100 mais renvoie None au lieu de sys.exit(3).

    Pour l'outillage bench : un échec de résolution (driver indisponible,
    helper manquant) ne doit pas planter un script au moment de l'import —
    l'appelant choisit alors son repli (les valeurs 30/10 : cuda_ordinal=0,
    smi_index=1) et le log.
    """
    cu = _cu_uuid_enum()
    smi = _smi_uuid_enum()
    if cu is None or smi is None or uuid not in cu or uuid not in smi:
        return None
    return {"uuid": uuid, "cuda_ordinal": cu[uuid], "smi_index": smi[uuid],
            "name": "Tesla V100-PCIE-32GB"}


def main():
    if "--json" in sys.argv:
        # sortie machine (json) : {"resolved": bool, ...} — pour gpu_uuid_select.py
        import json
        info = safe_resolve_v100()
        if info is None:
            print(json.dumps({"resolved": False}))
            return
        print(json.dumps(dict(info, resolved=True)))
        return
    if "--cvt-ordinal" in sys.argv:
        # sortie machine : seul l'ordinal CUDA du V100 (pour scripts shell)
        print(resolve_v100(verbose=False)["cuda_ordinal"])
        return
    if "--smi-index" in sys.argv:
        print(resolve_v100(verbose=False)["smi_index"])
        return
    cu = _cu_uuid_enum() or {}
    smi = _smi_uuid_enum() or {}
    allu = sorted(set(cu) | set(smi))
    print("=== TABLE GPU par UUID ===")
    print("%-42s  %-12s  %-12s" % ("UUID", "CUDA ordinal", "smi index"))
    for u in allu:
        print("%-42s  %-12s  %-12s"
              % (u, cu.get(u, "—"), smi.get(u, "—")))
    v = resolve_v100()
    print("\nOK : le V100 est résolu par UUID. Utilise gpu_uuid.resolve_v100() "
          "dans l'outillage, JAMAIS un index en dur.")


if __name__ == "__main__":
    main()
