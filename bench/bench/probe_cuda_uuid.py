#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_cuda_uuid.py — vérité de terrain : ordinal CUDA (driver) vs index nvidia-smi vs UUID.

Aucun contexte CUDA n'est ouvert (cuDeviceGet + propriétés seul) → impact prod = nul.
À lancer SANS CUDA_VISIBLE_DEVICES pour voir l'enum brute.

Usage : python3 probe_cuda_uuid.py
"""
import ctypes
import subprocess

cuda = ctypes.CDLL("libcuda.so.1")


class CUuuid(ctypes.Structure):
    _fields_ = [("bytes", ctypes.c_byte * 16)]


def fmt_uu(u: CUuuid) -> str:
    b = bytes(x & 0xFF for x in u.bytes)
    h = b.hex()
    return "GPU-" + "-".join([h[0:8], h[8:12], h[12:16], h[16:20], h[20:32]])


def main():
    assert cuda.cuInit(0) == 0
    nv = ctypes.c_int(0)
    assert cuda.cuDriverGetVersion(ctypes.byref(nv)) == 0
    print("cuDriverGetVersion :", nv.value, "(0x%X)" % nv.value)

    n = ctypes.c_int(0)
    assert cuda.cuDeviceGetCount(ctypes.byref(n)) == 0
    cuda_map = {}
    print("\n=== ENUMERATION CUDA (ordinaux, sans CVD) ===")
    for i in range(n.value):
        dev = ctypes.c_int(0)
        assert cuda.cuDeviceGet(ctypes.byref(dev), i) == 0
        name = ctypes.create_string_buffer(256)
        assert cuda.cuDeviceGetName(name, 256, dev) == 0
        u = CUuuid()
        assert cuda.cuDeviceGetUuid(ctypes.byref(u), dev) == 0
        mj = ctypes.c_int(0)
        mn = ctypes.c_int(0)
        assert cuda.cuDeviceComputeCapability(ctypes.byref(mj), ctypes.byref(mn), dev) == 0
        mem = ctypes.c_ulonglong(0)
        assert cuda.cuDeviceTotalMem_v2(ctypes.byref(mem), dev) == 0
        uuid = fmt_uu(u)
        cuda_map[uuid] = i
        print("   ordinal CUDA %d : %-24s cc %d.%d  %6d MiB  %s"
              % (i, name.value.decode(), mj.value, mn.value,
                 mem.value // (1024 * 1024), uuid))

    print("\n=== ENUMERATION NVIDIA-SMI (indexes) ===")
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,uuid,compute_cap,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True)
    smi_map = {}
    for line in r.stdout.strip().splitlines():
        idx, name, uuid, cc, mib = [p.strip() for p in line.split(",")]
        smi_map[uuid] = (idx, name, cc, mib)
        print("   index nvidia-smi %s : %-24s cc %s  %6s MiB  %s"
              % (idx, name, cc, mib, uuid))

    print("\n=== CROISEMENT PAR UUID ===")
    consistent = True
    for uuid, ord_i in sorted(cuda_map.items()):
        if uuid not in smi_map:
            print("   !! CUDA ordinal %d : UUID %s absent de nvidia-smi" % (ord_i, uuid))
            consistent = False
            continue
        smi_idx, name, cc, mib = smi_map[uuid]
        if int(smi_idx) == ord_i:
            print("   CUDA ordinal %d  ==  nvidia-smi index %s  [%s / %s]"
                  % (ord_i, smi_idx, name, uuid))
        else:
            print("   CUDA ordinal %d  !=  nvidia-smi index %s  ** DIVERGENCE **  [%s / %s]"
                  % (ord_i, smi_idx, name, uuid))
            consistent = False

    print("\nVERDICT :", "ordinal CUDA == index nvidia-smi (stable aujourd'hui, MAIS pas garanti au reboot)"
          if consistent else "DIVERGENCE ou flou → sélection par UUID OBLIGATOIRE")
    # carte utile pour l'outillage : UUID du GPU « compute » (V100)
    for uuid in cuda_map:
        if "V100" in smi_map.get(uuid, ("", "", "", ""))[1]:
            print("UUID V100 (cible bench/profilage) :", uuid)


if __name__ == "__main__":
    main()
