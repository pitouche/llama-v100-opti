#!/usr/bin/env python3
"""Probe : comment la lib cuda voit les devices (nom + index) selon CVD."""
import ctypes, os
lib = ctypes.CDLL("libcuda.so.1")
err = lib.cuInit(0)
print(f"cuInit -> {err} (CVD={os.environ.get('CUDA_VISIBLE_DEVICES', '<absent>')}, user={os.environ.get('USER','?')})")
if err:
    raise SystemExit(1)
count = ctypes.c_int(0)
lib.cuDeviceGetCount(ctypes.byref(count))
print(f"nb devices visibles: {count.value}")
for i in range(count.value):
    dev = ctypes.c_int(i)
    name = ctypes.create_string_buffer(128)
    lib.cuDeviceGetName(name, 128, dev)
    total = ctypes.c_ulonglong(0)
    lib.cuDeviceTotalMem_v2(ctypes.byref(total), dev)
    print(f"  device {i}: {name.value.decode()} — {total.value//1024//1024} MiB")
