#!/usr/bin/env python3
"""Échantillonneur nvidia-smi V100 pendant la nuit de bench.

Usage : gputimes_sampler.py <out.tsv> [periode_s]   (défaut 1.0)
"""
import subprocess, time, sys
from gpu_uuid_select import v100_cuda_ordinal, v100_smi_index  # UUID GPU (jamais d'index en dur — voir GPU_UUID_NSYS_CUPTI_20261030.md)
out = sys.argv[1]
PERIOD = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
FIELDS = "utilization.gpu,utilization.memory,memory.used,power.draw"
CMD = ["nvidia-smi", f"--query-gpu={FIELDS}", "--format=csv,noheader,nounits", "--id=" + v100_smi_index()]
with open(out, "w") as f:
    f.write("epoch_ms util_gpu util_mem mem_MiB power_W\n")
    while True:
        try:
            p = subprocess.run(CMD, capture_output=True, text=True, timeout=5)
            parts = [x.strip() for x in p.stdout.strip().split(",")]
            if len(parts) == 4:
                f.write(f"{int(time.time()*1000)} {' '.join(parts)}\n")
                f.flush()
        except Exception:
            pass
        time.sleep(max(0.05, PERIOD - 0.05))
