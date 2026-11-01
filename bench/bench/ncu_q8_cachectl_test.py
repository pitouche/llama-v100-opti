#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ncu_q8_cachectl_test.py — Test decisif : anomalie Q8_0 (DRAM read = 11.9 MB / 94.7 MB de poids).

Hypothese : le working set Q8 (~52 MB de trafic poids/passe) tient dans le L2 61 MB de la V100
-> en replay NCU sans flush L2, les passes se suivent chaudes et le DRAM s'effondre.
Q6 (~73 MB/passe) et Q5 depassent le L2 -> pas d'effet.

Test : meme lance, meme shape, --cache-control {none, all} :
  - si DRAM(q8, all) >> DRAM(q8, none) : L2 chaud entre passes (pas de flush par defaut).
  - q6 en temoin : DRAM stable ~73 MB dans les deux cas.

USAGE :
  cd /home/guile/llama-experiments/bench
  python3 supervisor_bench.py run ncu_q8_cachectl_test.py --detach
"""
import json
import os
import re
import sys
import time
from gpu_uuid_select import v100_cuda_ordinal, v100_smi_index  # UUID GPU (jamais d'index en dur — voir GPU_UUID_NSYS_CUPTI_20261030.md)

D = os.path.dirname(os.path.abspath(__file__))
ns = {"__file__": os.path.join(D, "night_bench.py"), "__name__": "ncu_cachectl_harness"}
src = open(os.path.join(D, "night_bench.py")).read().split('if __name__ == "__main__":')[0]
exec(src, ns)

NCU = "/home/guile/ncu-2024.1.1/usr/lib/nsight-compute/ncu"
MB = os.path.join(D, "mb_mmvq")
SO = ns["SO_PROD"]
OUT = "/tmp/ncu_cachectl"

log = ns["log"]

METRICS = ("dram__bytes_read.sum,gpu__time_duration.sum,"
           "lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum,"
           "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,"
           "lts__t_sectors_op_read.sum")


def run_ncu(shape, cache_ctl):
    raw = os.path.join(OUT, "cc_%s_%s.txt" % (shape, cache_ctl))
    cmd = ["sudo", "env", "LC_ALL=C", "CUDA_VISIBLE_DEVICES=" + str(v100_cuda_ordinal()), "MB_MMVQ_SO=" + SO, NCU,
           "--launch-skip", "5", "--launch-count", "3",
           "--cache-control", cache_ctl,
           "--kernel-name", "regex:mul_mat_vec_q",
           "--metrics", METRICS,
           MB, "60", shape, "4"]
    r = ns["run"](cmd, timeout=900)
    open(raw, "w").write("CMD : " + " ".join(cmd) + "\nRC=%d\n" % r.returncode + r.stdout + r.stderr)
    return raw


def metric_list(raw, name):
    pat = re.escape(name) + r"\s+[A-Za-z/]+\s+([\d.]+)"
    return [float(x) for x in re.findall(pat, open(raw).read())]


def first(lst):
    return lst[0] if lst else None


def main():
    import signal
    signal.signal(signal.SIGINT, ns["on_sig"])
    signal.signal(signal.SIGTERM, ns["on_sig"])

    os.makedirs(OUT, exist_ok=True)
    log("Test cache-control NCU (anomalie Q8) — stop service")
    ns["stop_and_free"]()
    try:
        rows = []
        for shape in ["q8_f17408x5120", "q6_f17408x5120"]:
            for cc in ["none", "all"]:
                log("  · %s cache-control=%s" % (shape, cc))
                raw = run_ncu(shape, cc)
                d = {
                    "shape": shape, "cc": cc,
                    "dram_MB": first(metric_list(raw, "dram__bytes_read.sum")),
                    "dur_us": first(metric_list(raw, "gpu__time_duration.sum")),
                    "l2miss_MB": first(metric_list(raw, "lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum")),
                    "l1_req_MB": first(metric_list(raw, "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum")),
                    "l2_req_MB": first(metric_list(raw, "lts__t_sectors_op_read.sum")),
                    "n": len(metric_list(raw, "dram__bytes_read.sum")),
                }
                rows.append(d)
                log("    DRAM=%s MB  dur=%s us  L2miss=%s MB  L1req=%s MB  L2req=%s MB"
                    % (d["dram_MB"], d["dur_us"], d["l2miss_MB"], d["l1_req_MB"], d["l2_req_MB"]))
        with open(os.path.join(OUT, "cachectl.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    finally:
        ns["restore_service"]()
    log("FIN test cache-control — %s" % ("service OK" if ns["_restored"] else "SERVICE KO"))
    sys.exit(0 if ns["_restored"] else 3)


if __name__ == "__main__":
    main()
