#!/usr/bin/env python3
"""
ncu_fa_t3.py — NCU (via sudo, counters perm) sur le kernel FA/T3 (prod .so fa3e75c2)
: MUR MÉMOIRE ou MARGE COMPUTE ?

Question (24/10) : FA = 14,2 % du busy @90k (run9e) = 6,02 ms/tok alors que le roofline
lecture KV (≈2,4 GB/tok de KV q8_0) est ≈2,7 ms au pic HBM2 900 GB/s → T3 tourne à ~45 %
du roofline mémoire. Est-ce un kernel au mur (→ CLOS confirmé, seule issue = KV q4_0) ou
compute-bound (→ il reste un vrai levier kernel dans les 14 %) ?

Méthode :
  - stop_and_free (la V100 doit être libre : clock control NCU + pas de runner prod dessous)
  - NCU 2024.1.1 --set full sur `flash_attn_ext_mma_volta_small` (T3, géométrie prod
    D=256 / 24 qh / 4 kvh / KV q8_0, n_kv=107351) via le harnais `mb_fa` route `t3`
    (dlopen du .so PROD = même dispatch que la prod). iters=1 : par T=1..8, le harnais
    lance 2 warmup + 1 timed → 24 instances au total ; les TIMED sont les blocs 3,6,..,24.
  - restore_service (garanti en finally) + vérification

Lancement : python3 supervisor_bench.py run ncu_fa_t3.py --detach
Sorties : bench/ncu_fa_t3_<ts>.txt (rapport complet) + bench/ncu_fa_t3_extract.md
"""
import os, re, signal, subprocess, sys, time
from gpu_uuid_select import v100_cuda_ordinal, v100_smi_index  # UUID GPU (jamais d'index en dur — voir GPU_UUID_NSYS_CUPTI_20261030.md)

D = os.path.dirname(os.path.abspath(__file__))
ns = {"__file__": os.path.join(D, "night_bench.py"), "__name__": "ncu_fa_t3"}
src = open(os.path.join(D, "night_bench.py")).read().split('if __name__ == "__main__":')[0]
exec(src, ns)
log = ns["log"]
svc_state = ns["svc_state"]
stop_and_free = ns["stop_and_free"]
restore_service = ns["restore_service"]

NCU = "/home/guile/ncu-2024.1.1/usr/lib/nsight-compute/ncu"
MB = os.path.join(D, "mb_fa")
SO_PROD = "/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so"
LDP = "/home/guile/llama-experiments/tools/cuda-12.6/usr/local/cuda-12.6/targets/x86_64-linux/lib"
ITERS = 1      # 2 warmup + 1 timed par T (T=1..8) -> 24 instances
N_KV = 107351
NCAP = 24

def extract(report: str) -> str:
    """Par bloc de profilage : SOL + Compute + Memory + Occupancy (lignes qui tranchent)."""
    blocks = re.split(r"==PROF== Profiling", report)[1:]
    out = []
    want = re.compile(r"(GPU Speed Of Light Throughput|Compute Workload Analysis|"
                      r"Memory Workload Analysis|Occupancy|Warp State Statistics)")
    for i, b in enumerate(blocks):
        T = (i // 3) + 1
        kind = "timed" if (i % 3) == 2 else "warmup"
        mname = re.search(r'void\s+(\S+)', b[:1500])
        sig = mname.group(1).split("(")[0] if mname else "?"
        keep = []
        insec = None
        for ln in b.splitlines():
            m = re.match(r"\s*Section: (.*)$", ln)
            if m:
                insec = m.group(1).strip()
                if want.search(insec):
                    keep.append(f"## {insec}")
                continue
            if insec and want.search(insec) and re.search(r"\d", ln.strip()):
                s = ln.strip()
                if re.search(r"(Memory Throughput|Compute \(SM\)|Duration|DRAM|L1/TEX|L2 Cache|"
                             r"Achieved|Theoretical|Issued Ipc|Executed Ipc|SM Busy|Mem Busy|"
                             r"Stall|Warp Cycles|Issued Warp|Long Scoreboard|Short Scoreboard|"
                             r"Wait\b|No Instruction|LG Throttle|Math Pipe|Pipe|Occupancy|Waves)", s):
                    keep.append(s)
        out.append(f"### bloc {i+1}/24 : T={T} ({kind}) {sig}\n" + "\n".join(keep[:70]))
    return "\n\n".join(out)

def main():
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_txt = os.path.join(D, f"ncu_fa_t3_{ts}.txt")
    out_md = os.path.join(D, "ncu_fa_t3_extract.md")
    md5 = subprocess.run(["md5sum", SO_PROD], capture_output=True, text=True).stdout.split()[0]
    log(f"NCU FA/T3 — prod {md5[:8]} · iters={ITERS} n_kv={N_KV} cap={NCAP} (sudo)")
    if not md5.startswith("fa3e75c2"):
        log("  · !! md5 prod inattendu — arrêt avant de toucher au service")
        sys.exit(4)
    signal.signal(signal.SIGINT, ns["on_sig"])
    signal.signal(signal.SIGTERM, ns["on_sig"])

    if not stop_and_free():
        log("  · V100 pas libérée — ABORT (pas de profilage sous le runner prod)")
        restore_service()
        sys.exit(3)
    log("  · V100 libre, lancement NCU (sudo)")
    envargs = [
        "CUDA_VISIBLE_DEVICES=" + str(v100_cuda_ordinal()),  # index LIB CUDA : 0 = V100 (inversé vs nvidia-smi)
        f"LD_LIBRARY_PATH={LDP}",
        f"MB_FA_SO={SO_PROD}",
    ]
    cmd = ["sudo", "-n", "env"] + envargs + [
        NCU, "--set", "full",
        "-k", "regex:flash_attn_ext_mma_volta_small",
        "-c", str(NCAP),
        "-f",
        MB, "t3", str(ITERS), str(N_KV)]
    log("  · " + " ".join(cmd))
    rc = -1
    t0 = time.time()
    try:
        with open(out_txt, "w") as f:
            p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=dict(os.environ),
                               timeout=1800)
        rc = p.returncode
        log(f"  · NCU rc={rc} en {time.time()-t0:.0f} s → {out_txt}")
    except subprocess.TimeoutExpired:
        log("  · !! timeout NCU (1800 s) — rapport partiel conservé")
    finally:
        restore_service()
    st = svc_state()
    log(f"  · service={st}")
    if st != "active":
        log("  · !! service pas active — Intervention humaine requise")
        sys.exit(1)

    report = open(out_txt).read()
    n_kern = report.count("==PROF== Profiling")
    log(f"  · kernels capturés : {n_kern}")
    with open(out_md, "w") as f:
        f.write(f"# NCU FA/T3 — extract ({ts})\n\n.so prod {md5} · n_kv={N_KV} · T=1..8, "
                f"blocs 3i = timed (i=1..8)\nCommande : `{' '.join(cmd)}`\nrc={rc}\n\n"
                + extract(report) + "\n")
    log(f"  · extract → {out_md}")
    if n_kern < 24:
        log(f"  · !! seulement {n_kern}/24 kernels capturés — vérifier rapport")
    if rc != 0 or n_kern == 0:
        sys.exit(2 if n_kern == 0 else 5)

if __name__ == "__main__":
    main()
