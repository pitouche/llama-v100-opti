#!/usr/bin/env python3
"""
FA16 NCU phase — confirmation du dispatch réel via le .so patché (d4954502).
Service stoppé, GPU idle. Profile EXACTEMENT le kernel FA :
  --kernel-name regex:flash_attn_ext_f16   (exclut flash_attn_mask_to_KV_max)
  --launch-skip 3                          (2 warmups + 1er iter réel)
Lancer via superviseur :  python3 supervisor_bench.py run bench_fa16_ncu.py --detach
"""
import json, os, re, signal, socket, subprocess, sys, time
from datetime import datetime

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
from gpu_uuid_select import v100_cuda_ordinal, v100_smi_index

CVD, SMI = str(v100_cuda_ordinal()), str(v100_smi_index())
SERVICE = "ollama-v100"
SO_DUAL = "/home/guile/llama-experiments/v100/dual-arch/lib/libggml-cuda-dual.so"
MB      = os.path.join(D, "mb_fa_pf")
NCU     = "/home/guile/ncu-2024.1.1/usr/lib/nsight-compute/ncu"
TS      = datetime.now().strftime("%Y%m%d-%H%M%S")
OUT_JSONL = os.path.join(D, f"fa16_ncu_{TS}.jsonl")
OUT_LOG   = os.path.join(D, f"fa16_ncu_{TS}.log")
CUDA_LIBS = ["/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12",
             "/home/guile/llama-experiments/ollama-v100/lib/ollama",
             "/usr/local/cuda-12.9/lib64"]

# (nom, env FA, T) — 2 iters seulement (NCU re-simule le kernel profilé)
CASES = [
    ("ncu_baseline",        {},                                             2048),
    ("ncu_16x2",            {"GGML_CUDA_VOLTA_FA_NCOLS1": "16"},            2048),
    ("ncu_compact_t2048",   {"GGML_CUDA_VOLTA_FA_COMPACT_TMAX": "8192"},    2048),
    ("ncu_compact_t1024",   {"GGML_CUDA_VOLTA_FA_COMPACT_TMAX": "2048"},    1024),
    ("ncu_16x2_t1024",      {"GGML_CUDA_VOLTA_FA_NCOLS1": "16"},            1024),
]

_restored = False

def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(OUT_LOG, "a") as f: f.write(line + "\n")

def svc_state():
    return subprocess.run(["systemctl", "is-active", SERVICE],
                          capture_output=True, text=True).stdout.strip()

def vram_mib():
    r = subprocess.run(["nvidia-smi", f"--id={SMI}", "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True)
    v = r.stdout.strip()
    return int(v) if v.isdigit() else -1

def v100_idle():
    try:
        r = subprocess.run(["nvidia-smi", f"--id={SMI}", "--query-gpu=utilization.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True)
        if r.stdout.strip().isdigit() and int(r.stdout.strip()) >= 5:
            return False
        r = subprocess.run(["nvidia-smi", f"--id={SMI}", "--query-compute-apps=pid,used_memory",
                            "--format=csv,noheader"], capture_output=True, text=True)
        for line in r.stdout.splitlines():
            p = line.split(",")
            if len(p) >= 2 and p[1].strip().split()[0].isdigit():
                if int(p[1].strip().split()[0]) > 8000:
                    return False
    except Exception:
        return False
    return True

def wait_idle(timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if v100_idle(): return True
        time.sleep(2)
    return v100_idle()

def restore_service():
    global _restored
    try:
        if svc_state() != "active":
            log("  · restauration : systemctl start " + SERVICE)
            subprocess.run(["sudo", "systemctl", "start", SERVICE], timeout=120, check=False)
        for _ in range(90):
            if svc_state() == "active": break
            time.sleep(2)
        _restored = svc_state() == "active"
        log(f"  · service {SERVICE} = {svc_state()}")
    except Exception as e:
        log(f"  · ERREUR restauration : {e}")
        _restored = False

def on_sig(s, f):
    restore_service(); sys.exit(128 + s)

def run_ncu(name, extra, T):
    out = os.path.join(D, f"fa16_ncu_{name}_{TS}.txt")
    envkv = " ".join(f"{k}={v}" for k, v in extra.items())
    ldp = ":".join(d for d in CUDA_LIBS if os.path.isdir(d))
    if not wait_idle():
        log(f"  · NCU {name}: V100 pas idle, abandon"); return {"case": name, "error": "gpu busy"}
    cmd = (f"sudo -n env CUDA_VISIBLE_DEVICES={CVD} MB_FA_SO={SO_DUAL} LD_LIBRARY_PATH='{ldp}'"
           + (f" {envkv}" if envkv else "") +
           f" {NCU} --launch-skip 3 --launch-count 1 --kernel-name 'regex:flash_attn_ext_f16' "
           f"--metrics gpu__time_duration.sum,launch__grid_size,launch__block_size,"
           f"launch__registers_per_thread,launch__shared_mem_per_block_dynamic,"
           f"launch__occupancy_limit_shared_mem,launch__occupancy_limit_registers,"
           f"launch__waves_per_multiprocessor,sm__inst_executed_pipe_tensor_op_hmma.sum "
           f"./mb_fa_pf {T} 114688 5 > {out} 2>&1")
    r = subprocess.run(cmd, shell=True, timeout=600)
    info = {"case": name, "T": T, "env": extra, "ncu_rc": r.returncode}
    try:
        txt = open(out, errors="replace").read()
        for line in txt.splitlines():
            if "void flash_attn_ext_f16" in line:
                info["kernel"] = line.strip()[:150]
                m = re.search(r"\((\d+), (\d+), (\d+)\)x\((\d+), (\d+), (\d+)\)", line)
                info["launch"] = (f"({m.group(1)},{m.group(2)},{m.group(3)})x"
                                  f"({m.group(4)},{m.group(5)},{m.group(6)})") if m else "?"
                break
        for pat, key in [(r"gpu__time_duration\.sum\s+[\w/]*\s+([\d.,]+)", "us_ncu"),
                         (r"launch__grid_size\s+(\d+)", "grid"),
                         (r"launch__block_size\s+(\d+)", "block"),
                         (r"launch__registers_per_thread\s+[\w/]*\s+([\d.]+)", "regs"),
                         (r"launch__shared_mem_per_block_dynamic\s+[\w/]*\s+([\d.,]+)", "smem_kb"),
                         (r"launch__occupancy_limit_shared_mem\s+[\w/]*\s+([\d.]+)", "lim_smem"),
                         (r"launch__occupancy_limit_registers\s+[\w/]*\s+([\d.]+)", "lim_regs"),
                         (r"launch__waves_per_multiprocessor\s+[\d.,]*\s+([\d.,]+)", "waves"),
                         (r"sm__inst_executed_pipe_tensor_op_hmma\.sum\s+[\w/]*\s+([\d.,]+)", "hmma")]:
            m = re.search(pat, txt)
            if m: info[key] = m.group(1)
        if "No kernels were profiled" in txt:
            info["warn"] = "no kernel profiled (filtre trop strict?)"
    except Exception as e:
        info["parse_err"] = str(e)
    log(f"  · NCU {name}: rc={r.returncode} kernel={info.get('kernel','?')[:72]} "
        f"launch={info.get('launch','?')} us={info.get('us_ncu','?')} regs={info.get('regs','?')} "
        f"smem={info.get('smem_kb','?')}KB lim_smem={info.get('lim_smem','?')}")
    return info

def main():
    signal.signal(signal.SIGINT, on_sig); signal.signal(signal.SIGTERM, on_sig)
    open(OUT_LOG, "a").close()
    log(f"FA16 NCU phase — {TS} — service stoppé")
    if not socket.socket() or True:
        s = socket.socket()
        try: s.bind(("127.0.0.1", 8917)); s.close()
        except OSError:
            s.close(); sys.exit("port 8917 occupé")
    log(f"  · stop {SERVICE}")
    subprocess.run(["sudo", "systemctl", "stop", SERVICE], timeout=120, check=False)
    t0 = time.time()
    while time.time() - t0 < 180:
        if vram_mib() < 1024 and v100_idle():
            log(f"  · V100 libérée + idle en {time.time()-t0:.0f}s"); break
        time.sleep(3)
    rows = []
    try:
        for name, extra, T in CASES:
            row = run_ncu(name, extra, T)
            rows.append(row)
            with open(OUT_JSONL, "a") as f: f.write(json.dumps(row) + "\n")
    finally:
        restore_service()
    log(f"FIN NCU phase — service restauré = {_restored}")
    sys.exit(0 if _restored else 2)

if __name__ == "__main__":
    main()
