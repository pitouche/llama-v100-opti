#!/usr/bin/env python3
"""
Bench FA microbench CLEAN — service stoppé, GPU idle, via dispatch prod.

Mesure le .so patché (dual) à travers le dispatch réel `ggml_cuda_flash_attn_ext`
(environnement FA16), sur la géométrie prod Qwen3.8-27B : D=256, qh=24, kvh=4.

Cas mesurés (mb_fa_pf, f16 KV, mask=0 = full attention) :
  1. f16_prod     : .so PROD, pas d'env FA             → 32x2, baseline absolue (~194 ms T=2048)
  2. baseline     : dual, pas d'env FA                 → 32x2 (doit = prod)
  3. f16_16x2     : dual, FA_NCOLS1=16                 → 16x2 (~163 ms T=2048 attendu)
  4. compact_t2048: dual, FA_COMPACT_TMAX=8192         → kernel compact 48 KB si gate OK (T=2048)
  5. baseline_t1024: dual, pas d'env FA, T=1024        → 32x2 (compact OFF par défaut)
  6. compact_t1024: dual, FA_COMPACT_TMAX=2048, T=1024 → kernel compact 48 KB
  7. f16_16x2_t1024: dual, FA_NCOLS1=16, T=1024        → 16x2
NB : en prod (KV q8_0), le kernel FA opère sur buffers f16 (dequant séparé)
     → la mesure f16 est fidèle au coût du kernel FA ; le q8 complet se mesure e2e (fa7).

Le superviseur (supervisor_bench.py) tient le lock ; on stoppe/restaure le service
ici, avec trap signal. Ne PAS lancer sans superviseur.

Sortie : fa16_clean_<ts>.jsonl + fa16_clean_<ts>_summary.md
"""
import json, os, signal, socket, subprocess, sys, time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpu_uuid_select import v100_cuda_ordinal, v100_smi_index

D = os.path.dirname(os.path.abspath(__file__))
CVD = str(v100_cuda_ordinal())          # V100 = UUID résolu (jamais d'index en dur)
SMI = str(v100_smi_index())

SERVICE = "ollama-v100"
SO_DUAL = "/home/guile/llama-experiments/v100/dual-arch/lib/libggml-cuda-dual.so"
SO_PROD = "/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so"
MB      = os.path.join(D, "mb_fa_pf")
TS = datetime.now().strftime("%Y%m%d-%H%M%S")
OUT_JSONL   = os.path.join(D, f"fa16_clean_{TS}.jsonl")
OUT_SUMMARY = os.path.join(D, f"fa16_clean_{TS}_summary.md")
OUT_LOG     = os.path.join(D, f"fa16_clean_{TS}.log")

CUDA_LIBS = [
    "/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12",
    "/home/guile/llama-experiments/ollama-v100/lib/ollama",
    "/usr/local/cuda-12.9/lib64",
]

# NB : en prod KV q8_0, le kernel FA opère sur des buffers f16 (dequant séparé,
# need_f16_K/V=true dans get_alloc_size pour BEST_FATTN_KERNEL_MMA_F16) → mesurer
# le FA en f16 ici est fidèle au coût du kernel FA prod. Pas de mode q8 dans mb_fa_pf
# (pas de pipeline dequant allocé) — le vrai q8 se mesure e2e (fa7).
#
# (nom, so, env FA, T, n_kv, iters, description)
CASES = [
    ("f16_prod",  SO_PROD, {},                        2048, 114688, 3, "baseline absolue : .so prod, f16 KV, 32x2"),
    ("baseline",  SO_DUAL, {},                        2048, 114688, 5, "dual sans env → 32x2 (comportement identique attendu)"),
    ("f16_16x2",  SO_DUAL, {"GGML_CUDA_VOLTA_FA_NCOLS1": "16"}, 2048, 114688, 5, "dual 16x2 forcé, f16 KV"),
    ("compact_t2048", SO_DUAL, {"GGML_CUDA_VOLTA_FA_COMPACT_TMAX": "8192"}, 2048, 114688, 5, "compact 48 KB (32x2) autorisé à T=2048"),
    ("baseline_t1024", SO_DUAL, {},                   1024, 114688, 5, "dual sans env, T=1024 (compact OFF par défaut : 1024<1024 faux)"),
    ("compact_t1024", SO_DUAL, {"GGML_CUDA_VOLTA_FA_COMPACT_TMAX": "2048"}, 1024, 114688, 5, "compact (32x2) à T=1024 (TMAX>1024)"),
    ("f16_16x2_t1024", SO_DUAL, {"GGML_CUDA_VOLTA_FA_NCOLS1": "16"}, 1024, 114688, 5, "16x2 forcé, T=1024"),
]

_restored = False

def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(OUT_LOG, "a") as f:
        f.write(line + "\n")

def svc_state():
    r = subprocess.run(["systemctl", "is-active", SERVICE], capture_output=True, text=True)
    return r.stdout.strip()

def vram_mib():
    r = subprocess.run(["nvidia-smi", f"--id={SMI}", "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True)
    v = r.stdout.strip()
    return int(v) if v.isdigit() else -1

def v100_idle():
    """V100 sans process GPU actif (util<5% et pas de compute app >8000 MiB)."""
    try:
        r = subprocess.run(["nvidia-smi", f"--id={SMI}",
                            "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True)
        if r.stdout.strip().isdigit() and int(r.stdout.strip()) >= 5:
            return False
        r = subprocess.run(["nvidia-smi", f"--id={SMI}", "--query-compute-apps=pid,used_memory",
                            "--format=csv,noheader"], capture_output=True, text=True)
        for line in r.stdout.splitlines():
            parts = line.split(",")
            if len(parts) >= 2 and parts[1].strip().split()[0].isdigit():
                if int(parts[1].strip().split()[0]) > 8000:
                    return False
    except Exception:
        return False
    return True

def restore_service():
    global _restored
    try:
        if svc_state() != "active":
            log(f"  · restauration : systemctl start {SERVICE}")
            subprocess.run(["sudo", "systemctl", "start", SERVICE], timeout=120, check=False)
        for _ in range(90):
            if svc_state() == "active":
                break
            time.sleep(2)
        _restored = svc_state() == "active"
        log(f"  · service {SERVICE} = {svc_state()}")
    except Exception as e:
        log(f"  · ERREUR restauration : {e}")
        _restored = False

def on_sig(sig, frm):
    restore_service()
    sys.exit(128 + sig)

def stop_and_free():
    log(f"  · stop {SERVICE}")
    subprocess.run(["sudo", "systemctl", "stop", SERVICE], timeout=120, check=False)
    t0 = time.time()
    while time.time() - t0 < 180:
        if vram_mib() < 1024 and v100_idle():
            log(f"  · V100 libérée + idle ({vram_mib()} MiB) en {time.time()-t0:.0f}s")
            return True
        time.sleep(3)
    log(f"  · ATTENTION V100 pas totalement libre (VRAM={vram_mib()} MiB) — on continue si idle")
    return v100_idle()

def wait_idle(timeout=45):
    """Attend que la V100 devienne idle (util<5%, pas de process >8000 MiB).
    nvidia-smi sample l'utilisation sur plusieurs secondes : juste après un
    bench, la valeur reste haute quelques instants — il faut patienter."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if v100_idle():
            return True
        time.sleep(2)
    return v100_idle()

def port_free(port):
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", port)); s.close(); return True
    except OSError:
        s.close(); return False

def run_case(name, so, extra, T, n_kv, iters):
    if not port_free(8917):
        return {"case": name, "error": "port 8917 occupé"}
    if not wait_idle():
        return {"case": name, "error": f"V100 pas idle avant {name} (VRAM={vram_mib()} MiB)"}
    env = dict(os.environ)
    for ddir in CUDA_LIBS:
        if os.path.isdir(ddir) and ddir not in env.get("LD_LIBRARY_PATH", ""):
            env["LD_LIBRARY_PATH"] = ddir + ":" + env.get("LD_LIBRARY_PATH", "")
    env["MB_FA_SO"] = so
    env["CUDA_VISIBLE_DEVICES"] = CVD
    env.update(extra)
    t0 = time.time()
    r = subprocess.run([MB, str(T), str(n_kv), str(iters)],
                       capture_output=True, text=True, env=env, timeout=600)
    dt = time.time() - t0
    row = {"case": name, "T": T, "n_kv": n_kv, "iters": iters, "kv": "f16",
           "env": extra, "wall_s": round(dt, 1), "rc": r.returncode}
    last = [l for l in r.stdout.strip().splitlines() if l.startswith("{")]
    if last:
        try:
            row.update(json.loads(last[-1]))
        except Exception as e:
            row["json_err"] = str(e)
    if r.returncode != 0 or not last:
        row["stderr"] = r.stderr[-400:]
    log(f"  · {name}: rc={r.returncode} us_med={row.get('us_med','?')} finite={row.get('finite','?')} ({dt:.0f}s)")
    return row

NCU = "/home/guile/ncu-2024.1.1/usr/lib/nsight-compute/ncu"
NCU_CASES = [
    ("ncu_baseline",        {},                                          2048),
    ("ncu_16x2",            {"GGML_CUDA_VOLTA_FA_NCOLS1": "16"},         2048),
    ("ncu_compact_t2048",   {"GGML_CUDA_VOLTA_FA_COMPACT_TMAX": "8192"}, 2048),
    ("ncu_compact_t1024",   {"GGML_CUDA_VOLTA_FA_COMPACT_TMAX": "2048"}, 1024),
]

def run_ncu(name, extra, T):
    """NCU minimal (nom kernel + launch) pour confirmer le dispatch réel via le .so patché."""
    out = os.path.join(D, f"fa16_ncu_{name}.txt")
    envkv = " ".join(f"{k}={v}" for k, v in extra.items())
    ldp = ":".join(ddir for ddir in CUDA_LIBS if os.path.isdir(ddir))
    if not wait_idle():
        log(f"  · NCU {name}: V100 pas idle, abandon du cas NCU")
        return {"case": name, "T": T, "env": extra, "ncu_rc": -1, "error": "gpu busy"}
    cmd = (f"sudo -n env CUDA_VISIBLE_DEVICES={CVD} MB_FA_SO={SO_DUAL} LD_LIBRARY_PATH='{ldp}'"
           + (f" {envkv}" if envkv else "") +
           f" {NCU} --launch-skip 2 --launch-count 1 --kernel-name 'regex:flash_attn' "
           f"--metrics gpu__time_duration.sum,launch__grid_size,launch__block_size,"
           f"launch__registers_per_thread,launch__shared_mem_per_block_dynamic,"
           f"launch__occupancy_limit_shared_mem,launch__occupancy_limit_registers "
           f"./mb_fa_pf {T} 114688 3 > {out} 2>&1")
    r = subprocess.run(cmd, shell=True, timeout=300)
    if r.returncode != 0:
        try:
            info["ncu_err"] = open(out, errors="replace").read()[-400:]
        except Exception:
            pass
    info = {"case": name, "T": T, "env": extra, "ncu_rc": r.returncode}
    try:
        txt = open(out, errors="replace").read()
        import re as _re
        for line in txt.splitlines():
            if "flash_attn_ext_f16" in line and "void" in line:
                info["kernel"] = line.strip()[:140]
                m = _re.search(r"\((\d+), (\d+), (\d+)\)x\((\d+), (\d+), (\d+)\)", line)
                info["launch"] = (f"({m.group(1)},{m.group(2)},{m.group(3)})x"
                                  f"({m.group(4)},{m.group(5)},{m.group(6)})") if m else "?"
                break
        for pat, key in [(r"gpu__time_duration\.sum\s+[\w/]*\s+([\d.,]+)", "us_ncu"),
                         (r"launch__grid_size\s+(\d+)", "grid"),
                         (r"launch__block_size\s+(\d+)", "block"),
                         (r"launch__registers_per_thread\s+[\w/]*\s+([\d.]+)", "regs"),
                         (r"launch__shared_mem_per_block_dynamic\s+[\w/]*\s+([\d.,]+)", "smem_kb"),
                         (r"launch__occupancy_limit_shared_mem\s+[\w/]*\s+([\d.]+)", "lim_smem"),
                         (r"launch__occupancy_limit_registers\s+[\w/]*\s+([\d.]+)", "lim_regs")]:
            m = _re.search(pat, txt)
            if m: info[key] = m.group(1)
    except Exception as e:
        info["parse_err"] = str(e)
    log(f"  · NCU {name}: kernel={info.get('kernel','?')[:84]} launch={info.get('launch','?')} us={info.get('us_ncu','?')} regs={info.get('regs','?')} smem={info.get('smem_kb','?')}KB")
    return info

def main():
    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)
    open(OUT_LOG, "a").close()
    log(f"FA16 CLEAN — service stoppé, GPU idle — {TS}")
    import hashlib
    for tag, p in (("dual", SO_DUAL), ("prod", SO_PROD)):
        h = hashlib.md5(open(p, "rb").read()).hexdigest()[:8]
        log(f"  md5 {tag}: {h}")
    if not port_free(8917):
        sys.exit("port 8917 occupé")
    if not stop_and_free():
        sys.exit("V100 pas libre — abandon")
    rows = []
    ncu_rows = []
    try:
        for name, so, extra, T, nkv, iters, desc in CASES:
            row = run_case(name, so, extra, T, nkv, iters)
            rows.append(row)
            with open(OUT_JSONL, "a") as f:
                f.write(json.dumps(row) + "\n")
        log("  --- phase NCU (confirmation dispatch) ---")
        for name, extra, T in NCU_CASES:
            ncu_rows.append(run_ncu(name, extra, T))
            with open(OUT_JSONL, "a") as f:
                f.write(json.dumps(ncu_rows[-1]) + "\n")
    finally:
        restore_service()

    # ---- summary ----
    def med(r): return r.get("us_med")
    lines = [f"# FA16 clean — bench microbench service stoppé — {TS}", "",
             "Géométrie : D=256, qh=24, kvh=4 (GQA6), n_kv=114688 (pad 256), mask=0 (full attn), -b 2048 prod.",
             f"GPU : V100 (SMI {SMI} / CUDA {CVD}), service stoppé, idle vérifié par cas.", ""]
    lines.append("| cas | KV | T | env | us_med | us_min | us_max | finite |")
    lines.append("|---|---|---:|---|---:|---:|---:|---|")
    for r in rows:
        lines.append(f"| {r.get('case')} | {r.get('kv','?')} | {r.get('T','?')} | {r.get('env') or '{}'} | "
                     f"{r.get('us_med','?')} | {r.get('us_min','?')} | {r.get('us_max','?')} | {r.get('finite','?')} |")
    # deltas
    def find(n):
        for r in rows:
            if r.get("case") == n and med(r):
                return med(r)
        return None
    def delta(a, b):
        va, vb = find(a), find(b)
        if va and vb:
            return (vb - va) / va * 100.0
        return None
    d1 = delta("baseline", "f16_16x2")
    d2 = delta("baseline", "compact_t2048")
    d3 = delta("baseline_t1024", "compact_t1024")
    d4 = delta("baseline_t1024", "f16_16x2_t1024")
    lines += ["", "## Deltas (vs baseline sans env, même T)"]
    if d1 is not None: lines.append(f"- T=2048 : 16x2 vs 32x2 = **{d1:+.1f} %**")
    if d2 is not None: lines.append(f"- T=2048 : compact(32x2) vs 32x2 = **{d2:+.1f} %**")
    if d3 is not None: lines.append(f"- T=1024 : compact(32x2) vs 32x2 = **{d3:+.1f} %**")
    if d4 is not None: lines.append(f"- T=1024 : 16x2 vs 32x2 = **{d4:+.1f} %**")
    lines += ["", "## Service restauré : {'OK' if _restored else 'ECHEC'}", ""]
    lines += ["", "## NCU (confirmation dispatch .so patché)", "",
              "| cas | T | env | kernel | launch | us_ncu | grid | regs | smem_KB | lim_smem | lim_regs |",
              "|---|---:|---|---|---|---:|---|---:|---:|---:|---:|"]
    for r in ncu_rows:
        lines.append(f"| {r.get('case')} | {r.get('T')} | {r.get('env') or '{}'} | {r.get('kernel','?')[:64]} | {r.get('launch','?')} | "
                     f"{r.get('us_ncu','?')} | {r.get('grid','?')} | {r.get('regs','?')} | {r.get('smem_kb','?')} | {r.get('lim_smem','?')} | {r.get('lim_regs','?')} |")
    with open(OUT_SUMMARY, "w") as f:
        f.write("\n".join(lines))
    log(f"SUMMARY : {OUT_SUMMARY}")
    log(f"service restauré = {_restored}")
    sys.exit(0 if _restored else 2)

if __name__ == "__main__":
    main()
