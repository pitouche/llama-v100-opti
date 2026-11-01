#!/usr/bin/env python3
# regime_probe.py — RECONCILIATION 32.6 (A/B frais) vs 63.13 (CUPTI run9b warm).
# Rejoue la config EXACTE de run9b (prompt 186k ~46k tok, MTP ON, KV q8_0, FA auto,
# -b 2048, -c 131000, .so prod) SANS CUPTI, sur un serveur standalone, et envoie
# N fois le meme prompt (64 gen chacune) pour lire la courbe de regime :
#   req1 (froid, 1er decode apres prefill)  ->  req2 (chaud, KV prefill reutilise)  -> reqN.
# Objectif : savoir si le decode prod reel est ~32 t/s (regime A/B) ou ~63 t/s (regime warm),
# et d'ou vient l'ecart 2x (premiere requete vs requete chaude).
# STOPPE le service ollama-v100 et se restaure (finally).
# USAGE : python3 supervisor_bench.py run regime_probe.py [NREQ] --detach
import json, os, socket, subprocess, sys, time, urllib.request
from gpu_uuid_select import v100_cuda_ordinal, v100_smi_index  # UUID GPU (jamais d'index en dur — voir GPU_UUID_NSYS_CUPTI_20261030.md)

D = os.path.dirname(os.path.abspath(__file__))
ROOT = "/home/guile/llama-experiments"
LLAMA = f"{ROOT}/ollama-v100/lib/ollama/llama-server"
GGUF  = "/usr/share/ollama/.ollama/models/blobs/sha256-493301830a596b8ad56dc1329f80bbcb578c8e910da395feafdc9cd8263430bb"
FILLER= f"{ROOT}/v100-repro/work/filler-200k.txt"
SO    = f"{ROOT}/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so"
PORT  = 44913
SERVICE = "ollama-v100"
GPU_SMI = v100_smi_index()  # résolu par UUID (ne pas hardcoder)
CTX_CHARS = 186000      # = run9b
N_GEN   = 64
NREQ    = int(sys.argv[1]) if len(sys.argv) > 1 else 4
SOUT    = f"/tmp/regime_probe_server.out"
OUT     = f"/tmp/regime_probe_{time.strftime('%Y%m%d-%H%M%S')}"
os.makedirs(OUT, exist_ok=True)

logf = open(os.path.join(D, "regime_probe.log"), "w")
def log(m):
    print(time.strftime("[%H:%M:%S] ")+m, flush=True)
    logf.write(time.strftime("[%H:%M:%S] ")+m+"\n"); logf.flush()

def run(c, t=120): return subprocess.run(c, capture_output=True, text=True, timeout=t)

def vram():
    r = run(["nvidia-smi", f"--id={GPU_SMI}", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"])
    v = r.stdout.strip(); return int(v) if v.isdigit() else -1

def http(url, data=None, t=1800):
    req = urllib.request.Request(url, data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=t) as r: return json.loads(r.read().decode())

restored = False
def restore():
    global restored
    log("  · restauration : stop server probe + start service")
    run(["pkill","-f",f"port {PORT}"]); time.sleep(3)
    for _ in range(90):
        if vram() < 1000: break
        run(["pkill","-9","-f","llama-server"]); time.sleep(1)
    run(["sudo","systemctl","start",SERVICE], t=180)
    for _ in range(120):
        try:
            http("http://127.0.0.1:11434/api/version", t=3); restored=True; break
        except Exception: time.sleep(1)
    log(f"  · service={subprocess.run(['systemctl','is-active',SERVICE],capture_output=True,text=True).stdout.strip()} "
        f"V100={vram()}MiB md5={subprocess.run(['md5sum',SO],capture_output=True,text=True).stdout.split()[0][:8]}")

try:
    import hashlib
    assert hashlib.md5(open(SO,'rb').read()).hexdigest()[:8]=="fa3e75c2", "prod .so modifie — abort"
    s = socket.socket(); s.bind(("127.0.0.1", PORT)); s.close()
    filler = open(FILLER, encoding="utf-8", errors="replace").read()
    prompt = filler[:CTX_CHARS]
    log(f"REGIME PROBE — prompt {CTX_CHARS} chars (~46k tok) — {NREQ} req × {N_GEN} gen — MTP ON config prod")
    log(f"  .so prod md5 = fa3e75c2 OK")

    log("  · stop service")
    run(["sudo","systemctl","stop",SERVICE], t=180)
    t0=time.time()
    while time.time()-t0<180:
        if vram()<1024: break
        time.sleep(3)
    log(f"  · V100 liberee ({vram()} MiB) en {time.time()-t0:.0f}s")

    env = dict(os.environ)
    env.update({
        "LD_LIBRARY_PATH": f"{ROOT}/ollama-v100/lib/ollama:{ROOT}/ollama-v100/lib/ollama/cuda_v12:"
                           f"{ROOT}/tools/cuda-12.6/usr/local/cuda-12.6/lib64:"
                           f"{ROOT}/tools/cuda-12.6/usr/local/cuda-12.6/targets/x86_64-linux/lib",
        "GGML_BACKEND_PATH": SO,
        "GGML_CUDA_VOLTA_Q6_W4R4": "1",
        "GGML_CUDA_VOLTA_Q8_FATTN_TC": "0",
        "GGML_CUDA_QWEN35_MTP_SHORTLIST": f"{ROOT}/src/llama-volta-merged/data/mtp-shortlists/qwen38-27b-exact-131072.i32",
        "LLAMA_ARG_SPEC_MTP_DEFER_PROMPT": "1",
        "LLAMA_ARG_FIT_TARGET": "1024",
        "CUDA_VISIBLE_DEVICES": str(v100_cuda_ordinal()),  # UUID, pas en dur
        # PAS de LD_PRELOAD (pas de CUPTI — mesure wall pure)
    })
    lf = open(SOUT, "w")
    args = [LLAMA, "--model", GGUF, "--port", str(PORT), "--host","127.0.0.1","--no-webui","-c","131000",
            "-np","1","-lv","4","--no-log-prefix","--no-log-timestamps",
            "--spec-type","draft-mtp","--spec-draft-n-max","3","--spec-draft-backend-sampling",
            "--cache-type-k","q8_0","--cache-type-v","q8_0","--flash-attn","auto","-b","2048","-ub","2048"]
    proc = subprocess.Popen(args, stdout=lf, stderr=subprocess.STDOUT, env=env)
    ok=0
    for _ in range(300):
        try: http(f"http://127.0.0.1:{PORT}/v1/models", t=3); ok=1; break
        except Exception: time.sleep(1)
    if not ok:
        log("server pas up"); sys.exit(2)
    MID = http(f"http://127.0.0.1:{PORT}/v1/models", t=10)["data"][0]["id"]
    log(f"  · server probe up (port {PORT}) — VRAM {vram()} MiB — model {MID[:24]}")

    # gate shortlist
    for _ in range(60):
        if "CUDA MTP shortlist" in open(SOUT, errors="replace").read(): break
        time.sleep(1)
    if "CUDA MTP shortlist" not in open(SOUT, errors="replace").read():
        log("WARN shortlist non charge"); 

    # warmup 16 tok (= run9b)
    http(f"http://127.0.0.1:{PORT}/v1/chat/completions",
         {"model":MID,"stream":False,"temperature":0,"max_tokens":16,
          "chat_template_kwargs":{"enable_thinking":False},
          "messages":[{"role":"user","content":"Say OK."}]}, t=1800)
    log("  · warmup 16 tok done")

    results = []
    for i in range(1, NREQ+1):
        data = {"model":MID,"stream":False,"temperature":0,"top_k":1,"ignore_eos":True,
                "max_tokens":N_GEN,"chat_template_kwargs":{"enable_thinking":False},
                "messages":[{"role":"user","content":prompt}]}
        t0=time.time()
        try:
            r = http(f"http://127.0.0.1:{PORT}/v1/chat/completions", data, t=1800)
            wall = time.time()-t0
            u = r.get("usage",{})
            gen = u.get("completion_tokens"); p = u.get("prompt_tokens")
            # eval time depuis le log (dernier "eval time")
            txt = open(SOUT, errors="replace").read()
            import re
            ev = re.findall(r"(?<!prompt )eval time = +([\d.]+) ms / +(\d+) tokens", txt)
            last = ev[-1] if ev else (None,None)
            etps = None; ems=None
            if last and last[1]:
                ems=float(last[0]); etps=round(int(last[1])/(ems/1000.0),2)
            acc = re.findall(r"draft acceptance = ([\d.]+) \(\s*(\d+) accepted / +(\d+) generated\), mean len = +([\d.]+)", txt)
            a = acc[-1] if acc else (None,None,None,None)
            row = {"req":i,"wall_s":round(wall,1),"prompt_tokens":p,"gen_tokens":gen,
                   "eval_tps":etps,"eval_ms":ems,"acc":a[0],"mean_len":a[3],
                   "accept":a[1],"gen_drafts":a[2]}
            results.append(row)
            log(f"  req{i}: wall {wall:.1f}s  prompt {p}  gen {gen}  eval {etps} t/s ({ems} ms)  "
                f"acc {a[0]} mean_len {a[3]}  ({a[1]}/{a[2]})")
        except Exception as e:
            log(f"  req{i} ECHEC: {e}"); results.append({"req":i,"error":str(e)})
    with open(os.path.join(OUT,"results.json"), "w") as f: json.dump(results,f,indent=2)
    logf.write("\n"+json.dumps(results,indent=2)+"\n")
    log(f"FIN — results : {OUT}/results.json")
    sys.exit(0)
except Exception as e:
    import traceback; logf.write(traceback.format_exc()); logf.flush()
    log(f"ERREUR: {e}"); 
finally:
    restore()
    logf.close()
