#!/usr/bin/env python3
"""
KV-CACHE-EN-RAM (multi-utilisateurs) - etude du 01/11

Objectif : mesurer le cout reel d'un prefill FROID (le mal) vs le cout quand
le contexte a ete stationne en RAM cache par le serveur (le bien), dans le
scenario exact "2 utilisateurs qui alternent leurs questions" sur un seul
runner V100 (1 slot, comme la prod OWUI).

Methode : llama-server STANDALONE sur la V100 (service ollama-v100 stoppe,
restauration en fin de run - protocole supervisor_bench_README). Le runner est
le modele texte qwen3.8-27b-q6-mtp (SANS mmproj) : l'usage prod OWUI est 100%
texte, on economise ~2-3 Go de VRAM 1070 et le runner est plus leger.
MTP OFF (pas de --spec-type) : prefill pur, sans bruit MTP.
KV q8_0 (prod), -b/-ub 2048 (prod), env stricte prod hors MTP, --cache-ram 8192.

Points (n_ctx serveur, taille du contexte A, n_gen) :
  P04 : 16k    /   4k
  P16 : 24k    /  16k
  P32 : 40k    /  32k
  P64 : 80k    /  64k
  P96 : 112k   /  96k    - regne reel OWUI (~114k tokens)
  P128: 140k   / 128k    - plafond test

Sequance par point (serveur FRAIS par point) :
  T0  : prefill FROID du contexte complet A (reference du prefill complet)
  T1  : A + petite question (tout est deja en cache, slot chaud) = residu
  T2  : U1 - A + Q1, generation 8 tok (le contexte U1 vit dans le slot)
  T3  : U2 - contexte B TOTALEMENT different (demi-prompt, seed B) : le slot
        idle de U1 est stationne (RAM cache / checkpoint) avant liberation.
        On mesure le prefill de B ET le temps de stationnement (log serveur).
  T4  : U1 revient - A + Q2 (Q2 != Q1) : LE cas multi-user. Le prefixe A doit
        etre restaure depuis la RAM ; on ne prefill QUE le delta Q2.
  T5  : serveur FRAIS, A + Q2 a froid = le cout SANS RAM cache.

  GAIN = T5_prefill / T4_prefill : factorisation obtenue par la station RAM.

USAGE (depuis l'agent, obligatoire --detach) :
  cd /home/guile/llama-experiments/bench
  python3 supervisor_bench.py run bench_kvcache_ram.py --detach
  ou direct (terminal) : python3 bench_kvcache_ram.py [--quick]
"""
import json, os, re, signal, socket, subprocess, sys, time
import urllib.request
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpu_uuid_select import v100_cuda_ordinal, v100_smi_index

D = os.path.dirname(os.path.abspath(__file__))

LLAMA_SERVER = "/home/guile/llama-experiments/ollama-v100/lib/ollama/llama-server"
GGUF = ("/usr/share/ollama/.ollama/models/blobs/"
        "sha256-493301830a596b8ad56dc1329f80bbcb578c8e910da395feafdc9cd8263430bb")
SO_PROD = "/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so"

PORT = 8918
SERVICE = "ollama-v100"
TS = datetime.now().strftime("%Y%m%d-%H%M%S")
OUT_JSONL = os.path.join(D, f"kvcache_ram_{TS}.jsonl")
OUT_SUMMARY = os.path.join(D, f"kvcache_ram_{TS}_summary.md")
OUT_SERVER_LOG = os.path.join(D, f"kvcache_ram_{TS}_server.log")
OUT_LOG = os.path.join(D, f"kvcache_ram_{TS}.log")

CUDA_LIBS = [
    "/home/guile/llama-experiments/ollama-v100/lib/ollama",
    "/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12",
    "/usr/local/cuda-12.9/lib64",
]

# env stricte prod (hors MTP, hors FA16-compact qui ne sert que le prefill long)
ENV_BASE = {
    "GGML_CUDA_VOLTA_Q6_W4R4": "1",
    "GGML_CUDA_VOLTA_Q8_FATTN_TC": "0",
    "LLAMA_ARG_FIT_TARGET": "1024",
}

# (nom, n_ctx serveur, contexte A en tokens, n_gen)
POINTS = [
    ("P04", 16384,   4096, 16),
    ("P16", 24576,  16384, 16),
    ("P32", 40960,  32768, 16),
    ("P64", 81920,  65536, 16),
    ("P96", 114688, 98304, 16),
    ("P128", 143360, 131072, 16),
]
if "--quick" in sys.argv:
    POINTS = POINTS[:2]

GPU_SMI_IDX = v100_smi_index()
GPU_CUDA_IDX = v100_cuda_ordinal()

FILL = ("La tour de controle surveille le passage des avions au crepuscule "
        "pendant que le chef de piste organise les depart de la soir.")

_state = {}
_restored = False
_srvlog_off = [0]


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(OUT_LOG, "a") as f:
        f.write(line + "\n")


def run(cmd, timeout=60):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def svc_state():
    return run(["systemctl", "is-active", SERVICE]).stdout.strip()


def vram_mib():
    r = run(["nvidia-smi", f"--id={GPU_SMI_IDX}",
             "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    v = r.stdout.strip()
    return int(v) if v.isdigit() else -1


def v100_proc_gone():
    r = run(["nvidia-smi", f"--id={GPU_SMI_IDX}",
             "--query-compute-apps=pid,used_memory", "--format=csv,noheader"])
    for line in r.stdout.splitlines():
        if line.strip():
            parts = line.split(",")
            if len(parts) >= 2:
                mem = parts[1].strip().split()[0]
                if mem.isdigit() and int(mem) > 8000:
                    return False
    return True


def restore_service():
    global _restored
    try:
        if svc_state() != "active":
            log(f"  * restauration : systemctl start {SERVICE}")
            run(["sudo", "systemctl", "start", SERVICE], timeout=120)
        for _ in range(90):
            if svc_state() == "active":
                break
            time.sleep(2)
        _restored = svc_state() == "active"
        log(f"  * service {SERVICE} = {svc_state()}")
    except Exception as e:
        log(f"  * ERREUR restauration : {e}")
        _restored = False


def on_sig(sig, frm):
    restore_service()
    sys.exit(128 + sig)


def stop_and_free():
    log(f"  * stop {SERVICE}")
    run(["sudo", "systemctl", "stop", SERVICE], timeout=120)
    t0 = time.time()
    while time.time() - t0 < 180:
        if vram_mib() < 1024 and v100_proc_gone():
            log(f"  * V100 liberee ({vram_mib()} MiB) en {time.time()-t0:.0f}s")
            return True
        time.sleep(3)
    log(f"  * ATTENTION V100 pas totalement liberee (VRAM={vram_mib()} MiB)")
    return False


def build_env():
    env = dict(os.environ)
    for ddir in CUDA_LIBS:
        if os.path.isdir(ddir) and ddir not in env.get("LD_LIBRARY_PATH", ""):
            env["LD_LIBRARY_PATH"] = ddir + ":" + env.get("LD_LIBRARY_PATH", "")
    env.update(ENV_BASE)
    env["GGML_BACKEND_PATH"] = SO_PROD
    env["CUDA_VISIBLE_DEVICES"] = str(GPU_CUDA_IDX)
    return env


def server_args(n_ctx):
    return [
        "--model", GGUF, "--port", str(PORT), "--host", "127.0.0.1",
        "--no-webui", "--offline",
        "-c", str(n_ctx), "-np", "1",
        "-b", "2048", "-ub", "2048",
        "--log-verbosity", "4",
        # MTP OFF : pas de --spec-type (mesure prefill pure)
        "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
        "--flash-attn", "auto",
        # RAM cache explicite : 8192 MiB (defaut du runner)
        "--cache-ram", "8192",
    ]


def port_free(port):
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", port)); s.close(); return True
    except OSError:
        s.close(); return False


def http_json(path, data=None, timeout=600):
    url = f"http://127.0.0.1:{PORT}{path}"
    req = urllib.request.Request(
        url, data=json.dumps(data).encode() if data is not None else None,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _srvlog_size():
    try:
        return os.path.getsize(OUT_SERVER_LOG)
    except OSError:
        return 0


def srvlog_tail():
    """Lignes du log serveur ecrites depuis le debut du /completion courant."""
    try:
        with open(OUT_SERVER_LOG, "rb") as f:
            f.seek(_srvlog_off[0])
            data = f.read().decode("utf-8", "replace")
    except OSError:
        return []
    keep = []
    for ln in data.splitlines():
        if re.search(r"prompt cache|cache state|restored context checkpoint|"
                     r"saving idle slot|parked|forked|removing oldest|"
                     r"release:|prompt eval time|n_prompt_tokens_cache|"
                     r"selected slot|slot update|cache reuse", ln, re.I):
            keep.append(ln.strip())
    return keep[-12:]


def complete(prompt, n, timeout=900):
    """POST /completion non-stream. Renvoie dict de mesures."""
    _srvlog_off[0] = _srvlog_size()
    t0 = time.time()
    d = http_json("/completion",
                  {"prompt": prompt, "n_predict": n, "stream": False,
                   "temperature": 0}, timeout=timeout)
    dt = time.time() - t0
    m = {
        "wall_s": round(dt, 3),
        "prompt_eval_count": d.get("prompt_eval_count"),
        "prompt_eval_duration_s": d.get("prompt_eval_duration", 0) / 1e9,
        "eval_count": d.get("eval_count"),
        "eval_duration_s": d.get("eval_duration", 0) / 1e9,
        "n_prompt_tokens_cache": d.get("n_prompt_tokens_cache"),
        "content": (d.get("content") or "")[:40],
    }
    pe, pec = m["prompt_eval_count"], m.get("n_prompt_tokens_cache")
    m["fresh_tokens"] = (pe - pec) if (pe is not None and pec is not None) else pe
    if m["prompt_eval_duration_s"] and m.get("fresh_tokens"):
        m["fresh_tps"] = round(m["fresh_tokens"] / m["prompt_eval_duration_s"], 1)
    m["srvlog"] = srvlog_tail()
    return m


def gen_prompt(n_tokens_target, seed):
    """Prompt de ~n_tokens_target tokens (approx 4.15 char/token)."""
    per_char = 4.15
    n_chars = int(n_tokens_target * per_char)
    text = (FILL * ((n_chars // len(FILL)) + 2))[:n_chars]
    return f"(seed {seed}) " + text


def start_server(n_ctx):
    env = build_env()
    lf = open(OUT_SERVER_LOG, "a")
    proc = subprocess.Popen([LLAMA_SERVER] + server_args(n_ctx),
                            stdout=lf, stderr=subprocess.STDOUT, env=env)
    _state["proc"] = proc
    t0 = time.time()
    while time.time() - t0 < 300:
        if proc.poll() is not None:
            raise RuntimeError("serveur mort au demarrage (voir server log)")
        if not port_free(PORT):
            try:
                http_json("/health", timeout=3)
                break
            except Exception:
                pass
        time.sleep(1)
    else:
        raise RuntimeError("serveur pas pret en 300 s")
    log(f"  * serveur pret en {time.time()-t0:.0f}s (n_ctx={n_ctx})")


def stop_server():
    proc = _state.get("proc")
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait(timeout=10)
    t0 = time.time()
    while time.time() - t0 < 120 and not v100_proc_gone():
        time.sleep(2)
    log(f"  * serveur arrete, V100 liberee en {time.time()-t0:.0f}s ({vram_mib()} MiB)")


def q(i, k):
    return f"Question {k} (variante {i}) : quel est le sens de la vie ? Reponds en un mot."


def bench_point(name, n_ctx, a_tokens, n_gen):
    log(f"  === {name} : n_ctx={n_ctx}, contexte A={a_tokens} tok, n_gen={n_gen}")
    start_server(n_ctx)
    rec = {"point": name, "n_ctx": n_ctx, "a_tokens_target": a_tokens,
           "vram_mib": vram_mib()}
    try:
        A = gen_prompt(a_tokens, name + "A")
        B = gen_prompt(int(a_tokens * 0.5), name + "B")  # contexte U2, seed distinct
        qa1, qa2 = q(1, 1), q(1, 2)

        # T0 : prefill FROID du contexte complet
        t0 = complete(A, n_gen)
        rec["T0_cold_full"] = t0
        log(f"  T0 FROID A  : pe={t0['prompt_eval_count']} fresh={t0['fresh_tokens']} "
            f"pe_dur={t0['prompt_eval_duration_s']}s -> {t0.get('fresh_tps')} t/s (wall {t0['wall_s']}s)")

        # T1 : A + petite question, slot chaud (tout est en cache)
        t1 = complete(A + " " + qa2, 8)
        rec["T1_warm"] = t1
        log(f"  T1 WARM     : pe={t1['prompt_eval_count']} cache={t1['n_prompt_tokens_cache']} "
            f"fresh={t1['fresh_tokens']} pe_dur={t1['prompt_eval_duration_s']}s (wall {t1['wall_s']}s)")

        # T2 : U1 - A + Q1 (le contexte U1 vit maintenant dans le slot)
        t2 = complete(A + " " + qa1, 8)
        rec["T2_u1"] = t2
        log(f"  T2 U1 Q1    : pe={t2['prompt_eval_count']} cache={t2['n_prompt_tokens_cache']} "
            f"fresh={t2['fresh_tokens']} pe_dur={t2['prompt_eval_duration_s']}s (wall {t2['wall_s']}s)")

        # T3 : U2 - contexte B totalement different : stationnement de U1 en RAM
        t3 = complete(B + " " + q(2, 1), 8)
        rec["T3_u2"] = t3
        log(f"  T3 U2 (B)   : pe={t3['prompt_eval_count']} cache={t3['n_prompt_tokens_cache']} "
            f"fresh={t3['fresh_tokens']} pe_dur={t3['prompt_eval_duration_s']}s (wall {t3['wall_s']}s)")
        log(f"    log serveur (stationnement) : {json.dumps(t3['srvlog'][-6:], ensure_ascii=False)}")

        # T4 : U1 revient - A + Q2 : restauration RAM + prefill du delta Q2
        t4 = complete(A + " " + qa2, 8)
        rec["T4_u1_back"] = t4
        log(f"  T4 U1 Q2    : pe={t4['prompt_eval_count']} cache={t4['n_prompt_tokens_cache']} "
            f"fresh={t4['fresh_tokens']} pe_dur={t4['prompt_eval_duration_s']}s (wall {t4['wall_s']}s)")
        log(f"    log serveur (restauration) : {json.dumps(t4['srvlog'][-8:], ensure_ascii=False)}")

        # T5 : serveur FRAIS - A + Q2 a froid = le cout SANS RAM cache
        stop_server()
        start_server(n_ctx)
        t5 = complete(A + " " + qa2, 8)
        rec["T5_cold_full_ctx"] = t5
        log(f"  T5 FROID A+Q2 : pe={t5['prompt_eval_count']} fresh={t5['fresh_tokens']} "
            f"pe_dur={t5['prompt_eval_duration_s']}s -> {t5.get('fresh_tps')} t/s (wall {t5['wall_s']}s)")

        if t4["prompt_eval_duration_s"] and t5["prompt_eval_duration_s"]:
            rec["gain_x_vs_cold"] = round(
                t5["prompt_eval_duration_s"] / max(t4["prompt_eval_duration_s"], 0.01), 1)
    finally:
        stop_server()

    with open(OUT_JSONL, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def main():
    signal.signal(signal.SIGTERM, on_sig)
    signal.signal(signal.SIGINT, on_sig)
    log(f"debut - {len(POINTS)} points, port {PORT}, GPU smi={GPU_SMI_IDX} cuda={GPU_CUDA_IDX}")
    log(f"jsonl: {OUT_JSONL}")
    stop_and_free()
    results = []
    try:
        for name, n_ctx, at, ng in POINTS:
            results.append(bench_point(name, n_ctx, at, ng))
            time.sleep(5)
    finally:
        restore_service()

    lines = ["# bench_kvcache_ram - resume (KV cache en RAM, multi-user)", "",
             f"Date : {datetime.now().isoformat()}",
             "Modele : qwen3.8-27b-q6-mtp (texte, sans mmproj) - MTP OFF, KV q8_0, b/ub 2048, cache-ram 8192 MiB",
             "",
             "Scenario : 1 slot (comme prod OWUI). U1 = contexte A (long), U2 = contexte B (demi-longueur, distinct).",
             "",
             "| point | ctx | T0 FROID A (s / t/s) | T1 warm resid (s, fresh) | T3 U2 (s) + station | T4 U1 retour (cache / fresh / pe s / wall s) | T5 FROID A+Q2 (s / t/s) | gain x |",
             "|---|---|---|---|---|---|---|---|"]
    for r in results:
        a = r.get("T0_cold_full", {}); b = r.get("T1_warm", {})
        c = r.get("T3_u2", {}); d = r.get("T4_u1_back", {}); e = r.get("T5_cold_full_ctx", {})
        lines.append(
            f"| {r['point']} | {r['n_ctx']} "
            f"| {a.get('prompt_eval_duration_s')}s / {a.get('fresh_tps')} t/s "
            f"| {b.get('prompt_eval_duration_s')}s / {b.get('fresh_tokens')} "
            f"| {c.get('prompt_eval_duration_s')}s "
            f"| {d.get('n_prompt_tokens_cache')} / {d.get('fresh_tokens')} / {d.get('prompt_eval_duration_s')}s / {d.get('wall_s')}s "
            f"| {e.get('prompt_eval_duration_s')}s / {e.get('fresh_tps')} t/s "
            f"| {r.get('gain_x_vs_cold', '-')} |")
    lines += ["",
              "## Lecture",
              "- T0 = le prefill complet a eviter (cout d'un contexte froid).",
              "- T4 = le multi-user : U1 revient apres que U2 ait pris le slot ;",
              "  le prefixe A est restaure depuis la RAM (logs 'prompt cache update took' /",
              "  'restored context checkpoint'), seul le delta Q2 se prefill.",
              "- wall T4 - pe_dur T4 ~ cout de restauration RAM + overhead (a comparer T4 vs T1).",
              "- T5 = meme demande a froid (serveur neuf) = cout SANS cache RAM.",
              "- gain x = T5 / T4 : ce que la station en RAM fait gagner sur le temps de prompt.",
              "",
              "## Cout de station (RAM) par contexte, deduits du log serveur",
              "Voir les lignes 'cache state: N prompts, X MiB' dans kvcache_ram_*.log -",
              "1 contexte ~a_tokens donne ~ a_tokens x 344.4 o (KV q8_0) de payload stationne."]
    with open(OUT_SUMMARY, "w") as f:
        f.write("\n".join(lines) + "\n")
    log(f"  FIN - resume : {OUT_SUMMARY}")
    log(f"  service restaure : {_restored}")


if __name__ == "__main__":
    main()
