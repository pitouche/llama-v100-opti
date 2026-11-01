#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench_direct.py — bench 2 niveaux, à LANCER PAR L'UTILISATEUR (l'assistant ne
l'exécute pas, pour ne pas interférer). L'assistant se charge UNIQUEMENT de
dépouiller bench_direct_summary.md + bench_direct_results.jsonl.

  Niveau 1 (DIRECT) : llama-server (MÊME build que le runner ollama, MÊME GGUF
                      Q6_K, KV q8_0, flash-attn auto, MTP draft=3, -b 512
                      -ub 512, 1 slot, -c 204800, context-shift, keep 4) sur
                      port 8917, GPU V100, API OpenAI-compatible /v1/chat.
  Niveau 2 (OLLAMA) : le même GGUF via ollama 11434 (runner -c 200192) sur
                      quelques points, pour comparer la courbe et voir si
                      Ollama déforme quelque chose.

Deux expériences séparées :
  A. COLD : préfill complet de taille C (2k → ~200k tokens).
     Chaque point a SON propre texte aléatoire (seed par taille) : aucun
     préfixe commun entre points → le checkpoint/LCP du runner ne peut rien
     restaurer → préfill VRAIEMENT complet. Vérifié au log : restored n_past ≈ 0.
     (Un reset court avant chaque point sert aussi de chauffe des CUDA graphs
     au premier point et laisse une trace de log de contrôle.)
  B. WARM : chaîne de préfixes croissants sur un SEUL corpus B :
     2k → 4k → 8k → 16k → 32k → 64k → 96k → 128k → 160k → ~200k tokens.
     Le runner restaure son checkpoint au préfixe commun et ne prefille QUE
     la diff → coût marginal du contexte chaud. C'est la mesure clé pour
     savoir si garder 200k de contexte est viable. Les tokens réellement
     pré-évalués sortent du log (prompt eval time).

Mesures par run (trois sources croisées) :
  - API du run (usage) : prompt_tokens, gen_tokens, durée
  - log du runner : n_ctx_slot, restored n_past (checkpoint), premier/dernier
    "cached n_tokens", prompt eval time (tokens RÉELLEMENT préfillés + temps),
    eval time, draft acceptance (MTP), stop n_tokens
    (phase direct : bench_direct_server.log ; phase ollama : journal du service)
  - nvidia-smi : VRAM V100 après chaque run (la pool KV est pré-allouée au
    contexte max au démarrage du server : la colonne sert de contrôle, elle ne
    "grandit" pas point par point)

ATTENTION AVANT DE LANCER :
  * Ne pas lancer pendant une génération active du modèle V100 (OpenWebUI en
    cours…) : la phase DIRECT décharge d'abord le runner ollama (keep_alive=0)
    pour libérer la VRAM. Le modèle se rechargera ~25 s au prochain usage.
  * Le script lance/tue lui-même son llama-server (port 8917). S'il refuse de
    démarrer (port occupé, crash), voir bench_direct_server.log.
  * Le script met dans l'env du server : le backend CUDA en plugin
    (GGML_BACKEND_PATH -> cuda_v12/libggml-cuda.so — OUBLIER ça fait reculer
    le server en CPU SILENCIEUSEMENT), les lib CUDA 12.6 + le tuning Volta du
    service ollama-v100 (W4R4 etc.) et ABORTE si le model atterrit en CPU
    (sonde VRAM V100 < 1 GiB après chargement). Sans ça, il a déjà reculé
    en CPU total deux fois (01/10 et 02/10) : 7 t/s au lieu de ~700.
  * Sélection GPU : CUDA_VISIBLE_DEVICES avec l'index de la LIB CUDA
    (cuDeviceGetName — sur cette machine V100 = CUDA 0, alors que
    nvidia-smi dit 1). PAS GGML_CUDA_VISIBLE_DEVICES (cette build ne le lit
    pas). Si on confond les deux index, la 1070 devient CUDA0 et le server
    crash sur "no kernel image" (build Volta, pas de SASS sm_61).
  * Durée totale estimée : ~25-35 min (direct ~15-20 min + ollama ~5-10 min,
    dont ~25 s de rechargement du modèle avant le premier point ollama).

USAGE :
  python3 bench_direct.py                     # tout (DIRECT puis OLLAMA)
  python3 bench_direct.py --quick             # pipeline minimal (~4 min) pour
                                              # valider la mécanique d'abord
  python3 bench_direct.py --phase direct      # seulement llama.cpp direct
  python3 bench_direct.py --phase ollama      # seulement ollama (pas de server)
  python3 bench_direct.py --cold 16384 196608 --warm 32768 98304   # points custom (en tokens)
  python3 bench_direct.py --no-ollama-warm    # sauter la chaîne warm ollama
  python3 bench_direct.py --leave-server      # ne pas tuer le llama-server
  python3 bench_direct.py --no-unload         # ne pas décharger le modèle
                                              # ollama avant (risque OOM)

SORTIES (dossier du script) :
  bench_direct_results.jsonl   une ligne JSON par run (api + log + vram)
  bench_direct_summary.md      tableau de synthèse lisible (à me coller)
  bench_direct_server.log      stdout/stderr du llama-server lancé
"""
import argparse
import json
import os
import random
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request

D = os.path.dirname(os.path.abspath(__file__))
OUT_JSONL = os.path.join(D, "bench_direct_results.jsonl")
OUT_SUMMARY = os.path.join(D, "bench_direct_summary.md")
OUT_SERVER_LOG = os.path.join(D, "bench_direct_server.log")

# ------------------- CONFIG (identique au runner ollama) ---------------------
LLAMA_SERVER = "/home/guile/llama-experiments/ollama-v100/lib/ollama/llama-server"
GGUF = ("/usr/share/ollama/.ollama/models/blobs/"
        "sha256-493301830a596b8ad56dc1329f80bbcb578c8e910da395feafdc9cd8263430bb")
MMPROJ = ("/usr/share/ollama/.ollama/models/blobs/"
          "sha256-ac3714bfdddeca31351f2752bf1a63f266f4df87c0b68c895e44945ca704448e")
MODEL_OLLAMA = "qwen3.8-27b-q6-mtp-vision:latest"

PORT_DIRECT = 8917
PORT_OLLAMA = 11434
OLLAMA_UNIT = "ollama-v100.service"
GPU_NAME = "V100"

# MÊME environnement CUDA que le service ollama-v100.service (sinon le binaire
# ne trouve pas libcudart/libcublas 12.6 et le server se lance EN CPU total —
# c'est ce qui est arrivé le 01/10 : 6.9 t/s sur le smoke, KV buffer CPU).
# Ordre identique à la LD_LIBRARY_PATH du service (voir /proc/<runner>/environ).
CUDA_LIBS = ["/home/guile/llama-experiments/ollama-v100/lib/ollama",
             "/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12",
             "/home/guile/llama-experiments/tools/cuda-12.6/usr/local/cuda-12.6/lib64",
             "/home/guile/llama-experiments/tools/cuda-12.6/usr/local/cuda-12.6/targets/x86_64-linux/lib"]
# CRITIQUE (bug du 02/10) : dans cette build, le backend CUDA est un PLUGIN
# isolé (cuda_v12/libggml-cuda.so), pas à côté du binaire llama-server. Sans
# GGML_BACKEND_PATH, le server démarre SILENCIEUSEMENT 100 % CPU (device_info
# sans CUDA, KV buffer CPU, 7 t/s) — c'est le Ollama serve qui met cette var
# dans l'env de son runner, il ne suffit pas de copier la LD_LIBRARY_PATH.
GGML_CUDA_BACKEND = "/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so"
# même tuning Volta que le runner ollama (W4R4 etc.) + fit target du service
GGML_ENV = {
    "GGML_CUDA_ALLREDUCE": "internal",
    "GGML_CUDA_AR_COPY_THRESHOLD": "131072",
    "GGML_CUDA_VOLTA_Q8_FATTN_TC": "1",
    "GGML_CUDA_VOLTA_Q5_X4": "1",
    "GGML_CUDA_VOLTA_Q6_W4R4": "1",
    "GGML_CUDA_VOLTA_GQA8_NCOLS2": "2",
    "LLAMA_ARG_FIT_TARGET": "1024",  # même que ollama-v100.service
}

def build_env(gpu_cuda_idx):
    """Env du llama-server DIRECT — à reproduire à l'IDENTIQUE partout
    (start_server, smoke_check) pour qu'un changement d'env n'ait pas de
    version parallèle qui pourrit les tests."""
    env = dict(os.environ)
    for d in CUDA_LIBS:
        if os.path.isdir(d) and d not in env.get("LD_LIBRARY_PATH", ""):
            env["LD_LIBRARY_PATH"] = d + ":" + env.get("LD_LIBRARY_PATH", "")
    env.update(GGML_ENV)
    env["GGML_BACKEND_PATH"] = globals().get("GGML_CUDA_BACKEND", GGML_CUDA_BACKEND)  # CRITIQUE: sans ça = CPU silencieux (bug 02/10)
    # Sélection GPU : CUDA_VISIBLE_DEVICES (PAS un "GGML_CUDA_VISIBLE_DEVICES" —
    # cette build ne le lit pas, vérifié par strings sur le plugin). L'index CVD
    # est l'index de la LIB CUDA, PAS l'index nvidia-smi (les deux diffèrent sur
    # cette machine : V100 = CUDA 0 / nvidia-smi 1). Après filtrage, la V100 est
    # device 0 du process — c'est le CUDA0 attendu dans le log.
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_cuda_idx)
    return env

SERVER_ARGS = [
    "--model", GGUF,
    "--port", str(PORT_DIRECT),
    "--host", "127.0.0.1",
    "--no-webui", "--offline",
    "-c", "204800",            # contexte max (runner ollama : -c 200000 → 200192)
    "-np", "1",                # 1 slot
    "--log-verbosity", "4",
    "--no-log-prefix", "--no-log-timestamps",
    "--mmproj", MMPROJ,
    "--spec-type", "draft-mtp",
    "--spec-draft-n-max", "3",
    "--spec-draft-backend-sampling",
    "--cache-type-k", "q8_0",
    "--cache-type-v", "q8_0",
    "--flash-attn", "auto",
    "-b", "512",
    "-ub", "512",
    "--context-shift",
    "--keep", "4",
]

# cibles en TOKENS (196608 ≈ "200k" avec marge prompt+gen sous 200192/204800)
COLD_SIZES = [2048, 4096, 8192, 16384, 32768, 65536, 98304, 131072, 163840, 196608]
WARM_SIZES = [2048, 4096, 8192, 16384, 32768, 65536, 98304, 131072, 163840, 196608]
OLLAMA_COLD = [16384, 65536, 131072, 196608]
OLLAMA_WARM = [32768, 98304, 163840]
QUICK_COLD = [2048, 16384]
QUICK_WARM = [4096, 32768]
QUICK_OLLAMA_COLD = [16384]
QUICK_OLLAMA_WARM = [32768]

N_PRED = 4            # gen minuscule (pas l'objet du bench)
TIMEOUT_S = 1800      # par requête (prefill ~200k ≈ 5-6 min au pire)
LOG_WAIT_S = 6.0      # attente max du flush des lignes de timing après le run

# ------------------- corpus déterministe (ratio tok/mot MESURÉ) -----------------
# Le "1.03" d'origine était FAUX (supposition) : le run 02/10 donne, sur les
# points réels préfillés, des ratios stables par corpus :
#   cold (WORDS)      : 1.427–1.447 selon seed (LS 1.4326, 6 points 2k→131k)
#   warm (WORDS_WARM) : 1.14–1.16 (diff warm_4096→warm_32768 du même run)
# Conséquence du bug : cold_163840 a généré 228162 tokens → 400 Bad Request
# ("exceeds the available context size (204800)") au 5e point du run complet.
# On prend le bord HAUT du ratio (mots en moins, C légèrement sous-cible mais
# jamais au-dessus du contexte — la colonne 'préfill réel' donne l'exact).
WORDS = ["le", "modele", "contexte", "memoire", "tenseur", "attention", "cache",
         "prompt", "decodage", "generation", "vram", "cuda", "kernel", "fenetre",
         "glissante", "latence", "debits", "buffer", "layer", "groupe", "recurrent",
         "etats", "passage", "eval", "charge", "tests", "resultats", "cible", "draft",
         "tokens", "seconds", "prefill", "infer", "quant", "q6", "mtp", "speculatif"]
WORDS_WARM = ["un", "vecteur", "gradient", "matrice", "somme", "produit", "norme",
              "ecart", "moyenne", "variance", "entropie", "logits", "softmax",
              "embedding", "position", "rope", "scale", "theta", "freq", "base",
              "alpha", "beta", "gamma", "delta", "omega", "sigma", "lambda", "mu",
              "rho", "phi", "kappa", "nu", "xi", "eta", "iota", "flux", "masse"]
WORDS_DUMMY = ["reset", "purge", "flush", "vider", "effacer", "reinitialiser",
               "zone", "tampon", "slot", "liberer", "memoire", "fragment",
               "tableau", "pointeur"]

PROMPT = "\nReponds exactement : OK-BENCH."

def _stream(seed, vocab, n_words):
    rng = random.Random(seed)
    return " ".join(rng.choice(vocab) for _ in range(n_words))

R_COLD = 1.45   # tok/mot du corpus cold (bord haut mesuré 1.4473)
R_WARM = 1.16   # tok/mot du corpus warm (bord haut mesuré 1.1579)

def words_for(n_tokens, r):
    """cible tokens → nb de mots du remplissage (ratio mesuré, bord haut)."""
    return int(n_tokens / r)

# Cold : UN seed par taille → aucun préfixe commun entre points (pas de LCP)
def cold_text(c_tokens):
    return _stream(20261000 + c_tokens, WORDS, words_for(c_tokens, R_COLD)) + PROMPT

# Warm : UN SEUL corpus croissant (le préfixe doit être partagé entre points)
MAXW_WARM = words_for(max(WARM_SIZES + OLLAMA_WARM + QUICK_WARM + QUICK_OLLAMA_WARM), R_WARM) + 6000
STREAM_WARM = _stream(20261002, WORDS_WARM, MAXW_WARM)

def warm_text(c_tokens):
    return " ".join(STREAM_WARM.split(" ")[:words_for(c_tokens, R_WARM)]) + PROMPT

def dummy_text(i):
    rng = random.Random(900000 + i)
    return " ".join(rng.choice(WORDS_DUMMY) for _ in range(64)) + f" RESETMARK_{i} done."

# ------------------- infra ----------------------------------------------------
_state = {"proc": None, "gpu_idx": None, "gpu_cuda_idx": None, "gpu_name": None,
          "run_start_ts": 0, "load_s": None, "model_id": None, "quick": False}

def log(msg):
    print(msg, flush=True)

def cuda_device_index(gpu_name):
    """Index du device dans l'énumération de la LIB CUDA (cuDeviceGetName) —
    C'EST CET INDEX que CUDA_VISIBLE_DEVICES attend. Il peut DIFFÉRER de
    l'index nvidia-smi (ici : nvidia-smi 1070=0, V100=1 ; lib CUDA V100=0,
    1070=1 — d'où le crash "no kernel image" quand on confondait les deux)."""
    import ctypes
    lib = ctypes.CDLL("libcuda.so.1")
    if lib.cuInit(0) != 0:
        raise RuntimeError("cuInit a échoué (driver NVIDIA)")
    count = ctypes.c_int(0)
    lib.cuDeviceGetCount(ctypes.byref(count))
    for i in range(count.value):
        dev = ctypes.c_int(i)
        buf = ctypes.create_string_buffer(128)
        lib.cuDeviceGetName(buf, 128, dev)
        if gpu_name in buf.value.decode():
            return i
    raise RuntimeError(f"GPU {gpu_name} absent de l'énumération CUDA")

def find_gpu():
    """Retourne (idx_nvidia_smi, cuda_idx, nom) de la V100.
    - idx_nvidia_smi : pour `nvidia-smi -i` (surveillance VRAM)
    - cuda_idx       : pour CUDA_VISIBLE_DEVICES (CVD) — l'index de la LIB CUDA"""
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    smi_idx, name = None, None
    for line in out.splitlines():
        if GPU_NAME in line:
            i, n = line.split(",", 1)
            smi_idx, name = int(i.strip()), n.strip()
    if smi_idx is None:
        raise RuntimeError(f"GPU {GPU_NAME} introuvable")
    return smi_idx, cuda_device_index(GPU_NAME), name

def vram_mib():
    out = subprocess.run(["nvidia-smi", "-i", str(_state["gpu_idx"]),
                          "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return int(out)

def http_json(url, payload, timeout):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return d, time.time() - t0

def http_get(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode()

def port_busy(port):
    s = socket.socket()
    s.settimeout(1)
    try:
        return s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        s.close()

# ------------------- log du runner ---------------------------------------------
RE_NCTX = re.compile(r"new prompt, n_ctx_slot = (\d+), n_keep = (\d+), task\.n_tokens = (\d+)")
RE_RESTORE = re.compile(r"restored context checkpoint .* n_past = (\d+)")
RE_CACHED = re.compile(r"cached n_tokens = (\d+)")
RE_PROMPT = re.compile(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens")
RE_EVAL = re.compile(r"(?<!prompt )eval time =\s*([\d.]+) ms /\s*(\d+) tokens \(.*?([\d.]+) tokens per second\)")
RE_ACCEPT = re.compile(r"draft acceptance = ([\d.]+) \(\s*(\d+) accepted /\s*(\d+) generated\), mean len =\s*([\d.]+)")
RE_STOP = re.compile(r"stop processing: n_tokens = (\d+)")
RE_TASK = re.compile(r"task (\d+) \|")
RE_SHIFT = re.compile(r"shift", re.I)

def journal_entries(since_epoch, unit, n=30000):
    cmd = ["journalctl", "-u", unit, "-o", "json", "--no-pager", "-n", str(n),
           f"--since=@{int(since_epoch)}"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except Exception:
        return []
    out = []
    for line in p.stdout.splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        rt = 0
        for k, v in e.items():
            if k.strip("_").upper() == "REALTIME_TIMESTAMP":
                rt = int(v or 0)
        out.append((rt / 1e6 if rt else time.time(), e.get("MESSAGE", "")))
    out.sort(key=lambda x: x[0])
    return out

def server_log_chunk(offset):
    """Nouvelles lignes du log du llama-server depuis offset (phase direct)."""
    try:
        with open(OUT_SERVER_LOG, "r", errors="replace") as f:
            f.seek(offset)
            return f.read().splitlines()
    except FileNotFoundError:
        return []

def extract_run(w):
    """w = lignes du run ((t, msg) déjà filtrées) : dernier 'new prompt' =
    le run mesuré. Renvoie le dict résumé (tokens réellement pré-évalués, etc.)."""
    run = {"n_ctx_slot": None, "task_n_tokens": None, "restored_n_past": None,
           "cached_first": None, "cached_last": None, "n_cached_lines": 0,
           "prompt_eval_ms": None, "prompt_eval_tokens": None,
           "eval_ms": None, "eval_tokens": None, "eval_tps": None,
           "acceptance": None, "acc_accepted": None, "acc_generated": None,
           "mean_len": None, "stop_n_tokens": None, "shift_hits": 0}
    for _, m in w:
        if RE_SHIFT.search(m):
            run["shift_hits"] += 1
    news = [(t, m) for (t, m) in w if RE_NCTX.search(m)]
    if not news:
        return run
    t_new, m_new = news[-1]
    mm = RE_NCTX.search(m_new)
    run["n_ctx_slot"] = int(mm.group(1))
    run["task_n_tokens"] = int(mm.group(3))
    task_m = RE_TASK.search(m_new)
    task = ("task " + task_m.group(1) + " |") if task_m else None
    for t, m in w:
        if t < t_new - 0.5:
            continue
        if task is not None and task not in m and not (
                RE_PROMPT.search(m) or RE_EVAL.search(m)
                or RE_ACCEPT.search(m) or RE_STOP.search(m)):
            continue
        if (mm2 := RE_RESTORE.search(m)):
            run["restored_n_past"] = int(mm2.group(1))
        if (mm2 := RE_CACHED.search(m)):
            v = int(mm2.group(1))
            if run["cached_first"] is None:
                run["cached_first"] = v
            run["cached_last"] = v
            run["n_cached_lines"] += 1
        if (mm2 := RE_PROMPT.search(m)):
            run["prompt_eval_ms"] = float(mm2.group(1))
            run["prompt_eval_tokens"] = int(mm2.group(2))
        if (mm2 := RE_EVAL.search(m)):
            run["eval_ms"] = float(mm2.group(1))
            run["eval_tokens"] = int(mm2.group(2))
            run["eval_tps"] = float(mm2.group(3))
        if (mm2 := RE_ACCEPT.search(m)):
            run["acceptance"] = float(mm2.group(1))
            run["acc_accepted"] = int(mm2.group(2))
            run["acc_generated"] = int(mm2.group(3))
            run["mean_len"] = float(mm2.group(4))
        if (mm2 := RE_STOP.search(m)):
            run["stop_n_tokens"] = int(mm2.group(1))
    return run

# ------------------- requêtes ----------------------------------------------------
def req_direct(text, tag, phase):
    payload = {
        "model": _state["model_id"],
        "messages": [{"role": "user", "content": text}],
        "max_tokens": N_PRED,
        "stream": False,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
    }
    d, wall = http_json(f"http://127.0.0.1:{PORT_DIRECT}/v1/chat/completions",
                        payload, TIMEOUT_S)
    u = d.get("usage", {})
    return {
        "prompt_tokens": u.get("prompt_tokens"),
        "gen_tokens": u.get("completion_tokens"),
        "wall_s": round(wall, 2),
    }

def req_ollama(text, tag, phase):
    payload = {
        "model": MODEL_OLLAMA,
        "messages": [{"role": "user", "content": text}],
        "stream": False,
        "options": {"num_predict": N_PRED, "keep_alive": -1,
                    "temperature": 0.0, "top_k": 1},
    }
    d, wall = http_json(f"http://127.0.0.1:{PORT_OLLAMA}/api/chat", payload, TIMEOUT_S)
    return {
        "prompt_tokens": d.get("prompt_eval_count"),
        "gen_tokens": d.get("eval_count"),
        "wall_s": round(wall, 2),
        "prompt_s": round(d.get("prompt_eval_duration", 0) / 1e9, 3),
        "eval_s": round(d.get("eval_duration", 0) / 1e9, 3),
        "load_s": round(d.get("load_duration", 0) / 1e9, 2),
    }

# ------------------- gestion du serveur ------------------------------------------
def unload_ollama_model():
    # ne décharger que si le modèle est réellement chargé (sinon generate
    # rechargerait juste pour rien)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT_OLLAMA}/api/ps", timeout=10) as r:
            loaded = [m.get("name", "") for m in json.loads(r.read()).get("models", [])]
    except Exception:
        log("  · WARN: /api/ps inatteignable — on tente le déchargement quand même")
        loaded = [MODEL_OLLAMA]
    if any(MODEL_OLLAMA in n or n.startswith(MODEL_OLLAMA.split(":")[0]) for n in loaded):
        try:
            # /api/generate est en STREAM par défaut (plusieurs objets JSON
            # séparés par \n) -> "stream": false obligatoire, sinon
            # json.loads() lève "Extra data: line 2 column 1"
            http_json(f"http://127.0.0.1:{PORT_OLLAMA}/api/generate",
                      {"model": MODEL_OLLAMA, "prompt": "x", "keep_alive": 0,
                       "stream": False,
                       "options": {"num_predict": 1}}, 120)
            log("  · modèle ollama déchargé (keep_alive=0)")
        except Exception as e:
            log(f"  · WARN: déchargement ollama impossible: {e}")
    else:
        log("  · modèle ollama pas chargé, rien à décharger")
    for _ in range(60):
        if vram_mib() < 6000:
            return
        time.sleep(2)
    log("  · WARN: VRAM pas encore libérée par ollama après 2 min")

def start_server():
    if port_busy(PORT_DIRECT):
        raise RuntimeError(f"port {PORT_DIRECT} déjà occupé — libérez-le avant de relancer")
    # anti OOM : si la V100 est encore pleine d'un autre process (runner ollama
    # pas encore libéré, génération en cours…), le server crashe en cudaMemGetInfo
    # OOM au chargement — on lève plus tôt, avec un message lisible
    if vram_mib() > 6000:
        raise RuntimeError(
            f"ABORT: VRAM {GPU_NAME} = {vram_mib()} MiB AVANT le lancement — un autre "
            "processus occupe le GPU (runner ollama encore chargé ? génération en "
            "cours ? nvidia-smi). Réessayez une fois qu'il est libéré.")
    env = build_env(_state["gpu_cuda_idx"])
    # le log de startup DOIT montrer la V100 comme CUDA0 ; sinon (1070) le
    # server va crasher sur "no kernel image" (build Volta, pas de SASS 610)
    log(f"  · GPU cible: CUDA dev {_state['gpu_cuda_idx']} ({_state['gpu_name']}) → CUDA0 filtré")
    f = open(OUT_SERVER_LOG, "w")
    t0 = time.time()
    proc = subprocess.Popen([LLAMA_SERVER] + SERVER_ARGS, stdout=f,
                            stderr=subprocess.STDOUT, env=env)
    _state["proc"] = proc
    log(f"  · llama-server lancé (pid {proc.pid}), chargement…")
    deadline = time.time() + 300
    gpu_ok = False
    def tail_log(n=8):
        try:
            with open(OUT_SERVER_LOG, "r", errors="replace") as lf:
                return " | ".join(lf.read().splitlines()[-n:])
        except FileNotFoundError:
            return "(log vide)"
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"llama-server est mort au démarrage (exit {proc.returncode}). "
                f"FIN DU LOG: {tail_log()}")
        # contrôle anti-recule-CPU / anti-mauvaise-carte : le log doit montrer
        # la V100 dans device_info ET "using device ... V100"
        try:
            with open(OUT_SERVER_LOG, "r", errors="replace") as lf:
                head = lf.read()
            if not gpu_ok and GPU_NAME in head:
                gpu_ok = True
        except FileNotFoundError:
            pass
        try:
            if "ok" in http_get(f"http://127.0.0.1:{PORT_DIRECT}/health", 3).lower():
                _state["load_s"] = round(time.time() - t0, 1)
                time.sleep(3)  # laisser les buffers GPU s'allouer avant la sonde VRAM
                if vram_mib() < 1024:
                    raise RuntimeError(
                        f"ABORT: VRAM {GPU_NAME} = {vram_mib()} MiB après chargement — "
                        "le model est SÛR le CPU (CUDA non résolu). Voir bench_direct_server.log.")
                d = json.loads(http_get(f"http://127.0.0.1:{PORT_DIRECT}/v1/models", 10))
                _state["model_id"] = d["data"][0]["id"]
                log(f"  · llama-server prêt (load {_state['load_s']}s, model id '{_state['model_id']}', "
                    f"GPU={GPU_NAME if gpu_ok else '??? (non confirmée)'}; VRAM {vram_mib()} MiB)")
                return
        except RuntimeError:
            raise
        except Exception:
            pass
        # le warmup peut faire crasher le server juste après "model loaded"
        # (ex: "no kernel image" si la 1070 est devenue CUDA0) — détecté plus tôt
        try:
            with open(OUT_SERVER_LOG, "r", errors="replace") as lf:
                if "no kernel image" in lf.read():
                    proc.kill()
                    raise RuntimeError(
                        f"ABORT: 'no kernel image' dans le log — mauvaise carte (build Volta, "
                        f"pas de SASS sm_61 pour la 1070). {tail_log()}")
        except FileNotFoundError:
            pass
        time.sleep(1)
    raise RuntimeError("timeout attente /health du llama-server")

def stop_server():
    p = _state.get("proc")
    if p and p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=15)
        except Exception:
            p.kill()
        log("  · llama-server arrêté")
    _state["proc"] = None

def on_sigint(signum, frame):
    log("\n^C — arrêt du script, nettoyage…")
    p = _state.get("proc")
    if p and p.poll() is None:
        p.terminate()
    sys.exit(130)

# ------------------- run générique -------------------------------------------------
def measure(tag, phase, mode, text, c_target, c_prev, req_fn, unit=None):
    t_start = time.time()
    # offset du log direct AVANT la requête (le chunk lu après = ce run)
    direct_offset = os.path.getsize(OUT_SERVER_LOG) if (phase == "direct" and os.path.exists(OUT_SERVER_LOG)) else 0
    api = req_fn(text, tag, phase)
    t_end = time.time()
    if phase == "direct":
        # attendre le flush des lignes de timing (print_timing/stop processing)
        lines = []
        deadline = time.time() + LOG_WAIT_S
        while time.time() < deadline:
            lines = server_log_chunk(direct_offset)
            blob = "\n".join(lines)
            if "prompt eval time" in blob or "stop processing" in blob:
                break
            time.sleep(0.2)
        w = list(enumerate(lines))
    else:
        # journald peut être en retard d'une seconde ou deux sur la réponse HTTP :_
        # poller jusqu'à la ligne de timing du run
        entries, w = [], []
        deadline = time.time() + LOG_WAIT_S
        while True:
            entries = journal_entries(t_start - 1, unit=unit)
            w = [(t, m) for (t, m) in entries if t_start - 2 <= t <= t_end + 10]
            blob = "\n".join(m for _, m in w)
            if ("prompt eval time" in blob or "stop processing" in blob) or time.time() >= deadline:
                break
            time.sleep(0.3)
    jr = extract_run(w)
    vram = vram_mib()
    row = {
        "ts": int(t_start), "phase": phase, "mode": mode, "tag": tag,
        "C_target_tok": c_target, "C_prev_tok": c_prev,
        "api": api, "journal": jr, "vram_mib": vram,
    }
    pt = api.get("prompt_tokens")
    row["prefill_tps_api"] = round(pt / api["wall_s"], 1) if pt and api.get("wall_s") else None
    if jr.get("prompt_eval_tokens") and jr.get("prompt_eval_ms"):
        row["prefill_tps_journal"] = round(jr["prompt_eval_tokens"] / (jr["prompt_eval_ms"] / 1000), 1)
        row["ms_per_tok_journal"] = round(jr["prompt_eval_ms"] / jr["prompt_eval_tokens"], 3)
    if api.get("eval_s"):
        row["gen_tps"] = round(api["gen_tokens"] / api["eval_s"], 1)
    rp = jr.get("restored_n_past") or 0
    if mode == "cold" and rp > 200:
        row["note"] = f"ATTENTION: checkpoint restauré à {rp} — ce n'est PAS un prefill complet"
    if mode == "warm_init" and rp > 300:
        row["note"] = (f"ATTENTION: warm_init a restauré un checkpoint à {rp} (run précédent "
                       f"probablement encore en pool) — ce point est une re-utilisation, pas un préfill")
    elif mode == "warm":
        expect = c_target - (c_prev or 0)
        if rp < 0.5 * max(1, expect - 3000):
            row["note"] = f"ATTENTION: restore={rp} en dessous de l'attendu (C_prev={c_prev})"
    with open(OUT_JSONL, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    jt, jms = jr.get("prompt_eval_tokens"), jr.get("prompt_eval_ms")
    spd = f"{jt/(jms/1000):.0f} t/s" if (jt and jms) else "?"
    log(f"  [{tag}] {mode} C={c_target} | préfill réel {jt if jt is not None else '?'} tok "
        f"en {(jms/1000 if jms else '?')}s ({spd}) | restore={rp} | gen={api.get('gen_tokens')}"
        + (f" | {row['note']}" if row.get("note") else ""))
    return row

# ------------------- phases ------------------------------------------------------------
def phase_direct(cold_sizes, warm_sizes):
    log(f"\n=== PHASE 1 — DIRECT (llama-server :{PORT_DIRECT}, V100) ===")
    unload_ollama_model()
    time.sleep(3)
    start_server()
    i = 0
    log("\n--- A. COLD : préfill complet (texte unique par taille, pas de LCP possible) ---")
    for c in cold_sizes:
        i += 1
        measure(f"reset_{i}", "direct", "dummy", dummy_text(i), None, None, req_direct)
        time.sleep(1)
        measure(f"cold_{c}", "direct", "cold", cold_text(c), c, None, req_direct)
        time.sleep(1)
    log("\n--- B. WARM : chaîne de préfixes croissants (seule la diff est prefilled) ---")
    prev = None
    for c in warm_sizes:
        mode = "warm_init" if prev is None else "warm"
        measure(f"warm_{c}", "direct", mode, warm_text(c), c, prev, req_direct)
        time.sleep(1)
        prev = c

def phase_ollama(cold_sizes, warm_sizes, do_warm=True):
    log(f"\n=== PHASE 2 — OLLAMA (port {PORT_OLLAMA}, runner -c 200192) ===")
    i = 0
    log("\n--- A. COLD (mêmes textes que direct) ---")
    for c in cold_sizes:
        i += 1
        measure(f"oreset_{i}", "ollama", "dummy", dummy_text(1000 + i), None, None,
                req_ollama, unit=OLLAMA_UNIT)
        time.sleep(1)
        measure(f"ocold_{c}", "ollama", "cold", cold_text(c), c, None,
                req_ollama, unit=OLLAMA_UNIT)
        time.sleep(1)
    if do_warm and warm_sizes:
        log("\n--- B. WARM (chaîne sur corpus warm, via ollama) ---")
        prev = None
        for c in warm_sizes:
            mode = "warm_init" if prev is None else "warm"
            measure(f"owarm_{c}", "ollama", mode, warm_text(c), c, prev,
                    req_ollama, unit=OLLAMA_UNIT)
            time.sleep(1)
            prev = c
    log("  · runner ollama laissé chargé (keep_alive=-1) — le modèle est de nouveau dispo")

# ------------------- synthèse -------------------------------------------------------------
def fnum(v, fmt="{:.0f}"):
    return fmt.format(v) if isinstance(v, (int, float)) else "·"

def write_summary(rows, quick=False):
    L = []
    L.append("# Bench direct — synthèse (à laisser à l'assistant pour dépouillement)")
    L.append(f"Généré le {time.strftime('%Y-%m-%d %H:%M:%S')} par bench_direct.py")
    L.append("")
    L.append("## A. COLD — coût d'un préfill complet")
    L.append("'préfill réel' = tokens réellement pré-évalués selon le log du runner "
             "(restore ≈ 0 vérifié). 't/s api' = prompt_tokens(total)/durée : biaisé à la "
             "hausse s'il reste du cache (artefact connu du 01/10).")
    L.append("")
    L.append("| phase | C cible (tok) | prompt (api) | restore | préfill réel (tok) | préfill (s) | t/s log | ms/tok | t/s api | VRAM (GiB) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r.get("mode") != "cold":
            continue
        a, j = r["api"], r["journal"]
        L.append(f"| {r['phase']} | {r['C_target_tok']} | {a.get('prompt_tokens')} | {j.get('restored_n_past')} "
                 f"| {j.get('prompt_eval_tokens')} | {fnum((j.get('prompt_eval_ms') or 0)/1000, '{:.2f}')} "
                 f"| {fnum(r.get('prefill_tps_journal'))} | {fnum(r.get('ms_per_tok_journal'), '{:.2f}')} "
                 f"| {fnum(r.get('prefill_tps_api'))} | {fnum((r.get('vram_mib') or 0)/1024, '{:.1f}')} |")
    L.append("")
    L.append("## B. WARM — coût marginal : Δ prefilled quand C_prev est déjà en KV")
    L.append("C'est LA mesure pour la viabilité du contexte 200k conservé.")
    L.append("")
    L.append("| phase | C_prev (tok) | C final (tok) | Δ attendu | restore (n_past) | diff réelle (tok) | diff (s) | t/s chaud | ms/tok | VRAM (GiB) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r.get("mode") not in ("warm", "warm_init"):
            continue
        j = r["journal"]
        dexp = (r["C_target_tok"] - (r["C_prev_tok"] or 0)) if r.get("C_prev_tok") else r["C_target_tok"]
        L.append(f"| {r['phase']} | {r.get('C_prev_tok') or '·'} | {r['C_target_tok']} | {dexp} "
                 f"| {j.get('restored_n_past')} | {j.get('prompt_eval_tokens')} "
                 f"| {fnum((j.get('prompt_eval_ms') or 0)/1000, '{:.2f}')} "
                 f"| {fnum(r.get('prefill_tps_journal'))} | {fnum(r.get('ms_per_tok_journal'), '{:.2f}')} "
                 f"| {fnum((r.get('vram_mib') or 0)/1024, '{:.1f}')} |")
    L.append("")
    L.append("## C. OLLAMA vs DIRECT (points communs, t/s du log)")
    L.append("")
    L.append("| point | direct | ollama | Δ ollama |")
    L.append("|---|---|---|---|")
    def tps_of(phase, mode, c):
        for r in rows:
            if r["phase"] == phase and r["mode"] == mode and r["C_target_tok"] == c:
                return r.get("prefill_tps_journal")
        return None
    if quick:
        sizes = sorted(set(QUICK_COLD + QUICK_WARM + QUICK_OLLAMA_COLD + QUICK_OLLAMA_WARM))
    else:
        sizes = sorted(set(COLD_SIZES) | set(WARM_SIZES) | set(OLLAMA_COLD) | set(OLLAMA_WARM))
    for c in sizes:
        for mode in ("cold", "warm"):
            d_, o_ = tps_of("direct", mode, c), tps_of("ollama", mode, c)
            if d_ and o_:
                L.append(f"| {mode} {c} | {d_:.0f} | {o_:.0f} | {(o_/d_-1)*100:+.1f}% |")
    L.append("")
    L.append("## D. MTP (draft) — acceptance par run")
    L.append("")
    L.append("| run | phase | acceptance | mean len | VRAM (GiB) |")
    L.append("|---|---|---|---|---|")
    for r in rows:
        j = r["journal"]
        if j.get("acceptance") is not None:
            L.append(f"| {r['tag']} | {r['phase']} | {j['acceptance']:.3f} | {j.get('mean_len')} "
                     f"| {fnum((r.get('vram_mib') or 0)/1024, '{:.1f}')} |")
    L.append("")
    L.append("## E. Notes de méthode")
    L.append("- Cold : chaque point a un texte aléatoire dédié (seed par taille) → aucun préfixe commun entre points → le checkpoint/LCP du runner ne restaure rien (restored n_past ≈ 0 vérifié). Si un point COLD a restore > 200, il est invalide.")
    L.append("- Warm : même corpus croissant (B, distinct des corpus cold) → le runner restaure le checkpoint au préfixe commun et ne prefille que la diff ; la colonne 'diff réelle' vient du log (prompt eval time).")
    L.append("- C est la cible en TOKENS ; le remplissage est dimensionné par les ratios tok/mot mesurés (R_COLD=1.45, R_WARM=1.16, bord haut) → C réel légèrement sous la cible ; les colonnes prompt/diff réelle donnent les tokens exacts mesurés.")
    L.append("- Run direct = MÊME build de llama-server, MÊME GGUF, MÊMES args que le runner ollama (-c 204800, q8_0 KV, flash-attn auto, MTP draft=3, -b 512 -ub 512, 1 slot, context-shift, keep 4).")
    L.append("- VRAM : la pool KV est pré-allouée au contexte max au démarrage du server → la colonne est un contrôle (constante attendue), pas une mesure de croissance.")
    L.append("- Le premier point ollama inclut ~25 s de rechargement du modèle dans wall_s (pas dans les t/s du log).")
    L.append(f"- Lignes brutes : {OUT_JSONL}")
    with open(OUT_SUMMARY, "w") as f:
        f.write("\n".join(L) + "\n")

def main():
    ap = argparse.ArgumentParser(description="bench 2 niveaux (llama.cpp direct vs ollama)")
    ap.add_argument("--phase", choices=["direct", "ollama", "all"], default="all")
    ap.add_argument("--quick", action="store_true", help="pipeline minimal de validation (~4 min)")
    ap.add_argument("--cold", type=int, nargs="*", default=None, help="cibles cold custom (en tokens)")
    ap.add_argument("--warm", type=int, nargs="*", default=None, help="chaîne warm custom, CROISSANTE (en tokens)")
    ap.add_argument("--no-ollama-warm", action="store_true", help="sauter la chaîne warm ollama")
    ap.add_argument("--leave-server", action="store_true", help="ne pas tuer le llama-server à la fin")
    ap.add_argument("--no-unload", action="store_true", help="ne pas décharger le modèle ollama avant (risque OOM)")
    args = ap.parse_args()
    signal.signal(signal.SIGINT, on_sigint)

    _state["gpu_idx"], _state["gpu_cuda_idx"], _state["gpu_name"] = find_gpu()
    _state["quick"] = args.quick
    log(f"Bench direct — GPU {GPU_NAME} (CUDA dev {_state['gpu_cuda_idx']}, nvidia-smi idx {_state['gpu_idx']}) — VRAM départ {vram_mib()} MiB")
    if not os.path.exists(LLAMA_SERVER):
        sys.exit(f"llama-server introuvable: {LLAMA_SERVER}")
    if not os.path.exists(GGUF):
        sys.exit(f"GGUF introuvable: {GGUF}")
    missing = [d for d in CUDA_LIBS if not os.path.isdir(d)]
    if missing:
        sys.exit(f"dossier(s) CUDA manquant(s) — le server partirait en CPU: {missing}")
    if not os.path.exists(GGML_CUDA_BACKEND):
        sys.exit(f"backend CUDA introuvable — le server partirait en CPU: {GGML_CUDA_BACKEND}")

    if args.quick:
        cold, warm = QUICK_COLD, QUICK_WARM
        ocold, owarm = QUICK_OLLAMA_COLD, QUICK_OLLAMA_WARM
    else:
        cold, warm = args.cold or COLD_SIZES, args.warm or WARM_SIZES
        ocold, owarm = OLLAMA_COLD, OLLAMA_WARM
    if not all(a <= b for a, b in zip(warm, warm[1:])):
        sys.exit("--warm doit être croissant (chaîne de préfixes)")
    # garde-fou anti-400 : avec le ratio au bord haut, l'estimation du prompt ≈
    # C + un overhead chat template (~100-150 tok) — on refuse si ça approche
    # du plafond de contexte (204800 direct / 200192 ollama)
    for c in set(cold) | set(warm):
        if c + 500 > 204800:
            sys.exit(f"cible C={c} trop proche de la limite de contexte (204800) — "
                     "baissez la (ex: 196608)")

    _state["run_start_ts"] = int(time.time())
    with open(OUT_JSONL, "a") as f:
        f.write(json.dumps({"tag": "run_start", "ts": _state["run_start_ts"],
                            "phase": args.phase, "quick": args.quick,
                            "vram_mib": vram_mib(), "server_args": SERVER_ARGS,
                            "cuda_libs": CUDA_LIBS, "ggml_env": GGML_ENV,
                            "model_ollama": MODEL_OLLAMA, "gguf": GGUF}) + "\n")

    t_all = time.time()
    if args.phase in ("direct", "all"):
        if args.no_unload:
            log("  · --no-unload : pas de déchargement ollama (risque OOM sur la V100)")
        phase_direct(cold, warm)
    if args.phase in ("ollama", "all"):
        if _state.get("proc") and _state["proc"].poll() is None:
            stop_server()
            time.sleep(5)
        phase_ollama(ocold, owarm, do_warm=not args.no_ollama_warm)
    if args.phase in ("direct", "all") and not args.leave_server:
        stop_server()

    rows = []
    with open(OUT_JSONL) as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("ts", 0) >= _state["run_start_ts"] and row.get("mode") not in (None, "dummy"):
                rows.append(row)
    write_summary(rows, quick=args.quick)
    log(f"\nFIN en {time.time()-t_all:.0f}s — {len(rows)} runs mesurés")
    log(f"  synthèse : {OUT_SUMMARY}")
    log(f"  brut     : {OUT_JSONL}")
    p = _state.get("proc")
    if args.leave_server and p and p.poll() is None:
        log(f"  llama-server LASSÉ en place (pid {p.pid}, port {PORT_DIRECT}) — arrêt manuel : kill {p.pid}")

if __name__ == "__main__":
    main()
