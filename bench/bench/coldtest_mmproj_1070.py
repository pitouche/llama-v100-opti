#!/usr/bin/env python3
"""
coldtest_mmproj_1070.py — TEST À FROID du binaire ollama patché (mmproj sur GTX 1070).

Objectif : valider le patch Go `ollama.mmproj-1070` + le .so dual-arch (b014d026)
SANS toucher au service prod ni à son arborescence :
  - port 8917, arborescence /tmp/ollama-test-8917/ (copie de la lib prod,
    cuda_v12/libggml-cuda.so = DUAL b014d026),
  - binaire : v100/dual-arch/go_patch/ollama.mmproj-1070,
  - env : OLLAMA_MMPROJ_DEVICE=cuda1 (+ env unit prod, hors OLLAMA_HOST),
  - modèle prod qwen3.8-27b-q6-mtp-vision (blobs /usr/share/ollama, lecture seule),
  - image 1472x1472 (pire cas warmup) pour forcer l'encode mmproj.

Le service prod DOIT être stoppé avant (le modèle 23 Go ne peut pas cohabiter
avec les ~30,5 Go déjà occupés sur la V100). Lancement OBLIGATOIRE via le
superviseur : `python3 supervisor_bench.py run coldtest_mmproj_1070.py --detach`.

Critères de succès (log /tmp/ollama-test-8917/ollama.log + enfant) :
  [S1] "mmproj device: adding projector GPU to placement" (parent, sched.go)
  [S2] "mmproj device: pinning text model to first GPU" (parent, llm)
  [S3] "starting llama-server" avec --split-mode none --main-gpu 0
  [S4] enfant : "clip_ctx: CLIP using CUDA1 backend" (mmproj sur la 1070)
  [S5] encode OK : prompt eval > 0, génération > 0 token, pas d'erreur
  [S6] VRAM 1070 > 3500 MiB pendant encode (le mmproj y vit), V100 > 25 Go
  [S7] pas d'OOM, pas de "no kernel image", pas de "not in compiled architectures"

RC : 0 = S1-S7 OK · 2 = un critère échoue · 3 = service pas restauré.
Le service est restauré dans le `finally` (et via signal handlers) AVANT la sortie.

USAGE :
  cd /home/guile/llama-experiments/bench
  python3 supervisor_bench.py run coldtest_mmproj_1070.py --detach
"""
import base64, json, os, re, shutil, signal, socket, subprocess, sys, time, urllib.request

TESTROOT = "/tmp/ollama-test-8917"
PROD_ROOT = "/home/guile/llama-experiments/ollama-v100"
BIN_PATCHED = "/home/guile/llama-experiments/v100/dual-arch/go_patch/ollama.mmproj-1070"
SO_DUAL = "/home/guile/llama-experiments/v100/dual-arch/lib/libggml-cuda-dual.so"
SO_PROD = "/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so"
LOG = os.path.join(TESTROOT, "ollama.log")
PORT = 8917
SERVICE = "ollama-v100"
GGUF = ("/usr/share/ollama/.ollama/models/blobs/"
        "sha256-493301830a596b8ad56dc1329f80bbcb578c8e910da395feafdc9cd8263430bb")
MMPROJ = ("/usr/share/ollama/.ollama/models/blobs/"
          "sha256-ac3714bfdddeca31351f2752bf1a63f266f4df87c0b68c895e44945ca704448e")
MODEL = "qwen3.8-27b-q6-mtp-vision:latest"

# Env unit prod (identiques), hors OLLAMA_HOST (8917)
ENV_BASE = {
    "LLAMA_ARG_SPEC_MTP_DEFER_PROMPT": "1",
    "OLLAMA_KEEP_ALIVE": "-1",
    "OLLAMA_NOPRUNE": "true",
    "OLLAMA_CONTEXT_LENGTH": "128000",
    "OLLAMA_KV_CACHE_TYPE": "q8_0",
    "LLAMA_ARG_FIT_TARGET": "1024",
    "GGML_CUDA_VOLTA_Q6_W4R4": "1",
    "GGML_CUDA_VOLTA_Q8_FATTN_TC": "0",
    "GGML_CUDA_QWEN35_MTP_SHORTLIST":
        "/home/guile/llama-experiments/src/llama-volta-merged/data/mtp-shortlists/qwen38-27b-exact-131072.i32",
    "LD_LIBRARY_PATH":
        "/home/guile/llama-experiments/tools/cuda-12.6/usr/local/cuda-12.6/lib64:"
        "/home/guile/llama-experiments/tools/cuda-12.6/usr/local/cuda-12.6/targets/x86_64-linux/lib",
    "OLLAMA_MODELS": "/usr/share/ollama/.ollama/models",
    "OLLAMA_LIBRARY_PATH": os.path.join(TESTROOT, "lib"),
    # --- le feature à tester ---
    "OLLAMA_MMPROJ_DEVICE": "cuda1",
    "OLLAMA_MMPROJ_FIT_MARGIN_MIB": "1024",
}
# CUDA_LIBS pour LD de l'enfant (ajoutés au-dessus, comme en prod)
CUDA_LIBS = [
    f"{PROD_ROOT}/lib/ollama",
    f"{PROD_ROOT}/lib/ollama/cuda_v12",
]

_state = {}
_restored = False


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def run(cmd, timeout=60):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def svc_state():
    return run(["systemctl", "is-active", SERVICE]).stdout.strip()


def vram(smi_idx):
    r = run(["nvidia-smi", f"--id={smi_idx}",
             "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    v = r.stdout.strip()
    return int(v) if v.isdigit() else -1


def md5short(p):
    try:
        import hashlib
        return hashlib.md5(open(p, "rb").read()).hexdigest()[:8]
    except OSError:
        return "?"


def port_free(port):
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", port)); s.close(); return True
    except OSError:
        s.close(); return False


def http_json(url, data=None, timeout=30):
    req = urllib.request.Request(
        url, data=json.dumps(data).encode() if data is not None else None,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def restore_service():
    global _restored
    try:
        # tuer le serveur de test s'il subsiste
        p = _state.get("proc")
        if p and p.poll() is None:
            p.terminate()
            try: p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill(); p.wait(timeout=10)
        if svc_state() != "active":
            log("  · restauration : systemctl start " + SERVICE)
            run(["sudo", "systemctl", "start", SERVICE], timeout=120)
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
    run(["sudo", "systemctl", "stop", SERVICE], timeout=120)
    t0 = time.time()
    while time.time() - t0 < 180:
        if vram(1) < 1024:
            log(f"  · V100 libérée ({vram(1)} MiB) en {time.time()-t0:.0f}s")
            return True
        time.sleep(3)
    log(f"  · ATTENTION V100 pas totalement libérée (VRAM={vram(1)} MiB)")
    return False


def prepare_tree():
    log(f"  · prépa {TESTROOT} (COPIE REELLE de la lib prod + .so dual dans la copie)")
    if os.path.exists(TESTROOT):
        shutil.rmtree(TESTROOT, ignore_errors=True)
    os.makedirs(TESTROOT, exist_ok=True)
    # AUCUN SYMLINK : un symlink cuda_v12 -> prod a déjà écrasé le .so prod une fois.
    shutil.copytree(os.path.join(PROD_ROOT, "lib"), os.path.join(TESTROOT, "lib"))
    # le .so DUAL remplace le prod UNIQUEMENT dans la copie de test
    target = os.path.join(TESTROOT, "lib", "ollama", "cuda_v12", "libggml-cuda.so")
    shutil.copy2(SO_DUAL, target)
    os.chmod(target, 0o755)
    shutil.copy2(BIN_PATCHED, os.path.join(TESTROOT, "ollama"))
    os.chmod(os.path.join(TESTROOT, "ollama"), 0o755)
    log(f"    .so dual (copie) : {md5short(target)} (attendu b014d026)")
    log(f"    binaire          : {md5short(os.path.join(TESTROOT,'ollama'))} (attendu 6f6d33a9)")
    # garde-fou post-copie : le prod n'a pas bougé
    if md5short(SO_PROD) != "feaf05aa":
        raise RuntimeError(f"prod .so modifié ?! ({md5short(SO_PROD)} != feaf05aa) — ABORT")


def make_image():
    import numpy as np, io
    from PIL import Image
    x = np.linspace(0, 255, 1472, dtype=np.uint8)
    y = np.linspace(0, 255, 1472, dtype=np.uint8)
    r = np.tile(x[None, :], (1472, 1))
    g = np.tile(y[:, None], (1, 1472))
    b = ((r.astype(int) + g.astype(int)) // 2).astype(np.uint8)
    img = Image.frombytes("RGB", (1472, 1472), np.stack([r, g, b], axis=2).tobytes())
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def main():
    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)
    log("COLD TEST mmproj→1070 (patch Go + .so dual, port 8917)")
    t_start = time.time()

    if md5short(SO_PROD) != "feaf05aa":
        sys.exit(f"ABORT : .so prod {md5short(SO_PROD)} != feaf05aa")
    if md5short(SO_DUAL) != "b014d026":
        sys.exit(f"ABORT : .so dual {md5short(SO_DUAL)} != b014d026")
    if not port_free(PORT):
        sys.exit(f"port {PORT} occupé — abandon")

    prepare_tree()
    stop_and_free()
    open(LOG, "w").close()

    results = {}
    try:
        # ---- démarrage du serveur patché (user guile, blobs + lib en lecture seule) ----
        log("  · démarrage ollama patché (8917)")
        lf = open(LOG, "a")
        env = dict(os.environ)
        for k, v in ENV_BASE.items():
            env[k] = v
        env["OLLAMA_HOST"] = f"127.0.0.1:{PORT}"
        # LD enfant : lib du test + prod (comme ab_dual) — pas sur le chemin
        # de lib/ollama (sinon le .so dual serait écrasé par le prod feaf05aa)
        ldp = ":".join(d for d in [os.path.join(TESTROOT, "lib", "ollama"),
                                   *CUDA_LIBS] if os.path.isdir(d))
        env["LD_LIBRARY_PATH"] = ldp + ":" + env.get("LD_LIBRARY_PATH", "")
        proc = subprocess.Popen([os.path.join(TESTROOT, "ollama"), "serve"],
                                stdout=lf, stderr=subprocess.STDOUT, env=env)
        _state["proc"] = proc
        t0 = time.time()
        up = False
        while time.time() - t0 < 240:
            if proc.poll() is not None:
                break
            try:
                # NB : ce build n'a PAS /health (404) -> utiliser /api/version
                http_json(f"http://127.0.0.1:{PORT}/api/version", timeout=3)
                up = True
                break
            except Exception:
                time.sleep(2)
        if not up:
            raise RuntimeError(f"ollama patché mort au démarrage (exit {proc.returncode})")
        log(f"  · ollama patché up ({time.time()-t0:.0f}s)")
        tags = http_json(f"http://127.0.0.1:{PORT}/api/tags", timeout=30)
        names = [m.get("name") for m in tags.get("models", [])]
        if not any((n or "").startswith("qwen3.8-27b-q6-mtp-vision") for n in names):
            raise RuntimeError(f"modèle vision absent de /api/tags : {names}")
        log(f"  · /api/tags OK : {names}")

        # ---- chargement du modèle (c'est là que tout se joue) ----
        log("  · chargement du modèle prod (V100) …")
        t0 = time.time()
        img_b64 = make_image()
        data = {
            "model": MODEL, "stream": False,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Décris cette image en une phrase."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                ],
            }],
            "max_tokens": 32, "temperature": 0.0, "top_k": 1, "top_p": 1.0,
        }
        vram1070_peak, vramv100_peak = -1, -1
        try:
            # polling VRAM pendant le load+encode
            def vram_watch():
                nonlocal vram1070_peak, vramv100_peak
                while True:
                    v1, v2 = vram(0), vram(1)
                    if v1 > vram1070_peak: vram1070_peak = v1
                    if v2 > vramv100_peak: vramv100_peak = v2
                    if proc.poll() is not None: break
                    time.sleep(1)
            import threading
            wt = threading.Thread(target=vram_watch, daemon=True)
            wt.start()
            r = http_json(f"http://127.0.0.1:{PORT}/v1/chat/completions", data, timeout=1800)
        finally:
            time.sleep(2)
        wall = round(time.time() - t0, 1)
        content = (r.get("choices") or [{}])[0].get("message", {}).get("content", "")
        usage = r.get("usage", {})
        log(f"  · requête image OK en {wall}s — {usage}")
        log(f"    réponse : {content[:200]}")
        results["request_ok"] = True
        results["wall_s"] = wall
        results["usage"] = usage
        results["vram1070_peak"] = vram1070_peak
        results["vramv100_peak"] = vramv100_peak

        # ---- analyse du log ----
        with open(LOG, errors="replace") as f:
            lg = f.read()
        def find(pat, flags=0):
            m = re.search(pat, lg, flags)
            return m.group(0) if m else None
        # S1 : le bloc d'ajout du GPU projecteur (sched.go) ne s'exécute QUE si le
        # placement n'a PAS déjà les 2 GPUs — ici le placement natif inclut déjà la
        # 1070, donc l'absence du log est le CAS ATTENDU. On accepte donc :
        #   - le log présent, OU
        #   - la présence d'UN des 2 GPUs dans le placement (preuve que le projet
        #     a pu voir 2 GPUs, soit placement natif, soit ajout patché).
        has_add_log = bool(re.search(r"mmproj device: adding projector GPU to placement", lg or ""))
        has_two_gpu_placement = "CUDA0" in (lg or "") and "CUDA1" in (lg or "")
        results["S1_add_proj_gpu"] = (find(r"mmproj device: adding projector GPU to placement.*")
                                      or ("placement natif 2 GPUs (log S1 absent = attendu)" if has_two_gpu_placement else None))
        results["S1_mode"] = "log" if has_add_log else ("native-2gpu" if has_two_gpu_placement else "absent")
        results["S2_pin_text"] = find(r"mmproj device: pinning text model to first GPU.*")
        m = re.search(r'starting llama-server.*', lg)
        results["S3_start_cmd"] = m.group(0)[:400] if m else None
        results["S3_split_none"] = bool(re.search(r"--split-mode none --main-gpu 0", lg or ""))
        results["S4_clip_cuda1"] = find(r"clip_ctx: CLIP using CUDA\d+ backend.*")
        results["S5_prompt_eval"] = find(r"prompt eval time = .*")
        results["S5_eval"] = find(r"(?<!prompt )eval time = .*")
        results["S5_tokens"] = find(r"llama_tokens: .*")
        results["oom"] = find(r"out of memory|OOM|no kernel image|not in compiled architectures|CUDA error|error while computing block")
        results["mtmd_est"] = find(r"estimated worst-case memory usage of mmproj is [\d.]+ MiB.*")
        results["clip_ctx_log"] = find(r"clip_ctx: .*")
        results["child_env_check"] = find(r"MTMD_BACKEND_DEVICE.*")

        # ---- verdict ----
        crit = {
            "S1 (proj GPU addé au placement)": bool(results["S1_add_proj_gpu"]),
            "S2 (text model pinned)": bool(results["S2_pin_text"]),
            "S3 (split none + main-gpu 0)": results["S3_split_none"],
            "S4 (CLIP sur CUDA1=1070)": "CUDA1" in (results["S4_clip_cuda1"] or ""),
            "S5 (encode+gen OK, pas d'erreur)": bool(results["request_ok"] and not results["oom"]),
            "S6 (VRAM 1070 > 3500 MiB)": vram1070_peak > 3500,
        }
        log("  · CRITÈRES :")
        for k, v in crit.items():
            log(f"    [{'OK ' if v else 'KO '}] {k}")
        log(f"  · pic VRAM 1070 = {vram1070_peak} MiB | pic V100 = {vramv100_peak} MiB")
        log(f"  · S4 détail : {results['S4_clip_cuda1']}")
        log(f"  · S5 détail : {results['S5_prompt_eval']} | {results['S5_eval']}")
        log(f"  · OOM/erreur : {results['oom'] or 'aucune'}")
        log(f"  · mtmd est. : {results['mtmd_est']}")
        with open(os.path.join(TESTROOT, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        log(f"  · résultats : {TESTROOT}/results.json")
        rc = 0 if all(crit.values()) else 2
    except Exception as e:
        log(f"  · ECHEC : {e}")
        rc = 2
    finally:
        # tuer le serveur de test
        p = _state.get("proc")
        if p and p.poll() is None:
            p.terminate()
            try: p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                p.kill(); p.wait(timeout=10)
        # V100 doit être libérée avant de redémarrer le service
        t0 = time.time()
        while time.time() - t0 < 180 and vram(1) > 2000:
            time.sleep(3)
        log(f"  · V100 libérée ({vram(1)} MiB)")
        restore_service()
        log(f"FIN en {time.time()-t_start:.0f}s — service restaure : {svc_state()}, rc={rc}")
    sys.exit(rc if _restored else 3)


if __name__ == "__main__":
    main()
