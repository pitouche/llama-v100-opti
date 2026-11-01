#!/usr/bin/env python3
"""
ROLLOUT FA16 — push to prod du lever prefill compact (31/10).

Ce que ça fait (tous steps idempotents, réversible via --rollback) :
  1. Vérif preflight : .so prod = 8035caf4 (sinon ABORT — état inconnu),
     .so dual = d4954502, GPU par UUID, service actif.
  2. Backup .so prod → /home/guile/llama-experiments/rollout_fa16/backups/
     (timestamp + md5 vérifié).
  3. cp .so dual → ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so
  4. Drop-in systemd fa16-compact.conf : GGML_CUDA_VOLTA_FA_COMPACT_TMAX=8192
     (seule nouvelle variable ; pas de daemon-reload si le fichier n'a pas changé).
  5. daemon-reload + restart ollama-v100 (si et seulement si .so ou drop-in changés).
  6. Vérifs post : active, logs propres (pas de CUDA/error), VRAM chargée,
     env TMAX visible dans le process llama-server.
  7. SMOKE MTP-ON (la seule vérif réelle restante, cf. DECISION_FA16_20261031 §5) :
     prompt ~60k → prefill t/s attendu ≈ 800-900 (baseline ~740, cible +19 %) ;
     génération 128 tok → decode attendu ≈ 22-23 t/s (pas de régression) ;
     réponse texte non vide.
  8. Résumé + état final.

ROLLBACK : python3 rollout_fa16.py --rollback
  → restaure le .so backup le plus récent + supprime le drop-in + restart + smoke.

USAGE :
  cd /home/guile/llama-experiments/bench
  python3 supervisor_bench.py run rollout_fa16.py --detach      # forward
  python3 supervisor_bench.py run rollout_fa16.py --rollback --detach
"""
import hashlib, json, os, re, shutil, signal, subprocess, sys, time
import urllib.request
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpu_uuid_select import v100_smi_index

D = os.path.dirname(os.path.abspath(__file__))
TS = datetime.now().strftime("%Y%m%d-%H%M%S")
OUT_LOG = os.path.join(D, f"rollout_fa16_{TS}.log")
OUT_JSONL = os.path.join(D, f"rollout_fa16_{TS}.jsonl")

SO_PROD = "/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so"
SO_DUAL = "/home/guile/llama-experiments/v100/dual-arch/lib/libggml-cuda-dual.so"
MD5_PROD_ATTENDU = "8035caf4"
MD5_DUAL_ATTENDU = "d4954502"
BACKUP_DIR = "/home/guile/llama-experiments/rollout_fa16/backups"
DROPIN_DIR = "/etc/systemd/system/ollama-v100.service.d"
DROPIN = os.path.join(DROPIN_DIR, "fa16-compact.conf")
DROPIN_CONTENT = (
    "[Service]\n"
    "# FA16 31/10 : lever prefill compact (kernel volta_compact 48 KB, 2 blocks/SM).\n"
    "# .so dual-arch d4954502 (rollback : rollout_fa16.py --rollback, cf. DECISION_FA16_20261031.md)\n"
    "Environment=\"GGML_CUDA_VOLTA_FA_COMPACT_TMAX=8192\"\n"
)
SERVICE = "ollama-v100"
PORT = 11434
ROLLBACK = "--rollback" in sys.argv
GPU_SMI_IDX = v100_smi_index()

_state = {"restored": False, "steps": []}


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(OUT_LOG, "a") as f:
        f.write(line + "\n")


def run(cmd, timeout=120, sudo=False, input_text=None):
    c = (["sudo"] + cmd) if sudo else cmd
    return subprocess.run(c, capture_output=True, text=True, timeout=timeout,
                          input=input_text)


def md5_short(p):
    h = hashlib.md5(open(p, "rb").read()).hexdigest()
    return h[:8], h


def svc_state():
    return run(["systemctl", "is-active", SERVICE]).stdout.strip()


def vram_mib():
    r = run(["nvidia-smi", f"--id={GPU_SMI_IDX}",
             "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    v = r.stdout.strip()
    return int(v) if v.isdigit() else -1


def stop_and_free():
    if svc_state() == "active":
        log(f"  · stop {SERVICE}")
        run(["systemctl", "stop", SERVICE], timeout=120, sudo=True)
    t0 = time.time()
    while time.time() - t0 < 180:
        if vram_mib() < 1024:
            log(f"  · V100 libérée ({vram_mib()} MiB) en {time.time()-t0:.0f}s")
            return
        time.sleep(3)
    log(f"  · ATTENTION V100 pas libérée (VRAM={vram_mib()} MiB) — on continue prudemment")


def restore_service():
    global _state
    try:
        if svc_state() != "active":
            log(f"  · restauration : systemctl start {SERVICE}")
            run(["systemctl", "start", SERVICE], timeout=180, sudo=True)
        for _ in range(90):
            if svc_state() == "active":
                break
            time.sleep(2)
        _state["restored"] = svc_state() == "active"
        log(f"  · service {SERVICE} = {svc_state()}")
    except Exception as e:
        log(f"  · ERREUR restauration : {e}")
        _state["restored"] = False


def on_sig(sig, frm):
    restore_service()
    sys.exit(128 + sig)


def step(name, ok, detail=""):
    _state["steps"].append({"step": name, "ok": ok, "detail": detail})
    log(f"  [{'OK' if ok else 'KO'}] {name}{' — ' + detail if detail else ''}")
    return ok


def preflight(rollback=False):
    log("=== PREFLIGHT ===")
    a, fa = md5_short(SO_PROD)
    b, fb = md5_short(SO_DUAL)
    # prod peut être 8035caf4 (pristine) OU d4954502 (déjà copié — déploiement partiel idempotent)
    step("md5 prod " + a + " ∈ {8035caf4, d4954502}", a in (MD5_PROD_ATTENDU, MD5_DUAL_ATTENDU))
    step("md5 dual " + b, b == MD5_DUAL_ATTENDU, "attendu d4954502")
    if a not in (MD5_PROD_ATTENDU, MD5_DUAL_ATTENDU) and not rollback:
        log("  · ABORT : .so prod inconnu — état à identifier avant d'avancer.")
        sys.exit(2)
    if rollback:
        backups = find_pristine_backups()
        step("backup pristine pour rollback", len(backups) > 0, backups[-1] if backups else "AUCUN")
        if not backups:
            sys.exit(2)
    if svc_state() != "active":
        log(f"  · service {svc_state()} — on le remet en marche d'abord")
        restore_service()
        step("service actif", _state["restored"])


def smoke_test():
    """MTP-ON réel (service prod) : prefill ~60k + decode 128 tok."""
    log("=== SMOKE MTP-ON (service prod) ===")
    base = f"http://127.0.0.1:{PORT}"
    models = json.load(urllib.request.urlopen(base + "/api/tags", timeout=30))
    name = next((m["name"] for m in models.get("models", [])
                 if m["name"].startswith("qwen3.8-27b-q6-mtp-vision")), None)
    if not name:
        step("smoke modèle", False, "qwen3.8-27b-q6-mtp-vision introuvable")
        return
    # ~60k tokens : le FILL du bench e2e fait ~4 tok/word, on recycle le ratio 5/4
    fill = "La tour de contrôle surveille le passage des avions au crépuscule pendant que le chef de piste organise les départs de la soirée. "
    n_words = int(60000 * 4 / 5)
    words = fill.split() * max(1, n_words // fill.count(" ") + 1)
    # sel TS en tête → invalide le prefix cache : le smoke préfill TOUJOURS les ~60k
    prompt = f"[run {TS}] " + " ".join(words[:n_words]) + "\nRéponds en une phrase."
    data = {"model": name, "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"num_predict": 128, "temperature": 0.0, "top_k": 1}}
    t0 = time.time()
    r = json.load(urllib.request.urlopen(
        urllib.request.Request(base + "/api/chat",
                               data=json.dumps(data).encode(),
                               headers={"Content-Type": "application/json"}),
        timeout=3600))
    wall = time.time() - t0
    pt = r.get("prompt_eval_count") or 0
    pc = r.get("prompt_eval_cached_count") or 0
    pt_s = (r.get("prompt_eval_duration") or 0) / 1e9   # ns -> s
    ct = r.get("eval_count") or r.get("completion_tokens", 0)
    ct_s = (r.get("eval_duration") or 0) / 1e9          # ns -> s
    # NB : prompt_eval_duration couvre les tokens NON cachés (le reste = prefix cache,
    # KEEP_ALIVE=-1) — un 2e smoke de suite ne préfill que la queue.
    prefill = round((pt - pc) / pt_s, 1) if pt_s else 0
    decode = round(ct / ct_s, 2) if ct_s else 0
    msg = r.get("message", {})
    content = (msg.get("content") or "") + (msg.get("thinking") or "")  # modèle thinking
    if isinstance(content, list):
        content = "".join(x.get("content", "") if isinstance(x, dict) else str(x) for x in content)
    row = {"model": name, "prompt_tokens": pt, "prompt_cached": pc,
           "prefill_tps": prefill, "gen_tokens": ct, "decode_tps": decode,
           "wall_s": round(wall, 1), "done_reason": r.get("done_reason"),
           "content_head": str(content)[:120]}
    with open(OUT_JSONL, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log(f"  · prefill {prefill} t/s ({pt} tok, {pc} cachés) | decode {decode} t/s ({ct} tok) | wall {row['wall_s']} s")
    log(f"  · réponse : {str(content)[:100]!r}")
    step("smoke prefill (cible ~700-950 t/s ; baseline prod ~785 @65k)", prefill > 650 and pt > 10000, f"{prefill} t/s")
    step("smoke decode MTP-ON (cible ≥ 40 t/s)", decode >= 40, f"{decode} t/s")
    step("smoke réponse non vide (content ou thinking)", len(str(content).strip()) > 0)


def verify_server_env():
    r = run(["pgrep", "-f", "llama-server --model"], timeout=30)
    for pid in r.stdout.split():
        try:
            env = run(["sh", "-c", f"tr '\\0' '\\n' < /proc/{pid}/environ"],
                      timeout=30, sudo=True).stdout
        except Exception:
            continue
        if "GGML_CUDA_VOLTA_FA_COMPACT_TMAX" in env:
            v = re.search(r"GGML_CUDA_VOLTA_FA_COMPACT_TMAX=(\S+)", env)
            step("env TMAX visible dans llama-server", bool(v) and v.group(1) == "8192",
                 f"pid {pid} TMAX={v.group(1) if v else '?'}")
            return
    step("env TMAX visible dans llama-server", False, "variable absente du process (sudo sh -c /proc environ)")


def check_logs():
    r = run(["journalctl", "-u", SERVICE, "-n", "200", "--no-pager"], timeout=60, sudo=True)
    bad = [l for l in r.stdout.splitlines()
           if re.search(r"cuda error|no device code|fatal|abort|unspecified launch|illegal memory", l, re.I)]
    step("logs service propres (pas de CUDA/fatal)", not bad,
         "; ".join(b[:120] for b in bad[:3]) if bad else "ok")
    return bad


def find_pristine_backups():
    out = []
    if os.path.isdir(BACKUP_DIR):
        for f in sorted(os.listdir(BACKUP_DIR)):
            if f.startswith("libggml-cuda.so.prod."):
                m, _ = md5_short(os.path.join(BACKUP_DIR, f))
                if m == MD5_PROD_ATTENDU:
                    out.append(f)
    return out


def forward():
    preflight()
    os.makedirs(BACKUP_DIR, exist_ok=True)
    stop_and_free()
    # 1. backup
    a_now, _ = md5_short(SO_PROD)
    if a_now == MD5_PROD_ATTENDU:
        bak = os.path.join(BACKUP_DIR, f"libggml-cuda.so.prod.{a_now}.{TS}")
        shutil.copy2(SO_PROD, bak)
        mb, _ = md5_short(bak)
        step("backup .so prod (pristine)", mb == MD5_PROD_ATTENDU, bak)
    elif a_now == MD5_DUAL_ATTENDU:
        pb = find_pristine_backups()
        step("backup .so prod (pristine)", len(pb) > 0,
             "déjà copié en dual — backup existant : " + (pb[-1] if pb else "ABSENT"))
        if not pb:
            log("  · ABORT : pas de backup 8035caf4 — impossible de garantir le rollback.")
            restore_service()
            sys.exit(2)
    # 2. copie
    changed = md5_short(SO_PROD)[1] != md5_short(SO_DUAL)[1]
    if changed:
        log("  · copie .so dual → prod")
        tmp = SO_PROD + ".fa16.tmp"
        shutil.copy2(SO_DUAL, tmp)
        os.chmod(tmp, 0o644)
        os.chown(tmp, 1000, 1000)
        os.replace(tmp, SO_PROD)
        a, _ = md5_short(SO_PROD)
        step(".so prod = dual", a == MD5_DUAL_ATTENDU, f"md5 {a}")
    else:
        log("  · .so prod déjà = dual, copie inutile")
    # 3. drop-in (écriture root : /etc/systemd → sudo tee)
    dropin_written = False
    if os.path.exists(DROPIN) and open(DROPIN).read() == DROPIN_CONTENT:
        log("  · drop-in déjà en place")
    else:
        log("  · écrit drop-in " + DROPIN + " (sudo)")
        r = run(["tee", DROPIN], input_text=DROPIN_CONTENT, sudo=True)
        if r.returncode != 0 or not os.path.exists(DROPIN) or open(DROPIN).read() != DROPIN_CONTENT:
            raise RuntimeError(f"écriture drop-in échouée : {r.stderr.strip()[:200]}")
        dropin_written = True
    step("drop-in TMAX=8192", open(DROPIN).read() == DROPIN_CONTENT, "(sudo tee)")
    # 4. reload + restart (uniquement si quelque chose a changé)
    if changed or dropin_written:
        log("  · daemon-reload + restart")
        run(["systemctl", "daemon-reload"], sudo=True)
        run(["systemctl", "restart", SERVICE], timeout=300, sudo=True)
    else:
        log("  · rien de changé : pas de restart")
    # 5. vérifs post
    for _ in range(60):
        if svc_state() == "active":
            break
        time.sleep(2)
    step("service active", svc_state() == "active", svc_state())
    for _ in range(90):
        if vram_mib() > 8000:
            break
        time.sleep(2)
    step("VRAM chargée (runner up)", vram_mib() > 8000, f"{vram_mib()} MiB")
    time.sleep(10)
    check_logs()
    verify_server_env()
    smoke_test()


def rollback():
    preflight(rollback=True)
    backups = find_pristine_backups()
    step("backup pristine 8035caf4 présent", len(backups) > 0, backups[-1] if backups else "AUCUN")
    if not backups:
        sys.exit(2)
    bak = os.path.join(BACKUP_DIR, backups[-1])
    a, _ = md5_short(bak)
    step("md5 backup", a == MD5_PROD_ATTENDU, bak)
    stop_and_free()
    tmp = SO_PROD + ".rb.tmp"
    shutil.copy2(bak, tmp)
    os.chmod(tmp, 0o644)
    os.chown(tmp, 1000, 1000)
    os.replace(tmp, SO_PROD)
    step(".so prod restauré", md5_short(SO_PROD)[0] == MD5_PROD_ATTENDU)
    if os.path.exists(DROPIN):
        r = run(["rm", "-f", DROPIN], sudo=True)
        step("drop-in supprimé", r.returncode == 0 and not os.path.exists(DROPIN),
             "(sudo rm)" + (f" — {r.stderr.strip()[:100]}" if r.returncode else ""))
    run(["systemctl", "daemon-reload"], sudo=True)
    run(["systemctl", "restart", SERVICE], timeout=300, sudo=True)
    for _ in range(60):
        if svc_state() == "active":
            break
        time.sleep(2)
    step("service active", svc_state() == "active")
    for _ in range(90):
        if vram_mib() > 8000:
            break
        time.sleep(2)
    step("VRAM chargée", vram_mib() > 8000, f"{vram_mib()} MiB")
    time.sleep(10)
    check_logs()


def main():
    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)
    open(OUT_LOG, "a").close()
    log(f"ROLLOUT FA16 {'ROLLBACK' if ROLLBACK else 'FORWARD'} — {TS}")
    try:
        (rollback if ROLLBACK else forward)()
    except SystemExit:
        raise
    except Exception as e:
        log(f"  · EXCEPTION : {e!r}")
        _state["steps"].append({"step": "exception", "ok": False, "detail": repr(e)[:200]})
    finally:
        restore_service()
    oks = all(s["ok"] for s in _state["steps"])
    with open(OUT_JSONL, "a") as f:
        f.write(json.dumps({"mode": "rollback" if ROLLBACK else "forward",
                            "steps": _state["steps"],
                            "service_restored": _state["restored"]},
                           ensure_ascii=False) + "\n")
    log(f"FIN — {'OK' if oks else 'ÉCHEC'} — service restauré = {_state['restored']} | log {OUT_LOG}")
    sys.exit(0 if (oks and _state["restored"]) else 3)


if __name__ == "__main__":
    main()
