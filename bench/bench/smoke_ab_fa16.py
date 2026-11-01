#!/usr/bin/env python3
"""
SMOKE A/B FA16 — le gain prefill tient-il en stack PROD MTP-ON ? (31/10)

Le rollout (rollout_fa16.py) est déjà fait : .so dual d4954502 + drop-in TMAX=8192.
Ce script fait l'unique vérif restante de DECISION_FA16_20261031 §5 :
comparer, sur le SERVICE PROD réel (MTP ON, SPEC_MTP_DEFER_PROMPT=1, KV q8_0,
ctx 128k), un prompt frais ~60k entre :
  BASE : .so 8035caf4 (pristine, backup) + pas de TMAX   → 32x2 standard
  NEW  : .so d4954502 (dual) + TMAX=8192                 → kernel compact

Séquence : BASE → NEW → état final restauré = NEW (le déploiement reste en place).
Chaque leg : stop service → config → start → wait runner → smoke (prompt salé,
~60k frais, 128 tok) → métriques. Le bench restaure le service avant de sortir.

USAGE :
  python3 supervisor_bench.py run smoke_ab_fa16.py --detach
"""
import hashlib, json, os, re, shutil, signal, subprocess, sys, time
import urllib.request
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpu_uuid_select import v100_smi_index

D = os.path.dirname(os.path.abspath(__file__))
TS = datetime.now().strftime("%Y%m%d-%H%M%S")
OUT_LOG = os.path.join(D, f"smoke_ab_fa16_{TS}.log")
OUT_JSONL = os.path.join(D, f"smoke_ab_fa16_{TS}.jsonl")
OUT_SUMMARY = os.path.join(D, f"smoke_ab_fa16_{TS}_summary.md")

SO_PROD = "/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so"
SO_DUAL = "/home/guile/llama-experiments/v100/dual-arch/lib/libggml-cuda-dual.so"
BACKUP_DIR = "/home/guile/llama-experiments/rollout_fa16/backups"
DROPIN = "/etc/systemd/system/ollama-v100.service.d/fa16-compact.conf"
DROPIN_CONTENT = (
    "[Service]\n"
    "# FA16 31/10 : lever prefill compact (kernel volta_compact 48 KB, 2 blocks/SM).\n"
    "# .so dual-arch d4954502 (rollback : rollout_fa16.py --rollback, cf. DECISION_FA16_20261031.md)\n"
    "Environment=\"GGML_CUDA_VOLTA_FA_COMPACT_TMAX=8192\"\n"
)
SERVICE = "ollama-v100"
PORT = 11434
MD5_BASE, MD5_NEW = "8035caf4", "d4954502"
GPU_SMI_IDX = v100_smi_index()

MODEL = "qwen3.8-27b-q6-mtp-vision:latest"
FILL = ("La tour de contrôle surveille le passage des avions au crépuscule "
        "pendant que le chef de piste organise les départs de la soirée. ")

_state = {"final_state": "inconnu"}
_restored = False


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
    return hashlib.md5(open(p, "rb").read()).hexdigest()[:8]


def svc_state():
    return run(["systemctl", "is-active", SERVICE]).stdout.strip()


def vram_mib():
    r = run(["nvidia-smi", f"--id={GPU_SMI_IDX}",
             "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    v = r.stdout.strip()
    return int(v) if v.isdigit() else -1


def stop_and_free():
    if svc_state() == "active":
        log("  · stop service")
        run(["systemctl", "stop", SERVICE], timeout=120, sudo=True)
    t0 = time.time()
    while time.time() - t0 < 180 and vram_mib() > 1024:
        time.sleep(3)
    log(f"  · V100 libérée ({vram_mib()} MiB) en {time.time()-t0:.0f}s")


def wait_ready():
    for _ in range(90):
        if svc_state() == "active":
            break
        time.sleep(2)
    if svc_state() != "active":
        raise RuntimeError(f"service {svc_state()} après start")
    for _ in range(180):
        if vram_mib() > 8000:
            return
        time.sleep(2)
    raise RuntimeError(f"runner pas chargé (VRAM={vram_mib()} MiB)")


def set_config(leg):
    """leg 'BASE' → .so 8035caf4 + drop-in absent ; leg 'NEW' → dual + drop-in."""
    if leg == "BASE":
        src = os.path.join(BACKUP_DIR,
                           [f for f in sorted(os.listdir(BACKUP_DIR))
                            if md5_short(os.path.join(BACKUP_DIR, f)) == MD5_BASE][-1])
        exp = MD5_BASE
    else:
        src, exp = SO_DUAL, MD5_NEW
    tmp = SO_PROD + ".smoke.tmp"
    shutil.copy2(src, tmp)
    os.chmod(tmp, 0o644)
    os.chown(tmp, 1000, 1000)
    os.replace(tmp, SO_PROD)
    if leg == "NEW":
        r = run(["tee", DROPIN], input_text=DROPIN_CONTENT, sudo=True)
        if r.returncode != 0 or open(DROPIN).read() != DROPIN_CONTENT:
            raise RuntimeError("drop-in NEW échoué")
    else:
        run(["rm", "-f", DROPIN], sudo=True)
    run(["systemctl", "daemon-reload"], sudo=True)
    if md5_short(SO_PROD) != exp:
        raise RuntimeError(f"md5 {md5_short(SO_PROD)} != {exp} (leg {leg})")


def smoke(leg):
    n_words = int(60000 * 4 / 5)
    words = FILL.split() * max(1, n_words // FILL.count(" ") + 1)
    # sel TS+leg en tête → prompt 100 % frais (pas de prefix cache)
    prompt = f"[smoke {TS} {leg}] " + " ".join(words[:n_words]) + "\nRéponds en une phrase."
    data = {"model": MODEL, "stream": False,
            "options": {"num_predict": 128, "temperature": 0.0, "top_k": 1},
            "messages": [{"role": "user", "content": prompt}]}
    t0 = time.time()
    r = json.load(urllib.request.urlopen(
        urllib.request.Request(f"http://127.0.0.1:{PORT}/api/chat",
                               data=json.dumps(data).encode(),
                               headers={"Content-Type": "application/json"}),
        timeout=3600))
    wall = time.time() - t0
    pt = r.get("prompt_eval_count") or 0
    pc = r.get("prompt_eval_cached_count") or 0
    pt_s = (r.get("prompt_eval_duration") or 0) / 1e9
    ct = r.get("eval_count") or 0
    ct_s = (r.get("eval_duration") or 0) / 1e9
    prefill = round((pt - pc) / pt_s, 1) if pt_s else 0
    decode = round(ct / ct_s, 2) if ct_s else 0
    msg = r.get("message", {})
    content = str(msg.get("content") or "") + str(msg.get("thinking") or "")
    row = {"leg": leg, "so": md5_short(SO_PROD), "tmax": "8192" if leg == "NEW" else "absent",
           "prompt_tokens": pt, "prompt_cached": pc, "prefill_tps": prefill,
           "gen_tokens": ct, "decode_tps": decode, "wall_s": round(wall, 1),
           "done_reason": r.get("done_reason"),
           "content_head": content[:120]}
    with open(OUT_JSONL, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log(f"  · {leg}: prefill {prefill} t/s ({pt} tok, {pc} cachés) | "
        f"decode {decode} t/s ({ct} tok) | wall {wall:.1f}s")
    log(f"    réponse : {content[:100]!r}")
    return row


def check_logs(tag):
    r = run(["journalctl", "-u", SERVICE, "-n", "150", "--no-pager"], timeout=60, sudo=True)
    bad = [l for l in r.stdout.splitlines()
           if re.search(r"cuda error|no device code|fatal|abort|unspecified launch|illegal memory", l, re.I)]
    log(f"  · logs {tag}: {'OK' if not bad else 'PROBLÈME : ' + bad[0][:120]}")
    return bad


def restore_final():
    """État final voulu = NEW (déploiement en place) + service actif."""
    global _restored
    try:
        need = False
        if md5_short(SO_PROD) != MD5_NEW:
            log("  · restauration .so NEW (dual)")
            tmp = SO_PROD + ".fin.tmp"
            shutil.copy2(SO_DUAL, tmp)
            os.chmod(tmp, 0o644)
            os.chown(tmp, 1000, 1000)
            os.replace(tmp, SO_PROD)
            need = True
        if not os.path.exists(DROPIN) or open(DROPIN).read() != DROPIN_CONTENT:
            log("  · restauration drop-in TMAX")
            r = run(["tee", DROPIN], input_text=DROPIN_CONTENT, sudo=True)
            if r.returncode != 0:
                raise RuntimeError("drop-in restauration échouée")
            need = True
        if need or svc_state() != "active":
            run(["systemctl", "daemon-reload"], sudo=True)
            if svc_state() == "active":
                run(["systemctl", "restart", SERVICE], timeout=300, sudo=True)
            else:
                run(["systemctl", "start", SERVICE], timeout=300, sudo=True)
        for _ in range(90):
            if svc_state() == "active":
                break
            time.sleep(2)
        _restored = svc_state() == "active"
        log(f"  · état final : service = {svc_state()}, .so = {md5_short(SO_PROD)}, "
            f"drop-in = {'oui' if os.path.exists(DROPIN) else 'NON'}")
    except Exception as e:
        log(f"  · ERREUR restauration finale : {e!r}")
        _restored = False


def on_sig(sig, frm):
    restore_final()
    sys.exit(128 + sig)


def main():
    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)
    open(OUT_LOG, "a").close()
    log(f"SMOKE A/B FA16 (stack prod MTP-ON) — {TS}")
    rows = {}
    try:
        for leg in ("BASE", "NEW"):
            log(f"=== leg {leg} ===")
            stop_and_free()
            set_config(leg)
            run(["systemctl", "start", SERVICE], timeout=300, sudo=True)
            wait_ready()
            time.sleep(8)
            check_logs(leg)
            rows[leg] = smoke(leg)
            with open(OUT_JSONL, "a") as f:
                f.write(json.dumps({"leg": leg, "checkpoint": "ok"}) + "\n")
        # comparatif
        b, n = rows["BASE"], rows["NEW"]
        dp = (n["prefill_tps"] - b["prefill_tps"]) / b["prefill_tps"] * 100
        dd = (n["decode_tps"] - b["decode_tps"]) / b["decode_tps"] * 100
        lines = [f"# SMOKE A/B FA16 — stack prod MTP-ON — {TS}", "",
                 f"Service prod réel (MTP ON, DEFER_PROMPT=1, KV q8_0, ctx 128k), "
                 f"prompt frais ~{b['prompt_tokens']} tok + 128 tok de génération.", "",
                 "| leg | .so | TMAX | prefill t/s | decode t/s (MTP) | wall s |",
                 "|---|---|---|---:|---:|---:|",
                 f"| BASE (prod avant) | {b['so']} | absent | {b['prefill_tps']} | {b['decode_tps']} | {b['wall_s']} |",
                 f"| **NEW (déploié)** | {n['so']} | {n['tmax']} | **{n['prefill_tps']}** | {n['decode_tps']} | {n['wall_s']} |",
                 "",
                 f"**Δ prefill : {dp:+.1f} %** | **Δ decode MTP : {dd:+.1f} %**", "",
                 "Caveat : 1 passage par leg (chauffe GPU légère BASE→NEW ; le prefill est "
                 "dominant sur 8 s de wall, l'effet thermique est mineur vs le Δ e2e mesuré)."]
        with open(OUT_SUMMARY, "w") as f:
            f.write("\n".join(lines) + "\n")
        log(f"  · Δ prefill {dp:+.1f} % | Δ decode {dd:+.1f} % → {OUT_SUMMARY}")
    except SystemExit:
        raise
    except Exception as e:
        log(f"  · EXCEPTION : {e!r}")
    finally:
        restore_final()
    ok = _restored and "BASE" in rows and "NEW" in rows
    with open(OUT_JSONL, "a") as f:
        f.write(json.dumps({"final": {"service_active": _restored,
                                      "so": md5_short(SO_PROD),
                                      "rows": {k: v for k, v in rows.items()}}},
                           ensure_ascii=False) + "\n")
    log(f"FIN — {'OK' if ok else 'ÉCHEC'} — service actif = {_restored}, .so final = {md5_short(SO_PROD)}")
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
