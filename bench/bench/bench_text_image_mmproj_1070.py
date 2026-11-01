#!/usr/bin/env python3
"""bench_text_image_mmproj_1070.py — bench complet texte + image sur le service
prod ollama-v100 (port 11434), config mmproj-1070 déployée le 25/10.

Ne stoppe PAS le service : mesure sur le service en place. Avant de mesurer,
attend la libération du slot (probe rapide répétée) pour éviter toute
contamination par un tir concurrent (ex. génération agent en cours).

Mesures :
  TEXTE  : prompt ~4k tokens, gen 256 tok  -> prompt eval tok/s + gen tok/s
  IMAGE  : 1472x1472 (3 cercles rouges), gen 128 tok
           -> encode estimé (wall - prompt_eval - gen), prompt eval vision, gen
  VRAM   : pic par phase sur chaque GPU (échantillonnage nvidia-smi 4 Hz)

Sortie : stdout (résumé chiffré) + bench/results_mmproj_1070.json
RC : 0 = mesures OK · 1 = échec
"""
import json, os, subprocess, sys, threading, time, urllib.request

HOST = "http://127.0.0.1:11434"
MODEL = "qwen3.8-27b-q6-mtp-vision:latest"
OUT = "/home/guile/llama-experiments/bench/results_mmproj_1070.json"


def log(m):
    print(time.strftime("[%H:%M:%S] ") + str(m), flush=True)


def chat(messages, num_predict, temperature=0):
    payload = json.dumps({"model": MODEL, "stream": False, "messages": messages,
                          "options": {"num_predict": num_predict,
                                      "temperature": temperature}}).encode()
    t0 = time.time()
    req = urllib.request.Request(HOST + "/api/chat", data=payload,
                                 headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=900))
    return d, time.time() - t0


class VramSampler(threading.Thread):
    """Échantillonne la VRAM des 2 GPUs (index 0=V100, 1=1070) à 4 Hz."""

    def __init__(self):
        super().__init__(daemon=True)
        self.samples = []
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10)
                vals = {}
                for line in r.stdout.strip().splitlines():
                    i, m = line.split(",")
                    vals[int(i.strip())] = int(m.strip())
                self.samples.append((time.time(), vals))
            except Exception:
                pass
            time.sleep(0.25)

    def stop(self):
        self._stop.set()
        self.join(timeout=5)

    def peak(self, gpu, t0, t1):
        vals = [v.get(gpu, 0) for t, v in self.samples if t0 <= t <= t1]
        return max(vals) if vals else None


def make_text_prompt(target_tokens=4096):
    base = ("Le systeme ollama-v100 heberge un modele de langage de 27 milliards de parametres "
            "sur une GPU V100 32 Go. Le projecteur multimodal est isole sur une GTX 1070 8 Go "
            "grace au placement dual-GPU. Le cache de cles-valeurs utilise la quantification "
            "q8_0 pour reduire lempreinte memoire. La speculation MTP accelere la generation "
            "avec un tirage de trois jetons par pas. Le projecteur consomme environ un milliard "
            "de parametres supplementaires pendant lecodage des images. ")
    reps = max(1, int(target_tokens * 4 / len(base)))
    text = base * reps
    return ("Analyse le texte ci-dessous et donne un resume en 3 lignes. Texte :\n" + text)


def make_image_b64():
    from PIL import Image, ImageDraw
    import io
    img = Image.new("RGB", (1472, 1472), "white")
    d = ImageDraw.Draw(img)
    d.ellipse([100, 100, 500, 500], fill="red")
    d.ellipse([600, 100, 1000, 500], fill="red")
    d.ellipse([1100, 100, 1400, 400], fill="red")
    d.rectangle([200, 700, 800, 1200], fill="blue")
    d.polygon([(1100, 700), (1500, 700), (1300, 1200)], fill="green")
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    import base64
    return base64.b64encode(buf.getvalue()).decode()


def wait_idle(max_wait_s=900):
    """Attends que le slot soit libre : une probe ~30 tok doit boucler < 10 s."""
    t_start = time.time()
    attempt = 0
    while time.time() - t_start < max_wait_s:
        attempt += 1
        try:
            d, wall = chat([{"role": "user", "content": "Reponds juste : OK"}],
                           num_predict=2)
        except Exception as e:
            log(f"probe {attempt} en erreur ({e}) — retry dans 5s")
            time.sleep(5)
            continue
        if wall < 10.0:
            log(f"slot libéré (probe {wall * 1000:.0f} ms, essai n°{attempt})")
            return True
        log(f"slot occupé (probe {wall:.1f} s) — essai n°{attempt}, on attend")
        time.sleep(5)
    return False


def sanity():
    try:
        v = json.load(urllib.request.urlopen(HOST + "/api/version", timeout=10))
        log(f"version ollama : {v.get('version', '?')}")
    except Exception as e:
        log(f"version inconnue : {e}")
    r = subprocess.run(["journalctl", "-u", "ollama-v100", "--since", "-12h",
                        "--no-pager"], capture_output=True, text=True, timeout=30)
    pin = "mmproj device: pinning text model to first GPU" in r.stdout
    clip = "clip_ctx: CLIP using CUDA1 backend" in r.stdout
    log(f"feature mmproj-1070 armé dans le journal : pinning={pin}, CLIP-CUDA1={clip}")
    return pin and clip


def main():
    t0 = time.time()
    log("BENCH complet TEXTE + IMAGE — prod ollama-v100 (config mmproj-1070)")
    if not sanity():
        log("AVERTISSEMENT : feature mmproj-1070 non confirmé dans le journal")
    time.sleep(30)  # laisse le tir agent en cours finir (la probe gère le reste)
    if not wait_idle():
        log("ECHEC : slot jamais libéré")
        sys.exit(1)

    # VRAM idle (baseline)
    r = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used",
                        "--format=csv,noheader,nounits"], capture_output=True,
                       text=True, timeout=10)
    base = {}
    for line in r.stdout.strip().splitlines():
        i, m = line.split(",")
        base[int(i.strip())] = int(m.strip())
    log(f"VRAM idle : V100={base.get(0)} MiB, 1070={base.get(1)} MiB")

    sam = VramSampler()
    sam.start()
    results = {"vrAM_idle_MiB": base}
    gpu_names = {0: "V100 (CUDA0)", 1: "GTX 1070 (CUDA1)"}

    # ---------- PHASE TEXTE ----------
    log("PHASE TEXTE : ~4k tok prompt + 256 tok gen")
    tt0 = time.time()
    d, wall = chat([{"role": "user", "content": make_text_prompt()}], num_predict=256)
    tt1 = time.time()
    pe_c, pe_d = d.get("prompt_eval_count", 0), d.get("prompt_eval_duration", 0) / 1e9
    ev_c, ev_d = d.get("eval_count", 0), d.get("eval_duration", 0) / 1e9
    results["text"] = {
        "prompt_tokens": pe_c, "prompt_eval_s": round(pe_d, 3),
        "prompt_tps": round(pe_c / pe_d, 1) if pe_d else None,
        "gen_tokens": ev_c, "gen_s": round(ev_d, 3),
        "gen_tps": round(ev_c / ev_d, 1) if ev_d else None,
        "wall_s": round(wall, 2),
        "content_head": (d.get("message", {}).get("content") or "")[:80],
    }
    log(f"  texte : {pe_c} tok prompt / {pe_d:.2f}s = {results['text']['prompt_tps']} tok/s"
        f" | gen {ev_c} tok / {ev_d:.2f}s = {results['text']['gen_tps']} tok/s"
        f" | wall {wall:.1f}s")

    time.sleep(3)

    # ---------- PHASE IMAGE ----------
    log("PHASE IMAGE : 1472x1472 (3 cercles rouges) + 128 tok gen")
    b64 = make_image_b64()
    ti0 = time.time()
    d, wall = chat([{"role": "user",
                     "content": "Combien de cercles rouges voit-on sur cette image ? "
                                "Reponds par un seul chiffre.",
                     "images": [b64]}], num_predict=128)
    ti1 = time.time()
    pe_c, pe_d = d.get("prompt_eval_count", 0), d.get("prompt_eval_duration", 0) / 1e9
    ev_c, ev_d = d.get("eval_count", 0), d.get("eval_duration", 0) / 1e9
    encode_est = max(0.0, wall - pe_d - ev_d)
    results["image"] = {
        "prompt_tokens": pe_c, "prompt_eval_s": round(pe_d, 3),
        "prompt_tps": round(pe_c / pe_d, 1) if pe_d else None,
        "gen_tokens": ev_c, "gen_s": round(ev_d, 3),
        "gen_tps": round(ev_c / ev_d, 1) if ev_d else None,
        "encode_est_s": round(encode_est, 2),
        "wall_s": round(wall, 2),
        "content_head": (d.get("message", {}).get("content") or "")[:40],
    }
    log(f"  image : encode est. {encode_est:.2f}s | {pe_c} tok prompt / {pe_d:.2f}s"
        f" = {results['image']['prompt_tps']} tok/s | gen {ev_c} tok / {ev_d:.2f}s"
        f" = {results['image']['gen_tps']} tok/s | wall {wall:.1f}s | reponse: "
        f"{results['image']['content_head']!r}")

    sam.stop()
    for phase, (p0, p1) in (("text", (tt0, tt1)), ("image", (ti0, ti1))):
        pk = {g: sam.peak(g, p0, p1) for g in (0, 1)}
        results[phase]["vram_peak_MiB"] = {gpu_names[g]: pk.get(g) for g in (0, 1)}
        log(f"  VRAM pic {phase} : " +
            " | ".join(f"{gpu_names[g]}={pk.get(g)} MiB" for g in (0, 1)))

    # cross-check journal (lignes prompt eval des 2 tasks)
    r = subprocess.run(["journalctl", "-u", "ollama-v100", "--since",
                        f"-{int(time.time() - t0) + 30}s", "--no-pager"],
                       capture_output=True, text=True, timeout=30)
    lines = [l for l in r.stdout.splitlines() if "prompt eval time" in l or
             "process_mtmd" in l]
    results["journal"] = lines[-6:]
    log("journal : " + " || ".join(l[-120:] for l in lines[-4:]))

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    log(f"résultats : {OUT}")
    log(f"=== FIN BENCH en {time.time() - t0:.0f}s — rc=0 ===")
    sys.exit(0)


if __name__ == "__main__":
    main()
