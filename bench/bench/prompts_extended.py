#!/usr/bin/env python3
"""
prompts_extended.py — 06/11/2026 — prompts étendus au-delà du canonique (107 340)
pour la campagne Q6_K 170k (paliers de 16k, borne haute 170000).

Méthode : même pipeline validé 02/11 (TRUNCATION_VALIDATION_20261102.md) :
  1. texte étendu = canonical * 3 (≈ 322 k tokens) — la répétition du même texte
     reste une charge déterministe et comparable (même contenu sémantique).
  2. tokenisation du texte étendu (llama-tokenize, même tokenizer embarqué GGUF)
     → ids_ext (séquence autoritative, gauche→droite : les 107 340 premiers ids
     doivent être identiques à la tokenisation du seul canonical).
  3. pour chaque point N : prompt = detok_fix(ids_ext[:N]) (U+0120 -> U+0020).
  4. vérifications dures :
     - pour N ≤ 107 340 : sha256 == valeur du manifest existant (16384/32768/65536/
       98304/107340) — sinon ABORT (comparabilité avec le rebench 02/11).
     - pour N > 107 340 : stabilité re-tokenize(detok(ids[:N])) == ids[:N].

Points de la campagne (16k en 16k, borne haute 170000) :
  16384 32768 49152 65536 81920 98304 114688 131072 147456 163840  (+167568)
  Le point "170k" utilise un prompt de 167568 tokens car prompt + 2048 gén +
  template doit rester ≤ 170000 (dépasser = plantage, consigne opérateur).

Sortie : bench/prompts/<N>.txt + manifest_ext.json (à côté de manifest.json,
dossier unique bench/prompts/ — canonical.json est aussi là).
USAGE : python3 bench/bench/prompts_extended.py
(chemins machine-dépendants en tête : TOKENIZE_BIN / GGUF / LIBDIRS — README §6)
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# Chemins repo : D = bench/bench/ → prompts dans bench/prompts/ (dossier unique)
PROMPTS = os.path.join(HERE, "..", "prompts")
CANONICAL = os.path.join(PROMPTS, "canonical.json")
OUTDIR = PROMPTS
EXISTING = PROMPTS   # manifest.json + points ≤ 107 340 vivent dans le même dossier

SHA_CANONICAL = "44b1737bb5d7034a0eaa18dd346567f78befa1b7949a6f8a227c129f1d159b91"
TOKENIZE_BIN = "/home/guile/llama-experiments/build/cuda-v100-volta/bin/llama-tokenize"
GGUF = ("/usr/share/ollama/.ollama/models/blobs/"
        "sha256-493301830a596b8ad56dc1329f80bbcb578c8e910da395feafdc9cd8263430bb")
LIBDIRS = [
    "/home/guile/llama-experiments/tools/cuda-12.6/usr/local/cuda-12.6/lib64",
    "/home/guile/llama-experiments/tools/cuda-12.6/usr/local/cuda-12.6/targets/x86_64-linux/lib",
]

NORM = "\u0120"

# paliers de la campagne (borne haute : prompt 167568 + 2048 gén ≤ 170000)
POINTS = [49152, 81920, 114688, 131072, 147456, 163840, 167568]
# points existants à re-vérifier contre la séquence étendue (comparabilité).
# 107340 exclu volontairement : son dernier token (BPE de jonction) diffère de
# la séquence étendue (attendu — le texte continue après). Fichier 107340.txt
# existant, inchangé, non utilisé par la campagne.
EXISTING_CHECK = {16384, 32768, 65536, 98304}


def tokenize_ids(text, env):
    p = subprocess.run([TOKENIZE_BIN, "-m", GGUF, "--no-bos", "--ids", "--stdin"],
                       env=env, input=text, capture_output=True, text=True, timeout=900)
    if p.returncode != 0:
        sys.exit(f"ABORT : llama-tokenize rc={p.returncode}")
    return [int(x) for x in re.findall(r"-?\d+", p.stdout.split("Total")[0])]


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    raw = open(CANONICAL, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    assert sha == SHA_CANONICAL, f"canonical sha {sha} != {SHA_CANONICAL}"
    text = json.loads(raw)["messages"][0]["content"]
    print(f"canonique : {len(text)} chars OK")

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(LIBDIRS) + os.pathsep + env.get("LD_LIBRARY_PATH", "")

    # 1. texte étendu = canonical * 3
    text_ext = text * 3
    print(f"texte étendu : {len(text_ext)} chars ({len(text_ext)//1024} Ko)")
    t0 = time.time()
    ids = tokenize_ids(text_ext, env)
    print(f"tokenisation étendue : {len(ids)} tokens en {time.time()-t0:.1f}s")

    # 2. vocab GGUF
    from gguf import GGUFReader
    G = GGUFReader(GGUF)
    pieces = G.fields["tokenizer.ggml.tokens"].contents()
    print(f"vocab GGUF : {len(pieces)} ids")

    # détoko complet de la séquence étendue
    detok = "".join(pieces[j] for j in ids).replace(NORM, " ")
    # contrôle caractères : detok(ids_ext) doit reproduire text*3 (BPE : les
    # tokens de jonction se mergeent/dés-merge mais la chaîne est invariante)
    assert detok == text * 3, "detok(ids_ext) != text*3 — séquence étendue invalide"

    # manifest existant (vraie source de sha pour ≤ 107 340)
    mf_old = json.load(open(os.path.join(EXISTING, "manifest.json")))

    mf = {}
    max_needed = max(max(POINTS), max(EXISTING_CHECK))
    assert max_needed <= len(ids), f"pas assez de tokens ({len(ids)} < {max_needed})"

    # 3. points à vérifier (existants) — sans écrire
    for N in sorted(EXISTING_CHECK):
        tr = "".join(pieces[j] for j in ids[:N]).replace(NORM, " ")
        h = hashlib.sha256(tr.encode()).hexdigest()
        expect = mf_old[str(N)]["prompt_sha256"]
        ok = h == expect
        print(f"  check {N:>7} : sha {h[:16]} {'== OK' if ok else '!= ABORT ' + expect[:16]}")
        if not ok:
            sys.exit(f"ABORT : dérive sur le point existant {N}")

    # 4. points neufs — écriture + stabilité
    for N in sorted(POINTS):
        if os.path.exists(os.path.join(OUTDIR, f"{N}.txt")):
            print(f"  {N:>7} : déjà présent, saut")
            continue
        tr = "".join(pieces[j] for j in ids[:N]).replace(NORM, " ")
        t0 = time.time()
        if tokenize_ids(tr, env) != ids[:N]:
            sys.exit(f"ABORT : instable à N={N}")
        stab_s = time.time() - t0
        h = hashlib.sha256(tr.encode()).hexdigest()
        with open(os.path.join(OUTDIR, f"{N}.txt"), "w") as f:
            f.write(tr)
        mf[str(N)] = {"prompt_sha256": h, "n_tokens": N, "n_chars": len(tr)}
        print(f"  {N:>7} tokens -> {len(tr):>8} chars  sha {h[:16]}  stable=OK ({stab_s:.1f}s)")

    mf["_meta"] = {
        "method": ("canonical*3, detok_fix(ids_ext[:N]), U+0120->U+0020 ; "
                   "stabilité re-tokenize vérifiée ; points ≤107340 validés contre le manifest 02/11"),
        "tokenizer": TOKENIZE_BIN,
        "n_tokens_ext_total": len(ids),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_mf = os.path.join(OUTDIR, "manifest_ext.json")
    old = {}
    if os.path.exists(out_mf):
        old = json.load(open(out_mf))
        old.pop("_meta", None)
    old.update(mf)
    old["_meta"] = mf["_meta"]
    with open(out_mf, "w") as f:
        json.dump(old, f, indent=1)
    print(f"manifest : {out_mf}")
    print("RESULTAT : OK")


if __name__ == "__main__":
    main()
