#!/usr/bin/env python3
"""Figure de synthese du bench 2026-10-01 (V100, Qwen3.8-27B Q6_K MTP+vision).
Version corrigee : prefill texte FROID vs mesure API gonflee par le cache de checkpoint."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/home/guile/llama-experiments/bench/"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8))
fig.suptitle("Auto-bench 2026-10-01 - Qwen3.8-27B Q6_K (MTP spec draft=3) + mmproj vision - V100 32GB, ollama custom (V100)", fontsize=11)

# --- Panel 1: prefill ---
# Texte FROID (pas de checkpoint exploitable : premier run, runner frais, prefixe different)
cold = [(73, 152.9, "73 (warmup graph)"),
        (1676, 779.5, ""),
        (3347, 860.9, "txt_equiv"),
        (20022, 836.9, "live_gauge"),
        (48130, 709.2, ""),
        (51030, 702.7, "diff_check E")]
ax1.plot([x/1000 for x, _, _ in cold], [y for _, y, _ in cold], "o-", color="#d62728",
         label="texte FROID (verifie: prefill complet)")
for x, y, lab in cold:
    ax1.annotate(f"{y:.0f}", (x/1000, y), textcoords="offset points", xytext=(0, 9), fontsize=8, color="#d62728")
# Mesure API quand le checkpoint restaure le prefixe (artefact: compte le prompt complet
# sur le temps de la diff seule) -> 1.2-1.5k t/s fictifs
artefact = [(3353, 1222.3), (6609, 1462.5), (13217, 1540.2), (26526, 1449.3), (44529, 1540.7)]
ax1.plot([x/1000 for x, y in artefact], [y for x, y in artefact], "s--", color="#ff9f43",
         label="mesure API avec checkpoint du prefixe (artefact, diff seule prefilled)")
for x, y in artefact:
    ax1.annotate(f"{y:.0f}", (x/1000, y), textcoords="offset points", xytext=(0, 9), fontsize=8, color="#e67e22")
# Vision (froid, seules ~22 tok de template cachees)
img = [(96, 149.7), (288, 388.2), (608, 453.2), (1056, 559.1), (1796, 515.4), (4128, 430.7)]
ax1.plot([x/1000 for x, y in img], [y for x, y in img], "s-", color="#1f77b4",
         label="vision (tokens image, froid)")
for x, y in img:
    ax1.annotate(f"{x}", (x/1000, y), textcoords="offset points", xytext=(0, -16), fontsize=7, color="#1f77b4")
ax1.set_xlabel("prompt (k tokens)")
ax1.set_ylabel("prompt eval (t/s)")
ax1.set_title("Prefill: texte froid ~700-860 t/s  |  vision 150-560 t/s")
ax1.legend(fontsize=8); ax1.grid(alpha=0.3); ax1.set_xlim(0, 55); ax1.set_ylim(0, 1750)

# --- Panel 2: gen vs contexte ---
ax2.plot([1.6, 6.0, 11.9, 23.8, 40.0], [69.2, 63.4, 50.4, 44.6, 39.6], "o-", color="#2ca02c",
         label="sweep counting 1..128 (runner 65k frais a chaque run)")
ax2.plot([77.0, 78.8], [51.8, 51.7], "s", color="#2ca02c", ms=9,
         label="ctxfill 77-79k (runner 200k, KV chaud)")
for x, y, lab in [(25.6, 38.2, "conversation reelle 25k"),
                  (43.4, 31.3, "conversation reelle 43k"),
                  (92.6, 28.2, "conversation reelle 92k")]:
    ax2.plot(x, y, "^", color="#9467bd", ms=10)
    ax2.annotate(f"{lab}\n{y:.0f} t/s", (x, y), textcoords="offset points", xytext=(0, 10),
                 fontsize=8, ha="center", color="#9467bd")
ax2.axhline(23, color="gray", ls=":", lw=1.5)
ax2.annotate("reference vanilla (sans MTP): 22-25 t/s", (1, 24.5), fontsize=8, color="gray")
ax2.set_xlabel("contexte final (k tokens)")
ax2.set_ylabel("generation (t/s)")
ax2.set_title("Generation MTP: la baisse avec le contexte est reelle (~0.75 t/s par k tok)")
ax2.grid(alpha=0.3); ax2.set_xlim(0, 100); ax2.set_ylim(0, 62)
ax2.legend(fontsize=8, loc="upper right")

plt.tight_layout()
out = D + "auto_bench_20261001.png"
plt.savefig(out, dpi=110)
print("saved", out)
