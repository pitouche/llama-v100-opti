#!/usr/bin/env python3
"""
plot_q6_170k.py — graphiques de la campagne sweep_q6_170k (06/11/2026).

Génère :
  charts/decode_off.png    t/s decode (MTP OFF) par état, vs ctx
  charts/decode_on.png     t/s decode (MTP ON) par état, vs ctx (médiane toutes fenêtres)
  charts/decode_on_plateau.png  t/s decode ON — plateau (médiane 6 dernières fenêtres)
  charts/decode_on_norm.png     t/s decode ON INDEXÉ : plateau × 3.4/(1+3A) (dérivé)
  charts/prefill_off.png   t/s prefill (MTP OFF) par état, vs ctx
  charts/acceptance.png    acceptance MTP par état, vs ctx
  charts/gain_s0_s6.png    gain S0→S6 (%) par ctx, off, on brut et on indexé
  charts/mult_on_off.png   multiplicateur MTP ON/OFF (plateau / off) par état

Quantité « plateau » : médiane des 6 dernières fenêtres tg_3s de R2 (la 1ʳᵉ
est exclue). Retire le warm-up MTP (les ~1200 premiers tok générés sont
lents : ramp de l'acceptance), qui biaisait la médiane toutes fenêtres —
surtout à 112k où il couvrait 11–14 des 15–26 fenêtres de R2.

INDEXÉ : plateau × 3.4/(1+3A) — t/s = (1+3A)/t_step validé à ±0.007 sur les
77 runs → la quantité indexée ≈ t_step (vitesse du pas de vérification),
indépendante de l'acceptance. Courbe dérivée, pas mesurée.

Données : sweep.jsonl (dernier record valide par (state, ctx, mtp)).
USAGE : cd bench/rebench-abba/20261106_sweep_q6 && python3 plot_q6_170k.py
"""
import json
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "sweep_q6_170k.jsonl")
OUT = os.path.join(HERE, "..", "charts")
os.makedirs(OUT, exist_ok=True)

STATES = ["S0", "S1", "S2", "S3", "S4", "S5", "S6"]
LABELS = {
    "S0": "S0 · 0 opti (pre-T3, b512)",
    "S1": "S1 · +batch (b2048)",
    "S2": "S2 · +T3",
    "S3": "S3 · +lba4",
    "S4": "S4 · +cvec",
    "S5": "S5 · +vldg",
    "S6": "S6 · +FA16 (prod)",
}
COLORS = ["#999999", "#cccccc", "#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#D32F2F"]


def load():
    rows = [json.loads(l) for l in open(DATA)]
    last = {}
    for r in rows:
        k = (r["state"], r["ctx_target"], r["mtp"])
        if r.get("steady_tps") is not None:
            last[k] = r
    return last


def ctxk(c):
    return c / 1024.0


def plateau_tps(r):
    """Médiane des 6 dernières fenêtres tg_3s de R2 (warm-up MTP retiré)."""
    tg = r["r2"]["log"]["tg"]
    v = [w[2] for w in (tg[-6:] if len(tg) >= 10 else tg)]
    return statistics.median(v)


def indexed_tps(r, a_ref=0.8):
    """Plateau indexé à acceptance a_ref : plateau × (1+3·a_ref)/(1+3·A)."""
    if r.get("mtp_acceptance") is None:
        return None
    return plateau_tps(r) * (1 + 3 * a_ref) / (1 + 3 * r["mtp_acceptance"])


def plot_decode_on(data, value, fname, title, ylabel, note=""):
    fig, ax = plt.subplots(figsize=(11, 7))
    for i, st in enumerate(STATES):
        xs, ys = [], []
        for c in sorted(set(k[1] for k in data)):
            r = data.get((st, c, "on"))
            v = value(r) if r else None
            if v:
                xs.append(ctxk(c))
                ys.append(v)
        ax.plot(xs, ys, "o-", color=COLORS[i], lw=2, ms=5, label=LABELS[st])
    ax.set_xlabel("contexte (k tokens)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Q6_K 27B — {title}" + (f"\n{note}" if note else ""))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=140)
    plt.close(fig)


def plot_decode(data, mtp):
    if mtp == "on":
        plot_decode_on(data, lambda r: r["steady_tps"], "decode_on.png",
                       "decode MTP ON (médiane TOUTES fenêtres — à titre de référence)",
                       "decode t/s (régime établi, 2048 tok)")
        return
    fig, ax = plt.subplots(figsize=(11, 7))
    for i, st in enumerate(STATES):
        xs, ys = [], []
        for c in sorted(set(k[1] for k in data)):
            r = data.get((st, c, mtp))
            if r:
                xs.append(ctxk(c))
                ys.append(r["steady_tps"])
        ax.plot(xs, ys, "o-", color=COLORS[i], lw=2, ms=5, label=LABELS[st])
    ax.set_xlabel("contexte (k tokens)")
    ax.set_ylabel("decode t/s (régime établi, 2048 tok)")
    ax.set_title(f"Q6_K 27B — decode {mtp.upper()} (V100, KV q8_0, FA auto, -c 170000)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"decode_{mtp}.png"), dpi=140)
    plt.close(fig)


def plot_prefill(data):
    fig, ax = plt.subplots(figsize=(11, 7))
    for i, st in enumerate(STATES):
        xs, ys = [], []
        for c in sorted(set(k[1] for k in data)):
            r = data.get((st, c, "off"))
            if r and r.get("prefill_tps"):
                xs.append(ctxk(c))
                ys.append(r["prefill_tps"])
        ax.plot(xs, ys, "o-", color=COLORS[i], lw=2, ms=5, label=LABELS[st])
    ax.set_xlabel("contexte (k tokens)")
    ax.set_ylabel("prefill t/s (froid, 1er passage)")
    ax.set_title("Q6_K 27B — prefill MTP OFF (V100, KV q8_0, FA auto)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "prefill_off.png"), dpi=140)
    plt.close(fig)


def plot_acceptance(data):
    fig, ax = plt.subplots(figsize=(11, 7))
    for i, st in enumerate(STATES):
        xs, ys = [], []
        for c in sorted(set(k[1] for k in data)):
            r = data.get((st, c, "on"))
            if r and r.get("mtp_acceptance") is not None:
                xs.append(ctxk(c))
                ys.append(r["mtp_acceptance"])
        if xs:
            ax.plot(xs, ys, "o-", color=COLORS[i], lw=2, ms=5, label=LABELS[st])
    ax.set_xlabel("contexte (k tokens)")
    ax.set_ylabel("draft acceptance")
    ax.set_ylim(0, 1)
    ax.set_title("Q6_K 27B — MTP draft acceptance (n_max=3, backend-sampling)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "acceptance.png"), dpi=140)
    plt.close(fig)


def plot_gain(data):
    fig, ax = plt.subplots(figsize=(11, 7))
    xs_all = sorted(set(k[1] for k in data))
    series = (("off", "#2196F3", "o", lambda r: r["steady_tps"], "gain S0→S6 MTP OFF"),
              ("on", "#D32F2F", "s", lambda r: r["steady_tps"], "gain S0→S6 MTP ON (brut)"),
              ("on_idx", "#1B5E20", "^", lambda r: indexed_tps(r), "gain S0→S6 MTP ON (indexé)"))
    for _, color, marker, value, label in series:
        xs, ys = [], []
        for c in xs_all:
            r0 = data.get(("S0", c, "on" if "on" in _ else "off"))
            r6 = data.get(("S6", c, "on" if "on" in _ else "off"))
            v0, v6 = (value(r0), value(r6)) if "on" in _ else (r0 and r0["steady_tps"], r6 and r6["steady_tps"])
            if v0 and v6:
                xs.append(ctxk(c))
                ys.append((v6 / v0 - 1) * 100)
        ax.plot(xs, ys, marker + "-", color=color, lw=2, ms=6, label=label)
    ax.set_xlabel("contexte (k tokens)")
    ax.set_ylabel("gain % (S0 → S6)")
    ax.set_title("Q6_K 27B — apport total des optimisations S0→S6\n"
                 "(ON indexé : plateau × 3.4/(1+3A) — le creux brut de 112k disparaît)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "gain_s0_s6.png"), dpi=140)
    plt.close(fig)


def plot_mult_on_off(data):
    fig, ax = plt.subplots(figsize=(11, 7))
    for i, st in enumerate(STATES):
        xs, ys = [], []
        for c in sorted(set(k[1] for k in data)):
            r0 = data.get((st, c, "off"))
            r1 = data.get((st, c, "on"))
            if r0 and r1 and r0["steady_tps"]:
                xs.append(ctxk(c))
                ys.append(plateau_tps(r1) / r0["steady_tps"])
        ax.plot(xs, ys, "o-", color=COLORS[i], lw=2, ms=5, label=LABELS[st])
    ax.set_xlabel("contexte (k tokens)")
    ax.set_ylabel("MTP ON (plateau) / OFF (×)")
    ax.set_title("Q6_K 27B — multiplicateur MTP ON/OFF par état (ON en plateau, warm-up retiré)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "mult_on_off.png"), dpi=140)
    plt.close(fig)


def plot_decode_norm(data):
    """Courbe INDEXÉE : plateau × 3.4/(1+3A) (≈ t_step, indépendant de l'acceptance)."""
    A_REF = 0.8
    plot_decode_on(data, lambda r: indexed_tps(r, A_REF), "decode_on_norm.png",
                   f"decode MTP ON INDEXÉ à acceptance {A_REF} (courbe dérivée)",
                   "t/s indexé (plateau × 3.4/(1+3A)) ≈ t_step",
                   note="warm-up MTP retiré (plateau) + acceptance égalisée — "
                        "reste la vitesse du pas de vérification")


def plot_decode_plateau(data):
    plot_decode_on(data, plateau_tps, "decode_on_plateau.png",
                   "decode MTP ON — plateau (médiane 6 dernières fenêtres de R2)",
                   "decode t/s (plateau, warm-up MTP retiré)",
                   note="acceptance toujours variable d'un ctx à l'autre — "
                        "voir la courbe indexée pour l'égaliser")


def main():
    data = load()
    n = len(data)
    print(f"points : {n} (attendu 154)")
    plot_decode(data, "off")
    plot_decode(data, "on")
    plot_decode_plateau(data)
    plot_decode_norm(data)
    plot_prefill(data)
    plot_acceptance(data)
    plot_gain(data)
    plot_mult_on_off(data)
    print("graphiques écrits dans charts/")


if __name__ == "__main__":
    main()
