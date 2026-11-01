#!/usr/bin/env python3
"""mtp_prof_stepA_analyze.py — ANALYSE FINALE de l'Étape A (bench du 28/10 18:20-18:33 UTC).

CORRIGÉ (2e passe, 28/10 ~20:35 UTC) — le docstring initial de ce script était FAUX :
  - gputimes N'EST PAS à 100 Hz : cadence effective = 13 Hz (dt_med=76 ms).
    Le sampler gputimes_sampler.py est passé en PERIOD=0.1 s (mtp_prof_stepA.py:108),
    mais nvidia-smi (fork+exec ~75 ms/tour) impose dt=76 ms -> 13 Hz.
  - Les fenêtres decode initialement "estimées à l'oeil" (decode = longues plages
    GPU 100 %) étaient FAUSES : les longues plages GPU ~100 % = le PREFILL.
    Le decode ne dure que 2.6-4.1 s par leg (voir eval_ms server.log).
  - Fenêtres decode ci-dessous = ANCRÉES sur horodatages superviseur + server.log.

Fenêtres decode retenues (t0 = 1er échantillon gputimes = 18:20:31.745 UTC) :
  L1 mtp-on 114k  : +248.4 .. +251.4 s   (eval 4.11 s)
  L2 mtp-on 160k  : +376.0 .. +377.2 s   (eval 2.64 s)
  L3 mtp-off 114k : +606.7 .. +609.2 s   (eval 2.79 s)
  L4 mtp-off 160k : +733.4 .. +736.2 s   (eval 2.95 s)
"""
import statistics as st, json

D = "/home/guile/llama-experiments/bench"
rows = []
with open(f"{D}/mtp_prof_stepA_gputimes.tsv") as f:
    next(f)
    for line in f:
        p = line.split()
        if len(p) == 5:
            rows.append((int(p[0]), int(p[1]), int(p[2]), int(p[3]), float(p[4])))
T0 = rows[0][0]
dt_med = st.median([rows[i+1][0]-rows[i][0] for i in range(len(rows)-1)])
DECODE_WINS = {
    "L1 mtp-on 114k dec (15.57 t/s)": (248.4, 251.4),
    "L2 mtp-on 160k dec (24.26 t/s)": (376.0, 377.2),
    "L3 mtp-off 114k dec (22.92 t/s)": (606.7, 609.2),
    "L4 mtp-off 160k dec (21.68 t/s)": (733.4, 736.2),
}

def win_stats(lo, hi):
    sel = [r for r in rows if lo <= (r[0]-T0)/1000 < hi]
    n = len(sel)
    gs = [r[1] for r in sel]; ms = [r[2] for r in sel]; ps = [r[4] for r in sel]
    return {
        "n": n,
        "gpu_med": st.median(gs),
        "gpu_p10": sorted(gs)[max(0, int(n*0.1))] if n else 0,
        "gpu>=95 %": sum(1 for x in gs if x >= 95)/max(1,n)*100,
        "mem_med %": st.median(ms),
        "W_med": st.median(ps),
    }

if __name__ == "__main__":
    print(f"dt_med={dt_med:.0f}ms (cadence effective {1000/dt_med:.0f} Hz - PAS 100 Hz) n={len(rows)}")
    print(f"{'window':34s} {'n':>3s} {'gpu_med':>7s} {'gpu_p10':>7s} {'%gpu>=95':>8s} {'mem_med':>7s} {'W_med':>6s}")
    for name,(lo,hi) in DECODE_WINS.items():
        s = win_stats(lo,hi)
        print(f"{name:34s} {s['n']:3d} {s['gpu_med']:7.1f} {s['gpu_p10']:7.1f} "
              f"{s['gpu>=95 %']:8.1f} {s['mem_med %']:7.1f} {s['W_med']:6.1f}")
    print()
    print("Lecture : MTP-ON = GPU-bas (~20-25 %, mem ~5-6 %, W~60) dans les 2 ctx ;")
    print("          MTP-OFF = GPU-haut (~98 %, mem ~88 %, W~225) dans les 2 ctx.")
    print("          -> decode MTP-ON apparemment host-bound / sous-occupe le GPU ;")
    print("             la periodicite fine du spec-step (~62 ms) n'est PAS resolue a 13 Hz.")
    print("          -> Etape B CUPTI (kernel-par-kernel) est la suite.")
