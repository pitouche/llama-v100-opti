# PROF MTP STEP A — macro (gputimes 100 ms) — 28/10

date : 2026-10-28 18:33 UTC
SO prod figé = 8035caf4 · service restauré : OUI
config prod exacte (H2 OFF, shortlist 131072, DEFER_PROMPT=1, Q6_W4R4, KV q8_0, FA auto, -b 2048)

| leg | ctx | decode t/s | acc rate/pos | dur(b,g,a) ms | VRAM pic |
|---|---|---|---|---|---|
| mtp-on | 114688 | 15.57 | (1.000,0.929,0.821) | (0.012,243.5,0.025) | 32438 |
| mtp-on | 160000 | 24.26 | (0.938,0.854,0.708) | (0.019,473.1,0.045) | 32438 |
| mtp-off | 114688 | 22.92 | — | — | 29768 |
| mtp-off | 160000 | 21.68 | — | — | 29768 |

Signaux à lire : (1) ON vs OFF par ctx ; (2) inversion ctx 114k vs 160k ;
(3) gputimes.tsv @100 ms : périodicité spec-step + gaps hôtes.
artefacts : /home/guile/llama-experiments/bench/mtp_prof_stepA_gputimes.tsv / /home/guile/llama-experiments/bench/mtp_prof_stepA_server.log / /home/guile/llama-experiments/bench/mtp_prof_stepA_results.jsonl
