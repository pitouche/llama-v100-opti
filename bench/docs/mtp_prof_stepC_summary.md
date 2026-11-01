# PROF MTP STEP C — où passe le temps MTP (harnais corrigé W1/W2) — 29/10

date : 2026-10-29 03:32 UTC (valeurs spec re-calculées le 29/10 après fix des 2 bugs harnais)
SO prod figé = 8035caf4 · service restauré : OUI
config lente prod exacte (H2 OFF, shortlist 131072, DEFER_PROMPT=1, KV q8_0, FA auto, -b 2048, c167000, mmproj)
point : 114688 · W1 = 64 gen prompt neuf (catch-up) · W2 = 128 gen même prompt checkpointé (steady)

| leg | W1 t/s | catchup ms | W2 steady t/s | W2 drafts | mean acc len | t_cycle | t_draft | t_verify+ |
|---|---|---|---|---|---|---|---|---|
| mtp-on | 15.87 | 2835.4 | 53.45 | 37 | 3.432 | 64.7 ms | 7.84 ms | 56.9 ms |
| mtp-off | 22.92 | -17.2 | 22.78 | ? | ? | ? ms | ? ms | ? ms |

Lecture : t_cycle = W2 eval_ms / n_drafts (1 cycle = 1 draft + 1 verify rows=1..4).
t_draft mesuré par le serveur (t_draft_us). t_verify+ = verify 27B + sampling + gaps hôtes.
Cross-check GPU (gputimes, 10 Hz) : W2 MTP-ON util méd. 96 % (223 W) · W2 MTP-OFF 99 % (221 W) · W1 catch-up 23 % (64 W).
artefacts : mtp_prof_stepC_gputimes.tsv / mtp_prof_stepC_server.log / mtp_prof_stepC_results.jsonl
