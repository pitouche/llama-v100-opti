# bench_kvcache_ram - resume (KV cache en RAM, multi-user)

Date : 2026-11-01T01:36:36.825837
Modele : qwen3.8-27b-q6-mtp (texte, sans mmproj) - MTP OFF, KV q8_0, b/ub 2048, cache-ram 8192 MiB

Scenario : 1 slot (comme prod OWUI). U1 = contexte A (long), U2 = contexte B (demi-longueur, distinct).

| point | ctx | T0 FROID A (s / t/s) | T1 warm resid (s, fresh) | T3 U2 (s) + station | T4 U1 retour (cache / fresh / pe s / wall s) | T5 FROID A+Q2 (s / t/s) | gain x |
|---|---|---|---|---|---|---|---|
| P04 | 16384 | 0.0s / None t/s | 0.0s / None | 0.0s | None / None / 0.0s / 0.733s | 0.0s / None t/s | - |
| P16 | 24576 | 0.0s / None t/s | 0.0s / None | 0.0s | None / None / 0.0s / 1.075s | 0.0s / None t/s | - |
| P32 | 40960 | 0.0s / None t/s | 0.0s / None | 0.0s | None / None / 0.0s / 1.371s | 0.0s / None t/s | - |
| P64 | 81920 | 0.0s / None t/s | 0.0s / None | 0.0s | None / None / 0.0s / 1.612s | 0.0s / None t/s | - |
| P96 | 114688 | 0.0s / None t/s | 0.0s / None | 0.0s | None / None / 0.0s / 2.059s | 0.0s / None t/s | - |
| P128 | 143360 | 0.0s / None t/s | 0.0s / None | 0.0s | None / None / 0.0s / 2.428s | 0.0s / None t/s | - |

## Lecture
- T0 = le prefill complet a eviter (cout d'un contexte froid).
- T4 = le multi-user : U1 revient apres que U2 ait pris le slot ;
  le prefixe A est restaure depuis la RAM (logs 'prompt cache update took' /
  'restored context checkpoint'), seul le delta Q2 se prefill.
- wall T4 - pe_dur T4 ~ cout de restauration RAM + overhead (a comparer T4 vs T1).
- T5 = meme demande a froid (serveur neuf) = cout SANS cache RAM.
- gain x = T5 / T4 : ce que la station en RAM fait gagner sur le temps de prompt.

## Cout de station (RAM) par contexte, deduits du log serveur
Voir les lignes 'cache state: N prompts, X MiB' dans kvcache_ram_*.log -
1 contexte ~a_tokens donne ~ a_tokens x 344.4 o (KV q8_0) de payload stationne.
