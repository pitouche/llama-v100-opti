# Verdict e2e — levier B (lba4) en tok/s global — 24/10 20:21 — **CONFIRMÉ (+1,1 à +1,7 % decode MTP-OFF)**

Question : le +6,8 % microbench du kernel FA @T=1 (482→450 µs) se traduit-il en gain
applicatif ? FA n'est que 14,2 % du busy decode (run9e), le reste étant mmvq (73,6 %) —
le gain FA doit donc être **dilué** au pas global (attente ~+0,7 à +1 %).

## Méthode (harnais `ab_lb4.py`, identique `ab_lba.py` de 22/10)

- 2 legs, serveur frais par leg, port 8917, CVD=0, `-b 2048` (batch prod), KV q8_0,
  FA auto, H2 OFF, **MTP OFF** (decode single-token = T=1 = le cas que lba4 améliore).
- Seule différence : le `.so` via `GGML_BACKEND_PATH` (le `.so` prod n'est JAMAIS touché) :
  - **A-lba4** = `feaf05aa` (nouveau prod, `launch_bounds` NT=1 → minblocks=4, 124 regs)
  - **B-ref**  = `fa3e75c2` (ancien prod, backup, minblocks=3, 162 regs)
- Même kernel T3, smem, launcher, pb auto : seule diff = minblocks NT=1 (4 vs 3).
- Points : 32k / 65k / 98k ctx, 64 gen, temp 0 / top_k 1, filler, 2 reps, ordre alterné.

## Résultats (decode, t/s, médiane)

| contexte | A-lba4 | B-ref (lba) | Δ | Δ par rep (repro) |
|---:|---:|---:|---:|---|
| 32 768 | 24,92 | 24,93 | **−0,0 %** | 24,92/24,92 vs 24,91/24,93 |
| 65 536 | 23,73 | 23,48 | **+1,1 %** | 23,73/23,73 vs 23,48/23,48 |
| 98 304 | 22,56 | 22,19 | **+1,7 %** | 22,56/22,56 vs 22,18/22,19 |

- Les reps sont **stables à ±0,01 t/s** → le signal est réel, pas du bruit.
- **Le gain croît avec le ctx** (0 → +1,1 → +1,7 %) — même forme que le microbench
  (FA = seul terme O(ctx) ; mmvq constant) et même forme que lba 22/10 (+1,3/+2,2/+2,7 %).
- Wall par requête (prompt+64 gen) : 154,1 vs 154,2 s @98k — cohérent (−0,1 s).

## Lecture

- **Le gain B se propage bien en tok/s global** : +1,7 % @98k, au-delà du bas de la fourchette
  attendue (+0,7 à +1 %) — FA = 14,2 % du busy, +6,8 % sur la partie T=1 ≈ +1,0 %, le solde
  venant du combine/prefetch qui profite aussi de l'occupancy 4 blocs/SM.
- **Effet marginal cohérent avec l'occupancy** : lba (baseline T3 → 3 blocs/SM) valait
  +1,3/+2,2/+2,7 % ; lba4 (3 → 4 blocs/SM, marginal) vaut 0/+1,1/+1,7 % — **rendements
  décroissants** en occupancy, comme attendu (chaque bloc/SM supplémentaire masque moins de
  latence que le précédent). @32k le gain est nul : à ce KV, la partie T=1 de FA est trop
  courte pour que l'occupancy supplémentaire change le pas.
- **Gain MTP-ON : non mesurable en e2e par construction (non-item, pas un caveat bloquant).**
  Le gain lba4 n'agit que sur T=1 (NT=1) ; en MTP-ON la majorité des pas sont des verify
  T=2..4 (NT≥2, minblocks=2 identique dans les deux variants) et le T=1 n'apparaît qu'aux
  rejets totaux. FA = 14,2 % du busy, seule une fraction T=1 → gain total sur un pas MTP-ON
  **bien < 1 %**, noyé dans la variance d'acceptance MTP (contenu/sampling). Le **mesurer**
  en e2e MTP-ON est un non-sens : il faudrait des dizaines de reps sur contenu identique et le
  chiffre resterait confondu avec l'acceptance. Le gain **existe** (prouvé par microbench
  +6,8 % @T=1 **et** e2e MTP-OFF +1,7 %) ; l'e2e MTP-OFF **est** la confirmation appropriée.
  Non à refaire.

## Conclusion

**Levier B confirmé en e2e.** Le `.so` `feaf05aa` (déployé 19:34) est le bon choix :
gain microbench (+6,8 % @T=1) **et** gain applicatif (+1,1 à +1,7 % decode MTP-OFF,
croissant avec le ctx), sans régression sur le prefill/wall, sans toucher aux autres NT.
Le déploiement tient ; pas de rollback.

## Invariants post-bench
service `ollama-v100` active · `.so` prod `feaf05aa` **intact** (le bench ne le touchait pas,
chaque leg pointait `GGML_BACKEND_PATH`) · V100 libérée · superviseur `rc=0`.

## Artefacts
`ab_lb4.py` · `ab_lb4_results.jsonl` · `ab_lb4_summary.md` · `ab_lb4_server.log`.
