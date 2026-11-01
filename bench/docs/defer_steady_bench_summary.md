# DEFER STEADY — le coût DEFER_PROMPT est-il ponctuel ?

date : 2026-10-29 01:11 UTC
config : lente EXACTE (shortlist+DEFER+H2off), MTP ON, b2048, c167000, mmproj, .so 8035caf4
service restauré : OUI

| req | prompt tok traités | decode t/s | généré |
|---|---|---|---|
| A-contam | 111713 | 14.72 | 64 |
| B-steady | 4 | 53.85 | 128 |

Lignes serveur (checkpoint/timings, extrait) :
    slot   operator(): id  0 | task 15 | cached n_tokens = 98346, memory_seq_rm [98346, end)
    slot   operator(): id  0 | task 15 | cached n_tokens = 100394, memory_seq_rm [100394, end)
    slot   operator(): id  0 | task 15 | cached n_tokens = 102442, memory_seq_rm [102442, end)
    slot   operator(): id  0 | task 15 | cached n_tokens = 104490, memory_seq_rm [104490, end)
    slot   operator(): id  0 | task 15 | cached n_tokens = 106538, memory_seq_rm [106538, end)
    slot   operator(): id  0 | task 15 | cached n_tokens = 108586, memory_seq_rm [108586, end)
    slot   operator(): id  0 | task 15 | cached n_tokens = 109707, memory_seq_rm [109707, end)
    slot   operator(): id  0 | task 15 | cached n_tokens = 111751, memory_seq_rm [111751, end)
    slot print_timing: id  0 | task 15 | prompt eval time =  179750.27 ms / 111713 tokens (    1.61 ms per token,   621.49 tokens per second)
    slot print_timing: id  0 | task 15 |        eval time =    4346.49 ms /    64 tokens (   68.99 ms per token,    14.49 tokens per second)
    slot   operator(): id  0 | task 89 | restored context checkpoint (pos_min = 111750, pos_max = 111750, n_tokens = 111751, n_past = 111751, hits = 1, size = 149.6
    slot   operator(): id  0 | task 89 | cached n_tokens = 111751, memory_seq_rm [111751, end)
    slot print_timing: id  0 | task 89 | prompt eval time =     140.44 ms /     4 tokens (   35.11 ms per token,    28.48 tokens per second)
    slot print_timing: id  0 | task 89 |        eval time =    2377.05 ms /   128 tokens (   18.72 ms per token,    53.43 tokens per second)

Lecture : REQ B >> REQ A → le coût DEFER est PONCTUEL (catch-up 1× au 1er token).
REQ B ≈ 56 t/s → le steady-state MTP .so 8035caf4 est PLUS RAPIDE que le 18/10 (45.7).
