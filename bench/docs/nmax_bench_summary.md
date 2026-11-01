# BENCH nmax — n-max 3 vs 4 — .so T3 PROD (d2df07b8) — 19/10 — MTP ON (config prod), -b 2048, interleave

date : 2026-10-19 00:19 UTC
reps : 2 · service restauré : OUI
config : -c 167000, --spec-type draft-mtp --spec-draft-backend-sampling, KV q8_0, FA auto
binaire : SO_T3 PROD — A = --spec-draft-n-max 3 (prod actuelle), B = --spec-draft-n-max 4

## ctx 114688
| leg | rep | decode t/s | prefill t/s | acc (dernier slot) |
|---|---|---|---|---|
| A-nmax3 | 1 | 45.72 | 609.4 | slot print_timing: id  0 | task 18 | draft acceptance = 0.92000 (   46 accepted /    50 generated), mean len =  3.71 |
| B-nmax4 | 1 | 39.58 | 590.8 | slot print_timing: id  0 | task 17 | draft acceptance = 0.85714 (   48 accepted /    56 generated), mean len =  4.43 |
| A-nmax3 | 2 | 45.71 | 609.5 | slot print_timing: id  0 | task 18 | draft acceptance = 0.92000 (   46 accepted /    50 generated), mean len =  3.71 |
| B-nmax4 | 2 | 39.56 | 591.0 | slot print_timing: id  0 | task 17 | draft acceptance = 0.85714 (   48 accepted /    56 generated), mean len =  4.43 |

Δ decode vs A-nmax3 (médiane) :
  - B-nmax4: -13.43%

## ctx 160000
| leg | rep | decode t/s | prefill t/s | acc (dernier slot) |
|---|---|---|---|---|
| A-nmax3 | 1 | 40.87 | 369.6 | slot print_timing: id  0 | task 92 | draft acceptance = 0.92000 (   46 accepted /    50 generated), mean len =  3.71 |
| B-nmax4 | 1 | 36.25 | 362.9 | slot print_timing: id  0 | task 89 | draft acceptance = 0.81356 (   48 accepted /    59 generated), mean len =  4.20 |
| A-nmax3 | 2 | 40.89 | 369.7 | slot print_timing: id  0 | task 92 | draft acceptance = 0.92000 (   46 accepted /    50 generated), mean len =  3.71 |
| B-nmax4 | 2 | 36.25 | 362.9 | slot print_timing: id  0 | task 89 | draft acceptance = 0.81356 (   48 accepted /    59 generated), mean len =  4.20 |

Δ decode vs A-nmax3 (médiane) :
  - B-nmax4: -11.35%

## stats MTP brutes (spec statistics, par leg)
- A-nmax3 rep1 ctx114688: spec common_specu: statistics        draft-mtp: #calls(b,g,a) =    1     14     14, #gen drafts =     14, #acc drafts =    13, #gen tokens =     42, #acc tokens =    38, #mean acc len = 3.71, #acc rate/pos = (0.929, 0.929, 0.857), dur(b,g,a) = 0.003, 109.566, 0.012 ms || spec common_specu: statistics        draft-mtp: #calls(b,g,a) =    2     31     31, #gen drafts =     31, #acc drafts =    30, #gen tokens =     92, #acc tokens =    84, #mean acc len = 3.71, #acc rate/pos = (0.968, 0.903, 0.839), dur(b,g,a) = 0.004, 274.318, 0.024 ms
- A-nmax3 rep1 ctx160000: spec common_specu: statistics        draft-mtp: #calls(b,g,a) =    2     31     31, #gen drafts =     31, #acc drafts =    30, #gen tokens =     92, #acc tokens =    84, #mean acc len = 3.71, #acc rate/pos = (0.968, 0.903, 0.839), dur(b,g,a) = 0.004, 274.318, 0.024 ms || spec common_specu: statistics        draft-mtp: #calls(b,g,a) =    3     48     48, #gen drafts =     48, #acc drafts =    47, #gen tokens =    142, #acc tokens =   130, #mean acc len = 3.71, #acc rate/pos = (0.979, 0.896, 0.833), dur(b,g,a) = 0.004, 453.689, 0.042 ms
- B-nmax4 rep1 ctx114688: spec common_specu: statistics        draft-mtp: #calls(b,g,a) =    1     13     13, #gen drafts =     13, #acc drafts =    11, #gen tokens =     52, #acc tokens =    40, #mean acc len = 4.08, #acc rate/pos = (0.846, 0.769, 0.769, 0.692), dur(b,g,a) = 0.002, 134.437, 0.016 ms || spec common_specu: statistics        draft-mtp: #calls(b,g,a) =    2     27     27, #gen drafts =     27, #acc drafts =    24, #gen tokens =    108, #acc tokens =    88, #mean acc len = 4.26, #acc rate/pos = (0.889, 0.815, 0.815, 0.741), dur(b,g,a) = 0.003, 318.475, 0.028 ms
- B-nmax4 rep1 ctx160000: spec common_specu: statistics        draft-mtp: #calls(b,g,a) =    2     27     27, #gen drafts =     27, #acc drafts =    24, #gen tokens =    108, #acc tokens =    88, #mean acc len = 4.26, #acc rate/pos = (0.889, 0.815, 0.815, 0.741), dur(b,g,a) = 0.003, 318.475, 0.028 ms || spec common_specu: statistics        draft-mtp: #calls(b,g,a) =    3     42     42, #gen drafts =     42, #acc drafts =    38, #gen tokens =    167, #acc tokens =   136, #mean acc len = 4.24, #acc rate/pos = (0.905, 0.810, 0.810, 0.714), dur(b,g,a) = 0.003, 528.777, 0.039 ms
- B-nmax4 rep2 ctx114688: spec common_specu: statistics        draft-mtp: #calls(b,g,a) =    1     13     13, #gen drafts =     13, #acc drafts =    11, #gen tokens =     52, #acc tokens =    40, #mean acc len = 4.08, #acc rate/pos = (0.846, 0.769, 0.769, 0.692), dur(b,g,a) = 0.003, 134.876, 0.014 ms || spec common_specu: statistics        draft-mtp: #calls(b,g,a) =    2     27     27, #gen drafts =     27, #acc drafts =    24, #gen tokens =    108, #acc tokens =    88, #mean acc len = 4.26, #acc rate/pos = (0.889, 0.815, 0.815, 0.741), dur(b,g,a) = 0.004, 318.465, 0.027 ms
- B-nmax4 rep2 ctx160000: spec common_specu: statistics        draft-mtp: #calls(b,g,a) =    2     27     27, #gen drafts =     27, #acc drafts =    24, #gen tokens =    108, #acc tokens =    88, #mean acc len = 4.26, #acc rate/pos = (0.889, 0.815, 0.815, 0.741), dur(b,g,a) = 0.004, 318.465, 0.027 ms || spec common_specu: statistics        draft-mtp: #calls(b,g,a) =    3     42     42, #gen drafts =     42, #acc drafts =    38, #gen tokens =    167, #acc tokens =   136, #mean acc len = 4.24, #acc rate/pos = (0.905, 0.810, 0.810, 0.714), dur(b,g,a) = 0.005, 529.242, 0.040 ms
- A-nmax3 rep2 ctx114688: spec common_specu: statistics        draft-mtp: #calls(b,g,a) =    1     14     14, #gen drafts =     14, #acc drafts =    13, #gen tokens =     42, #acc tokens =    38, #mean acc len = 3.71, #acc rate/pos = (0.929, 0.929, 0.857), dur(b,g,a) = 0.002, 108.588, 0.011 ms || spec common_specu: statistics        draft-mtp: #calls(b,g,a) =    2     31     31, #gen drafts =     31, #acc drafts =    30, #gen tokens =     92, #acc tokens =    84, #mean acc len = 3.71, #acc rate/pos = (0.968, 0.903, 0.839), dur(b,g,a) = 0.003, 273.491, 0.025 ms
- A-nmax3 rep2 ctx160000: spec common_specu: statistics        draft-mtp: #calls(b,g,a) =    2     31     31, #gen drafts =     31, #acc drafts =    30, #gen tokens =     92, #acc tokens =    84, #mean acc len = 3.71, #acc rate/pos = (0.968, 0.903, 0.839), dur(b,g,a) = 0.003, 273.491, 0.025 ms || spec common_specu: statistics        draft-mtp: #calls(b,g,a) =    3     48     48, #gen drafts =     48, #acc drafts =    47, #gen tokens =    142, #acc tokens =   130, #mean acc len = 3.71, #acc rate/pos = (0.979, 0.896, 0.833), dur(b,g,a) = 0.005, 452.436, 0.038 ms
