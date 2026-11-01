# PROF PRD — analyse captures (run 20261030-135141)

date : 20261030-135141
so : 8035caf4 | ctx ~115k (prompt 112060 tok) | b2048 | KV q8_0 | H2 OFF

| config | phase | wall s | tok/s (prompt / gén.) | ms/tok gén. | kernels | span GPU | busy GPU | top gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MTP OFF | prefill | 184.6 | 617.4 / 21.32 | 46.9 | 179025 | 182.13 s | 98.4 % | 100.0 ms |
| MTP OFF | decode | 6.6 | 21.60 | 46.3 | 180115 | 4.45 s | 95.8 % | 7.1 ms |
| MTP ON | prefill | 186.8 | 619.2 / 4.35 | 230.0 | 173221 | 184.34 s | 97.4 % | 386.6 ms |
| MTP ON | decode | 6.1 | 41.94 | 23.8 | 139175 | 3.85 s | 91.6 % | 14.0 ms |

## MTP OFF — prefill

- span GPU (1er→dernier kernel) : 182.13 s, busy union : 179.13 s (98.4 %)
- kernels : 179025, sync stream : 2112 (68.34 s), memcpy : 1112 (1.23 s)

| classe | # | temps | %GPU |
|---|---:|---:|---:|
| flash_attn | 2272 | 87.96 s | 49.1 |
| autre | 106750 | 82.00 s | 45.8 |
| quant/cvt | 38694 | 5.78 s | 3.2 |
| norm/rms | 21655 | 2.76 s | 1.5 |
| matvec (proj decode) | 7382 | 531.4 ms | 0.3 |
| rope | 2272 | 101.3 ms | 0.1 |

top kernels :
- `flash_attn_ext_f16` ×880 = 87.84 s
- `Kernel2` ×27248 = 65.09 s
- `gated_delta_net_cuda_128x4_volta` ×2688 = 7.58 s
- `convert_unary` ×27280 = 3.69 s
- `dequantize_block_q6_K_grouped` ×12320 = 3.16 s
- `unary_gated_op_kernel` ×9238 = 2.61 s
- `add_rms_norm_mul_f32` ×7832 = 1.52 s
- `dequantize_block_q8_0_f16` ×12650 = 1.38 s

top runtime API :
- `cudaLaunchKernel_v7000` ×127415 = 84.96 s
- `cuKernelGetFunction` ×27248 = 14.2 ms
- `cuLaunchKernel` ×27248 = 25.02 s
- `cudaStreamSynchronize_v3020` ×2112 = 68.35 s
- `cudaMemsetAsync_v3020` ×1776 = 1.44 s
- `cudaMemcpyAsync_v3020` ×1112 = 77.6 ms
- `cuGetProcAddress_v2` ×810 = 121 µs
- `cudaEventCreateWithFlags_v3020` ×18 = 13 µs

gaps GPU >100 µs : 75, total 2.83 s ; plus gros : 100.0 ms, 82.1 ms, 79.1 ms, 77.0 ms, 75.8 ms

## MTP ON — prefill

- span GPU (1er→dernier kernel) : 184.34 s, busy union : 179.63 s (97.4 %)
- kernels : 173221, sync stream : 2732 (65.33 s), memcpy : 1636 (1.91 s)

| classe | # | temps | %GPU |
|---|---:|---:|---:|
| flash_attn | 2044 | 88.45 s | 49.2 |
| autre | 110060 | 82.20 s | 45.8 |
| quant/cvt | 35333 | 5.78 s | 3.2 |
| norm/rms | 19565 | 2.77 s | 1.5 |
| matvec (proj decode) | 4112 | 324.1 ms | 0.2 |
| rope | 2107 | 101.0 ms | 0.1 |

top kernels :
- `flash_attn_ext_f16` ×880 = 88.35 s
- `Kernel2` ×27411 = 65.18 s
- `gated_delta_net_cuda_128x4_volta` ×3024 = 7.61 s
- `convert_unary` ×27445 = 3.71 s
- `dequantize_block_q6_K_grouped` ×12375 = 3.17 s
- `unary_gated_op_kernel` ×8078 = 2.62 s
- `add_rms_norm_mul_f32` ×8001 = 1.52 s
- `dequantize_block_q8_0_f16` ×12760 = 1.38 s

top runtime API :
- `cudaLaunchKernel_v7000` ×133593 = 49.66 s
- `cuKernelGetFunction` ×27258 = 15.7 ms
- `cuLaunchKernel` ×27258 = 62.98 s
- `cudaStreamSynchronize_v3020` ×2732 = 65.34 s
- `cudaMemsetAsync_v3020` ×1800 = 41.7 ms
- `cudaMemcpyAsync_v3020` ×1610 = 65.3 ms
- `cuGetProcAddress_v2` ×810 = 142 µs
- `cudaGraphLaunch_v10000` ×77 = 18.9 ms

gaps GPU >100 µs : 145, total 4.55 s ; plus gros : 386.6 ms, 206.8 ms, 91.0 ms, 53.3 ms, 44.4 ms

## MTP OFF — decode

- span GPU (1er→dernier kernel) : 4.45 s, busy union : 4.27 s (95.8 %)
- kernels : 180115, sync stream : 2501 (4.12 s), memcpy : 1152 (36.7 ms)

| classe | # | temps | %GPU |
|---|---:|---:|---:|
| matvec (proj decode) | 44102 | 3.16 s | 74.1 |
| flash_attn | 3072 | 598.4 ms | 14.0 |
| autre | 53415 | 220.5 ms | 5.2 |
| norm/rms | 29280 | 149.6 ms | 3.5 |
| quant/cvt | 47174 | 125.1 ms | 2.9 |
| rope | 3072 | 11.3 ms | 0.3 |

top kernels :
- `mul_mat_vec_q` ×44102 = 3.16 s
- `flash_attn_ext_mma_volta_small` ×1536 = 581.5 ms
- `quantize_q8_1` ×44102 = 108.1 ms
- `rms_norm_f32` ×15377 = 89.0 ms
- `gated_delta_net_cuda` ×4560 = 48.2 ms
- `unary_gated_op_kernel` ×13238 = 37.8 ms
- `cpy_scalar` ×6144 = 34.8 ms
- `add_rms_norm_mul_f32` ×4687 = 34.4 ms

top runtime API :
- `cudaLaunchKernel_v7000` ×9581 = 61.7 ms
- `cudaStreamSynchronize_v3020` ×2501 = 4.12 s
- `cudaMemcpyAsync_v3020` ×1152 = 30.7 ms
- `cudaGraphLaunch_v10000` ×93 = 196.6 ms
- `cudaStreamBeginCapture_v10000` ×2 = 5 µs
- `cudaGraphDestroy_v10000` ×2 = 1.0 ms
- `cudaStreamEndCapture_v10000` ×2 = 91 µs
- `cudaGraphExecUpdate_v10020` ×2 = 1.0 ms

gaps GPU >100 µs : 95, total 97.7 ms ; plus gros : 7.1 ms, 6.7 ms, 3.7 ms, 3.0 ms, 1.2 ms

## MTP ON — decode

- span GPU (1er→dernier kernel) : 3.85 s, busy union : 3.52 s (91.6 %)
- kernels : 139175, sync stream : 4649 (3.39 s), memcpy : 2609 (37.9 ms)

| classe | # | temps | %GPU |
|---|---:|---:|---:|
| matvec (proj decode) | 31460 | 2.44 s | 69.1 |
| flash_attn | 2192 | 620.5 ms | 17.6 |
| autre | 50169 | 255.1 ms | 7.2 |
| norm/rms | 19689 | 104.8 ms | 3.0 |
| quant/cvt | 33412 | 98.6 ms | 2.8 |
| rope | 2253 | 8.4 ms | 0.2 |

top kernels :
- `mul_mat_vec_q` ×31340 = 2.33 s
- `flash_attn_ext_mma_volta_small` ×1096 = 611.3 ms
- `mul_mat_vec_q_mapped_rows` ×120 = 107.7 ms
- `quantize_q8_1` ×31460 = 81.4 ms
- `add_rms_norm_mul_f32` ×7747 = 57.9 ms
- `gated_delta_net_cuda_128x4_volta` ×2928 = 51.6 ms
- `cpy_scalar` ×9880 = 45.2 ms
- `rms_norm_f32` ×6086 = 29.8 ms

top runtime API :
- `cudaLaunchKernel_v7000` ×11035 = 79.6 ms
- `cudaStreamSynchronize_v3020` ×4649 = 3.39 s
- `cudaMemcpyAsync_v3020` ×2371 = 23.6 ms
- `cudaGraphLaunch_v10000` ×236 = 161.5 ms
- `cudaMemsetAsync_v3020` ×12 = 29 µs
- `cudaStreamBeginCapture_v10000` ×4 = 9 µs
- `cudaGraphDestroy_v10000` ×4 = 1.4 ms
- `cudaStreamEndCapture_v10000` ×4 = 73 µs

gaps GPU >100 µs : 241, total 250.2 ms ; plus gros : 14.0 ms, 9.5 ms, 7.3 ms, 7.1 ms, 6.9 ms

## MTP ON — décomposition par cycle (steady-state, après transition)

capture = préfill 4 tok (checkpoint restore) + génération 160 tok ;
steady-state = 234 graph launches après le gap de transition (58 verifies ≈ 160 tok, soit 2.76 tok/verify)

| phase | # | busy/cycle | busy tot |
|---|---:|---:|---:|
| verify | 58 | 53.1 ms | 3.08 s |
| draft | 117 | 2.3 ms | 272.7 ms |
| sampling | 59 | 146 µs | 8.6 ms |

- **wall/launch = 17.2 ms** (busy GPU 14.4 ms + host gap 1.3 ms) ; wall/verify = 69.3 ms (≈4.0 launches/verify) → **39.8 tok/s** (2.76 tok/verify) — cohérent avec 41.94 tok/s mesurés

gaps host par transition (moy / max) :
- sampling → draft : n=58, moy 1.5 ms, max 7.3 ms
- draft → verify : n=58, moy 1.3 ms, max 14.0 ms
- draft → sampling : n=1, moy 57.6 ms, max 57.6 ms
- draft → draft : n=58, moy 679 µs, max 7.1 ms
- verify → sampling : n=58, moy 590 µs, max 966 µs
  - (n=1, 57.6 ms : artefact de transition / hiccup unique, à écarter du budget)

breakdown GPU **à l'intérieur d'un verify** (busy/verify) :
- matvec (proj decode) : 36.9 ms (496 kern/verify)
- flash_attn : 9.0 ms (32 kern/verify)
- autre : 3.9 ms (786 kern/verify)
- norm/rms : 1.6 ms (305 kern/verify)
- quant/cvt : 1.6 ms (529 kern/verify)
- rope : 122 µs (32 kern/verify)

Côté CPU (perf 99 Hz, fenêtre decode MTP ON) : 71 % dans `ggml_graph_compute_secondary_thread` (libggml-cpu, worker spin) + 24 % `cudaStreamSynchronize` — le host **attend le GPU**, pas l'inverse. Les gaps host (~1.3 ms/launch) viennent du sampling + bookkeeping, pas d'un goulot CPU.

## Sampler nvidia-smi 100 ms (busy = échantillons avec util>0)

| fenêtre | échantillons | busy % | util moy |
|---|---:|---:|---:|
| MTP OFF prefill | 2372 | 99 % | 98 |
| MTP ON prefill | 2404 | 99 % | 97 |
| MTP OFF decode | 87 | 72 % | 67 |
| MTP ON decode | 80 | 66 % | 60 |

## Verdict MTP ON vs OFF (ctx ~112k, début de decode)

- decode : MTP OFF 46.30 ms/tok (21.60 t/s) → MTP ON 23.80 ms/tok (41.94 t/s) = **×1.95**
- acceptance MTP (fenêtre decode) : 0.825 (99/120, moy 2.65 pos, per-pos 0.883/0.767)
- prefill : ~identique (617.6 vs 619.7 t/s) — le MTP ne touche pas le prefill (defer prompt). Le « 3.9 t/s » de la ligne prefill MTP ON = les 16 tok générés après le prompt (1er cycle MTP froid, acceptance 0.40), pas le prompt.

