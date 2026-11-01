# PROF PRD — analyse captures (run 20261030-164453)

date : 20261030-164453
so : 8035caf4 | ctx ~115k (prompt 112060 tok) | b2048 | KV q8_0 | H2 OFF

| config | phase | wall s | tok/s (prompt / gén.) | ms/tok gén. | kernels | span GPU | busy GPU | top gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MTP OFF | prefill | 184.5 | 617.8 / 20.75 | 48.2 | 179025 | 182.05 s | 98.3 % | 100.5 ms |
| MTP OFF | decode | 6.7 | 21.59 | 46.3 | 180115 | 4.46 s | 95.8 % | 8.5 ms |
| MTP ON | prefill | 187.0 | 619.2 / 4.14 | 241.8 | 173221 | 184.51 s | 97.4 % | 369.3 ms |
| MTP ON | decode | 6.0 | 42.26 | 23.7 | 139175 | 3.82 s | 92.3 % | 9.4 ms |

## MTP OFF — prefill

- span GPU (1er→dernier kernel) : 182.05 s, busy union : 179.00 s (98.3 %)
- kernels : 179025, sync stream : 2112 (68.28 s), memcpy : 1112 (1.23 s)

| classe | # | temps | %GPU |
|---|---:|---:|---:|
| flash_attn | 2272 | 87.92 s | 49.1 |
| autre | 106750 | 81.91 s | 45.8 |
| quant/cvt | 38694 | 5.78 s | 3.2 |
| norm/rms | 21655 | 2.76 s | 1.5 |
| matvec (proj decode) | 7382 | 531.4 ms | 0.3 |
| rope | 2272 | 101.0 ms | 0.1 |

top kernels :
- `flash_attn_ext_f16` ×880 = 87.80 s
- `Kernel2` ×27248 = 65.00 s
- `gated_delta_net_cuda_128x4_volta` ×2688 = 7.58 s
- `convert_unary` ×27280 = 3.69 s
- `dequantize_block_q6_K_grouped` ×12320 = 3.16 s
- `unary_gated_op_kernel` ×9238 = 2.61 s
- `add_rms_norm_mul_f32` ×7832 = 1.52 s
- `dequantize_block_q8_0_f16` ×12650 = 1.38 s

top runtime API :
- `cudaLaunchKernel_v7000` ×127415 = 84.92 s
- `cuKernelGetFunction` ×27248 = 13.7 ms
- `cuLaunchKernel` ×27248 = 24.99 s
- `cudaStreamSynchronize_v3020` ×2112 = 68.29 s
- `cudaMemsetAsync_v3020` ×1776 = 1.43 s
- `cudaMemcpyAsync_v3020` ×1112 = 78.3 ms
- `cuGetProcAddress_v2` ×810 = 1.3 ms
- `cudaEventCreateWithFlags_v3020` ×18 = 12 µs

gaps GPU >100 µs : 76, total 2.89 s ; plus gros : 100.5 ms, 80.0 ms, 79.7 ms, 77.7 ms, 76.5 ms

## MTP ON — prefill

- span GPU (1er→dernier kernel) : 184.51 s, busy union : 179.66 s (97.4 %)
- kernels : 173221, sync stream : 2732 (65.41 s), memcpy : 1636 (1.91 s)

| classe | # | temps | %GPU |
|---|---:|---:|---:|
| flash_attn | 2044 | 88.43 s | 49.2 |
| autre | 110060 | 82.25 s | 45.8 |
| quant/cvt | 35333 | 5.78 s | 3.2 |
| norm/rms | 19565 | 2.77 s | 1.5 |
| matvec (proj decode) | 4112 | 324.9 ms | 0.2 |
| rope | 2107 | 101.0 ms | 0.1 |

top kernels :
- `flash_attn_ext_f16` ×880 = 88.33 s
- `Kernel2` ×27411 = 65.22 s
- `gated_delta_net_cuda_128x4_volta` ×3024 = 7.62 s
- `convert_unary` ×27445 = 3.72 s
- `dequantize_block_q6_K_grouped` ×12375 = 3.17 s
- `unary_gated_op_kernel` ×8078 = 2.62 s
- `add_rms_norm_mul_f32` ×8001 = 1.52 s
- `dequantize_block_q8_0_f16` ×12760 = 1.38 s

top runtime API :
- `cudaLaunchKernel_v7000` ×133593 = 49.70 s
- `cuKernelGetFunction` ×27258 = 14.3 ms
- `cuLaunchKernel` ×27258 = 62.96 s
- `cudaStreamSynchronize_v3020` ×2732 = 65.41 s
- `cudaMemsetAsync_v3020` ×1800 = 41.7 ms
- `cudaMemcpyAsync_v3020` ×1610 = 65.9 ms
- `cuGetProcAddress_v2` ×810 = 6.4 ms
- `cudaGraphLaunch_v10000` ×77 = 19.6 ms

gaps GPU >100 µs : 145, total 4.69 s ; plus gros : 369.3 ms, 124.3 ms, 55.0 ms, 52.4 ms, 51.8 ms

## MTP OFF — decode

- span GPU (1er→dernier kernel) : 4.46 s, busy union : 4.27 s (95.8 %)
- kernels : 180115, sync stream : 2501 (4.13 s), memcpy : 1152 (36.8 ms)

| classe | # | temps | %GPU |
|---|---:|---:|---:|
| matvec (proj decode) | 44102 | 3.16 s | 74.1 |
| flash_attn | 3072 | 598.5 ms | 14.0 |
| autre | 53415 | 220.5 ms | 5.2 |
| norm/rms | 29280 | 149.7 ms | 3.5 |
| quant/cvt | 47174 | 125.2 ms | 2.9 |
| rope | 3072 | 11.3 ms | 0.3 |

top kernels :
- `mul_mat_vec_q` ×44102 = 3.16 s
- `flash_attn_ext_mma_volta_small` ×1536 = 581.6 ms
- `quantize_q8_1` ×44102 = 108.2 ms
- `rms_norm_f32` ×15377 = 89.1 ms
- `gated_delta_net_cuda` ×4560 = 48.2 ms
- `unary_gated_op_kernel` ×13238 = 37.7 ms
- `cpy_scalar` ×6144 = 34.8 ms
- `add_rms_norm_mul_f32` ×4687 = 34.3 ms

top runtime API :
- `cudaLaunchKernel_v7000` ×9581 = 62.0 ms
- `cudaStreamSynchronize_v3020` ×2501 = 4.13 s
- `cudaMemcpyAsync_v3020` ×1152 = 30.3 ms
- `cudaGraphLaunch_v10000` ×93 = 190.0 ms
- `cudaStreamBeginCapture_v10000` ×2 = 5 µs
- `cudaGraphDestroy_v10000` ×2 = 1.2 ms
- `cudaStreamEndCapture_v10000` ×2 = 114 µs
- `cudaGraphExecUpdate_v10020` ×2 = 1.2 ms

gaps GPU >100 µs : 95, total 99.4 ms ; plus gros : 8.5 ms, 7.2 ms, 3.6 ms, 3.0 ms, 1.1 ms

## MTP ON — decode

- span GPU (1er→dernier kernel) : 3.82 s, busy union : 3.52 s (92.3 %)
- kernels : 139175, sync stream : 4649 (3.39 s), memcpy : 2609 (38.0 ms)

| classe | # | temps | %GPU |
|---|---:|---:|---:|
| matvec (proj decode) | 31460 | 2.44 s | 69.1 |
| flash_attn | 2192 | 620.9 ms | 17.6 |
| autre | 50169 | 255.1 ms | 7.2 |
| norm/rms | 19689 | 104.8 ms | 3.0 |
| quant/cvt | 33412 | 98.8 ms | 2.8 |
| rope | 2253 | 8.4 ms | 0.2 |

top kernels :
- `mul_mat_vec_q` ×31340 = 2.33 s
- `flash_attn_ext_mma_volta_small` ×1096 = 611.7 ms
- `mul_mat_vec_q_mapped_rows` ×120 = 107.7 ms
- `quantize_q8_1` ×31460 = 81.5 ms
- `add_rms_norm_mul_f32` ×7747 = 57.9 ms
- `gated_delta_net_cuda_128x4_volta` ×2928 = 51.6 ms
- `cpy_scalar` ×9880 = 45.2 ms
- `rms_norm_f32` ×6086 = 29.8 ms

top runtime API :
- `cudaLaunchKernel_v7000` ×11035 = 79.8 ms
- `cudaStreamSynchronize_v3020` ×4649 = 3.39 s
- `cudaMemcpyAsync_v3020` ×2371 = 23.5 ms
- `cudaGraphLaunch_v10000` ×236 = 157.2 ms
- `cudaMemsetAsync_v3020` ×12 = 28 µs
- `cudaStreamBeginCapture_v10000` ×4 = 8 µs
- `cudaGraphDestroy_v10000` ×4 = 1.2 ms
- `cudaStreamEndCapture_v10000` ×4 = 82 µs

gaps GPU >100 µs : 241, total 220.4 ms ; plus gros : 9.4 ms, 7.2 ms, 7.1 ms, 7.0 ms, 7.0 ms

## MTP ON — décomposition par cycle (steady-state, après transition)

capture = préfill 4 tok (checkpoint restore) + génération 160 tok ;
steady-state = 234 graph launches après le gap de transition (58 verifies ≈ 160 tok, soit 2.76 tok/verify)

| phase | # | busy/cycle | busy tot |
|---|---:|---:|---:|
| verify | 58 | 53.1 ms | 3.08 s |
| draft | 117 | 2.3 ms | 272.8 ms |
| sampling | 59 | 146 µs | 8.6 ms |

- **wall/launch = 16.9 ms** (busy GPU 14.4 ms + host gap 1.1 ms) ; wall/verify = 68.4 ms (≈4.0 launches/verify) → **40.3 tok/s** (2.76 tok/verify) — cohérent avec 42.26 tok/s mesurés

gaps host par transition (moy / max) :
- sampling → draft : n=58, moy 1.7 ms, max 7.2 ms
- draft → sampling : n=1, moy 56.7 ms, max 56.7 ms
- draft → verify : n=58, moy 883 µs, max 9.4 ms
- verify → sampling : n=58, moy 565 µs, max 734 µs
- draft → draft : n=58, moy 531 µs, max 6.4 ms
  - (n=1, 56.7 ms : artefact de transition / hiccup unique, à écarter du budget)

breakdown GPU **à l'intérieur d'un verify** (busy/verify) :
- matvec (proj decode) : 37.0 ms (497 kern/verify)
- flash_attn : 9.0 ms (32 kern/verify)
- autre : 3.9 ms (786 kern/verify)
- norm/rms : 1.6 ms (305 kern/verify)
- quant/cvt : 1.6 ms (529 kern/verify)
- rope : 122 µs (32 kern/verify)

Côté CPU (perf 99 Hz, fenêtre decode MTP ON) : 71 % dans `ggml_graph_compute_secondary_thread` (libggml-cpu, worker spin) + 24 % `cudaStreamSynchronize` — le host **attend le GPU**, pas l'inverse. Les gaps host (~1.1 ms/launch) viennent du sampling + bookkeeping, pas d'un goulot CPU.

## Sampler nvidia-smi 100 ms (busy = échantillons avec util>0)

| fenêtre | échantillons | busy % | util moy |
|---|---:|---:|---:|
| MTP OFF prefill | 2375 | 99 % | 98 |
| MTP ON prefill | 2413 | 99 % | 97 |
| MTP OFF decode | 88 | 72 % | 67 |
| MTP ON decode | 79 | 67 % | 62 |

## Verdict MTP ON vs OFF (ctx ~112k, début de decode)

- decode : MTP OFF 46.30 ms/tok (21.59 t/s) → MTP ON 23.70 ms/tok (42.26 t/s) = **×1.95**
- acceptance MTP (fenêtre decode) : 0.825 (99/120, moy 2.65 pos, per-pos 0.883/0.767)
- prefill : ~identique (617.6 vs 619.7 t/s) — le MTP ne touche pas le prefill (defer prompt). Le « 3.9 t/s » de la ligne prefill MTP ON = les 16 tok générés après le prompt (1er cycle MTP froid, acceptance 0.40), pas le prompt.

