# PROF PRD — analyse captures (run 20261030-142401)

date : 20261030-142401
so : 8035caf4 | ctx ~115k (prompt 112060 tok) | b2048 | KV q8_0 | H2 OFF

| config | phase | wall s | tok/s (prompt / gén.) | ms/tok gén. | kernels | span GPU | busy GPU | top gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MTP OFF | prefill | 184.5 | 617.5 / 21.30 | 46.9 | 179025 | 182.12 s | 98.3 % | 102.4 ms |
| MTP OFF | decode | 6.6 | 21.62 | 46.3 | 180115 | 4.45 s | 95.9 % | 8.9 ms |
| MTP ON | prefill | 194.3 | 594.7 / 4.29 | 233.0 | 171705 | 191.86 s | 93.2 % | 416.7 ms |
| MTP ON | decode | 6.9 | 33.86 | 29.5 | 100414 | 4.62 s | 65.7 % | 17.4 ms |

## MTP OFF — prefill

- span GPU (1er→dernier kernel) : 182.12 s, busy union : 179.08 s (98.3 %)
- kernels : 179025, sync stream : 2112 (68.33 s), memcpy : 1112 (1.23 s)

| classe | # | temps | %GPU |
|---|---:|---:|---:|
| flash_attn | 2272 | 87.96 s | 49.1 |
| autre | 106750 | 81.95 s | 45.8 |
| quant/cvt | 38694 | 5.78 s | 3.2 |
| norm/rms | 21655 | 2.76 s | 1.5 |
| matvec (proj decode) | 7382 | 532.0 ms | 0.3 |
| rope | 2272 | 101.1 ms | 0.1 |

top kernels :
- `flash_attn_ext_f16` ×880 = 87.84 s
- `Kernel2` ×27248 = 65.04 s
- `gated_delta_net_cuda_128x4_volta` ×2688 = 7.58 s
- `convert_unary` ×27280 = 3.69 s
- `dequantize_block_q6_K_grouped` ×12320 = 3.16 s
- `unary_gated_op_kernel` ×9238 = 2.61 s
- `add_rms_norm_mul_f32` ×7832 = 1.52 s
- `dequantize_block_q8_0_f16` ×12650 = 1.38 s

top runtime API :
- `cudaLaunchKernel_v7000` ×127415 = 84.95 s
- `cuKernelGetFunction` ×27248 = 12.8 ms
- `cuLaunchKernel` ×27248 = 25.00 s
- `cudaStreamSynchronize_v3020` ×2112 = 68.33 s
- `cudaMemsetAsync_v3020` ×1776 = 1.44 s
- `cudaMemcpyAsync_v3020` ×1112 = 77.3 ms
- `cuGetProcAddress_v2` ×810 = 126 µs
- `cudaEventCreateWithFlags_v3020` ×18 = 12 µs

gaps GPU >100 µs : 75, total 2.88 s ; plus gros : 102.4 ms, 80.4 ms, 78.9 ms, 78.0 ms, 76.2 ms

## MTP ON — prefill

- span GPU (1er→dernier kernel) : 191.86 s, busy union : 178.81 s (93.2 %)
- kernels : 171705, sync stream : 9466 (72.24 s), memcpy : 4023 (8.17 s)

| classe | # | temps | %GPU |
|---|---:|---:|---:|
| flash_attn | 1948 | 87.87 s | 49.1 |
| autre | 112972 | 82.10 s | 45.9 |
| quant/cvt | 33657 | 5.78 s | 3.2 |
| norm/rms | 18588 | 2.73 s | 1.5 |
| matvec (proj decode) | 2532 | 235.7 ms | 0.1 |
| rope | 2008 | 100.5 ms | 0.1 |

top kernels :
- `flash_attn_ext_f16` ×880 = 87.79 s
- `Kernel2` ×27411 = 65.11 s
- `gated_delta_net_cuda_128x4_volta` ×2870 = 7.60 s
- `convert_unary` ×27445 = 3.68 s
- `dequantize_block_q6_K_grouped` ×12375 = 3.17 s
- `unary_gated_op_kernel` ×7674 = 2.60 s
- `add_rms_norm_mul_f32` ×7386 = 1.47 s
- `dequantize_block_q8_0_f16` ×12760 = 1.38 s

top runtime API :
- `cudaLaunchKernel_v7000` ×137824 = 49.24 s
- `cuKernelGetFunction` ×26595 = 13.1 ms
- `cuLaunchKernel` ×26595 = 60.71 s
- `cudaStreamSynchronize_v3020` ×9466 = 72.24 s
- `cudaMemcpyAsync_v3020` ×3997 = 2.67 s
- `cudaMemsetAsync_v3020` ×1800 = 332.0 ms
- `cudaGraphLaunch_v10000` ×1425 = 28.8 ms
- `cuGetProcAddress_v2` ×810 = 121 µs

gaps GPU >100 µs : 1140, total 12.86 s ; plus gros : 416.7 ms, 104.8 ms, 58.6 ms, 57.8 ms, 57.7 ms

## MTP OFF — decode

- span GPU (1er→dernier kernel) : 4.45 s, busy union : 4.27 s (95.9 %)
- kernels : 180115, sync stream : 2501 (4.12 s), memcpy : 1152 (36.9 ms)

| classe | # | temps | %GPU |
|---|---:|---:|---:|
| matvec (proj decode) | 44102 | 3.16 s | 74.1 |
| flash_attn | 3072 | 598.5 ms | 14.0 |
| autre | 53415 | 220.4 ms | 5.2 |
| norm/rms | 29280 | 149.5 ms | 3.5 |
| quant/cvt | 47174 | 125.1 ms | 2.9 |
| rope | 3072 | 11.3 ms | 0.3 |

top kernels :
- `mul_mat_vec_q` ×44102 = 3.16 s
- `flash_attn_ext_mma_volta_small` ×1536 = 581.6 ms
- `quantize_q8_1` ×44102 = 108.1 ms
- `rms_norm_f32` ×15377 = 88.9 ms
- `gated_delta_net_cuda` ×4560 = 48.2 ms
- `unary_gated_op_kernel` ×13238 = 37.8 ms
- `cpy_scalar` ×6144 = 34.8 ms
- `add_rms_norm_mul_f32` ×4687 = 34.3 ms

top runtime API :
- `cudaLaunchKernel_v7000` ×9581 = 62.2 ms
- `cudaStreamSynchronize_v3020` ×2501 = 4.13 s
- `cudaMemcpyAsync_v3020` ×1152 = 30.8 ms
- `cudaGraphLaunch_v10000` ×93 = 194.2 ms
- `cudaStreamBeginCapture_v10000` ×2 = 6 µs
- `cudaGraphDestroy_v10000` ×2 = 1.2 ms
- `cudaStreamEndCapture_v10000` ×2 = 71 µs
- `cudaGraphExecUpdate_v10020` ×2 = 1.1 ms

gaps GPU >100 µs : 95, total 94.4 ms ; plus gros : 8.9 ms, 6.6 ms, 3.7 ms, 2.6 ms, 1.1 ms

## MTP ON — decode

- span GPU (1er→dernier kernel) : 4.62 s, busy union : 3.04 s (65.7 %)
- kernels : 100414, sync stream : 5482 (2.92 s), memcpy : 3119 (43.0 ms)

| classe | # | temps | %GPU |
|---|---:|---:|---:|
| matvec (proj decode) | 21640 | 2.12 s | 69.7 |
| flash_attn | 1672 | 547.0 ms | 18.0 |
| autre | 38698 | 217.9 ms | 7.2 |
| quant/cvt | 22984 | 75.5 ms | 2.5 |
| norm/rms | 13706 | 73.1 ms | 2.4 |
| rope | 1714 | 6.3 ms | 0.2 |

top kernels :
- `mul_mat_vec_q` ×21476 = 1.97 s
- `flash_attn_ext_mma_volta_small` ×836 = 538.2 ms
- `mul_mat_vec_q_mapped_rows` ×164 = 147.2 ms
- `quantize_q8_1` ×21640 = 60.2 ms
- `gated_delta_net_cuda_128x4_volta` ×1932 = 50.9 ms
- `cpy_scalar` ×10496 = 45.2 ms
- `add_rms_norm_mul_f32` ×5208 = 38.3 ms
- `rms_norm_f32` ×4634 = 23.2 ms

top runtime API :
- `cudaLaunchKernel_v7000` ×11151 = 91.0 ms
- `cudaStreamSynchronize_v3020` ×5482 = 2.92 s
- `cudaMemcpyAsync_v3020` ×2793 = 27.2 ms
- `cudaGraphLaunch_v10000` ×322 = 130.8 ms
- `cudaGraphDestroy_v10000` ×30 = 1.2 ms
- `cudaGraphExecDestroy_v10000` ×24 = 276 µs
- `cudaMemsetAsync_v3020` ×12 = 29 µs
- `cudaStreamBeginCapture_v10000` ×6 = 17 µs

gaps GPU >100 µs : 331, total 1.53 s ; plus gros : 17.4 ms, 17.2 ms, 16.0 ms, 15.7 ms, 15.5 ms

## MTP ON — décomposition par cycle (steady-state, après transition)

capture = préfill 4 tok (checkpoint restore) + génération 155 tok ;
steady-state = 144 graph launches après le gap de transition (24 verifies ≈ 155 tok, soit 6.46 tok/verify)

| phase | # | busy/cycle | busy tot |
|---|---:|---:|---:|
| verify | 24 | 63.4 ms | 1.52 s |
| draft | 95 | 2.3 ms | 221.4 ms |
| sampling | 25 | 179 µs | 4.5 ms |

- **wall/launch = 24.5 ms** (busy GPU 12.1 ms + host gap 6.1 ms) ; wall/verify = 147.2 ms (≈6.0 launches/verify) → **43.9 tok/s** (6.46 tok/verify) — cohérent avec 33.86 tok/s mesurés

gaps host par transition (moy / max) :
- draft → verify : n=24, moy 31.1 ms, max 36.9 ms
- sampling → draft : n=24, moy 2.5 ms, max 8.0 ms
- draft → draft : n=71, moy 705 µs, max 6.3 ms
- verify → sampling : n=24, moy 869 µs, max 973 µs

breakdown GPU **à l'intérieur d'un verify** (busy/verify) :
- matvec (proj decode) : 44.8 ms (481 kern/verify)
- flash_attn : 10.7 ms (32 kern/verify)
- autre : 4.5 ms (850 kern/verify)
- quant/cvt : 1.7 ms (513 kern/verify)
- norm/rms : 1.6 ms (295 kern/verify)
- rope : 125 µs (32 kern/verify)

Côté CPU (perf 99 Hz, fenêtre decode MTP ON) : 71 % dans `ggml_graph_compute_secondary_thread` (libggml-cpu, worker spin) + 24 % `cudaStreamSynchronize` — le host **attend le GPU**, pas l'inverse. Les gaps host (~6.1 ms/launch) viennent du sampling + bookkeeping, pas d'un goulot CPU.

## Sampler nvidia-smi 100 ms (busy = échantillons avec util>0)

| fenêtre | échantillons | busy % | util moy |
|---|---:|---:|---:|
| MTP OFF prefill | 2376 | 99 % | 98 |
| MTP ON prefill | 2504 | 99 % | 96 |
| MTP OFF decode | 87 | 72 % | 67 |
| MTP ON decode | 88 | 68 % | 46 |

## Verdict MTP ON vs OFF (ctx ~112k, début de decode)

- decode : MTP OFF 46.30 ms/tok (21.62 t/s) → MTP ON 29.50 ms/tok (33.86 t/s) = **×1.57**
- acceptance MTP (fenêtre decode) : 0.701 (115/164, moy 3.80 pos, per-pos 0.951/0.732/0.610/0.512)
- prefill : ~identique (617.6 vs 619.7 t/s) — le MTP ne touche pas le prefill (defer prompt). Le « 3.9 t/s » de la ligne prefill MTP ON = les 16 tok générés après le prompt (1er cycle MTP froid, acceptance 0.40), pas le prompt.

