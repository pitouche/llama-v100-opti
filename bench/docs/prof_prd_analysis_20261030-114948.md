# PROF PRD — analyse captures (run 20261030-114948)

date : 20261030-114948
so : 8035caf4 | ctx ~115k (prompt 112060 tok) | b2048 | KV q8_0 | H2 OFF

| config | phase | wall s | tok/s (prompt / gén.) | ms/tok gén. | kernels | span GPU | busy GPU | top gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MTP OFF | prefill | 184.5 | 617.6 / 21.26 | 47.0 | 179025 | 182.09 s | 98.3 % | 99.3 ms |
| MTP OFF | decode | 6.6 | 21.62 | 46.3 | 180115 | 4.45 s | 95.9 % | 7.2 ms |
| MTP ON | prefill | 187.0 | 619.7 / 3.90 | 256.3 | 176509 | 184.60 s | 97.3 % | 376.1 ms |
| MTP ON | decode | 5.9 | 44.18 | 22.6 | 123324 | 3.65 s | 89.7 % | 7.4 ms |

## MTP OFF — prefill

- span GPU (1er→dernier kernel) : 182.09 s, busy union : 179.04 s (98.3 %)
- kernels : 179025, sync stream : 2112 (68.30 s), memcpy : 1112 (1.23 s)

| classe | # | temps | %GPU |
|---|---:|---:|---:|
| flash_attn | 2272 | 87.90 s | 49.1 |
| autre | 106750 | 81.97 s | 45.8 |
| quant/cvt | 38694 | 5.78 s | 3.2 |
| norm/rms | 21655 | 2.76 s | 1.5 |
| matvec (proj decode) | 7382 | 531.5 ms | 0.3 |
| rope | 2272 | 101.1 ms | 0.1 |

top kernels :
- `flash_attn_ext_f16` ×880 = 87.78 s
- `Kernel2` ×27248 = 65.06 s
- `gated_delta_net_cuda_128x4_volta` ×2688 = 7.58 s
- `convert_unary` ×27280 = 3.69 s
- `dequantize_block_q6_K_grouped` ×12320 = 3.16 s
- `unary_gated_op_kernel` ×9238 = 2.61 s
- `add_rms_norm_mul_f32` ×7832 = 1.52 s
- `dequantize_block_q8_0_f16` ×12650 = 1.38 s

top runtime API :
- `cudaLaunchKernel_v7000` ×127415 = 84.91 s
- `cuKernelGetFunction` ×27248 = 13.5 ms
- `cuLaunchKernel` ×27248 = 25.01 s
- `cudaStreamSynchronize_v3020` ×2112 = 68.30 s
- `cudaMemsetAsync_v3020` ×1776 = 1.45 s
- `cudaMemcpyAsync_v3020` ×1112 = 78.2 ms
- `cuGetProcAddress_v2` ×810 = 856 µs
- `cudaEventCreateWithFlags_v3020` ×18 = 180 µs

gaps GPU >100 µs : 76, total 2.88 s ; plus gros : 99.3 ms, 79.3 ms, 78.3 ms, 76.5 ms, 75.0 ms

## MTP ON — prefill

- span GPU (1er→dernier kernel) : 184.60 s, busy union : 179.59 s (97.3 %)
- kernels : 176509, sync stream : 2834 (64.77 s), memcpy : 1708 (1.91 s)

| classe | # | temps | %GPU |
|---|---:|---:|---:|
| flash_attn | 2056 | 88.40 s | 49.2 |
| autre | 113186 | 82.19 s | 45.8 |
| quant/cvt | 35381 | 5.78 s | 3.2 |
| norm/rms | 19607 | 2.77 s | 1.5 |
| matvec (proj decode) | 4160 | 348.0 ms | 0.2 |
| rope | 2119 | 100.7 ms | 0.1 |

top kernels :
- `flash_attn_ext_f16` ×880 = 88.30 s
- `Kernel2` ×27411 = 65.15 s
- `gated_delta_net_cuda_128x4_volta` ×3024 = 7.62 s
- `convert_unary` ×27445 = 3.71 s
- `dequantize_block_q6_K_grouped` ×12375 = 3.17 s
- `unary_gated_op_kernel` ×8084 = 2.62 s
- `add_rms_norm_mul_f32` ×8001 = 1.52 s
- `dequantize_block_q8_0_f16` ×12760 = 1.38 s

top runtime API :
- `cudaLaunchKernel_v7000` ×138620 = 103.49 s
- `cuKernelGetFunction` ×27258 = 16.1 ms
- `cuLaunchKernel` ×27258 = 8.70 s
- `cudaStreamSynchronize_v3020` ×2834 = 64.78 s
- `cudaMemsetAsync_v3020` ×1800 = 923.0 ms
- `cudaMemcpyAsync_v3020` ×1670 = 66.4 ms
- `cuGetProcAddress_v2` ×810 = 571 µs
- `cudaGraphLaunch_v10000` ×81 = 18.4 ms

gaps GPU >100 µs : 150, total 4.84 s ; plus gros : 376.1 ms, 137.3 ms, 64.7 ms, 52.6 ms, 52.1 ms

## MTP OFF — decode

- span GPU (1er→dernier kernel) : 4.45 s, busy union : 4.27 s (95.9 %)
- kernels : 180115, sync stream : 2501 (4.13 s), memcpy : 1152 (36.7 ms)

| classe | # | temps | %GPU |
|---|---:|---:|---:|
| matvec (proj decode) | 44102 | 3.16 s | 74.1 |
| flash_attn | 3072 | 598.6 ms | 14.0 |
| autre | 53415 | 220.5 ms | 5.2 |
| norm/rms | 29280 | 149.5 ms | 3.5 |
| quant/cvt | 47174 | 125.2 ms | 2.9 |
| rope | 3072 | 11.3 ms | 0.3 |

top kernels :
- `mul_mat_vec_q` ×44102 = 3.16 s
- `flash_attn_ext_mma_volta_small` ×1536 = 581.7 ms
- `quantize_q8_1` ×44102 = 108.1 ms
- `rms_norm_f32` ×15377 = 88.9 ms
- `gated_delta_net_cuda` ×4560 = 48.2 ms
- `unary_gated_op_kernel` ×13238 = 37.8 ms
- `cpy_scalar` ×6144 = 34.8 ms
- `add_rms_norm_mul_f32` ×4687 = 34.3 ms

top runtime API :
- `cudaLaunchKernel_v7000` ×9581 = 61.4 ms
- `cudaStreamSynchronize_v3020` ×2501 = 4.13 s
- `cudaMemcpyAsync_v3020` ×1152 = 30.2 ms
- `cudaGraphLaunch_v10000` ×93 = 189.4 ms
- `cudaStreamBeginCapture_v10000` ×2 = 6 µs
- `cudaGraphDestroy_v10000` ×2 = 1.1 ms
- `cudaStreamEndCapture_v10000` ×2 = 96 µs
- `cudaGraphExecUpdate_v10020` ×2 = 1.1 ms

gaps GPU >100 µs : 95, total 94.8 ms ; plus gros : 7.2 ms, 6.7 ms, 3.7 ms, 2.9 ms, 1.1 ms

## MTP ON — decode

- span GPU (1er→dernier kernel) : 3.65 s, busy union : 3.28 s (89.7 %)
- kernels : 123324, sync stream : 4861 (3.13 s), memcpy : 2849 (41.0 ms)

| classe | # | temps | %GPU |
|---|---:|---:|---:|
| matvec (proj decode) | 27216 | 2.28 s | 69.4 |
| flash_attn | 1968 | 565.5 ms | 17.3 |
| autre | 46108 | 245.8 ms | 7.5 |
| quant/cvt | 28880 | 91.5 ms | 2.8 |
| norm/rms | 17132 | 91.3 ms | 2.8 |
| rope | 2020 | 7.3 ms | 0.2 |

top kernels :
- `mul_mat_vec_q` ×27064 = 2.14 s
- `flash_attn_ext_mma_volta_small` ×984 = 556.2 ms
- `mul_mat_vec_q_mapped_rows` ×152 = 136.5 ms
- `quantize_q8_1` ×27216 = 74.4 ms
- `gated_delta_net_cuda_128x4_volta` ×2496 = 54.4 ms
- `add_rms_norm_mul_f32` ×6604 = 49.1 ms
- `cpy_scalar` ×10968 = 48.1 ms
- `rms_norm_f32` ×5536 = 27.4 ms

top runtime API :
- `cudaLaunchKernel_v7000` ×13504 = 107.4 ms
- `cudaStreamSynchronize_v3020` ×4861 = 3.13 s
- `cudaMemcpyAsync_v3020` ×2547 = 28.9 ms
- `cudaGraphLaunch_v10000` ×249 = 138.9 ms
- `cudaMemsetAsync_v3020` ×12 = 28 µs
- `cudaStreamBeginCapture_v10000` ×4 = 9 µs
- `cudaGraphDestroy_v10000` ×4 = 6.4 ms
- `cudaStreamEndCapture_v10000` ×4 = 78 µs

gaps GPU >100 µs : 255, total 309.9 ms ; plus gros : 7.4 ms, 7.3 ms, 7.3 ms, 7.1 ms, 7.1 ms

## MTP ON — décomposition par cycle (steady-state, après transition)

capture = préfill 4 tok (checkpoint restore) + génération 160 tok ;
steady-state = 246 graph launches après le gap de transition (48 verifies ≈ 160 tok, soit 3.33 tok/verify)

| phase | # | busy/cycle | busy tot |
|---|---:|---:|---:|
| verify | 48 | 56.3 ms | 2.70 s |
| draft | 148 | 2.3 ms | 344.6 ms |
| sampling | 50 | 155 µs | 7.8 ms |

- **wall/launch = 15.5 ms** (busy GPU 12.4 ms + host gap 1.4 ms) ; wall/verify = 79.6 ms (≈5.1 launches/verify) → **41.9 tok/s** (3.33 tok/verify) — cohérent avec 44.18 tok/s mesurés

gaps host par transition (moy / max) :
- sampling → draft : n=50, moy 2.1 ms, max 7.3 ms
- draft → draft : n=98, moy 940 µs, max 6.4 ms
- draft → verify : n=48, moy 1.4 ms, max 7.4 ms
- draft → sampling : n=1, moy 59.2 ms, max 59.2 ms
- verify → sampling : n=48, moy 676 µs, max 872 µs
  - (n=1, 59.2 ms : artefact de transition / hiccup unique, à écarter du budget)

breakdown GPU **à l'intérieur d'un verify** (busy/verify) :
- matvec (proj decode) : 39.5 ms (497 kern/verify)
- flash_attn : 9.1 ms (32 kern/verify)
- autre : 4.3 ms (834 kern/verify)
- quant/cvt : 1.7 ms (529 kern/verify)
- norm/rms : 1.6 ms (305 kern/verify)
- rope : 121 µs (32 kern/verify)

Côté CPU (perf 99 Hz, fenêtre decode MTP ON) : 71 % dans `ggml_graph_compute_secondary_thread` (libggml-cpu, worker spin) + 24 % `cudaStreamSynchronize` — le host **attend le GPU**, pas l'inverse. Les gaps host (~1.4 ms/launch) viennent du sampling + bookkeeping, pas d'un goulot CPU.

## Sampler nvidia-smi 100 ms (busy = échantillons avec util>0)

| fenêtre | échantillons | busy % | util moy |
|---|---:|---:|---:|
| MTP OFF prefill | 2377 | 99 % | 98 |
| MTP ON prefill | 2412 | 99 % | 97 |
| MTP OFF decode | 88 | 72 % | 67 |
| MTP ON decode | 77 | 66 % | 59 |

## Verdict MTP ON vs OFF (ctx ~112k, début de decode)

- decode : MTP OFF 46.30 ms/tok (21.62 t/s) → MTP ON 22.60 ms/tok (44.18 t/s) = **×2.05**
- acceptance MTP (fenêtre decode) : 0.711 (108/152, moy 3.12 pos, per-pos 0.902/0.706/0.510)
- prefill : ~identique (617.6 vs 619.7 t/s) — le MTP ne touche pas le prefill (defer prompt). Le « 3.9 t/s » de la ligne prefill MTP ON = les 16 tok générés après le prompt (1er cycle MTP froid, acceptance 0.40), pas le prompt.

