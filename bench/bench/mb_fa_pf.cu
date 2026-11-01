// (c) 2026 guile — MIT. mb_fa_pf.cu — 30/10
//
// Microbench PREFILL du kernel FA f16 de la V100 — SANS rebuild du .so prod,
// SANS toucher au service ollama-v100.
//
// Principe : dlopen du .so prod (libggml-cuda.so md5 8035caf4…) + libggml-base.so.0,
//   puis pilotage de ggml_cuda_flash_attn_ext(ctx, dst) — le MÊME dispatch que la prod.
//   Contrairement à mb_fa.cu (decode T=1..8, KV q8_0), ici on reproduit le PREFILL :
//     - K / V en F16 (cache-type-k/v f16 pour le chemin mma-f16 prod)
//     - T paramétrable (défaut 2048 = taille de chunk prod -b 2048)
//     - n_kv long (défaut 114688 = 56*2048, ≈ prompt prod ~115k, pad 256)
//
// Le dispatch mène au kernel prod :
//     flash_attn_ext_f16<256,256,32,2,false,false,false,false>
//   (D=256, 24 qh, 4 kvh, gqa 6, T=2048 -> grid (768,1,1), block (32,4,1), 255 regs,
//    smem dyn 67584 B). Confirmed against the prod nsys trace.
//
// CONVENTION MASK (documentée, à garder en tête dans l'analyse) :
//   mask F16 = 0.0 partout (AUCUNE -inf) -> le pre-pass flash_attn_mask_to_KV_max ne
//   trouve aucun KV "totalement maské" -> KV_max = n_kv complet -> le kernel FA fait le
//   travail "full attention" (CHACUNE des T queries scanne tout le KV). C'est le cas de
//   charge MAJORE (pas causal). Pour regs/occupancy/tensor-core c'est le régime
//   stationnaire le plus pur ; la DURÉE ne sera PAS égale à la prod causale (~2x moins
//   de FLOPs attention en causal). Le mask=0 est volontairement conservé.
//
// GPU : on ne force PAS CUDA_VISIBLE_DEVICES ici. Le wrapper shell résout le V100 par
//   UUID (gpu_uuid.py --cvt-ordinal) et fixe CVD AVANT l'exec ; le code use device 0
//   (= le V100 après filtrage CVD). Jamais d'index en dur.
//
// Usage :
//   CVD=$(python3 gpu_uuid.py --cvt-ordinal); CUDA_VISIBLE_DEVICES=$CVD ./mb_fa_pf [T] [n_kv] [iters]
//   T     défaut 2048 ; n_kv défaut 114688 ; iters défaut 3 (préfill = gros kernels)
// Sortie : JSONL stdout, 1 ligne : {"T":..,"n_kv":..,"us_med":..,"us_min":..,"us_max":..,"out":[...]}
//   out0..out3 = 4 premiers floats de la sortie (fini, borné, pas NaN/inf attendu).

#include <dlfcn.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>

#define CK(x) do{ cudaError_t e_=(x); if(e_){fprintf(stderr,"CUDA err @%d: %s\n",__LINE__,cudaGetErrorString(e_));exit(1);} }while(0)

// ---------------------------------------------------------------- types ggml minimaux
// (mêmes layouts EXACTS que mb_fa.cu — ggml-impl.h / ggml-backend-impl.h, llama-volta-merged)
struct ggml_tensor {
    int32_t  type;
    void    *buffer;
    int64_t  ne[4];
    size_t   nb[4];
    int32_t  op;
    int32_t  op_params[16];
    int32_t  flags;
    ggml_tensor *src[10];
    ggml_tensor *view_src;
    size_t   view_offs;
    void    *data;
    char     name[64];
    void    *extra;
    char     padding[8];
};
static_assert(sizeof(ggml_tensor) % 8 == 0, "align 8");
struct ggml_context;     // opaque
typedef struct ggml_backend_buffer ggml_backend_buffer;
typedef ggml_backend_buffer * ggml_backend_buffer_t;

enum ggml_type_e { GGML_TYPE_F32=0, GGML_TYPE_F16=1 };

struct ggml_init_params { size_t mem_size; void * mem_buffer; bool no_alloc; };

// ---------------------------------------------------------------- API ggml (ptrs)
typedef struct ggml_context * (*f_ggml_init)(ggml_init_params);
typedef ggml_backend_buffer_t  (*f_buft_alloc_buffer)(void *, size_t);
typedef void *                 (*f_buf_get_base)(ggml_backend_buffer_t);
typedef void                   (*f_buf_free)(ggml_backend_buffer_t);
typedef void                   (*f_backend_free)(void *);

// .so CUDA chargé : défaut = prod ; override via env MB_FA_SO.
static const char * SO_CUDA() {
    static const char * p = getenv("MB_FA_SO") ? getenv("MB_FA_SO")
        : "/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so";
    return p;
}
static const char * SO_BASE = "/home/guile/llama-experiments/ollama-v100/lib/ollama/libggml-base.so.0";

// ggml_cuda_flash_attn_ext(ggml_backend_cuda_context&, ggml_tensor*)  [symbole exporté]
static void (*g_fa_ext)(void *, ggml_tensor *) = nullptr;

struct Env {
    void * hbase, * hcuda;
    f_ggml_init           ggml_init;
    f_buft_alloc_buffer   buft_alloc_buffer;
    f_buf_get_base        buf_get_base;
    f_buf_free            buf_free;
    f_backend_free        backend_free;
    void   * backend = nullptr;
    void   * ctx_cuda = nullptr;
    cudaStream_t stream = nullptr;
    ggml_context * gctx = nullptr;
    ggml_backend_buffer_t buft_buf = nullptr;
    void   * base = nullptr;
    void   * buft = nullptr;
};

static void *must_dlsym(void *h, const char *name) {
    void *p = dlsym(h, name);
    if (!p) { fprintf(stderr, "dlsym %s manquant : %s\n", name, dlerror()); exit(1); }
    return p;
}

static void env_open(Env & E) {
    E.hbase = dlopen(SO_BASE, RTLD_NOW | RTLD_LOCAL);
    E.hcuda = dlopen(SO_CUDA(), RTLD_NOW | RTLD_LOCAL);
    if (!E.hbase || !E.hcuda) { fprintf(stderr, "dlopen échoué (%s) : %s\n", SO_CUDA(), dlerror()); exit(1); }
    E.ggml_init         = (f_ggml_init)must_dlsym(E.hbase, "ggml_init");
    E.buf_free          = (f_buf_free)must_dlsym(E.hbase, "ggml_backend_buffer_free");
    E.backend_free      = (f_backend_free)must_dlsym(E.hbase, "ggml_backend_free");
    g_fa_ext            = (void (*)(void*, ggml_tensor*))must_dlsym(E.hcuda, "_Z24ggml_cuda_flash_attn_extR25ggml_backend_cuda_contextP11ggml_tensor");

    void *init_p = must_dlsym(E.hcuda, "ggml_backend_cuda_init");
    E.backend = ((ggml_backend_buffer_t(*)(int))init_p)(0);   // device 0 = V100 (CVD fixé par le wrapper)
    if (!E.backend) { fprintf(stderr,"ggml_backend_cuda_init(0) a échoué\n"); exit(1); }

    // ggml_backend = { guid(8), iface(16 ptrs=128B), device(8B)@136, context(8B)@144 }
    E.ctx_cuda = *(void **)((char *)E.backend + 144);
    void * dev = *(void **)((char *)E.backend + 136);
    void * get_buft = *(void **)((char *)dev + 6 * sizeof(void *));   // dev_i.get_buffer_type (7e slot)
    void * buft = ((void *(*)(void *))get_buft)(dev);
    E.buft_alloc_buffer = (f_buft_alloc_buffer)(*(void **)((char *)buft + sizeof(void *)));
    E.buft = buft;
    if (!E.ctx_cuda) { fprintf(stderr,"ctx_cuda null (offset 144 mauvais?)\n"); exit(1); }
    fprintf(stderr, "[mb_pf] route ok (backend=%p ctx=%p)\n", E.backend, E.ctx_cuda);
}

static void env_close(Env & E) {
    if (E.buft_buf)  E.buf_free(E.buft_buf);
    if (E.gctx)      ((void(*)(ggml_context*))must_dlsym(E.hbase,"ggml_free"))(E.gctx);
    if (E.backend)   E.backend_free(E.backend);
    if (E.hcuda)     dlclose(E.hcuda);
    if (E.hbase)     dlclose(E.hbase);
}

struct Geo { int D=256, qh=24, kvh=4; int n_kv=114688, n_kv_pad=114688; float scale; };

int main(int argc, char ** argv) {
    int T      = argc > 1 ? atoi(argv[1]) : 2048;
    int n_kv   = argc > 2 ? atoi(argv[2]) : 114688;
    int iters  = argc > 3 ? atoi(argv[3]) : 3;
    Geo g; g.n_kv = n_kv; g.n_kv_pad = ((n_kv + 255) / 256) * 256;
    g.scale = 1.0f / sqrtf((float)g.D);

    // device 0 = V100 (CVD résolu par UUID dans le wrapper). Pas de setenv ici.
    CK(cudaSetDevice(0));
    int dev; CK(cudaGetDevice(&dev));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
    fprintf(stderr, "[mb_pf] device %d = %s (SM %d.%d, %d SM, %.1f GB)\n",
            dev, prop.name, prop.major, prop.minor, prop.multiProcessorCount,
            prop.totalGlobalMem/1073741824.0);
    fprintf(stderr, "[mb_pf] T=%d n_kv=%d (pad %d) iters=%d\n", T, n_kv, g.n_kv_pad, iters);

    Env E;
    env_open(E);

    // --- tailles (KV F16, mask F16) ---
    size_t nq8  = (size_t)g.D*T*g.qh*4;                 // Q  f32 [D,T,qh]
    size_t nkv  = (size_t)g.D*g.n_kv_pad*g.kvh*2;       // K  f16 [D,n_kv_pad,kvh]  (2 octets)
    size_t nm8  = (size_t)T*g.n_kv_pad*2;               // mask f16 [n_kv_pad,T]
    size_t nout8= (size_t)g.D*g.qh*T*4;                 // out f32 [D,qh,T]
    // marge après out : temporaires du pool (KV_max négligeable ; pas de f16 K/V convert à fournir)
    const size_t OUT_MARGIN = 64ull*1024*1024;
    size_t total = nq8 + nkv + nkv + nm8 + nout8 + OUT_MARGIN;

    E.buft_buf = E.buft_alloc_buffer(E.buft, total);
    if (!E.buft_buf) { fprintf(stderr,"alloc buffer (%zu oct) échoué\n", total); return 1; }
    E.buf_get_base = (f_buf_get_base)(*(void **)((char *)E.buft_buf + 1 * sizeof(void *)));   // buffer_i.get_base (2e slot)
    E.base = E.buf_get_base(E.buft_buf);
    if (!E.base) { fprintf(stderr,"get_base nul\n"); return 1; }
    fprintf(stderr, "[mb_pf] buffer %zu octets (%.1f MB), base=%p (Q=%.1f KV=%.1f mask=%.1f out=%.1f MB)\n",
            total, total/1048576.0, E.base,
            (double)nq8/1048576.0, (double)nkv/1048576.0, (double)nm8/1048576.0, (double)nout8/1048576.0);

    // --- données (déterministe + borné) ---
    std::vector<float>  hq(nq8/4);
    std::vector<__half> hkv(nkv/2);
    std::vector<__half> hm(nm8/2);
    srand(98765);
    for (auto & x : hq)  x = (rand()/(float)RAND_MAX - 0.5f);
    for (auto & x : hkv) x = __float2half((rand()/(float)RAND_MAX - 0.5f));
    for (auto & x : hm)  x = __float2half(0.0f);   // mask=0 -> full attention (voir en-tête)
    CK(cudaMemcpy(E.base,                    hq.data(),  nq8,  cudaMemcpyHostToDevice));
    CK(cudaMemcpy((char*)E.base + nq8,       hkv.data(), nkv,  cudaMemcpyHostToDevice));  // K
    CK(cudaMemcpy((char*)E.base + nq8 + nkv, hkv.data(), nkv,  cudaMemcpyHostToDevice));  // V
    CK(cudaMemcpy((char*)E.base + nq8 + 2*nkv, hm.data(), nm8, cudaMemcpyHostToDevice));

    ggml_init_params p { 0, nullptr, true };
    E.gctx = E.ggml_init(p);
    if (!E.gctx) { fprintf(stderr,"ggml_init échoué\n"); return 1; }

    auto mk = [&](int type, int64_t ne0, int64_t ne1, int64_t ne2, size_t nb0, size_t nb1, void *data) -> ggml_tensor * {
        ggml_tensor * t = (ggml_tensor *)calloc(1, sizeof(ggml_tensor));
        t->type = type; t->buffer = (void *)E.buft_buf;
        t->ne[0]=ne0; t->ne[1]=ne1; t->ne[2]=ne2; t->ne[3]=1;
        t->nb[0]=nb0; t->nb[1]=nb1; t->nb[2]=nb1*ne1; t->nb[3]=nb1*ne1*ne2;
        t->data = data;
        return t;
    };
    ggml_tensor * q   = mk(GGML_TYPE_F32, g.D, T, g.qh, 4, (size_t)g.D*4, E.base);
    ggml_tensor * k   = mk(GGML_TYPE_F16, g.D, g.n_kv_pad, g.kvh, 2, (size_t)g.D*2, (char*)E.base + nq8);
    ggml_tensor * v   = mk(GGML_TYPE_F16, g.D, g.n_kv_pad, g.kvh, 2, (size_t)g.D*2, (char*)E.base + nq8 + nkv);
    ggml_tensor * m   = mk(GGML_TYPE_F16, g.n_kv_pad, T, 1, 2, (size_t)g.n_kv_pad*2, (char*)E.base + nq8 + 2*nkv);
    ggml_tensor * out = mk(GGML_TYPE_F32, g.D, g.qh, T, 4, (size_t)g.D*4, (char*)E.base + nq8 + 2*nkv + nm8);

    out->op = 74; // GGML_OP_FLASH_ATTN_EXT
    float params[3] = { g.scale, 0.0f, 0.0f };
    memcpy(out->op_params, params, 12);
    out->src[0]=q; out->src[1]=k; out->src[2]=v; out->src[3]=m; out->src[4]=nullptr;

    // warmup (2 iters hors mesure) — crée le stream ggml (lazy) + heuristique cuBLAS/occupancy
    for (int w=0; w<2; ++w) g_fa_ext(E.ctx_cuda, out);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    // relecture du stream ggml (créé lazily au 1er launch)
    cudaStream_t gst = *(cudaStream_t *)((char *)E.ctx_cuda + 48);
    if (!gst) { fprintf(stderr,"stream ggml toujours null après warmup — offset 48 mauvais\n"); return 1; }
    E.stream = gst;
    { cudaEvent_t se; CK(cudaEventCreate(&se)); CK(cudaEventRecord(se, gst));
      CK(cudaEventSynchronize(se)); CK(cudaEventDestroy(se)); }

    cudaEvent_t ev0, ev1; CK(cudaEventCreate(&ev0)); CK(cudaEventCreate(&ev1));
    std::vector<float> samples;
    for (int it=0; it<iters; ++it) {
        CK(cudaEventRecord(ev0, E.stream));
        g_fa_ext(E.ctx_cuda, out);
        CK(cudaEventRecord(ev1, E.stream));
        CK(cudaEventSynchronize(ev1));
        CK(cudaGetLastError());
        float ms; CK(cudaEventElapsedTime(&ms, ev0, ev1));
        samples.push_back(ms*1000.0f);
    }
    std::sort(samples.begin(), samples.end());
    float us_med = samples[samples.size()/2];
    float us_min = samples.front();
    float us_max = samples.back();

    float o[4] = {0,0,0,0};
    CK(cudaMemcpy(o, (char*)E.base + nq8 + 2*nkv + nm8, 16, cudaMemcpyDeviceToHost));
    bool finite = std::isfinite(o[0]) && std::isfinite(o[1]) && std::isfinite(o[2]) && std::isfinite(o[3]);

    printf("{\"T\":%d,\"n_kv\":%d,\"us_med\":%.1f,\"us_min\":%.1f,\"us_max\":%.1f,\"finite\":%s,\"out\":[%.6g,%.6g,%.6g,%.6g]}\n",
           T, g.n_kv_pad, us_med, us_min, us_max, finite?"true":"false", o[0], o[1], o[2], o[3]);
    fflush(stdout);

    env_close(E);
    return 0;
}
