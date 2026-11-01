// (c) 2026 guile — MIT. Microbench kernel FA rows 1-8 (wmma stock / H2 BC=8 / T3) — 21/10.
//
// mb_fa.cu — harnais standalone des kernels FA q8_0 de la V100, SANS rebuild du .so prod.
//
// Principe : dlopen du .so prod T3 (libggml-cuda.so) + libggml-base.so.0, puis pilotage de
//   ggml_cuda_flash_attn_ext(ctx, dst)  (symbole exporté, T) qui fait le MÊME dispatch que
//   la prod (ggml_cuda_get_best_fattn_kernel). Une route par processus, séparée par env
//   (forcé au démarrage du processus, avant le 1er appel) :
//     t3    : GGML_CUDA_FA_NO_VOLTA_SMALL=0 (défaut) + GGML_CUDA_VOLTA_Q8_FATTN_TC=0 -> T3 (rows 1-8)
//     h2    : GGML_CUDA_FA_NO_VOLTA_SMALL=1  + GGML_CUDA_VOLTA_Q8_FATTN_TC=1 -> H2 (rows==4 uniquement)
//     stock : les deux off -> vec (rows*gqa_eff<=2) / tile (sinon) = le "wmma stock"
//   Le .so est le même partout : la différence mesure le kernel, pas le binaire.
//
// Géométrie prod (Qwen3.8-27B decode/verify) : D=256, 24 qh, 4 kvh (gqa 6), KV q8_0, Q f32,
//   mask f16 = 0, n_kv paramétrable (défaut 107351, pad 256 -> 107520).
//
// Usage : ./mb_fa <mode> [iters] [n_kv]
//   mode : t3 | h2 | stock | probe
// Sortie : JSONL stdout, 1 ligne par rows (T=1..8, ou T=4 pour probe).
//   out0..out3 = 4 premiers floats de la sortie (validation inter-kernel : les routes doivent
//   donner la MÊME attention -> mêmes floats, tolérance fp16 ~1e-2 relatif).

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
#include <string>

#define CK(x) do{ cudaError_t e_=(x); if(e_){fprintf(stderr,"CUDA err @%d: %s\n",__LINE__,cudaGetErrorString(e_));exit(1);} }while(0)

// ---------------------------------------------------------------- types ggml minimaux
// Layouts EXACTS de ggml-base .so.0 (ggml-impl.h / ggml-backend-impl.h, llama-volta-merged).
// La tête de tensor est EXACTEMENT le layout ci-dessous (ggml-impl.h). On la définit directe
// sous le nom ggml_tensor : le .so écrit/dit dans cette structure en mémoire (pas d'ABI
// cachée au-delà du layout). C'est notre allouation (calloc), pas ggml_new_tensor.
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
struct ggml_context;     // opaque : on ne garde que le pointeur
struct ggml_cgraph;
typedef struct ggml_backend_buffer ggml_backend_buffer;
typedef ggml_backend_buffer * ggml_backend_buffer_t;

enum ggml_type_e { GGML_TYPE_F32=0, GGML_TYPE_F16=1, GGML_TYPE_Q8_0=8 };

struct block_q8_0 { __half d; int8_t qs[32]; };   // 34 octets, align 2

struct ggml_init_params { size_t mem_size; void * mem_buffer; bool no_alloc; };

// ---------------------------------------------------------------- API ggml (ptrs)
typedef struct ggml_context * (*f_ggml_init)(ggml_init_params);
typedef ggml_backend_buffer_t  (*f_buft_alloc_buffer)(void *, size_t);
typedef void *                 (*f_buf_get_base)(ggml_backend_buffer_t);
typedef void                   (*f_buf_free)(ggml_backend_buffer_t);
typedef void                   (*f_backend_free)(void *);

// .so CUDA chargé : défaut = prod ; override via env MB_FA_SO (candidat T3 v2, sans toucher la prod).
static const char * SO_CUDA() {
    static const char * p = getenv("MB_FA_SO") ? getenv("MB_FA_SO")
        : "/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so";
    return p;
}
static const char * SO_BASE = "/home/guile/llama-experiments/ollama-v100/lib/ollama/libggml-base.so.0";

// ggml_cuda_flash_attn_ext(ggml_backend_cuda_context&, ggml_tensor*)  [symbole exporté]
static void (*g_fa_ext)(void *, ggml_tensor *) = nullptr;

struct Env {
    const char * mode;
    void * hbase, * hcuda;
    f_ggml_init           ggml_init;
    f_buft_alloc_buffer   buft_alloc_buffer;
    f_buf_get_base        buf_get_base;
    f_buf_free            buf_free;
    f_backend_free        backend_free;
    void   * backend = nullptr;
    void   * ctx_cuda = nullptr;   // ggml_backend_cuda_context *
    cudaStream_t stream = nullptr;
    ggml_context * gctx = nullptr;
    ggml_backend_buffer_t buft_buf = nullptr;
    void   * base = nullptr;       // base du buffer device
    void   * buft = nullptr;       // ggml_backend_buffer_type *
};

static void *must_dlsym(void *h, const char *name) {
    void *p = dlsym(h, name);
    if (!p) { fprintf(stderr, "dlsym %s manquant\n", name); exit(1); }
    return p;
}

static void env_open(Env & E, const char * mode) {
    E.mode = mode;
    if      (strcmp(mode,"t3")==0)     { setenv("GGML_CUDA_FA_NO_VOLTA_SMALL","0",1); setenv("GGML_CUDA_VOLTA_Q8_FATTN_TC","0",1); }
    else if (strcmp(mode,"h2")==0)     { setenv("GGML_CUDA_FA_NO_VOLTA_SMALL","1",1); setenv("GGML_CUDA_VOLTA_Q8_FATTN_TC","1",1); }
    else if (strcmp(mode,"stock")==0 || strcmp(mode,"probe")==0) { setenv("GGML_CUDA_FA_NO_VOLTA_SMALL","1",1); setenv("GGML_CUDA_VOLTA_Q8_FATTN_TC","0",1); }
    else { fprintf(stderr,"mode inconnu %s\n", mode); exit(1); }
    unsetenv("GGML_CUDA_FA_SKIP_MASKED");
    unsetenv("GGML_CUDA_FATTN_NCBLOCKS");
    unsetenv("GGML_CUDA_N_STREAMS");

    E.hbase = dlopen(SO_BASE, RTLD_NOW | RTLD_LOCAL);
    E.hcuda = dlopen(SO_CUDA(), RTLD_NOW | RTLD_LOCAL);
    if (!E.hbase || !E.hcuda) { fprintf(stderr, "dlopen échoué (%s) : %s\n", SO_CUDA(), dlerror()); exit(1); }
    E.ggml_init         = (f_ggml_init)must_dlsym(E.hbase, "ggml_init");
    E.buf_free          = (f_buf_free)must_dlsym(E.hbase, "ggml_backend_buffer_free");
    E.backend_free      = (f_backend_free)must_dlsym(E.hbase, "ggml_backend_free");
    g_fa_ext            = (void (*)(void*, ggml_tensor*))must_dlsym(E.hcuda, "_Z24ggml_cuda_flash_attn_extR25ggml_backend_cuda_contextP11ggml_tensor");

    void *init_p = must_dlsym(E.hcuda, "ggml_backend_cuda_init");
    E.backend = ((ggml_backend_buffer_t(*)(int))init_p)(0);   // device 0 = V100 (CUDA_VISIBLE_DEVICES=0)
    if (!E.backend) { fprintf(stderr,"ggml_backend_cuda_init(0) a échoué\n"); exit(1); }

    // ggml_guid_t = ggml_guid* = 8 octets (POINTEUR, pas 16). ggml_backend = { guid(8), iface(16 ptrs=128B), device(8B), context(8B) }
    //   -> context @ 8+128+8 = 144
    E.ctx_cuda = *(void **)((char *)E.backend + 144);
    // ggml_backend_cuda_context = { int device(4) ; pad(4) ; std::string name(32) ; cudaEvent_t copy_event(8) ; streams[8][8] }
    //   -> streams @ 4+4+32+8 = 48 ; streams[0][0] @ 48
    E.stream = *(cudaStream_t *)((char *)E.ctx_cuda + 48);
    // NB: le stream ggml est créé LAZILY au 1er ctx.stream() (dans le kernel launch).
    // Donc ici il peut être nullptr ; on le relit APRÈS le warmup dans main().
    if (!E.ctx_cuda) { fprintf(stderr,"ctx_cuda null (offset 144 mauvais?)\n"); exit(1); }
    if (E.stream) fprintf(stderr, "[mb] stream pré-rempli %p (pas lazy)\n", (void*)E.stream);

    // buffer type device
    // ggml_backend = { guid(8), iface(16 ptrs=128B), device(8B)@136, context(8B)@144 }
    void * dev = *(void **)((char *)E.backend + 136);
    void * get_buft = *(void **)((char *)dev + 6 * sizeof(void *));   // dev_i.get_buffer_type = 7e slot (idx 6)
    void * buft = ((void *(*)(void *))get_buft)(dev);
    E.buft_alloc_buffer = (f_buft_alloc_buffer)(*(void **)((char *)buft + sizeof(void *)));
    E.buft = buft;
    // buf_get_base (buffer_i.get_base, 2e slot du buffer) est lu après alloc.
    fprintf(stderr, "[mb] route=%s ok (backend=%p ctx=%p stream=%p)\n", mode, E.backend, E.ctx_cuda, (void*)E.stream);
}

static void env_close(Env & E) {
    if (E.buft_buf)  E.buf_free(E.buft_buf);
    if (E.gctx)      ((void(*)(ggml_context*))must_dlsym(E.hbase,"ggml_free"))(E.gctx);
    if (E.backend)   E.backend_free(E.backend);
    if (E.hcuda)     dlclose(E.hcuda);
    if (E.hbase)     dlclose(E.hbase);
}

struct Geo { int D=256, qh=24, kvh=4; int n_kv=107351, n_kv_pad=107520; float scale; };

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <t3|h2|stock|probe> [iters] [n_kv]\n", argv[0]); return 2; }
    int iters = argc > 2 ? atoi(argv[2]) : 200;
    Geo g;
    if (argc > 3) { g.n_kv = atoi(argv[3]); g.n_kv_pad = ((g.n_kv + 255) / 256) * 256; }
    g.scale = 1.0f / sqrtf((float)g.D);

    setenv("CUDA_VISIBLE_DEVICES", "0", 1);
    CK(cudaSetDevice(0));
    int dev; CK(cudaGetDevice(&dev));

    Env E;
    env_open(E, argv[1]);

    // --- taille du buffer (T max = 8) ---
    const int TMAX = 8;
    size_t nq8  = (size_t)g.D*TMAX*g.qh*4;
    size_t nkv  = (size_t)(g.D/32)*g.n_kv_pad*g.kvh*sizeof(block_q8_0);
    size_t nm8  = (size_t)TMAX*g.n_kv_pad*2;
    // marge après `out` : les temporaires des routes (pool VMM écrit juste après dst->data+nbytes)
    //   - tile/vec q8 : conversion K,V -> f16 = 2 x n_kv_pad*kvh*D*2 octets (LE GROS)
    //   - H2 : pacc/pm/pl ~ splits*T*qh*D*2 (8 Mo max)
    const size_t f16_kv = (size_t)g.n_kv_pad*g.kvh*g.D*2;
    const size_t OUT_MARGIN = f16_kv*2 + 80ull*1024*1024;
    size_t nout8 = nq8 + OUT_MARGIN;
    size_t total = nq8 + nkv + nkv + nm8 + nout8;

    E.buft_buf = E.buft_alloc_buffer(E.buft, total);
    if (!E.buft_buf) { fprintf(stderr,"alloc buffer (%zu oct) échoué\n", total); return 1; }
    E.buf_get_base = (f_buf_get_base)(*(void **)((char *)E.buft_buf + 1 * sizeof(void *)));   // buffer_i.get_base (2e slot)
    E.base = E.buf_get_base(E.buft_buf);
    if (!E.base) { fprintf(stderr,"get_base nul\n"); return 1; }
    fprintf(stderr, "[mb] buffer %zu octets ok, base=%p (Q=%.1fMB KV=%.1fMB out=%.1fMB)\n",
            total, E.base, (double)nq8/1e6, (double)nkv/1e6, (double)nout8/1e6);

    // --- données (déterministe + borné) ---
    std::vector<float>  hq(nq8/4);
    std::vector<char>   hkv(nkv);
    std::vector<__half> hm(nm8/2);
    srand(12345);
    for (auto & x : hq) x = (rand()/(float)RAND_MAX - 0.5f);
    {
        block_q8_0 * bk = (block_q8_0 *)hkv.data();
        for (size_t b=0; b<nkv/sizeof(block_q8_0); ++b) {
            bk[b].d = __float2half((rand()/(float)RAND_MAX - 0.5f)*0.5f);
            for (int j=0;j<32;++j) bk[b].qs[j] = (int8_t)((rand()%251)-125);
        }
    }
    for (auto & x : hm) x = __float2half(0.0f);
    CK(cudaMemcpy(E.base,                       hq.data(),  nq8,  cudaMemcpyHostToDevice));
    CK(cudaMemcpy((char*)E.base + nq8,          hkv.data(), nkv,  cudaMemcpyHostToDevice));
    CK(cudaMemcpy((char*)E.base + nq8 + nkv,    hkv.data(), nkv,  cudaMemcpyHostToDevice));
    CK(cudaMemcpy((char*)E.base + nq8 + 2*nkv,  hm.data(),  nm8,  cudaMemcpyHostToDevice));

    ggml_init_params p { 0, nullptr, true };
    E.gctx = E.ggml_init(p);
    if (!E.gctx) { fprintf(stderr,"ggml_init échoué\n"); return 1; }

    cudaEvent_t ev0, ev1;
    CK(cudaEventCreate(&ev0)); CK(cudaEventCreate(&ev1));

    const bool probe = (strcmp(argv[1],"probe")==0);
    int T_list[8] = {1,2,3,4,5,6,7,8};
    int nT = probe ? 1 : 8;
    if (probe) T_list[0] = 4;

    for (int ti = 0; ti < nT; ++ti) {
        const int T = T_list[ti];
        size_t nm  = (size_t)T*g.n_kv_pad*2;

        auto mk = [&](int type, int64_t ne0, int64_t ne1, int64_t ne2, size_t nb0, size_t nb1, void *data) -> ggml_tensor * {
            ggml_tensor * t = (ggml_tensor *)calloc(1, sizeof(ggml_tensor));
            t->type = type; t->buffer = (void *)E.buft_buf;
            t->ne[0]=ne0; t->ne[1]=ne1; t->ne[2]=ne2; t->ne[3]=1;
            t->nb[0]=nb0; t->nb[1]=nb1; t->nb[2]=nb1*ne1; t->nb[3]=nb1*ne1*ne2;
            t->data = data;
            return t;
        };
        ggml_tensor * q   = mk(GGML_TYPE_F32,  g.D, T, g.qh, 4, (size_t)g.D*4, E.base);
        ggml_tensor * k   = mk(GGML_TYPE_Q8_0, g.D, g.n_kv_pad, g.kvh, sizeof(block_q8_0), (size_t)(g.D/32)*sizeof(block_q8_0), (char*)E.base + nq8);
        ggml_tensor * v   = mk(GGML_TYPE_Q8_0, g.D, g.n_kv_pad, g.kvh, sizeof(block_q8_0), (size_t)(g.D/32)*sizeof(block_q8_0), (char*)E.base + nq8 + nkv);

        // mask : zéros. MÊME layout pour les 3 routes (assertion H2 : mask->ne[0]==K->ne[1],
        // mask->ne[1]>=Q->ne[1]) : [key][tok] -> ne[0]=n_kv_pad, ne[1]=T, nb[0]=2, nb[1]=n_kv_pad*2.
        ggml_tensor * m = mk(GGML_TYPE_F16, g.n_kv_pad, T, 1, 2, (size_t)g.n_kv_pad*2, (char*)E.base + nq8 + 2*nkv);
        ggml_tensor * out = mk(GGML_TYPE_F32, g.D, g.qh, T, 4, (size_t)g.D*4, (char*)E.base + nq8 + 2*nkv + nm8);

        out->op = 74; // GGML_OP_FLASH_ATTN_EXT
        float params[3] = { g.scale, 0.0f, 0.0f };
        memcpy(out->op_params, params, 12);
        out->src[0]=q; out->src[1]=k; out->src[2]=v; out->src[3]=m; out->src[4]=nullptr;

        // warmup (2 iters hors mesure) — crée aussi le stream ggml (lazy)
        for (int w=0; w<2; ++w) g_fa_ext(E.ctx_cuda, out);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());

        // relecture du stream ggml (créé lazily au 1er launch) + validation
        cudaStream_t gst = *(cudaStream_t *)((char *)E.ctx_cuda + 48);
        if (!gst) { fprintf(stderr,"stream ggml toujours null après warmup — offset 48 mauvais\n"); return 1; }
        if (E.stream == nullptr) E.stream = gst;   // mémoire : même stream pour toutes les T
        {
            cudaEvent_t se; CK(cudaEventCreate(&se));
            CK(cudaEventRecord(se, gst));
            CK(cudaEventSynchronize(se));
            CK(cudaEventDestroy(se));
        }

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

        printf("{\"mode\":\"%s\",\"T\":%d,\"n_kv\":%d,\"us_med\":%.1f,\"us_min\":%.1f,\"us_max\":%.1f,\"out\":[%.6g,%.6g,%.6g,%.6g]}\n",
               argv[1], T, g.n_kv_pad, us_med, us_min, us_max, o[0], o[1], o[2], o[3]);
        fflush(stdout);
    }

    env_close(E);
    return 0;
}
