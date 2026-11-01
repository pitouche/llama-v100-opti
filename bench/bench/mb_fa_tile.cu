// (c) 2026 guile — MIT. mb_fa_tile.cu — 31/10
//
// A/B des VARIANTES DE TILE du FA f16 directement dans le .so PROD (dlopen, SANS
// rebuild, SANS toucher au service ollama-v100). Meme setup que mb_fa_pf.cu
// (Q f32, K/V f16, mask f16 = 0 partout -> full attention majore), mais on dlsym
// les instantiations exportees du .so au lieu du dispatch ggml_cuda_flash_attn_ext.
//
// IMPORTANT — lecture des templates flash_attn_ext_f16<DKQ, DV, ncols1, ncols2,...> :
//   DKQ/DV = DIMENSION DE TETE (head_length=256 du modele) -> FIXES a 256,256.
//   ncols1 = tokens Q par CTA (le vrai "tile" sur l'axe T)
//   ncols2 = tetes KV par CTA (GQA, doit diviser gqa=6 -> 1 ou 2 ici)
// Les "variantes (128,128,32,2)" / "(128,256,32,2)" demandees ne SONT PAS ces
// combos : elles changeraient DKQ/DV et ne calculeraient pas la tete de dim 256.
// Les vrais re-tiles dispo SANS rebuild = les instantiations (ncols1,ncols2)
// deja compilees dans le .so prod :
//
//   32x2  = case<256,256,32,2>  = kernel BASELINE PROD (ncols=64, 128 thr/4 warps,
//                                  config Volta ncols64: nbatch_fa=32, nstages=2,
//                                  regs 255, smem 67.58 KB -> 1 block/SM)
//   64x1  = case<256,256,64,1>  (ncols=64, 128 thr, nbatch_fa=32, 1 tete KV/CTA)
//   16x2  = case<256,256,16,2>  (ncols=32, 128 thr, config "NInfer sm70 32-col":
//                                  nbatch_fa=32, nstages=1 -> smem ~2x plus petite)
//   32x1  = case<256,256,32,1>  (ncols=32, 128 thr, 1 tete KV/CTA)
//   16x1  = case<256,256,16,1>  (ncols=16, 64 thr, config ampere fallback ncols16)
//
// Usage :
//   CUDA_VISIBLE_DEVICES=$CVD ./mb_fa_tile [variant|all] [T] [n_kv] [iters] [maskmode]
//   variant defaut "all" (= 32x2,16x2,32x1,64x1 ; 16x1 mort sur sm_70, exclu) ;
//   T defaut 512 ; n_kv defaut 114688 ; iters defaut 5 ; maskmode 0=full(mask=0), 1=causal.
// Sortie JSONL : 1 ligne par variante testee.

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
// (identiques a mb_fa_pf.cu — meme ABI que le .so prod)
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
struct ggml_context;
typedef struct ggml_backend_buffer ggml_backend_buffer;
typedef ggml_backend_buffer * ggml_backend_buffer_t;
enum ggml_type_e { GGML_TYPE_F32=0, GGML_TYPE_F16=1 };
struct ggml_init_params { size_t mem_size; void * mem_buffer; bool no_alloc; };

typedef struct ggml_context * (*f_ggml_init)(ggml_init_params);
typedef ggml_backend_buffer_t  (*f_buft_alloc_buffer)(void *, size_t);
typedef void *                 (*f_buf_get_base)(ggml_backend_buffer_t);
typedef void                   (*f_buf_free)(ggml_backend_buffer_t);
typedef void                   (*f_backend_free)(void *);
typedef void                   (*f_case)(void *, ggml_tensor *);

static const char * SO_CUDA() {
    static const char * p = getenv("MB_FA_SO") ? getenv("MB_FA_SO")
        : "/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so";
    return p;
}
static const char * SO_BASE = "/home/guile/llama-experiments/ollama-v100/lib/ollama/libggml-base.so.0";

// mangled reels (verifies par `nm -D libggml-cuda.so | c++filt`)
// ggml_cuda_flash_attn_ext_mma_f16_case<256, 256, N1, N2>(ggml_backend_cuda_context&, ggml_tensor*)
struct Variant { const char * name; const char * sym; };
static Variant VARIANTS[] = {
    { "32x2", "_Z37ggml_cuda_flash_attn_ext_mma_f16_caseILi256ELi256ELi32ELi2EEvR25ggml_backend_cuda_contextP11ggml_tensor" },
    { "64x1", "_Z37ggml_cuda_flash_attn_ext_mma_f16_caseILi256ELi256ELi64ELi1EEvR25ggml_backend_cuda_contextP11ggml_tensor" },
    { "16x2", "_Z37ggml_cuda_flash_attn_ext_mma_f16_caseILi256ELi256ELi16ELi2EEvR25ggml_backend_cuda_contextP11ggml_tensor" },
    { "32x1", "_Z37ggml_cuda_flash_attn_ext_mma_f16_caseILi256ELi256ELi32ELi1EEvR25ggml_backend_cuda_contextP11ggml_tensor" },
    { "16x1", "_Z37ggml_cuda_flash_attn_ext_mma_f16_caseILi256ELi256ELi16ELi1EEvR25ggml_backend_cuda_contextP11ggml_tensor" },
};
static const int NVARIANTS = sizeof(VARIANTS)/sizeof(VARIANTS[0]);
// "all" = variants vivants sur sm_70 (16x1 exclu : mort, pas de device code 700)
static const char *ALL_LIST[] = { "32x2", "16x2", "32x1", "64x1" };

struct Env {
    void * hbase, * hcuda;
    f_ggml_init           ggml_init;
    f_buft_alloc_buffer   buft_alloc_buffer;
    f_buf_get_base        buf_get_base;
    f_buf_free            buf_free;
    f_backend_free        backend_free;
    f_case                fn[NVARIANTS];
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
    for (int i = 0; i < NVARIANTS; ++i) E.fn[i] = (f_case)must_dlsym(E.hcuda, VARIANTS[i].sym);

    void *init_p = must_dlsym(E.hcuda, "ggml_backend_cuda_init");
    E.backend = ((ggml_backend_buffer_t(*)(int))init_p)(0);
    if (!E.backend) { fprintf(stderr,"ggml_backend_cuda_init(0) a échoué\n"); exit(1); }
    E.ctx_cuda = *(void **)((char *)E.backend + 144);
    void * dev = *(void **)((char *)E.backend + 136);
    void * get_buft = *(void **)((char *)dev + 6 * sizeof(void *));
    void * buft = ((void *(*)(void *))get_buft)(dev);
    E.buft_alloc_buffer = (f_buft_alloc_buffer)(*(void **)((char *)buft + sizeof(void *)));
    E.buft = buft;
    if (!E.ctx_cuda) { fprintf(stderr,"ctx_cuda null\n"); exit(1); }
    fprintf(stderr, "[mb_tile] route ok\n");
}

static void env_close(Env & E) {
    if (E.buft_buf)  E.buf_free(E.buft_buf);
    if (E.gctx)      ((void(*)(ggml_context*))must_dlsym(E.hbase,"ggml_free"))(E.gctx);
    if (E.backend)   E.backend_free(E.backend);
    if (E.hcuda)     dlclose(E.hcuda);
    if (E.hbase)     dlclose(E.hbase);
}

struct Geo { int D=256, qh=24, kvh=4; int n_kv=114688, n_kv_pad=114688; float scale; };

static void run_case(Env & E, int idx, ggml_tensor * out, int T, int n_kv_pad,
                     int iters, size_t off_out, int maskmode) {
    f_case fn = E.fn[idx];
    const char * name = VARIANTS[idx].name;
    // warmup 2 (hors mesure) : set-attr smem + caches
    for (int w=0; w<2; ++w) fn(E.ctx_cuda, out);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    { cudaEvent_t se; CK(cudaEventCreate(&se)); CK(cudaEventRecord(se, E.stream));
      CK(cudaEventSynchronize(se)); CK(cudaEventDestroy(se)); }
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    std::vector<float> samples;
    for (int it=0; it<iters; ++it) {
        CK(cudaEventRecord(e0, E.stream));
        fn(E.ctx_cuda, out);
        CK(cudaEventRecord(e1, E.stream));
        CK(cudaEventSynchronize(e1));
        CK(cudaGetLastError());
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
        samples.push_back(ms*1000.0f);
    }
    std::sort(samples.begin(), samples.end());
    float us_med = samples[samples.size()/2];
    float us_min = samples.front();
    float us_max = samples.back();
    float o[4];
    CK(cudaMemcpy(o, (char*)E.base + off_out, 16, cudaMemcpyDeviceToHost));
    bool finite = std::isfinite(o[0]) && std::isfinite(o[1]) && std::isfinite(o[2]) && std::isfinite(o[3]);
    printf("{\"case\":\"%s\",\"T\":%d,\"n_kv\":%d,\"mask\":%d,\"us_med\":%.1f,\"us_min\":%.1f,\"us_max\":%.1f,\"finite\":%s,\"out\":[%.6g,%.6g,%.6g,%.6g]}\n",
           name, T, n_kv_pad, maskmode, us_med, us_min, us_max, finite?"true":"false",
           o[0], o[1], o[2], o[3]);
    fflush(stdout);
}

int main(int argc, char ** argv) {
    const char * sel  = argc > 1 ? argv[1] : "all";
    int T      = argc > 2 ? atoi(argv[2]) : 512;
    int n_kv   = argc > 3 ? atoi(argv[3]) : 114688;
    int iters  = argc > 4 ? atoi(argv[4]) : 5;
    int maskmode = argc > 5 ? atoi(argv[5]) : 0;  // 0=full (mask=0) ; 1=causal
    Geo g; g.n_kv_pad = ((n_kv + 255) / 256) * 256;
    g.n_kv = g.n_kv_pad;
    g.scale = 1.0f / sqrtf((float)g.D);

    CK(cudaSetDevice(0));
    int dev; CK(cudaGetDevice(&dev));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
    fprintf(stderr, "[mb_tile] device %d = %s (SM %d.%d, %d SM, %.1f GB)\n",
            dev, prop.name, prop.major, prop.minor, prop.multiProcessorCount,
            prop.totalGlobalMem/1073741824.0);
    fprintf(stderr, "[mb_tile] sel=%s T=%d n_kv=%d (pad %d) iters=%d mask=%d\n", sel, T, n_kv, g.n_kv_pad, iters, maskmode);

    Env E;
    env_open(E);

    size_t nq8  = (size_t)g.D*T*g.qh*4;
    size_t nkv  = (size_t)g.D*g.n_kv_pad*g.kvh*2;
    size_t nm8  = (size_t)T*g.n_kv_pad*2;
    size_t nout8= (size_t)g.D*g.qh*T*4;
    const size_t OUT_MARGIN = 64ull*1024*1024;
    size_t total = nq8 + nkv + nkv + nm8 + nout8 + OUT_MARGIN;

    E.buft_buf = E.buft_alloc_buffer(E.buft, total);
    if (!E.buft_buf) { fprintf(stderr,"alloc échoué (%zu oct)\n", total); return 1; }
    E.buf_get_base = (f_buf_get_base)(*(void **)((char *)E.buft_buf + 1 * sizeof(void *)));
    E.base = E.buf_get_base(E.buft_buf);
    if (!E.base) { fprintf(stderr,"get_base nul\n"); return 1; }
    fprintf(stderr, "[mb_tile] buffer %zu octets (%.1f MB)\n", total, total/1048576.0);

    std::vector<float>  hq(nq8/4);
    std::vector<__half> hkv(nkv/2);
    std::vector<__half> hm(nm8/2);
    srand(98765);
    for (auto & x : hq)  x = (rand()/(float)RAND_MAX - 0.5f);
    for (auto & x : hkv) x = __float2half((rand()/(float)RAND_MAX - 0.5f));
    for (auto & x : hm)  x = __float2half(0.0f);   // mask=0 -> full attention (majoré)
    // Causal (maskmode==1) : le query iq (position globale = (n_kv-T)+iq) ne voit que ikv <= (n_kv-T)+iq.
    // mask layout : ne=[n_kv_pad, T, 1], nb=[2, n_kv_pad*2, ...] ; element (ikv, iq) -> offset iq*n_kv_pad + ikv (half).
    if (maskmode == 1) {
        for (long iq = 0; iq < T; ++iq)
            for (long ikv = 0; ikv < g.n_kv_pad; ++ikv)
                hm[iq*(long)g.n_kv_pad + ikv] =
                    __float2half( ikv <= ((long)g.n_kv - T) + iq ? 0.0f : -65504.0f );
    }
    CK(cudaMemcpy(E.base,                    hq.data(),  nq8,  cudaMemcpyHostToDevice));
    CK(cudaMemcpy((char*)E.base + nq8,       hkv.data(), nkv,  cudaMemcpyHostToDevice));
    CK(cudaMemcpy((char*)E.base + nq8 + nkv, hkv.data(), nkv,  cudaMemcpyHostToDevice));
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

    // 1er launch (hors mesure) pour creer le stream ggml (lazy), puis relecture
    E.fn[0](E.ctx_cuda, out);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    E.stream = *(cudaStream_t *)((char *)E.ctx_cuda + 48);
    if (!E.stream) { fprintf(stderr,"stream ggml null apres warmup — offset 48 mauvais?\n"); return 1; }
    { cudaEvent_t se; CK(cudaEventCreate(&se)); CK(cudaEventRecord(se, E.stream));
      CK(cudaEventSynchronize(se)); CK(cudaEventDestroy(se)); }

    size_t off_out = nq8 + 2*nkv + nm8;
    // liste des variants a tester
    std::vector<const char*> to_run;
    if (strcmp(sel, "all") == 0) {
        for (auto n : ALL_LIST) to_run.push_back(n);
    } else {
        to_run.push_back(sel);
    }
    for (const char * nm : to_run) {
        int i = -1;
        for (int k = 0; k < NVARIANTS; ++k) if (!strcmp(VARIANTS[k].name, nm)) { i = k; break; }
        if (i < 0) { fprintf(stderr, "variant inconnu: %s\n", nm); continue; }
        run_case(E, i, out, T, g.n_kv_pad, iters, off_out, maskmode);
    }

    env_close(E);
    return 0;
}
