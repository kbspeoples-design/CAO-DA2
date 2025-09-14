#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <limits.h>
#include <inttypes.h>
#include <signal.h>
#ifdef _OPENMP
#include <omp.h>
#else
static inline int omp_get_thread_num(){ return 0; }
static inline double omp_get_wtime(){ return 0.0; }
static inline int omp_get_max_threads(){ return 1; }
static inline void omp_set_num_threads(int n){ (void)n; }
#endif

#ifdef _WIN32
#include <windows.h>
#include <malloc.h>
#include <io.h>
#else
#include <sys/mman.h>
#endif

#if defined(__linux__)
#include <unistd.h>
#include <fcntl.h>
#include <numa.h>
#include <libaio.h>
#elif defined(__APPLE__)
#include <sys/types.h>
#include <sys/sysctl.h>
#include <aio.h>
#endif

// -------- Defaults / Tunables ----------
int BLOCK_OUTER_I = 56, BLOCK_OUTER_J = 56, BLOCK_OUTER_K = 56;
int MICRO_I = 8, MICRO_J = 16;  // prefer 8x16 microkernel by default
int DATASETS = 10000;
int BATCH = 16;
int DEFAULT_M = 500;
int DEFAULT_K = 500;
int DEFAULT_N = 500;
int THREADS = 0;
int PIN_BUF = 0;
int AUTOTUNE = 0; // 0 off, 1 quick, 2 thorough
#define NUM_BUFFERS 3
#define ALIGN_BYTES 64

// Safety minimum measurement time for autotune (seconds)
#define AUTOTUNE_MIN_TIME 0.02
// Minimum repeats if too short
#define AUTOTUNE_MIN_REPEATS 3

// -------- Prefetch macro ----------
#if defined(__GNUC__) || defined(__clang__)
#define PREFETCH(addr) __builtin_prefetch(addr)
#else
#define PREFETCH(addr) ((void)0)
#endif

// -------- Loader sync ----------
pthread_mutex_t loader_mutex[NUM_BUFFERS] = {
    PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER
};
pthread_cond_t loader_cond[NUM_BUFFERS] = {
    PTHREAD_COND_INITIALIZER, PTHREAD_COND_INITIALIZER, PTHREAD_COND_INITIALIZER
};
volatile int buf_ready[NUM_BUFFERS];
volatile int buf_done[NUM_BUFFERS];
volatile int loader_error = 0;

typedef struct {
#ifndef _WIN32
    FILE *fa, *fb;
#else
    HANDLE fa_h, fb_h;
#endif
    double *A_buf[NUM_BUFFERS];
    double *B_buf[NUM_BUFFERS];
    int M,K,N;
    int batch;
    int datasets;
} loader_args_t;

// -------- Allocation registry (to match numa_alloc_onnode + numa_free) ----------
typedef struct alloc_record {
    void *ptr;             // user pointer returned to caller
    size_t bytes;
    int numa_used;         // 1 if allocated with numa_alloc_onnode
    struct alloc_record *next;
} alloc_record_t;

static alloc_record_t *alloc_list = NULL;
static pthread_mutex_t alloc_list_mutex = PTHREAD_MUTEX_INITIALIZER;

static void register_alloc(void *ptr, size_t bytes, int numa_used) {
    alloc_record_t *r = (alloc_record_t*)malloc(sizeof(alloc_record_t));
    if (!r) { fprintf(stderr,"register_alloc malloc failed\n"); exit(1); }
    r->ptr = ptr; r->bytes = bytes; r->numa_used = numa_used;
    pthread_mutex_lock(&alloc_list_mutex);
    r->next = alloc_list;
    alloc_list = r;
    pthread_mutex_unlock(&alloc_list_mutex);
}
static alloc_record_t* find_and_remove_alloc(void *ptr) {
    pthread_mutex_lock(&alloc_list_mutex);
    alloc_record_t *p = alloc_list, *prev = NULL;
    while (p) {
        if (p->ptr == ptr) {
            if (prev) prev->next = p->next;
            else alloc_list = p->next;
            pthread_mutex_unlock(&alloc_list_mutex);
            return p;
        }
        prev = p; p = p->next;
    }
    pthread_mutex_unlock(&alloc_list_mutex);
    return NULL;
}

// -------- NUMA/allocation helpers ----------
int get_numa_nodes() {
#if defined(_WIN32)
    return 1;
#elif defined(__linux__)
    if (numa_available() != -1) return numa_max_node()+1;
    return 1;
#else
    return 1;
#endif
}
int get_numa_node_for_thread(int tid) {
    int nodes = get_numa_nodes();
    return (nodes>0) ? (tid % nodes) : 0;
}

/* safe allocation with overflow checks; on Linux will try numa_alloc_onnode() */
double* alloc_numa_matrix(int rows, int cols, int batch, int pin, int node) {
    if (rows < 0 || cols < 0 || batch < 0) { fprintf(stderr,"Bad alloc dims\n"); exit(1); }
    size_t r = (size_t) rows;
    size_t c = (size_t) cols;
    size_t b = (size_t) batch;
    size_t elsize = sizeof(double);

    if (r == 0 || c == 0 || b == 0) {
        size_t bytes = elsize;
#if defined(_WIN32)
        double* mat = (double*)_aligned_malloc(bytes, ALIGN_BYTES);
        if (!mat) { fprintf(stderr,"alloc failed\n"); exit(1); }
        if (pin) VirtualLock(mat, bytes);
        memset(mat, 0, bytes);
        register_alloc(mat, bytes, 0);
        return mat;
#else
        double* mat = NULL;
        if (posix_memalign((void**)&mat, ALIGN_BYTES, bytes) != 0 || !mat) { fprintf(stderr,"posix_memalign failed\n"); exit(1); }
        memset(mat, 0, bytes);
        if (pin) mlock(mat, bytes);
        register_alloc(mat, bytes, 0);
        return mat;
#endif
    }

    if (r > SIZE_MAX / c) { fprintf(stderr,"alloc overflow\n"); exit(1); }
    size_t rc = r * c;
    if (rc > SIZE_MAX / b) { fprintf(stderr,"alloc overflow\n"); exit(1); }
    size_t rcb = rc * b;
    if (rcb > SIZE_MAX / elsize) { fprintf(stderr,"alloc overflow\n"); exit(1); }
    size_t bytes = rcb * elsize;

#if defined(__linux__)
    double* mat = NULL;
    if (numa_available() != -1 && node < get_numa_nodes()) {
        // Try NUMA allocation
        mat = (double*)numa_alloc_onnode(bytes, node);
        if (!mat) {
            // fall back to posix_memalign
            mat = NULL;
            if (posix_memalign((void**)&mat, ALIGN_BYTES, bytes) != 0 || !mat) { fprintf(stderr,"alloc failed\n"); exit(1); }
            memset(mat, 0, bytes);
            if (pin) mlock(mat, bytes);
            register_alloc(mat, bytes, 0);
            return mat;
        } else {
            memset(mat, 0, bytes);
            if (pin) mlock(mat, bytes);
            register_alloc(mat, bytes, 1); // mark numa_used
            return mat;
        }
    } else {
        if (posix_memalign((void**)&mat, ALIGN_BYTES, bytes) != 0 || !mat) { fprintf(stderr,"posix_memalign failed\n"); exit(1); }
        memset(mat, 0, bytes);
        if (pin) mlock(mat, bytes);
        register_alloc(mat, bytes, 0);
        return mat;
    }
#elif defined(_WIN32)
    double* mat = (double*)_aligned_malloc(bytes, ALIGN_BYTES);
    if (!mat) { fprintf(stderr,"alloc failed\n"); exit(1); }
    if (pin) VirtualLock(mat, bytes);
    memset(mat, 0, bytes);
    register_alloc(mat, bytes, 0);
    return mat;
#else
    double* mat = NULL;
    if (posix_memalign((void**)&mat, ALIGN_BYTES, bytes) != 0 || !mat) { fprintf(stderr,"posix_memalign failed\n"); exit(1); }
    memset(mat, 0, bytes);
    if (pin) mlock(mat, bytes);
    register_alloc(mat, bytes, 0);
    return mat;
#endif
}

void free_matrix(double* mat) {
    if (!mat) return;
    alloc_record_t *r = find_and_remove_alloc(mat);
    if (r) {
#if defined(__linux__)
        if (r->numa_used) {
            numa_free(mat, r->bytes);
        } else {
            free(mat);
        }
#else
        free(mat);
#endif
        free(r);
    } else {
        // Not found in registry: fallback to free()
#if defined(_WIN32)
        _aligned_free(mat);
#else
        free(mat);
#endif
    }
}

// -------- small helpers ----------
static inline uint64_t xorshift64star(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 2685821657736338717ULL;
}
void random_matrix(double* restrict mat, int rows, int cols, uint64_t seed) {
    uint64_t st = seed ? seed : 0xdeadbeefcafebabeULL;
    long long total = (long long)rows * (long long)cols;
    for (long long i=0;i<total;++i) {
        uint64_t v = xorshift64star(&st);
        mat[i] = (double)(v & 0xFFFFFFFFULL) / (double)0x100000000ULL;
    }
}
void parallel_memset(double* restrict mat, int rows, int cols) {
    #pragma omp parallel for
    for (long long i=0;i<(long long)rows*(long long)cols;++i) mat[i]=0.0;
}

// -------- transpose (blocked) ----------
void transpose_matrix_rect(const double* restrict B, double* restrict BT, int K, int N, int block) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i0=0;i0<K;i0+=block) {
        for (int j0=0;j0<N;j0+=block) {
            int imax = (i0+block>K)?(K-i0):block;
            int jmax = (j0+block>N)?(N-j0):block;
            for (int i=i0;i<i0+imax;++i) {
                for (int j=j0;j<j0+jmax;++j) {
                    BT[(size_t)j * K + i] = B[(size_t)i * N + j];
                }
            }
        }
    }
}

// -------- Packing helpers ----------
static inline int round_up(int x, int a) { return ((x + a - 1) / a) * a; }

void pack_A_block_panel_padded(const double* restrict A, double* restrict AP,
                        int M, int K, int i_start, int k_start,
                        int block_i, int block_k, int pad_i)
{
    for (int i=0;i<pad_i;++i) {
        if (i < block_i) PREFETCH(&A[(size_t)(i_start + i) * K + k_start]);
        for (int k=0;k<block_k;++k) {
            if (i < block_i && (k_start + k) < K) {
                AP[(size_t)i * block_k + k] = A[(size_t)(i_start + i) * K + (k_start + k)];
            } else {
                AP[(size_t)i * block_k + k] = 0.0;
            }
        }
    }
}

void pack_B_block_panel_padded(const double* restrict BT, double* restrict BP,
                        int K, int N, int j_start, int k_start,
                        int block_j, int block_k, int pad_j)
{
    for (int kk=0; kk<block_k; ++kk) {
        PREFETCH(&BT[(size_t)j_start * K + (k_start + kk)]);
        for (int j=0;j<pad_j;++j) {
            if (j < block_j && (j_start + j) < N && (k_start + kk) < K) {
                BP[(size_t)kk * pad_j + j] = BT[(size_t)(j_start + j) * K + (k_start + kk)];
            } else {
                BP[(size_t)kk * pad_j + j] = 0.0;
            }
        }
    }
}

// -------- per-thread panel buffers ----------
typedef struct {
    double* AP;
    double* BP;
    int block_i, block_j, block_k;
    int pad_i, pad_j;
} thread_panel_buffers_t;

thread_panel_buffers_t* alloc_panel_buffers(int nthreads, int block_i, int block_j, int block_k, int micro_i, int micro_j) {
    thread_panel_buffers_t* bufs = (thread_panel_buffers_t*)malloc((size_t)nthreads * sizeof(thread_panel_buffers_t));
    if (!bufs) { fprintf(stderr,"alloc panel bufs failed\n"); exit(1); }
    int pad_i = round_up(block_i, micro_i);
    int pad_j = round_up(block_j, micro_j);
    size_t asize = (size_t)pad_i * block_k * sizeof(double);
    size_t bsize = (size_t)block_k * pad_j * sizeof(double);
    for (int t=0;t<nthreads;++t) {
#if defined(_WIN32)
        bufs[t].AP = (double*)_aligned_malloc(asize, ALIGN_BYTES);
        bufs[t].BP = (double*)_aligned_malloc(bsize, ALIGN_BYTES);
        if (!bufs[t].AP || !bufs[t].BP) { fprintf(stderr,"aligned malloc panel buf failed\n"); exit(1); }
#else
        if (posix_memalign((void**)&bufs[t].AP, ALIGN_BYTES, asize) != 0) {
            for (int q=0;q<t;++q) free(bufs[q].AP), free(bufs[q].BP);
            free(bufs); fprintf(stderr,"posix_memalign panel AP failed\n"); exit(1);
        }
        if (posix_memalign((void**)&bufs[t].BP, ALIGN_BYTES, bsize) != 0) {
            free(bufs[t].AP);
            for (int q=0;q<t;++q) free(bufs[q].AP), free(bufs[q].BP);
            free(bufs); fprintf(stderr,"posix_memalign panel BP failed\n"); exit(1);
        }
#endif
        memset(bufs[t].AP, 0, asize);
        memset(bufs[t].BP, 0, bsize);
        bufs[t].block_i = block_i; bufs[t].block_j = block_j; bufs[t].block_k = block_k;
        bufs[t].pad_i = pad_i; bufs[t].pad_j = pad_j;
    }
    return bufs;
}
void free_panel_buffers(thread_panel_buffers_t* bufs, int nthreads) {
    if (!bufs) return;
    for (int t=0;t<nthreads;++t) {
#if defined(_WIN32)
        _aligned_free(bufs[t].AP);
        _aligned_free(bufs[t].BP);
#else
        free(bufs[t].AP);
        free(bufs[t].BP);
#endif
    }
    free(bufs);
}

// -------- Aligned load/store helpers ----------
static inline int is_aligned32(const void *p) { return (((uintptr_t)p) & 31) == 0; }
static inline __m256d load256_al(const double *p) { return is_aligned32(p) ? _mm256_load_pd(p) : _mm256_loadu_pd(p); }
static inline void store256_al(double *p, __m256d v) { if (is_aligned32(p)) _mm256_store_pd(p, v); else _mm256_storeu_pd(p, v); }

// -------- AVX2 microkernels (FMA in-place; aligned when possible) ----------
static inline void micro_kernel_4x8_avx2(const double* restrict AP, const double* restrict BP,
                                         double* restrict C, int ldc, int block_k, int ldp)
{
    __m256d cacc[4][2];
    for (int i=0;i<4;++i) {
        cacc[i][0] = load256_al(&C[i*(size_t)ldc + 0]);
        cacc[i][1] = load256_al(&C[i*(size_t)ldc + 4]);
    }

    for (int kk=0; kk<block_k; ++kk) {
        __m256d a0 = _mm256_broadcast_sd(&AP[0*block_k + kk]);
        __m256d a1 = _mm256_broadcast_sd(&AP[1*block_k + kk]);
        __m256d a2 = _mm256_broadcast_sd(&AP[2*block_k + kk]);
        __m256d a3 = _mm256_broadcast_sd(&AP[3*block_k + kk]);
        const double *bptr = &BP[(size_t)kk * ldp];
        __m256d b0 = load256_al(&bptr[0]);
        __m256d b1 = load256_al(&bptr[4]);
        cacc[0][0] = _mm256_fmadd_pd(a0,b0,cacc[0][0]); cacc[0][1] = _mm256_fmadd_pd(a0,b1,cacc[0][1]);
        cacc[1][0] = _mm256_fmadd_pd(a1,b0,cacc[1][0]); cacc[1][1] = _mm256_fmadd_pd(a1,b1,cacc[1][1]);
        cacc[2][0] = _mm256_fmadd_pd(a2,b0,cacc[2][0]); cacc[2][1] = _mm256_fmadd_pd(a2,b1,cacc[2][1]);
        cacc[3][0] = _mm256_fmadd_pd(a3,b0,cacc[3][0]); cacc[3][1] = _mm256_fmadd_pd(a3,b1,cacc[3][1]);
    }

    for (int i=0;i<4;++i) {
        store256_al(&C[i*(size_t)ldc + 0], cacc[i][0]);
        store256_al(&C[i*(size_t)ldc + 4], cacc[i][1]);
    }
}

static inline void micro_kernel_6x8_avx2(const double* restrict AP, const double* restrict BP,
                                         double* restrict C, int ldc, int block_k, int ldp)
{
    __m256d cacc[6][2];
    for (int i=0;i<6;++i) {
        cacc[i][0] = load256_al(&C[i*(size_t)ldc + 0]);
        cacc[i][1] = load256_al(&C[i*(size_t)ldc + 4]);
    }
    for (int kk=0; kk<block_k; ++kk) {
        __m256d a[6];
        for (int r=0;r<6;++r) a[r] = _mm256_broadcast_sd(&AP[r*block_k + kk]);
        const double *bptr = &BP[(size_t)kk * ldp];
        __m256d b0 = load256_al(&bptr[0]);
        __m256d b1 = load256_al(&bptr[4]);
        for (int r=0;r<6;++r) {
            cacc[r][0] = _mm256_fmadd_pd(a[r], b0, cacc[r][0]);
            cacc[r][1] = _mm256_fmadd_pd(a[r], b1, cacc[r][1]);
        }
    }
    for (int i=0;i<6;++i) {
        store256_al(&C[i*(size_t)ldc + 0], cacc[i][0]);
        store256_al(&C[i*(size_t)ldc + 4], cacc[i][1]);
    }
}

static inline void micro_kernel_8x8_avx2(const double* restrict AP, const double* restrict BP,
                                         double* restrict C, int ldc, int block_k, int ldp)
{
    __m256d cacc[8][2];
    for (int i=0;i<8;++i) {
        cacc[i][0] = load256_al(&C[i*(size_t)ldc + 0]);
        cacc[i][1] = load256_al(&C[i*(size_t)ldc + 4]);
    }
    for (int kk=0; kk<block_k; ++kk) {
        __m256d a[8];
        for (int r=0;r<8;++r) a[r] = _mm256_broadcast_sd(&AP[r*block_k + kk]);
        const double *bptr = &BP[(size_t)kk * ldp];
        __m256d b0 = load256_al(&bptr[0]);
        __m256d b1 = load256_al(&bptr[4]);
        for (int r=0;r<8;++r) {
            cacc[r][0] = _mm256_fmadd_pd(a[r], b0, cacc[r][0]);
            cacc[r][1] = _mm256_fmadd_pd(a[r], b1, cacc[r][1]);
        }
    }
    for (int i=0;i<8;++i) {
        store256_al(&C[i*(size_t)ldc + 0], cacc[i][0]);
        store256_al(&C[i*(size_t)ldc + 4], cacc[i][1]);
    }
}

static inline void micro_kernel_8x16_avx2(const double* restrict AP, const double* restrict BP,
                                          double* restrict C, int ldc, int block_k, int ldp)
{
    __m256d cacc[8][4];
    for (int r=0;r<8;++r) {
        cacc[r][0] = load256_al(&C[r*(size_t)ldc + 0]);
        cacc[r][1] = load256_al(&C[r*(size_t)ldc + 4]);
        cacc[r][2] = load256_al(&C[r*(size_t)ldc + 8]);
        cacc[r][3] = load256_al(&C[r*(size_t)ldc + 12]);
    }

    for (int kk=0; kk<block_k; ++kk) {
        __m256d a[8];
        for (int r=0;r<8;++r) a[r] = _mm256_broadcast_sd(&AP[r*block_k + kk]);
        const double *bptr = &BP[(size_t)kk * ldp];
        __m256d b0 = load256_al(&bptr[0]);
        __m256d b1 = load256_al(&bptr[4]);
        __m256d b2 = load256_al(&bptr[8]);
        __m256d b3 = load256_al(&bptr[12]);
        for (int r=0;r<8;++r) {
            cacc[r][0] = _mm256_fmadd_pd(a[r], b0, cacc[r][0]);
            cacc[r][1] = _mm256_fmadd_pd(a[r], b1, cacc[r][1]);
            cacc[r][2] = _mm256_fmadd_pd(a[r], b2, cacc[r][2]);
            cacc[r][3] = _mm256_fmadd_pd(a[r], b3, cacc[r][3]);
        }
    }

    for (int r=0;r<8;++r) {
        store256_al(&C[r*(size_t)ldc + 0], cacc[r][0]);
        store256_al(&C[r*(size_t)ldc + 4], cacc[r][1]);
        store256_al(&C[r*(size_t)ldc + 8], cacc[r][2]);
        store256_al(&C[r*(size_t)ldc + 12], cacc[r][3]);
    }
}

// -------- Blocked GEMM (BT path) ----------
void blocked_matmul_packed_BT(const double* restrict A, const double* restrict BT, double* restrict C,
    int M, int K, int N,
    int block_outer_i, int block_outer_j, int block_outer_k,
    int micro_i, int micro_j,
    thread_panel_buffers_t* thread_bufs, int nthreads)
{
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i0=0;i0<M;i0+=block_outer_i)
    for (int j0=0;j0<N;j0+=block_outer_j) {
        int tid = omp_get_thread_num();
        tid = (nthreads>0) ? (tid % nthreads) : tid;
        thread_panel_buffers_t* buf = &thread_bufs[tid];
        int pad_i = buf->pad_i;
        int pad_j = buf->pad_j;
        for (int k0=0;k0<K;k0+=block_outer_k) {
            pack_A_block_panel_padded(A, buf->AP, M, K, i0, k0, block_outer_i, block_outer_k, pad_i);
            pack_B_block_panel_padded(BT, buf->BP, K, N, j0, k0, block_outer_j, block_outer_k, pad_j);
            for (int ii=0; ii<pad_i; ii+=micro_i) {
                int mi = (ii+micro_i > block_outer_i) ? (block_outer_i - ii) : micro_i;
                if (mi <= 0) break;
                for (int jj=0; jj<pad_j; jj+=micro_j) {
                    int mj = (jj+micro_j > block_outer_j) ? (block_outer_j - jj) : micro_j;
                    if (mj <= 0) break;
                    double* Cblock = &C[(size_t)(i0+ii) * N + (j0+jj)];
                    const double* AP = &buf->AP[(size_t)ii * block_outer_k];
                    const double* BP = &buf->BP[jj]; // BP indexed as kk*pad_j + j
                    if (mi==8 && mj>=16) {
                        micro_kernel_8x16_avx2(AP, BP, Cblock, N, block_outer_k, pad_j);
                    } else if (mi==8 && mj>=8) {
                        micro_kernel_8x8_avx2(AP, BP, Cblock, N, block_outer_k, pad_j);
                    } else if (mi==6 && mj>=8) {
                        micro_kernel_6x8_avx2(AP, BP, Cblock, N, block_outer_k, pad_j);
                    } else if (mi==4 && mj>=8) {
                        micro_kernel_4x8_avx2(AP, BP, Cblock, N, block_outer_k, pad_j);
                    } else {
                        for (int i=0;i<mi;++i)
                            for (int j=0;j<mj;++j)
                                for (int kk=0; kk<block_outer_k; ++kk)
                                    Cblock[i*(size_t)N + j] += AP[(size_t)i * block_outer_k + kk] * BP[(size_t)kk * pad_j + j];
                    }
                }
            }
        }
    }
}

// -------- Fallback no-transpose blocked path ----------
void pack_B_original(const double* restrict B, double* restrict BP,
                     int K, int N, int j_start, int k_start,
                     int block_j, int block_k)
{
    for (int kk=0; kk<block_k; ++kk) {
        PREFETCH(&B[(size_t)(k_start + kk) * N + j_start]);
        for (int j=0;j<block_j;++j) {
            BP[(size_t)kk * block_j + j] = B[(size_t)(k_start + kk) * N + (j_start + j)];
        }
    }
}

void blocked_matmul_noT(const double* restrict A, const double* restrict B, double* restrict C,
    int M, int K, int N,
    int block_outer_i, int block_outer_j, int block_outer_k,
    int micro_i, int micro_j,
    thread_panel_buffers_t* thread_bufs, int nthreads)
{
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i0=0; i0<M; i0+=block_outer_i)
    for (int j0=0; j0<N; j0+=block_outer_j) {
        int tid = omp_get_thread_num();
        tid = (nthreads>0) ? (tid % nthreads) : tid;
        thread_panel_buffers_t* buf = &thread_bufs[tid];
        int pad_i = buf->pad_i;
        int pad_j = buf->pad_j;
        for (int k0=0;k0<K;k0+=block_outer_k) {
            pack_A_block_panel_padded(A, buf->AP, M, K, i0, k0, block_outer_i, block_outer_k, pad_i);
            pack_B_original(B, buf->BP, K, N, j0, k0, block_outer_j, block_outer_k);
            for (int ii=0; ii<block_outer_i; ii+=micro_i) {
                int mi = (ii+micro_i > block_outer_i) ? (block_outer_i-ii) : micro_i;
                for (int jj=0; jj<block_outer_j; jj+=micro_j) {
                    int mj = (jj+micro_j > block_outer_j) ? (block_outer_j-jj) : micro_j;
                    double* Cblock = &C[(size_t)(i0+ii) * N + (j0+jj)];
                    const double* AP = &buf->AP[(size_t)ii * block_outer_k];
                    const double* BP = &buf->BP[jj]; // indexing: BP[kk*block_j + j]
                    if (mi==8 && mj>=16) micro_kernel_8x16_avx2(AP,BP,Cblock,N,block_outer_k,block_outer_j);
                    else if (mi==8 && mj>=8) micro_kernel_8x8_avx2(AP,BP,Cblock,N,block_outer_k,block_outer_j);
                    else if (mi==6 && mj>=8) micro_kernel_6x8_avx2(AP,BP,Cblock,N,block_outer_k,block_outer_j);
                    else if (mi==4 && mj>=8) micro_kernel_4x8_avx2(AP,BP,Cblock,N,block_outer_k,block_outer_j);
                    else {
                        for (int i=0;i<mi;++i)
                            for (int j=0;j<mj;++j)
                                for (int kk=0; kk<block_outer_k; ++kk)
                                    Cblock[i*(size_t)N + j] += AP[(size_t)i * block_outer_k + kk] * buf->BP[(size_t)kk * block_outer_j + (jj + j)];
                    }
                }
            }
        }
    }
}

// -------- Async loader thread ----------
void signal_loader_done(int i) {
    pthread_mutex_lock(&loader_mutex[i]);
    buf_done[i] = 1;
    buf_ready[i] = 0;
    pthread_cond_signal(&loader_cond[i]);
    pthread_mutex_unlock(&loader_mutex[i]);
}

/* wait_loader_ready: returns 0 on success, -1 on loader_error */
int wait_loader_ready(int i) {
    pthread_mutex_lock(&loader_mutex[i]);
    while (!buf_ready[i] && !loader_error) {
        pthread_cond_wait(&loader_cond[i], &loader_mutex[i]);
    }
    int err = loader_error ? -1 : 0;
    pthread_mutex_unlock(&loader_mutex[i]);
    return err;
}

#if defined(__linux__)
void async_prefetch_linux(int fd, void *buf, size_t offset, size_t length, struct iocb *iocb, io_context_t *ctx) {
    io_prep_pread(iocb, fd, buf, length, offset);
    struct iocb *cbs[1] = {iocb};
    io_submit(*ctx, 1, cbs);
}
void wait_async_linux(io_context_t ctx, struct iocb *iocb) {
    struct io_event events[1];
    io_getevents(ctx, 1, 1, events, NULL);
}
#endif

void* loader_thread_fn(void* arg) {
    loader_args_t* args = (loader_args_t*)arg;
    int M=args->M, K=args->K, N=args->N;
    int batch=args->batch, datasets=args->datasets;

#if defined(__linux__)
    io_context_t ctxs[NUM_BUFFERS];
    struct iocb iocb_a[NUM_BUFFERS], iocb_b[NUM_BUFFERS];
    int fa_fd = fileno(args->fa), fb_fd = fileno(args->fb);
    for (int i=0;i<NUM_BUFFERS;++i) io_setup(1, &ctxs[i]);
#elif defined(_WIN32)
    OVERLAPPED ov_a[NUM_BUFFERS], ov_b[NUM_BUFFERS];
#endif

    for (int d=0; d<datasets && !loader_error; d+=batch) {
        int buf = (d / batch) % NUM_BUFFERS;
        // Wait until this buffer is free (main thread signals when done)
        pthread_mutex_lock(&loader_mutex[buf]);
        while (!buf_done[buf] && !loader_error) pthread_cond_wait(&loader_cond[buf], &loader_mutex[buf]);
        pthread_mutex_unlock(&loader_mutex[buf]);
        if (loader_error) break;

        size_t n_this = (d + batch > datasets) ? (datasets - d) : batch;
        size_t bytes_a = n_this * (size_t)M * K * sizeof(double);
        size_t bytes_b = n_this * (size_t)K * N * sizeof(double);
        size_t offset_a = (size_t)d * (size_t)M * K * sizeof(double);
        size_t offset_b = (size_t)d * (size_t)K * N * sizeof(double);

#if defined(__linux__)
        async_prefetch_linux(fa_fd, args->A_buf[buf], offset_a, bytes_a, &iocb_a[buf], &ctxs[buf]);
        async_prefetch_linux(fb_fd, args->B_buf[buf], offset_b, bytes_b, &iocb_b[buf], &ctxs[buf]);
        wait_async_linux(ctxs[buf], &iocb_a[buf]);
        wait_async_linux(ctxs[buf], &iocb_b[buf]);
#elif defined(_WIN32)
        memset(&ov_a[buf],0,sizeof(OVERLAPPED)); memset(&ov_b[buf],0,sizeof(OVERLAPPED));
        ov_a[buf].Offset = (DWORD)(offset_a & 0xFFFFFFFF); ov_a[buf].OffsetHigh = (DWORD)((offset_a>>32)&0xFFFFFFFF);
        ov_b[buf].Offset = (DWORD)(offset_b & 0xFFFFFFFF); ov_b[buf].OffsetHigh = (DWORD)((offset_b>>32)&0xFFFFFFFF);
        if (!ReadFile(args->fa_h, args->A_buf[buf], (DWORD)bytes_a, NULL, &ov_a[buf])) {
            DWORD err = GetLastError();
            if (err != ERROR_IO_PENDING) { fprintf(stderr,"ReadFile A failed: %lu\n",err); loader_error=1; break; }
        }
        if (!ReadFile(args->fb_h, args->B_buf[buf], (DWORD)bytes_b, NULL, &ov_b[buf])) {
            DWORD err = GetLastError();
            if (err != ERROR_IO_PENDING) { fprintf(stderr,"ReadFile B failed: %lu\n",err); loader_error=1; break; }
        }
        DWORD bytes;
        GetOverlappedResult(args->fa_h, &ov_a[buf], &bytes, TRUE);
        GetOverlappedResult(args->fb_h, &ov_b[buf], &bytes, TRUE);
#else
        FILE *fa = args->fa, *fb = args->fb;
        if (fseeko(fa, (off_t)offset_a, SEEK_SET) != 0) { perror("fseeko A"); loader_error=1; break; }
        if (fread(args->A_buf[buf], 1, bytes_a, fa) != bytes_a) { fprintf(stderr,"fread A short\n"); loader_error=1; break; }
        if (fseeko(fb, (off_t)offset_b, SEEK_SET) != 0) { perror("fseeko B"); loader_error=1; break; }
        if (fread(args->B_buf[buf], 1, bytes_b, fb) != bytes_b) { fprintf(stderr,"fread B short\n"); loader_error=1; break; }
#endif

        // signal ready
        pthread_mutex_lock(&loader_mutex[buf]);
        buf_ready[buf] = 1;
        buf_done[buf] = 0;
        pthread_cond_signal(&loader_cond[buf]);
        pthread_mutex_unlock(&loader_mutex[buf]);
    }

#if defined(__linux__)
    for (int i=0;i<NUM_BUFFERS;++i) io_destroy(ctxs[i]);
#endif
    return NULL;
}

// -------- Autotune (unchanged except uses alloc_numa_matrix) ----------
void autotune_blocks(int M, int K, int N, int nthreads,
    int* block_outer_i, int* block_outer_j, int* block_outer_k,
    int* micro_i, int* micro_j)
{
    int cand_i_quick[] = {32,48,64};
    int cand_j_quick[] = {32,48,64,80,96,112,128};
    int cand_k_quick[] = {32,48,64};
    int cand_i_th[] = {32,40,48,56,64,72};
    int cand_j_th[] = {32,40,48,56,64,80,96,128};
    int cand_k_th[] = {32,40,48,56,64};

    int *cand_i, *cand_j, *cand_k;
    int ni,nj,nk;
    if (AUTOTUNE==2) { cand_i = cand_i_th; ni = sizeof(cand_i_th)/sizeof(int);
                       cand_j = cand_j_th; nj = sizeof(cand_j_th)/sizeof(int);
                       cand_k = cand_k_th; nk = sizeof(cand_k_th)/sizeof(int); }
    else { cand_i = cand_i_quick; ni = sizeof(cand_i_quick)/sizeof(int);
           cand_j = cand_j_quick; nj = sizeof(cand_j_quick)/sizeof(int);
           cand_k = cand_k_quick; nk = sizeof(cand_k_quick)/sizeof(int); }

    int cand_mi[] = {4,6,8};
    int nm = sizeof(cand_mi)/sizeof(int);
    double best_g = 0.0;
    int best_i=cand_i[0], best_j=cand_j[0], best_k=cand_k[0];
    int best_mi=4, best_mj=8;

    int testM = (AUTOTUNE==2)?M : (M>256?256:M);
    int testK = (AUTOTUNE==2)?K : (K>256?256:K);
    int testN = (AUTOTUNE==2)?N : (N>256?256:N);

    printf("Autotune: mode=%d test dims=%dx%dx%d threads=%d\n", AUTOTUNE, testM,testK,testN,nthreads);
    fflush(stdout);

    for (int ii=0; ii<ni; ++ii) {
        for (int jj=0; jj<nj; ++jj) {
            for (int kk=0; kk<nk; ++kk) {
                int bi=cand_i[ii], bj=cand_j[jj], bk=cand_k[kk];
                if (bi>testM || bk>testK || bj>testN) continue;
                for (int mi_idx=0; mi_idx<nm; ++mi_idx) {
                    int mi = cand_mi[mi_idx];
                    int mj = (mi==8) ? 16 : 8;
                    int bi_round = round_up(bi, mi);
                    int bj_round = round_up(bj, mj);
                    int bk_round = bk;
                    printf(" try %dx%dx%d micro=%dx%d (padded %dx%dx%d)...\n", bi,bj,bk,mi,mj,bi_round,bj_round,bk_round); fflush(stdout);

                    thread_panel_buffers_t* bufs = alloc_panel_buffers(nthreads, bi_round, bj_round, bk_round, mi, mj);
                    double *A = alloc_numa_matrix(testM, testK, 1, 0, 0);
                    double *B = alloc_numa_matrix(testK, testN, 1, 0, 0);
                    double *C = alloc_numa_matrix(testM, testN, 1, 0, 0);
                    double *BT = alloc_numa_matrix(testN, testK, 1, 0, 0);
                    random_matrix(A, testM, testK, 123);
                    random_matrix(B, testK, testN, 456);
                    transpose_matrix_rect(B, BT, testK, testN, bk_round);

                    blocked_matmul_packed_BT(A, BT, C, testM, testK, testN, bi_round, bj_round, bk_round, mi, mj, bufs, nthreads);

                    int repeats = 0;
                    double acc_t = 0.0;
                    double best_local = 0.0;
                    while ((acc_t < AUTOTUNE_MIN_TIME && repeats < 50) || repeats < AUTOTUNE_MIN_REPEATS) {
                        double t0 = omp_get_wtime();
                        blocked_matmul_packed_BT(A, BT, C, testM, testK, testN, bi_round, bj_round, bk_round, mi, mj, bufs, nthreads);
                        double t1 = omp_get_wtime();
                        double dt = t1 - t0;
                        if (dt < 1e-9) dt = 1e-9;
                        acc_t += dt; repeats++;
                        double g = 2.0 * (double)testM * (double)testK * (double)testN / (1e9 * dt);
                        if (g > best_local) best_local = g;
                    }
                    double chosen_g = best_local;
                    printf("   -> best GFLOPS=%.2f (repeats=%d)\n", chosen_g, repeats); fflush(stdout);
                    if (chosen_g > best_g) {
                        best_g = chosen_g;
                        best_i = bi_round; best_j = bj_round; best_k = bk_round;
                        best_mi = mi; best_mj = mj;
                    }
                    free_panel_buffers(bufs, nthreads);
                    free_matrix(A); free_matrix(B); free_matrix(C); free_matrix(BT);
                }
            }
        }
    }
    printf("Autotune selected I=%d J=%d K=%d micro=%d x %d (%.2f GFLOPS)\n", best_i,best_j,best_k,best_mi,best_mj,best_g);
    *block_outer_i = best_i; *block_outer_j = best_j; *block_outer_k = best_k;
    *micro_i = best_mi; *micro_j = best_mj;
}

// -------- CLI / CSV / compare ----------
int compare_matrices(const double* restrict A, const double* restrict B, int rows, int cols) {
    const double atol = 1e-6;
    const double rtol = 1e-12;
    long long total = (long long)rows * (long long)cols;
    for (long long i=0;i<total;++i) {
        double a = A[i], b = B[i];
        double diff = fabs(a-b);
        double tol = atol + rtol * fmax(1.0, fmax(fabs(a), fabs(b)));
        if (diff > tol) {
            printf("First mismatch at idx %lld: %.12f vs %.12f (diff %.12e tol %.12e)\n", i, A[i], B[i], diff, tol);
            return 0;
        }
    }
    return 1;
}

void print_csv_header(FILE *csv, int M,int K,int N) {
    fprintf(csv, "# BLOCK_I=%d,BLOCK_J=%d,BLOCK_K=%d,MICRO_I=%d,MICRO_J=%d,THREADS=%d,BATCH=%d,AUTOTUNE=%d\n",
        BLOCK_OUTER_I,BLOCK_OUTER_J,BLOCK_OUTER_K,MICRO_I,MICRO_J, THREADS, BATCH, AUTOTUNE);
    fprintf(csv, "Dataset,Seq_NoT(s),Par_NoT(s),Speedup_NoT,GFLOPS_NoT,Seq_T(s),Par_T(s),Trans_T(s),Speedup_T,GFLOPS_T,GFLOPS_Par_NoT,GFLOPS_Par_T\n");
}

void parse_cli(int argc, char** argv, int* M_, int* K_, int* N_) {
    for (int i=1;i<argc;++i) {
        if (strncmp(argv[i],"--block_outer_i=",16)==0) BLOCK_OUTER_I=atoi(argv[i]+16);
        else if (strncmp(argv[i],"--block_outer_j=",16)==0) BLOCK_OUTER_J=atoi(argv[i]+16);
        else if (strncmp(argv[i],"--block_outer_k=",16)==0) BLOCK_OUTER_K=atoi(argv[i]+16);
        else if (strncmp(argv[i],"--micro_i=",10)==0) MICRO_I=atoi(argv[i]+10);
        else if (strncmp(argv[i],"--micro_j=",10)==0) MICRO_J=atoi(argv[i]+10);
        else if (strncmp(argv[i],"--threads=",10)==0) THREADS=atoi(argv[i]+10);
        else if (strncmp(argv[i],"--BATCH=",8)==0) BATCH=atoi(argv[i]+8);
        else if (strncmp(argv[i],"--AUTOTUNE=",10)==0) AUTOTUNE=atoi(argv[i]+10);
        else if (strncmp(argv[i],"--M=",4)==0) *M_=atoi(argv[i]+4);
        else if (strncmp(argv[i],"--K=",4)==0) *K_=atoi(argv[i]+4);
        else if (strncmp(argv[i],"--N=",4)==0) *N_=atoi(argv[i]+4);
        else if (strncmp(argv[i],"--DATASETS=",11)==0) DATASETS=atoi(argv[i]+11);
    }
}

// -------- simple file-generator used if files missing ----------
void generate_input_files(const char* fa_name, const char* fb_name, int M, int K, int N, int datasets) {
    printf("Generating input files for matrices A and B...\n");
    double t0 = omp_get_wtime();
    double* bufA = malloc((size_t)M*K*sizeof(double));
    double* bufB = malloc((size_t)K*N*sizeof(double));
    FILE *fa = fopen(fa_name,"wb"), *fb = fopen(fb_name,"wb");
    if (!fa || !fb) { fprintf(stderr,"Cannot create input files\n"); exit(1); }
    for (int d=0; d<datasets; ++d) {
        random_matrix(bufA, M, K, 1000 + d);
        random_matrix(bufB, K, N, 2000 + d);
        fwrite(bufA, sizeof(double), (size_t)M*K, fa);
        fwrite(bufB, sizeof(double), (size_t)K*N, fb);
    }
    fclose(fa); fclose(fb); free(bufA); free(bufB);
    printf("Files written in %.6f s\n", omp_get_wtime() - t0);
}

// signal handler to flush stdio on Ctrl-C/abort
static void flush_on_signal(int sig) {
    fflush(NULL);                 /* flush all open stdio streams */
    signal(sig, SIG_DFL);         /* restore default and re-raise if needed */
    raise(sig);
}

// -------- main ----------
int main(int argc, char** argv) {
    for (int i=0;i<NUM_BUFFERS;++i){ buf_ready[i]=0; buf_done[i]=1; }
    int m = DEFAULT_M, k = DEFAULT_K, n = DEFAULT_N;
    parse_cli(argc, argv, &m, &k, &n);

    int nthreads = THREADS > 0 ? THREADS : omp_get_max_threads();
    omp_set_num_threads(nthreads);

    /* --- FORCE UNBUFFERED OUTPUT FOR IMMEDIATE FLUSHING --- */
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    signal(SIGINT,  flush_on_signal);
    signal(SIGTERM, flush_on_signal);
    signal(SIGABRT, flush_on_signal);
    /* ---------------------------------------------------- */

    if (AUTOTUNE) {
        autotune_blocks(m,k,n,nthreads, &BLOCK_OUTER_I, &BLOCK_OUTER_J, &BLOCK_OUTER_K, &MICRO_I, &MICRO_J);
    }
    if (m<=0 || k<=0 || n<=0) { fprintf(stderr,"Bad matrix dims\n"); return 2; }
    if (BATCH > DATASETS) BATCH = DATASETS;

#ifndef _WIN32
    FILE *fa = fopen("A.bin","rb");
    FILE *fb = fopen("B.bin","rb");
    if (!fa || !fb) {
        if (fa) fclose(fa); if (fb) fclose(fb);
        generate_input_files("A.bin","B.bin", m,k,n, DATASETS);
        fa = fopen("A.bin","rb"); fb = fopen("B.bin","rb");
        if (!fa || !fb) { fprintf(stderr,"Cannot open A.bin/B.bin\n"); return 1; }
    }
    setvbuf(fa, NULL, _IOFBF, 64<<20);
    setvbuf(fb, NULL, _IOFBF, 64<<20);
#else
    HANDLE fa_h = CreateFileA("A.bin", GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_FLAG_OVERLAPPED, NULL);
    HANDLE fb_h = CreateFileA("B.bin", GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_FLAG_OVERLAPPED, NULL);
    if (fa_h==INVALID_HANDLE_VALUE || fb_h==INVALID_HANDLE_VALUE) {
        if (fa_h!=INVALID_HANDLE_VALUE) CloseHandle(fa_h);
        if (fb_h!=INVALID_HANDLE_VALUE) CloseHandle(fb_h);
        generate_input_files("A.bin","B.bin", m,k,n, DATASETS);
        fa_h = CreateFileA("A.bin", GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_FLAG_OVERLAPPED, NULL);
        fb_h = CreateFileA("B.bin", GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_FLAG_OVERLAPPED, NULL);
        if (fa_h==INVALID_HANDLE_VALUE || fb_h==INVALID_HANDLE_VALUE) { fprintf(stderr,"Cannot open files\n"); return 1; }
    }
#endif

    printf("Threads=%d Blocks I=%d J=%d K=%d micro=%dx%d batch=%d datasets=%d\n",
           nthreads, BLOCK_OUTER_I,BLOCK_OUTER_J,BLOCK_OUTER_K, MICRO_I, MICRO_J, BATCH, DATASETS);

    // allocate double-buffers for A and B
    double *A_buf[NUM_BUFFERS], *B_buf[NUM_BUFFERS], *BT_buf, *C_seq, *C_par;
    for (int i=0;i<NUM_BUFFERS;++i) {
        int node = get_numa_node_for_thread(i);
        A_buf[i] = alloc_numa_matrix(m,k,BATCH, PIN_BUF, node);
        B_buf[i] = alloc_numa_matrix(k,n,BATCH, PIN_BUF, node);
    }
    BT_buf = alloc_numa_matrix(n,k,BATCH,0,0); // per-batch transposed B (N x K)
    C_seq = alloc_numa_matrix(m,n,BATCH,0,0);
    C_par = alloc_numa_matrix(m,n,BATCH,0,0);

    loader_args_t loader_args;
#ifndef _WIN32
    loader_args.fa = fa; loader_args.fb = fb;
#else
    loader_args.fa_h = fa_h; loader_args.fb_h = fb_h;
#endif
    for (int i=0;i<NUM_BUFFERS;++i) { loader_args.A_buf[i] = A_buf[i]; loader_args.B_buf[i] = B_buf[i]; }
    loader_args.M = m; loader_args.K = k; loader_args.N = n;
    loader_args.batch = BATCH; loader_args.datasets = DATASETS;

    pthread_t loader_thread;
    if (pthread_create(&loader_thread, NULL, loader_thread_fn, &loader_args) != 0) {
        fprintf(stderr,"Failed to create loader thread\n"); return 1;
    }

    FILE *csv = fopen("matrix_perf.csv","w");
    if (!csv) csv = stdout;
    print_csv_header(csv, m,k,n);

    thread_panel_buffers_t* thread_bufs = alloc_panel_buffers(nthreads, BLOCK_OUTER_I, BLOCK_OUTER_J, BLOCK_OUTER_K, MICRO_I, MICRO_J);

    double total_seq_noT=0.0, total_par_noT=0.0, total_seq_T=0.0, total_par_T=0.0, total_trans_T=0.0;
    double min_seq_noT=1e9, min_par_noT=1e9, min_seq_T=1e9, min_par_T=1e9, min_trans_T=1e9;
    double max_seq_noT=0.0, max_par_noT=0.0, max_seq_T=0.0, max_par_T=0.0, max_trans_T=0.0;
    const double GFLOPS_FACTOR = 2.0 * (double)m * (double)k * (double)n / 1e9;
    int correctness = 1;

    double start_time = omp_get_wtime();
    for (int d=0; d<DATASETS; d+=BATCH) {
        int n_this = (d + BATCH > DATASETS) ? (DATASETS - d) : BATCH;
        int buf_idx = (d / BATCH) % NUM_BUFFERS;
        if (wait_loader_ready(buf_idx) != 0) { fprintf(stderr,"Loader thread error\n"); signal_loader_done(buf_idx); break; }

        for (int b=0;b<n_this;++b) {
            double *Aptr = A_buf[buf_idx] + (size_t)b * m * k;
            double *Bptr = B_buf[buf_idx] + (size_t)b * k * n;
            double *BTptr = BT_buf + (size_t)b * n * k;
            double *Cseq = C_seq + (size_t)b * m * n;
            double *Cpar = C_par + (size_t)b * m * n;

            // Sequential (no transpose) - force 1 thread for true baseline
            parallel_memset(Cseq, m, n);
            omp_set_num_threads(1);
            double t0 = omp_get_wtime();
            blocked_matmul_noT(Aptr, Bptr, Cseq, m,k,n, BLOCK_OUTER_I,BLOCK_OUTER_J,BLOCK_OUTER_K, MICRO_I, MICRO_J, thread_bufs, /*nthreads*/1);
            double t1 = omp_get_wtime();
            double seq_noT = t1 - t0;
            omp_set_num_threads(nthreads);

            // Parallel (no transpose)
            parallel_memset(Cpar, m, n);
            double t2 = omp_get_wtime();
            blocked_matmul_noT(Aptr, Bptr, Cpar, m,k,n, BLOCK_OUTER_I,BLOCK_OUTER_J,BLOCK_OUTER_K, MICRO_I, MICRO_J, thread_bufs, nthreads);
            double t3 = omp_get_wtime();
            double par_noT = t3 - t2;

            if (!compare_matrices(Cseq, Cpar, m, n)) { correctness = 0; printf("Mismatch noT dataset %d\n", d+b+1); }

            // Transpose B for BT path (blocked transpose)
            double t0t = omp_get_wtime();
            transpose_matrix_rect(Bptr, BTptr, k, n, 64);
            double t1t = omp_get_wtime();
            double trans_T = t1t - t0t;

            // Using packed/padded BT path (sequential)
            parallel_memset(Cseq, m, n);
            omp_set_num_threads(1);
            double t4 = omp_get_wtime();
            blocked_matmul_packed_BT(Aptr, BTptr, Cseq, m,k,n, BLOCK_OUTER_I,BLOCK_OUTER_J,BLOCK_OUTER_K, MICRO_I, MICRO_J, thread_bufs, /*nthreads*/1);
            double t5 = omp_get_wtime();
            double seq_T = trans_T + (t5 - t4);
            omp_set_num_threads(nthreads);

            // Parallel using BT
            parallel_memset(Cpar, m, n);
            double t6 = omp_get_wtime();
            blocked_matmul_packed_BT(Aptr, BTptr, Cpar, m,k,n, BLOCK_OUTER_I,BLOCK_OUTER_J,BLOCK_OUTER_K, MICRO_I, MICRO_J, thread_bufs, nthreads);
            double t7 = omp_get_wtime();
            double par_T = trans_T + (t7 - t6);

            if (!compare_matrices(Cseq, Cpar, m, n)) { correctness = 0; printf("Mismatch T dataset %d\n", d+b+1); }

            double speed_noT = (par_noT>0.0) ? (seq_noT/par_noT) : 0.0;
            double speed_T = (par_T>0.0) ? (seq_T/par_T) : 0.0;

            double gflops_noT = (seq_noT>0.0) ? GFLOPS_FACTOR / seq_noT : 0.0;
            double gflops_T = (seq_T>0.0) ? GFLOPS_FACTOR / seq_T : 0.0;
            double gflops_par_noT = (par_noT>0.0) ? GFLOPS_FACTOR / par_noT : 0.0;
            double gflops_par_T = (par_T>0.0) ? GFLOPS_FACTOR / par_T : 0.0;

            fprintf(csv, "%d,%.8lf,%.8lf,%.4lf,%.4lf,%.8lf,%.8lf,%.8lf,%.4lf,%.4lf,%.4lf,%.4lf\n",
                    d + b + 1, seq_noT, par_noT, speed_noT, gflops_noT,
                    seq_T, par_T, trans_T, speed_T, gflops_T, gflops_par_noT, gflops_par_T);

            total_seq_noT += seq_noT; total_par_noT += par_noT;
            total_seq_T += seq_T; total_par_T += par_T; total_trans_T += trans_T;
            if (seq_noT < min_seq_noT) min_seq_noT = seq_noT;
            if (par_noT < min_par_noT) min_par_noT = par_noT;
            if (seq_T < min_seq_T) min_seq_T = seq_T;
            if (par_T < min_par_T) min_par_T = par_T;
            if (trans_T < min_trans_T) min_trans_T = trans_T;
            if (seq_noT > max_seq_noT) max_seq_noT = seq_noT;
            if (par_noT > max_par_noT) max_par_noT = par_noT;
            if (seq_T > max_seq_T) max_seq_T = seq_T;
            if (par_T > max_par_T) max_par_T = par_T;
            if (trans_T > max_trans_T) max_trans_T = trans_T;
        }

        if ((d+1) % 100 == 0) {
            double elapsed = omp_get_wtime() - start_time;
            double percent = 100.0 * (d+1) / DATASETS;
            double estimated_total = elapsed / (percent / 100.0);
            double remaining = estimated_total - elapsed;
            printf("Processed %d/%d datasets (%.2f%%), elapsed: %.2fs, remaining: %.2fs\n", d+1, DATASETS, percent, elapsed, remaining);
            fflush(stdout);
        }
        signal_loader_done(buf_idx);
    }

    int total_items = DATASETS;
    double avg_seq_noT = total_seq_noT / total_items;
    double avg_par_noT = total_par_noT / total_items;
    double avg_seq_T = total_seq_T / total_items;
    double avg_par_T = total_par_T / total_items;
    double avg_trans_T = total_trans_T / total_items;
    double avg_speed_noT = (avg_par_noT>0)?(avg_seq_noT/avg_par_noT):0.0;
    double avg_speed_T = (avg_par_T>0)?(avg_seq_T/avg_par_T):0.0;
    double avg_gflops_noT = (avg_seq_noT>0)? GFLOPS_FACTOR / avg_seq_noT : 0.0;
    double avg_gflops_T = (avg_seq_T>0)? GFLOPS_FACTOR / avg_seq_T : 0.0;
    double avg_gflops_par_noT = (avg_par_noT>0)? GFLOPS_FACTOR / avg_par_noT : 0.0;
    double avg_gflops_par_T = (avg_par_T>0)? GFLOPS_FACTOR / avg_par_T : 0.0;

    printf("\n--- Summary (averages over %d items) ---\n", total_items);
    printf("Avg Seq (noT): %.6f s (%.2f GFLOPS)\n", avg_seq_noT, avg_gflops_noT);
    printf("Avg Par (noT): %.6f s (speedup %.2f) (%.2f GFLOPS)\n", avg_par_noT, avg_speed_noT, avg_gflops_par_noT);
    printf("Avg Seq (T)  : %.6f s (%.2f GFLOPS)\n", avg_seq_T, avg_gflops_T);
    printf("Avg Par (T)  : %.6f s (speedup %.2f) (%.2f GFLOPS)\n", avg_par_T, avg_speed_T, avg_gflops_par_T);
    printf("----------------------------------------\n");
    printf("Min Seq noT: %.6f  Max Seq noT: %.6f\n", min_seq_noT, max_seq_noT);
    printf("Min Par noT: %.6f  Max Par noT: %.6f\n", min_par_noT, max_par_noT);
    printf("Min Seq T:   %.6f  Max Seq T:   %.6f\n", min_seq_T, max_seq_T);
    printf("Min Par T:   %.6f  Max Par T:   %.6f\n", min_par_T, max_par_T);
    printf("Min Trans:   %.6f  Max Trans:   %.6f\n", min_trans_T, max_trans_T);

    fprintf(csv, "Average,%.8lf,%.8lf,%.4lf,%.4lf,%.8lf,%.8lf,%.8lf,%.4lf,%.4lf,%.4lf,%.4lf\n",
            avg_seq_noT, avg_par_noT, avg_speed_noT, avg_gflops_noT,
            avg_seq_T, avg_par_T, avg_trans_T, avg_speed_T, avg_gflops_T, avg_gflops_par_noT, avg_gflops_par_T);
    if (csv != stdout) fclose(csv);

    pthread_join(loader_thread, NULL);

    for (int i=0;i<NUM_BUFFERS;++i) { free_matrix(A_buf[i]); free_matrix(B_buf[i]); }
    free_matrix(BT_buf); free_matrix(C_seq); free_matrix(C_par);
    free_panel_buffers(thread_bufs, nthreads);

#ifndef _WIN32
    fclose(fa); fclose(fb);
#else
    CloseHandle(fa_h); CloseHandle(fb_h);
#endif

    printf("\nDone. Correctness: %s\n", correctness ? "PASSED" : "FAILED");
    printf("Results in matrix_perf.csv\n");
    return 0;
}
