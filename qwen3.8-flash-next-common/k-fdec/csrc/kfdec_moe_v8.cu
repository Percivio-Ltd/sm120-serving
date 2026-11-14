// K-FDEC v8: v7 plus unrouted slots. A routing id outside [0, E) (vLLM's topk_ids == -1, "unrouted token slot")
// contributes nothing: both kernels read it as -1, kernel A's CTA for that slot exits, kernel B's dedupe skips it, so
// no weights, act row or topk weight of that slot is read; a token with no valid id gets an output row of zeros.
// v7 indexed the expert tensors with it (illegal memory access in the engine's M = 4 warmup, where all ids are -1).
//
// K-FDEC v7: v6 plus an optional calls counter for the serving evidence (count = true): thread 0 of kernel B's CTA
// (0, 0) increments a device counter and copies the new value to a word of mapped pinned host memory, which
// calls() reads without any CUDA call (so a log thread can show the count rising during CUDA-graph replay, when the
// Python wrapper no longer runs). counter_init() allocates both once, before any counting call or graph capture.
//
// K-FDEC v6: small-M routed-MoE decode for Flash NVFP4 experts in vLLM's Marlin layout (sm_120).
//
// v6 = v3 with explicit per-kernel register caps (__maxnreg__, mode `regs`) instead of v4/v5's
// __launch_bounds__(256, 2), so an A CTA and a B CTA fit on one SM together (v3: A 120 + B 148 registers x 256
// threads > 64 K) without spilling and B's dedupe and first weight batch overlap A under PDL; the per-expert
// global scales are loaded before the row loop; v4's pfb (B L2-prefetches its rows past the first batch before
// waiting on A); v3's A-side pf (slower) is dropped.
// v3 = v2 plus: kernel A finds its expert by slot (CTA y = routing slot y; duplicate slots exit after one
// __syncthreads_or).
// Same math, layout and numerics as v1 (see the layout notes below); v2 changed the schedule:
//  - kernel A loads RBA k16 rows per warp per round trip (template), kernel B loads RBB rows per expert;
//  - kernel B reduces the expert groups of an output chunk across a thread-block cluster (1, G, 1) through
//    distributed shared memory (fixed order, no global workspace or ticket counter);
//  - optional programmatic dependent launch: A waits on its predecessor before reading routing, then lets B launch;
//    B dedupes routing and loads its expert weights and scales (independent of A) before waiting for A's act.
//
// Reads the Marlin tensors already resident for the incumbent (no second weight copy):
//   w13 [E, H/16, 4I] int32, s13 [E, H/16, 2I] S0E5M3 bytes, g13 [E] fp32 (Marlin-processed global scale),
//   w2  [E, I/16, 2H] int32, s2  [E, I/16, H]  bytes,       g2  [E] fp32.
// In a Marlin k16 row, 64-column chunk c is 128 int32 words; lane l (g = l / 4, t = l % 4) owns words 4l..4l+3 and
// word j is exactly the m16n8k16 A fragment of the 16x16 tile (n = 64c + 16j + g (+8), k = 16r + 2t (+1, +8, +9)):
// nibble pairs (q, q + 4) sit in the two bf16 halves. The lane's 8 scales for the chunk are the 8 bytes at
// 64c + 8g; tile j uses byte 4(j >> 1) + (j & 1) for row g and that + 2 for row g + 8.
// Swap-AB: weights are the MMA A operand (16 output rows), tokens the N operand (8 per MMA).
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

namespace cg = cooperative_groups;

constexpr int WARPS = 8;
constexpr int THREADS = WARPS * 32;
constexpr int MAX_SLOTS = 256;

// Calls counter (see counter_init): device counter, mapped pinned host word (host and device views), its device.
unsigned long long* g_cnt = nullptr;
unsigned long long* g_cnt_h = nullptr;
unsigned long long* g_cnt_hd = nullptr;
int g_cnt_dev = -1;
constexpr int MAX_E = 512;
// A = e2m1 * 2^-126 (raw bf16 bits) times h * 2^110 (scale as bf16), i.e. e2m1 * h * 2^-16; the epilogue applies
// g * 2^-110, so value = e2m1 * h * g * 2^-126 exactly as Marlin defines it.
constexpr float POST = 7.703719777548943e-34f;  // 2^-110

__device__ __forceinline__ void mma16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t bmul2(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}

// Two S0E5M3 bytes (at bits 0 and 16 of v) -> bf16x2 of h * 2^110; 0 stays 0 (nonzero bytes have bit 7 set).
__device__ __forceinline__ uint32_t scale_pair(uint32_t v) {
  const uint32_t hi = (v >> 7) & 0x00010001u;
  return ((v & 0x007F007Fu) << 4) + hi * 0x7700u;
}

// Dequantize one Marlin word into the A fragment, scaled by sg (row g) and sg8 (row g + 8), both bf16x2 duplicated.
__device__ __forceinline__ void dequant(uint32_t w, uint32_t sg, uint32_t sg8, uint32_t (&a)[4]) {
  const uint32_t x0 = w & 0x000F000Fu, x2 = (w >> 4) & 0x000F000Fu;
  const uint32_t x1 = (w >> 8) & 0x000F000Fu, x3 = (w >> 12) & 0x000F000Fu;
  auto raw = [](uint32_t x) { return ((x & 0x00070007u) << 6) | ((x & 0x00080008u) << 12); };
  a[0] = bmul2(raw(x0), sg);
  a[1] = bmul2(raw(x1), sg8);
  a[2] = bmul2(raw(x2), sg);
  a[3] = bmul2(raw(x3), sg8);
}

// Scales of tile j from the lane's 8 bytes (lo = bytes 0-3, hi = bytes 4-7): row g / g + 8, duplicated halves.
__device__ __forceinline__ void tile_scales(uint32_t lo, uint32_t hi, uint32_t (&sg)[4], uint32_t (&sg8)[4]) {
  const uint32_t p0 = scale_pair(lo & 0x00FF00FFu), p1 = scale_pair((lo >> 8) & 0x00FF00FFu);
  const uint32_t p2 = scale_pair(hi & 0x00FF00FFu), p3 = scale_pair((hi >> 8) & 0x00FF00FFu);
  sg[0] = __byte_perm(p0, 0, 0x1010); sg8[0] = __byte_perm(p0, 0, 0x3232);
  sg[1] = __byte_perm(p1, 0, 0x1010); sg8[1] = __byte_perm(p1, 0, 0x3232);
  sg[2] = __byte_perm(p2, 0, 0x1010); sg8[2] = __byte_perm(p2, 0, 0x3232);
  sg[3] = __byte_perm(p3, 0, 0x1010); sg8[3] = __byte_perm(p3, 0, 0x3232);
}

__device__ __forceinline__ int4 ldw(const int32_t* p) {
  int4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

__device__ __forceinline__ uint2 lds8(const uint8_t* p) {
  uint2 r;
  asm volatile("ld.global.nc.v2.u32 {%0,%1}, [%2];\n" : "=r"(r.x), "=r"(r.y) : "l"(p));
  return r;
}

// Routing id as read by both kernels: the expert, or -1 for an id outside [0, E) (unrouted slot).
template <typename IdT>
__device__ __forceinline__ int routed_id(IdT v, int E) {
  return (v >= 0 && v < E) ? static_cast<int>(v) : -1;
}

// Distinct experts in slot order: s_ids[S] (expert per slot, -1 unrouted), s_own[D] (owner = first slot of each
// distinct expert; unrouted slots own nothing).
template <typename IdT>
__device__ int distinct_experts(const IdT* ids, int S, int E, int* s_ids, int* s_own, int* s_first, int* s_cnt) {
  const int tid = threadIdx.x;
  for (int i = tid; i < E; i += THREADS) s_first[i] = MAX_SLOTS;
  if (tid < S) s_ids[tid] = routed_id(ids[tid], E);
  __syncthreads();
  const bool routed = tid < S && s_ids[tid] >= 0;
  if (routed) atomicMin(s_first + s_ids[tid], tid);
  __syncthreads();
  const bool owner = routed && s_first[s_ids[tid]] == tid;
  const unsigned bal = __ballot_sync(0xffffffffu, owner);
  const int warp = tid >> 5, lane = tid & 31;
  if (lane == 0) s_cnt[warp] = __popc(bal);
  __syncthreads();
  int base = 0, total = 0;
  for (int w = 0; w < WARPS; ++w) {
    base += (w < warp) ? s_cnt[w] : 0;
    total += s_cnt[w];
  }
  if (owner) s_own[base + __popc(bal & ((1u << lane) - 1))] = tid;
  __syncthreads();
  return total;
}

// Slot (t * topk + k) of token t for expert e, or -1.
__device__ __forceinline__ int token_slot(const int* s_ids, int t, int M, int topk, int e) {
  if (t >= M) return -1;
  int r = -1;
  for (int k = 0; k < topk; ++k) r = (s_ids[t * topk + k] == e) ? t * topk + k : r;
  return r;
}


__device__ __forceinline__ void prefetch_l2(const void* p, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(p), "r"(bytes) : "memory");
}

// CTA `part` of `parts` prefetches its share of [base, base + bytes) into L2 in 4 KiB pieces (bytes % 16 == 0).
__device__ __forceinline__ void prefetch_share(const char* base, size_t bytes, int part, int parts) {
  constexpr size_t PIECE = 4096;
  const size_t n = (bytes + PIECE - 1) / PIECE;
  for (size_t k = part + (size_t)parts * threadIdx.x; k < n; k += (size_t)parts * THREADS) {
    const size_t off = k * PIECE;
    prefetch_l2(base + off, static_cast<uint32_t>(bytes - off < PIECE ? bytes - off : PIECE));
  }
}

__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }
__device__ __forceinline__ void pdl_launch() { asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); }

template <int NT, typename IdT, int RBA, int RA>
__global__ void __maxnreg__(RA) moe_up_kernel(
    const __nv_bfloat16* __restrict__ x, int M, const int32_t* __restrict__ w13, const uint8_t* __restrict__ s13,
    const float* __restrict__ g13, const IdT* __restrict__ ids, int topk, __nv_bfloat16* __restrict__ act, int H,
    int I, int E) {
  constexpr int PAIRS = 4 * NT * 4;
  __shared__ int s_ids[MAX_SLOTS];
  __shared__ float red[WARPS][32][PAIRS + 1];
  __shared__ float gsum[32][PAIRS + 1];
  pdl_wait();
  pdl_launch();
  const int S = M * topk, y = blockIdx.y;
  if (threadIdx.x < S) s_ids[threadIdx.x] = routed_id(ids[threadIdx.x], E);
  __syncthreads();
  const int e = s_ids[y];
  if (e < 0) return;  // unrouted slot (uniform over the CTA): no act row
  if (__syncthreads_or(threadIdx.x < y && s_ids[threadIdx.x] == e)) return;  // expert handled by an earlier slot
  const int cp = blockIdx.x;  // gate chunk cp, up chunk cp + I / 64
  const float alpha = __ldg(g13 + e) * POST;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tq = lane & 3;
  const int R = H >> 4, N = 2 * I, cu = cp + I / 64;
  const int32_t* wb = w13 + (size_t)e * R * (2 * N) + 4 * lane;
  const uint8_t* sb = s13 + (size_t)e * R * N + 8 * g;
  const __nv_bfloat16* xr[NT];
  bool tv[NT];
#pragma unroll
  for (int n = 0; n < NT; ++n) {
    tv[n] = (g + 8 * n) < M;
    xr[n] = x + (size_t)(tv[n] ? g + 8 * n : 0) * H + 2 * tq;
  }
  float acc[2][4][NT][4];
#pragma unroll
  for (int h = 0; h < 2; ++h)
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int n = 0; n < NT; ++n)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[h][j][n][i] = 0.f;
  for (int r0 = warp; r0 < R; r0 += WARPS * RBA) {
    int4 wv[RBA][2];
    uint2 sv[RBA][2];
    uint32_t xb[RBA][NT][2];
#pragma unroll
    for (int u = 0; u < RBA; ++u) {
      const int r = r0 + u * WARPS;
      if (r < R) {
        wv[u][0] = ldw(wb + (size_t)r * (2 * N) + 128 * cp);
        wv[u][1] = ldw(wb + (size_t)r * (2 * N) + 128 * cu);
        sv[u][0] = lds8(sb + (size_t)r * N + 64 * cp);
        sv[u][1] = lds8(sb + (size_t)r * N + 64 * cu);
#pragma unroll
        for (int n = 0; n < NT; ++n) {
          xb[u][n][0] = tv[n] ? __ldg(reinterpret_cast<const uint32_t*>(xr[n] + 16 * r)) : 0u;
          xb[u][n][1] = tv[n] ? __ldg(reinterpret_cast<const uint32_t*>(xr[n] + 16 * r + 8)) : 0u;
        }
      }
    }
#pragma unroll
    for (int u = 0; u < RBA; ++u) {
      const int r = r0 + u * WARPS;
      if (r < R) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          uint32_t sg[4], sg8[4];
          tile_scales(sv[u][h].x, sv[u][h].y, sg, sg8);
          const uint32_t wq[4] = {(uint32_t)wv[u][h].x, (uint32_t)wv[u][h].y, (uint32_t)wv[u][h].z,
                                  (uint32_t)wv[u][h].w};
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            uint32_t a[4];
            dequant(wq[j], sg[j], sg8[j], a);
#pragma unroll
            for (int n = 0; n < NT; ++n) mma16816(acc[h][j][n], a, xb[u][n][0], xb[u][n][1]);
          }
        }
      }
    }
  }
  // Fixed-order reduction over warps: gate sums first, then up sums and the epilogue.
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int i = 0; i < 4; ++i) red[warp][lane][(j * NT + n) * 4 + i] = acc[0][j][n][i];
  __syncthreads();
  for (int p = threadIdx.x; p < 32 * PAIRS; p += THREADS) {
    const int l = p / PAIRS, v = p % PAIRS;
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) s += red[w][l][v];
    gsum[l][v] = s;
  }
  __syncthreads();
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int i = 0; i < 4; ++i) red[warp][lane][(j * NT + n) * 4 + i] = acc[1][j][n][i];
  __syncthreads();
  for (int p = threadIdx.x; p < 32 * PAIRS; p += THREADS) {
    const int l = p / PAIRS, v = p % PAIRS;  // v = (j * NT + n) * 4 + i
    float us = 0.f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) us += red[w][l][v];
    const int i = v & 3, n = (v >> 2) % NT, j = (v >> 2) / NT;
    const int row = 64 * cp + 16 * j + (l >> 2) + 8 * (i >> 1);  // index in [0, I)
    const int t = 8 * n + 2 * (l & 3) + (i & 1);
    const int slot = token_slot(s_ids, t, M, topk, e);
    if (slot >= 0) {
      const float gv = gsum[l][v] * alpha, uv = us * alpha;
      act[(size_t)slot * I + row] = __float2bfloat16(gv / (1.f + expf(-gv)) * uv);
    }
  }
}

// Grid (H / 64, G) with cluster (1, G, 1): CTA (c, gi) sums experts [gi D / G, (gi + 1) D / G) for output chunk c.
template <int NT, typename IdT, int RBB, int RB>
__global__ void __maxnreg__(RB) moe_down_kernel(
    const __nv_bfloat16* __restrict__ act, int M, const int32_t* __restrict__ w2, const uint8_t* __restrict__ s2,
    const float* __restrict__ g2, const IdT* __restrict__ ids, const float* __restrict__ tw, int topk,
    __nv_bfloat16* __restrict__ out, int H, int I, int E, bool pfb, unsigned long long* __restrict__ cnt,
    volatile unsigned long long* cnt_h) {
  constexpr int VALS = 4 * NT * 4, P = 32 * VALS, PT = (P + THREADS - 1) / THREADS;
  __shared__ int s_ids[MAX_SLOTS], s_own[MAX_SLOTS], s_first[MAX_E], s_cnt[WARPS];
  __shared__ float red[WARPS][32][VALS + 1];
  __shared__ float part_s[P];
  const int S = M * topk;
  // The counter's atomic is issued before the dedupe and its result used after it, hiding the round trip.
  const bool counts = cnt != nullptr && blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0;
  unsigned long long calls = 0;
  if (counts) calls = atomicAdd(cnt, 1ULL) + 1;
  const int D = distinct_experts(ids, S, E, s_ids, s_own, s_first, s_cnt);
  if (counts) *cnt_h = calls;
  const int c = blockIdx.x, G = gridDim.y, gi = blockIdx.y;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tq = lane & 3;
  const int R = I >> 4;
  const int d0 = (gi * D) / G, d1 = ((gi + 1) * D) / G;
  float acc[4][NT][4];
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[j][n][i] = 0.f;
  for (int d = d0 + warp; d < d1; d += WARPS) {
    const int e = s_ids[s_own[d]];
    const float alpha = __ldg(g2 + e) * POST;
    const int32_t* wb = w2 + (size_t)e * R * (2 * H) + 128 * c + 4 * lane;
    const uint8_t* sb = s2 + (size_t)e * R * H + 64 * c + 8 * g;
    if (pfb && d == d0 + warp) {  // rows past the first register batch: L2 prefetch (weights 512 B, scales 64 B)
      for (int r = RBB + lane; r < R; r += 32) {
        prefetch_l2(wb - 4 * lane + (size_t)r * (2 * H), 512);
        prefetch_l2(sb - 8 * g + (size_t)r * H, 64);
      }
    }
    const __nv_bfloat16* ar[NT];
    bool av[NT];
    float twv[NT][2];
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      const int sl = token_slot(s_ids, g + 8 * n, M, topk, e);
      av[n] = sl >= 0;
      ar[n] = act + (size_t)(av[n] ? sl : 0) * I + 2 * tq;
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int s2l = token_slot(s_ids, 8 * n + 2 * tq + i, M, topk, e);
        twv[n][i] = s2l >= 0 ? tw[s2l] : 0.f;
      }
    }
    float cacc[4][NT][4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int n = 0; n < NT; ++n)
#pragma unroll
        for (int i = 0; i < 4; ++i) cacc[j][n][i] = 0.f;
    for (int r0 = 0; r0 < R; r0 += RBB) {
      int4 wv[RBB];
      uint2 sv[RBB];
      uint32_t ab[RBB][NT][2];
#pragma unroll
      for (int u = 0; u < RBB; ++u) {
        if (r0 + u < R) {
          wv[u] = ldw(wb + (size_t)(r0 + u) * (2 * H));
          sv[u] = lds8(sb + (size_t)(r0 + u) * H);
        }
      }
      pdl_wait();  // act is kernel A's output; everything above is independent of it
#pragma unroll
      for (int u = 0; u < RBB; ++u) {
        if (r0 + u < R) {
#pragma unroll
          for (int n = 0; n < NT; ++n) {
            ab[u][n][0] = av[n] ? __ldcg(reinterpret_cast<const unsigned int*>(ar[n] + 16 * (r0 + u))) : 0u;
            ab[u][n][1] = av[n] ? __ldcg(reinterpret_cast<const unsigned int*>(ar[n] + 16 * (r0 + u) + 8)) : 0u;
          }
        }
      }
#pragma unroll
      for (int u = 0; u < RBB; ++u) {
        if (r0 + u < R) {
          uint32_t sg[4], sg8[4];
          tile_scales(sv[u].x, sv[u].y, sg, sg8);
          const uint32_t wq[4] = {(uint32_t)wv[u].x, (uint32_t)wv[u].y, (uint32_t)wv[u].z, (uint32_t)wv[u].w};
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            uint32_t a[4];
            dequant(wq[j], sg[j], sg8[j], a);
#pragma unroll
            for (int n = 0; n < NT; ++n) mma16816(cacc[j][n], a, ab[u][n][0], ab[u][n][1]);
          }
        }
      }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int n = 0; n < NT; ++n)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[j][n][i] += cacc[j][n][i] * alpha * twv[n][i & 1];
  }
  pdl_wait();  // warps without an expert, before out is written
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int i = 0; i < 4; ++i) red[warp][lane][(j * NT + n) * 4 + i] = acc[j][n][i];
  __syncthreads();
#pragma unroll
  for (int q = 0; q < PT; ++q) {
    const int p = threadIdx.x + q * THREADS;
    if (p < P) {
      float s = 0.f;
#pragma unroll
      for (int w = 0; w < WARPS; ++w) s += red[w][p / VALS][p % VALS];
      part_s[p] = s;
    }
  }
  cg::cluster_group cl = cg::this_cluster();
  cl.sync();
  const int rank = static_cast<int>(cl.block_rank());
  const float* parts[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) parts[k] = k < G ? cl.map_shared_rank(part_s, k) : part_s;
  for (int p = threadIdx.x * G + rank; p < P; p += THREADS * G) {
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k) s += k < G ? parts[k][p] : 0.f;
    const int l = p / VALS, vv = p % VALS;
    const int i = vv & 3, n = (vv >> 2) % NT, j = (vv >> 2) / NT;
    const int col = 64 * c + 16 * j + (l >> 2) + 8 * (i >> 1);
    const int t = 8 * n + 2 * (l & 3) + (i & 1);
    if (t < M) out[(size_t)t * H + col] = __float2bfloat16(s);
  }
  cl.sync();  // keep this CTA's shared memory alive until the whole cluster has read it
}

template <typename K, typename... Args>
void launch_ex(K kernel, dim3 grid, int cluster_y, bool pdl, cudaStream_t stream, Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = dim3(THREADS);
  cfg.stream = stream;
  cudaLaunchAttribute attr[2];
  int na = 0;
  if (pdl) {
    attr[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[na].val.programmaticStreamSerializationAllowed = 1;
    ++na;
  }
  if (cluster_y > 0) {
    attr[na].id = cudaLaunchAttributeClusterDimension;
    attr[na].val.clusterDim.x = 1;
    attr[na].val.clusterDim.y = cluster_y;
    attr[na].val.clusterDim.z = 1;
    ++na;
  }
  cfg.attrs = attr;
  cfg.numAttrs = na;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, args...));
}

template <int NT, typename IdT, int RBA, int RBB, int RA, int RB>
void launch(const torch::Tensor& x, const torch::Tensor& w13, const torch::Tensor& s13, const torch::Tensor& g13,
            const torch::Tensor& w2, const torch::Tensor& s2, const torch::Tensor& g2, const torch::Tensor& ids,
            const torch::Tensor& tw, torch::Tensor& out, torch::Tensor& act, int groups, bool pdl, bool pfb,
            bool count) {
  const int M = x.size(0), H = x.size(1), topk = ids.size(1), I = w2.size(1) * 16, E = w13.size(0);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  const auto* idp = reinterpret_cast<const IdT*>(ids.data_ptr());
  auto* actp = reinterpret_cast<__nv_bfloat16*>(act.data_ptr());
  launch_ex(moe_up_kernel<NT, IdT, RBA, RA>, dim3(I / 64, M * topk), 0, pdl, stream, xp, M,
            static_cast<const int32_t*>(w13.data_ptr<int32_t>()), reinterpret_cast<const uint8_t*>(s13.data_ptr()),
            static_cast<const float*>(g13.data_ptr<float>()), idp, topk, actp, H, I, E);
  launch_ex(moe_down_kernel<NT, IdT, RBB, RB>, dim3(H / 64, groups), groups, pdl, stream,
            static_cast<const __nv_bfloat16*>(actp), M, static_cast<const int32_t*>(w2.data_ptr<int32_t>()),
            reinterpret_cast<const uint8_t*>(s2.data_ptr()), static_cast<const float*>(g2.data_ptr<float>()), idp,
            static_cast<const float*>(tw.data_ptr<float>()), topk,
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), H, I, E, pfb, count ? g_cnt : nullptr,
            static_cast<volatile unsigned long long*>(count ? g_cnt_hd : nullptr));
}

// (rba, rbb, regs) cases. regs: 0 uncapped (255 / 255), 1 A 112 / B 136, 2 A 128 / B 128, 3 A 96 / B 152 (per-thread
// registers; any of 1-3 lets one A and one B CTA of 256 threads share an SM's 64 K registers). NT = 2 (M > 8) has
// only (4, 10, 0).
template <int NT, typename IdT>
void dispatch_rb(int rba, int rbb, int regs, const torch::Tensor& x, const torch::Tensor& w13,
                 const torch::Tensor& s13, const torch::Tensor& g13, const torch::Tensor& w2, const torch::Tensor& s2,
                 const torch::Tensor& g2, const torch::Tensor& ids, const torch::Tensor& tw, torch::Tensor& out,
                 torch::Tensor& act, int groups, bool pdl, bool pfb, bool count) {
#define KFDEC_CASE(A, B, G, RA, RB)                                                                \
  if (rba == A && rbb == B && regs == G) {                                                         \
    launch<NT, IdT, A, B, RA, RB>(x, w13, s13, g13, w2, s2, g2, ids, tw, out, act, groups, pdl, pfb, count); \
    return;                                                                                        \
  }
  KFDEC_CASE(4, 10, 0, 255, 255)
  if constexpr (NT == 1) {
    KFDEC_CASE(4, 10, 1, 112, 136)
    KFDEC_CASE(4, 10, 2, 128, 128)
    KFDEC_CASE(4, 10, 3, 96, 152)
    KFDEC_CASE(4, 5, 0, 255, 255)
    KFDEC_CASE(4, 5, 1, 112, 136)
  }
#undef KFDEC_CASE
  TORCH_CHECK(false, "kfdec: unsupported (rba, rbb, regs)");
}

}  // namespace

// One-time setup of the calls counter on the current device (idempotent; call outside graph capture).
void counter_init() {
  if (g_cnt != nullptr) return;
  int dev = 0;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  const unsigned long long zero = 0;
  C10_CUDA_CHECK(cudaMalloc(&g_cnt, sizeof(zero)));
  C10_CUDA_CHECK(cudaMemcpy(g_cnt, &zero, sizeof(zero), cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaHostAlloc(&g_cnt_h, sizeof(zero), cudaHostAllocMapped));
  *g_cnt_h = 0;
  C10_CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&g_cnt_hd), g_cnt_h, 0));
  g_cnt_dev = dev;
}

// Counted calls so far (kernel B launches with count = true that have reached the dedupe), or -1 before
// counter_init. Reads host memory only.
int64_t calls() {
  return g_cnt_h == nullptr ? -1 : static_cast<int64_t>(*static_cast<volatile unsigned long long*>(g_cnt_h));
}

// out[M, H] = sum over routed k (ids[m, k] in [0, E)) of tw[m, k] * W2[ids[m, k]] (silu(W13g x) * W13u x), Marlin-layout
// NVFP4 experts, M <= 16.
// act: [>= M * topk, I] bf16 scratch; groups: expert groups (= cluster size, 1..8) per output chunk; pdl:
// programmatic dependent launch; rba / rbb: rows per load batch (up / down); pfb: kernel B L2-prefetches its rows
// past the first batch before waiting on A; regs: register-cap mode (see dispatch_rb); count: bump the calls
// counter (needs counter_init on this device).
void moe_decode(torch::Tensor x, torch::Tensor w13, torch::Tensor s13, torch::Tensor g13, torch::Tensor w2,
                torch::Tensor s2, torch::Tensor g2, torch::Tensor ids, torch::Tensor tw, torch::Tensor out,
                torch::Tensor act, int64_t groups, bool pdl, int64_t rba, int64_t rbb, bool pfb, int64_t regs,
                bool count) {
  const int M = x.size(0), H = x.size(1), topk = ids.size(1), I = w2.size(1) * 16, E = w13.size(0);
  TORCH_CHECK(M >= 1 && M <= 16 && M * topk <= MAX_SLOTS && E <= MAX_E, "kfdec: M or E out of range");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.is_contiguous() && out.is_contiguous(), "kfdec: x/out");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.size(0) == M && out.size(1) == H, "kfdec: out shape");
  TORCH_CHECK(H % 64 == 0 && I % 64 == 0, "kfdec: H and I must be multiples of 64");
  TORCH_CHECK(w13.scalar_type() == at::kInt && w13.size(1) == H / 16 && w13.size(2) == 4 * I, "kfdec: w13");
  TORCH_CHECK(s13.element_size() == 1 && s13.size(0) == E && s13.size(1) == H / 16 && s13.size(2) == 2 * I,
              "kfdec: s13");
  TORCH_CHECK(w2.scalar_type() == at::kInt && w2.size(0) == E && w2.size(2) == 2 * H, "kfdec: w2");
  TORCH_CHECK(s2.element_size() == 1 && s2.size(1) == I / 16 && s2.size(2) == H, "kfdec: s2");
  TORCH_CHECK(g13.scalar_type() == at::kFloat && g2.scalar_type() == at::kFloat && g13.numel() == E &&
                  g2.numel() == E, "kfdec: global scales");
  TORCH_CHECK(tw.scalar_type() == at::kFloat && tw.is_contiguous() && ids.is_contiguous() && tw.size(0) == M &&
                  tw.size(1) == topk && ids.size(0) == M, "kfdec: routing");
  TORCH_CHECK(act.scalar_type() == at::kBFloat16 && act.is_contiguous() && act.numel() >= (int64_t)M * topk * I,
              "kfdec: act");
  TORCH_CHECK(groups >= 1 && groups <= 8, "kfdec: groups must be 1..8");
  TORCH_CHECK(w13.is_contiguous() && s13.is_contiguous() && w2.is_contiguous() && s2.is_contiguous(),
              "kfdec: weights must be contiguous");
  TORCH_CHECK(!count || (g_cnt != nullptr && g_cnt_dev == x.device().index()),
              "kfdec: count needs counter_init on this device");
  const at::cuda::CUDAGuard guard(x.device());
  const bool i64 = ids.scalar_type() == at::kLong;
  TORCH_CHECK(i64 || ids.scalar_type() == at::kInt, "kfdec: ids dtype");
  const int NT = M <= 8 ? 1 : 2;
#define KFDEC_RB(NTV, T) \
  dispatch_rb<NTV, T>(rba, rbb, regs, x, w13, s13, g13, w2, s2, g2, ids, tw, out, act, groups, pdl, pfb, count)
  if (NT == 1) {
    if (i64) KFDEC_RB(1, int64_t);
    else KFDEC_RB(1, int32_t);
  } else {
    if (i64) KFDEC_RB(2, int64_t);
    else KFDEC_RB(2, int32_t);
  }
#undef KFDEC_RB
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_decode", &moe_decode, "K-FDEC v8 small-M Marlin-layout MoE (unrouted slots: ids outside [0, E))");
  m.def("counter_init", &counter_init, "allocate the calls counter on the current device (once)");
  m.def("calls", &calls, "counted calls so far (host read, -1 before counter_init)");
}
