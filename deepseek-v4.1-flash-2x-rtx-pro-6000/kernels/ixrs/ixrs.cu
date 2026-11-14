// ixrs.cu: an exact multi-CTA top-k for DeepSeek-V4.1-Flash's attention indexer on sm_120a (RTX PRO 6000 Blackwell).
// Copyright 2026 Percivio Ltd. Apache-2.0 (see LICENSE at the repository root). Written by Percivio's AI agents.
//
// It replaces the decode indexer top-k of diffbot's recipe, rtx8::rowtopk (recipe/kernels/rowtopk/rowtopk.cu, MIT), with the
// same op schema, argument checks and output bytes; only the work split differs. The recipe's kernel runs one 1024-thread
// CTA per row, so a 6-row verify step uses 6 of 188 SMs. Here each row gets a thread-block cluster of C CTAs (C = 8/4/2/1,
// the largest whose clusters all fit at once). Each CTA owns a contiguous, 4-aligned chunk of the row's valid prefix, reads
// it from global memory once (pass 0, U float4 loads in flight per thread) into a shared-memory key cache, and builds a
// local histogram. At C >= 2 each pass's cluster histogram is reduce-scattered: CTA r sums only slice r (nb/C bins counted
// from the top) over the peers into red[] and its slice total; after a cluster barrier every CTA reads the C totals, finds
// the slice holding the threshold, reads that slice's sums from its owner and scans them, so all CTAs agree on the
// threshold bin, prefix, mask and remaining count after each 11/11/10-bit pass, with the recipe's early stop. Barriers at
// C >= 2: S1 and S1b per pass; S2 is a CTA barrier on passes that don't stop and a cluster barrier on the stop pass; S3 as
// before. Index order across CTAs: CTA q starts its strictly-above and tie outputs at the counts of CTAs 0..q-1; each CTA
// derives its counts from its own histograms and publishes them once through DSMEM.
// Graph-safe: grid and cluster size depend only on the row count, lengths are read on the device, no host sync, no global
// scratch. The first call per (k, alignment, C) sets the dynamic shared-memory limit and queries cluster occupancy, so the
// first call should be eager (the engine's warmup is). One device per process.
// Build: build_ixrs.py (library and C++ namespace ixrs, so it loads beside the recipe's rtx8). IXRS_FORCE_C (unset in the
// served build) forces a cluster size, for benchmarking only; IXTK_MUTANT_LAST_TIES builds a negative control for tests.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_reduce.cuh>
#include <climits>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace cg = cooperative_groups;

// IXTK_NS: the op library and C++ namespace (default ixrs). build_ixrs.py with IXTK_MUTANT=1 adds -DIXTK_MUTANT_LAST_TIES
// -DIXTK_NS=ixrsm: the bench's negative control, identical except that it keeps the LAST `remaining` threshold ties
// instead of the first (it loads beside ixrs, ixtk and rtx8).
#ifndef IXTK_NS
#define IXTK_NS ixrs
#endif

namespace IXTK_NS {
constexpr int T = 1024;
constexpr int U = 4;                    // VEC groups in flight per thread in the pass-0 global loads
constexpr int NB = 2048;                // histogram bins (pass 2 uses the first 1024)
constexpr uint32_t NONE = 0xffffffffu;

__device__ __forceinline__ uint32_t key_of(float x) {
  uint32_t u = __float_as_uint(x);
  if ((u & 0x7fffffffu) > 0x7f800000u) return 0u;           // NaN -> lowest
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// warp-aggregated histogram add; the whole warp calls it (bin = NONE: no add); no match when no lane adds
__device__ __forceinline__ void hist_add(uint32_t* hist, uint32_t bin) {
  if (!__any_sync(0xffffffffu, bin != NONE)) return;
  const unsigned m = __match_any_sync(0xffffffffu, bin);
  if (bin != NONE && (int)(threadIdx.x & 31) == __ffs(m) - 1) atomicAdd(&hist[bin], (uint32_t)__popc(m));
}

template <int VEC>
__device__ __forceinline__ void load_keys(const float* __restrict__ x, int i, int n, uint32_t (&k)[VEC]) {
  if constexpr (VEC == 4) {
    if (i + 3 < n) {
      const float4 v = __ldg(reinterpret_cast<const float4*>(x + i));
      k[0] = key_of(v.x); k[1] = key_of(v.y); k[2] = key_of(v.z); k[3] = key_of(v.w);
      return;
    }
  }
#pragma unroll
  for (int j = 0; j < VEC; ++j) k[j] = (i + j < n) ? key_of(__ldg(x + i + j)) : 0u;
}

template <int VEC>
__device__ __forceinline__ void load_cached(const uint32_t* kc, int i, int n, uint32_t (&k)[VEC]) {
  if constexpr (VEC == 4) {
    if (i + 3 < n) {
      const uint4 v = *reinterpret_cast<const uint4*>(kc + i);
      k[0] = v.x; k[1] = v.y; k[2] = v.z; k[3] = v.w;
      return;
    }
  }
#pragma unroll
  for (int j = 0; j < VEC; ++j) k[j] = (i + j < n) ? kc[i + j] : 0u;
}

template <int K, int VEC, int C>
__global__ void __launch_bounds__(T) ixtk_kernel(const float* __restrict__ X, long ldx, int ncols,
                                                 const int* __restrict__ lens, int* __restrict__ out, long ldo, int cap) {
  const int r = blockIdx.x / C, rank = blockIdx.x % C, tid = threadIdx.x;
  int n = lens[r];
  n = n < 0 ? 0 : (n > ncols ? ncols : n);
  const float* x = X + (long)r * ldx;
  int* o = out + (long)r * ldo;
  if (n <= K) {                                   // the whole cluster leaves here (n is per row): no cluster barrier
    for (int i = rank * T + tid; i < K; i += C * T) o[i] = i < n ? i : -1;
    return;
  }
  extern __shared__ __align__(16) uint32_t smem[];
  uint32_t* hist = smem;                          // NB local bins, read by the other CTAs of the cluster
  uint32_t* kc = smem + NB;                       // key cache: the first `cap` keys of this CTA's chunk
  __shared__ uint32_t s_bin, s_above, s_inbin, s_la, s_gt, s_eq;
  __shared__ uint32_t s_slot[2];                  // this CTA's (above, tie) counts at the stop, read by higher ranks
  __shared__ uint32_t red[C > 1 ? NB / C : 1];    // C >= 2: this CTA's slice sums of the pass, read by the other CTAs
  __shared__ uint32_t s_tot, s_tots[C];           // this CTA's slice total (read by the others); all C totals
  using Scan = cub::BlockScan<uint32_t, T, cub::BLOCK_SCAN_WARP_SCANS>;
  using Reduce = cub::BlockReduce<uint32_t, T>;
  __shared__ typename Scan::TempStorage ts;
  __shared__ typename Reduce::TempStorage rs;
  cg::cluster_group cl = cg::this_cluster();

  const int cs = ((n + C - 1) / C + 3) & ~3;      // chunk size, a multiple of 4 (float4 alignment)
  const int lo = min(n, rank * cs), len = min(n, lo + cs) - lo;
  const int nc = min(len, cap);                   // cached part [0, nc) of the chunk; [nc, len) is re-read from global
  const float* xc = x + lo;

  uint32_t prefix = 0, pmask = 0, my_gt = 0, my_eq = 0, e_tot = 0;
  int remaining = K;
#pragma unroll 1
  for (int p = 0; p < 3; ++p) {
    const int sh = p == 0 ? 21 : (p == 1 ? 10 : 0);
    const int nb = p == 2 ? 1024 : 2048;
    for (int i = tid; i < nb; i += T) hist[i] = 0;
    __syncthreads();
    if (p == 0) {
      for (int base = 0; base < len; base += T * VEC * U) {
        uint32_t k[U][VEC];
#pragma unroll
        for (int u = 0; u < U; ++u) load_keys<VEC>(xc, base + (u * T + tid) * VEC, len, k[u]);
#pragma unroll
        for (int u = 0; u < U; ++u) {
          const int i = base + (u * T + tid) * VEC;
          bool stored = false;
          if constexpr (VEC == 4) {
            if (i + 3 < nc) { *reinterpret_cast<uint4*>(kc + i) = make_uint4(k[u][0], k[u][1], k[u][2], k[u][3]); stored = true; }
          }
          if (!stored) {
#pragma unroll
            for (int j = 0; j < VEC; ++j) if (i + j < nc) kc[i + j] = k[u][j];
          }
#pragma unroll
          for (int j = 0; j < VEC; ++j) hist_add(hist, i + j < len ? (k[u][j] >> 21) : NONE);
        }
      }
    } else {
      for (int base = 0; base < nc; base += T * VEC) {
        const int i = base + tid * VEC;
        uint32_t k[VEC];
        load_cached<VEC>(kc, i, nc, k);
#pragma unroll
        for (int j = 0; j < VEC; ++j) {
          const bool ok = (i + j < nc) && ((k[j] & pmask) == prefix);
          hist_add(hist, ok ? ((k[j] >> sh) & (uint32_t)(nb - 1)) : NONE);
        }
      }
      for (int base = nc; base < len; base += T * VEC) {
        const int i = base + tid * VEC;
        uint32_t k[VEC];
        load_keys<VEC>(xc, i, len, k);
#pragma unroll
        for (int j = 0; j < VEC; ++j) {
          const bool ok = (i + j < len) && ((k[j] & pmask) == prefix);
          hist_add(hist, ok ? ((k[j] >> sh) & (uint32_t)(nb - 1)) : NONE);
        }
      }
    }
    cl.sync();                                    // S1: every local histogram of this pass is complete
    const int per = nb / T;                       // 2 or 1
    uint32_t own[2] = {0u, 0u};                   // this CTA's counts of the bins this thread owns (for la)
    if constexpr (C == 1) {
      // cluster-wide counts of the bins this thread owns, counted from the top: b = nb-1 - (t*per + j); then the recipe's scan
      uint32_t loc[2] = {0u, 0u}, lsum = 0;
      for (int j = 0; j < per; ++j) {
        const int bin = nb - 1 - (tid * per + j);
        uint32_t s = 0;
#pragma unroll
        for (int q = 0; q < C; ++q) s += cl.map_shared_rank(hist, q)[bin];
        loc[j] = s; lsum += s; own[j] = hist[bin];
      }
      uint32_t c;
      Scan(ts).ExclusiveSum(lsum, c);
      for (int j = 0; j < per; ++j) {
        if (c < (uint32_t)remaining && c + loc[j] >= (uint32_t)remaining) {
          s_bin = nb - 1 - (tid * per + j); s_above = c; s_inbin = loc[j];
        }
        c += loc[j];
      }
    } else {
      // reduce-scatter: slice r is positions [r*S, (r+1)*S) counted from the top, bin = nb-1 - position
      const int S = nb / C;                       // 2048/C or 1024/C bins, at most T
      uint32_t s = 0;
      if (tid < S) {
        const int bin = nb - 1 - (rank * S + tid);
#pragma unroll
        for (int q = 0; q < C; ++q) s += cl.map_shared_rank(hist, q)[bin];
        red[tid] = s;
      }
      const uint32_t tot = Reduce(rs).Sum(s);
      if (tid == 0) s_tot = tot;
      for (int j = 0; j < per; ++j) own[j] = hist[nb - 1 - (tid * per + j)];
      cl.sync();                                  // S1b: every slice's sums and total are written
      if (tid < C) s_tots[tid] = *cl.map_shared_rank(&s_tot, tid);
      __syncthreads();
      int o = 0;                                  // the owner: the slice where the count from the top reaches remaining
      uint32_t ab = 0;                            // the keys in the slices above slice o
      while (o < C - 1 && ab + s_tots[o] < (uint32_t)remaining) { ab += s_tots[o]; ++o; }
      const uint32_t v = tid < S ? cl.map_shared_rank(red, o)[tid] : 0u;
      uint32_t c;
      Scan(ts).ExclusiveSum(v, c);
      c += ab;
      if (tid < S && c < (uint32_t)remaining && c + v >= (uint32_t)remaining) {
        s_bin = nb - 1 - (o * S + tid); s_above = c; s_inbin = v;
      }
    }
    __syncthreads();
    const uint32_t b = s_bin, above = s_above, inbin = s_inbin;
    e_tot = inbin;                                // at the stop: all ties (masked key == prefix) in the row
    uint32_t la = 0;                              // this CTA's keys in bins above b
    for (int j = 0; j < per; ++j) la += (uint32_t)(nb - 1 - (tid * per + j)) > b ? own[j] : 0u;
    la = Reduce(rs).Sum(la);
    if (tid == 0) s_la = la;
    __syncthreads();
    my_gt += s_la;
    my_eq = hist[b];
    prefix |= (b << sh);
    pmask |= ((uint32_t)(nb - 1) << sh);
    remaining -= (int)above;                      // still to take among keys matching the new prefix
    const bool stop = (int)inbin == remaining || p == 2;
    if (stop && tid == 0) { s_slot[0] = my_gt; s_slot[1] = my_eq; }
    // S2: nobody zeroes or leaves while others read its histogram. At C >= 2 the peers' reads of it ended before S1b, so a
    // pass that doesn't stop needs only this CTA's threads (my_eq read hist[b] above); the stop pass (uniform in the
    // cluster: every CTA has the same inbin and remaining) keeps the cluster barrier, so s_slot is visible to higher ranks
    if (C == 1 || stop) cl.sync();
    else __syncthreads();
    if (stop) break;
  }
  // output offsets of this CTA: the above and tie counts of CTAs 0..rank-1 (index order across chunks)
  uint32_t g0 = 0, e0 = 0;
  for (int q = 0; q < rank; ++q) {
    const uint32_t* s = cl.map_shared_rank(s_slot, q);
    g0 += s[0]; e0 += s[1];
  }
  // compaction as in the recipe's kernel: gt = (k & pmask) > prefix in index order from g0; eq = (k & pmask) == prefix, global tie index
  // e0 + local, written only below `remaining`
  const uint32_t n_gt = (uint32_t)(K - remaining);
#ifdef IXTK_MUTANT_LAST_TIES
  const uint32_t e_skip = e_tot - (uint32_t)remaining;   // negative control: ties e_skip..e_tot-1 instead of 0..remaining-1
#else
  const uint32_t e_skip = 0;
  (void)e_tot;
#endif
  if (tid == 0) { s_gt = 0; s_eq = 0; }
  __syncthreads();
  auto done = [&]() { return s_gt >= my_gt && (s_eq >= my_eq || e0 + s_eq >= e_skip + (uint32_t)remaining); };
  auto step = [&](int i, const uint32_t (&k)[VEC], int end) {
    uint32_t mine = 0;                            // (gt << 16) | eq
    uint8_t fl[VEC];
#pragma unroll
    for (int j = 0; j < VEC; ++j) {
      const uint32_t km = k[j] & pmask;
      fl[j] = (i + j < end) ? (km > prefix ? 2 : (km == prefix ? 1 : 0)) : 0;
      mine += fl[j] == 2 ? (1u << 16) : (fl[j] == 1 ? 1u : 0u);
    }
    uint32_t off, tot;
    Scan(ts).ExclusiveSum(mine, off, tot);
    uint32_t g = g0 + s_gt + (off >> 16), e = e0 + s_eq + (off & 0xffffu);
#pragma unroll
    for (int j = 0; j < VEC; ++j) {
      if (fl[j] == 2) { o[g] = lo + i + j; ++g; }
      else if (fl[j] == 1) { if (e >= e_skip && e - e_skip < (uint32_t)remaining) o[n_gt + e - e_skip] = lo + i + j; ++e; }
    }
    __syncthreads();
    if (tid == 0) { s_gt += tot >> 16; s_eq += tot & 0xffffu; }
    __syncthreads();
  };
#pragma unroll 1
  for (int base = 0; base < nc; base += T * VEC) {
    if (done()) break;                            // uniform (read after the barrier in step)
    const int i = base + tid * VEC;
    uint32_t k[VEC];
    load_cached<VEC>(kc, i, nc, k);
    step(i, k, nc);
  }
#pragma unroll 1
  for (int base = nc; base < len; base += T * VEC) {
    if (done()) break;
    const int i = base + tid * VEC;
    uint32_t k[VEC];
    load_keys<VEC>(xc, i, len, k);
    step(i, k, len);
  }
  cl.sync();                                      // S3: nobody leaves while a higher rank may still read its s_slot
}

struct Cfg { int cap = -1; size_t dyn = 0; int maxc = 0; };

template <int K, int VEC, int C>
Cfg& cfg_of() {
  static Cfg c;
  if (c.cap < 0) {
    auto kern = ixtk_kernel<K, VEC, C>;
    int dev = 0, optin = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    cudaFuncAttributes fa;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, kern));
    const long avail = (long)optin - (long)fa.sharedSizeBytes - (long)NB * 4;
    TORCH_CHECK(avail >= 4L * T * 4, "ixtk::rowtopk: too little shared memory (", optin, " B opt-in)");
    c.dyn = (size_t)NB * 4 + (size_t)((avail / 4) & ~3L) * 4;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)c.dyn));
    cudaLaunchConfig_t lc = {};
    lc.gridDim = dim3(C); lc.blockDim = dim3(T); lc.dynamicSmemBytes = c.dyn;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = C; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    lc.attrs = at; lc.numAttrs = 1;
    if (cudaOccupancyMaxActiveClusters(&c.maxc, kern, &lc) != cudaSuccess) {   // cluster size refused: never chosen
      (void)cudaGetLastError();
      TORCH_CHECK(C > 1, "ixtk::rowtopk: cluster occupancy query failed for C = 1");
      c.maxc = 0;
    }
    c.cap = (int)((avail / 4) & ~3L);
  }
  return c;
}

template <int K, int VEC, int C>
void launch_c(const torch::Tensor& logits, const torch::Tensor& lens, torch::Tensor& out, cudaStream_t st) {
  const Cfg& c = cfg_of<K, VEC, C>();
  const int rows = logits.size(0), ncols = logits.size(1);
  const long ldx = logits.stride(0), ldo = out.stride(0);
  cudaLaunchConfig_t lc = {};
  lc.gridDim = dim3(rows * C); lc.blockDim = dim3(T); lc.dynamicSmemBytes = c.dyn; lc.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = C; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  lc.attrs = at; lc.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, ixtk_kernel<K, VEC, C>, logits.data_ptr<float>(), ldx, ncols,
                                    lens.data_ptr<int>(), out.data_ptr<int>(), ldo, c.cap));
}

// the largest cluster size whose clusters for all rows are resident at once (6 rows at c1: C = 8)
template <int K, int VEC>
void launch_v(const torch::Tensor& logits, const torch::Tensor& lens, torch::Tensor& out, cudaStream_t st) {
  const int rows = logits.size(0);
#ifdef IXRS_FORCE_C
  // a bench arm only (built with -DIXRS_FORCE_C=4 -DIXTK_NS=ixrs4): this cluster size whenever
  // the call's clusters at it all fit, else the choice below. Without the macro the dispatch is ixtk's.
  if (rows <= cfg_of<K, VEC, IXRS_FORCE_C>().maxc) { launch_c<K, VEC, IXRS_FORCE_C>(logits, lens, out, st); return; }
#endif
  if (rows <= cfg_of<K, VEC, 8>().maxc) launch_c<K, VEC, 8>(logits, lens, out, st);
  else if (rows <= cfg_of<K, VEC, 4>().maxc) launch_c<K, VEC, 4>(logits, lens, out, st);
  else if (rows <= cfg_of<K, VEC, 2>().maxc) launch_c<K, VEC, 2>(logits, lens, out, st);
  else launch_c<K, VEC, 1>(logits, lens, out, st);
}

template <int K>
void launch(const torch::Tensor& logits, const torch::Tensor& lens, torch::Tensor& out, cudaStream_t st) {
  const long ldx = logits.stride(0);
  const bool v4 = (ldx % 4 == 0) && (reinterpret_cast<uintptr_t>(logits.data_ptr()) % 16 == 0);
  if (v4) launch_v<K, 4>(logits, lens, out, st);
  else launch_v<K, 1>(logits, lens, out, st);
}
void rowtopk_op(torch::Tensor logits, torch::Tensor lens, torch::Tensor out, int64_t k) {
  TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == torch::kFloat32 && logits.dim() == 2 && logits.stride(1) == 1,
              "ixtk::rowtopk: logits must be CUDA fp32 [rows, width] with unit column stride");
  TORCH_CHECK(lens.is_cuda() && lens.scalar_type() == torch::kInt32 && lens.is_contiguous() && lens.numel() == logits.size(0),
              "ixtk::rowtopk: lens must be CUDA int32 contiguous [rows]");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kInt32 && out.dim() == 2 && out.stride(1) == 1 &&
              out.size(0) >= logits.size(0) && out.size(1) >= k, "ixtk::rowtopk: out must be CUDA int32 [rows, >=k]");
  if (logits.size(0) == 0) return;
  const at::cuda::OptionalCUDAGuard guard(logits.device());
  auto st = at::cuda::getCurrentCUDAStream();
  if (k == 512) launch<512>(logits, lens, out, st);
  else if (k == 1024) launch<1024>(logits, lens, out, st);
  else if (k == 2048) launch<2048>(logits, lens, out, st);
  else TORCH_CHECK(false, "ixtk::rowtopk: k must be 512/1024/2048");
}
}  // namespace IXTK_NS

// one level of indirection so IXTK_NS expands before TORCH_LIBRARY pastes and stringifies it
#define IXTK_LIBRARY(ns, m) TORCH_LIBRARY(ns, m)
#define IXTK_LIBRARY_IMPL(ns, key, m) TORCH_LIBRARY_IMPL(ns, key, m)
IXTK_LIBRARY(IXTK_NS, m) { m.def("rowtopk(Tensor logits, Tensor lens, Tensor(a!) out, int k) -> ()"); }
IXTK_LIBRARY_IMPL(IXTK_NS, CUDA, m) { m.impl("rowtopk", &IXTK_NS::rowtopk_op); }
IXTK_LIBRARY_IMPL(IXTK_NS, Meta, m) { m.impl("rowtopk", [](torch::Tensor, torch::Tensor, torch::Tensor, int64_t) {}); }
