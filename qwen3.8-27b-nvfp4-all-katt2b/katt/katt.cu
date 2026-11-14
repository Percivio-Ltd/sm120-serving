#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math_constants.h>
#include <cstdint>
#include <cstdio>

// D256, GQA6, BF16 Q/P, exact e4m3 -> BF16 K/V, FP32 MMA/online softmax.
// The source is standalone: no torch ABI and no FlashInfer kernel dependency.
using bf16 = __nv_bfloat16;

struct Args {
  const bf16* q;
  const uint8_t *k, *v;
  bf16* o;
  const int *qo, *indptr, *pages, *last;
  int page_size;
  int64_t qs0, qs1, ks0, ks1, ks2, vs0, vs1, vs2, os0, os1;
  float sm_scale, v_scale;
};

__device__ __forceinline__ uint32_t sm_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp16(void* dst, const void* src, bool valid) {
  const uint32_t a = sm_addr(dst);
  const int bytes = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::
               "r"(a), "l"(src), "r"(bytes));
}
__device__ __forceinline__ void commit() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}
__device__ __forceinline__ void wait_copy() {
  asm volatile("cp.async.wait_group 0;" ::: "memory");
}
__device__ __forceinline__ uint32_t ld2(const bf16* p) {
  return *reinterpret_cast<const uint32_t*>(p);
}
__device__ __forceinline__ void st2(bf16* p, float x, float y) {
  *reinterpret_cast<__nv_bfloat162*>(p) = __floats2bfloat162_rn(x, y);
}
__device__ __forceinline__ bf16 fp8_bf16(uint8_t x) {
  // All finite e4m3 values are exactly representable in BF16. Use the CUDA
  // conversion so zero/subnormal/NaN handling stays the NVIDIA contract.
  __nv_fp8_e4m3 f;
  f.__x = x;
  return static_cast<bf16>(f);
}
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4],
                                   const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

template<int M>
struct Shared {
  // Padding gives conflict-free 32-bit fragment loads for m16n8k16 lanes.
  bf16 q[M][264];
  bf16 k[32][264];
  bf16 vt[256][40];
  bf16 p[M][40];
  alignas(16) uint8_t raw[2][32][256];
};

template<int M>
__device__ __forceinline__ void prefetch(Shared<M>& s, const Args& a,
                                        int kv0, int kvlen, int pi0, int kh) {
  constexpr int NT = M / 16 * 32;
  for (int i = threadIdx.x; i < 2 * 32 * 16; i += NT) {
    const int side = i / (32 * 16);
    const int row = (i / 16) % 32;
    const int col = (i % 16) * 16;
    const int pos = kv0 + row;
    const bool valid = pos < kvlen;
    const int page = valid ? a.pages[pi0 + pos / a.page_size] : 0;
    const int token = valid ? pos % a.page_size : 0;
    const uint8_t* src = side ? a.v + page*a.vs0 + kh*a.vs1 + token*a.vs2 + col
                              : a.k + page*a.ks0 + kh*a.ks1 + token*a.ks2 + col;
    cp16(&s.raw[side][row][col], src, valid);
  }
  commit();
}

template<int M>
__global__ __launch_bounds__(M / 16 * 32, 1)
void katt_prefill_kernel(Args a) {
  extern __shared__ __align__(16) char storage[];
  auto& s = *reinterpret_cast<Shared<M>*>(storage);
  const int seq = blockIdx.z, kh = blockIdx.y;
  const int q0 = a.qo[seq], qlen = a.qo[seq+1] - q0;
  const int flat0 = blockIdx.x * M;
  if (flat0 >= qlen * 6) return;
  const int pi0 = a.indptr[seq], np = a.indptr[seq+1] - pi0;
  const int kvlen = (np-1) * a.page_size + a.last[seq];
  const int prefix = kvlen - qlen;
  const int kvend = min(kvlen, prefix + (flat0 + M-1) / 6 + 1);
  constexpr int NT = M / 16 * 32;
  for (int i = threadIdx.x; i < M * 32; i += NT) {
    const int row = i / 32, col = (i % 32) * 8;
    const int flat = flat0 + row;
    const int tok = flat / 6;
    const int head = kh * 6 + flat % 6;
    const bool valid = tok < qlen;
    const bf16* src = a.q + (q0 + (valid ? tok : 0))*a.qs0 + head*a.qs1 + col;
    cp16(&s.q[row][col], src, valid);
  }
  prefetch<M>(s, a, 0, kvlen, pi0, kh);
  wait_copy();
  __syncthreads();
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int row0 = warp * 16 + lane / 4, row1 = row0 + 8;
  const int lc = (lane % 4) * 2, bc = lane / 4;
  const int causal0 = prefix + (flat0 + row0) / 6;
  const int causal1 = prefix + (flat0 + row1) / 6;
  float out[32][4] = {};
  float oldm0 = -CUDART_INF_F, oldm1 = -CUDART_INF_F;
  float den0 = 0, den1 = 0;
  const float scale = a.sm_scale * 1.4426950408889634f;
  for (int kv0 = 0; kv0 < kvend; kv0 += 32) {
    wait_copy();
    __syncthreads();
    for (int i = threadIdx.x; i < 32 * 256; i += NT) {
      const int row = i / 256, col = i % 256;
      s.k[row][col] = fp8_bf16(s.raw[0][row][col]);
      s.vt[col][row] = fp8_bf16(s.raw[1][row][col]);
    }
    __syncthreads();
    if (kv0 + 32 < kvend) prefetch<M>(s, a, kv0 + 32, kvlen, pi0, kh);
    float score[4][4] = {};
    #pragma unroll
    for (int d = 0; d < 256; d += 16) {
      const uint32_t qa[4] = {ld2(&s.q[row0][d+lc]), ld2(&s.q[row1][d+lc]),
                              ld2(&s.q[row0][d+lc+8]), ld2(&s.q[row1][d+lc+8])};
      #pragma unroll
      for (int n = 0; n < 4; ++n) {
        const uint32_t kb[2] = {ld2(&s.k[n*8+bc][d+lc]), ld2(&s.k[n*8+bc][d+lc+8])};
        mma(score[n], qa, kb);
      }
    }
    const bool diagonal = kv0 + 31 > prefix + flat0 / 6;
    float max0 = -CUDART_INF_F, max1 = -CUDART_INF_F;
    #pragma unroll
    for (int n = 0; n < 4; ++n) {
      const int pos = kv0 + n*8 + lc;
      #pragma unroll
      for (int j = 0; j < 2; ++j) {
        score[n][j] = pos+j >= kvlen || (diagonal && pos+j > causal0)
                        ? -CUDART_INF_F : score[n][j] * scale;
        score[n][j+2] = pos+j >= kvlen || (diagonal && pos+j > causal1)
                        ? -CUDART_INF_F : score[n][j+2] * scale;
        max0 = fmaxf(max0, score[n][j]);
        max1 = fmaxf(max1, score[n][j+2]);
      }
    }
    max0 = fmaxf(max0, __shfl_xor_sync(0xffffffff, max0, 1));
    max0 = fmaxf(max0, __shfl_xor_sync(0xffffffff, max0, 2));
    max1 = fmaxf(max1, __shfl_xor_sync(0xffffffff, max1, 1));
    max1 = fmaxf(max1, __shfl_xor_sync(0xffffffff, max1, 2));
    max0 = fmaxf(oldm0, max0); max1 = fmaxf(oldm1, max1);
    const float alpha0 = exp2f(oldm0 - max0), alpha1 = exp2f(oldm1 - max1);
    float sum0 = 0, sum1 = 0;
    #pragma unroll
    for (int n = 0; n < 4; ++n) {
      #pragma unroll
      for (int j = 0; j < 2; ++j) {
        score[n][j] = exp2f(score[n][j] - max0);
        score[n][j+2] = exp2f(score[n][j+2] - max1);
        sum0 += score[n][j]; sum1 += score[n][j+2];
      }
      st2(&s.p[row0][n*8+lc], score[n][0], score[n][1]);
      st2(&s.p[row1][n*8+lc], score[n][2], score[n][3]);
    }
    sum0 += __shfl_xor_sync(0xffffffff, sum0, 1);
    sum0 += __shfl_xor_sync(0xffffffff, sum0, 2);
    sum1 += __shfl_xor_sync(0xffffffff, sum1, 1);
    sum1 += __shfl_xor_sync(0xffffffff, sum1, 2);
    den0 = den0 * alpha0 + sum0; den1 = den1 * alpha1 + sum1;
    oldm0 = max0; oldm1 = max1;
    #pragma unroll
    for (int n = 0; n < 32; ++n) {
      out[n][0] *= alpha0; out[n][1] *= alpha0;
      out[n][2] *= alpha1; out[n][3] *= alpha1;
    }
    __syncwarp();
    #pragma unroll
    for (int kk = 0; kk < 32; kk += 16) {
      const uint32_t pa[4] = {ld2(&s.p[row0][kk+lc]), ld2(&s.p[row1][kk+lc]),
                              ld2(&s.p[row0][kk+lc+8]), ld2(&s.p[row1][kk+lc+8])};
      #pragma unroll
      for (int n = 0; n < 32; ++n) {
        const uint32_t vb[2] = {ld2(&s.vt[n*8+bc][kk+lc]), ld2(&s.vt[n*8+bc][kk+lc+8])};
        mma(out[n], pa, vb);
      }
    }
    // Every warp has finished consuming the current BF16 K/V before conversion.
    __syncthreads();
  }
  #pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int flat = flat0 + (r ? row1 : row0), tok = flat / 6;
    const int head = kh*6 + flat%6;
    if (tok < qlen) {
      const float inv = 1.f / (r ? den1 : den0);
      #pragma unroll
      for (int n = 0; n < 32; ++n) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
          bf16 val = __float2bfloat16_rn(out[n][r*2+j] * inv);
          // Match FlashInfer's BF16 output followed by in-place v_scale.
          if (a.v_scale != 1.f) val = __float2bfloat16_rn(__bfloat162float(val) * a.v_scale);
          a.o[(q0+tok)*a.os0 + head*a.os1 + n*8+lc+j] = val;
        }
      }
    }
  }
}

__global__ void dequant_kernel(const uint8_t* k, const uint8_t* v, bf16* ko, bf16* vo,
                               int np, int ps, int64_t ks0, int64_t ks1, int64_t ks2,
                               int64_t vs0, int64_t vs1, int64_t vs2) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (int64_t)np * 4 * ps * 256) return;
  const int col = i % 256, tok = i / 256 % ps, head = i / (256*ps) % 4;
  const int page = i / (256*ps*4);
  ko[i] = fp8_bf16(k[page*ks0 + head*ks1 + tok*ks2 + col]);
  vo[i] = fp8_bf16(v[page*vs0 + head*vs1 + tok*vs2 + col]);
}

extern "C" int katt_init() {
  auto e = cudaFuncSetAttribute(katt_prefill_kernel<64>, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Shared<64>));
  if (e != cudaSuccess) return int(e);
  return int(cudaFuncSetAttribute(katt_prefill_kernel<32>, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Shared<32>)));
}
extern "C" int katt_run(const void* q, const void* k, const void* v, void* o,
                         const int* qo, const int* indptr, const int* pages, const int* last,
                         int batch, int maxq, int ps,
                         int64_t qs0, int64_t qs1, int64_t ks0, int64_t ks1, int64_t ks2,
                         int64_t vs0, int64_t vs1, int64_t vs2, int64_t os0, int64_t os1,
                         float sm_scale, float v_scale, void* stream, int tile_m) {
  Args a{(const bf16*)q, (const uint8_t*)k, (const uint8_t*)v, (bf16*)o,
         qo, indptr, pages, last, ps, qs0, qs1, ks0, ks1, ks2, vs0, vs1, vs2, os0, os1,
         sm_scale, v_scale};
  if (tile_m == 32) {
    katt_prefill_kernel<32><<<dim3((maxq*6+31)/32,4,batch),64,sizeof(Shared<32>),(cudaStream_t)stream>>>(a);
  } else {
    katt_prefill_kernel<64><<<dim3((maxq*6+63)/64,4,batch),128,sizeof(Shared<64>),(cudaStream_t)stream>>>(a);
  }
  return int(cudaGetLastError());
}
extern "C" int katt_dequant(const void* k, const void* v, void* ko, void* vo,
                            int np, int ps, int64_t ks0, int64_t ks1, int64_t ks2,
                            int64_t vs0, int64_t vs1, int64_t vs2, void* stream) {
  dequant_kernel<<<((int64_t)np*4*ps*256+255)/256,256,0,(cudaStream_t)stream>>>(
      (const uint8_t*)k,(const uint8_t*)v,(bf16*)ko,(bf16*)vo,np,ps,ks0,ks1,ks2,vs0,vs1,vs2);
  return int(cudaGetLastError());
}

__global__ void gather_dequant_kernel(const uint8_t* k, const uint8_t* v,
                                      bf16* ko, bf16* vo, const int* pages,
                                      int np, int ps, int64_t ks0, int64_t ks1, int64_t ks2,
                                      int64_t vs0, int64_t vs1, int64_t vs2) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (int64_t)np * 4 * ps * 256) return;
  const int col = i % 256, tok = i / 256 % ps, head = i / (256*ps) % 4;
  const int page = pages[i / (256*ps*4)];
  ko[i] = fp8_bf16(k[page*ks0 + head*ks1 + tok*ks2 + col]);
  vo[i] = fp8_bf16(v[page*vs0 + head*vs1 + tok*vs2 + col]);
}

extern "C" int katt_gather(const void* k, const void* v, void* ko, void* vo,
                           const int* pages, int np, int ps,
                           int64_t ks0, int64_t ks1, int64_t ks2,
                           int64_t vs0, int64_t vs1, int64_t vs2, void* stream) {
  gather_dequant_kernel<<<((int64_t)np*4*ps*256+255)/256,256,0,(cudaStream_t)stream>>>(
      (const uint8_t*)k,(const uint8_t*)v,(bf16*)ko,(bf16*)vo,pages,np,ps,
      ks0,ks1,ks2,vs0,vs1,vs2);
  return int(cudaGetLastError());
}
