# Marlin量化矩乘集成设计

## 技术背景

**论文**：Marlin: Mixed-Precision Matrix Multiplication
**链接**：https://arxiv.org/abs/2408.11743

**核心优势**：
- 专门针对4-bit量化权重优化的CUDA kernel
- 重新设计内存访问模式，提高memory coalescing
- 支持group-wise量化的高效计算
- **性能提升**：2.1-3.2x矩阵乘法加速

## 当前llama.cpp矩阵乘法架构分析

### 现有量化矩阵乘法调度
```cpp
// /ggml/src/ggml-cuda/mmq.cu
static void ggml_cuda_mul_mat_q_switch_type(
    ggml_backend_cuda_context & ctx, 
    const mmq_args & args, 
    cudaStream_t stream) {
    
    switch (args.type_x) {
        case GGML_TYPE_Q4_0:
            mul_mat_q_case<GGML_TYPE_Q4_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_q_case<GGML_TYPE_Q4_1>(ctx, args, stream);
            break;
        // ... 其他量化格式
    }
}
```

### 集成策略设计

#### 1. 新增Marlin量化格式支持
```cpp
// 扩展 /ggml/include/ggml.h
enum ggml_type {
    // ... 现有类型
    GGML_TYPE_Q4_0_MARLIN = 41,  // Marlin优化的Q4_0格式
    GGML_TYPE_Q4_1_MARLIN = 42,  // Marlin优化的Q4_1格式
    GGML_TYPE_Q4_K_MARLIN = 43,  // Marlin优化的Q4_K格式
    // ...
};

// Marlin量化块结构
typedef struct {
    uint32_t qs[4];        // 4-bit量化权重（16个权重打包）
    half scale;            // 缩放因子
    half zero_point;       // 零点（如果需要）
    uint16_t group_idx;    // 组索引（用于group-wise量化）
} block_q4_marlin;

#define QK_MARLIN 16  // Marlin块大小
```

#### 2. Marlin Kernel实现
```cuda
// 新增文件 /ggml/src/ggml-cuda/marlin-mmq.cu
#include "common.cuh"
#include "mmq.cuh"

// Marlin配置常量
#define MARLIN_TILE_K 64
#define MARLIN_TILE_M 16  
#define MARLIN_TILE_N 64
#define MARLIN_GROUP_SIZE 128

// Marlin量化权重重排函数
__device__ void marlin_reorder_weights(
    const uint32_t* weight_packed,
    uint32_t* weight_reordered,
    int k_tiles, int n_tiles) {
    
    // Marlin特有的权重重排，优化内存访问模式
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    int total_tiles = k_tiles * n_tiles;
    
    for (int tile_idx = thread_id; tile_idx < total_tiles; tile_idx += gridDim.x * blockDim.x) {
        int k_tile = tile_idx / n_tiles;
        int n_tile = tile_idx % n_tiles;
        
        // 计算重排后的索引
        int src_offset = (k_tile * n_tiles + n_tile) * (MARLIN_TILE_K * MARLIN_TILE_N / 8);
        int dst_offset = marlin_compute_reorder_index(k_tile, n_tile, k_tiles, n_tiles);
        
        // 执行重排（处理多个uint32_t）
        for (int i = 0; i < MARLIN_TILE_K * MARLIN_TILE_N / 32; ++i) {
            weight_reordered[dst_offset + i] = weight_packed[src_offset + i];
        }
    }
}

// Marlin核心GEMM kernel
template<int MARLIN_M, int MARLIN_N, int MARLIN_K>
__global__ void marlin_gemm_kernel(
    const half* __restrict__ A,     // 激活 [M, K]
    const uint32_t* __restrict__ B, // 量化权重 [K, N] (4-bit packed)
    const half* __restrict__ scales, // 缩放因子 [K/group_size, N]
    half* __restrict__ C,           // 输出 [M, N]
    int M, int N, int K,
    int group_size) {
    
    // 线程块和线程索引
    int block_m = blockIdx.x * MARLIN_M;
    int block_n = blockIdx.y * MARLIN_N;
    int thread_m = threadIdx.x;
    int thread_n = threadIdx.y;
    
    // 共享内存声明
    __shared__ half A_shared[MARLIN_M][MARLIN_K];
    __shared__ uint32_t B_shared[MARLIN_K][MARLIN_N / 8]; // 4-bit packed
    __shared__ half scales_shared[MARLIN_K / MARLIN_GROUP_SIZE][MARLIN_N];
    
    // 寄存器累积器
    half accum[MARLIN_M / blockDim.x][MARLIN_N / blockDim.y];
    #pragma unroll
    for (int i = 0; i < MARLIN_M / blockDim.x; ++i) {
        #pragma unroll
        for (int j = 0; j < MARLIN_N / blockDim.y; ++j) {
            accum[i][j] = __float2half(0.0f);
        }
    }
    
    // 主循环：处理K维度的tile
    for (int k_block = 0; k_block < K; k_block += MARLIN_K) {
        __syncthreads();
        
        // 加载A到共享内存
        #pragma unroll
        for (int i = 0; i < MARLIN_M / blockDim.x; ++i) {
            int global_m = block_m + thread_m + i * blockDim.x;
            if (global_m < M) {
                #pragma unroll
                for (int k = thread_n; k < MARLIN_K && k_block + k < K; k += blockDim.y) {
                    A_shared[thread_m + i * blockDim.x][k] = 
                        A[global_m * K + k_block + k];
                }
            }
        }
        
        // 加载量化权重B到共享内存
        #pragma unroll
        for (int k = thread_m; k < MARLIN_K && k_block + k < K; k += blockDim.x) {
            #pragma unroll
            for (int n = thread_n * 8; n < MARLIN_N && block_n + n < N; n += blockDim.y * 8) {
                int global_n = block_n + n;
                if (global_n < N) {
                    // 加载8个4-bit权重（打包在一个uint32_t中）
                    B_shared[k][n / 8] = 
                        B[(k_block + k) * (N / 8) + global_n / 8];
                }
            }
        }
        
        // 加载缩放因子
        int scale_k = (k_block + MARLIN_K - 1) / group_size;
        if (thread_m == 0 && k_block % group_size == 0) {
            #pragma unroll
            for (int n = thread_n; n < MARLIN_N && block_n + n < N; n += blockDim.y) {
                scales_shared[scale_k % (MARLIN_K / group_size)][n] = 
                    scales[scale_k * N + block_n + n];
            }
        }
        
        __syncthreads();
        
        // 计算这个K块的部分乘积
        #pragma unroll
        for (int k = 0; k < MARLIN_K && k_block + k < K; ++k) {
            // 获取当前组的缩放因子
            int scale_idx = k / group_size;
            
            #pragma unroll
            for (int i = 0; i < MARLIN_M / blockDim.x; ++i) {
                half a_val = A_shared[thread_m + i * blockDim.x][k];
                
                #pragma unroll
                for (int j = 0; j < MARLIN_N / blockDim.y; ++j) {
                    int n_idx = thread_n + j * blockDim.y;
                    if (n_idx < MARLIN_N && block_n + n_idx < N) {
                        // 解包4-bit权重
                        uint32_t packed_weights = B_shared[k][n_idx / 8];
                        int shift = (n_idx % 8) * 4;
                        int weight_4bit = (packed_weights >> shift) & 0xF;
                        
                        // 转换为half并应用缩放
                        half weight_fp16 = __int2half_rz(weight_4bit - 8); // 假设对称量化
                        weight_fp16 = __hmul(weight_fp16, scales_shared[scale_idx][n_idx]);
                        
                        // 累积
                        accum[i][j] = __hfma(a_val, weight_fp16, accum[i][j]);
                    }
                }
            }
        }
    }
    
    // 写回结果
    #pragma unroll
    for (int i = 0; i < MARLIN_M / blockDim.x; ++i) {
        int global_m = block_m + thread_m + i * blockDim.x;
        if (global_m < M) {
            #pragma unroll
            for (int j = 0; j < MARLIN_N / blockDim.y; ++j) {
                int global_n = block_n + thread_n + j * blockDim.y;
                if (global_n < N) {
                    C[global_m * N + global_n] = accum[i][j];
                }
            }
        }
    }
}

// Marlin调度函数
void marlin_cuda_mul_mat_q4(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0,  // 量化权重
    const ggml_tensor * src1,  // 激活
    ggml_tensor * dst) {
    
    const int M = src1->ne[1];  // batch * seq_len
    const int N = src0->ne[1];  // output features
    const int K = src0->ne[0];  // input features
    
    // 检查Marlin支持的条件
    if (K % MARLIN_TILE_K != 0 || N % MARLIN_TILE_N != 0) {
        // 回退到标准MMQ
        mul_mat_q_case<GGML_TYPE_Q4_0>(ctx, {/* standard args */}, ctx.stream());
        return;
    }
    
    // 权重重排（如果需要）
    uint32_t* d_weights_reordered = nullptr;
    if (needs_reordering(src0)) {
        CUDA_CHECK(cudaMalloc(&d_weights_reordered, 
                              ggml_nbytes(src0)));
        
        dim3 reorder_grid((K * N / 32 + 255) / 256);
        dim3 reorder_block(256);
        
        marlin_reorder_weights<<<reorder_grid, reorder_block, 0, ctx.stream()>>>(
            (const uint32_t*)src0->data,
            d_weights_reordered,
            K / MARLIN_TILE_K,
            N / MARLIN_TILE_N
        );
    }
    
    // 启动Marlin GEMM kernel
    dim3 grid((M + MARLIN_TILE_M - 1) / MARLIN_TILE_M,
              (N + MARLIN_TILE_N - 1) / MARLIN_TILE_N);
    dim3 block(16, 4);  // 64线程每block
    
    // 确定使用的权重指针
    const uint32_t* weights_ptr = d_weights_reordered ? 
        d_weights_reordered : (const uint32_t*)src0->data;
    
    // 获取缩放因子指针（假设存储在src0的额外数据中）
    const half* scales_ptr = get_marlin_scales(src0);
    
    marlin_gemm_kernel<MARLIN_TILE_M, MARLIN_TILE_N, MARLIN_TILE_K>
        <<<grid, block, 0, ctx.stream()>>>(
        (const half*)src1->data,
        weights_ptr,
        scales_ptr,
        (half*)dst->data,
        M, N, K,
        MARLIN_GROUP_SIZE
    );
    
    CUDA_CHECK(cudaGetLastError());
    
    // 清理临时内存
    if (d_weights_reordered) {
        CUDA_CHECK(cudaFree(d_weights_reordered));
    }
}
```

#### 3. 集成到MMQ调度器
```cpp
// 修改 /ggml/src/ggml-cuda/mmq.cu

// 添加Marlin支持检查
bool marlin_support_available(const ggml_tensor* src0, int cc) {
    // 检查硬件支持（需要Compute Capability >= 7.5）
    if (cc < GGML_CUDA_CC_TURING) {
        return false;
    }
    
    // 检查尺寸对齐
    if (src0->ne[0] % MARLIN_TILE_K != 0 || src0->ne[1] % MARLIN_TILE_N != 0) {
        return false;
    }
    
    // 检查量化类型
    switch (src0->type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q4_K:
            return true;
        default:
            return false;
    }
}

// 修改调度函数
static void ggml_cuda_mul_mat_q_switch_type(
    ggml_backend_cuda_context & ctx, 
    const mmq_args & args, 
    cudaStream_t stream) {
    
    // 检查是否可以使用Marlin优化
    if (marlin_support_available(get_tensor_from_args(args), 
                                ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) {
        
        switch (args.type_x) {
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q4_1:
            case GGML_TYPE_Q4_K:
                marlin_cuda_mul_mat_q4(ctx, 
                                      get_src0_from_args(args),
                                      get_src1_from_args(args), 
                                      get_dst_from_args(args));
                return;
        }
    }
    
    // 回退到标准MMQ实现
    switch (args.type_x) {
        case GGML_TYPE_Q4_0:
            mul_mat_q_case<GGML_TYPE_Q4_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_q_case<GGML_TYPE_Q4_1>(ctx, args, stream);
            break;
        // ... 其他情况
    }
}
```

#### 4. 模型转换支持
```cpp
// 新增文件 /ggml/src/ggml-marlin.c
#include "ggml-impl.h"
#include "ggml-marlin.h"

// 将标准量化格式转换为Marlin优化格式
void ggml_marlin_quantize_q4_0(
    const float* src,
    void* dst,
    int n_elements,
    int group_size) {
    
    block_q4_marlin* blocks = (block_q4_marlin*)dst;
    int n_blocks = n_elements / QK_MARLIN;
    
    for (int i = 0; i < n_blocks; ++i) {
        const float* x = src + i * QK_MARLIN;
        block_q4_marlin* block = &blocks[i];
        
        // 计算组统计信息
        float min_val = FLT_MAX;
        float max_val = -FLT_MAX;
        
        for (int j = 0; j < QK_MARLIN; ++j) {
            min_val = fminf(min_val, x[j]);
            max_val = fmaxf(max_val, x[j]);
        }
        
        // 计算缩放因子和零点
        const float scale = (max_val - min_val) / 15.0f;
        const float zero_point = -min_val / scale;
        
        block->scale = GGML_FP32_TO_FP16(scale);
        block->zero_point = GGML_FP32_TO_FP16(zero_point);
        block->group_idx = i / (group_size / QK_MARLIN);
        
        // 量化权重并按Marlin格式打包
        for (int j = 0; j < QK_MARLIN; j += 8) {
            uint32_t packed = 0;
            for (int k = 0; k < 8; ++k) {
                if (j + k < QK_MARLIN) {
                    float val = x[j + k];
                    int quantized = (int)roundf(val / scale + zero_point);
                    quantized = CLAMP(quantized, 0, 15);
                    packed |= (quantized << (k * 4));
                }
            }
            block->qs[j / 8] = packed;
        }
    }
}

// 运行时格式转换
void ggml_marlin_convert_weights(
    const ggml_tensor* src,
    ggml_tensor* dst) {
    
    if (src->type == dst->type) {
        // 已经是Marlin格式，仅需内存重排
        marlin_reorder_tensor_data(src, dst);
    } else {
        // 需要格式转换
        switch (src->type) {
            case GGML_TYPE_Q4_0:
                convert_q4_0_to_marlin(src, dst);
                break;
            case GGML_TYPE_Q4_1:
                convert_q4_1_to_marlin(src, dst);
                break;
            default:
                GGML_ABORT("Unsupported conversion to Marlin format");
        }
    }
}
```

## 性能优化策略

### 1. 自适应调度
```cpp
// 性能启发式函数
bool should_use_marlin(const ggml_tensor* src0, const ggml_tensor* src1, int cc) {
    const int M = src1->ne[1];
    const int N = src0->ne[1];
    const int K = src0->ne[0];
    
    // 硬件要求
    if (cc < GGML_CUDA_CC_TURING) return false;
    
    // 尺寸对齐要求
    if (K % MARLIN_TILE_K != 0 || N % MARLIN_TILE_N != 0) return false;
    
    // 性能收益预测（基于实验数据）
    float marlin_perf = predict_marlin_performance(M, N, K);
    float standard_perf = predict_standard_mmq_performance(M, N, K);
    
    return marlin_perf > standard_perf * 1.1f;  // 至少10%提升才使用
}
```

### 2. 内存管理优化
```cpp
// Marlin内存池
struct marlin_memory_pool {
    void* reorder_buffer;
    size_t buffer_size;
    bool in_use;
};

static marlin_memory_pool g_marlin_pool = {nullptr, 0, false};

void* marlin_get_temp_buffer(size_t size) {
    if (g_marlin_pool.buffer_size < size) {
        if (g_marlin_pool.reorder_buffer) {
            cudaFree(g_marlin_pool.reorder_buffer);
        }
        CUDA_CHECK(cudaMalloc(&g_marlin_pool.reorder_buffer, size));
        g_marlin_pool.buffer_size = size;
    }
    return g_marlin_pool.reorder_buffer;
}
```

### 3. 精度验证
```cpp
// Marlin精度测试
void marlin_accuracy_test(const ggml_tensor* weights, const ggml_tensor* input) {
    // 标准实现结果
    ggml_tensor* ref_output = run_standard_mmq(weights, input);
    
    // Marlin实现结果  
    ggml_tensor* marlin_output = run_marlin_mmq(weights, input);
    
    // 计算误差
    float max_error = compute_max_error(ref_output, marlin_output);
    float rms_error = compute_rms_error(ref_output, marlin_output);
    
    printf("Marlin accuracy: max_error=%.6f, rms_error=%.6f\n", 
           max_error, rms_error);
    
    // 如果误差过大，禁用Marlin
    if (max_error > 1e-3 || rms_error > 1e-4) {
        disable_marlin_for_model(weights);
    }
}
```

## 集成时间表

### Phase 2A (Week 5-6): Marlin Kernel开发
- Marlin GEMM核心kernel实现
- 权重重排和内存优化
- 基础性能测试

### Phase 2B (Week 7): 系统集成  
- MMQ调度器集成
- 自适应调度逻辑
- 格式转换支持

### Phase 2C (Week 8): 优化和验证
- 性能调优和精度验证
- 多模型兼容性测试
- 内存管理优化

## 预期收益

- **矩阵乘法加速**：2.1-3.2x（针对4-bit量化模型）
- **端到端推理**：20-30%性能提升
- **内存带宽利用率**：提升40-50%
- **模型兼容性**：支持主流4-bit量化模型

这个Marlin集成设计提供了完整的技术路径，既保持了与现有架构的兼容性，又充分利用了Marlin的性能优势。