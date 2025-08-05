# llama.cpp 开发模式指南

基于真实llama.cpp代码分析的开发模式总结。

## 🏗️ 项目架构模式

### 目录结构约定
```
llama.cpp/
├── src/                    # 核心实现
│   ├── llama.cpp          # 主API实现
│   ├── llama-*.cpp        # 功能模块
│   └── llama-*.h          # 模块头文件
├── ggml/                  # 张量操作库
│   ├── include/           # 公共头文件
│   └── src/              # 实现文件
├── examples/              # 示例程序
├── tests/                 # 单元测试
└── tools/                # 工具程序
```

### 模块分离原则
- **功能模块化**: 每个模块负责特定功能（如quant、vocab、model等）
- **接口清晰**: 公共API在include/目录，内部实现在src/
- **依赖最小化**: 模块间依赖关系清晰，避免循环依赖

## 🔧 编码模式

### 1. 错误处理模式

```cpp
// 使用LLAMA_LOG_*宏记录日志
LLAMA_LOG_ERROR("%s: failed to load model\n", __func__);
LLAMA_LOG_WARN("%s: unsupported feature\n", __func__);
LLAMA_LOG_INFO("%s: model loaded successfully\n", __func__);

// 函数返回值约定
bool load_model(const std::string& path) {
    if (path.empty()) {
        LLAMA_LOG_ERROR("%s: empty path\n", __func__);
        return false;
    }
    // ... 实现
    return true;
}

// 异常处理（仅在构造函数中使用）
class model_loader {
public:
    explicit model_loader(const std::string& path) {
        if (!load(path)) {
            throw std::runtime_error("failed to load model");
        }
    }
};
```

### 2. 内存管理模式

```cpp
// RAII资源管理
class cuda_context {
public:
    cuda_context() {
        CHECK_CUDA_ERROR(cudaStreamCreate(&stream_));
    }
    
    ~cuda_context() {
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
    }
    
    // 禁用拷贝，允许移动
    cuda_context(const cuda_context&) = delete;
    cuda_context& operator=(const cuda_context&) = delete;
    cuda_context(cuda_context&&) = default;
    cuda_context& operator=(cuda_context&&) = default;
    
private:
    cudaStream_t stream_ = nullptr;
};

// 智能指针使用
std::unique_ptr<ggml_context, decltype(&ggml_free)> 
    ctx(ggml_init(params), &ggml_free);
```

### 3. 命名约定

```cpp
// 函数和变量: snake_case
int n_vocab = 0;
float compute_score(const float* data, int size);

// 类型: snake_case with suffix
struct llama_model {};
enum ggml_type {};

// 常量: UPPER_CASE
#define LLAMA_MAX_DEVICES 16
const int QK_K = 256;

// 私有成员: trailing underscore
class example {
private:
    int value_;
    std::string name_;
};
```

### 4. 参数验证模式

```cpp
static size_t quantize_row(const float * src, void * dst, int n) {
    // 输入验证
    if (!src || !dst) {
        LLAMA_LOG_ERROR("%s: null pointer\n", __func__);
        return 0;
    }
    
    if (n <= 0 || n % QK_K != 0) {
        LLAMA_LOG_ERROR("%s: invalid size %d\n", __func__, n);
        return 0;
    }
    
    // 实现...
    return processed_bytes;
}
```

## ⚡ 性能优化模式

### 1. SIMD使用模式

```cpp
// 编译时特性检测
#if defined(__AVX2__)
static void process_avx2(const float* src, float* dst, int n) {
    const int n_vec = n & ~7; // 8个float一组
    
    for (int i = 0; i < n_vec; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        // 处理...
        _mm256_storeu_ps(dst + i, v);
    }
    
    // 处理剩余元素
    for (int i = n_vec; i < n; ++i) {
        dst[i] = process_scalar(src[i]);
    }
}
#endif

// 运行时分发
void process_data(const float* src, float* dst, int n) {
#if defined(__AVX2__)
    if (ggml_cpu_has_avx2()) {
        process_avx2(src, dst, n);
        return;
    }
#endif
    process_fallback(src, dst, n);
}
```

### 2. 内存访问模式

```cpp
// 缓存友好的循环顺序
void matrix_multiply(const float* A, const float* B, float* C, 
                    int M, int N, int K) {
    // 分块处理，优化缓存命中率
    const int BLOCK_SIZE = 64;
    
    for (int ii = 0; ii < M; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
            for (int kk = 0; kk < K; kk += BLOCK_SIZE) {
                // 内核计算
                int i_max = std::min(ii + BLOCK_SIZE, M);
                int j_max = std::min(jj + BLOCK_SIZE, N);
                int k_max = std::min(kk + BLOCK_SIZE, K);
                
                for (int i = ii; i < i_max; ++i) {
                    for (int j = jj; j < j_max; ++j) {
                        float sum = 0.0f;
                        for (int k = kk; k < k_max; ++k) {
                            sum += A[i*K + k] * B[k*N + j];
                        }
                        C[i*N + j] += sum;
                    }
                }
            }
        }
    }
}
```

### 3. 并行化模式

```cpp
// OpenMP并行化
static void compute_parallel(const float* input, float* output, 
                            int n_elements, int n_threads) {
    #pragma omp parallel for num_threads(n_threads)
    for (int i = 0; i < n_elements; ++i) {
        output[i] = expensive_computation(input[i]);
    }
}

// 手动线程池
class thread_pool {
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;

public:
    explicit thread_pool(size_t threads) : stop(false) {
        for (size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this] {
                // 工作线程实现...
            });
        }
    }
};
```

## 🧪 测试模式

### 1. 单元测试结构

```cpp
// tests/test-quantize-fns.cpp 模式
#include "ggml.h"
#undef NDEBUG
#include <assert.h>

static const char* RESULT_STR[] = {"ok", "FAILED"};

// 测试数据生成
static void generate_data(float offset, size_t n, float * dst) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = 0.1f + 2.0f * cosf(i + offset);
    }
}

// 精度验证
static float calculate_rmse(const float* a, const float* b, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrtf(sum / n);
}

// 测试函数
static bool test_quantization_accuracy() {
    const size_t test_size = 1024;
    std::vector<float> src(test_size);
    std::vector<uint8_t> quantized(test_size);
    std::vector<float> dst(test_size);
    
    generate_data(0.0f, test_size, src.data());
    
    // 量化和反量化
    quantize_row_q4_0(src.data(), quantized.data(), test_size);
    dequantize_row_q4_0(quantized.data(), dst.data(), test_size);
    
    // 验证精度
    float rmse = calculate_rmse(src.data(), dst.data(), test_size);
    return rmse < MAX_QUANTIZATION_ERROR;
}

int main() {
    bool success = true;
    
    success &= test_quantization_accuracy();
    success &= test_edge_cases();
    success &= test_performance();
    
    printf("Overall result: %s\n", RESULT_STR[!success]);
    return success ? 0 : 1;
}
```

### 2. 性能测试模式

```cpp
#include <chrono>

static void benchmark_operation() {
    const int iterations = 1000;
    const int data_size = 4096;
    
    std::vector<float> input(data_size);
    std::vector<float> output(data_size);
    
    // 预热
    for (int i = 0; i < 10; ++i) {
        test_operation(input.data(), output.data(), data_size);
    }
    
    // 基准测试
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        test_operation(input.data(), output.data(), data_size);
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double throughput = (double)(iterations * data_size) / duration.count(); // MB/s
    
    printf("Throughput: %.2f MB/s\n", throughput);
}
```

## 🔌 扩展模式

### 1. 新操作添加模式

```cpp
// 1. 在ggml.h中声明
GGML_API struct ggml_tensor * ggml_new_operation(
    struct ggml_context * ctx,
    struct ggml_tensor  * a);

// 2. 在ggml.c中实现
struct ggml_tensor * ggml_new_operation_impl(
    struct ggml_context * ctx,
    struct ggml_tensor  * a) {
    
    // 验证输入
    GGML_ASSERT(a);
    
    // 创建结果张量
    struct ggml_tensor * result = ggml_new_tensor(ctx, a->type, a->n_dims, a->ne);
    
    // 设置操作类型
    result->op = GGML_OP_NEW_OPERATION;
    result->src[0] = a;
    
    return result;
}

// 3. 添加到计算图执行
static void ggml_compute_forward_new_operation_f32(
    const struct ggml_compute_params * params,
    const struct ggml_tensor * src0,
    struct ggml_tensor * dst) {
    
    // 实现具体计算逻辑
}
```

### 2. 后端集成模式

```cpp
// 后端注册
static struct ggml_backend_i new_backend_i = {
    /* .get_name                = */ ggml_backend_new_name,
    /* .free                    = */ ggml_backend_new_free,
    /* .get_default_buffer_type = */ ggml_backend_new_get_default_buffer_type,
    /* .set_tensor              = */ ggml_backend_new_set_tensor,
    /* .get_tensor              = */ ggml_backend_new_get_tensor,
    /* .cpy_tensor_from         = */ ggml_backend_new_cpy_tensor_from,
    /* .cpy_tensor_to           = */ ggml_backend_new_cpy_tensor_to,
    /* .graph_plan_create       = */ ggml_backend_new_graph_plan_create,
    /* .graph_plan_free         = */ ggml_backend_new_graph_plan_free,
    /* .graph_plan_compute      = */ ggml_backend_new_graph_plan_compute,
    /* .supports_op             = */ ggml_backend_new_supports_op,
};
```

## 📦 构建系统模式

### CMake模式

```cmake
# 条件编译
option(LLAMA_NEW_FEATURE "Enable new feature" OFF)

if (LLAMA_NEW_FEATURE)
    add_compile_definitions(LLAMA_NEW_FEATURE)
    list(APPEND LLAMA_SOURCES src/llama-new-feature.cpp)
endif()

# 依赖查找
find_package(NewLibrary QUIET)
if (NewLibrary_FOUND)
    target_link_libraries(llama PRIVATE NewLibrary::NewLibrary)
    add_compile_definitions(LLAMA_USE_NEW_LIBRARY)
endif()

# 测试添加
if (LLAMA_BUILD_TESTS)
    add_executable(test-new-feature tests/test-new-feature.cpp)
    target_link_libraries(test-new-feature PRIVATE llama)
    add_test(NAME test-new-feature COMMAND test-new-feature)
endif()
```

这些模式基于真实的llama.cpp代码分析得出，遵循这些模式可以确保你的代码与项目风格保持一致，并获得最佳的性能和可维护性。