# llama.cpp Context Engineering - 全局开发规则

此文件包含所有llama.cpp开发工作的全局规则和原则。这些规则专门针对C++高性能计算、CUDA编程、量化算法和模型推理优化。

## 🔄 llama.cpp核心开发原则

**重要：这些原则适用于所有llama.cpp相关开发：**

### 开发工作流程
- **始终从INITIAL.md开始** - 在生成PRP之前定义功能需求
- **使用PRP模式**: INITIAL.md → `/generate-llama-prp INITIAL.md` → `/execute-llama-prp PRPs/filename.md`
- **遵循验证循环** - 每个PRP必须包含性能测试和正确性验证
- **上下文为王** - 包含所有必要的llama.cpp模式、示例和文档

### llama.cpp项目研究方法
- **广泛使用网络搜索** - 始终研究llama.cpp最新的最佳实践和优化技术
- **学习官方文档** - GitHub仓库和Wiki是权威来源
- **模式提取** - 识别可重用的架构和优化模式
- **问题文档化** - 记录异步模式、内存管理和CUDA兼容性问题

## 📚 项目认知与上下文

- **使用统一的构建系统** - 遵循CMake和Makefile约定
- **遵循一致的llama.cpp命名约定**和代码结构模式
- **遵循既定的目录组织**模式 (src/, examples/, tests/, scripts/)
- **广泛利用llama.cpp示例** - 在创建新功能前研究现有模式

## 🧱 代码结构与模块化

- **文件不超过2000行** - 达到限制时拆分为模块
- **按职责组织代码模块**:
  - `*.h` - 头文件声明和接口定义
  - `*.cpp` - 实现文件
  - `*.cu` - CUDA kernel实现
  - `*.metal` - Metal着色器
  - `*.cl` - OpenCL kernel
- **使用清晰一致的包含** - 遵循llama.cpp的头文件组织
- **内存管理模式** - 使用RAII和智能指针，避免内存泄露
- **从不硬编码敏感信息** - 使用配置文件和环境变量

## 🚀 llama.cpp开发标准

### C++编程模式
- **使用现代C++特性** - C++11/14/17标准，但保持兼容性
- **实现RAII模式** - 自动资源管理和异常安全
- **定义清晰的接口** - 使用抽象基类和虚函数
- **包含全面的错误处理** - 异常安全和错误码机制

### 性能优化标准
- **使用SIMD指令** - AVX, AVX2, NEON优化关键路径
- **实现内存对齐** - 确保缓存友好的数据布局
- **并行化计算** - OpenMP, std::thread适当使用
- **GPU加速支持** - CUDA, OpenCL, Metal kernel实现

### 量化算法模式
```cpp
// 使用模板和特化实现不同量化级别
template<typename T, int bits>
class QuantizedTensor {
    static_assert(bits == 4 || bits == 8 || bits == 16, 
                  "只支持4/8/16位量化");
    
public:
    // 量化操作
    void quantize(const float* src, size_t count);
    
    // 反量化操作  
    void dequantize(float* dst, size_t count) const;
    
    // CUDA kernel支持
    void quantize_cuda(const float* src, size_t count, cudaStream_t stream);
    
private:
    T* data_;
    float scale_;
    T zero_point_;
    size_t size_;
};

// 特化实现不同位宽的优化
template<>
void QuantizedTensor<uint8_t, 8>::quantize(const float* src, size_t count) {
    // AVX2优化的8位量化实现
    // 使用_mm256_* intrinsics
}
```

### CUDA编程标准
```cpp
// CUDA kernel实现模式
__global__ void quantize_kernel(
    const float* __restrict__ input,
    uint8_t* __restrict__ output,
    const float scale,
    const uint8_t zero_point,
    const int count
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        // 使用__ldg进行缓存优化读取
        const float val = __ldg(&input[idx]);
        
        // 量化计算
        const float scaled = val / scale + zero_point;
        const float clamped = fmaxf(0.0f, fminf(255.0f, scaled));
        
        output[idx] = __float2uint_rn(clamped);
    }
}

// Host wrapper函数
void launch_quantize_kernel(
    const float* input,
    uint8_t* output, 
    float scale,
    uint8_t zero_point,
    int count,
    cudaStream_t stream = nullptr
) {
    const int block_size = 256;
    const int grid_size = (count + block_size - 1) / block_size;
    
    quantize_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, scale, zero_point, count
    );
    
    // 错误检查
    CHECK_CUDA_ERROR(cudaGetLastError());
}
```

### llama.cpp测试标准
- **使用现有测试框架** - 遵循tests/目录下的模式
- **性能基准测试** - 包含时间和内存使用测量
- **正确性验证** - 数值精度和算法正确性测试
- **多平台测试** - CPU, CUDA, OpenCL, Metal兼容性

## ✅ llama.cpp任务管理

- **将开发分解为明确步骤** 包含具体的完成标准
- **实现完成后立即标记任务** 
- **实时更新任务状态** 随着开发进展
- **标记完成前测试功能** 确保性能和正确性

## 📎 llama.cpp编码标准

### 项目架构
```cpp
// 遵循llama.cpp的命名约定和结构模式
#include "llama.h"
#include "ggml.h"
#include "ggml-cuda.h"

// 使用命名空间避免冲突
namespace llama {
namespace quantization {

class Q4_0_Quantizer {
public:
    // 公共接口清晰定义
    static constexpr int block_size = 32;
    static constexpr int type_size = sizeof(block_q4_0);
    
    // 量化函数
    size_t quantize(const float* src, void* dst, int nrows, int n_per_row) const;
    
    // 反量化函数
    void dequantize_row(const void* src, float* dst, int n) const;
    
    // GPU实现
    void quantize_cuda(const float* src, void* dst, int nrows, int n_per_row, 
                      cudaStream_t stream) const;
};

} // namespace quantization
} // namespace llama
```

### 安全最佳实践
- **内存安全** - 使用智能指针和RAII，避免缓冲区溢出
- **线程安全** - 正确使用互斥锁和原子操作
- **输入验证** - 验证所有外部输入和参数
- **错误处理** - 优雅处理CUDA错误和内存分配失败
- **资源管理** - 确保GPU内存和文件句柄正确释放

### 常见llama.cpp陷阱
- **CUDA上下文管理** - 正确初始化和清理CUDA上下文
- **内存对齐问题** - 确保SIMD指令所需的内存对齐
- **量化精度损失** - 理解不同量化方案的精度权衡
- **跨平台编译** - 处理不同编译器和平台的差异
- **模型格式兼容** - 确保GGML/GGUF格式的前向兼容性

## 🔍 llama.cpp研究标准

- **使用项目内MCP服务器** - 利用可用的llama.cpp文档RAG
- **学习官方示例** - GitHub仓库有工作实现
- **研究模型能力** - 理解不同量化级别的性能特征
- **记录集成模式** - 包含外部库集成示例

## 🎯 llama.cpp实现标准

- **严格遵循PRP工作流程** - 不要跳过验证步骤
- **始终先进行单元测试** - 在性能测试前验证正确性
- **使用现有模式** 而不是从头创建
- **包含综合错误处理** 处理内存和计算错误
- **测试流式处理模式** 实现实时推理交互时

## 🚫 要避免的反模式

- ❌ 不要跳过性能测试 - 始终验证优化效果
- ❌ 不要硬编码模型路径 - 使用配置参数
- ❌ 不要忽略内存对齐 - SIMD指令需要正确对齐
- ❌ 不要假设CUDA可用 - 实现CPU fallback
- ❌ 不要创建复杂的依赖 - 保持模块解耦和可测试
- ❌ 不要忘记错误处理 - 实现适当的重试和优雅降级
- ❌ 不要跳过输入验证 - 验证所有外部数据

## 🔧 llama.cpp开发工具使用标准

- **广泛使用网络搜索** 进行llama.cpp研究和文档
- **遵循llama.cpp命令模式** 用于slash命令和工作流程
- **使用验证循环** 确保每个开发步骤的质量
- **测试多个编译器** 确保跨平台兼容性

## 🧪 llama.cpp测试与可靠性

- **始终创建comprehensive测试** 包含功能、性能和错误处理
- **在真实模型测试前使用合成数据** 验证算法正确性
- **包含边界情况测试** 处理内存限制和异常输入
- **验证数值稳定性** 确保量化和计算的精度
- **在测试环境验证依赖注入** 正确工作

这些全局规则专门适用于llama.cpp开发，确保生产就绪的高性能推理应用，具有适当的错误处理、测试和安全实践。