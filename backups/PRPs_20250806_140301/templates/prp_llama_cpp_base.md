name: "llama.cpp PRP模板 v1 - C++高性能计算与验证循环"
description: |

## 目的
专为llama.cpp开发优化的模板，包含充分的上下文和自验证能力，通过迭代改进实现高质量C++代码。

## 核心原则
1. **上下文为王**: 包含所有必要的文档、示例和注意事项
2. **验证循环**: 提供可执行的测试/检查，AI可以运行和修复
3. **信息密集**: 使用代码库的关键字和模式
4. **渐进成功**: 从简单开始，验证，然后增强
5. **全局规则**: 确保遵循CLAUDE.md中的所有规则

---

## 目标
[需要构建什么 - 具体说明最终状态和期望]

## 为什么
- [业务价值和用户影响]
- [与现有功能的集成]
- [解决的问题和受益对象]

## 什么
[用户可见的行为和技术需求]

### 成功标准
- [ ] [具体可衡量的结果]

## 所需的全部上下文

### 文档与参考资料 (列出实现功能所需的所有上下文)
```yaml
# 必须阅读 - 包含在你的上下文窗口中
- url: https://github.com/ggerganov/llama.cpp
  why: 主仓库，了解最新架构和模式
  
- url: https://github.com/ggerganov/llama.cpp/wiki
  why: 官方文档和最佳实践
  
- file: examples/[相关示例文件]
  why: 遵循的模式，避免的陷阱
  
- url: https://github.com/ggerganov/ggml
  why: GGML库文档，理解底层张量操作
  
- file: src/llama.cpp
  why: 核心API实现模式
  
- file: ggml-cuda.cu
  why: CUDA实现参考 (如果需要GPU支持)
  
- docfile: documentation/[相关文档].md
  why: 用户粘贴到项目中的文档
```

### 当前代码库树状结构 (在项目根目录运行 `find . -name "*.h" -o -name "*.cpp" -o -name "*.cu" | head -20` 获取概览)
```bash

```

### 期望的代码库树状结构与要添加的文件及文件职责
```bash

```

### 已知的代码库陷阱与库特性
```cpp
// 关键: llama.cpp使用特定的内存布局
// 示例: 所有量化块必须32字节对齐以支持AVX2
// 示例: CUDA kernel需要处理任意大小的张量，不只是2的幂
// 示例: ggml_tensor结构体有特定的生命周期管理要求
// 示例: 不同平台的SIMD指令支持需要运行时检测
```

## 实现蓝图

### 数据模型和结构

创建核心数据模型，确保类型安全和一致性。
```cpp
示例:
 - ggml_tensor结构扩展
 - 量化参数结构体
 - CUDA context管理类
 - 性能计数器结构

// 示例量化结构
struct quantization_params {
    float scale;
    int32_t zero_point;
    int bits;
    enum ggml_type type;
};

// RAII CUDA上下文管理
class cuda_context {
public:
    cuda_context();
    ~cuda_context();
    
    // 禁用拷贝
    cuda_context(const cuda_context&) = delete;
    cuda_context& operator=(const cuda_context&) = delete;
    
    // 允许移动
    cuda_context(cuda_context&&) noexcept;
    cuda_context& operator=(cuda_context&&) noexcept;
    
    cudaStream_t get_stream() const { return stream_; }
    
private:
    cudaStream_t stream_;
    bool initialized_;
};
```

### 按完成顺序列出要完成的任务列表

```yaml
任务 1: 设置基础结构
创建 src/new_feature.h:
  - 模式: 遵循 src/llama.h 的结构
  - 包含: 必要的前向声明
  - 保持: 与现有API的一致性

创建 src/new_feature.cpp:
  - 镜像模式来自: src/llama.cpp
  - 修改: 核心逻辑实现
  - 保持: 错误处理模式相同

任务 2: 实现CPU版本
修改 src/new_feature.cpp:
  - 查找模式: "现有类似功能"
  - 注入位置: 在现有函数模板之后
  - 保持: 现有函数签名

...

任务 N: CUDA支持 (如果需要)
创建 ggml-cuda-new-feature.cu:
  - 模式: 遵循 ggml-cuda.cu 的kernel模式
  - 包含: 适当的错误检查
  - 保持: 与CPU版本的数值一致性
```

### 每个任务的伪代码 (根据需要添加到每个任务)
```cpp

// 任务 1: 基础结构
// 伪代码包含关键细节，不要写完整代码
class new_feature_impl {
public:
    // 模式: 总是先验证输入 (参见 src/llama.cpp)
    bool initialize(const params& p);
    
    // 陷阱: 需要处理不同的张量类型
    template<typename T>
    ggml_tensor* process_tensor(ggml_context* ctx, ggml_tensor* input) {
        // 模式: 使用现有的张量创建模式
        ggml_tensor* result = ggml_new_tensor_2d(ctx, input->type, input->ne[0], input->ne[1]);
        
        // 关键: 设置操作类型用于计算图
        ggml_set_op_params(result, &params_, sizeof(params_));
        
        return result;
    }
    
    // 模式: 标准化的释放模式
    ~new_feature_impl() {
        // 清理GPU内存 (如果使用)
        cleanup_cuda_resources();
    }
};

// 任务 N: CUDA支持
__global__ void new_feature_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int n_elements,
    const feature_params params
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;
    
    // 关键: 使用__ldg进行缓存优化读取
    const float val = __ldg(&input[idx]);
    
    // 处理逻辑 (具体实现)
    output[idx] = process_value(val, params);
}
```

### 集成点
```yaml
构建系统:
  - 修改: CMakeLists.txt
  - 模式: "find_package(CUDA QUIET)" 用于可选CUDA支持
  
配置:
  - 添加到: common/common.h
  - 模式: "GGML_NEW_FEATURE_MAX_BATCH_SIZE"环境变量支持
  
API:
  - 添加到: llama.h
  - 模式: "llama_[feature_name]_*" 函数命名约定

测试:
  - 添加到: tests/
  - 模式: test-[feature-name].cpp命名约定
```

## 验证循环

### 级别 1: 语法与风格
```bash
# 首先运行这些 - 继续之前修复任何错误
make -j$(nproc) 2>&1 | grep -E "(error|Error)"  # 编译错误检查
cppcheck --enable=all --std=c++11 src/new_feature.cpp  # 静态分析

# 预期: 无错误。如果有错误，阅读并修复。
```

### 级别 2: 单元测试 每个新功能/文件/函数使用现有测试模式
```cpp
// 创建 tests/test-new-feature.cpp 包含这些测试用例:
void test_basic_functionality() {
    // 基本功能工作
    ggml_context* ctx = ggml_init({.mem_size = 1024*1024, .mem_buffer = nullptr});
    
    // 创建测试张量
    ggml_tensor* input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 100);
    
    // 应用新功能
    ggml_tensor* result = new_feature_forward(ctx, input);
    
    // 验证结果
    assert(result != nullptr);
    assert(result->type == GGML_TYPE_F32);
    
    ggml_free(ctx);
}

void test_cuda_implementation() {
    // CUDA实现与CPU匹配
    if (!ggml_cuda_available()) return;
    
    // 测试相同输入在CPU和GPU上产生相同结果
    // ... 数值精度测试
}

void test_edge_cases() {
    // 处理边界情况
    // 空张量、单元素张量、大张量
    // ... 边界条件测试
}
```

```bash
# 运行并迭代直到通过:
cd tests && make test-new-feature && ./test-new-feature

# 如果失败: 阅读错误，理解根本原因，修复代码，重新运行
```

### 级别 3: 集成测试
```bash
# 测试与主程序的集成
cd examples/main && make

# 测试新功能
echo "测试提示" | ./main -m ../../models/test-model.gguf --new-feature-option

# 预期: 正常运行没有崩溃，输出符合预期
# 如果错误: 检查 stderr 输出获取堆栈跟踪
```

### 级别 4: 性能验证
```bash
# 性能基准测试
cd tests && make test-new-feature-perf && ./test-new-feature-perf

# 预期性能指标:
# - CPU版本: 不慢于现有类似功能的110%
# - CUDA版本: 至少比CPU版本快2倍 (如果适用)
# - 内存使用: 不超过现有方案的120%

# 如果性能不达标: 分析瓶颈，优化热点路径
```

## 最终验证检查清单
- [ ] 所有测试通过: `cd tests && make && make test`
- [ ] 无编译错误: `make clean && make -j$(nproc)`
- [ ] 无内存泄漏: `valgrind --leak-check=full ./test-new-feature`
- [ ] 性能符合目标: 基准测试达到预期
- [ ] CUDA支持正常 (如果适用): 在GPU上测试
- [ ] 错误情况优雅处理: 测试异常输入
- [ ] 文档更新 (如果需要): README或注释
- [ ] 向后兼容: 不破坏现有API

---

## 要避免的反模式
- ❌ 不要创建新模式，现有模式有效时
- ❌ 不要跳过验证，认为"应该能工作"
- ❌ 不要忽略失败的测试 - 修复它们
- ❌ 不要在异步上下文中使用同步函数
- ❌ 不要硬编码应该配置的值
- ❌ 不要捕获所有异常 - 要具体
- ❌ 不要假设CUDA总是可用 - 提供CPU回退
- ❌ 不要忽略内存对齐要求 - SIMD需要对齐
- ❌ 不要跳过数值精度验证 - 确保正确性

## 置信度分数: [1-10]

高置信度因为:
- 遵循既定的llama.cpp模式
- 包含全面的验证门控
- 详细的性能和正确性测试
- 处理常见的C++/CUDA陷阱

不确定性在:
- 具体的性能优化可能需要迭代
- CUDA兼容性取决于硬件配置