## FEATURE:

基于深入的论文分析，为llama.cpp融合最有价值的推理优化技术。

具体功能包括：
- 分析42篇推理优化论文的完整技术细节和llama.cpp融合可行性
- 实现INT-FlashAttention：将INT8量化与FlashAttention融合，实现72%内存节省
- 实现KV-Compress：分页KV缓存压缩，支持不同attention head的可变压缩率
- 实现Sorting-Free GPU Kernels：优化采样过程的GPU内核
- 算子融合优化：将dequant和后续计算融合，减少内存访问
- 评估并集成其他高ROI的优化技术（如Training-Free Activation Sparsity）

## TECHNICAL REQUIREMENTS:

**融合约束条件**：
- **架构兼容性**: 必须与现有GGML框架和llama.cpp架构完全兼容
- **向后兼容性**: 不能破坏现有模型文件格式和API接口
- **渐进式集成**: 作为可选功能添加，不影响现有功能稳定性

**性能目标**：
- **INT-FlashAttention**: 目标72%内存节省 + 1.7x推理加速
- **KV-Compress**: 目标60-80%KV缓存内存节省，特别是长文本场景
- **整体优化**: 在保持精度的前提下，推理速度提升1.5-2x

**硬件支持**：
- **CPU支持**: x86_64 (AVX2), ARM64 (NEON)
- **GPU支持**: CUDA (Compute Capability 6.0+, 特别是Ampere架构)
- **内存要求**: 支持大模型在有限内存设备上运行

**精度要求**：
- INT8量化精度损失 < 2%
- KV缓存压缩精度损失 < 1%
- 整体模型输出质量不明显下降

## IMPLEMENTATION SCOPE:

[描述实现范围，包括需要修改/创建的文件]

示例：
- **核心实现**: src/llama-new-feature.cpp, src/llama-new-feature.h
- **量化支持**: ggml/src/ggml-quants.c中添加新类型
- **CUDA实现**: ggml/src/ggml-cuda/new-feature.cu
- **测试文件**: tests/test-new-feature.cpp
- **示例程序**: examples/new-feature/main.cpp
- **CMake更新**: 更新构建配置以包含新文件

## EXAMPLES TO REFERENCE:

[列出examples/目录下相关的参考示例]

本模板包含以下示例可供参考：
- examples/quantization/q4_k_example.cpp - Q4_K量化实现模式
- examples/optimization/simd_optimized_ops.cpp - SIMD优化模式
- examples/cuda_kernels/quantize_cuda_example.cu - CUDA kernel实现模式
- examples/build_system/CMakeLists_example.txt - CMake构建模式
- examples/model_formats/gguf_handler_example.cpp - GGUF文件处理模式

[如果你有其他相关的代码示例，也可以放在examples/目录下并在这里引用]

## DOCUMENTATION TO REFERENCE:

[列出需要参考的文档和资源]

### llama.cpp官方资源:
- https://github.com/ggerganov/llama.cpp - 主仓库
- https://github.com/ggerganov/llama.cpp/wiki - 官方Wiki
- https://github.com/ggerganov/ggml - GGML库文档

### 技术规范:
- GGUF格式规范: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- CUDA编程指南: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- Intel Intrinsics指南: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/

### 算法参考:
[根据你的具体功能添加相关论文或技术文档]
- 量化算法论文链接
- 优化技术文档
- 模型架构论文

## OTHER CONSIDERATIONS:

[其他重要考虑事项，特别是llama.cpp开发中常见的陷阱]

### 常见陷阱提醒:
- **内存对齐**: SIMD操作需要32字节对齐，否则会崩溃
- **CUDA上下文**: 正确管理CUDA context和stream
- **线程安全**: ggml_context不是线程安全的，需要适当同步
- **数值精度**: 不同量化级别的精度权衡
- **向后兼容**: 确保不破坏现有API和文件格式兼容性

### 性能优化要点:
- **内存访问模式**: 优化缓存局部性
- **分支预测**: 减少条件分支
- **向量化**: 充分利用SIMD指令
- **GPU内存**: 合并内存访问，减少传输

### 测试策略:
- **单元测试**: 每个函数的正确性验证
- **性能测试**: 基准测试和回归检测
- **边界测试**: 极端输入的处理
- **平台测试**: 多平台兼容性验证

### 代码风格遵循:
- 使用llama.cpp的命名约定（snake_case）
- 遵循现有的错误处理模式
- 添加适当的日志输出（LLAMA_LOG_*）
- 包含详细的函数文档注释

---

**填写完成后，运行 `/generate-llama-prp PRPs/INITIAL.md` 来生成详细的实现计划！**