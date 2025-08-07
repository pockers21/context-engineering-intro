## FEATURE:

[在这里详细描述你想要为llama.cpp实现的功能。越具体越好！]

示例：
- 实现新的Q3_K量化算法，支持CPU和CUDA实现
- 优化现有的RMSNorm操作，使用AVX-512指令集
- 添加新的模型架构支持（如Mamba、RetNet等）
- 实现内存优化的KV-cache管理
- 添加新的采样策略

## TECHNICAL REQUIREMENTS:

[描述技术要求和约束条件]

示例：
- **性能目标**: 比现有实现快20%，内存使用不超过110%
- **平台支持**: Linux, Windows, macOS
- **硬件支持**: CPU (AVX2), CUDA (Compute Capability 6.0+)
- **精度要求**: 数值误差不超过1e-4
- **内存对齐**: 支持32字节对齐以优化SIMD
- **线程安全**: 支持多线程并发访问

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