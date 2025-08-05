# llama.cpp Context Engineering 模板

专为llama.cpp开发优化的Context Engineering模板，基于真实的llama.cpp项目结构和编码风格设计。

## 🚀 快速开始

### 1. 复制模板到你的llama.cpp项目

```bash
# 复制整个模板到你的llama.cpp项目根目录
cp -r llama-cpp-context-engineering/* /path/to/your/llama.cpp/

# 或者只复制需要的部分
cp -r llama-cpp-context-engineering/.claude /path/to/your/llama.cpp/
cp llama-cpp-context-engineering/CLAUDE.md /path/to/your/llama.cpp/
cp -r llama-cpp-context-engineering/PRPs /path/to/your/llama.cpp/
```

### 2. 使用Context Engineering工作流

```bash
# 1. 定义你要实现的功能
# 编辑 PRPs/INITIAL.md，描述你想要的llama.cpp功能

# 2. 生成详细的实现计划
/generate-llama-prp PRPs/INITIAL.md

# 3. 执行实现
/execute-llama-prp PRPs/your-generated-prp.md
```

## 📁 模板结构

```
llama-cpp-context-engineering/
├── CLAUDE.md                          # llama.cpp专用开发规则
├── .claude/commands/
│   ├── generate-llama-prp.md          # 生成llama.cpp PRP命令
│   └── execute-llama-prp.md           # 执行llama.cpp PRP命令
├── PRPs/
│   ├── templates/
│   │   └── prp_llama_cpp_base.md      # llama.cpp PRP基础模板
│   └── INITIAL.md                     # 功能需求模板
├── examples/                          # llama.cpp开发示例
│   ├── quantization/                  # 量化算法示例
│   ├── optimization/                  # SIMD优化示例
│   ├── cuda_kernels/                  # CUDA kernel示例
│   ├── build_system/                  # CMake构建示例
│   └── model_formats/                 # GGUF格式处理示例
├── documentation/                     # 开发文档
└── README.md                          # 本文件
```

## 🎯 核心特性

### Context Engineering优势
- **10倍效率提升**: 相比传统提示工程，提供结构化的开发流程
- **基于真实代码**: 所有示例都基于实际的llama.cpp代码风格
- **验证循环**: 内置测试和验证机制，确保代码质量
- **渐进式开发**: 从简单到复杂，逐步实现功能

### llama.cpp专业化
- **遵循项目约定**: 命名、结构、错误处理都与llama.cpp保持一致
- **性能优化**: 包含SIMD、CUDA优化的实际示例
- **内存管理**: RAII、智能指针等现代C++模式
- **跨平台支持**: 处理不同编译器和平台的差异

## 📚 使用示例

### 示例1: 添加新的量化算法

1. **定义需求** (PRPs/INITIAL.md):
```markdown
## FEATURE:
实现Q3_K量化算法，支持CPU和CUDA两种实现，性能要求达到Q4_K的90%

## TOOLS:
- 量化函数：将F32转换为Q3_K格式
- 反量化函数：将Q3_K转换回F32
- 向量点积：Q3_K与Q8_K的高效点积运算
- CUDA kernel：GPU加速版本

## DEPENDENCIES:
- ggml-quants.h：量化类型定义
- CUDA runtime：GPU支持
- AVX2指令集：CPU SIMD优化
```

2. **生成PRP**:
```bash
/generate-llama-prp PRPs/INITIAL.md
```

3. **执行实现**:
```bash
/execute-llama-prp PRPs/q3_k_quantization-2024-01-15.md
```

### 示例2: 优化现有操作

1. **定义需求**:
```markdown
## FEATURE:
优化RMSNorm操作，使用AVX-512指令集，目标性能提升30%

## TOOLS:
- SIMD实现：使用AVX-512的RMSNorm
- 基准测试：性能对比工具
- 单元测试：数值精度验证
```

2. **直接生成并执行**:
```bash
/generate-llama-prp PRPs/INITIAL.md && /execute-llama-prp PRPs/rmsnorm_optimization-*.md
```

## 🛠️ 开发工具

### Slash命令

#### `/generate-llama-prp`
根据INITIAL.md生成详细的实现计划:
- 研究现有llama.cpp模式
- 收集相关文档和示例
- 创建分步实现计划
- 包含验证和测试策略

#### `/execute-llama-prp`
执行PRP中的实现计划:
- 创建任务列表并跟踪进度
- 按步骤实现功能
- 运行测试和验证
- 处理错误和边界情况

### 示例库

所有示例都基于真实的llama.cpp代码：

1. **量化示例** (`examples/quantization/`):
   - Q4_K量化实现模式
   - 遵循ggml-quants.c的结构
   - 包含性能测试和验证

2. **SIMD优化** (`examples/optimization/`):
   - AVX2/NEON向量操作
   - 内存对齐和缓存优化
   - 跨平台兼容性处理

3. **CUDA Kernel** (`examples/cuda_kernels/`):
   - GPU量化实现
   - 内存传输优化
   - 错误处理和设备管理

4. **构建系统** (`examples/build_system/`):
   - CMake配置模式
   - 跨平台编译设置
   - 依赖管理和选项定义

5. **文件格式** (`examples/model_formats/`):
   - GGUF文件解析
   - 内存映射优化
   - 元数据处理

## 🔧 开发规则 (CLAUDE.md)

模板包含专门为llama.cpp定制的开发规则：

### C++编程标准
- 使用现代C++11/14特性
- RAII资源管理
- 智能指针和异常安全
- 跨平台兼容性

### 性能优化
- SIMD指令集使用
- 内存对齐要求
- 缓存友好的数据布局
- GPU/CPU混合计算

### 代码质量
- 统一的错误处理
- 全面的单元测试
- 性能基准测试
- 内存泄漏检查

## 🧪 验证和测试

每个PRP都包含多层验证：

### Level 1: 编译检查
```bash
make clean && make -j$(nproc)
cppcheck --enable=all --std=c++11 src/new_feature.cpp
```

### Level 2: 单元测试
```bash
cd tests && make test-new-feature && ./test-new-feature
```

### Level 3: 性能测试
```bash
./test-new-feature-perf
# 验证性能目标是否达到
```

### Level 4: 集成测试
```bash
cd examples/main && make
echo "test prompt" | ./main -m ../../models/test.gguf --new-feature
```

## 📈 性能预期

使用此模板可以期待：

- **开发效率**: 提升60-80%
- **代码质量**: 更少的bug，更好的性能
- **学习曲线**: 快速掌握llama.cpp开发模式
- **维护成本**: 降低长期维护难度

## 🔗 相关资源

- [llama.cpp官方仓库](https://github.com/ggerganov/llama.cpp)
- [GGML文档](https://github.com/ggerganov/ggml)
- [Context Engineering方法论](../README.md)
- [Claude Code文档](https://docs.anthropic.com/en/docs/claude-code)

## 🤝 贡献指南

欢迎贡献新的示例和改进：

1. 确保示例基于真实的llama.cpp代码
2. 包含完整的测试和文档
3. 遵循项目的代码风格
4. 添加性能基准数据

## 📝 许可证

本模板遵循MIT许可证，可自由使用和修改。

---

**准备好优化你的llama.cpp开发了吗？** 从复制模板和编辑`PRPs/INITIAL.md`开始吧！ 🚀