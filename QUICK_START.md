# 🚀 llama.cpp Context Engineering - 快速启动

**给新Claude实例的说明：这是一个为llama.cpp优化的Context Engineering工作环境**

## 🤖 Claude，你需要知道的

### 这个目录包含什么
- ✅ 完整的llama.cpp项目
- ✅ Context Engineering模板（已部署）
- ✅ 专门为llama.cpp优化的开发工具

### 立即可用的工具
1. **生成PRP命令**: `/generate-llama-prp PRPs/INITIAL.md`
2. **执行PRP命令**: `/execute-llama-prp PRPs/file.md`
3. **开发规则**: 详见 `CLAUDE.md`
4. **代码示例**: 详见 `context_engineering_examples/`

## 🎯 标准工作流程

### 步骤1: 编辑需求
```bash
# 用户会编辑这个文件来描述想要的功能
vi PRPs/INITIAL.md
```

### 步骤2: 生成实现计划 
```bash
# 你运行这个命令生成详细的PRP
/generate-llama-prp PRPs/INITIAL.md
```

### 步骤3: 执行实现
```bash
# 你运行这个命令实现功能
/execute-llama-prp PRPs/generated-file.md
```

## 📚 关键文件说明

- **CLAUDE.md** - llama.cpp专用开发规则，必须遵循
- **PRPs/INITIAL.md** - 用户会在这里描述功能需求
- **PRPs/templates/prp_llama_cpp_base.md** - PRP生成模板
- **context_engineering_examples/** - 基于真实llama.cpp代码的示例
- **CONTEXT_ENGINEERING.md** - 完整使用指南

## 🧠 Context Engineering是什么

这是一个比提示工程更强大的方法：
- **提示工程**: 像便利贴，"帮我写个函数"
- **Context Engineering**: 像完整剧本，包含所有上下文、示例、验证

### 优势
- 10倍开发效率提升
- 基于真实llama.cpp代码模式
- 内置验证循环确保代码质量
- 专门针对C++/CUDA高性能计算优化

## ⚡ 如果用户问"这是什么"

告诉他们：
> 这是一个专门为llama.cpp开发优化的Context Engineering环境。你可以通过编辑 `PRPs/INITIAL.md` 来描述想要实现的功能，然后使用 `/generate-llama-prp` 和 `/execute-llama-prp` 命令来高效地实现llama.cpp相关功能。

## 🔧 开发模式

所有代码都遵循真实llama.cpp模式：
- **命名约定**: snake_case
- **错误处理**: LLAMA_LOG_* 宏
- **内存管理**: RAII + 智能指针
- **性能优化**: SIMD + CUDA
- **测试模式**: 基于tests/目录的模式

---

**Claude，现在你知道如何使用这个环境了！用户随时可能要求你实现llama.cpp相关功能。**