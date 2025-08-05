#!/usr/bin/env python3
"""
一键部署 llama.cpp Context Engineering 模板
用法: python deploy_to_llama_cpp.py /path/to/your/llama.cpp
"""

import os
import shutil
import sys
import argparse
from pathlib import Path

def deploy_to_llama_cpp(target_dir: str):
    """一键部署Context Engineering模板到任何llama.cpp项目"""
    
    template_dir = Path(__file__).parent
    target_path = Path(target_dir)
    
    # 验证目标目录
    if not target_path.exists():
        print(f"❌ 目标目录不存在: {target_dir}")
        return False
    
    print(f"🚀 部署llama.cpp Context Engineering模板到: {target_path}")
    print(f"📦 从模板目录: {template_dir}")
    
    try:
        # 1. 部署CLAUDE.md (强制覆盖)
        claude_src = template_dir / "CLAUDE.md"
        claude_dst = target_path / "CLAUDE.md"
        if claude_src.exists():
            shutil.copy2(claude_src, claude_dst)
            print(f"   ✅ CLAUDE.md 已部署")
        
        # 2. 部署.claude目录 (强制覆盖)
        claude_dir_src = template_dir / ".claude"
        claude_dir_dst = target_path / ".claude"
        if claude_dir_src.exists():
            if claude_dir_dst.exists():
                shutil.rmtree(claude_dir_dst)
            shutil.copytree(claude_dir_src, claude_dir_dst)
            print(f"   ✅ .claude/ 命令目录已部署")
        
        # 3. 部署PRPs目录
        prp_src = template_dir / "PRPs"
        prp_dst = target_path / "PRPs"
        if prp_src.exists():
            if prp_dst.exists():
                # 只覆盖模板，保留用户的INITIAL.md
                templates_src = prp_src / "templates"
                templates_dst = prp_dst / "templates"
                if templates_dst.exists():
                    shutil.rmtree(templates_dst)
                shutil.copytree(templates_src, templates_dst)
                
                # 如果没有INITIAL.md才复制
                initial_dst = prp_dst / "INITIAL.md"
                if not initial_dst.exists():
                    initial_src = prp_src / "INITIAL.md"
                    shutil.copy2(initial_src, initial_dst)
            else:
                shutil.copytree(prp_src, prp_dst)
            print(f"   ✅ PRPs/ 模板已部署")
        
        # 4. 部署示例代码
        examples_src = template_dir / "examples"
        examples_dst = target_path / "context_engineering_examples"
        if examples_src.exists():
            if examples_dst.exists():
                shutil.rmtree(examples_dst)
            shutil.copytree(examples_src, examples_dst)
            print(f"   ✅ 示例代码已部署到 context_engineering_examples/")
        
        # 5. 部署文档
        docs_src = template_dir / "documentation"
        docs_dst = target_path / "context_engineering_docs"
        if docs_src.exists():
            if docs_dst.exists():
                shutil.rmtree(docs_dst)
            shutil.copytree(docs_src, docs_dst)
            print(f"   ✅ 开发文档已部署到 context_engineering_docs/")
        
        # 6. 部署使用指南
        readme_src = template_dir / "README.md"
        readme_dst = target_path / "CONTEXT_ENGINEERING.md"
        if readme_src.exists():
            shutil.copy2(readme_src, readme_dst)
            print(f"   ✅ 使用指南已部署为 CONTEXT_ENGINEERING.md")
        
        # 7. 创建快速启动指南
        quick_start_content = f"""# 🚀 llama.cpp Context Engineering - 已就绪

**这个llama.cpp项目已配置Context Engineering环境**

## ⚡ 立即开始使用

### 3步工作流程:
1. **编辑需求**: `vi PRPs/INITIAL.md`
2. **生成计划**: `/generate-llama-prp PRPs/INITIAL.md`  
3. **执行实现**: `/execute-llama-prp PRPs/your-prp.md`

### 📚 资源位置:
- **开发规则**: CLAUDE.md
- **代码示例**: context_engineering_examples/
- **开发文档**: context_engineering_docs/
- **完整指南**: CONTEXT_ENGINEERING.md

部署时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
部署到: {target_path.absolute()}
"""
        
        quick_start_dst = target_path / "QUICK_START.md"
        with open(quick_start_dst, 'w', encoding='utf-8') as f:
            f.write(quick_start_content)
        print(f"   ✅ 快速启动指南已创建")
        
        print("\n" + "="*60)
        print("🎉 部署完成！llama.cpp Context Engineering环境已就绪")
        print("="*60)
        print(f"📁 目标目录: {target_path.absolute()}")
        print(f"🤖 在目录下打开Claude Code即可使用专用命令")
        print(f"📝 开始方法: 编辑 PRPs/INITIAL.md 描述功能需求")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"❌ 部署失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="一键部署llama.cpp Context Engineering模板",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python deploy_to_llama_cpp.py /root/llama.cpp-main
  python deploy_to_llama_cpp.py /root/llama.cpp-quantization
  python deploy_to_llama_cpp.py /root/llama.cpp-cuda-optimization
        """
    )
    
    parser.add_argument(
        "target_dir",
        help="目标llama.cpp项目目录"
    )
    
    args = parser.parse_args()
    
    if not deploy_to_llama_cpp(args.target_dir):
        sys.exit(1)

if __name__ == "__main__":
    main()