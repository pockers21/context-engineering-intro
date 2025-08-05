#!/usr/bin/env python3
"""
llama.cpp Context Engineering 模板部署脚本

将模板文件复制到目标llama.cpp项目中，并提供交互式配置选项。
"""

import os
import shutil
import sys
import argparse
from pathlib import Path
from typing import List, Optional

class LlamaCppTemplateDeployer:
    def __init__(self, template_dir: str, target_dir: str):
        self.template_dir = Path(template_dir)
        self.target_dir = Path(target_dir)
        
        # 检查模板目录
        if not self.template_dir.exists():
            raise FileNotFoundError(f"模板目录不存在: {template_dir}")
            
        # 检查目标目录
        if not self.target_dir.exists():
            raise FileNotFoundError(f"目标目录不存在: {target_dir}")
            
        # 验证目标是llama.cpp项目
        if not self._is_llama_cpp_project():
            print(f"警告: {target_dir} 不像是llama.cpp项目")
            response = input("是否继续? (y/N): ")
            if response.lower() != 'y':
                sys.exit(1)
    
    def _is_llama_cpp_project(self) -> bool:
        """检查目标目录是否是llama.cpp项目"""
        indicators = [
            "CMakeLists.txt",
            "src/llama.cpp", 
            "ggml",
            "examples",
            "Makefile"
        ]
        
        found = 0
        for indicator in indicators:
            if (self.target_dir / indicator).exists():
                found += 1
        
        return found >= 3  # 至少找到3个指标文件
    
    def deploy(self, components: Optional[List[str]] = None, force: bool = False):
        """部署模板组件"""
        if components is None:
            components = ['all']
        
        print(f"🚀 将llama.cpp Context Engineering模板部署到: {self.target_dir}")
        
        if 'all' in components:
            self._deploy_all(force)
        else:
            for component in components:
                self._deploy_component(component, force)
        
        print("\n✅ 部署完成！")
        self._print_next_steps()
    
    def _deploy_all(self, force: bool):
        """部署所有组件"""
        components = {
            'claude_config': '全局开发规则和Claude配置',
            'prp_templates': 'PRP模板和命令',
            'examples': '代码示例库',
            'documentation': '开发文档'
        }
        
        for component, description in components.items():
            print(f"\n📦 部署: {description}")
            self._deploy_component(component, force)
    
    def _deploy_component(self, component: str, force: bool):
        """部署特定组件"""
        if component == 'claude_config':
            self._copy_claude_config(force)
        elif component == 'prp_templates':
            self._copy_prp_templates(force)
        elif component == 'examples':
            self._copy_examples(force)
        elif component == 'documentation':
            self._copy_documentation(force)
        else:
            print(f"❌ 未知组件: {component}")
    
    def _copy_claude_config(self, force: bool):
        """复制Claude配置文件"""
        # 复制 CLAUDE.md
        src_claude = self.template_dir / "CLAUDE.md"
        dst_claude = self.target_dir / "CLAUDE.md"
        self._copy_file(src_claude, dst_claude, force)
        
        # 复制 .claude 目录
        src_claude_dir = self.template_dir / ".claude"
        dst_claude_dir = self.target_dir / ".claude"
        self._copy_directory(src_claude_dir, dst_claude_dir, force)
        
        print("   ✓ Claude配置和命令已复制")
    
    def _copy_prp_templates(self, force: bool):
        """复制PRP模板"""
        src_prp = self.template_dir / "PRPs"
        dst_prp = self.target_dir / "PRPs"
        self._copy_directory(src_prp, dst_prp, force)
        print("   ✓ PRP模板已复制")
    
    def _copy_examples(self, force: bool):
        """复制示例代码"""
        src_examples = self.template_dir / "examples"
        dst_examples = self.target_dir / "context_engineering_examples"
        
        # 避免与llama.cpp原有的examples目录冲突
        self._copy_directory(src_examples, dst_examples, force)
        print("   ✓ 示例代码已复制到 context_engineering_examples/")
    
    def _copy_documentation(self, force: bool):
        """复制文档"""
        src_docs = self.template_dir / "documentation"
        if src_docs.exists():
            dst_docs = self.target_dir / "context_engineering_docs"
            self._copy_directory(src_docs, dst_docs, force)
            print("   ✓ 文档已复制")
        
        # 复制README
        src_readme = self.template_dir / "README.md"
        dst_readme = self.target_dir / "CONTEXT_ENGINEERING.md"
        self._copy_file(src_readme, dst_readme, force)
        print("   ✓ 使用指南已复制为 CONTEXT_ENGINEERING.md")
    
    def _copy_file(self, src: Path, dst: Path, force: bool):
        """复制单个文件"""
        if dst.exists() and not force:
            print(f"     ⚠️  文件已存在，跳过: {dst.name}")
            return
        
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"     📄 {src.name} → {dst}")
    
    def _copy_directory(self, src: Path, dst: Path, force: bool):
        """复制目录"""
        if dst.exists():
            if force:
                shutil.rmtree(dst)
            else:
                print(f"     ⚠️  目录已存在，跳过: {dst.name}")
                return
        
        shutil.copytree(src, dst)
        print(f"     📁 {src.name}/ → {dst}/")
    
    def _print_next_steps(self):
        """打印后续步骤"""
        print("\n" + "="*60)
        print("🎉 llama.cpp Context Engineering 模板已成功部署！")
        print("="*60)
        print()
        print("📋 接下来的步骤:")
        print()
        print("1. 📝 编辑功能需求:")
        print(f"   vi {self.target_dir}/PRPs/INITIAL.md")
        print()
        print("2. 🤖 生成实现计划 (在Claude Code中):")
        print("   /generate-llama-prp PRPs/INITIAL.md")
        print()
        print("3. ⚡执行实现:")
        print("   /execute-llama-prp PRPs/your-generated-prp.md")
        print()
        print("4. 📚 参考资源:")
        print("   - 开发规则: CLAUDE.md")
        print("   - 代码示例: context_engineering_examples/")
        print("   - 使用指南: CONTEXT_ENGINEERING.md")
        print()
        print("🚀 开始你的高效llama.cpp开发之旅吧！")

def main():
    parser = argparse.ArgumentParser(
        description="部署llama.cpp Context Engineering模板",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python copy_template.py /path/to/llama.cpp                    # 部署全部组件
  python copy_template.py /path/to/llama.cpp --components prp_templates claude_config
  python copy_template.py /path/to/llama.cpp --force            # 强制覆盖现有文件
        """
    )
    
    parser.add_argument(
        "target_dir",
        help="目标llama.cpp项目目录"
    )
    
    parser.add_argument(
        "--components",
        nargs="+",
        choices=["claude_config", "prp_templates", "examples", "documentation", "all"],
        default=["all"],
        help="要部署的组件 (默认: all)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制覆盖现有文件"
    )
    
    parser.add_argument(
        "--template-dir",
        default=".",
        help="模板目录路径 (默认: 当前目录)"
    )
    
    args = parser.parse_args()
    
    try:
        deployer = LlamaCppTemplateDeployer(args.template_dir, args.target_dir)
        deployer.deploy(args.components, args.force)
    except Exception as e:
        print(f"❌ 部署失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()