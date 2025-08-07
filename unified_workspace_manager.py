#!/usr/bin/env python3
"""
统一工作空间管理器 - 消除复杂的同步流程
创建符号链接架构，让Claude Code在任何地方工作都能直接修改源文件
"""

import os
import sys
import shutil
from pathlib import Path
import subprocess

class UnifiedWorkspaceManager:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.projects_dir = self.base_dir / "projects"
        
    def setup_unified_workspace(self):
        """设置统一工作空间"""
        print("🏗️  设置统一Context Engineering工作空间")
        print("🎯  目标: 消除文件来回同步，让所有工作直接在GitHub仓库中进行")
        
        # 1. 创建projects目录
        self.projects_dir.mkdir(exist_ok=True)
        print(f"📁 创建项目目录: {self.projects_dir}")
        
        # 2. 检查现有llama.cpp项目
        existing_projects = self._find_existing_projects()
        
        if existing_projects:
            print(f"\n📋 发现现有llama.cpp项目:")
            for proj in existing_projects:
                print(f"   • {proj}")
            
            choice = input(f"\n❓ 要迁移到统一工作空间吗？这将:\n"
                          f"   1. 移动项目到 projects/ 目录\n"
                          f"   2. 创建符号链接保持兼容性\n"
                          f"   3. 所有修改直接在Git仓库中\n"
                          f"   (y/N): ").lower().strip()
            
            if choice == 'y':
                self._migrate_projects(existing_projects)
                
        self._create_workspace_tools()
        self._update_deployment_strategy()
        
        print(f"\n🎉 统一工作空间设置完成!")
        print(f"📁 所有项目现在都在: {self.projects_dir}")
        print(f"🔄 无需同步文件 - 直接工作即可")
        
    def _find_existing_projects(self):
        """查找现有的llama.cpp项目"""
        potential_locations = [
            Path("/root/llama.cpp-clip"),
            Path("/root/llama.cpp-main"),
            Path("/root/llama.cpp-debug")
        ]
        
        existing = []
        for loc in potential_locations:
            if loc.exists() and (loc / "src" / "llama.cpp").exists():
                existing.append(loc)
        
        return existing
    
    def _migrate_projects(self, projects):
        """迁移现有项目到统一工作空间"""
        print(f"\n🚚 迁移项目到统一工作空间...")
        
        for proj_path in projects:
            target_path = self.projects_dir / proj_path.name
            
            if target_path.exists():
                print(f"   ⚠️  {target_path} 已存在，跳过")
                continue
                
            print(f"   📦 迁移: {proj_path.name}")
            
            try:
                # 移动项目到统一目录
                shutil.move(str(proj_path), str(target_path))
                
                # 创建符号链接保持向后兼容
                os.symlink(str(target_path), str(proj_path))
                
                print(f"   ✅ {proj_path.name} → {target_path}")
                print(f"   🔗 符号链接: {proj_path} → {target_path}")
                
            except Exception as e:
                print(f"   ❌ 迁移失败: {e}")
    
    def _create_workspace_tools(self):
        """创建工作空间管理工具"""
        
        # 1. 创建项目切换器
        switcher_path = self.base_dir / "switch_project.py"
        switcher_content = '''#!/usr/bin/env python3
"""
项目切换器 - 直接在项目目录中启动Claude Code
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    base_dir = Path(__file__).parent
    projects_dir = base_dir / "projects"
    
    if len(sys.argv) < 2:
        print("📋 可用项目:")
        if projects_dir.exists():
            for proj in projects_dir.iterdir():
                if proj.is_dir() and not proj.name.startswith("."):
                    print(f"   • {proj.name}")
        else:
            print("   ⚠️  没有找到项目目录")
        
        print("\\n🚀 用法:")
        print("   python switch_project.py <project_name>")
        print("   python switch_project.py llama.cpp-clip")
        return
    
    project_name = sys.argv[1]
    project_path = projects_dir / project_name
    
    if not project_path.exists():
        print(f"❌ 项目不存在: {project_name}")
        return
    
    print(f"🚀 切换到项目: {project_name}")
    print(f"📁 项目路径: {project_path}")
    print(f"💡 在此目录中启动Claude Code，所有修改直接保存到Git仓库")
    
    # 切换到项目目录
    os.chdir(project_path)
    
    # 启动shell
    subprocess.run(["/bin/bash"], cwd=project_path)

if __name__ == "__main__":
    main()
'''
        
        switcher_path.write_text(switcher_content)
        switcher_path.chmod(0o755)
        print(f"✅ 项目切换器: {switcher_path}")
        
        # 2. 创建统一部署脚本
        deploy_all_path = self.base_dir / "deploy_all_projects.py"
        deploy_all_content = '''#!/usr/bin/env python3
"""
统一部署脚本 - 将Context Engineering模板部署到所有项目
"""

import os
import sys
from pathlib import Path
import subprocess

def main():
    base_dir = Path(__file__).parent
    projects_dir = base_dir / "projects"
    deploy_script = base_dir / "deploy_to_llama_cpp.py"
    
    if not deploy_script.exists():
        print("❌ 找不到部署脚本")
        return
    
    if not projects_dir.exists():
        print("❌ 没有找到projects目录")
        return
    
    projects = [p for p in projects_dir.iterdir() 
                if p.is_dir() and not p.name.startswith(".")]
    
    if not projects:
        print("❌ 没有找到任何项目")
        return
    
    print(f"🚀 部署到所有项目:")
    for proj in projects:
        print(f"   📦 {proj.name}")
    
    for proj in projects:
        print(f"\\n🔄 部署到: {proj.name}")
        result = subprocess.run([
            sys.executable, str(deploy_script), str(proj)
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"   ✅ {proj.name} 部署成功")
        else:
            print(f"   ❌ {proj.name} 部署失败: {result.stderr}")

if __name__ == "__main__":
    main()
'''
        
        deploy_all_path.write_text(deploy_all_content)
        deploy_all_path.chmod(0o755)
        print(f"✅ 统一部署脚本: {deploy_all_path}")
    
    def _update_deployment_strategy(self):
        """更新部署策略说明"""
        strategy_path = self.base_dir / "UNIFIED_WORKFLOW.md"
        strategy_content = f"""# 统一Context Engineering工作流程

## 🎯 核心理念

**消除文件来回同步，让所有工作直接在GitHub仓库中进行**

## 📁 新的目录结构

```
{self.base_dir.name}/                          (Git仓库根目录)
├── projects/                           (所有llama.cpp项目)
│   ├── llama.cpp-clip/                (项目1 - 直接在Git中)
│   ├── llama.cpp-main/                (项目2 - 直接在Git中)
│   └── llama.cpp-quantization/        (项目3 - 直接在Git中)
├── PRPs/                              (分析结果直接保存这里)
├── switch_project.py                  (项目切换器)
├── deploy_all_projects.py             (统一部署)
└── sync_back_from_llama_cpp.py        (不再需要!)
```

## 🚀 新工作流程

### 1. 初始设置 (仅需一次)
```bash
python unified_workspace_manager.py
```

### 2. 日常工作流程
```bash
# 切换到项目
python switch_project.py llama.cpp-clip

# 在项目目录中启动Claude Code
# 所有修改直接保存在Git仓库中
```

### 3. 提交工作成果
```bash
git add .
git commit -m "完成XXX功能"
git push
```

## ✅ 优势

1. **🔄 无需同步**: 所有工作直接在Git仓库中
2. **📁 统一管理**: 多个llama.cpp项目在同一个仓库
3. **🎯 简化流程**: 一次commit覆盖所有工作
4. **🔍 完整历史**: 项目演进过程完整记录
5. **⚡ 提高效率**: 消除复杂的文件拷贝步骤

## 🆚 对比旧流程

### 旧流程 (复杂)
母模板 → CC生成子模板 → 部署 → 工作 → 手动同步 → 提交

### 新流程 (简单)
直接工作 → 提交

## 💡 最佳实践

- 使用`switch_project.py`切换项目
- Claude Code工作目录直接设为项目目录
- 所有PRPs分析直接保存到仓库根目录
- 定期commit保存进度
"""
        
        strategy_path.write_text(strategy_content)
        print(f"✅ 工作流程说明: {strategy_path}")

def main():
    manager = UnifiedWorkspaceManager()
    manager.setup_unified_workspace()

if __name__ == "__main__":
    main()