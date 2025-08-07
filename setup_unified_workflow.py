#!/usr/bin/env python3
"""
设置统一的Context Engineering工作流程
将所有llama.cpp项目作为子项目管理，避免来回同步
"""

import os
import shutil
import sys
from pathlib import Path
import subprocess

def setup_unified_workflow():
    """设置统一工作流程"""
    
    base_dir = Path(__file__).parent
    projects_dir = base_dir / "projects"
    
    print("🏗️  设置统一Context Engineering工作流程")
    print(f"📁 基础目录: {base_dir}")
    
    # 1. 创建项目管理结构
    projects_dir.mkdir(exist_ok=True)
    
    # 2. 检查现有的llama.cpp项目
    existing_projects = []
    
    llama_cpp_locations = [
        "/root/llama.cpp-clip",
        "/root/llama.cpp-main", 
        "/root/llama.cpp-debug"
    ]
    
    for loc in llama_cpp_locations:
        if Path(loc).exists():
            existing_projects.append(Path(loc))
    
    if existing_projects:
        print(f"\n📋 发现现有llama.cpp项目:")
        for i, proj in enumerate(existing_projects):
            print(f"   {i+1}. {proj}")
        
        print(f"\n🤖 建议的统一结构:")
        for proj in existing_projects:
            target = projects_dir / proj.name
            print(f"   {proj} → {target}")
        
        print(f"\n📝 这样的好处:")
        print(f"   • 所有项目在同一个Git仓库中")
        print(f"   • 无需来回同步文件") 
        print(f"   • 统一的版本控制")
        print(f"   • 简化的工作流程")
        
        choice = input(f"\n❓ 是否要迁移现有项目到统一结构? (y/N): ").lower().strip()
        
        if choice == 'y':
            return migrate_existing_projects(existing_projects, projects_dir)
    
    # 3. 创建新项目模板
    return setup_new_project_template(projects_dir)

def migrate_existing_projects(existing_projects, projects_dir):
    """迁移现有项目"""
    
    print(f"\n🚚 开始迁移现有项目...")
    
    for proj in existing_projects:
        target = projects_dir / proj.name
        
        if target.exists():
            print(f"   ⚠️  {target} 已存在，跳过")
            continue
        
        print(f"   📦 迁移: {proj.name}")
        try:
            # 移动整个项目
            shutil.move(str(proj), str(target))
            
            # 创建符号链接保持向后兼容
            os.symlink(str(target), str(proj))
            print(f"   ✅ {proj.name} 已迁移，创建符号链接保持兼容")
            
        except Exception as e:
            print(f"   ❌ 迁移失败: {e}")
    
    # 更新工作流程脚本
    create_unified_workflow_scripts(projects_dir.parent)
    
    return True

def setup_new_project_template(projects_dir):
    """为新项目设置模板"""
    
    template_dir = projects_dir / ".template"
    template_dir.mkdir(exist_ok=True)
    
    # 创建项目模板
    template_readme = template_dir / "README.md"
    template_readme.write_text("""# llama.cpp项目模板

这是在context-engineering-intro中管理的llama.cpp项目模板。

## 使用方法

1. 复制此模板创建新项目
2. 使用统一的Context Engineering工具
3. 所有工作自动同步到GitHub

## 优势

- 无需文件来回同步
- 统一版本控制
- 简化工作流程
""")
    
    create_unified_workflow_scripts(projects_dir.parent)
    
    print(f"✅ 新项目模板已创建")
    return True

def create_unified_workflow_scripts(base_dir):
    """创建统一工作流程脚本"""
    
    # 1. 项目管理脚本
    project_manager = base_dir / "manage_projects.py"
    project_manager.write_text(f'''#!/usr/bin/env python3
"""
统一项目管理器
"""

import os
import sys
from pathlib import Path
import subprocess

def main():
    """主函数"""
    base_dir = Path(__file__).parent
    projects_dir = base_dir / "projects"
    
    if len(sys.argv) < 2:
        print("📋 可用项目:")
        for proj in projects_dir.glob("*"):
            if proj.is_dir() and not proj.name.startswith("."):
                print(f"   • {{proj.name}}")
        
        print("\\n🚀 用法:")
        print("   python manage_projects.py <project_name> [command]")
        print("   python manage_projects.py llama.cpp-clip build")
        return
    
    project_name = sys.argv[1]
    project_path = projects_dir / project_name
    
    if not project_path.exists():
        print(f"❌ 项目不存在: {{project_name}}")
        return
    
    # 切换到项目目录执行命令
    os.chdir(project_path)
    
    if len(sys.argv) > 2:
        command = sys.argv[2:]
        print(f"🏃 在 {{project_name}} 中执行: {{' '.join(command)}}")
        subprocess.run(command)
    else:
        print(f"📁 已切换到项目: {{project_path}}")
        subprocess.run(["bash"], cwd=project_path)

if __name__ == "__main__":
    main()
''')
    project_manager.chmod(0o755)
    
    # 2. 快速部署脚本
    quick_deploy = base_dir / "quick_deploy.py"
    quick_deploy.write_text(f'''#!/usr/bin/env python3
"""
快速部署Context Engineering到项目
"""

import sys
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        base_dir = Path(__file__).parent
        projects_dir = base_dir / "projects"
        
        print("📋 可用项目:")
        for proj in projects_dir.glob("*"):
            if proj.is_dir() and not proj.name.startswith("."):
                print(f"   • {{proj.name}}")
        
        print("\\n🚀 用法:")
        print("   python quick_deploy.py <project_name>")
        return
    
    project_name = sys.argv[1]
    base_dir = Path(__file__).parent
    target_dir = base_dir / "projects" / project_name
    
    if not target_dir.exists():
        print(f"❌ 项目不存在: {{project_name}}")
        return
    
    # 使用现有的deploy脚本
    deploy_script = base_dir / "deploy_to_llama_cpp.py"
    if deploy_script.exists():
        import subprocess
        subprocess.run([sys.executable, str(deploy_script), str(target_dir)])
    else:
        print("❌ 找不到部署脚本")

if __name__ == "__main__":
    main()
''')
    quick_deploy.chmod(0o755)
    
    print("✅ 统一工作流程脚本已创建")

if __name__ == "__main__":
    setup_unified_workflow()