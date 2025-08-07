#!/usr/bin/env python3
"""
智能工作空间管理器 - 保持项目分离，但简化工作流程
"""

import os
import sys
import shutil
from pathlib import Path
import subprocess
import json

class SmartWorkspaceManager:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.config_file = self.base_dir / ".workspace_config.json"
        
    def setup_smart_workspace(self):
        """设置智能工作空间 - 保持项目分离但简化流程"""
        print("🎯 设置智能Context Engineering工作空间")
        print("💡 理念：项目分离 + 流程简化 + 自动同步")
        
        # 1. 发现现有llama.cpp项目
        existing_projects = self._find_llama_projects()
        
        if not existing_projects:
            print("❌ 没有找到llama.cpp项目")
            print("请先确保有llama.cpp项目在 /root/llama.cpp-* 目录中")
            return False
            
        print(f"\n📋 发现llama.cpp项目:")
        for proj in existing_projects:
            print(f"   • {proj}")
        
        # 2. 创建工作空间配置
        config = {
            "projects": {proj.name: str(proj) for proj in existing_projects},
            "template_dir": str(self.base_dir),
            "auto_sync": True,
            "git_auto_commit": False  # 用户手动控制Git提交
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ 工作空间配置保存到: {self.config_file}")
        
        # 3. 创建智能工作工具
        self._create_smart_tools()
        
        print(f"\n🎉 智能工作空间设置完成!")
        print(f"📁 项目保持独立：不会污染Git仓库")
        print(f"🔄 自动同步：工作结果自动回流")
        print(f"⚡ 简化流程：一键切换和部署")
        
        return True
    
    def _find_llama_projects(self):
        """查找llama.cpp项目"""
        search_paths = [Path("/root").glob("llama.cpp*"), Path("/root").glob("llama-*")]
        projects = []
        
        for pattern in search_paths:
            for path in pattern:
                if path.is_dir() and (path / "src" / "llama.cpp").exists():
                    projects.append(path)
        
        return projects
    
    def _create_smart_tools(self):
        """创建智能工作工具"""
        
        # 1. 智能项目切换器
        self._create_project_switcher()
        
        # 2. 自动同步工具
        self._create_auto_sync_tool()
        
        # 3. 一键工作流工具
        self._create_workflow_tool()
        
        # 4. 更新现有的部署脚本，使其支持自动同步
        self._enhance_deploy_script()
    
    def _create_project_switcher(self):
        """创建智能项目切换器"""
        switcher_path = self.base_dir / "work_on_project.py"
        switcher_content = '''#!/usr/bin/env python3
"""
智能项目工作器 - 切换项目 + 自动部署 + 自动同步
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def main():
    base_dir = Path(__file__).parent
    config_file = base_dir / ".workspace_config.json"
    
    if not config_file.exists():
        print("❌ 工作空间未配置，请先运行: python smart_workspace_manager.py")
        return
        
    with open(config_file) as f:
        config = json.load(f)
    
    projects = config.get("projects", {})
    
    if len(sys.argv) < 2:
        print("📋 可用项目:")
        for name in projects.keys():
            print(f"   • {name}")
        
        print("\\n🚀 用法:")
        print("   python work_on_project.py <project_name>")
        print("\\n💡 这会自动:")
        print("   1. 部署Context Engineering模板到项目")
        print("   2. 提示你在项目目录启动Claude Code")
        print("   3. 设置自动同步（工作完成后运行sync命令）")
        return
    
    project_name = sys.argv[1]
    
    if project_name not in projects:
        print(f"❌ 项目不存在: {project_name}")
        print(f"可用项目: {list(projects.keys())}")
        return
    
    project_path = Path(projects[project_name])
    
    if not project_path.exists():
        print(f"❌ 项目目录不存在: {project_path}")
        return
    
    print(f"🎯 准备在项目中工作: {project_name}")
    print(f"📁 项目路径: {project_path}")
    
    # 1. 自动部署Context Engineering模板
    print(f"\\n🚀 部署Context Engineering模板...")
    deploy_script = base_dir / "deploy_to_llama_cpp.py"
    if deploy_script.exists():
        result = subprocess.run([
            sys.executable, str(deploy_script), str(project_path)
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 模板部署成功")
        else:
            print(f"⚠️  部署警告: {result.stderr}")
    
    # 2. 提供工作指引
    print(f"\\n" + "="*60)
    print(f"🎉 项目 {project_name} 已准备就绪!")
    print(f"="*60)
    print(f"📁 项目位置: {project_path}")
    print(f"\\n📋 接下来的步骤:")
    print(f"   1. 在新终端中切换到项目目录:")
    print(f"      cd {project_path}")
    print(f"   2. 启动Claude Code (在项目目录中)")
    print(f"   3. 开始工作 (编辑PRPs, 实现功能等)")
    print(f"   4. 工作完成后，运行同步命令:")
    print(f"      python {base_dir}/sync_results.py {project_name}")
    print(f"="*60)
    
    # 3. 可选：直接打开项目目录
    choice = input("\\n❓ 要打开项目目录吗? (y/N): ").lower().strip()
    if choice == 'y':
        os.chdir(project_path)
        print(f"📂 已切换到: {project_path}")
        subprocess.run(["/bin/bash"], cwd=project_path)

if __name__ == "__main__":
    main()
'''
        
        switcher_path.write_text(switcher_content)
        switcher_path.chmod(0o755)
        print(f"✅ 智能项目切换器: {switcher_path}")
    
    def _create_auto_sync_tool(self):
        """创建自动同步工具"""
        sync_path = self.base_dir / "sync_results.py"
        sync_content = '''#!/usr/bin/env python3
"""
智能结果同步器 - 从项目同步工作结果到模板仓库
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

def main():
    base_dir = Path(__file__).parent
    config_file = base_dir / ".workspace_config.json"
    
    if not config_file.exists():
        print("❌ 工作空间未配置")
        return
        
    with open(config_file) as f:
        config = json.load(f)
    
    projects = config.get("projects", {})
    
    if len(sys.argv) < 2:
        print("🔄 结果同步器")
        print("\\n📋 可同步的项目:")
        for name in projects.keys():
            print(f"   • {name}")
        
        print("\\n🚀 用法:")
        print("   python sync_results.py <project_name>")
        print("   python sync_results.py --all  # 同步所有项目")
        return
    
    if sys.argv[1] == "--all":
        for project_name in projects.keys():
            sync_project(project_name, projects[project_name], base_dir)
    else:
        project_name = sys.argv[1]
        if project_name not in projects:
            print(f"❌ 项目不存在: {project_name}")
            return
        
        sync_project(project_name, projects[project_name], base_dir)
    
    print(f"\\n🎉 同步完成! 现在可以提交到Git:")
    print(f"   git add .")
    print(f"   git commit -m \\"完成{sys.argv[1]}项目工作\\"")
    print(f"   git push")

def sync_project(project_name, project_path, base_dir):
    """同步单个项目的结果"""
    project_path = Path(project_path)
    
    print(f"\\n🔄 同步项目: {project_name}")
    print(f"📁 源路径: {project_path}")
    
    if not project_path.exists():
        print(f"   ❌ 项目路径不存在")
        return False
    
    # 同步PRPs目录
    prps_src = project_path / "PRPs"
    prps_dst = base_dir / "PRPs"
    
    if prps_src.exists():
        # 备份现有PRPs
        backup_dir = base_dir / f"backups/PRPs_{project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_dir.parent.mkdir(exist_ok=True)
        
        if prps_dst.exists():
            shutil.copytree(prps_dst, backup_dir)
            print(f"   📋 备份到: {backup_dir.relative_to(base_dir)}")
        
        # 智能合并PRPs内容
        merge_prps_content(prps_src, prps_dst)
        print(f"   ✅ PRPs内容已同步")
    
    # 同步其他工作成果
    sync_items = [
        ("CONTEXT_ENGINEERING.md", "项目文档"),
        ("context_engineering_docs", "开发文档"),
        ("context_engineering_examples", "示例代码")
    ]
    
    for item_name, description in sync_items:
        item_src = project_path / item_name
        item_dst = base_dir / item_name
        
        if item_src.exists() and (not item_dst.exists() or 
            item_src.stat().st_mtime > item_dst.stat().st_mtime):
            
            if item_dst.exists():
                if item_dst.is_dir():
                    shutil.rmtree(item_dst)
                else:
                    item_dst.unlink()
            
            if item_src.is_dir():
                shutil.copytree(item_src, item_dst)
            else:
                shutil.copy2(item_src, item_dst)
            
            print(f"   ✅ {description}已更新")
    
    return True

def merge_prps_content(src_dir, dst_dir):
    """智能合并PRPs内容"""
    dst_dir.mkdir(exist_ok=True)
    
    # 保留原有模板
    templates_dst = dst_dir / "templates"
    if not templates_dst.exists() and (src_dir / "templates").exists():
        shutil.copytree(src_dir / "templates", templates_dst)
    
    # 同步所有分析结果
    for item in src_dir.iterdir():
        if item.name == "templates":
            continue  # 跳过模板，保留原有的
            
        item_dst = dst_dir / item.name
        
        if item.is_dir():
            if item_dst.exists():
                shutil.rmtree(item_dst)
            shutil.copytree(item, item_dst)
        else:
            shutil.copy2(item, item_dst)

if __name__ == "__main__":
    main()
'''
        
        sync_path.write_text(sync_content)
        sync_path.chmod(0o755)
        print(f"✅ 智能同步工具: {sync_path}")
    
    def _create_workflow_tool(self):
        """创建一键工作流工具"""
        workflow_path = self.base_dir / "quick_workflow.py"
        workflow_content = '''#!/usr/bin/env python3
"""
一键工作流程 - 项目选择 → 部署 → 工作 → 同步 → 提交
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def main():
    base_dir = Path(__file__).parent
    
    print("🚀 Context Engineering 一键工作流程")
    print("=" * 50)
    
    # 显示可用项目
    config_file = base_dir / ".workspace_config.json"
    if not config_file.exists():
        print("❌ 请先运行: python smart_workspace_manager.py")
        return
        
    with open(config_file) as f:
        config = json.load(f)
    
    projects = config.get("projects", {})
    project_names = list(projects.keys())
    
    if not project_names:
        print("❌ 没有找到项目")
        return
    
    print("📋 可用项目:")
    for i, name in enumerate(project_names, 1):
        print(f"   {i}. {name}")
    
    # 让用户选择
    try:
        choice = int(input("\\n选择项目 (输入数字): ")) - 1
        if choice < 0 or choice >= len(project_names):
            print("❌ 无效选择")
            return
            
        project_name = project_names[choice]
        
    except (ValueError, KeyboardInterrupt):
        print("\\n👋 已取消")
        return
    
    print(f"\\n🎯 选择了项目: {project_name}")
    
    # 执行工作流程
    work_script = base_dir / "work_on_project.py"
    sync_script = base_dir / "sync_results.py"
    
    if work_script.exists():
        print("\\n🚀 启动项目工作...")
        subprocess.run([sys.executable, str(work_script), project_name])
        
        # 询问是否同步结果
        sync_choice = input("\\n❓ 工作完成了吗? 要同步结果吗? (y/N): ").lower().strip()
        
        if sync_choice == 'y':
            print("\\n🔄 同步工作结果...")
            subprocess.run([sys.executable, str(sync_script), project_name])
            
            # 询问是否提交到Git
            git_choice = input("\\n❓ 要提交到Git吗? (y/N): ").lower().strip()
            
            if git_choice == 'y':
                commit_msg = input("请输入提交信息 (回车使用默认): ").strip()
                if not commit_msg:
                    commit_msg = f"完成{project_name}项目工作"
                
                print("\\n📝 提交到Git...")
                subprocess.run(["git", "add", "."], cwd=base_dir)
                subprocess.run(["git", "commit", "-m", commit_msg], cwd=base_dir)
                
                push_choice = input("\\n❓ 要推送到GitHub吗? (y/N): ").lower().strip()
                if push_choice == 'y':
                    subprocess.run(["git", "push"], cwd=base_dir)
                    print("🎉 已推送到GitHub!")

if __name__ == "__main__":
    main()
'''
        
        workflow_path.write_text(workflow_content)
        workflow_path.chmod(0o755)
        print(f"✅ 一键工作流: {workflow_path}")
    
    def _enhance_deploy_script(self):
        """增强现有的部署脚本"""
        # 这里可以修改deploy_to_llama_cpp.py，使其支持更智能的部署
        # 但为了不破坏现有功能，我们保持原样
        print(f"✅ 部署脚本增强: 保持现有deploy_to_llama_cpp.py")

def main():
    manager = SmartWorkspaceManager()
    if manager.setup_smart_workspace():
        print(f"\n🎯 推荐的新工作流程:")
        print(f"   1. python work_on_project.py llama.cpp-clip")
        print(f"   2. 在项目中使用Claude Code工作")
        print(f"   3. python sync_results.py llama.cpp-clip")
        print(f"   4. git add . && git commit && git push")
        print(f"\n⚡ 或者使用一键流程:")
        print(f"   python quick_workflow.py")

if __name__ == "__main__":
    main()