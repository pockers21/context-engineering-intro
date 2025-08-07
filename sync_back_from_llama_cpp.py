#!/usr/bin/env python3
"""
从llama.cpp项目同步分析结果回context-engineering-intro
用法: python sync_back_from_llama_cpp.py /path/to/llama.cpp
"""

import os
import shutil
import sys
import argparse
from pathlib import Path
from datetime import datetime

def sync_back_from_llama_cpp(llama_cpp_dir: str):
    """从llama.cpp项目同步分析结果回context-engineering-intro"""
    
    template_dir = Path(__file__).parent
    source_path = Path(llama_cpp_dir)
    
    # 验证源目录
    if not source_path.exists():
        print(f"❌ 源目录不存在: {llama_cpp_dir}")
        return False
    
    print(f"🔄 从llama.cpp项目同步分析结果: {source_path}")
    print(f"📦 同步到模板目录: {template_dir}")
    
    try:
        # 1. 同步PRPs分析结果 (保留原有模板)
        prps_src = source_path / "PRPs"
        prps_dst = template_dir / "PRPs"
        
        if prps_src.exists():
            # 创建备份目录
            backup_dir = template_dir / f"backups/PRPs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_dir.parent.mkdir(exist_ok=True)
            
            # 备份现有PRPs
            if prps_dst.exists():
                shutil.copytree(prps_dst, backup_dir)
                print(f"   📋 已备份现有PRPs到: {backup_dir.relative_to(template_dir)}")
            
            # 同步分析结果但保留模板
            analysis_items = [
                "paper_analysis",  # 完整的论文分析目录
                "INITIAL.md",      # 如果有更新
                "*.md"             # 其他分析文档
            ]
            
            for item in analysis_items:
                if item == "paper_analysis":
                    # 完整同步paper_analysis目录
                    pa_src = prps_src / "paper_analysis"
                    pa_dst = prps_dst / "paper_analysis"
                    if pa_src.exists():
                        if pa_dst.exists():
                            shutil.rmtree(pa_dst)
                        shutil.copytree(pa_src, pa_dst)
                        print(f"   ✅ paper_analysis/ 已同步")
                
                elif item == "INITIAL.md":
                    # 同步INITIAL.md如果有更新
                    initial_src = prps_src / "INITIAL.md"
                    initial_dst = prps_dst / "INITIAL.md"
                    if initial_src.exists():
                        # 检查文件时间戳，只有更新时才同步
                        if (not initial_dst.exists() or 
                            initial_src.stat().st_mtime > initial_dst.stat().st_mtime):
                            shutil.copy2(initial_src, initial_dst)
                            print(f"   ✅ INITIAL.md 已更新")
        
        # 2. 同步其他可能的工作成果
        work_items = [
            "CONTEXT_ENGINEERING.md",
            "QUICK_START.md", 
            "context_engineering_docs",
            "context_engineering_examples"
        ]
        
        for item in work_items:
            item_src = source_path / item
            item_dst = template_dir / item
            
            if item_src.exists():
                # 检查是否有更新
                if (not item_dst.exists() or 
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
                    print(f"   ✅ {item} 已更新")
        
        # 3. 创建同步日志
        sync_log_path = template_dir / "sync_log.md"
        log_content = f"""# 同步日志

## 最近同步记录

### {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **源目录**: {source_path.absolute()}
- **目标目录**: {template_dir.absolute()}
- **同步内容**: 
  - paper_analysis/ (完整论文分析结果)
  - 工作文档更新
  - 示例代码更新

### 同步说明
此日志记录从llama.cpp项目向context-engineering-intro的反向同步。
确保所有分析工作都能保存到GitHub。

---
*自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        with open(sync_log_path, 'w', encoding='utf-8') as f:
            f.write(log_content)
        
        print("\n" + "="*60)
        print("🎉 反向同步完成！分析结果已保存到context-engineering-intro")
        print("="*60)
        print(f"📁 目标目录: {template_dir.absolute()}")
        print(f"📊 论文分析: PRPs/paper_analysis/")
        print(f"🔄 现在可以commit并push到GitHub")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"❌ 同步失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="从llama.cpp项目反向同步分析结果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python sync_back_from_llama_cpp.py /root/llama.cpp-clip
  python sync_back_from_llama_cpp.py /root/llama.cpp-main
        """
    )
    
    parser.add_argument(
        "llama_cpp_dir",
        help="llama.cpp项目目录（包含分析结果）"
    )
    
    args = parser.parse_args()
    
    if not sync_back_from_llama_cpp(args.llama_cpp_dir):
        sys.exit(1)

if __name__ == "__main__":
    main()