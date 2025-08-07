#!/usr/bin/env python3
"""
快速论文分析工具 - 重点分析最关键的论文
"""

import subprocess
import re
import json
from datetime import datetime

def get_paper_info(url):
    """快速获取论文基础信息"""
    try:
        cmd = f'curl -s -m 10 "{url}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        content = result.stdout
        
        # 提取标题
        title_match = re.search(r'<title>(.*?)</title>', content)
        title = title_match.group(1).strip() if title_match else "未获取到标题"
        title = re.sub(r'\[.*?\]\s*', '', title)  # 移除 [2408.11743] 这样的前缀
        
        # 提取摘要
        abstract_match = re.search(r'<div class="ltx_abstract">.*?<p[^>]*>(.*?)</p>', content, re.DOTALL)
        abstract = ""
        if abstract_match:
            abstract = re.sub(r'<[^>]+>', '', abstract_match.group(1))
            abstract = re.sub(r'\s+', ' ', abstract).strip()
        
        # 查找性能数据
        performance = []
        text_for_perf = abstract.lower()
        
        # 查找加速比
        speedup_matches = re.findall(r'(\d+\.?\d*)\s*[×x]\s*(?:speedup|faster|improvement)', text_for_perf)
        performance.extend([f"{m}x加速" for m in speedup_matches])
        
        # 查找百分比
        percent_matches = re.findall(r'(\d+)%\s*(?:improvement|better|reduction|faster)', text_for_perf)
        performance.extend([f"{m}%改进" for m in percent_matches])
        
        return {
            "title": title,
            "abstract": abstract,
            "performance": performance,
            "url": url
        }
    except Exception as e:
        return {"title": f"获取失败: {e}", "abstract": "", "performance": [], "url": url}

def analyze_key_papers():
    """分析关键论文"""
    
    # 重点论文列表 (从paper.md中提取的关键链接)
    key_papers = [
        "https://arxiv.org/html/2409.16997v1",  # INT-FlashAttention
        "https://arxiv.org/html/2410.02367v1",  # SageAttention  
        "https://arxiv.org/html/2410.00161v2",  # KV-Compress
        "https://arxiv.org/abs/2408.11743",     # MARLIN
        "https://arxiv.org/html/2408.14690v1",  # Training-Free Activation Sparsity
        "https://arxiv.org/html/2410.09426v1",  # FlatQuant
        "https://arxiv.org/abs/2405.03917",     # KV Cache 1-bit
        "https://arxiv.org/html/2410.11305v1",  # QSpec
        "https://arxiv.org/html/2503.01840",    # EAGLE-3 (你提到的)
        "https://flashinfer.ai/2025/03/10/sampling.html"  # Sorting-Free
    ]
    
    print("🚀 快速分析关键论文")
    print("="*50)
    
    results = []
    
    for i, url in enumerate(key_papers, 1):
        print(f"[{i}/{len(key_papers)}] 分析: {url}")
        
        info = get_paper_info(url)
        
        # 基于真实内容评估
        title = info['title']
        abstract = info['abstract']
        
        # 技术分类
        tech_category = "其他"
        if any(word in title.lower() + abstract.lower() for word in ['flash', 'attention']):
            tech_category = "注意力优化"
        elif any(word in title.lower() + abstract.lower() for word in ['quantiz', 'int8', 'int4']):
            tech_category = "量化技术"  
        elif any(word in title.lower() + abstract.lower() for word in ['kv', 'cache']):
            tech_category = "KV缓存优化"
        elif any(word in title.lower() + abstract.lower() for word in ['speculative', 'eagle']):
            tech_category = "推测采样"
        elif any(word in title.lower() + abstract.lower() for word in ['sparse', 'sparsity']):
            tech_category = "稀疏化"
        
        # llama.cpp相关性评分
        relevance = 0.3  # 基础分
        if any(word in title.lower() + abstract.lower() for word in 
               ['inference', 'generation', 'llm', 'transformer']):
            relevance += 0.3
        if any(word in title.lower() + abstract.lower() for word in
               ['gpu', 'cuda', 'memory', 'efficient']):
            relevance += 0.2
        if any(word in title.lower() + abstract.lower() for word in
               ['quantization', 'attention', 'cache']):
            relevance += 0.2
        relevance = min(relevance, 1.0)
        
        # 实施复杂度估算
        complexity = "中等"
        if any(word in title.lower() + abstract.lower() for word in
               ['novel', 'new', 'training', 'architecture']):
            complexity = "高"
        elif any(word in title.lower() + abstract.lower() for word in
                 ['plug-and-play', 'compatible', 'efficient']):
            complexity = "低"
        
        result = {
            "title": title,
            "url": url,
            "category": tech_category,
            "abstract_snippet": abstract[:200] + "..." if len(abstract) > 200 else abstract,
            "performance_claims": info['performance'],
            "llama_cpp_relevance": relevance,
            "estimated_complexity": complexity,
            "analysis_status": "成功" if abstract else "部分成功"
        }
        
        results.append(result)
        
        print(f"✅ {title[:50]}...")
        print(f"   分类: {tech_category} | 相关性: {relevance:.2f} | 复杂度: {complexity}")
        print(f"   性能: {', '.join(info['performance'][:2]) if info['performance'] else '待确认'}")
        print()
    
    return results

def generate_priority_ranking(results):
    """生成优先级排序"""
    def calculate_score(paper):
        score = paper['llama_cpp_relevance'] * 5  # 相关性权重
        
        # 性能收益加分
        if paper['performance_claims']:
            score += len(paper['performance_claims']) * 1.5
        
        # 复杂度调整
        if paper['estimated_complexity'] == '低':
            score += 1
        elif paper['estimated_complexity'] == '高':
            score -= 1
        
        # 技术重要性加分
        important_categories = ['注意力优化', 'KV缓存优化', '量化技术']
        if paper['category'] in important_categories:
            score += 1
        
        return score
    
    # 计算得分并排序
    for paper in results:
        paper['priority_score'] = calculate_score(paper)
    
    results.sort(key=lambda x: x['priority_score'], reverse=True)
    return results

def main():
    # 分析关键论文
    results = analyze_key_papers()
    
    # 生成优先级排序
    ranked_results = generate_priority_ranking(results)
    
    # 生成报告
    report = {
        "analysis_timestamp": datetime.now().isoformat(),
        "method": "基于真实论文内容的快速分析",
        "papers_analyzed": len(results),
        "priority_ranking": ranked_results,
        "summary": {
            "total_papers": len(results),
            "successful_analysis": len([r for r in results if r['analysis_status'] == '成功']),
            "top_categories": {}
        }
    }
    
    # 统计技术分类
    categories = {}
    for paper in results:
        cat = paper['category']
        categories[cat] = categories.get(cat, 0) + 1
    report['summary']['top_categories'] = categories
    
    # 保存结果
    with open('/root/llama.cpp-clip/PRPs/paper_analysis/results/quick_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print("="*50)
    print("🎯 快速分析完成!")
    print(f"📊 成功分析: {report['summary']['successful_analysis']}/{report['summary']['total_papers']} 篇论文")
    print("\n🏆 TOP 5 优先级排序:")
    
    for i, paper in enumerate(ranked_results[:5], 1):
        print(f"{i}. {paper['title'][:55]}...")
        print(f"   得分: {paper['priority_score']:.1f} | {paper['category']} | 复杂度: {paper['estimated_complexity']}")
        if paper['performance_claims']:
            print(f"   性能: {', '.join(paper['performance_claims'][:2])}")
        print()
    
    print("💾 详细报告已保存到: quick_analysis.json")

if __name__ == "__main__":
    main()