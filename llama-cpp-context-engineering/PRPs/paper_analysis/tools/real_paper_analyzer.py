#!/usr/bin/env python3
"""
基于真实论文内容的推理优化分析工具
通过网络访问获取论文实际内容，提供准确的技术评估
"""

import subprocess
import re
import json
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass 
class RealPaperAnalysis:
    title: str
    url: str
    abstract: str
    key_contributions: List[str]
    technical_details: List[str]
    performance_claims: List[str]
    implementation_complexity: str
    llama_cpp_relevance: float

class RealPaperAnalyzer:
    def __init__(self):
        self.papers = []
        self.analysis_results = {}
    
    def fetch_arxiv_content(self, url: str) -> Dict[str, str]:
        """获取arxiv论文的实际内容"""
        try:
            cmd = f'curl -s "{url}"'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            html_content = result.stdout
            
            # 提取标题
            title_match = re.search(r'<title>(.*?)</title>', html_content)
            title = title_match.group(1).strip() if title_match else "未找到标题"
            
            # 提取摘要
            abstract_match = re.search(r'<div class="ltx_abstract">.*?<p class="ltx_p"[^>]*>(.*?)</p>', html_content, re.DOTALL)
            abstract = ""
            if abstract_match:
                abstract = re.sub(r'<[^>]+>', '', abstract_match.group(1))
                abstract = re.sub(r'\s+', ' ', abstract).strip()
            
            return {
                "title": title,
                "abstract": abstract,
                "html_content": html_content
            }
        except Exception as e:
            print(f"获取论文内容失败 {url}: {e}")
            return {"title": "", "abstract": "", "html_content": ""}
    
    def analyze_paper_from_list(self, paper_line: str) -> Optional[RealPaperAnalysis]:
        """从论文列表中的一行分析真实论文内容"""
        
        # 提取URL
        url_match = re.search(r'https://arxiv\.org/[^\s]+', paper_line)
        if not url_match:
            return None
            
        url = url_match.group(0)
        print(f"正在分析论文: {url}")
        
        # 获取真实内容
        content = self.fetch_arxiv_content(url)
        if not content["title"]:
            return None
        
        # 基于真实内容分析
        title = content["title"]
        abstract = content["abstract"]
        
        # 分析技术贡献
        key_contributions = self.extract_contributions(abstract, content["html_content"])
        
        # 分析性能声明
        performance_claims = self.extract_performance_claims(abstract)
        
        # 评估实施复杂度
        complexity = self.assess_implementation_complexity(title, abstract, key_contributions)
        
        # 评估与llama.cpp的相关性
        relevance = self.assess_llama_cpp_relevance(title, abstract, key_contributions)
        
        return RealPaperAnalysis(
            title=title,
            url=url,
            abstract=abstract,
            key_contributions=key_contributions,
            technical_details=[],  # 需要进一步分析HTML内容
            performance_claims=performance_claims,
            implementation_complexity=complexity,
            llama_cpp_relevance=relevance
        )
    
    def extract_contributions(self, abstract: str, html_content: str) -> List[str]:
        """从摘要和内容中提取技术贡献"""
        contributions = []
        
        # 从摘要中提取关键技术点
        if "introduce" in abstract.lower():
            intro_part = abstract.lower().split("introduce")[1].split(".")[0]
            contributions.append(f"核心技术: {intro_part.strip()}")
        
        # 查找方法关键词
        methods = []
        if "speculative" in abstract.lower():
            methods.append("推测采样技术")
        if "quantization" in abstract.lower():
            methods.append("量化技术")  
        if "attention" in abstract.lower():
            methods.append("注意力优化")
        if "cache" in abstract.lower():
            methods.append("缓存优化")
        
        contributions.extend(methods)
        return contributions
    
    def extract_performance_claims(self, abstract: str) -> List[str]:
        """提取性能提升声明"""
        claims = []
        
        # 查找加速比
        speedup_matches = re.findall(r'(\d+\.?\d*)\s*x\s*(speedup|improvement|faster)', abstract.lower())
        for match in speedup_matches:
            claims.append(f"{match[0]}x {match[1]}")
        
        # 查找百分比提升
        percent_matches = re.findall(r'(\d+)%\s*(improvement|better|reduction)', abstract.lower())
        for match in percent_matches:
            claims.append(f"{match[0]}% {match[1]}")
            
        return claims
    
    def assess_implementation_complexity(self, title: str, abstract: str, contributions: List[str]) -> str:
        """评估实施复杂度"""
        complexity_indicators = 0
        
        # 技术复杂度指标
        if any(word in title.lower() + abstract.lower() for word in ["novel", "new", "training", "fine-tuning"]):
            complexity_indicators += 2
        
        if any(word in title.lower() + abstract.lower() for word in ["kernel", "cuda", "optimization"]):
            complexity_indicators += 1
            
        if any(word in title.lower() + abstract.lower() for word in ["framework", "system", "architecture"]):
            complexity_indicators += 2
        
        if complexity_indicators >= 4:
            return "高复杂度"
        elif complexity_indicators >= 2:
            return "中等复杂度"
        else:
            return "低复杂度"
    
    def assess_llama_cpp_relevance(self, title: str, abstract: str, contributions: List[str]) -> float:
        """评估与llama.cpp的相关性 (0-1)"""
        relevance_score = 0.0
        
        # 直接相关的技术
        if any(word in title.lower() + abstract.lower() for word in 
               ["inference", "generation", "sampling", "decoding"]):
            relevance_score += 0.3
        
        if any(word in title.lower() + abstract.lower() for word in
               ["quantization", "compression", "acceleration"]):
            relevance_score += 0.3
            
        if any(word in title.lower() + abstract.lower() for word in
               ["attention", "transformer", "llm"]):
            relevance_score += 0.2
        
        if any(word in title.lower() + abstract.lower() for word in
               ["gpu", "cuda", "kernel", "memory"]):
            relevance_score += 0.2
        
        return min(relevance_score, 1.0)

def main():
    """分析论文列表中的真实内容"""
    print("🔍 开始基于真实论文内容的分析")
    print("="*50)
    
    analyzer = RealPaperAnalyzer()
    
    # 从paper.md读取论文列表
    try:
        with open('/root/llama.cpp-clip/PRPs/paper_analysis/paper.md', 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print("未找到paper.md文件")
        return
    
    # 提取arxiv链接并分析
    arxiv_urls = re.findall(r'https://arxiv\.org/[^\s]+', content)
    
    print(f"发现 {len(arxiv_urls)} 个arxiv论文链接")
    print()
    
    real_analyses = []
    for i, url in enumerate(arxiv_urls[:5], 1):  # 先分析前5篇
        print(f"[{i}/5] 分析: {url}")
        analysis = analyzer.analyze_paper_from_list(url)
        if analysis:
            real_analyses.append(analysis)
            print(f"✅ 标题: {analysis.title[:60]}...")
            print(f"   复杂度: {analysis.implementation_complexity}")
            print(f"   相关性: {analysis.llama_cpp_relevance:.2f}")
            print(f"   性能声明: {', '.join(analysis.performance_claims[:2])}")
        else:
            print(f"❌ 分析失败")
        print()
    
    # 保存真实分析结果
    results = {
        "analysis_timestamp": datetime.now().isoformat(),
        "method": "基于真实论文内容分析",
        "papers_analyzed": len(real_analyses),
        "real_analyses": [
            {
                "title": p.title,
                "url": p.url,
                "abstract": p.abstract[:200] + "..." if len(p.abstract) > 200 else p.abstract,
                "key_contributions": p.key_contributions,
                "performance_claims": p.performance_claims,
                "implementation_complexity": p.implementation_complexity,
                "llama_cpp_relevance": p.llama_cpp_relevance
            } for p in real_analyses
        ]
    }
    
    with open('/root/llama.cpp-clip/PRPs/paper_analysis/results/real_paper_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("📊 基于真实内容的分析完成!")
    print("结果已保存到 real_paper_analysis.json")

if __name__ == "__main__":
    main()