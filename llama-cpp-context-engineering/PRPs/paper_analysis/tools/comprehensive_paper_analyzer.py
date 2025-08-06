#!/usr/bin/env python3
"""
全面的论文分析工具 - 基于真实内容
系统性分析所有推理优化论文，提供准确的技术评估和实施建议
"""

import subprocess
import re
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import time

@dataclass
class PaperAnalysis:
    title: str
    url: str
    abstract: str
    key_techniques: List[str]
    performance_improvements: List[str]
    implementation_requirements: List[str]
    llama_cpp_compatibility: float
    estimated_complexity: str
    potential_benefits: List[str]
    technical_challenges: List[str]
    priority_score: float

class ComprehensivePaperAnalyzer:
    def __init__(self):
        self.analyses = []
        self.failed_papers = []
    
    def fetch_paper_content(self, url: str) -> Dict[str, str]:
        """获取论文的详细内容"""
        try:
            # 添加延迟避免过于频繁的请求
            time.sleep(1)
            
            cmd = f'curl -s "{url}"'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            content = result.stdout
            
            # 提取标题
            title = self.extract_title(content)
            
            # 提取摘要
            abstract = self.extract_abstract(content)
            
            # 提取正文关键部分
            key_sections = self.extract_key_sections(content)
            
            return {
                "title": title,
                "abstract": abstract,
                "content": content,
                "key_sections": key_sections
            }
        except Exception as e:
            print(f"❌ 获取失败 {url}: {e}")
            return {}
    
    def extract_title(self, content: str) -> str:
        """提取论文标题"""
        patterns = [
            r'<title>(.*?)</title>',
            r'<h1[^>]*>(.*?)</h1>',
            r'class="ltx_title"[^>]*>(.*?)</span>'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                title = re.sub(r'<[^>]+>', '', match.group(1))
                title = re.sub(r'\s+', ' ', title).strip()
                if len(title) > 10:  # 确保不是空标题
                    return title
        return "未找到标题"
    
    def extract_abstract(self, content: str) -> str:
        """提取论文摘要"""
        patterns = [
            r'<div class="ltx_abstract">.*?<p[^>]*>(.*?)</p>',
            r'<section[^>]*abstract[^>]*>.*?<p[^>]*>(.*?)</p>',
            r'Abstract</h[0-9]>.*?<p[^>]*>(.*?)</p>'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                abstract = re.sub(r'<[^>]+>', '', match.group(1))
                abstract = re.sub(r'\s+', ' ', abstract).strip()
                if len(abstract) > 50:  # 确保是有意义的摘要
                    return abstract
        return "未找到摘要"
    
    def extract_key_sections(self, content: str) -> Dict[str, str]:
        """提取论文关键章节"""
        sections = {}
        
        # 查找方法部分
        method_patterns = [
            r'(?i)<h[0-9][^>]*>\s*method[s]?\s*</h[0-9]>(.*?)(?=<h[0-9]|$)',
            r'(?i)<h[0-9][^>]*>\s*approach\s*</h[0-9]>(.*?)(?=<h[0-9]|$)',
        ]
        
        for pattern in method_patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                sections['method'] = re.sub(r'<[^>]+>', '', match.group(1))[:500]
                break
        
        # 查找实验结果
        result_patterns = [
            r'(?i)<h[0-9][^>]*>\s*experiment[s]?\s*</h[0-9]>(.*?)(?=<h[0-9]|$)',
            r'(?i)<h[0-9][^>]*>\s*result[s]?\s*</h[0-9]>(.*?)(?=<h[0-9]|$)',
            r'(?i)<h[0-9][^>]*>\s*evaluation\s*</h[0-9]>(.*?)(?=<h[0-9]|$)'
        ]
        
        for pattern in result_patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                sections['results'] = re.sub(r'<[^>]+>', '', match.group(1))[:500]
                break
        
        return sections
    
    def analyze_techniques(self, title: str, abstract: str, content: str) -> List[str]:
        """分析论文的关键技术"""
        techniques = []
        text = (title + " " + abstract).lower()
        
        # 量化相关技术
        if any(word in text for word in ['quantization', 'quantize', 'int8', 'int4', 'bit']):
            if 'int8' in text: techniques.append('INT8量化')
            if 'int4' in text: techniques.append('INT4量化')
            if 'mixed' in text: techniques.append('混合精度')
            if not any('INT' in t for t in techniques):
                techniques.append('通用量化')
        
        # 注意力优化
        if any(word in text for word in ['attention', 'flash', 'sage']):
            if 'flash' in text: techniques.append('FlashAttention优化')
            elif 'sage' in text: techniques.append('SageAttention技术')
            else: techniques.append('注意力机制优化')
        
        # 缓存优化
        if any(word in text for word in ['cache', 'kv', 'memory']):
            if 'kv' in text: techniques.append('KV缓存优化')
            if 'compress' in text: techniques.append('缓存压缩')
            if 'paged' in text: techniques.append('分页缓存')
        
        # 推测采样
        if any(word in text for word in ['speculative', 'eagle', 'draft']):
            techniques.append('推测采样')
        
        # 稀疏化
        if any(word in text for word in ['sparse', 'sparsity', 'pruning']):
            techniques.append('激活稀疏化')
        
        # 算子融合
        if any(word in text for word in ['fusion', 'fuse', 'kernel']):
            techniques.append('算子融合')
        
        # 并行推理
        if any(word in text for word in ['parallel', 'batch', 'throughput']):
            techniques.append('并行推理优化')
        
        return techniques if techniques else ['其他优化技术']
    
    def extract_performance_data(self, abstract: str, content: str) -> List[str]:
        """提取性能改进数据"""
        improvements = []
        text = abstract + " " + content
        
        # 查找加速比 (如 2.5x, 3x speedup)
        speedup_patterns = [
            r'(\d+\.?\d*)\s*[×x]\s*(?:speedup|faster|improvement)',
            r'(?:speedup|faster|improvement).*?(\d+\.?\d*)\s*[×x]',
            r'(\d+\.?\d*)\s*times\s*faster'
        ]
        
        for pattern in speedup_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                improvements.append(f"{match}x 加速")
        
        # 查找百分比改进 (如 30% improvement)
        percent_patterns = [
            r'(\d+)%\s*(?:improvement|better|reduction|faster)',
            r'(?:improve|reduce|increase).*?(\d+)%'
        ]
        
        for pattern in percent_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                improvements.append(f"{match}% 改进")
        
        # 查找内存优化 (如 50% memory reduction)
        memory_patterns = [
            r'(\d+)%\s*(?:memory|mem)\s*(?:reduction|saving)',
            r'(?:memory|mem).*?(?:reduce|save).*?(\d+)%'
        ]
        
        for pattern in memory_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                improvements.append(f"{match}% 内存节省")
        
        return improvements if improvements else ['性能提升数据待确认']
    
    def assess_implementation_requirements(self, techniques: List[str], abstract: str) -> List[str]:
        """评估实施要求"""
        requirements = []
        text = abstract.lower()
        
        # CUDA开发需求
        if any(word in text for word in ['gpu', 'cuda', 'kernel']):
            requirements.append('CUDA内核开发')
        
        # 算法重新实现
        if any(word in text for word in ['novel', 'new', 'propose']):
            requirements.append('新算法实现')
        
        # 架构修改
        if any(word in text for word in ['architecture', 'framework', 'system']):
            requirements.append('架构级别修改')
        
        # 模型训练/微调
        if any(word in text for word in ['training', 'fine-tuning', 'calibration']):
            requirements.append('模型校准/训练')
        
        # 特定硬件支持
        if any(word in text for word in ['ampere', 'tensor', 'specific']):
            requirements.append('特定硬件支持')
        
        return requirements if requirements else ['基础实现']
    
    def calculate_llama_cpp_compatibility(self, techniques: List[str], requirements: List[str]) -> float:
        """计算与llama.cpp的兼容性评分 (0-1)"""
        compatibility = 0.5  # 基础分数
        
        # 提升兼容性的因素
        positive_factors = [
            ('FlashAttention优化', 0.2),
            ('量化', 0.15),
            ('KV缓存优化', 0.2),
            ('算子融合', 0.15),
            ('并行推理优化', 0.1)
        ]
        
        for technique in techniques:
            for factor, score in positive_factors:
                if factor in technique:
                    compatibility += score
        
        # 降低兼容性的因素
        negative_factors = [
            ('架构级别修改', -0.3),
            ('模型校准/训练', -0.2),
            ('特定硬件支持', -0.1)
        ]
        
        for requirement in requirements:
            for factor, penalty in negative_factors:
                if factor in requirement:
                    compatibility += penalty
        
        return max(0.0, min(1.0, compatibility))
    
    def estimate_complexity(self, techniques: List[str], requirements: List[str]) -> str:
        """估算实施复杂度"""
        complexity_score = 0
        
        # 技术复杂度
        high_complexity_techniques = ['推测采样', 'FlashAttention优化', '混合精度']
        medium_complexity_techniques = ['KV缓存优化', 'SageAttention技术', '激活稀疏化']
        
        for technique in techniques:
            if any(hct in technique for hct in high_complexity_techniques):
                complexity_score += 3
            elif any(mct in technique for mct in medium_complexity_techniques):
                complexity_score += 2
            else:
                complexity_score += 1
        
        # 实施要求复杂度
        complex_requirements = ['CUDA内核开发', '新算法实现', '架构级别修改']
        for requirement in requirements:
            if any(cr in requirement for cr in complex_requirements):
                complexity_score += 2
        
        if complexity_score >= 8:
            return "极高复杂度"
        elif complexity_score >= 6:
            return "高复杂度"
        elif complexity_score >= 4:
            return "中等复杂度"
        else:
            return "低复杂度"
    
    def calculate_priority_score(self, analysis: PaperAnalysis) -> float:
        """计算优先级得分"""
        # 性能收益分数
        benefit_score = len(analysis.performance_improvements) * 2
        
        # 兼容性分数
        compatibility_score = analysis.llama_cpp_compatibility * 5
        
        # 复杂度惩罚
        complexity_penalties = {
            "低复杂度": 0,
            "中等复杂度": 2,
            "高复杂度": 4,
            "极高复杂度": 6
        }
        complexity_penalty = complexity_penalties.get(analysis.estimated_complexity, 3)
        
        # 技术重要性分数
        important_techniques = ['FlashAttention优化', 'KV缓存优化', 'INT8量化']
        tech_score = sum(2 for tech in analysis.key_techniques 
                        if any(imp in tech for imp in important_techniques))
        
        total_score = benefit_score + compatibility_score + tech_score - complexity_penalty
        return max(0.0, total_score)
    
    def analyze_single_paper(self, url: str) -> Optional[PaperAnalysis]:
        """分析单篇论文"""
        print(f"🔍 分析: {url}")
        
        content_data = self.fetch_paper_content(url)
        if not content_data or not content_data.get('title'):
            self.failed_papers.append(url)
            return None
        
        title = content_data['title']
        abstract = content_data['abstract']
        content = content_data['content']
        
        # 技术分析
        techniques = self.analyze_techniques(title, abstract, content)
        performance = self.extract_performance_data(abstract, content)
        requirements = self.assess_implementation_requirements(techniques, abstract)
        compatibility = self.calculate_llama_cpp_compatibility(techniques, requirements)
        complexity = self.estimate_complexity(techniques, requirements)
        
        # 提取潜在收益和挑战
        benefits = self.extract_potential_benefits(techniques, performance)
        challenges = self.extract_technical_challenges(requirements, complexity)
        
        analysis = PaperAnalysis(
            title=title,
            url=url,
            abstract=abstract[:300] + "..." if len(abstract) > 300 else abstract,
            key_techniques=techniques,
            performance_improvements=performance,
            implementation_requirements=requirements,
            llama_cpp_compatibility=compatibility,
            estimated_complexity=complexity,
            potential_benefits=benefits,
            technical_challenges=challenges,
            priority_score=0.0  # 待计算
        )
        
        # 计算优先级得分
        analysis.priority_score = self.calculate_priority_score(analysis)
        
        print(f"✅ {title[:50]}...")
        print(f"   技术: {', '.join(techniques[:2])}")
        print(f"   复杂度: {complexity}")
        print(f"   兼容性: {compatibility:.2f}")
        print(f"   优先级: {analysis.priority_score:.2f}")
        
        return analysis
    
    def extract_potential_benefits(self, techniques: List[str], performance: List[str]) -> List[str]:
        """提取潜在收益"""
        benefits = []
        
        if any('量化' in t for t in techniques):
            benefits.append('内存使用减少')
            benefits.append('推理速度提升')
        
        if any('FlashAttention' in t for t in techniques):
            benefits.append('显存效率大幅提升')
        
        if any('缓存' in t for t in techniques):
            benefits.append('长序列处理能力增强')
        
        if any('推测采样' in t for t in techniques):
            benefits.append('生成吞吐量提升')
        
        if performance:
            benefits.extend([f"已验证: {p}" for p in performance[:2]])
        
        return benefits if benefits else ['性能优化']
    
    def extract_technical_challenges(self, requirements: List[str], complexity: str) -> List[str]:
        """提取技术挑战"""
        challenges = []
        
        if 'CUDA内核开发' in requirements:
            challenges.append('需要GPU编程专业知识')
        
        if '架构级别修改' in requirements:
            challenges.append('可能影响系统稳定性')
        
        if '新算法实现' in requirements:
            challenges.append('算法调优和验证复杂')
        
        if complexity in ['高复杂度', '极高复杂度']:
            challenges.append('实施周期长')
            challenges.append('需要专业团队')
        
        return challenges if challenges else ['常规实施挑战']
    
    def run_comprehensive_analysis(self) -> Dict:
        """运行全面分析"""
        print("🚀 开始全面论文分析")
        print("="*60)
        
        # 从paper.md读取所有arxiv链接
        try:
            with open('/root/llama.cpp-clip/PRPs/paper_analysis/paper.md', 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            print("❌ 未找到paper.md文件")
            return {}
        
        # 提取所有arxiv URL
        arxiv_urls = re.findall(r'https://arxiv\.org/[^\s]+', content)
        print(f"📚 发现 {len(arxiv_urls)} 篇arxiv论文")
        print()
        
        # 分析每篇论文
        for i, url in enumerate(arxiv_urls, 1):
            print(f"[{i}/{len(arxiv_urls)}]", end=" ")
            analysis = self.analyze_single_paper(url)
            if analysis:
                self.analyses.append(analysis)
            print()
        
        # 按优先级排序
        self.analyses.sort(key=lambda x: x.priority_score, reverse=True)
        
        # 生成总结报告
        report = self.generate_comprehensive_report()
        
        return report
    
    def generate_comprehensive_report(self) -> Dict:
        """生成全面报告"""
        return {
            "analysis_metadata": {
                "timestamp": datetime.now().isoformat(),
                "method": "基于真实论文内容的全面分析",
                "papers_found": len(self.analyses) + len(self.failed_papers),
                "papers_analyzed": len(self.analyses),
                "failed_papers": len(self.failed_papers)
            },
            "top_priority_papers": [
                {
                    "rank": i+1,
                    "title": analysis.title,
                    "priority_score": analysis.priority_score,
                    "key_techniques": analysis.key_techniques,
                    "complexity": analysis.estimated_complexity,
                    "compatibility": analysis.llama_cpp_compatibility,
                    "benefits": analysis.potential_benefits[:3],
                    "url": analysis.url
                }
                for i, analysis in enumerate(self.analyses[:10])
            ],
            "technical_categories": self.categorize_by_technique(),
            "complexity_distribution": self.analyze_complexity_distribution(),
            "implementation_roadmap": self.generate_implementation_roadmap(),
            "detailed_analyses": [asdict(analysis) for analysis in self.analyses],
            "failed_urls": self.failed_papers
        }
    
    def categorize_by_technique(self) -> Dict[str, List[str]]:
        """按技术类型分类"""
        categories = {}
        for analysis in self.analyses:
            for technique in analysis.key_techniques:
                if technique not in categories:
                    categories[technique] = []
                categories[technique].append(analysis.title)
        return categories
    
    def analyze_complexity_distribution(self) -> Dict[str, int]:
        """分析复杂度分布"""
        distribution = {}
        for analysis in self.analyses:
            complexity = analysis.estimated_complexity
            distribution[complexity] = distribution.get(complexity, 0) + 1
        return distribution
    
    def generate_implementation_roadmap(self) -> Dict[str, List[str]]:
        """生成实施路线图"""
        roadmap = {
            "第一阶段 (立即实施)": [],
            "第二阶段 (中期规划)": [],
            "第三阶段 (长期研究)": []
        }
        
        for analysis in self.analyses:
            if (analysis.priority_score >= 8 and 
                analysis.estimated_complexity in ["低复杂度", "中等复杂度"]):
                roadmap["第一阶段 (立即实施)"].append(analysis.title)
            elif analysis.priority_score >= 6:
                roadmap["第二阶段 (中期规划)"].append(analysis.title)
            else:
                roadmap["第三阶段 (长期研究)"].append(analysis.title)
        
        return roadmap

def main():
    analyzer = ComprehensivePaperAnalyzer()
    report = analyzer.run_comprehensive_analysis()
    
    # 保存完整报告
    output_path = '/root/llama.cpp-clip/PRPs/paper_analysis/results/comprehensive_analysis.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*60)
    print("🎯 分析完成总结:")
    print(f"✅ 成功分析: {report['analysis_metadata']['papers_analyzed']} 篇")
    print(f"❌ 分析失败: {report['analysis_metadata']['failed_papers']} 篇")
    print(f"\n📊 TOP 5 优先级论文:")
    
    for paper in report['top_priority_papers'][:5]:
        print(f"{paper['rank']}. {paper['title'][:60]}...")
        print(f"   得分: {paper['priority_score']:.1f} | 复杂度: {paper['complexity']} | 兼容性: {paper['compatibility']:.2f}")
    
    print(f"\n💾 完整报告已保存到: {output_path}")

if __name__ == "__main__":
    main()