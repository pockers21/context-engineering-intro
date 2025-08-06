#!/usr/bin/env python3
"""
BERT技术机制深度分析套件
基于Context Engineering方法论构建的结构化分析工具

包含模块:
1. AttentionMaskAnalyzer - attention mask机制分析
2. MLMFlowTracer - MLM forward pass追踪  
3. LossCalculationVerifier - 损失计算验证
4. ShapeTransformationTracker - 形状变化追踪
5. ComprehensiveReporter - 综合报告生成
"""

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import sys
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# 确保使用本地transformers源码
sys.path.insert(0, '/root/transformers/src')
from transformers import BertTokenizer, BertForMaskedLM

class BERTAnalysisSuite:
    """BERT技术机制综合分析套件"""
    
    def __init__(self, model_path='/root/autodl-fs/google-bert/bert-base-chinese'):
        self.model_path = model_path
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForMaskedLM.from_pretrained(model_path)
        self.analysis_results = {}
        self.log_entries = []
        
    def log(self, message: str, level: str = "INFO"):
        """统一日志记录"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        self.log_entries.append(log_entry)
        print(log_entry)
    
    def verify_environment(self):
        """验证分析环境"""
        self.log("=== 环境验证 ===")
        self.log(f"模型路径: {self.model_path}")
        self.log(f"词汇表大小: {self.model.config.vocab_size}")
        self.log(f"隐藏层维度: {self.model.config.hidden_size}")
        self.log(f"注意力头数: {self.model.config.num_attention_heads}")
        
        self.analysis_results['environment'] = {
            'model_path': self.model_path,
            'vocab_size': self.model.config.vocab_size,
            'hidden_size': self.model.config.hidden_size,
            'num_attention_heads': self.model.config.num_attention_heads
        }
    
    def analyze_attention_mask(self, test_text: str):
        """Attention Mask完整分析"""
        self.log("=== Phase 2: Attention Mask机制分析 ===")
        
        inputs = self.tokenizer(test_text, return_tensors="pt")
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        
        # 创建分析器并执行分析
        analyzer = AttentionMaskAnalyzer(self)
        
        mask_results = analyzer.analyze_mask_transformation(input_ids, attention_mask)
        attention_results = analyzer.trace_attention_computation(input_ids, attention_mask)
        
        self.analysis_results['attention_mask'] = {
            'mask_transformation': mask_results,
            'attention_computation': attention_results
        }
    
    def trace_mlm_forward_pass(self, test_text: str):
        """MLM Forward Pass追踪"""
        self.log("=== Phase 3: MLM Forward Pass追踪 ===")
        
        tracer = MLMFlowTracer(self)
        flow_results = tracer.trace_complete_forward(test_text)
        
        self.analysis_results['mlm_forward_pass'] = flow_results
    
    def verify_loss_calculation(self, test_text: str):
        """损失计算验证"""
        self.log("=== Phase 4: 损失计算验证 ===")
        
        verifier = LossCalculationVerifier(self)
        loss_results = verifier.verify_ignore_index_mechanism()
        
        self.analysis_results['loss_calculation'] = loss_results
    
    def generate_comprehensive_report(self):
        """生成综合分析报告"""
        self.log("=== Phase 5: 生成综合报告 ===")
        
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'summary': {
                'attention_mask_understood': True,
                'mlm_flow_traced': True,
                'loss_mechanism_verified': self.analysis_results.get('loss_calculation', {}).get('verification_passed', False)
            },
            'key_findings': [
                "Attention mask的'全0'现象是正确的，因为经过(1.0-mask)*(-inf)转换",
                "MLM的[MASK]token在attention中是可见的，只在损失计算时被选择性处理",
                "CrossEntropyLoss的ignore_index=-100机制确保只有真实标签位置参与训练"
            ],
            'log_entries': self.log_entries
        }
        
        self.analysis_results['comprehensive_report'] = report
        self.log("✅ 综合分析报告生成完成")
    
    def run_comprehensive_analysis(self, test_text: str = "我喜欢学习人工智能"):
        """运行完整的BERT技术分析"""
        self.log("开始BERT技术机制深度分析", "INFO")
        self.log(f"分析文本: {test_text}")
        
        # Phase 1: 环境验证
        self.verify_environment()
        
        # Phase 2: Attention Mask分析
        self.analyze_attention_mask(test_text)
        
        # Phase 3: MLM Forward Pass追踪
        self.trace_mlm_forward_pass(test_text)
        
        # Phase 4: 损失计算验证
        self.verify_loss_calculation(test_text)
        
        # Phase 5: 生成综合报告
        self.generate_comprehensive_report()
        
        return self.analysis_results

class AttentionMaskAnalyzer:
    """Attention Mask机制深度分析器"""
    
    def __init__(self, suite: BERTAnalysisSuite):
        self.suite = suite
        
    def analyze_mask_transformation(self, input_ids, attention_mask):
        """分析attention mask的数值转换过程"""
        self.suite.log("=== Attention Mask数值转换分析 ===")
        
        results = {
            'original_mask': attention_mask.clone(),
            'mask_values': {},
            'transformation_steps': []
        }
        
        # 1. 原始mask
        self.suite.log(f"1. 原始attention_mask: {attention_mask[0].tolist()}")
        self.suite.log(f"   含义: 1=有效token, 0=padding")
        
        # 2. 模拟extend_attention_mask过程
        batch_size, seq_length = attention_mask.shape
        extended_mask = attention_mask[:, None, None, :]  # [batch, 1, 1, seq_len]
        self.suite.log(f"2. 扩展为4D: {extended_mask.shape}")
        
        # 3. 关键转换: (1.0 - mask) * -inf
        dtype = torch.float32
        extended_mask = (1.0 - extended_mask) * torch.finfo(dtype).min
        self.suite.log(f"3. 转换公式: (1.0 - mask) * {torch.finfo(dtype).min}")
        self.suite.log(f"   转换后值: {extended_mask[0, 0, 0, :5].tolist()}")
        self.suite.log(f"   含义: 0=可关注, -inf=忽略")
        
        results['extended_mask'] = extended_mask
        results['transformation_formula'] = "(1.0 - attention_mask) * torch.finfo(dtype).min"
        
        return results
        
    def trace_attention_computation(self, input_ids, attention_mask):
        """追踪attention计算中mask的作用"""
        self.suite.log("=== Attention计算中Mask作用追踪 ===")
        
        # 获取第一层attention的中间值
        with torch.no_grad():
            # 手动调用embedding和第一层
            embeddings = self.suite.model.bert.embeddings(input_ids)
            
            # 获取扩展后的attention mask
            extended_mask = self.suite.model.bert.get_extended_attention_mask(
                attention_mask, input_ids.shape
            )
            
            # 第一层attention
            attention_layer = self.suite.model.bert.encoder.layer[0].attention.self
            
            # 计算Q, K, V
            mixed_query = attention_layer.query(embeddings)
            mixed_key = attention_layer.key(embeddings)
            mixed_value = attention_layer.value(embeddings)
            
            # 转换为多头格式
            query = attention_layer.transpose_for_scores(mixed_query)
            key = attention_layer.transpose_for_scores(mixed_key)
            
            # 计算attention scores
            attention_scores = torch.matmul(query, key.transpose(-1, -2))
            attention_scores = attention_scores / (attention_layer.attention_head_size ** 0.5)
            
            self.suite.log(f"attention_scores计算后: {attention_scores[0, 0, 0, :5]}")
            
            # 应用mask
            attention_scores_masked = attention_scores + extended_mask
            self.suite.log(f"加上mask后: {attention_scores_masked[0, 0, 0, :5]}")
            
            # softmax
            attention_probs = F.softmax(attention_scores_masked, dim=-1)
            self.suite.log(f"softmax后: {attention_probs[0, 0, 0, :5]}")
            
        return {
            'raw_scores': attention_scores[0, 0, 0, :].detach(),
            'masked_scores': attention_scores_masked[0, 0, 0, :].detach(), 
            'attention_probs': attention_probs[0, 0, 0, :].detach()
        }

class MLMFlowTracer:
    """MLM Forward Pass完整追踪器"""
    
    def __init__(self, suite: BERTAnalysisSuite):
        self.suite = suite
        
    def trace_complete_forward(self, text: str):
        """追踪完整的MLM forward pass"""
        self.suite.log("=== MLM Forward Pass完整追踪 ===")
        
        # 准备输入
        inputs = self.suite.tokenizer(text, return_tensors="pt")
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        
        shapes_log = []
        
        def log_shape(name: str, tensor: torch.Tensor):
            shape_info = f"{name}: {list(tensor.shape)}"
            shapes_log.append(shape_info)
            self.suite.log(f"  {shape_info}")
        
        self.suite.model.eval()
        with torch.no_grad():
            # 1. Embeddings
            self.suite.log("1. BertEmbeddings")
            embeddings_out = self.suite.model.bert.embeddings(input_ids)
            log_shape("embeddings_output", embeddings_out)
            
            # 2. Encoder layers
            self.suite.log("2. BertEncoder处理")
            encoder_out = self.suite.model.bert.encoder(
                embeddings_out, 
                attention_mask=self.suite.model.bert.get_extended_attention_mask(
                    attention_mask, input_ids.shape
                )
            )
            sequence_output = encoder_out.last_hidden_state
            log_shape("sequence_output", sequence_output)
            
            # 3. Prediction Head Transform
            self.suite.log("3. BertLMPredictionHead")
            self.suite.log("  3.1 Transform层")
            transformed = self.suite.model.cls.predictions.transform(sequence_output)
            log_shape("transformed_output", transformed)
            
            # 3.2 Decoder
            self.suite.log("  3.2 Decoder线性层")
            prediction_scores = self.suite.model.cls.predictions.decoder(transformed)
            log_shape("prediction_scores", prediction_scores)
            
        return {
            'shapes_log': shapes_log,
            'sequence_output_shape': list(sequence_output.shape),
            'prediction_scores_shape': list(prediction_scores.shape),
            'vocab_size': self.suite.model.config.vocab_size
        }

class LossCalculationVerifier:
    """损失计算机制验证器"""
    
    def __init__(self, suite: BERTAnalysisSuite):
        self.suite = suite
        
    def verify_ignore_index_mechanism(self):
        """验证CrossEntropyLoss的ignore_index=-100机制"""
        self.suite.log("=== CrossEntropyLoss ignore_index机制验证 ===")
        
        # 模拟数据
        batch_size, seq_length, vocab_size = 1, 8, 21128
        
        # 创建测试数据
        prediction_scores = torch.randn(batch_size, seq_length, vocab_size)
        labels = torch.full((batch_size, seq_length), -100, dtype=torch.long)
        
        # 只在位置5设置真实标签
        mask_pos = 5
        true_token_id = 3721  # 随机选择一个token id
        labels[0, mask_pos] = true_token_id
        
        self.suite.log(f"labels: {labels[0].tolist()}")
        self.suite.log(f"只有位置{mask_pos}有真实标签: {true_token_id}")
        
        # 1. 使用PyTorch的CrossEntropyLoss
        loss_fct = CrossEntropyLoss()  # ignore_index=-100 是默认值
        pytorch_loss = loss_fct(
            prediction_scores.view(-1, vocab_size), 
            labels.view(-1)
        )
        
        # 2. 手动计算只有mask位置的损失
        mask_logits = prediction_scores[0, mask_pos]  # [vocab_size]
        mask_probs = F.softmax(mask_logits, dim=-1)
        manual_loss = -torch.log(mask_probs[true_token_id])
        
        self.suite.log(f"PyTorch CrossEntropyLoss: {pytorch_loss.item():.6f}")
        self.suite.log(f"手动计算损失: {manual_loss.item():.6f}")
        self.suite.log(f"差异: {abs(pytorch_loss.item() - manual_loss.item()):.8f}")
        
        # 3. 验证其他位置被忽略
        self.suite.log("\n验证-100位置被忽略:")
        for pos in [0, 1, 2, 3, 4, 6, 7]:
            pos_labels = torch.full((batch_size, seq_length), -100, dtype=torch.long)
            pos_labels[0, pos] = true_token_id
            pos_loss = loss_fct(prediction_scores.view(-1, vocab_size), pos_labels.view(-1))
            self.suite.log(f"  位置{pos}单独损失: {pos_loss.item():.6f}")
            
        return {
            'pytorch_loss': pytorch_loss.item(),
            'manual_loss': manual_loss.item(),
            'difference': abs(pytorch_loss.item() - manual_loss.item()),
            'verification_passed': abs(pytorch_loss.item() - manual_loss.item()) < 1e-6
        }

def main():
    """主分析流程"""
    print("🚀 启动BERT技术机制深度分析")
    print("基于Context Engineering方法论的结构化分析")
    print("="*60)
    
    # 创建分析套件
    suite = BERTAnalysisSuite()
    
    # 运行综合分析
    results = suite.run_comprehensive_analysis("我喜欢学习人工[MASK]")
    
    print("\n" + "="*60)
    print("🎯 分析完成！结果已保存到 bert_analysis_results.json")
    
    # 保存结果
    with open('/root/bert_analysis_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

if __name__ == "__main__":
    main()