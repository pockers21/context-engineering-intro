# BERT技术机制深度分析项目

## FEATURE:
深入理解和分析BERT模型的核心技术机制，包括attention mask处理、MLM训练、损失计算等关键组件的工作原理

## TOOLS:
- 代码追踪脚本：分析attention mask在各层的数值变化和处理逻辑
- 调试工具：实时打印中间值、张量形状、数据流向
- 对比分析：训练模式vs推理模式的行为差异
- 损失计算验证：手动实现并验证CrossEntropyLoss的-100忽略机制
- 形状变化追踪：从sequence_output到prediction_scores的详细变换
- 文档生成：整理发现的技术原理和最佳实践

## DEPENDENCIES:
- transformers库源码 (/root/transformers/src/)
- BERT中文预训练模型 (/root/autodl-fs/google-bert/bert-base-chinese)
- PyTorch调试和可视化工具
- 现有调试脚本 (trace_*.py文件)

## 研究目标:
1. 彻底理解attention mask的"全0"现象及其正确性
2. 区分attention masking vs MLM token masking的不同作用
3. 追踪BERT forward pass中的数据流和形状变化
4. 验证损失计算中-100标签的控制机制
5. 建立完整的BERT技术机制知识体系

## 预期成果:
- 完整的BERT技术分析报告
- 可复用的调试工具集
- 清晰的代码级别技术文档
- 验证过的技术结论和最佳实践指南