昨天提到的论文是abq-llm,另外qserve也是类似的。可以看一下

我记得上周你说过有几个开源的项目，宣称自己的的cuda性能可以追上超越cublas，有链接吗
neuralmagic kernel, marlin kernel ?
marlin 刚出论文了 https://arxiv.org/abs/2408.11743 看看，回头分享一下

https://github.com/linkedin/Liger-Kernel这个仓库实现了加速算子实现

INT-FlashAttention: Enabling Flash Attention for INT8 Quantization
https://arxiv.org/html/2409.16997v1

SageAttention: Accurate 8-bit attention for Plug-and-Play Inference Acceleration
https://arxiv.org/html/2410.02367v1

KV-Compress: Paged KV-Cache Compression with Variable Compression Rates per Attention Head
https://arxiv.org/html/2410.00161v2

https://arxiv.org/html/2408.14690v1  Training-Free Activation Sparsity in Large Language Models

w4a4 最新的一篇
FlatQuant: Flatness Matters for LLM Quantization
 https://arxiv.org/html/2410.09426v1  

kv cache 相关的
KV Cache is 1 Bit Per Channel: Efficient Large Language Model Inference with Coupled Quantization

 https://arxiv.org/abs/2405.03917

Keyformer: KV Cache Reduction through Key Tokens Selection for Efficient Generative Inference

  https://arxiv.org/abs/2403.09054v2 

 https://arxiv.org/html/2409.13731v3

刚才说的 QSpec https://arxiv.org/html/2410.11305v1  

Rl 实战代码
https://huggingface.co/learn/deep-rl-course/unit0/introduction

 https://arxiv.org/html/2503.02812v1  

这个论文挺有意思，通过惩罚不成熟的推理时策略转变来加速推理和提高精度 https://arxiv.org/html/2501.18585v2 

下面是一些最近看到的，包含一些推理时 scaling 的观察，有助于理解如何去对 LRM 模型的推理做改进。传统的模型的稀疏化、pruning、训练时压缩等都不在这个范围：

- 减少token 输出，可以用 prompting（COD？），减少 alternative 数量（https://arxiv.org/html/2501.18585v2），或者https://github.com/hao-ai-lab/Dynasor 
- 推测采样：eagle3
- 提高 LRM 精度：Laconic decoding: Run the model 5 times (in parallel) and pick the answer with the smallest number of tokens.
- https://arxiv.org/html/2503.13288v1 ϕ-Decoding: Adaptive Foresight Sampling for Balanced Inference-Time Exploration and Exploitation
- https://flashinfer.ai/2025/03/10/sampling.html Sorting-Free GPU Kernels for LLM Sampling
- Thoughts Are All Over the Place: On the Underthinking of o1-Like LLMs https://arxiv.org/html/2501.18585v2
- Scaling Test-Time Compute Without Verification or RL is Suboptimal https://arxiv.org/html/2502.12118v2
- From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step https://arxiv.org/html/2405.14838v1
- https://arxiv.org/html/2408.03314v1 
- https://arxiv.org/abs/2501.19393
- https://arxiv.org/pdf/2502.08235 
- llama.cpp：
  - 异构推理（ktransformers 特性移植）
  - 实现 https://flashinfer.ai/2025/03/10/sampling.html Sorting-Free GPU Kernels for LLM Sampling
  - 算子/层融合（awq）：llama.cpp 已经支持 flash-attn，所以得看其他算子。调研 autoawq 的方式是否可以有效用于 llama.cpp
  - dequant 和后续计算是否可以融合
  - fuse 官方也有一些讨论，例如：https://github.com/ggml-org/llama.cpp/pull/5413

另外，QServe 有新进展 https://github.com/mit-han-lab/omniserve  
这个实验室有很大精力在优化模型效率
还有一个方向是有没有可以用一些第三方的cuda 算子替换 ggml 的算子。例如 llama.cpp 引入了 flash_attn 算子替换之前对应的的 ggml 计算图部分。
https://arxiv.org/pdf/2411.02355

暂时无法在飞书文档外展示此内容

电路追踪：揭示语言模型中的计算图
https://transformer-circuits.pub/2025/attribution-graphs/methods.html

Cuda mode 教程
https://zhuanlan.zhihu.com/p/706469164

一口气刷完DeepSeek开源周！5天带你吃透FlashMLA、DeepEP、DeepGEMM、DualPipe&EPLB、3FS&smallpond等通俗易懂
https://www.bilibili.com/video/BV1e2XfYcEkH?spm_id_from=333.788.videopod.episodes&vd_source=e9d2c78cfdedcc7ab55152a4751709a1

计算机体系结构
https://www.bilibili.com/video/BV1sV411b7c1?spm_id_from=333.788.videopod.sections&vd_source=e9d2c78cfdedcc7ab55152a4751709a1

哲学随笔
https://zhuanlan.zhihu.com/p/110240540

T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge
https://www.arxiv.org/pdf/2407.00088

通向AGI
https://zhuanlan.zhihu.com/p/597586623
https://arxiv.org/pdf/2012.14913
https://arxiv.org/pdf/2202.05262

KTransformers 团队分享异构推理架构思路：基于内存的大模型推理成本优化
https://www.bilibili.com/video/BV1VNQrYGEad/?spm_id_from=333.337.search-card.all.click&vd_source=0db9e326710307dd821c44d742b119a8


大型语言模型中的超级权重
https://hjfy.top/arxiv/2411.07191
miniCPM4 的发布有一系列的技术，例如量化（SpecMQuant) 和推测解码 (FR-Spec) ，https://arxiv.org/html/2502.14856v2  FR-Spec 是对 eagle2 的改进  


https://arxiv.org/html/2407.03157v2  这篇论文用于实时编辑场景，本质是解决 KV 重用的，我在想能否用于改进推测解码的的 KV 重用？  

大语言模型推理加速：硬件视角的全面解析
https://zhuanlan.zhihu.com/p/1895888543127154854

DeepSeek推理最高提速6倍！开源研究：加装「思维进度条」，计算量减少30%
https://zhuanlan.zhihu.com/p/1925920556538115717?share_code=15feDAdY6hTgy&utm_psn=1925983651352524101