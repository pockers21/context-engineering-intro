# KV-Compress集成策略

## 技术背景

**论文**：KV-Compress: Paged KV-Cache Compression with Variable Compression Rates per Attention Head  
**链接**：https://arxiv.org/html/2410.00161v2

**核心创新**：
- 不同注意力头的变压缩率
- 分页内存管理
- 动态压缩策略
- **性能收益**：60-80%KV缓存内存节省

## 当前KV Cache架构分析

### 现有llama_kv_cache_unified架构
从`/src/llama-kv-cache-unified.cpp`分析可以看到：

#### 核心数据结构
```cpp
class llama_kv_cache_unified : public llama_memory_i {
    struct layer_data {
        uint32_t il;  // layer index
        ggml_tensor * k;  // K cache tensor [n_embd_k_gqa, kv_size, n_stream]
        ggml_tensor * v;  // V cache tensor [n_embd_v_gqa, kv_size, n_stream]
        std::vector<ggml_tensor *> k_stream;  // per-stream K views
        std::vector<ggml_tensor *> v_stream;  // per-stream V views
    };
    
    std::vector<layer_data> layers;          // 每层的KV缓存
    std::vector<llama_kv_cells> v_cells;     // 缓存单元管理
    std::vector<uint32_t> v_heads;           // 每个流的头指针
    uint32_t n_stream;                       // 流数量
    uint32_t n_seq_max;                      // 最大序列数
};
```

#### 现有内存管理模式
- **分层存储**：每层独立管理K、V缓存
- **流式设计**：支持多序列并行推理
- **统一格式**：所有头使用相同的数据类型和布局
- **连续分配**：大块连续内存分配模式

### KV-Compress集成挑战
1. **头间差异化压缩**：需要为不同注意力头设置不同压缩率
2. **动态压缩决策**：运行时根据头重要性调整压缩策略
3. **分页内存管理**：实现高效的压缩页面管理
4. **解压缩开销**：平衡压缩率和计算开销

## KV-Compress集成设计

### 1. 压缩KV Cache架构扩展

#### 头级别压缩参数结构
```cpp
// 新增文件 /src/llama-kv-compress.h
#pragma once

#include "llama-kv-cache-unified.h"
#include <memory>
#include <unordered_map>

// 每个注意力头的压缩配置
struct kv_head_compress_config {
    float compression_ratio;     // 压缩率 [0.1, 1.0]
    uint32_t min_tokens;        // 最小保留token数
    uint32_t max_tokens;        // 最大保留token数  
    float importance_threshold; // 重要性阈值
    bool enable_dynamic_adjust; // 是否启用动态调整
};

// 压缩页面管理
struct kv_compressed_page {
    uint32_t page_id;           // 页面ID
    uint32_t layer_id;          // 层ID
    uint32_t head_id;           // 头ID
    uint32_t seq_start;         // 起始序列位置
    uint32_t seq_end;           // 结束序列位置
    uint32_t compressed_size;   // 压缩后大小
    uint32_t original_size;     // 原始大小
    void* compressed_data;      // 压缩数据指针
    float* importance_scores;   // 重要性分数
    uint64_t timestamp;         // 时间戳
};

// KV压缩管理器
class llama_kv_compress_manager {
private:
    // 头级别压缩配置
    std::vector<std::vector<kv_head_compress_config>> head_configs; // [layer][head]
    
    // 压缩页面池
    std::vector<std::unique_ptr<kv_compressed_page>> compressed_pages;
    std::unordered_map<uint64_t, size_t> page_lookup; // page_key -> page_index
    
    // 重要性评估器
    std::unique_ptr<class kv_importance_evaluator> importance_evaluator;
    
    // 压缩算法引擎  
    std::unique_ptr<class kv_compression_engine> compression_engine;
    
    // 统计信息
    struct {
        size_t total_compressed_bytes;
        size_t total_original_bytes;
        size_t num_compressions;
        size_t num_decompressions;
        float average_compression_ratio;
        double total_compression_time;
        double total_decompression_time;
    } stats;

public:
    llama_kv_compress_manager(uint32_t n_layers, const std::vector<uint32_t>& n_heads_per_layer);
    
    // 配置管理
    void set_head_compression_config(uint32_t layer, uint32_t head, 
                                   const kv_head_compress_config& config);
    kv_head_compress_config get_head_compression_config(uint32_t layer, uint32_t head) const;
    
    // 压缩操作
    bool should_compress(uint32_t layer, uint32_t head, uint32_t seq_len) const;
    size_t compress_kv_data(uint32_t layer, uint32_t head,
                           const void* k_data, const void* v_data,
                           uint32_t seq_start, uint32_t seq_end,
                           void** compressed_data_out);
    
    // 解压缩操作
    bool decompress_kv_data(uint64_t page_key,
                           void** k_data_out, void** v_data_out,
                           uint32_t* seq_len_out);
    
    // 重要性评估
    void update_importance_scores(uint32_t layer, uint32_t head,
                                const float* attention_weights,
                                uint32_t seq_len);
    
    // 动态调整
    void adjust_compression_strategy(uint32_t layer, uint32_t head,
                                   float performance_feedback);
    
    // 内存管理
    void garbage_collect(float memory_pressure_threshold = 0.8f);
    size_t get_memory_usage() const;
    size_t get_compressed_memory_savings() const;
    
    // 统计信息
    void get_compression_stats(/* output stats */);
};
```

#### 重要性评估器
```cpp
// 重要性评估算法
class kv_importance_evaluator {
private:
    // 历史注意力权重记录
    struct attention_history {
        std::vector<float> weights_history;  // 历史权重
        std::vector<uint64_t> timestamps;    // 时间戳
        float moving_average;                // 移动平均
        float variance;                      // 方差
    };
    
    std::vector<std::vector<std::vector<attention_history>>> history; // [layer][head][seq_pos]
    
    // 评估参数
    struct eval_params {
        float temporal_decay = 0.95f;       // 时间衰减因子
        float frequency_weight = 0.3f;      // 频率权重
        float recency_weight = 0.4f;        // 最近性权重  
        float magnitude_weight = 0.3f;      // 幅度权重
        uint32_t history_window = 1000;     // 历史窗口大小
    } params;

public:
    kv_importance_evaluator(uint32_t n_layers, const std::vector<uint32_t>& n_heads);
    
    // 更新注意力权重历史
    void update_attention_weights(uint32_t layer, uint32_t head,
                                const float* weights, uint32_t seq_len);
    
    // 计算token重要性分数
    std::vector<float> compute_importance_scores(uint32_t layer, uint32_t head,
                                               uint32_t seq_start, uint32_t seq_end);
    
    // 选择要压缩的token
    std::vector<bool> select_tokens_for_compression(uint32_t layer, uint32_t head,
                                                   const std::vector<float>& importance_scores,
                                                   float compression_ratio);
    
    // 自适应阈值调整
    void adapt_importance_threshold(uint32_t layer, uint32_t head,
                                  float performance_feedback);
    
private:
    float compute_temporal_importance(const attention_history& hist, uint64_t current_time);
    float compute_frequency_importance(const attention_history& hist);
    float compute_magnitude_importance(const attention_history& hist);
};
```

#### 压缩引擎
```cpp
// KV数据压缩引擎
class kv_compression_engine {
public:
    enum compression_algorithm {
        ALGORITHM_QUANTIZATION,    // 量化压缩
        ALGORITHM_SPARSIFICATION,  // 稀疏化
        ALGORITHM_LOW_RANK,        // 低秩近似
        ALGORITHM_HYBRID           // 混合算法
    };
    
private:
    compression_algorithm current_algorithm;
    
    // 量化压缩器
    struct quantization_compressor {
        uint8_t target_bits = 4;        // 目标比特数
        float quantization_scale;       // 量化缩放
        int8_t zero_point;              // 零点
        
        size_t compress(const float* data, uint32_t size, void** output);
        void decompress(const void* compressed, uint32_t compressed_size, 
                       float* output, uint32_t original_size);
    } quant_compressor;
    
    // 稀疏化压缩器
    struct sparsification_compressor {
        float sparsity_ratio = 0.7f;    // 稀疏率
        uint32_t* sparse_indices;       // 稀疏索引
        float* sparse_values;           // 稀疏值
        
        size_t compress(const float* data, uint32_t size, 
                       const std::vector<bool>& keep_mask, void** output);
        void decompress(const void* compressed, uint32_t compressed_size,
                       float* output, uint32_t original_size);
    } sparse_compressor;
    
    // 低秩近似压缩器
    struct lowrank_compressor {
        uint32_t rank = 32;             // 近似秩
        float* U_matrix;                // U矩阵
        float* V_matrix;                // V矩阵
        
        size_t compress(const float* data, uint32_t rows, uint32_t cols, void** output);
        void decompress(const void* compressed, uint32_t compressed_size,
                       float* output, uint32_t rows, uint32_t cols);
    } lowrank_compressor;

public:
    kv_compression_engine(compression_algorithm algo = ALGORITHM_HYBRID);
    
    // 压缩K、V数据
    size_t compress_kv_pair(const float* k_data, const float* v_data,
                           uint32_t seq_len, uint32_t head_dim,
                           const std::vector<bool>& keep_mask,
                           void** compressed_k, void** compressed_v);
    
    // 解压缩K、V数据
    bool decompress_kv_pair(const void* compressed_k, const void* compressed_v,
                          uint32_t compressed_k_size, uint32_t compressed_v_size,
                          uint32_t seq_len, uint32_t head_dim,
                          float** k_data_out, float** v_data_out);
    
    // 选择最优压缩算法
    compression_algorithm select_optimal_algorithm(const float* sample_data,
                                                  uint32_t size,
                                                  float target_compression_ratio);
    
    // 压缩质量评估
    float evaluate_compression_quality(const float* original, const float* reconstructed,
                                     uint32_t size);
};
```

### 2. 扩展llama_kv_cache_unified

#### 压缩缓存集成
```cpp
// 修改 /src/llama-kv-cache-unified.h

class llama_kv_cache_unified_compressed : public llama_kv_cache_unified {
private:
    // 压缩管理器
    std::unique_ptr<llama_kv_compress_manager> compress_manager;
    
    // 压缩缓存层
    struct compressed_layer_data : public layer_data {
        // 压缩状态跟踪
        std::vector<std::vector<bool>> is_compressed; // [stream][head]
        std::vector<std::vector<uint64_t>> compressed_page_keys; // [stream][head]
        std::vector<std::vector<uint32_t>> original_seq_lens; // [stream][head]
        
        // 压缩统计
        std::vector<std::vector<float>> compression_ratios; // [stream][head] 
        std::vector<std::vector<uint64_t>> last_access_times; // [stream][head]
    };
    
    std::vector<compressed_layer_data> compressed_layers;
    
    // 压缩策略参数
    struct compression_strategy {
        bool enable_compression = true;
        float memory_pressure_threshold = 0.75f;  // 内存压力阈值
        uint32_t min_seq_len_for_compression = 512;  // 最小压缩序列长度
        uint32_t compression_check_interval = 100;   // 压缩检查间隔
        float performance_degradation_threshold = 0.05f; // 性能下降阈值
    } compression_strategy;
    
    // 运行时状态
    uint64_t global_timestamp = 0;
    uint32_t operations_since_last_gc = 0;

public:
    llama_kv_cache_unified_compressed(
        const llama_model & model,
        layer_filter_cb && filter,
        ggml_type type_k,
        ggml_type type_v,
        bool v_trans,
        bool offload,
        bool unified,
        uint32_t kv_size,
        uint32_t n_seq_max,
        uint32_t n_pad,
        uint32_t n_swa,
        llama_swa_type swa_type,
        const compression_strategy& comp_strategy = {});
    
    // 重写基类方法以支持压缩
    void update(
        const llama_batch & batch,
        slot_info_vec_t & slot_info,
        const std::vector<ggml_tensor *> & k_in,
        const std::vector<ggml_tensor *> & v_in) override;
    
    void find_slot_info(
        slot_info_vec_t & slot_info,
        const llama_batch & batch) override;
    
    // 压缩相关方法
    bool try_compress_layer_head(uint32_t layer, uint32_t stream, uint32_t head);
    bool decompress_layer_head(uint32_t layer, uint32_t stream, uint32_t head);
    
    void set_head_compression_config(uint32_t layer, uint32_t head,
                                   const kv_head_compress_config& config);
    
    // 性能监控
    void update_attention_importance(uint32_t layer,
                                   const std::vector<ggml_tensor*>& attention_weights);
    
    // 内存管理
    void trigger_compression_gc();
    float get_memory_pressure() const;
    
    // 统计信息
    struct compression_stats {
        size_t total_memory_saved;
        size_t total_compressed_data;
        float average_compression_ratio;
        uint32_t num_compressed_heads;
        uint32_t total_heads;
        double total_compression_time_ms;
        double total_decompression_time_ms;
    };
    
    compression_stats get_compression_statistics() const;
};
```

#### 压缩缓存更新逻辑
```cpp
// 实现 /src/llama-kv-cache-unified-compressed.cpp

void llama_kv_cache_unified_compressed::update(
    const llama_batch & batch,
    slot_info_vec_t & slot_info,
    const std::vector<ggml_tensor *> & k_in,
    const std::vector<ggml_tensor *> & v_in) {
    
    ++global_timestamp;
    ++operations_since_last_gc;
    
    // 执行标准KV缓存更新
    llama_kv_cache_unified::update(batch, slot_info, k_in, v_in);
    
    // 检查是否需要压缩
    if (compression_strategy.enable_compression && 
        operations_since_last_gc >= compression_strategy.compression_check_interval) {
        
        float memory_pressure = get_memory_pressure();
        
        if (memory_pressure > compression_strategy.memory_pressure_threshold) {
            // 尝试压缩一些头
            try_compress_candidates();
        }
        
        operations_since_last_gc = 0;
    }
}

void llama_kv_cache_unified_compressed::try_compress_candidates() {
    std::vector<compression_candidate> candidates;
    
    // 收集压缩候选
    for (uint32_t il = 0; il < compressed_layers.size(); ++il) {
        const auto& layer = compressed_layers[il];
        
        for (uint32_t s = 0; s < n_stream; ++s) {
            for (uint32_t h = 0; h < hparams.n_head_kv(il); ++h) {
                if (!layer.is_compressed[s][h] && 
                    should_compress_head(il, s, h)) {
                    
                    float priority = compute_compression_priority(il, s, h);
                    candidates.push_back({il, s, h, priority});
                }
            }
        }
    }
    
    // 按优先级排序
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return a.priority > b.priority; });
    
    // 压缩高优先级候选
    size_t compressed_count = 0;
    size_t max_compressions = candidates.size() * 0.2; // 每次最多压缩20%
    
    for (const auto& candidate : candidates) {
        if (compressed_count >= max_compressions) break;
        
        if (try_compress_layer_head(candidate.layer, candidate.stream, candidate.head)) {
            ++compressed_count;
        }
    }
}

bool llama_kv_cache_unified_compressed::try_compress_layer_head(
    uint32_t layer, uint32_t stream, uint32_t head) {
    
    auto& comp_layer = compressed_layers[layer];
    
    if (comp_layer.is_compressed[stream][head]) {
        return false; // 已经压缩
    }
    
    // 获取K、V数据
    const auto& layer_data = layers[map_layer_ids[layer]];
    ggml_tensor* k_tensor = layer_data.k_stream[stream];
    ggml_tensor* v_tensor = layer_data.v_stream[stream];
    
    // 计算头在张量中的偏移
    uint32_t head_dim = hparams.n_embd_head_k(layer);
    uint32_t head_offset = head * head_dim;
    uint32_t seq_len = v_heads[stream]; // 当前序列长度
    
    if (seq_len < compression_strategy.min_seq_len_for_compression) {
        return false; // 序列太短，不值得压缩
    }
    
    // 提取这个头的K、V数据
    std::vector<float> k_data(head_dim * seq_len);
    std::vector<float> v_data(head_dim * seq_len);
    
    extract_head_data(k_tensor, v_tensor, head, k_data.data(), v_data.data());
    
    // 获取重要性分数并确定保留掩码
    auto importance_scores = compress_manager->get_importance_evaluator()
        ->compute_importance_scores(layer, head, 0, seq_len);
    
    auto keep_mask = compress_manager->get_importance_evaluator()
        ->select_tokens_for_compression(layer, head, importance_scores,
                                       compress_manager->get_head_compression_config(layer, head).compression_ratio);
    
    // 执行压缩
    void* compressed_k = nullptr;
    void* compressed_v = nullptr;
    
    size_t compressed_size = compress_manager->get_compression_engine()
        ->compress_kv_pair(k_data.data(), v_data.data(), seq_len, head_dim,
                          keep_mask, &compressed_k, &compressed_v);
    
    if (compressed_size == 0) {
        return false; // 压缩失败
    }
    
    // 创建压缩页面
    uint64_t page_key = generate_page_key(layer, stream, head, global_timestamp);
    
    auto compressed_page = std::make_unique<kv_compressed_page>();
    compressed_page->page_id = page_key;
    compressed_page->layer_id = layer;
    compressed_page->head_id = head;
    compressed_page->seq_start = 0;
    compressed_page->seq_end = seq_len;
    compressed_page->compressed_size = compressed_size;
    compressed_page->original_size = head_dim * seq_len * 2 * sizeof(float);
    compressed_page->compressed_data = compressed_k; // 包含K和V的压缩数据
    compressed_page->timestamp = global_timestamp;
    
    // 更新压缩状态
    comp_layer.is_compressed[stream][head] = true;
    comp_layer.compressed_page_keys[stream][head] = page_key;
    comp_layer.original_seq_lens[stream][head] = seq_len;
    comp_layer.compression_ratios[stream][head] = 
        (float)compressed_size / (float)compressed_page->original_size;
    comp_layer.last_access_times[stream][head] = global_timestamp;
    
    // 清空原始缓存数据（可选，用于节省内存）
    clear_head_data(k_tensor, v_tensor, head);
    
    return true;
}

bool llama_kv_cache_unified_compressed::decompress_layer_head(
    uint32_t layer, uint32_t stream, uint32_t head) {
    
    auto& comp_layer = compressed_layers[layer];
    
    if (!comp_layer.is_compressed[stream][head]) {
        return true; // 未压缩，无需解压
    }
    
    uint64_t page_key = comp_layer.compressed_page_keys[stream][head];
    
    // 从压缩管理器解压缩数据
    void* k_data = nullptr;
    void* v_data = nullptr;
    uint32_t seq_len = 0;
    
    if (!compress_manager->decompress_kv_data(page_key, &k_data, &v_data, &seq_len)) {
        return false; // 解压缩失败
    }
    
    // 恢复到KV缓存张量
    const auto& layer_data = layers[map_layer_ids[layer]];
    restore_head_data(layer_data.k_stream[stream], layer_data.v_stream[stream],
                     head, (float*)k_data, (float*)v_data, seq_len);
    
    // 更新状态
    comp_layer.is_compressed[stream][head] = false;
    comp_layer.compressed_page_keys[stream][head] = 0;
    comp_layer.last_access_times[stream][head] = global_timestamp;
    
    // 清理临时解压缩数据
    free(k_data);
    free(v_data);
    
    return true;
}
```

### 3. 自适应压缩策略

#### 动态压缩率调整
```cpp
// 自适应压缩策略
class adaptive_compression_strategy {
private:
    struct head_performance_metrics {
        float attention_utilization;    // 注意力利用率
        float compression_benefit;      // 压缩收益
        float decompression_cost;       // 解压缩成本
        float accuracy_impact;          // 精度影响
        uint32_t access_frequency;      // 访问频率
        uint64_t last_update_time;     // 最后更新时间
    };
    
    std::vector<std::vector<head_performance_metrics>> head_metrics; // [layer][head]
    
    // 全局策略参数
    struct strategy_params {
        float base_compression_ratio = 0.5f;    // 基础压缩率
        float max_compression_ratio = 0.9f;     // 最大压缩率  
        float min_compression_ratio = 0.1f;     // 最小压缩率
        float adaptation_rate = 0.1f;           // 适应速率
        float performance_weight = 0.6f;        // 性能权重
        float memory_weight = 0.4f;             // 内存权重
    } params;

public:
    adaptive_compression_strategy(uint32_t n_layers, const std::vector<uint32_t>& n_heads);
    
    // 更新头性能指标
    void update_head_metrics(uint32_t layer, uint32_t head,
                           float attention_scores_entropy,
                           float memory_pressure,
                           float inference_latency);
    
    // 计算最优压缩率
    float compute_optimal_compression_ratio(uint32_t layer, uint32_t head);
    
    // 全局策略调整
    void adjust_global_strategy(float overall_performance_change,
                              float memory_usage_change);
    
private:
    float compute_attention_utilization(const float* attention_weights, uint32_t seq_len);
    float estimate_compression_benefit(uint32_t layer, uint32_t head, float compression_ratio);
    float estimate_decompression_cost(uint32_t layer, uint32_t head, uint32_t seq_len);
};
```

### 4. 实施时间表

#### Phase 3A (Week 13-14): 核心架构开发
- **Week 13**：
  - 压缩管理器基础架构实现
  - 重要性评估器开发
  - 基础压缩算法集成
- **Week 14**：
  - 分页内存管理实现
  - 压缩引擎优化
  - 单元测试开发

#### Phase 3B (Week 15): 系统集成
- 扩展llama_kv_cache_unified支持压缩
- 实现压缩/解压缩调度逻辑
- 自适应策略算法开发

#### Phase 3C (Week 16): 优化和验证
- 压缩性能调优
- 多模型兼容性测试
- 内存节省效果验证

### 5. 风险控制

#### 精度保护机制
```cpp
// 精度监控和保护
class compression_quality_monitor {
private:
    struct quality_metrics {
        float attention_score_mse;      // 注意力分数MSE
        float output_deviation;         // 输出偏差
        float perplexity_change;        // 困惑度变化
        uint32_t quality_violations;   // 质量违规次数
    };
    
    std::vector<std::vector<quality_metrics>> head_quality; // [layer][head]
    
    const float quality_thresholds[3] = {0.01f, 0.05f, 0.1f}; // 质量阈值

public:
    bool validate_compression_quality(uint32_t layer, uint32_t head,
                                    const float* original_attention,
                                    const float* compressed_attention,
                                    uint32_t seq_len);
    
    void trigger_quality_fallback(uint32_t layer, uint32_t head);
};
```

### 6. 预期收益

#### 内存优化指标
- **KV缓存内存节省**：60-80%
- **总内存使用降低**：30-50%
- **支持更长序列**：4K → 16K+ tokens

#### 性能指标
- **压缩开销**：<5%推理延迟增加
- **解压缩延迟**：<2ms per head
- **动态调整效果**：自适应压缩率优化

#### 质量保证
- **精度损失控制**：<3% perplexity增加  
- **注意力模式保持**：>95%重要注意力权重保留
- **长文本处理能力**：支持32K+ token序列

这个KV-Compress集成策略提供了完整的分页KV缓存压缩解决方案，通过头级别的差异化压缩和自适应策略，在显著节省内存的同时保持了模型的推理质量。

<function_calls>
<invoke name="Glob">
<parameter name="pattern">**/llama*kv*cache*.cpp