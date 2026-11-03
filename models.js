/* Official config.json snapshots, verified 2026-11-03. No network requests or model code. */
(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.ModelCatalog = api;
})(typeof window !== 'undefined' ? window : globalThis, function () {
  'use strict';

  const SNAPSHOT_DATE = '2026-11-03';
  // These dormant fields satisfy the simulator schema. They describe no MLA
  // dimensions in GQA models, and do not participate in their memory estimates.
  const inactiveMLA = { kvLoraRank: 512, qkRopeHeadDim: 64 };
  const gqaNote = 'GQA 模型中的 MLA latent / RoPE 字段仅为未启用的界面默认值，不参与此模型估算。';
  const approximationNote = '结构参数来自官方配置；容量仍采用本工具简化结构，不含 checkpoint 文件大小、激活、工作区及全部模型专有参数。';

  const LEGACY_MODELS = [
    {
      id: 'qwen3-8b', name: 'Qwen3-8B', organization: 'Qwen', architecture: 'Qwen3ForCausalLM',
      source: 'https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json',
      url: 'https://huggingface.co/Qwen/Qwen3-8B', verifiedAt: SNAPSHOT_DATE,
      config: {
        modelName: 'Qwen3-8B', attentionType: 'gqa', layers: 36, hiddenSize: 4096,
        heads: 32, kvHeads: 8, headDim: 128, intermediateSize: 12288,
        experts: 0, topK: 0, moeLayers: 0, sharedExperts: 0,
        moeIntermediateSize: 12288, moeLayerPlacement: 'uniform', vocabSize: 151936,
        precision: 'bf16', kvPrecision: 'bf16', ...inactiveMLA
      },
      notes: ['Dense 模型；所有专家相关数量均关闭，未启用的专家 FFN dim 沿用 Dense FFN dim。', gqaNote, approximationNote]
    },
    {
      id: 'qwen3-32b', name: 'Qwen3-32B', organization: 'Qwen', architecture: 'Qwen3ForCausalLM',
      source: 'https://huggingface.co/Qwen/Qwen3-32B/raw/main/config.json',
      url: 'https://huggingface.co/Qwen/Qwen3-32B', verifiedAt: SNAPSHOT_DATE,
      config: {
        modelName: 'Qwen3-32B', attentionType: 'gqa', layers: 64, hiddenSize: 5120,
        heads: 64, kvHeads: 8, headDim: 128, intermediateSize: 25600,
        experts: 0, topK: 0, moeLayers: 0, sharedExperts: 0,
        moeIntermediateSize: 25600, moeLayerPlacement: 'uniform', vocabSize: 151936,
        precision: 'bf16', kvPrecision: 'bf16', ...inactiveMLA
      },
      notes: ['官方 head_dim=128；Q 投影维度不等于 hidden_size，不能用 hidden_size / heads 替代。Dense 模型没有 MoE，专家 FFN dim 为未启用的 Dense 同值占位。', gqaNote, approximationNote]
    },
    {
      id: 'qwen3-30b-a3b', name: 'Qwen3-30B-A3B', organization: 'Qwen', architecture: 'Qwen3MoeForCausalLM',
      source: 'https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/config.json',
      url: 'https://huggingface.co/Qwen/Qwen3-30B-A3B', verifiedAt: SNAPSHOT_DATE,
      config: {
        modelName: 'Qwen3-30B-A3B', attentionType: 'gqa', layers: 48, hiddenSize: 2048,
        heads: 32, kvHeads: 4, headDim: 128, intermediateSize: 6144,
        experts: 128, topK: 8, moeLayers: 48, sharedExperts: 0,
        moeIntermediateSize: 768, moeLayerPlacement: 'uniform', vocabSize: 151936,
        precision: 'bf16', kvPrecision: 'bf16', ...inactiveMLA
      },
      notes: ['decoder_sparse_step=1 且 mlp_only_layers 为空：48 个 Transformer 层均为 MoE；没有共享专家。', '官方 head_dim=128，Q 投影维度与 hidden_size 不同。', gqaNote, approximationNote]
    },
    {
      id: 'qwen3-235b-a22b', name: 'Qwen3-235B-A22B', organization: 'Qwen', architecture: 'Qwen3MoeForCausalLM',
      source: 'https://huggingface.co/Qwen/Qwen3-235B-A22B/raw/main/config.json',
      url: 'https://huggingface.co/Qwen/Qwen3-235B-A22B', verifiedAt: SNAPSHOT_DATE,
      config: {
        modelName: 'Qwen3-235B-A22B', attentionType: 'gqa', layers: 94, hiddenSize: 4096,
        heads: 64, kvHeads: 4, headDim: 128, intermediateSize: 12288,
        experts: 128, topK: 8, moeLayers: 94, sharedExperts: 0,
        moeIntermediateSize: 1536, moeLayerPlacement: 'uniform', vocabSize: 151936,
        precision: 'bf16', kvPrecision: 'bf16', ...inactiveMLA
      },
      notes: ['decoder_sparse_step=1 且 mlp_only_layers 为空：94 个 Transformer 层均为 MoE；没有共享专家。', '官方 head_dim=128，Q 投影维度与 hidden_size 不同。', gqaNote, approximationNote]
    },
    {
      id: 'deepseek-v3', name: 'DeepSeek-V3', organization: 'deepseek-ai', architecture: 'DeepseekV3ForCausalLM',
      source: 'https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json',
      url: 'https://huggingface.co/deepseek-ai/DeepSeek-V3', verifiedAt: SNAPSHOT_DATE,
      config: {
        modelName: 'DeepSeek-V3', attentionType: 'mla', layers: 61, hiddenSize: 7168,
        heads: 128, kvHeads: 128, headDim: 192, intermediateSize: 18432,
        experts: 256, topK: 8, moeLayers: 58, sharedExperts: 1,
        moeIntermediateSize: 2048, moeLayerPlacement: 'trailing', vocabSize: 129280,
        precision: 'bf16', kvPrecision: 'bf16', kvLoraRank: 512, qkRopeHeadDim: 64
      },
      notes: [
        'first_k_dense_replace=3、moe_layer_freq=1：L0–L2 为 Dense，L3–L60 共 58 个 MoE 层；每层 256 个路由专家、1 个共享专家。',
        'MLA 的 KV 使用 512 latent + 64 RoPE 估算。headDim=192 来自 QK 的 128 non-RoPE + 64 RoPE；实际 V head dim=128。当前引擎用统一 head dim 的展开投影近似，未建模 q_lora_rank=1536 的低秩权重，Attention 容量不是该模型的真实参数统计。',
        '官方仓库含 FP8 量化配置；这里使用 BF16 理想位宽作为可调分析基线，不复现 FP8 checkpoint 的占用。',
        '只覆盖 61 个主干 Transformer 层，不包含配置中的额外 MTP / next-token prediction 层。', approximationNote
      ]
    },
    {
      id: 'glm-4.5', name: 'GLM-4.5', organization: 'zai-org', architecture: 'Glm4MoeForCausalLM',
      source: 'https://huggingface.co/zai-org/GLM-4.5/raw/main/config.json',
      url: 'https://huggingface.co/zai-org/GLM-4.5', verifiedAt: SNAPSHOT_DATE,
      config: {
        modelName: 'GLM-4.5', attentionType: 'gqa', layers: 92, hiddenSize: 5120,
        heads: 96, kvHeads: 8, headDim: 128, intermediateSize: 12288,
        experts: 160, topK: 8, moeLayers: 89, sharedExperts: 1,
        moeIntermediateSize: 1536, moeLayerPlacement: 'trailing', vocabSize: 151552,
        precision: 'bf16', kvPrecision: 'bf16', ...inactiveMLA
      },
      notes: [
        'first_k_dense_replace=3：L0–L2 为 Dense，L3–L91 共 89 个 MoE 层；每层 160 个路由专家、1 个共享专家。',
        '官方 head_dim=128、Q heads=96；Q 投影维度与 hidden_size 不同。Attention bias、QK norm 与额外 MTP 层未完整计入容量。',
        gqaNote, approximationNote
      ]
    }
  ];

  const partialNote = '部分结构估算：Attention 权重、KV / 线性注意力状态和专有辅助模块尚未建模；零表示未计入。合计只覆盖已建模部分，不能用于判断完整模型能否装入显存。';
  const glmDsa = name => ({
    modelName: name, attentionType: 'mla', layers: 78, hiddenSize: 6144,
    heads: 64, kvHeads: 64, headDim: 256, intermediateSize: 12288,
    experts: 256, topK: 8, moeLayers: 75, sharedExperts: 1,
    moeIntermediateSize: 2048, moeLayerPlacement: 'trailing', vocabSize: 154880,
    precision: 'bf16', kvPrecision: 'bf16', kvLoraRank: 512, qkRopeHeadDim: 64
  });
  const deepseekV4 = (name, layers, hiddenSize, heads, experts, moeIntermediateSize) => ({
    modelName: name, attentionType: 'unmodeled', layers, hiddenSize,
    heads, kvHeads: 1, headDim: 512, intermediateSize: moeIntermediateSize,
    experts, topK: 6, moeLayers: layers, sharedExperts: 1,
    moeIntermediateSize, moeLayerPlacement: 'trailing', vocabSize: 129280,
    precision: 'bf16', kvPrecision: 'bf16', ...inactiveMLA
  });
  const miniMax = (name, attentionType) => ({
    modelName: name, attentionType, layers: 60, hiddenSize: 6144,
    heads: 64, kvHeads: 4, headDim: 128, intermediateSize: 12288,
    experts: 128, topK: 4, moeLayers: 57, sharedExperts: 1,
    moeIntermediateSize: 3072, moeLayerPlacement: 'trailing', vocabSize: 200064,
    precision: 'bf16', kvPrecision: 'bf16', ...inactiveMLA
  });
  const CURRENT_MODELS = [
    {
      id: 'glm-5.2', name: 'GLM-5.2', organization: 'zai-org', architecture: 'GlmMoeDsaForCausalLM',
      source: 'https://huggingface.co/zai-org/GLM-5.2/raw/main/config.json', url: 'https://huggingface.co/zai-org/GLM-5.2',
      support: 'projection-proxy', attentionDescription: 'DSA / MLA：展开投影近似；稀疏索引器未计入',
      attentionLayers: Array(78).fill('DSA'), config: glmDsa('GLM-5.2'),
      notes: [
        '官方配置为 78 层，L0–L2 Dense，L3–L77 为 75 个 MoE 层；每层 256 个路由专家、Top-8、1 个共享专家。',
        'DSA 的有效 QK head dim=192 non-RoPE + 64 RoPE=256；配置的通用 head_dim=192 不能替代该值。MLA KV 按 512 latent + 64 RoPE 近似。',
        'Attention 权重采用展开投影代理；q_lora_rank=2048 的低秩路径、共享稀疏索引器及其缓存、额外 MTP 层没有精确建模。', approximationNote
      ]
    },
    {
      id: 'glm-5.3', name: 'GLM-5.3', organization: 'zai-org', architecture: 'GlmMoeDsaForCausalLM',
      source: 'https://huggingface.co/zai-org/GLM-5.3/raw/main/config.json', url: 'https://huggingface.co/zai-org/GLM-5.3',
      support: 'projection-proxy', attentionDescription: 'DSA / MLA：展开投影近似；稀疏索引器未计入',
      attentionLayers: Array(78).fill('DSA'), config: glmDsa('GLM-5.3'),
      notes: [
        '官方配置为 78 层，L0–L2 Dense，75 个 MoE 层；每层 256 个路由专家、Top-8、1 个共享专家。',
        '有效 QK head dim=192 non-RoPE + 64 RoPE=256；MLA KV 按 512 latent + 64 RoPE 近似。Attention 是展开投影代理，低秩路径、共享稀疏索引器及额外 MTP 未精确建模。',
        '官方配置包含 FP8 混合量化；BF16 是可调整的分析起点，不是发布 checkpoint 的实际占用。', approximationNote
      ]
    },
    {
      id: 'glm-5.3-flash', name: 'GLM-5.3-Flash', organization: 'zai-org', architecture: 'Glm5NextForConditionalGeneration',
      source: 'https://huggingface.co/zai-org/GLM-5.3-Flash/raw/main/config.json', url: 'https://huggingface.co/zai-org/GLM-5.3-Flash',
      support: 'partial', attentionDescription: '34 层 KDA + 11 层 DSA；Attention / KV 尚未建模',
      attentionLayers: Array.from({ length: 45 }, (_, i) => i % 4 === 3 ? 'DSA' : 'KDA'),
      config: {
        modelName: 'GLM-5.3-Flash', attentionType: 'unmodeled', layers: 45, hiddenSize: 4096,
        heads: 64, kvHeads: 64, headDim: 256, intermediateSize: 12288,
        experts: 288, topK: 8, moeLayers: 42, sharedExperts: 1,
        moeIntermediateSize: 2048, moeLayerPlacement: 'trailing', vocabSize: 154880,
        precision: 'bf16', kvPrecision: 'bf16', kvLoraRank: 512, qkRopeHeadDim: 0
      },
      notes: [
        '读取官方 text_config：45 个语言主干层，前三层 Dense，42 层 MoE；288 个路由专家、Top-8、1 个共享专家。',
        '混合 34 层线性注意力 KDA 与 11 层 DSA（L3、L7、…、L43）。heads=64、headDim=256 描述 DSA 分支，不能套用于 KDA 的 64×128 状态。',
        'mHC、视觉编码器、线性注意力卷积 / 状态与稀疏索引器均未计入。官方含 FP8 配置，BF16 起点不复现 checkpoint。', partialNote
      ]
    },
    {
      id: 'deepseek-v4-flash-0731', name: 'DeepSeek-V4-Flash-0731', organization: 'deepseek-ai', architecture: 'DeepseekV4ForCausalLM',
      source: 'https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/raw/main/config.json', url: 'https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731',
      support: 'partial', attentionDescription: 'SWA / CSA / HCA 压缩稀疏 Attention；状态尚未建模',
      attentionLayers: Array.from({ length: 43 }, (_, i) => i < 2 ? 'SWA' : i % 2 === 0 ? 'CSA×4' : 'HCA×128'),
      config: { ...deepseekV4('DeepSeek-V4-Flash-0731', 43, 4096, 64, 256, 2048), hashMoeLayers: 3 },
      notes: [
        '43 个主干层均为 MoE：256 路由专家、Top-6、1 个共享专家；前 3 个 MoE 层采用静态 token→expert Hash Router，EP 只展示逻辑专家分配。',
        '滑动窗口为 128；主干压缩比例为 0、0，随后 4 / 128 交替。压缩注意力、低秩投影、索引器、mHC 和 DSpark / 额外草稿模块未计入。',
        '官方未给 intermediate_size 或 kv_lora_rank：Dense FFN dim 以专家 FFN dim 作未启用占位，latent=512 仅为未启用的界面默认值。所有层均为 MoE，且 Attention / KV 不参与容量统计。',
        '发布 checkpoint 为 FP4 专家与 FP8 等混合位宽；BF16 是可调整分析起点。', partialNote
      ]
    },
    {
      id: 'deepseek-v4-pro', name: 'DeepSeek-V4-Pro', organization: 'deepseek-ai', architecture: 'DeepseekV4ForCausalLM',
      source: 'https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/raw/main/config.json', url: 'https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro',
      support: 'partial', attentionDescription: 'CSA / HCA 压缩稀疏 Attention；状态尚未建模',
      attentionLayers: Array.from({ length: 61 }, (_, i) => i < 2 || i % 2 ? 'HCA×128' : 'CSA×4'),
      config: { ...deepseekV4('DeepSeek-V4-Pro', 61, 7168, 128, 384, 3072), hashMoeLayers: 3 },
      notes: [
        '61 个主干层均为 MoE：384 路由专家、Top-6、1 个共享专家；前 3 个 MoE 层为 Hash Router，不计常规 Dense Router 矩阵，Hash 表占用尚未建模。',
        '滑动窗口为 128；前两层压缩比例为 128，随后 4 / 128 交替。压缩注意力、低秩投影、索引器、mHC 与额外草稿 / DSpark 模块未计入。',
        '官方未给 intermediate_size 或 kv_lora_rank；对应 Dense FFN / latent 字段为未启用的界面占位，不代表官方结构。',
        '官方混合量化 checkpoint 的占用不等于本工具 BF16 起点。', partialNote
      ]
    },
    {
      id: 'deepseek-v4.1-flash-1017', name: 'DeepSeek-V4.1-Flash-1017', organization: 'deepseek-ai', architecture: 'DeepseekV41ForCausalLM',
      source: 'https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/raw/main/config.json', url: 'https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017',
      support: 'partial', attentionDescription: 'CED：20 层编码器 + 20 层解码器，跨层共享 KV / 索引',
      attentionLayers: Array.from({ length: 40 }, (_, i) => i < 2 ? 'CED encoder · SWA' : i < 20 ? 'CED encoder · CSA×2' : 'CED decoder · shared KV'),
      config: deepseekV4('DeepSeek-V4.1-Flash-1017', 40, 5120, 64, 384, 2304),
      notes: [
        '读取官方 text_config：40 个语言主干层均为 MoE，384 路由专家、Top-6、1 个共享专家。该配置未声明 num_hash_layers，不能套用 V4 的前三层 Hash Router。',
        'CED 的 20 个编码层 + 20 个解码层采用跨层 KV / 索引共享；不能按普通逐层 GQA / MLA 计算缓存。',
        'Engram 条件记忆（官方约 196B 参数）、视觉编码器、mHC、压缩注意力及草稿 / DSpark 模块均未计入，因此这里只显示语言主干中已建模的部分。',
        'intermediate_size / kv_lora_rank 未在官方主干配置给出，界面值为未启用占位；BF16 只是可调分析起点。', partialNote
      ]
    },
    {
      id: 'kimi-k3', name: 'Kimi-K3', organization: 'moonshotai', architecture: 'KimiK3ForConditionalGeneration',
      source: 'https://huggingface.co/moonshotai/Kimi-K3/raw/main/config.json', url: 'https://huggingface.co/moonshotai/Kimi-K3',
      support: 'partial', attentionDescription: '69 层 KDA + 24 层 MLA；LatentMoE 专家维度 3584',
      attentionLayers: Array.from({ length: 93 }, (_, i) => (i + 1) % 4 === 0 || i === 92 ? 'MLA' : 'KDA'),
      config: {
        modelName: 'Kimi-K3', attentionType: 'unmodeled', layers: 93, hiddenSize: 7168,
        heads: 96, kvHeads: 96, headDim: 192, intermediateSize: 33792,
        experts: 896, topK: 16, moeLayers: 92, sharedExperts: 2,
        moeIntermediateSize: 3072, routedExpertHiddenSize: 3584, moeLayerPlacement: 'trailing', vocabSize: 163840,
        precision: 'bf16', kvPrecision: 'bf16', kvLoraRank: 512, qkRopeHeadDim: 64
      },
      notes: [
        '官方 text_config 为 93 层，L0 Dense，L1–L92 为 92 个 MoE 层；896 路由专家、Top-16、2 个共享专家。',
        '路由专家实际输入 / 输出维度为 routed_expert_hidden_size=3584，共享专家与 Router 使用 hidden_size=7168；额外适配投影按简化结构单独估算。',
        '官方 full_attn_layers 为从 1 开始的层号，转成界面 0-based 后为 L3、L7、…、L91、L92，共 24 层 MLA，其余 69 层 KDA。',
        'MLA 分支 QK dim=128+64，V dim=128；KDA 头维度、卷积状态、Attention Residual / 视觉模块没有计入。官方混合低位宽量化不等于 BF16 起点。', partialNote
      ]
    },
    {
      id: 'minimax-m3', name: 'MiniMax-M3', organization: 'MiniMaxAI', architecture: 'MiniMaxM3SparseForConditionalGeneration',
      source: 'https://huggingface.co/MiniMaxAI/MiniMax-M3/raw/main/config.json', url: 'https://huggingface.co/MiniMaxAI/MiniMax-M3',
      support: 'projection-proxy', attentionDescription: '前三层 GQA，57 层 MSA 稀疏 Attention；按 GQA 投影近似',
      attentionLayers: Array.from({ length: 60 }, (_, i) => i < 3 ? 'GQA' : 'MSA'),
      config: miniMax('MiniMax-M3', 'gqa'),
      notes: [
        '官方 text_config 为 60 层，前三层 FFN Dense，57 层 MoE；128 路由专家、Top-4、1 个共享专家。dense_intermediate_size=12288，专家与共享 FFN dim=3072。',
        'MSA 稀疏 Attention 用 GQA Q/K/V/O 投影代理；未计入稀疏索引器，缓存按完整 GQA 序列估算，不能当作 MSA 的真实缓存布局。',
        '视觉编码器、MTP / next-token 模块、QK norm 和模型专有结构未完整计入；BF16 是可调整分析起点。', gqaNote, approximationNote
      ]
    },
    {
      id: 'minimax-m3.1-flash', name: 'MiniMax-M3.1-Flash', organization: 'MiniMaxAI', architecture: 'MiniMaxM3SparseForConditionalGeneration',
      source: 'https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/raw/main/config.json', url: 'https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash',
      releasedAt: '2026-10-19', releaseSource: 'https://www.minimax.io/blog/minimax-m31f',
      support: 'partial', attentionDescription: '60 层 MSA 稀疏 Attention；Attention / KV 尚未建模',
      attentionLayers: Array(60).fill('MSA'), config: miniMax('MiniMax-M3.1-Flash', 'unmodeled'),
      notes: [
        '2026-10-19 正式发布。官方 text_config 为 60 层，所有层均为稀疏 Attention；前三层 FFN 仍为 Dense，其余 57 层 MoE。128 路由专家、Top-4、1 个共享专家。',
        '原生视觉 / 视频、DSpark、稀疏索引器和 Attention 状态尚未建模；不能将 MiniMax-M3 的前三层普通 Attention 布局直接套用。',
        '官方 NVFP4 专家与 MXFP8 Attention 等混合量化不由本工具 BF16 起点复现；MLA latent 字段是未启用的界面占位。', partialNote
      ]
    }
  ];

  const MODELS = [
    ...CURRENT_MODELS.map(model => ({ generation: 'current', verifiedAt: SNAPSHOT_DATE, ...model })),
    ...LEGACY_MODELS.map(model => ({ generation: 'legacy', support: 'projection-proxy', attentionDescription: model.config.attentionType === 'mla' ? 'MLA：展开投影近似' : 'GQA：统一 Q/K/V/O 投影模型', ...model }))
  ].map(model => Object.freeze({
    ...model,
    config: Object.freeze({ routedExpertHiddenSize: 0, hashMoeLayers: 0, ...model.config }),
    ...(model.attentionLayers ? { attentionLayers: Object.freeze(model.attentionLayers) } : {}),
    notes: Object.freeze(model.notes)
  }));

  Object.freeze(MODELS);
  const getModel = id => MODELS.find(model => model.id === id);
  return Object.freeze({ MODELS, SNAPSHOT_DATE, getModel });
});
