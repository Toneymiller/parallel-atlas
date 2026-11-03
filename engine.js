/* Parallel Atlas: deterministic, framework-aware educational placement engine. */
(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.ParallelEngine = api;
})(typeof window !== 'undefined' ? window : globalThis, function () {
  'use strict';

  const DEFAULT_CONFIG = Object.freeze({
    profile: 'generic', modelName: 'MoE · 教学模型 · 64 experts', attentionType: 'gqa',
    layers: 32, hiddenSize: 4096, heads: 32, kvHeads: 8, headDim: 128,
    experts: 64, topK: 4, intermediateSize: 11008, moeIntermediateSize: 2048, routedExpertHiddenSize: 0,
    moeLayers: 24, moeLayerPlacement: 'uniform', hashMoeLayers: 0, sharedExperts: 1, vocabSize: 32000, sequenceLength: 8192,
    batchSize: 8, precision: 'bf16', kvPrecision: 'bf16', kvLoraRank: 512,
    qkRopeHeadDim: 64, tp: 4, pp: 1, dp: 2, cp: 1, pcp: 1, dcp: 1,
    ep: 8, etp: 1, sp: false, gpusPerNode: 8, memoryGB: 80, hardwareId: 'custom',
    expertPlacement: 'contiguous'
  });

  const PRESETS = Object.freeze([
    { id: 'moe-eight', name: 'MoE · TP4 + DP2 + EP8', description: '8 张卡：Attention 按 TP 切分，64 个专家分布在 EP8。', config: { ...DEFAULT_CONFIG } },
    { id: 'dense-tp', name: 'Dense · TP4', description: '4 张卡：观察 Q/K/V/O 与稠密 FFN 的张量切分。', config: { ...DEFAULT_CONFIG, modelName: 'Dense GQA · 教学模型', tp: 4, dp: 1, ep: 1, etp: 4, experts: 0, topK: 0, moeLayers: 0, sharedExperts: 0 } },
    { id: 'context', name: '长上下文 · TP2 + CP4', description: '8 张卡：独立 CP 轴切分序列，权重在 CP 卡间复制。', config: { ...DEFAULT_CONFIG, modelName: 'Long-context GQA · 教学模型', tp: 2, dp: 1, cp: 4, ep: 2, etp: 1, sequenceLength: 65536, batchSize: 2 } },
    { id: 'pipeline', name: '流水线 · TP2 + PP4', description: '8 张卡：按层划分 4 个流水级，级间传输激活。', config: { ...DEFAULT_CONFIG, tp: 2, pp: 4, dp: 1, ep: 2, etp: 1 } },
    { id: 'ascend-pcp', name: 'Ascend · TP4 + PCP2 + DCP2', description: '8 张卡：PCP 增加进程；DCP 复用 PCP 卡切分 KV。', config: { ...DEFAULT_CONFIG, profile: 'vllm-ascend', tp: 4, dp: 1, pcp: 2, dcp: 2, ep: 8, etp: 1, sequenceLength: 32768 } },
    { id: 'decode', name: 'Decode · TP8 + DCP2', description: '8 张卡：GQA 的 KV 头在 TP 中复制，DCP 切分缓存序列。', config: { ...DEFAULT_CONFIG, profile: 'vllm-ascend', modelName: 'GQA · 教学模型', tp: 8, dp: 1, kvHeads: 4, dcp: 2, ep: 8, etp: 1, sequenceLength: 32768 } },
    { id: 'expert-tensor', name: 'Hybrid · EP4 × ETP2 + SP', description: '8 张卡：专家分配到 EP4，同一专家再由 ETP2 切分。', config: { ...DEFAULT_CONFIG, tp: 4, dp: 2, ep: 4, etp: 2, sp: true } },
    { id: 'mla', name: 'MLA · TP2 + PCP2 + DCP4', description: '4 张卡：DCP 复用完整 TP×PCP 网格，展示压缩 latent KV。', config: { ...DEFAULT_CONFIG, profile: 'vllm-ascend', modelName: 'MLA · 拓扑教学模型', attentionType: 'mla', tp: 2, dp: 1, pcp: 2, dcp: 4, ep: 4, etp: 1, sequenceLength: 65536, batchSize: 2 } }
  ]);

  const SOURCES = Object.freeze([
    { title: 'vLLM · Parallelism and Scaling', url: 'https://docs.vllm.ai/en/latest/serving/parallelism_scaling/' },
    { title: 'vLLM · Data Parallel Deployment', url: 'https://docs.vllm.ai/en/v0.13.0/serving/data_parallel_deployment/' },
    { title: 'vLLM Ascend · Context Parallel Guide', url: 'https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/context_parallel.html' },
    { title: 'vLLM Ascend · Context Parallel Design', url: 'https://docs.vllm.ai/projects/ascend/en/main/developer_guide/Design_Documents/context_parallel.html' },
    { title: 'vLLM Ascend · Sequence Parallelism', url: 'https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/sequence_parallelism.html' },
    { title: 'vLLM · Fused MoE Parallel Configuration', url: 'https://docs.vllm.ai/en/v0.25.0/api/vllm/model_executor/layers/fused_moe/' },
    { title: 'vLLM · Expert Map Manager', url: 'https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/fused_moe/expert_map_manager/' },
    { title: 'DeepSeek-V4 · Official Gate and Hash Routing Implementation', url: 'https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/raw/main/inference/model.py' },
    { title: 'NVIDIA Megatron Core · Context Parallel', url: 'https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html' },
    { title: 'NVIDIA Megatron Core · Parallelism Strategies', url: 'https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html' }
  ]);

  const RANGES = {
    layers: [1, 256], hiddenSize: [1, 65536], heads: [1, 512], kvHeads: [1, 512], headDim: [1, 1024],
    experts: [0, 2048], topK: [0, 64], intermediateSize: [1, 262144], moeIntermediateSize: [1, 262144], routedExpertHiddenSize: [0, 65536],
    moeLayers: [0, 256], hashMoeLayers: [0, 256], sharedExperts: [0, 64], vocabSize: [1, 2000000], sequenceLength: [1, 4194304],
    batchSize: [1, 4096], kvLoraRank: [1, 8192], qkRopeHeadDim: [0, 1024],
    tp: [1, 512], pp: [1, 256], dp: [1, 512], cp: [1, 512], pcp: [1, 512], dcp: [1, 512],
    ep: [1, 512], etp: [1, 512], gpusPerNode: [1, 512], memoryGB: [1, 16384]
  };
  const ENUMS = {
    profile: ['generic', 'vllm-ascend'], precision: ['bf16', 'fp16', 'int8', 'int4'],
    kvPrecision: ['bf16', 'fp16', 'int8'], attentionType: ['gqa', 'mla', 'unmodeled'], expertPlacement: ['contiguous', 'round-robin'],
    moeLayerPlacement: ['uniform', 'trailing']
  };
  const WIDTHS = { bf16: 2, fp16: 2, int8: 1, int4: 0.5 };
  const GIB = 1024 ** 3;

  function normalizeConfig(input) {
    if (input === undefined) input = {};
    if (!input || typeof input !== 'object' || Array.isArray(input)) throw Object.assign(new Error('配置必须是 JSON 对象。'), { errors: ['配置必须是 JSON 对象。'] });
    const result = {};
    const errors = [];
    for (const key of Object.keys(DEFAULT_CONFIG)) {
      const own = Object.prototype.hasOwnProperty.call(input, key);
      const raw = own ? input[key] : DEFAULT_CONFIG[key];
      if (key in RANGES) {
        const n = typeof raw === 'number' ? raw : typeof raw === 'string' && raw.trim() !== '' ? Number(raw) : NaN;
        const [min, max] = RANGES[key];
        if (!Number.isFinite(n) || !Number.isInteger(n) || n < min || n > max) errors.push(`${key} 必须是 ${min}–${max} 之间的整数。`);
        else result[key] = n;
      } else if (key in ENUMS) {
        if (!ENUMS[key].includes(raw)) errors.push(`${key} 必须是 ${ENUMS[key].join(' / ')}。`);
        else result[key] = raw;
      } else if (key === 'sp') {
        if (typeof raw !== 'boolean') errors.push('sp 必须是布尔值。');
        else result[key] = raw;
      } else if (key === 'modelName') {
        if (typeof raw !== 'string' || !raw.trim() || raw.length > 120) errors.push('modelName 必须是 1–120 字符的名称。');
        else result[key] = raw.trim();
      } else if (key === 'hardwareId') {
        if (typeof raw !== 'string' || !/^[a-z0-9-]{1,64}$/.test(raw)) errors.push('hardwareId 必须是 1–64 字符的小写字母、数字或连字符。');
        else result[key] = raw;
      }
    }
    if (errors.length) throw Object.assign(new Error(errors.join('\n')), { errors });
    return result;
  }

  const span = (total, partitions, index) => ({ start: Math.floor(total * index / partitions), end: Math.floor(total * (index + 1) / partitions) - 1, count: Math.floor(total * (index + 1) / partitions) - Math.floor(total * index / partitions) });
  const expertSpan = (total, partitions, index) => {
    const base = Math.floor(total / partitions), remainder = total % partitions;
    const count = base + (index < remainder ? 1 : 0), start = index * base + Math.min(index, remainder);
    return { start, end: start + count - 1, count };
  };
  const ints = (start, end) => Array.from({ length: Math.max(0, end - start + 1) }, (_, i) => start + i);
  const isMoeLayer = (layer, cfg) => cfg.moeLayers > 0 && (cfg.moeLayerPlacement === 'trailing'
    ? layer >= cfg.layers - cfg.moeLayers
    : Math.floor((layer + 1) * cfg.moeLayers / cfg.layers) > Math.floor(layer * cfg.moeLayers / cfg.layers));

  function analyze(input) {
    const errors = [], warnings = [];
    let config;
    try { config = normalizeConfig(input); } catch (error) {
      return { config: null, valid: false, errors: error.errors || [error.message], warnings, worldSize: 0, cards: [], groups: [], stats: {}, assumptions: [], sources: SOURCES };
    }
    const c = config;
    const ascend = c.profile === 'vllm-ascend';
    const attentionUnmodeled = c.attentionType === 'unmodeled';
    const partialEstimate = attentionUnmodeled || c.hashMoeLayers > 0;
    const routedHidden = c.routedExpertHiddenSize || c.hiddenSize;
    const hasExpertAdapters = routedHidden !== c.hiddenSize;
    const worldSize = c.tp * c.pp * c.dp * c.cp * c.pcp;
    const expertAxis = c.tp * c.dp * (ascend ? c.pcp : 1);
    const effectiveETP = ascend ? (c.ep === 1 ? expertAxis : 1) : c.etp;
    const expertReplicas = expertAxis / (c.ep * effectiveETP);
    if (worldSize > 512) errors.push(`需要 ${worldSize} 张卡；为保证浏览器响应，本工具最多展示 512 张。`);
    if (c.layers < c.pp) errors.push('PP 不能大于层数；每个流水级至少需要一层。');
    if (c.heads % c.tp) errors.push('Q 头数 heads 必须能被 TP 整除。');
    if (c.hiddenSize % c.tp) errors.push('hiddenSize 必须能被 TP 整除。');
    if (!attentionUnmodeled && c.heads % c.kvHeads) errors.push('Q 头数 heads 必须是 KV 头数 kvHeads 的整数倍。');
    if (!attentionUnmodeled && (c.kvHeads >= c.tp ? c.kvHeads % c.tp !== 0 : c.tp % c.kvHeads !== 0)) errors.push('KV 头数须能被 TP 整除，或 TP 是 KV 头数的整数倍以复制 KV 头。');
    if (c.moeLayers < c.layers && c.intermediateSize % c.tp) errors.push('稠密 FFN intermediateSize 必须能被 TP 整除。');
    if (c.moeLayers > c.layers) errors.push('moeLayers 不能超过 layers。');
    if (c.hashMoeLayers > c.moeLayers) errors.push('hashMoeLayers 不能超过 MoE 层数。');
    if (c.moeLayers > 0 && (c.experts === 0 || c.topK < 1 || c.topK > c.experts)) errors.push('启用 MoE 时需要 experts ≥ topK ≥ 1。');
    if (c.experts === 0 && (c.moeLayers > 0 || c.sharedExperts > 0 || c.topK > 0)) errors.push('Dense 模型请同时设 experts、moeLayers、sharedExperts、topK 为 0。');
    if (c.cp > 1 && c.pcp > 1) errors.push('CP 与 PCP 在本工具中是两种上下文方案，不能同时大于 1。');
    if (ascend && c.cp !== 1) errors.push('vLLM Ascend 模式不接受独立 CP 轴；请使用 PCP / DCP，或切换通用教学模式。');
    if (ascend && c.ep !== 1 && c.ep !== expertAxis) errors.push(`vLLM Ascend 的 EP 为启用开关：EP 应为 1 或 TP×DP×PCP = ${expertAxis}。`);
    if (ascend && c.etp !== 1) errors.push('vLLM Ascend 模式下 ETP 不作为独立自由参数，请设为 1；关闭 EP 时自动模拟框架的 MoE 张量切分。');
    if (!ascend && expertAxis % (c.ep * c.etp) !== 0) errors.push(`通用模式要求 EP×ETP 整除 TP×DP = ${expertAxis}；EP/ETP 复用现有卡。`);
    if (c.moeLayers > 0 && c.moeIntermediateSize % effectiveETP) errors.push(`moeIntermediateSize 必须能被有效 ETP = ${effectiveETP} 整除。`);
    if (c.sharedExperts > 0 && c.moeIntermediateSize % c.tp) errors.push('共享专家按 TP 切分，moeIntermediateSize 必须能被 TP 整除。');
    if (hasExpertAdapters && c.moeLayers > 0 && routedHidden % c.tp) errors.push('LatentMoE routedExpertHiddenSize 必须能被 TP 整除以展示适配投影切分。');
    if (c.pcp === 1) {
      if (c.tp % c.dcp) errors.push('PCP=1 时 DCP 必须整除 TP，且不会增加卡数。');
      if (c.attentionType === 'gqa' && c.dcp > 1 && (c.tp < c.kvHeads * c.dcp || (c.tp / c.kvHeads) % c.dcp !== 0)) errors.push('GQA/MQA 的独立 DCP 需要 TP/KV头数 为 DCP 的整数倍，确保同组拥有相同 KV 头。');
    } else if (c.dcp !== 1) {
      if (attentionUnmodeled && c.dcp !== c.pcp && c.dcp !== c.pcp * c.tp) errors.push('当前逻辑设备网格要求 DCP=PCP 或 DCP=PCP×TP。');
      if (c.attentionType === 'gqa' && c.dcp !== c.pcp) errors.push('GQA 的 PCP+DCP 组合要求 DCP=PCP；DCP 不再额外增加卡数。');
      if (c.attentionType === 'mla' && c.dcp !== c.pcp && c.dcp !== c.pcp * c.tp) errors.push('MLA 的 PCP+DCP 组合支持 DCP=PCP 或 DCP=PCP×TP。');
    }
    if (ascend && c.sp && (c.tp === 1 || c.ep === 1 || c.moeLayers === 0)) errors.push('vLLM Ascend 的 SP MoE 路径需要 TP>1、启用 EP 且包含 MoE 层。');
    const assumptions = [
      '这是确定性拓扑与容量教学模型，不运行大模型；通信线表示逻辑依赖，不是物理链路、实测吞吐或时延。',
      'Rank 顺序由内至外为 TP → PCP → CP → PP → DP；卡按顺序放置到节点，实际启动器可采用不同编号。',
      `PP 按层数近似均分；MoE 层采用${c.moeLayerPlacement === 'trailing' ? '末尾连续分布（前部为 Dense 层）' : '均匀分布'}。嵌入在首级、未绑定的 LM Head 在末级，均按词表切分。`,
      '权重以 Q/K/V/O、三矩阵 SwiGLU、路由器、共享专家、LayerNorm 估算；不含偏置、量化元数据、激活、通信缓冲、运行时工作区与 KV 分页碎片。',
      '每类 Attention/FFN 使用统一维度，不模拟逐层异构的头数、专家大小、滑动窗口、混合 Attention、额外预测头或共享嵌入；模型目录中的参数也按该容量近似解释。',
      'batchSize 是全局并发序列数，近似均分到 DP；每条序列按 sequenceLength 分配 KV。权重量化与 KV 精度分别设置。',
      'CP 用连续片段示意持久 KV 分片；PCP 只切分 prefill 计算，PCP 本身不缩减持久 KV。DCP 用交错 token 所有权示意，实际框架还受 block/interleave 配置影响。',
      ascend
        ? 'Ascend SP 表示 TP 组内的 SP MoE token 切分，Attention 边界使用 AllGather/ReduceScatter，减少重复专家计算；实际启用还需 FlashComm 开关、支持的模型与后端。SP 不增加卡数或进一步切分权重。'
        : '通用 SP 指 TP 组内 LayerNorm/激活的序列切分，不增加卡数或进一步切分权重；不代表 Ulysses 或扩散模型的 SP。'
    ];
    assumptions.push(ascend
      ? 'vLLM Ascend 模式：world=TP×PP×DP×PCP；EP 开启时跨 TP×DP×PCP 分专家，关闭时专家张量跨该组切分；DCP 复用现有 TP/PCP。按当前默认设置，Embedding/LM Head 跨 TP×PCP 切词表，其他 GQA/MLA Attention 权重在 PCP 轴复制；模型后端可有额外选项。本工具校验部分结构约束，不保证任意模型、后端和版本均可部署。'
      : '通用教学模式：world=TP×PP×DP×CP×PCP；EP×ETP 覆盖 TP×DP 并复用卡，剩余维度复制专家；这是显式抽象拓扑，不是某个框架的启动参数生成器。');
    if (c.attentionType === 'mla') {
      warnings.push('MLA 的 KV 按 latent+RoPE 维度估算；Attention 权重暂用展开后的 GQA 投影近似，无法等同真实 MLA 参数量。');
      assumptions.push('MLA 的 Q/K/V/O 权重视图为展开投影的教学近似；压缩 KV 每 token 使用 (kvLoraRank+qkRopeHeadDim) 个元素，不按 TP 头数再除。');
    }
    if (attentionUnmodeled) {
      warnings.unshift('部分估算：Attention 权重、KV / 线性注意力状态及模型特有辅助模块未建模，数值中的 0 表示未计入，不表示不存在。合计仅为已建模部分的结构估算，不能据此判断模型可装入显存或证明不会 OOM。');
      assumptions.unshift('未建模 Attention：不生成真实 Q/K/V/O 权重或 KV 容量估算。TP 仅说明已计量的 FFN 与适配投影分片；未建模的上下文与状态通信不作为已知通信关系展示。');
    }
    if (c.hashMoeLayers > 0) {
      warnings.push(`前 ${c.hashMoeLayers} 个 MoE 层使用 Hash 专家映射；门控评分仍保留 hidden×experts Router 矩阵。Hash 查找表与其专用开销未建模，合计是部分结构估算。`);
      assumptions.push('hashMoeLayers 计数按全局 MoE 层顺序；Hash 只改变专家编号的选择，门控权重仍由 Router 矩阵评分，所有 MoE 层均计入该矩阵。未知 Hash 查找表占用不以零占用解释。');
    }
    if (hasExpertAdapters) assumptions.push(`LatentMoE：路由专家使用 ${routedHidden} 维输入/输出，shared experts 与 router 仍使用 hiddenSize=${c.hiddenSize}；每个 MoE 层额外估算两个 ${c.hiddenSize}↔${routedHidden} 投影（按 TP 切分）与复制的 latent norm。`);
    if (!attentionUnmodeled && c.hiddenSize !== c.heads * c.headDim) warnings.push('hiddenSize 与 heads×headDim 不同：允许非标准投影结构，O 投影仍返回 hiddenSize。');
    if (c.layers % c.pp) warnings.push('层数不能被 PP 整除，各流水级层数与容量存在轻微差异。');
    if (c.moeLayers && c.experts % c.ep) warnings.push('专家数不能被 EP 整除，专家分配不均衡。');
    if (c.moeLayers && c.ep > c.experts) warnings.push('EP 大于专家数，部分卡不持有路由专家。');
    if (ascend && c.expertPlacement === 'round-robin' && c.ep > 1) warnings.push('Round-robin 展示请求的专家映射；实际 vLLM 后端可能不支持并退回连续分配，请核对所用 MoE backend。');
    if (c.batchSize < c.dp) warnings.push('batchSize 小于 DP，部分 DP 副本本轮没有请求，但仍占权重内存。');
    if (c.batchSize % c.dp) warnings.push('batchSize 不能被 DP 整除，各副本 KV 容量按实际分配数量估算。');
    if (c.sp && c.tp === 1) warnings.push('TP=1 时 SP 没有可切分的 TP 同伴。');
    if (c.precision === 'int4' || c.precision === 'int8') warnings.push('量化权重按理想位宽估算，实际 scale、zero-point、混合精度和打包对齐会增加占用。');
    if (c.moeLayers === 0 && c.ep > 1) warnings.push('当前没有 MoE 层，EP 不产生实际专家通信。');
    if (ascend && c.ep === 1 && c.moeLayers) warnings.push(`EP 关闭：按 vLLM 的 MoE 规则，专家张量使用有效 ETP=${effectiveETP}，并非在所有 TP/DP/PCP 卡上完整复制。`);
    if (errors.length) return { config, valid: false, errors, warnings, worldSize, cards: [], groups: [], stats: {}, assumptions, sources: SOURCES };

    const weightWidth = WIDTHS[c.precision], kvWidth = WIDTHS[c.kvPrecision];
    const hashMoeLayers = new Set(ints(0, c.layers - 1).filter(layer => isMoeLayer(layer, c)).slice(0, c.hashMoeLayers));
    const cards = [];
    const groups = [];
    for (let rank = 0; rank < worldSize; rank++) {
      let remainder = rank;
      const coords = {};
      for (const axis of ['tp', 'pcp', 'cp', 'pp', 'dp']) { coords[axis] = remainder % c[axis]; remainder = Math.floor(remainder / c[axis]); }
      const expertLinear = (coords.dp * (ascend ? c.pcp : 1) + (ascend ? coords.pcp : 0)) * c.tp + coords.tp;
      coords.etp = expertLinear % effectiveETP;
      coords.ep = Math.floor(expertLinear / effectiveETP) % c.ep;
      coords.expertReplica = Math.floor(expertLinear / (effectiveETP * c.ep));
      coords.dcp = c.pcp > 1 ? (c.dcp === c.pcp * c.tp ? coords.tp * c.pcp + coords.pcp : c.dcp > 1 ? coords.pcp : 0) : coords.tp % c.dcp;
      const layerSpan = span(c.layers, c.pp, coords.pp);
      const localMoeLayers = ints(layerSpan.start, layerSpan.end).filter(layer => isMoeLayer(layer, c));
      const localHashMoeLayers = localMoeLayers.filter(layer => hashMoeLayers.has(layer));
      const qRange = span(c.heads, c.tp, coords.tp);
      const qHeads = ints(qRange.start, qRange.end);
      const kvRange = c.kvHeads >= c.tp ? span(c.kvHeads, c.tp, coords.tp) : { start: Math.floor(coords.tp / (c.tp / c.kvHeads)), end: Math.floor(coords.tp / (c.tp / c.kvHeads)) };
      const kvHeads = ints(kvRange.start, kvRange.end);
      const expertRange = expertSpan(c.experts, c.ep, coords.ep);
      const experts = c.expertPlacement === 'round-robin' ? ints(0, c.experts - 1).filter(i => i % c.ep === coords.ep) : ints(expertRange.start, expertRange.end);
      const expertShard = span(c.moeIntermediateSize, effectiveETP, coords.etp);
      const qParams = c.hiddenSize * qHeads.length * c.headDim;
      const kvParams = c.hiddenSize * kvHeads.length * c.headDim;
      const oParams = qHeads.length * c.headDim * c.hiddenSize;
      const attentionBytes = attentionUnmodeled ? 0 : (qParams + kvParams * 2 + oParams) * layerSpan.count * weightWidth;
      const moeBytes = localMoeLayers.length * experts.length * 3 * routedHidden * expertShard.count * weightWidth;
      const expertAdapterBytes = hasExpertAdapters ? localMoeLayers.length * (2 * c.hiddenSize * routedHidden / c.tp + routedHidden) * weightWidth : 0;
      const sharedBytes = localMoeLayers.length * c.sharedExperts * 3 * c.hiddenSize * c.moeIntermediateSize / c.tp * weightWidth;
      const denseBytes = (layerSpan.count - localMoeLayers.length) * 3 * c.hiddenSize * c.intermediateSize / c.tp * weightWidth;
      const routerBytes = localMoeLayers.length * c.hiddenSize * c.experts * weightWidth;
      const vocabularyShards = c.tp * (ascend ? c.pcp : 1);
      const vocabularyShardIndex = (ascend ? coords.pcp * c.tp : 0) + coords.tp;
      const vocabulary = span(c.vocabSize, vocabularyShards, vocabularyShardIndex);
      const embeddingBytes = ((coords.pp === 0 ? 1 : 0) + (coords.pp === c.pp - 1 ? 1 : 0)) * vocabulary.count * c.hiddenSize * weightWidth;
      const normBytes = (2 * layerSpan.count + (coords.pp === c.pp - 1 ? 1 : 0)) * c.hiddenSize * weightWidth;
      const weightBytes = attentionBytes + moeBytes + sharedBytes + denseBytes + routerBytes + embeddingBytes + normBytes + expertAdapterBytes;
      const localBatch = span(c.batchSize, c.dp, coords.dp);
      const cpRange = span(c.sequenceLength, c.cp, coords.cp);
      const kvTokenCount = Math.max(0, Math.floor((cpRange.count - 1 - coords.dcp) / c.dcp) + 1);
      const kvStart = kvTokenCount ? cpRange.start + coords.dcp : -1;
      const kvEnd = kvTokenCount ? kvStart + (kvTokenCount - 1) * c.dcp : -1;
      const kvElements = attentionUnmodeled ? 0 : c.attentionType === 'mla' ? c.kvLoraRank + c.qkRopeHeadDim : 2 * kvHeads.length * c.headDim;
      const kvBytes = localBatch.count * kvTokenCount * layerSpan.count * kvElements * kvWidth;
      const prefillSpan = span(cpRange.count, c.pcp, coords.pcp);
      const tensor = (shape, shard, params) => ({ shape, shard, modeled: !attentionUnmodeled, bytes: attentionUnmodeled ? 0 : params * weightWidth, layersBytes: attentionUnmodeled ? 0 : params * weightWidth * layerSpan.count });
      cards.push({
        rank, node: Math.floor(rank / c.gpusPerNode), localRank: rank % c.gpusPerNode, coords, partialEstimate,
        layers: { ...layerSpan, moe: localMoeLayers, denseCount: layerSpan.count - localMoeLayers.length },
        attention: { qHeads, kvHeads, modeled: !attentionUnmodeled, kvReplicated: !attentionUnmodeled && c.tp > c.kvHeads, weightsBytes: attentionBytes,
          approximation: attentionUnmodeled ? 'unmodeled' : c.attentionType === 'mla' ? 'expanded-projection proxy' : null,
          tensors: {
            q: tensor([c.hiddenSize, qHeads.length * c.headDim], `列切分 · Q heads ${qRange.start}–${qRange.end}`, qParams),
            k: tensor([c.hiddenSize, kvHeads.length * c.headDim], `${c.tp > c.kvHeads ? '复制' : '列切分'} · KV heads ${kvRange.start}–${kvRange.end}`, kvParams),
            v: tensor([c.hiddenSize, kvHeads.length * c.headDim], `${c.tp > c.kvHeads ? '复制' : '列切分'} · KV heads ${kvRange.start}–${kvRange.end}`, kvParams),
            o: tensor([qHeads.length * c.headDim, c.hiddenSize], '行切分 · 局部结果归并', oParams)
          }
        },
        moe: { experts: localMoeLayers.length ? experts : [], expertShards: { index: coords.etp, count: effectiveETP, intermediateStart: expertShard.start, intermediateEnd: expertShard.end }, weightsBytes: moeBytes, sharedBytes, routerBytes, expertAdapterBytes, routedHiddenSize: routedHidden, sharedHiddenSize: c.hiddenSize, sharedExperts: c.sharedExperts, layers: localMoeLayers, hashLayers: localHashMoeLayers, topK: c.topK },
        embedding: { vocabStart: vocabulary.start, vocabEnd: vocabulary.end, count: vocabulary.count, shardIndex: vocabularyShardIndex, shardCount: vocabularyShards, input: coords.pp === 0, output: coords.pp === c.pp - 1, weightsBytes: embeddingBytes },
        kv: { tokens: kvTokenCount, start: kvStart, end: kvEnd, stride: c.dcp, batch: localBatch.count, bytes: kvBytes, modeled: !attentionUnmodeled, precision: c.kvPrecision, mode: attentionUnmodeled ? 'unmodeled' : c.dcp > 1 ? 'interleaved' : c.cp > 1 ? 'contiguous' : 'full', elementsPerToken: kvElements },
        tokens: { prefillStart: cpRange.start + prefillSpan.start, prefillEnd: cpRange.start + prefillSpan.end, prefillCount: prefillSpan.count, batchStart: localBatch.start, batchEnd: localBatch.end },
        memory: { attention: attentionBytes, moe: moeBytes, shared: sharedBytes, dense: denseBytes, embedding: embeddingBytes, norm: normBytes, router: routerBytes, expertAdapters: expertAdapterBytes, weights: weightBytes, kv: kvBytes, total: weightBytes + kvBytes, capacity: c.memoryGB * GIB },
        groups: {}
      });
    }

    const addGroup = (type, members, collective, description, phase, active, extra) => {
      if (attentionUnmodeled && ['CP', 'PCP', 'DCP', 'SP'].includes(type)) return;
      const id = `${type}-${groups.filter(g => g.type === type).length}`;
      const group = { id, type, ranks: members.map(card => card.rank), collective, description, phase: phase || 'both', active: !!active && members.length > 1, ...extra };
      groups.push(group);
      for (const card of members) card.groups[type] = id;
    };
    const axisGroups = (type, varying, collective, description, phase, active) => {
      const bins = new Map();
      for (const card of cards) {
        const key = ['tp', 'pcp', 'cp', 'pp', 'dp'].filter(axis => !varying.includes(axis)).map(axis => card.coords[axis]).join(':');
        if (!bins.has(key)) bins.set(key, []);
        bins.get(key).push(card);
      }
      for (const members of bins.values()) addGroup(type, members, collective, description, phase, active);
    };
    axisGroups('TP', ['tp'], c.sp ? 'AllGather / ReduceScatter' : 'AllReduce', attentionUnmodeled ? '已计量的 Dense/shared FFN 按 TP 切分中间维度，LatentMoE 适配投影按 TP 切分；本组不表示 Attention 的切分或通信。' : 'Q/K/V 列切分；O 与稠密 FFN 下投影归并局部结果。', 'both', c.tp > 1);
    axisGroups('PP', ['pp'], 'Send / Recv', '仅相邻流水级传递前向激活；最后一级输出 token logits。', 'both', c.pp > 1);
    axisGroups('DP', ['dp'], 'Request routing', 'Attention 副本处理独立请求；推理无需梯度 AllReduce。MoE 同步另见 EP / ETP。', 'both', false);
    axisGroups('CP', ['cp'], 'Ring KV exchange', '通用 CP：交换其他序列片段的 KV，保留各自 token 所有权。', 'both', c.cp > 1);
    axisGroups('PCP', ['pcp'], 'AllGather / P2P KV', 'PCP 分配 prefill 查询片段；交换 KV 或缓存更新输入。框架可额外切分 Embedding/LM Head。', 'prefill', c.pcp > 1);
    if (c.sp) axisGroups('SP', ['tp'], 'AllGather / ReduceScatter', ascend ? 'SP MoE 在 TP 组中切分专家输入 token；Attention 边界聚合/重分片。' : 'TP 组内切分 LayerNorm/激活序列；需要时聚合到完整输入。', 'both', c.tp > 1);

    const dcpBins = new Map();
    const epBins = new Map(), etpBins = new Map();
    for (const card of cards) {
      const p = card.coords;
      const dcpKey = c.pcp > 1 ? [p.dp, p.pp, p.cp, c.dcp === c.pcp * c.tp ? 'all' : p.tp, c.dcp === 1 ? p.pcp : 'all'].join(':') : [p.dp, p.pp, p.cp, p.pcp, Math.floor(p.tp / c.dcp)].join(':');
      if (!dcpBins.has(dcpKey)) dcpBins.set(dcpKey, []);
      dcpBins.get(dcpKey).push(card);
      const scope = [p.pp, p.cp, ascend ? '*' : p.pcp, p.expertReplica];
      const epKey = [...scope, p.etp].join(':');
      const etpKey = [...scope, p.ep].join(':');
      if (!epBins.has(epKey)) epBins.set(epKey, []);
      if (!etpBins.has(etpKey)) etpBins.set(etpKey, []);
      epBins.get(epKey).push(card);
      etpBins.get(etpKey).push(card);
    }
    for (const members of dcpBins.values()) {
      members.sort((a, b) => a.coords.dcp - b.coords.dcp);
      addGroup('DCP', members, 'Query exchange / LSE merge', 'DCP 按序列切分 KV，交换查询或部分注意力结果，利用 LSE 合并局部 softmax。', 'decode', c.dcp > 1, { prefillNote: '有缓存的 prefill 也可能访问历史 DCP 分片。' });
    }
    for (const members of epBins.values()) addGroup('EP', members, 'Dispatch / Combine · AllToAll', 'Router 评分或 Hash 映射选择专家编号，按门控权重分发 token 到专家所有者，随后合并专家输出；逻辑依赖不等于固定 all-to-all 内核。', 'both', members.some(card => card.layers.moe.length > 0) && c.ep > 1);
    for (const members of etpBins.values()) addGroup('ETP', members, 'AllReduce', '同一专家的 gate/up 按中间维度切分，down 投影结果归并。', 'both', members.some(card => card.layers.moe.length > 0) && effectiveETP > 1);
    const weights = cards.map(card => card.memory.weights);
    const totals = cards.map(card => card.memory.total);
    const maxMemory = Math.max(...totals);
    const minMemory = Math.min(...totals);
    if (maxMemory > c.memoryGB * GIB) warnings.push(`${partialEstimate ? '已建模部分' : '权重+KV'}的结构估算最高为 ${(maxMemory / GIB).toFixed(1)} GiB，超过单卡 ${c.memoryGB} GiB；实际运行还有未计入模块、激活和工作区。`);
    const uniqueAttn = attentionUnmodeled ? 0 : c.layers * c.hiddenSize * c.headDim * (2 * c.heads + 2 * c.kvHeads) * weightWidth;
    const uniqueMoe = c.moeLayers * c.experts * 3 * routedHidden * c.moeIntermediateSize * weightWidth;
    const uniqueShared = c.moeLayers * c.sharedExperts * 3 * c.hiddenSize * c.moeIntermediateSize * weightWidth;
    const uniqueDense = (c.layers - c.moeLayers) * 3 * c.hiddenSize * c.intermediateSize * weightWidth;
    const uniqueExpertAdapters = hasExpertAdapters ? c.moeLayers * (2 * c.hiddenSize * routedHidden + routedHidden) * weightWidth : 0;
    const uniqueOther = (c.moeLayers * c.hiddenSize * c.experts + 2 * c.vocabSize * c.hiddenSize + (2 * c.layers + 1) * c.hiddenSize) * weightWidth;
    const uniqueWeightBytes = uniqueAttn + uniqueMoe + uniqueShared + uniqueDense + uniqueOther + uniqueExpertAdapters;
    const stats = {
      partialEstimate, estimateScope: partialEstimate ? 'modeled-subset' : 'structural-approximation',
      excludedModules: [...(attentionUnmodeled ? ['Attention（含线性/压缩/索引模块）', 'KV / 线性注意力状态', '模型特有辅助模块'] : []), ...(c.hashMoeLayers > 0 ? ['Hash Router 查找表'] : [])],
      worldSize, nodes: Math.ceil(worldSize / c.gpusPerNode), weightsPerCardMin: Math.min(...weights), weightsPerCardMax: Math.max(...weights),
      totalWeightBytes: weights.reduce((a, b) => a + b, 0), uniqueWeightBytes, uniqueExpertAdapterBytes: uniqueExpertAdapters,
      kvPerCardMax: Math.max(...cards.map(card => card.kv.bytes)), totalKVBytes: cards.reduce((sum, card) => sum + card.kv.bytes, 0),
      memoryPerCardMin: minMemory, memoryPerCardMax: maxMemory, capacityBytes: c.memoryGB * GIB,
      memoryUtilizationMax: maxMemory / (c.memoryGB * GIB), imbalance: maxMemory ? (maxMemory - minMemory) / maxMemory : 0,
      expertReplication: expertReplicas * c.cp * (ascend ? 1 : c.pcp), effectiveETP,
      expertCopiesAcrossPP: expertReplicas * c.cp * (ascend ? 1 : c.pcp), attentionReplication: c.dp * c.cp * c.pcp,
      activeGroups: groups.filter(group => group.active).length, totalGroups: groups.length,
      estimatedParameters: uniqueWeightBytes / weightWidth,
      activatedExpertParameters: c.moeLayers * c.topK * 3 * routedHidden * c.moeIntermediateSize,
      formula: ascend ? 'TP × PP × DP × PCP' : 'TP × PP × DP × CP × PCP'
    };
    return { config, valid: true, errors, warnings, worldSize, cards, groups, stats, assumptions, sources: SOURCES };
  }

  return Object.freeze({ DEFAULT_CONFIG, PRESETS, normalizeConfig, analyze, SOURCES });
});
