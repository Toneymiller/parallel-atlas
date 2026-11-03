(function () {
  'use strict';
  const E = window.ParallelEngine;
  const modelLibrary = window.ModelCatalog?.MODELS || [];
  const hardwareLibrary = window.HardwareCatalog?.HARDWARE || [];
  const $ = id => document.getElementById(id);
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const GIB = 1024 ** 3;
  const colors = { TP: '#8c75da', PP: '#b280ba', DP: '#8ba2be', CP: '#56b3a6', PCP: '#48a7c7', DCP: '#dd8b7d', EP: '#e2a14f', ETP: '#ac88cc', SP: '#88b85f' };
  const strategies = [
    ['TP', 'tp', '张量并行', '切分 Attention 头及投影矩阵。Q/K/V 采用列切分，O 采用行切分并归并结果；KV 头不足时会复制。'],
    ['PP', 'pp', '流水线并行', '将网络层分到不同流水级。只有相邻级传递激活，单卡持有本级网络层。'],
    ['DP', 'dp', '数据并行', 'Attention 副本处理不同请求。推理阶段没有训练梯度同步；启用跨 DP 的 EP 时仍需专家通信。'],
    ['EP', 'ep', '专家并行', '将路由专家分到不同卡。先按 top-k 路由分发 token，再合并专家输出。EP 复用现有设备。'],
    ['CP', 'cp', '上下文并行', '独立的上下文轴切分序列，复制模型权重，交换 KV 或注意力中间结果。这里采用逻辑 Ring KV 交换。'],
    ['PCP', 'pcp', '预填充上下文并行', '增加参与 Prefill 的设备，分配不同的查询 token。单独开启 PCP 不会自动减少持久 KV Cache。'],
    ['DCP', 'dcp', '解码上下文并行', '复用 TP/PCP 中的设备，沿序列维度切分 KV，通过查询交换和 LSE 归并计算注意力，不增加卡数。'],
    ['ETP', 'etp', '专家张量并行', '同一个专家的中间维度再做张量切分。通用模式要求 EP×ETP 整除 TP×DP。'],
    ['SP', 'sp', '序列并行', '通用模式采用 Megatron 风格的 TP 组内激活 / LayerNorm 序列切分；vLLM Ascend 模式表示 SP MoE token 切分，减少重复专家计算。复用 TP 卡，不额外切分持久权重；实际支持取决于模型与后端。']
  ];
  const modelFields = [['layers', '网络层数', 1, 256], ['moeLayers', 'MoE 层数', 0, 256], ['hiddenSize', 'Hidden size', 1, 65536], ['headDim', 'Head dim', 1, 1024], ['heads', 'Q 头数', 1, 512], ['kvHeads', 'KV 头数', 1, 512], ['experts', '路由专家数', 0, 2048], ['topK', '路由 Top-K', 0, 64]];
  const runtimeFields = [['sequenceLength', '序列长度', 1, 4194304], ['batchSize', '全局序列数', 1, 4096], ['gpusPerNode', '每节点设备数', 1, 512], ['memoryGB', '单卡容量 GiB', 1, 16384]];
  const advancedFields = [['intermediateSize', 'Dense FFN dim', 1, 262144], ['moeIntermediateSize', 'MoE FFN dim', 1, 262144], ['routedExpertHiddenSize', '路由专家 Hidden · 0=默认', 0, 65536], ['hashMoeLayers', '前置 Hash MoE 层', 0, 256], ['sharedExperts', '共享专家数', 0, 64], ['vocabSize', '词表大小', 1, 2000000], ['kvLoraRank', 'MLA latent dim', 1, 8192], ['qkRopeHeadDim', 'MLA RoPE dim', 0, 1024]];
  const initialModel = modelLibrary.find(m => m.name === 'GLM-5.3');
  const initialConfig = { ...E.DEFAULT_CONFIG, ...initialModel?.config, ...(initialModel ? { pp: 2, hardwareId: 'h200-141gb-sxm', memoryGB: 141 } : {}) };
  let config = { ...initialConfig }, result, selectedRank = 0, selectedGroup = null, selectedType = 'ALL', phase = 'prefill', view = 'topology', zoom = 1, layer = 0, baseline = null, toastTimer, updateTimer, latestSvg = '', svgSize = { width: 800, height: 500 };
  let animate = !window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function formatBytes(bytes, digits = 2) {
    if (!Number.isFinite(bytes)) return '—';
    if (bytes === 0) return '0';
    if (bytes >= GIB) return (bytes / GIB).toFixed(digits) + ' GiB';
    if (bytes >= 1024 ** 2) return (bytes / 1024 ** 2).toFixed(digits > 1 ? 1 : digits) + ' MiB';
    return (bytes / 1024).toFixed(0) + ' KiB';
  }
  function formatCount(n) { return Number(n).toLocaleString('en-US'); }
  function rangeText(values, prefix = '') {
    if (!values?.length) return '—';
    let chunks = [], start = values[0], end = start;
    for (let i = 1; i <= values.length; i++) {
      if (i < values.length && values[i] === end + 1) end = values[i];
      else { chunks.push(prefix + start + (end > start ? '–' + prefix + end : '')); start = end = values[i]; }
    }
    return chunks.length > 6 ? chunks.slice(0, 5).join(', ') + ` … (${values.length})` : chunks.join(', ');
  }
  const dot = type => `<i class="color-dot" style="--group-color:${colors[type] || colors.TP}"></i>`;
  function notify(message) { $('toast').textContent = message; $('toast').classList.add('show'); clearTimeout(toastTimer); toastTimer = setTimeout(() => $('toast').classList.remove('show'), 3500); }
  function fieldMarkup([key, label, min, max]) { return `<div><label class="field-label" for="${key}">${label}</label><input id="${key}" name="${key}" type="number" min="${min}" max="${max}" step="1" inputmode="numeric"></div>`; }

  function initializeForm() {
    for (const generation of ['current', 'legacy']) $('model-select').innerHTML += `<optgroup label="${generation === 'current' ? '最新模型 · 官方结构已核实' : '经典模型 · 兼容保留'}">${modelLibrary.filter(m => (m.generation || 'legacy') === generation).map(m => `<option value="${esc(m.id)}">${esc(m.name)}</option>`).join('')}</optgroup>`;
    for (const vendor of ['Huawei', 'NVIDIA']) $('hardware-select').innerHTML += `<optgroup label="${vendor === 'Huawei' ? '华为 Ascend NPU' : '英伟达 NVIDIA GPU'}">${hardwareLibrary.filter(h=>h.vendor===vendor).map(h=>`<option value="${esc(h.id)}">${esc(h.name)}</option>`).join('')}</optgroup>`;
    $('preset-select').innerHTML = '<option value="custom">自定义配置</option>' + E.PRESETS.map(p => `<option value="${p.id}">${esc(p.name)}</option>`).join('');
    $('parallel-fields').innerHTML = strategies.filter(s => s[0] !== 'SP').map(([type, key, name]) => `<div class="parallel-field"><label for="${key}" title="${name}">${dot(type)}${type}</label><input id="${key}" name="${key}" type="number" min="1" max="${key === 'pp' ? 256 : 512}" step="1" inputmode="numeric" aria-label="${name} ${type}"><span class="field-description">${name}</span></div>`).join('');
    $('model-fields').innerHTML = modelFields.map(fieldMarkup).join('');
    $('runtime-fields').innerHTML = runtimeFields.map(fieldMarkup).join('');
    $('advanced-fields').innerHTML = advancedFields.map(fieldMarkup).join('');
    $('animate-toggle').checked = animate;
    try {
      const shared = location.hash.startsWith('#config=') ? location.hash.slice(8) : '';
      if (shared.length > 12000) throw new Error('分享配置过大');
      if (shared) { config = E.normalizeConfig(JSON.parse(new TextDecoder().decode(Uint8Array.from(atob(shared), c => c.charCodeAt(0))))); notify('已加载分享配置'); }
      else { const saved = localStorage.getItem('parallel-atlas-config-v1'); if (saved) config = E.normalizeConfig(JSON.parse(saved)); }
    } catch { config = { ...initialConfig }; notify('未能读取已保存的配置，已使用默认示例'); }
    layer = firstMoeLayer(config);
    fillForm(config);
  }
  function firstMoeLayer(c) { return c.moeLayers ? c.moeLayerPlacement === 'trailing' ? c.layers - c.moeLayers : Math.max(0, Math.ceil(c.layers / c.moeLayers) - 1) : 0; }
  function fillForm(value) {
    for (const [key, v] of Object.entries(value)) { const input = $(key); if (input) { if (input.type === 'checkbox') input.checked = v; else input.value = v; } }
    updatePresetLabel();
  }
  function updatePresetLabel() {
    const p = E.PRESETS.find(p => Object.keys(E.DEFAULT_CONFIG).every(k => p.config[k] === config[k]));
    $('preset-select').value = p?.id || 'custom';
    $('preset-description').textContent = p?.description || '调整参数，实时探索自定义并行布局。';
    const model = currentModel();
    $('model-select').value = model?.id || 'custom';
    $('model-source').innerHTML = model ? `<a href="${esc(model.source)}" target="_blank" rel="noopener">${esc(model.organization)} · 官方 config.json ↗</a><p><b>${model.support === 'partial' ? '部分结构估算' : '投影结构近似'}</b> · ${esc(model.attentionDescription || model.architecture)}</p><p>${model.notes.map(esc).join(' ')}</p>` : '<p>选择公开模型自动填入结构；并行参数可继续独立调整。</p>';
    const hardware = hardwareLibrary.find(h=>h.id===config.hardwareId);
    $('hardware-select').value = hardware?.id || 'custom';
    $('hardware-source').innerHTML = hardware ? `<a href="${esc(hardware.source)}" target="_blank" rel="noopener">${esc(hardware.vendor)} · ${hardware.status === 'unverified' ? '公开产品目录' : '规格来源'} ↗</a>${hardware.status === 'unverified' ? '<p><b>规格待核实 · 当前使用手动容量与节点预算</b></p>' : ''}<p>${hardware.notes.map(esc).join(' ')}</p>${hardware.status !== 'unverified' && (config.memoryGB!==hardware.memoryGB||config.gpusPerNode!==hardware.gpusPerNode)?'<p>当前容量或设备数已手动调整。</p>':''}` : '<p>选择硬件自动填入容量预算与节点设备数；可手动调整。</p>';
  }
  function deviceLabel() { return hardwareLibrary.find(h=>h.id===config.hardwareId)?.vendor === 'Huawei' ? 'NPU' : 'GPU'; }
  function currentModel() { const matches = modelLibrary.filter(m => Object.entries(m.config).filter(([k])=>!['precision','kvPrecision','modelName'].includes(k)).every(([k,v]) => config[k] === v)); return matches.find(m => m.config.modelName === config.modelName) || (matches.length === 1 ? matches[0] : undefined); }
  function attentionLabel() { return currentModel()?.attentionLayers?.[layer] || (config.attentionType === 'unmodeled' ? '混合 ATTN' : config.attentionType.toUpperCase()); }
  function readForm() {
    const next = {};
    for (const key of Object.keys(E.DEFAULT_CONFIG)) { const input = $(key); next[key] = input ? input.type === 'checkbox' ? input.checked : input.value : config[key]; }
    return next;
  }
  function recompute() {
    result = E.analyze(readForm());
    if (result.config) config = result.config;
    selectedRank = Math.min(selectedRank, Math.max(0, result.cards.length - 1));
    selectedGroup = null;
    layer = Math.min(layer, (result.config?.layers || 1) - 1);
    $('layer-select').innerHTML = Array.from({ length: result.config?.layers || 1 }, (_, i) => `<option value="${i}">L${i}</option>`).join('');
    $('layer-select').value = layer;
    updatePresetLabel();
    if (result.valid) { try { localStorage.setItem('parallel-atlas-config-v1', JSON.stringify(config)); } catch { /* Storage is optional. */ } }
    renderAll();
  }
  function renderAll() {
    $('model-summary').textContent = `${config.modelName} · ${currentModel()?.attentionDescription || '探索每张卡的权重、上下文与通信路径。'}`;
    renderMetrics(); renderValidation(); renderFilters(); renderCanvas(); renderCommunications(); renderInspector(); renderInsights(); renderComparison();
    $('world-formula').textContent = result.valid ? `${result.stats.formula} = ${result.worldSize} devices` : '请调整参数以生成有效布局';
    $('download-button').disabled = !result.valid;
    $('baseline-button').disabled = !result.valid;
  }
  function renderMetrics() {
    const s = result.stats;
    const partial = s?.partialEstimate;
    const metrics = [
      ['集群规模', '▦', result.valid ? result.worldSize : '—', 'DEVICES', result.valid ? `${s.nodes} 个节点 · 每节点 ${config.gpusPerNode} 个设备` : '等待有效配置'],
      [partial ? '单卡权重 · 已建模部分' : '单卡权重 · 最高', '▧', result.valid ? (s.weightsPerCardMax / GIB).toFixed(2) : '—', 'GiB', result.valid ? `最小 ${(s.weightsPerCardMin / GIB).toFixed(2)} GiB · ${config.precision.toUpperCase()}` : 'Attention + FFN + 其他权重'],
      ['单卡 KV Cache · 最高', '▤', config.attentionType === 'unmodeled' ? '未计量' : result.valid ? (s.kvPerCardMax / GIB).toFixed(2) : '—', config.attentionType === 'unmodeled' ? '' : 'GiB', `${formatCount(config.sequenceLength)} tokens · ${config.kvPrecision.toUpperCase()}`],
      [partial ? '部分占用 / 容量 · 最高' : '估算显存占比 · 最高', '◴', result.valid ? (s.memoryUtilizationMax * 100).toFixed(1) : '—', '%', partial ? '已建模部分 · 不能判断完整模型能否装入' : `权重 + KV / ${config.memoryGB} GiB · 不含工作区`]
    ];
    $('metrics').innerHTML = metrics.map(([label, icon, value, unit, detail]) => `<div class="metric"><div class="metric-label">${label}<span class="metric-icon">${icon}</span></div><div class="metric-value">${value}<small>${unit}</small></div><div class="metric-detail">${esc(detail)}</div></div>`).join('');
  }
  function renderValidation() {
    $('validation').innerHTML = result.errors.length ? `<div class="validation-box error"><strong>参数组合需要调整</strong>${result.errors.map(e => `<p>${esc(e)}</p>`).join('')}</div>` : result.stats.partialEstimate ? `<div class="validation-box"><strong>当前为已建模部分的结构估算</strong><p>未计入：${result.stats.excludedModules.map(esc).join('、')}。数值不能用于判断完整模型是否装入显存${config.attentionType === 'unmodeled' ? '；Attention 相关通信仅展示候选逻辑组' : ''}。</p></div>` : '';
  }
  function visibleGroups() {
    if (!result.valid) return [];
    return result.groups.filter(g => g.ranks.length > 1 && (g.active || g.type === 'DP') && (g.phase === 'both' || g.phase === phase) && (selectedType === 'ALL' || g.type === selectedType) && (!['EP', 'ETP'].includes(g.type) || g.ranks.some(r => result.cards[r].layers.moe.includes(layer))));
  }
  function renderFilters() {
    const available = result.valid ? strategies.map(s => s[0]).filter(type => result.groups.some(g => g.type === type && g.ranks.length > 1 && (g.active || type === 'DP'))) : [];
    if (selectedType !== 'ALL' && !available.includes(selectedType)) selectedType = 'ALL';
    $('group-filters').innerHTML = `<button class="filter-chip all ${selectedType === 'ALL' ? 'active' : ''}" style="--group-color:#8c75da" data-type="ALL">全部关系</button>` + available.map(type => `<button class="filter-chip ${selectedType === type ? 'active' : ''}" data-type="${type}" style="--group-color:${colors[type]}">${dot(type)}${type}</button>`).join('') + `<span class="muted small" style="margin-left:auto">○ 选卡查看详情</span>`;
  }
  function topologySVG() {
    const n = result.worldSize, cols = n <= 4 ? 2 : n <= 32 ? 4 : 8;
    const cardW = 140, cardH = 117, gapX = 32, gapY = 52, padX = 30, top = 68;
    const rows = Math.ceil(n / cols), width = Math.max(420, padX * 2 + cols * cardW + (cols - 1) * gapX), height = Math.max(390, top + rows * (cardH + gapY) - gapY + 55);
    svgSize = { width, height };
    const positions = result.cards.map((card, i) => ({ x: padX + (i % cols) * (cardW + gapX), y: top + Math.floor(i / cols) * (cardH + gapY) }));
    const fitScale = Math.min(1, Math.max(280, $('canvas-viewport').clientWidth - 12) / width);
    let svg = `<svg class="topology-svg" id="topology-svg" xmlns="http://www.w3.org/2000/svg" width="${width * fitScale * zoom}" height="${height * fitScale * zoom}" viewBox="0 0 ${width} ${height}" role="group" aria-label="${n} 个设备的逻辑权重和通信拓扑"><defs>`;
    for (const [type, color] of Object.entries(colors)) svg += `<marker id="arrow-${type}" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0L6 3L0 6" fill="${color}"/></marker>`;
    svg += '</defs>';
    // Node boundaries are ordinal rank placement, independent from logical collectives.
    for (let r = 0; r < rows; r++) {
      const rowCards = result.cards.slice(r * cols, (r + 1) * cols);
      let segments = [];
      for (const card of rowCards) { const last = segments[segments.length - 1]; if (last?.node === card.node) last.cards.push(card); else segments.push({ node: card.node, cards: [card] }); }
      for (const segment of segments) {
        const p = positions[segment.cards[0].rank], sw = segment.cards.length * (cardW + gapX) - gapX + 22;
        svg += `<rect x="${p.x - 11}" y="${p.y - 28}" width="${sw}" height="${cardH + 44}" rx="9" fill="#f7f7fc" stroke="#e7e5f1" stroke-dasharray="4 4"/><text x="${p.x + 1}" y="${p.y - 12}" fill="#aaa5bc" font-size="8" letter-spacing="1.1">NODE ${String(segment.node).padStart(2, '0')} <tspan fill="#c3bfd0" font-size="7">/ LOGICAL PLACEMENT</tspan></text>`;
      }
    }
    const allGroups = visibleGroups(), selected = allGroups.find(g => g.id === selectedGroup);
    // Lines aggregate large collectives as a ring to avoid O(N²) edge rendering.
    const lineGroups = selected ? [selected] : allGroups.filter(g => g.active && g.ranks.includes(selectedRank));
    let links = 0;
    for (const g of lineGroups) {
      if (!g.active && g.type === 'DP') continue;
      let pairs = [];
      const ranks = g.ranks;
      if (g.type === 'PP') pairs = ranks.slice(1).map((r, i) => [ranks[i], r]);
      else if (ranks.length <= 4) { for (let i = 0; i < ranks.length; i++) for (let j = i + 1; j < ranks.length; j++) pairs.push([ranks[i], ranks[j]]); }
      else pairs = ranks.map((r, i) => [r, ranks[(i + 1) % ranks.length]]);
      for (const [a, b] of pairs.slice(0, 512)) {
        const p = positions[a], q = positions[b], startX = p.x + cardW / 2, endX = q.x + cardW / 2, sameRow = p.y === q.y;
        let d;
        if (sameRow) { const y = p.y + cardH + 3; const bend = 14 + Math.min(12, Math.abs(a - b) * 3) + Object.keys(colors).indexOf(g.type) * 2; d = `M${startX},${y} C${startX},${y + bend} ${endX},${y + bend} ${endX},${y}`; }
        else { const startY = p.y + cardH / 2, endY = q.y + cardH / 2; const bendX = Math.min(startX, endX) - cardW / 2 - 12; d = `M${p.x + 1},${startY} C${bendX},${startY} ${bendX},${endY} ${q.x + 1},${endY}`; }
        svg += `<path class="comm-line ${animate ? 'animated' : ''} ${selectedType === 'ALL' && !selected ? 'faded' : ''}" d="${d}" stroke="${colors[g.type]}" stroke-width="${selected ? 2 : 1.5}" marker-end="url(#arrow-${g.type})"><title>${esc(g.id)} · ${esc(g.collective)} · R${a} ↔ R${b}</title></path>`;
        links++;
      }
    }
    for (const card of result.cards) {
      const { x, y } = positions[card.rank], isSelected = card.rank === selectedRank, dim = selected && !selected.ranks.includes(card.rank), inLayer = layer >= card.layers.start && layer <= card.layers.end;
      const activeMoE = card.layers.moe.includes(layer), expertText = activeMoE ? rangeText(card.moe.experts, 'E') : inLayer ? 'Dense FFN' : `L${card.layers.start}–L${card.layers.end}`;
      const mem = card.memory, utilization = Math.min(1, mem.total / mem.capacity), wbar = (cardW - 22) * utilization;
      svg += `<g class="gpu-card ${isSelected ? 'selected' : ''} ${dim ? 'dim' : ''}" data-rank="${card.rank}" tabindex="0" role="button" aria-label="${deviceLabel()} ${card.rank}, 节点 ${card.node}, 选择查看权重详情"><rect class="gpu-bg" x="${x}" y="${y}" width="${cardW}" height="${cardH}" rx="8" fill="white" stroke="${isSelected ? '#8c75da' : '#e5e2ef'}"/>
        <rect x="${x + 10}" y="${y + 11}" width="18" height="18" rx="4" fill="${isSelected ? '#efeafb' : '#f4f3f9'}"/><text x="${x + 14}" y="${y + 24}" font-size="13" fill="#a08cce">▧</text><text x="${x + 35}" y="${y + 24}" fill="${isSelected ? '#7861ba' : '#716b87'}" font-size="11" font-weight="600">${deviceLabel()} ${String(card.rank).padStart(2, '0')}</text><circle cx="${x + cardW - 15}" cy="${y + 20}" r="2.5" fill="${mem.total > mem.capacity ? '#df887a' : inLayer ? '#76baa5' : '#d2cfde'}"/>
        <rect x="${x + 10}" y="${y + 39}" width="${cardW - 20}" height="18" rx="3" fill="#eef8fc"/><text x="${x + 17}" y="${y + 51}" font-size="8" fill="#69a9bc">${esc(attentionLabel())}</text><text x="${x + 72}" y="${y + 51}" font-size="7" fill="#5999ad">${card.attention.modeled === false ? '未计量' : 'Q' + esc(rangeText(card.attention.qHeads))}</text>
        <rect x="${x + 10}" y="${y + 62}" width="${cardW - 20}" height="18" rx="3" fill="${activeMoE ? '#fff6e8' : '#f5f2fb'}"/><text x="${x + 17}" y="${y + 74}" font-size="8" fill="${activeMoE ? '#c8a16b' : '#a69bbd'}">${activeMoE ? 'MoE' : 'FFN'}</text><text x="${x + 47}" y="${y + 74}" font-size="${expertText.length > 18 ? 6 : 8}" fill="${activeMoE ? '#be985f' : '#a69bbd'}">${esc(expertText)}</text>
        <rect x="${x + 11}" y="${y + 91}" width="${cardW - 22}" height="3" rx="1.5" fill="#f1eef7"/><rect x="${x + 11}" y="${y + 91}" width="${Math.max(0, wbar)}" height="3" rx="1.5" fill="${mem.total > mem.capacity ? '#d58b80' : '#b7a4df'}"/><text x="${x + 11}" y="${y + 107}" font-size="7" fill="#b0a9c0">${card.partialEstimate ? '部分 ' : ''}${(mem.total / GIB).toFixed(2)} / ${config.memoryGB} GiB</text><text x="${x + cardW - 11}" y="${y + 107}" text-anchor="end" font-size="7" fill="#b0a9c0">TP${card.coords.tp} · EP${card.coords.ep}</text></g>`;
    }
    svg += `<text x="${padX}" y="${height - 15}" font-size="7" fill="#b8b0c9">L${layer} · ${phase.toUpperCase()} · ${result.stats.partialEstimate ? 'MODELED SUBSET ONLY · 部分结构估算' : 'WEIGHTS + KV ESTIMATE'} · ${links} LOGICAL EDGES</text></svg>`;
    return svg;
  }
  function renderCanvas() {
    if (!result.valid) { $('canvas-content').innerHTML = '<div class="empty-state" style="padding-top:150px">调整左侧参数，生成集群拓扑与逐卡分布。<br>也可以选择一个预设开始探索。</div>'; latestSvg = ''; return; }
    latestSvg = topologySVG();
    if (view === 'topology') $('canvas-content').innerHTML = latestSvg;
    else if (view === 'weights') renderWeightView();
    else renderMatrix();
    $('zoom-value').textContent = view === 'topology' ? Math.round(zoom * 100) + '%' : '—';
    $('zoom-in').disabled = view !== 'topology'; $('zoom-out').disabled = view !== 'topology';
    $('canvas-caption').textContent = view === 'matrix' ? '逻辑通信邻接 · 包括同组关系，无物理带宽含义' : `L${layer} · ${phase} · ${selectedGroup || '当前卡相关通信组'}${result.worldSize > 4 ? ' · 大组以环示意' : ''}`;
  }
  function renderWeightView() {
    $('canvas-content').innerHTML = `<table class="weight-table"><thead><tr><th>设备 / 节点</th><th>模型层</th><th>Q / KV 头</th><th>路由专家</th><th>权重${result.stats.partialEstimate ? ' · 部分' : ''}</th><th>KV Cache</th><th>${result.stats.partialEstimate ? '已计量合计' : '合计'}</th></tr></thead><tbody>${result.cards.map(c => `<tr data-rank="${c.rank}" tabindex="0" role="button" aria-label="查看 ${deviceLabel()} ${c.rank} 权重" class="${selectedRank === c.rank ? 'selected' : ''}"><td>${deviceLabel()} ${c.rank}<br><span class="muted">NODE ${c.node}</span></td><td>L${c.layers.start}–${c.layers.end}</td><td>${c.attention.modeled === false ? '特殊结构<br><span class="muted">见单卡说明</span>' : esc(rangeText(c.attention.qHeads)) + '<br><span class="muted">' + esc(rangeText(c.attention.kvHeads)) + '</span>'}</td><td title="${esc(c.moe.experts.join(', '))}">${esc(rangeText(c.moe.experts, 'E'))}</td><td>${formatBytes(c.memory.weights)}</td><td>${c.kv.modeled === false ? '未计量' : formatBytes(c.kv.bytes)}</td><td>${formatBytes(c.memory.total)}</td></tr>`).join('')}</tbody></table>`;
  }
  function renderMatrix() {
    const groups = visibleGroups().filter(g => g.active);
    // Matrix pages follow selected rank; every card remains inspectable in table/topology.
    const pageStart = Math.floor(selectedRank / 32) * 32, cards = result.cards.slice(pageStart, pageStart + 32);
    let html = `<div class="matrix-wrap"><p class="matrix-info">R${pageStart}–R${pageStart + cards.length - 1} · 单元格显示共同通信组；点击选中目标卡。${result.worldSize > 32 ? '大集群按当前选中卡分为 32 卡一页。' : ''}</p><table class="matrix-table"><thead><tr><th>FROM / TO</th>${cards.map(c => `<th>${c.rank}</th>`).join('')}</tr></thead><tbody>`;
    for (const a of cards) { html += `<tr><th>R${a.rank}</th>`; for (const b of cards) { const matches = groups.filter(g => g.ranks.includes(a.rank) && g.ranks.includes(b.rank) && (g.type !== 'PP' || Math.abs(a.coords.pp - b.coords.pp) === 1)); html += `<td data-rank="${b.rank}" tabindex="0" role="button" aria-label="R${a.rank} 到 R${b.rank} 的通信关系" class="${a.rank === b.rank ? 'matrix-diagonal' : matches.length ? 'matrix-on' : ''}" style="--cell-color:${colors[matches[0]?.type] || '#eae7f3'}" title="R${a.rank} → R${b.rank}: ${esc(a.rank === b.rank ? '本卡' : matches.map(g => g.id).join(', ') || '无当前筛选关系')}">${a.rank === b.rank ? '·' : matches.length ? matches.length > 1 ? '+' : matches[0].type : ''}</td>`; } html += '</tr>'; }
    html += '</tbody></table>' + (result.worldSize > 32 ? `<div class="matrix-pages"><button class="button compact" data-matrix-page="${Math.max(0,pageStart-32)}" ${pageStart===0?'disabled':''}>← 上一页</button><span>${Math.floor(pageStart/32)+1} / ${Math.ceil(result.worldSize/32)}</span><button class="button compact" data-matrix-page="${pageStart+32}" ${pageStart+32>=result.worldSize?'disabled':''}>下一页 →</button></div>` : '') + '</div>'; $('canvas-content').innerHTML = html;
  }
  function renderCommunications() {
    const groups = visibleGroups();
    $('group-count').textContent = groups.length;
    if (!groups.length) { $('communication-list').innerHTML = '<div class="empty-state">当前阶段、模型层或筛选下没有多卡通信组。</div>'; return; }
    $('communication-list').innerHTML = `<div class="table-scroll"><table class="comm-table"><thead><tr><th>通信组</th><th>参与 Ranks</th><th>通信操作</th></tr></thead><tbody>${groups.map(g => `<tr class="comm-row ${g.id === selectedGroup ? 'selected' : ''}" data-group="${g.id}" title="${esc(g.description)}" tabindex="0" role="button" aria-label="高亮 ${g.id}"><td><span class="comm-type" style="color:${colors[g.type]}">${dot(g.type)}${g.id}</span></td><td class="rank-list" title="${g.ranks.join(', ')}">${esc(rangeText(g.ranks))}</td><td class="collective">${esc(g.collective)}${!g.active ? ' · 无集体同步' : ''}</td></tr>`).join('')}</tbody></table></div>${selectedGroup ? `<div class="more-note">${esc(groups.find(g => g.id === selectedGroup)?.description || '')}</div>` : ''}`;
  }
  function renderInspector() {
    const c = result.cards[selectedRank];
    if (!c) { $('card-inspector').innerHTML = '<div class="empty-state">有效配置生成后，点击任一卡片查看。</div>'; return; }
    const inLayer = layer >= c.layers.start && layer <= c.layers.end, activeMoE = c.layers.moe.includes(layer);
    const localExperts = new Set(activeMoE ? c.moe.experts : []), expertPage = Math.floor((c.moe.experts[0] || 0) / 64) * 64, shownExperts = Math.min(64, Math.max(0, config.experts - expertPage));
    const capacity = c.memory.capacity, memory = [['attention', 'Attention', '#67b9cf'], ['moe', '路由专家', '#e7b56f'], ['expertAdapters', 'LatentMoE 适配投影', '#d6a6b7'], ['shared', '共享专家', '#cdb58b'], ['dense', 'Dense FFN', '#b7a0d7'], ['embedding', 'Embedding / LM Head', '#c5bedc'], ['router', 'Router', '#a9c1b5'], ['norm', 'Norm', '#d1d1dd'], ['kv', 'KV Cache', '#8c78c5']];
    const model = currentModel();
    const attentionContent = c.attention.modeled === false ? `<p class="inspector-note"><b>L${layer} · ${esc(attentionLabel())}</b><br>${esc(model?.attentionDescription || '此模型采用特殊 / 混合注意力。')}<br>Attention 权重、KV / 线性状态尚未计量。TP 仅表示候选逻辑分组，未假定普通 Q/K/V/O 矩阵切分。</p>` : `${['q', 'k', 'v', 'o'].map(k => { const t = c.attention.tensors[k]; return `<div class="tensor-row" title="${esc(t.shard)} · ${t.shape.join('×')} · 每层 ${formatBytes(t.bytes)}"><span class="tensor-name">W${k.toUpperCase()}</span><span class="tensor-bar">${Array.from({ length: Math.min(8, k === 'q' || k === 'o' ? c.attention.qHeads.length : c.attention.kvHeads.length) }, () => '<i></i>').join('')}</span><span class="tensor-detail">${t.shape[0]}×${t.shape[1]}</span></div>`; }).join('')}<p class="inspector-note">Q 头 ${esc(rangeText(c.attention.qHeads))} · KV 头 ${esc(rangeText(c.attention.kvHeads))}${c.attention.kvReplicated ? '（复制）' : ''}<br>${config.attentionType === 'mla' ? 'MLA 权重为展开投影近似。' : '矩阵形状为每层局部分片。'}${model?.attentionDescription ? '<br>' + esc(model.attentionDescription) : ''}</p>`;
    let html = `<div class="inspector-body"><div class="inspector-identity"><div class="gpu-identity"><span class="gpu-symbol">▧</span><div><h3>${deviceLabel()} ${String(c.rank).padStart(2, '0')}</h3><small>NODE ${String(c.node).padStart(2, '0')} / LOCAL ${c.localRank}</small></div></div><span class="rank-badge">RANK ${c.rank}</span></div><div class="coords">${['tp', 'pp', 'dp', 'cp', 'pcp', 'dcp', 'ep', 'etp'].filter(k => config[k] > 1 || k === 'tp' || (k === 'etp' && c.moe.expertShards.count > 1)).map(k => `<span class="coord">${k.toUpperCase()} : ${c.coords[k]}</span>`).join('')}</div>
      <div class="inspector-section"><h4>${dot('TP')} Attention <small>L${c.layers.start}–L${c.layers.end}</small></h4>${attentionContent}</div>
      <div class="inspector-section"><h4>${dot('EP')} ${activeMoE ? 'MoE 专家分布' : 'FFN 分布'}<small>${inLayer ? activeMoE ? `TOP-${config.topK} / ${c.moe.experts.length} LOCAL` : 'DENSE LAYER' : '不持有当前层'}</small></h4>`;
    if (activeMoE) html += `<div class="expert-grid">${Array.from({ length: shownExperts }, (_, i) => { const id = i + expertPage; return `<span class="expert-block ${localExperts.has(id) ? '' : 'off'}" title="Expert ${id} ${localExperts.has(id) ? '在此卡' : '在其他卡'}">${id}</span>`; }).join('')}</div><p class="inspector-note">拥有：${esc(rangeText(c.moe.experts, 'E'))}<br>专家中间维度 ${c.moe.expertShards.intermediateStart}–${c.moe.expertShards.intermediateEnd} · ETP ${c.moe.expertShards.index}/${c.moe.expertShards.count}${config.experts > 64 ? '<br>格子显示首个本地专家所在的 64 专家窗口。' : ''}<br>共享专家 ${config.sharedExperts} 个 · 按 TP 切分</p><details class="expert-list"><summary>展开全部 ${c.moe.experts.length} 个本地专家编号</summary><p>${c.moe.experts.map(id=>'E'+id).join(', ')}</p></details>`;
    else html += `<p class="inspector-note">${inLayer ? '本层采用 Dense FFN，按 TP 切分中间维度。' : `当前 L${layer} 位于其他 PP 流水级。`}<br>本卡共有 ${c.layers.moe.length} 个 MoE 层、${c.layers.denseCount} 个 Dense 层。</p>`;
    if (activeMoE && c.moe.routedHiddenSize !== config.hiddenSize) html += `<p class="inspector-note">LatentMoE：路由专家 Hidden ${c.moe.routedHiddenSize}，共享专家 Hidden ${c.moe.sharedHiddenSize}。已计入双向适配投影。</p>`;
    if (activeMoE && c.moe.hashLayers?.includes(layer)) html += '<p class="inspector-note">本层使用 Hash Router；查找表占用未计量。</p>';
    html += `</div><div class="inspector-section"><h4>${dot('PCP')} 上下文与 KV <small>${c.kv.modeled === false ? 'UNMODELED' : c.kv.precision.toUpperCase()}</small></h4><p class="inspector-note">${c.kv.modeled === false ? '仅按配置分配查询逻辑范围：' : 'Prefill 查询：'}${c.tokens.prefillStart}–${c.tokens.prefillEnd}<br>${c.kv.modeled === false ? '压缩 / 共享 KV 或线性状态未建模，不能以 0 占用解释。' : `KV token：${c.kv.tokens ? `${c.kv.start}–${c.kv.end}` : '无'}${c.kv.stride > 1 ? `，步长 ${c.kv.stride}` : ''}<br>本卡 ${formatCount(c.kv.tokens)} tokens / 序列 × ${c.kv.batch} 序列`}${config.sp ? '<br>SP 为候选激活分组，不额外切分权重。' : ''}</p></div>
      <div class="inspector-section"><h4>${c.partialEstimate ? '已建模部分构成' : '显存构成'} <small>ALL LOCAL LAYERS</small></h4><div class="memory-bar">${memory.map(([key,,color]) => `<span class="memory-segment" style="width:${Math.max(0, c.memory[key] / Math.max(capacity, c.memory.total) * 100)}%;background:${color}"></span>`).join('')}</div>${memory.filter(([key]) => c.memory[key] > 0).map(([key, label, color]) => `<div class="memory-key"><span><i class="color-dot" style="--group-color:${color}"></i>${label}</span><strong>${formatBytes(c.memory[key])}</strong></div>`).join('')}<div class="memory-total"><span>${c.partialEstimate ? '部分结构估算' : '权重 + KV'}</span><b>${formatBytes(c.memory.total)}</b></div><p class="inspector-note" style="margin-top:6px">${c.partialEstimate ? '未计量模块见上方提示，不能据此判断完整显存需求。' : '不含激活、工作区与通信缓冲。'}</p></div></div>`;
    $('card-inspector').innerHTML = html;
  }
  function renderInsights() {
    if (!result.valid) { $('insights').innerHTML = '<div class="insights"><p class="insight">参数通过校验后展示复制关系与容量洞察。</p></div>'; return; }
    const s = result.stats, entries = [];
    if (config.ep > 1) entries.push(`${config.attentionType === 'unmodeled' ? '候选 Attention 逻辑组为' : 'Attention 按'} <strong>TP${config.tp}</strong>；专家按 <strong>EP${config.ep}</strong> 分配，两者复用设备。`);
    entries.push(`${config.attentionType === 'unmodeled' ? 'Attention 仅展示候选分组，权重与缓存未计量' : `Attention 权重共 <strong>${s.attentionReplication} 份</strong>逻辑副本${config.attentionType === 'mla' ? '（投影近似）' : ''}`}；路由专家复制 ${s.expertReplication} 份。`);
    if (config.pcp > 1 && config.attentionType !== 'unmodeled') entries.push('<strong>PCP 切分计算</strong>，持久 KV 是否减少取决于 DCP。');
    if (config.dcp > 1 && config.attentionType !== 'unmodeled') entries.push(`<strong>DCP${config.dcp}</strong> 复用已有卡，将 KV 序列分为 ${config.dcp} 份。`);
    if (s.imbalance > .01) entries.push(`单卡总容量差异为 <strong>${(s.imbalance * 100).toFixed(1)}%</strong>，请检查层、专家和请求的均衡性。`);
    if (config.pp > 1) entries.push('逐卡容量包含该 PP 级所有网络层；当前层决定 FFN / MoE 视图。');
    entries.push('通信线为逻辑关系；大通信组使用环形摘要，<strong>不表示真实路由或带宽</strong>。');
    $('insights').innerHTML = `<div class="insights">${entries.map(t => `<div class="insight"><span class="insight-icon">✧</span><p>${t}</p></div>`).join('')}${result.warnings.map(t => `<div class="insight warning"><span class="insight-icon">△</span><p>${esc(t)}</p></div>`).join('')}</div>`;
  }
  function renderComparison() {
    if (!baseline) return;
    if (!result.valid) { $('comparison-content').textContent = '基线已保留；当前参数通过校验后显示对比。'; return; }
    if (baseline.stats.partialEstimate !== result.stats.partialEstimate || (baseline.stats.partialEstimate && baseline.config.modelName !== config.modelName)) { $('comparison-content').textContent = '两组配置的计量范围不同，请为同一模型重新保存基线后对比。'; return; }
    const delta = (a, b, format) => `${a > b ? '+' : ''}${format(a - b)}`;
    $('comparison-content').innerHTML = `<div class="comparison-cards"><div class="comparison-card"><span>基线 · ${esc(baseline.config.modelName)}</span><br><strong>${baseline.worldSize} cards · ${formatBytes(baseline.stats.memoryPerCardMax)}</strong><br>${result.stats.partialEstimate ? '最高单卡已建模部分' : '最高单卡权重 + KV'}</div><div class="comparison-card"><span>当前方案 · 与基线差值</span><br><strong>${delta(result.worldSize, baseline.worldSize, n => n)} cards · ${delta(result.stats.memoryPerCardMax, baseline.stats.memoryPerCardMax, n => (n / GIB).toFixed(2))} GiB</strong><br>当前 ${result.worldSize} 卡 · 最高 ${formatBytes(result.stats.memoryPerCardMax)}</div></div>`;
  }
  function showGuide() {
    $('guide-content').innerHTML = `<p>参数和矩阵索引均从 0 开始。该工具用于解释逻辑并行与容量，不自动生成可部署的框架启动命令。</p><div class="guide-grid" style="margin-top:18px">${strategies.map(([type,, name, description]) => `<article class="guide-card"><h3>${dot(type)}<code>${type}</code> ${name}</h3><p>${description}</p></article>`).join('')}</div><div class="guide-assumptions"><h3>计算口径与模型边界</h3><ul>${(result.assumptions.length ? result.assumptions : E.analyze(E.DEFAULT_CONFIG).assumptions).map(t => `<li>${esc(t)}</li>`).join('')}</ul><h3>官方参考资料</h3><ul>${(result.sources || E.SOURCES).map(s => `<li><a href="${esc(s.url)}" target="_blank" rel="noopener">${esc(s.title)} ↗</a></li>`).join('')}</ul></div>`;
    $('guide-dialog').showModal();
  }
  function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob), a = document.createElement('a');
    a.href = url; a.download = filename; document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 2000);
  }
  function filename(ext) { return `parallel-atlas-${phase}-${new Date().toISOString().slice(0, 10)}.${ext}`; }
  function svgText(text, x, y, size = 11, color = '#857e99', extra = '') { return `<text x="${x}" y="${y}" font-size="${size}" fill="${color}" font-family="Segoe UI,Microsoft YaHei,sans-serif" ${extra}>${esc(text)}</text>`; }
  function exportSVG() {
    const graphW = svgSize.width, graphH = svgSize.height;
    let width, height, content;
    const title = `${config.modelName} · L${layer} · ${phase.toUpperCase()}`;
    if (view === 'topology') {
      width = graphW; height = graphH + 115;
      const inner = latestSvg.replace(/^<svg[^>]*>/, '').replace(/<\/svg>$/, '').replaceAll('class="comm-line animated', 'class="comm-line');
      content = `<g transform="translate(0,70)">${inner}</g>`;
    } else if (view === 'weights') {
      width = 1180; height = 160 + result.cards.length * 32;
      const columns = [25, 150, 285, 440, 700, 855, 1005];
      const headers = ['设备 / 节点', '模型层', 'Q / KV 头', '路由专家', result.stats.partialEstimate ? '权重 · 部分' : '权重', 'KV Cache', result.stats.partialEstimate ? '已计量合计' : '合计'];
      content = '<rect x="15" y="85" width="1150" height="32" rx="4" fill="#eeeaf8"/>' + headers.map((t,i) => svgText(t, columns[i], 106, 11)).join('');
      for (const [i,c] of result.cards.entries()) { const y = 139 + i * 32; content += `<path d="M15 ${y + 11}H1165" stroke="#ece8f3"/>` + [`${deviceLabel()} ${c.rank} / N${c.node}`, `L${c.layers.start}–L${c.layers.end}`, c.attention.modeled === false ? '特殊结构 · 未计量' : `${rangeText(c.attention.qHeads)} / ${rangeText(c.attention.kvHeads)}`, rangeText(c.moe.experts,'E'), formatBytes(c.memory.weights),c.kv.modeled === false ? '未计量' : formatBytes(c.kv.bytes),formatBytes(c.memory.total)].map((t,j)=> svgText(t,columns[j],y,10)).join(''); }
    } else {
      const start = Math.floor(selectedRank / 32) * 32, cards = result.cards.slice(start,start+32), cell = 26, offset = 80;
      width = Math.max(650,offset + cards.length*cell+30); height = 190 + cards.length*cell;
      content = svgText(`通信矩阵 R${start}–R${start+cards.length-1} · 筛选 ${selectedType}`,25,98);
      for (let i=0;i<cards.length;i++) {content+=svgText(cards[i].rank,offset+i*cell+7,124,9)+svgText('R'+cards[i].rank,28,145+i*cell,9);for(let j=0;j<cards.length;j++){const matches=visibleGroups().filter(g=>g.active&&g.ranks.includes(cards[i].rank)&&g.ranks.includes(cards[j].rank)&&(g.type!=='PP'||Math.abs(cards[i].coords.pp-cards[j].coords.pp)===1));content+=`<rect x="${offset+j*cell}" y="${130+i*cell}" width="22" height="22" rx="3" fill="${i===j?'#e9e6f1':matches.length?colors[matches[0].type]:'#f0eef7'}"/>`;}}
    }
    const header = svgText('PARALLEL ATLAS',25,27,10,'#8c75da','letter-spacing="2"')+svgText(title,25,52,16,'#4c4368')+svgText(`${result.worldSize} devices · TP${config.tp} PP${config.pp} DP${config.dp} CP${config.cp} PCP${config.pcp} DCP${config.dcp} EP${config.ep} ETP${result.stats.effectiveETP}`,25,72,9);
    return {width,height,svg:`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}"><style>.comm-line{fill:none;opacity:.55;stroke-width:1.5}.comm-line.faded{opacity:.24}.gpu-card.dim{opacity:.28}.gpu-card.selected .gpu-bg{stroke:#8c75da;stroke-width:1.8}</style><rect width="100%" height="100%" fill="#fcfbff"/>${header}${content}${svgText(result.stats.partialEstimate ? '部分结构估算 · Attention / KV 或辅助模块未计量 · 不能判断完整显存需求 · 通信线非物理路由' : (currentModel() ? '官方结构快照 · 投影结构近似，非完整 checkpoint 占用 · 不含激活 / 工作区 · 通信线非物理路由' : '逻辑分布模拟 · 显存为估算，不含激活/工作区 · 通信线非物理路由 · parallel-atlas'),25,height-14,8,'#a8a0b9')}</svg>`};
  }
  async function exportImage(type) {
    if (!result.valid) return;
    $('download-menu').hidden = true;
    try {
      const output = exportSVG();
      if (type === 'svg') { downloadBlob(new Blob([output.svg], { type: 'image/svg+xml;charset=utf-8' }), filename('svg')); notify('已导出当前视图 SVG'); return; }
      const scale = Math.min(2, 15000 / output.height, 15000 / output.width, Math.sqrt(30000000/(output.width*output.height)));
      const canvas = document.createElement('canvas'); canvas.width = Math.ceil(output.width*scale); canvas.height = Math.ceil(output.height*scale);
      const url = URL.createObjectURL(new Blob([output.svg], {type:'image/svg+xml;charset=utf-8'}));
      try { const img = new Image(); await new Promise((resolve,reject) => { img.onload=resolve;img.onerror=()=>reject(new Error('图片渲染失败'));img.src=url; }); const ctx=canvas.getContext('2d'); if(!ctx)throw new Error('浏览器不支持 Canvas');ctx.drawImage(img,0,0,canvas.width,canvas.height);const blob=await new Promise(resolve=>canvas.toBlob(resolve,'image/png'));if(!blob)throw new Error('无法生成 PNG');downloadBlob(blob,filename('png'));notify('已导出当前视图 PNG'); } finally { URL.revokeObjectURL(url); }
    } catch(error) { notify(`导出失败：${error.message}`); }
  }
  function selectCard(rank) {
    selectedRank = rank; selectedGroup = null; renderCanvas(); renderInspector(); renderCommunications();
  }
  function bindEvents() {
    $('config-form').addEventListener('submit',e=>e.preventDefault());
    $('config-form').addEventListener('input',()=>{clearTimeout(updateTimer);updateTimer=setTimeout(recompute,180);});
    $('config-form').addEventListener('change',()=>{clearTimeout(updateTimer);recompute();});
    $('model-select').addEventListener('change',e=>{const model=modelLibrary.find(m=>m.id===e.target.value);if(!model)return;config={...config,...model.config};selectedRank=0;layer=firstMoeLayer(config);fillForm(config);recompute();notify(`已载入 ${model.name} 的公开结构参数`);});
    $('hardware-select').addEventListener('change',e=>{const hardware=hardwareLibrary.find(h=>h.id===e.target.value);config.hardwareId=hardware?.id||'custom';if(Number.isInteger(hardware?.memoryGB))config.memoryGB=hardware.memoryGB;if(Number.isInteger(hardware?.gpusPerNode))config.gpusPerNode=hardware.gpusPerNode;fillForm(config);recompute();notify(hardware?.status==='unverified'?'950Server 规格待核实，已保留当前容量与节点预算':hardware?`已载入 ${hardware.name} 的容量预算`:'已切换为自定义硬件');});
    $('preset-select').addEventListener('change',e=>{const p=E.PRESETS.find(p=>p.id===e.target.value);if(!p)return;config={...p.config};selectedRank=0;layer=firstMoeLayer(config);zoom=1;selectedType='ALL';fillForm(config);recompute();});
    $('reset-button').addEventListener('click',()=>{config={...initialConfig};selectedRank=0;layer=firstMoeLayer(config);zoom=1;selectedType='ALL';fillForm(config);recompute();notify('已恢复默认配置');});
    for(const id of ['guide-button','nav-guide','assumptions-button'])$(id).addEventListener('click',showGuide);
    $('close-guide').addEventListener('click',()=>$('guide-dialog').close());
    $('guide-dialog').addEventListener('click',e=>{if(e.target===$('guide-dialog')){const r=$('guide-dialog').getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)$('guide-dialog').close();}});
    $('nav-explore').addEventListener('click',()=>$('canvas-viewport').scrollIntoView({behavior:'smooth',block:'center'}));
    document.querySelectorAll('[data-view]').forEach(b=>b.addEventListener('click',()=>{view=b.dataset.view;document.querySelectorAll('[data-view]').forEach(x=>{x.classList.toggle('active',x===b);x.setAttribute('aria-selected',x===b?'true':'false');});renderCanvas();}));
    document.querySelectorAll('[data-phase]').forEach(b=>b.addEventListener('click',()=>{phase=b.dataset.phase;selectedGroup=null;document.querySelectorAll('[data-phase]').forEach(x=>x.classList.toggle('active',x===b));renderCanvas();renderCommunications();}));
    $('group-filters').addEventListener('click',e=>{const b=e.target.closest('[data-type]');if(!b)return;selectedType=b.dataset.type;selectedGroup=null;renderFilters();renderCanvas();renderCommunications();});
    $('layer-select').addEventListener('change',()=>{layer=Number($('layer-select').value);selectedGroup=null;renderCanvas();renderCommunications();renderInspector();});
    $('animate-toggle').addEventListener('change',()=>{animate=$('animate-toggle').checked;renderCanvas();});
    $('canvas-content').addEventListener('click',e=>{const page=e.target.closest('[data-matrix-page]');if(page){selectCard(Number(page.dataset.matrixPage));return;}const c=e.target.closest('[data-rank]');if(c)selectCard(Number(c.dataset.rank));});
    $('canvas-content').addEventListener('keydown',e=>{const c=e.target.closest('[data-rank]');if(c&&(e.key==='Enter'||e.key===' ')){e.preventDefault();selectCard(Number(c.dataset.rank));const current=$('canvas-content').querySelector(`[data-rank="${selectedRank}"]`);current?.focus();}});
    const groupSelect=e=>{const g=e.target.closest('[data-group]');if(!g)return;selectedGroup=selectedGroup===g.dataset.group?null:g.dataset.group;renderCanvas();renderCommunications();};
    $('communication-list').addEventListener('click',groupSelect);
    $('communication-list').addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();groupSelect(e);$('communication-list').querySelector(`[data-group="${selectedGroup}"]`)?.focus();}});
    $('zoom-in').addEventListener('click',()=>{zoom=Math.min(2,Math.round((zoom+.1)*10)/10);renderCanvas();});
    $('zoom-out').addEventListener('click',()=>{zoom=Math.max(.4,Math.round((zoom-.1)*10)/10);renderCanvas();});
    $('fit-button').addEventListener('click',()=>{zoom=1;renderCanvas();$('canvas-viewport').scrollTo(0,0);});
    $('download-button').addEventListener('click',()=>$('download-menu').hidden=!$('download-menu').hidden);
    document.addEventListener('click',e=>{if(!e.target.closest('.download-wrap'))$('download-menu').hidden=true;});
    document.querySelectorAll('[data-export]').forEach(b=>b.addEventListener('click',()=>exportImage(b.dataset.export)));
    $('export-button').addEventListener('click',()=>{if(!result.valid){notify('请先修正配置参数');return;}downloadBlob(new Blob([JSON.stringify({schemaVersion:1,application:'parallel-atlas',config},null,2)],{type:'application/json'}),'parallel-atlas-config.json');notify('已导出 JSON 配置');});
    $('import-button').addEventListener('click',()=>$('import-file').click());
    $('import-file').addEventListener('change',async e=>{const file=e.target.files[0];if(!file)return;try{if(file.size>100000)throw new Error('配置文件不能超过 100 KB');const data=JSON.parse(await file.text());if(data.schemaVersion!==undefined&&data.schemaVersion!==1)throw new Error('不支持此配置版本');const candidate=E.analyze(Object.prototype.hasOwnProperty.call(data,'config')?data.config:data);if(!candidate.valid)throw new Error(candidate.errors.join('；'));config=candidate.config;selectedRank=0;layer=firstMoeLayer(config);selectedType='ALL';fillForm(config);recompute();notify('配置已导入');}catch(error){notify('导入失败：'+error.message);}finally{e.target.value='';}});
    $('share-button').addEventListener('click',async()=>{if(!result.valid){notify('请先修正配置参数');return;}const bytes=new TextEncoder().encode(JSON.stringify(config));const hash=btoa(Array.from(bytes,b=>String.fromCharCode(b)).join(''));const url=location.href.split('#')[0]+'#config='+hash;try{await navigator.clipboard.writeText(url);notify('配置分享链接已复制');}catch{location.hash='config='+hash;notify('配置已写入地址栏，可复制链接分享');}});
    $('baseline-button').addEventListener('click',()=>{if(!result.valid)return;baseline=E.analyze(config);renderComparison();notify('当前配置已保存为对比基线');});
  }
  initializeForm();bindEvents();recompute();
})();
