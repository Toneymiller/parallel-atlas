const test = require('node:test');
const assert = require('node:assert/strict');
const { DEFAULT_CONFIG, PRESETS, normalizeConfig, analyze } = require('./engine.js');

const baseline = changes => ({ ...DEFAULT_CONFIG, ...changes });
const near = (actual, expected) => assert.ok(Math.abs(actual - expected) <= Math.max(1, expected) * 1e-12, `${actual} != ${expected}`);

test('every shipped preset has a complete topology and finite estimates', () => {
  for (const preset of PRESETS) {
    const result = analyze(preset.config);
    assert.equal(result.valid, true, `${preset.id}: ${result.errors.join('; ')}`);
    assert.equal(result.cards.length, result.worldSize);
    assert.equal(new Set(result.cards.map(card => card.rank)).size, result.worldSize);
    for (const card of result.cards) {
      assert.equal(card.node, Math.floor(card.rank / result.config.gpusPerNode));
      for (const bytes of Object.values(card.memory)) assert.ok(Number.isFinite(bytes) && bytes >= 0);
      near(card.memory.total, card.memory.weights + card.memory.kv);
      near(card.attention.weightsBytes, Object.values(card.attention.tensors).reduce((sum, tensor) => sum + tensor.layersBytes, 0));
      for (const [type, id] of Object.entries(card.groups)) {
        const group = result.groups.find(item => item.id === id);
        assert.ok(group && group.type === type && group.ranks.includes(card.rank));
      }
    }
    for (const group of result.groups) {
      assert.equal(new Set(group.ranks).size, group.ranks.length);
      for (const rank of group.ranks) assert.ok(rank >= 0 && rank < result.worldSize);
    }
  }
});

test('TP attention weights partition heads without dropping KV replication', () => {
  const result = analyze(baseline({ tp: 8, dp: 1, kvHeads: 4, ep: 8 }));
  assert.equal(result.valid, true);
  const allQHeads = result.cards.flatMap(card => card.attention.qHeads);
  assert.deepEqual(allQHeads, Array.from({ length: 32 }, (_, i) => i));
  for (let head = 0; head < 4; head++) assert.equal(result.cards.filter(card => card.attention.kvHeads.includes(head)).length, 2);
  assert.ok(result.cards.every(card => card.attention.kvReplicated));
  const oneHeadBytes = result.config.hiddenSize * result.config.headDim * 2 * result.config.layers;
  const totalK = result.cards.reduce((sum, card) => sum + card.attention.tensors.k.layersBytes, 0);
  assert.equal(totalK, oneHeadBytes * 4 * 2);
});

test('generic EP×ETP covers every expert shard once per expert replica', () => {
  const result = analyze(baseline({ tp: 4, dp: 2, ep: 4, etp: 2 }));
  assert.equal(result.valid, true);
  for (let expert = 0; expert < result.config.experts; expert++) {
    const owners = result.cards.filter(card => card.moe.experts.includes(expert));
    assert.equal(owners.length, 2);
    assert.deepEqual(owners.map(card => card.coords.etp).sort(), [0, 1]);
    const shardWidths = owners.map(card => card.moe.expertShards.intermediateEnd - card.moe.expertShards.intermediateStart + 1);
    assert.equal(shardWidths.reduce((sum, width) => sum + width, 0), result.config.moeIntermediateSize);
  }
  const expectedMoe = result.config.moeLayers * result.config.experts * 3 * result.config.hiddenSize * result.config.moeIntermediateSize * 2;
  near(result.cards.reduce((sum, card) => sum + card.moe.weightsBytes, 0), expectedMoe);
  assert.equal(result.stats.expertReplication, 1);
});

test('uneven and round-robin expert placement preserve all experts', () => {
  for (const expertPlacement of ['contiguous', 'round-robin']) {
    const result = analyze(baseline({ experts: 65, expertPlacement }));
    assert.equal(result.valid, true);
    assert.equal(result.cards.flatMap(card => card.moe.experts).length, 65);
    assert.equal(new Set(result.cards.flatMap(card => card.moe.experts)).size, 65);
    assert.ok(result.warnings.some(message => message.includes('不均衡')));
  }
  const uneven = analyze(baseline({ profile: 'vllm-ascend', experts: 67 }));
  assert.equal(uneven.valid, true);
  assert.deepEqual(uneven.cards.map(card => card.moe.experts.length), [9, 9, 9, 8, 8, 8, 8, 8]);
  assert.deepEqual(uneven.cards[0].moe.experts, [0, 1, 2, 3, 4, 5, 6, 7, 8]);
  assert.equal(uneven.cards[1].moe.experts[0], 9);
});

test('PCP increases workers and compute sharding, but alone leaves persisted KV replicated', () => {
  const original = analyze(baseline({ profile: 'vllm-ascend', tp: 4, dp: 1, pcp: 1, ep: 4, batchSize: 3 }));
  const pcp = analyze(baseline({ profile: 'vllm-ascend', tp: 4, dp: 1, pcp: 2, ep: 8, batchSize: 3 }));
  assert.equal(original.valid && pcp.valid, true);
  assert.equal(pcp.worldSize, original.worldSize * 2);
  near(pcp.stats.totalKVBytes, original.stats.totalKVBytes * 2);
  assert.equal(pcp.cards[0].tokens.prefillCount * 2, original.cards[0].tokens.prefillCount);
  assert.equal(pcp.cards[0].kv.tokens, original.cards[0].kv.tokens);
});

test('PCP+DCP reuses cards and partitions KV tokens exactly even for odd sequence lengths', () => {
  const noDcp = analyze(baseline({ profile: 'vllm-ascend', tp: 4, dp: 1, pcp: 2, ep: 8, sequenceLength: 8193 }));
  const withDcp = analyze(baseline({ profile: 'vllm-ascend', tp: 4, dp: 1, pcp: 2, dcp: 2, ep: 8, sequenceLength: 8193 }));
  assert.equal(withDcp.valid, true);
  assert.equal(withDcp.worldSize, noDcp.worldSize);
  near(withDcp.stats.totalKVBytes * 2, noDcp.stats.totalKVBytes);
  for (const group of withDcp.groups.filter(group => group.type === 'DCP')) {
    assert.equal(group.ranks.length, 2);
    const cards = group.ranks.map(rank => withDcp.cards[rank]);
    assert.equal(cards.reduce((sum, card) => sum + card.kv.tokens, 0), 8193);
    const positions = cards.flatMap(card => Array.from({ length: card.kv.tokens }, (_, i) => card.kv.start + i * card.kv.stride));
    assert.equal(new Set(positions).size, 8193);
    assert.equal(Math.min(...positions), 0);
    assert.equal(Math.max(...positions), 8192);
    assert.deepEqual(cards[0].attention.kvHeads, cards[1].attention.kvHeads);
  }
});

test('standalone GQA DCP only shares ranks with replicated KV heads', () => {
  const result = analyze(baseline({ profile: 'vllm-ascend', tp: 8, dp: 1, kvHeads: 4, dcp: 2, ep: 8 }));
  assert.equal(result.valid, true);
  for (const group of result.groups.filter(group => group.type === 'DCP')) {
    const cards = group.ranks.map(rank => result.cards[rank]);
    assert.equal(cards.length, 2);
    assert.deepEqual(cards[0].attention.kvHeads, cards[1].attention.kvHeads);
  }
  assert.equal(analyze(baseline({ tp: 4, dp: 1, ep: 4, kvHeads: 8, dcp: 2 })).valid, false);
});

test('MLA full-grid DCP and compressed KV use the correct reuse and latent dimensions', () => {
  const preset = PRESETS.find(item => item.id === 'mla');
  const result = analyze(preset.config);
  assert.equal(result.valid, true);
  assert.equal(result.worldSize, 4);
  const groups = result.groups.filter(group => group.type === 'DCP');
  assert.equal(groups.length, 1);
  assert.deepEqual(groups[0].ranks, [0, 2, 1, 3]);
  assert.equal(result.cards[0].kv.elementsPerToken, 576);
  assert.ok(result.warnings.some(message => message.includes('MLA')));
});

test('Ascend EP disabled tensor-shards experts over TP×DP×PCP', () => {
  const result = analyze(baseline({ profile: 'vllm-ascend', tp: 2, dp: 2, pcp: 2, ep: 1 }));
  assert.equal(result.valid, true);
  assert.equal(result.stats.effectiveETP, 8);
  assert.ok(result.cards.every(card => card.moe.experts.length === 64));
  assert.deepEqual(new Set(result.cards.map(card => card.coords.etp)), new Set([0, 1, 2, 3, 4, 5, 6, 7]));
});

test('PP assigns each layer and each MoE layer exactly once per model replica', () => {
  const result = analyze(baseline({ layers: 33, moeLayers: 25, pp: 4, tp: 1, dp: 1, ep: 1, etp: 1 }));
  assert.equal(result.valid, true);
  assert.equal(result.cards.reduce((sum, card) => sum + card.layers.count, 0), 33);
  assert.equal(result.cards.reduce((sum, card) => sum + card.layers.moe.length, 0), 25);
  near(result.stats.totalWeightBytes, result.stats.uniqueWeightBytes);
  const layers = result.cards.flatMap(card => Array.from({ length: card.layers.count }, (_, index) => card.layers.start + index));
  assert.deepEqual(layers, Array.from({ length: 33 }, (_, index) => index));
});

test('trailing MoE placement preserves real first-dense layers across arbitrary PP stages', () => {
  for (const [layers, moeLayers] of [[61, 58], [92, 89], [4, 0], [4, 4]]) {
    const result = analyze(baseline({ layers, moeLayers, moeLayerPlacement: 'trailing', pp: 4, tp: 1, dp: 1, ep: 1, etp: 1 }));
    assert.equal(result.valid, true);
    const actual = result.cards.flatMap(card => card.layers.moe);
    const expected = Array.from({ length: moeLayers }, (_, i) => layers - moeLayers + i);
    assert.deepEqual(actual, expected);
    for (const card of result.cards) assert.equal(card.layers.denseCount + card.layers.moe.length, card.layers.count);
    near(result.stats.totalWeightBytes, result.stats.uniqueWeightBytes);
  }
  for (const strategy of [{ ep: 2, etp: 1 }, { ep: 1, etp: 2 }]) {
    const result = analyze(baseline({ layers: 4, moeLayers: 2, moeLayerPlacement: 'trailing', pp: 4, tp: 1, dp: 2, ...strategy }));
    assert.equal(result.valid, true);
    for (const group of result.groups.filter(group => ['EP', 'ETP'].includes(group.type))) {
      const hasMoE = group.ranks.some(rank => result.cards[rank].layers.moe.length);
      assert.equal(group.active, hasMoE && group.ranks.length > 1);
    }
  }
  assert.throws(() => normalizeConfig({ moeLayerPlacement: 'unsupported' }));
});

test('weight precision does not alter KV precision; DP accounts for uneven requests', () => {
  const fp = analyze(baseline({ batchSize: 3 }));
  const int4 = analyze(baseline({ precision: 'int4', batchSize: 3 }));
  assert.equal(fp.valid && int4.valid, true);
  near(int4.stats.totalWeightBytes * 4, fp.stats.totalWeightBytes);
  near(int4.stats.totalKVBytes, fp.stats.totalKVBytes);
  assert.equal(fp.cards.find(card => card.coords.dp === 0).kv.batch, 1);
  assert.equal(fp.cards.find(card => card.coords.dp === 1).kv.batch, 2);
});

test('CP and SP add only their documented dimensions and communication', () => {
  const original = analyze(baseline({ tp: 2, dp: 1, ep: 2 }));
  const cp = analyze(baseline({ tp: 2, dp: 1, ep: 2, cp: 4 }));
  const sp = analyze(baseline({ tp: 2, dp: 1, ep: 2, sp: true }));
  assert.equal(cp.valid && sp.valid, true);
  assert.equal(cp.worldSize, original.worldSize * 4);
  near(cp.stats.totalWeightBytes, original.stats.totalWeightBytes * 4);
  near(cp.stats.totalKVBytes, original.stats.totalKVBytes);
  assert.equal(sp.worldSize, original.worldSize);
  near(sp.stats.totalWeightBytes, original.stats.totalWeightBytes);
  assert.ok(sp.groups.some(group => group.type === 'SP' && group.active));
  assert.ok(sp.groups.filter(group => group.type === 'DP').every(group => !group.active && group.collective === 'Request routing'));
});

test('invalid dimensions and unsupported framework combinations never create partial cards', () => {
  const cases = [
    { tp: 3 }, { pp: 40 }, { ep: 3 }, { cp: 2, pcp: 2 }, { tp: 512, dp: 2 },
    { profile: 'vllm-ascend', cp: 2 }, { profile: 'vllm-ascend', ep: 2 },
    { profile: 'vllm-ascend', etp: 2 }, { profile: 'vllm-ascend', tp: 4, dp: 1, pcp: 2, dcp: 4, ep: 8 },
    { profile: 'vllm-ascend', sp: true, ep: 1 },
    { heads: 31 }, { kvHeads: 7 }, { topK: 0 }, { moeLayers: 33 }, { experts: 0 }
  ];
  for (const changes of cases) {
    const result = analyze(baseline(changes));
    assert.equal(result.valid, false, JSON.stringify(changes));
    assert.ok(result.errors.length > 0);
    assert.equal(result.cards.length, 0);
    assert.equal(result.groups.length, 0);
  }
});

test('unused Dense FFN dimensions do not restrict fully MoE models', () => {
  for (const attentionType of ['gqa', 'unmodeled']) {
    const config = baseline({ attentionType, moeLayers: DEFAULT_CONFIG.layers, intermediateSize: 3 });
    const result = analyze(config);
    assert.equal(result.valid, true, result.errors.join('; '));
    assert.ok(result.cards.every(card => card.layers.denseCount === 0 && card.memory.dense === 0));
    near(result.stats.totalWeightBytes, analyze({ ...config, intermediateSize: 4 }).stats.totalWeightBytes);
    const withDense = analyze({ ...config, moeLayers: config.layers - 1 });
    assert.equal(withDense.valid, false);
    assert.ok(withDense.errors.some(error => error.includes('intermediateSize')));
    assert.equal(withDense.cards.length, 0);
  }
});

test('current Ascend PCP supports SP MoE and defaults to TP×PCP resident vocab shards', () => {
  const result = analyze(baseline({ profile: 'vllm-ascend', tp: 4, dp: 1, pcp: 2, ep: 8, sp: true, vocabSize: 32003 }));
  assert.equal(result.valid, true);
  assert.ok(result.groups.some(group => group.type === 'SP' && group.active));
  const shards = result.cards.map(card => card.embedding);
  assert.equal(shards.reduce((sum, shard) => sum + shard.count, 0), 32003);
  assert.equal(new Set(shards.map(shard => shard.shardIndex)).size, 8);
  assert.ok(shards.every(shard => shard.shardCount === 8));
  const expectedEmbedding = 2 * 32003 * result.config.hiddenSize * 2;
  near(result.cards.reduce((sum, card) => sum + card.memory.embedding, 0), expectedEmbedding);
  for (const changes of [
    { profile: 'vllm-ascend', sp: true, tp: 1, dp: 2, ep: 2 },
    { profile: 'vllm-ascend', sp: true, experts: 0, topK: 0, moeLayers: 0, sharedExperts: 0 }
  ]) assert.equal(analyze(baseline(changes)).valid, false);
});

test('normalization rejects unsafe and malformed inputs and whitelists imported fields', () => {
  for (const value of [null, [], false, 'text']) assert.throws(() => normalizeConfig(value));
  for (const value of ['', '  ', Infinity, NaN, -1, 1.5, true, {}, 10000000]) assert.throws(() => normalizeConfig({ tp: value }));
  assert.throws(() => normalizeConfig({ sp: 'false' }));
  assert.throws(() => normalizeConfig({ precision: 'fp8' }));
  assert.throws(() => normalizeConfig({ modelName: 'x'.repeat(121) }));
  assert.equal(normalizeConfig({ tp: '8' }).tp, 8);
  assert.equal(normalizeConfig({ hardwareId: 'future-gpu-123' }).hardwareId, 'future-gpu-123');
  for (const hardwareId of ['', 'A100', '../bad', 'x'.repeat(65), 123]) assert.throws(() => normalizeConfig({ hardwareId }));
  const imported = normalizeConfig(JSON.parse('{"__proto__":{"polluted":true},"unknown":"ignored","tp":4}'));
  assert.equal(imported.unknown, undefined);
  assert.equal({}.polluted, undefined);
  assert.equal(analyze({ tp: 'bad' }).valid, false);
});

test('unmodeled modern attention reports only its modeled subset and never fake MLA or KV bytes', () => {
  const result = analyze(baseline({ attentionType: 'unmodeled', kvHeads: 7, dcp: 2, sp: true }));
  assert.equal(result.valid, true, result.errors.join('; '));
  assert.equal(result.stats.partialEstimate, true);
  assert.equal(result.stats.estimateScope, 'modeled-subset');
  assert.ok(result.stats.excludedModules.some(item => item.includes('Attention')));
  assert.ok(result.stats.excludedModules.some(item => item.includes('线性注意力状态')));
  assert.ok(result.warnings.some(item => item.includes('不能据此判断')));
  assert.ok(result.warnings.every(item => !item.includes('容量下界')));
  assert.equal(result.stats.totalKVBytes, 0);
  for (const card of result.cards) {
    assert.equal(card.partialEstimate, true);
    assert.equal(card.attention.modeled, false);
    assert.equal(card.attention.approximation, 'unmodeled');
    assert.equal(card.memory.attention, 0);
    assert.equal(card.memory.kv, 0);
    assert.equal(card.kv.bytes, 0);
    assert.equal(card.kv.modeled, false);
    assert.equal(card.kv.mode, 'unmodeled');
    assert.ok(Object.values(card.attention.tensors).every(tensor => !tensor.modeled && tensor.bytes === 0 && tensor.layersBytes === 0));
    const sum = ['attention', 'moe', 'shared', 'dense', 'embedding', 'norm', 'router', 'expertAdapters'].reduce((sum, field) => sum + card.memory[field], 0);
    near(card.memory.weights, sum);
  }
  assert.ok(result.groups.every(group => !['CP', 'PCP', 'DCP', 'SP'].includes(group.type)));
  assert.ok(result.cards.every(card => ['CP', 'PCP', 'DCP', 'SP'].every(type => !(type in card.groups))));
  for (const group of result.groups.filter(group => group.type === 'TP')) {
    assert.match(group.description, /已计量的 Dense\/shared FFN/);
    assert.match(group.description, /不表示 Attention 的切分或通信/);
    assert.doesNotMatch(group.description, /候选/);
  }
  assert.ok(result.assumptions.every(assumption => !assumption.includes('候选')));
  assert.equal(analyze(baseline({ attentionType: 'unmodeled', heads: 31 })).valid, false);
  const known = analyze(DEFAULT_CONFIG);
  assert.equal(known.stats.partialEstimate, false);
  assert.ok(known.stats.totalKVBytes > 0);
});

test('unknown attention omits context and sequence communication groups for every logical grid', () => {
  const unknownTypes = ['CP', 'PCP', 'DCP', 'SP'];
  for (const changes of [{ cp: 2, sp: true }, { pcp: 2, dcp: 2, sp: true }, { pcp: 2, dcp: 8, sp: true }]) {
    const result = analyze(baseline({ attentionType: 'unmodeled', ...changes }));
    assert.equal(result.valid, true, result.errors.join('; '));
    assert.ok(result.groups.every(group => !unknownTypes.includes(group.type)));
    assert.ok(result.cards.every(card => unknownTypes.every(type => !(type in card.groups))));
    assert.ok(result.groups.some(group => group.type === 'TP' && group.active));
    assert.ok(result.groups.some(group => group.type === 'EP' && group.active));
  }
  const invalid = analyze(baseline({ attentionType: 'unmodeled', pcp: 2, dcp: 4 }));
  assert.equal(invalid.valid, false);
  assert.ok(invalid.errors.includes('当前逻辑设备网格要求 DCP=PCP 或 DCP=PCP×TP。'));
  for (const attentionType of ['gqa', 'mla']) {
    const result = analyze(baseline({ attentionType, pcp: 2, dcp: 2, sp: true }));
    assert.equal(result.valid, true, result.errors.join('; '));
    for (const type of unknownTypes) assert.ok(result.groups.some(group => group.type === type));
    for (const type of ['PCP', 'DCP', 'SP']) assert.ok(result.groups.some(group => group.type === type && group.active));
    const cp = analyze(baseline({ attentionType, cp: 2 }));
    assert.equal(cp.valid, true, cp.errors.join('; '));
    assert.ok(cp.groups.some(group => group.type === 'CP' && group.active));
  }
});

test('LatentMoE routes through smaller expert hidden dimension while preserving shared experts', () => {
  const original = analyze(baseline({ tp: 4, dp: 2, ep: 4, etp: 2 }));
  const latent = analyze(baseline({ tp: 4, dp: 2, ep: 4, etp: 2, routedExpertHiddenSize: 2048 }));
  assert.equal(latent.valid, true);
  near(latent.stats.activatedExpertParameters * 2, original.stats.activatedExpertParameters);
  for (let rank = 0; rank < latent.worldSize; rank++) {
    const before = original.cards[rank], card = latent.cards[rank];
    near(card.memory.moe * 2, before.memory.moe);
    near(card.memory.shared, before.memory.shared);
    near(card.memory.router, before.memory.router);
    near(card.memory.attention, before.memory.attention);
    near(card.memory.kv, before.memory.kv);
    near(card.memory.expertAdapters, card.layers.moe.length * (2 * 4096 * 2048 / 4 + 2048) * 2);
    assert.equal(card.moe.routedHiddenSize, 2048);
    assert.equal(card.moe.sharedHiddenSize, 4096);
  }
  const oneReplica = analyze(baseline({ tp: 1, dp: 1, pp: 4, ep: 1, etp: 1, routedExpertHiddenSize: 2048 }));
  near(oneReplica.stats.totalWeightBytes, oneReplica.stats.uniqueWeightBytes);
  near(oneReplica.cards.reduce((sum, card) => sum + card.memory.expertAdapters, 0), oneReplica.stats.uniqueExpertAdapterBytes);
  assert.equal(analyze(baseline({ routedExpertHiddenSize: 4096 })).cards[0].memory.expertAdapters, 0);
  assert.equal(analyze(baseline({ routedExpertHiddenSize: 2047 })).valid, false);
  assert.throws(() => normalizeConfig({ routedExpertHiddenSize: 65537 }));
});

test('Hash MoE preserves every gate matrix across PP while only lookup tables remain excluded', () => {
  const config = baseline({ attentionType: 'unmodeled', layers: 8, moeLayers: 5, moeLayerPlacement: 'trailing', hashMoeLayers: 3, tp: 1, pp: 4, dp: 1, ep: 1, etp: 1 });
  const hashed = analyze(config), standard = analyze({ ...config, hashMoeLayers: 0 });
  assert.equal(hashed.valid, true);
  assert.deepEqual(hashed.cards.flatMap(card => card.moe.hashLayers), [3, 4, 5]);
  near(hashed.stats.uniqueWeightBytes, standard.stats.uniqueWeightBytes);
  near(hashed.stats.totalWeightBytes, standard.stats.totalWeightBytes);
  for (const card of hashed.cards) {
    near(card.memory.router, card.layers.moe.length * config.hiddenSize * config.experts * 2);
    near(card.memory.router, standard.cards[card.rank].memory.router);
  }
  near(hashed.stats.totalWeightBytes, hashed.stats.uniqueWeightBytes);
  assert.ok(hashed.stats.excludedModules.includes('Hash Router 查找表'));
  assert.ok(hashed.warnings.some(warning => warning.includes('仍保留 hidden×experts Router 矩阵')));
  assert.ok(hashed.assumptions.some(assumption => assumption.includes('所有 MoE 层均计入该矩阵')));
  const knownAttention = analyze({ ...config, attentionType: 'gqa' });
  assert.equal(knownAttention.stats.partialEstimate, true);
  assert.ok(knownAttention.stats.totalKVBytes > 0);
  assert.equal(analyze({ ...config, hashMoeLayers: 6 }).valid, false);
});

test('maximum supported world stays bounded and groups cover each rank once per type', () => {
  const result = analyze(baseline({ tp: 8, pp: 8, dp: 8, ep: 64, etp: 1 }));
  assert.equal(result.valid, true);
  assert.equal(result.worldSize, 512);
  assert.ok(result.groups.length < 5000);
  for (const type of new Set(result.groups.map(group => group.type))) {
    const ranks = result.groups.filter(group => group.type === type).flatMap(group => group.ranks);
    assert.equal(ranks.length, result.worldSize, type);
    assert.equal(new Set(ranks).size, result.worldSize, type);
  }
});
