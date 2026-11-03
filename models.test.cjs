'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const E = require('./engine.js');
const C = require('./models.js');

test('catalog is available in plain browsers without imports or network access', () => {
  const sandbox = { window: {} };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, 'models.js'), 'utf8'), sandbox);
  assert.equal(sandbox.window.ModelCatalog.MODELS.length, 15);
  assert.equal(sandbox.window.ModelCatalog.getModel('deepseek-v3').config.attentionType, 'mla');
  assert.equal(sandbox.window.ModelCatalog.getModel('unknown'), undefined);
});

test('each model has dated official source provenance and immutable model-only fields', () => {
  const ids = new Set();
  const modelFields = new Set(['modelName', 'attentionType', 'layers', 'hiddenSize', 'heads', 'kvHeads', 'headDim', 'intermediateSize', 'experts', 'topK', 'moeLayers', 'sharedExperts', 'moeIntermediateSize', 'routedExpertHiddenSize', 'hashMoeLayers', 'moeLayerPlacement', 'vocabSize', 'precision', 'kvPrecision', 'kvLoraRank', 'qkRopeHeadDim']);
  for (const model of C.MODELS) {
    assert.ok(!ids.has(model.id)); ids.add(model.id);
    assert.equal(new URL(model.source).origin, 'https://huggingface.co');
    assert.ok(model.source.includes(`/${model.organization}/`));
    assert.ok(model.source.endsWith('/raw/main/config.json'));
    assert.equal(model.verifiedAt, '2026-11-03');
    assert.match(model.architecture, /For(?:CausalLM|ConditionalGeneration)$/);
    assert.ok(['current', 'legacy'].includes(model.generation));
    assert.ok(['projection-proxy', 'partial'].includes(model.support));
    assert.ok(model.attentionDescription.length > 5);
    assert.ok(model.notes.length >= 2);
    assert.ok(Object.isFrozen(model) && Object.isFrozen(model.config) && Object.isFrozen(model.notes));
    assert.ok(Object.keys(model.config).every(key => modelFields.has(key)));
    assert.ok(Object.keys(model.config).every(key => Object.prototype.hasOwnProperty.call(E.DEFAULT_CONFIG, key)), `${model.id} contains an unsupported engine field`);
    assert.equal(model.config.precision, 'bf16');
  }
  assert.equal(ids.size, 15);
  assert.equal(C.MODELS.filter(m => m.generation === 'current').length, 9);
  assert.equal(C.MODELS.filter(m => m.generation === 'legacy').length, 6);
  assert.ok(C.MODELS.slice(0, 9).every(m => m.generation === 'current'));
});

test('all catalog structures normalize and produce finite layouts with an independent TP1 plan', () => {
  for (const model of C.MODELS) {
    const result = E.analyze({ ...E.DEFAULT_CONFIG, ...model.config, tp: 1, pp: 1, dp: 1, cp: 1, pcp: 1, dcp: 1, ep: 1, etp: 1, sp: false });
    assert.equal(result.valid, true, `${model.id}: ${result.errors.join('; ')}`);
    assert.equal(result.cards.length, 1);
    assert.equal(result.config.moeLayerPlacement, model.config.moeLayerPlacement);
    assert.ok(Number.isFinite(result.stats.memoryPerCardMax));
    assert.equal(result.cards[0].layers.moe.length, model.config.moeLayers);
  }
});

test('Dense catalog entries cannot retain phantom MoE parameters', () => {
  for (const id of ['qwen3-8b', 'qwen3-32b']) {
    const c = C.getModel(id).config;
    for (const key of ['experts', 'topK', 'moeLayers', 'sharedExperts']) assert.equal(c[key], 0);
    const result = E.analyze({ ...E.DEFAULT_CONFIG, ...c, tp: 4, dp: 1, ep: 1, etp: 1 });
    assert.equal(result.valid, true);
    assert.ok(result.cards.every(card => card.memory.moe === 0 && card.memory.shared === 0 && card.memory.router === 0));
  }
});

test('Qwen source dimensions preserve explicit head_dim rather than hidden_size / heads', () => {
  const expected = [
    ['qwen3-8b', 36, 4096, 32, 8, 12288, 0, 0],
    ['qwen3-32b', 64, 5120, 64, 8, 25600, 0, 0],
    ['qwen3-30b-a3b', 48, 2048, 32, 4, 6144, 128, 768],
    ['qwen3-235b-a22b', 94, 4096, 64, 4, 12288, 128, 1536]
  ];
  for (const [id, layers, hiddenSize, heads, kvHeads, intermediateSize, experts, moeIntermediateSize] of expected) {
    const c = C.getModel(id).config;
    assert.deepEqual([c.layers, c.hiddenSize, c.heads, c.kvHeads, c.headDim, c.intermediateSize, c.experts, c.vocabSize], [layers, hiddenSize, heads, kvHeads, 128, intermediateSize, experts, 151936]);
    if (experts) {
      assert.equal(c.moeIntermediateSize, moeIntermediateSize);
      assert.equal(c.moeLayers, layers);
      assert.equal(c.topK, 8);
      assert.equal(c.sharedExperts, 0);
    }
  }
});

test('DeepSeek and GLM preserve first three Dense layers and verified shared-expert counts', () => {
  const deepseek = C.getModel('deepseek-v3').config;
  const glm = C.getModel('glm-4.5').config;
  assert.deepEqual([deepseek.layers, deepseek.moeLayers, deepseek.hiddenSize, deepseek.experts, deepseek.sharedExperts, deepseek.topK, deepseek.vocabSize], [61, 58, 7168, 256, 1, 8, 129280]);
  assert.deepEqual([glm.layers, glm.moeLayers, glm.hiddenSize, glm.experts, glm.sharedExperts, glm.topK, glm.vocabSize], [92, 89, 5120, 160, 1, 8, 151552]);
  for (const c of [deepseek, glm]) {
    assert.equal(c.moeLayerPlacement, 'trailing');
    const result = E.analyze({ ...E.DEFAULT_CONFIG, ...c, tp: 1, pp: 2, dp: 1, ep: 1, etp: 1 });
    assert.equal(result.valid, true);
    const layers = result.cards.flatMap(card => card.layers.moe);
    assert.equal(Math.min(...layers), 3);
    assert.equal(Math.max(...layers), c.layers - 1);
    assert.equal(new Set(layers).size, c.moeLayers);
  }
});

test('DeepSeek MLA KV uses source latent dimensions while weight approximation stays explicit', () => {
  const model = C.getModel('deepseek-v3');
  assert.equal(model.config.kvLoraRank, 512);
  assert.equal(model.config.qkRopeHeadDim, 64);
  assert.equal(model.config.headDim, 128 + 64);
  const result = E.analyze({ ...E.DEFAULT_CONFIG, ...model.config, tp: 1, dp: 1, ep: 1, etp: 1 });
  assert.equal(result.cards[0].kv.elementsPerToken, 576);
  assert.equal(result.cards[0].attention.approximation, 'expanded-projection proxy');
  assert.ok(model.notes.some(note => note.includes('V head dim=128')));
  assert.ok(model.notes.some(note => note.includes('FP8')));
});

test('current GLM DSA entries preserve effective QK dimension and first three Dense layers', () => {
  for (const id of ['glm-5.2', 'glm-5.3']) {
    const model = C.getModel(id), c = model.config;
    assert.deepEqual([c.layers, c.hiddenSize, c.heads, c.kvHeads, c.headDim, c.experts, c.topK, c.moeLayers, c.sharedExperts, c.vocabSize], [78, 6144, 64, 64, 256, 256, 8, 75, 1, 154880]);
    assert.equal(c.attentionType, 'mla');
    assert.equal(model.support, 'projection-proxy');
    assert.ok(model.notes.some(note => note.includes('192') && note.includes('64')));
    const r = E.analyze({ ...E.DEFAULT_CONFIG, ...c, tp: 4, dp: 2, ep: 8, etp: 1 });
    assert.equal(r.valid, true);
    assert.equal(r.cards[0].layers.moe[0], 3);
    assert.equal(r.cards[0].kv.elementsPerToken, 576);
  }
  assert.notEqual(C.getModel('glm-5.2').config.modelName, C.getModel('glm-5.3').config.modelName);
});

test('hybrid attention arrays are immutable and retain source layer indexing', () => {
  for (const model of C.MODELS.filter(m => m.generation === 'current')) {
    assert.equal(model.attentionLayers.length, model.config.layers, model.id);
    assert.ok(Object.isFrozen(model.attentionLayers));
  }
  const flash = C.getModel('glm-5.3-flash');
  assert.equal(flash.attentionLayers.filter(t => t === 'KDA').length, 34);
  assert.equal(flash.attentionLayers.filter(t => t === 'DSA').length, 11);
  assert.equal(flash.attentionLayers[3], 'DSA');
  assert.equal(flash.attentionLayers[44], 'KDA');
  const kimi = C.getModel('kimi-k3');
  assert.equal(kimi.attentionLayers.filter(t => t === 'KDA').length, 69);
  assert.equal(kimi.attentionLayers.filter(t => t === 'MLA').length, 24);
  assert.equal(kimi.attentionLayers[0], 'KDA');
  assert.equal(kimi.attentionLayers[3], 'MLA');
  assert.deepEqual(kimi.attentionLayers.slice(-2), ['MLA', 'MLA']);
});

test('partial models explicitly omit unknown attention and state instead of estimating full fit', () => {
  const models = C.MODELS.filter(m => m.support === 'partial');
  assert.equal(models.length, 6);
  for (const model of models) {
    assert.equal(model.config.attentionType, 'unmodeled');
    const r = E.analyze({ ...E.DEFAULT_CONFIG, ...model.config, tp: 4, pp: 2, dp: 2, ep: 8, etp: 1 });
    assert.equal(r.valid, true, model.id);
    assert.equal(r.stats.partialEstimate, true);
    assert.equal(r.stats.estimateScope, 'modeled-subset');
    for (const card of r.cards) {
      assert.equal(card.partialEstimate, true);
      assert.equal(card.attention.modeled, false);
      assert.equal(card.kv.modeled, false);
      assert.equal(card.memory.attention, 0);
      assert.equal(card.memory.kv, 0);
      assert.ok(card.memory.total > 0);
    }
    assert.ok(model.notes.some(note => note.includes('未计入') && note.includes('显存')));
  }
});

test('DeepSeek V4 has all-MoE backbones and only V4 has source-confirmed Hash Router layers', () => {
  const expected = [
    ['deepseek-v4-flash-0731', 43, 4096, 64, 256, 2048, 3],
    ['deepseek-v4-pro', 61, 7168, 128, 384, 3072, 3],
    ['deepseek-v4.1-flash-1017', 40, 5120, 64, 384, 2304, 0]
  ];
  for (const [id, layers, hidden, heads, experts, dim, hash] of expected) {
    const c = C.getModel(id).config;
    assert.deepEqual([c.layers, c.moeLayers, c.hiddenSize, c.heads, c.experts, c.moeIntermediateSize, c.hashMoeLayers, c.topK, c.sharedExperts], [layers, layers, hidden, heads, experts, dim, hash, 6, 1]);
    const r = E.analyze({ ...E.DEFAULT_CONFIG, ...c, tp: 1, dp: 1, ep: 1, etp: 1, pp: 2 });
    assert.equal(r.valid, true);
    assert.equal(r.cards.reduce((n, card) => n + card.layers.denseCount, 0), 0);
    assert.deepEqual(r.cards.flatMap(card => card.moe.hashLayers), Array.from({ length: hash }, (_, i) => i));
  }
  const v41 = C.getModel('deepseek-v4.1-flash-1017');
  assert.ok(v41.notes.some(note => note.includes('196B') && note.includes('未计入')));
});

test('Kimi LatentMoE uses narrow routed experts while retaining full-width shared experts and adapters', () => {
  const c = C.getModel('kimi-k3').config;
  assert.deepEqual([c.layers, c.hiddenSize, c.routedExpertHiddenSize, c.experts, c.topK, c.sharedExperts, c.moeLayers, c.moeIntermediateSize, c.intermediateSize], [93, 7168, 3584, 896, 16, 2, 92, 3072, 33792]);
  const r = E.analyze({ ...E.DEFAULT_CONFIG, ...c, tp: 1, dp: 1, ep: 1, etp: 1 });
  const card = r.cards[0];
  assert.equal(card.layers.moe[0], 1);
  assert.equal(card.moe.routedHiddenSize, 3584);
  assert.equal(card.moe.sharedHiddenSize, 7168);
  assert.equal(card.memory.moe, 92 * 896 * 3 * 3584 * 3072 * 2);
  assert.equal(card.memory.shared, 92 * 2 * 3 * 7168 * 3072 * 2);
  assert.ok(card.memory.expertAdapters > 0);
});

test('MiniMax releases retain separate Dense FFN dimensions and distinct sparse-attention support', () => {
  const m3 = C.getModel('minimax-m3'), newer = C.getModel('minimax-m3.1-flash');
  for (const model of [m3, newer]) {
    const c = model.config;
    assert.deepEqual([c.layers, c.hiddenSize, c.heads, c.kvHeads, c.headDim, c.experts, c.topK, c.sharedExperts, c.intermediateSize, c.moeIntermediateSize, c.moeLayers, c.vocabSize], [60, 6144, 64, 4, 128, 128, 4, 1, 12288, 3072, 57, 200064]);
  }
  assert.equal(m3.support, 'projection-proxy');
  assert.equal(m3.attentionLayers.filter(t => t === 'GQA').length, 3);
  assert.equal(newer.support, 'partial');
  assert.ok(newer.attentionLayers.every(t => t === 'MSA'));
  assert.equal(newer.releasedAt, '2026-10-19');
  assert.equal(new URL(newer.releaseSource).hostname, 'www.minimax.io');
});
