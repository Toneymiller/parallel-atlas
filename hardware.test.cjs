'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const E = require('./engine.js');
const H = require('./hardware.js');

test('hardware presets load offline in browsers and CommonJS', () => {
  const sandbox = { window: {} };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, 'hardware.js'), 'utf8'), sandbox);
  assert.equal(sandbox.window.HardwareCatalog.HARDWARE.length, H.HARDWARE.length);
  assert.equal(sandbox.window.HardwareCatalog.getHardware('atlas-800t-a3').gpusPerNode, 16);
  assert.equal(H.getHardware('unknown'), undefined);
});

test('hardware catalog has immutable dated manufacturer provenance and editable engine inputs', () => {
  assert.equal(H.SNAPSHOT_DATE, '2026-11-03');
  const ids = new Set();
  const officialHosts = new Set(['www.nvidia.com', 'docs.nvidia.com', 'e.huawei.com', 'www.hiascend.com', 'docs.vllm.ai', 'www.dell.com']);
  for (const hardware of H.HARDWARE) {
    assert.ok(!ids.has(hardware.id)); ids.add(hardware.id);
    assert.ok(['Huawei', 'NVIDIA'].includes(hardware.vendor));
    if (hardware.status === 'unverified') {
      assert.equal(hardware.memoryGB, null);
      assert.equal(hardware.gpusPerNode, null);
      assert.equal(hardware.budgetOnly, true);
      assert.match(hardware.name, /待核实/);
    } else {
      assert.ok(Number.isInteger(hardware.memoryGB) && hardware.memoryGB > 0);
      assert.ok(Number.isInteger(hardware.gpusPerNode) && hardware.gpusPerNode > 0);
    }
    assert.ok(Object.isFrozen(hardware) && Object.isFrozen(hardware.notes) && Object.isFrozen(hardware.sources));
    assert.ok(hardware.sources.some(source => source.url === hardware.source));
    for (const source of hardware.sources) {
      assert.ok(officialHosts.has(new URL(source.url).hostname));
      assert.equal(new URL(source.url).protocol, 'https:');
      assert.ok(Object.isFrozen(source));
    }
    if (hardware.status !== 'unverified') {
      assert.ok(hardware.notes.some(note => note.includes('GiB') && note.includes('实际可用')));
      const config = E.normalizeConfig({ ...E.DEFAULT_CONFIG, memoryGB: hardware.memoryGB, gpusPerNode: hardware.gpusPerNode });
      assert.equal(config.memoryGB, hardware.memoryGB);
      assert.equal(config.gpusPerNode, hardware.gpusPerNode);
    }
  }
  assert.equal(ids.size, 10);
  assert.ok(Object.isFrozen(H.HARDWARE));
});

test('Atlas A3 ranks use logical 64 GB devices rather than physical 128 GB processors', () => {
  const a3 = H.getHardware('atlas-800t-a3');
  assert.deepEqual([a3.memoryGB, a3.gpusPerNode], [64, 16]);
  assert.match(a3.unitLabel, /逻辑/);
  assert.ok(a3.notes.some(note => note.includes('8 颗昇腾 910') && note.includes('8 × 128 GB')));
  assert.ok(a3.notes.some(note => note.includes('16 颗物理处理器')));
  assert.equal(a3.memoryGB * a3.gpusPerNode, 8 * 128);
  const result = E.analyze({ ...E.DEFAULT_CONFIG, tp: 8, dp: 2, ep: 8, memoryGB: a3.memoryGB, gpusPerNode: a3.gpusPerNode });
  assert.equal(result.valid, true);
  assert.equal(result.cards.length, 16);
  assert.equal(new Set(result.cards.map(card => card.node)).size, 1);
});

test('NVIDIA SXM presets retain specific product variants and sourced HGX node sizes', () => {
  const expected = [['a100-80gb-sxm', 80], ['h100-80gb-sxm', 80], ['h200-141gb-sxm', 141], ['b200-180gb-sxm', 180]];
  for (const [id, memory] of expected) {
    const hardware = H.getHardware(id);
    assert.equal(hardware.memoryGB, memory);
    assert.equal(hardware.gpusPerNode, 8);
    assert.match(hardware.name, /SXM/);
  }
});

test('consumer GPU node count is explicitly a simulator default', () => {
  const rtx = H.getHardware('rtx-4090-24gb');
  assert.equal(rtx.memoryGB, 24);
  assert.ok(rtx.notes.some(note => note.includes('布局默认值') && note.includes('非 RTX 4090 产品规格')));
});

test('950DT uses the verified Atlas 650E variant while 950Server has no invented logical specifications', () => {
  const dt = H.getHardware('ascend-950dt-96gb');
  assert.deepEqual([dt.memoryGB, dt.gpusPerNode], [96, 8]);
  assert.match(dt.name, /Atlas 650E/);
  assert.ok(dt.notes.some(note => note.includes('144 GB') && note.includes('不同产品规格')));
  const server = H.getHardware('ascend-950server');
  assert.equal(server.status, 'unverified');
  assert.equal(server.memoryGB, null);
  assert.equal(server.gpusPerNode, null);
  assert.ok(server.notes.some(note => note.includes('保留当前自定义')));
});

test('H20 uses nominal 96 GB and an eight-GPU OEM reference node', () => {
  const h20 = H.getHardware('h20-96gb-sxm');
  assert.deepEqual([h20.memoryGB, h20.gpusPerNode], [96, 8]);
  assert.equal(h20.capacityUnit, undefined);
  assert.match(h20.source, /nim\/large-language-models\/1\.15\.0\/supported-models\.html$/);
  assert.ok(h20.sources.some(source => new URL(source.url).hostname === 'www.dell.com'));
  assert.ok(h20.notes.some(note => note.includes('标称 GB') && note.includes('GiB') && note.includes('未作十进制/二进制换算')));
});
