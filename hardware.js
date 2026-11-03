/* Official capacity/topology snapshots, verified 2026-11-03. No live requests. */
(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.HardwareCatalog = api;
})(typeof window !== 'undefined' ? window : globalThis, function () {
  'use strict';

  const SNAPSHOT_DATE = '2026-11-03';
  const nominalCapacityNote = '容量采用厂商标称 GB 数字，近似填入本工具的 GiB 容量输入，未作十进制/二进制换算；不代表实际可用显存，请按实测余量修改。';
  const hgxSource = { title: 'NVIDIA HGX H100 / H200 / B200：每 GPU 容量与 8-GPU 节点', url: 'https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html' };
  const HARDWARE = [
    {
      id: 'atlas-800t-a2', name: 'Huawei Atlas 800T A2 · 64 GB / NPU', vendor: 'Huawei',
      memoryGB: 64, gpusPerNode: 8, unitLabel: 'NPU 逻辑设备',
      source: 'https://www.hiascend.com/doc_center/source/zh/HDK/2610/A2/A2DCMI/dcmi_050.html',
      sources: [
        { title: '华为 DCMI：Atlas 800T A2 单芯片 HBM 容量 64G', url: 'https://www.hiascend.com/doc_center/source/zh/HDK/2610/A2/A2DCMI/dcmi_050.html' },
        { title: '华为 MindCluster：Atlas 800T A2 单机 1–8 个 NPU', url: 'https://www.hiascend.com/document/detail/zh/mindcluster/60rc3/clustersched/reference/api/dlug_api_055.html' },
        { title: 'vLLM-Ascend：Atlas A2 / Ascend-cann-kernels-910b 产品系列', url: 'https://docs.vllm.ai/projects/ascend/en/v0.13.0/faqs.html' }
      ],
      notes: [
        '按 Atlas 800T A2 的 8 个 NPU 设备、每设备标称 64 GB 建模；属于 Atlas A2 / Ascend 910B 软件产品系列。',
        '华为 DCMI 文档说明无业务时系统亦占用部分 HBM；此处容量是标称上限，未扣系统和框架预留。',
        nominalCapacityNote
      ]
    },
    {
      id: 'atlas-800t-a3', name: 'Huawei Atlas 800T A3 · 64 GB / 逻辑 NPU', vendor: 'Huawei',
      memoryGB: 64, gpusPerNode: 16, unitLabel: 'NPU 逻辑设备',
      source: 'https://e.huawei.com/cn/products/computing/ascend/atlas-800t-a3',
      sources: [
        { title: '华为 Atlas 800T A3：单机 8 颗昇腾 910、8 × 128 GB', url: 'https://e.huawei.com/cn/products/computing/ascend/atlas-800t-a3' },
        { title: 'vLLM-Ascend v0.26.0rc1：Atlas 800T A3 的 64 GB × 16 设备部署', url: 'https://docs.vllm.ai/projects/ascend/en/v0.26.0rc1/tutorials/features/dynamic_chunked_pipeline_parallel.html' }
      ],
      notes: [
        '本预设按 vLLM-Ascend 文档的单机 16 × 64 GB 软件可见逻辑设备建模；每节点设备数 16 指逻辑 NPU。',
        '华为产品规格的物理口径是单机 8 颗昇腾 910、8 × 128 GB HBM；请勿把这里的 16 个逻辑设备当成 16 颗物理处理器或 16 张物理卡。',
        '图中按逻辑设备分配 rank 和容量；未模拟同一物理处理器内的具体互联结构。',
        nominalCapacityNote
      ]
    },
    {
      id: 'a100-80gb-sxm', name: 'NVIDIA A100 SXM · 80 GB', vendor: 'NVIDIA',
      memoryGB: 80, gpusPerNode: 8, unitLabel: 'GPU',
      source: 'https://www.nvidia.com/en-gb/data-center/a100/',
      sources: [{ title: 'NVIDIA A100：80 GB SXM 与 DGX A100 的 8-GPU 配置', url: 'https://www.nvidia.com/en-gb/data-center/a100/' }],
      notes: ['选用 80 GB HBM2e 的 SXM 版本，节点按 DGX A100 的 8 GPU 配置预设，可按实际系统修改。', nominalCapacityNote]
    },
    {
      id: 'h100-80gb-sxm', name: 'NVIDIA H100 SXM · 80 GB', vendor: 'NVIDIA',
      memoryGB: 80, gpusPerNode: 8, unitLabel: 'GPU', source: hgxSource.url, sources: [hgxSource],
      notes: ['选用 H100 SXM 的 80 GB HBM3 版本；节点采用官方 HGX 8-GPU 参考配置，可按实际系统修改。', nominalCapacityNote]
    },
    {
      id: 'h200-141gb-sxm', name: 'NVIDIA H200 SXM · 141 GB', vendor: 'NVIDIA',
      memoryGB: 141, gpusPerNode: 8, unitLabel: 'GPU', source: hgxSource.url, sources: [hgxSource],
      notes: ['选用 H200 SXM 的 141 GB HBM3e 版本；节点采用官方 HGX 8-GPU 参考配置，可按实际系统修改。', nominalCapacityNote]
    },
    {
      id: 'b200-180gb-sxm', name: 'NVIDIA B200 SXM · 180 GB', vendor: 'NVIDIA',
      memoryGB: 180, gpusPerNode: 8, unitLabel: 'GPU', source: hgxSource.url, sources: [hgxSource],
      notes: ['选用 B200 SXM 的 180 GB HBM3e 版本；节点采用官方 HGX 8-GPU 参考配置，可按实际系统修改。', nominalCapacityNote]
    },
    {
      id: 'rtx-4090-24gb', name: 'NVIDIA GeForce RTX 4090 · 24 GB', vendor: 'NVIDIA',
      memoryGB: 24, gpusPerNode: 8, unitLabel: 'GPU',
      source: 'https://www.nvidia.com/en-au/geforce/graphics-cards/40-series/rtx-4090/',
      sources: [{ title: 'NVIDIA GeForce RTX 4090：24 GB GDDR6X', url: 'https://www.nvidia.com/en-au/geforce/graphics-cards/40-series/rtx-4090/' }],
      notes: ['选用标准 24 GB GDDR6X 版本。每节点 8 个 GPU 是本工具的可编辑布局默认值，非 RTX 4090 产品规格；请按实际主机修改。', nominalCapacityNote]
    }
  ];

  for (const hardware of HARDWARE) {
    Object.freeze(hardware.notes);
    for (const source of hardware.sources) Object.freeze(source);
    Object.freeze(hardware.sources);
    Object.freeze(hardware);
  }
  Object.freeze(HARDWARE);
  const getHardware = id => HARDWARE.find(hardware => hardware.id === id);
  return Object.freeze({ HARDWARE, SNAPSHOT_DATE, getHardware });
});
