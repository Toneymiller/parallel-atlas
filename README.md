# Parallel Atlas · 大模型并行实验台

在浏览器里配置大模型推理并行策略，查看每张 GPU / NPU 上的 Attention、MoE 权重、KV Cache 和卡间通信关系。适合学习并行原理、讨论部署方案，以及比较参数变化对资源布局的影响。

**[在线使用](https://toneymiller.github.io/parallel-atlas/)** · [GitHub 仓库](https://github.com/Toneymiller/parallel-atlas) · MIT License

纯静态 HTML / CSS / JavaScript，无第三方运行依赖，无需账户或推理服务。配置和计算在当前浏览器内完成。

## 快速开始

直接打开 `index.html` 即可使用。也可以在安装 Node.js 18 或更新版本后启动本地服务：

```sh
git clone https://github.com/Toneymiller/parallel-atlas.git
cd parallel-atlas
npm start
```

打开 <http://127.0.0.1:4173>。无需执行 `npm install`。本地服务仅监听 `127.0.0.1`；可通过环境变量 `PORT` 更换端口。浏览器对 `file://` 下本地存储的支持不同，建议使用本地服务或在线版本保存工作状态。

## 能做什么

- 设置 TP、PP、DP、CP、PCP、DCP、EP、ETP 和 SP，并实时校验组合约束。
- 按卡查看 rank 坐标、所在并行组、Attention heads、专家分配、权重占用与 KV Cache 估算。
- 在集群拓扑、逐卡权重表和通信矩阵之间切换，按 Prefill / Decode、模型层和通信组筛选。
- 设置 GQA / MLA、模型层数、hidden size、专家数、Top-K、精度、序列长度与批量等参数；混合注意力模型展示逐层类型与估算范围。
- 在模型结构中选择现有开源模型，自动填入从官方配置核验的结构参数，并查看来源与近似说明。
- 选择华为 Ascend NPU 或 NVIDIA GPU 硬件预设，设置容量预算与每节点设备数，也可自定义硬件。
- 通过预设快速探索部署布局；导入、导出 JSON 配置；下载 PNG / SVG 图片。
- 将当前方案设为基线，对比设备数和最高单卡占用；通过链接分享参数配置。
- 在通用概念布局与 vLLM Ascend PCP / DCP 语义之间切换。

## 选择现有模型

模型结构区域内置 **9 个当前模型 / 版本与 6 个旧版参考模型**，按两组排列。结构参数是 **2026-11-03 核验的官方 `config.json` 快照**，选择后可以继续修改；页面不下载模型权重、不执行模型仓库代码，也不依赖在线获取配置。

| 当前模型 / 官方来源 | 主干层数 | Hidden size | 路由专家 / Top-K | Attention 与支持范围 |
| --- | --- | --- | --- | --- |
| [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2/raw/main/config.json) | 78 | 6144 | 256 / 8 | DSA / MLA，展开投影代理 |
| [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3/raw/main/config.json) | 78 | 6144 | 256 / 8 | DSA / MLA，展开投影代理 |
| [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash/raw/main/config.json) | 45 | 4096 | 288 / 8 | 34 KDA + 11 DSA，部分估算 |
| [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/raw/main/config.json) | 43 | 4096 | 256 / 6 | SWA / CSA / HCA，部分估算 |
| [DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/raw/main/config.json) | 61 | 7168 | 384 / 6 | CSA / HCA，部分估算 |
| [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/raw/main/config.json) | 40 | 5120 | 384 / 6 | CED 跨层共享 KV / 索引，部分估算 |
| [Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3/raw/main/config.json) | 93 | 7168 | 896 / 16 | 69 KDA + 24 MLA，部分估算 |
| [MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3/raw/main/config.json) | 60 | 6144 | 128 / 4 | 3 GQA + 57 MSA，GQA 投影代理 |
| [MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/raw/main/config.json) | 60 | 6144 | 128 / 4 | 60 MSA，部分估算；[2026-10-19 发布](https://www.minimax.io/blog/minimax-m31f) |

**部分估算的含义：**GLM-5.3-Flash、DeepSeek V4 / V4.1、Kimi-K3 和 MiniMax-M3.1-Flash 保留已核验的层、专家、FFN 和词表结构；Attention 权重、KV / 线性注意力状态以及尚未支持的专有模块均排除。界面中的“未计入”或零值不表示这些模块不存在，合计仅为已建模部分的结构估算，不能用于证明完整模型可装入显存。CP / PCP / DCP 只显示候选逻辑参与组，实际算子、状态布局与部署支持仍需对照后端验证。

GLM-5.2 / GLM-5.3 保留前三层 Dense、75 层 MoE，以及每层一个共享专家；有效 QK head dim 为 `192 + 64 = 256`。MLA KV 按 `512 latent + 64 RoPE` 近似，Attention 使用展开投影代理，低秩路径与共享稀疏索引器尚未精确建模。MiniMax-M3 的稀疏 Attention 使用 GQA 投影代理，缓存按完整 GQA 序列估算，不表示 MSA 的真实缓存布局；两个 MiniMax 版本均保留前三层 Dense FFN 与后续 57 层 MoE。

DeepSeek V4 的主干全部为 MoE，前三个 MoE 层采用 Hash Router，常规 Router 矩阵省略，Hash 表本身未计入；V4.1 配置没有声明该规则，因此没有沿用。V4.1 的 Engram 条件记忆（官方约 196B 参数）、视觉与草稿模块也未计入。Kimi-K3 的路由专家使用实际 `3584` 维输入 / 输出，共享专家使用 `7168` 维，额外适配投影单独近似；其官方从 1 开始的 Attention 层号已转换为页面的 0-based 层号。所有条目的详细说明可在模型来源区域查看。

旧版参考模型继续保留：

| 模型 / 官方来源 | 主干层数 | Hidden size | Q / KV heads | 路由专家 / Top-K | MoE 层 |
| --- | --- | --- | --- | --- | --- |
| [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json) | 36 | 4096 | 32 / 8 | 无 | 无 |
| [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B/raw/main/config.json) | 64 | 5120 | 64 / 8 | 无 | 无 |
| [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/config.json) | 48 | 2048 | 32 / 4 | 128 / 8 | L0–L47 |
| [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B/raw/main/config.json) | 94 | 4096 | 64 / 4 | 128 / 8 | L0–L93 |
| [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json) | 61 | 7168 | 128 / 128 | 256 / 8 | L3–L60 |
| [GLM-4.5](https://huggingface.co/zai-org/GLM-4.5/raw/main/config.json) | 92 | 5120 | 96 / 8 | 160 / 8 | L3–L91 |

旧版 DeepSeek-V3 和 GLM-4.5 均保留前三层 Dense 与每个 MoE 层的一个共享专家。Qwen3 MoE 模型没有共享专家。显式 `head_dim` 或具体 Attention 的有效投影维度按官方结构读取，不能用 `hidden_size / num_attention_heads` 替代。

模型条目使用 BF16 权重与 BF16 KV 作为可调分析起点；部分估算条目的 KV 仍排除。多个模型的官方 checkpoint 使用 FP8 / FP4 / MXFP4 / NVFP4 等混合量化，BF16 起点不复现这些发布文件的实际占用。旧版 DeepSeek-V3 的 MLA KV 使用 `512 latent + 64 RoPE`，Attention 权重为展开投影近似：QK head dim 取 `128 + 64 = 192`，实际 V head dim 为 `128`，低秩投影没有精确建模。额外 MTP、草稿、视觉及其他辅助模块以各条目的排除说明为准。

GQA 条目中的 MLA latent / RoPE 字段，以及 Dense 条目中的 MoE FFN dim，是未启用的界面占位值，不参与对应模型估算。DeepSeek V4 / V4.1 官方没有提供的 Dense FFN / latent 字段也仅作未启用占位，不能当作官方参数。模型选择提供结构起点；已有并行参数仍需通过校验，容量警告也需结合实际部署条件解读。

## 选择硬件

“请求与硬件”区域内置以下条目，公开规格核验日期为 **2026-11-03**。已核实的硬件选择填入容量预算和节点设备数；可以直接修改这两个值，也可以选择自定义硬件。950Server 单独标记为规格待核实，选择时保留当前容量与节点输入。预设影响容量比较与 rank 的逻辑节点分组，不模拟硬件算力、互联带宽或实际延迟。

| 硬件 / 官方来源 | 单设备标称容量 | 每节点默认设备数 |
| --- | --- | --- |
| [Huawei Atlas 800T A2](https://www.hiascend.com/doc_center/source/zh/HDK/2610/A2/A2DCMI/dcmi_050.html) | 64 GB | 8 个 NPU 设备 |
| [Huawei Atlas 800T A3 · 软件部署口径](https://docs.vllm.ai/projects/ascend/en/v0.26.0rc1/tutorials/features/dynamic_chunked_pipeline_parallel.html) | 64 GB / 逻辑 NPU | 16 个逻辑 NPU 设备 |
| [Huawei Ascend 950DT · Atlas 650E 配置](https://www.hiascend.com/hardware/ai-server) | 96 GB / NPU | 8 个 NPU（整机 768 GB HBM） |
| Huawei 950Server · 规格待核实 | 沿用当前输入 | 沿用当前输入；没有填入推测规格 |
| [NVIDIA A100 SXM](https://www.nvidia.com/en-gb/data-center/a100/) | 80 GB | 8 个 GPU（DGX 参考布局） |
| [NVIDIA H100 SXM](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html) | 80 GB | 8 个 GPU（HGX 参考布局） |
| [NVIDIA H20 SXM5](https://docs.nvidia.com/nim/large-language-models/1.15.0/supported-models.html) | 96 GB | 8 个 GPU（[Dell PowerEdge XE9680 参考布局](https://www.dell.com/support/manuals/en-my/poweredge-xe9680/xe9680_ism_pub/gpu-specifications?guid=guid-1f91b9fb-b8fd-492e-863f-45febc9355c6&lang=en-us)） |
| [NVIDIA H200 SXM](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html) | 141 GB | 8 个 GPU（HGX 参考布局） |
| [NVIDIA B200 SXM](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html) | 180 GB | 8 个 GPU（HGX 参考布局） |
| [NVIDIA GeForce RTX 4090](https://www.nvidia.com/en-au/geforce/graphics-cards/40-series/rtx-4090/) | 24 GB | 8 个 GPU（工具默认，可修改） |

Atlas 800T A3 的 **16 × 64 GB 是软件可见逻辑设备口径**；[华为产品规格](https://e.huawei.com/cn/products/computing/ascend/atlas-800t-a3)的物理口径为 8 颗昇腾 910、8 × 128 GB HBM。本工具的 16 个 rank 设备不表示 16 颗物理处理器或 16 张物理卡，也未描述处理器内部互联。A2 的 8 设备布局另见[华为 MindCluster 节点说明](https://www.hiascend.com/document/detail/zh/mindcluster/60rc3/clustersched/reference/api/dlug_api_055.html)。RTX 4090 的每节点 8 设备是可编辑布局默认值，不是产品保证的服务器配置。

950DT 条目具体对应当前官方 Atlas 650E 的 **8 × 96 GB HBM** 配置，不能把整个 950DT 系列视作统一容量。[华为此前公开路线图](https://www.huawei.com/en/news/2025/9/hc-xu-keynote-speech)也描述过 144 GB 的 950DT 芯片规格；本预设没有把该规格套入 650E。**950Server 尚未核实到对应的独立官方型号与精确容量**，保留为待核实条目供手动设置；它不表示已确认等同 Atlas 650E，也不采用鲲鹏 / TaiShan 950 的 CPU 系统内存作为 NPU HBM。H20 选用 96 GB 版本，不混用其他可能的容量款型；每节点数量仍可按实际服务器修改。

**单位口径：**硬件预设把厂商标称 GB 数字近似填入本工具的 GiB 输入，未做十进制 / 二进制换算。容量输入按 `1024³` 字节计算，因此预设不是精确的可用显存测量；系统、驱动和框架也会预留内存，请按实际可用容量调整预算。

## 并行策略的含义

| 策略 | 本工具中的含义 | 主要观察点 |
| --- | --- | --- |
| TP · Tensor Parallel | 在一个层内部切分 Attention 和矩阵参数 | Q / KV heads、权重分片和组内归约 |
| PP · Pipeline Parallel | 将模型层分配给流水线阶段 | 每张卡的层区间、阶段间传递 |
| DP · Data Parallel | 建立处理不同请求的 Attention 副本 | 参数复制、请求分工；MoE 可跨 DP 组构成 EP |
| CP · Context Parallel | 通用概念模式的独立上下文轴 | 序列分片、KV 分片与上下文交换 |
| PCP · Prefill Context Parallel | 将 prefill token 计算分散到更多 rank | prefill 计算分工；仅启用 PCP 不会自动分片持久 KV |
| DCP · Decode Context Parallel | 在已有 TP / PCP rank 上沿序列维度分片 KV | decode KV 所有权、Attention 结果合并；不额外增加卡数 |
| EP · Expert Parallel | 将 routed experts 分配给专家组 | 每卡专家 ID、dispatch / combine 关系 |
| ETP · Expert Tensor Parallel | 切分单个专家内部的矩阵 | 同一专家的权重分片与专家组副本 |
| SP · Sequence Parallel | 在 TP 组内切分部分序列激活 | 激活的 ReduceScatter / AllGather；不新增权重分片轴 |

通用概念模式中的设备数为 `TP × PP × DP × CP × PCP`，CP 与 PCP 不同时扩展。`EP × ETP` 必须整除 `TP × DP`，剩余维度形成专家副本。DCP 和 SP 复用已有 rank，不把这些参数再次乘入设备数。具体可用组合以界面校验结果为准。

vLLM Ascend 模式采用 `TP × PP × DP × PCP` 的设备网格，DCP 复用其中的 rank。EP 设为 `1` 表示关闭，或设为 `TP × DP × PCP` 表示启用；ETP 输入固定为 `1`，关闭 EP 时引擎会自动模拟跨 `TP × DP × PCP` 的专家张量切分。该模式还对 GQA / MLA、PCP / DCP 和 SP 组合进行校验；它表达选定文档中的布局规则，并不代表所有模型、后端或版本均支持这些组合。

`PCP=1` 时 DCP 必须整除 TP，GQA / MQA 还要求同组 KV 头兼容。启用 PCP 后，GQA 的 DCP 可为 `1` 或 `PCP`，MLA 的 DCP 可为 `1`、`PCP` 或 `TP × PCP`。页面将 DCP 通信主要归在 Decode 视图；带历史 KV 的 prefill 也可能需要访问 DCP 分片。

Ascend 模式还采用当前文档中 Embedding / LM Head 沿 `TP × PCP` 常驻分片的默认行为；因此 PCP 并不意味着每类权重都完整复制。SP 在该模式中表示 MoE token 路径，需要 `TP > 1`、MoE 与 EP；实际框架还需启用 FlashComm。通用模式的 SP 则表示 Megatron 风格的 TP 组内激活切分。

## 如何解读结果

这是**并行布局概念模拟器**，输入参数驱动的是明确的分片模型，没有连接实际推理引擎、读取模型 checkpoint 或采集通信 trace。

- 权重按参数尺寸和所选数值精度估算，包含 Q / K / V / O、三矩阵 SwiGLU、Router、共享专家、LayerNorm，以及首级 Embedding 和末级未绑定 LM Head 的简化结构。模型特有的 bias、特殊投影、权重绑定及其他结构差异没有完整建模。
- MLA 的持久 KV 使用 latent 维度估算，Attention 权重采用展开投影的近似模型，不能把它当作特定 MLA 模型的精确参数统计。
- 混合 / 压缩 Attention 尚未支持的条目排除全部 Attention 权重和缓存 / 线性状态；界面显示部分估算，通信组也仅表示候选 ranks。层类型列表用于阅读模型结构，不会把这些异构层错误地按普通 GQA 计算。
- KV Cache 的精度独立于权重精度；将权重量化到 INT4 / INT8 不会自动把 KV Cache 改为相同精度。
- 显存 / HBM 估算不含激活、算子工作区、框架预留、分配器碎片、padding、通信 buffer、量化元数据等完整开销。它用于比较布局，不是实际 OOM 预测。
- 通信连线展示逻辑参与关系，不表示 NCCL / HCCL 在硬件上的实际路由、消息调度或物理连接。DP 副本不会因普通推理自动执行训练式梯度 AllReduce。
- 本工具不提供实测通信量、时延或吞吐；真实性能取决于拓扑、实现、调度、负载与硬件。

所有层、head、expert 和 token 索引从 **0** 开始，区间的首尾均包含在内。PP 近似均分模型层；MoE 层可均匀散布，或集中在尾部以保留开头的 Dense 层。全局并发序列近似均分到 DP。容量使用 **GiB（1024³ 字节）**，配置字段 `memoryGB` 也按 GiB 解释。为保证浏览器响应，设备总数上限为 **512**。

参数组合有误时先修正校验提示，再解读拓扑和资源结果。部署前应对照所用框架版本、模型实现与硬件约束进行验证。

## 配置与图片

通过界面导出 JSON 保存实验，再使用导入功能恢复。导入同时接受包含 `schemaVersion: 1` 和 `config` 的导出文件，以及直接的参数对象；超过 100 KB、未知版本或不通过布局校验的文件会被拒绝。配置描述的是本工具的模型参数与布局语义，不是可以直接交给 vLLM、MindIE 或其他推理引擎的启动配置。

有效配置会保存在浏览器本地存储中，包含硬件选择和手动调整的容量 / 节点预算。分享按钮把配置编码进 URL 的 `#config=` 片段；接收方打开在线或 HTTP 链接即可恢复参数。分享内容不包含当前选择的卡片、视图或对比基线；对比基线在当前页面会话内保留。

PNG 适合插入文档，SVG 适合缩放与后续编辑。导出跟随当前拓扑、权重表或矩阵视图及通信筛选；大图会自动限制 PNG 像素尺寸。拓扑中的大型通信组以环形摘要表示；通信矩阵每页展示 32 张卡，可以上下翻页，也会跟随当前选中卡。单卡专家网格展示一个 64 专家窗口，下面可以展开完整的本地专家编号。

## 开发与验证

```sh
npm run check
npm test
```

`engine.js` 包含计算、校验、分组和分配逻辑，`app.js` 负责交互与可视化，`models.js` 和 `hardware.js` 保存核验后的模型 / 硬件快照。测试覆盖核心分片约束、资源计算、15 个模型的参数来源、Dense / MoE 层位置、混合 Attention 层型、部分估算标记、LatentMoE / Hash Router 与硬件容量口径。`scripts/serve.cjs` 是零依赖本地静态服务，仅提供公开页面资源。

| 文件 | 用途 |
| --- | --- |
| `index.html` | 页面结构与参数表单 |
| `styles.css` | 响应式界面与减少动态效果支持 |
| `app.js` | 浏览器交互、图形渲染与导入导出 |
| `engine.js` | 并行拓扑与分片估算 |
| `models.js` | 带官方来源的模型结构快照 |
| `hardware.js` | 带官方来源与逻辑 / 物理口径说明的硬件快照 |
| `engine.test.cjs` | Node.js 内置测试 |
| `models.test.cjs` | 模型快照与层结构测试 |
| `hardware.test.cjs` | 硬件快照与设备容量口径测试 |
| `.github/workflows/pages.yml` | 静态验证与 GitHub Pages 部署 |

## GitHub Pages 部署

仓库提供 GitHub Actions 工作流，推送到 `main` 或手动运行 **Deploy GitHub Pages** 后，工作流检查代码并发布静态页面。仓库的 **Settings → Pages → Build and deployment → Source** 需要选择 **GitHub Actions**。

也可使用 **Deploy from a branch**，选择 `main` 的 `/ (root)`。两种部署方式选择一种即可；静态网页无需后端、构建产物或访问令牌。

## 语义参考

本工具根据以下一手文档选择概念和约束；后端实现仍会随版本演进：

- [vLLM：Parallelism and Scaling](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/)
- [vLLM Ascend：Context Parallel 设计文档](https://docs.vllm.ai/projects/ascend/en/main/developer_guide/Design_Documents/context_parallel.html)
- [vLLM Ascend：Context Parallel 用户指南](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/context_parallel.html)
- [vLLM Ascend：Sequence Parallelism 用户指南](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/sequence_parallelism.html)

## 许可证

[MIT](LICENSE) © 2026 Toneymiller
