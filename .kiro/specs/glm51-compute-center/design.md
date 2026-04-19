# Design — GLM-5.1 训练 / 推理算力中心讲解网站

> Kiro Spec Workflow · Phase 2 — Design
>
> 本文件给出技术架构、关键算法、数据契约和讲解网站的信息架构 (IA)。

---

## 1. 系统架构总览

```
┌──────────────────────────────────────────────────────────────┐
│                      讲解网站 (静态 SPA)                      │
│   web/index.html  +  web/styles.css  +  web/app.js           │
│   ├── KaTeX (公式)        ├── Chart.js (图表)                 │
│   └── 章节路由 (hash router)  └── 互动控件 (vanilla JS)        │
└────────────────▲─────────────────────────────────────────────┘
                 │  fetch('/web/data/computed.json')
┌────────────────┴─────────────────────────────────────────────┐
│                  数据层 (确定性 JSON)                          │
│   web/data/computed.json   (由 Python 模型生成)               │
└────────────────▲─────────────────────────────────────────────┘
                 │  python models/run_all.py
┌────────────────┴─────────────────────────────────────────────┐
│                   计算层 (Python 3.10+ 纯标准库)              │
│   models/                                                    │
│   ├── scaling_laws.py        (Chinchilla / Hoffmann)         │
│   ├── cluster_sizing.py      (FLOPs → GPU·hours)             │
│   ├── memory_parallelism.py  (3D + EP + ZeRO)                │
│   ├── network_bandwidth.py   (AllReduce / A2A 时间)           │
│   ├── storage_io.py          (Checkpoint / dataset BW)       │
│   ├── power_cooling.py       (PUE / WUE / TDP)               │
│   ├── tco.py                 (CAPEX / OPEX 三年)              │
│   ├── inference.py           (KV-Cache, vLLM 吞吐)            │
│   ├── constants.py           (硬件参数表 / 单价表)             │
│   └── run_all.py             (orchestrator)                  │
└──────────────────────────────────────────────────────────────┘
```

设计原则：
- **零后端**：所有页面是静态资源，可部署到任何静态站托管 (GitHub Pages / Vercel / Nginx)。
- **数据可重放**：`computed.json` 是单一可信源 (single source of truth)。
- **无构建工具**：不依赖 npm / webpack；Tailwind 走 CDN（讲课现场只要能联网即可）。
- **可备份本地化**：CDN 资源可在 `web/vendor/` 提前缓存，支持彻底离线放映。

## 2. 关键技术选型

| 层 | 选型 | 理由 |
|---|---|---|
| 前端框架 | 原生 HTML5 + ES Modules | 学生易读、无构建链路、上课现场无版本风险 |
| 样式 | 自写设计系统（CSS 变量）+ Tailwind CDN（utility 救急） | 兼顾原创性和速度 |
| 图表 | **Chart.js v4** | 体积小、tree-shake 友好、暗色友好 |
| 公式 | **KaTeX 0.16** | 比 MathJax 快 30×，渲染稳定 |
| 矢量图 | **手写 SVG + Mermaid (CDN)** | 投影 4K 不糊 |
| 计算层 | **Python 3.10+ 标准库** | 0 依赖（不需要 numpy），任何机器都能跑 |
| 数据交换 | **UTF-8 JSON** | 通用、可读 |

> 决策记录：放弃 Next.js / React，因为 (a) 内容是讲稿性质静态页；(b) 现场避免任何 Node 版本/网络问题；(c) 静态站方便分发到学校内网。

## 3. 模型与公式（核心）

> 完整代码见 `models/`，以下为公式摘要。

### 3.1 GLM-5.1 参数推断

GLM 系列从 GLM-4 → GLM-4.5 → GLM-4.6 演进路径已开源公开（Zhipu AI Tech Reports, 2024-2025），GLM-4.5 / 4.6 采用 **MoE 355B-A32B**。我们在没有官方 GLM-5.1 公开规格的前提下，**基于 Zhipu 公开 roadmap 与开源社区共识做工程性外推**，给出**三档假设**让讲师选择：

| 档位 | 总参数 N | 激活参数 N_act | 上下文 | 备注 |
|---|---|---|---|---|
| Conservative | 600 B | 40 B | 256 K | 在 GLM-4.6 基础上 1.7× |
| **Baseline (本课主用)** | **1.0 T** | **64 B** | **512 K** | 1T 量级、稀疏度 ~6.4 % |
| Frontier | 2.0 T | 128 B | 1 M | 与 Kimi K2 / DeepSeek V4 对齐 |

### 3.2 训练算力 (Chinchilla 法则)

经典近似（Hoffmann et al., 2022, "Training Compute-Optimal Large Language Models"）：

$$
C_{\text{train}} \approx 6 \cdot N_{\text{act}} \cdot D
$$

- $N_{\text{act}}$：MoE 激活参数量 (≈ 64 B)
- $D$：训练 Token 数 (取 Chinchilla 最优 $D \approx 20 N$ 的 MoE 修正版 $D \approx 20 N_{\text{total}}$, 即 20 T tokens)
- 因子 6 = 2 (前向) + 4 (反向)

补充开销：
- MoE Router/Gating：+5–10 %
- 重计算 (recompute) 反向：+33 %
- 系统额外（数据加载/通信尾延）：+15 %

实际工程系数取 **8.0** 替代 6.0 较稳妥（参考 Megatron-LM 论文 Sec.5）：

$$
C_{\text{eff}} = 8 \cdot N_{\text{act}} \cdot D \approx 8 \times 6.4\!\times\!10^{10} \times 2.0\!\times\!10^{13} = 1.024 \times 10^{25} \text{ FLOPs}
$$

**约 10²⁵ FLOPs（10 YottaFLOPs / 10 千万亿亿 FLOPs）**——已经处于美国政府 EO 14110 注意力门槛 (10²⁶) 的下沿。

### 3.3 GPU 集群与训练时长

$$
\text{GPU·s} = \frac{C_{\text{eff}}}{\text{TFLOPS}_{\text{peak}} \times \text{MFU}}
$$

- MFU (Model FLOPs Utilization)：H100 32–40 %, B200 35–45 %（最新 NeMo 数据）
- BF16 Tensor Core 峰值 (TFLOPS)：H100 989 / H200 989 / B200 2 250 / GB200 2 500（per GPU，dense, w/o sparsity）

> 注：NVIDIA 营销常引"4.9 PFLOPS"是 FP4 数字。BF16 才是训练真实可用值。

### 3.4 显存占用 (3D 并行 + ZeRO-1)

每 GPU 内存约束：

$$
M_{\text{gpu}} = \underbrace{\frac{N_{\text{act}} \cdot 18}{\text{TP} \cdot \text{PP}}}_{\text{params + grad + optim (BF16+FP32)}} + \underbrace{\frac{B \cdot S \cdot H \cdot L \cdot \alpha}{\text{TP} \cdot \text{CP}}}_{\text{激活}} + M_{\text{kv-cache}}
$$

- 18 字节/参数 = 2(BF16 weight) + 2(grad) + 4+4+4+2(Adam state in FP32 + master weight)
- $\alpha$：激活系数（~ 12，使用选择性重计算后 ~ 4）
- ZeRO-1 把 optimizer state 切到 DP 维度，可再除以 DP

### 3.5 网络通信时间

AllReduce 时间（Ring 算法，每 GPU 参与）：

$$
T_{\text{ar}} = 2 \cdot (P-1) \cdot \frac{M / P}{B}
$$

EP MoE 的 All-to-All：

$$
T_{a2a} = \frac{M_{\text{tokens}}}{B_{\text{bisection}}}
$$

InfiniBand NDR 单端口 400 Gbps = 50 GB/s；H100 配置 8×400 G = 3.2 TB/s 节点出口。

### 3.6 PUE 与电费

机房年耗电：

$$
E_{\text{annual}} = P_{\text{IT}} \times \text{PUE} \times 8760
$$

- 北京气候 + 风冷 PUE ≈ 1.4；液冷 ≈ 1.15；浸没式 ≈ 1.08
- 工业电价取 0.6 ¥/kWh（北京一般工商业电价 2025）

### 3.7 三年 TCO

$$
\text{TCO}_{3y} = \text{CAPEX}_{\text{server}} + \text{CAPEX}_{\text{network}} + \text{CAPEX}_{\text{facility}} + 3 \cdot (\text{OPEX}_{\text{power}} + \text{OPEX}_{\text{salary}} + \text{OPEX}_{\text{maint}})
$$

GPU 单价 (2025 Q4 公开渠道)：H100 SXM ≈ 25 万 ¥；H200 ≈ 32 万 ¥；B200 ≈ 45 万 ¥；GB200 NVL72 整机柜 ≈ 2 800 万 ¥。

### 3.8 推理吞吐 (vLLM, KV cache 制约)

- 单请求 KV：$2 \cdot L \cdot H \cdot S \cdot \text{bytes}$
- 并发上限 = $\frac{M_{\text{available}}}{KV_{\text{per-req}}}$
- Decode 吞吐 ≈ $\frac{N_{\text{act}} \cdot 2}{\text{TFLOPS} \cdot \text{MFU}}$ (per token, per req)

## 4. 数据契约 (web/data/computed.json)

```jsonc
{
  "version": "1.0",
  "generated_at": "2026-04-19T00:00:00Z",
  "model": {
    "name": "GLM-5.1",
    "scenarios": [
      { "id": "conservative", "N": 6.0e11, "N_act": 4.0e10, "ctx": 262144, "tokens": 1.2e13 },
      { "id": "baseline",     "N": 1.0e12, "N_act": 6.4e10, "ctx": 524288, "tokens": 2.0e13 },
      { "id": "frontier",     "N": 2.0e12, "N_act": 1.28e11,"ctx": 1048576,"tokens": 4.0e13 }
    ],
    "flops_total": 1.024e25
  },
  "gpus": { "h100": {...}, "h200": {...}, "b200": {...}, "gb200": {...} },
  "cluster": {
    "baseline_b200": { "n_gpu": 8192, "days": 78, "mfu": 0.40, ... }
  },
  "network": { ... },
  "memory": { ... },
  "power": { "pue": 1.15, "annual_mwh": 86_500, "annual_co2_t": 47_000 },
  "tco": { "capex_total_yuan": 4.7e9, "opex_3y_yuan": 1.6e9 },
  "inference": { ... }
}
```

## 5. 信息架构 (网站章节路线)

| # | 章节 | 时长 | 关键交互 |
|---|---|---|---|
| 0 | Hero (项目封面) | 2 min | 大字号副标题 + 60 min 倒计时 |
| 1 | 课程导论 & AI 算力格局 | 5 min | 全球十大算力中心地图 |
| 2 | GLM-5.1 模型解构 | 5 min | MoE 架构 SVG 动画 |
| 3 | Scaling Law 推导 | 6 min | **互动滑块计算器** |
| 4 | NVIDIA GPU 选型 | 6 min | **4 卡型对比卡片** |
| 5 | 集群架构 | 7 min | NVL72 + IB Fat-tree SVG |
| 6 | 并行策略 | 6 min | 3D 并行示意图 + ZeRO 阶段图 |
| 7 | 存储与数据 | 4 min | Checkpoint 时间柱状图 |
| 8 | 电力 & 液冷 | 5 min | PUE 滑块 + 年耗电柱状 |
| 9 | 推理部署 | 5 min | vLLM 吞吐曲线 |
|10 | TCO 经济模型 | 5 min | **CAPEX/OPEX 堆叠图** + 滑块 |
|11 | 12 个月施工路线图 + Q&A 引导 | 4 min | 甘特图 |

> 总计 **60 min** 整。每章最后预留 30 s 思考题，引导课堂互动。

## 6. 设计系统 (Design Tokens)

```css
:root {
  --bg-0: #0b0f17;
  --bg-1: #111826;
  --bg-2: #1a2332;
  --fg-0: #e6edf6;
  --fg-1: #a6b3c5;
  --fg-2: #6b7a91;
  --accent-nv: #76b900;   /* NVIDIA Green */
  --accent-zhipu: #2563eb;/* Zhipu Blue   */
  --accent-warn: #f59e0b;
  --accent-danger: #ef4444;
  --grid: #1f2a3c;
  --radius: 12px;
  --font-cn: "PingFang SC", "Microsoft YaHei", system-ui, sans-serif;
  --font-mono: "JetBrains Mono", "SF Mono", Consolas, monospace;
}
```

排版：标题 Inter / 系统无衬线；正文行高 1.7；段宽 ≤ 72 ch。

## 7. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| GLM-5.1 官方未发布、参数推断主观 | 学生质疑 | 明确"工程外推"标注，给出 3 档 + 公式可调 |
| 现场断网导致 KaTeX/Chart.js 加载失败 | 演讲翻车 | `web/vendor/` 预下载 fallback |
| Python 计算未运行直接看页面 | 数字为占位 | 仓库内提交一份预算好的 `computed.json` |
| 学生对 MoE / 3D 并行无背景 | 听不懂 | 章节顺序由浅到深，先架构后并行 |

---

> 进入 Phase 3 → `tasks.md`
