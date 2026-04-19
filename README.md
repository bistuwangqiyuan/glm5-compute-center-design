# Project Stardust · GLM-5.1 算力中心讲解网站

> **教育部一流课程示范项目** · 北京信息科技大学（BISTU）大三专业课
> 60 分钟把"如何用 NVIDIA GPU 建造能训练 GLM-5.1 的算力中心"讲透。

[![Spec](https://img.shields.io/badge/spec-Kiro%20Workflow-76b900)](.kiro/specs/glm51-compute-center/)
[![Models](https://img.shields.io/badge/models-Python%203.10%2B%20stdlib-2c7eff)](models/)
[![Web](https://img.shields.io/badge/web-Static%20HTML-ef4444)](web/)

---

## 5 分钟 Quickstart

```powershell
# 1. 进入项目
cd "glm5训练用算力中心设计详细讲解网站"

# 2. (Windows) 临时设置 UTF-8 输出，避免控制台 GBK 报错
$env:PYTHONIOENCODING = "utf-8"

# 3. 跑全部 8 个 Python 模型自检（< 5 秒）
python -m models.run_all --check

# 4. 重新生成网站数据
python -m models.run_all

# 5. 起一个本地 HTTP 服务（KaTeX 与 Chart.js 必须经 HTTP，不能 file://）
python -m http.server 8000

# 6. 浏览器打开
start http://localhost:8000/web/
```

> Linux/macOS 同样适用，去掉 `$env:` 那一行即可。

## 项目结构

```text
.
├── .kiro/
│   └── specs/glm51-compute-center/
│       ├── requirements.md     # Phase 1 EARS 用户故事 + 验收标准
│       ├── design.md           # Phase 2 架构 / 公式 / 数据契约
│       └── tasks.md            # Phase 3 可执行任务清单
│
├── models/                     # 计算层（0 第三方依赖，纯 stdlib）
│   ├── constants.py            # 硬件 / 物价 / 物理常量表
│   ├── scaling_laws.py         # Chinchilla / FLOPs 推导
│   ├── cluster_sizing.py       # GPU·hours / 训练天数
│   ├── memory_parallelism.py   # 3D + EP + ZeRO 显存账本
│   ├── network_bandwidth.py    # AllReduce / All-to-All 时间
│   ├── storage_io.py           # 三层存储 + Checkpoint I/O
│   ├── power_cooling.py        # PUE / WUE / 年耗电
│   ├── tco.py                  # CAPEX / OPEX / 三年 TCO
│   ├── inference.py            # vLLM / KV-Cache / 推理吞吐
│   └── run_all.py              # 主调度，输出 web/data/computed.json
│
├── web/                        # 讲解网站
│   ├── index.html              # 11 章节全部内联
│   ├── styles.css              # 设计系统 (NVIDIA Green / Zhipu Blue)
│   ├── app.js                  # 数据加载 / KaTeX / Chart.js / 滑块
│   └── data/
│       ├── computed.json       # ← Python 模型输出
│       └── references.json     # 35 条 IEEE 风格参考文献
│
├── docs/
│   └── SPEAKER_NOTES.md        # 60 分钟逐章节讲稿建议节奏
│
└── README.md                   # 本文件
```

## 演讲建议节奏（60 min）

| 段 | 时长 | 章节 | 教学要点 |
|----|------|------|---------|
| 开场 | 2′ | Hero | 介绍课程目标 + 60 min 路线图 |
| §1 | 5′ | 课程导论 | 三个时代对照表 + 全球 Top 10 集群地图 |
| §2 | 5′ | GLM-5.1 模型 | MoE 路由动画 + 三档参数表 |
| §3 | 6′ | Scaling Law | **现场让同学拨滑块**回答练习题 |
| §4 | 6′ | GPU 选型 | 给学生看 4 张卡的横向条图 |
| §5 | 7′ | 集群架构 | NVLink 域 / IB Fat-tree 两张 SVG |
| §6 | 6′ | 并行策略 | DP/TP/PP/EP/SP/CP 六维表 + 显存饼图 |
| §7 | 4′ | 存储 | 三层存储 + Checkpoint 0.08% 占比 |
| §8 | 5′ | 电力 | PUE 滑量分析 + 选址三选 |
| §9 | 5′ | 推理 | Prefill/Decode 解耦 + 框架对比 |
| §10 | 5′ | TCO | **现场拖滑块**调到学校实际预算 |
| §11 | 4′ | 路线图 | 12 个月里程碑 + 6 句话总结 + Q&A |

详见 [`docs/SPEAKER_NOTES.md`](docs/SPEAKER_NOTES.md)。

## 重新计算与定制

### 调整模型档位
编辑 [`models/constants.py`](models/constants.py) 中 `SCENARIOS` 字典，可改 `n_total / n_active / train_tokens / ctx_len`。

### 调整 GPU / 网络 / 单价
同文件中 `GPUS / NICS / SERVERS` 字典，所有价格 / 功率 / 算力都集中在常量表，方便课堂"假设我们打折 30 % 拿货"等沙盘演练。

### 重生成
```powershell
python -m models.run_all
```
会更新 `web/data/computed.json`，刷新页面即生效。

## 离线放映（断网讲课）

如果讲课现场无法访问 CDN（KaTeX 与 Chart.js 默认走 jsdelivr），可手动下载到 `web/vendor/`：

```powershell
mkdir web/vendor
# KaTeX
curl -L -o web/vendor/katex.min.css  https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css
curl -L -o web/vendor/katex.min.js   https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js
curl -L -o web/vendor/auto-render.min.js https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js
# Chart.js
curl -L -o web/vendor/chart.umd.min.js https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js
```

然后把 `web/index.html` 顶部 4 行 `<script src="https://cdn.jsdelivr.net/...">` 改为 `vendor/...` 本地路径即可。

## 数据可信度声明

- 所有 GPU / 网络 / 存储 / 价格数据来自 NVIDIA 官方文档、MLPerf 公开结果、Semianalysis 行业报告。
- GLM-5.1 参数为基于 Zhipu 公开 GLM-4.5/4.6 架构的**工程外推**，不代表官方规格。
- 训练 FLOPs 公式基于 Hoffmann 等 2022 (Chinchilla) 经典推导。
- TCO 单价为 2025 Q4 国内大型集群采购中位数 (industry consensus)，实际项目以最新报价为准。

完整参考文献见 `web/data/references.json` 与网站第 11 章。

## 验收清单 (Acceptance)

- [x] AC-1 `python -m models.run_all` < 30s 完成全部计算并输出 JSON
- [x] AC-2 网站 11 章节齐全，每章有图表 / 公式 / 工程结论
- [x] AC-3 三个交互组件：Scaling Law 计算器、GPU 选型对比、TCO 滑块
- [x] AC-4 暗色主题 + NVIDIA/Zhipu 双色辅助
- [x] AC-5 所有 Python 模块有 `--selftest` 入口
- [x] AC-6 README + SPEAKER_NOTES 完整
- [x] AC-7 35 条 IEEE 参考文献，覆盖 NVIDIA 官档 / arXiv / 行业报告 / 国家政策

## 致谢

本项目内容由 Cursor Agent (Opus 4.7) 基于 Kiro Spec Workflow 编制；
模型与硬件参数综合来自 NVIDIA、Zhipu AI、DeepSeek、MLCommons、Semianalysis 等公开技术资料；
课程结构参考北京信息科技大学计算机学院 2026 春季"高性能计算与大模型系统"课程大纲。
