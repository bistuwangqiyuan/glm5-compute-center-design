# Tasks — GLM-5.1 算力中心讲解网站 实施计划

> Kiro Spec Workflow · Phase 3 — Tasks
>
> 用 Markdown checklist 表达；Cursor agent 按顺序执行。

---

## T0. 项目骨架

- [x] T0.1 仓库目录初始化（`.kiro/`, `models/`, `web/`, `web/data/`, `web/vendor/`, `docs/`）
- [x] T0.2 README.md 与 LICENSE 占位

## T1. 计算层 (Python · 0 依赖)

- [x] T1.1 `models/constants.py` — 硬件参数表（GPU / 网络 / 电价 / 物价）
- [x] T1.2 `models/scaling_laws.py` — Chinchilla, FLOPs, token budget
- [x] T1.3 `models/cluster_sizing.py` — GPU·hours, 训练天数
- [x] T1.4 `models/memory_parallelism.py` — 显存占用与 3D + EP + ZeRO 推荐配置
- [x] T1.5 `models/network_bandwidth.py` — AllReduce / All-to-All 时间
- [x] T1.6 `models/storage_io.py` — 数据集吞吐 + Checkpoint 时间
- [x] T1.7 `models/power_cooling.py` — PUE / WUE / 年用电 / CO₂
- [x] T1.8 `models/tco.py` — CAPEX / OPEX / 三年 TCO
- [x] T1.9 `models/inference.py` — vLLM 吞吐 / KV-Cache 上限
- [x] T1.10 `models/run_all.py` — orchestrator → `web/data/computed.json`
- [x] T1.11 每个模块 `--selftest` 入口；`python models/run_all.py --check` 通过

## T2. 数据契约

- [x] T2.1 `web/data/computed.json` 由 T1.10 输出，schema 与 design.md §4 对齐
- [x] T2.2 `web/data/references.json` —— ≥ 30 条 IEEE 风格参考文献

## T3. 网站骨架

- [x] T3.1 `web/index.html` — 语义化骨架、章节锚点、跳转导航
- [x] T3.2 `web/styles.css` — 设计 token + 暗色主题 + 响应式
- [x] T3.3 `web/app.js` — 数据加载、章节路由、滑块、Chart.js / KaTeX 渲染器封装

## T4. 章节内容（11 章 + Hero）

- [x] T4.0 Hero / 课程封面
- [x] T4.1 §1 课程导论 & AI 算力格局
- [x] T4.2 §2 GLM-5.1 模型解构
- [x] T4.3 §3 Scaling Law 推导（**含交互式计算器**）
- [x] T4.4 §4 NVIDIA GPU 选型对比（**4 卡型联动表**）
- [x] T4.5 §5 集群架构与拓扑（NVL72 + IB Fat-tree）
- [x] T4.6 §6 并行策略（3D + EP + ZeRO）
- [x] T4.7 §7 存储与数据流水线
- [x] T4.8 §8 电力 / 制冷 / 液冷机房
- [x] T4.9 §9 推理部署（vLLM / TensorRT-LLM / Dynamo）
- [x] T4.10 §10 TCO 经济模型（**CAPEX/OPEX 滑块**）
- [x] T4.11 §11 12 个月施工路线图 + 风险 + 参考文献

## T5. 体验打磨

- [x] T5.1 KaTeX 离线兜底（`web/vendor/katex/`）
- [x] T5.2 Chart.js 离线兜底（`web/vendor/chartjs/`）
- [x] T5.3 投影模式（按 `P` 全屏 / 按 `→` 进入下一章）
- [x] T5.4 打印样式 (CSS @media print) — 可一键导出 PDF 讲义

## T6. 文档与交付

- [x] T6.1 `README.md` — Quickstart、目录、演讲建议
- [x] T6.2 `docs/SPEAKER_NOTES.md` — 60 min 逐分钟讲解稿
- [x] T6.3 `docs/REFERENCES.md` — IEEE 风格参考文献全文
- [x] T6.4 本地预览：`python -m http.server 8000` → `http://localhost:8000/web/`

## T7. 自检

- [x] T7.1 11 章节全部存在、可滚动、导航高亮
- [x] T7.2 任一交互组件改变都能联动至少 1 个图表
- [x] T7.3 `computed.json` 与页面文字一致
- [x] T7.4 Lighthouse Performance ≥ 90（人工核验）

---

> 任务执行完毕即视为符合 requirements.md §4 验收标准。
