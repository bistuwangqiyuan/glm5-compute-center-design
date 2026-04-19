# Requirements — GLM-5.1 训练 / 推理算力中心讲解网站

> Kiro Spec Workflow · Phase 1 — Requirements
>
> 项目代号：**Project Stardust** （北京信息科技大学 · 教育部示范课程）
>
> 版本：v1.0 · 日期：2026-04-19 · 作者：Cursor Agent (Opus 4.7)

---

## 1. 项目目标 (Mission)

构建一套**世界级、可复用、可演讲、可计算**的教育型网站，用于在 **60 分钟** 内向北京信息科技大学计算机 / 人工智能 / 软件工程专业大三学生系统讲解：

> **"如何用 NVIDIA GPU 设计、建造并运营一个能够完成 GLM-5.1 量级（万亿参数 MoE）大模型预训练 + 推理服务的国家级算力中心"**

要求做到：

- **学术严谨**：所有数字均由 Python 模型计算得出，公式可溯源。
- **工业可信**：架构选型与最新一线工业实践对齐（2025-2026 时点）。
- **教学友好**：知识从科普 → 工程 → 经济三层递进，可视化主导。
- **示范课程**：满足教育部"示范性虚拟仿真 / 一流课程"的展示要求。

## 2. 干系人 (Stakeholders)

| 角色 | 关切点 |
|---|---|
| 大三本科生（主要受众） | 听得懂、看得懂、记得住，能把"算力中心"的全栈拼图拼起来 |
| 授课教师 | 60 min 节奏可控、PPT 替代品、可现场互动 |
| 教育部 / 学校评估专家 | 内容深度、思想性、原创性、产学研结合 |
| 行业专家（旁听） | 技术不能错、不出现常识性错误（这点最关键） |

## 3. 顶层用户故事 (User Stories)

> 采用 **EARS** 语法（Easy Approach to Requirements Syntax）：`When/While/Where/If <trigger>, the <system> shall <response>`

### 3.1 内容侧

- **US-1** While 用户首次进入站点, the 系统 shall 在 ≤ 1.5 s 内渲染首屏 Hero 区域，并展示项目封面、副标题和 60 分钟讲解结构索引。
- **US-2** When 用户点击章节锚点, the 系统 shall 平滑滚动至对应 section，并在导航条高亮当前章节。
- **US-3** Where 数值类内容（参数量、FLOPs、TCO 等）出现, the 系统 shall 标注 **数据来源** 与 **计算公式**，且数据均来自仓库内 `models/` 中 Python 脚本输出。
- **US-4** When 用户在 Scaling Law 章节调整滑块（参数量 N、Token 数 D、GPU 类型）, the 系统 shall 实时联动重算训练所需 GPU·小时与电费。
- **US-5** When 用户切换 GPU 型号 (H100 / H200 / B200 / GB200), the 系统 shall 同步更新集群规模、训练时长、CAPEX、PUE 后机房功耗。
- **US-6** Where 出现网络拓扑、机柜布局、并行策略示意, the 系统 shall 提供 SVG 矢量图（不使用模糊位图），保证投影清晰。
- **US-7** Where 出现数学公式, the 系统 shall 通过 KaTeX 渲染，保证可在大屏 4K 投影下清晰阅读。
- **US-8** Where 出现核心定量结论, the 系统 shall 同时给出 **柱状/折线/面积图**（Chart.js）+ **数据表** 双视图。

### 3.2 内容覆盖范围 (Scope)

系统讲解必须覆盖以下 **11 个章节**，每章节都需包含：背景 → 公式 → 计算 → 工程结论。

1. 课程导论与全球 AI 算力格局
2. GLM-5.1 模型推断（架构 / 参数 / 训练目标 / 数据配比）
3. Scaling Law 与训练算力推导
4. NVIDIA GPU 选型与对比（Hopper / Blackwell 全家族）
5. 集群架构：节点、NVLink Domain、InfiniBand / Spectrum-X 拓扑
6. 并行策略：DP / TP / PP / EP / SP / CP + ZeRO + 重计算
7. 存储与数据流水线：并行文件系统、Checkpoint、数据预处理
8. 电力、制冷、液冷与机房物理设计（PUE / WUE）
9. 推理部署：vLLM / TensorRT-LLM / Triton / NVIDIA Dynamo
10. TCO 经济模型：CAPEX / OPEX / 三年总拥有成本
11. 落地路线图、风险与参考文献

### 3.3 技术与质量

- **NFR-1 性能**：首屏 Lighthouse Performance ≥ 90；总下载 ≤ 5 MB（不含 KaTeX/Chart.js CDN）。
- **NFR-2 离线可用**：网站为纯静态站点，可通过 `python -m http.server` 在断网环境（仅依赖 CDN 缓存或本地化 KaTeX）演讲。
- **NFR-3 兼容性**：Chrome 120+ / Edge 120+ / Safari 17+ / Firefox 121+ 均可正常运行；推荐 1920×1080 与 3840×2160 投影。
- **NFR-4 可复算**：任何受众下载仓库后，运行 `python models/run_all.py` 即可重新生成 `web/data/*.json`，刷新页面后所有数字一致。
- **NFR-5 无机密依赖**：不调用任何付费 API；不要求 GPU；普通笔记本可全栈构建。
- **NFR-6 中文友好**：全站中文（含图例），字体使用系统 Sans-Serif 兜底，必要英文术语保留原文 + 中文注释。
- **NFR-7 学术诚信**：参考文献按 IEEE 风格列出 ≥ 30 条，覆盖 NVIDIA 官方文档、arXiv 论文、Zhipu AI 技术报告、MLPerf、Semianalysis 行业报告等。

### 3.4 非目标 (Out of Scope)

- 不实现真实部署脚本（Slurm / Kubernetes manifest 为示意）。
- 不替代 NVIDIA / Zhipu 官方文档（仅做教学性综合）。
- 不提供与 GLM-5.1 的实时对话（作为讲解站，不是 Chat UI）。

## 4. 验收标准 (Acceptance Criteria)

- [ ] AC-1 仓库执行 `python models/run_all.py` 可在 < 30 s 内完成全部 8 个模型的计算并生成 `web/data/computed.json`。
- [ ] AC-2 网站 11 个章节全部存在，每章 ≥ 1 张数据图表 + ≥ 1 段数学公式 + ≥ 1 段工程结论。
- [ ] AC-3 网站含 ≥ 3 个交互组件（Scaling Law 计算器、GPU 选型对比器、TCO 滑块）。
- [ ] AC-4 设计风格统一（暗色主题 + NVIDIA Green / Zhipu Blue 双色辅助），版面专业。
- [ ] AC-5 所有 Python 模型有 docstring 与单元自检 (`python -m models.<name> --selftest`)。
- [ ] AC-6 README.md 给出 5 分钟 Quickstart 与 60 min 演讲建议节奏。
- [ ] AC-7 参考文献 ≥ 30 条，每条含一级机构来源链接。

---

> 进入 Phase 2 → `design.md`
