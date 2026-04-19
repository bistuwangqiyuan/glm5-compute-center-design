/* ===========================================================================
 * Project Stardust — GLM-5.1 算力中心讲解网站
 * 前端运行时：数据加载 → KaTeX/Chart.js 渲染 → 交互组件
 * =========================================================================== */

const $  = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));

const fmt = {
  int: n => n.toLocaleString("zh-CN", { maximumFractionDigits: 0 }),
  fixed: (n, d = 1) => n.toLocaleString("zh-CN", { maximumFractionDigits: d, minimumFractionDigits: d }),
  cny: n => {
    if (n >= 1e8) return (n / 1e8).toFixed(2) + " 亿元";
    if (n >= 1e4) return (n / 1e4).toFixed(1) + " 万元";
    return n.toFixed(0) + " 元";
  },
  watts: w => w >= 1000 ? (w / 1000).toFixed(1) + " kW" : w.toFixed(0) + " W",
  bytes: b => {
    if (b >= 1e15) return (b / 1e15).toFixed(2) + " PB";
    if (b >= 1e12) return (b / 1e12).toFixed(1) + " TB";
    if (b >= 1e9)  return (b / 1e9).toFixed(1)  + " GB";
    if (b >= 1e6)  return (b / 1e6).toFixed(1)  + " MB";
    return b.toFixed(0) + " B";
  },
  flops: f => {
    if (f >= 1e24) return (f / 1e24).toFixed(2) + " YFLOPs";
    if (f >= 1e21) return (f / 1e21).toFixed(2) + " ZFLOPs";
    if (f >= 1e18) return (f / 1e18).toFixed(2) + " EFLOPs";
    if (f >= 1e15) return (f / 1e15).toFixed(2) + " PFLOPs";
    return f.toExponential(2);
  },
};

/* ---------- Chart.js 全局主题 ---------- */
function initChartTheme() {
  if (!window.Chart) return;
  Chart.defaults.color = "#aeb9cd";
  Chart.defaults.font.family = '"PingFang SC","Microsoft YaHei UI",system-ui,sans-serif';
  Chart.defaults.font.size = 12;
  Chart.defaults.borderColor = "#1f2a3e";
  Chart.defaults.scale.grid.color = "#1f2a3e";
  Chart.defaults.scale.ticks.color = "#aeb9cd";
}

const COLORS = {
  nv: "#76b900", zh: "#2c7eff",
  warn: "#f59e0b", danger: "#ef4444", ok: "#10b981",
  fg: "#e9eef7", line: "#2b3a55", bg: "#141a28",
  palette: ["#76b900", "#2c7eff", "#f59e0b", "#10b981", "#ef4444", "#a78bfa", "#ec4899", "#22d3ee"],
};

/* ---------- 数据加载 ---------- */
let DATA = null, REFS = null;
async function loadData() {
  const [d, r] = await Promise.all([
    fetch("data/computed.json").then(r => r.json()),
    fetch("data/references.json").then(r => r.json()),
  ]);
  DATA = d; REFS = r;
  return { DATA, REFS };
}

/* ---------- KaTeX ---------- */
function renderMath(root = document.body) {
  if (!window.renderMathInElement) return;
  renderMathInElement(root, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "$",  right: "$",  display: false },
    ],
    throwOnError: false,
  });
}

/* ---------- 章节导航高亮 ---------- */
function initNavSpy() {
  const links = $$(".nav-list a");
  const map   = new Map(links.map(a => [a.getAttribute("href").slice(1), a]));
  const obs   = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        links.forEach(a => a.classList.remove("active"));
        const a = map.get(e.target.id);
        if (a) a.classList.add("active");
      }
    });
  }, { rootMargin: "-40% 0px -55% 0px", threshold: 0 });
  $$("section.chapter, #hero").forEach(s => obs.observe(s));
}

/* ===========================================================================
 * §3  Scaling Law 交互式计算器
 * =========================================================================== */
function initScalingCalculator() {
  const root = $("#calc-scaling");
  if (!root) return;

  const elN     = $("#sl-n");
  const elD     = $("#sl-d");
  const elGPU   = $("#sl-gpu");
  const elMfu   = $("#sl-mfu");
  const elDays  = $("#sl-days");
  const out     = $("#sl-out");

  function compute() {
    const N = parseFloat(elN.value);          // active params
    const D = parseFloat(elD.value);          // tokens
    const gk = elGPU.value;
    const mfu = parseFloat(elMfu.value);
    const days = parseFloat(elDays.value);

    const overhead = 8.0 / 6.0;
    const flops = 6 * N * D * overhead;       // FLOPs
    const tflops = DATA.constants.gpus[gk].bf16_tflops;
    const gpu_seconds = flops / (tflops * 1e12 * mfu);
    const n_gpu = Math.ceil(gpu_seconds / (days * 86400) / 8) * 8;
    const sustained = n_gpu * tflops * mfu / 1e3;  // PFLOPS
    const tdp = DATA.constants.gpus[gk].tdp_w * n_gpu / 1000; // kW
    const energy_kwh = tdp * 1.15 * days * 24;     // PUE 1.15
    const cost = energy_kwh * 0.6;

    $("#sl-flops").textContent = fmt.flops(flops);
    $("#sl-ngpu").textContent = fmt.int(n_gpu);
    $("#sl-sustained").textContent = fmt.fixed(sustained, 1) + " PFLOPS";
    $("#sl-tdp").textContent = fmt.fixed(tdp, 0) + " kW IT";
    $("#sl-energy").textContent = fmt.fixed(energy_kwh / 1000, 1) + " MWh";
    $("#sl-cost").textContent = fmt.cny(cost);

    // value indicators
    $("#sl-n-v").textContent  = (N / 1e10).toFixed(1) + " × 10¹⁰";
    $("#sl-d-v").textContent  = (D / 1e12).toFixed(1) + " T tokens";
    $("#sl-mfu-v").textContent = (mfu * 100).toFixed(0) + " %";
    $("#sl-days-v").textContent = days + " 天";
  }

  [elN, elD, elGPU, elMfu, elDays].forEach(el => el.addEventListener("input", compute));
  compute();
}

/* ===========================================================================
 * §4  GPU 选型 — 综合对比柱状图
 * =========================================================================== */
function initGPUCompareChart() {
  const ctx = $("#gpu-bench")?.getContext("2d");
  if (!ctx) return;
  const gpus = DATA.constants.gpus;
  const keys = ["h100", "h200", "b200", "gb200"];
  const labels = keys.map(k => gpus[k].name);
  const tflops = keys.map(k => gpus[k].bf16_tflops);
  const hbm    = keys.map(k => gpus[k].hbm_gb);
  const bw     = keys.map(k => gpus[k].hbm_bw_tbs);
  const tdp    = keys.map(k => gpus[k].tdp_w);

  new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        { label: "BF16 TFLOPS", data: tflops, backgroundColor: COLORS.nv },
        { label: "HBM (GB)",    data: hbm,    backgroundColor: COLORS.zh, yAxisID: "y2" },
        { label: "HBM 带宽 (TB/s × 100)", data: bw.map(v => v * 100), backgroundColor: COLORS.warn, yAxisID: "y2" },
      ],
    },
    options: {
      responsive: true,
      plugins: { legend: { position: "top" }, title: { display: false } },
      scales: {
        y:  { beginAtZero: true, title: { display: true, text: "TFLOPS (BF16, dense)" } },
        y2: { beginAtZero: true, position: "right", title: { display: true, text: "GB / (TB/s × 100)" }, grid: { drawOnChartArea: false } },
      },
    },
  });
}

/* ===========================================================================
 * §3.6  集群规模 vs GPU 数量 折线图
 * =========================================================================== */
function initClusterFixedChart() {
  const ctx = $("#cluster-fixed")?.getContext("2d");
  if (!ctx) return;
  const fixed = DATA.cluster.fixed_gpu_baseline;
  const xs = fixed.h100.map(p => p.n_gpu);
  const palette = { h100: COLORS.fg, h200: COLORS.warn, b200: COLORS.nv, gb200: COLORS.zh };
  const labelMap = { h100: "H100", h200: "H200", b200: "B200", gb200: "GB200 NVL72" };

  new Chart(ctx, {
    type: "line",
    data: {
      labels: xs,
      datasets: Object.keys(fixed).map(k => ({
        label: labelMap[k],
        data: fixed[k].map(p => p.days),
        borderColor: palette[k],
        backgroundColor: palette[k] + "33",
        tension: 0.25, borderWidth: 2, pointRadius: 4,
      })),
    },
    options: {
      responsive: true,
      plugins: {
        title: { display: true, text: "训练 GLM-5.1 Baseline (1 T 总参 / 64 B 激活 / 20 T tokens) 所需天数" },
        tooltip: { callbacks: { label: c => `${c.dataset.label}: ${c.parsed.y.toFixed(1)} 天` } },
      },
      scales: {
        x: { title: { display: true, text: "GPU 数量" } },
        y: { type: "logarithmic", title: { display: true, text: "训练天数 (log)" } },
      },
    },
  });
}

/* ===========================================================================
 * §6  显存分布饼图 (Baseline B200)
 * =========================================================================== */
function initMemoryPie() {
  const ctx = $("#mem-pie")?.getContext("2d");
  if (!ctx) return;
  const bd = DATA.memory.breakdown.b200;
  const labels = ["BF16 权重", "BF16 梯度", "FP32 优化器", "激活", "通信缓冲"];
  const data = [bd.gb.weight_bf16, bd.gb.grad_bf16, bd.gb.optim_fp32, bd.gb.activation, bd.gb.buffer];

  new Chart(ctx, {
    type: "doughnut",
    data: {
      labels,
      datasets: [{ data, backgroundColor: COLORS.palette.slice(0, 5), borderColor: COLORS.bg, borderWidth: 2 }],
    },
    options: {
      cutout: "55%",
      plugins: {
        legend: { position: "right" },
        tooltip: { callbacks: { label: c => `${c.label}: ${c.parsed.toFixed(1)} GB` } },
        title: { display: true, text: `每 GPU 显存占用合计 ${bd.gb.total.toFixed(1)} GB / B200 192 GB` },
      },
    },
  });
}

/* ===========================================================================
 * §8  PUE → 年耗电柱状
 * =========================================================================== */
function initPueChart() {
  const ctx = $("#pue-chart")?.getContext("2d");
  if (!ctx) return;
  const sw = DATA.power.pue_sweep;
  new Chart(ctx, {
    type: "bar",
    data: {
      labels: sw.map(p => "PUE " + p.pue.toFixed(2)),
      datasets: [
        { label: "年耗电 (MWh)",   data: sw.map(p => p.annual_mwh),         backgroundColor: COLORS.zh },
        { label: "年 CO₂ (吨)",    data: sw.map(p => p.annual_co2_t),       backgroundColor: COLORS.warn, yAxisID: "y2" },
      ],
    },
    options: {
      plugins: { title: { display: true, text: "B200 集群 IT 负载固定下，PUE 对年能耗与碳排的放大效应" } },
      scales: {
        y: { title: { display: true, text: "年耗电 (MWh)" } },
        y2: { position: "right", title: { display: true, text: "年 CO₂ (吨)" }, grid: { drawOnChartArea: false } },
      },
    },
  });
}

/* ===========================================================================
 * §10  TCO 堆叠柱状图 + 滑块联动
 * =========================================================================== */
function initTcoChart() {
  const ctx = $("#tco-chart")?.getContext("2d");
  if (!ctx) return;
  const cmp = DATA.tco.comparison_at_baseline;
  const keys = Object.keys(cmp);
  const labels = keys.map(k => k.replace("|", " × ").replace("hgx_", "").replace("ib_", "IB-").toUpperCase());
  const cap = keys.map(k => cmp[k].capex_total_cny / 1e8);
  const op  = keys.map(k => cmp[k].opex_3y_cny / 1e8);

  const chart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        { label: "CAPEX (亿元)",     data: cap, backgroundColor: COLORS.nv },
        { label: "OPEX 3 年 (亿元)", data: op,  backgroundColor: COLORS.zh },
      ],
    },
    options: {
      plugins: {
        title: { display: true, text: "三年 TCO 对比：5 种硬件 × 网络组合 (固定 2 048 张 GPU 同等规模)" },
        tooltip: { callbacks: { label: c => `${c.dataset.label}: ${c.parsed.y.toFixed(2)} 亿元` } },
      },
      responsive: true,
      scales: {
        x: { stacked: true },
        y: { stacked: true, title: { display: true, text: "成本 (亿元)" } },
      },
    },
  });

  // 滑块：调整规模 → 重算
  const sn = $("#tco-n-gpu");
  const so = $("#tco-n-out");
  const sg = $("#tco-server");
  if (sn) {
    function update() {
      const n = parseInt(sn.value, 10);
      so.textContent = n.toLocaleString("zh-CN") + " 张 GPU";
      const sk = sg.value;
      // 简化估算：线性按比例放缩 baseline 数据
      const baseN = DATA.tco.selected_plan_b200.n_gpu;
      const ratio = n / baseN;
      const base = sk === "gb200" ? DATA.tco.selected_plan_gb200 : DATA.tco.selected_plan_b200;
      const c = base.capex_total_cny * ratio;
      const o = base.opex_3y_cny * ratio;
      $("#tco-capex-out").textContent = (c / 1e8).toFixed(2) + " 亿元";
      $("#tco-opex-out").textContent = (o / 1e8).toFixed(2) + " 亿元";
      $("#tco-total-out").textContent = ((c + o) / 1e8).toFixed(2) + " 亿元";
      $("#tco-cph-out").textContent = base.cny_per_gpu_hour.toFixed(1) + " ¥";
    }
    sn.addEventListener("input", update);
    sg.addEventListener("change", update);
    update();
  }
}

/* ===========================================================================
 * §9  推理吞吐对比柱状
 * =========================================================================== */
function initInferenceChart() {
  const ctx = $("#inf-chart")?.getContext("2d");
  if (!ctx) return;
  const cmp = DATA.inference.comparison_8k;
  const keys = ["h100", "h200", "b200", "gb200"];
  const labels = keys.map(k => DATA.constants.gpus[k].name);

  new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        { label: "Decode 吞吐 (tok/s, TP=8)", data: keys.map(k => cmp[k].decode_throughput_tok_s), backgroundColor: COLORS.nv },
        { label: "并发请求上限 (8K seq)",     data: keys.map(k => cmp[k].max_concurrent),         backgroundColor: COLORS.zh, yAxisID: "y2" },
      ],
    },
    options: {
      plugins: { title: { display: true, text: "推理服务能力（GLM-5.1 baseline，TP=8 单副本）" } },
      scales: {
        y:  { title: { display: true, text: "Decode 吞吐 (tok/s)" } },
        y2: { position: "right", title: { display: true, text: "并发请求上限" }, grid: { drawOnChartArea: false } },
      },
    },
  });
}

/* ===========================================================================
 * §11  参考文献渲染
 * =========================================================================== */
function renderReferences() {
  const ol = $("#refs-list");
  if (!ol || !REFS) return;
  ol.innerHTML = REFS.items.map((r, i) =>
    `<li id="${r.id}">[${i + 1}] ${r.text} ${r.url ? `<a href="${r.url}" target="_blank" rel="noopener">[link]</a>` : ""}</li>`
  ).join("");
}

/* ===========================================================================
 * 文末数据溯源徽章
 * =========================================================================== */
function fillProvenance() {
  const stamp = $("#data-stamp");
  if (stamp && DATA) {
    const t = new Date(DATA.generated_at).toLocaleString("zh-CN");
    stamp.textContent = `数据生成于 ${t} · schema v${DATA.schema_version} · generator v${DATA.generator_version}`;
  }
}

/* ===========================================================================
 * 关键 KPI 数值填充（Hero / 章节内 data-kpi 占位符）
 * =========================================================================== */
function fillKPIs() {
  if (!DATA) return;
  const baseline = DATA.scaling.baseline;
  // 优先用"selected"（人为整定到 2048）的方案；若没有则回退到 90 天等效卡数
  const cluster90_b200 = DATA.cluster.target_90d.baseline.b200_selected
                       || DATA.cluster.target_90d.baseline.b200;
  const cluster90_gb200 = DATA.cluster.target_90d.baseline.gb200_selected
                        || DATA.cluster.target_90d.baseline.gb200;
  const power_b200 = DATA.power.report_b200;
  const tco_b200 = DATA.tco.selected_plan_b200;

  const map = {
    "kpi-flops":          fmt.flops(baseline.flops_engineering),
    "kpi-pflopdays":      fmt.int(baseline.petaflop_day),
    "kpi-tokens":         (baseline.train_tokens / 1e12).toFixed(0) + " T tokens",
    "kpi-nact":           (baseline.n_active / 1e9).toFixed(0) + " B",
    "kpi-ntotal":         (baseline.n_total / 1e9).toFixed(0) + " B",
    "kpi-b200-ngpu":      fmt.int(cluster90_b200.n_gpu),
    "kpi-b200-pf":        fmt.fixed(cluster90_b200.sustained_pflops, 0) + " PFLOPS",
    "kpi-b200-days":      cluster90_b200.days.toFixed(0) + " 天",
    "kpi-gb200-ngpu":     fmt.int(cluster90_gb200.n_gpu),
    "kpi-gb200-days":     cluster90_gb200.days.toFixed(0) + " 天",
    "kpi-power-it":       fmt.fixed(power_b200.p_it_kw, 0) + " kW",
    "kpi-power-total":    fmt.fixed(power_b200.p_total_kw, 0) + " kW",
    "kpi-power-mwh":      fmt.fixed(power_b200.annual_kwh / 1000, 0) + " MWh",
    "kpi-power-cny":      fmt.cny(power_b200.annual_cny),
    "kpi-power-co2":      fmt.fixed(power_b200.annual_co2_t, 0) + " 吨",
    "kpi-tco-capex":      fmt.cny(tco_b200.capex_total_cny),
    "kpi-tco-opex":       fmt.cny(tco_b200.opex_3y_cny),
    "kpi-tco-total":      fmt.cny(tco_b200.tco_3y_cny),
    "kpi-tco-cph":        tco_b200.cny_per_gpu_hour.toFixed(1) + " ¥",
  };
  Object.entries(map).forEach(([id, val]) => {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  });
}

/* ===========================================================================
 * 启动
 * =========================================================================== */
async function main() {
  initChartTheme();
  await loadData();
  fillKPIs();
  renderReferences();
  fillProvenance();
  initNavSpy();

  // 等 KaTeX/Chart.js (defer) 都到位
  const ready = () => {
    renderMath();
    initScalingCalculator();
    initGPUCompareChart();
    initClusterFixedChart();
    initMemoryPie();
    initPueChart();
    initTcoChart();
    initInferenceChart();
  };
  if (document.readyState === "complete") ready();
  else window.addEventListener("load", ready);
}

main().catch(err => {
  console.error(err);
  document.body.insertAdjacentHTML("afterbegin",
    `<div style="background:#7a1010;color:#fff;padding:14px 20px;font-family:monospace">
       数据加载失败：${err.message}。请确认已运行 <code>python -m models.run_all</code>，并通过 HTTP 访问（非 file://）。
     </div>`);
});
