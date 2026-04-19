"""硬件 / 物理 / 经济常数表
================================

所有数值都给出来源。若无法精确公开（如 OEM 渠道单价），按行业普遍接受
的 2025 Q4 中位数取值，并注明 "industry consensus 2025"。

单位约定：
    - 算力 (compute)        : FLOPS  (1 TFLOPS = 1e12)
    - 显存 (memory)         : Bytes  (1 GB = 1e9, 1 GiB = 2**30)
    - 带宽 (bandwidth)      : Bytes/s (1 GB/s = 1e9)
    - 网络速率 (link rate)  : bit/s  (1 Gbps = 1e9 bit/s = 0.125 GB/s)
    - 功率                  : Watt
    - 货币                  : 人民币元 (¥, CNY)
    - 时间                  : 秒；输出时换算到天/小时
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, List

# ---------------------------------------------------------------------------
# 1. NVIDIA Datacenter GPU 参数
# ---------------------------------------------------------------------------
# 来源：
#   * NVIDIA H100/H200 Datasheet  (2023-09 / 2024-08)
#   * NVIDIA Blackwell GTC 2024 Keynote, B200/GB200 Architecture Whitepaper
#   * NVIDIA HGX/GB200 NVL72 Product Brief
#   * MLPerf Training v4.0 / v4.1 results
# 注：所有 BF16 FLOPS 均为 dense（不含 sparsity 2× 加成），训练实际可用值
# 注：单价均为 2025 Q4 大陆渠道行情 (industry consensus, 含税不含运)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GPUSpec:
    name: str
    arch: str
    bf16_tflops: float          # dense BF16 Tensor Core (TFLOPS)
    fp8_tflops: float           # dense FP8
    fp4_tflops: float           # dense FP4 (Blackwell+)
    hbm_gb: float               # HBM 容量 (GB, 厂商十进制)
    hbm_bw_tbs: float           # HBM 带宽 (TB/s, 厂商十进制)
    nvlink_bw_gbs: float        # 单 GPU NVLink 双向 (GB/s)
    tdp_w: int                  # 单卡 TDP (W)
    price_cny: float            # 中位渠道单价 (¥)
    available_year: int         # 量产可得年份


GPUS: Dict[str, GPUSpec] = {
    "h100": GPUSpec(
        name="H100 SXM5",
        arch="Hopper",
        bf16_tflops=989.0,
        fp8_tflops=1979.0,
        fp4_tflops=0.0,
        hbm_gb=80.0,
        hbm_bw_tbs=3.35,
        nvlink_bw_gbs=900.0,
        tdp_w=700,
        price_cny=250_000,
        available_year=2023,
    ),
    "h200": GPUSpec(
        name="H200 SXM5",
        arch="Hopper",
        bf16_tflops=989.0,
        fp8_tflops=1979.0,
        fp4_tflops=0.0,
        hbm_gb=141.0,
        hbm_bw_tbs=4.80,
        nvlink_bw_gbs=900.0,
        tdp_w=700,
        price_cny=320_000,
        available_year=2024,
    ),
    "b200": GPUSpec(
        name="B200 SXM",
        arch="Blackwell",
        bf16_tflops=2250.0,    # 2.25 PFLOPS dense BF16
        fp8_tflops=4500.0,
        fp4_tflops=9000.0,
        hbm_gb=192.0,
        hbm_bw_tbs=8.00,
        nvlink_bw_gbs=1800.0,
        tdp_w=1000,
        price_cny=450_000,
        available_year=2025,
    ),
    "gb200": GPUSpec(
        # 注：GB200 Superchip = 1 Grace + 2 B200 ；NVL72 整机柜 = 36 superchip = 72 GPU
        # 这里 spec 取 "单 GPU" 视角以便对比
        name="GB200 (per GPU, NVL72)",
        arch="Blackwell + Grace",
        bf16_tflops=2500.0,
        fp8_tflops=5000.0,
        fp4_tflops=10000.0,
        hbm_gb=192.0,
        hbm_bw_tbs=8.00,
        nvlink_bw_gbs=1800.0,
        tdp_w=1200,             # 含 Grace 分摊
        price_cny=550_000,      # 折算自 NVL72 整柜 ≈ 2800 万 ¥ / 72
        available_year=2025,
    ),
}

# ---------------------------------------------------------------------------
# 2. 服务器 / 机柜参数
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ServerSpec:
    name: str
    gpu_key: str
    gpu_per_node: int
    cpu_cores: int
    sysmem_gb: int
    nvme_tb: int
    node_tdp_w: int            # 整机 TDP (含 CPU/IB/风扇)
    node_price_cny: float      # 整机 BOM (含 GPU)
    rack_u: int                # 机柜 U 数
    rack_gpu_density: int      # 单机柜 GPU 数

SERVERS: Dict[str, ServerSpec] = {
    "hgx_h100": ServerSpec(
        name="HGX H100 8-GPU 节点 (Air Cooled)",
        gpu_key="h100",
        gpu_per_node=8,
        cpu_cores=112,         # 2× Xeon 8480C
        sysmem_gb=2048,
        nvme_tb=30,
        node_tdp_w=10_200,
        node_price_cny=2_400_000,
        rack_u=42,
        rack_gpu_density=32,   # 4 节点 / 机柜，风冷
    ),
    "hgx_h200": ServerSpec(
        name="HGX H200 8-GPU 节点 (Liquid Cooled)",
        gpu_key="h200",
        gpu_per_node=8,
        cpu_cores=128,
        sysmem_gb=2048,
        nvme_tb=30,
        node_tdp_w=10_400,
        node_price_cny=2_950_000,
        rack_u=42,
        rack_gpu_density=64,   # 8 节点 / 机柜，液冷板
    ),
    "hgx_b200": ServerSpec(
        name="HGX B200 8-GPU 节点 (DLC)",
        gpu_key="b200",
        gpu_per_node=8,
        cpu_cores=128,
        sysmem_gb=2048,
        nvme_tb=30,
        node_tdp_w=14_300,
        node_price_cny=4_100_000,
        rack_u=42,
        rack_gpu_density=64,
    ),
    "gb200_nvl72": ServerSpec(
        name="GB200 NVL72 (整机柜液冷)",
        gpu_key="gb200",
        gpu_per_node=72,       # 整机柜视为单"逻辑节点"（NVLink 域）
        cpu_cores=36 * 72,     # 36 Grace, 72 cores each
        sysmem_gb=17_280,      # 36 × 480 GB LPDDR5X
        nvme_tb=200,
        node_tdp_w=120_000,    # ≈ 120 kW per rack (含 CDU 不在内)
        node_price_cny=28_000_000,
        rack_u=42,
        rack_gpu_density=72,
    ),
}

# ---------------------------------------------------------------------------
# 3. 网络 / 互联
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NICSpec:
    name: str
    rate_gbps: int           # 单端口速率 (Gbps)
    ports_per_gpu: float     # 平均每 GPU 出口
    switch_radix: int        # 单交换机端口数
    switch_price_cny: float
    nic_price_cny: float
    cable_price_cny: float   # 含 transceiver

NICS: Dict[str, NICSpec] = {
    "ib_ndr": NICSpec(
        name="InfiniBand NDR (400 Gbps)",
        rate_gbps=400,
        ports_per_gpu=1.0,
        switch_radix=64,
        switch_price_cny=950_000,
        nic_price_cny=22_000,
        cable_price_cny=4_500,
    ),
    "ib_xdr": NICSpec(
        name="InfiniBand XDR (800 Gbps)",
        rate_gbps=800,
        ports_per_gpu=1.0,
        switch_radix=144,
        switch_price_cny=1_900_000,
        nic_price_cny=42_000,
        cable_price_cny=8_000,
    ),
    "spectrum_x_400": NICSpec(
        name="Spectrum-X Ethernet 400 Gbps",
        rate_gbps=400,
        ports_per_gpu=1.0,
        switch_radix=64,
        switch_price_cny=750_000,
        nic_price_cny=18_000,
        cable_price_cny=4_000,
    ),
    "spectrum_x_800": NICSpec(
        name="Spectrum-X Ethernet 800 Gbps",
        rate_gbps=800,
        ports_per_gpu=1.0,
        switch_radix=128,
        switch_price_cny=1_600_000,
        nic_price_cny=36_000,
        cable_price_cny=7_200,
    ),
}

# ---------------------------------------------------------------------------
# 4. 存储参考方案
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StorageSpec:
    name: str
    raw_pb: float
    read_gbs: float       # 聚合读吞吐 (GB/s)
    write_gbs: float
    price_per_tb_cny: float

STORAGE: Dict[str, StorageSpec] = {
    "weka_nvme": StorageSpec(
        name="WekaFS All-NVMe Tier-0",
        raw_pb=10.0,
        read_gbs=2_000.0,
        write_gbs=1_500.0,
        price_per_tb_cny=22_000,
    ),
    "lustre_hybrid": StorageSpec(
        name="Lustre + DDN ES400NVX2 Tier-1",
        raw_pb=50.0,
        read_gbs=1_200.0,
        write_gbs=900.0,
        price_per_tb_cny=4_500,
    ),
    "ceph_archive": StorageSpec(
        name="Ceph QLC Archive Tier-2",
        raw_pb=200.0,
        read_gbs=200.0,
        write_gbs=120.0,
        price_per_tb_cny=900,
    ),
}

# ---------------------------------------------------------------------------
# 5. 物理 / 机房
# ---------------------------------------------------------------------------

POWER_PRICE_CNY_PER_KWH = 0.60      # 北京一般工商业平均电价 (含税, 2025)
WATER_PRICE_CNY_PER_M3 = 6.0        # 北京工业用水
PUE_AIR = 1.40                      # 冷通道封闭风冷
PUE_LIQUID = 1.15                   # 板式液冷 (Direct Liquid Cooling)
PUE_IMMERSION = 1.08                # 单相浸没式
WUE_LIQUID = 0.30                   # L / kWh
GRID_CO2_KG_PER_KWH = 0.581         # 中国华北电网 2024 年平均排放因子

RACK_POWER_AIR_KW = 35              # 风冷机柜上限
RACK_POWER_LIQUID_KW = 80
RACK_POWER_GB200_KW = 132           # NVL72 含 CDU

# ---------------------------------------------------------------------------
# 6. OPEX：人工 / 运维
# ---------------------------------------------------------------------------
HEADCOUNT_AVG_SALARY_CNY = 600_000  # 含五险一金 / 年
MAINT_RATE_OF_CAPEX = 0.04          # 年均运维费率 (硬件保险 + 备件 + 工程)

# ---------------------------------------------------------------------------
# 7. GLM-5.1 模型三档假设
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelScenario:
    id: str
    label: str
    n_total: float       # 总参数量
    n_active: float      # 激活参数量 (MoE)
    n_layers: int
    hidden: int
    n_heads: int
    head_dim: int
    n_experts: int
    top_k: int
    ctx_len: int
    train_tokens: float

SCENARIOS: Dict[str, ModelScenario] = {
    "conservative": ModelScenario(
        id="conservative", label="保守 (Conservative)",
        n_total=6.0e11, n_active=4.0e10,
        n_layers=72, hidden=8192, n_heads=64, head_dim=128,
        n_experts=128, top_k=8,
        ctx_len=262_144, train_tokens=1.2e13,   # 12 T tokens
    ),
    "baseline": ModelScenario(
        id="baseline", label="基准 (Baseline · 本课主用)",
        n_total=1.0e12, n_active=6.4e10,
        n_layers=88, hidden=10_240, n_heads=80, head_dim=128,
        n_experts=160, top_k=8,
        ctx_len=524_288, train_tokens=2.0e13,   # 20 T tokens
    ),
    "frontier": ModelScenario(
        id="frontier", label="前沿 (Frontier)",
        n_total=2.0e12, n_active=1.28e11,
        n_layers=120, hidden=12_288, n_heads=96, head_dim=128,
        n_experts=256, top_k=8,
        ctx_len=1_048_576, train_tokens=4.0e13, # 40 T tokens
    ),
}

# ---------------------------------------------------------------------------
# 8. 训练实测系数（来源：MLPerf v4.1 / NeMo Performance Guide / DeepSeek-V3 报告）
# ---------------------------------------------------------------------------
MFU_BY_GPU = {
    "h100": 0.36,    # 36 % MFU on Hopper, BF16 mixed
    "h200": 0.40,    # HBM 大 → I/O 紧度低
    "b200": 0.42,
    "gb200": 0.45,   # NVL72 域内通信几乎免费
}

# 含工程开销（路由 / 重计算 / 通信尾延）的折算系数
COMPUTE_OVERHEAD_FACTOR = 8.0 / 6.0   # 约 1.33×, 见 design.md §3.2

# ---------------------------------------------------------------------------
# 9. 序列化助手（dataclass → dict）
# ---------------------------------------------------------------------------

def all_constants_as_dict() -> dict:
    return {
        "gpus": {k: asdict(v) for k, v in GPUS.items()},
        "servers": {k: asdict(v) for k, v in SERVERS.items()},
        "nics": {k: asdict(v) for k, v in NICS.items()},
        "storage": {k: asdict(v) for k, v in STORAGE.items()},
        "scenarios": {k: asdict(v) for k, v in SCENARIOS.items()},
        "physics": {
            "power_price_cny_per_kwh": POWER_PRICE_CNY_PER_KWH,
            "water_price_cny_per_m3": WATER_PRICE_CNY_PER_M3,
            "pue_air": PUE_AIR,
            "pue_liquid": PUE_LIQUID,
            "pue_immersion": PUE_IMMERSION,
            "wue_liquid": WUE_LIQUID,
            "grid_co2_kg_per_kwh": GRID_CO2_KG_PER_KWH,
            "rack_power_air_kw": RACK_POWER_AIR_KW,
            "rack_power_liquid_kw": RACK_POWER_LIQUID_KW,
            "rack_power_gb200_kw": RACK_POWER_GB200_KW,
        },
        "opex": {
            "headcount_salary_cny": HEADCOUNT_AVG_SALARY_CNY,
            "maint_rate_of_capex": MAINT_RATE_OF_CAPEX,
        },
        "training": {
            "mfu": MFU_BY_GPU,
            "compute_overhead_factor": COMPUTE_OVERHEAD_FACTOR,
        },
    }


if __name__ == "__main__":
    import json, sys
    json.dump(all_constants_as_dict(), sys.stdout, ensure_ascii=False, indent=2)
