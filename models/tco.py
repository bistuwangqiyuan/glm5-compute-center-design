"""TCO / CAPEX / OPEX 三年总拥有成本建模
==========================================

CAPEX 项：
  1. GPU 服务器（含 GPU/CPU/RAM/NVMe 等整机）
  2. 网络（交换机 + 网卡 + 线缆）
  3. 存储（三层）
  4. 机房设施（土建 + 配电 + UPS + 制冷 CAPEX 摊销，本模型按 CAPEX_RATIO_FACILITY 折算）
  5. 软件 / 集群管理（折算 NVIDIA AI Enterprise / Slurm 商业支持）

OPEX 项 (年化，× 3)：
  - 电力
  - 运维人员
  - 硬件保险 + 备件
  - 网络出口（科研网带宽 100 Gbps × N 月）
  - 软件订阅
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List

from . import constants as C
from . import power_cooling as PC
from . import network_bandwidth as NW
from . import storage_io as ST


# 经验摊销系数
CAPEX_RATIO_FACILITY = 0.18         # 机房 CAPEX ≈ 服务器 CAPEX 的 18 %（含液冷 CDU、配电、UPS、土建摊销）
CAPEX_SOFTWARE_PER_GPU_CNY = 5_000  # NVAIE / Slurm Pro / observability 一次性
HEADCOUNT_PER_1000_GPU = 8          # 平均每 1000 GPU 8 个 FTE（含 SRE / 平台 / 数据 / 网络 / 安全）
NETWORK_EGRESS_CNY_PER_YEAR = 4_000_000   # 100 Gbps 国内运营商专线 × 12 月 估算
SOFTWARE_OPEX_PER_GPU_PER_YEAR = 8_000    # NVAIE 订阅 / 监控 SaaS / W&B-like


@dataclass
class TCOReport:
    server_key: str
    nic_key: str
    n_gpu: int
    capex: dict
    opex_annual: dict
    capex_total_cny: float
    opex_3y_cny: float
    tco_3y_cny: float
    cny_per_gpu_hour: float       # 折算到每 GPU·小时成本


def capex_breakdown(server_key: str, nic_key: str, n_gpu: int,
                    storage_keys: List[str] = None) -> dict:
    s = C.SERVERS[server_key]
    n_node = (n_gpu + s.gpu_per_node - 1) // s.gpu_per_node
    server_capex = n_node * s.node_price_cny
    net = NW.topology_estimate(n_gpu, nic_key)["cost_cny"]["total"]
    storage_keys = storage_keys or ["weka_nvme", "lustre_hybrid", "ceph_archive"]
    tier = ST.tiered_design("baseline")
    storage_capex = (
        tier["tier0_hot_pb"] * 1000 * C.STORAGE["weka_nvme"].price_per_tb_cny
        + tier["tier1_warm_pb"] * 1000 * C.STORAGE["lustre_hybrid"].price_per_tb_cny
        + tier["tier2_cold_pb"] * 1000 * C.STORAGE["ceph_archive"].price_per_tb_cny
    )
    facility_capex = server_capex * CAPEX_RATIO_FACILITY
    software_capex = n_gpu * CAPEX_SOFTWARE_PER_GPU_CNY
    total = server_capex + net + storage_capex + facility_capex + software_capex
    return {
        "server": server_capex,
        "network": net,
        "storage": storage_capex,
        "facility": facility_capex,
        "software": software_capex,
        "total": total,
    }


def opex_annual(server_key: str, n_gpu: int) -> dict:
    pc = PC.report(server_key, n_gpu)
    headcount = max(8, n_gpu / 1000 * HEADCOUNT_PER_1000_GPU)
    salary = headcount * C.HEADCOUNT_AVG_SALARY_CNY
    cap = capex_breakdown(server_key, "ib_ndr", n_gpu)["total"]
    maint = cap * C.MAINT_RATE_OF_CAPEX
    soft = n_gpu * SOFTWARE_OPEX_PER_GPU_PER_YEAR
    egress = NETWORK_EGRESS_CNY_PER_YEAR
    return {
        "power": pc.annual_cny,
        "salary": salary,
        "headcount": headcount,
        "maintenance": maint,
        "software": soft,
        "network_egress": egress,
        "total": pc.annual_cny + salary + maint + soft + egress,
    }


def report(server_key: str, nic_key: str, n_gpu: int,
           years: int = 3, util: float = 0.85) -> TCOReport:
    cap = capex_breakdown(server_key, nic_key, n_gpu)
    op = opex_annual(server_key, n_gpu)
    op3 = op["total"] * years
    tco = cap["total"] + op3
    gpu_hours_3y = n_gpu * 8760 * years * util
    cphr = tco / gpu_hours_3y
    return TCOReport(
        server_key=server_key, nic_key=nic_key, n_gpu=n_gpu,
        capex=cap, opex_annual=op,
        capex_total_cny=cap["total"], opex_3y_cny=op3, tco_3y_cny=tco,
        cny_per_gpu_hour=cphr,
    )


def comparison(n_gpu: int = 2048) -> Dict[str, dict]:
    pairs = [
        ("hgx_h100", "ib_ndr"),
        ("hgx_h200", "ib_ndr"),
        ("hgx_b200", "ib_ndr"),
        ("hgx_b200", "ib_xdr"),
        ("gb200_nvl72", "ib_xdr"),
    ]
    out = {}
    for sk, nk in pairs:
        out[f"{sk}|{nk}"] = asdict(report(sk, nk, n_gpu))
    return out


def _selftest() -> None:
    r = report("hgx_b200", "ib_ndr", 2048)
    print(f"[tco] B200 2048GPU CAPEX={r.capex_total_cny/1e8:.2f} 亿 CNY   "
          f"OPEX 3y={r.opex_3y_cny/1e8:.2f} 亿 CNY   TCO={r.tco_3y_cny/1e8:.2f} 亿 CNY   "
          f"GPU·h 单价={r.cny_per_gpu_hour:.1f} CNY  [OK]")
    # B200 2048 整机 256 节点 × 4.1 M = 1049 M (10.5 亿)
    assert 8e8 < r.capex_total_cny < 25e8


if __name__ == "__main__":
    import sys, json
    if "--selftest" in sys.argv:
        _selftest()
    else:
        json.dump(comparison(), sys.stdout, ensure_ascii=False, indent=2)
