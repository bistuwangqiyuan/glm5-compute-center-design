"""通信 / 网络带宽建模
==========================

公式来源：
    * Patarasuk & Yuan 2009, "Bandwidth Optimal All-reduce ..."
      Ring AllReduce: T = 2(P-1) · M / (P · B)
    * Sergeev & Del Balso 2018, "Horovod"
    * NVIDIA Mellanox Sharp 加速：可让 P 项消失（视作 1）

我们对：
  - 跨节点 AllReduce（梯度同步）
  - MoE All-to-All（专家路由）
分别给出时间估算。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

from . import constants as C


@dataclass
class CommReport:
    nic_key: str
    n_gpu: int
    bytes_total: float
    bandwidth_per_gpu_gbs: float    # GB/s 单端口可用带宽（双向算单向）
    allreduce_time_s: float
    a2a_time_s: float
    nic_bw_utilization: float        # 我们假设有效带宽 = 80 % 线速


def link_bandwidth_gbs(nic_key: str, util: float = 0.80) -> float:
    """单端口有效带宽 (GB/s)。"""
    n = C.NICS[nic_key]
    return n.rate_gbps * util / 8.0   # bit→Byte


def allreduce_time(bytes_msg: float, n_gpu: int, bw_gbs: float, sharp: bool = False) -> float:
    """Ring AllReduce 时间。SHARP 加速时近似 = bytes/bw。"""
    if sharp:
        return bytes_msg / (bw_gbs * 1e9)
    p = max(2, n_gpu)
    return 2 * (p - 1) / p * bytes_msg / (bw_gbs * 1e9)


def a2a_time(tokens_per_step: int, hidden: int, ep: int, bw_gbs: float) -> float:
    """MoE All-to-All：每步搬运 tokens·H·2 字节 (BF16) 跨 ep 路。"""
    bytes_a2a = tokens_per_step * hidden * 2 * (ep - 1) / ep
    return bytes_a2a / (bw_gbs * 1e9)


def grad_size_bytes(n_active: float) -> float:
    """每次反传需要 AllReduce 的梯度量 (BF16)."""
    return n_active * 2.0


def report(nic_key: str, n_gpu: int, scenario_id: str = "baseline",
           tokens_per_step: int = 4_000_000, ep: int = 8) -> CommReport:
    s = C.SCENARIOS[scenario_id]
    bw = link_bandwidth_gbs(nic_key)
    grad_bytes = grad_size_bytes(s.n_active)
    t_ar = allreduce_time(grad_bytes, n_gpu, bw, sharp=False)
    t_a2a = a2a_time(tokens_per_step, s.hidden, ep, bw)
    return CommReport(
        nic_key=nic_key, n_gpu=n_gpu,
        bytes_total=grad_bytes, bandwidth_per_gpu_gbs=bw,
        allreduce_time_s=t_ar, a2a_time_s=t_a2a,
        nic_bw_utilization=0.80,
    )


def comparison_grid(n_gpu: int = 2048) -> Dict[str, dict]:
    out = {}
    for nk in C.NICS:
        out[nk] = asdict(report(nk, n_gpu))
    return out


def topology_estimate(n_gpu: int, nic_key: str) -> dict:
    """估算 Fat-tree (3-tier) 所需交换机数量与端口。

    简化：
      Leaf : 每 leaf 服务 32 GPU (4 节点)，对上 32 端口  → leaf_n = ceil(n_gpu / 32)
      Spine: leaf_n × 32 上行端口需要 spine 端口；按每 spine radix port → spine_n = ceil(leaf_n*32 / radix)
      Core : 同理一层
    """
    n = C.NICS[nic_key]
    radix = n.switch_radix
    leaf_n = (n_gpu + 31) // 32
    spine_n = max(1, (leaf_n * 32 + radix - 1) // radix)
    core_n = max(1, (spine_n * radix // 2 + radix - 1) // radix)
    cables = n_gpu + leaf_n * 32 + spine_n * (radix // 2)
    cost_switch = (leaf_n + spine_n + core_n) * n.switch_price_cny
    cost_nic = n_gpu * n.nic_price_cny
    cost_cable = cables * n.cable_price_cny
    return {
        "nic_key": nic_key, "n_gpu": n_gpu, "radix": radix,
        "leaf_switches": leaf_n, "spine_switches": spine_n, "core_switches": core_n,
        "total_switches": leaf_n + spine_n + core_n,
        "total_nics": n_gpu,
        "total_cables": cables,
        "cost_cny": {
            "switches": cost_switch, "nics": cost_nic, "cables": cost_cable,
            "total": cost_switch + cost_nic + cost_cable,
        },
    }


def _selftest() -> None:
    r = report("ib_ndr", 2048)
    print(f"[network] IB NDR 2048 GPU AllReduce 64 GB grad: {r.allreduce_time_s*1000:.1f} ms  ✔")
    assert r.allreduce_time_s > 0
    t = topology_estimate(2048, "ib_ndr")
    print(f"[network] Fat-tree 2048 GPU IB-NDR: leaf={t['leaf_switches']} spine={t['spine_switches']} "
          f"cost={t['cost_cny']['total']/1e6:.1f} M¥  ✔")


if __name__ == "__main__":
    import sys, json
    if "--selftest" in sys.argv:
        _selftest()
    else:
        json.dump({
            "comm_2048": comparison_grid(2048),
            "topology_2048_ndr": topology_estimate(2048, "ib_ndr"),
            "topology_2048_xdr": topology_estimate(2048, "ib_xdr"),
        }, sys.stdout, ensure_ascii=False, indent=2)
