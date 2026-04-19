"""电力 / PUE / 制冷 / 机房物理设计建模
========================================

E_annual = P_IT × PUE × 8760  (kWh/年)

WUE_annual = P_IT × PUE × WUE × 8760 / 1000   (m³/年)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List

from . import constants as C


@dataclass
class PowerReport:
    server_key: str
    n_gpu: int
    n_node: int
    n_rack: int
    p_it_kw: float                # 服务器 IT 功率
    pue: float
    p_total_kw: float             # IT × PUE
    annual_kwh: float
    annual_cny: float
    annual_co2_t: float
    annual_water_m3: float
    cooling_type: str
    rack_layout: str


def _pue_for(server_key: str) -> tuple[float, str]:
    if server_key == "hgx_h100":
        return C.PUE_AIR, "风冷 (Air Cooled, 冷通道封闭)"
    if server_key in ("hgx_h200", "hgx_b200"):
        return C.PUE_LIQUID, "板式液冷 (Direct-to-Chip Liquid Cooling)"
    if server_key == "gb200_nvl72":
        return C.PUE_LIQUID, "整机柜液冷 (NVL72 全液冷, CDU 冗余)"
    return C.PUE_AIR, "风冷"


def report(server_key: str, n_gpu: int) -> PowerReport:
    s = C.SERVERS[server_key]
    g = C.GPUS[s.gpu_key]
    n_node = (n_gpu + s.gpu_per_node - 1) // s.gpu_per_node
    n_rack = (n_gpu + s.rack_gpu_density - 1) // s.rack_gpu_density
    p_it = n_node * s.node_tdp_w / 1000.0   # kW
    pue, cool = _pue_for(server_key)
    p_total = p_it * pue
    annual_kwh = p_total * 8760
    cny = annual_kwh * C.POWER_PRICE_CNY_PER_KWH
    co2_t = annual_kwh * C.GRID_CO2_KG_PER_KWH / 1000
    water_m3 = annual_kwh * C.WUE_LIQUID if pue < 1.30 else 0.0
    layout = (f"{n_rack} 个 42U 机柜，每柜 {s.rack_gpu_density} 张 GPU，"
              f"约 {p_it/n_rack:.0f} kW/柜")
    return PowerReport(
        server_key=server_key, n_gpu=n_gpu, n_node=n_node, n_rack=n_rack,
        p_it_kw=p_it, pue=pue, p_total_kw=p_total,
        annual_kwh=annual_kwh, annual_cny=cny, annual_co2_t=co2_t,
        annual_water_m3=water_m3,
        cooling_type=cool, rack_layout=layout,
    )


def grid_for_n_gpu(n_gpu_by_server: Dict[str, int]) -> Dict[str, dict]:
    out = {}
    for sk, n in n_gpu_by_server.items():
        out[sk] = asdict(report(sk, n))
    return out


def pue_sweep(p_it_kw: float, pue_values: List[float] = None) -> List[dict]:
    """供前端 PUE 滑块使用：固定 IT 负载下不同 PUE 的能耗。"""
    if pue_values is None:
        pue_values = [1.08, 1.10, 1.15, 1.20, 1.30, 1.40, 1.50]
    out = []
    for pue in pue_values:
        kwh = p_it_kw * pue * 8760
        out.append({
            "pue": pue,
            "annual_kwh": kwh,
            "annual_mwh": kwh / 1000,
            "annual_cny": kwh * C.POWER_PRICE_CNY_PER_KWH,
            "annual_co2_t": kwh * C.GRID_CO2_KG_PER_KWH / 1000,
        })
    return out


def _selftest() -> None:
    r = report("hgx_b200", 2048)
    print(f"[power] B200 2048GPU IT={r.p_it_kw:.0f} kW Total={r.p_total_kw:.0f} kW "
          f"年电费 {r.annual_cny/1e6:.1f} M¥  CO₂ {r.annual_co2_t:.0f} t  ✔")
    # 2048 GPU / 8 = 256 节点 × 14.3 kW = 3661 kW IT
    assert 3000 < r.p_it_kw < 4500
    assert r.pue == C.PUE_LIQUID


if __name__ == "__main__":
    import sys, json
    if "--selftest" in sys.argv:
        _selftest()
    else:
        json.dump({
            "report_b200_2048": asdict(report("hgx_b200", 2048)),
            "report_gb200_2304": asdict(report("gb200_nvl72", 2304)),  # 32 racks × 72
            "pue_sweep": pue_sweep(3700.0),
        }, sys.stdout, ensure_ascii=False, indent=2)
