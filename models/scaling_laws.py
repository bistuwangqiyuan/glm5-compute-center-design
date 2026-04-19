"""Scaling Law / 训练算力推导
============================

公式来源：
    Hoffmann et al. 2022,
    "Training Compute-Optimal Large Language Models" (a.k.a. Chinchilla)
    arXiv:2203.15556 §3 Eq. (1)–(3)

经典关系：
    C ≈ 6 · N · D                         (dense, FLOPs)
    D_optimal ≈ 20 · N                    (Chinchilla)

MoE 修正：
    C_MoE ≈ 6 · N_active · D
    D_MoE 推荐 ≈ 20 · N_total （ScalingMoE, DeepSeek-V2/V3 经验）

工程系数：训练时还有
    - MoE 路由开销           ≈ 5–10 %
    - 选择性激活重计算       ≈ +33 %
    - 通信尾延 / 数据加载   ≈ +15 %
合并约 1.33×，故引入 COMPUTE_OVERHEAD_FACTOR = 8/6。

每个函数都返回 *FLOPs* 浮点数；调用者自行换算到 PetaFLOP-day 等单位。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

from . import constants as C


@dataclass
class TrainComputeReport:
    scenario_id: str
    n_total: float
    n_active: float
    train_tokens: float
    flops_dense_naive: float        # 6·N·D（仅供对比）
    flops_moe_naive: float          # 6·N_act·D
    flops_engineering: float        # 含工程系数 8·N_act·D
    petaflop_day: float             # PetaFLOP-day = 8.64e19 FLOPs
    yotta_flops: float              # 1 YFLOP = 1e24


def chinchilla_optimal_tokens(n_params: float, ratio: float = 20.0) -> float:
    """Chinchilla 给出的最优 token 数（Hoffmann §3.2 Approach 3）"""
    return ratio * n_params


def train_flops(
    n_active: float,
    train_tokens: float,
    overhead_factor: float = C.COMPUTE_OVERHEAD_FACTOR,
) -> float:
    """训练总 FLOPs。

    overhead_factor = 8/6 ≈ 1.333，把 MoE 路由 / 重计算 / 通信尾延等都计入。
    """
    return 6.0 * n_active * train_tokens * overhead_factor


def report(scenario_id: str) -> TrainComputeReport:
    s = C.SCENARIOS[scenario_id]
    flops_dense = 6.0 * s.n_total * s.train_tokens
    flops_moe   = 6.0 * s.n_active * s.train_tokens
    flops_eng   = train_flops(s.n_active, s.train_tokens)
    pf_day      = flops_eng / 8.64e19
    yflop       = flops_eng / 1.0e24
    return TrainComputeReport(
        scenario_id=scenario_id,
        n_total=s.n_total,
        n_active=s.n_active,
        train_tokens=s.train_tokens,
        flops_dense_naive=flops_dense,
        flops_moe_naive=flops_moe,
        flops_engineering=flops_eng,
        petaflop_day=pf_day,
        yotta_flops=yflop,
    )


def all_reports() -> Dict[str, dict]:
    return {sid: asdict(report(sid)) for sid in C.SCENARIOS}


def _selftest() -> None:
    r = report("baseline")
    # baseline: N_act = 6.4e10, D = 2.0e13
    # 6 * 6.4e10 * 2.0e13 * (8/6) = 8 * 6.4e10 * 2.0e13 = 1.024e25
    expected = 8.0 * 6.4e10 * 2.0e13
    assert abs(r.flops_engineering - expected) / expected < 1e-9, r
    assert r.flops_engineering > 1.0e25 * 0.9
    assert r.flops_engineering < 1.1e25
    print(f"[scaling_laws] baseline FLOPs = {r.flops_engineering:.3e} = "
          f"{r.petaflop_day:,.0f} PetaFLOP-day  [OK]")


if __name__ == "__main__":
    import sys, json
    if "--selftest" in sys.argv:
        _selftest()
    else:
        json.dump(all_reports(), sys.stdout, ensure_ascii=False, indent=2)
