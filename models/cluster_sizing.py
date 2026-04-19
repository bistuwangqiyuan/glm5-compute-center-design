"""GPU 集群规模与训练时长建模
==================================

核心公式（design.md §3.3）：

    GPU_seconds = C_eff / (TFLOPS_peak · MFU)
    GPU_hours   = GPU_seconds / 3600
    days        = GPU_seconds / (n_gpu · 86400)

我们对每个场景 × 每种 GPU，给出 *若希望 X 天完成训练，需要多少 GPU* 以及
*若手头已有 N 张卡，需要多少天*。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List

from . import constants as C
from . import scaling_laws as SL


@dataclass
class ClusterPoint:
    scenario_id: str
    gpu_key: str
    n_gpu: int
    days: float
    gpu_hours: float
    mfu: float
    sustained_pflops: float        # 集群可持续算力


def gpus_for_target_days(scenario_id: str, gpu_key: str, target_days: float = 90.0) -> ClusterPoint:
    """给定希望训练时长，反算需要多少张卡（向上取整到 8 的倍数）。"""
    s = C.SCENARIOS[scenario_id]
    g = C.GPUS[gpu_key]
    mfu = C.MFU_BY_GPU[gpu_key]
    flops = SL.train_flops(s.n_active, s.train_tokens)
    gpu_seconds = flops / (g.bf16_tflops * 1e12 * mfu)
    n_gpu_raw = gpu_seconds / (target_days * 86400)
    # 向上取整到 8 倍数（一台 HGX 节点 8 卡）
    n_gpu = int((n_gpu_raw + 7) // 8 * 8)
    if gpu_key == "gb200":
        # GB200 NVL72：向上取整到 72 的倍数
        n_gpu = int((n_gpu_raw + 71) // 72 * 72)
    days = gpu_seconds / (n_gpu * 86400)
    sustained = n_gpu * g.bf16_tflops * mfu / 1e3   # PFLOPS
    return ClusterPoint(
        scenario_id=scenario_id, gpu_key=gpu_key,
        n_gpu=n_gpu, days=days, gpu_hours=gpu_seconds / 3600,
        mfu=mfu, sustained_pflops=sustained,
    )


def days_for_n_gpu(scenario_id: str, gpu_key: str, n_gpu: int) -> ClusterPoint:
    s = C.SCENARIOS[scenario_id]
    g = C.GPUS[gpu_key]
    mfu = C.MFU_BY_GPU[gpu_key]
    flops = SL.train_flops(s.n_active, s.train_tokens)
    gpu_seconds = flops / (g.bf16_tflops * 1e12 * mfu)
    days = gpu_seconds / (n_gpu * 86400)
    sustained = n_gpu * g.bf16_tflops * mfu / 1e3
    return ClusterPoint(
        scenario_id=scenario_id, gpu_key=gpu_key,
        n_gpu=n_gpu, days=days, gpu_hours=gpu_seconds / 3600,
        mfu=mfu, sustained_pflops=sustained,
    )


def comparison_grid(target_days: float = 90.0) -> Dict[str, dict]:
    """对每个 scenario × 每个 GPU 给出"目标 X 天"的 GPU 需求矩阵。"""
    grid = {}
    for sid in C.SCENARIOS:
        grid[sid] = {}
        for gk in C.GPUS:
            grid[sid][gk] = asdict(gpus_for_target_days(sid, gk, target_days))
    return grid


def fixed_gpu_grid(n_gpus: List[int]) -> Dict[str, dict]:
    """对 baseline scenario，给固定 GPU 数量 → 训练天数。供前端滑块使用。"""
    out = {}
    for gk in C.GPUS:
        out[gk] = []
        for n in n_gpus:
            cp = days_for_n_gpu("baseline", gk, n)
            out[gk].append({"n_gpu": n, "days": cp.days, "sustained_pflops": cp.sustained_pflops})
    return out


def _selftest() -> None:
    cp = gpus_for_target_days("baseline", "b200", target_days=90)
    # baseline FLOPs ~ 1.024e25
    # 1.024e25 / (2250e12 * 0.42 * 90 * 86400) = 1.024e25 / 7.35e21 ≈ 1394
    print(f"[cluster_sizing] B200 90d → n_gpu={cp.n_gpu}, sustained={cp.sustained_pflops:.1f} PFLOPS  [ok]")
    assert 1000 <= cp.n_gpu <= 2500
    cp_h = gpus_for_target_days("baseline", "h100", target_days=90)
    print(f"[cluster_sizing] H100 90d → n_gpu={cp_h.n_gpu}  [ok]")
    assert cp_h.n_gpu > cp.n_gpu * 2  # H100 慢得多


if __name__ == "__main__":
    import sys, json
    if "--selftest" in sys.argv:
        _selftest()
    else:
        json.dump(comparison_grid(), sys.stdout, ensure_ascii=False, indent=2)
