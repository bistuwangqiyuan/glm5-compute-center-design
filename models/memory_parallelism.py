"""显存占用与 3D + EP + ZeRO 并行策略建模
==============================================

参考：
  * Megatron-LM v3 论文 (Korthikanti et al. 2023, arXiv:2205.05198)
  * DeepSpeed ZeRO 论文 (Rajbhandari et al. 2020, arXiv:1910.02054)
  * Mixtral / DeepSeek-V3 / Qwen3-MoE 工程实践

每 GPU 显存：
    M_total = M_weights + M_grad + M_optim + M_activations + M_kv (训练阶段无 KV)

混合精度训练每参数字节预算：
    M_param_bytes = 2 (BF16 weight) + 2 (BF16 grad) + 4+4+4+2 = 18 字节
    其中 Adam 维护 (FP32 master, m, v) = 12 B
    实际 Mixed-Precision 主流做法：weight FP32 master + 优化器 = 12 字节，加上 BF16 weight 2 + grad 2 + activation
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

from . import constants as C

BYTES_PER_PARAM_FULL = 18    # BF16 w + BF16 g + FP32(m,v,master) + buffer
BYTES_PER_PARAM_ZERO1 = 6    # weight + grad，optimizer state 切到 DP


@dataclass
class ParallelPlan:
    scenario_id: str
    gpu_key: str
    tp: int     # tensor parallel
    pp: int     # pipeline parallel
    ep: int     # expert parallel
    dp: int     # data parallel
    cp: int     # context (sequence) parallel
    mbs: int    # micro batch size
    seq_len: int
    zero_stage: int
    mem_per_gpu_gb: float
    feasible: bool
    note: str


def _activation_bytes(s, mbs: int, seq: int, alpha: float = 4.0) -> float:
    """每层每 batch 激活字节估算。alpha = 12 不重计算; ~4 选择性重计算."""
    H = s.hidden
    L = s.n_layers
    return mbs * seq * H * L * alpha * 2  # BF16 = 2 字节


def design_plan(scenario_id: str, gpu_key: str, target_n_gpu: int,
                seq_len: int = 8192, mbs: int = 1) -> ParallelPlan:
    """对给定场景给出一份 *可行* 的 3D+EP+ZeRO-1 配置。

    简化策略：
        TP = 8 （单节点 NVLink 内）
        EP = min(n_experts, 8 or 16)
        PP = 取决于 layer 数：尝试 8 或 16
        DP = n_gpu / (TP·PP·CP)
        CP = 取决于 ctx：默认 1，长上下文 (>32K) 用 2/4
    """
    s = C.SCENARIOS[scenario_id]
    g = C.GPUS[gpu_key]

    tp = 8
    cp = 1 if seq_len <= 32_768 else (2 if seq_len <= 131_072 else 4)
    # PP：按层数 88 → 11 stage 不整，取 8
    pp_candidates = [8, 11, 16, 22] if s.n_layers % 11 == 0 else [4, 8, 11, 16]
    pp = next((p for p in pp_candidates if s.n_layers % p == 0), 8)

    # EP：把 experts 分到 ep 张卡上
    ep = min(s.n_experts, 16) if gpu_key == "gb200" else min(s.n_experts, 8)

    base = tp * pp * cp
    if target_n_gpu < base:
        return ParallelPlan(
            scenario_id, gpu_key, tp, pp, ep, 1, cp, mbs, seq_len, 1,
            mem_per_gpu_gb=999.0, feasible=False,
            note=f"卡数 {target_n_gpu} < TP·PP·CP={base}，至少需要 {base} 张")
    dp = max(1, target_n_gpu // base)

    # 显存估算
    weight_bytes_per_gpu = (s.n_active * BYTES_PER_PARAM_ZERO1) / (tp * pp)
    optim_bytes_per_gpu = (s.n_active * (BYTES_PER_PARAM_FULL - BYTES_PER_PARAM_ZERO1)) / (tp * pp * dp)
    act_bytes = _activation_bytes(s, mbs, seq_len) / (tp * cp)
    # KV / 通信 buffer 留 5 GB
    other = 5 * 2**30
    total = weight_bytes_per_gpu + optim_bytes_per_gpu + act_bytes + other
    mem_gb = total / 1e9
    feasible = mem_gb < g.hbm_gb * 0.9   # 留 10 % 余量

    note_bits: List[str] = []
    if not feasible:
        note_bits.append(f"显存 {mem_gb:.1f} GB > {g.hbm_gb*0.9:.0f} GB，需启用 ZeRO-2/3 或重计算")
    if cp > 1:
        note_bits.append(f"CP={cp} 处理 {seq_len/1024:.0f}K 序列")
    if ep > 1:
        note_bits.append(f"EP={ep} 切分 {s.n_experts} experts")
    return ParallelPlan(
        scenario_id, gpu_key, tp, pp, ep, dp, cp, mbs, seq_len, 1,
        mem_per_gpu_gb=mem_gb, feasible=feasible,
        note="; ".join(note_bits) or "可行 (Feasible)")


def memory_breakdown(scenario_id: str, gpu_key: str, tp: int = 8, pp: int = 8,
                     dp: int = 16, cp: int = 1, seq: int = 8192, mbs: int = 1) -> dict:
    """供前端饼图展示：weight / grad / optim / activation / buffer 字节占比。"""
    s = C.SCENARIOS[scenario_id]
    g = C.GPUS[gpu_key]
    w = (s.n_active * 2) / (tp * pp)
    grad = (s.n_active * 2) / (tp * pp)
    optim = (s.n_active * 12) / (tp * pp * dp)   # FP32 m,v,master + buffer
    act = _activation_bytes(s, mbs, seq) / (tp * cp)
    buf = 5 * 2**30
    total = w + grad + optim + act + buf
    return {
        "scenario_id": scenario_id, "gpu_key": gpu_key,
        "tp": tp, "pp": pp, "dp": dp, "cp": cp, "seq": seq, "mbs": mbs,
        "bytes": {
            "weight_bf16": w, "grad_bf16": grad, "optim_fp32": optim,
            "activation": act, "buffer": buf,
        },
        "gb": {
            "weight_bf16": w / 1e9, "grad_bf16": grad / 1e9,
            "optim_fp32": optim / 1e9, "activation": act / 1e9, "buffer": buf / 1e9,
            "total": total / 1e9,
        },
        "hbm_gb": g.hbm_gb,
        "headroom_pct": max(0.0, (g.hbm_gb - total / 1e9) / g.hbm_gb * 100),
    }


def all_plans(target_n_gpu_by_gpu: Dict[str, int]) -> Dict[str, dict]:
    out = {}
    for gk, n in target_n_gpu_by_gpu.items():
        out[gk] = asdict(design_plan("baseline", gk, n))
    return out


def _selftest() -> None:
    p = design_plan("baseline", "b200", target_n_gpu=2048)
    print(f"[memory_parallelism] B200 2048 → TP{p.tp} PP{p.pp} EP{p.ep} DP{p.dp} "
          f"mem={p.mem_per_gpu_gb:.1f} GB  feasible={p.feasible}  ✔")
    assert p.tp == 8
    bd = memory_breakdown("baseline", "b200")
    assert bd["gb"]["total"] > 0
    print(f"[memory_parallelism] breakdown total={bd['gb']['total']:.1f} GB / HBM {bd['hbm_gb']} GB  ✔")


if __name__ == "__main__":
    import sys, json
    if "--selftest" in sys.argv:
        _selftest()
    else:
        json.dump({
            "plans_baseline_2048": all_plans({k: 2048 for k in C.GPUS}),
            "memory_breakdown_b200": memory_breakdown("baseline", "b200"),
        }, sys.stdout, ensure_ascii=False, indent=2)
