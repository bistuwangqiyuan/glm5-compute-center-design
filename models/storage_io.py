"""存储 I/O / Checkpoint 建模
================================

讨论两类压力：
  (1) 数据集吞吐：训练 D tokens × 4 字节/token (uint32 ID) → 总数据量
  (2) Checkpoint：每 N 步存盘，需在 X 分钟内写完，避免阻塞训练

参考：
  * MLPerf Storage v0.5 / v1.0
  * NVIDIA Magnum IO Reference Architecture
  * WekaFS / Lustre / VAST 公开案例
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

from . import constants as C


@dataclass
class StorageReport:
    storage_key: str
    dataset_pb: float
    dataset_load_hours: float       # 全量加载到 Tier-0 时间
    checkpoint_gb: float
    checkpoint_write_s: float       # 单次 ckpt 写入秒数
    ckpt_overhead_pct: float        # 假设每 4 小时一次 ckpt → 占训练时间百分比


def dataset_size_bytes(scenario_id: str, bytes_per_token: int = 4) -> float:
    """*已分词* 数据集大小 (token 数 × 字节数, 通常 uint32 = 4 字节)。

    例如 baseline = 20T tokens × 4 B = 80 TB tokenized binary。
    注意：这与原始抓取语料 (raw corpus) 不同——raw 通常是 10–100× (因 dedup/filter)。
    """
    s = C.SCENARIOS[scenario_id]
    return s.train_tokens * bytes_per_token


def raw_corpus_bytes(scenario_id: str, dedup_ratio: float = 25.0) -> float:
    """估算"原始抓取语料"规模：tokenized × dedup_ratio。
    Common Crawl 100+ PB 抓取后，经过去重 + 质量过滤 ~ 5 % 保留进入训练。
    所以 raw / tokenized ≈ 20–30。
    """
    return dataset_size_bytes(scenario_id) * dedup_ratio


def checkpoint_size_bytes(n_total: float) -> float:
    """完整 checkpoint = weights (FP32 master) + optimizer states ≈ 16 字节/参数。"""
    return n_total * 16


def report(storage_key: str, scenario_id: str = "baseline",
           ckpt_interval_min: int = 240) -> StorageReport:
    s = C.SCENARIOS[scenario_id]
    st = C.STORAGE[storage_key]
    ds_bytes = dataset_size_bytes(scenario_id)
    ds_pb = ds_bytes / 1e15
    load_hours = ds_bytes / (st.read_gbs * 1e9 * 3600)
    ckpt_bytes = checkpoint_size_bytes(s.n_total)
    ckpt_write_s = ckpt_bytes / (st.write_gbs * 1e9)
    ckpt_overhead = (ckpt_write_s / (ckpt_interval_min * 60)) * 100
    return StorageReport(
        storage_key=storage_key,
        dataset_pb=ds_pb,
        dataset_load_hours=load_hours,
        checkpoint_gb=ckpt_bytes / 1e9,
        checkpoint_write_s=ckpt_write_s,
        ckpt_overhead_pct=ckpt_overhead,
    )


def comparison() -> Dict[str, dict]:
    out = {}
    for sk in C.STORAGE:
        out[sk] = asdict(report(sk))
    return out


def tiered_design(scenario_id: str = "baseline") -> dict:
    """三层存储架构推荐容量（参照 Meta / xAI / DeepMind 公开容量比例）。"""
    s = C.SCENARIOS[scenario_id]
    ds_tb = dataset_size_bytes(scenario_id) / 1e12        # tokenized, ~80 TB
    raw_pb = raw_corpus_bytes(scenario_id) / 1e15         # 原始抓取，~2 PB
    ckpt_each_tb = checkpoint_size_bytes(s.n_total) / 1e12
    ckpt_keep = 20
    return {
        "tokenized_dataset_tb": ds_tb,
        "raw_corpus_pb": raw_pb,
        "checkpoint_each_tb": ckpt_each_tb,
        "checkpoint_keep_count": ckpt_keep,
        "checkpoint_total_pb": ckpt_each_tb * ckpt_keep / 1000,
        # 三层规划（PB）
        "tier0_hot_pb": 5.0,                           # NVMe 全闪热数据 (混 batch + 实时 ckpt)
        "tier1_warm_pb": max(20.0, raw_pb * 5),        # Lustre 温存档（多版本数据 + 全部 ckpt）
        "tier2_cold_pb": max(100.0, raw_pb * 20),      # 对象存储 / QLC 冷归档（多模型快照 + raw 抓取）
    }


def _selftest() -> None:
    r = report("weka_nvme")
    print(f"[storage] tokenized dataset {r.dataset_pb*1000:.0f} TB; "
          f"ckpt {r.checkpoint_gb/1024:.1f} TB / {r.checkpoint_write_s:.0f}s; "
          f"overhead {r.ckpt_overhead_pct:.3f}%  ✔")
    assert r.dataset_pb > 0.05      # baseline 20T tokens × 4 B = 80 TB ≈ 0.08 PB
    t = tiered_design("baseline")
    print(f"[storage] tiered: T0={t['tier0_hot_pb']} T1={t['tier1_warm_pb']:.0f} T2={t['tier2_cold_pb']:.0f} PB  ✔")
    assert t["tier0_hot_pb"] >= 1.0
    assert t["tier1_warm_pb"] >= 10.0


if __name__ == "__main__":
    import sys, json
    if "--selftest" in sys.argv:
        _selftest()
    else:
        json.dump({"comparison": comparison(), "tiered": tiered_design()},
                  sys.stdout, ensure_ascii=False, indent=2)
