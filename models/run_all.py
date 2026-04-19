"""主调度脚本：运行所有计算模型 → 写出 web/data/computed.json
================================================================

用法：
    python -m models.run_all                # 生成 JSON
    python -m models.run_all --check        # 只跑 selftest
    python -m models.run_all --pretty       # 美化输出
"""
from __future__ import annotations

import json
import os
import sys
import datetime as dt
from dataclasses import asdict
from pathlib import Path

from . import __version__
from . import constants as C
from . import scaling_laws as SL
from . import cluster_sizing as CS
from . import memory_parallelism as MP
from . import network_bandwidth as NW
from . import storage_io as ST
from . import power_cooling as PC
from . import tco as TCO
from . import inference as INF


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "web" / "data" / "computed.json"


def build_payload() -> dict:
    # 1) Scaling Law / 算力 ----------------------------------------------------
    scaling = SL.all_reports()

    # 2) 集群规模：每场景 × 每 GPU，目标 90 天 ---------------------------------
    cluster_90 = CS.comparison_grid(target_days=90.0)
    cluster_60 = CS.comparison_grid(target_days=60.0)
    cluster_120 = CS.comparison_grid(target_days=120.0)

    # 滑块用：固定 GPU 数 → 天数
    fixed = CS.fixed_gpu_grid([512, 1024, 2048, 4096, 8192, 16384])

    # 3) 显存 / 并行 -----------------------------------------------------------
    plans = MP.all_plans({k: cluster_90["baseline"][k]["n_gpu"] for k in C.GPUS})
    mem_breakdown = {
        gk: MP.memory_breakdown("baseline", gk) for gk in C.GPUS
    }

    # 4) 网络 ------------------------------------------------------------------
    n_baseline = cluster_90["baseline"]["b200"]["n_gpu"]
    network = {
        "comm": NW.comparison_grid(n_baseline),
        "topology": {
            nic: NW.topology_estimate(n_baseline, nic) for nic in C.NICS
        },
    }

    # 5) 存储 ------------------------------------------------------------------
    storage = {
        "comparison": ST.comparison(),
        "tiered_baseline": ST.tiered_design("baseline"),
    }

    # —— 选定主集群规模：B200 = 2048 (256 节点 HGX，整数美观)，GB200 = 2304 (32 NVL72) ——
    SELECTED_N_B200 = 2048
    SELECTED_N_GB200 = 2304

    # 把"选定方案"对应的训练天数写回 cluster.selected_plan
    sel_b200_cp = CS.days_for_n_gpu("baseline", "b200", SELECTED_N_B200)
    sel_gb200_cp = CS.days_for_n_gpu("baseline", "gb200", SELECTED_N_GB200)

    # 6) 电力 / 制冷 ----------------------------------------------------------
    power = {
        "report_b200": asdict(PC.report("hgx_b200", SELECTED_N_B200)),
        "report_gb200": asdict(PC.report("gb200_nvl72", SELECTED_N_GB200)),
        "report_h200": asdict(PC.report("hgx_h200", cluster_90["baseline"]["h200"]["n_gpu"])),
        "report_h100": asdict(PC.report("hgx_h100", cluster_90["baseline"]["h100"]["n_gpu"])),
        "pue_sweep": PC.pue_sweep(PC.report("hgx_b200", SELECTED_N_B200).p_it_kw),
    }

    # 7) TCO -------------------------------------------------------------------
    tco = {
        "comparison_at_baseline": TCO.comparison(SELECTED_N_B200),
        "selected_plan_b200": asdict(TCO.report("hgx_b200", "ib_xdr", SELECTED_N_B200)),
        "selected_plan_gb200": asdict(TCO.report("gb200_nvl72", "ib_xdr", SELECTED_N_GB200)),
    }

    # 同步把 selected 写到 cluster 字段，前端直接读
    cluster_90["baseline"]["b200_selected"] = asdict(sel_b200_cp)
    cluster_90["baseline"]["gb200_selected"] = asdict(sel_gb200_cp)

    # 8) 推理 ------------------------------------------------------------------
    inference = {
        "comparison_8k": INF.comparison(seq_len=8192),
        "comparison_32k": INF.comparison(seq_len=32_768),
        "cluster_b200_256": INF.cluster_qps("b200", 256),
        "cluster_gb200_144": INF.cluster_qps("gb200", 144, tp_per_replica=8),
    }

    # ---- 汇总 --------------------------------------------------------------
    payload = {
        "schema_version": "1.0",
        "generator_version": __version__,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "constants": C.all_constants_as_dict(),
        "scaling": scaling,
        "cluster": {
            "target_60d": cluster_60,
            "target_90d": cluster_90,
            "target_120d": cluster_120,
            "fixed_gpu_baseline": fixed,
        },
        "memory": {"plans": plans, "breakdown": mem_breakdown},
        "network": network,
        "storage": storage,
        "power": power,
        "tco": tco,
        "inference": inference,
    }
    return payload


def write_json(payload: dict, pretty: bool = True) -> Path:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        if pretty:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        else:
            json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    return OUTPUT


def selftest_all() -> None:
    print("=== run_all selftest ===")
    SL._selftest()
    CS._selftest()
    MP._selftest()
    NW._selftest()
    ST._selftest()
    PC._selftest()
    TCO._selftest()
    INF._selftest()
    print("ALL OK ✔")


def main() -> None:
    if "--check" in sys.argv:
        selftest_all()
        return
    pretty = "--pretty" in sys.argv or True
    payload = build_payload()
    path = write_json(payload, pretty=pretty)
    size = path.stat().st_size
    print(f"✔ 写入 {path.relative_to(REPO_ROOT)}  ({size/1024:.1f} KB)")
    print(f"  schema_version = {payload['schema_version']}")
    print(f"  generated_at   = {payload['generated_at']}")


if __name__ == "__main__":
    main()
