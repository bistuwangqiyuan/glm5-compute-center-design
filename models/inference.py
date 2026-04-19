"""推理吞吐 / KV-Cache / 并发建模
==================================

参考：
  * Kwon et al. 2023 "Efficient Memory Management for LLM Serving with PagedAttention" (vLLM)
  * NVIDIA TensorRT-LLM Performance Tuning Guide
  * Pope et al. 2023 "Efficiently Scaling Transformer Inference"

关键量：
  - Prefill 阶段算力主导   ：~ 2 · N_act · L_in   FLOPs / request
  - Decode 阶段带宽主导   ：~ 2 · N_act           FLOPs / token
  - KV-Cache 字节        ：2 · L · H_kv · L_seq · 2 (BF16)，其中 H_kv = n_kv_heads · head_dim

并发上限：
  N_concurrent = (M_avail) / KV_per_request_full
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List

from . import constants as C


@dataclass
class InferenceReport:
    gpu_key: str
    n_gpu_serving: int            # 模型并行度 (TP)
    kv_per_token_bytes: int
    kv_per_request_bytes: float
    max_concurrent: int
    decode_throughput_tok_s: float
    prefill_latency_ms_8k: float


def kv_per_token_bytes(scenario_id: str = "baseline", n_kv_heads_ratio: int = 8) -> int:
    """GQA: n_kv_heads = n_heads / ratio"""
    s = C.SCENARIOS[scenario_id]
    n_kv_heads = max(1, s.n_heads // n_kv_heads_ratio)
    return 2 * s.n_layers * n_kv_heads * s.head_dim * 2   # K + V, BF16


def report(gpu_key: str, scenario_id: str = "baseline",
           tp: int = 8, seq_len: int = 8192,
           hbm_reserve_gb: float = 20.0) -> InferenceReport:
    s = C.SCENARIOS[scenario_id]
    g = C.GPUS[gpu_key]

    weight_bytes = s.n_active * 2     # BF16 weight (decode 实际加载激活专家)
    weight_per_gpu = weight_bytes / tp
    kv_per_tok = kv_per_token_bytes(scenario_id)
    kv_per_req = kv_per_tok * seq_len

    avail = g.hbm_gb * 1e9 * tp - weight_bytes - hbm_reserve_gb * 1e9 * tp
    n_conc = max(0, int(avail // kv_per_req))

    # Decode：受 HBM BW 限制，吞吐 ≈ HBM_BW × tp / weight_bytes
    decode_tok_s = (g.hbm_bw_tbs * 1e12 * tp) / weight_bytes

    # Prefill：受算力限制，时长 ≈ (2·N_act·L_in) / (TFLOPS·MFU·tp)
    mfu_inf = 0.55
    prefill_latency = (2 * s.n_active * 8192) / (g.bf16_tflops * 1e12 * mfu_inf * tp) * 1000

    return InferenceReport(
        gpu_key=gpu_key, n_gpu_serving=tp,
        kv_per_token_bytes=kv_per_tok,
        kv_per_request_bytes=kv_per_req,
        max_concurrent=n_conc,
        decode_throughput_tok_s=decode_tok_s,
        prefill_latency_ms_8k=prefill_latency,
    )


def comparison(seq_len: int = 8192) -> Dict[str, dict]:
    out = {}
    for gk in C.GPUS:
        out[gk] = asdict(report(gk, seq_len=seq_len))
    return out


def cluster_qps(gpu_key: str, n_gpu_total: int, seq_len: int = 8192,
                tp_per_replica: int = 8) -> dict:
    r = report(gpu_key, seq_len=seq_len, tp=tp_per_replica)
    n_replica = n_gpu_total // tp_per_replica
    total_concurrent = r.max_concurrent * n_replica
    qps = total_concurrent * r.decode_throughput_tok_s / 256  # 假设 256 token/请求
    return {
        "gpu_key": gpu_key, "n_gpu_total": n_gpu_total,
        "n_replica": n_replica, "tp_per_replica": tp_per_replica,
        "total_concurrent": total_concurrent,
        "decode_tok_s_per_replica": r.decode_throughput_tok_s,
        "qps_estimate_256tok": qps,
    }


def _selftest() -> None:
    r = report("h200", seq_len=8192)
    print(f"[inference] H200 TP8 8K-seq concurrent={r.max_concurrent} "
          f"decode={r.decode_throughput_tok_s:.0f} tok/s prefill={r.prefill_latency_ms_8k:.1f} ms  ✔")
    assert r.max_concurrent > 0
    c = cluster_qps("b200", 256)
    print(f"[inference] B200 256GPU cluster QPS≈{c['qps_estimate_256tok']:.0f}  ✔")


if __name__ == "__main__":
    import sys, json
    if "--selftest" in sys.argv:
        _selftest()
    else:
        json.dump({
            "comparison": comparison(),
            "cluster_b200_256": cluster_qps("b200", 256),
        }, sys.stdout, ensure_ascii=False, indent=2)
