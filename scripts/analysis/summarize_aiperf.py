#!/usr/bin/env python3
"""summarize_aiperf.py POINT_DIR... : one row per AIPerf point (+ per-turn TTFT for agentic points).

Reads the raw AIPerf artifacts of a point directory (profile_export_aiperf.json, profile_export.jsonl and,
if present, server_metrics_export.json) and prints the same columns as data/points.json:
  req/s, output tok/s per GPU, TTFT and ITL p90 (p50 / p99), tok/s per user at p90 (1000 / ITL p90),
  SLO-ok %, goodput per GPU, prefix-cache hit %, KV peak, preemptions, waiting-queue p90.
Environment: NGPU (default 8), SLO_TTFT ms (default 2000), SLO_ITL ms (default 50), TURNS=0 to hide per-turn rows.
"""
import json, os, re, sys

NG = int(os.environ.get("NGPU", 8)); SLO_T = float(os.environ.get("SLO_TTFT", 2000)); SLO_I = float(os.environ.get("SLO_ITL", 50))


def pct(a, p):
    a = sorted(a); return a[min(len(a) - 1, int(round(p / 100 * (len(a) - 1))))] if a else float("nan")


def ser(m, k, stat="total"):
    v = m.get(k); return sum(s["stats"].get(stat, 0) for s in v["series"]) if v else None


print("point | c | req/s | out tok/s/GPU | TTFT p90 (p50/p99) ms | ITL p90 (p50/p99) ms | tok/s/user p90 | SLO-ok % | goodput/GPU | hit % | KV peak | preempt | wait p90")
for d in sys.argv[1:]:
    try:
        a = json.load(open(f"{d}/profile_export_aiperf.json"))
    except Exception as e:
        print(os.path.basename(d), "NO DATA", e); continue
    ok = n = 0; turns = {}
    for line in open(f"{d}/profile_export.jsonl"):
        r = json.loads(line); mm = r.get("metrics", {})
        if "time_to_first_token" not in mm:
            continue
        n += 1; t = mm["time_to_first_token"]["value"]; i = mm.get("inter_token_latency", {}).get("value", 0)
        ok += (t <= SLO_T and i <= SLO_I); turns.setdefault(r["metadata"].get("turn_index", 0), []).append(t)
    rps = a["request_throughput"]["avg"]; ot = a["output_token_throughput"]["avg"]
    g = lambda k, p: a[k][p] if k in a else float("nan")
    try:
        m = json.load(open(f"{d}/server_metrics_export.json"))["metrics"]
    except Exception:
        m = {}
    h = ser(m, "vllm:prefix_cache_hits"); q = ser(m, "vllm:prefix_cache_queries")
    kv = max([s["stats"]["max"] for s in m.get("vllm:kv_cache_usage_perc", {"series": []})["series"]] or [float("nan")])
    if h is None and "trtllm_kv_cache_reused_blocks" in m:  # TensorRT-LLM counts 64-token blocks, not tokens
        h = ser(m, "trtllm_kv_cache_reused_blocks"); q = h + ser(m, "trtllm_kv_cache_missed_blocks")
        kv = max([s["stats"]["max"] for s in m.get("trtllm_kv_cache_utilization", {"series": []})["series"]] or [float("nan")])
    pre = ser(m, "vllm:num_preemptions"); w = ser(m, "vllm:num_requests_waiting", "p90")
    c = int(re.search(r"-c(\d+)", os.path.basename(d)).group(1))
    print(f"{os.path.basename(d)} | {c} | {rps:.2f} | {ot / NG:.1f} | "
          f"{g('time_to_first_token', 'p90'):,.0f} ({g('time_to_first_token', 'p50'):,.0f}/{g('time_to_first_token', 'p99'):,.0f}) | "
          f"{g('inter_token_latency', 'p90'):.1f} ({g('inter_token_latency', 'p50'):.1f}/{g('inter_token_latency', 'p99'):.1f}) | "
          f"{1000 / g('inter_token_latency', 'p90'):.1f} | {100 * ok / max(n, 1):.1f} | {rps * ok / max(n, 1) / NG:.3f} | "
          f"{100 * h / q if h and q else float('nan'):.1f} | {kv:.2f} | {pre or 0:.0f} | {w if w is not None else float('nan'):.0f}")
    if len(turns) > 1 and os.environ.get("TURNS", "1") == "1":
        print("   per-turn TTFT p90 (p50):", " ".join(f"t{k}:{pct(v, 90):.0f}({pct(v, 50):.0f})" for k, v in sorted(turns.items())))
