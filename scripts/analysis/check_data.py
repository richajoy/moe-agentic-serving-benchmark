#!/usr/bin/env python3
"""Recompute every published number from data/ and print the tables the docs use.

  python3 scripts/analysis/check_data.py            # verify all points, print the summary table
  python3 scripts/analysis/check_data.py --frontier # also print the operating frontier

Checks, for each of the 37 points in data/points.json:
  - SLO-ok % and goodput recomputed from records.jsonl.gz (TTFT <= 2000 ms AND per-request mean ITL <= 50 ms)
  - a point meets the target when SLO-ok >= 90% (DistServe convention); the last column prints yes/no
  - req/s, TTFT and ITL percentiles equal to summary.json (AIPerf's own aggregates)
Exits 1 on any mismatch. Stdlib only.
"""
import gzip, json, os, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
NGPU, SLO_TTFT_MS, SLO_ITL_MS, POINT_OK_PCT = 8, 2000.0, 50.0, 90.0
# TTFT on this point is not a valid measurement: the load generator repeated prompts, so the prefix cache served
# almost every prompt token. It cannot be judged against a TTFT target.
TTFT_INVALID = {"vllm-tp8-matrix/S1-c64"}


def meets(p, slo_ok_pct):
    if f"{p['run']}/{p['point']}" in TTFT_INVALID:
        return "n/a (TTFT invalid)"
    return "yes" if slo_ok_pct >= POINT_OK_PCT else "no"


def load_points():
    return json.load(open(os.path.join(ROOT, "points.json")))["points"]


def recompute(p):
    d = os.path.join(ROOT, p["run"], p["point"])
    recs = [json.loads(l) for l in gzip.open(os.path.join(d, "records.jsonl.gz"), "rt")]
    ok_recs = [r for r in recs if r["ttft_ms"] is not None]
    slo = sum(r["ttft_ms"] <= SLO_TTFT_MS and (r["itl_ms"] or 0) <= SLO_ITL_MS for r in ok_recs)
    s = json.load(open(os.path.join(d, "summary.json")))
    rps = s["request_throughput"]["avg"]
    return {
        "records": len(recs),
        "slo_ok_pct": 100.0 * slo / len(ok_recs),
        "goodput": rps * slo / len(ok_recs) / NGPU,
        "req_s": rps,
        "ttft_p50": s["time_to_first_token"]["p50"], "ttft_p90": s["time_to_first_token"]["p90"],
        "ttft_p99": s["time_to_first_token"]["p99"],
        "itl_p50": s["inter_token_latency"]["p50"], "itl_p90": s["inter_token_latency"]["p90"],
        "itl_p99": s["inter_token_latency"]["p99"], "itl_mean": s["inter_token_latency"]["avg"],
        "out_tok_s": s["output_token_throughput"]["avg"], "total_tok_s": s["total_token_throughput"]["avg"],
        "eff_c": s["effective_concurrency"]["avg"], "e2e_p50_s": s["request_latency"]["p50"] / 1000,
    }


def close(a, b, rel=1e-6):
    return a is not None and b is not None and abs(a - b) <= rel * max(1.0, abs(a), abs(b))


def main():
    pts = load_points(); bad = 0
    hdr = ("point", "layout", "KV", "c", "eff.c", "req/s", "out tok/s/GPU", "total tok/s/GPU",
           "TTFT p90 (p50/p99) ms", "ITL p90 (p50/p99) ms", "tok/s/user p90 (p50)", "SLO-ok %", "goodput", "turns ran/planned", "meets target")
    print(" | ".join(hdr))
    for p in pts:
        r = recompute(p)
        checks = [("slo_ok_pct", r["slo_ok_pct"], p["slo_ok_pct"]), ("goodput", r["goodput"], p["goodput_req_s_gpu"]),
                  ("req_s", r["req_s"], p["req_s"]), ("ttft_p50", r["ttft_p50"], p["ttft_p50_ms"]),
                  ("ttft_p99", r["ttft_p99"], p["ttft_p99_ms"]), ("itl_p50", r["itl_p50"], p["itl_p50_ms"]),
                  ("ttft_p90", r["ttft_p90"], p["ttft_p90_ms"]), ("itl_p90", r["itl_p90"], p["itl_p90_ms"]),
                  ("itl_p99", r["itl_p99"], p["itl_p99_ms"]), ("tok_s_user_p90", round(1000 / r["itl_p90"], 3), p["tok_s_user_p90"]),
                  ("records", r["records"], p["records"])]
        for name, got, want in checks:
            if not close(got, want):
                bad += 1; print(f"MISMATCH {p['run']}/{p['point']} {name}: data {got} vs points.json {want}")
        print(" | ".join(str(x) for x in (
            f"{p['run']}/{p['point']}", p["layout"], p["kv_dtype"], p["concurrency"], f"{r['eff_c']:.1f}",
            f"{r['req_s']:.2f}", f"{r['out_tok_s']/NGPU:.1f}", f"{r['total_tok_s']/NGPU:,.0f}",
            f"{r['ttft_p90']:,.0f} ({r['ttft_p50']:,.0f}/{r['ttft_p99']:,.0f})", f"{r['itl_p90']:.1f} ({r['itl_p50']:.1f}/{r['itl_p99']:.1f})",
            f"{1000/r['itl_p90']:.1f} ({1000/r['itl_p50']:.1f})", f"{r['slo_ok_pct']:.1f}", f"{r['goodput']:.3f}",
            f"{p['records']}/{p['turns_planned_total']}" if p.get("turns_planned_total") else str(p["records"]),
            meets(p, r["slo_ok_pct"]))))
    if "--frontier" in sys.argv:
        ag = [p for p in pts if p["point"].endswith("-t8")]
        front = [p for p in ag if not any(
            q is not p and q["out_tok_s_gpu"] >= p["out_tok_s_gpu"] and q["tok_s_user_p90"] >= p["tok_s_user_p90"]
            and (q["out_tok_s_gpu"] > p["out_tok_s_gpu"] or q["tok_s_user_p90"] > p["tok_s_user_p90"]) for q in ag)]
        print("\nOperating frontier (8-turn agentic points; output tok/s/GPU vs tok/s per user at p90 = 1000 / ITL p90):")
        for p in sorted(front, key=lambda p: -p["tok_s_user_p90"]):
            print(f"  {p['run']}/{p['point']}: {p['tok_s_user_p90']:.1f} tok/s/user (p90), {p['out_tok_s_gpu']:.1f} out tok/s/GPU, "
                  f"TTFT p90 {p['ttft_p90_ms']:,.0f} ms, ITL p90 {p['itl_p90_ms']:.1f} ms, SLO-ok {p['slo_ok_pct']:.1f}%, "
                  f"meets target (>= {POINT_OK_PCT:.0f}%): {meets(p, p['slo_ok_pct'])}")
    print(f"\n{len(pts)} points checked, {bad} mismatches")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
