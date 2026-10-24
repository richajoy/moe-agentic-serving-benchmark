#!/usr/bin/env python3
"""Static figures f01..f09, regenerated from data/ (stdlib only).

  python3 scripts/figures/make_figures.py      # writes figures/f0*.svg

Headline percentile is p90 everywhere: TTFT p90, ITL p90, and tok/s per user = 1000 / ITL p90
(the slow tail, the same way InferenceX computes P90 interactivity). Where room allows, a thin bar
spans p50 to p99 around the p90 marker. A hollow marker means the run stopped at its duration cap
with sessions unfinished, so its latency is a lower bound.
"""
import json
from svglib import (CONDITIONS, DATA, Fig, L, LinScale, LogScale, P, R, T, C, fmt_ms, lower_bound, marker, pct, records)

SERIES = [  # label, color, shape, points
    ("TP8, BF16 KV", 1, "o", ["vllm-tp8/AG-c16-t4", "vllm-tp8/AG-c64-t8", "vllm-tp8/AG-c256-t8",
                               "vllm-tp8-matrix/AG-c512-t8", "vllm-tp8-matrix/AG-c1000-t8"]),
    ("TP8+EP8, BF16 KV", 2, "s", ["vllm-tp8ep8/AG-c64-t8", "vllm-tp8ep8/AG-c256-t8"]),
    ("TP8+EP8, FP8 KV", 3, "^", ["vllm-tp8ep8-fp8kv/AG-c256-t8", "vllm-tp8ep8-fp8kv/AG-c1000-t8"]),
]
CONC = [16, 64, 256, 512, 1000]
SRC_LADDER = ["data/points.json rows: " + ", ".join(k for _, _, _, ks in SERIES for k in ks)]


def legend(f, x, y, items, hollow_note=True):
    for i, (lab, col, shp) in enumerate(items):
        f.add(marker(shp, x + i * 190, y - 4, col, size=5), T(x + 12 + i * 190, y, lab, "ink s"))
    if hollow_note:
        f.add(marker("o", x + len(items) * 190, y - 4, 1, hollow=True, size=5),
              T(x + 12 + len(items) * 190, y, "hollow = stopped at time cap (lower bound)", "mut s"))


def ladder_panel(f, x0, y0, w, h, metric, title, lo, hi, slo, slo_label, ticks, tick_fmt):
    xs = LogScale(12, 1300, x0 + 10, x0 + w - 10)
    ys = LogScale(lo, hi, y0 + h, y0)
    f.add(T(x0, y0 - 14, title, "ink h"))
    for t in ticks:
        f.add(L(x0, ys(t), x0 + w, ys(t), "grid"), T(x0 - 6, ys(t) + 4, tick_fmt(t), "mut s", "end"))
    for c in CONC:
        f.add(L(xs(c), y0 + h, xs(c), y0 + h + 4), T(xs(c), y0 + h + 18, f"{c:,}", "mut s", "middle"))
    f.add(L(x0, y0 + h, x0 + w, y0 + h))
    f.add(R(x0, ys(slo), w, y0 + h - ys(slo), "ok", 0))
    f.add(L(x0, ys(slo), x0 + w, ys(slo), "sr", 1.4, 'stroke-dasharray="5 4"'),
          T(x0 + w - 4, ys(slo) - 6, slo_label, "cr s", "end"))
    off = {1: -9, 2: 0, 3: 9}
    for lab, col, shp, keys in SERIES:
        pts = []
        for k in keys:
            p = P(k); cx = xs(p["concurrency"]) + off[col]
            v50, v90, v99 = (p[f"{metric}_p50_ms"], p[f"{metric}_p90_ms"], p[f"{metric}_p99_ms"])
            if metric == "ttft":
                v50, v90, v99 = v50 / 1000, v90 / 1000, v99 / 1000
            f.add(L(cx, ys(max(v50, lo)), cx, ys(min(v99, hi)), f"s{col}", 2.2, 'stroke-opacity=".45"'))
            pts.append((cx, ys(v90), lower_bound(p)))
        f.add('<polyline fill="none" stroke-width="1.6" class="s%d" points="%s"/>' % (col, " ".join(f"{a:.1f},{b:.1f}" for a, b, _ in pts)))
        for cx, cy, lb in pts:
            f.add(marker(shp, cx, cy, col, hollow=lb, size=5.5))
    f.add(T(x0 + w / 2, y0 + h + 36, "concurrent agentic sessions (log scale)", "mut s", "middle"))


def f01():
    f = Fig(960, 470, "Agentic TTFT p90 and ITL p90 as concurrent sessions grow, three layouts",
            "Two log-scale panels. TTFT p90 in seconds and ITL p90 in milliseconds versus 16 to 1,000 concurrent agentic sessions on one "
            "8x H100 node. Markers are p90; thin bars span p50 to p99. At 64 sessions every layout is inside the SLO "
            "(TP8+EP8 BF16 KV: TTFT p90 0.85 s, ITL p90 25.6 ms). At 256 sessions every layout misses it at p90; FP8 KV comes closest "
            "(TTFT p90 4.1 s, ITL p90 56.4 ms). At 1,000 sessions TTFT p90 is 72.5 s or more and ITL p90 265 ms or more, and both runs are lower bounds.",
            SRC_LADDER)
    f.add(T(24, 32, "Latency vs load: the SLO holds at 64 sessions and breaks by 256", "ink t"))
    f.add(T(24, 52, "Agentic sessions (2,000-token shared prompt, +600 in / 200 out per turn, 8 ± 2 turns). Marker = p90, bar = p50 to p99.", "mut"))
    ladder_panel(f, 80, 100, 360, 260, "ttft", "TTFT p90 (s)", 0.15, 200, 2.0, "SLO 2 s",
                 [0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200], lambda t: f"{t:g}")
    ladder_panel(f, 560, 100, 360, 260, "itl", "ITL p90 (ms)", 12, 400, 50.0, "SLO 50 ms",
                 [15, 20, 30, 50, 100, 200, 400], lambda t: f"{t:g}")
    legend(f, 80, 430, [(l, c, s) for l, c, s, _ in SERIES])
    f.add(T(24, 458, "16 sessions used 4-turn sessions. " + CONDITIONS + ".", "faint xs"))
    f.write("f01-latency-vs-sessions.svg")


def f02():
    f = Fig(960, 450, "Output throughput per GPU and goodput per GPU versus concurrent agentic sessions",
            "Two panels. Left: output tokens per second per GPU. Right: goodput, requests per second per GPU that met TTFT 2 s and ITL 50 ms. "
            "FP8 KV gives the most throughput (401.9 output tok/s/GPU at 256 sessions) but goodput peaks at 64 sessions "
            "(1.218 req/s/GPU on TP8+EP8 BF16 KV) and is zero at 1,000 sessions.",
            SRC_LADDER)
    f.add(T(24, 32, "More sessions buy throughput only until KV fills; goodput peaks at 64", "ink t"))
    f.add(T(24, 52, "Same points as Figure 1. Goodput counts only requests that met TTFT ≤ 2 s and ITL ≤ 50 ms, per GPU.", "mut"))
    for (x0, key, title, hi, ticks, fmt) in [(80, "out_tok_s_gpu", "Output tok/s per GPU", 450, [0, 100, 200, 300, 400], "{:.0f}"),
                                              (560, "goodput_req_s_gpu", "Goodput (SLO-meeting req/s per GPU)", 1.4, [0, 0.4, 0.8, 1.2], "{:.1f}")]:
        y0, w, h = 100, 360, 250
        xs = LogScale(12, 1300, x0 + 10, x0 + w - 10); ys = LinScale(0, hi, y0 + h, y0)
        f.add(T(x0, y0 - 14, title, "ink h"))
        for t in ticks:
            f.add(L(x0, ys(t), x0 + w, ys(t), "grid"), T(x0 - 6, ys(t) + 4, fmt.format(t), "mut s", "end"))
        for c in CONC:
            f.add(L(xs(c), y0 + h, xs(c), y0 + h + 4), T(xs(c), y0 + h + 18, f"{c:,}", "mut s", "middle"))
        f.add(L(x0, y0 + h, x0 + w, y0 + h))
        for lab, col, shp, keys in SERIES:
            pts = [(xs(P(k)["concurrency"]), ys(P(k)[key]), lower_bound(P(k)), P(k)[key]) for k in keys]
            f.add('<polyline fill="none" stroke-width="1.6" class="s%d" points="%s"/>' % (col, " ".join(f"{a:.1f},{b:.1f}" for a, b, _, _ in pts)))
            for cx, cy, lb, v in pts:
                f.add(marker(shp, cx, cy, col, hollow=lb, size=5.5))
        f.add(T(x0 + w / 2, y0 + h + 36, "concurrent agentic sessions (log scale)", "mut s", "middle"))
    # direct labels on the points that carry the message
    a, b = P("vllm-tp8ep8-fp8kv/AG-c256-t8"), P("vllm-tp8ep8/AG-c64-t8")
    xs = LogScale(12, 1300, 90, 430); ys = LinScale(0, 450, 350, 100)
    f.add(T(xs(256) + 10, ys(a["out_tok_s_gpu"]) + 4, f"{a['out_tok_s_gpu']:.0f} (FP8 KV, 18% in SLO)", "ink s"))
    xs2 = LogScale(12, 1300, 570, 910); ys2 = LinScale(0, 1.4, 350, 100)
    f.add(T(xs2(64) + 10, ys2(b["goodput_req_s_gpu"]) - 8, f"{b['goodput_req_s_gpu']:.2f} (TP8+EP8, 100% in SLO)", "ink s"))
    legend(f, 80, 412, [(l, c, s) for l, c, s, _ in SERIES])
    f.add(T(24, 440, CONDITIONS + ". Per-GPU values divide by all 8 GPUs.", "faint xs"))
    f.write("f02-throughput-and-goodput.svg")


IDEAL = {4: 84.2, 8: 88.9, 20: 94.1}


def f03():
    rows = [("TP8, c16 (4 turns)", "vllm-tp8/AG-c16-t4", 4), ("TP8, c64", "vllm-tp8/AG-c64-t8", 8),
            ("TP8+EP8, c64", "vllm-tp8ep8/AG-c64-t8", 8), ("TRT-LLM TP8+EP8, c64 (blocks)", "trtllm-tp8ep8/AG-c64-t8", 8),
            ("P/D FP8 KV, c64", "vllm-pd-1p1d-fp8kv/AG-c64-t8", 8), ("TP8, c64 (20 turns)", "vllm-tp8-matrix/AG-c64-t20", 20),
            ("TP8+EP8 FP8 KV, c256", "vllm-tp8ep8-fp8kv/AG-c256-t8", 8),
            ("DP8+EP8 + KV router, c64", "vllm-dp8ep8-kvrouter/AG-c64-t8", 8), ("DP8+EP8 internal balancer, c64", "vllm-dp8ep8/AG-c64-t8", 8),
            ("P/D BF16 KV, c64", "vllm-pd-1p1d/AG-c64-t8", 8), ("TP8+EP8 BF16 KV, c256", "vllm-tp8ep8/AG-c256-t8", 8),
            ("TRT-LLM TP8+EP8, c256 (blocks)", "trtllm-tp8ep8/AG-c256-t8", 8), ("TP8, c256", "vllm-tp8/AG-c256-t8", 8),
            ("TP8+EP8 FP8 KV, c1000", "vllm-tp8ep8-fp8kv/AG-c1000-t8", 8), ("TP8, c1000", "vllm-tp8-matrix/AG-c1000-t8", 8),
            ("P/D FP8 KV, c256", "vllm-pd-1p1d-fp8kv/AG-c256-t8", 8), ("DP8+EP8 internal balancer, c256", "vllm-dp8ep8/AG-c256-t8", 8)]
    rh, y0 = 22, 96
    f = Fig(960, y0 + rh * len(rows) + 70, "Measured prefix-cache hit rate versus the ideal for each agentic point",
            "One row per agentic point. A tick marks the ideal hit rate for the session length (84.2% for 4 turns, 88.9% for 8, 94.1% for 20) "
            "and a dot marks the measured value. While the KV pool has room the measured hit matches the ideal (84.0 to 94.4%). When the pool "
            "is full, when a data-parallel balancer scatters a session's turns, or when both P/D pools evict at 256 sessions, it falls to 37.6 to 71.4%.",
            ["data/points.json hit_pct, kv_max for: " + ", ".join(k for _, k, _ in rows),
             "Ideal: 1 - 600n / (sum of n turn inputs), docs/00-method.md"])
    f.add(T(24, 32, "Prefix-cache hit vs the ideal: at the ceiling while KV fits, about half once it does not", "ink t"))
    f.add(T(24, 52, "Dot = measured hit (filled: KV pool had room, peak use < 0.99; hollow: pool full). Tick = ideal for that session length.", "mut"))
    xs = LinScale(30, 100, 330, 840)
    for t in range(30, 101, 10):
        f.add(L(xs(t), y0 - 8, xs(t), y0 + rh * len(rows) - 6, "grid"), T(xs(t), y0 + rh * len(rows) + 10, f"{t}%", "mut s", "middle"))
    f.add(T(850, y0 - 16, "KV peak", "mut s"))
    for i, (lab, k, n) in enumerate(rows):
        p = P(k); y = y0 + i * rh; hit = p["hit_pct"]; full = p["kv_max"] >= 0.99
        f.add(T(320, y + 4, lab, "ink s", "end"))
        f.add(L(xs(min(hit, IDEAL[n])), y, xs(max(hit, IDEAL[n])), y, "axis", 3, 'stroke-opacity=".35"'))
        f.add(L(xs(IDEAL[n]), y - 8, xs(IDEAL[n]), y + 8, "sm", 2.2))
        f.add(marker("o", xs(hit), y, 2 if full else 1, hollow=full, size=5.5))
        f.add(T(xs(hit) + (10 if hit >= IDEAL[n] else -10), y - 7, f"{hit:.1f}", "ink xs", "start" if hit >= IDEAL[n] else "end"))
        f.add(T(852, y + 4, f"{p['kv_max']:.2f}", "mut s"))
    yb = y0 + rh * len(rows) + 34
    f.add(T(24, yb, "TRT-LLM counts reused 64-token blocks, vLLM counts tokens; P/D pools the prefill and decode workers' counters. " + "Single runs.", "faint xs"))
    f.add(T(24, yb + 16, CONDITIONS + ".", "faint xs"))
    f.write("f03-prefix-hit-vs-ideal.svg")


def f04():
    st = json.load(open(DATA / "engine-startup.json"))
    kv = lambda run, w="worker0": st[run]["workers"][w]["kv_tokens"]
    rows = [("TP8+EP8, FP8 KV (vLLM)", kv("vllm-tp8ep8-fp8kv"), 3),
            ("TP8, BF16 KV (vLLM)", kv("vllm-tp8"), 1),
            ("TP8+EP8, BF16 KV (vLLM)", kv("vllm-tp8ep8"), 1),
            ("TP8+EP8, BF16 KV (TensorRT-LLM)", kv("trtllm-tp8ep8"), 1),
            ("DP8+EP8, BF16 KV (8 ranks)", kv("vllm-dp8ep8"), 1),
            ("P/D decode pool, FP8 KV (4 GPUs)", kv("vllm-pd-1p1d-fp8kv"), 2),
            ("P/D decode pool, BF16 KV (4 GPUs)", kv("vllm-pd-1p1d"), 2)]
    f = Fig(960, 430, "KV cache capacity in tokens by layout, against estimated session demand at 256 sessions",
            "Horizontal bars of KV pool size from engine startup logs: TP8+EP8 FP8 KV 1,812,224 tokens; TP8 BF16 914,240; TP8+EP8 BF16 906,304; "
            "TensorRT-LLM TP8+EP8 860,608; DP8+EP8 747,008; P/D decode pool 587,648 (FP8) and 294,016 (BF16). A shaded band marks estimated "
            "demand at 256 sessions (1.64 to 2.15 million tokens). Only the FP8 KV pool reaches the band; demand at 1,000 sessions (6.4 to 8.4 million) is off the scale.",
            ["data/engine-startup.json kv_tokens (worker0) for " + ", ".join(sorted({'vllm-tp8ep8-fp8kv', 'vllm-tp8', 'vllm-tp8ep8', 'trtllm-tp8ep8', 'vllm-dp8ep8', 'vllm-pd-1p1d-fp8kv', 'vllm-pd-1p1d'})),
             "Demand band: 256 sessions x about 8,400 tokens at the end of an 8-turn session, with the 2,000-token shared prompt stored per session (2.15M) or once (1.64M)"])
    f.add(T(24, 32, "KV capacity by layout: layout moves it by tens of percent, KV dtype by 2x", "ink t"))
    f.add(T(24, 52, "Pool size from each engine's startup log. Band: estimated KV demand of 256 full-length sessions.", "mut"))
    x0, x1 = 290, 900; xs = LinScale(0, 2.4e6, x0, x1); y0, rh = 96, 34
    for t in [0, 0.5e6, 1.0e6, 1.5e6, 2.0e6]:
        f.add(L(xs(t), y0 - 10, xs(t), y0 + rh * len(rows), "grid"), T(xs(t), y0 + rh * len(rows) + 16, f"{t / 1e6:.1f}M", "mut s", "middle"))
    f.add(R(xs(1.64e6), y0 - 10, xs(2.15e6) - xs(1.64e6), rh * len(rows) + 10, "band", 0))
    f.add(T(xs(1.895e6), y0 - 16, "256-session demand", "c4 s b", "middle"))
    for i, (lab, v, col) in enumerate(rows):
        y = y0 + i * rh
        f.add(T(x0 - 10, y + 17, lab, "ink s", "end"), R(x0, y + 4, xs(v) - x0, 20, f"c{col}", 2), T(xs(v) + 6, y + 19, f"{v:,}", "ink s"))
    f.add(T(x1, y0 + rh * len(rows) + 40, "1,000-session demand (6.4M to 8.4M tokens) is off the scale →", "c4 s", "end"))
    f.add(T(24, 402, "Demand band is arithmetic (8,400 tokens at session end), so it overstates the average resident context.", "faint xs"))
    f.add(T(24, 418, CONDITIONS + ". GPU memory utilization 0.92, block size 64.", "faint xs"))
    f.write("f04-kv-capacity-by-layout.svg")


def bars_panel(f, x0, y0, w, h, groups, metric, title, hi, ticks, unit, slo=None, log=False, lo=None):
    """groups: [(label, [(key, color), ...])]. Plots p90 bars with a p50 tick, value label on each bar."""
    ys = (LogScale(lo, hi, y0 + h, y0) if log else LinScale(0, hi, y0 + h, y0))
    base = y0 + h
    f.add(T(x0, y0 - 14, title, "ink h"))
    for t in ticks:
        f.add(L(x0, ys(t), x0 + w, ys(t), "grid"), T(x0 - 6, ys(t) + 4, f"{t:g}", "mut s", "end"))
    if slo:
        f.add(L(x0, ys(slo[0]), x0 + w, ys(slo[0]), "sr", 1.4, 'stroke-dasharray="5 4"'))
    gw = w / len(groups)
    for gi, (glab, items) in enumerate(groups):
        bw = min(34, (gw - 20) / len(items))
        gx = x0 + gi * gw + (gw - bw * len(items)) / 2
        for j, (k, col) in enumerate(items):
            p = P(k); v = p[metric]; v50 = p[metric.replace("p90", "p50")] if "p90" in metric else None
            scale = 1000 if (metric.startswith("ttft") and unit == "s") else 1
            bx = gx + j * bw
            top = ys(max(v / scale, lo or 0.0001))
            hollow = lower_bound(p)
            f.add(R(bx + 2, top, bw - 4, base - top, f"c{col}" if not hollow else f"hollow s{col}", 1, 'stroke-width="2"' if hollow else ""))
            if v50 is not None:
                f.add(L(bx + 2, ys(v50 / scale), bx + bw - 2, ys(v50 / scale), "sm", 2))
            vv = v / scale
            lab = f"{vv:.2f}" if (unit == "s" and vv < 10) else (f"{vv:.1f}" if vv < 100 else f"{vv:,.0f}")
            f.add(T(bx + bw / 2, top - 5, lab, "ink xs", "middle"))
        f.add(T(x0 + gi * gw + gw / 2, base + 16, glab, "mut s", "middle"))
    f.add(L(x0, base, x0 + w, base))



def f05():
    groups = [("S2-c16", [("vllm-tp8ep8/S2-c16", 1), ("trtllm-tp8ep8/S2-c16", 2)]),
              ("AG-c64", [("vllm-tp8ep8/AG-c64-t8", 1), ("trtllm-tp8ep8/AG-c64-t8", 2)]),
              ("S1-c64", [("vllm-tp8ep8/S1-c64", 1), ("trtllm-tp8ep8/S1-c64", 2)]),
              ("AG-c256", [("vllm-tp8ep8/AG-c256-t8", 1), ("trtllm-tp8ep8/AG-c256-t8", 2)])]
    f = Fig(960, 432, "vLLM 0.31.0 versus TensorRT-LLM 1.3.0rc29 at TP8+EP8: TTFT p90 and ITL p90 at four points",
            "Grouped bars, vLLM blue and TensorRT-LLM orange, p90 with a dark tick at p50. TTFT p90 in seconds: S2-c16 3.71 vs 3.61, AG-c64 0.85 vs 0.73, "
            "S1-c64 1.13 vs 1.04, AG-c256 13.96 vs 14.59. ITL p90 in ms: S2-c16 45.9 vs 45.5, AG-c64 25.6 vs 27.7, S1-c64 20.0 vs 23.6, AG-c256 164.6 vs 142.7. "
            "Neither engine wins both panels; most gaps are within run-to-run noise for single runs.",
            ["data/points.json rows vllm-tp8ep8/* and trtllm-tp8ep8/* (S2-c16, AG-c64-t8, S1-c64, AG-c256-t8)"])
    f.add(T(24, 32, "vLLM vs TensorRT-LLM, same TP8+EP8 layout and default knobs: no overall winner", "ink t"))
    f.add(T(24, 52, "Bar = p90, dark tick = p50. S2 = 16,384 in / 256 out, S1 = 512 in / 4,096 out, AG = agentic sessions.", "mut"))
    bars_panel(f, 80, 100, 370, 240, groups, "ttft_p90_ms", "TTFT p90 (s)", 16, [0, 2, 4, 8, 12, 16], "s", slo=(2.0, "SLO 2 s"))
    bars_panel(f, 560, 100, 370, 240, groups, "itl_p90_ms", "ITL p90 (ms)", 180, [0, 25, 50, 100, 150], "ms", slo=(50.0, "SLO 50 ms"))
    f.add(R(80, 376, 12, 12, "c1", 2), T(98, 386, "vLLM 0.31.0", "ink s"), R(200, 376, 12, 12, "c2", 2), T(218, 386, "TensorRT-LLM 1.3.0rc29 (PyTorch backend)", "ink s"),
          L(520, 382, 548, 382, "sr", 1.6, 'stroke-dasharray="5 4"'), T(556, 386, "SLO: TTFT 2 s, ITL 50 ms", "ink s"))
    f.add(T(24, 404, "TRT-LLM KV pool is 5% smaller (860,608 vs 906,304 tokens). Default knobs on both engines.", "faint xs"))
    f.add(T(24, 420, CONDITIONS + ".", "faint xs"))
    f.write("f05-vllm-vs-trtllm.svg")


def f06():
    groups = [("64 sessions", [("vllm-tp8ep8/AG-c64-t8", 1), ("vllm-pd-1p1d/AG-c64-t8", 2), ("vllm-pd-1p1d-fp8kv/AG-c64-t8", 4)]),
              ("256 sessions", [("vllm-tp8ep8-fp8kv/AG-c256-t8", 3), ("vllm-pd-1p1d-fp8kv/AG-c256-t8", 4)])]
    f = Fig(960, 440, "Aggregated serving versus prefill/decode disaggregation (1 prefill : 1 decode) on agentic load",
            "Three panels of bars: TTFT p90 (log seconds), ITL p90 (ms) and output tokens per second per GPU. At 256 sessions with FP8 KV, "
            "P/D cuts ITL p90 from 56.4 to 38.5 ms but raises TTFT p90 from 4.1 s to at least 44.4 s (the P/D run hit its time cap) and cuts output from 401.9 to 167.5 tok/s/GPU. "
            "At 64 sessions the two are close; P/D with BF16 KV has a TTFT p90 of 3.25 s versus 0.85 s aggregated.",
            ["data/points.json rows vllm-tp8ep8/AG-c64-t8, vllm-pd-1p1d/AG-c64-t8, vllm-pd-1p1d-fp8kv/AG-c64-t8, vllm-tp8ep8-fp8kv/AG-c256-t8, vllm-pd-1p1d-fp8kv/AG-c256-t8"])
    f.add(T(24, 32, "P/D trades TTFT and throughput for steadier token streaming", "ink t"))
    f.add(T(24, 52, "Aggregated = one TP8+EP8 engine on 8 GPUs. P/D = prefill TP4+EP4 on 4 GPUs, decode TP4+EP4 on 4 GPUs, KV moved over NVLink.", "mut"))
    bars_panel(f, 70, 110, 250, 230, groups, "ttft_p90_ms", "TTFT p90 (s, log)", 60, [0.5, 1, 2, 5, 10, 20, 50], "s", slo=(2.0, "SLO 2 s"), log=True, lo=0.3)
    bars_panel(f, 380, 110, 250, 230, groups, "itl_p90_ms", "ITL p90 (ms)", 70, [0, 20, 40, 60], "ms", slo=(50.0, "SLO 50 ms"))
    bars_panel(f, 690, 110, 250, 230, groups, "out_tok_s_gpu", "Output tok/s per GPU", 450, [0, 100, 200, 300, 400], "")
    items = [("Agg TP8+EP8, BF16 KV", 1), ("P/D, BF16 KV", 2), ("P/D, FP8 KV", 4), ("Agg TP8+EP8, FP8 KV", 3)]
    for i, (lab, c) in enumerate(items):
        f.add(R(70 + i * 175, 380, 12, 12, f"c{c}", 2), T(88 + i * 175, 390, lab, "ink s"))
    f.add(L(770, 386, 796, 386, "sr", 1.6, 'stroke-dasharray="5 4"'), T(802, 390, "SLO", "ink s"))
    f.add(T(24, 414, "Hollow bar: P/D at 256 sessions stopped at its time cap (2,152 of 2,208 turns), so its TTFT is a lower bound. Bars p90, ticks p50.", "faint xs"))
    f.add(T(24, 430, CONDITIONS + ". Per-GPU values divide by all 8 GPUs for both topologies.", "faint xs"))
    f.write("f06-pd-vs-aggregated.svg")


def f07():
    nx = json.load(open(DATA / "nixl.json"))["points"]
    rows = [("FP8 KV, 64 sessions", "vllm-pd-1p1d-fp8kv/AG-c64-t8"), ("BF16 KV, 64 sessions", "vllm-pd-1p1d/AG-c64-t8"),
            ("FP8 KV, 256 sessions", "vllm-pd-1p1d-fp8kv/AG-c256-t8"), ("FP8 KV, S2-c16 (16k-token prompts)", "vllm-pd-1p1d-fp8kv/S2-c16")]
    f = Fig(960, 330, "KV transfer cost per transfer between the prefill and decode pools",
            "Per-transfer means from the decode pool's NIXL histograms. Megabytes per transfer "
            + ", ".join(f"{nx[k]['mb_per_transfer']:.1f}" for _, k in rows) + "; time per transfer "
            + ", ".join(f"{nx[k]['xfer_ms']:.1f}" for _, k in rows) + " ms, of which CPU post time is "
            + ", ".join(f"{nx[k]['post_ms']:.1f}" for _, k in rows) + " ms. One transfer is one request's KV for one of "
            "the four decode GPUs. Zero failed transfers at every point.",
            ["data/nixl.json points: " + ", ".join(k for _, k in rows)])
    ag = [nx[k]["xfer_ms"] for _, k in rows[:3]]
    f.add(T(24, 32, f"KV hand-off cost in P/D: {min(ag):.0f} to {max(ag):.0f} ms per transfer on agentic load, "
                    f"{nx[rows[3][1]]['xfer_ms']:.0f} ms for 16k-token prompts", "ink t"))
    f.add(T(24, 52, "Mean per transfer (one request x one decode GPU). Time is split into CPU post time and the rest of the copy.", "mut"))
    xa = LinScale(0, 420, 300, 560); xb = LinScale(0, 150, 620, 900); y0, rh = 104, 44
    f.add(T(300, 90, "MB per transfer", "ink h"), T(620, 90, "ms per transfer", "ink h"))
    for i, (lab, k) in enumerate(rows):
        n = nx[k]; y = y0 + i * rh
        f.add(T(290, y + 15, lab, "ink s", "end"))
        f.add(R(300, y, xa(n["mb_per_transfer"]) - 300, 22, "c1", 2), T(xa(n["mb_per_transfer"]) + 6, y + 15, f"{n['mb_per_transfer']:.1f}", "ink s"))
        f.add(R(620, y, xb(n["post_ms"]) - 620, 22, "c2", 2), R(xb(n["post_ms"]), y, xb(n["xfer_ms"]) - xb(n["post_ms"]), 22, "c4", 2))
        f.add(T(xb(n["xfer_ms"]) + 6, y + 15, f"{n['xfer_ms']:.1f} ({n['post_share_pct']:.0f}% post)", "ink s"))
    f.add(R(620, 290, 12, 12, "c2", 2), T(638, 300, "CPU post time", "ink s"), R(750, 290, 12, 12, "c4", 2), T(768, 300, "rest of transfer", "ink s"))
    f.add(T(24, 300, "A per-stage cost: TTFT and ITL for these runs are in Figure 6.", "mut s"))
    f.add(T(24, 320, CONDITIONS + ". NIXL 1.4.1, UCX cuda_ipc over NVLink, block size 64.", "faint xs"))
    f.write("f07-kv-transfer-cost.svg")


def f08():
    keys = [("TP8, BF16 KV", "vllm-tp8-matrix/AG-c1000-t8", 1, "o"), ("TP8+EP8, FP8 KV", "vllm-tp8ep8-fp8kv/AG-c1000-t8", 3, "^")]
    per = {}
    for lab, k, col, shp in keys:
        by = {}
        for r in records(k):
            by.setdefault(r["turn_index"], []).append(r)
        per[k] = {t: (pct([r["ttft_ms"] for r in rs], 90) / 1000, pct([r["itl_ms"] for r in rs], 90), len(rs)) for t, rs in sorted(by.items())}
    f = Fig(960, 440, "TTFT p90 and ITL p90 by turn index at 1,000 concurrent sessions",
            "Two panels by turn index within a session (0-based). TTFT p90 climbs turn after turn on both configurations and never levels off; "
            "the runs stopped at their time cap after turn 4 (TP8 BF16) and turn 5 (TP8+EP8 FP8), so whole-run percentiles at 1,000 sessions are lower bounds. "
            f"ITL p90 stays between {min(v[1] for d in per.values() for v in d.values()):.0f} and {max(v[1] for d in per.values() for v in d.values()):.0f} ms on every turn, "
            "several times the 50 ms SLO.",
            [f"data/{k}/records.jsonl.gz (per-turn p90, nearest rank)" for _, k, _, _ in keys])
    f.add(T(24, 32, "At 1,000 sessions latency is still climbing when the run ends", "ink t"))
    f.add(T(24, 52, "p90 per turn index. Each later turn carries more history and waits longer; turns 6 and later were never reached.", "mut"))
    for x0, idx, title, lo, hi, ticks, slo in [(80, 0, "TTFT p90 (s)", 0, 130, [0, 25, 50, 75, 100, 125], 2.0),
                                               (560, 1, "ITL p90 (ms)", 0, 320, [0, 50, 100, 200, 300], 50.0)]:
        y0, w, h = 100, 360, 230
        xs = LinScale(-0.4, 8.4, x0, x0 + w); ys = LinScale(lo, hi, y0 + h, y0)
        f.add(T(x0, y0 - 14, title, "ink h"))
        f.add(R(xs(5.6), y0, xs(8.4) - xs(5.6), h, "pan", 0), T(xs(7.0), y0 + h / 2, "not reached", "mut s", "middle"))
        for t in ticks:
            f.add(L(x0, ys(t), x0 + w, ys(t), "grid"), T(x0 - 6, ys(t) + 4, f"{t:g}", "mut s", "end"))
        for t in range(9):
            f.add(T(xs(t), y0 + h + 18, f"t{t}", "mut s", "middle"))
        f.add(L(x0, y0 + h, x0 + w, y0 + h))
        f.add(L(x0, ys(slo), x0 + w, ys(slo), "sr", 1.4, 'stroke-dasharray="5 4"'), T(x0 + w - 4, ys(slo) - 5, f"SLO {slo:g}{' s' if idx == 0 else ' ms'}", "cr s", "end"))
        for lab, k, col, shp in keys:
            pts = [(xs(t), ys(v[idx])) for t, v in per[k].items()]
            f.add('<polyline fill="none" stroke-width="1.8" class="s%d" points="%s"/>' % (col, " ".join(f"{a:.1f},{b:.1f}" for a, b in pts)))
            for a, b in pts:
                f.add(marker(shp, a, b, col, hollow=True, size=5))
        f.add(T(x0 + w / 2, y0 + h + 36, "turn index within a session", "mut s", "middle"))
    for i, (lab, k, col, shp) in enumerate(keys):
        p = P(k)
        f.add(marker(shp, 86 + i * 400, 386, col, hollow=True, size=5),
              T(98 + i * 400, 390, f"{lab}: {p['records']:,} of {p['turns_planned_total']:,} planned turns ran", "ink s"))
    f.add(T(24, 412, "The two runs differ in layout and KV dtype. Hollow = run stopped at its time cap.", "faint xs"))
    f.add(T(24, 428, CONDITIONS + ".", "faint xs"))
    f.write("f08-per-turn-latency-at-1000.svg")


KNOBS = [  # label, baseline key, variant key, note
    ("BF16 → FP8 KV cache (TP8+EP8, 256 sessions)", "vllm-tp8ep8/AG-c256-t8", "vllm-tp8ep8-fp8kv/AG-c256-t8", "like-for-like"),
    ("TP8 → TP8+EP8 (64 sessions)", "vllm-tp8/AG-c64-t8", "vllm-tp8ep8/AG-c64-t8", "like-for-like"),
    ("TP8 → DP8+EP8, vLLM balancer (64)", "vllm-tp8/AG-c64-t8", "vllm-dp8ep8/AG-c64-t8", "like-for-like"),
    ("DP8 balancer → Dynamo KV router (64)", "vllm-dp8ep8/AG-c64-t8", "vllm-dp8ep8-kvrouter/AG-c64-t8", "like-for-like"),
    ("Default → tuned MoE config, TP8 (S2-c16)", "vllm-tp8-moe-default/S2-c16", "vllm-tp8-moe-tuned/S2-c16", "like-for-like"),
    ("Default → tuned MoE config, TP8 (64 sessions)", "vllm-tp8-moe-default/AG-c64-t8", "vllm-tp8-moe-tuned/AG-c64-t8", "like-for-like"),
    ("vLLM → TensorRT-LLM, TP8+EP8 (64)", "vllm-tp8ep8/AG-c64-t8", "trtllm-tp8ep8/AG-c64-t8", "KV pool -5%"),
    ("Aggregated → P/D 1P:1D, BF16 KV (64)", "vllm-tp8ep8/AG-c64-t8", "vllm-pd-1p1d/AG-c64-t8", "like-for-like"),
    ("Aggregated → P/D 1P:1D, FP8 KV (256)", "vllm-tp8ep8-fp8kv/AG-c256-t8", "vllm-pd-1p1d-fp8kv/AG-c256-t8", "P/D capped"),
    ("8 → 20 turns per session (TP8, 64)", "vllm-tp8/AG-c64-t8", "vllm-tp8-matrix/AG-c64-t20", "rules differ"),
    ("64 → 256 sessions (TP8, BF16 KV)", "vllm-tp8/AG-c64-t8", "vllm-tp8/AG-c256-t8", "like-for-like"),
]
COLS = [("TTFT p90", "ttft_p90_ms", True), ("ITL p90", "itl_p90_ms", True),
        ("output tok/s/GPU", "out_tok_s_gpu", False), ("goodput/GPU", "goodput_req_s_gpu", False)]


def f09():
    rh, y0 = 36, 108
    f = Fig(960, y0 + rh * len(KNOBS) + 52, "Knob map: measured effect of each single change on TTFT p90, ITL p90, output throughput and goodput",
            "A table of colored cells. Each row is one change from a baseline point to a variant point; each cell shows baseline and variant and the "
            "percent change, blue when the change helps and red when it hurts, with latency signs flipped so lower latency counts as better. FP8 KV is the only "
            "change that helps all four columns by a large margin; P/D trades TTFT and throughput for ITL; going from 64 to 256 sessions on BF16 KV hurts everything.",
            ["data/points.json rows: " + "; ".join(f"{a} -> {b}" for _, a, b, _ in KNOBS)])
    f.add(T(24, 32, "Knob map: which change moves which metric (p90 latency)", "ink t"))
    f.add(T(24, 52, "Each cell: baseline → variant, then the change. Blue helps, red hurts (lower latency counts as better). Color saturates at ±100%.", "mut"))
    cx = [380, 520, 660, 800]; cw = 134
    for (lab, _, _), x in zip(COLS, cx):
        f.add(T(x + cw / 2, y0 - 14, lab, "ink h", "middle"))
    f.add(T(24, y0 - 14, "change (concurrency)", "ink h"))
    for i, (lab, a, b, note) in enumerate(KNOBS):
        y = y0 + i * rh; pa, pb = P(a), P(b)
        f.add(T(24, y + 15, lab, "ink s"), T(24, y + 29, note, "faint xs"))
        for (cl, key, lower_better), x in zip(COLS, cx):
            va, vb = pa[key], pb[key]
            if va <= 0:
                ch, txt = 0, "n/a"
            else:
                r = vb / va; ch = (r - 1) * 100
                txt = (f"{r:.1f}x" if r >= 2 else f"{ch:+.0f}%")
            good = (-ch if lower_better else ch)
            cls = "c1" if good > 0 else "cr"
            op = min(abs(good), 100) / 100 * 0.55 + 0.06
            f.add(R(x, y + 2, cw, rh - 4, cls, 3, f'fill-opacity="{op:.2f}"'))
            if key == "itl_p90_ms":
                fv = lambda v: f"{v:.1f} ms" if v < 100 else f"{v:.0f} ms"
            elif key.endswith("_ms"):
                fv = fmt_ms
            else:
                fv = lambda v: f"{v:.0f}" if v >= 10 else f"{v:.2f}" if v >= 0.1 or v == 0 else f"{v:.3f}" if v >= 0.01 else f"{v:.4f}"
            f.add(T(x + cw / 2, y + 14, f"{fv(va)} → {fv(vb)}", "ink xs", "middle"), T(x + cw / 2, y + 29, txt, "ink s b", "middle"))
    yb = y0 + rh * len(KNOBS) + 18
    f.add(T(24, yb, "Notes: the TRT-LLM row also has a 5% smaller KV pool; the P/D 256 row compares a capped run with a complete one; 8 vs 20 turns mixes completion rules.", "faint xs"))
    f.add(T(24, yb + 16, CONDITIONS + ". Differences under about 10% are within run-to-run noise.", "faint xs"))
    f.write("f09-knob-map.svg")


if __name__ == "__main__":
    f01(); f02(); f03(); f04(); f05(); f06(); f07(); f08(); f09()
