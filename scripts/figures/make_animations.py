#!/usr/bin/env python3
"""Animated SVGs a01..a06 (hand-laid SVG + CSS keyframes, no JavaScript), regenerated from data/.

  python3 scripts/figures/make_animations.py      # writes figures/anim/a0*.svg

Conventions:
- Light palette by default, prefers-color-scheme: dark override.
- prefers-reduced-motion: animation stops and every element shows its summary state (the static
  opacity attribute of each animated element is its state at the summary time, near the end).
- Every measured number is read from data/ at build time; the XML comment at the top of each file
  lists the files. Shapes such as KV cells, queues and step boxes are schematic.
- Headline percentile is p90: TTFT p90, ITL p90, tok/s per user = 1000 / ITL p90.
"""
import json
from svglib import CSS, MOTION_CSS, DATA, FIG, P, points, records, pct, esc, T, R, L, C, marker, fmt_ms

W, H = 960, 540
POINT_OK_PCT = 90.0  # a point meets the target when at least 90% of its requests are SLO-ok (DistServe convention)


class Anim:
    def __init__(self, period, title, desc, sources, still=95.0):
        self.period, self.title, self.desc, self.sources, self.still = period, title, desc, sources, still
        self.kf, self.body, self.n = [], [], 0

    def win(self, windows, fade=0.6):
        """Opacity keyframes, visible inside [(a, b), ...] percent windows. Returns (class, static opacity)."""
        pts = {0: 1 if any(a <= 0 for a, b in windows) else 0}
        for a, b in windows:
            if a > 0:
                pts[max(a - fade, 0.01)] = 0; pts[a] = 1
            if b < 100:
                pts[b] = 1; pts[min(b + fade, 99.99)] = 0
        pts[100] = 1 if any(b >= 100 for a, b in windows) else 0
        name = f"k{self.n}"; self.n += 1
        self.kf.append(f"@keyframes {name}{{" + " ".join(f"{k:.2f}%{{opacity:{pts[k]}}}" for k in sorted(pts)) + "}"
                       f".{name}{{animation:{name} {self.period}s linear infinite}}")
        static = 1 if any(a <= self.still <= b for a, b in windows) else 0
        return name, static

    def frames(self, prop_frames):
        name = f"k{self.n}"; self.n += 1
        self.kf.append(f"@keyframes {name}{{" + " ".join(f"{p:.2f}%{{{c}}}" for p, c in prop_frames) + "}"
                       f".{name}{{animation:{name} {self.period}s linear infinite}}")
        return name

    def show(self, windows, inner):
        k, s = self.win(windows)
        self.body.append(f'<g class="{k}" opacity="{s}">{inner}</g>')

    def add(self, *s):
        self.body.extend(s)

    def svg(self):
        src = "\n".join("  " + s.replace("--", "-") for s in self.sources)
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-labelledby="title desc">\n'
                f'<title id="title">{esc(self.title)}</title>\n<desc id="desc">{esc(self.desc)}</desc>\n'
                f'<!--\nData sources (every measured number in this animation is read from these files):\n{src}\n-->\n'
                f'<style>{CSS}{MOTION_CSS}\n' + "\n".join(self.kf) + '\n</style>\n'
                f'<rect class="bg" x="0" y="0" width="{W}" height="{H}"/>\n' + "\n".join(self.body) + "\n</svg>\n")

    def write(self, name):
        p = FIG / "anim" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.svg()); print("wrote", name, p.stat().st_size, "bytes")


def header(a, title, sub):
    a.add(T(24, 32, title, "ink t"), T(24, 54, sub, "mut"))


def footer(a, lines):
    y = H - 14 - 17 * (len(lines) - 1)
    for i, s in enumerate(lines):
        a.add(T(24, y + 17 * i, s, "faint xs"))


def kv_tokens(run, worker="worker0"):
    return json.load(open(DATA / "engine-startup.json"))[run]["workers"][worker]["kv_tokens"]


def per_turn(key, max_turn):
    by = {}
    for r in records(key):
        if r["turn_index"] <= max_turn:
            by.setdefault(r["turn_index"], []).append(r)
    return {t: dict(n=len(rs), isl=pct([r["isl"] for r in rs], 50), ttft90=pct([r["ttft_ms"] for r in rs], 90),
                    itl90=pct([r["itl_ms"] for r in rs], 90)) for t, rs in sorted(by.items())}


# ------------------------------------------------------------------ a01 KV cliff
def a01():
    lo, hi = P("vllm-tp8/AG-c64-t8"), P("vllm-tp8/AG-c256-t8")
    fp8 = P("vllm-tp8ep8-fp8kv/AG-c256-t8")
    pool = kv_tokens("vllm-tp8")
    a = Anim(16, "The KV cliff: from 64 to 256 agentic sessions the KV pool fills and prefix hits collapse",
             f"A 40-cell KV pool ({pool:,} tokens, vLLM TP8, BF16 KV). At 64 sessions 15 of 40 cells are used (KV peak {lo['kv_max']:.2f}), prefix hit "
             f"{lo['hit_pct']:.1f}%, TTFT p90 {fmt_ms(lo['ttft_p90_ms'])}, ITL p90 {lo['itl_p90_ms']:.1f} ms, all requests in SLO. At 256 sessions every cell "
             f"fills, cached history is evicted and recomputed, requests queue (waiting p90 {hi['wait_p90']:.0f}), the hit falls to {hi['hit_pct']:.1f}%, "
             f"TTFT p90 rises to {fmt_ms(hi['ttft_p90_ms'])} and ITL p90 to {hi['itl_p90_ms']:.0f} ms; {hi['slo_ok_pct']:.2f}% of requests meet the SLO.",
             ["data/engine-startup.json vllm-tp8 kv_tokens", "data/points.json vllm-tp8/AG-c64-t8, vllm-tp8/AG-c256-t8 (count-capped, 2,048 of 2,208 turns)",
              "data/points.json vllm-tp8ep8-fp8kv/AG-c256-t8 (FP8 note)"], still=92)
    header(a, "The KV cliff: sessions outgrow the KV pool, hits collapse, work is redone",
           "vLLM TP8, BF16 KV cache, agentic sessions (2,000-token shared prompt, +600 in / 200 out per turn)")
    P1, P2, P3 = (0, 34), (34, 70), (70, 100)
    x0, y0, cw, ch, gap = 40, 120, 44, 34, 5
    a.add(T(x0, 106, f"KV pool: {pool:,} tokens (1 cell = {pool / 40 / 1000:.1f}k tokens)", "ink h"))
    cells = [(x0 + (i % 10) * (cw + gap), y0 + (i // 10) * (ch + gap)) for i in range(40)]
    for x, y in cells:
        a.add(R(x, y, cw, ch, "emp"))
    for x, y in cells[:15]:
        a.add(R(x, y, cw, ch, "c1"))
    for i, (x, y) in enumerate(cells[15:]):
        a.show([(36 + i * 0.36, 100)], R(x, y, cw, ch, "c1"))
    for j, i in enumerate([3, 7, 12, 18, 22, 27, 31, 36]):
        x, y = cells[i]
        a.show([(50 + j * 0.5, 58 + j * 0.5)], R(x, y, cw, ch, "cr") + T(x + cw / 2, y + 22, "evict", "w s", "middle"))
        a.show([(58 + j * 0.5, 100)], R(x + 3, y + 3, cw - 6, ch - 6, "c3", 2) + T(x + cw / 2, y + 22, "redo", "w s", "middle"))
    bw = 10 * (cw + gap) - gap; ybar = y0 + 4 * (ch + gap) + 6
    a.add(R(x0, ybar, bw, 10, "emp", 2))
    a.show([P1], R(x0, ybar, lo["kv_max"] * bw, 10, "c1", 2) + T(x0, ybar + 30, f"KV in use {lo['kv_max'] * 100:.0f}% (64 sessions)", "ink"))
    a.show([(P2[0], 100)], R(x0, ybar, bw, 10, "c1", 2) + T(x0, ybar + 30, "KV in use 100% (256 sessions): new turns evict cached history", "ink"))
    ly = ybar + 52
    a.add(R(x0, ly - 11, 14, 14, "c1", 2), T(x0 + 20, ly, "cached session KV", "mut s"))
    a.add(R(x0 + 150, ly - 11, 14, 14, "cr", 2), T(x0 + 170, ly, "evicted", "mut s"))
    a.add(R(x0 + 250, ly - 11, 14, 14, "c3", 2), T(x0 + 270, ly, "recomputed as prefill", "mut s"))
    qx = 545
    a.add(T(qx - 6, 106, "waiting", "ink h"), L(qx, 120, qx, 270, "axis", 1), L(qx + 50, 120, qx + 50, 270, "axis", 1))
    a.add(C(qx + 25, 258, 6, "c2"))
    for i in range(1, 9):
        a.show([(44 + i * 2.6, 100)], C(qx + 25, 258 - i * 16, 6, "c2"))
    a.add(T(qx - 6, 290, "waiting p90", "mut s"))
    a.show([P1], T(qx - 6, 308, f"{lo['wait_p90']:.0f} requests", "ink h"))
    a.show([(P2[0], 100)], T(qx - 6, 308, f"{hi['wait_p90']:.0f} requests", "cr h"))
    mx = 640
    a.add(R(mx, 84, 296, 292, "pan", 6))
    rows = [("concurrent sessions", "64", "256"), ("KV peak use", f"{lo['kv_max']:.2f}", f"{hi['kv_max']:.2f}"),
            ("prefix hit (ideal 88.9%)", f"{lo['hit_pct']:.1f}%", f"{hi['hit_pct']:.1f}%"),
            ("preemptions", f"{lo['preemptions']:.0f}", f"{hi['preemptions']:.0f}"),
            ("TTFT p90", fmt_ms(lo["ttft_p90_ms"]), fmt_ms(hi["ttft_p90_ms"])),
            ("ITL p90", f"{lo['itl_p90_ms']:.1f} ms", f"{hi['itl_p90_ms']:.0f} ms"),
            ("requests in SLO", f"{lo['slo_ok_pct']:.0f}%", f"{hi['slo_ok_pct']:.2f}%")]
    for i, (lab, v1, v2) in enumerate(rows):
        y = 116 + i * 38
        a.add(T(mx + 14, y, lab, "mut"))
        a.show([P1], T(mx + 282, y, v1, "ink h", "end"))
        a.show([(P2[0], 100)], T(mx + 282, y, v2, "cr h" if i >= 2 else "ink h", "end"))
    a.add(T(mx + 14, 366, "measured, whole request, client side", "faint xs"))
    a.show([P1], T(40, 412, f"1. 64 sessions fit (KV peak {lo['kv_max']:.2f}). Each turn reuses its cached history: hit {lo['hit_pct']:.1f}%.", "ink h"))
    a.show([P2], T(40, 412, "2. 256 sessions need about 1.6M to 2.2M tokens of KV against a 0.91M pool: history is evicted and redone.", "ink h"))
    a.show([P3], T(40, 412, f"3. Result: hit {lo['hit_pct']:.0f}% → {hi['hit_pct']:.0f}%, TTFT p90 {fmt_ms(lo['ttft_p90_ms'])} → {fmt_ms(hi['ttft_p90_ms'])}, "
                             f"ITL p90 {lo['itl_p90_ms']:.0f} → {hi['itl_p90_ms']:.0f} ms.", "ink h"))
    a.show([P3], T(40, 434, f"What moved it: an FP8 KV cache doubles the pool (TP8+EP8, 256 sessions: hit {fp8['hit_pct']:.1f}%, KV peak {fp8['kv_max']:.2f}, "
                            f"{fp8['preemptions']:.0f} preemptions),", "ink"))
    a.show([P3], T(40, 452, f"but TTFT p90 is still {fmt_ms(fp8['ttft_p90_ms'])} and ITL p90 {fp8['itl_p90_ms']:.1f} ms there: only {fp8['slo_ok_pct']:.0f}% of requests meet the SLO.", "ink"))
    footer(a, ["Cells and queue are schematic; panel values are measured. 256-session run is count-capped (2,048 of 2,208 turns). Single runs.",
               "Qwen3-235B-A22B-Instruct-2507 FP8 | 8x H100 SXM5 | Dynamo 1.6.0 + vLLM 0.31.0 | SLO: TTFT ≤ 2 s and ITL ≤ 50 ms per request"])
    a.write("a01-kv-cliff.svg")


# ------------------------------------------------------------------ a02 DP scatter vs KV router
def a02():
    rr, kv = P("vllm-dp8ep8/AG-c64-t8"), P("vllm-dp8ep8-kvrouter/AG-c64-t8")
    rr2, kv2 = P("vllm-dp8ep8/AG-c256-t8"), P("vllm-dp8ep8-kvrouter/AG-c256-t8")
    a = Anim(14, "Data-parallel ranks: vLLM's balancer scatters a session's turns, the Dynamo KV router keeps them on one rank",
             f"Two panels of 8 data-parallel ranks. Left: vLLM's internal balancer sends each turn of one session to a different rank, so most turns "
             f"miss their history: prefix hit {rr['hit_pct']:.1f}%, TTFT p90 {fmt_ms(rr['ttft_p90_ms'])}, ITL p90 {rr['itl_p90_ms']:.1f} ms at 64 sessions. "
             f"Right: the Dynamo KV router sends turns to the rank that holds the session's blocks: hit {kv['hit_pct']:.1f}%, TTFT p90 "
             f"{fmt_ms(kv['ttft_p90_ms'])}, ITL p90 {kv['itl_p90_ms']:.1f} ms.",
             ["data/points.json vllm-dp8ep8/AG-c64-t8, vllm-dp8ep8-kvrouter/AG-c64-t8, and both AG-c256-t8 rows",
              "data/configs/vllm-dp8ep8*/cmd.txt (router mode); the ranks chosen for the one session drawn are illustrative"], still=96)
    header(a, "Data-parallel ranks: where do one session's turns land?",
           "vLLM DP8+EP8 (8 separate KV caches of about 93k tokens each), 64 agentic sessions; one session's turns t0 to t5 shown")
    panels = [(20, "vLLM internal balancer (round-robin frontend)", [0, 3, 6, 3, 1, 7], ["cold", "miss", "miss", "partial", "miss", "miss"], rr),
              (490, "Dynamo KV router (--router-mode kv)", [2, 2, 2, 2, 2, 2], ["cold", "hit", "hit", "hit", "hit", "hit"], kv)]
    for px, ttl, ranks, res, p in panels:
        a.add(R(px, 70, 450, 340, "pan", 8), T(px + 225, 94, ttl, "ink h", "middle"))
        rx, ry = px + 175, 108
        a.add(R(rx, ry, 100, 32, "emp", 6), T(rx + 50, ry + 21, "router", "ink", "middle"))
        bx = [px + 18 + i * 53 for i in range(8)]
        for i, x in enumerate(bx):
            a.add(R(x, 230, 45, 70, "emp", 4), T(x + 22, 318, f"rank {i}", "mut s", "middle"))
        a.add(L(rx + 50, ry + 32, rx + 50, 200, "axis", 1))
        for t, (rk, rs) in enumerate(zip(ranks, res)):
            t0 = 2 + t * 14.0; tx = bx[rk] + 22
            k = a.frames([(0, f"transform:translate({rx + 50}px,150px);opacity:0"), (t0, f"transform:translate({rx + 50}px,150px);opacity:0"),
                          (t0 + 0.5, f"transform:translate({rx + 50}px,150px);opacity:1"), (t0 + 5, f"transform:translate({tx}px,222px);opacity:1"),
                          (t0 + 5.5, f"transform:translate({tx}px,222px);opacity:0"), (100, f"transform:translate({tx}px,222px);opacity:0")])
            a.add(f'<g class="{k} motion" opacity="0"><circle r="9" class="c1"/><text y="4" class="w s" text-anchor="middle">t{t}</text></g>')
            stack = sum(1 for q in ranks[:t] if q == rk)
            a.show([(t0 + 5, 100)], R(bx[rk] + 6, 290 - 12 * min(stack, 4) - 8, 33, 9, "c1" if rs != "miss" else "c2", 2))
            dot = {"miss": "cr", "partial": "c2", "hit": "cg", "cold": "faint"}[rs]
            a.show([(t0 + 5, 100)], C(px + 14 + t * 72, 348, 4, dot) + T(px + 20 + t * 72, 352, f"t{t}: {rs}", "ink s"))
        a.add(T(px + 20, 382, f"prefix hit {p['hit_pct']:.1f}%, requests in SLO {p['slo_ok_pct']:.0f}%", "ink h"))
        a.add(T(px + 20, 400, f"TTFT p90 {fmt_ms(p['ttft_p90_ms'])}, ITL p90 {p['itl_p90_ms']:.1f} ms (measured, 64 sessions)", "ink s"))
    a.add(T(24, 438, "Left: turns land on ranks that never saw the session, so its history is prefilled again. Right: the router scores each rank's", "ink"))
    a.add(T(24, 456, f"cached blocks and picks the owner. Still below the 88.9% ideal (KV peak was only {kv['kv_max']:.2f}, so this is routing, not capacity).", "ink"))
    a.add(T(24, 474, f"At 256 sessions routing cannot help: both full, hit {rr2['hit_pct']:.1f}% vs {kv2['hit_pct']:.1f}%.", "ink"))
    footer(a, ["Blue chip = history cached on that rank; orange = history prefilled again after a miss; t0 is cold everywhere. Rank choices are illustrative.",
               "Qwen3-235B-A22B-Instruct-2507 FP8 | 8x H100 SXM5 | Dynamo 1.6.0 + vLLM 0.31.0 | single runs"])
    a.write("a02-dp-scatter-vs-kv-router.svg")


# ------------------------------------------------------------------ a03 aggregated step vs P/D
def a03():
    ag, pd = P("vllm-tp8ep8-fp8kv/AG-c256-t8"), P("vllm-pd-1p1d-fp8kv/AG-c256-t8")
    nx = json.load(open(DATA / "nixl.json"))["points"]
    x64, x256 = nx["vllm-pd-1p1d-fp8kv/AG-c64-t8"]["xfer_ms"], nx["vllm-pd-1p1d-fp8kv/AG-c256-t8"]["xfer_ms"]
    a = Anim(12, "Aggregated engine steps mix prefill chunks into decode steps; P/D keeps decode steps short but queues prefill",
             f"Top lane: one aggregated TP8+EP8 engine at 256 sessions, FP8 KV. Many steps carry a prefill chunk of up to 8,192 tokens next to the "
             f"running decodes, so the time between tokens stretches. Lower lanes: P/D with 4 prefill GPUs and 4 decode GPUs; decode steps carry only "
             f"decodes while the prefill queue grows. Measured end to end: aggregated TTFT p90 {fmt_ms(ag['ttft_p90_ms'])}, ITL p90 {ag['itl_p90_ms']:.1f} ms, "
             f"{ag['req_s']:.2f} req/s; P/D TTFT p90 {fmt_ms(pd['ttft_p90_ms'])}, ITL p90 {pd['itl_p90_ms']:.1f} ms, {pd['req_s']:.2f} req/s.",
             ["data/points.json vllm-tp8ep8-fp8kv/AG-c256-t8 (complete), vllm-pd-1p1d-fp8kv/AG-c256-t8 (stopped at its time cap, 2,152 of 2,208 turns)",
              "data/nixl.json per-transfer time for the two P/D agentic FP8 points", "Step widths are schematic; the 8,192-token step budget is the vLLM default"], still=96)
    header(a, "Why P/D steadies streaming but stretches time to first token (256 sessions, FP8 KV)",
           "Each box is one engine step; every running request gets one token per step, so a longer step means a longer gap between tokens")
    X0, X1 = 150, 930
    for y, l1, l2 in [(80, "Aggregated", "8 GPUs, one engine"), (200, "P/D prefill", "4 GPUs (TP4+EP4)"), (284, "P/D decode", "4 GPUs (TP4+EP4)")]:
        a.add(T(24, y + 22, l1, "ink h"), T(24, y + 40, l2, "mut s"), L(X0, y + 56, X1, y + 56, "axis", 1))
    reveal = lambda x: 2 + 86 * (x - X0) / (X1 - X0)
    xs = X0
    for pflag in [1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1]:
        dw, pw = 18, (34 if pflag else 0)
        if xs + dw + pw > X1:
            break
        base = R(xs, 84, dw, 48, "c1", 2) + (R(xs + dw, 84, pw, 48, "c2", 2) if pw else "")
        a.add(f'<g opacity="0.22">{base}</g>'); a.show([(reveal(xs), 100)], base)
        xs += dw + pw + 4
    a.add(R(X0, 146, 12, 12, "c1", 2), T(X0 + 18, 156, "decode tokens (one per running request)", "mut s"))
    a.add(R(X0 + 290, 146, 12, 12, "c2", 2), T(X0 + 308, 156, "prefill chunk (up to 8,192 prompt tokens per step)", "mut s"))
    xs = X0
    while xs + 52 <= X1:
        base = R(xs, 204, 52, 48, "c2", 2)
        a.add(f'<g opacity="0.22">{base}</g>'); a.show([(reveal(xs), 100)], base); xs += 56
    xs = X0
    while xs + 18 <= X1:
        base = R(xs, 288, 18, 48, "c1", 2)
        a.add(f'<g opacity="0.22">{base}</g>'); a.show([(reveal(xs), 100)], base); xs += 22
    for xx in (X0 + 40, X0 + 520, X0 + 720):
        a.show([(reveal(xx), 100)], L(xx, 254, xx, 284, "s3", 2) + f'<path d="M{xx - 5},278 L{xx},286 L{xx + 5},278 Z" class="c3"/>')
    a.add(T(X0 + 52, 274, f"KV hand-off over NVLink: {x256:.0f} ms per transfer here ({x64:.0f} ms at 64 sessions), measured per stage", "mut s"))
    a.add(T(24, 376, "P/D prefill queue", "ink h"), T(24, 394, f"waiting p90 {pd['wait_p90']:.0f} (aggregated {ag['wait_p90']:.0f})", "mut s"))
    a.add(C(X0 + 210, 382, 7, "c2"))
    for i in range(1, 26):
        a.show([(4 + i * 3.4, 100)], C(X0 + 210 + i * 22, 382, 7, "c2"))
    # measured results box, whole request, not attached to any lane
    a.add(R(24, 414, 912, 70, "pan", 6), T(40, 436, "Measured end to end (client side, whole request):", "ink h"))
    a.add(T(40, 458, f"Aggregated: TTFT p90 {fmt_ms(ag['ttft_p90_ms'])}, ITL p90 {ag['itl_p90_ms']:.1f} ms (over the 50 ms SLO), {ag['out_tok_s_gpu']:.0f} output tok/s/GPU", "ink"))
    a.add(T(40, 476, f"P/D 1P:1D: TTFT p90 {fmt_ms(pd['ttft_p90_ms'])} (lower bound, run hit its time cap), ITL p90 {pd['itl_p90_ms']:.1f} ms, {pd['out_tok_s_gpu']:.0f} output tok/s/GPU", "ink"))
    k = a.frames([(0, "transform:translateX(0px);opacity:1"), (2, "transform:translateX(0px);opacity:1"), (88, f"transform:translateX({X1 - X0}px);opacity:1"),
                  (89, f"transform:translateX({X1 - X0}px);opacity:0"), (100, f"transform:translateX({X1 - X0}px);opacity:0")])
    a.add(f'<g class="{k} motion"><line x1="{X0}" y1="70" x2="{X0}" y2="346" class="sr" stroke-width="2" stroke-dasharray="4 3"/></g>')
    footer(a, ["Box widths are schematic; only the numbers are measured. Both runs FP8 KV; the only change is the topology. Single runs.",
               "Qwen3-235B-A22B-Instruct-2507 FP8 | 8x H100 SXM5 | Dynamo 1.6.0 + vLLM 0.31.0 | NIXL 1.4.1"])
    a.write("a03-agg-step-vs-pd.svg")


# ------------------------------------------------------------------ a04 MoE block refinement
def a04():
    tp, ep = P("vllm-tp8-matrix/S2-c16"), P("vllm-tp8ep8/S2-c16")
    a = Anim(16, "Qwen3-235B FP8 on pure TP8: 1536 / 8 = 192 forces [64,64] scale blocks and a default Triton config; TP8+EP8 keeps native [128,128]",
             "Left: pure TP8 splits each expert's 1536-wide intermediate dimension into 8 slices of 192. 192 is not a multiple of the checkpoint's "
             "128-wide FP8 scale block, so vLLM refines the scales to [64,64] blocks, which only its Triton MoE kernel accepts, and vLLM 0.31.0 ships no "
             "tuned Triton config for that shape on H100, so it logs that it uses a default config. Right: TP8 attention plus EP8 experts gives each GPU "
             f"16 whole experts, native [128,128] blocks and the FLASHINFER_CUTLASS kernel. Measured on the prefill-heavy S2 shape at 16 concurrent "
             f"requests: {tp['req_s']:.2f} to {ep['req_s']:.2f} req/s, TTFT p90 {fmt_ms(tp['ttft_p90_ms'])} to {fmt_ms(ep['ttft_p90_ms'])}, "
             f"ITL p90 {tp['itl_p90_ms']:.1f} to {ep['itl_p90_ms']:.1f} ms.",
             ["data/points.json vllm-tp8-matrix/S2-c16, vllm-tp8ep8/S2-c16", "data/engine-startup.json moe_backend and moe_block_refinement for vllm-tp8 and vllm-tp8ep8",
              "Model config: 128 experts, expert intermediate size 1536"], still=97)
    header(a, "Why pure TP8 runs this FP8 MoE on an untuned kernel, and TP8+EP8 does not",
           "Expert FFN intermediate size 1536; the checkpoint's FP8 weight scales cover [128 x 128] blocks")
    LX, RX, CW = 24, 492, 444
    a.add(R(LX, 66, CW, 384, "pan", 8), R(RX, 66, CW, 384, "pan", 8))
    a.add(T(LX + 14, 90, "Pure TP8 (each expert sliced 8 ways)", "ink h"), T(RX + 14, 90, "TP8 attention + EP8 experts (whole experts per GPU)", "ink h"))
    sy = [104, 172, 236, 300, 356]
    seg = 400 / 8
    a.add(T(LX + 14, sy[0] + 14, "1536 / 8 GPUs = 192 columns per GPU", "ink"))
    for i in range(8):
        k = a.frames([(0, "transform:translateX(0px)"), (4, "transform:translateX(0px)"), (9, f"transform:translateX({(i - 3.5) * 3:.1f}px)"),
                      (96, f"transform:translateX({(i - 3.5) * 3:.1f}px)"), (100, "transform:translateX(0px)")])
        a.add(f'<g class="{k}">{R(LX + 22 + i * seg, sy[0] + 24, seg - 1, 22, "c1", 2)}{T(LX + 22 + i * seg + seg / 2, sy[0] + 40, "192", "w s", "middle")}</g>')
    a.add(T(LX + 14, sy[1] + 14, "192 is not a multiple of the 128-wide scale block:", "ink"))
    a.add(R(LX + 22, sy[1] + 24, 204.8, 22, "c1", 2), T(LX + 22 + 102.4, sy[1] + 40, "128", "w s", "middle"))
    a.add(R(LX + 228.8, sy[1] + 24, 102.4, 22, "cr", 2), T(LX + 280, sy[1] + 40, "64 left over", "w s", "middle"))
    a.add(T(LX + 14, sy[2] + 14, "vLLM refines the scales: gcd(128, 192) = 64, block [64, 64]", "ink"))
    for i in range(3):
        a.add(R(LX + 22 + i * 104.4, sy[2] + 24, 102.4, 22, "c3", 2), T(LX + 73 + i * 104.4, sy[2] + 40, "64", "w s", "middle"))
    a.add(T(LX + 14, sy[3] + 14, "Only the Triton MoE kernel accepts [64,64] blocks", "ink"), T(LX + 14, sy[3] + 34, "→ TRITON FP8 MoE backend", "ink h"))
    a.add(T(LX + 14, sy[4] + 10, "Tuned-config lookup for this shape on H100:", "ink"))
    a.add(T(LX + 14, sy[4] + 28, "E=128,N=192,device_name=NVIDIA_H100_80GB_HBM3,", "ink mono"), T(LX + 14, sy[4] + 44, "dtype=fp8_w8a8,block_shape=[64,64].json", "ink mono"))
    a.add(T(LX + 14, sy[4] + 70, "not shipped → vLLM logs 'Using default MoE config'", "cr h"))
    a.add(T(RX + 14, sy[0] + 14, "128 experts / 8 GPUs = 16 whole experts per GPU", "ink"))
    for i in range(16):
        a.add(R(RX + 22 + i * 25, sy[0] + 24, 22, 22, "c1", 2))
    a.add(T(RX + 14, sy[1] + 14, "each keeps 1536 = 12 x 128: native [128, 128] blocks", "ink"))
    for i in range(12):
        a.add(R(RX + 22 + i * 33.5, sy[1] + 24, 32, 22, "c3", 2))
    a.add(T(RX + 14, sy[2] + 14, "no refinement and no default-config warning in the log", "ink"))
    a.add(T(RX + 14, sy[3] + 14, "native 128 blocks keep the faster kernels eligible", "ink"), T(RX + 14, sy[3] + 34, "→ FLASHINFER_CUTLASS FP8 MoE backend", "ink h"))
    a.add(T(RX + 14, sy[4] + 10, "Measured end to end, S2 (16,384 in / 256 out), 16 concurrent:", "ink"))
    a.add(T(RX + 14, sy[4] + 34, f"{tp['req_s']:.2f} → {ep['req_s']:.2f} req/s (+{(ep['req_s'] / tp['req_s'] - 1) * 100:.0f}%)", "cg big"))
    a.add(T(RX + 14, sy[4] + 56, f"TTFT p90 {fmt_ms(tp['ttft_p90_ms'])} → {fmt_ms(ep['ttft_p90_ms'])}, ITL p90 {tp['itl_p90_ms']:.1f} → {ep['itl_p90_ms']:.1f} ms", "ink h"))
    order = []
    for i, y in enumerate(sy):
        order += [(LX + 6, y - 4), (RX + 6, y - 4)]
    n = len(order); fr = [(0, f"transform:translate({order[0][0]}px,{order[0][1]}px);opacity:0")]
    for j, (x, y) in enumerate(order):
        p0 = 2 + j * (90 / n)
        fr += [(p0, f"transform:translate({x}px,{y}px);opacity:1"), (p0 + 90 / n - 0.8, f"transform:translate({x}px,{y}px);opacity:1")]
    fr += [(94, f"transform:translate({order[-1][0]}px,{order[-1][1]}px);opacity:0"), (100, f"transform:translate({order[-1][0]}px,{order[-1][1]}px);opacity:0")]
    k = a.frames(fr)
    a.add(f'<g class="{k} motion" opacity="0"><rect x="0" y="0" width="{CW - 12}" height="58" rx="6" fill="none" class="sy" stroke-width="3"/></g>')
    footer(a, ["The end-to-end gain also includes the change in MoE communication (all-to-all instead of all-reduce), so it is not the kernel alone.",
               "Qwen3-235B-A22B-Instruct-2507 FP8 | 8x H100 SXM5 | Dynamo 1.6.0 + vLLM 0.31.0 | single runs"])
    a.write("a04-moe-block-refinement.svg")


# ------------------------------------------------------------------ a05 session growth (new)
def a05():
    kb, kf = "vllm-tp8ep8/AG-c256-t8", "vllm-tp8ep8-fp8kv/AG-c256-t8"
    bf, f8 = per_turn(kb, 8), per_turn(kf, 8)
    pb, pf = kv_tokens("vllm-tp8ep8"), kv_tokens("vllm-tp8ep8-fp8kv")
    jump = max(range(1, 9), key=lambda t: bf[t]["ttft90"] / bf[t - 1]["ttft90"])
    a = Anim(16, "As agentic sessions grow turn by turn, BF16 KV runs out and latency jumps; FP8 KV holds",
             f"Left: one session's prompt grows from {bf[0]['isl']:,} tokens at turn 0 to {bf[7]['isl']:,} at turn 7, because every turn re-sends the "
             f"2,000-token shared prompt and the whole history. Right: at 256 concurrent sessions on TP8+EP8, per-turn TTFT p90 and ITL p90. With BF16 KV "
             f"({pb:,}-token pool) TTFT p90 jumps from {fmt_ms(bf[jump - 1]['ttft90'])} at turn {jump - 1} to {fmt_ms(bf[jump]['ttft90'])} at turn {jump} and ITL p90 "
             f"passes 150 ms. With FP8 KV ({pf:,}-token pool) both stay flat: TTFT p90 {fmt_ms(min(v['ttft90'] for v in f8.values()))} to "
             f"{fmt_ms(max(v['ttft90'] for v in f8.values()))}, ITL p90 {min(v['itl90'] for v in f8.values()):.0f} to {max(v['itl90'] for v in f8.values()):.0f} ms.",
             [f"data/{kb}/records.jsonl.gz and data/{kf}/records.jsonl.gz: per-turn median ISL, TTFT p90 and ITL p90 (nearest rank)",
              "data/engine-startup.json kv_tokens for vllm-tp8ep8 and vllm-tp8ep8-fp8kv"], still=96)
    header(a, "Sessions grow every turn; at 256 sessions BF16 KV runs out mid-session",
           "TP8+EP8, 256 concurrent agentic sessions; same flags except the KV cache dtype; both runs finished every planned turn")
    # left: prompt growth
    a.add(R(24, 70, 400, 384, "pan", 8), T(40, 94, "One session's prompt per turn (tokens)", "ink h"))
    xs = lambda v: 40 + v / 9000 * 300
    tmax = 7
    for t in range(tmax + 1):
        y = 110 + t * 40; t0 = 4 + t * 10
        isl = bf[t]["isl"]
        hist = max(isl - 2000 - 600, 0)
        bar = (R(xs(0), y, xs(2000) - xs(0), 22, "c4", 1) + R(xs(2000), y, xs(2000 + hist) - xs(2000), 22, "c1", 1)
               + R(xs(2000 + hist), y, xs(isl) - xs(2000 + hist), 22, "c3", 1) + T(xs(isl) + 6, y + 16, f"{isl:,}", "ink s"))
        a.add(T(34, y + 16, f"t{t}", "mut s", "end"))
        a.show([(t0, 100)], bar)
    a.add(R(40, 428, 12, 12, "c4", 2), T(58, 438, "shared prompt", "mut s"), R(150, 428, 12, 12, "c1", 2), T(168, 438, "history", "mut s"),
          R(230, 428, 12, 12, "c3", 2), T(248, 438, "new input (600)", "mut s"))
    # right: two mini charts
    def chart(y0, h, key, title, hi, ticks, slo, fmt):
        x0, x1 = 500, 910
        xs2 = lambda t: x0 + t / 8 * (x1 - x0)
        ys2 = lambda v: y0 + h - min(v, hi) / hi * h
        a.add(T(x0 - 30, y0 - 10, title, "ink h"))
        for tk in ticks:
            a.add(L(x0, ys2(tk), x1, ys2(tk), "grid"), T(x0 - 6, ys2(tk) + 4, fmt(tk), "mut s", "end"))
        a.add(L(x0, ys2(slo), x1, ys2(slo), "sr", 1.4, 'stroke-dasharray="5 4"'))
        a.add(L(x0, y0 + h, x1, y0 + h))
        for t in range(9):
            a.add(T(xs2(t), y0 + h + 15, f"t{t}", "mut xs", "middle"))
        for d, col, shp in [(bf, 2, "s"), (f8, 3, "^")]:
            prev = None
            for t in range(9):
                v = d[t][key]; t0 = 4 + t * 10
                seg = (L(prev[0], prev[1], xs2(t), ys2(v), f"s{col}", 2) if prev else "")
                lab = ""
                if d is bf and t == jump:
                    lab = T(xs2(t), ys2(v) - 10, fmt_ms(v) if key == "ttft90" else f"{v:.0f} ms", "ink xs b halo", "middle")
                a.show([(t0, 100)], seg + marker(shp, xs2(t), ys2(v), col, size=4.5) + lab)
                prev = (xs2(t), ys2(v))
    chart(100, 118, "ttft90", "TTFT p90 (s)", 32000, [0, 10000, 20000, 30000], 2000, lambda v: f"{v / 1000:.0f}")
    chart(276, 118, "itl90", "ITL p90 (ms)", 190, [0, 50, 100, 150], 50, lambda v: f"{v:.0f}")
    a.add(marker("s", 476, 428, 2, size=5), T(486, 432, f"BF16 KV, {pb / 1e6:.2f}M-token pool", "ink s"),
          marker("^", 676, 428, 3, size=5), T(686, 432, f"FP8 KV, {pf / 1e6:.2f}M-token pool", "ink s"))
    a.add(L(470, 448, 496, 448, "sr", 1.4, 'stroke-dasharray="5 4"'), T(502, 452, "SLO: TTFT 2 s, ITL 50 ms", "mut s"))
    kb_max, kf_max = P(kb)["kv_max"], P(kf)["kv_max"]
    a.show([(90, 100)], T(24, 482, f"Measured KV peak: {kb_max:.2f} of the BF16 pool (it filled), {kf_max:.2f} of the FP8 pool (it never filled). "
                                   f"Same layout, same 2,208 turns.", "ink"))
    footer(a, ["Per-turn p90 over all requests at that turn index (n = 256 at t0 to 130 at t8). FP8 KV used an uncalibrated scale, so output quality is not verified.",
               "Qwen3-235B-A22B-Instruct-2507 FP8 | 8x H100 SXM5 | Dynamo 1.6.0 + vLLM 0.31.0 | single runs"])
    a.write("a05-session-growth.svg")


# ------------------------------------------------------------------ a06 operating frontier (new)
def a06():
    pts = [p for k, p in sorted(points().items()) if k.split("/")[1].endswith("-t8")]
    front = [p for p in pts if not any(q is not p and q["out_tok_s_gpu"] >= p["out_tok_s_gpu"] and q["tok_s_user_p90"] >= p["tok_s_user_p90"]
                                       and (q["out_tok_s_gpu"] > p["out_tok_s_gpu"] or q["tok_s_user_p90"] > p["tok_s_user_p90"]) for q in pts)]
    front.sort(key=lambda p: p["tok_s_user_p90"])
    lo, hi = front[-1], front[0]  # most interactive, most throughput
    a = Anim(16, "Measured operating frontier for agentic serving on one 8x H100 node: throughput per GPU versus per-user speed",
             f"Each of the {len(pts)} eight-turn agentic points is plotted as output tokens per second per GPU against tokens per second per user "
             f"(1000 / ITL p90). Solid markers meet the target: at least {POINT_OK_PCT:.0f}% of requests met TTFT 2 s and ITL 50 ms. Hollow markers did not. "
             f"Only two points are on the frontier: "
             f"TP8+EP8 BF16 KV at 64 sessions ({lo['tok_s_user_p90']:.1f} tok/s/user, {lo['out_tok_s_gpu']:.1f} tok/s/GPU, {lo['slo_ok_pct']:.0f}% in SLO) and TP8+EP8 "
             f"FP8 KV at 256 sessions ({hi['tok_s_user_p90']:.1f} tok/s/user, {hi['out_tok_s_gpu']:.1f} tok/s/GPU, {hi['slo_ok_pct']:.1f}% in SLO). "
             f"Concurrencies 96 to 192 were not measured, so the curve between them is unknown. Dashed lines mark 20 tok/s/user (the 50 ms ITL SLO) "
             f"and the InferenceX reference tiers of 35 and 50 tok/s/user.",
             ["data/points.json: every row whose point name ends in -t8 (out_tok_s_gpu, tok_s_user_p90, slo_ok_pct, ttft_p90_ms)",
              "Reference tiers: InferenceX calculator default 35 tok/s/user and default tier 50 tok/s/user (docs/07-how-this-compares.md)"], still=96)
    header(a, "The measured operating frontier: two points, and a gap between them",
           "8-turn agentic sessions on one 8x H100 node. x = tok/s per user at p90 (1000 / ITL p90), y = output tok/s per GPU")
    x0, x1, y0, y1 = 110, 680, 92, 420
    xs = lambda v: x0 + (v - 0) / 50 * (x1 - x0)
    ys = lambda v: y1 - v / 450 * (y1 - y0)
    for t in range(0, 51, 10):
        a.add(L(xs(t), y0, xs(t), y1, "grid"), T(xs(t), y1 + 18, f"{t}", "mut s", "middle"))
    for t in range(0, 451, 100):
        a.add(L(x0, ys(t), x1, ys(t), "grid"), T(x0 - 8, ys(t) + 4, f"{t}", "mut s", "end"))
    a.add(L(x0, y1, x1, y1), L(x0, y0, x0, y1))
    a.add(T((x0 + x1) / 2, y1 + 38, "tok/s per user at p90 (1000 / ITL p90): higher = faster streaming for the slowest 10%", "mut s", "middle"))
    a.add(T(x0 - 52, (y0 + y1) / 2, "output tok/s per GPU", "mut s", "middle", f'transform="rotate(-90 {x0 - 52} {(y0 + y1) / 2})"'))
    for v, lab in [(20, "20: ITL 50 ms (our SLO)"), (35, "35: InferenceX calculator default"), (50, "50: InferenceX default tier")]:
        a.add(L(xs(v), y0, xs(v), y1, "sr" if v == 20 else "s4", 1.3, 'stroke-dasharray="5 4"'))
        a.add(T(xs(v) - 4, y1 - 8 - (16 if v == 35 else 0), lab, ("cr" if v == 20 else "c4") + " xs halo", "end"))
    order = sorted(pts, key=lambda p: (p["concurrency"], p["run"]))
    style = {"TP8": (1, "o"), "TP8+EP8": (2, "s"), "DP8+EP8": (4, "o")}
    for i, p in enumerate(order):
        t0 = 4 + i * 2.8
        col, shp = style.get(p["layout"], (4, "^"))
        if p["kv_dtype"] == "fp8" and p["layout"] == "TP8+EP8":
            col, shp = 3, "^"
        if p["layout"].startswith("P/D"):
            col, shp = 4, "^" if p["kv_dtype"] == "fp8" else "s"
        if p["engine"] == "trtllm":
            col, shp = 2, "o"
        good = p["slo_ok_pct"] >= POINT_OK_PCT
        a.show([(t0, 100)], marker(shp, xs(p["tok_s_user_p90"]), ys(p["out_tok_s_gpu"]), col, hollow=not good, size=6))
    # frontier line and gap
    pl = " ".join(f"{xs(p['tok_s_user_p90']):.1f},{ys(p['out_tok_s_gpu']):.1f}" for p in front)
    a.show([(62, 100)], f'<polyline points="{pl}" fill="none" class="sy" stroke-width="3" stroke-dasharray="2 5" stroke-linecap="round"/>')
    ly = ys(lo["out_tok_s_gpu"]) + 82
    a.show([(62, 100)], L(xs(lo["tok_s_user_p90"]), ys(lo["out_tok_s_gpu"]) + 8, 672, ly - 14, "axis", 1)
           + T(676, ly, f"64 sessions, BF16 KV: {lo['out_tok_s_gpu']:.0f} tok/s/GPU,", "ink s b halo", "end")
           + T(676, ly + 16, f"{lo['tok_s_user_p90']:.0f} tok/s/user, TTFT p90 {fmt_ms(lo['ttft_p90_ms'])}, {lo['slo_ok_pct']:.0f}% in SLO", "ink xs halo", "end"))
    a.show([(62, 100)], T(xs(hi["tok_s_user_p90"]) + 10, ys(hi["out_tok_s_gpu"]) - 4, f"256 sessions, FP8 KV: {hi['out_tok_s_gpu']:.0f} tok/s/GPU, {hi['tok_s_user_p90']:.0f} tok/s/user,", "ink s b halo")
           + T(xs(hi["tok_s_user_p90"]) + 10, ys(hi["out_tok_s_gpu"]) + 12, f"TTFT p90 {fmt_ms(hi['ttft_p90_ms'])}, only {hi['slo_ok_pct']:.0f}% in SLO", "ink xs halo"))
    mx, my = (xs(lo["tok_s_user_p90"]) + xs(hi["tok_s_user_p90"])) / 2, (ys(lo["out_tok_s_gpu"]) + ys(hi["out_tok_s_gpu"])) / 2
    a.show([(74, 100)], C(mx, my, 26, "cy", 'fill-opacity=".18"') + T(mx - 30, my + 44, "96 to 192 sessions:", "cy s b", "middle")
           + T(mx - 30, my + 60, "not measured", "cy s b", "middle"))
    # right-hand reading panel
    px = 708
    a.add(R(px, 84, 236, 336, "pan", 8), T(px + 14, 108, "How to read it", "ink h"))
    lines = ["Up = more output per GPU", "(cheaper tokens).", "Right = faster streaming", "for the slowest 10% of users.",
             f"Solid = meets the target:", f"at least {POINT_OK_PCT:.0f}% of requests had", "TTFT ≤ 2 s and ITL ≤ 50 ms.", "Hollow = fewer.", "",
             "Every point past 64", "sessions is hollow: the", "last load that meets the", "target lies between 64", "and 256 sessions."]
    for i, s in enumerate(lines):
        a.add(T(px + 14, 132 + i * 19, s, "ink s" if s else "ink s"))
    leg = [("TP8, BF16 KV", 1, "o"), ("TP8+EP8, BF16 KV", 2, "s"), ("TP8+EP8, FP8 KV", 3, "^"), ("TRT-LLM TP8+EP8", 2, "o"), ("DP8+EP8", 4, "o"), ("P/D, FP8 KV", 4, "^"), ("P/D, BF16 KV", 4, "s")]
    for i, (lab, c, s) in enumerate(leg):
        lx = 24 + i * 132
        a.add(marker(s, lx + 6, 488, c, size=5), T(lx + 16, 492, lab, "ink xs"))
    footer(a, ["Points: every 8-turn agentic run (16 to 1,000 sessions). Per-GPU values divide by all 8 GPUs. FP8 KV used an uncalibrated scale (quality not verified).",
               "Qwen3-235B-A22B-Instruct-2507 FP8 | 8x H100 SXM5 | Dynamo 1.6.0 | vLLM 0.31.0 and TensorRT-LLM 1.3.0rc29 | single runs"])
    a.write("a06-operating-frontier.svg")


if __name__ == "__main__":
    a01(); a02(); a03(); a04(); a05(); a06()
