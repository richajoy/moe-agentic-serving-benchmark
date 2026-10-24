"""Tiny SVG helpers shared by make_figures.py and make_animations.py (stdlib only).

Every figure is plain SVG with an internal <style>: a light palette by default, a
prefers-color-scheme: dark override, and prefers-reduced-motion support. No external
fonts, images or scripts, so GitHub renders it through <img>.
"""
import gzip, json, math, os, pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
DATA = REPO / "data"
FIG = REPO / "figures"

CSS = """
  .bg{fill:#fcfcfb} .ink{fill:#16161a} .mut{fill:#55544f} .faint{fill:#8a8985}
  .pan{fill:#f3f3f0;stroke:#d6d5cd} .grid{stroke:#e3e2db} .axis{stroke:#8a8985} .emp{fill:#fcfcfb;stroke:#c3c2b7}
  .c1{fill:#2a78d6} .c2{fill:#eb6834} .c3{fill:#1baf7a} .c4{fill:#6250d6} .cr{fill:#d03b3b} .cg{fill:#0f8a3c} .cy{fill:#eda100}
  .s1{stroke:#2a78d6} .s2{stroke:#eb6834} .s3{stroke:#1baf7a} .s4{stroke:#6250d6} .sr{stroke:#d03b3b} .sg{stroke:#0f8a3c} .sy{stroke:#eda100}
  .sm{stroke:#55544f} .w{fill:#ffffff} .hollow{fill:#fcfcfb}
  .halo{paint-order:stroke;stroke:#fcfcfb;stroke-width:4px;stroke-linejoin:round}
  .band{fill:#6250d6;fill-opacity:.14} .ok{fill:#0f8a3c;fill-opacity:.08}
  text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;font-size:13px}
  .mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px}
  .t{font-size:18px;font-weight:600} .h{font-size:14px;font-weight:600} .s{font-size:12px} .xs{font-size:11.5px}
  .big{font-size:22px;font-weight:700} .b{font-weight:600}
  @media (prefers-color-scheme: dark){
    .bg{fill:#0d1117} .ink{fill:#ececea} .mut{fill:#a8a7a3} .faint{fill:#7d7c78}
    .pan{fill:#161b22;stroke:#30363d} .grid{stroke:#262c34} .axis{stroke:#6e7681} .emp{fill:#0d1117;stroke:#3a3f47}
    .c1{fill:#4a90e8} .c2{fill:#f07a4a} .c3{fill:#2bc48e} .c4{fill:#9085e9} .cr{fill:#ef6b6b} .cg{fill:#3fb950} .cy{fill:#f2b53a}
    .s1{stroke:#4a90e8} .s2{stroke:#f07a4a} .s3{stroke:#2bc48e} .s4{stroke:#9085e9} .sr{stroke:#ef6b6b} .sg{stroke:#3fb950} .sy{stroke:#f2b53a}
    .sm{stroke:#a8a7a3} .hollow{fill:#0d1117} .halo{stroke:#0d1117} .band{fill:#9085e9;fill-opacity:.20} .ok{fill:#3fb950;fill-opacity:.10}
  }
"""

MOTION_CSS = """
  .still{display:none}
  @media (prefers-reduced-motion: reduce){ *{animation:none !important} .still{display:inline} .motion{display:none} }
"""


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def T(x, y, s, cls="ink", anchor="start", extra=""):
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}" {extra}>{esc(s)}</text>'


def R(x, y, w, h, cls, rx=2, extra=""):
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{max(h, 0):.1f}" rx="{rx}" class="{cls}" {extra}/>'


def L(x1, y1, x2, y2, cls="axis", w=1.0, extra=""):
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="{cls}" stroke-width="{w}" {extra}/>'


def C(x, y, r, cls, extra=""):
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" class="{cls}" {extra}/>'


def marker(shape, x, y, color, hollow=False, size=6):
    """shape: o (circle), s (square), ^ (triangle). color: 1..4. Hollow = white fill, colored stroke."""
    fill = "hollow" if hollow else f"c{color}"
    st = f'class="{fill} s{color}" stroke-width="2"'
    if shape == "o":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{size}" {st}/>'
    if shape == "s":
        return f'<rect x="{x - size:.1f}" y="{y - size:.1f}" width="{2 * size}" height="{2 * size}" {st}/>'
    h = size * 1.15
    return f'<path d="M{x:.1f},{y - h:.1f} L{x + h:.1f},{y + h * 0.8:.1f} L{x - h:.1f},{y + h * 0.8:.1f} Z" {st}/>'


class Fig:
    def __init__(self, w, h, title, desc, sources, extra_css=""):
        self.w, self.h, self.title, self.desc, self.sources, self.extra_css = w, h, title, desc, sources, extra_css
        self.body = []

    def add(self, *s):
        self.body.extend(s)

    def svg(self):
        src = "\n".join("  " + s.replace("--", "-") for s in self.sources)
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}" '
                f'role="img" aria-labelledby="title desc">\n'
                f'<title id="title">{esc(self.title)}</title>\n<desc id="desc">{esc(self.desc)}</desc>\n'
                f'<!--\nData sources (every number in this figure is read from these files):\n{src}\n-->\n'
                f'<style>{CSS}{self.extra_css}</style>\n'
                f'<rect class="bg" x="0" y="0" width="{self.w}" height="{self.h}"/>\n' + "\n".join(self.body) + "\n</svg>\n")

    def write(self, rel):
        p = FIG / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.svg())
        print("wrote", rel, p.stat().st_size, "bytes")


class LogScale:
    def __init__(self, lo, hi, a, b):
        self.lo, self.hi, self.a, self.b = math.log10(lo), math.log10(hi), a, b

    def __call__(self, v):
        return self.a + (math.log10(v) - self.lo) / (self.hi - self.lo) * (self.b - self.a)


class LinScale:
    def __init__(self, lo, hi, a, b):
        self.lo, self.hi, self.a, self.b = lo, hi, a, b

    def __call__(self, v):
        return self.a + (v - self.lo) / (self.hi - self.lo) * (self.b - self.a)


_POINTS = None


def points():
    global _POINTS
    if _POINTS is None:
        _POINTS = {f"{p['run']}/{p['point']}": p for p in json.load(open(DATA / "points.json"))["points"]}
    return _POINTS


def P(key):
    return points()[key]


def records(key):
    return [json.loads(l) for l in gzip.open(DATA / key / "records.jsonl.gz", "rt")]


def pct(a, p):
    """Nearest-rank percentile, the same rule as points.json per_turn_ttft."""
    a = sorted(a)
    return a[min(len(a) - 1, int(round(p / 100 * (len(a) - 1))))] if a else None


def lower_bound(p):
    return p["completion_rule"].startswith("duration") and not p["ran_to_completion"]


def fmt_ms(v):
    return f"{v / 1000:.1f} s" if v >= 10000 else (f"{v / 1000:.2f} s" if v >= 2000 else f"{v:,.0f} ms")


CONDITIONS = "Qwen3-235B-A22B-Instruct-2507 FP8 | one node, 8x H100 SXM5 | Dynamo 1.6.0 | single run per point"
