#!/usr/bin/env python3
"""Render lightweight SVG figures without optional plotting dependencies."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/experiments/qaego4d_e2_closed500_b1_b2_amendment6_v1"

def esc(value: object) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def svg_shell(title: str, body: str, width: int = 760, height: int = 440) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/><text x="30" y="32" font-family="sans-serif" font-size="20" font-weight="bold">{esc(title)}</text>{body}</svg>'''

def axes() -> str:
    return '<line x1="70" y1="390" x2="730" y2="390" stroke="#333"/><line x1="70" y1="70" x2="70" y2="390" stroke="#333"/>'

def main() -> None:
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    fig = OUT / "figures"; fig.mkdir(exist_ok=True)
    methods = ["b0", "b1", "b2"]
    labels = ["B0", "B1", "B2"]
    vals = [summary["accuracy"][m]["accuracy"] for m in methods]
    body = axes()
    for i, (label, value, method) in enumerate(zip(labels, vals, methods)):
        x = 130 + i * 190; h = value * 280; y = 390 - h
        lo, hi = summary["accuracy"][method]["wilson_95ci"]
        body += f'<rect x="{x}" y="{y:.1f}" width="90" height="{h:.1f}" fill="#4c78a8"/><text x="{x+45}" y="410" text-anchor="middle" font-family="sans-serif">{label}</text><text x="{x+45}" y="{y-8:.1f}" text-anchor="middle" font-family="sans-serif">{value:.1%}</text>'
        body += f'<line x1="{x+45}" y1="{390-hi*280:.1f}" x2="{x+45}" y2="{390-lo*280:.1f}" stroke="#222" stroke-width="3"/><line x1="{x+35}" y1="{390-hi*280:.1f}" x2="{x+55}" y2="{390-hi*280:.1f}" stroke="#222"/><line x1="{x+35}" y1="{390-lo*280:.1f}" x2="{x+55}" y2="{390-lo*280:.1f}" stroke="#222"/>'
    body += '<text x="25" y="80" font-family="sans-serif" font-size="12">Accuracy (Wilson 95% CI)</text>'
    (fig / "accuracy_wilson.svg").write_text(svg_shell("Formal E2 Closed-500 Accuracy", body), encoding="utf-8")

    fields = ["retrieval_time_s", "online_cached_frame_read_time_s", "answer_preprocess_time_s", "answer_model_time_s"]
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    body = axes(); bottom = [0.0, 0.0, 0.0]
    for field, color in zip(fields, colors):
        values = [summary["efficiency"][m]["latency"][field]["mean"] or 0.0 for m in methods]
        for i, value in enumerate(values):
            scale = 280 / max(5.0, max(sum(summary["efficiency"][m]["latency"][f]["mean"] or 0.0 for f in fields) for m in methods))
            x = 130 + i * 190; h = value * scale; y = 390 - bottom[i] * scale - h
            body += f'<rect x="{x}" y="{y:.1f}" width="90" height="{h:.1f}" fill="{color}"/>'
            bottom[i] += value
    for i, label in enumerate(labels): body += f'<text x="{175+i*190}" y="410" text-anchor="middle" font-family="sans-serif">{label}</text>'
    for i, field in enumerate(fields): body += f'<rect x="540" y="{90+i*25}" width="14" height="14" fill="{colors[i]}"/><text x="560" y="{102+i*25}" font-family="sans-serif" font-size="12">{esc(field)}</text>'
    (fig / "latency_decomposition.svg").write_text(svg_shell("Mean Online Latency Decomposition", body), encoding="utf-8")

    be = summary["break_even"]["per_clip"]; body = axes()
    for idx, method in enumerate(("b1", "b2")):
        vals = [row["N_star"] for row in be if row["method"] == method and row["N_star"] is not None]
        if not vals: continue
        maximum = max(vals) or 1.0
        for j, value in enumerate(sorted(vals)):
            x = 85 + (j / max(1, len(vals)-1)) * 640; y = 390 - min(280, value / maximum * 280)
            body += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.5" fill="{"#4c78a8" if method=="b1" else "#f58518"}"/>'
        body += f'<text x="560" y="{90+idx*25}" font-family="sans-serif"><tspan fill="{"#4c78a8" if method=="b1" else "#f58518"}">●</tspan> {method.upper()} (n={len(vals)}, mean={sum(vals)/len(vals):.1f})</text>'
    body += '<text x="270" y="420" font-family="sans-serif">sorted clips</text><text x="20" y="80" font-family="sans-serif" font-size="12">N* (offline cost / online saving)</text>'
    (fig / "break_even.svg").write_text(svg_shell("Break-even Queries per Clip", body), encoding="utf-8")

if __name__ == "__main__": main()
