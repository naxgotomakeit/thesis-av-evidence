"""Human-first HTML for photometric-robust Fine-to-Medium frontiers."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


METHOD_LABELS = {
    "current_fluid_loose": "A Current Fluid Loose",
    "local_absolute_strict": "B1 Local Absolute Strict",
    "local_absolute_balanced": "B2 Local Absolute Balanced",
    "local_absolute_loose": "B3 Local Absolute Loose",
    "photo_strict": "C1 Photo Strict",
    "photo_balanced": "C2 Photo Balanced",
    "photo_loose": "C3 Photo Loose",
    "photo_temporal_strict": "D1 Photo + Temporal Strict",
    "photo_temporal_balanced": "D2 Photo + Temporal Balanced",
    "photo_temporal_loose": "D3 Photo + Temporal Loose",
}


def e(value: Any) -> str:
    return html.escape(str(value))


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def review_controls(prefix: str) -> str:
    return f"""
    <div class="review"><b>Human review</b>
      <label><input type="radio" name="{e(prefix)}_judgment"> Too fragmented</label>
      <label><input type="radio" name="{e(prefix)}_judgment"> Natural Medium grouping</label>
      <label><input type="radio" name="{e(prefix)}_judgment"> Overmerged</label>
      <label><input type="radio" name="{e(prefix)}_judgment"> Unsure</label><br>
      <label>Reason <select><option>unreviewed</option><option>lighting/flicker artifact</option><option>camera motion</option><option>real scene change</option><option>real action/state change</option><option>repetitive/stable region</option><option>unclear</option></select></label>
      <label>Notes <textarea></textarea></label>
    </div>"""


def timeline(ids: list[str], nodes: dict[str, dict[str, Any]], duration: float, start: float = 0.0, end: float | None = None) -> str:
    end = duration if end is None else end
    span = max(end - start, 1e-9)
    blocks = []
    for index, node_id in enumerate(ids):
        node = nodes[node_id]
        left = max(start, float(node["start"]))
        right = min(end, float(node["end"]))
        if right <= left:
            continue
        blocks.append(
            f'<span class="block b{index % 6}" style="left:{100*(left-start)/span:.4f}%;width:{100*(right-left)/span:.4f}%" '
            f'title="{e(node_id)} {float(node["start"]):.1f}-{float(node["end"]):.1f}s"></span>'
        )
    return '<div class="timeline">' + "".join(blocks) + "</div>"


def node_cards(ids: list[str], nodes: dict[str, dict[str, Any]], assets: dict[str, dict[str, Any]]) -> str:
    cards = []
    for node_id in ids:
        node = nodes[node_id]
        asset = assets[node_id]
        cards.append(
            f'<article class="node"><img loading="lazy" src="{e(asset["html_relative_path"])}">'
            f'<b>{e(node_id)}</b><span>{float(node["start"]):.1f}-{float(node["end"]):.1f}s · {float(node["duration"]):.1f}s</span>'
            f'<small>representative {float(asset["timestamp"]):.1f}s</small></article>'
        )
    return '<div class="nodes">' + "".join(cards) + "</div>"


def summary_table(per_video: list[dict[str, Any]]) -> str:
    rows = []
    for item in per_video:
        metrics = item["metrics"]
        reduction = item["projected_reduction_vs_current_ratio"]
        rows.append(
            "<tr>"
            f'<td>{e(item["video_id"])}</td><td>{e(METHOD_LABELS[item["method"]])}</td>'
            f'<td>{item["fine_count"]}</td><td>{metrics["medium_count"]}</td><td>{metrics["medium_count"]}</td>'
            f'<td>{100*reduction:.1f}%</td><td>{metrics["median_duration_sec"]:.1f}s</td>'
            f'<td>{fmt(metrics["adjacent_similarity_median"])}</td><td>{metrics["overmerge_risk_flag_count"]}</td></tr>'
        )
    return """<div class="table-scroll"><table><thead><tr><th>Video</th><th>Method</th><th>Fine</th><th>Medium</th><th>Projected VLM calls</th><th>Reduction vs A</th><th>Median Medium</th><th>Adjacent similarity</th><th>Risk flags</th></tr></thead><tbody>""" + "".join(rows) + "</tbody></table></div>"


def flash_section(flash: dict[str, Any], hierarchies: dict[str, dict[str, Any]], assets: dict[str, dict[str, dict[str, Any]]]) -> str:
    hierarchy = hierarchies[flash["video_id"]]
    nodes = {row["node_id"]: row for row in hierarchy["nodes"]}
    rows = []
    for item in flash["methods"]:
        ids = item["medium_ids_in_interval"]
        rows.append(
            f'<article class="method-row"><h3>{e(METHOD_LABELS[item["method"]])}</h3>'
            f'<p><b>{len(ids)} Mediums</b> · photo crossed {item["suspected_photometric_boundaries_crossed"]} · '
            f'persistent crossed {item["persistent_boundaries_crossed"]} · risk flags {item["risk_flags"]}</p>'
            + timeline(ids, nodes, float(hierarchy["video_duration"]), flash["start_sec"], flash["end_sec"])
            + node_cards(ids, nodes, assets[flash["video_id"]])
            + review_controls(f'flash_{item["method"]}') + "</article>"
        )
    diag_rows = "".join(
        f'<tr><td>{item["timestamp"]:.1f}</td><td>{fmt(item["original_dino_change"])}</td><td>{fmt(item["normalized_dino_change"])}</td>'
        f'<td>{fmt(item["luminance_mean_change"])}</td><td>{fmt(item["color_histogram_change"])}</td>'
        f'<td>{e(item["suspected_photometric_by_profile"])}</td><td>{item["persistent_state_change"]}</td><td>{item["transient_or_reverting_change"]}</td></tr>'
        for item in flash["boundary_diagnostics"]
    )
    return f"""<section id="flash"><h2>Targeted flashing-light review — FIRST</h2>
    <p><b>{e(flash['video_id'])}</b> · {flash['start_sec']:.1f}-{flash['end_sec']:.1f}s. This video was selected deterministically by maximum frozen Fixed-Reference overlap, not by inspecting new results.</p>
    {''.join(rows)}
    <details><summary>Advanced boundary diagnostics</summary><div class="table-scroll"><table><tr><th>t</th><th>Original DINO change</th><th>Normalized DINO change</th><th>Luminance Δ</th><th>Color Δ</th><th>Photo flags</th><th>Persistent</th><th>Transient</th></tr>{diag_rows}</table></div></details></section>"""


def transition_section(title: str, rows: list[dict[str, Any]], kind: str) -> str:
    cards = []
    for index, item in enumerate(rows):
        cards.append(
            f'<article class="transition"><h3>{e(item["video_id"])} · {float(item.get("timestamp", item.get("start", 0))):.1f}s</h3>'
            f'<div class="pair"><figure><img loading="lazy" src="{e(item["left_asset"])}"><figcaption>before</figcaption></figure>'
            f'<figure><img loading="lazy" src="{e(item["right_asset"])}"><figcaption>after</figcaption></figure></div>'
            f'<p>Original Δ {fmt(item.get("original_dino_change"))} · Normalized Δ {fmt(item.get("normalized_dino_change"))} · persistent {e(item.get("persistent_state_change"))}</p>'
            + review_controls(f'{kind}_{index}') + "</article>"
        )
    return f'<section><h2>{e(title)}</h2><div class="transition-grid">{"".join(cards)}</div></section>'


def full_video_sections(
    per_video: list[dict[str, Any]], hierarchies: dict[str, dict[str, Any]],
    assets: dict[str, dict[str, dict[str, Any]]], method_order: list[str],
) -> str:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in per_video:
        grouped.setdefault(item["video_id"], []).append(item)
    sections = []
    for video_id, items in grouped.items():
        hierarchy = hierarchies[video_id]
        nodes = {row["node_id"]: row for row in hierarchy["nodes"]}
        by_method = {item["method"]: item for item in items}
        methods = []
        for method in method_order:
            item = by_method[method]
            ids = item["medium_ids"]
            methods.append(
                f'<details class="method"><summary>{e(METHOD_LABELS[method])} · {len(ids)} Mediums · median {item["metrics"]["median_duration_sec"]:.1f}s</summary>'
                + timeline(ids, nodes, float(hierarchy["video_duration"]))
                + node_cards(ids, nodes, assets[video_id])
                + review_controls(f'full_{video_id}_{method}')
                + '<details><summary>Advanced numerical diagnostics / decision trace</summary><pre>'
                + e(json.dumps({"metrics": item["metrics"], "decision_trace": item["decision_trace"]}, indent=2))
                + "</pre></details></details>"
            )
        sections.append(
            f'<section id="v_{e(video_id)}"><h2>{e(video_id)}</h2><p>{float(hierarchy["video_duration"]):.1f}s · Fine {len(hierarchy["cuts"]["fine"]["node_ids"])}</p>{"".join(methods)}</section>'
        )
    return "".join(sections)


def build_html(
    *, output_path: Path, per_video: list[dict[str, Any]], hierarchies: dict[str, dict[str, Any]],
    assets: dict[str, dict[str, dict[str, Any]]], flash: dict[str, Any],
    real_transitions: list[dict[str, Any]], stable_regions: list[dict[str, Any]],
    method_order: list[str], runtime: dict[str, Any],
) -> dict[str, Any]:
    nav = " ".join(f'<a href="#v_{e(video_id)}">{e(video_id[:16])}</a>' for video_id in hierarchies)
    css = """
*{box-sizing:border-box}body{margin:0;background:#edf1f4;color:#17212b;font:13px/1.45 system-ui,sans-serif}header{position:sticky;top:0;z-index:10;background:#172b3b;color:white;padding:14px}header a{color:#b9e6ff;margin-right:9px}main{max-width:1600px;margin:auto;padding:16px}section,.summary{background:white;border:1px solid #c9d2da;border-radius:11px;padding:15px;margin:16px 0}table{border-collapse:collapse;width:100%;font-size:11px}th,td{border:1px solid #ccd4dc;padding:5px}th{background:#e6edf2}.table-scroll{overflow:auto}.method-row,.method{border:2px solid #738594;border-radius:8px;padding:9px;margin:10px 0}.timeline{position:relative;height:26px;background:repeating-linear-gradient(to right,#f1f4f6 0,#f1f4f6 calc(10% - 1px),#b9c4cc calc(10% - 1px),#b9c4cc 10%)}.block{position:absolute;height:24px;top:1px;border-right:2px solid white}.b0{background:#4c78a8}.b1{background:#72b7b2}.b2{background:#f58518}.b3{background:#e45756}.b4{background:#54a24b}.b5{background:#b279a2}.nodes{display:flex;gap:7px;overflow-x:auto;padding:8px 0}.node{min-width:155px;border:1px solid #bfc9d1;border-radius:6px;padding:5px;display:flex;flex-direction:column}.node img{width:145px;height:95px;object-fit:cover}.node small,.node span{font-size:10px}.review{background:#f6f8fa;border:1px dashed #9aa8b3;padding:7px}.review label{margin-right:10px}.review textarea{display:block;width:100%;height:35px}.transition-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:8px}.transition{border:1px solid #bcc8d1;padding:7px}.pair{display:flex;gap:5px}.pair figure{margin:0}.pair img{width:145px;height:95px;object-fit:cover}pre{white-space:pre-wrap;max-height:500px;overflow:auto;font-size:10px}details summary{cursor:pointer;font-weight:700}#flash{border:4px solid #d47700}
"""
    body = f"""<!doctype html><html><head><meta charset="utf-8"><title>Photometric robust fluid Medium v0.1</title><style>{css}</style></head><body>
    <header><h1>Photometric-robust Fluid Medium v0.1</h1><nav><a href="#flash">Flashing case</a> {nav}</nav></header><main>
    <div class="summary"><h2>Method summary</h2><p>Fine leaves and Safe-Merge topology are frozen. All values are structural diagnostics; no winner is selected and projected VLM calls are counts only.</p>{summary_table(per_video)}
    <details><summary>Runtime</summary><pre>{e(json.dumps(runtime, indent=2))}</pre></details></div>
    {flash_section(flash, hierarchies, assets)}
    {transition_section('Real-transition safety review (structural proxy)', real_transitions, 'real')}
    {transition_section('Stable long-region review', stable_regions, 'stable')}
    <section><h2>Full-video timelines</h2><p>Open each method to inspect all retained Medium nodes. Advanced metrics and decision traces are collapsed by default.</p></section>
    {full_video_sections(per_video, hierarchies, assets, method_order)}
    </main></body></html>"""
    output_path.write_text(body, encoding="utf-8")
    image_refs = [part.split('"', 1)[0] for part in body.split('src="')[1:]]
    missing = [item for item in image_refs if not (output_path.parent / item).is_file()]
    return {
        "video_count": len(hierarchies), "method_video_sections": len(per_video),
        "image_reference_count": len(image_refs), "missing_image_reference_count": len(missing),
        "missing_image_references": missing,
    }
