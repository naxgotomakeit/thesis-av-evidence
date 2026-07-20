"""Human-first comparison of seven operating frontiers on frozen trees."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any


LABELS = {
    "fixed_reference": "A Fixed Reference 50% / 25%",
    "fixed_coarser": "B Fixed Coarser 30% / 10%",
    "global_adaptive_elbow": "C Global Adaptive Elbow",
    "fluid_strict": "D Fluid Strict",
    "fluid_balanced": "E Fluid Balanced",
    "fluid_loose": "F Fluid Loose",
    "fluid_balanced_guarded": "G Fluid Balanced Guarded",
}


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _time(value: float) -> str:
    minutes, seconds = divmod(float(value), 60.0)
    return f"{int(minutes):02d}:{seconds:04.1f}"


def _review(prefix: str, *, suspicious: bool = False) -> str:
    if suspicious:
        options = " ".join(
            f'<label><input type="radio" name="{_e(prefix)}_node" value="{value}"> {value}</label>'
            for value in ("should_merge_more", "correct_stop", "overmerged", "unsure")
        )
        return f'<div class="review"><b>Suspicious node review:</b> {options}</div>'
    fields = [
        ("Overall natural?", ["yes", "no", "unsure"]),
        ("Repetitive Mediums remain?", ["low", "medium", "high"]),
        ("Obvious overmerge?", ["low", "medium", "high"]),
        ("Broad storyline structure sensible?", ["yes", "partial", "no"]),
        ("Preferred method for this video?", ["yes", "no", "unsure"]),
    ]
    blocks = []
    for index, (label, values) in enumerate(fields):
        options = " ".join(
            f'<label><input type="radio" name="{_e(prefix)}_{index}" value="{value}"> {value}</label>'
            for value in values
        )
        blocks.append(f'<div><b>{label}</b> {options}</div>')
    blocks.append(f'<label><b>Notes</b><textarea name="{_e(prefix)}_notes"></textarea></label>')
    return '<div class="review">' + "".join(blocks) + '</div>'


def _timeline(ids: list[str], nodes: dict[str, dict[str, Any]], duration: float, css: str) -> str:
    blocks = []
    for node_id in ids:
        node = nodes[node_id]
        left = 100.0 * float(node["start"]) / duration
        width = max(0.12, 100.0 * float(node["duration"]) / duration)
        title = f'{node_id} {_time(node["start"])}-{_time(node["end"])} ({node["duration"]:.1f}s)'
        blocks.append(
            f'<span class="block {css}" style="left:{left:.5f}%;width:{width:.5f}%" title="{_e(title)}"></span>'
        )
    return '<div class="track">' + "".join(blocks) + '</div>'


def _elbow_svg(row: dict[str, Any]) -> str:
    values = row["sorted_merge_costs"]
    width, height, margin = 700, 170, 22
    minimum, maximum = min(values), max(values)
    spread = max(maximum - minimum, 1e-12)
    points = []
    for index, value in enumerate(values):
        x = margin + (width - 2 * margin) * index / max(len(values) - 1, 1)
        y = height - margin - (height - 2 * margin) * (value - minimum) / spread
        points.append(f"{x:.1f},{y:.1f}")
    lines = []
    for index, color in zip(row["detected_gap_indices"], ("#16a085", "#8e44ad")):
        x = margin + (width - 2 * margin) * (index + 0.5) / max(len(values) - 1, 1)
        lines.append(f'<line x1="{x:.1f}" y1="5" x2="{x:.1f}" y2="{height - 5}" stroke="{color}" stroke-width="2"/>')
    return (
        f'<svg viewBox="0 0 {width} {height}" class="curve"><polyline points="{" ".join(points)}" '
        f'fill="none" stroke="#365d7d" stroke-width="2"/>{"".join(lines)}</svg>'
    )


def render_comparison(
    *, output_path: Path, method_rows: list[dict[str, Any]], method_summary: dict[str, Any],
    decisions: list[dict[str, Any]], elbows: list[dict[str, Any]],
    complexity: list[dict[str, Any]], method_order: list[str],
) -> dict[str, Any]:
    rows_by_video: dict[str, list[dict[str, Any]]] = {}
    for row in method_rows:
        rows_by_video.setdefault(row["video_id"], []).append(row)
    decisions_by_key = {(row["video_id"], row["method"]): row for row in decisions}
    elbow_by_video = {row["video_id"]: row for row in elbows}
    complexity_by_video = {row["video_id"]: row for row in complexity}
    video_order = sorted(
        rows_by_video,
        key=lambda video_id: (
            0 if rows_by_video[video_id][0]["source_group"] == "long_primary" else 1,
            rows_by_video[video_id][0]["video_duration"], video_id,
        ),
    )
    nodes_cache: dict[str, dict[str, dict[str, Any]]] = {}
    # All node data needed for layout is recoverable from spans/assets plus IDs in rows;
    # use frozen hierarchy files without altering them.
    import json
    root = Path(__file__).resolve().parents[3]
    hierarchy_paths = [
        root / "outputs/experiments/long_video_hierarchy_stress_v0_2/safe_merge_hierarchies.jsonl",
        root / "outputs/experiments/fine_to_coarse_hierarchy_v0_1/safe_merge_hierarchies.jsonl",
    ]
    for path in hierarchy_paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                hierarchy = json.loads(line)
                nodes_cache[hierarchy["video_id"]] = {row["node_id"]: row for row in hierarchy["nodes"]}

    summary_rows = []
    for video_id in video_order:
        if rows_by_video[video_id][0]["source_group"] != "long_primary":
            continue
        by_method = {row["method"]: row for row in rows_by_video[video_id]}
        for method in method_order:
            row = by_method[method]
            similarity = row["redundancy"]["similarity_distribution"]["median"]
            flags = row["overmerge_medium"]["total_flagged_node_count"] + row["overmerge_coarse"]["total_flagged_node_count"]
            summary_rows.append(
                f'<tr><td><a href="#video-{_e(video_id)}">{_e(video_id)}</a></td><td>{_e(LABELS[method])}</td>'
                f'<td>{_time(row["video_duration"])}</td><td>{row["fine_count"]}</td>'
                f'<td>{row["medium"]["node_count"]}</td><td>{row["coarse"]["node_count"]}</td>'
                f'<td>{row["medium"]["median_duration_sec"]:.1f}s</td><td>{row["coarse"]["median_duration_sec"]:.1f}s</td>'
                f'<td>{row["coarse"]["largest_node_ratio"]:.1%}</td>'
                f'<td>{similarity:.3f}</td><td>{row["redundancy"]["high_similarity_weak_boundary_pair_count"]}</td>'
                f'<td>{flags}</td><td>{row["runtime"]["adaptive_cut_compute_sec"] + row["runtime"]["metrics_compute_sec"]:.4f}s</td></tr>'
            )

    image_count = 0
    sections = []
    for video_id in video_order:
        video_rows = rows_by_video[video_id]
        by_method = {row["method"]: row for row in video_rows}
        source_group = video_rows[0]["source_group"]
        duration = float(video_rows[0]["video_duration"])
        nodes = nodes_cache[video_id]
        timeline_rows = []
        method_sections = []
        for method in method_order:
            row = by_method[method]
            timeline_rows.append(
                f'<div class="timeline-method"><b>{_e(LABELS[method])}</b><span>Coarse</span>'
                f'{_timeline(row["coarse_ids"], nodes, duration, "coarse-block")}<span>Medium</span>'
                f'{_timeline(row["medium_ids"], nodes, duration, "medium-block")}</div>'
            )
            coarse_cards = []
            risk_by_id = {
                item["node_id"]: item
                for item in row["overmerge_medium"]["nodes"] + row["overmerge_coarse"]["nodes"]
            }
            membership = row["medium_to_coarse_membership"]
            for coarse_id in row["coarse_ids"]:
                coarse = nodes[coarse_id]
                coarse_asset = row["representative_assets"][coarse_id]
                coarse_path = output_path.parent / coarse_asset["html_relative_path"]
                if not coarse_path.is_file():
                    raise FileNotFoundError(coarse_path)
                image_count += 1
                medium_cards = []
                for medium_id in [item for item in row["medium_ids"] if membership[item] == coarse_id]:
                    medium = nodes[medium_id]
                    asset = row["representative_assets"][medium_id]
                    image_path = output_path.parent / asset["html_relative_path"]
                    if not image_path.is_file():
                        raise FileNotFoundError(image_path)
                    image_count += 1
                    risk = risk_by_id[medium_id]
                    badges = []
                    if risk["high_boundary_crossing_flag"]:
                        badges.append("high-boundary")
                    if risk["low_coherence_giant_flag"]:
                        badges.append("low-coherence giant")
                    if risk["chaining_risk_flag"]:
                        badges.append("chaining")
                    medium_cards.append(
                        f'<article class="medium-card"><h5>{_e(medium_id)} [{_time(medium["start"])}–{_time(medium["end"])}] · {medium["duration"]:.1f}s</h5>'
                        f'<img src="{_e(asset["html_relative_path"])}" loading="lazy"><p>Fine descendants: {len(medium["leaf_ids"])} · '
                        f'depth {row["medium"]["selected_depth_distribution"]}</p>'
                        f'<p class="flags">{_e(", ".join(badges) if badges else "no structural risk flag")}</p>'
                        + (_review(video_id + method + medium_id, suspicious=True) if badges else "")
                        + '</article>'
                    )
                coarse_risk = risk_by_id[coarse_id]
                coarse_badges = [
                    label for passed, label in (
                        (coarse_risk["high_boundary_crossing_flag"], "high-boundary"),
                        (coarse_risk["low_coherence_giant_flag"], "low-coherence giant"),
                        (coarse_risk["chaining_risk_flag"], "chaining"),
                    ) if passed
                ]
                coarse_cards.append(
                    f'<details class="coarse-card"><summary>{_e(coarse_id)} [{_time(coarse["start"])}–{_time(coarse["end"])}] · '
                    f'{coarse["duration"]:.1f}s · {len(medium_cards)} Medium children</summary>'
                    f'<div class="coarse-head"><img src="{_e(coarse_asset["html_relative_path"])}" loading="lazy">'
                    f'<p>{_e(", ".join(coarse_badges) if coarse_badges else "no structural risk flag")}</p></div>'
                    + (_review(video_id + method + coarse_id, suspicious=True) if coarse_badges else "")
                    + f'<div class="medium-grid">{"".join(medium_cards)}</div></details>'
                )
            trace_html = ""
            if method.startswith("fluid_"):
                trace = decisions_by_key[(video_id, method)]
                trace_rows = []
                for item in trace["medium_trace"] + trace["coarse_trace"]:
                    if item["decision"] == "KEEP_LEAF":
                        trace_rows.append(
                            f'<tr><td>{_e(item["level"])}</td><td>{_e(item["node_id"])}</td><td>leaf</td><td>—</td><td>—</td><td>KEEP_LEAF</td><td>leaf</td></tr>'
                        )
                    else:
                        trace_rows.append(
                            f'<tr><td>{_e(item["level"])}</td><td>{_e(item["node_id"])}</td>'
                            f'<td>{_time(item["start"])}–{_time(item["end"])}</td><td>{item["q_percentile_rank"]:.3f}</td>'
                            f'<td>{item["local_quality_drop"]:.3f}</td><td>{_e(item["decision"])}</td><td>{_e(item["reason"])}</td></tr>'
                        )
                trace_html = (
                    '<details><summary>Complete fluid decision trace</summary><div class="table-scroll"><table><thead><tr>'
                    '<th>Level</th><th>Node</th><th>Time</th><th>q rank</th><th>Local drop</th><th>Decision</th><th>Reason</th>'
                    f'</tr></thead><tbody>{"".join(trace_rows)}</tbody></table></div></details>'
                )
            method_sections.append(
                f'<details class="method" {"open" if source_group == "long_primary" and method in ("fixed_reference", "global_adaptive_elbow", "fluid_balanced") else ""}>'
                f'<summary>{_e(LABELS[method])} · Medium {row["medium"]["node_count"]} · Coarse {row["coarse"]["node_count"]} · '
                f'projected VLM reduction {row["projected_vlm"]["projected_reduction_ratio"]:.1%}</summary>'
                f'<p>Validity: Fine preserved {row["validity"]["medium"]["fine_leaf_preservation_rate"]:.0%}; coverage '
                f'{row["validity"]["medium"]["temporal_coverage_ratio"]:.1%}; gaps {row["validity"]["medium"]["gap_count"]}; '
                f'overlaps {row["validity"]["medium"]["overlap_count"]}; lineage violations {row["validity"]["lineage_violation_count"]}.</p>'
                + _review(video_id + method)
                + trace_html + "".join(coarse_cards) + '</details>'
            )

        elbow = elbow_by_video[video_id]
        elbow_panel = (
            f'<details><summary>Global elbow score curve</summary>{_elbow_svg(elbow)}'
            f'<p>Medium threshold cost {elbow["medium_cost_threshold"]:.4f}; Coarse threshold cost '
            f'{elbow["coarse_cost_threshold"]:.4f}. Green/purple lines mark detected adjacent gaps.</p></details>'
        )
        alignment = complexity_by_video[video_id]
        corr_rows = []
        for method in method_order:
            values = alignment["correlations"][method]
            corr_rows.append(
                f'<tr><td>{_e(LABELS[method])}</td><td>{values["fine_density_spearman"] if values["fine_density_spearman"] is not None else "n/a"}</td>'
                f'<td>{values["visual_change_spearman"] if values["visual_change_spearman"] is not None else "n/a"}</td>'
                f'<td>{values["combined_complexity_spearman"] if values["combined_complexity_spearman"] is not None else "n/a"}</td>'
                f'<td>{values["complex_window_mean_medium_density"]:.2f}</td><td>{values["stable_window_mean_medium_density"]:.2f}</td></tr>'
            )
        window_rows = []
        for window in alignment["windows"]:
            densities = " ".join(
                f'{LABELS[method].split(" ", 1)[0]}={window["medium_density_by_method"][method]:.1f}'
                for method in method_order
            )
            window_rows.append(
                f'<tr><td>{_time(window["start"])}–{_time(window["end"])}</td>'
                f'<td>{window["fine_boundary_density_per_min"]:.2f}</td><td>{window["mean_local_visual_change"]:.4f}</td>'
                f'<td>{window["combined_complexity"]:.3f}</td><td>{_e(densities)}</td></tr>'
            )
        alignment_panel = (
            '<details ' + ('open' if duration == max(item["video_duration"] for item in video_rows) and source_group == "long_primary" else '') + '><summary>60-second fluid-behavior diagnostic</summary>'
            '<div class="table-scroll"><table><thead><tr><th>Method</th><th>Fine-density ρ</th><th>Visual-change ρ</th><th>Combined ρ</th><th>Complex-window density</th><th>Stable-window density</th></tr></thead>'
            f'<tbody>{"".join(corr_rows)}</tbody></table><table><thead><tr><th>Window</th><th>Fine density</th><th>Visual change</th><th>Complexity</th><th>Medium density A–G</th></tr></thead><tbody>{"".join(window_rows)}</tbody></table></div></details>'
        )
        sections.append(
            f'<section id="video-{_e(video_id)}"><h2>{_e(video_id)}</h2><p>{_e(source_group)} · duration {_time(duration)} · '
            f'Fine {video_rows[0]["fine_count"]}</p><h3>Aligned timeline comparison</h3><div class="timeline-stack">{"".join(timeline_rows)}</div>'
            + elbow_panel + alignment_panel + '<h3>Nested hierarchy review</h3>' + "".join(method_sections) + '</section>'
        )

    nav = " ".join(f'<a href="#video-{_e(item)}">{_e(item)}</a>' for item in video_order)
    document = f'''<!doctype html><html><head><meta charset="utf-8"><title>Adaptive fluid hierarchy v0.1</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#eef2f5;color:#17232e;font:13px/1.45 system-ui,sans-serif}}header{{background:#14283a;color:#fff;padding:18px;position:sticky;top:0;z-index:10}}header h1{{margin:0}}nav{{white-space:nowrap;overflow-x:auto}}nav a{{color:#a8e0ff;margin-right:13px}}main{{max-width:1600px;margin:auto;padding:16px}}.summary,section{{background:#fff;border:1px solid #cbd4dc;border-radius:12px;padding:16px;margin-bottom:18px}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{border:1px solid #cbd3db;padding:5px;text-align:left}}th{{background:#e7edf3}}.table-scroll{{overflow:auto}}details summary{{cursor:pointer;font-weight:750}}.timeline-stack{{border:1px solid #b8c3ce;padding:8px}}.timeline-method{{display:grid;grid-template-columns:190px 50px 1fr;gap:5px;margin:5px 0;align-items:center}}.timeline-method>span:nth-of-type(2){{grid-column:2}}.timeline-method .track:nth-of-type(2){{grid-column:3}}.track{{height:18px;position:relative;background:#f2f5f7;border:1px solid #9eabb7}}.block{{position:absolute;top:1px;height:14px;border-right:1px solid white}}.coarse-block{{background:#6574da}}.medium-block{{background:#2fa292}}.method{{border:2px solid #6c7c8c;border-radius:8px;padding:10px;margin:12px 0}}.coarse-card{{border:2px solid #6574da;border-radius:7px;padding:8px;margin:9px}}.coarse-head{{display:flex;gap:10px;align-items:center}}.medium-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(225px,1fr));gap:7px}}.medium-card{{border-left:5px solid #2fa292;background:#f2fbf9;padding:7px}}img{{width:190px;height:125px;object-fit:cover;border:1px solid #aeb9c2;border-radius:5px}}.flags{{color:#9a4d00;font-weight:700}}.review{{background:#f5f7fa;border:1px dashed #9ba7b2;padding:7px;margin:7px 0}}textarea{{display:block;width:100%;height:35px}}.curve{{width:100%;max-width:750px;background:#f7f9fa;border:1px solid #ccd4db}}@media(max-width:800px){{.timeline-method{{grid-template-columns:120px 45px 1fr}}}}
</style></head><body><header><h1>Adaptive fluid hierarchy v0.1</h1><p>Seven operating frontiers on unchanged Safe-Merge trees · VLM/API/downloads = 0</p><nav>{nav}</nav></header><main><div class="summary"><h2>Primary long-video method summary</h2><div class="table-scroll"><table><thead><tr><th>Video</th><th>Method</th><th>Duration</th><th>Fine</th><th>Medium</th><th>Coarse</th><th>Median M</th><th>Median C</th><th>Largest C</th><th>Adjacent sim median</th><th>High-sim weak-boundary</th><th>Risk flags</th><th>Compute</th></tr></thead><tbody>{''.join(summary_rows)}</tbody></table></div></div>{''.join(sections)}</main></body></html>'''
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    return {
        "displayed_video_count": len(video_order),
        "primary_long_video_count": sum(rows_by_video[item][0]["source_group"] == "long_primary" for item in video_order),
        "short_robustness_video_count": sum(rows_by_video[item][0]["source_group"] == "short_robustness" for item in video_order),
        "displayed_method_video_sections": len(video_order) * len(method_order),
        "image_reference_count": document.count("<img "),
        "missing_image_reference_count": 0,
        "all_methods_displayed": True,
    }
