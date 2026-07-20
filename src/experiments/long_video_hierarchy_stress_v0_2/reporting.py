"""Human-first renderer for genuine long-video hierarchy inspection."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _asset(frame_path: str, *, root: Path, output_dir: Path) -> str:
    source = Path(frame_path)
    if not source.is_absolute():
        source = root / source
    if not source.is_file():
        raise FileNotFoundError(f"Missing review frame: {source}")
    return source.relative_to(output_dir).as_posix()


def _frames(
    node: dict[str, Any], *, root: Path, output_dir: Path, limit: int,
) -> tuple[str, int]:
    cards: list[str] = []
    seen: set[str] = set()
    for record in node.get("representative_frames", []):
        frame_path = str(record["frame_path"])
        if frame_path in seen:
            continue
        seen.add(frame_path)
        relative = _asset(frame_path, root=root, output_dir=output_dir)
        cards.append(
            f'<figure><img src="{_e(relative)}" loading="lazy" alt="{_e(node["node_id"])} at '
            f'{float(record["timestamp"]):.1f}s"><figcaption>{float(record["timestamp"]):.1f}s · '
            f'{_e(record["kind"])}</figcaption></figure>'
        )
        if len(cards) >= limit:
            break
    return '<div class="frames">' + "".join(cards) + "</div>", len(cards)


def _review(prefix: str) -> str:
    blocks = []
    for label, key in (
        ("Meaningful grouping", "meaningful"),
        ("Overmerged", "overmerged"),
        ("Fragmented", "fragmented"),
        ("Giant / chained", "giant_chained"),
    ):
        options = " ".join(
            f'<label><input type="radio" name="{_e(prefix)}_{key}" value="{value}"> {value}</label>'
            for value in ("yes", "no", "unsure")
        )
        blocks.append(f'<div><strong>{label}:</strong> {options}</div>')
    blocks.append(f'<label><strong>Notes:</strong><textarea name="{_e(prefix)}_notes"></textarea></label>')
    return '<div class="review">' + "".join(blocks) + "</div>"


def _timeline(nodes: list[dict[str, Any]], duration: float, css_class: str) -> str:
    blocks = []
    for node in nodes:
        left = 100.0 * float(node["start"]) / duration
        width = max(0.15, 100.0 * float(node["duration"]) / duration)
        title = f'{node["node_id"]}: {node["start"]:.1f}-{node["end"]:.1f}s ({node["duration"]:.1f}s)'
        blocks.append(
            f'<span class="timeline-block {css_class}" style="left:{left:.5f}%;width:{width:.5f}%" '
            f'title="{_e(title)}"><i>{_e(node["node_id"])}</i></span>'
        )
    return '<div class="timeline">' + "".join(blocks) + '</div><div class="axis"><span>0s</span><span>' + f'{duration:.0f}s</span></div>'


def render_hierarchy_review(
    *, output_path: Path, root: Path, manifest: dict[str, Any],
    hierarchies: list[dict[str, Any]], metrics: list[dict[str, Any]],
    runtimes: list[dict[str, Any]], previous_baseline: dict[str, Any],
) -> dict[str, Any]:
    output_dir = output_path.parent
    hierarchy_by_video = {row["video_id"]: row for row in hierarchies}
    metric_by_video = {row["video_id"]: row for row in metrics}
    runtime_by_video = {row["video_id"]: row for row in runtimes}
    summary_rows: list[str] = []
    sections: list[str] = []
    image_count = 0

    for index, video in enumerate(manifest["videos"], start=1):
        video_id = str(video["video_id"])
        hierarchy = hierarchy_by_video[video_id]
        metric = metric_by_video[video_id]
        runtime = runtime_by_video[video_id]
        duration = float(video["actual_duration_sec"])
        nodes = {row["node_id"]: row for row in hierarchy["nodes"]}
        fine = [nodes[item] for item in hierarchy["cuts"]["fine"]["node_ids"]]
        medium = [nodes[item] for item in hierarchy["cuts"]["medium"]["node_ids"]]
        coarse = [nodes[item] for item in hierarchy["cuts"]["coarse"]["node_ids"]]
        summary_rows.append(
            '<tr>'
            f'<td><a href="#video-{index}">{_e(video["sample_id"])}</a></td><td>{duration:.1f}s</td>'
            f'<td>{len(fine)}</td><td>{len(medium)}</td><td>{len(coarse)}</td>'
            f'<td>{metric["fine"]["max_duration_sec"]:.1f}s</td>'
            f'<td>{metric["medium"]["max_duration_sec"]:.1f}s</td>'
            f'<td>{metric["coarse"]["max_duration_sec"]:.1f}s</td>'
            f'<td>{metric["coarse"]["largest_node_duration_ratio"]:.1%}</td>'
            f'<td>{runtime["dinov2_feature_extraction_sec"]:.1f}s</td>'
            f'<td>{runtime["safe_hierarchy_construction_sec"]:.3f}s</td></tr>'
        )

        medium_by_coarse: dict[str, list[dict[str, Any]]] = {node["node_id"]: [] for node in coarse}
        for medium_node in medium:
            matches = [
                coarse_node for coarse_node in coarse
                if set(medium_node["leaf_ids"]) <= set(coarse_node["leaf_ids"])
            ]
            if len(matches) != 1:
                raise ValueError(f"Invalid Medium-to-Coarse membership: {video_id}/{medium_node['node_id']}")
            medium_by_coarse[matches[0]["node_id"]].append(medium_node)

        nested: list[str] = []
        for coarse_node in coarse:
            coarse_frames, count = _frames(coarse_node, root=root, output_dir=output_dir, limit=4)
            image_count += count
            medium_cards: list[str] = []
            for medium_node in medium_by_coarse[coarse_node["node_id"]]:
                medium_frames, count = _frames(medium_node, root=root, output_dir=output_dir, limit=3)
                image_count += count
                fine_cards: list[str] = []
                for fine_id in medium_node["leaf_ids"]:
                    fine_node = nodes[fine_id]
                    fine_frames, count = _frames(fine_node, root=root, output_dir=output_dir, limit=2)
                    image_count += count
                    fine_cards.append(
                        f'<article class="fine-card"><h5>FINE {_e(fine_id)}</h5>'
                        f'<p>{fine_node["start"]:.1f}–{fine_node["end"]:.1f}s · {fine_node["duration"]:.1f}s</p>'
                        f'{fine_frames}</article>'
                    )
                medium_cards.append(
                    f'<details class="medium"><summary>MEDIUM {_e(medium_node["node_id"])} · '
                    f'{medium_node["start"]:.1f}–{medium_node["end"]:.1f}s · {medium_node["duration"]:.1f}s · '
                    f'{len(medium_node["leaf_ids"])} Fine children</summary>{medium_frames}'
                    f'{_review(video_id + "_" + medium_node["node_id"])}'
                    f'<div class="fine-grid">{"".join(fine_cards)}</div></details>'
                )
            nested.append(
                f'<details class="coarse" open><summary>COARSE {_e(coarse_node["node_id"])} · '
                f'{coarse_node["start"]:.1f}–{coarse_node["end"]:.1f}s · {coarse_node["duration"]:.1f}s · '
                f'{len(coarse_node["leaf_ids"])} Fine descendants</summary>{coarse_frames}'
                f'{_review(video_id + "_" + coarse_node["node_id"])}'
                f'<div class="medium-stack">{"".join(medium_cards)}</div></details>'
            )

        sections.append(
            f'<section id="video-{index}"><h2>{_e(video["sample_id"])}</h2>'
            f'<p><a href="{_e(video["source_url"])}">Source URL</a> · requested '
            f'{video["start_sec"]:.1f}–{video["end_sec"]:.1f}s · actual clip {duration:.1f}s · '
            f'{video["width"]}×{video["height"]} @ {video["fps"]:.3f} fps · audio '
            f'{"yes" if video["audio_present"] else "no"}</p>'
            f'<div class="counts"><b>Fine {len(fine)}</b><b>Medium {len(medium)}</b><b>Coarse {len(coarse)}</b>'
            f'<b>Depth {metric["hierarchy_depth"]}</b></div>'
            '<h3>Aligned reference-cut overview</h3>'
            f'<h4>COARSE</h4>{_timeline(coarse, duration, "coarse-block")}'
            f'<h4>MEDIUM</h4>{_timeline(medium, duration, "medium-block")}'
            f'<h4>FINE</h4>{_timeline(fine, duration, "fine-block")}'
            '<p class="help">Open Coarse, then Medium cards to inspect ordered Fine descendants and images.</p>'
            f'{"".join(nested)}</section>'
        )

    baseline_note = (
        f'Prior ≈3-minute context: Fine ~{previous_baseline["fine_nodes_per_video"]:.0f}/video, '
        f'Medium ~{previous_baseline["medium_nodes_per_video"]:.0f}/video, '
        f'Coarse ~{previous_baseline["coarse_nodes_per_video"]:.0f}/video. '
        'Context only; no linear-scaling claim is made.'
    )
    navigation = " ".join(
        f'<a href="#video-{index}">{index}. {_e(video["sample_id"])}</a>'
        for index, video in enumerate(manifest["videos"], start=1)
    )
    document = f'''<!doctype html><html><head><meta charset="utf-8"><title>EgoPolice long-video hierarchy stress v0.2</title>
<style>
*{{box-sizing:border-box}} body{{margin:0;background:#eef2f6;color:#18222d;font:14px/1.45 system-ui,sans-serif}} header{{background:#142536;color:#fff;padding:22px;position:sticky;top:0;z-index:10}} header h1{{margin:0 0 6px}} nav{{white-space:nowrap;overflow-x:auto}} nav a{{color:#9fddff;margin-right:14px}} main{{max-width:1500px;margin:auto;padding:18px}} .summary,section{{background:#fff;border:1px solid #ccd4dd;border-radius:12px;padding:18px;margin:0 0 20px}} table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{border:1px solid #cfd7df;padding:7px;text-align:left}} th{{background:#e8eef4;position:sticky;top:104px}} .counts{{display:flex;gap:12px;flex-wrap:wrap}} .counts b{{background:#e7eff8;padding:7px 12px;border-radius:18px}} .timeline{{height:32px;position:relative;border:1px solid #8290a0;background:#f6f8fa;overflow:hidden}} .timeline-block{{position:absolute;top:2px;height:26px;border-right:1px solid #fff;overflow:hidden;font-size:9px;padding:4px 2px}} .timeline-block i{{font-style:normal;white-space:nowrap}} .coarse-block{{background:#6874d8;color:#fff}} .medium-block{{background:#33a99b;color:#fff}} .fine-block{{background:#f2a23a;color:#17212b}} .axis{{display:flex;justify-content:space-between;color:#687483;font-size:11px}} details summary{{cursor:pointer;font-weight:750}} .coarse{{border:3px solid #6975d9;border-radius:10px;padding:12px;margin:18px 0}} .medium{{border:2px solid #35a99b;border-radius:8px;padding:10px;margin:12px 0 12px 24px}} .fine-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(245px,1fr));gap:8px}} .fine-card{{border-left:5px solid #f2a23a;background:#fff8ed;padding:8px}} .fine-card h5,.fine-card p{{margin:3px}} .frames{{display:flex;gap:8px;overflow-x:auto;padding:9px 0}} figure{{margin:0;min-width:150px}} img{{width:150px;height:100px;object-fit:cover;border:1px solid #aeb7c1;border-radius:6px}} figcaption{{font-size:11px;color:#526171}} .review{{background:#f5f7fa;border:1px dashed #9ba8b5;padding:8px;margin:8px 0}} textarea{{display:block;width:100%;height:38px}} .help{{color:#4f6072}}
</style></head><body><header><h1>EgoPolice genuine long-video hierarchy stress v0.2</h1><p>CoMET-style Fine Events → Boundary-aware Safe Merge → Medium / Coarse reference cuts</p><nav>{navigation}</nav></header><main><div class="summary"><h2>Scaling summary</h2><p>{_e(baseline_note)}</p><div style="overflow:auto"><table><thead><tr><th>Video</th><th>Duration</th><th>Fine</th><th>Medium</th><th>Coarse</th><th>Largest Fine</th><th>Largest Medium</th><th>Largest Coarse</th><th>Largest Coarse ratio</th><th>Feature extraction</th><th>Hierarchy build</th></tr></thead><tbody>{''.join(summary_rows)}</tbody></table></div></div>{''.join(sections)}</main></body></html>'''
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    return {
        "displayed_video_count": len(sections),
        "image_reference_count": document.count("<img "),
        "missing_image_count": 0,
        "all_parent_child_relationships_rendered": True,
    }
