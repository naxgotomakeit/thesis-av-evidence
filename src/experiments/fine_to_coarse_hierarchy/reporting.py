"""Expandable human-review report for offline temporal hierarchies."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _thumb_src(project_relative: str) -> str:
    value = project_relative.replace("\\", "/")
    return "../../" + value[len("outputs/") :] if value.startswith("outputs/") else value


def _timeline(
    label: str,
    nodes: list[dict[str, Any]],
    duration: float,
    css_class: str,
    *,
    compact_labels: bool = False,
) -> str:
    bars = []
    for index, node in enumerate(nodes):
        left = 100.0 * float(node["start"]) / duration
        width = max(0.15, 100.0 * float(node["duration"]) / duration)
        title = (
            f"{node.get('node_id', node.get('segment_id'))} | "
            f"{float(node['start']):.1f}–{float(node['end']):.1f}s | {float(node['duration']):.1f}s"
        )
        text = "" if compact_labels else str(index + 1)
        bars.append(
            f'<span class="bar {css_class}" style="left:{left:.5f}%;width:{width:.5f}%" title="{_e(title)}">{_e(text)}</span>'
        )
    return (
        f'<div class="timeline-row"><div class="timeline-label">{_e(label)} <b>({len(nodes)})</b></div>'
        '<div class="axis"><span class="t0">0s</span><span class="t60">60</span><span class="t120">120</span><span class="t180">180s</span>'
        + "".join(bars) + "</div></div>"
    )


def _thumbs(node: dict[str, Any]) -> str:
    return '<div class="thumbs">' + "".join(
        f'<figure><img loading="lazy" src="{_e(_thumb_src(row["frame_path"]))}"><figcaption>{_e(row["kind"])} · {float(row["timestamp"]):.1f}s</figcaption></figure>'
        for row in node.get("representative_frames", [])
    ) + "</div>"


def _node_summary(node: dict[str, Any]) -> str:
    return (
        f'<span class="node-id">{_e(node["node_id"])}</span> '
        f'{float(node["start"]):.1f}–{float(node["end"]):.1f}s · '
        f'{len(node["leaf_ids"])} Fine leaves · variance {float(node["internal_variability"]):.4f}'
    )


def _expand_tree(hierarchy: dict[str, Any], method_label: str) -> str:
    nodes = {row["node_id"]: row for row in hierarchy["nodes"]}
    medium = [nodes[node_id] for node_id in hierarchy["cuts"]["medium"]["node_ids"]]
    coarse = [nodes[node_id] for node_id in hierarchy["cuts"]["coarse"]["node_ids"]]
    sections = []
    for coarse_node in coarse:
        coarse_leaves = set(coarse_node["leaf_ids"])
        medium_children = [node for node in medium if set(node["leaf_ids"]).issubset(coarse_leaves)]
        medium_html = []
        for medium_node in medium_children:
            fine_html = "".join(
                f'<details class="fine-detail"><summary>{_node_summary(nodes[fine_id])}</summary>{_thumbs(nodes[fine_id])}</details>'
                for fine_id in medium_node["leaf_ids"]
            )
            medium_html.append(
                f'<details class="medium-detail"><summary>Medium: {_node_summary(medium_node)}</summary>{_thumbs(medium_node)}'
                f'<div class="fine-list"><h5>Immutable Fine descendants</h5>{fine_html}</div></details>'
            )
        sections.append(
            f'<details class="coarse-detail"><summary>{_e(method_label)} Coarse: {_node_summary(coarse_node)}</summary>'
            f'{_thumbs(coarse_node)}<div class="medium-list">{"".join(medium_html)}</div></details>'
        )
    return "".join(sections)


def _select(storage: str, choices: list[str]) -> str:
    return (
        f'<select class="manual" data-storage="{_e(storage)}">'
        + "".join(f'<option>{_e(choice)}</option>' for choice in choices)
        + "</select>"
    )


def render_comparison(
    *,
    output_path: Path,
    manifest: dict[str, Any],
    ward_hierarchies: list[dict[str, Any]],
    safe_hierarchies: list[dict[str, Any]],
    kts_records: list[dict[str, Any]],
    per_video_metrics: list[dict[str, Any]],
    aggregate_metrics: dict[str, Any],
    boundary_survival: list[dict[str, Any]],
) -> None:
    ward_by_video = {row["video_id"]: row for row in ward_hierarchies}
    safe_by_video = {row["video_id"]: row for row in safe_hierarchies}
    kts_by_video = {row["video_id"]: row for row in kts_records}
    metric_by_video = {row["video_id"]: row for row in per_video_metrics}
    boundary_by_video = {row["video_id"]: row for row in boundary_survival}
    cards = []
    for index, item in enumerate(manifest["videos"], start=1):
        video_id = str(item["video_id"])
        ward = ward_by_video[video_id]
        safe = safe_by_video[video_id]
        kts = kts_by_video[video_id]
        metrics = metric_by_video[video_id]
        boundaries = boundary_by_video[video_id]
        ward_nodes = {row["node_id"]: row for row in ward["nodes"]}
        safe_nodes = {row["node_id"]: row for row in safe["nodes"]}
        fine = [ward_nodes[node_id] for node_id in ward["cuts"]["fine"]["node_ids"]]
        ward_medium = [ward_nodes[node_id] for node_id in ward["cuts"]["medium"]["node_ids"]]
        ward_coarse = [ward_nodes[node_id] for node_id in ward["cuts"]["coarse"]["node_ids"]]
        safe_fine = [safe_nodes[node_id] for node_id in safe["cuts"]["fine"]["node_ids"]]
        safe_medium = [safe_nodes[node_id] for node_id in safe["cuts"]["medium"]["node_ids"]]
        safe_coarse = [safe_nodes[node_id] for node_id in safe["cuts"]["coarse"]["node_ids"]]
        duration = float(ward["video_duration"])
        key = video_id.replace("-", "_")
        diag = metrics["failure_mode_diagnostics"]
        flags = [name for name, value in diag.items() if isinstance(value, bool) and value]
        cards.append(
            f'''<article class="video" id="video-{_e(key)}">
<div class="video-head"><div><span class="number">{index}</span><strong>{_e(video_id)}</strong> <span class="pill">{_e(item['group'])}</span></div><a href="#top">Back to summary</a></div>
<div class="headline"><b>Fine leaves {metrics['fine']['node_count']}</b> · Ward M/C {metrics['ward']['medium']['node_count']}/{metrics['ward']['coarse']['node_count']} · Safe M/C {metrics['safe']['medium']['node_count']}/{metrics['safe']['coarse']['node_count']} · KTS {metrics['kts_reference']['node_count']}</div>
<div class="flags">{''.join(f'<span class="flag">{_e(flag)}</span>' for flag in flags) or '<span class="muted">No automatic structural risk flag</span>'}</div>
<div class="timelines">
{_timeline('KTS reference', kts['segments'], duration, 'kts')}
{_timeline('Raw CoMET Fine', fine, duration, 'fine', compact_labels=True)}
<div class="method-title">Adjacent-Ward hierarchy</div>
{_timeline('Ward Fine', fine, duration, 'fine', compact_labels=True)}
{_timeline('Ward Medium', ward_medium, duration, 'ward-medium')}
{_timeline('Ward Coarse', ward_coarse, duration, 'ward-coarse')}
<div class="method-title">Boundary-aware Safe Merge hierarchy</div>
{_timeline('Safe Fine', safe_fine, duration, 'fine', compact_labels=True)}
{_timeline('Safe Medium', safe_medium, duration, 'safe-medium')}
{_timeline('Safe Coarse', safe_coarse, duration, 'safe-coarse')}
</div>
<div class="metric-grid">
 <div><h4>Ward</h4><p>Medium compression {100*(1-metrics['ward']['medium']['compression_ratio_vs_fine']):.1f}% · max ratio {metrics['ward']['medium']['largest_region_ratio']:.3f}</p><p>Strong-boundary survival M/C {100*boundaries['ward']['medium']['strong_survival_rate']:.1f}% / {100*boundaries['ward']['coarse']['strong_survival_rate']:.1f}%</p><p>Risk nodes M/C {metrics['ward']['medium']['overmerge_risk_node_count']} / {metrics['ward']['coarse']['overmerge_risk_node_count']}</p></div>
 <div><h4>Safe Merge</h4><p>Medium compression {100*(1-metrics['safe']['medium']['compression_ratio_vs_fine']):.1f}% · max ratio {metrics['safe']['medium']['largest_region_ratio']:.3f}</p><p>Strong-boundary survival M/C {100*boundaries['safe']['medium']['strong_survival_rate']:.1f}% / {100*boundaries['safe']['coarse']['strong_survival_rate']:.1f}%</p><p>Risk nodes M/C {metrics['safe']['medium']['overmerge_risk_node_count']} / {metrics['safe']['coarse']['overmerge_risk_node_count']}</p></div>
</div>
<h3>Expandable Coarse → Medium → immutable Fine lineage</h3>
<div class="tree-grid"><section><h4>Ward lineage</h4>{_expand_tree(ward,'Ward')}</section><section><h4>Safe lineage</h4>{_expand_tree(safe,'Safe')}</section></div>
<div class="manual"><h3>Manual review — intentionally blank</h3>
 <label>1. More coherent Medium {_select(key+':coherent',['Unreviewed','Ward','Safe Merge','Similar','Neither'])}</label>
 <label>2. Better redundancy reduction {_select(key+':redundancy',['Unreviewed','Ward','Safe Merge','Similar','Neither'])}</label>
 <label>3a. Ward incorrectly merged actions? {_select(key+':ward_bad_merge',['Unreviewed','Yes','No','Unclear'])}</label>
 <label>3b. Safe incorrectly merged actions? {_select(key+':safe_bad_merge',['Unreviewed','Yes','No','Unclear'])}</label>
 <label>4a. Ward stable-background details {_select(key+':ward_detail',['Unreviewed','Good','Partial','Poor'])}</label>
 <label>4b. Safe stable-background details {_select(key+':safe_detail',['Unreviewed','Good','Partial','Poor'])}</label>
 <label>5a. Ward Medium more useful than KTS? {_select(key+':ward_vs_kts',['Unreviewed','Yes','No','Unclear'])}</label>
 <label>5b. Safe Medium more useful than KTS? {_select(key+':safe_vs_kts',['Unreviewed','Yes','No','Unclear'])}</label>
 <label>6a. Ward Coarse useful overview? {_select(key+':ward_coarse',['Unreviewed','Yes','No','Unclear'])}</label>
 <label>6b. Safe Coarse useful overview? {_select(key+':safe_coarse',['Unreviewed','Yes','No','Unclear'])}</label>
 <label>7. Plausible resolution/descend/fallback hierarchy? {_select(key+':plausible',['Unreviewed','Yes','No','Unclear'])}</label>
 <label class="wide">8. Notes<textarea class="manual" data-storage="{_e(key+':notes')}" rows="3"></textarea></label>
</div>
<details><summary>Structural diagnostics JSON</summary><pre>{_e(json.dumps(metrics, indent=2, ensure_ascii=False))}</pre></details>
</article>'''
        )

    all_metrics = aggregate_metrics["all"]
    html_text = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fine to Coarse Hierarchy v0.1</title><style>
:root{{--bg:#f4f6f8;--paper:#fff;--ink:#1c2733;--line:#d6dee7;--blue:#315f8f;--safe:#28856d;--warn:#9c5c00}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,Segoe UI,sans-serif}} header,main{{max-width:1550px;margin:auto}} header{{padding:25px}} main{{padding:0 25px 60px}}
h1{{margin:0}} .notice{{background:#fff7dc;border-left:5px solid #d69b00;padding:12px;margin:15px 0}} .summary{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px}} .stat,.video{{background:white;border:1px solid var(--line);border-radius:10px}} .stat{{padding:13px}} .stat b{{display:block;font-size:23px;color:var(--blue)}}
.video{{padding:17px;margin:22px 0;scroll-margin-top:10px}} .video-head{{display:flex;justify-content:space-between;align-items:center}} .number{{display:inline-grid;place-items:center;width:28px;height:28px;border-radius:50%;background:var(--blue);color:white;margin-right:8px}} .pill,.flag{{display:inline-block;background:#e8eef5;border-radius:999px;padding:3px 8px;margin-left:6px}} .flag{{background:#ffedcf;color:#704500;margin:8px 5px 0 0}}
.headline{{font-size:16px;margin:10px 0}} .timelines{{border:1px solid var(--line);padding:10px;border-radius:8px;margin-top:12px;overflow-x:auto}} .timeline-row{{display:grid;grid-template-columns:150px minmax(900px,1fr);align-items:center;margin:7px 0}} .timeline-label{{padding-right:8px}} .axis{{position:relative;height:31px;background:repeating-linear-gradient(to right,#eef2f5 0,#eef2f5 calc(33.333% - 1px),#c8d2dc calc(33.333% - 1px),#c8d2dc 33.333%);border:1px solid #b8c4d0}} .axis>span:not(.bar){{position:absolute;top:-17px;font-size:10px;color:#627180}} .t0{{left:0}}.t60{{left:33.333%}}.t120{{left:66.666%}}.t180{{right:0}} .bar{{position:absolute;top:4px;height:21px;border:1px solid rgba(0,0,0,.32);overflow:hidden;text-align:center;font-size:10px;color:white}} .kts{{background:#7b67a7}}.fine{{background:#8c98a5}}.ward-medium{{background:#4f7fab}}.ward-coarse{{background:#285178}}.safe-medium{{background:#48a88c}}.safe-coarse{{background:#176d58}} .method-title{{font-weight:700;margin:14px 0 5px;color:var(--blue)}}
.metric-grid,.tree-grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:12px 0}} .metric-grid>div,.tree-grid>section{{border:1px solid var(--line);border-radius:8px;padding:11px}} details{{margin:7px 0}} summary{{cursor:pointer}} .coarse-detail{{border-left:4px solid var(--blue);padding-left:8px}} .medium-detail{{border-left:3px solid #67a1cc;padding-left:8px}} .fine-detail{{margin-left:12px}} .thumbs{{display:flex;gap:7px;overflow-x:auto;margin:8px 0}} figure{{margin:0;min-width:120px}} img{{width:120px;height:75px;object-fit:cover;border:1px solid #b8c3ce;border-radius:4px}} figcaption{{font-size:10px;color:#596879}} .node-id{{font-family:ui-monospace,monospace}}
.manual{{display:grid;grid-template-columns:1fr 1fr;gap:9px;background:#f2f6fa;padding:12px;border-radius:8px;margin-top:14px}} .manual h3,.wide{{grid-column:1/-1}} label{{display:flex;justify-content:space-between;gap:8px}} textarea{{width:100%}} pre{{white-space:pre-wrap;max-height:450px;overflow:auto;background:#111827;color:#e5edf5;padding:12px}} .muted{{color:#687787}}
@media(max-width:900px){{.metric-grid,.tree-grid,.manual{{grid-template-columns:1fr}}.manual h3,.wide{{grid-column:auto}}}}
</style></head><body><header id="top"><h1>Fine → Medium → Coarse hierarchy v0.1</h1><p>Offline proof-of-concept over immutable full-video CoMET-style Fine leaves on the frozen 10 videos.</p>
<div class="notice"><b>Interpretation boundary:</b> Medium≈50% and Coarse≈25% are transparent visualization/reference cuts, not ground-truth granularity. KTS is reference only. No questions, gold, options, Planner, retrieval, final QA, or API calls are used.</div>
<div class="summary">
<div class="stat"><b>{aggregate_metrics['video_count']}</b>frozen videos</div>
<div class="stat"><b>{all_metrics['fine']['total_nodes']}</b>immutable Fine leaves</div>
<div class="stat"><b>{all_metrics['ward']['medium']['mean_node_count']:.1f}</b>mean Ward Medium nodes</div>
<div class="stat"><b>{all_metrics['safe']['medium']['mean_node_count']:.1f}</b>mean Safe Medium nodes</div>
<div class="stat"><b>{100*all_metrics['ward']['medium']['strong_boundary_survival_rate']:.1f}%</b>Ward strong-boundary survival</div>
<div class="stat"><b>{100*all_metrics['safe']['medium']['strong_boundary_survival_rate']:.1f}%</b>Safe strong-boundary survival</div>
</div><p>Jump to: {''.join(f'<a href="#video-{_e(str(row["video_id"]).replace("-","_"))}">{index}</a> ' for index,row in enumerate(manifest['videos'],1))}</p></header><main>{''.join(cards)}</main>
<script>document.querySelectorAll('.manual').forEach(el=>{{const k='fine-coarse-review:'+el.dataset.storage;const v=localStorage.getItem(k);if(v!==null)el.value=v;el.addEventListener('change',()=>localStorage.setItem(k,el.value));}});</script></body></html>'''
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
