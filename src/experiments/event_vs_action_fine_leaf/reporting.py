"""Human-review HTML for Event-level versus Action-level leaves."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _thumb_src(path: str) -> str:
    normalized = path.replace("\\", "/")
    if normalized.startswith("outputs/"):
        return "../../" + normalized[len("outputs/"):]
    return normalized


def _timeline(label: str, segments: list[dict[str, Any]], duration: float, css: str) -> str:
    bars = []
    for index, segment in enumerate(segments, start=1):
        start, end = float(segment["start"]), float(segment["end"])
        title = f"{segment.get('action_id', segment.get('segment_id'))} | {start:.1f}-{end:.1f}s"
        bars.append(
            f'<span class="bar {css}" style="left:{100*start/duration:.6f}%;width:{max(.18,100*(end-start)/duration):.6f}%" '
            f'title="{_e(title)}">{index}</span>'
        )
    return (
        f'<div class="timeline-row"><div class="timeline-label">{_e(label)} <b>({len(segments)})</b></div>'
        '<div class="axis"><i class="x0">0s</i><i class="x60">60</i><i class="x120">120</i><i class="x180">180s</i>'
        + "".join(bars) + "</div></div>"
    )


def _thumbs(rows: list[dict[str, Any]]) -> str:
    figures = []
    for row in rows:
        figures.append(
            f'<figure><img loading="lazy" src="{_e(_thumb_src(row["frame_path"]))}">'
            f'<figcaption>{_e(row["kind"])} · {float(row["timestamp"]):.1f}s</figcaption></figure>'
        )
    return '<div class="thumbs">' + "".join(figures) + "</div>"


def _motion_svg(decision: dict[str, Any], signal: dict[str, Any]) -> str:
    start, end = float(decision["parent_start"]), float(decision["parent_end"])
    intervals = signal["interval_starts"]
    values = signal["smoothed_motion"]
    points = [(float(t), float(v)) for t, v in zip(intervals, values) if start <= float(t) < end]
    if not points:
        return '<div class="muted">No motion observations</div>'
    lo = min(value for _, value in points)
    hi = max(value for _, value in points)
    scale = max(hi - lo, 1e-8)
    poly = " ".join(
        f"{100*(t-start)/max(end-start,1e-8):.3f},{38-32*(v-lo)/scale:.3f}" for t, v in points
    )
    boundaries = "".join(
        f'<line x1="{100*(float(row["timestamp"])-start)/max(end-start,1e-8):.3f}" y1="2" '
        f'x2="{100*(float(row["timestamp"])-start)/max(end-start,1e-8):.3f}" y2="40" class="cp"><title>'
        f'{float(row["timestamp"]):.1f}s · score {float(row["motion_change_score"]):.2f}</title></line>'
        for row in decision["change_points"]
    )
    return (
        '<svg class="motion" viewBox="0 0 100 42" preserveAspectRatio="none">'
        f'<polyline points="{poly}" />{boundaries}</svg>'
        '<div class="curve-key">smoothed frame-difference proxy · red = accepted Action boundary</div>'
    )


def _select(key: str, choices: list[str]) -> str:
    return '<select class="manual" data-storage="' + _e(key) + '">' + "".join(
        f'<option>{_e(choice)}</option>' for choice in choices
    ) + "</select>"


def render_comparison(
    *, output_path: Path, manifest: dict[str, Any], event_rows: list[dict[str, Any]],
    action_rows: list[dict[str, Any]], kts_rows: list[dict[str, Any]],
    motion_rows: list[dict[str, Any]], per_video_metrics: list[dict[str, Any]],
    aggregate_metrics: dict[str, Any], subtle_candidates: list[dict[str, Any]],
    fragmentation: list[dict[str, Any]],
) -> None:
    events = {row["video_id"]: row for row in event_rows}
    actions = {row["video_id"]: row for row in action_rows}
    kts = {row["video_id"]: row for row in kts_rows}
    motions = {row["video_id"]: row for row in motion_rows}
    metrics = {row["video_id"]: row for row in per_video_metrics}
    subtle_by_video: dict[str, list[dict[str, Any]]] = {}
    for row in subtle_candidates:
        subtle_by_video.setdefault(row["video_id"], []).append(row)
    fragment_by_video = {row["video_id"]: row for row in fragmentation}
    cards = []
    for number, item in enumerate(manifest["videos"], start=1):
        video_id = item["video_id"]
        event_row, action_row = events[video_id], actions[video_id]
        duration = float(event_row["video_duration"])
        decision_by_parent = {row["parent_event_id"]: row for row in motions[video_id]["event_decisions"]}
        actions_by_parent: dict[str, list[dict[str, Any]]] = {}
        for action in action_row["segments"]:
            actions_by_parent.setdefault(action["parent_event_id"], []).append(action)
        split_blocks = []
        for event in event_row["segments"]:
            children = actions_by_parent[event["segment_id"]]
            if len(children) <= 1:
                continue
            child_html = []
            for child in children:
                child_html.append(
                    f'<div class="action-child"><b>{_e(child["action_id"])}</b> '
                    f'{float(child["start"]):.1f}-{float(child["end"]):.1f}s '
                    f'({float(child["duration"]):.1f}s){_thumbs(child["representative_frames"])}</div>'
                )
            split_blocks.append(
                f'<details class="split" open><summary><b>{_e(event["segment_id"])}</b> '
                f'{float(event["start"]):.1f}-{float(event["end"]):.1f}s → {len(children)} Actions</summary>'
                f'<h4>Parent Event · 25/50/75%</h4>{_thumbs(event["representative_frames"])}'
                f'{_motion_svg(decision_by_parent[event["segment_id"]], motions[video_id])}'
                f'<div class="children">{"".join(child_html)}</div></details>'
            )
        metric = metrics[video_id]
        frag = fragment_by_video[video_id]
        key = video_id.replace("-", "_")
        subtle = subtle_by_video.get(video_id, [])
        cards.append(f'''<article class="video" id="video-{key}">
<div class="video-head"><div><span class="number">{number}</span><b>{_e(video_id)}</b> <span class="pill">{_e(item['group'])}</span></div><a href="#top">Back to summary</a></div>
<div class="quick"><b>Events {metric['event_level']['segment_count']}</b> → <b>Actions {metric['action_level']['segment_count']}</b> · expansion {metric['action_expansion_ratio']:.2f}× · split parents {100*metric['parent_split_rate']:.1f}% · subtle flags {len(subtle)} · redundant-fragment flags {frag['possible_redundant_fragment_count']}</div>
<div class="timelines">
{_timeline('EVENT_LEVEL', event_row['segments'], duration, 'event')}
{_timeline('ACTION_LEVEL', action_row['segments'], duration, 'action')}
{_timeline('KTS reference only', kts[video_id]['segments'], duration, 'kts')}
</div>
<div class="metric-grid"><div><h3>Event-level</h3><p>Mean/median/max {metric['event_level']['mean_duration_sec']:.1f} / {metric['event_level']['median_duration_sec']:.1f} / {metric['event_level']['max_duration_sec']:.1f}s</p><p>&gt;20s {100*metric['event_level']['fraction_gt_20s']:.1f}% · &lt;4s {100*metric['event_level']['fraction_lt_4s']:.1f}%</p></div><div><h3>Action-level</h3><p>Mean/median/max {metric['action_level']['mean_duration_sec']:.1f} / {metric['action_level']['median_duration_sec']:.1f} / {metric['action_level']['max_duration_sec']:.1f}s</p><p>&gt;20s {100*metric['action_level']['fraction_gt_20s']:.1f}% · &lt;4s {100*metric['action_level']['fraction_lt_4s']:.1f}%</p></div></div>
<div class="notice"><b>Stable-appearance / motion-change diagnostic:</b> {len(subtle)} candidate Event(s). This is a signal proxy, not a semantic action judgment.</div>
<h3>Split Event → Action children and motion curve</h3>
{"".join(split_blocks) if split_blocks else '<p class="muted">No Event was split by the frozen penalized objective.</p>'}
<div class="manual-panel"><h3>Manual review — intentionally blank</h3>
<label>1. Meaningful missed motion revealed? {_select(key+':revealed',['Unreviewed','Yes','No','Unclear'])}</label>
<label>2. New boundaries semantically useful? {_select(key+':useful',['Unreviewed','Mostly useful','Mixed','Mostly redundant','Unclear'])}</label>
<label>3. Continuous action over-fragmented? {_select(key+':overfragment',['Unreviewed','Yes','No','Unclear'])}</label>
<label>4. Event level already enough? {_select(key+':event_enough',['Unreviewed','Yes','No','Unclear'])}</label>
<label>5. Better future Fine leaf? {_select(key+':future_leaf',['Unreviewed','Event-level','Action-level','Hybrid-adaptive','Unclear'])}</label>
<label>6. Stable-background details captured by? {_select(key+':stable_detail',['Unreviewed','Event','Action','Similar','Unclear'])}</label>
<label class="wide">7. Notes<textarea class="manual" data-storage="{key}:notes" rows="3"></textarea></label>
</div><details><summary>Structural diagnostics JSON</summary><pre>{_e(json.dumps({'metrics':metric,'fragmentation':frag,'subtle_candidates':subtle},indent=2,ensure_ascii=False))}</pre></details></article>''')
    all_group = aggregate_metrics["all"]
    links = " ".join(
        f'<a href="#video-{item["video_id"].replace("-", "_")}">{index}</a>'
        for index, item in enumerate(manifest["videos"], start=1)
    )
    document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Event vs Action Fine Leaf v0.1</title><style>
:root{{--bg:#f3f6f8;--paper:#fff;--ink:#1d2935;--line:#ccd7e2;--event:#347a93;--action:#c26532;--kts:#77649d}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,Segoe UI,sans-serif}}header,main{{max-width:1500px;margin:auto}}header{{padding:24px}}main{{padding:0 24px 60px}}h1{{margin:0}}.notice{{background:#fff7dc;border-left:5px solid #d79b00;padding:11px;margin:12px 0}}.summary{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}}.stat,.video{{background:var(--paper);border:1px solid var(--line);border-radius:10px}}.stat{{padding:12px}}.stat b{{font-size:22px;color:#2c6179;display:block}}.video{{padding:17px;margin:22px 0;scroll-margin-top:10px}}.video-head{{display:flex;justify-content:space-between}}.number{{display:inline-grid;place-items:center;width:28px;height:28px;border-radius:50%;background:#2c6179;color:#fff;margin-right:7px}}.pill{{background:#e9eef4;border-radius:999px;padding:3px 8px}}.quick{{font-size:16px;margin:10px 0}}.timelines{{border:1px solid var(--line);border-radius:8px;padding:20px 10px 8px;overflow-x:auto}}.timeline-row{{display:grid;grid-template-columns:165px minmax(900px,1fr);align-items:center;margin:8px 0}}.axis{{height:31px;position:relative;border:1px solid #aebbc7;background:repeating-linear-gradient(to right,#edf2f5 0,#edf2f5 calc(33.333% - 1px),#c5d0db calc(33.333% - 1px),#c5d0db 33.333%)}}.axis i{{position:absolute;top:-18px;font-size:10px;font-style:normal;color:#687684}}.x0{{left:0}}.x60{{left:33.333%}}.x120{{left:66.666%}}.x180{{right:0}}.bar{{position:absolute;top:4px;height:21px;color:white;border:1px solid rgba(0,0,0,.35);text-align:center;font-size:9px;overflow:hidden}}.event{{background:var(--event)}}.action{{background:var(--action)}}.kts{{background:var(--kts)}}.metric-grid,.children{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px}}.metric-grid>div,.action-child{{border:1px solid var(--line);padding:9px;border-radius:7px}}.split{{border-left:5px solid var(--action);padding:8px 10px;margin:10px 0;background:#fffaf7}}.thumbs{{display:flex;gap:6px;overflow-x:auto;margin:7px 0}}figure{{margin:0;min-width:130px}}img{{width:130px;height:80px;object-fit:cover;border:1px solid #bac5cf;border-radius:4px}}figcaption{{font-size:10px;color:#61707e}}.motion{{width:100%;height:90px;background:#eef3f6;border:1px solid #cad4de}}.motion polyline{{fill:none;stroke:#2c6179;stroke-width:1}}.motion .cp{{stroke:#d13c33;stroke-width:.55}}.curve-key{{font-size:11px;color:#687684}}.manual-panel{{display:grid;grid-template-columns:1fr 1fr;gap:9px;background:#edf4f8;padding:12px;border-radius:8px;margin-top:14px}}.manual-panel h3,.wide{{grid-column:1/-1}}label{{display:flex;justify-content:space-between;gap:8px}}textarea{{width:100%}}pre{{white-space:pre-wrap;max-height:450px;overflow:auto;background:#111a24;color:#e7edf3;padding:12px}}.muted{{color:#697986}}@media(max-width:850px){{.manual-panel{{grid-template-columns:1fr}}.manual-panel h3,.wide{{grid-column:auto}}}}
</style></head><body><header id="top"><h1>Event-level vs Action-level Fine leaves v0.1</h1><p>Frozen Event segments versus within-Event <b>LIGHTWEIGHT_MOTION_PROXY</b> refinement on the same 10 videos.</p><div class="notice"><b>Interpretation boundary:</b> Action-level uses 1 FPS frame differences, not RAFT and not full CoMET Action segmentation. KTS is visual reference only. No question, option, gold, Planner, QA, or external API is used.</div><div class="summary"><div class="stat"><b>{aggregate_metrics['video_count']}</b>videos</div><div class="stat"><b>{all_group['event_level']['total_segments']}</b>frozen Events</div><div class="stat"><b>{all_group['action_level']['total_segments']}</b>Action leaves</div><div class="stat"><b>{all_group['action_expansion_ratio']:.2f}×</b>Action expansion</div><div class="stat"><b>{100*all_group['parent_split_rate']:.1f}%</b>parents split</div><div class="stat"><b>{aggregate_metrics['subtle_action_candidate_count']}</b>subtle-motion flags</div></div><p>Jump to: {links}</p></header><main>{''.join(cards)}</main><script>document.querySelectorAll('.manual').forEach(el=>{{const k='event-action-review:'+el.dataset.storage;const v=localStorage.getItem(k);if(v!==null)el.value=v;el.addEventListener('change',()=>localStorage.setItem(k,el.value));}});</script></body></html>'''
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
