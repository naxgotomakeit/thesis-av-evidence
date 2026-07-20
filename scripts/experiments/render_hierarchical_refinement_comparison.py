"""Render a self-contained human-review report for hierarchical refinement."""

from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/experiments/hierarchical_refinement_comparison_v0_1"
PREVIOUS = ROOT / "outputs/experiments/coarse_segmentation_3way_v0_1"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def h(value: Any) -> str:
    return html.escape(str(value))


def frame_path(video_id: str, timestamp: float) -> Path:
    index = max(0, min(179, int(round(timestamp))))
    return ROOT / "outputs/visual_index" / video_id / "frames_1fps" / f"frame_{index:06d}.jpg"


def image_data(path: Path) -> str:
    with Image.open(path).convert("RGB") as image:
        image.thumbnail((180, 112))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=70, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def selected_gallery(video_id: str, selected: list[dict[str, Any]], label: str) -> str:
    cards = []
    for row in selected:
        start, end = float(row["start"]), float(row["end"])
        for fraction, position in ((0.25, "25%"), (0.50, "50%"), (0.75, "75%")):
            timestamp = min(179.0, start + (end - start) * fraction)
            cards.append(
                f"<figure><img loading='lazy' src='{image_data(frame_path(video_id, timestamp))}'>"
                f"<figcaption><b>{h(label)} rank {row.get('rank', '—')}</b><br>"
                f"{h(row['segment_id'])}<br>{start:.1f}–{end:.1f}s · {position} @ {timestamp:.1f}s"
                f"<br>score {float(row.get('score', 0.0)):.5f}</figcaption></figure>"
            )
    return "<div class='gallery'>" + "".join(cards) + "</div>"


def axis() -> str:
    return "<div class='axis'><span>0s</span><span>30</span><span>60</span><span>90</span><span>120</span><span>150</span><span>180s</span></div>"


def timeline(
    rows: list[dict[str, Any]],
    *,
    css_class: str,
    selected_ids: set[str] | None = None,
    score_by_id: dict[str, dict[str, Any]] | None = None,
) -> str:
    selected_ids = selected_ids or set()
    score_by_id = score_by_id or {}
    bars = []
    for row in rows:
        start, end = float(row["start"]), float(row["end"])
        identifier = str(row.get("segment_id", row.get("zone_id")))
        score = score_by_id.get(identifier, {}).get("score")
        rank = score_by_id.get(identifier, {}).get("rank")
        selected = " selected" if identifier in selected_ids else ""
        left, width = start / 1.8, (end - start) / 1.8
        title = f"{identifier}: {start:.1f}–{end:.1f}s"
        if score is not None:
            title += f" | rank {rank}, score {float(score):.5f}"
        label = f"{start:.0f}–{end:.0f}" if width >= 5 else ""
        bars.append(
            f"<div class='bar {css_class}{selected}' style='left:{left:.5f}%;width:{width:.5f}%' "
            f"title='{h(title)}'>{h(label)}</div>"
        )
    return "<div class='track'>" + "".join(bars) + "</div>"


def ranking_table(rows: list[dict[str, Any]], selected_ids: set[str]) -> str:
    body = []
    for row in rows:
        identifier = str(row["segment_id"])
        body.append(
            f"<tr class='{'picked' if identifier in selected_ids else ''}'><td>{row['rank']}</td>"
            f"<td>{h(identifier)}</td><td>{float(row['start']):.1f}–{float(row['end']):.1f}s</td>"
            f"<td>{float(row['duration']):.1f}s</td><td>{float(row['score']):.5f}</td>"
            f"<td>{'Top-3' if identifier in selected_ids else 'not selected'}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Rank</th><th>ID</th><th>Interval</th><th>Duration</th>"
        "<th>CLIP score</th><th>Decision</th></tr></thead><tbody>" + "".join(body) + "</tbody></table>"
    )


def metric_table(aggregate: dict[str, Any]) -> str:
    rows = []
    for group in ("all", "problematic", "control"):
        value = aggregate["groups"][group]
        rows.append(
            f"<tr><td>{h(group)}</td><td>{value['case_count']}</td>"
            f"<td>{value['kts_candidates_scored']:.1f}</td><td>{value['kts_selected_duration']:.1f}s / {value['kts_selected_coverage_ratio']:.1%}</td>"
            f"<td>{value['full_comet_candidates_scored']:.1f}</td><td>{value['full_comet_selected_duration']:.1f}s / {value['full_comet_selected_coverage_ratio']:.1%}</td>"
            f"<td>{value['candidate_zone_count']:.1f}</td><td>{value['refinement_input_seconds']:.1f}s / {value['refinement_input_ratio']:.1%}</td>"
            f"<td>{value['local_comet_candidates_generated']:.1f}</td><td>{value['local_final_duration']:.1f}s / {value['local_final_coverage_ratio']:.1%}</td>"
            f"<td>{value['fine_candidate_reduction']:.1%}</td><td>{value['fine_temporal_processing_reduction']:.1%}</td>"
            f"<td>{value['full_comet_top1_containment']:.1%}</td><td>{value['full_comet_top3_containment_rate']:.1%}</td>"
            f"<td>{value['temporal_overlap_with_full_comet_top3_ratio']:.1%}</td></tr>"
        )
    return "".join(rows)


def sensitivity_table(aggregate: dict[str, Any]) -> str:
    rows = []
    for group in ("all", "problematic", "control"):
        for top_m in ("1", "2", "3"):
            value = aggregate["top_m_sensitivity"][group][top_m]
            rows.append(
                f"<tr><td>{h(group)}</td><td>{top_m}</td><td>{value['candidate_zone_count']:.1f}</td>"
                f"<td>{value['candidate_zone_duration']:.1f}s</td><td>{value['candidate_zone_coverage_ratio']:.1%}</td>"
                f"<td>{value['local_comet_segment_count']:.1f}</td><td>{value['full_comet_top1_contained']:.1%}</td>"
                f"<td>{value['full_comet_top3_containment_rate']:.1%}</td></tr>"
            )
    return "".join(rows)


def manual_review(video_id: str) -> str:
    options = {
        "method": "<option>unreviewed</option><option>KTS_ONLY</option><option>FULL_COMET</option><option>KTS_LOCAL_COMET</option><option>unclear</option>",
        "quality": "<option>unreviewed</option><option>Good</option><option>Partial</option><option>Poor</option>",
        "redundancy": "<option>unreviewed</option><option>Low</option><option>Medium</option><option>High</option>",
        "yn": "<option>unreviewed</option><option>Yes</option><option>No</option><option>Unclear</option>",
        "compare": "<option>unreviewed</option><option>Better</option><option>Similar</option><option>Worse</option><option>Unclear</option>",
    }
    return f"""<div class='manual' data-video='{h(video_id)}'><h3>Manual review — intentionally unfilled</h3>
<label>Fine-detail localization<select>{options['method']}</select></label>
<label>KTS_ONLY context sufficiency<select>{options['quality']}</select></label>
<label>FULL_COMET context sufficiency<select>{options['quality']}</select></label>
<label>KTS_LOCAL_COMET context sufficiency<select>{options['quality']}</select></label>
<label>KTS_ONLY redundancy<select>{options['redundancy']}</select></label>
<label>FULL_COMET redundancy<select>{options['redundancy']}</select></label>
<label>KTS_LOCAL_COMET redundancy<select>{options['redundancy']}</select></label>
<label>Important evidence lost by KTS gating?<select>{options['yn']}</select></label>
<label>Does local CoMET add useful detail?<select>{options['yn']}</select></label>
<label>Hybrid comparable to FULL_COMET?<select>{options['compare']}</select></label>
<label>Best quality-efficiency trade-off<select>{options['method']}</select></label>
<label class='notes'>Notes<textarea></textarea></label></div>"""


def main() -> int:
    aggregate = load(OUT / "aggregate_metrics.json")
    metrics = {row["video_id"]: row for row in load(OUT / "per_case_metrics.json")}
    kts_rows = load_jsonl(OUT / "kts_only_results.jsonl")
    full_rows = load_jsonl(OUT / "full_comet_results.jsonl")
    local_rows = load_jsonl(OUT / "kts_local_comet_results.jsonl")
    sensitivity = load(OUT / "top_m_sensitivity.json")["per_case"]
    kts_segments = {row["video_id"]: row for row in load_jsonl(PREVIOUS / "kts_segments.jsonl")}
    comet_segments = {row["video_id"]: row for row in load_jsonl(PREVIOUS / "comet_segments.jsonl")}
    kts_map, full_map, local_map = (
        {row["video_id"]: row for row in rows} for rows in (kts_rows, full_rows, local_rows)
    )
    sections, navigation = [], []
    for index, kts in enumerate(kts_rows, start=1):
        video_id = kts["video_id"]
        full, local, metric = full_map[video_id], local_map[video_id], metrics[video_id]
        navigation.append(f"<a href='#{h(video_id)}'>{index}</a>")
        kts_selected = {row["segment_id"] for row in kts["selected_top_k"]}
        full_selected = {row["segment_id"] for row in full["selected_top_k"]}
        local_selected = {row["segment_id"] for row in local["selected_top_k"]}
        kts_scores = {row["segment_id"]: row for row in kts["ranking"]}
        full_scores = {row["segment_id"]: row for row in full["ranking"]}
        local_scores = {row["segment_id"]: row for row in local["local_ranking"]}
        zone_rows = local["candidate_zones"]
        zone_list = "".join(
            f"<li><b>{h(zone['zone_id'])}</b> {zone['start']:.1f}–{zone['end']:.1f}s: "
            f"{h(zone['selected_kts_ids'])}; ranks {h(zone['selection_ranks'])}; "
            f"merged edges {len(zone['merge_decisions'])}</li>" for zone in zone_rows
        )
        local_sensitivity = sorted(
            [row for row in sensitivity if row["video_id"] == video_id], key=lambda row: row["top_m"]
        )
        sensitivity_rows = "".join(
            f"<tr><td>{row['top_m']}</td><td>{row['candidate_zone_count']}</td>"
            f"<td>{row['candidate_zone_duration']:.1f}s / {row['candidate_zone_coverage_ratio']:.1%}</td>"
            f"<td>{row['local_comet_segment_count']}</td><td>{'yes' if row['full_comet_top1_contained'] else 'no'}</td>"
            f"<td>{row['full_comet_top3_containment_rate']:.1%}</td></tr>" for row in local_sensitivity
        )
        section = f"""<section id='{h(video_id)}'>
<header><div><span class='case-no'>Case {index}/10 · {h(kts['group'])}</span><h2>{h(kts['question'])}</h2><code>{h(video_id)}</code></div><a href='#top'>Back to summary</a></header>
<div class='method kts'><h3>A. KTS_ONLY</h3><p>All {kts['total_kts_units']} KTS units scored; Top-3 covers <b>{kts['selected_unique_duration']:.1f}s ({kts['selected_coverage_ratio']:.1%})</b>.</p>
{axis()}{timeline(kts_segments[video_id]['segments'], css_class='kts-bar', selected_ids=kts_selected, score_by_id=kts_scores)}
{ranking_table(kts['ranking'], kts_selected)}<h4>Selected unit context: 25% / 50% / 75%</h4>{selected_gallery(video_id, kts['selected_top_k'], 'KTS')}</div>
<div class='method comet'><h3>B. FULL_COMET</h3><p>Existing full-video fine segmentation: {full['total_full_comet_segments']} candidates scored; Top-3 covers <b>{full['selected_unique_duration']:.1f}s ({full['selected_coverage_ratio']:.1%})</b>.</p>
{axis()}{timeline(comet_segments[video_id]['segments'], css_class='comet-bar', selected_ids=full_selected, score_by_id=full_scores)}
{ranking_table(full['ranking'], full_selected)}<h4>Selected fine context: 25% / 50% / 75%</h4>{selected_gallery(video_id, full['selected_top_k'], 'FULL_COMET')}</div>
<div class='method hybrid'><h3>C. KTS_LOCAL_COMET</h3>
<div class='stats'><span>KTS units scored <b>{local['kts_units_scored']}</b></span><span>zones <b>{local['candidate_zone_count']}</b></span><span>refinement input <b>{local['refinement_input_seconds']:.1f}s / {local['refinement_input_ratio']:.1%}</b></span><span>local candidates <b>{local['local_comet_candidate_count']}</b></span><span>candidate reduction <b>{local['fine_candidate_reduction']:.1%}</b></span><span>temporal reduction <b>{local['fine_temporal_processing_reduction']:.1%}</b></span></div>
<h4>Level 1 — KTS retrieval</h4>{axis()}{timeline(kts_segments[video_id]['segments'], css_class='kts-bar', selected_ids=kts_selected, score_by_id=kts_scores)}
<h4>Selected/merged candidate zones</h4>{axis()}{timeline(zone_rows, css_class='zone-bar')}<ul>{zone_list}</ul>
<h4>Level 2 — local CoMET only inside zones</h4>{axis()}{timeline(local['local_comet_segments'], css_class='local-bar', selected_ids=local_selected, score_by_id=local_scores)}
{ranking_table(local['local_ranking'], local_selected)}<h4>Final local fine evidence: 25% / 50% / 75%</h4>{selected_gallery(video_id, local['selected_top_k'], 'LOCAL_COMET')}
<div class='proxy'><b>DIAGNOSTIC PROXY ONLY:</b> FULL_COMET Top-1 contained = {str(local['containment_proxy']['full_comet_top1_contained']).lower()}; Top-3 midpoint containment = {local['containment_proxy']['full_comet_top3_containment_rate']:.1%}; final temporal overlap = {local['containment_proxy']['temporal_overlap_with_full_comet_top3_ratio']:.1%}.</div>
<h4>Top-M sensitivity</h4><table><thead><tr><th>M</th><th>Zones</th><th>Input</th><th>Local candidates</th><th>Top-1 proxy</th><th>Top-3 proxy</th></tr></thead><tbody>{sensitivity_rows}</tbody></table></div>
{manual_review(video_id)}</section>"""
        sections.append(section)
    if len(sections) != 10:
        raise RuntimeError(f"Expected 10 rendered cases, got {len(sections)}")
    summary = f"""<div id='top' class='summary'><h1>Hierarchical refinement comparison v0.1</h1>
<div class='notice'><b>Structural experiment only.</b> FULL_COMET containment is a diagnostic proxy, not ground-truth recall. No Planner, Task5C, Task6, final QA, answer options, gold, previous correctness, or paid API is used. Human judgments below are deliberately unfilled.</div>
<h2>Primary Top-3 comparison</h2><table><thead><tr><th>Group</th><th>N</th><th>KTS candidates</th><th>KTS Top-3</th><th>Full fine candidates</th><th>Full fine Top-3</th><th>Hybrid zones</th><th>Hybrid input</th><th>Local candidates</th><th>Local Top-3</th><th>Candidate reduction</th><th>Temporal reduction</th><th>Top-1 proxy</th><th>Top-3 proxy</th><th>Overlap proxy</th></tr></thead><tbody>{metric_table(aggregate)}</tbody></table>
<h2>Top-M sensitivity</h2><table><thead><tr><th>Group</th><th>M</th><th>Zones</th><th>Duration</th><th>Coverage</th><th>Local candidates</th><th>Top-1 proxy</th><th>Top-3 proxy</th></tr></thead><tbody>{sensitivity_table(aggregate)}</tbody></table>
<p class='legend'><span class='swatch kts-bar'></span>KTS <span class='swatch comet-bar'></span>full CoMET <span class='swatch zone-bar'></span>candidate zone <span class='swatch local-bar'></span>local CoMET <span class='selected-demo'>outlined</span> Top-3 selected</p></div>"""
    document = f"""<!doctype html><html><head><meta charset='utf-8'><title>Hierarchical refinement comparison v0.1</title><style>
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:#f3f5f8;color:#182230;font:14px/1.45 system-ui,sans-serif}}nav{{position:sticky;top:0;z-index:50;background:#162033;padding:8px;display:flex;gap:5px}}nav a{{color:#fff;text-decoration:none;background:#394a63;border-radius:5px;padding:5px 10px}}main{{max-width:1500px;margin:auto;padding:18px}}section,.summary{{background:#fff;border:1px solid #d6dde8;border-radius:12px;padding:20px;margin:18px 0;box-shadow:0 2px 9px #0001}}header{{display:flex;justify-content:space-between;align-items:flex-start;gap:20px}}header h2{{max-width:1100px;margin:5px 0}}header>a{{white-space:nowrap}}.case-no{{font-weight:700;color:#526173}}.notice,.proxy{{padding:12px;background:#fff5d6;border-left:5px solid #e0a000;margin:12px 0}}.method{{border-top:4px solid #cfd8e4;padding:14px;margin-top:22px}}.method.kts{{border-color:#6d8eae}}.method.comet{{border-color:#e39a38}}.method.hybrid{{border-color:#2b9b73}}.axis{{display:flex;justify-content:space-between;color:#64748b;font-size:11px}}.track{{height:44px;position:relative;background:repeating-linear-gradient(to right,#edf1f6 0,#edf1f6 calc(16.666% - 1px),#b9c5d4 calc(16.666% - 1px),#b9c5d4 16.666%);overflow:hidden;margin-bottom:12px}}.bar{{position:absolute;top:4px;height:36px;border-right:2px solid #fff;padding:8px 2px;font-size:10px;overflow:hidden;white-space:nowrap}}.kts-bar{{background:#7597b7}}.comet-bar{{background:#eba64b}}.zone-bar{{background:#46b58d}}.local-bar{{background:#b47dd1}}.bar.selected{{outline:4px solid #101820;z-index:2}}.swatch{{display:inline-block;width:22px;height:12px;margin:0 5px 0 16px}}.selected-demo{{border:3px solid #101820;padding:1px 4px;margin-left:16px}}table{{border-collapse:collapse;width:100%;margin:10px 0 18px}}th,td{{border:1px solid #cbd4df;padding:6px;text-align:right}}th:nth-child(2),td:nth-child(2){{text-align:left}}tr.picked{{background:#e5f5ec;font-weight:650}}.gallery{{display:flex;gap:8px;overflow-x:auto;padding:8px 0 18px}}figure{{margin:0;min-width:180px}}figure img{{width:180px;height:112px;object-fit:cover;border-radius:6px}}figcaption{{font-size:10px}}.stats{{display:flex;flex-wrap:wrap;gap:8px}}.stats span{{background:#e9f6f1;padding:7px;border-radius:6px}}.manual{{display:grid;grid-template-columns:repeat(3,1fr);gap:9px;background:#edf2f7;padding:14px;border-radius:8px;margin-top:20px}}.manual h3,.manual .notes{{grid-column:1/-1}}label{{display:flex;flex-direction:column;gap:4px}}textarea{{min-height:75px}}code{{font-size:11px}}@media(max-width:800px){{.manual{{grid-template-columns:1fr}}.manual h3,.manual .notes{{grid-column:auto}}}}
</style></head><body><nav><a href='#top'>Summary</a>{''.join(navigation)}</nav><main>{summary}{''.join(sections)}</main><script>
document.querySelectorAll('.manual').forEach(box=>{{let key='hier-refine-'+box.dataset.video;let cs=[...box.querySelectorAll('select,textarea')];let saved=JSON.parse(localStorage.getItem(key)||'null');if(saved)cs.forEach((c,i)=>c.value=saved[i]||c.value);cs.forEach(c=>c.addEventListener('change',()=>localStorage.setItem(key,JSON.stringify(cs.map(x=>x.value)))));}});
</script></body></html>"""
    path = OUT / "comparison.html"
    path.write_text(document, encoding="utf-8")
    print(json.dumps({"rendered_cases": len(sections), "output": path.as_posix(), "bytes": len(document.encode("utf-8"))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
