"""Render a self-contained human-review report for the three-way experiment."""

from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/experiments/coarse_segmentation_3way_v0_1"
MANIFEST = ROOT / "data/manifests/coarse_segmentation_3way_10.json"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def h(value: Any) -> str:
    return html.escape(str(value))


def image_data(path: Path) -> str:
    with Image.open(path).convert("RGB") as image:
        image.thumbnail((150, 95))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=72, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def representative_path(video_id: str, timestamp: float) -> Path:
    index = max(0, min(179, int(round(timestamp))))
    return ROOT / "outputs/visual_index" / video_id / "frames_1fps" / f"frame_{index:06d}.jpg"


def metric_table(aggregate: dict[str, Any], group: str) -> str:
    rows = []
    for method, values in aggregate[group].items():
        rows.append(
            f"<tr><td>{h(method)}</td><td>{values['number_of_segments']:.2f}</td>"
            f"<td>{values['mean_segment_duration']:.2f}s</td><td>{values['median_segment_duration']:.2f}s</td>"
            f"<td>{values['max_segment_duration']:.2f}s</td><td>{values['largest_region_ratio']:.3f}</td>"
            f"<td>{values['short_segment_fraction_lt_4s']:.3f}</td><td>{values['boundaries_per_minute']:.2f}</td></tr>"
        )
    return "".join(rows)


def retrieval_table(aggregate: dict[str, Any], group: str) -> str:
    return "".join(
        f"<tr><td>{h(method)}</td><td>{values['refinement_search_space_seconds']:.1f}s</td>"
        f"<td>{values['refinement_search_space_ratio']:.3f}</td>"
        f"<td>{values['top_k_overlap_redundancy_seconds']:.1f}s</td></tr>"
        for method, values in aggregate[group].items()
    )


def segment_timeline(
    record: dict[str, Any], selected_ids: set[str]
) -> str:
    duration = float(record["video_duration"])
    bars = []
    for segment in record["segments"]:
        start, end = float(segment["start"]), float(segment["end"])
        left = 100.0 * start / duration
        width = 100.0 * (end - start) / duration
        selected = " selected" if segment["segment_id"] in selected_ids else ""
        label = f"{start:.0f}–{end:.0f}s" if width > 6 else ""
        bars.append(
            f"<div class='seg {record['method']}{selected}' style='left:{left:.5f}%;width:{width:.5f}%' "
            f"title='{h(segment['segment_id'])}: {start:.1f}–{end:.1f}s ({end-start:.1f}s)'>{label}</div>"
        )
    return "<div class='timeline-track'>" + "".join(bars) + "</div>"


def segment_gallery(record: dict[str, Any]) -> str:
    cards = []
    for segment in record["segments"]:
        timestamp = float(segment["representative_frame_timestamp"])
        path = representative_path(record["video_id"], timestamp)
        cards.append(
            f"<figure><img loading='lazy' src='{image_data(path)}'><figcaption>"
            f"{h(segment['segment_id'])}<br>{segment['start']:.1f}–{segment['end']:.1f}s "
            f"({segment['duration']:.1f}s)<br>keyframe {timestamp:.1f}s</figcaption></figure>"
        )
    return "<div class='gallery'>" + "".join(cards) + "</div>"


def curve_svg(raw: list[float], smooth: list[float], comet_bounds: list[float], kts_bounds: list[float]) -> str:
    width, height, pad = 1100, 230, 30
    values = [*raw, *smooth]
    low, high = min(values), max(values)
    span = max(high - low, 1e-9)

    def points(series: list[float]) -> str:
        return " ".join(
            f"{pad + index/(len(series)-1)*(width-2*pad):.2f},{height-pad-(value-low)/span*(height-2*pad):.2f}"
            for index, value in enumerate(series)
        )

    lines = []
    for value in comet_bounds:
        x = pad + value / 180.0 * (width - 2 * pad)
        lines.append(f"<line x1='{x:.2f}' x2='{x:.2f}' y1='{pad}' y2='{height-pad}' class='comet-bound'/>")
    for value in kts_bounds:
        x = pad + value / 180.0 * (width - 2 * pad)
        lines.append(f"<line x1='{x:.2f}' x2='{x:.2f}' y1='{pad}' y2='{height-pad}' class='kts-bound'/>")
    return (
        f"<svg viewBox='0 0 {width} {height}' class='curve'><polyline points='{points(raw)}' class='raw'/>"
        f"<polyline points='{points(smooth)}' class='smooth'/>{''.join(lines)}"
        f"<text x='35' y='18'>DINO adjacent cosine: raw gray / smoothed blue; CoMET orange / KTS purple</text></svg>"
    )


def objective_svg(values: list[float]) -> str:
    width, height, pad = 500, 150, 22
    low, high = min(values), max(values)
    span = max(high - low, 1e-9)
    points = " ".join(
        f"{pad + index/max(1,len(values)-1)*(width-2*pad):.2f},{height-pad-(value-low)/span*(height-2*pad):.2f}"
        for index, value in enumerate(values)
    )
    return f"<svg viewBox='0 0 {width} {height}' class='objective'><polyline points='{points}'/><text x='25' y='16'>KTS objective by segment count</text></svg>"


def main() -> int:
    manifest = load(MANIFEST)
    aggregate = load(OUT / "aggregate_metrics.json")
    retrieval_aggregate = load(OUT / "aggregate_retrieval.json")
    metrics = load(OUT / "per_video_metrics.json")
    diagnostics = {row["video_id"]: row for row in load(OUT / "method_diagnostics.json")}
    retrieval = load_jsonl(OUT / "retrieval_replay.jsonl")
    segmentations = {}
    for filename in ("current_segments.jsonl", "comet_segments.jsonl", "kts_segments.jsonl"):
        for record in load_jsonl(OUT / filename):
            segmentations[(record["video_id"], record["method"])] = record
    metric_map = {(row["video_id"], row["method"]): row for row in metrics}
    retrieval_map = {(row["video_id"], row["method"]): row for row in retrieval}
    sections = []
    navigation = []
    labels = {"current": "CURRENT Ours-v0.1", "comet_style_dinov2": "CoMET-style DINOv2", "kts_dinov2": "DINOv2 + KTS"}
    for row in manifest["videos"]:
        video_id = row["video_id"]
        navigation.append(f"<a href='#{video_id}'>{h(row['group'][0].upper())}{row['selection_rank']}</a>")
        method_blocks = []
        for method in ("current", "comet_style_dinov2", "kts_dinov2"):
            record = segmentations[(video_id, method)]
            metric = metric_map[(video_id, method)]
            replay = retrieval_map[(video_id, method)]
            selected = {item["segment_id"] for item in replay["selected_top_k"]}
            method_blocks.append(
                f"<div class='method'><h3>{labels[method]}</h3>"
                f"<p>segments={metric['number_of_segments']} · max={metric['max_segment_duration']:.1f}s · "
                f"largest ratio={metric['largest_region_ratio']:.3f} · boundaries/min={metric['boundaries_per_minute']:.2f} · "
                f"Top-3 search={replay['refinement_search_space_seconds']:.1f}s ({replay['refinement_search_space_ratio']:.1%})</p>"
                f"{segment_timeline(record, selected)}{segment_gallery(record)}</div>"
            )
        diag = diagnostics[video_id]
        comet_bounds = diag["comet"]["boundaries"]
        kts_bounds = diag["kts"]["boundaries"]
        review = (
            f"<div class='manual' data-video='{video_id}'><h3>Manual review</h3>"
            "<label>Boundary quality <select><option>unreviewed</option><option>poor</option><option>mixed</option><option>good</option></select></label>"
            "<label>Within-segment semantic coherence <select><option>unreviewed</option><option>poor</option><option>mixed</option><option>good</option></select></label>"
            "<label>Under-segmentation <select><option>unreviewed</option><option>none</option><option>some</option><option>severe</option></select></label>"
            "<label>Over-segmentation <select><option>unreviewed</option><option>none</option><option>some</option><option>severe</option></select></label>"
            "<label>Most useful event-level index <select><option>unreviewed</option><option>current</option><option>comet_style_dinov2</option><option>kts_dinov2</option></select></label>"
            "<label>Notes <textarea></textarea></label></div>"
        )
        sections.append(
            f"<section id='{video_id}'><header><h2>{h(video_id)}</h2><span>{h(row['group'])} rank {row['selection_rank']} · current ratio {row['current_largest_region_ratio']:.3f}</span></header>"
            f"<div class='axis-labels'><span>0s</span><span>30</span><span>60</span><span>90</span><span>120</span><span>150</span><span>180s</span></div>"
            f"{''.join(method_blocks)}<h3>Shared DINO signal and boundary overlays</h3>"
            f"{curve_svg(diag['comet']['raw_similarity'],diag['comet']['smoothed_similarity'],comet_bounds,kts_bounds)}"
            f"<div class='diag'><p><b>CoMET boundaries:</b> {h(comet_bounds)}</p><p><b>KTS boundaries:</b> {h(kts_bounds)}</p>"
            f"{objective_svg(diag['kts']['diagnostics']['objective_by_segment_count'])}</div>{review}</section>"
        )
    tables = []
    for group in ("all", "problematic", "control"):
        tables.append(
            f"<h3>{group.title()} segmentation metrics</h3><table><thead><tr><th>Method</th><th>Segments</th><th>Mean duration</th><th>Median</th><th>Max</th><th>Largest ratio</th><th>&lt;4s fraction</th><th>Boundaries/min</th></tr></thead><tbody>{metric_table(aggregate,group)}</tbody></table>"
            f"<h3>{group.title()} Top-3 refinement search space</h3><table><thead><tr><th>Method</th><th>Seconds</th><th>Ratio</th><th>Overlap redundancy</th></tr></thead><tbody>{retrieval_table(retrieval_aggregate,group)}</tbody></table>"
        )
    document = f"""<!doctype html><html><head><meta charset='utf-8'><title>Coarse segmentation 3-way v0.1</title><style>
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:#f4f6fa;color:#172033;font:14px/1.45 system-ui,sans-serif}}nav{{position:sticky;top:0;z-index:20;background:#172033;padding:8px;display:flex;gap:6px}}nav a{{color:white;background:#445269;padding:5px 9px;border-radius:5px;text-decoration:none}}main{{max-width:1450px;margin:auto;padding:20px}}section,.summary{{background:white;border:1px solid #d8deea;border-radius:12px;padding:20px;margin:20px 0;box-shadow:0 2px 8px #0001}}header{{display:flex;justify-content:space-between;align-items:center}}table{{border-collapse:collapse;width:100%;margin-bottom:18px}}th,td{{border:1px solid #cbd2dc;padding:6px;text-align:right}}th:first-child,td:first-child{{text-align:left}}.axis-labels{{display:flex;justify-content:space-between;margin:0 0 3px 0;color:#667085}}.method{{border-top:1px solid #dae0e9;padding-top:8px}}.timeline-track{{height:42px;position:relative;background:repeating-linear-gradient(to right,#eef2f7 0,#eef2f7 calc(16.666% - 1px),#bdc7d5 calc(16.666% - 1px),#bdc7d5 16.666%);overflow:hidden}}.seg{{position:absolute;top:4px;height:34px;border-right:2px solid white;padding:7px 2px;font-size:10px;overflow:hidden;white-space:nowrap}}.seg.current{{background:#6f8fae}}.seg.comet_style_dinov2{{background:#e8a144}}.seg.kts_dinov2{{background:#9b72cf}}.seg.selected{{outline:4px solid #111;z-index:2}}.gallery{{display:flex;gap:7px;overflow-x:auto;padding:8px 0 16px}}figure{{margin:0;min-width:150px}}figure img{{width:150px;height:95px;object-fit:cover;border-radius:5px}}figcaption{{font-size:10px}}svg{{width:100%;background:#f9fafc;border:1px solid #d6dce7}}.curve .raw{{fill:none;stroke:#999;stroke-width:1}}.curve .smooth{{fill:none;stroke:#1769aa;stroke-width:2}}.comet-bound{{stroke:#ef6c00;stroke-width:1.3}}.kts-bound{{stroke:#6a1b9a;stroke-width:1.3;stroke-dasharray:4 2}}.objective{{max-width:520px}}.objective polyline{{fill:none;stroke:#6a1b9a;stroke-width:2}}.manual{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;background:#eef3f8;padding:12px;border-radius:8px}}.manual h3{{grid-column:1/-1}}label{{display:flex;flex-direction:column;gap:4px}}textarea{{min-height:70px}}.notice{{padding:12px;background:#fff4ce;border-left:5px solid #e6a700}}@media(max-width:800px){{.manual{{grid-template-columns:1fr}}}}
</style></head><body><nav>{''.join(navigation)}</nav><main><div class='summary'><h1>Coarse temporal segmentation 3-way v0.1</h1><div class='notice'>No automatic winner is declared. Metrics expose under-segmentation, fragmentation, and CLIP Top-3 refinement search-space trade-offs. Gold answers, options, previous correctness, Planner, Gemini, and paid APIs were not used.</div>{''.join(tables)}</div>{''.join(sections)}</main><script>
document.querySelectorAll('.manual').forEach(box=>{{let key='coarse-review-'+box.dataset.video;let controls=[...box.querySelectorAll('select,textarea')];let saved=JSON.parse(localStorage.getItem(key)||'null');if(saved)controls.forEach((c,i)=>c.value=saved[i]||c.value);controls.forEach(c=>c.addEventListener('change',()=>localStorage.setItem(key,JSON.stringify(controls.map(x=>x.value)))));}});
</script></body></html>"""
    (OUT / "comparison.html").write_text(document, encoding="utf-8")
    print(json.dumps({"videos": len(sections), "output": (OUT / 'comparison.html').as_posix(), "bytes": len(document.encode('utf-8'))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
