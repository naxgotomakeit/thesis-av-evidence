#!/usr/bin/env python3
"""Build a static visual audit from existing YKI08 segmentation artifacts.

This script performs visualization only. It never loads DINOv2, C-RADIO, or
Qwen and never recomputes embeddings or segmentation.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import decord
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
THESIS_ROOT = ROOT.parents[1]
DEFAULT_RESULT = ROOT / "outputs/diagnostics/cradio_v4/dino_comparison/dino_comparison_results.json"
DEFAULT_VIDEO = THESIS_ROOT / "data/EgoPolice_1.0.0/videos/pasadena/YKI08.mp4"
DEFAULT_HTML = ROOT / "docs/reports/CRADIO_DINO_SEGMENTATION_VISUAL_AUDIT.html"


def _record_map(replay: dict[str, Any]) -> dict[float, dict[str, Any]]:
    return {float(row["timestamp"]): row for row in replay["boundary_records"]}


def _nearest(values: Iterable[float], target: float) -> tuple[float, float]:
    value = min(values, key=lambda item: (abs(item - target), item))
    return value, abs(value - target)


def _strength(record: dict[str, Any]) -> float:
    return float(record["smoothed_change_strength"])


def _take_with_forced(
    candidates: list[Any], count: int, forced_predicate: Any, score: Any
) -> list[Any]:
    ordered = sorted(candidates, key=lambda row: (-score(row), repr(row)))
    chosen: list[Any] = []
    forced = next((row for row in ordered if forced_predicate(row)), None)
    if forced is not None:
        chosen.append(forced)
    for row in ordered:
        if row not in chosen:
            chosen.append(row)
        if len(chosen) == count:
            break
    if len(chosen) != count:
        raise RuntimeError(f"Could not select {count} boundary examples")
    return chosen


def select_boundary_examples(result: dict[str, Any]) -> list[dict[str, Any]]:
    dino = result["representations"]["dinov2"]
    cradio = result["representations"]["cradio"]
    d_records = _record_map(dino)
    c_records = _record_map(cradio)
    d_times = sorted(d_records)
    c_times = sorted(c_records)
    matches = result["boundary_agreement"]["exact_frozen_threshold_outputs"]["2.0"]["matches"]
    pairs = [(float(row["left"]), float(row["right"]), float(row["distance_sec"])) for row in matches]

    exact = [row for row in pairs if row[2] == 0.0]
    close = [row for row in pairs if 0.0 < row[2] <= 2.0]
    d_far = [(timestamp, *_nearest(c_times, timestamp)) for timestamp in d_times]
    d_far = [row for row in d_far if row[2] > 5.0]
    c_far = [(timestamp, *_nearest(d_times, timestamp)) for timestamp in c_times]
    c_far = [row for row in c_far if row[2] > 5.0]

    exact_selected = _take_with_forced(
        exact, 4, lambda row: row[0] == 1345.0,
        lambda row: _strength(d_records[row[0]]) + _strength(c_records[row[1]]),
    )
    close_selected = _take_with_forced(
        close, 4, lambda row: row[0] == 1094.0 or row[1] == 1094.0,
        lambda row: _strength(d_records[row[0]]) + _strength(c_records[row[1]]),
    )
    dino_only = _take_with_forced(
        d_far, 4, lambda row: row[0] == 369.0, lambda row: _strength(d_records[row[0]])
    )
    cradio_only = _take_with_forced(
        c_far, 4, lambda row: row[0] == 186.0, lambda row: _strength(c_records[row[0]])
    )
    used_d = {row[0] for row in dino_only}
    used_c = {row[0] for row in cradio_only}
    large_pool: list[tuple[str, float, float, float]] = []
    large_pool.extend(("dino", row[0], row[1], row[2]) for row in d_far if row[0] not in used_d)
    large_pool.extend(("cradio", row[0], row[1], row[2]) for row in c_far if row[0] not in used_c)
    large_selected: list[tuple[str, float, float, float]] = []
    for method in ("dino", "cradio"):
        pool = [row for row in large_pool if row[0] == method]
        records = d_records if method == "dino" else c_records
        large_selected.extend(sorted(pool, key=lambda row: (-_strength(records[row[1]]), row[1]))[:2])

    examples: list[dict[str, Any]] = []

    def add(category: str, d_time: float, c_time: float, distance: float, anchor: float) -> None:
        d_record = d_records[d_time]
        c_record = c_records[c_time]
        examples.append({
            "example_number": len(examples) + 1,
            "category": category,
            "anchor_sec": anchor,
            "dino_boundary_sec": d_time,
            "cradio_boundary_sec": c_time,
            "nearest_distance_sec": distance,
            "dino": d_record,
            "cradio": c_record,
        })

    for d_time, c_time, distance in exact_selected:
        add("strong shared — exact", d_time, c_time, distance, d_time)
    for d_time, c_time, distance in close_selected:
        add("shared — 1–2 s disagreement", d_time, c_time, distance, (d_time + c_time) / 2)
    for d_time, c_time, distance in dino_only:
        add("DINO-only strong (>5 s to C-RADIO)", d_time, c_time, distance, d_time)
    for c_time, d_time, distance in cradio_only:
        add("C-RADIO-only strong (>5 s to DINO)", d_time, c_time, distance, c_time)
    for method, primary, nearest, distance in large_selected:
        if method == "dino":
            add(">5 s disagreement — DINO anchor", primary, nearest, distance, primary)
        else:
            add(">5 s disagreement — C-RADIO anchor", nearest, primary, distance, primary)
    if len(examples) != 20:
        raise RuntimeError(f"Expected exactly 20 examples, got {len(examples)}")
    return examples


def _segments_in_window(segments: list[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
    return [row for row in segments if float(row["end"]) > start and float(row["start"]) < end]


def select_segment_windows(result: dict[str, Any]) -> list[dict[str, Any]]:
    d_segments = result["representations"]["dinov2"]["fine_segmentation"]["segments"]
    c_segments = result["representations"]["cradio"]["fine_segmentation"]["segments"]
    duration = float(result["input"]["duration_sec"])

    def around(center: float, width: float) -> tuple[float, float]:
        start = max(0.0, center - width / 2)
        end = min(duration, start + width)
        start = max(0.0, end - width)
        return start, end

    longest_d = max(d_segments, key=lambda row: (float(row["duration"]), -float(row["start"])))
    longest_c = max(c_segments, key=lambda row: (float(row["duration"]), -float(row["start"])))
    shortest_d = min(d_segments, key=lambda row: (float(row["duration"]), float(row["start"])))
    shortest_d_center = (float(shortest_d["start"]) + float(shortest_d["end"])) / 2
    shortest_c = next(
        row for row in sorted(c_segments, key=lambda item: (float(item["duration"]), float(item["start"])))
        if abs((float(row["start"]) + float(row["end"])) / 2 - shortest_d_center) >= 40.0
    )

    scan: list[tuple[float, int, int]] = []
    for start in range(0, int(duration) - 59, 20):
        end = start + 60.0
        scan.append((float(start), len(_segments_in_window(d_segments, start, end)), len(_segments_in_window(c_segments, start, end))))
    d_dense = max(scan, key=lambda row: (row[1] - row[2], row[1], -row[0]))
    c_dense = max(scan, key=lambda row: (row[2] - row[1], row[2], -row[0]))

    specs = [
        ("Longest DINO Fine segment context", *around((float(longest_d["start"]) + float(longest_d["end"])) / 2, 80.0)),
        ("Longest C-RADIO Fine segment context", *around((float(longest_c["start"]) + float(longest_c["end"])) / 2, 70.0)),
        ("Shortest DINO Fine segment context", *around((float(shortest_d["start"]) + float(shortest_d["end"])) / 2, 40.0)),
        ("Shortest C-RADIO Fine segment context", *around((float(shortest_c["start"]) + float(shortest_c["end"])) / 2, 40.0)),
        ("DINO splits more in this 60 s window", d_dense[0], d_dense[0] + 60.0),
        ("C-RADIO splits more in this 60 s window", c_dense[0], c_dense[0] + 60.0),
    ]
    windows = []
    for number, (label, start, end) in enumerate(specs, 1):
        windows.append({
            "window_number": number,
            "label": label,
            "start_sec": start,
            "end_sec": end,
            "dino_segments": _segments_in_window(d_segments, start, end),
            "cradio_segments": _segments_in_window(c_segments, start, end),
        })
    return windows


def _timestamp_filename(timestamp: float) -> str:
    return f"frame_{timestamp:08.3f}".replace(".", "p") + ".jpg"


def decode_visual_frames(
    video_path: Path, timestamps: Iterable[float], assets_dir: Path
) -> dict[float, str]:
    assets_dir.mkdir(parents=True, exist_ok=True)
    unique = sorted({round(float(value), 3) for value in timestamps})
    reader = decord.VideoReader(str(video_path), ctx=decord.cpu(0), num_threads=4)
    fps = float(reader.get_avg_fps())
    max_time = (len(reader) - 1) / fps
    result: dict[float, str] = {}
    for offset in range(0, len(unique), 32):
        batch_times = [min(max(value, 0.0), max_time) for value in unique[offset:offset + 32]]
        indices = [min(len(reader) - 1, int(math.floor(value * fps))) for value in batch_times]
        arrays = reader.get_batch(indices).asnumpy()
        for requested, actual, array in zip(unique[offset:offset + 32], batch_times, arrays):
            path = assets_dir / _timestamp_filename(requested)
            image = Image.fromarray(array, mode="RGB")
            image.thumbnail((320, 180), Image.Resampling.LANCZOS)
            image.save(path, "JPEG", quality=78, optimize=True)
            result[requested] = path.name
    return result


def _marker_html(timestamp: float, duration: float, css_class: str, title: str) -> str:
    left = 100.0 * timestamp / duration
    return f'<span class="marker {css_class}" style="left:{left:.5f}%" title="{html.escape(title)}"></span>'


def _timeline_html(result: dict[str, Any]) -> str:
    duration = float(result["input"]["duration_sec"])
    d_times = [float(row["timestamp"]) for row in result["representations"]["dinov2"]["boundary_records"]]
    c_times = [float(row["timestamp"]) for row in result["representations"]["cradio"]["boundary_records"]]
    matches = result["boundary_agreement"]["exact_frozen_threshold_outputs"]["2.0"]["matches"]
    d_match = {float(row["left"]): float(row["distance_sec"]) for row in matches}
    c_match = {float(row["right"]): float(row["distance_sec"]) for row in matches}

    def markers(times: list[float], match: dict[float, float], method: str) -> str:
        rows = []
        for timestamp in times:
            if timestamp in match and match[timestamp] == 0:
                category = "exact"
            elif timestamp in match:
                category = "near"
            else:
                category = "dino-only" if method == "DINOv2" else "cradio-only"
            rows.append(_marker_html(timestamp, duration, category, f"{method} {timestamp:.1f}s — {category}"))
        return "".join(rows)

    ticks = "".join(
        f'<span class="tick" style="left:{100*t/duration:.4f}%"><i></i><b>{int(t)}s</b></span>'
        for t in range(0, 1601, 200)
    )
    return f"""
    <div class="timeline-scale">{ticks}</div>
    <div class="timeline-row"><div class="track-label">DINOv2</div><div class="track">{markers(d_times, d_match, 'DINOv2')}</div></div>
    <div class="timeline-row"><div class="track-label">C-RADIO</div><div class="track">{markers(c_times, c_match, 'C-RADIO')}</div></div>
    """


def _metric_line(label: str, record: dict[str, Any]) -> str:
    return (
        f"<strong>{label}</strong> boundary {float(record['timestamp']):.1f}s · "
        f"strength {float(record['smoothed_change_strength']):.4f} · "
        f"raw cosine {float(record['raw_adjacent_cosine_similarity']):.4f} · "
        f"smoothed cosine {float(record['smoothed_adjacent_cosine_similarity']):.4f}"
    )


def _boundary_card(example: dict[str, Any], frames: dict[float, str], assets_rel: str) -> str:
    center = round(float(example["anchor_sec"]))
    times = [round(center + delta, 3) for delta in range(-3, 4)]
    strip = "".join(
        f'<figure><img src="{assets_rel}/{frames[t]}" alt="YKI08 at {t:.1f} seconds"><figcaption>{t:.1f}s</figcaption></figure>'
        for t in times
    )
    start, end = center - 3.0, center + 3.0
    rulers = []
    dino_time = float(example["dino_boundary_sec"])
    cradio_time = float(example["cradio_boundary_sec"])
    if dino_time == cradio_time and start <= dino_time <= end:
        left = 100 * (dino_time - start) / (end - start)
        rulers.append(
            f'<span class="audit-line shared-line" style="left:{left:.3f}%">'
            f'<b>DINO + C-RADIO {dino_time:.1f}s</b></span>'
        )
    else:
        for label, timestamp, css_class in (
            ("DINO", dino_time, "dino-line"),
            ("C-RADIO", cradio_time, "cradio-line"),
        ):
            if start <= timestamp <= end:
                left = 100 * (timestamp - start) / (end - start)
                rulers.append(f'<span class="audit-line {css_class}" style="left:{left:.3f}%"><b>{label} {timestamp:.1f}s</b></span>')
            else:
                direction = "←" if timestamp < start else "→"
                rulers.append(f'<span class="offstrip {css_class}">{label} nearest {timestamp:.1f}s {direction} outside strip</span>')
    return f"""
    <article class="boundary-card" id="example-{example['example_number']}">
      <header><span class="number">#{example['example_number']:02d}</span><h3>{html.escape(example['category'])}</h3><span class="distance">nearest distance {example['nearest_distance_sec']:.1f}s</span></header>
      <div class="strip">{strip}</div>
      <div class="mini-ruler">{''.join(rulers)}</div>
      <div class="metrics"><div>{_metric_line('DINOv2', example['dino'])}</div><div>{_metric_line('C-RADIO', example['cradio'])}</div></div>
      <div class="manual"><strong>Manual impression:</strong> □ DINO more reasonable &nbsp; □ C-RADIO more reasonable &nbsp; □ both reasonable &nbsp; □ neither / ambiguous</div>
    </article>
    """


def _segment_track(
    method: str, segments: list[dict[str, Any]], start: float, end: float,
    frames: dict[float, str], assets_rel: str,
) -> str:
    width = end - start
    cards = []
    for segment in segments:
        seg_start = float(segment["start"])
        seg_end = float(segment["end"])
        visible_start = max(start, seg_start)
        visible_end = min(end, seg_end)
        flex = max(visible_end - visible_start, 0.2)
        representative = round(float(segment["representative_frame_timestamp"]), 3)
        cards.append(f"""
        <div class="segment {method.lower()}" style="flex:{flex} 1 0">
          <img src="{assets_rel}/{frames[representative]}" alt="{method} segment representative at {representative:.1f}s">
          <div><b>{seg_start:.1f}–{seg_end:.1f}s</b><small>{float(segment['duration']):.1f}s · rep {representative:.1f}s</small></div>
        </div>""")
    return f'<div class="segment-row"><div class="track-label">{method}</div><div class="segments" data-window-width="{width:.1f}">{"".join(cards)}</div></div>'


def _segment_window(window: dict[str, Any], frames: dict[float, str], assets_rel: str) -> str:
    start, end = float(window["start_sec"]), float(window["end_sec"])
    return f"""
    <article class="segment-window">
      <header><h3>{html.escape(window['label'])}</h3><span>{start:.1f}–{end:.1f}s</span></header>
      {_segment_track('DINOv2', window['dino_segments'], start, end, frames, assets_rel)}
      {_segment_track('C-RADIO', window['cradio_segments'], start, end, frames, assets_rel)}
      <div class="manual"><strong>Manual impression:</strong> □ DINO events more coherent &nbsp; □ C-RADIO events more coherent &nbsp; □ comparable &nbsp; □ ambiguous</div>
    </article>
    """


CSS = r"""
:root { --ink:#172033; --muted:#637083; --paper:#fff; --line:#d7deea; --dino:#2474d2; --cradio:#e76f32; --shared:#7847c8; --near:#2e9b67; }
* { box-sizing:border-box; }
body { margin:0; background:#eef2f7; color:var(--ink); font:14px/1.45 Inter,Arial,sans-serif; }
main { max-width:1380px; margin:auto; background:var(--paper); padding:36px 42px 64px; }
h1 { margin:0 0 4px; font-size:30px; }
h2 { margin-top:40px; padding-bottom:7px; border-bottom:2px solid #263a5a; }
h3 { margin:0; font-size:16px; }
.subtitle,.note { color:var(--muted); }
.scope { border-left:4px solid #e2a018; background:#fff8e7; padding:10px 14px; }
.legend { display:flex; flex-wrap:wrap; gap:16px; margin:12px 0; font-size:13px; }
.swatch { width:12px; height:12px; display:inline-block; margin-right:5px; vertical-align:-1px; }
.exact-bg{background:var(--shared)} .near-bg{background:var(--near)} .dino-bg{background:var(--dino)} .cradio-bg{background:var(--cradio)}
.timeline-wrap { border:1px solid var(--line); border-radius:8px; padding:24px 16px 14px; overflow:hidden; }
.timeline-scale { position:relative; height:28px; margin-left:90px; }
.tick { position:absolute; top:0; transform:translateX(-50%); color:var(--muted); font-size:10px; }
.tick i { display:block; height:9px; border-left:1px solid #9aa5b6; margin:auto; }
.tick b { font-weight:500; }
.timeline-row { display:flex; align-items:center; margin:10px 0; }
.track-label { flex:0 0 86px; font-weight:700; }
.track { position:relative; height:42px; flex:1; background:linear-gradient(#f8fafc,#edf1f6); border:1px solid var(--line); }
.marker { position:absolute; top:0; height:100%; width:2px; opacity:.88; }
.marker.exact{background:var(--shared);width:3px}.marker.near{background:var(--near)}.marker.dino-only{background:var(--dino)}.marker.cradio-only{background:var(--cradio)}
.boundary-card,.segment-window { break-inside:avoid; page-break-inside:avoid; border:1px solid var(--line); border-radius:9px; margin:18px 0; padding:14px; box-shadow:0 2px 7px #15284a12; }
.boundary-card header,.segment-window header { display:flex; align-items:center; gap:12px; margin-bottom:10px; }
.number { border-radius:16px; padding:3px 9px; background:#223655; color:white; font-weight:700; }
.distance,.segment-window header span { margin-left:auto; color:var(--muted); }
.strip { display:grid; grid-template-columns:repeat(7,1fr); gap:5px; }
figure { margin:0; min-width:0; background:#111; }
figure img { width:100%; display:block; aspect-ratio:16/9; object-fit:cover; }
figcaption { color:#fff; text-align:center; font-size:11px; padding:2px; }
.mini-ruler { position:relative; height:42px; margin-top:5px; border-top:1px solid #aeb8c7; }
.audit-line { position:absolute; top:0; height:24px; border-left:3px solid; }
.audit-line b { position:absolute; white-space:nowrap; top:12px; transform:translateX(-50%); font-size:10px; background:#fff; }
.dino-line { border-color:var(--dino); color:var(--dino); }.dino-line b { top:8px; }
.cradio-line { border-color:var(--cradio); color:var(--cradio); }.cradio-line b { top:23px; }
.shared-line { border-color:var(--shared); color:var(--shared); }.shared-line b { top:12px; }
.offstrip { display:inline-block; margin:7px 16px 0 0; font-size:11px; border-left:0; }
.metrics { display:grid; grid-template-columns:1fr 1fr; gap:10px; background:#f5f7fa; padding:8px; font-size:12px; }
.manual { margin-top:9px; padding:8px 10px; background:#fffdf3; border:1px dashed #bba964; }
.segment-row { display:flex; align-items:stretch; margin:10px 0; }
.segments { display:flex; flex:1; gap:3px; min-height:110px; }
.segment { min-width:85px; border:2px solid; background:#f7f9fc; overflow:hidden; }
.segment.dinov2 { border-color:var(--dino); }.segment.c-radio { border-color:var(--cradio); }
.segment img { width:100%; height:72px; object-fit:cover; display:block; }
.segment div { padding:4px; font-size:10px; }.segment small { display:block; color:var(--muted); }
.stats { border-collapse:collapse; width:100%; margin:12px 0; }
.stats th,.stats td { border:1px solid var(--line); padding:7px 9px; text-align:right; }
.stats th:first-child,.stats td:first-child { text-align:left; }
.callout { padding:12px 14px; background:#eef7ff; border-left:4px solid var(--dino); font-weight:600; }
.existing-list { columns:2; }
footer { margin-top:36px; color:var(--muted); font-size:12px; }
@media print {
  @page { size:A4 landscape; margin:9mm; }
  body { background:white; font-size:11px; }
  main { max-width:none; padding:0; }
  h1 { font-size:24px; } h2 { margin-top:22px; }
  .boundary-card,.segment-window { box-shadow:none; margin:10px 0; }
  .strip { gap:3px; }
  .metrics { font-size:9px; }
  .segments { min-height:88px; }.segment img { height:54px; }
}
"""


def build_html(
    result: dict[str, Any], examples: list[dict[str, Any]], windows: list[dict[str, Any]],
    frames: dict[float, str], output_html: Path, assets_dir: Path,
) -> None:
    output_html.parent.mkdir(parents=True, exist_ok=True)
    assets_rel = assets_dir.relative_to(output_html.parent).as_posix()
    agreement = result["boundary_agreement"]["exact_frozen_threshold_outputs"]
    dino = result["representations"]["dinov2"]
    cradio = result["representations"]["cradio"]
    d_dist = dino["adjacent_similarity"]["raw_distribution"]
    c_dist = cradio["adjacent_similarity"]["raw_distribution"]
    compatibility = result["absolute_frozen_threshold_compatibility"]
    boundary_cards = "".join(_boundary_card(row, frames, assets_rel) for row in examples)
    segment_cards = "".join(_segment_window(row, frames, assets_rel) for row in windows)
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>C-RADIOv4 vs DINOv2 segmentation visual audit</title><style>{CSS}</style></head>
<body><main>
<h1>C-RADIOv4 vs DINOv2 segmentation visual audit</h1>
<p class="subtitle">YKI08 · 1611.584 seconds · existing deterministic 1 FPS diagnostic replay · generated without model or segmentation recomputation</p>
<p class="scope"><strong>Manual inspection only.</strong> This report does not decide which representation is better. Judge whether boundaries correspond to real scene/action/event changes, whether C-RADIO ignores useful fine changes, whether DINO reacts to camera/photometric motion, and which segmentation yields coherent Medium-level events.</p>

<h2>A. Global timeline comparison</h2>
<div class="legend"><span><i class="swatch exact-bg"></i>exact/shared</span><span><i class="swatch near-bg"></i>matched within ±2 s</span><span><i class="swatch dino-bg"></i>DINO-only</span><span><i class="swatch cradio-bg"></i>C-RADIO-only</span></div>
<div class="timeline-wrap">{_timeline_html(result)}</div>
<p class="note">Markers are all existing Fine boundaries. “Only” means unmatched by the deterministic one-to-one ±2 s comparison; it is not a correctness label.</p>

<h2>B. Side-by-side boundary examples</h2>
<p>Twenty deterministic examples: four exact strong shared, four shared with 1–2 s displacement, four strong DINO-only, four strong C-RADIO-only, and four additional >5 s disagreement anchors. Seven source frames around each anchor provide ±3 s context.</p>
{boundary_cards}

<h2>C. Segment-level comparison</h2>
<p>Six representative windows cover both models' longest and shortest Fine segments and windows where either method splits more often. Each box reports the full segment interval/duration and displays its existing representative timestamp frame. Boxes are clipped visually to the selected window but labels retain full boundaries.</p>
{segment_cards}

<h2>D. Existing strongest examples</h2>
<ul class="existing-list">
<li><a href="#example-{next(row['example_number'] for row in examples if row['dino_boundary_sec']==1094.0)}">Strongest DINO boundary near 1094 s</a></li>
<li><a href="#example-{next(row['example_number'] for row in examples if row['cradio_boundary_sec']==1345.0)}">Strongest C-RADIO boundary near 1345 s</a></li>
<li><a href="#example-{next(row['example_number'] for row in examples if row['dino_boundary_sec']==369.0)}">Strong DINO-only boundary near 369 s</a></li>
<li><a href="#example-{next(row['example_number'] for row in examples if row['cradio_boundary_sec']==186.0)}">Strong C-RADIO-only boundary near 186 s</a></li>
</ul>
<p>Each linked card includes seven newly decoded context frames. Existing three-frame contact sheets remain preserved in the diagnostic output directory but are not needed to render this report.</p>

<h2>E. Summary statistics</h2>
<table class="stats"><thead><tr><th>Metric</th><th>DINOv2</th><th>C-RADIOv4</th></tr></thead><tbody>
<tr><td>Fine segments</td><td>{dino['fine_metrics']['number_of_segments']}</td><td>{cradio['fine_metrics']['number_of_segments']}</td></tr>
<tr><td>Medium segments</td><td>{dino['medium_duration_summary']['count']}</td><td>{cradio['medium_duration_summary']['count']}</td></tr>
<tr><td>Adjacent cosine mean</td><td>{d_dist['mean']:.6f}</td><td>{c_dist['mean']:.6f}</td></tr>
<tr><td>Adjacent cosine std</td><td>{d_dist['std']:.6f}</td><td>{c_dist['std']:.6f}</td></tr>
<tr><td>Adjacent cosine MAD</td><td>{d_dist['mad']:.6f}</td><td>{c_dist['mad']:.6f}</td></tr>
<tr><td>Smoothed MAD</td><td>{compatibility['dinov2_smoothed_mad']:.6f}</td><td>{compatibility['cradio_smoothed_mad']:.6f}</td></tr>
</tbody></table>
<table class="stats"><thead><tr><th>Boundary tolerance</th><th>Matches</th><th>F1</th></tr></thead><tbody>
<tr><td>Exact</td><td>{agreement['0.0']['matched_count']}</td><td>{agreement['0.0']['f1']:.3f}</td></tr>
<tr><td>±1 s</td><td>{agreement['1.0']['matched_count']}</td><td>{agreement['1.0']['f1']:.3f}</td></tr>
<tr><td>±2 s</td><td>{agreement['2.0']['matched_count']}</td><td>{agreement['2.0']['f1']:.3f}</td></tr>
<tr><td>±5 s</td><td>{agreement['5.0']['matched_count']}</td><td>{agreement['5.0']['f1']:.3f}</td></tr>
</tbody></table>
<p class="callout">The 8.49× MAD difference indicates different similarity-signal scales, not an 8.49× performance difference.</p>

<footer>Visualization-only audit. Source: <code>outputs/diagnostics/cradio_v4/dino_comparison/dino_comparison_results.json</code>. No embeddings, thresholds, segments, B0/B1/B2 definitions, or frozen artifacts were changed.</footer>
</main></body></html>"""
    output_html.write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    examples = select_boundary_examples(result)
    windows = select_segment_windows(result)
    assets_dir = args.html.parent / "assets/cradio_dino_segmentation_audit"

    requested: list[float] = []
    duration = float(result["input"]["duration_sec"])
    for example in examples:
        center = round(float(example["anchor_sec"]))
        requested.extend(min(max(center + delta, 0.0), duration - 0.001) for delta in range(-3, 4))
    for window in windows:
        for method in ("dino_segments", "cradio_segments"):
            requested.extend(float(row["representative_frame_timestamp"]) for row in window[method])
    frames = decode_visual_frames(args.video, requested, assets_dir)
    build_html(result, examples, windows, frames, args.html, assets_dir)

    manifest = {
        "schema_version": "cradio-dino-segmentation-visual-audit-v1",
        "visualization_only": True,
        "models_rerun": False,
        "embeddings_recomputed": False,
        "segmentation_recomputed": False,
        "source_result": str(args.result),
        "source_video": str(args.video),
        "boundary_example_count": len(examples),
        "segment_window_count": len(windows),
        "decoded_unique_frame_count": len(frames),
        "boundary_examples": examples,
        "segment_windows": windows,
    }
    (assets_dir / "visual_audit_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"HTML: {args.html}")
    print(f"Boundary examples: {len(examples)}")
    print(f"Segment windows: {len(windows)}")
    print(f"Unique decoded frames: {len(frames)}")


if __name__ == "__main__":
    main()
