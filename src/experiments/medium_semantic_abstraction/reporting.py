"""Human-readable Medium/Coarse semantic abstraction comparison."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _timing(timing: dict[str, Any]) -> str:
    return (
        f'decode {timing["image_decode_sec"]:.3f}s · preprocess {timing["image_preprocess_sec"]:.3f}s · '
        f'inference {timing["model_inference_sec"]:.3f}s · total {timing["total_semantic_generation_sec"]:.3f}s'
    )


def _medium_review(prefix: str) -> str:
    radios = lambda name, values: " ".join(  # noqa: E731
        f'<label><input type="radio" name="{_e(prefix)}_{name}" value="{_e(value)}"> {_e(value)}</label>'
        for value in values
    )
    return f'''<div class="review">
<div><b>1-frame summary correct?</b> {radios("one_correct", ["yes", "partial", "no"])}</div>
<div><b>3-frame summary correct?</b> {radios("three_correct", ["yes", "partial", "no"])}</div>
<div><b>Better:</b> {radios("better", ["1-frame", "3-frame", "similar"])}</div>
<label><b>Notes:</b><textarea name="{_e(prefix)}_notes"></textarea></label></div>'''


def _coarse_review(prefix: str) -> str:
    radios = lambda name, values: " ".join(  # noqa: E731
        f'<label><input type="radio" name="{_e(prefix)}_{name}" value="{_e(value)}"> {_e(value)}</label>'
        for value in values
    )
    return f'''<div class="review">
<div><b>1-frame-derived Coarse summary correct?</b> {radios("one_correct", ["yes", "partial", "no"])}</div>
<div><b>3-frame-derived Coarse summary correct?</b> {radios("three_correct", ["yes", "partial", "no"])}</div>
<div><b>Better:</b> {radios("better", ["1-frame", "3-frame", "similar"])}</div>
<label><b>Notes:</b><textarea name="{_e(prefix)}_notes"></textarea></label></div>'''


def _condition(record: dict[str, Any], condition: str) -> str:
    data = record["conditions"][condition]
    images = "".join(
        f'<figure><img src="{_e(item["html_relative_path"])}" loading="lazy">'
        f'<figcaption>{item["timestamp"]:.1f}s · {_e(item["selection_rule"])}</figcaption></figure>'
        for item in data["selected_images"]
    )
    return f'''<div class="condition {condition}"><h5>{"1-FRAME" if condition == "one_frame" else "3-FRAME"} CONDITION</h5>
<div class="frames">{images}</div><p><b>Raw Qwen output:</b> {_e(data["description"])}</p>
<p class="timing">{_timing(data["timing"])}</p></div>'''


def render_comparison(
    *, output_path: Path, medium_results: list[dict[str, Any]],
    coarse_results: list[dict[str, Any]], aggregate: dict[str, Any], model: dict[str, Any],
) -> dict[str, Any]:
    medium_by_id = {(row["video_id"], row["medium_id"]): row for row in medium_results}
    coarse_by_id = {(row["video_id"], row["coarse_id"]): row for row in coarse_results}
    video_ids = sorted({row["video_id"] for row in medium_results})
    sections = []
    image_refs = 0
    missing = []
    for video_id in video_ids:
        video_medium = sorted(
            [row for row in medium_results if row["video_id"] == video_id],
            key=lambda row: (row["start"], row["medium_id"]),
        )
        video_coarse = sorted(
            [row for row in coarse_results if row["video_id"] == video_id],
            key=lambda row: (row["start"], row["coarse_id"]),
        )
        coarse_cards = []
        for coarse in video_coarse:
            medium_cards = []
            for medium_id in coarse["medium_ids"]:
                row = medium_by_id[(video_id, medium_id)]
                for condition in ("one_frame", "three_frame"):
                    for item in row["conditions"][condition]["selected_images"]:
                        image_refs += 1
                        path = output_path.parent / item["html_relative_path"]
                        if not path.is_file():
                            missing.append(path.as_posix())
                medium_cards.append(
                    f'<article class="medium"><h4>MEDIUM {_e(medium_id)} '
                    f'[{row["start"]:.1f}–{row["end"]:.1f}s] · {row["duration"]:.1f}s</h4>'
                    f'<p>Fine children: {_e(", ".join(row["fine_ids"]))}</p>'
                    f'<div class="conditions">{_condition(row, "one_frame")}{_condition(row, "three_frame")}</div>'
                    f'{_medium_review(video_id + "_" + medium_id)}</article>'
                )
            coarse_result = coarse_by_id[(video_id, coarse["coarse_id"])]
            one_lines = "".join(
                f'<li><b>{_e(mid)}</b> → {_e(medium_by_id[(video_id, mid)]["conditions"]["one_frame"]["description"])}</li>'
                for mid in coarse["medium_ids"]
            )
            three_lines = "".join(
                f'<li><b>{_e(mid)}</b> → {_e(medium_by_id[(video_id, mid)]["conditions"]["three_frame"]["description"])}</li>'
                for mid in coarse["medium_ids"]
            )
            coarse_cards.append(
                f'<details class="coarse" open><summary>COARSE {_e(coarse["coarse_id"])} '
                f'[{coarse["start"]:.1f}–{coarse["end"]:.1f}s] · {coarse["duration"]:.1f}s</summary>'
                f'<div class="medium-stack">{"".join(medium_cards)}</div>'
                f'<div class="coarse-compare"><div><h4>Using 1-frame Medium summaries</h4><ol>{one_lines}</ol>'
                f'<p><b>Coarse summary:</b> {_e(coarse_result["conditions"]["one_frame"]["summary"])}</p>'
                f'<p class="timing">{_timing(coarse_result["conditions"]["one_frame"]["timing"])}</p></div>'
                f'<div><h4>Using 3-frame Medium summaries</h4><ol>{three_lines}</ol>'
                f'<p><b>Coarse summary:</b> {_e(coarse_result["conditions"]["three_frame"]["summary"])}</p>'
                f'<p class="timing">{_timing(coarse_result["conditions"]["three_frame"]["timing"])}</p></div></div>'
                f'{_coarse_review(video_id + "_" + coarse["coarse_id"])}</details>'
            )
        sections.append(
            f'<section id="video-{_e(video_id)}"><h2>VIDEO {_e(video_id)}</h2>'
            f'<p>{len(video_medium)} Medium nodes · {len(video_coarse)} Coarse nodes</p>{"".join(coarse_cards)}</section>'
        )
    if missing:
        raise RuntimeError(f"Missing semantic review images: {missing[:5]}")
    links = " ".join(f'<a href="#video-{_e(video_id)}">{_e(video_id[:8])}</a>' for video_id in video_ids)
    one = aggregate["conditions"]["one_frame"]
    three = aggregate["conditions"]["three_frame"]
    document = f'''<!doctype html><html><head><meta charset="utf-8"><title>Medium semantic abstraction v0.1</title>
<style>
body{{font:14px/1.45 system-ui;margin:0;background:#f2f4f7;color:#17212d}}header{{background:#17283b;color:white;padding:22px;position:sticky;top:0;z-index:3}}nav a{{color:#9edcff;margin-right:12px}}main{{max-width:1500px;margin:auto;padding:18px}}section{{background:white;border:1px solid #ccd5df;border-radius:12px;margin:20px 0;padding:18px}}.coarse{{border:3px solid #5c6bc0;border-radius:10px;padding:12px;margin:14px 0}}summary{{font-size:18px;font-weight:800;cursor:pointer}}.medium{{border:2px solid #26a69a;border-radius:9px;padding:12px;margin:12px 0 12px 24px}}.conditions,.coarse-compare{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}.condition,.coarse-compare>div{{background:#f8fafc;border:1px solid #ccd5df;border-radius:8px;padding:10px}}.three_frame{{background:#f3fbf7}}.frames{{display:flex;gap:8px;overflow:auto}}figure{{margin:0}}img{{width:210px;height:145px;object-fit:cover;border-radius:6px;border:1px solid #999}}figcaption{{font-size:12px}}.timing{{font-family:ui-monospace,monospace;color:#526170}}.review{{background:#fff8e5;border:1px dashed #c59b32;padding:10px;margin:10px 0}}textarea{{display:block;width:98%;height:42px}}.metrics{{display:flex;gap:24px;flex-wrap:wrap}}.metric{{background:#253b52;padding:9px;border-radius:6px}}@media(max-width:900px){{.conditions,.coarse-compare{{grid-template-columns:1fr}}}}
</style></head><body><header><h1>Medium Semantic Abstraction: 1 frame vs 3 frames</h1>
<div class="metrics"><div class="metric">Model: {_e(model["model"])}</div><div class="metric">Medium nodes: {aggregate["medium_node_count"]}</div>
<div class="metric">Images: {one["image_count"]} vs {three["image_count"]}</div><div class="metric">Mean Medium total: {one["mean_medium_total_sec"]:.2f}s vs {three["mean_medium_total_sec"]:.2f}s</div></div>
<p>Fine nodes were not captioned. Coarse summaries consume ordered Medium text only. Human judgment fields are intentionally blank.</p><nav>{links}</nav></header>
<main>{''.join(sections)}</main></body></html>'''
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    return {"video_count": len(video_ids), "medium_count": len(medium_results), "coarse_count": len(coarse_results), "image_reference_count": image_refs, "missing_image_count": 0}
