"""Human-first semantic chain report: keyframe -> Medium -> Coarse -> story."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _time(value: float) -> str:
    minutes, seconds = divmod(float(value), 60.0)
    return f"{int(minutes):02d}:{seconds:04.1f}"


def _review(prefix: str, fields: list[tuple[str, list[str]]]) -> str:
    blocks = []
    for label, options in fields:
        inputs = " ".join(
            f'<label><input type="radio" name="{_e(prefix)}_{index}" value="{_e(option)}"> {_e(option)}</label>'
            for option in options for index in [fields.index((label, options))]
        )
        blocks.append(f'<div><strong>{_e(label)}:</strong> {inputs}</div>')
    blocks.append(f'<label><strong>Notes:</strong><textarea name="{_e(prefix)}_notes"></textarea></label>')
    return '<div class="review">' + "".join(blocks) + "</div>"


def _image(relative: str, timestamp: float, label: str, *, output_dir: Path, css: str = "") -> str:
    path = output_dir / relative
    if not path.is_file():
        raise FileNotFoundError(f"Missing semantic-map image: {path}")
    return (
        f'<figure class="{_e(css)}"><img src="{_e(relative)}" loading="lazy" alt="{_e(label)}">'
        f'<figcaption>{_time(timestamp)} · {_e(label)}</figcaption></figure>'
    )


def render_semantic_map(
    *, output_path: Path, source_manifest: dict[str, Any], hierarchies: list[dict[str, Any]],
    keyframes: list[dict[str, Any]], semantic_groups: list[dict[str, Any]],
    medium_captions: list[dict[str, Any]], coarse_captions: list[dict[str, Any]],
    stories: list[dict[str, Any]], efficiency: list[dict[str, Any]],
    runtimes: list[dict[str, Any]],
) -> dict[str, Any]:
    output_dir = output_path.parent
    manifest_by_video = {row["video_id"]: row for row in source_manifest["videos"]}
    keyframe_by_id = {(row["video_id"], row["medium_id"]): row for row in keyframes}
    medium_by_id = {(row["video_id"], row["medium_id"]): row for row in medium_captions}
    efficiency_by_video = {row["video_id"]: row for row in efficiency}
    runtime_by_video = {row["video_id"]: row for row in runtimes}
    story_by_video = {row["video_id"]: row for row in stories}
    group_by_video: dict[str, list[dict[str, Any]]] = {}
    for group in semantic_groups:
        group_by_video.setdefault(group["video_id"], []).append(group)
    coarse_by_video: dict[str, list[dict[str, Any]]] = {}
    for coarse in coarse_captions:
        coarse_by_video.setdefault(coarse["video_id"], []).append(coarse)
    for values in coarse_by_video.values():
        values.sort(key=lambda row: (row["start"], row["coarse_id"]))

    summary_rows: list[str] = []
    sections: list[str] = []
    image_references = 0
    selected_image_references = 0
    candidate_image_references = 0
    group_image_references = 0
    for video_index, hierarchy in enumerate(hierarchies, start=1):
        video_id = str(hierarchy["video_id"])
        source = manifest_by_video[video_id]
        stats = efficiency_by_video[video_id]
        runtime = runtime_by_video[video_id]
        story = story_by_video[video_id]
        summary_rows.append(
            f'<tr><td><a href="#video-{video_index}">{_e(source["sample_id"])}</a></td>'
            f'<td>{_time(stats["duration_sec"])}</td><td>{stats["medium_count"]}</td>'
            f'<td>{stats["candidate_frame_count"]}</td><td>{stats["semantic_group_count"]}</td>'
            f'<td>{stats["actual_direct_vlm_image_calls"]}</td><td>{stats["vlm_calls_avoided"]}</td>'
            f'<td>{stats["vlm_call_reduction_ratio"]:.1%}</td><td>{runtime["steady_state_semantic_indexing_sec"]:.1f}s</td></tr>'
        )

        timeline = []
        coarse_cards = []
        for coarse_index, coarse in enumerate(coarse_by_video[video_id], start=1):
            timeline.append(
                f'<article class="timeline-item"><div class="badge">C{coarse_index}</div>'
                f'<div><b>{_time(coarse["start"])}–{_time(coarse["end"])}</b>'
                f'<p>{_e(coarse["cleaned_caption"])}</p></div></article>'
            )
            medium_cards = []
            for medium_id in coarse["medium_ids"]:
                keyframe = keyframe_by_id[(video_id, medium_id)]
                caption = medium_by_id[(video_id, medium_id)]
                selected = keyframe["selected_keyframe"]
                selected_html = _image(
                    selected["html_relative_path"], selected["timestamp"], "selected keyframe",
                    output_dir=output_dir, css="selected-frame",
                )
                image_references += 1
                selected_image_references += 1
                candidate_cards = []
                for candidate in keyframe["candidates"]:
                    candidate_cards.append(
                        '<article class="candidate">'
                        + _image(
                            candidate["html_relative_path"], candidate["timestamp"], candidate["kind"],
                            output_dir=output_dir,
                        )
                        + '<div class="score-grid">'
                        + "".join(
                            f'<span>{_e(name)}<b>{float(value):.3f}</b></span>'
                            for name, value in candidate["normalized_score_components"].items()
                        )
                        + f'<span>Total<b>{candidate["selection_score"]:.3f}</b></span></div></article>'
                    )
                    image_references += 1
                    candidate_image_references += 1
                source_class = "direct" if caption["caption_source"] == "direct_vlm" else "propagated"
                medium_cards.append(
                    f'<article class="medium"><h4>MEDIUM {_e(medium_id)} '
                    f'[{_time(keyframe["start"])}–{_time(keyframe["end"])}]</h4>'
                    f'<div class="medium-main">{selected_html}<div class="caption-panel">'
                    f'<div class="source {source_class}">{_e(caption["caption_source"])}</div>'
                    f'<h5>VLM caption</h5><p class="caption">{_e(caption["cleaned_caption"])}</p>'
                    f'<dl><dt>Selected timestamp</dt><dd>{_time(selected["timestamp"])}</dd>'
                    f'<dt>Semantic group</dt><dd>{_e(caption["semantic_group_id"])}</dd>'
                    f'<dt>Caption source Medium</dt><dd>{_e(caption["caption_source_medium_id"])}</dd>'
                    f'<dt>Similarity to representative</dt><dd>{caption["similarity_to_group_representative"]:.6f}</dd>'
                    f'<dt>Direct inference time</dt><dd>{caption["inference_timing"]["total_semantic_generation_sec"]:.3f}s</dd></dl>'
                    f'</div></div><details><summary>All {keyframe["candidate_count"]} keyframe candidates, scores, and rule</summary>'
                    f'<p>{_e(keyframe["selection_reason"])}</p><div class="candidate-grid">{"".join(candidate_cards)}</div></details>'
                    f'<details><summary>Raw local-Qwen output</summary><pre>{_e(caption["raw_vlm_output"])}</pre></details>'
                    + _review(
                        video_id + "_" + medium_id,
                        [
                            ("Keyframe representative?", ["yes", "no", "unsure"]),
                            ("Image informative?", ["yes", "no", "unsure"]),
                            ("Caption correct?", ["yes", "partial", "no"]),
                            ("Important information missing?", ["yes", "no", "unsure"]),
                        ],
                    )
                    + '</article>'
                )
            coarse_cards.append(
                f'<details class="coarse" open><summary>COARSE {_e(coarse["coarse_id"])} '
                f'[{_time(coarse["start"])}–{_time(coarse["end"])}]</summary>'
                f'<div class="coarse-caption"><b>COARSE CAPTION</b><p>{_e(coarse["cleaned_caption"])}</p>'
                f'<small>Text-only inference: {coarse["inference_timing"]["total_semantic_generation_sec"]:.3f}s</small></div>'
                f'<details><summary>Ordered Medium caption input and raw output</summary><ol>'
                + "".join(f'<li>{_e(value)}</li>' for value in coarse["ordered_medium_caption_input"])
                + f'</ol><pre>{_e(coarse["raw_model_output"])}</pre></details>'
                + _review(
                    video_id + "_" + coarse["coarse_id"],
                    [
                        ("Caption correct?", ["yes", "partial", "no"]),
                        ("Coherent storyline?", ["yes", "no", "unsure"]),
                        ("Too broad?", ["yes", "no", "unsure"]),
                        ("Too fragmented?", ["yes", "no", "unsure"]),
                    ],
                )
                + f'<div class="medium-stack">{"".join(medium_cards)}</div></details>'
            )

        group_cards = []
        for group in sorted(group_by_video.get(video_id, []), key=lambda row: row["semantic_group_id"]):
            representative = group["representative_keyframe"]
            representative_medium = medium_by_id[(video_id, group["representative_medium_id"])]
            members = []
            for member in group["members"]:
                frame = member["selected_keyframe"]
                members.append(
                    '<article class="group-member">'
                    + _image(
                        frame["html_relative_path"], frame["timestamp"], member["medium_id"],
                        output_dir=output_dir,
                    )
                    + f'<p>similarity: {member["similarity_to_representative"]:.6f}</p></article>'
                )
                image_references += 1
                group_image_references += 1
            group_cards.append(
                f'<details class="semantic-group"><summary>{_e(group["semantic_group_id"])} · '
                f'{group["member_count"]} member(s) · representative {_e(group["representative_medium_id"])}</summary>'
                f'<p><b>Representative caption:</b> {_e(representative_medium["cleaned_caption"])}</p>'
                f'<p>Complete-link minimum similarity: {group["minimum_pairwise_similarity"]:.6f}</p>'
                f'<div class="group-grid">{"".join(members)}</div></details>'
            )

        storyline_html = (
            '<ol>' + "".join(f'<li>{_e(item)}</li>' for item in story["high_level_storyline"]) + '</ol>'
            if story["high_level_storyline"] else '<p class="muted">Model did not emit a separately parseable numbered list; inspect raw output below.</p>'
        )
        sections.append(
            f'<section id="video-{video_index}"><h2>{_e(source["sample_id"])}</h2>'
            f'<p class="video-meta">Duration {_time(stats["duration_sec"])} · Fine {stats["fine_count"]} · '
            f'Medium {stats["medium_count"]} · Coarse {stats["coarse_count"]}</p>'
            f'<div class="stat-row"><b>Semantic groups {stats["semantic_group_count"]}</b>'
            f'<b>Actual image calls {stats["actual_direct_vlm_image_calls"]}</b>'
            f'<b>Avoided calls {stats["vlm_calls_avoided"]}</b><b>Reduction {stats["vlm_call_reduction_ratio"]:.1%}</b>'
            f'<b>Selected images {stats["selected_keyframe_count"]}</b></div>'
            f'<article class="story"><h3>Overall Story</h3><p>{_e(story["overall_story"])}</p>'
            f'<h3>High-level storyline</h3><p><small>source: {_e(story.get("high_level_storyline_source", "parsed_model_output"))}</small></p>{storyline_html}'
            f'<details><summary>Exact ordered Coarse-text input and raw output</summary><ol>'
            + "".join(f'<li>{_e(value)}</li>' for value in story["ordered_coarse_caption_input"])
            + f'</ol><pre>{_e(story["raw_model_output"])}</pre></details>'
            + _review(
                video_id + "_story",
                [
                    ("Overall story understandable?", ["yes", "partial", "no"]),
                    ("Major events preserved?", ["yes", "no", "unsure"]),
                    ("Repetitive?", ["yes", "no", "unsure"]),
                    ("Missing important transition?", ["yes", "no", "unsure"]),
                ],
            )
            + f'</article><h3>High-level timeline</h3><div class="storyline">{"".join(timeline)}</div>'
            f'<h3>Coarse → Medium semantic chain</h3>{"".join(coarse_cards)}'
            f'<h3>Semantic group audit</h3><p>Groups never change structural membership. Propagation is explicitly marked above.</p>'
            f'{"".join(group_cards)}</section>'
        )

    navigation = " ".join(
        f'<a href="#video-{index}">{index}. {_e(manifest_by_video[h["video_id"]]["sample_id"])}</a>'
        for index, h in enumerate(hierarchies, start=1)
    )
    document = f'''<!doctype html><html><head><meta charset="utf-8"><title>Efficient semantic map v0.1</title>
<style>
*{{box-sizing:border-box}} body{{margin:0;background:#eef2f5;color:#172331;font:14px/1.48 system-ui,sans-serif}} header{{background:#122638;color:white;padding:20px;position:sticky;top:0;z-index:10}} header h1{{margin:0 0 5px}} nav{{overflow-x:auto;white-space:nowrap}} nav a{{color:#a6dfff;margin-right:16px}} main{{max-width:1500px;margin:auto;padding:18px}} .summary,section{{background:white;border:1px solid #cbd4dd;border-radius:13px;padding:18px;margin-bottom:22px}} table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{border:1px solid #ccd5dd;padding:7px;text-align:left}} th{{background:#e6edf3}} .video-meta{{font-size:16px}} .stat-row{{display:flex;flex-wrap:wrap;gap:8px}} .stat-row b{{background:#e4edf7;border-radius:18px;padding:7px 11px}} .story{{background:linear-gradient(135deg,#102e49,#174d63);color:white;padding:18px;border-radius:12px;margin:15px 0}} .story details,.story .review{{background:#f5f8fb;color:#172331}} .storyline{{border-left:4px solid #367c98;margin-left:15px}} .timeline-item{{display:grid;grid-template-columns:50px 1fr;gap:10px;padding:8px 10px}} .timeline-item .badge{{background:#367c98;color:white;border-radius:50%;height:38px;width:38px;text-align:center;padding-top:8px}} .timeline-item p{{margin:3px 0}} details summary{{cursor:pointer;font-weight:750}} .coarse{{border:3px solid #6671d7;border-radius:10px;padding:12px;margin:16px 0}} .coarse-caption{{background:#eef0ff;border-left:5px solid #6671d7;padding:10px;margin:8px 0}} .medium{{border:2px solid #2e9f91;border-radius:9px;padding:10px;margin:12px 0 12px 24px}} .medium-main{{display:grid;grid-template-columns:minmax(220px,360px) 1fr;gap:16px}} .caption-panel{{padding:8px}} .caption{{font-size:18px}} dl{{display:grid;grid-template-columns:max-content 1fr;gap:3px 10px}} dt{{font-weight:700}} dd{{margin:0}} .source{{display:inline-block;padding:5px 9px;border-radius:15px;font-weight:750}} .direct{{background:#d9f3e7;color:#176540}} .propagated{{background:#fff0c7;color:#785300}} figure{{margin:0}} img{{width:180px;height:120px;object-fit:cover;border:1px solid #abb7c2;border-radius:6px}} .selected-frame img{{width:340px;height:220px;border:4px solid #2e9f91}} figcaption{{font-size:11px;color:#536477}} .candidate-grid,.group-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(195px,1fr));gap:9px}} .candidate,.group-member{{background:#f6f8fa;border:1px solid #d4dbe1;border-radius:7px;padding:7px}} .score-grid{{display:grid;grid-template-columns:repeat(2,1fr);font-size:11px}} .score-grid span{{display:flex;justify-content:space-between;gap:3px}} .semantic-group{{border:1px solid #a984c5;background:#faf6fd;border-radius:8px;padding:9px;margin:8px}} .review{{background:#f5f7f9;border:1px dashed #9ba7b2;padding:8px;margin:8px 0}} textarea{{display:block;width:100%;height:42px}} pre{{white-space:pre-wrap;background:#172331;color:#dbe8f2;padding:9px;border-radius:6px}} .muted{{color:#6a7683}} @media(max-width:750px){{.medium-main{{grid-template-columns:1fr}} .selected-frame img{{width:100%;height:auto}}}}
</style></head><body><header><h1>Efficient semantic map v0.1</h1><p>Selected keyframe → Medium caption → Coarse caption → whole-video story</p><nav>{navigation}</nav></header><main><div class="summary"><h2>Efficiency summary</h2><p>One local Qwen2-VL image call per conservative visual group; every propagated caption remains explicit. Coarse and story stages are text-only.</p><div style="overflow:auto"><table><thead><tr><th>Video</th><th>Duration</th><th>Medium</th><th>Candidates</th><th>Groups</th><th>Image calls</th><th>Avoided</th><th>Reduction</th><th>Steady semantic time</th></tr></thead><tbody>{''.join(summary_rows)}</tbody></table></div></div>{''.join(sections)}</main></body></html>'''
    output_path.write_text(document, encoding="utf-8")
    return {
        "displayed_video_count": len(sections),
        "displayed_medium_selected_image_count": selected_image_references,
        "candidate_image_reference_count": candidate_image_references,
        "semantic_group_image_reference_count": group_image_references,
        "total_image_reference_count": image_references,
        "missing_image_reference_count": 0,
        "semantic_chain_complete": True,
    }
