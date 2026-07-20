"""Human-first semantic chain report."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def _e(value: Any) -> str:
    return html.escape(str(value))


def _clock(seconds: float) -> str:
    minutes, sec = divmod(float(seconds), 60.0)
    return f"{int(minutes):02d}:{sec:05.2f}"


def render_html(*, videos: list[dict[str, Any]], aggregate: dict[str, Any], output_path: Path) -> dict[str, Any]:
    sections = []
    image_count = 0
    missing = []
    for video in videos:
        medium_by_id = {row["medium_id"]: row for row in video["medium"]}
        coarse_html = []
        for coarse in video["coarse"]:
            children_html = []
            for medium_id in coarse["child_medium_ids"]:
                medium = medium_by_id[medium_id]
                relative = medium["selected_keyframe"]["html_relative_path"]
                image_path = output_path.parent / relative
                image_count += 1
                if not image_path.is_file():
                    missing.append(relative)
                children_html.append(f"""
                <article class='medium'>
                  <img src='{_e(relative)}' loading='lazy' alt='{_e(medium_id)} keyframe'>
                  <div><h4>{_e(medium_id)} · {_clock(medium['start'])}–{_clock(medium['end'])}</h4>
                  <p class='caption'>{_e(medium['caption'])}</p>
                  <p class='meta'>selected {_clock(medium['selected_keyframe']['timestamp'])} · one direct local-Qwen image call · {medium['timing']['model_inference_sec']:.2f}s inference</p>
                  <details><summary>Keyframe candidates and selection trace</summary><pre>{_e(json.dumps(medium['keyframe_trace'], ensure_ascii=False, indent=2))}</pre></details>
                  </div>
                </article>""")
            coarse_html.append(f"""
            <section class='coarse'>
              <header><h3>{_e(coarse['coarse_id'])} · {_clock(coarse['start'])}–{_clock(coarse['end'])} · {coarse['child_count']} Medium(s)</h3>
              <p class='coarse-caption'>{_e(coarse['summary'])}</p>
              <span class='pill'>{_e(coarse['summary_source'])}</span></header>
              <div class='children'>{''.join(children_html)}</div>
              <div class='review'><b>Manual review:</b> grouping meaningful? ☐ yes ☐ no ☐ unsure &nbsp; overmerged? ☐ yes ☐ no ☐ unsure<br>summary correct? ☐ yes ☐ partial ☐ no<br>notes: <span class='line'></span></div>
            </section>""")
        timeline = "".join(
            f"<div title='{_e(c['coarse_id'])} {_clock(c['start'])}–{_clock(c['end'])}' style='width:{100*c['duration']/video['video_duration']:.4f}%'></div>"
            for c in video["coarse"]
        )
        sections.append(f"""
        <section class='video' id='{_e(video['video_id'])}'>
          <h2>{_e(video['video_id'])}</h2>
          <p class='stats'>{_clock(video['video_duration'])} · Fine {video['fine_count']} → frozen Fluid Loose Medium {video['medium_count']} → semantic Coarse {video['coarse_count']}</p>
          <div class='story'><h3>Overall storyline</h3><pre>{_e(video['story']['story'])}</pre></div>
          <div class='timeline'>{timeline}</div>
          {''.join(coarse_html)}
          <div class='review video-review'><b>Whole-video review:</b> understandable? ☐ yes ☐ partial ☐ no &nbsp; major phases preserved? ☐ yes ☐ unsure ☐ no<br>notes: <span class='line'></span></div>
        </section>""")
    rows = "".join(
        f"<tr><td><a href='#{_e(v['video_id'])}'>{_e(v['video_id'])}</a></td><td>{_clock(v['video_duration'])}</td><td>{v['medium_count']}</td><td>{v['coarse_count']}</td><td>{v['calls']['medium_vlm']}</td><td>{v['calls']['boundary_decision']}</td><td>{v['calls']['coarse_summary']}</td><td>{v['calls']['singleton_copy_through']}</td><td>{v['runtime']['steady_state_total_sec']:.1f}s</td></tr>"
        for v in videos
    )
    document = f"""<!doctype html><html><head><meta charset='utf-8'><title>semantic_coarse_v0_1</title>
    <style>
    :root{{--ink:#172033;--muted:#64748b;--bg:#f4f7fb;--card:#fff;--blue:#2563eb;--border:#d8e1ee}}
    *{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,Segoe UI,sans-serif}}
    main{{max-width:1500px;margin:auto;padding:24px}} h1,h2,h3,h4{{margin:.35rem 0}} table{{width:100%;border-collapse:collapse;background:white}} th,td{{padding:8px;border:1px solid var(--border);text-align:left}}
    .video{{background:#fff;margin:28px 0;padding:22px;border-radius:14px;box-shadow:0 5px 22px #1e293b12}}
    .stats,.meta{{color:var(--muted)}} .story{{background:#eef5ff;border-left:5px solid var(--blue);padding:12px 16px;margin:12px 0}} pre{{white-space:pre-wrap;overflow-wrap:anywhere}}
    .timeline{{height:22px;display:flex;border-radius:8px;overflow:hidden;margin:14px 0;background:#dbeafe}} .timeline div{{border-right:2px solid white;background:#60a5fa}} .timeline div:nth-child(even){{background:#93c5fd}}
    .coarse{{border:2px solid #bfdbfe;border-radius:12px;margin:18px 0;overflow:hidden}} .coarse>header{{background:#eff6ff;padding:12px 16px}} .coarse-caption{{font-size:1.08rem;font-weight:650}}
    .pill{{background:#dbeafe;border-radius:999px;padding:2px 9px;font-size:.8rem}} .children{{padding:10px 16px}}
    .medium{{display:grid;grid-template-columns:260px 1fr;gap:16px;border-top:1px solid var(--border);padding:14px 0}} .medium:first-child{{border-top:0}} .medium img{{width:260px;height:150px;object-fit:cover;border-radius:8px;background:#111}}
    .caption{{font-size:1.05rem}} details{{margin-top:8px}} details pre{{max-height:420px;overflow:auto;background:#f8fafc;padding:8px;font-size:.75rem}}
    .review{{background:#fffceb;border-top:1px dashed #d6b947;padding:10px 16px}} .line{{display:inline-block;border-bottom:1px solid #555;width:60%}}
    @media(max-width:720px){{.medium{{grid-template-columns:1fr}}.medium img{{width:100%;height:auto}}}}
    </style></head><body><main>
    <h1>Semantic Coarse v0.1 — frozen Fluid Loose Medium → semantic Coarse → storyline</h1>
    <p>Source is exclusively <b>A Current Fluid Loose</b>. Fine leaves, Safe-Merge topology, and Medium intervals are frozen. Images are read only for Medium captions; Coarse summaries and the storyline use ordered text only.</p>
    <table><thead><tr><th>Video</th><th>Duration</th><th>Medium</th><th>Coarse</th><th>Image calls</th><th>Boundary calls</th><th>Coarse calls</th><th>Copy-through</th><th>Steady runtime</th></tr></thead><tbody>{rows}</tbody></table>
    <p><b>Total:</b> Medium {aggregate['medium_count']}, semantic Coarse {aggregate['coarse_count']}, Medium image calls {aggregate['calls']['medium_vlm']}, Coarse summary calls {aggregate['calls']['coarse_summary']}, singleton copy-through savings {aggregate['calls']['singleton_copy_through']}.</p>
    {''.join(sections)}
    </main></body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    return {"videos_displayed": len(videos), "medium_images_displayed": image_count, "missing_image_references": missing}
