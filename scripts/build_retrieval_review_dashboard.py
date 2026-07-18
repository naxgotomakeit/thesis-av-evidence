from __future__ import annotations

import html
import json
import os
import sys
from pathlib import Path
from typing import Any

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.three_channel import temporal_distance, temporal_overlap  # noqa: E402

CASES = ["00002_7", "00004_1", "00018_1", "00003_2", "00006_3", "00061_5"]
LABELS = ["Not reviewed", "Semantic hit", "Partial hit", "Temporal-only hit", "Miss", "Unsure"]
OUT = ROOT / "outputs/retrieval"
ASSETS = OUT / "human_review_assets/audio_clips"
DATASET_ROOT = Path("D:/ThesisData/EgoSound")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def relative_href(target: Path) -> str:
    return Path(os.path.relpath(target, OUT)).as_posix()


def weak_metrics(result: dict[str, Any], start: float, end: float) -> tuple[bool, float]:
    overlap = temporal_overlap(result["start_time"], result["end_time"], start, end)
    distance = temporal_distance(result["start_time"], result["end_time"], start, end)
    return overlap, distance


def badge(overlap: bool) -> str:
    label = "YES — overlaps weak reference" if overlap else "NO — no weak-reference overlap"
    style = "overlap" if overlap else "no-overlap"
    return f'<span class="badge {style}">{label}</span>'


def review_controls(case_id: str, modality: str, rank: int, identifier: str) -> str:
    key = f"{case_id}|{modality}|{rank}|{identifier}"
    options = "".join(f'<option value="{html.escape(label)}">{html.escape(label)}</option>' for label in LABELS)
    return f'''<div class="review-controls" data-review-key="{html.escape(key)}" data-case-id="{case_id}" data-modality="{modality}" data-rank="{rank}" data-region-id="{html.escape(identifier)}">
      <label>Manual semantic judgment<select class="manual-label">{options}</select></label>
      <label>Optional note<textarea class="manual-note" maxlength="500" placeholder="Short human-review note"></textarea></label>
    </div>'''


def make_audio_clip(case: dict[str, Any], result: dict[str, Any], rank: int) -> Path:
    source = DATASET_ROOT / Path(case["audio_path"])
    info = sf.info(source)
    start = float(result["start_time"]); end = float(result["end_time"])
    start_frame = max(0, int(round(start * info.samplerate)))
    end_frame = min(info.frames, int(round(end * info.samplerate)))
    with sf.SoundFile(source) as handle:
        handle.seek(start_frame)
        audio = handle.read(end_frame - start_frame, dtype="float32", always_2d=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    target = ASSETS / f"{case['case_id']}_acoustic_rank{rank}_{start:.3f}_{end:.3f}.wav"
    sf.write(target, audio, info.samplerate, subtype="PCM_16")
    return target


def result_card(case: dict[str, Any], modality: str, result: dict[str, Any], ref_start: float, ref_end: float) -> str:
    overlap, distance = weak_metrics(result, ref_start, ref_end)
    rank = int(result["rank"]); score = float(result["similarity_score"])
    start = float(result["start_time"]); end = float(result["end_time"]); duration = float(result["duration"])
    if modality == "visual":
        identifier = result["visual_region_id"]
        image_path = ROOT / result["representative_keyframe_path"]
        href = relative_href(image_path)
        content = f'''<a class="thumb-link" href="{html.escape(href)}" target="_blank"><img class="keyframe" src="{html.escape(href)}" alt="{identifier} representative keyframe"></a>
        <p><strong>Visual region:</strong> {html.escape(identifier)}</p>'''
    elif modality == "speech":
        identifier = result["transcript_segment_id"]
        content = f'''<div class="transcript">{html.escape(result.get("transcript_text", ""))}</div>
        <p><strong>Transcript segment:</strong> {html.escape(identifier)} &nbsp; <strong>Speech region:</strong> {html.escape(str(result.get("speech_region_id") or "unavailable"))}</p>
        <p><strong>ASR language:</strong> {html.escape(str(result.get("asr_language") or "unknown"))}</p>'''
    else:
        identifier = result["acoustic_region_id"]
        clip = make_audio_clip(case, result, rank)
        href = relative_href(clip)
        content = f'''<p><strong>Acoustic region:</strong> {html.escape(identifier)}</p>
        <p><strong>Mean RMS:</strong> {result['mean_rms']:.6f} &nbsp; <strong>Peak RMS:</strong> {result['peak_rms']:.6f}</p>
        <p><strong>Mean spectral change:</strong> {result['mean_spectral_change_score']:.6f} &nbsp; <strong>Speech overlap ratio:</strong> {result['speech_overlap_ratio']:.4f}</p>
        <p><strong>Low-information/silence flag:</strong> {str(result['low_information_or_silence']).upper()}</p>
        <audio controls preload="metadata" src="{html.escape(href)}">Your browser does not support local WAV playback.</audio>
        <p class="small">Audio clip uses the exact retrieved boundary {start:.3f}–{end:.3f}s; no context padding.</p>'''
    return f'''<article class="result-card {modality}">
      <div class="card-head"><span class="rank">Rank {rank}</span><span>Cosine similarity: <strong>{score:.6f}</strong></span></div>
      {content}
      <p><strong>Interval:</strong> {start:.3f}–{end:.3f}s &nbsp; <strong>Duration:</strong> {duration:.3f}s</p>
      <p>{badge(overlap)} <strong>Temporal distance:</strong> {distance:.3f}s</p>
      {review_controls(case['case_id'], modality, rank, identifier)}
    </article>'''


def main() -> int:
    frozen_cases = {row["case_id"]: row for row in read_json(ROOT / "data/manifests/mvp_cases_6.json")}
    bundles = []
    template = []
    for case_id in CASES:
        case = frozen_cases[case_id]
        result = read_json(OUT / case_id / "three_channel_retrieval.json")
        # Use frozen ranked lists verbatim; never sort or recompute similarity.
        channels = {"visual": result["visual_results"], "speech": result["speech_results"], "acoustic": result["acoustic_results"]}
        ref = result["weak_reference_interval"]
        for modality, rows in channels.items():
            for row in rows:
                identifier = row.get("visual_region_id") or row.get("transcript_segment_id") or row.get("acoustic_region_id")
                template.append({"case_id": case_id, "modality": modality, "rank": row["rank"], "region_or_segment_id": identifier, "manual_label": "Not reviewed", "manual_note": ""})
        bundles.append((case, result, channels, float(ref["start"]), float(ref["end"])))
    (OUT / "human_review_template.json").write_text(json.dumps(template, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_rows = []
    sections = []
    md_rows = []
    for case, result, channels, ref_start, ref_end in bundles:
        d = result["diagnostics"]; u = result["union_diagnostics"]
        known_warning = "; ".join(result.get("warnings", [])) or "None"
        summary_rows.append(f'''<tr><td><a href="#{case['case_id']}">{case['case_id']}</a></td><td>{html.escape(case['question'])}</td><td>{ref_start:.3f}–{ref_end:.3f}s</td><td>{str(d['visual']['top1_overlap'])}/{str(d['visual']['top3_overlap'])}</td><td>{str(d['speech']['top1_overlap'])}/{str(d['speech']['top3_overlap'])}</td><td>{str(d['acoustic']['top1_overlap'])}/{str(d['acoustic']['top3_overlap'])}</td><td>{str(u['union_top1_overlap'])}/{str(u['union_top3_overlap'])}</td><td>{html.escape(known_warning)}</td></tr>''')
        md_rows.append(f"| {case['case_id']} | {case['question'].replace('|','/')} | {ref_start:.3f}–{ref_end:.3f}s | {d['visual']['top1_overlap']}/{d['visual']['top3_overlap']} | {d['speech']['top1_overlap']}/{d['speech']['top3_overlap']} | {d['acoustic']['top1_overlap']}/{d['acoustic']['top3_overlap']} | {u['union_top1_overlap']}/{u['union_top3_overlap']} |")
        timeline = relative_href(OUT / case["case_id"] / "retrieval_timeline.png")
        warning_items = "".join(f"<li>{html.escape(x)}</li>" for x in result.get("warnings", [])) or "<li>None</li>"
        known_61 = ""
        if case["case_id"] == "00061_5":
            known_61 = '''<div class="known-warning"><strong>Known difficult distant/overlapping-speech case.</strong><br>Frozen ASR did not recover the target phrase.<br>Suspected speaker-attribution inconsistency in the dataset.<br>The original annotation was not modified.</div>'''
        modality_html = []
        for modality, title in [("visual", "Visual Top-3"), ("speech", "Speech Top-3"), ("acoustic", "Acoustic Top-3")]:
            rows = channels[modality]
            cards = "".join(result_card(case, modality, row, ref_start, ref_end) for row in rows)
            if len(rows) < 3:
                cards += f'<div class="fewer">Only {len(rows)} result(s): the frozen {modality} index contains fewer than Top-3 items.</div>'
            modality_html.append(f'<section class="modality-section"><h3>{title}</h3><div class="cards">{cards}</div></section>')
        sections.append(f'''<section class="case" id="{case['case_id']}">
          <h2>{case['case_id']} · video {case['video_id']}</h2>
          <p class="question">{html.escape(case['question'])}</p>
          <div class="facts"><span>Video duration: {float(case['video_duration']):.3f}s</span><span>Audio duration: {float(case['audio_duration']):.3f}s</span><span>Weak reference: {ref_start:.3f}–{ref_end:.3f}s</span></div>
          {known_61}<details><summary>Known warnings</summary><ul>{warning_items}</ul></details>
          <a href="{timeline}" target="_blank"><img class="timeline" src="{timeline}" alt="Retrieval timeline for {case['case_id']}"></a>
          {''.join(modality_html)}
        </section>''')

    css = '''body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f3f4f6;color:#172033}main{max-width:1500px;margin:auto;padding:24px}h1,h2,h3{color:#111827}.intro,.case{background:white;border-radius:12px;padding:22px;margin-bottom:24px;box-shadow:0 2px 10px #0001}.notice{border-left:5px solid #ec4899;background:#fdf2f8;padding:14px}table{width:100%;border-collapse:collapse;font-size:13px}th,td{border:1px solid #d1d5db;padding:8px;vertical-align:top}th{background:#eef2ff;position:sticky;top:0}.table-wrap{overflow:auto}.question{font-size:19px;font-weight:600}.facts{display:flex;gap:20px;flex-wrap:wrap;background:#f8fafc;padding:10px}.known-warning{background:#fff7ed;border:2px solid #fb923c;padding:14px;margin:14px 0}.timeline{width:100%;max-height:620px;object-fit:contain;margin:15px 0;border:1px solid #ddd}.modality-section{border-top:3px solid #ddd;margin-top:24px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:15px}.result-card{border:1px solid #cbd5e1;border-top-width:6px;border-radius:9px;padding:14px;background:#fff}.result-card.visual{border-top-color:#8b5cf6}.result-card.speech{border-top-color:#2563eb}.result-card.acoustic{border-top-color:#0f9d76}.card-head{display:flex;justify-content:space-between;gap:8px}.rank{font-size:18px;font-weight:bold}.keyframe{width:100%;height:210px;object-fit:contain;background:#111}.transcript{font-size:19px;line-height:1.45;background:#eff6ff;padding:15px;margin:12px 0;min-height:55px}audio{width:100%}.badge{display:inline-block;padding:4px 8px;border-radius:14px;font-weight:700;font-size:12px}.overlap{background:#dcfce7;color:#166534}.no-overlap{background:#e5e7eb;color:#374151}.review-controls{margin-top:14px;background:#f8fafc;padding:10px}.review-controls label{display:block;font-weight:600;margin:7px 0}.review-controls select,.review-controls textarea{display:block;width:100%;box-sizing:border-box;margin-top:5px;padding:7px}.review-controls textarea{height:65px}.small{font-size:12px;color:#475569}.fewer{padding:15px;background:#f1f5f9}.toolbar{position:sticky;bottom:0;background:#111827;color:white;padding:12px;display:flex;gap:10px;align-items:center;z-index:5}.toolbar button{padding:9px 14px;font-weight:bold}'''
    script = '''const STORAGE_KEY="task4_retrieval_human_review_v1";
function controls(){return Array.from(document.querySelectorAll(".review-controls"));}
function collect(){return controls().map(c=>({case_id:c.dataset.caseId,modality:c.dataset.modality,rank:Number(c.dataset.rank),region_or_segment_id:c.dataset.regionId,manual_label:c.querySelector(".manual-label").value,manual_note:c.querySelector(".manual-note").value}));}
function save(){localStorage.setItem(STORAGE_KEY,JSON.stringify(collect()));document.getElementById("save-status").textContent="Saved locally at "+new Date().toLocaleTimeString();}
function restore(){let raw=localStorage.getItem(STORAGE_KEY);if(!raw)return;let map=new Map(JSON.parse(raw).map(x=>[[x.case_id,x.modality,x.rank,x.region_or_segment_id].join("|"),x]));controls().forEach(c=>{let x=map.get(c.dataset.reviewKey);if(x){c.querySelector(".manual-label").value=x.manual_label;c.querySelector(".manual-note").value=x.manual_note||"";}});}
function exportJson(){save();let blob=new Blob([JSON.stringify(collect(),null,2)],{type:"application/json"});let a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="task4_manual_retrieval_judgments.json";a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}
document.addEventListener("DOMContentLoaded",()=>{restore();controls().forEach(c=>{c.querySelector("select").addEventListener("change",save);c.querySelector("textarea").addEventListener("input",save);});document.getElementById("save-btn").addEventListener("click",save);document.getElementById("export-btn").addEventListener("click",exportJson);});'''
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Task 4 Retrieval Human Review</title><style>{css}</style></head><body><main>
    <section class="intro"><h1>Task 4 three-channel retrieval · human review</h1><ol><li>Retrieval has already been performed using embedding cosine similarity.</li><li>This page is only for checking whether retrieved content is genuinely relevant to the question.</li><li>“Overlap” means temporal overlap with the dataset reference only.</li><li>Human labels assess semantic relevance.</li></ol><div class="notice"><strong>The reference interval was not used for retrieval and is not a precise gold event boundary.</strong><br>Similarity is not confidence. Temporal overlap is not semantic correctness. No answers or answer options are shown.</div>
    <h2>Overall summary</h2><div class="table-wrap"><table><thead><tr><th>Case</th><th>Raw question</th><th>Weak reference</th><th>Visual T1/T3</th><th>Speech T1/T3</th><th>Acoustic T1/T3</th><th>Union T1/T3</th><th>Known warning</th></tr></thead><tbody>{''.join(summary_rows)}</tbody></table></div></section>
    {''.join(sections)}
    <div class="toolbar"><button id="save-btn">Save in browser</button><button id="export-btn">Export judgments JSON</button><span id="save-status">All controls start as Not reviewed.</span></div>
    </main><script>{script}</script></body></html>'''
    html_path = OUT / "human_review_summary.html"
    html_path.write_text(page, encoding="utf-8")
    md = ["# Task 4 retrieval human-review summary", "", "Open `human_review_summary.html` locally for timelines, keyframes, audio playback, review controls, localStorage persistence, and JSON export.", "", "The reference interval was not used for retrieval and is not a precise gold event boundary.", "", "| Case | Raw question | Weak reference | Visual T1/T3 | Speech T1/T3 | Acoustic T1/T3 | Union T1/T3 |", "|---|---|---|---|---|---|---|"] + md_rows + ["", "No answers or answer options are included. Similarity is not confidence; temporal overlap is not semantic correctness."]
    (OUT / "human_review_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"html": str(html_path), "cases": len(bundles), "review_records": len(template), "audio_clips": len(list(ASSETS.glob('*.wav')))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
