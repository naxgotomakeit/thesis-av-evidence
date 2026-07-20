"""Human-review report for Fine-only versus hierarchy-guided retrieval."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def thumb_src(path: str) -> str:
    value = path.replace("\\", "/")
    return "../../" + value[len("outputs/"):] if value.startswith("outputs/") else value


def timeline(label: str, nodes: list[dict[str, Any]], duration: float, selected: set[str], visited: set[str] | None = None) -> str:
    bars = []
    visited = visited or set()
    for node in nodes:
        identifier = node["node_id"]
        css = "selected" if identifier in selected else ("visited" if identifier in visited else "pruned")
        bars.append(
            f'<span class="bar {css}" style="left:{100*float(node["start"])/duration:.5f}%;width:{max(.2,100*float(node["duration"])/duration):.5f}%" title="{e(identifier)} | {float(node["start"]):.1f}-{float(node["end"]):.1f}s"></span>'
        )
    return f'<div class="timeline-row"><b>{e(label)}</b><div class="axis">{"".join(bars)}<i>0</i><i class="m">90s</i><i class="r">180s</i></div></div>'


def thumbnails(selected: list[dict[str, Any]], nodes: dict[str, dict[str, Any]]) -> str:
    figures = []
    for result in selected:
        node = nodes[result["node_id"]]
        reps = node.get("representative_frames", [])
        representative = next((row for row in reps if "medoid" in row.get("kind", "")), reps[len(reps)//2] if reps else None)
        if representative:
            figures.append(
                f'<figure><img loading="lazy" src="{e(thumb_src(representative["frame_path"]))}"><figcaption><b>{e(result["node_id"])}</b><br>{float(result["start"]):.1f}-{float(result["end"]):.1f}s · {float(result["score"]):.4f}</figcaption></figure>'
            )
    return '<div class="thumbs">' + "".join(figures) + "</div>"


def score_table(rows: list[dict[str, Any]], selected: set[str], visited: set[str] | None = None) -> str:
    visited = visited or set()
    body = []
    for row in rows:
        state = "SELECTED FINAL" if row["node_id"] in selected else ("VISITED" if row["node_id"] in visited else "PRUNED")
        body.append(f'<tr class="{state.lower().replace(" ","-")}"><td>{row["rank"]}</td><td>{e(row["node_id"])}</td><td>{float(row["start"]):.1f}-{float(row["end"]):.1f}</td><td>{float(row["score"]):.5f}</td><td>{state}</td></tr>')
    return '<table><thead><tr><th>Rank</th><th>Node</th><th>Interval</th><th>Score</th><th>State</th></tr></thead><tbody>' + "".join(body) + '</tbody></table>'


def tree(parent_id: str, medium_ids: list[str], nodes: dict[str, dict[str, Any]], guided: dict[str, Any]) -> str:
    parent = nodes[parent_id]
    selected_coarse = {row["node_id"] for row in guided["selected_coarse"]}
    selected_medium = {row["node_id"] for row in guided["selected_medium"]}
    visited_fine = set(guided["visited_fine_ids"])
    final = {row["node_id"] for row in guided["selected_final"]}
    contained = [identifier for identifier in medium_ids if set(nodes[identifier]["leaf_ids"]).issubset(set(parent["leaf_ids"]))]
    blocks = []
    def node_thumb(node: dict[str, Any]) -> str:
        reps = node.get("representative_frames", [])
        rep = next((row for row in reps if "medoid" in row.get("kind", "")), reps[len(reps)//2] if reps else None)
        return f'<img class="tree-thumb" loading="lazy" src="{e(thumb_src(rep["frame_path"]))}" title="{e(rep["kind"])} · {float(rep["timestamp"]):.1f}s">' if rep else ""
    for mid in contained:
        leaves = nodes[mid]["leaf_ids"]
        leaf_text = " ".join(
            f'<span class="tag {"selected" if leaf in final else ("visited" if leaf in visited_fine else "pruned")}">{e(leaf)}</span>'
            for leaf in leaves
        )
        blocks.append(f'<details><summary><span class="tag {"visited" if mid in selected_medium else "pruned"}">{e(mid)}</span> {float(nodes[mid]["start"]):.1f}-{float(nodes[mid]["end"]):.1f}s · {len(leaves)} Fine descendants</summary>{node_thumb(nodes[mid])}<div>{leaf_text}</div></details>')
    merge = parent.get("merge_record") or {}
    return f'<details class="branch"><summary><span class="tag {"visited" if parent_id in selected_coarse else "pruned"}">{e(parent_id)}</span> {float(parent["start"]):.1f}-{float(parent["end"]):.1f}s · {len(parent["leaf_ids"])} Fine · merge {merge.get("merge_score","leaf")} · variability {float(parent.get("internal_variability",0)):.4f}</summary>{node_thumb(parent)}{"".join(blocks)}</details>'


def select(key: str, values: list[str]) -> str:
    return f'<select class="manual" data-key="{e(key)}">' + "".join(f'<option>{e(v)}</option>' for v in values) + '</select>'


def render(
    *, output: Path, aggregate: dict[str, Any], results: list[dict[str, Any]],
    hierarchies: dict[str, dict[str, Any]], metrics: list[dict[str, Any]], old_reference: dict[str, dict[str, Any]],
) -> None:
    metric_by_video = {row["video_id"]: row for row in metrics}
    cards = []
    for index, row in enumerate(results, start=1):
        video_id = row["video_id"]
        hierarchy = hierarchies[video_id]
        nodes = {node["node_id"]: node for node in hierarchy["nodes"]}
        fine = [nodes[identifier] for identifier in hierarchy["cuts"]["fine"]["node_ids"]]
        medium_ids = hierarchy["cuts"]["medium"]["node_ids"]
        medium = [nodes[identifier] for identifier in medium_ids]
        coarse_ids = hierarchy["cuts"]["coarse"]["node_ids"]
        coarse = [nodes[identifier] for identifier in coarse_ids]
        a, b = row["fine_only"], row["hierarchy_guided"]
        a_selected = {x["node_id"] for x in a["selected_final"]}
        b_selected = {x["node_id"] for x in b["selected_final"]}
        b_visited = set(b["visited_fine_ids"])
        metric = metric_by_video[video_id]
        key = video_id.replace("-", "_")
        old = old_reference.get(video_id)
        cards.append(f'''<article id="case-{key}"><div class="head"><div><span class="num">{index}</span><b>{e(video_id)}</b> <span class="pill">{e(row['group'])}</span> <span class="pill">scope: {e(row.get('frozen_scope') or 'unavailable')}</span></div><a href="#top">Back to top</a></div><h2>{e(row['question'])}</h2>
<div class="timeline-box">{timeline('Fine-only · all Fine scored',fine,180,a_selected,set(x['node_id'] for x in a['ranking']))}{timeline('Hierarchy · Coarse view',coarse,180,{x['node_id'] for x in b['selected_coarse']},{x['node_id'] for x in b['coarse_ranking']})}{timeline('Hierarchy · Medium view',medium,180,{x['node_id'] for x in b['selected_medium']},set(b['medium_candidates']))}{timeline('Hierarchy · Fine reached',fine,180,b_selected,b_visited)}</div>
<div class="summary-grid"><div><h3>A · Fine-only</h3><p><b>{a['fine_nodes_scored']}</b> Fine comparisons · <b>{a['total_vector_comparisons']}</b> total vector comparisons · {a['activated_temporal_duration']:.1f}s final duration</p>{thumbnails(a['selected_final'],nodes)}</div><div><h3>B · Hierarchy-guided</h3><p><b>{b['fine_nodes_scored']}</b> Fine comparisons + <b>{b['operating_view_nodes_scored']}</b> parent/cut nodes · <b>{b['total_vector_comparisons']}</b> vector comparisons · {b['activated_temporal_duration']:.1f}s final duration</p>{thumbnails(b['selected_final'],nodes)}</div></div>
<div class="diagnostic"><b>Fine-only Top-3 diagnostic proxy:</b> Top-1 reached = {b['diagnostic_proxy']['fine_only_top1_reached']} · Top-3 branch reach = {100*b['diagnostic_proxy']['fine_only_top3_reached_rate']:.1f}% · final-ID overlap = {100*b['diagnostic_proxy']['final_selected_id_overlap_rate']:.1f}% · useful-proxy leaves pruned: {e(b['diagnostic_proxy']['useful_fine_pruned'])}</div>
<div class="columns"><section><h3>Fine-only ranking</h3>{score_table(a['ranking'],a_selected,set(x['node_id'] for x in a['ranking']))}</section><section><h3>Hierarchy branch scores</h3><h4>Coarse operating view</h4>{score_table(b['coarse_ranking'],{x['node_id'] for x in b['selected_coarse']},{x['node_id'] for x in b['selected_coarse']})}<h4>Visited Medium candidates</h4>{score_table(b['medium_ranking'],{x['node_id'] for x in b['selected_medium']},{x['node_id'] for x in b['selected_medium']})}<h4>Reached Fine ranking</h4>{score_table(b['fine_ranking'],b_selected,b_visited)}</section></div>
<h3>Inspectable persistent hierarchy</h3><p class="legend"><span class="tag visited">VISITED</span><span class="tag pruned">PRUNED</span><span class="tag selected">SELECTED FINAL</span></p>{''.join(tree(identifier,medium_ids,nodes,b) for identifier in coarse_ids)}
<div class="failure"><b>Automatic structural categories (not human truth):</b> {e(', '.join(metric['structural_failure_categories']) or 'none')}<br>{e(metric['structural_explanation'])}</div>{f'<div class="old">Historical KTS→local-CoMET reference: {old["local_comet_segments_scored"]} local Fine scored; {old["selected_coverage_ratio"]*100:.1f}% final coverage. Not part of A vs B.</div>' if old else ''}
<div class="manual-panel"><h3>Manual review — blank</h3><label>Better retrieval {select(key+':better',['Unreviewed','Fine-only','Hierarchy-guided','Similar','Neither','Unclear'])}</label><label>Needed Fine evidence preserved? {select(key+':preserved',['Unreviewed','Yes','No','Unclear'])}</label><label>Necessary branch pruned? {select(key+':pruned',['Unreviewed','Yes','No','Unclear'])}</label><label>Parent grouping meaningful? {select(key+':meaningful',['Unreviewed','Yes','No','Unclear'])}</label><label>Obvious over-merge? {select(key+':overmerge',['Unreviewed','Yes','No','Unclear'])}</label><label>Broader coverage needed? {select(key+':coverage',['Unreviewed','Yes','No','Unclear'])}</label><label class="wide">Notes<textarea class="manual" data-key="{key}:notes" rows="3"></textarea></label></div><details><summary>Raw deterministic trace</summary><pre>{e(json.dumps(row,indent=2,ensure_ascii=False))}</pre></details></article>''')
    a = aggregate["fine_only"]
    b = aggregate["hierarchy_guided_primary"]
    document = f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Fine-only vs hierarchy-guided v0.1</title><style>
:root{{--bg:#f4f6f8;--paper:#fff;--ink:#1c2834;--line:#ccd6df;--blue:#315f8f;--green:#26846d;--gray:#929eaa;--red:#b45151}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,Segoe UI,sans-serif}}header,main{{max-width:1550px;margin:auto}}header{{padding:24px}}main{{padding:0 24px 60px}}article,.stat{{background:var(--paper);border:1px solid var(--line);border-radius:10px}}article{{padding:17px;margin:22px 0}}h1{{margin:0}}.notice,.diagnostic,.failure,.old{{padding:11px;margin:10px 0;border-left:5px solid #d59a00;background:#fff7dc}}.failure{{border-color:var(--red);background:#fff1f1}}.old{{border-color:#777;background:#f0f2f4}}.aggregate{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:9px}}.stat{{padding:12px}}.stat b{{font-size:22px;color:var(--blue);display:block}}.head{{display:flex;justify-content:space-between}}.num{{display:inline-grid;place-items:center;width:28px;height:28px;border-radius:50%;background:var(--blue);color:#fff}}.pill,.tag{{display:inline-block;border-radius:999px;padding:3px 8px;background:#e8eef4;margin:2px}}.timeline-box{{border:1px solid var(--line);padding:18px 10px 8px;overflow-x:auto}}.timeline-row{{display:grid;grid-template-columns:190px minmax(900px,1fr);margin:8px 0;align-items:center}}.axis{{height:28px;position:relative;border:1px solid #aeb9c4;background:linear-gradient(to right,#edf2f5 49.8%,#bfcbd6 50%,#edf2f5 50.2%)}}.axis i{{position:absolute;top:-17px;font-size:10px;font-style:normal}}.axis .m{{left:49%}}.axis .r{{right:0}}.bar{{position:absolute;top:3px;height:21px;border:1px solid rgba(0,0,0,.3)}}.bar.visited,.tag.visited{{background:#56a98f}}.bar.pruned,.tag.pruned{{background:#aab3bc}}.bar.selected,.tag.selected{{background:#e67e43}}.summary-grid,.columns{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:12px 0}}.summary-grid>div,.columns>section{{border:1px solid var(--line);padding:10px;border-radius:8px;overflow:auto}}.thumbs{{display:flex;gap:8px;overflow:auto}}figure{{margin:0;min-width:145px}}img{{width:145px;height:90px;object-fit:cover;border-radius:4px}}figcaption{{font-size:11px}}table{{border-collapse:collapse;width:100%;font-size:11px}}th,td{{padding:4px;border-bottom:1px solid #dde4ea;text-align:left}}tr.selected-final{{background:#fff0e5}}.branch{{border-left:4px solid var(--blue);padding:6px;margin:7px 0}}.manual-panel{{display:grid;grid-template-columns:1fr 1fr;gap:8px;background:#edf4f8;padding:12px}}.manual-panel h3,.wide{{grid-column:1/-1}}label{{display:flex;justify-content:space-between}}textarea{{width:100%}}pre{{white-space:pre-wrap;max-height:500px;overflow:auto;background:#111b25;color:#e9eef3;padding:12px}}@media(max-width:900px){{.summary-grid,.columns,.manual-panel{{grid-template-columns:1fr}}.manual-panel h3,.wide{{grid-column:auto}}}}</style></head><body><header id="top"><h1>Fine-only vs reusable hierarchy-guided retrieval v0.1</h1><p>Primary beam = {aggregate['primary_beam_width']} · same immutable Fine leaves, CLIP query encoder, and final Top-{aggregate['final_evidence_budget']} budget.</p><div class="notice"><b>Reference operating cuts only:</b> 50%/25% cuts are navigation views, not natural semantic levels. Fine-only Top-3 is a diagnostic reference, not ground-truth evidence. No options, gold, final QA, Planner, or API calls.</div><div class="aggregate"><div class="stat"><b>{aggregate['case_count']}</b>cases</div><div class="stat"><b>{a['mean_fine_nodes_scored']:.1f}</b>Fine-only Fine scores</div><div class="stat"><b>{b['mean_fine_nodes_scored']:.1f}</b>guided Fine scores</div><div class="stat"><b>{100*b['fine_score_reduction']:.1f}%</b>Fine-score reduction</div><div class="stat"><b>{b['mean_total_node_score_operations']:.1f}</b>guided node scores</div><div class="stat"><b>{100*b['total_node_score_reduction']:.1f}%</b>total-node reduction</div><div class="stat"><b>{100*b['fine_only_top1_reach_rate']:.1f}%</b>Fine-only Top-1 reached</div><div class="stat"><b>{100*b['fine_only_top3_reach_rate']:.1f}%</b>Fine-only Top-3 branch reach</div></div><p>Cases: {' '.join(f'<a href="#case-{r["video_id"].replace("-","_")}">{i}</a>' for i,r in enumerate(results,1))}</p></header><main>{''.join(cards)}</main><script>document.querySelectorAll('.manual').forEach(el=>{{const k='fine-hierarchy-review:'+el.dataset.key,v=localStorage.getItem(k);if(v!==null)el.value=v;el.addEventListener('change',()=>localStorage.setItem(k,el.value));}});</script></body></html>'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
