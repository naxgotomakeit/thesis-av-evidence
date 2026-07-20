"""Human-review HTML renderer for question-only taxonomy annotations."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _pct(value: float | None) -> str:
    return "N/A" if value is None else f"{100 * float(value):.1f}%"


def _distribution_table(distribution: dict[str, Any], axis: str) -> str:
    groups = distribution["groups"]
    names = (
        ("EgoSchema representative", "egoschema_representative"),
        ("EgoSound representative", "egosound_representative"),
        ("Combined representative", "combined_representative"),
        ("Combined stress (not prevalence)", "combined_diversity_stress"),
    )
    rows = []
    for label, key in names:
        summary = groups[key][axis]
        counts = ", ".join(f"{_e(name)}={count}" for name, count in summary["counts"].items())
        rows.append(
            f"<tr><td>{_e(label)}</td><td>{summary['count']}</td><td>{counts}</td>"
            f"<td>{float(summary['mean_confidence']):.3f}</td>"
            f"<td>{_pct(summary['low_confidence_rate'])}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Group</th><th>N</th><th>Labels</th><th>Mean confidence</th>"
        "<th>Low confidence</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _select(field: str, choices: list[str], storage_key: str) -> str:
    options = "".join(f'<option value="{_e(value)}">{_e(value)}</option>' for value in choices)
    return (
        f'<select class="manual" data-storage="{_e(storage_key)}" name="{_e(field)}">'
        f"{options}</select>"
    )


def render_review_html(
    *,
    output_path: Path,
    annotations: list[dict[str, Any]],
    distribution: dict[str, Any],
    cross_axis: dict[str, Any],
    stability_metrics: dict[str, Any],
    stability_annotations: list[dict[str, Any]],
    routing: dict[str, Any],
    axis_assessments: dict[str, Any],
    runtime: dict[str, Any],
) -> None:
    disagreement_ids = {
        row["record_id"] for row in stability_annotations if row["any_disagreement"]
    }
    stability_by_id = {row["record_id"]: row for row in stability_annotations}
    cards = []
    for index, row in enumerate(sorted(annotations, key=lambda item: (item["dataset"], item["sample_type"], item["question_id"]))):
        low = any(float(value) < 0.70 for value in row["confidence"].values())
        disagreement = row["record_id"] in disagreement_ids
        flags = []
        if low:
            flags.append("low confidence")
        if row["nature_ambiguity"]:
            flags.append("nature ambiguity")
        if row["taxonomy_failure"]:
            flags.append("taxonomy failure")
        if disagreement:
            flags.append("stability disagreement")
        secondary = row["nature_secondary"] if row["nature_secondary"] is not None else "—"
        key = row["record_id"].replace(":", "_")
        stability = stability_by_id.get(row["record_id"])
        stability_html = ""
        if stability:
            second = stability["pass_2"]
            agreement_text = ", ".join(
                f"{name}={'agree' if value else 'DIFFER'}"
                for name, value in stability["agreements"].items()
            )
            stability_html = (
                '<div class="stability"><strong>Independent stability pass:</strong> '
                f"scope={_e(second['scope'])}; nature={_e(second['nature_primary'])}; "
                f"ambiguity={_e(second['nature_ambiguity'])}; modality={_e(second['modality'])}"
                f"<br><span class=\"muted\">{_e(agreement_text)}</span></div>"
            )
        cards.append(
            f'''<article class="card" id="q-{_e(key)}"
 data-dataset="{_e(row['dataset'])}" data-sample="{_e(row['sample_type'])}"
 data-scope="{_e(row['scope'])}" data-nature="{_e(row['nature_primary'])}"
 data-modality="{_e(row['modality'])}" data-low="{str(low).lower()}"
 data-ambiguity="{str(bool(row['nature_ambiguity'])).lower()}"
 data-failure="{str(bool(row['taxonomy_failure'])).lower()}"
 data-disagreement="{str(disagreement).lower()}">
 <div class="card-head"><div><span class="index">{index + 1}</span>
 <strong>{_e(row['question_id'])}</strong> <span class="pill">{_e(row['dataset'])}</span>
 <span class="pill sample">{_e(row['sample_type'])}</span></div>
 <a href="#top">Back to top</a></div>
 <h3>{_e(row['question'])}</h3>
 <div class="flags">{''.join(f'<span class="flag">{_e(flag)}</span>' for flag in flags) or '<span class="muted">No automatic review flags</span>'}</div>
 <div class="labels">
   <section><h4>Temporal scope</h4><div class="label">{_e(row['scope'])}</div>
   <meter min="0" max="1" value="{float(row['confidence']['scope']):.3f}"></meter>
   <div>confidence {float(row['confidence']['scope']):.2f}</div><p>{_e(row['reason_short']['scope'])}</p></section>
   <section><h4>Evidence nature</h4><div class="label">{_e(row['nature_primary'])}</div>
   <div>secondary: {_e(secondary)} · ambiguity: {_e(str(row['nature_ambiguity']).lower())}</div>
   <meter min="0" max="1" value="{float(row['confidence']['nature']):.3f}"></meter>
   <div>confidence {float(row['confidence']['nature']):.2f}</div><p>{_e(row['reason_short']['nature'])}</p></section>
   <section><h4>Modality</h4><div class="label">{_e(row['modality'])}</div>
   <meter min="0" max="1" value="{float(row['confidence']['modality']):.3f}"></meter>
   <div>confidence {float(row['confidence']['modality']):.2f}</div><p>{_e(row['reason_short']['modality'])}</p></section>
 </div>
 {stability_html}
 <div class="manual-review"><h4>Manual review (saved only in this browser)</h4>
  <label>1. Scope {_select('scope_review', ['unreviewed','Correct','Wrong','Ambiguous'], key + ':scope_review')}</label>
  <label>2. Better scope {_select('better_scope', ['unreviewed','local','multi_event','global','unclear'], key + ':better_scope')}</label>
  <label>3. Nature {_select('nature_review', ['unreviewed','Correct','Wrong','Ambiguous'], key + ':nature_review')}</label>
  <label>4. Better nature {_select('better_nature', ['unreviewed','static','dynamic','uncertain','mixed'], key + ':better_nature')}</label>
  <label>5. Modality {_select('modality_review', ['unreviewed','Correct','Wrong','Cannot infer from question'], key + ':modality_review')}</label>
  <label>6. Better modality {_select('better_modality', ['unreviewed','visual','audio','audio_visual','indeterminate'], key + ':better_modality')}</label>
  <label>7. Taxonomy itself insufficient? {_select('taxonomy_insufficient', ['unreviewed','Yes','No'], key + ':taxonomy_insufficient')}</label>
  <label class="wide">8. Missing concept / note<textarea class="manual" data-storage="{_e(key + ':note')}" rows="2"></textarea></label>
 </div>
 <details><summary>Technical annotation record</summary><pre>{_e(json.dumps(row, indent=2, ensure_ascii=False))}</pre></details>
</article>'''
        )

    combos = "".join(
        f"<tr><td>{_e(row['scope'])}</td><td>{_e(row['nature'])}</td><td>{_e(row['modality'])}</td>"
        f"<td>{row['count']}</td><td>{_pct(row['rate'])}</td><td>{float(row['mean_min_axis_confidence']):.3f}</td></tr>"
        for row in cross_axis["combination_frequencies"][:20]
    )
    assessments = "".join(
        f"<tr><td>{_e(axis)}</td><td><strong>{_e(result['assessment'])}</strong></td>"
        f"<td>{'<br>'.join(_e(value) for value in result['evidence'])}</td></tr>"
        for axis, result in axis_assessments.items()
    )
    filters = '''<div class="filters">
<label>Dataset <select id="dataset-filter"><option value="all">all</option><option>egoschema</option><option>egosound</option></select></label>
<label>Sample <select id="sample-filter"><option value="all">all</option><option value="representative">representative</option><option value="diversity_stress">diversity stress</option></select></label>
<label>Scope <select id="scope-filter"><option value="all">all</option><option>local</option><option>multi_event</option><option>global</option><option>unclear</option></select></label>
<label>Nature <select id="nature-filter"><option value="all">all</option><option>static</option><option>dynamic</option><option>uncertain</option></select></label>
<label>Modality <select id="modality-filter"><option value="all">all</option><option>visual</option><option>audio</option><option>audio_visual</option><option>indeterminate_from_question</option></select></label>
<label>Flag <select id="flag-filter"><option value="all">all</option><option value="low">low confidence</option><option value="ambiguity">ambiguity</option><option value="failure">taxonomy failure</option><option value="disagreement">stability disagreement</option></select></label>
<strong id="visible-count"></strong></div>'''
    html_text = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Planner Taxonomy Validation v0.1</title>
<style>
:root{{--ink:#17202a;--muted:#637083;--line:#d9e0e8;--paper:#fff;--bg:#f3f6f9;--blue:#235d9f;--amber:#8b5b00}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,Segoe UI,sans-serif}}
header,main{{max-width:1500px;margin:auto}} header{{padding:28px 24px 10px}} h1{{margin:0 0 8px}} h2{{margin-top:30px}}
.notice{{background:#fff8df;border-left:5px solid #d99a00;padding:12px 15px;margin:14px 0}}
.summary-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}} .stat,.panel{{background:white;border:1px solid var(--line);border-radius:9px;padding:14px}}
.stat b{{display:block;font-size:24px;color:var(--blue)}} table{{border-collapse:collapse;width:100%;background:white}} th,td{{border:1px solid var(--line);padding:7px;text-align:left;vertical-align:top}} th{{background:#edf3f8}}
.filters{{position:sticky;top:0;z-index:10;background:#e8f0f8;border:1px solid #b9cadc;padding:10px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}}
.filters select{{max-width:230px}} main{{padding:0 24px 60px}} .card{{background:var(--paper);border:1px solid var(--line);border-radius:10px;margin:18px 0;padding:18px;scroll-margin-top:80px}}
.card-head{{display:flex;justify-content:space-between}} .index{{display:inline-grid;place-items:center;background:var(--blue);color:white;width:28px;height:28px;border-radius:50%;margin-right:8px}}
.pill,.flag{{display:inline-block;border-radius:999px;padding:3px 8px;background:#e8eef5;margin-left:5px}} .sample{{background:#eef5e8}} .flag{{background:#fff0ce;color:#684500;margin:6px 5px 0 0}}
.labels{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:14px 0}} .labels section{{border:1px solid var(--line);border-radius:8px;padding:12px}} .labels h4{{margin:0 0 7px}} .label{{font-size:18px;font-weight:700;color:var(--blue)}} meter{{width:100%}}
.stability{{border-left:4px solid #7b61a8;background:#f4f0fa;padding:10px 12px;margin:12px 0}}
.manual-review{{display:grid;grid-template-columns:repeat(2,minmax(260px,1fr));gap:9px;background:#f6f8fb;border-radius:8px;padding:12px}} .manual-review h4{{grid-column:1/-1;margin:0}} label{{display:flex;justify-content:space-between;gap:8px;align-items:center}} .wide{{grid-column:1/-1;display:block}} textarea{{width:100%;margin-top:5px}} pre{{white-space:pre-wrap;word-break:break-word;background:#111827;color:#e5edf5;padding:12px;max-height:400px;overflow:auto}}
.muted{{color:var(--muted)}} @media(max-width:850px){{.labels,.manual-review{{grid-template-columns:1fr}}.manual-review h4,.wide{{grid-column:auto}}}}
</style></head><body><header id="top"><h1>Planner taxonomy validation v0.1</h1>
<p>Question-only teacher annotations across EgoSchema and EgoSound. This page supports human review; labels are hypotheses, not ground truth.</p>
<div class="notice"><strong>Leakage boundary:</strong> no gold answers, options, correctness, timestamps, video, audio, or retrieved evidence were supplied. Stress-set distributions are not natural prevalence estimates. No routing was implemented.</div>
<div class="summary-grid">
 <div class="stat"><b>{len(annotations)}</b>first-pass questions</div>
 <div class="stat"><b>{stability_metrics['count']}</b>independent stability reannotations</div>
 <div class="stat"><b>{runtime['total_api_calls']}</b>API calls</div>
 <div class="stat"><b>{runtime['total_input_tokens'] + runtime['total_output_tokens']:,}</b>total tokens</div>
 <div class="stat"><b>{cross_axis['unique_combinations']}</b>observed representative tuples</div>
 <div class="stat"><b>{distribution['groups']['combined_representative']['taxonomy_failure_count']}</b>representative taxonomy failures</div>
</div></header><main>
<h2>Axis assessment</h2><table><thead><tr><th>Axis</th><th>Assessment</th><th>Evidence</th></tr></thead><tbody>{assessments}</tbody></table>
<h2>Scope distribution</h2>{_distribution_table(distribution,'scope')}
<h2>Nature distribution</h2>{_distribution_table(distribution,'nature')}
<h2>Modality distribution</h2>{_distribution_table(distribution,'modality')}
<h2>Stability</h2><div class="panel"><p>Scope {_pct(stability_metrics['agreement']['scope'])} · primary nature {_pct(stability_metrics['agreement']['nature_primary'])} · nature ambiguity {_pct(stability_metrics['agreement']['nature_ambiguity'])} · modality {_pct(stability_metrics['agreement']['modality'])} · full tuple {_pct(stability_metrics['agreement']['full_3_axis_tuple'])}</p>
<p>Common confusion pairs: {_e(json.dumps(stability_metrics['confusion_pairs'], ensure_ascii=False))}</p></div>
<h2>Most common representative tuples</h2><table><thead><tr><th>Scope</th><th>Nature</th><th>Modality</th><th>N</th><th>Rate</th><th>Mean min confidence</th></tr></thead><tbody>{combos}</tbody></table>
<h2>Routing actionability (analysis only)</h2><div class="notice">{routing['observed_taxonomy_combinations']} observed tuples imply {routing['distinct_provisional_policy_combinations']} provisional policy combinations. This is not a runtime router and proves no downstream benefit.</div>
<h2 id="questions">Question cards</h2>{filters}<div id="cards">{''.join(cards)}</div>
</main><script>
const fields=['dataset','sample','scope','nature','modality'];
function applyFilters(){{
 let visible=0; const flag=document.getElementById('flag-filter').value;
 document.querySelectorAll('.card').forEach(card=>{{let show=true;
  fields.forEach(f=>{{const v=document.getElementById(f+'-filter').value;if(v!=='all'&&card.dataset[f]!==v)show=false;}});
  if(flag!=='all'&&card.dataset[flag]!=='true')show=false;
  card.style.display=show?'block':'none';if(show)visible++;
 }}); document.getElementById('visible-count').textContent=visible+' / {len(annotations)} visible';
}}
document.querySelectorAll('.filters select').forEach(el=>el.addEventListener('change',applyFilters));
document.querySelectorAll('.manual').forEach(el=>{{const key='taxonomy-review:'+el.dataset.storage;const saved=localStorage.getItem(key);if(saved!==null)el.value=saved;el.addEventListener('change',()=>localStorage.setItem(key,el.value));}});
applyFilters();
</script></body></html>'''
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
