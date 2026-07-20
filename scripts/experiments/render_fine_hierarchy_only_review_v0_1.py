"""Render the saved Fine/Medium/Coarse hierarchy without retrieval content."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/experiments/fine_to_coarse_hierarchy_v0_1/safe_merge_hierarchies.jsonl"
OUTPUT = ROOT / "outputs/experiments/fine_only_vs_hierarchy_guided_v0_1/hierarchy_review.html"


def e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def image_src(path: str) -> str:
    value = path.replace("\\", "/")
    if not value.startswith("outputs/"):
        raise ValueError(f"Unexpected frame provenance: {value}")
    return "../../" + value[len("outputs/"):]


def thumbnails(node: dict[str, Any], *, maximum: int = 4) -> str:
    rows = node.get("representative_frames", [])[:maximum]
    return '<div class="thumbs">' + "".join(
        f'<figure><img loading="lazy" src="{e(image_src(row["frame_path"]))}">'
        f'<figcaption>{e(row["kind"])} · {float(row["timestamp"]):.1f}s</figcaption></figure>'
        for row in rows
    ) + "</div>"


def review_fields(key: str) -> str:
    def select(name: str) -> str:
        return (
            f'<select class="manual" data-key="{e(key+":"+name)}">'
            '<option>Unreviewed</option><option>Yes</option><option>No</option><option>Unclear</option></select>'
        )
    return (
        '<div class="review"><b>Human review — intentionally blank</b>'
        f'<label>Grouping meaningful? {select("meaningful")}</label>'
        f'<label>Over-merged? {select("overmerged")}</label>'
        f'<label>Too fragmented? {select("fragmented")}</label>'
        f'<label class="wide">Notes<textarea class="manual" data-key="{e(key+":notes")}" rows="2"></textarea></label>'
        '</div>'
    )


def timeline(label: str, nodes: list[dict[str, Any]], duration: float, css: str) -> str:
    bars = []
    for index, node in enumerate(nodes, start=1):
        start, end = float(node["start"]), float(node["end"])
        bars.append(
            f'<span class="bar {css}" style="left:{100*start/duration:.6f}%;width:{max(.15,100*(end-start)/duration):.6f}%" '
            f'title="{e(node["node_id"])} | {start:.1f}–{end:.1f}s">{index}</span>'
        )
    return (
        f'<div class="timeline-row"><div><b>{e(label)}</b> ({len(nodes)})</div>'
        '<div class="axis"><i class="zero">0s</i><i class="sixty">60</i><i class="one-twenty">120</i><i class="end">180s</i>'
        + "".join(bars) + '</div></div>'
    )


def node_title(node: dict[str, Any]) -> str:
    return (
        f'<span class="node-id">{e(node["node_id"])}</span> '
        f'<span>{float(node["start"]):.1f}–{float(node["end"]):.1f}s</span> '
        f'<span class="duration">{float(node["duration"]):.1f}s</span>'
    )


def main() -> None:
    hierarchies = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line]
    if len(hierarchies) != 10:
        raise RuntimeError(f"Expected 10 frozen hierarchies, found {len(hierarchies)}")
    cards = []
    expected = {"fine": 0, "medium": 0, "coarse": 0}
    rendered = {"fine": 0, "medium": 0, "coarse": 0}
    expected_images: list[Path] = []
    for video_index, hierarchy in enumerate(hierarchies, start=1):
        if not hierarchy["invariants"]["valid"] or hierarchy["invariants"]["fine_leaf_preservation_rate"] != 1.0:
            raise RuntimeError(f"Invalid frozen hierarchy: {hierarchy['video_id']}")
        nodes = {row["node_id"]: row for row in hierarchy["nodes"]}
        fine_ids = hierarchy["cuts"]["fine"]["node_ids"]
        medium_ids = hierarchy["cuts"]["medium"]["node_ids"]
        coarse_ids = hierarchy["cuts"]["coarse"]["node_ids"]
        fine = [nodes[identifier] for identifier in fine_ids]
        medium = [nodes[identifier] for identifier in medium_ids]
        coarse = [nodes[identifier] for identifier in coarse_ids]
        expected["fine"] += len(fine_ids); expected["medium"] += len(medium_ids); expected["coarse"] += len(coarse_ids)
        key = hierarchy["video_id"].replace("-", "_")
        coarse_sections = []
        assigned_medium: set[str] = set()
        assigned_fine: set[str] = set()
        for coarse_node in coarse:
            coarse_leaves = set(coarse_node["leaf_ids"])
            medium_children = [node for node in medium if set(node["leaf_ids"]).issubset(coarse_leaves)]
            assigned_medium.update(node["node_id"] for node in medium_children)
            medium_sections = []
            for medium_node in medium_children:
                fine_children = [nodes[identifier] for identifier in medium_node["leaf_ids"]]
                assigned_fine.update(node["node_id"] for node in fine_children)
                fine_sections = []
                for fine_node in fine_children:
                    rendered["fine"] += 1
                    expected_images.extend(ROOT / row["frame_path"] for row in fine_node.get("representative_frames", [])[:4])
                    fine_sections.append(
                        f'<div class="fine-card"><div class="fine-title">└─ FINE · {node_title(fine_node)}</div>'
                        f'{thumbnails(fine_node, maximum=4)}</div>'
                    )
                rendered["medium"] += 1
                expected_images.extend(ROOT / row["frame_path"] for row in medium_node.get("representative_frames", [])[:4])
                medium_sections.append(
                    f'<details class="medium-card" open><summary>├─ MEDIUM · {node_title(medium_node)} · Fine children: {e(", ".join(medium_node["leaf_ids"]))}</summary>'
                    f'{thumbnails(medium_node, maximum=4)}{review_fields(key+":"+medium_node["node_id"])}'
                    f'<div class="fine-list">{"".join(fine_sections)}</div></details>'
                )
            rendered["coarse"] += 1
            expected_images.extend(ROOT / row["frame_path"] for row in coarse_node.get("representative_frames", [])[:4])
            coarse_sections.append(
                f'<details class="coarse-card" open><summary>COARSE · {node_title(coarse_node)} · Medium children: {len(medium_children)}</summary>'
                f'{thumbnails(coarse_node, maximum=4)}{review_fields(key+":"+coarse_node["node_id"])}'
                f'<div class="medium-list">{"".join(medium_sections)}</div></details>'
            )
        if assigned_medium != set(medium_ids) or assigned_fine != set(fine_ids):
            raise RuntimeError(f"Reference-cut containment incomplete for {hierarchy['video_id']}")
        cards.append(
            f'<article id="video-{key}"><div class="video-head"><div><span class="number">{video_index}</span>'
            f'<b>{e(hierarchy["video_id"])}</b></div><div>Duration: <b>{float(hierarchy["video_duration"]):.1f}s</b> · '
            f'Coarse {len(coarse)} · Medium {len(medium)} · Fine {len(fine)}</div><a href="#top">Back to top</a></div>'
            f'<div class="timelines">{timeline("COARSE",coarse,180,"coarse")}{timeline("MEDIUM",medium,180,"medium")}{timeline("FINE",fine,180,"fine")}</div>'
            f'<div class="tree">{"".join(coarse_sections)}</div></article>'
        )
    missing = [path for path in expected_images if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} referenced thumbnails; first: {missing[0]}")
    if rendered != expected:
        raise RuntimeError(f"Rendered node counts differ: expected={expected}, rendered={rendered}")
    links = " ".join(
        f'<a href="#video-{row["video_id"].replace("-","_")}">{index}</a>'
        for index, row in enumerate(hierarchies, start=1)
    )
    document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Fine–Medium–Coarse hierarchy review</title><style>
:root{{--bg:#f3f5f7;--paper:#fff;--ink:#202a34;--line:#ccd6df;--coarse:#315f8f;--medium:#338b75;--fine:#b76e3d}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,Segoe UI,sans-serif}}header,main{{max-width:1600px;margin:auto}}header{{padding:24px}}main{{padding:0 24px 60px}}h1{{margin:0}}.notice{{background:#fff7d9;border-left:5px solid #d69a00;padding:11px;margin:12px 0}}.summary{{display:flex;gap:10px;flex-wrap:wrap}}.stat,article{{background:var(--paper);border:1px solid var(--line);border-radius:10px}}.stat{{padding:10px 16px}}.stat b{{font-size:22px;color:var(--coarse)}}article{{padding:16px;margin:22px 0;scroll-margin-top:10px}}.video-head{{display:grid;grid-template-columns:1fr auto auto;gap:15px;align-items:center}}.number{{display:inline-grid;place-items:center;width:28px;height:28px;background:var(--coarse);color:#fff;border-radius:50%;margin-right:8px}}.timelines{{margin:15px 0;border:1px solid var(--line);padding:20px 10px 8px;overflow-x:auto}}.timeline-row{{display:grid;grid-template-columns:135px minmax(1000px,1fr);align-items:center;margin:8px 0}}.axis{{position:relative;height:31px;border:1px solid #adb9c4;background:repeating-linear-gradient(to right,#eef2f5 0,#eef2f5 calc(33.333% - 1px),#c6d0d9 calc(33.333% - 1px),#c6d0d9 33.333%)}}.axis i{{position:absolute;top:-18px;font-size:10px;font-style:normal}}.axis .zero{{left:0}}.axis .sixty{{left:33.333%}}.axis .one-twenty{{left:66.666%}}.axis .end{{right:0}}.bar{{position:absolute;top:4px;height:22px;color:#fff;border:1px solid rgba(0,0,0,.3);font-size:9px;text-align:center;overflow:hidden}}.bar.coarse{{background:var(--coarse)}}.bar.medium{{background:var(--medium)}}.bar.fine{{background:var(--fine)}}details summary{{cursor:pointer}}.coarse-card{{border:2px solid var(--coarse);border-radius:9px;padding:10px;margin:15px 0;background:#f7fbff}}.coarse-card>summary{{font-size:17px;font-weight:700;color:var(--coarse)}}.medium-list{{margin-left:28px;border-left:3px solid #86b9aa;padding-left:16px}}.medium-card{{border:1px solid var(--medium);border-radius:8px;padding:9px;margin:12px 0;background:#f7fcfa}}.medium-card>summary{{font-size:15px;font-weight:700;color:#246d5c}}.fine-list{{margin-left:30px;border-left:2px solid #d8a27e;padding-left:15px}}.fine-card{{border:1px solid #dfc1ad;background:#fffaf6;border-radius:6px;padding:8px;margin:9px 0}}.fine-title{{font-weight:650;color:#8d4f27}}.node-id{{font-family:ui-monospace,Consolas,monospace}}.duration{{background:#e9eef3;padding:2px 6px;border-radius:999px}}.thumbs{{display:flex;gap:8px;overflow-x:auto;margin:8px 0}}figure{{margin:0;min-width:150px}}img{{width:150px;height:94px;object-fit:cover;border:1px solid #b9c4ce;border-radius:4px}}figcaption{{font-size:10px;color:#5c6976}}.review{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;background:#edf3f7;padding:9px;border-radius:6px;margin:8px 0}}.review>b,.review .wide{{grid-column:1/-1}}label{{display:flex;justify-content:space-between;gap:8px}}textarea{{width:100%}}@media(max-width:850px){{.review{{grid-template-columns:1fr}}.review>b,.review .wide{{grid-column:auto}}.video-head{{grid-template-columns:1fr}}}}</style></head><body><header id="top"><h1>Fine → Medium → Coarse hierarchy-only review</h1><p>Visual inspection of the existing frozen Boundary-aware Safe Merge hierarchy. No questions or search/evaluation content is shown.</p><div class="notice"><b>Interpretation:</b> Medium and Coarse are reference operating cuts, not claimed natural semantic levels. Human review fields are blank and stored only in this browser.</div><div class="summary"><div class="stat"><b>{len(hierarchies)}</b>videos</div><div class="stat"><b>{rendered['coarse']}</b>Coarse nodes</div><div class="stat"><b>{rendered['medium']}</b>Medium nodes</div><div class="stat"><b>{rendered['fine']}</b>Fine leaves</div><div class="stat"><b>{len(expected_images)}</b>thumbnails</div></div><p>Videos: {links}</p></header><main>{''.join(cards)}</main><script>document.querySelectorAll('.manual').forEach(el=>{{const key='hierarchy-only-review:'+el.dataset.key,value=localStorage.getItem(key);if(value!==null)el.value=value;el.addEventListener('change',()=>localStorage.setItem(key,el.value));}});</script></body></html>'''
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(document, encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "videos": len(hierarchies), **rendered, "thumbnails": len(expected_images), "missing": 0}, indent=2))


if __name__ == "__main__":
    main()
