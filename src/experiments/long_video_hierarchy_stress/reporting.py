"""Human-first nested hierarchy review rendering."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _asset_path(frame_path: str, *, root: Path, output_dir: Path) -> str:
    path = Path(frame_path)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise FileNotFoundError(f"Missing hierarchy review image: {path}")
    return path.relative_to(output_dir).as_posix()


def _frames(node: dict[str, Any], *, root: Path, output_dir: Path, limit: int = 4) -> str:
    seen: set[str] = set()
    cards = []
    for item in node.get("representative_frames", []):
        source = str(item["frame_path"])
        if source in seen:
            continue
        seen.add(source)
        relative = _asset_path(source, root=root, output_dir=output_dir)
        cards.append(
            f'<figure><img src="{_e(relative)}" loading="lazy"><figcaption>'
            f'{float(item["timestamp"]):.1f}s · {_e(item["kind"])}</figcaption></figure>'
        )
        if len(cards) >= limit:
            break
    return '<div class="frames">' + "".join(cards) + "</div>"


def _review_fields(prefix: str) -> str:
    groups = []
    for label, name in (
        ("Meaningful grouping", "meaningful"),
        ("Overmerged", "overmerged"),
        ("Fragmented", "fragmented"),
    ):
        radios = " ".join(
            f'<label><input type="radio" name="{_e(prefix)}_{name}" value="{value}"> {value}</label>'
            for value in ("yes", "no", "unsure")
        )
        groups.append(f"<div><b>{label}:</b> {radios}</div>")
    groups.append(f'<label><b>Notes:</b><textarea name="{_e(prefix)}_notes"></textarea></label>')
    return '<div class="review">' + "".join(groups) + "</div>"


def render_hierarchy_review(
    *, output_path: Path, root: Path, manifest: dict[str, Any],
    hierarchies: list[dict[str, Any]], metrics: list[dict[str, Any]],
    inventory: dict[str, Any],
) -> dict[str, Any]:
    by_metric = {row["video_id"]: row for row in metrics}
    hierarchy_by_video = {row["video_id"]: row for row in hierarchies}
    sections = []
    image_count = 0
    for manifest_row in manifest["videos"]:
        video_id = str(manifest_row["video_id"])
        hierarchy = hierarchy_by_video[video_id]
        metric = by_metric[video_id]
        nodes = {row["node_id"]: row for row in hierarchy["nodes"]}
        medium_ids = hierarchy["cuts"]["medium"]["node_ids"]
        coarse_ids = hierarchy["cuts"]["coarse"]["node_ids"]
        medium_by_coarse: dict[str, list[str]] = {item: [] for item in coarse_ids}
        for medium_id in medium_ids:
            medium_leaves = set(nodes[medium_id]["leaf_ids"])
            matches = [cid for cid in coarse_ids if medium_leaves <= set(nodes[cid]["leaf_ids"])]
            if len(matches) != 1:
                raise ValueError(f"Invalid Medium→Coarse membership for {video_id}/{medium_id}: {matches}")
            medium_by_coarse[matches[0]].append(medium_id)
        coarse_html = []
        for coarse_id in coarse_ids:
            coarse = nodes[coarse_id]
            image_count += len({item["frame_path"] for item in coarse["representative_frames"]})
            medium_html = []
            for medium_id in medium_by_coarse[coarse_id]:
                medium = nodes[medium_id]
                image_count += len({item["frame_path"] for item in medium["representative_frames"]})
                fine_html = []
                for fine_id in medium["leaf_ids"]:
                    fine = nodes[fine_id]
                    image_count += len({item["frame_path"] for item in fine["representative_frames"]})
                    fine_html.append(
                        f'<article class="fine"><h5>FINE {_e(fine_id)} '
                        f'[{fine["start"]:.1f}–{fine["end"]:.1f}s] · {fine["duration"]:.1f}s</h5>'
                        f'{_frames(fine, root=root, output_dir=output_path.parent, limit=4)}</article>'
                    )
                medium_html.append(
                    f'<details class="medium" open><summary>MEDIUM {_e(medium_id)} '
                    f'[{medium["start"]:.1f}–{medium["end"]:.1f}s] · {medium["duration"]:.1f}s '
                    f'· Fine children: {_e(", ".join(medium["leaf_ids"]))}</summary>'
                    f'{_frames(medium, root=root, output_dir=output_path.parent, limit=4)}'
                    f'{_review_fields(video_id + "_" + medium_id)}'
                    f'<div class="fine-grid">{"".join(fine_html)}</div></details>'
                )
            coarse_html.append(
                f'<details class="coarse" open><summary>COARSE {_e(coarse_id)} '
                f'[{coarse["start"]:.1f}–{coarse["end"]:.1f}s] · {coarse["duration"]:.1f}s</summary>'
                f'{_frames(coarse, root=root, output_dir=output_path.parent, limit=4)}'
                f'{_review_fields(video_id + "_" + coarse_id)}'
                f'<div class="medium-stack">{"".join(medium_html)}</div></details>'
            )
        sections.append(
            f'<section id="video-{_e(video_id)}"><h2>{_e(video_id)}</h2>'
            f'<p class="meta">Duration {manifest_row["duration_sec"]:.1f}s · '
            f'Fine {metric["fine"]["node_count"]} · Medium {metric["medium"]["node_count"]} · '
            f'Coarse {metric["coarse"]["node_count"]} · hierarchy depth {metric["hierarchy_depth"]}</p>'
            f'<p class="warning">This is a held-out 180s generalization check, not a true 5–20 minute stress test.</p>'
            f'{"".join(coarse_html)}</section>'
        )
    links = " ".join(
        f'<a href="#video-{_e(row["video_id"])}">{_e(row["video_id"])}</a>' for row in manifest["videos"]
    )
    document = f'''<!doctype html><html><head><meta charset="utf-8"><title>Long-video hierarchy stress v0.1</title>
<style>
body{{font:14px/1.45 system-ui;margin:0;background:#f3f5f8;color:#1d2733}}header{{padding:22px;background:#182433;color:white;position:sticky;top:0;z-index:3}}nav a{{color:#aee1ff;margin-right:12px}}main{{max-width:1500px;margin:auto;padding:18px}}section{{background:white;border:1px solid #ccd5df;border-radius:12px;padding:18px;margin:18px 0}}summary{{cursor:pointer;font-weight:750}}.coarse{{border:3px solid #536dfe;border-radius:10px;padding:12px;margin:14px 0}}.medium{{border:2px solid #26a69a;border-radius:8px;padding:10px;margin:12px 0 12px 28px}}.fine{{border-left:4px solid #ffa726;background:#fff8ec;padding:8px;margin:8px}}.fine-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:8px}}.frames{{display:flex;gap:8px;overflow-x:auto;padding:8px 0}}figure{{margin:0;min-width:150px}}img{{width:150px;height:105px;object-fit:cover;border-radius:6px;border:1px solid #aaa}}figcaption{{font-size:12px}}.review{{background:#f7f8fb;border:1px dashed #abb5c2;padding:9px;margin:8px 0}}textarea{{display:block;width:98%;height:38px}}.warning{{color:#9a4e00;font-weight:700}}.meta{{font-size:16px}}
</style></head><body><header><h1>Hierarchy stress review</h1><p>Fine → Boundary-aware Safe Merge → Medium / Coarse reference cuts</p><p>{_e(inventory["limitation"])}</p><nav>{links}</nav></header><main>{''.join(sections)}</main></body></html>'''
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    return {
        "video_count": len(sections),
        "image_reference_count": document.count("<img "),
        "missing_image_count": 0,
    }
