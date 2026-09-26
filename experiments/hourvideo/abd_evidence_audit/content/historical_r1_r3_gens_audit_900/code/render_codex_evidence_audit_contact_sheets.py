#!/usr/bin/env python3
"""Render review-only contact sheets from exactly the frames in frozen packages."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit/codex_evidence_audit_r1_r3_gens_v3_v1"
CELL_W, CELL_H, COLS = 220, 155, 5


def main() -> None:
    batches = json.loads((AUDIT / "batch_manifest.json").read_text(encoding="utf-8"))["batches"]
    out = AUDIT / "contact_sheets"
    out.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    for batch in batches:
        target = out / f"{batch['batch_id']}.jpg"
        if target.exists():
            continue
        rows = []
        for neutral_id in batch["neutral_ids"]:
            package = json.loads((AUDIT / "review_packages" / f"{neutral_id}.json").read_text(encoding="utf-8"))
            frames = package["images"]
            row_count = max(1, (len(frames) + COLS - 1) // COLS)
            height = 28 + row_count * CELL_H
            panel = Image.new("RGB", (COLS * CELL_W, height), "white")
            draw = ImageDraw.Draw(panel)
            draw.text((4, 4), f"{neutral_id}  images={len(frames)}", fill="black", font=font)
            for index, frame in enumerate(frames):
                image = Image.open(frame["frame_path"]).convert("RGB")
                image.thumbnail((CELL_W - 8, CELL_H - 28))
                x = (index % COLS) * CELL_W + 4
                y = 28 + (index // COLS) * CELL_H
                panel.paste(image, (x, y + 16))
                ts = frame.get("resolved_timestamp_sec")
                draw.text((x, y), f"#{index + 1} t={ts}s", fill="black", font=font)
            rows.append(panel)
        sheet = Image.new("RGB", (COLS * CELL_W, sum(row.height for row in rows)), "white")
        y = 0
        for row in rows:
            sheet.paste(row, (0, y)); y += row.height
        sheet.save(target, quality=92)
    print(json.dumps({"status": "PASS", "sheets": len(batches), "source": "frozen review package frame paths"}))


if __name__ == "__main__":
    main()
