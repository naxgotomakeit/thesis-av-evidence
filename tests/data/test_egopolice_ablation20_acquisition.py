from __future__ import annotations

from pathlib import Path

from scripts.data.download_egopolice_ablation20 import _blocker, _resolved_rows


ROOT = Path(__file__).resolve().parents[2]


def test_download_targets_are_exactly_the_frozen_ablation20():
    rows = _resolved_rows(
        ROOT / "config/data/egopolice_ablation20_v1.json",
        ROOT / "config/data/egopolice_50videos.json",
    )

    assert len(rows) == 20
    assert len({row["video_id"] for row in rows}) == 20
    assert all(row["source_url"].startswith("https://") for row in rows)
    assert all(row["relative_video_path"].startswith("videos/") for row in rows)


def test_downloader_classifies_auth_and_rate_limit_stop_conditions():
    assert _blocker("ERROR: HTTP Error 429: Too Many Requests") == "vimeo_rate_limited_http_429"
    assert _blocker("HTTP Error 401; OAuth token required") == (
        "vimeo_authentication_required_http_401_or_oauth"
    )
    assert _blocker("unrelated network failure") is None
