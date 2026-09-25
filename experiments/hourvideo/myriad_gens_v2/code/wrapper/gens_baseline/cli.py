from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_profile, resolve_profile_path, verify_frozen_artifacts
from .gens_stage import validate_qwen_interface
from .pipeline import build_access_policy, run_one_question
from .query import load_selector


def command_validate(args: argparse.Namespace) -> int:
    profile = load_profile(args.profile)
    frozen = verify_frozen_artifacts(profile, full_manifests=args.full_hash)
    policy, _, _ = build_access_policy(profile)
    selector = load_selector(profile, policy)
    interface = validate_qwen_interface(
        resolve_profile_path(profile, profile["gens"]["local_path"])
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "profile": profile["profile"],
                "profile_sha256": profile["_profile_sha256"],
                "selector_count": len(selector),
                "frozen_artifacts": frozen,
                "qwen_interface": interface,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def command_smoke(args: argparse.Namespace) -> int:
    profile = load_profile(args.profile)
    verify_frozen_artifacts(profile, full_manifests=True)
    result = run_one_question(profile, qa_uid=args.qa_uid, device=args.device)
    print(
        json.dumps(
            {
                "qa_uid": result["qa_uid"],
                "status": result["status"],
                "actual_frame_count": result["actual_frame_count"],
                "error": result["error"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "ok" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frozen GenS-Hybrid-cap16 wrapper")
    parser.add_argument("--profile", required=True, type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="CPU-only local contract validation")
    validate.add_argument("--full-hash", action="store_true")
    validate.set_defaults(function=command_validate)
    smoke = subparsers.add_parser("smoke", help="exactly one public Eval300 question")
    smoke.add_argument("--qa-uid", required=True)
    smoke.add_argument("--device", default="cuda:0")
    smoke.set_defaults(function=command_smoke)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
