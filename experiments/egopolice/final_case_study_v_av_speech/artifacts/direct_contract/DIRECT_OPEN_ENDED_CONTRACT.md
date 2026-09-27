# Direct-v1.2 open-ended interface contract for 226

Status: `PROMPT_INTERFACE_FROZEN_NO_EXECUTION`

## Canonical identity

This adapter is derived from the final Eval300 Direct-v1.2 runtime—the Direct/C shared navigation method underlying the later A/B/D comparisons—not from a pilot, GenS, smoke prompt, or diagnostic implementation.

Authoritative freeze chain:

- Formal namespace: `source://school-project/msc_thesis/main_system/isolated_workspaces/hourvideo_direct_api_eval300_v1/outputs/direct_v1_formal/direct_v1_2_3x16_r1_r3_eval300_formal_v1`
- Launch lock: `formal_launch_lock_v3.json` (`ebd2014fcd952779444064510d39147149c174bc1ff2bc7fa231dfd589c09d5c`)
- Final candidate manifest: `formal_manifest_final_candidate_no_api_v3.json` (`c73aeaf41d5bf284ade54198525d8396bf3a9fa5e45587aedd49dd3744dc557d`)
- Formal runner: `scripts/execute_direct_v1_2_3x16_r1_r3_eval300_formal_v1.py`
- Prompt: `src/direct_api_v1/prompt.py`
- Provider/tool construction: `src/direct_api_v1/anthropic_provider.py`
- Actions: `src/direct_api_v1/actions.py`
- Controller: `src/direct_api_v1/controller.py`
- Policy: `src/direct_api_v1/policy.py`
- State/history: `src/direct_api_v1/state.py`
- Input loader: `src/direct_api_v1/maps.py`
- Frame resolver: `src/direct_api_v1/frame_resolver.py`
- Provider config: `config/direct_v1_anthropic_smoke.json`; despite the historical filename/schema label, its SHA is explicitly frozen by the final formal manifest.

Every current source SHA named by the formal manifest was recomputed and matched exactly. The canonical runtime system-prompt text SHA is `6e938f9fc09af7d3ce94d75ee7c6d1588b053aeb79f6b0855137b581e1b3a22d`; the prompt source-file SHA is `6b574c6d8b5e549ea72138404decb2bd437ab646d2a36904a0128e6e9e711330`.

## Frozen unchanged navigation policy

- Question-conditioned, agent-driven navigation over the verbatim supplied map.
- The map is the primary semantic/navigation guide; the controller performs no ranking or deterministic retrieval.
- Each accepted `inspect_frames` action requests 1–3 timestamps. The provider schema dynamically uses `maxItems=min(3, remaining budget)` and omits the inspect tool when the remaining budget is zero.
- The cumulative per-question ceiling is 16 unique physical frames. It is a hard ceiling, not a target; zero or fewer than 16 frames are valid and there is no padding.
- Requested timestamps resolve to the nearest available original 1-FPS frame within the video duration. An equal-distance floor/ceil tie chooses the earlier index.
- Physical identity is the resolved `frame_path`. Duplicate paths within a turn or seen earlier in the same session are not retransmitted and do not consume new budget.
- If all newly resolved unique paths would take the route beyond 16, the complete action is rejected before transmission.
- Newly accepted frames are presented chronologically by `(resolved_timestamp_sec, requested_timestamp_sec, frame_path)`, independent of request order.
- The same provider agent retains the entire message/tool-use/tool-result history. After an inspection only new physical frames are sent, followed by reassessment.
- Model/config stay `claude-haiku-4-5-20251001`, temperature `0`, max output `512`, timeout `120s`, 32 route turns, one provider transport retry, one structural correction, native tool use with `tool_choice={"type":"any"}`, and the same cached verbatim-map system block.

The frozen `inspect_frames_schema.json` is the actual canonical full-budget tool definition. Its only runtime variation is the canonical reduction of `maxItems` to 2 or 1 as budget falls; at zero it is omitted. No Planner, Medium Top-K, Fine reranker, SigLIP query ranking, deterministic retrieval, or new temporal-diversity heuristic is introduced.

## Minimal open-ended interface delta

Only MCQ-bound interface assumptions are replaced:

1. The initial user message carries the exact question text and budget state, with no options block.
2. System-prompt references to options, option elimination, answer choice, and A–E selection become equivalent open-ended answer wording.
3. The final-answer tool changes from `{selected_option_id: enum[A..E], reason}` to `{answer: non-empty string, reason: non-empty short string}`.
4. The input loader must retain the existing no-gold safety checks and require question identity/text, but not five A–E options.
5. Final-answer parsing/controller storage must accept the non-empty answer string; MCQ-only invalid-option validation and correction wording become open-ended equivalents.
6. For questions explicitly asking what is visible or visibly observable, the prompt requires confirmation in inspected original frames rather than treating map or speech-derived map content as visual confirmation. This applies symmetrically to both map conditions.

These are interface changes, not navigation-policy changes. The exact system-prompt changes are frozen in `canonical_to_open_ended_diff.txt`.

## V/AV symmetry

Both conditions use the exact same open-ended prompt, schemas, controller policy, provider/model configuration, frame resolver, deduplication, history mechanism, retry policy, and budgets:

- V: exact question + frozen V-v2.2 map.
- AV: exact same question + frozen AV-Speech-v2.2 map.

The only upstream difference is which frozen map is supplied. No routes were executed and no API, model, or GPU was used during this freeze.
