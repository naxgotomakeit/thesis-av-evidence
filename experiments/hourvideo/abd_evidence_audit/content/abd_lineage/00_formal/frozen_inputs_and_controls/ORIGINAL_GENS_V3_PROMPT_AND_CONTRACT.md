# Original GenS v3 direct-parser-aligned contract (verbatim export)

This file describes the already-run formal GenS v3 contract. It does not contain the proposed map modification.

## System prompt — verbatim

```text
You are a long-video evidence-reasoning agent.

Answer one multiple-choice question about a long video.
You receive the question, options A-E, and provided original
video frames. The provided video frames are presented in
chronological order.

The goal is to select the option best supported by the
available evidence, not exhaustive verification or proving
every alternative false. If the provided video frames do not
fully resolve the question, answer from the best available
evidence.

Return exactly one structured external action. For
final_answer, select exactly one of A, B, C, D, E and give
one short sentence stating only the basis. Do not provide
a verbose reasoning trace.
```

The source file has one terminal newline; the request builder removes only that terminal newline with `rstrip("\n")`.

## User question/options template — exact expansion

```text
Question: {question}
Options:
A. {options[A]}
B. {options[B]}
C. {options[C]}
D. {options[D]}
E. {options[E]}
Return the final_answer action.
```

No timestamp label or selector metadata is sent. In the completed formal GenS run, the user content is: 1–16 verbatim original JPEG base64 image blocks, globally chronological, followed by exactly one text block using the template above. The new A/B/D draft separately allows the two required zero-image rows; that is not being attributed to the old GenS run.

## Tool schema — verbatim JSON

```json
{
  "name": "final_answer",
  "description": "Finish with exactly one answer option.",
  "input_schema": {
    "type": "object",
    "additionalProperties": false,
    "properties": {
      "selected_option_id": {
        "type": "string",
        "enum": [
          "A",
          "B",
          "C",
          "D",
          "E"
        ]
      },
      "reason": {
        "type": "string",
        "minLength": 1,
        "maxLength": 240
      }
    },
    "required": [
      "selected_option_id",
      "reason"
    ]
  }
}
```

## Model and transport settings

```json
{
  "provider": "anthropic",
  "model": "claude-haiku-4-5-20251001",
  "max_output_tokens": 512,
  "temperature": 0.0,
  "timeout_sec": 120.0,
  "tool_choice": {
    "type": "any"
  },
  "single_turn": true
}
```

- Cache: `disabled_no_stable_large_prefix_single_turn_question_specific_request`; no `cache_control` field. Formal cache creation/read tokens were both zero.
- Retry: `max_retries=1`, hence at most 2 transport attempts. A received response is not repaired or retried for parser invalidity.
- Output limit: 512 tokens.

## Actual parser source — verbatim

```python
def parse_structured_response(content: Any) -> dict[str, Any]:
    """Apply Direct's actual final-answer parsing; never infer from free text.

    The provider tool schema retains its ``maxLength=240`` guidance, but the
    frozen Direct-v1.2 Python path does not revalidate that upper bound.  It
    converts option/reason values with ``str()``, rejects a blank/whitespace
    reason, ignores additional tool-input keys, and lets the controller enforce
    the A--E option set.  GenS v3 mirrors those effective semantics here while
    retaining its single-turn/no-correction protocol.
    """
    blocks = list(content or [])
    tool_blocks = [block for block in blocks if _field(block, "type") == "tool_use"]
    non_tool_blocks = [block for block in blocks if _field(block, "type") != "tool_use"]
    text = "".join(str(_field(block, "text", "")) for block in non_tool_blocks if _field(block, "type") == "text")
    base = {
        "tool_use_count": len(tool_blocks),
        "content_block_types": [_field(block, "type") for block in blocks],
        "accompanying_text": text,
        "tool_name": None,
        "tool_use_id": None,
        "tool_input_runtime_type": None,
        "normalised_tool_input": None,
        "prediction": None,
        "reason": None,
    }
    if not tool_blocks:
        return {**base, "result_class": "invalid_format", "failure_category": "missing_tool_call"}
    if len(tool_blocks) != 1:
        return {**base, "result_class": "invalid_format", "failure_category": "multiple_tool_calls"}
    if non_tool_blocks:
        return {**base, "result_class": "invalid_format", "failure_category": "tool_plus_text_or_other_content"}
    block = tool_blocks[0]
    name = _field(block, "name")
    identifier = _field(block, "id")
    native_input = _field(block, "input")
    normalized = normalise_tool_input(native_input)
    base.update({
        "tool_name": name,
        "tool_use_id": identifier,
        "tool_input_runtime_type": type(native_input).__name__,
        "normalised_tool_input": normalized,
    })
    if name != "final_answer":
        return {**base, "result_class": "invalid_format", "failure_category": "wrong_tool_name"}
    if not isinstance(identifier, str) or not identifier:
        return {**base, "result_class": "invalid_format", "failure_category": "missing_tool_use_id"}
    if normalized is None:
        return {**base, "result_class": "invalid_format", "failure_category": "invalid_tool_input_type"}
    option = str(normalized.get("selected_option_id", ""))
    reason = str(normalized.get("reason", ""))
    base["reason"] = reason
    # Direct's FinalAnswerAction validates the reason before the controller
    # validates the option, so preserve that order for mixed-invalid inputs.
    if not reason.strip():
        return {**base, "result_class": "invalid_format", "failure_category": "invalid_reason"}
    if option not in CHOICES:
        return {**base, "result_class": "invalid_format", "failure_category": "invalid_option"}
    return {**base, "prediction": option, "result_class": "valid_answer", "failure_category": None}
```

The provider schema contains `reason.maxLength=240`, but the actual Python parser does not enforce that upper bound and does not truncate. It converts the option and reason with `str()`, requires a nonblank reason, ignores extra input keys, and then requires A–E.

## Source SHA anchors

```json
{
  "formal_config": "bc647345593fb23eebaac87201fb064bb72735f6a7fcc8104fc16023ed31a287",
  "formal_preflight_manifest": "a2eae93163df6b4ccb78e9ddd236d8ccf6dc1ba0c918468564f970b65a240c93",
  "system_prompt_file": "ea841c205ea7d88cf40bae05d2b88cdecea2235047845ff08a80ca5afb8721b8",
  "structured_v3_runtime": "f92cb727556038ef018d9af6cbdae95f5bc255484c370e57e3657fb870be127f",
  "provider_transport_base": "b59aa200f4a2a9d222375898f0b0c69b1471016b1bfe5f4bc3b174af25da1617",
  "formal_runner": "f55a0626f357dfd25a261560f72f5c67ece97d7939c22fe1e06d9bd96f1f510e",
  "formal_scorer": "529358bb2b6e2918324f9ece235a33c7ac276bc2b074841ea5175a8b3a1615fb",
  "direct_final_manifest": "c73aeaf41d5bf284ade54198525d8396bf3a9fa5e45587aedd49dd3744dc557d",
  "direct_config": "581f7bb119660096820011bea6c03970c6af3391f3a2a3bff428377ff77f4c8a"
}
```
