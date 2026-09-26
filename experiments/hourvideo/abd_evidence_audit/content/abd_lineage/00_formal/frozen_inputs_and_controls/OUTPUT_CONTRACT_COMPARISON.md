# Output contract comparison

| Item | Direct v1.2 final answer | GenS v3 direct-parser-aligned / current offline A-B-D shared parser |
|---|---|---|
| External action | exactly one `final_answer` tool call | exactly one `final_answer` tool call |
| Fields | `selected_option_id`, `reason` | same |
| Provider schema | option enum A–E; nonempty reason; advertised max 240; no additional properties | byte-for-byte same schema |
| Content blocks | exactly one tool block; text/other blocks rejected | same |
| Tool-use id | nonempty string required | same |
| Runtime input normalization | mapping or HTTP-fallback `SimpleNamespace` | same |
| Runtime option handling | `str(value)`, then controller membership in the five actual option ids | `str(value)`, then A–E membership |
| Runtime reason handling | `str(value)`, reject blank/whitespace | same |
| 240-character enforcement | **not executed locally**; no truncation | **not executed locally**; no truncation |
| Extra tool-input keys | provider schema asks provider not to emit them; Direct runtime ignores them | same |
| Invalid structure | Direct permits one structural correction turn | GenS v3 and A/B/D single-turn contract: invalid result, no correction |
| Scoring | one A–E prediction or missing/failure | same |
| Accuracy denominator | all 300 questions; missing/failure is incorrect | same |

“Output consistency” therefore means the same response shape, effective parser acceptance, and 300-question scoring denominator. It does not mean identical generated answers or identical reasons.
