# Format-only final-answer recovery summary

Status: `FORMAT_ONLY_RECOVERY_DIAGNOSTIC`; recovery PASS: `4/4`.

Primary Direct open-ended v1 statistics remain permanently unchanged: 12 attempted, 8 protocol-complete, 4 format-failed.

Each recovery used the exact frozen conversation and evidence, offered only `final_answer`, made one provider request, and added zero inspections and zero frames.

| Route | Primary status | Recovery | Response ID | Reason chars |
|---|---|---|---|---:|
| q_global_summary__V | FAIL_REASON_LENGTH | PASS | msg_011Cf3uMs2J1YMabcdWwj85k | 243 |
| q_global_summary__AV_SPEECH | FAIL_REASON_LENGTH | PASS | msg_011Cf3uN2Y3humSkKG5i67DM | 205 |
| q_medical_assistance__AV_SPEECH | FAIL_REASON_LENGTH | PASS | msg_011Cf3uNBS5Y5QCJr3XyHBki | 304 |
| q_handcuff_before_medical__V | FAIL_REASON_LENGTH | PASS | msg_011Cf3uNQ8ZgSR9tdS2mdMMQ | 253 |

Recovered answers are diagnostic and do not replace primary route outcomes. No gold/correctness was read and no accuracy was computed.
