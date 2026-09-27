# Final 226 case-study qualitative audit

## Scope and integrity

This is a post-hoc, no-gold qualitative audit of the frozen Direct open-ended v1 routes. No model/API/GPU was used; no route, map, answer, timestamp, or frame selection was changed. The bundle, primary route, recovery, V map, AV map, and matched-pair SHA chains were revalidated. The primary result remains **8/12 protocol-complete** with four reason-length-only failures; format-only recovery remains a separate **4/4 diagnostic** with zero new inspections and zero new frames.

All 59 route-frame presentations (25 distinct physical frames) were opened and visually reviewed. Each source frame SHA matched the frozen route provenance.

## Headline findings

- **Global summary:** V navigation is poor (three late frames, one unusable); AV navigation is poor because it inspected no frames. The AV recovered answer is map-only and includes known-limit-risk details.
- **Weapon visible:** both routes are excellent and identical. The gun is clear at 547s and 562s; 577s is heavily occluded. This is the clean control.
- **Visible injury:** V is good and AV excellent. AV's 637s frame is materially clearer than V's 652s, while 727s decisively supports both. AV nevertheless overclaims injury source from speech-derived map context.
- **Medical assistance:** V targets the right transition but the stills do not cleanly distinguish checking from restraint. AV changes navigation substantially, yet no tourniquet is visible at 720s and later leg holding remains ambiguous. Both answers overclaim.
- **Handcuffing:** both routes scan broad restraint periods but never secure a clean wrist view. Their sampled observation (no identifiable cuff) is weaker than their whole-video negative wording.
- **Handcuff-before-medical:** AV uses six frames versus V's twelve, but neither resolves both temporal predicates. V's recovered negative medical claim conflicts with V's medical-assistance answer.

## Navigation delta

Meaningful changes are: (1) AV's 637s substitution for visible injury, which improves evidence clarity; (2) AV's much broader medical-assistance navigation, which is map-influenced but does not produce clearer treatment evidence; and (3) AV's concentrated six-frame temporal route versus V's twelve-frame search. Weapon navigation is exactly unchanged; handcuffing differs only by 765s versus 900s and gains no cuff visibility.

## Answer support and limitations

Overclaim is present in AV global summary, AV visible injury, both medical-assistance answers, both handcuffing answers, and both handcuff-before-medical answers. The most concrete cases are the unobserved 720s tourniquet, speech used as visual injury-source confirmation, whole-video no-handcuff claims from sparse/occluded samples, and medical assistance asserted at an occluded 585s frame.

Known v2.2 limitations are preserved, not repaired: chest/groin, tourniquet, fence temporal misattributions and knife-to-injury semantic overinterpretation. No payload corruption, V/AV visual mismatch, or ASR-alignment implementation error was found in this audit.

## Recommended paper cases

1. **q_weapon_visible** — control; identical navigation and clear evidence at 547s/562s.
2. **q_visible_injury** — a modest speech-map navigation gain (637s versus 652s) alongside a transparent semantic-overreach limitation; use 637s, 652s, and common 727s.
3. **q_handcuff_before_medical** — temporal-reasoning/failure-mode case; use common 600s/675s/720s and V-only 1200s to show evidence ambiguity and search expansion.

Use q_medical_assistance as an appendix limitation case rather than a headline success because its navigation delta is large but the claimed tourniquet is not visually present.
