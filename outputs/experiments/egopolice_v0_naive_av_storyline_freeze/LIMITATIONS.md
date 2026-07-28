# V0 known limitations

1. Keyframes are not guaranteed to represent the concrete incident.
2. Captions can omit important information and contain template-like redundancy such as “In this image...”.
3. Speech/VAD/ASR coverage is incomplete and there is no non-speech audio event detection.
4. Repeated Medium→Coarse compression can lose important information.
5. The storyline has no salience weighting, so important events can be diluted by ordinary or repeated content.
6. ASR can contain wrong words, incorrect language identification, or translation hallucinations.
7. Audio and visual evidence are fused mainly through text proxies; the system does not return to the original video for double verification.
8. Global identity tracking across people is unreliable, especially under occlusion, absent faces, and multi-person scenes.

These limitations are part of the frozen V0 baseline and are not fixes proposed in this freeze.
