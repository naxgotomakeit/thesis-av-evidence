# Corrected Qwen boundary contract smoke v0.1

- Boundaries: 10
- Qwen boundary calls: 10
- Deterministic: {'MERGE': 4, 'STOP': 6}
- Qwen: {'MERGE': 10}
- Agreement: 4/10

| # | Intent | Time | DINO | Deterministic | Qwen | Qwen reason |
|---:|---|---:|---:|---|---|---|
| 1 | lighting_exposure_change_activity_continuous | 119.0s | 0.9758 | MERGE | MERGE | Both captions describe the same crime scene activity, with the police vehicle and crime scene tape indicating ongoing investigation or emergency response. |
| 2 | lighting_exposure_change_activity_continuous | 124.0s | 0.9723 | MERGE | MERGE | Both captions describe the same immediate activity of a police vehicle with emergency lights flashing, indicating an ongoing emergency response. |
| 3 | obvious_merge_candidate | 180.0s | 0.9781 | MERGE | MERGE | Both captions describe the same activity of a police vehicle responding to an emergency at a street intersection. |
| 4 | obvious_merge_candidate | 868.0s | 0.9057 | MERGE | MERGE | Both captions describe the same immediate activity of a police officer providing medical attention to a person lying on the ground. |
| 5 | obvious_stop_candidate | 17.0s | 0.8056 | STOP | MERGE | Both captions describe the same activity of a person interacting with the dashboard of a vehicle, suggesting a moment of inspection or adjustment. |
| 6 | obvious_stop_candidate | 61.0s | 0.5237 | STOP | MERGE | Both captions describe the same activity of a person interacting with a dashboard in a vehicle, with the second caption providing additional context of a nighttime setting and a gas station. |
| 7 | obvious_stop_candidate | 194.0s | 0.4332 | STOP | MERGE | Both captions describe the same activity of a police officer interacting with a person on a table. |
| 8 | same_broad_topic_different_continuous_phase | 181.0s | 0.9242 | STOP | MERGE | Both captions describe the same activity of a police officer standing next to a car at night. |
| 9 | same_broad_topic_different_continuous_phase | 1001.0s | 0.9572 | STOP | MERGE | Both captions describe the same immediate activity of a police officer in a reflective state, likely documenting information or taking notes. |
| 10 | same_broad_topic_different_continuous_phase | 1079.0s | 0.9612 | STOP | MERGE | Both captions describe the same activity of police officers interacting with a vehicle at night, with one officer writing or reviewing notes and the other holding a clipboard. |
