# Efficient semantic map v0.1

A strictly local semantic layer over the frozen EgoPolice long-video Fine → Medium → Coarse hierarchy. Cheap quality and existing DINOv2 features select one Medium keyframe. Conservative complete-link groups may reuse a directly generated Qwen2-VL caption; all propagation is explicit. Coarse summaries read ordered Medium text only, and whole-video stories read ordered Coarse text only.

No hierarchy boundary or membership is changed. No question, answer, retrieval, Planner, audio, QA, external API, or canonical pipeline path is used.

The frozen cosine threshold produced 149 singleton groups from 149 Medium nodes, so conservative semantic deduplication avoided 0 image calls. This negative efficiency result was retained without threshold retuning. Qwen story-format noncompliance is exposed through raw outputs and a clearly labeled ordered-Coarse-caption fallback in the review page.
