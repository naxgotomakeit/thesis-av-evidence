from __future__ import annotations

import copy
import hashlib
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any


def load_plans_immutable(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return isolated copies so Task 5B cannot mutate frozen Task 5A records."""
    return copy.deepcopy(rows)


def overlap_seconds(a: float, b: float, c: float, d: float) -> float:
    return max(0.0, min(b, d) - max(a, c))


def interval_distance(a: float, b: float, c: float, d: float) -> float:
    return max(0.0, c - b, a - d)


def union_duration(items: list[dict[str, Any]]) -> float:
    spans = sorted((float(x["start_time"]), float(x["end_time"])) for x in items)
    total, end = 0.0, float("-inf")
    for start, stop in spans:
        if start > end: total += stop - start
        elif stop > end: total += stop - end
        end = max(end, stop)
    return total


def stable_id(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha1("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def resolve_question_intervals(question: str, cues: dict[str, Any], temporal_relation: str, duration: float) -> dict[str, Any]:
    raw, interpretations, expanded, trace = [], [], [], []
    for cue in cues.get("time_cues", []):
        if cue.get("start_sec") is None: continue
        start, end = float(cue["start_sec"]), float(cue.get("end_sec", cue["start_sec"]))
        raw.append({"raw_text": cue["raw_text"], "start_sec": start, "end_sec": end, "source": "question_text", "hard_constraint": True})
    lower = question.casefold()
    lexical = next((word for word in ("followed", "following", "after", "before") if re.search(rf"\b{word}\b", lower)), None)
    relation = temporal_relation
    if lexical in {"followed", "following", "after"}: relation = "after"
    elif lexical == "before": relation = "before"
    if lexical: trace.append({"event":"lexical_temporal_guardrail","lexical_cue":lexical,"plan_relation":temporal_relation,"executed_relation":relation})
    if not raw and re.search(r"\bat the very start\b", lower):
        raw.append({"raw_text":"at the very start","start_sec":0.0,"end_sec":min(5.0,duration),"source":"question_text","hard_constraint":True})
        interpretations.append({"raw_text":"at the very start","deterministic_interpretation":"first_5_seconds","start_sec":0.0,"end_sec":min(5.0,duration),"not_ground_truth":True})
    for item in raw:
        start, end = item["start_sec"], item["end_sec"]
        if item["raw_text"].casefold()=="at the very start": before = after = 0.0; reason = "deterministic first_5_seconds interval preserved without expansion"
        elif relation in {"between", "count_within"}: before = after = 0.0; reason = "explicit range preserved for between/count_within"
        elif relation in {"after", "sequence"}: before, after, reason = .5, 2.0, "post-event context for after/followed/sequence"
        elif relation == "before": before, after, reason = 2.0, .5, "pre-event context for before"
        else: before, after, reason = 1.0, 1.0, "symmetric context for during/none/uncertain"
        final_start, final_end = max(0.0, start-before), min(duration, end+after)
        expanded.append({"original_start_sec":start,"original_end_sec":end,"expand_before_sec":before,"expand_after_sec":after,"reason":reason,"start_sec":final_start,"end_sec":final_end})
    return {"raw_question_intervals":raw,"deterministic_interpretations":interpretations,"expanded_intervals":expanded,"final_search_intervals":[{"start_sec":x["start_sec"],"end_sec":x["end_sec"],"source":"question_cue_expansion"} for x in expanded],"executed_relation":relation,"routing_trace":trace}


def _norm(text: str) -> str: return unicodedata.normalize("NFKC", text).strip()
def _punctless(text: str) -> str: return " ".join("".join(ch if ch.isalnum() else " " for ch in _norm(text).casefold()).split())


def match_quoted_phrase(phrase: str, segments: list[dict[str, Any]], interval: tuple[float,float] | None = None, fuzzy_threshold: float = .6) -> list[dict[str, Any]]:
    pool = [x for x in segments if interval is None or overlap_seconds(float(x["start_time"]),float(x["end_time"]),*interval)>0]
    results=[]; target=_norm(phrase); target_fold=target.casefold(); target_plain=_punctless(target)
    for row in pool:
        text=_norm(str(row.get("text",""))); method=None; score=0.0
        if target in text: method,score="unicode_normalized_exact",1.0
        elif target_fold in text.casefold(): method,score="case_insensitive",.98
        elif target_plain and target_plain in _punctless(text): method,score="punctuation_insensitive",.95
        else:
            score=SequenceMatcher(None,target_plain,_punctless(text)).ratio()
            if score>=fuzzy_threshold: method="token_fuzzy_fallback"
        if method:
            results.append({"transcript_segment_id":row["transcript_segment_id"],"matched_transcript_text":row.get("text",""),"start_time":float(row["start_time"]),"end_time":float(row["end_time"]),"match_method":method,"match_score":round(score,6)})
    results.sort(key=lambda x:(-x["match_score"],x["start_time"],x["transcript_segment_id"]))
    for item in results:
        item["number_of_possible_matches"]=len(results); item["ambiguity_warning"]="multiple_possible_matches" if len(results)>1 else None
    return results


def modalities_to_execute(plan: dict[str, Any], fallback: set[str] | None = None) -> tuple[list[str],list[dict[str,str]]]:
    required=set(plan.get("resolver_modalities",[])); fallback=fallback or set(); executed=sorted(required|fallback)
    skipped=[]
    for modality in ("visual","speech","acoustic"):
        if modality not in executed: skipped.append({"modality":modality,"reason":"not requested by Task 5A resolver_modalities and no fallback triggered"})
    return executed,skipped


def clip_candidate(row: dict[str, Any], intervals: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not intervals: return dict(row)
    overlaps=[]
    for interval in intervals:
        amount=overlap_seconds(float(row["start_time"]),float(row["end_time"]),float(interval["start_sec"]),float(interval["end_sec"]))
        if amount>0: overlaps.append((amount,max(float(row["start_time"]),float(interval["start_sec"])),min(float(row["end_time"]),float(interval["end_sec"]))))
    if not overlaps: return None
    _,start,end=max(overlaps)
    result=dict(row); result["source_start_time"]=float(row["start_time"]); result["source_end_time"]=float(row["end_time"]); result["start_time"]=start; result["end_time"]=end; result["hard_interval_overlap_sec"]=end-start
    return result


def link_candidates(candidates: list[dict[str, Any]], relation: str, max_proximity: float=2.0) -> list[dict[str, Any]]:
    if not candidates: return []
    ordered=sorted(candidates,key=lambda x:(x["start_time"],x["end_time"],x["candidate_id"])); groups=[]
    for candidate in ordered:
        compatible=None
        for group in groups:
            distance=interval_distance(group["start_time"],group["end_time"],candidate["start_time"],candidate["end_time"])
            if distance<=max_proximity: compatible=group; break
        if compatible is None:
            compatible={"members":[],"start_time":candidate["start_time"],"end_time":candidate["end_time"]}; groups.append(compatible)
        compatible["members"].append(candidate); compatible["start_time"]=min(compatible["start_time"],candidate["start_time"]); compatible["end_time"]=max(compatible["end_time"],candidate["end_time"])
    linked=[]
    for group in groups:
        members=group["members"]; ids=[x["candidate_id"] for x in members]; pairs=[]
        for i,left in enumerate(members):
            for right in members[i+1:]: pairs.append({"left":left["candidate_id"],"right":right["candidate_id"],"overlap_sec":overlap_seconds(left["start_time"],left["end_time"],right["start_time"],right["end_time"]),"proximity_sec":interval_distance(left["start_time"],left["end_time"],right["start_time"],right["end_time"])})
        linked.append({"linked_window_id":stable_id("linked",*ids),"source_candidate_ids":ids,"linked_modalities":sorted({x["modality"] for x in members}),"union_start_time":group["start_time"],"union_end_time":group["end_time"],"pairwise_temporal_metrics":pairs,"relation_used":relation,"reason_for_linking":"temporal overlap or proximity <= 2 seconds among required resolver candidates"})
    return linked


DEFAULT_LIMITS={"maximum_linked_windows":3,"maximum_total_selected_duration_sec":12.0,"maximum_visual_micro_windows":2,"maximum_acoustic_candidates":3,"maximum_speech_candidates":5}


def apply_budget(candidates: list[dict[str, Any]], operation: str) -> tuple[list[dict[str, Any]],dict[str,Any]]:
    if operation in {"count_occurrences","measure_delay"}:
        return list(candidates),{"policy":f"{operation}_operation_exception_preserves_all_occurrences_or_sequence_candidates","limits":{},"removed_candidates":[]}
    priority=sorted(candidates,key=lambda x:(-int(bool(x.get("hard_timestamp_consistent"))),float(x.get("anchor_distance_sec",0)), -float(x.get("similarity_score") if x.get("similarity_score") is not None else -1),float(x["end_time"])-float(x["start_time"]),x["candidate_id"]))
    selected=[]; removed=[]; counts={"visual_micro":0,"acoustic":0,"speech":0}
    for item in priority:
        kind="visual_micro" if item["modality"]=="visual" and item.get("candidate_type")=="micro_window" else item["modality"]
        limit={"visual_micro":2,"acoustic":3,"speech":5}.get(kind)
        reason=None
        if limit is not None and counts[kind]>=limit: reason=f"{kind} candidate limit reached"
        elif union_duration(selected+[item])>12.0+1e-9: reason="maximum total selected duration (12s) would be exceeded"
        if reason: removed.append({"candidate_id":item["candidate_id"],"reason_removed":reason,"score":item.get("similarity_score"),"rank":item.get("rank"),"might_affect_evidence_sufficiency":True})
        else: selected.append(item); counts[kind]=counts.get(kind,0)+1
    return selected,{"policy":"deterministic operation-aware identify/describe/compare budget","limits":dict(DEFAULT_LIMITS),"removed_candidates":removed}


def package_micro_frames(candidate: dict[str, Any], source_microclip: dict[str, Any], fps: float | None = None) -> dict[str, Any]:
    """Separate source and selected frames using half-open interval semantics."""
    result = copy.deepcopy(candidate)
    source_paths = list(source_microclip.get("frame_paths") or candidate.get("source_frame_paths") or candidate.get("frame_paths") or [])
    timestamps = source_microclip.get("frame_timestamps")
    warning = None
    if timestamps is not None and len(timestamps) == len(source_paths):
        source_timestamps = [float(x) for x in timestamps]
    else:
        indices = source_microclip.get("frame_indices")
        if fps and fps > 0 and indices is not None and len(indices) == len(source_paths):
            source_timestamps = [float(index) / float(fps) for index in indices]
        else:
            source_timestamps = []
            warning = "unable_to_resolve_micro_frame_timestamps"
    start, end = float(candidate["start_time"]), float(candidate["end_time"])
    selected_pairs = [(path, timestamp) for path, timestamp in zip(source_paths, source_timestamps) if start <= timestamp < end]
    result["source_frame_paths"] = source_paths
    result["source_frame_timestamps"] = source_timestamps
    result["selected_frame_paths"] = [path for path, _ in selected_pairs]
    result["selected_frame_timestamps"] = [timestamp for _, timestamp in selected_pairs]
    result["frame_paths"] = list(result["selected_frame_paths"])
    result.setdefault("warnings", [])
    if warning and warning not in result["warnings"]: result["warnings"].append(warning)
    result["frame_path_semantics"] = "frame_paths aliases selected_frame_paths; source_frame_paths preserves the original microclip frames"
    result["frame_interval_semantics"] = "selected_start <= frame_timestamp < selected_end"
    return result


def canonical_visual_frames(
    selected_micro_candidates: list[dict[str, Any]],
    dense_frames: list[dict[str, Any]],
    video_id: str | None = None,
) -> list[dict[str, Any]]:
    """Deduplicate by ``video_id + millisecond timestamp``, preferring dense frames."""
    merged: dict[tuple[str, int], dict[str, Any]] = {}
    for candidate in selected_micro_candidates:
        for path, timestamp in zip(candidate.get("selected_frame_paths", []), candidate.get("selected_frame_timestamps", [])):
            timestamp_ms = int(round(float(timestamp) * 1000.0))
            frame_video_id = str(candidate.get("video_id") or video_id or "unknown_video")
            key = (frame_video_id, timestamp_ms)
            item = merged.setdefault(key, {"video_id": frame_video_id, "timestamp": timestamp_ms / 1000.0, "normalized_timestamp_ms": timestamp_ms, "canonical_frame_path": path, "provenance": "micro_window", "source_candidate_ids": [], "provenance_records": []})
            if candidate["candidate_id"] not in item["source_candidate_ids"]: item["source_candidate_ids"].append(candidate["candidate_id"])
            item["provenance_records"].append({"type":"micro_window","frame_path":path,"source_candidate_id":candidate["candidate_id"]})
    for frame in dense_frames:
        timestamp_ms = int(round(float(frame["timestamp"]) * 1000.0)); path = frame["frame_path"]
        frame_video_id = str(frame.get("video_id") or video_id or "unknown_video")
        key = (frame_video_id, timestamp_ms)
        item = merged.setdefault(key, {"video_id": frame_video_id, "timestamp": timestamp_ms / 1000.0, "normalized_timestamp_ms": timestamp_ms, "canonical_frame_path": path, "provenance": "dense_frame", "source_candidate_ids": [], "provenance_records": []})
        had_micro = any(x["type"] == "micro_window" for x in item["provenance_records"])
        item["canonical_frame_path"] = path
        item["provenance"] = "both" if had_micro else "dense_frame"
        item["provenance_records"].append({"type":"dense_frame","frame_path":path,"dense_frame_id":frame.get("dense_frame_id")})
    return [merged[key] for key in sorted(merged)]
