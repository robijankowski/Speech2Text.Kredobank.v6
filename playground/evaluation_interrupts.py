
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


from app.core.config import settings
from app.openai_tools.openai_client_transcribe import async_transcribe_audio_diarized

from app.transcribe.utilities.audio_tools import clean_audio_file
from app.transcribe.utilities.scenario_tools import async_classify_agent_or_client_prefix


# ----------------------------
# utils
# ----------------------------
def fmt_ts(seconds: float) -> str:
    if seconds is None:
        return "??:??.??"
    m, s = divmod(float(seconds), 60.0)
    return f"{int(m):02d}:{s:05.2f}"


def _word_count(text: str) -> int:
    return len([w for w in (text or "").strip().split() if w])


def format_segments(segs: List[Dict[str, Any]], label: str = "") -> str:
    lines = [""]
    for i, s in enumerate(segs, 1):
        lines.append(f"Speaker: {s['text']}")
    return "\n".join(lines)

def format_segments_2print(segs: List[Dict[str, Any]], label: str = "") -> str:
    lines = [""]
    for i, s in enumerate(segs, 1):
        lines.append(f"[{i:04d}] {label} {fmt_ts(s['start'])}–{fmt_ts(s['end'])}: {s['text']}")
    return "\n".join(lines)




def merge_close_segments(
    segs: List[Dict[str, Any]],
    *,
    gap_ms: int = 300,
    join_with: str = " ",
) -> List[Dict[str, Any]]:
    """
    Merge consecutive segments (already from the SAME group/speaker list) if the time gap between
    them is <= gap_ms. Keeps speaker label from the first segment and updates timestamps.

    Expected input: list sorted by start time.
    Output: new list (sorted), merged where applicable.

    Rules:
      - If next.start - curr.end <= gap_ms => merge
      - start = curr.start
      - end   = next.end
      - text  = curr.text + join_with + next.text
    """
    if not segs:
        return []

    gap_sec = gap_ms / 1000.0
    segs_sorted = sorted(segs, key=lambda x: (float(x["start"]), float(x["end"])))

    merged: List[Dict[str, Any]] = []
    cur = dict(segs_sorted[0])

    for nxt in segs_sorted[1:]:
        nxt = dict(nxt)

        # normalize
        cur_start = float(cur["start"])
        cur_end = float(cur["end"])
        nxt_start = float(nxt["start"])
        nxt_end = float(nxt["end"])

        gap = nxt_start - cur_end

        if gap <= gap_sec:
            # merge
            cur["end"] = max(cur_end, nxt_end)
            cur_text = str(cur.get("text", "") or "").strip()
            nxt_text = str(nxt.get("text", "") or "").strip()

            if cur_text and nxt_text:
                cur["text"] = f"{cur_text}{join_with}{nxt_text}"
            elif nxt_text:
                cur["text"] = nxt_text  # current empty, next has text
            # speaker stays as-is
        else:
            merged.append(cur)
            cur = nxt

    merged.append(cur)
    return merged


def merged_dialogue_text(
    agent_segs: List[Dict[str, Any]],
    client_segs: List[Dict[str, Any]],
    *,
    agent_label: str = "AG",
    client_label: str = "CL",
    include_timestamps: bool = True,
) -> str:
    """
    Merge two segment lists into one time-ordered dialogue string.
    Example:
      AG: ...
      CL: ...
    """
    items: List[Dict[str, Any]] = []
    for s in agent_segs:
        items.append({"start": float(s["start"]), "end": float(s["end"]), "text": s["text"], "label": agent_label})
    for s in client_segs:
        items.append({"start": float(s["start"]), "end": float(s["end"]), "text": s["text"], "label": client_label})

    items.sort(key=lambda x: (x["start"], x["end"]))

    lines: List[str] = []
    for it in items:
        txt = str(it["text"]).strip()
        if not txt:
            continue
        if include_timestamps:
            lines.append(f"{it['label']} {fmt_ts(it['start'])}–{fmt_ts(it['end'])}: {txt}")
        else:
            lines.append(f"{it['label']}: {txt}")
    return "\n".join(lines)


# ----------------------------
# diarized transcription
# ----------------------------
async def async_transcript_audio_file_verbose_o4_diarize(
    file_name: str,
    temperature: float = 0.0,
    chunking_strategy: str = "auto",
) -> List[Dict[str, Any]]:
    """
    Uses gpt-4o-transcribe-diarize (or whatever is in settings.OPENAI_MODEL_TRANSCRIBE_DIARIZE).
    Returns LIST of diarized segments with fields: speaker, start, end, text (and possibly more).
    """

    def _get(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    # with open(file_name, "rb") as audio_file:
    #     transcript = openai_client.audio.transcriptions.create(
    #         file=audio_file,
    #         model=model,
    #         response_format="diarized_json",
    #         temperature=temperature,
    #         chunking_strategy=chunking_strategy,
    #     )

    transcript = await async_transcribe_audio_diarized(
        audio=file_name,
        temperature=temperature,
        chunking_strategy=chunking_strategy
    )

    segments = _get(transcript, "segments", []) or []
    speakers = sorted({(_get(s, "speaker", "unknown")) for s in segments})
    print(f"Segments: {len(segments)} | Speakers: {speakers}")

    # Normalize each segment to plain dict
    out: List[Dict[str, Any]] = []
    for s in segments:
        if isinstance(s, dict):
            sp = s.get("speaker", "unknown")
            st = float(s.get("start", 0.0))
            en = float(s.get("end", 0.0))
            tx = str(s.get("text", "") or "").strip()
        else:
            sp = getattr(s, "speaker", "unknown")
            st = float(getattr(s, "start", 0.0))
            en = float(getattr(s, "end", 0.0))
            tx = str(getattr(s, "text", "") or "").strip()

        if en > st and tx:
            out.append({"speaker": str(sp), "start": st, "end": en, "text": tx})

    out.sort(key=lambda x: x["start"])
    return out


def split_by_speaker(
    diarized_segments: List[Dict[str, Any]],
    *,
    a_speaker: str = "A",
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns (A_segments, other_segments) where:
      - A_segments: speaker == a_speaker
      - other_segments: everyone else (B, C, unknown, ...)
    """
    a_norm = str(a_speaker).strip().upper()
    a_list: List[Dict[str, Any]] = []
    other_list: List[Dict[str, Any]] = []

    for s in diarized_segments:
        sp = str(s.get("speaker", "unknown")).strip().upper()
        if sp == a_norm:
            a_list.append(s)
        else:
            other_list.append(s)

    a_list.sort(key=lambda x: x["start"])
    other_list.sort(key=lambda x: x["start"])
    return a_list, other_list


# ----------------------------
# overlap + interruption detectors
# ----------------------------
def detect_any_overlaps(
    left: List[Dict[str, Any]],
    right: List[Dict[str, Any]],
    *,
    min_overlap_ms: int = 200,
    eps_ms: int = 30,
    min_segment_ms_left: int = 0,
    min_segment_ms_right: int = 0,
    min_words_left: int = 0,
    min_words_right: int = 0,
) -> List[Dict[str, Any]]:
    """
    Any overlap windows between any left/right segments (regardless of who started first),
    with optional filters to ignore short segments / short utterances.
    """
    min_overlap = min_overlap_ms / 1000.0
    eps = eps_ms / 1000.0
    min_seg_left = min_segment_ms_left / 1000.0
    min_seg_right = min_segment_ms_right / 1000.0

    def left_ok(seg: Dict[str, Any]) -> bool:
        dur = float(seg["end"]) - float(seg["start"])
        if dur + eps < min_seg_left:
            return False
        if _word_count(str(seg.get("text", ""))) < min_words_left:
            return False
        return True

    def right_ok(seg: Dict[str, Any]) -> bool:
        dur = float(seg["end"]) - float(seg["start"])
        if dur + eps < min_seg_right:
            return False
        if _word_count(str(seg.get("text", ""))) < min_words_right:
            return False
        return True

    i = j = 0
    out: List[Dict[str, Any]] = []

    while i < len(left) and j < len(right):
        L = left[i]
        R = right[j]

        s = max(float(L["start"]), float(R["start"]))
        e = min(float(L["end"]), float(R["end"]))
        overlap = e - s

        if overlap + eps >= min_overlap and left_ok(L) and right_ok(R):
            out.append(
                {
                    "overlap_start": s,
                    "overlap_end": e,
                    "overlap_sec": round(overlap, 3),
                    "left_seg": L,
                    "right_seg": R,
                }
            )

        if float(L["end"]) < float(R["end"]):
            i += 1
        else:
            j += 1

    return out


def detect_starts_while_other_speaks(
    starter_segs: List[Dict[str, Any]],
    while_segs: List[Dict[str, Any]],
    *,
    min_overlap_ms: int = 250,
    min_starter_segment_ms: int = 200,
    min_other_lead_ms: int = 0,
    eps_ms: int = 30,
) -> List[Dict[str, Any]]:
    """
    Events where STARTER starts while OTHER is ongoing:
      other.start + lead <= starter.start < other.end
    and overlap >= threshold.
    """
    min_overlap_sec = min_overlap_ms / 1000.0
    min_starter_seg_sec = min_starter_segment_ms / 1000.0
    min_other_lead_sec = min_other_lead_ms / 1000.0
    eps = eps_ms / 1000.0

    events: List[Dict[str, Any]] = []
    j = 0
    nS = len(starter_segs)

    for o in while_segs:
        o_s, o_e, o_text = o["start"], o["end"], o["text"]

        while j < nS and starter_segs[j]["end"] <= o_s + eps:
            j += 1

        k = j
        while k < nS and starter_segs[k]["start"] < o_e + eps:
            st = starter_segs[k]
            st_s, st_e, st_text = st["start"], st["end"], st["text"]

            if (st_e - st_s) < min_starter_seg_sec:
                k += 1
                continue

            if (o_s + min_other_lead_sec) <= st_s < (o_e + eps):
                overlap = min(st_e, o_e) - st_s
                if overlap + eps >= min_overlap_sec:
                    events.append(
                        {
                            "starter_start": st_s,
                            "starter_end": st_e,
                            "other_start": o_s,
                            "other_end": o_e,
                            "overlap_sec": round(overlap, 3),
                            "other_lead_sec": round(st_s - o_s, 3),
                            "starter_text": st_text,
                            "other_text": o_text,
                        }
                    )
            k += 1

    return events


# ----------------------------
# main analysis (NO split L/R)
# ----------------------------
async def async_analyze_diarized_groups(
    audio_file: str,
    *,
    a_speaker: str = "A",
    min_overlap_ms: int = 250,
    eps_ms: int = 30,
    min_agent_segment_ms: int = 200,
    min_client_segment_ms: int = 200,
    min_other_lead_ms: int = 0,

    # overlap filters (optional)
    min_segment_ms_agent: int = 0,
    min_segment_ms_client: int = 0,
    min_words_agent: int = 0,
    min_words_client: int = 0
) -> Dict[str, Any]:
    
    # labels for merged dialogue output
    agent_label: str = "AG"
    client_label: str = "CL"
    
    # 1) diarize original file
    clean_audio_file_path = clean_audio_file(str(audio_file))
    diarized = await async_transcript_audio_file_verbose_o4_diarize(file_name=clean_audio_file_path)

    # 2) create two lists: A and (B + all rest)
    agent_segs, client_segs = split_by_speaker(diarized, a_speaker=a_speaker)

    agent_segs = merge_close_segments(agent_segs, gap_ms=350)
    client_segs = merge_close_segments(client_segs, gap_ms=350)

    # print(f"\nagents segs\n{agent_segs}")
    a_segs_role = await async_classify_agent_or_client_prefix(format_segments(agent_segs))
    print(f"\n'A' speaker segment role: {a_segs_role}")
    if not a_segs_role == "AG":
        agent_segs, client_segs = client_segs, agent_segs  # swap

    # print(format_segments_2print(agent_segs, f"Speaker {agent_label}"))
    # print(format_segments_2print(client_segs, f"Speaker {client_label}"))

    # 3) overlaps (A vs rest)
    overlaps = detect_any_overlaps(
        agent_segs,
        client_segs,
        min_overlap_ms=min_overlap_ms,
        eps_ms=eps_ms,
        min_segment_ms_left=min_segment_ms_agent,
        min_segment_ms_right=min_segment_ms_client,
        min_words_left=min_words_agent,
        min_words_right=min_words_client,
    )

    print(f"\n=== Any overlaps >= {min_overlap_ms}ms ({len(overlaps)}) ===")
    for idx, o in enumerate(overlaps, 1):
        print(
            f"[{idx:03d}] {fmt_ts(o['overlap_start'])}–{fmt_ts(o['overlap_end'])} "
            f"({o['overlap_sec']}s) | "
            f"{agent_label}: {o['left_seg']['text']} | {client_label}: {o['right_seg']['text']}"
        )

    # interruptions in both directions
    agent_starts_while_client = detect_starts_while_other_speaks(
        starter_segs=agent_segs,
        while_segs=client_segs,
        min_overlap_ms=min_overlap_ms,
        min_starter_segment_ms=min_agent_segment_ms,
        min_other_lead_ms=min_other_lead_ms,
        eps_ms=eps_ms,
    )
    client_starts_while_agent = detect_starts_while_other_speaks(
        starter_segs=client_segs,
        while_segs=agent_segs,
        min_overlap_ms=min_overlap_ms,
        min_starter_segment_ms=min_client_segment_ms,
        min_other_lead_ms=min_other_lead_ms,
        eps_ms=eps_ms,
    )

    def _print_events(title: str, events: List[Dict[str, Any]]):
        print(f"\n=== {title} ({len(events)}) ===")
        for i, e in enumerate(events, 1):
            print(
                f"[{i:03d}] START {fmt_ts(e['starter_start'])}  "
                f"overlap={e['overlap_sec']}s lead={e['other_lead_sec']}s | "
                f"starter: {e['starter_text']}  | while other: {e['other_text']}"
            )

    _print_events(f"{agent_label} starts while {client_label} is speaking", agent_starts_while_client)
    _print_events(f"{client_label} starts while {agent_label} is speaking", client_starts_while_agent)

    # merged dialogue (time order)
    dialogue = merged_dialogue_text(
        agent_segs=agent_segs,
        client_segs=client_segs,
        agent_label=agent_label,
        client_label=client_label,
        include_timestamps=True,
    )
    print(f"\n=== Merged Dialogue ({len(dialogue)} chars) ===\n{dialogue}")

    return {
        "segments": {"agent": agent_segs, "client": client_segs},
        "overlaps": overlaps,
        "events": {
            "agent_starts_while_client": agent_starts_while_client,
            "client_starts_while_agent": client_starts_while_agent,
        },
        "dialogue": dialogue,
        "stats": {
            "agent_segments": len(agent_segs),
            "client_segments": len(client_segs),
            "any_overlaps": len(overlaps),
            "agent_interrupts": len(agent_starts_while_client),
            "client_interrupts": len(client_starts_while_agent),
            "min_overlap_ms": min_overlap_ms,
            "eps_ms": eps_ms,
            "min_other_lead_ms": min_other_lead_ms,
            "a_speaker": a_speaker,
        },
    }

async def async_detect_agent_interruptions(file_name: str) -> Dict[str, Any]:
    """
    Detect whether a call recording contains meaningful overlapping speech / interruptions.

    The function runs a diarized transcription analysis on the given audio file using
    `analyze_diarized_groups()`. It treats speaker "A" as the agent group and aggregates
    all other diarized speakers ("B" + any others) as the client group. It then computes
    overlaps/interruptions between these two groups and returns a simple YES/NO decision
    based on the number of detected overlaps.

    Parameters
    ----------
    file_name : str
        Path to the audio file to analyze (e.g., WAV/MP3).

    Returns
    -------
    Optional[str]
        - "YES" if the analysis detected more than one overlap event
          (`res["stats"]["any_overlaps"] > 1`).
        - "NO" if overlap events are <= 1.
        - None if an exception occurs during processing (the error is printed).

    Notes
    -----
    This function is a thin wrapper over `analyze_diarized_groups()` and uses the following
    tuning parameters:

    - a_speaker="A":
        Speaker label assumed to represent the agent in diarized output.
    - min_overlap_ms=450:
        Minimum overlap duration (ms) required to count an overlap event. Higher values
        ignore short overlaps (e.g., brief backchannel words).
    - eps_ms=30:
        Timestamp tolerance (ms) to account for diarization/ASR timing jitter.
    - min_agent_segment_ms=200 / min_client_segment_ms=200:
        Minimum segment duration (ms) for interruption detection logic.
    - min_other_lead_ms=0:
        Allows detecting interruptions even when both parties start nearly simultaneously.
        Increase this value to focus on "true" interruptions where the other speaker has
        already been speaking for some time.
    - min_segment_ms_agent=200 / min_segment_ms_client=200 and min_words_agent=1 / min_words_client=1:
        Filters for overlap detection to ignore very short segments and extremely short utterances.

    The function prints the `stats` returned by `analyze_diarized_groups()` for debugging.

    Examples
    --------
    >>> result = detect_agent_interruptions("call.wav")
    >>> if result == "YES":
    ...     print("Overlapping speech detected")
    ... elif result == "NO":
    ...     print("No meaningful overlaps detected")
    ... else:
    ...     print("Analysis failed")
    """
    try:
        res = await async_analyze_diarized_groups(
            file_name,
            a_speaker="A",          # group A vs (B+rest)
            min_overlap_ms=450,
            eps_ms=30,
            min_agent_segment_ms=200,
            min_client_segment_ms=200,
            min_other_lead_ms=0,

            # OPTIONAL: ignore short “backchannel”:
            min_segment_ms_agent=200,
            min_segment_ms_client=200,
            min_words_agent=1,
            min_words_client=1,
        )
    
        return res
    
    except Exception as e:
        print(f"Error processing file '{file_name}': {e}")
        return None
    


