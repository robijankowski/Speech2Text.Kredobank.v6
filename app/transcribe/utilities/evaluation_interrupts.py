from __future__ import annotations

from typing import Any, Dict, List

from app.transcribe.utilities.scenario_tools import Turn



def _word_count(text: str) -> int:
    return len([w for w in (text or "").strip().split() if w])


def _turns_to_segs(turns: List[Turn], role: str) -> List[Dict[str, Any]]:
    """
    Zamiana Turn -> segment dict zgodny z evaluation_interrupts.py:
      {"start":..., "end":..., "text":...}
    """
    out: List[Dict[str, Any]] = []
    role_norm = role.strip().upper()
    for t in turns:
        if (t.role or "").strip().upper() != role_norm:
            continue
        st = float(t.start)
        en = float(t.end)
        txt = (t.text or "").strip()
        if en > st and txt:
            out.append({"start": st, "end": en, "text": txt})
    out.sort(key=lambda x: (x["start"], x["end"]))
    return out


def merge_close_segments(
    segs: List[Dict[str, Any]],
    *,
    gap_ms: int = 300,
    join_with: str = " ",
) -> List[Dict[str, Any]]:
    """
    Sklej segmenty TEJ SAMEJ roli jeśli przerwa <= gap_ms.
    To jest ta sama logika co w evaluation_interrupts.py. :contentReference[oaicite:3]{index=3}
    """
    if not segs:
        return []

    gap_sec = gap_ms / 1000.0
    segs_sorted = sorted(segs, key=lambda x: (float(x["start"]), float(x["end"])))

    merged: List[Dict[str, Any]] = []
    cur = dict(segs_sorted[0])

    for nxt in segs_sorted[1:]:
        nxt = dict(nxt)
        cur_end = float(cur["end"])
        nxt_start = float(nxt["start"])
        gap = nxt_start - cur_end

        if gap <= gap_sec:
            cur["end"] = max(cur_end, float(nxt["end"]))
            cur_txt = str(cur.get("text", "") or "").strip()
            nxt_txt = str(nxt.get("text", "") or "").strip()

            if cur_txt and nxt_txt:
                cur["text"] = f"{cur_txt}{join_with}{nxt_txt}"
            elif nxt_txt:
                cur["text"] = nxt_txt
        else:
            merged.append(cur)
            cur = nxt

    merged.append(cur)
    return merged




def detect_any_overlaps_turns(
    ag_segs: List[Dict[str, Any]],
    cl_segs: List[Dict[str, Any]],

    # minimalny realny overlap
    min_overlap_ms: int = 250,
    # tolerancja na błędy timestampów (żeby nie gubić borderline)
    eps_ms: int = 30,

    # filtry jakości segmentów (żeby odsiać backchannel/śmieci)
    min_segment_ms_ag: int = 0,
    min_segment_ms_cl: int = 0,
    min_words_ag: int = 0,
    min_words_cl: int = 0,

    # ✅ NOWE: ignoruj overlap, jeśli zaczyna się w "ogonku" segmentu (tail)
    # typowo dla call-center: ignoruj ogon AG (bo CL często wchodzi "tak/aha" na końcu)
    ignore_tail_ms_ag: int = 0,
    ignore_tail_ms_cl: int = 0,
) -> List[Dict[str, Any]]:
    """
    Szuka wszystkich nakładających się odcinków pomiędzy segmentami AG i CL.
    Zwraca listę overlap-okien: overlap_start/end, overlap_sec + referencje do segów.

    Wersja bazuje na logice z evaluation_interrupts.py, ale:
    - ma dodatkowe filtry (min_words/min_segment)
    - ma filtr ignore_tail_ms_* do wycinania overlapów na końcówkach wypowiedzi
      (typowy backchannel: klient wchodzi w ostatnie 0.5–1.0s wypowiedzi AG)
    """

    min_overlap = min_overlap_ms / 1000.0
    eps = eps_ms / 1000.0

    min_seg_ag = min_segment_ms_ag / 1000.0
    min_seg_cl = min_segment_ms_cl / 1000.0

    tail_ag = ignore_tail_ms_ag / 1000.0
    tail_cl = ignore_tail_ms_cl / 1000.0

    def ag_ok(seg: Dict[str, Any]) -> bool:
        dur = float(seg["end"]) - float(seg["start"])
        if dur + eps < min_seg_ag:
            return False
        if _word_count(str(seg.get("text", ""))) < min_words_ag:
            return False
        return True

    def cl_ok(seg: Dict[str, Any]) -> bool:
        dur = float(seg["end"]) - float(seg["start"])
        if dur + eps < min_seg_cl:
            return False
        if _word_count(str(seg.get("text", ""))) < min_words_cl:
            return False
        return True

    i = j = 0
    out: List[Dict[str, Any]] = []

    # zakładamy że listy są posortowane po start
    # (jeśli nie masz pewności, sortuj przed wywołaniem)
    while i < len(ag_segs) and j < len(cl_segs):
        A = ag_segs[i]
        C = cl_segs[j]

        a_s = float(A["start"])
        a_e = float(A["end"])
        c_s = float(C["start"])
        c_e = float(C["end"])

        # wyznacz część wspólną
        s = max(a_s, c_s)
        e = min(a_e, c_e)
        overlap = e - s

        if overlap + eps >= min_overlap and ag_ok(A) and cl_ok(C):
            # ✅ filtr: ignoruj overlap jeśli startuje w "tail" jednego z segmentów
            # tail liczymy jako: (end - overlap_start) <= tail_window
            in_tail_ag = (tail_ag > 0.0) and ((a_e - s) <= tail_ag)
            in_tail_cl = (tail_cl > 0.0) and ((c_e - s) <= tail_cl)

            if not (in_tail_ag or in_tail_cl):
                out.append(
                    {
                        "overlap_start": s,
                        "overlap_end": e,
                        "overlap_sec": round(overlap, 3),
                        "ag_seg": A,
                        "cl_seg": C,
                    }
                )

        # przesuń wskaźnik krótszego segmentu (klasyczny merge-scan)
        if a_e < c_e:
            i += 1
        else:
            j += 1

    return out






def detect_starts_while_other_speaks_turns(
    starter_segs: List[Dict[str, Any]],
    while_segs: List[Dict[str, Any]],
    *,
    min_overlap_ms: int = 250,
    min_starter_segment_ms: int = 200,
    min_other_lead_ms: int = 0,
    eps_ms: int = 30,

    # ✅ NOWE: filtry "backchannel / tail"
    min_starter_words: int = 0,
    ignore_tail_ms_other: int = 0,
    min_other_remaining_ms: int = 0,
) -> List[Dict[str, Any]]:
    """
    Wykrywa przypadki, gdy starter zaczyna mówić w trakcie wypowiedzi other.
    Zwraca listę eventów (interruptions).

    Dodane filtry:
      - min_starter_words: ignoruj wtrącenia zbyt krótkie tekstowo
      - ignore_tail_ms_other: ignoruj, jeśli starter zaczyna w "ogonku" other
      - min_other_remaining_ms: ignoruj, jeśli other ma za mało czasu do końca w momencie wejścia startera
    """

    min_overlap_sec = min_overlap_ms / 1000.0
    min_starter_seg_sec = min_starter_segment_ms / 1000.0
    min_other_lead_sec = min_other_lead_ms / 1000.0
    eps = eps_ms / 1000.0

    tail_other_sec = ignore_tail_ms_other / 1000.0
    min_other_remaining_sec = min_other_remaining_ms / 1000.0

    # helper: starter ok?
    def starter_ok(seg: Dict[str, Any]) -> bool:
        dur = float(seg["end"]) - float(seg["start"])
        if dur + eps < min_starter_seg_sec:
            return False
        if _word_count(str(seg.get("text", ""))) < min_starter_words:
            return False
        return True

    events: List[Dict[str, Any]] = []
    j = 0
    nS = len(starter_segs)

    # zakładamy posortowane po czasie
    for o in while_segs:
        o_s = float(o["start"])
        o_e = float(o["end"])
        o_text = str(o.get("text", ""))

        # przesuń startery, które kończą się przed startem "other"
        while j < nS and float(starter_segs[j]["end"]) <= o_s + eps:
            j += 1

        k = j
        # iteruj startery, które mogą nachodzić na "other"
        while k < nS and float(starter_segs[k]["start"]) < o_e + eps:
            st = starter_segs[k]
            st_s = float(st["start"])
            st_e = float(st["end"])
            st_text = str(st.get("text", ""))

            # filtr długości + słów startera
            if not starter_ok(st):
                k += 1
                continue

            # starter musi zacząć w trakcie other, ale nie wcześniej niż min_other_lead
            # (czyli other musi już chwilę mówić)
            if (o_s + min_other_lead_sec) <= st_s < (o_e + eps):

                # ✅ filtr "ogon": jeśli starter wszedł w ostatnie X ms other -> pomiń
                if tail_other_sec > 0.0 and (o_e - st_s) <= tail_other_sec:
                    k += 1
                    continue

                # ✅ filtr: other musi mieć jeszcze minimalny "czas do końca"
                if min_other_remaining_sec > 0.0 and (o_e - st_s) < min_other_remaining_sec:
                    k += 1
                    continue

                # overlap liczony od startu startera
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







def analyze_turn_overlaps(
    turns: List[Turn],
    min_overlap_ms: int = 250,
    eps_ms: int = 30,
    # dla “interruptions”
    min_agent_segment_ms: int = 200,
    min_client_segment_ms: int = 200,
    min_other_lead_ms: int = 0,
    # opcjonalne filtry w detect_any_overlaps
    min_segment_ms_agent: int = 0,
    min_segment_ms_client: int = 0,
    min_words_agent: int = 0,
    min_words_client: int = 0,
    # pre-merge (jak w evaluation_interrupts.py)
    merge_gap_ms_agent: int = 350,
    merge_gap_ms_client: int = 350,
    ignore_tail_ms_ag: int = 0,
    ignore_tail_ms_cl: int = 0,
) -> Dict[str, Any]:
    """
    Analiza overlapów/interruptions na podstawie listy Turn(role in {"AG","CL"}).

    Zwraca strukturę podobną do analyze_diarized_groups() z evaluation_interrupts.py,
    tylko bez diarization/transkrypcji – operuje na gotowych turnach. :contentReference[oaicite:6]{index=6}
    """
    ag_segs = _turns_to_segs(turns, "AG")
    cl_segs = _turns_to_segs(turns, "CL")

    # jak w evaluation_interrupts.py: merge_close_segments(...) :contentReference[oaicite:7]{index=7}
    ag_segs = merge_close_segments(ag_segs, gap_ms=merge_gap_ms_agent)
    cl_segs = merge_close_segments(cl_segs, gap_ms=merge_gap_ms_client)

    overlaps = detect_any_overlaps_turns(
        ag_segs,
        cl_segs,
        min_overlap_ms=min_overlap_ms,
        eps_ms=eps_ms,
        min_segment_ms_ag=min_segment_ms_agent,
        min_segment_ms_cl=min_segment_ms_client,
        min_words_ag=min_words_agent,
        min_words_cl=min_words_client,
        ignore_tail_ms_ag=ignore_tail_ms_ag,
        ignore_tail_ms_cl=ignore_tail_ms_cl,
    )

    ag_interrupts = detect_starts_while_other_speaks_turns(
        starter_segs=ag_segs,
        while_segs=cl_segs,
        min_overlap_ms=min_overlap_ms,
        min_starter_segment_ms=min_agent_segment_ms,
        min_other_lead_ms=min_other_lead_ms,
        eps_ms=eps_ms,
        ignore_tail_ms_other=ignore_tail_ms_cl,
    )

    cl_interrupts = detect_starts_while_other_speaks_turns(
        starter_segs=cl_segs,
        while_segs=ag_segs,
        min_overlap_ms=min_overlap_ms,
        min_starter_segment_ms=min_client_segment_ms,
        min_other_lead_ms=min_other_lead_ms,
        eps_ms=eps_ms,
        ignore_tail_ms_other=ignore_tail_ms_ag,
    )

    return {
        "segments": {"agent": ag_segs, "client": cl_segs},
        "overlaps": overlaps,
        "events": {
            "agent_starts_while_client": ag_interrupts,
            "client_starts_while_agent": cl_interrupts,
        },
        "stats": {
            "agent_segments": len(ag_segs),
            "client_segments": len(cl_segs),
            "any_overlaps": len(overlaps),
            "agent_interrupts": len(ag_interrupts),
            "client_interrupts": len(cl_interrupts),
            "min_overlap_ms": min_overlap_ms,
            "eps_ms": eps_ms,
            "min_other_lead_ms": min_other_lead_ms,
        },
    }
