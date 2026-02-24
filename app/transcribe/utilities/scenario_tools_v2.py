from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Sequence

from app.core.config import settings
from app.core.logger import log



DiarSeg = Tuple[float, float, str, str]  

@dataclass(frozen=False)
class Turn:
    role: str           # "AGENT" | "CLIENT" | "AG" | "CL"
    start: float
    end: float
    text: str
    file: str

    # optional / defaults must be AFTER required fields
    text_diar: str = ""
    start_ext: Optional[float] = None
    end_ext: Optional[float] = None

    def __post_init__(self):
        # default ext to tight bounds if not provided
        if self.start_ext is None:
            self.start_ext = self.start
        if self.end_ext is None:
            self.end_ext = self.end
        if self.text_diar is None:
            self.text_diar = self.text





def _fmt_ts(sec: float) -> str:
    """Format seconds as MM:SS.mmm."""
    ms_total = int(round(sec * 1000))
    mm, rem = divmod(ms_total, 60_000)
    ss, ms = divmod(rem, 1000)
    return f"{mm:02d}:{ss:02d}.{ms:03d}"



def render_timestamped_script_from_turns(
    turns: Sequence[Turn],
    timestamp_on: bool = True,
    role_on: bool = True,
) -> str:
    lines: List[str] = []
    for t in turns:
        ts = f"[{_fmt_ts(t.start)}–{_fmt_ts(t.end)}] " if timestamp_on else ""
        rl = f"{t.role}" if role_on else ""
        lines.append(f"{ts}{rl}: {t.text}")
    return "\n".join(lines) + "\n"



def render_timestamped_script_from_diar_segs(
    segs: Iterable[Any],
    *,
    show_speaker: bool = True,
    show_index: bool = False,
) -> str:
    """
    Same timestamp formatting as render_timestamped_script_from_turns: [{_fmt_ts(st)}–{_fmt_ts(en)}]
    """
    def s_get(s: Any, k: str) -> Any:
        return s.get(k) if isinstance(s, dict) else getattr(s, k, None)

    lines: List[str] = []
    for i, s in enumerate(segs or []):
        st = s_get(s, "start")
        en = s_get(s, "end")
        txt = (s_get(s, "text") or "").strip()

        spk = (
            s_get(s, "speaker")
            or s_get(s, "speaker_id")
            or s_get(s, "spk")
            or ""
        )
        spk = str(spk).strip()

        if st is None or en is None:
            continue
        if not txt:
            continue

        prefix = f"{i:03d} " if show_index else ""
        ts = f"[{_fmt_ts(float(st))}–{_fmt_ts(float(en))}] "

        if show_speaker and spk:
            line = f"{prefix}{ts}{spk}: {txt}"
        else:
            line = f"{prefix}{ts}{txt}"

        lines.append(line)

    return "\n".join(lines) + "\n"



def render_turns_tight_vs_ext(
    turns: List[Turn],
    title: str = "\n==== Extended turns (tight -> ext) ====",
    limit: Optional[int] = None,
    show_text_diar: bool = False,
    max_text: int = 120,
    max_diar: int = 80,
) -> str:
    """
    Uses the SAME time formatting as render_timestamped_script_from_diar_segs:
      [{_fmt_ts(st)}–{_fmt_ts(en)}]
    and aligns columns by padding the pre-formatted timestamp strings.

    Alignment assumes calls are short and _fmt_ts length is stable for the dataset.
    """
    def clip(s: str, n: int) -> str:
        s = (s or "").replace("\n", " ").strip()
        return s if len(s) <= n else (s[: n - 1] + "…")

    it = turns if limit is None else turns[:limit]

    # pre-format to determine column widths
    tight_pairs = []
    ext_pairs = []
    for t in it:
        st, en = t.start, t.end
        stx = t.start_ext if t.start_ext is not None else st
        enx = t.end_ext if t.end_ext is not None else en
        tight_pairs.append(f"[{_fmt_ts(float(st))}–{_fmt_ts(float(en))}]")
        ext_pairs.append(f"[{_fmt_ts(float(stx))}–{_fmt_ts(float(enx))}]")

    w_tight = max([len(x) for x in tight_pairs], default=0)
    w_ext = max([len(x) for x in ext_pairs], default=0)

    lines: List[str] = [title]
    for t, tight_ts, ext_ts in zip(it, tight_pairs, ext_pairs):
        st, en = t.start, t.end
        stx = t.start_ext if t.start_ext is not None else st
        enx = t.end_ext if t.end_ext is not None else en

        line = (
            f"{tight_ts.ljust(w_tight)} ({(en - st):5.2f}s) "
            f"-> "
            f"{ext_ts.ljust(w_ext)} ({(enx - stx):5.2f}s)   "
            f"{(t.role or ''):4}  "
            f"{clip(t.text, max_text)}"
        )

        if show_text_diar and getattr(t, "text_diar", ""):
            line += f" | diar: {clip(t.text_diar, max_diar)}"

        lines.append(line)

    return "\n".join(lines) + "\n"




def format_scenario_md(text):
    """
    Convert dialogue text to a simple Markdown table.
    Each line becomes one row in the table.
    """
    lines = text.strip().split('\n')
    
    # Table header
    result = "| Role | Text |\n|-|----------|\n"
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if ':' in line:
            speaker, text = line.split(':', 1)
            speaker = speaker.strip()
            text = text.strip()
            result += f"| {speaker} | {text} |\n"
    
    return result





