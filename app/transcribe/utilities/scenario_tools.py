import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Sequence

from app.core.config import settings
from app.core.logger import log

# @dataclass(frozen=True)
# class Turn:
#     role: str  # "AG" | "CL"
#     start: float
#     end: float
#     text: str

DiarSeg = Tuple[float, float, str, str]  # (start, end, text, speaker)

@dataclass(frozen=False)
class Turn:
    role: str           # "AG" | "CL"
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

from app.transcribe.utilities.stats import set_stats
from app.openai_tools.openai_token_utilities import num_tokens_from_text

from app.openai_tools.openai_client_text import async_chat_completion, async_chat_completion_with_format


def _default_detect_speaker_role_model() -> str:
    return settings.AZURE_MODEL_CHAT_TRS_DETECT_PLAYER if settings.USE_AZURE_OPENAI == "Y" else settings.OPENAI_MODEL_CHAT_TRS_DETECT_PLAYER

def _default_split_into_roles_model() -> str:
    return settings.AZURE_MODEL_CHAT_TRS_SPLIT_INTO_ROLES if settings.USE_AZURE_OPENAI == "Y" else settings.OPENAI_MODEL_CHAT_TRS_SPLIT_INTO_ROLES

def _default_transcription_model() -> str:
    return settings.AZURE_MODEL_TRANSCRIBE_STEREO if settings.USE_AZURE_OPENAI == "Y" else settings.OPENAI_MODEL_TRANSCRIBE_STEREO



def _fmt_ts(sec: float) -> str:
    """Format seconds as MM:SS.mmm."""
    ms_total = int(round(sec * 1000))
    mm, rem = divmod(ms_total, 60_000)
    ss, ms = divmod(rem, 1000)
    return f"{mm:02d}:{ss:02d}.{ms:03d}"


from typing import Any, Iterable, List, Optional, Sequence

# assumes you already have:
# def _fmt_ts(x: float) -> str: ...


def remap_turn_roles(turns: List[Turn]) -> List[Turn]:
    """
    remaps roles: {"AGENT": "AG", "CLIENT": "CL"}
    """
    role_map = {"AGENT": "AG", "CLIENT": "CL"}

    for t in turns:
        r = (t.role or "").strip().upper()
        if r in role_map:
            t.role = role_map[r]
    return turns


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








async def async_split_transcription_into_roles_4o(agent_text, client_text, stereo_text, 
                                      metadata_text="", 
                                      model: str = "") -> str:
    SYSTEM_PROMPT = """
You are a highly accurate transcription and dialogue reconstruction assistant with specialized expertise in speaker identification.

You receive separate transcriptions of a bilingual phone call between two speakers — a bank **agent** and a **client** — from different audio channels (agent, client, stereo). Your task is to **merge them into a complete, speaker-labeled script** with PERFECT speaker assignment accuracy.

This conversation takes place between a **KredoBank Ukraine collections agent** and a **KredoBank client**, most likely concerning **outstanding debt, negative balance, account restrictions, or enforcement actions**.

CRITICAL REQUIREMENTS:
1. **COMPLETE PRESERVATION**: You MUST preserve EVERY SINGLE PHRASE, WORD, NAMES of people and UTTERANCE from the STEREO transcript
2. **ACCURATE SPEAKER ASSIGNMENT**: You MUST verify each utterance's speaker by cross-referencing with individual channel transcripts
3. **ORDER OF SENTENCES**: You MUST preserve order of sentences in STEREO transcript.

Your methodology:
* Use the **STEREO channel transcript as the complete foundation** — every word from it must appear in your output
* Use the **AGENT channel** to identify which parts were spoken by the agent
* Use the **CLIENT channel** to identify which parts were spoken by the client
* When in doubt, use contextual clues (professional language, banking terms, formal address patterns)
* Format as a script with clear speaker labels: `AG:` for agent, `CL:` for client

NEVER omit content from the stereo transcript! Your output should be a COMPLETE transcription with 100% accurate speaker assignments.
"""

    USER_PROMPT = f"""
You are given three transcripts of a phone call between a KredoBank Ukraine agent and a KredoBank client. The conversation is likely from the bank's collections department and concerns an account debt or negative balance.

Your task is to create a COMPLETE speaker-labeled conversation script that includes EVERY SINGLE PHRASE from the STEREO transcript with PERFECT speaker identification.

Conversation Metadata. Known entities canonical names:
{metadata_text}

Input Sources:
AGENT CHANNEL (AG): <AGENT_CHANNEL>{agent_text}</AGENT_CHANNEL>  
CLIENT CHANNEL (CL): <CLIENT_CHANNEL>{client_text}</CLIENT_CHANNEL>
STEREO CHANNEL (combined): <STEREO_CHANNEL>{stereo_text}</STEREO_CHANNEL>

MANDATORY SPEAKER VERIFICATION PROCESS:

1. **PRIMARY RULE**: Every word from STEREO transcript MUST appear in your final output

2. **SPEAKER IDENTIFICATION METHODOLOGY**:
   - For each phrase/utterance in the STEREO transcript, CHECK if it appears in:
     * AGENT channel → Label as AG:
     * CLIENT channel → Label as CL:
     * Both channels → Use contextual analysis (see below)
     * Neither channel clearly → Use contextual analysis

3. **CONTEXTUAL SPEAKER CLUES**:
   - **AGENT typically says**: Banking terminology, policy explanations, formal procedures, account details, payment instructions, "KredoBank", professional phrases
   - **CLIENT typically says**: Personal explanations, questions about their account, emotional responses, informal language, requests for help
   - **Formal address patterns**: "Пані/Пан [Name]" usually from agent to client
   - **Ukrainian politeness markers**: Note who uses formal vs informal speech

4. **VERIFICATION CROSS-CHECK**:
   - After assigning speakers to ALL stereo content, verify each assignment by asking:
     * "Does this phrase match the tone/content pattern in the AGENT channel?"
     * "Does this phrase match the tone/content pattern in the CLIENT channel?"
     * "Is this consistent with bank agent vs client behavior?"

5. **QUALITY CONTROL REQUIREMENTS**:
   - NO phrase from stereo transcript should be omitted
   - NO speaker should be assigned randomly - each assignment must be evidence-based
   - Maintain conversation flow and logical turn-taking
   - Preserve all natural speech elements (hesitations, fillers, repetitions)

6. **ENHANCEMENT GUIDELINES**: Use AGENT and CLIENT transcripts to:
   - Clarify unclear words from stereo (but keep original if unclear)
   - Confirm speaker identity through content matching
   - Resolve ambiguous speaker assignments
   - BUT NEVER remove or skip content from stereo

Output Format:
AG: [complete agent statement - verified against agent channel]
CL: [complete client statement - verified against client channel]
AG: [next agent statement - cross-referenced for accuracy]
CL: [next client statement - cross-referenced for accuracy]
...

FINAL VERIFICATION CHECKLIST before submitting:
✓ Every phrase from STEREO transcript appears in output
✓ Each speaker assignment is supported by channel evidence or strong contextual clues
✓ Conversation flow is logical and natural
✓ No content has been omitted, condensed, or summarized

Remember: Your accuracy in speaker identification is critical. Take time to cross-reference each utterance with the individual channel transcripts before making speaker assignments.

Do not include timestamps, metadata, or section headers. Output only the final reconstructed script with verified speaker labels.
"""

    if not model:
        model = _default_split_into_roles_model()

    response = await async_chat_completion(
        model=model,    
        temperature=0,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, 
                  {"role": "user", "content": USER_PROMPT}]       
        )
    
    log.info(f"Split into roles done with model: {model}:")
    log.info("\n" + str(response.usage))   

    result = response.choices[0].message.content
    result = result.replace("```plaintext", "").replace("```", "").replace("\n\n","\n").strip()
    return result



SCHEMA_SPEAKER_DETECTION = {
    "type": "object",
    "properties": {
        "text1_speaker": {
            "type": "string",
            "enum": ["AGENT", "CLIENT"],
            "description": "Whether text 1 is spoken by AGENT or CLIENT"
        },
        "text2_speaker": {
            "type": "string",
            "enum": ["AGENT", "CLIENT"],
            "description": "Whether text 2 is spoken by AGENT or CLIENT"
        },
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
            "description": "Confidence level of the classification"
        },
        "reasoning": {
            "type": "string",
            "description": "Explanation of why this classification was made"
        }
    },
    "required": ["text1_speaker", "text2_speaker", "confidence", "reasoning"],
    "additionalProperties": False
}

async def async_detect_speaker_roles(text1: str, text2: str, model: str = "") -> dict:
    """
    Detect which transcribed text belongs to the bank AGENT and which to the CLIENT.
    
    Args:
        text1: First transcribed text
        text2: Second transcribed text
        model: OpenAI model to use
        
    Returns:
        Dictionary with speaker classifications and reasoning
    """
    # Construct the prompt

    SYSTEM_PROMPT = "You are an expert at analyzing call transcripts and identifying speaker roles in banking conversations."

    USER_PROMPT = f"""You are analyzing transcripts from a phone call between a bank agent and a client.

TEXT 1:
{text1}

TEXT 2:
{text2}

Analyze both texts and determine which speaker is the bank AGENT and which is the CLIENT.

Consider these clues:
- Agents typically introduce themselves, mention the bank name, ask how they can help
- Agents provide solutions, explain policies, access account information
- Clients describe problems, ask questions about their accounts, request services
- Agents use more formal/professional language patterns
- Clients may express frustration, confusion, or gratitude

Classify each text as either AGENT or CLIENT."""


    if not model:
        model = _default_detect_speaker_role_model()

    # Call the API with structured output
    response = await async_chat_completion_with_format(
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, 
                  {"role": "user", "content": USER_PROMPT}],      
        format_schema=SCHEMA_SPEAKER_DETECTION,
        schema_name="speaker_detection",
        model=model,
        temperature=0.0,
    )
    log.info(f"Speaker detected with model {model}:")
    log.info("\n" + str(response.usage))   

    result = json.loads(response.choices[0].message.content)
    print(f"Speaker detection result: {result}")
    
    agent_text = text1 if result['text1_speaker'] == 'AGENT' else text2
    client_text = text2 if result['text2_speaker'] == 'CLIENT' else text1

    return agent_text, client_text





def add_prefix_to_sentences(text, prefix):   
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    prefixed_sentences = []
    for sentence in sentences:
        if sentence.strip():
            prefixed_sentences.append(f"{prefix} {sentence.strip()}")
    
    return ' '.join(prefixed_sentences)

def consolidate_dialogue(dialogue_text):
    lines = dialogue_text.strip().split('\n')
    consolidated = []
    current_speaker = None
    current_content = []
    
    for line in lines:
        line = line.strip()
        if not line:  # Skip empty lines
            continue
            
        # Check if line starts with AG: or CL:
        if line.startswith('AG:'):
            speaker = 'AG'
            content = line[3:].strip()  # Remove 'AG:' prefix
        elif line.startswith('CL:'):
            speaker = 'CL'
            content = line[3:].strip()  # Remove 'CL:' prefix
        else:
            # Line doesn't have speaker label, skip or treat as continuation
            continue
        
        # If same speaker as previous line, accumulate content
        if speaker == current_speaker:
            current_content.append(content)
        else:
            # Different speaker, save previous speaker's consolidated content
            if current_speaker and current_content:
                consolidated_line = f"{current_speaker}: {' '.join(current_content)}"
                consolidated.append(consolidated_line)
            
            # Start new speaker section
            current_speaker = speaker
            current_content = [content]
    
    # Don't forget the last speaker's content
    if current_speaker and current_content:
        consolidated_line = f"{current_speaker}: {' '.join(current_content)}"
        consolidated.append(consolidated_line)
    
    return '\n'.join(consolidated)

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







SCHEMA_SINGLE_SPEAKER_ROLE = {
    "type": "object",
    "properties": {
        "speaker": {
            "type": "string",
            "enum": ["AGENT", "CLIENT"],
            "description": "Whether the provided text is spoken by a bank AGENT or a CLIENT"
        },
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"]
        },
        "reason": {
            "type": "string",
            "description": "Short reason for the classification"
        }
    },
    "required": ["speaker", "confidence", "reason"],
    "additionalProperties": False
}

async def async_classify_agent_or_client_prefix(text: str, model: str = "") -> str:
    """
    Classify a single transcript text block as AGENT or CLIENT and return prefix:
      - 'AGENT' if agent
      - 'CLIENT' if client
    """
    system_prompt = (
        "You are an expert at analyzing bank call transcripts and identifying whether the speaker "
        "is a bank agent or a bank client."
    )

    user_prompt = f"""You are analyzing ONE transcript block from a phone call.

TEXT:
{text}

Decide whether this text is spoken by the bank AGENT or by the CLIENT.

Clues:
- AGENT: introduces themselves, mentions bank/company, policy/procedure, verification, payment instructions, professional tone
- CLIENT: asks what is going on, describes personal situation, reacts emotionally, requests help, complains/confusion

Return only the structured classification.
"""

    if not model:
        model = (
            settings.AZURE_MODEL_CHAT_TRS_DETECT_PLAYER
            if settings.USE_AZURE_OPENAI == "Y"
            else settings.OPENAI_MODEL_CHAT_TRS_DETECT_PLAYER
        )

    resp = await async_chat_completion_with_format(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        format_schema=SCHEMA_SINGLE_SPEAKER_ROLE,
        schema_name="single_speaker_role",
        model=model,
        temperature=0.0,
    )

    result = json.loads(resp.choices[0].message.content)
    return result["speaker"]
    return "AG" if result["speaker"] == "AGENT" else "CL"

