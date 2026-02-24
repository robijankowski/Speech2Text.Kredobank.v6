import json
from typing import Any, Dict, List, Tuple, Optional

from app.core.config import settings
from app.core.logger import log

from app.transcribe.utlities.scenario_tools import (DiarSeg, 
                                                 _default_detect_speaker_role_model,
                                                 _default_transcription_model,
                                                 _default_split_into_roles_model)


from app.openai_tools.openai_client_transcribe import async_transcribe_audio, Transcription
from app.openai_tools.openai_client_text import async_chat_completion, async_chat_completion_with_format





SINGLE_CHANNEL_UNKNOWN_ROLE_PROMPT_EN = """
This is an isolated single-channel recording from a phone call between a KredoBank Ukraine collections AGENT and a CLIENT.
Only ONE participant is audible in this file, but you DO NOT know whether it is the AGENT or the CLIENT.

Your task: produce a verbatim transcript of the audible speaker ONLY.

CRITICAL: PRESERVE THE ORIGINAL SPOKEN LANGUAGE (NO TRANSLATION)
- Transcribe EXACTLY what is said, without translating or "correcting" language.
- If a word is spoken in Ukrainian → write it in Ukrainian.
- If a word is spoken in Russian → write it in Russian.
- If speech is mixed (code-switching / surzhyk) → preserve each word in its original language.
- NEVER replace Russian words with Ukrainian equivalents (e.g., do not change "да" to "так").

SINGLE-SPEAKER RULES
- Do NOT invent the other speaker’s lines.
- Do NOT add placeholders like “(other speaker)” or imagined replies.
- If the speaker repeats themselves, hesitates, or uses fillers (“ну”, “ммм”, “это”, “вот”, etc.), keep them.

PUNCTUATION & STYLE
- Add natural punctuation and capitalization suitable for spoken dialogue.
- Keep numbers, names, dates, amounts exactly as spoken.
- If a word is unclear, keep your best guess; if truly unintelligible, use [unclear].

CALL CONTEXT (may appear in either language)
- Debt collection / arrears / negative balance / account restriction / enforcement topics.
- Typical terms: "від'ємний баланс" / "отрицательный баланс", "рахунок" / "счет", "арешт" / "арест"
- Typical phrases: "Алло", "Мене звати" / "Меня зовут", "Наша розмова записується" / "Наш разговор записывается"

Context metadata to help recognition (names, bank, etc.):
Known entities canonical names: {metadata}
"""


STEREO_PROMPT_UA = """
Це запис розмови між клієнтом та оператором KredoBank Україна.

🔸 КРИТИЧНО ВАЖЛИВО - ЗБЕРЕЖЕННЯ ОРИГІНАЛЬНОЇ МОВИ:
- Транскрибуй ТОЧНО ТАК, ЯК СКАЗАНО, без жодних перекладів. Наприклад, не замінюйте слово «да» на «так».
- Якщо слово вимовлено українською → записуй українською
- Якщо слово вимовлено російською → записуй російською  
- Якщо фраза змішана (суржик) → зберігай кожне слово в оригінальній мові
- НІКОЛИ не замінюй російські слова українськими еквівалентами
- НІКОЛИ не "виправляй" мову - записуй автентично

Цей виклик, ймовірно, пов'язаний із роботою відділу стягнення заборгованості банку, 
і може містити згадки про заборгованість, від'ємний баланс, арешт рахунків, виконавчу службу тощо.

🔹 Контекстні дані (метадані) для покращення розпізнавання:
Known entities canonical names: {metadata}

🔹 У розмові можуть зустрічатися ОБОМА МОВАМИ:
- Банківські терміни: "від'ємний баланс" / "отрицательный баланс", "рахунок" / "счет", "арешт" / "арест"
- Звернення: "добрий день" / "добрый день", "дякую" / "спасибо"
- Known entities canonical names: {metadata}
- Назви банківських додатків: "KredoBank", "Кредобанк", "KredoMobile"
- Службові фрази: "Алло", "Мене звати" / "Меня зовут", "Наша розмова записується" / "Наш разговор записывается"

🔹 ПРИКЛАДИ правильної транскрипції змішаного мовлення:
- "Добрий день, у мене отрицательный баланс на счету" ✓
- "Алло, можете сказать, чому в мене від'ємний баланс?" ✓  
- "Спасибо, до побачення" ✓

🔹 Інструкції для транскрипції:
- Зберігай всі слова, включно зі словами-паразитами ("ну", "добре", "може", "трошки", "ммм", "это", "вот")
- Не пропускай повторення або вагання
- Використовуй правильну пунктуацію для природного усного мовлення
- Мовна автентичність важливіша за мовну "чистоту". 
- ЗАВЖДИ ЗБЕРІГАЙТЕ РОСІЙСЬКІ СЛОВА, ЯКЩО ВОНИ З'ЯВИЛИСЯ В РОЗМОВІ!

ПАМ'ЯТАЙ: Твоє завдання - точно відтворити ТЕ, ЩО БУЛО СКАЗАНО, а не те, що "повинно було б" бути сказано!
"""




async def async_transcript_audio_file_verbose_o4_single_channel(
    file_name: str,
    o4_metadata_text: str = "",
    temperature: float = 0.0,
    model: str= "",
) -> Transcription:
    """
    Transcribe a single isolated channel (only one speaker audible),
    without assuming whether it is AGENT or CLIENT.
    """
    prompt = SINGLE_CHANNEL_UNKNOWN_ROLE_PROMPT_EN.format(metadata=o4_metadata_text or "")

    if not model:
        model = _default_transcription_model()

    transcription = await async_transcribe_audio(
        audio=file_name,
        model=model,
        prompt=prompt,
        temperature=temperature,
        response_format="json",
        timestamp_granularities=["segment"],
    )

    log.info(f"Single channel transcription done with model: {model}:")
    log.info("\n" + str(transcription.usage))   

    return transcription


async def async_transcript_audio_file_verbose_o4_stereo(file_name: str, 
                                            o4_metadata_text="", 
                                            temperature=0.0,
                                            model: str = ""
                                            ) -> Transcription:
    
    prompt = STEREO_PROMPT_UA.format(metadata=o4_metadata_text or "")

    if not model:
        model = _default_transcription_model()

    transcription = await async_transcribe_audio(
        audio=file_name,
        model=model,
        prompt=prompt,
        temperature=temperature,
        response_format="json",
        timestamp_granularities=["segment"],
    )
    log.info(f"Stereo file transcription done with model: {model}:")
    log.info("\n" + str(transcription.usage))   

    return transcription




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




# Schema: classify EACH detected speaker as AGENT or CLIENT
SCHEMA_MULTI_SPEAKER_ROLES = {
    "type": "object",
    "properties": {
        "roles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "speaker": {"type": "string"},
                    "role": {"type": "string", "enum": ["AGENT", "CLIENT"]},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "reason": {"type": "string"},
                },
                "required": ["speaker", "role", "confidence", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["roles"],
    "additionalProperties": False,
}


async def async_classify_all_speakers_agent_or_client(
    segs: List[DiarSeg],
    max_chars_per_speaker: int = 2400,
) -> Dict[str, str]:
    """
    Classify each diarized speaker as AG (agent) or CL (client).
    Input: List[DiarSeg] = (start, end, text, speaker)
    Output: mapping like {"A":"AGENT", "B":"CLIENT", ...}
    """
    if not segs:
        return {}

    # group by speaker
    by: Dict[str, List[DiarSeg]] = {}
    for st, en, txt, spk in segs:
        txt = (txt or "").strip()
        if not txt:
            continue
        spk = str(spk or "S0").strip()
        by.setdefault(spk, []).append((float(st), float(en), txt, spk))

    speakers = sorted(by.keys())
    if not speakers:
        return {}

    def speaker_stats(items: List[DiarSeg]) -> Tuple[float, int]:
        dur = 0.0
        words = 0
        for st, en, txt, _ in items:
            dur += max(0.0, float(en) - float(st))
            words += len((txt or "").split())
        return dur, words

    def speaker_excerpt(items: List[DiarSeg], max_chars: int) -> str:
        # chronological, concatenate until max chars
        items = sorted(items, key=lambda x: (x[0], x[1]))
        out = []
        total = 0
        for _, _, txt, _ in items:
            if not txt:
                continue
            piece = txt.strip()
            if not piece:
                continue
            add = (piece + "\n")
            if total + len(add) > max_chars:
                remaining = max_chars - total
                if remaining > 0:
                    out.append(add[:remaining])
                break
            out.append(add)
            total += len(add)
        return "".join(out).strip()

    blocks: List[str] = []
    for spk in speakers:
        dur, words = speaker_stats(by[spk])
        excerpt = speaker_excerpt(by[spk], max_chars=max_chars_per_speaker)
        blocks.append(
            f"SPEAKER {spk}\n"
            f"- total_duration_sec: {dur:.2f}\n"
            f"- approx_words: {words}\n"
            f"- excerpt:\n{excerpt}\n"
        )

    system_prompt = (
        "You classify speakers in bank calls.\n"
        "For EACH speaker, output role AGENT or CLIENT.\n"
        "Return JSON only."
    )
    # user_prompt = (
    #     "Diarized speakers from one call:\n\n"
    #     + "\n".join(blocks)
    #     + "\nReturn JSON: {\"roles\":[{\"speaker\":\"A\",\"role\":\"AGENT\"},...]}"
    # )
    user_prompt = (
        "Diarized speakers from one call.\n"
        f"Valid speaker ids are ONLY: {', '.join(speakers)}.\n"
        "In the JSON, 'speaker' MUST be exactly one of those ids (e.g., 'A', not 'SPEAKER A').\n\n"
        + "\n".join(blocks)
        + "\nReturn JSON: {\"roles\":[{\"speaker\":\"A\",\"role\":\"AGENT\",\"confidence\":\"high\",\"reason\":\"...\"},...]}"
    )
    # keep your existing schema + call helper
    model = _default_detect_speaker_role_model()
    resp = await async_chat_completion_with_format(
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}],
        format_schema=SCHEMA_MULTI_SPEAKER_ROLES,
        schema_name="multi_speaker_roles",
        model=model,
        temperature=0.0,
    )

    data = json.loads(resp.choices[0].message.content)
    log.info("\nRoles classification analysis result:\n" + str(data))
    mapping: Dict[str, str] = {}
    for row in data.get("roles", []):
        spk = str(row.get("speaker", "")).strip()
        role = str(row.get("role", "")).strip().upper()
        if not spk:
            continue
        mapping[spk] = role

    # ensure all speakers covered (fallback CL)
    for spk in speakers:
        mapping.setdefault(spk, "CLIENT")

    return mapping



