import logging
from transcribe.core.tr_config import tr_settings

log = logging.getLogger(tr_settings.TR_LOGGER_NAME)

from openai_tools.openai_client_transcribe import transcribe_audio, Transcription

from core.config import settings
from transcribe.utilities.stats import set_stats




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




def transcript_audio_file_verbose_o4_single_channel(
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
        model = settings.AZURE_MODEL_TRANSCRIBE_STEREO if settings.USE_AZURE_OPENAI == "Y" else settings.OPENAI_MODEL_TRANSCRIBE_STEREO

    transcription = transcribe_audio(
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



def transcript_audio_file_verbose_o4_stereo(file_name: str, 
                                            o4_metadata_text="", 
                                            temperature=0.0,
                                            model: str = ""
                                            ) -> Transcription:
    
    prompt = STEREO_PROMPT_UA.format(metadata=o4_metadata_text or "")

    if not model:
        model = settings.AZURE_MODEL_TRANSCRIBE_STEREO if settings.USE_AZURE_OPENAI == "Y" else settings.OPENAI_MODEL_TRANSCRIBE_STEREO

    transcription = transcribe_audio(
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




