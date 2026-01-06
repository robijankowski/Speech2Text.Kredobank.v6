from openai import OpenAI
import re
import base64

from transcribe.utilities.stats import set_stats

openai_client = OpenAI(api_key="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")



def transcript_audio_file_verbose_o4_diarize(
    file_name,
    temperature=0.0,
    language="uk",
    chunking_strategy="auto",
):
    """
    Uses gpt-4o-transcribe-diarize and prints diarization markers (speaker, start, end) per segment.
    Note: diarize model returns speaker annotations only with response_format="diarized_json"
    and does NOT support prompt or timestamp_granularities. :contentReference[oaicite:2]{index=2}
    """
    model = "gpt-4o-transcribe-diarize"
    print(f"Transcribing file: {file_name} using '{model}'")

    def _fmt_ts(seconds: float) -> str:
        # 00:00.00 formatting
        if seconds is None:
            return "??:??.??"
        m, s = divmod(float(seconds), 60.0)
        return f"{int(m):02d}:{s:05.2f}"

    def _get(obj, key, default=None):
        # supports dicts or objects
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _to_data_url(path: str, mime: str = "audio/wav") -> str:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    with open(file_name, "rb") as audio_file:
        transcript = openai_client.audio.transcriptions.create(
            file=audio_file,
            model=model,
            response_format="diarized_json",
            # language=language,
            temperature=temperature,
            chunking_strategy=chunking_strategy
            # extra_body={
            #     "known_speaker_names": ["agent"],
            #     "known_speaker_references": [_to_data_url(agent_file_name)],
            #     },
        )

    # Print usage if present (guarded)
    usage = _get(transcript, "usage")
    if usage is not None:
        print("Usage:", usage)

    # Print ALL diarization markers (speaker/start/end/text)
    segments = _get(transcript, "segments", []) or []
    print(f"Segments: {len(segments)}")
    speakers_seen = set()

    transcript_text = ""
    for i, seg in enumerate(segments, start=1):
        speaker = _get(seg, "speaker", "unknown")
        start = _get(seg, "start", None)
        end = _get(seg, "end", None)
        text = _get(seg, "text", "")

        speakers_seen.add(speaker)
        # print(f"[{i:04d}] {speaker} {_fmt_ts(start)}–{_fmt_ts(end)}: {text}")
        transcript_text += f"{speaker}: {text}\n"

    print("Speakers found:", ", ".join(map(str, sorted(speakers_seen))))
    print(transcript_text)
    
    return transcript_text






