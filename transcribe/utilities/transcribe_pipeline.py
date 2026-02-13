# transcribe/utilities/transcribe_mono.py

from __future__ import annotations
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple
from datetime import date
import json

from pydub import AudioSegment

from core.config import settings
from core.logger import log

from transcribe.utilities.summary_tools import generate_crm_summary_for_call_scenario
from transcribe.utilities.transcribe_mono import transcribe_mono_audio_file_to_scenario
from transcribe.utilities.transcribe_stereo import transcribe_stereo_audio_file_to_scenario
from transcribe.utilities.evaluation_engine import run_scheme
from transcribe.utilities.evaluation_engine_regs import load_active_scheme
from transcribe.utilities.call_analysis_engine import async_analyze_transcription_questions



# -----------------------------
# Universal wrapper: AUTO stereo/mono -> scenario
# -----------------------------
def transcribe_audio_file_to_scenario_pipeline(
    *,
    source_file: str,
    metadata: Any = None,
    temp_root_dir: str = None,
    temperature: float = 0.0,
    timeout: float = 120.0,
    force_mono: bool = False,
) -> str:
    """
    One entry point for your app:
      - If stereo and not force_mono -> use your existing stereo pipeline
      - Else -> mono pipeline above
    """
    audio = AudioSegment.from_file(source_file)
    is_stereo = (audio.channels == 2)

    if not temp_root_dir:
        temp_root_dir = settings.TR_TEMP_ROOT_DIR

    if is_stereo and not force_mono:
        return transcribe_stereo_audio_file_to_scenario(source_file=source_file,
                                                        temp_root_dir=temp_root_dir,
                                                        metadata=metadata 
                                                        )



    return transcribe_mono_audio_file_to_scenario(  source_file=source_file,
                                                    temp_root_dir=temp_root_dir,
                                                    metadata=metadata,
                                                    temperature=temperature,
                                                    timeout=timeout
                                                    )




def generate_scenario_summary_pipeline(
    *,
    scenario: str, 
    model_override: str = None) -> str:
      
    log.info("\n\n\n" + "="*30 + " Generating summary " + "="*30)
    summary = generate_crm_summary_for_call_scenario(scenario, model=model_override)
    log.info("\n" + summary)
    return summary



def evaluate_transcripted_scenario_pipeline(
    *,
    scenario: str,
    metadata: Any = None,
    system_code: str,
    call_date: date,
    call_info: Optional[Dict[str, Any]] = None,
    prev_result: Optional[Dict[str, Any]] = None,
    model_override: Optional[str] = None,
):      
    # run_scheme expects metadata to be a string (it calls `.strip()` in prompts)
    if metadata is None:
        metadata_text = ""
    elif isinstance(metadata, str):
        metadata_text = metadata
    else:
        # keep unicode (UA/RU names etc.) readable in logs/prompts
        try:
            metadata_text = json.dumps(metadata, ensure_ascii=False)
        except Exception:
            metadata_text = str(metadata)

    log.info("\n\n" + "=" * 30 + "\nLoading current evaluation scheme " + "=" * 30)

    scheme = load_active_scheme(
        config_root=settings.TR_EVALUATION_CONFIGS_ROOT,
        system_code=system_code,
        call_date=call_date,
        call_info=call_info,
        prev_result=prev_result,
    )


    log.info(f"\nUsing scheme: {scheme.system_code} v{scheme.version}\n")
    log.debug(
        f"\n{json.dumps(asdict(scheme), indent=2, ensure_ascii=False)}"
    )

    log.info("\n\n" + "=" * 30 + "\nRunning evaluation scheme " + "=" * 30)
    result, success = run_scheme(
        transcript_text=scenario,
        metadata=metadata_text,
        scheme=scheme,
        model_override=model_override,
        prev_result=prev_result,
    )

    return result, success



async def async_run_analysis_of_the_transcription_pipeline(
    request_json: str | Dict[str, Any],
    *,
    model_override: str = "",
    parallel_requests: int = None,
    prev_result: Optional[Dict[str, Any]] = None,
    timeout: float = 120.0,
) -> Tuple[Dict[str, Any], bool]:
    
    result, success = await async_analyze_transcription_questions(
        request_json=request_json,
        model=model_override,
        parallel_requests=parallel_requests,  
        prev_result=prev_result,
        timeout=timeout,
    )
    return result, success
