# transcribe/utilities/transcribe_mono.py

from __future__ import annotations
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple
from datetime import date
import json

from pydub import AudioSegment

from core.config import settings
from core.logger import log

from transcribe.utilities.scenario_tools import Turn

from transcribe.utilities.summary_tools import async_generate_crm_summary_for_call_scenario
from transcribe.utilities.transcribe_mono_lr import async_transcribe_mono_audio_file_to_scenario_lr
from transcribe.utilities.transcribe_stereo_lr import async_transcribe_stereo_lr_timestamped
from transcribe.utilities.evaluation_interrupts_lr import analyze_turn_overlaps_lr # no async need - just calc
from transcribe.utilities.evaluation_engine import async_run_scheme

from transcribe.utilities.evaluation_engine_regs import load_active_scheme # no async need - just calc

from transcribe.utilities.call_analysis_engine import async_analyze_transcription_questions



# -----------------------------
# Universal wrapper: AUTO stereo/mono -> scenario
# -----------------------------
async def async_transcribe_audio_file_to_scenario_pipeline_lr(
    *,
    source_file: str,
    metadata: Any = None,
    temp_root_dir: str = None,
    temperature: float = 0.0,
    timeout: float = 120.0,
    force_mono: bool = False,
) -> Tuple[List[Turn], str]:
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
        return await async_transcribe_stereo_lr_timestamped(source_file=source_file,
                                                        temp_root_dir=temp_root_dir,
                                                        metadata=metadata 
                                                        )



    return await async_transcribe_mono_audio_file_to_scenario_lr(  source_file=source_file,
                                                    temp_root_dir=temp_root_dir,
                                                    metadata=metadata,
                                                    temperature=temperature,
                                                    timeout=timeout
                                                    )




async def async_generate_scenario_summary_pipeline_lr(
    *,
    scenario: str, 
    model_override: str = None) -> str:
      
    log.info("\n\n\n" + "="*30 + " Generating summary " + "="*30)
    summary = await async_generate_crm_summary_for_call_scenario(scenario, model=model_override)
    # log.info("\n" + str(summary))
    return summary





def evaluate_conversation_interrupts_pipeline_lr(
    turns: List[Turn],
    model_override: str = ""
) -> Dict[str, Any]:
    """Rule-based interruption/overlap evaluation for LR timestamped turns.

    Returns a dict with the SAME SHAPE as `evaluation_engine.run_check(...)`.
    This is intentionally model-free (deterministic), but keeps the `model` field
    for downstream compatibility.
    """
    if not settings.TR_EVALUATE_INTERRUPTS == "Y":
        return None
    
    overlaps_res = analyze_turn_overlaps_lr(
        turns,
        min_overlap_ms=450,
        eps_ms=30,
        min_agent_segment_ms=200,
        min_client_segment_ms=800,
        min_other_lead_ms=0,
        min_segment_ms_agent=200,
        min_segment_ms_client=200,
        min_words_agent=1,
        min_words_client=4,
        # if CL starts talking before AG finished, ignore up to 1500ms of AG tail as possible overlap
        ignore_tail_ms_ag=1500,
        # if AG starts talking before CL finished, ignore up to 100ms of CL tail as possible overlap
        ignore_tail_ms_cl=100,
    )

    stats = (overlaps_res or {}).get("stats") or {}
    any_overlaps = int(stats.get("any_overlaps") or 0)
    agent_interrupts = int(stats.get("agent_interrupts") or 0)
    client_interrupts = int(stats.get("client_interrupts") or 0)
    total_interrupts = agent_interrupts + client_interrupts

    # --- scoring (simple + stable) ---
    max_points = 1
    weight = 1.0
    score = 1
    if any_overlaps>1:
        score = 0

    model_label = model_override or "rules"


    return {
        "id": "conversation_interrupts_lr",
        "desc": "Conversation overlaps / interruptions (LR timestamped, rule-based)",
        "score": int(score),
        "max_points": int(max_points),
        "weight": float(weight),
        "weighted_score": float(score) * float(weight),
        "weighted_max": float(max_points) * float(weight),
        "model": model_label,
        "raw": {
            "overlaps": overlaps_res["overlaps"],
            "events": overlaps_res["events"],
            "stats": overlaps_res["stats"],
        },
    }



async def async_evaluate_transcripted_scenario_pipeline_lr(
    scenario: str,
    system_code: str,
    call_date: date,
    metadata: Any = None,
    call_info: Optional[Dict[str, Any]] = None,
    prev_result: Optional[Dict[str, Any]] = None,
    model_override: Optional[str] = None,
    interrupts_analysis: Optional[Dict[str, Any]] = None,
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
    result, success = await async_run_scheme(
        transcript_text=scenario,
        metadata=metadata_text,
        scheme=scheme,
        model_override=model_override,
        prev_result=prev_result,
        interrupts_analysis=interrupts_analysis
    )

    return result, success



async def async_run_analysis_of_the_transcription_pipeline(
    request_json: str | Dict[str, Any],
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
