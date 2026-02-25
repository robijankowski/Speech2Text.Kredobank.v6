from __future__ import annotations
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple
from datetime import date
import json
import asyncio

from pydub import AudioSegment

from app.core.config import settings
from app.core.logger import log

from app.transcribe.utilities.scenario_tools import Turn

from app.transcribe.utilities.transcribe_stereo_v2 import async_transcribe_stereo_hq_roles_to_turns_v2
from app.transcribe.utilities.summary_tools import async_generate_crm_summary_for_call_scenario_ext
from app.transcribe.utilities.evaluation_interrupts import analyze_turn_overlaps 
from app.transcribe.utilities.call_analysis_engine import async_analyze_transcription_questions
from app.transcribe.utilities.evaluation_engine_regs import load_active_scheme 
from app.transcribe.utilities.evaluation_engine import async_run_scheme



# -----------------------------
# Universal wrapper: AUTO stereo/mono -> scenario
# -----------------------------
async def async_transcribe_audio_file_to_scenario_pipeline(
    *,
    source_file: str,
    metadata: Any = None,
    temp_root_dir: str = None
) -> Tuple[List[Turn], str]:
    """
    One entry point for your app:
      - If stereo and not force_mono -> use your existing stereo pipeline
      - Else -> mono pipeline above
    """
    log.info("\n\n" + "="*20 + f" Starting transcription of the file: '{source_file}' " + "="*20)

    temp_root_dir = settings.TR_TEMP_ROOT_DIR

    diar_segs_turns, scenario = await async_transcribe_stereo_hq_roles_to_turns_v2(source_file=source_file,
                                                                            temp_root_dir=temp_root_dir,
                                                                            metadata=metadata 
                                                                            )
                                                                
    log.info(f"\n\nFINAL RESULT - SCENARIO FROM THE TRANSCRIPTION OF: '{source_file}':\n\n" + str(scenario) + "\n")
    return diar_segs_turns, scenario 






async def async_generate_scenario_summary_pipeline(
    scenario: str, 
    model_override: str = None) -> str:
      
    log.info("\n\n" + "="*20 + " Starting generation of the summary " + "="*20)
    summary = await async_generate_crm_summary_for_call_scenario_ext(scenario, model=model_override)
    log.info("\n\nFINAL RESULT - SUMMARY\n\n" + json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return json.dumps(summary, ensure_ascii=False, indent=2)






async def async_evaluate_conversation_interrupts_pipeline(
    diar_segs_turns: List[Turn] = None,
    file_name: str = "",
    model_override: str = ""
) -> Dict[str, Any]:
    """Rule-based interruption/overlap evaluation for LR timestamped turns.

    Returns a dict with the SAME SHAPE as `evaluation_engine.run_check(...)`.
    This is intentionally model-free (deterministic), but keeps the `model` field
    for downstream compatibility.
    """

    log.info("\n\n" + "="*20 + " Starting interrupts analysis " + "="*20)

    if not settings.TR_EVALUATE_INTERRUPTS == "Y":
        log.info(f"Interrupts analysis canceled due to: settings.TR_EVALUATE_INTERRUPTS = '{settings.TR_EVALUATE_INTERRUPTS}'")
        return None
    


    overlaps_res = await asyncio.to_thread(
        analyze_turn_overlaps,
        diar_segs_turns,
        # IMPORTANT: disable internal per-role merge to avoid spanning across the other speaker :contentReference[oaicite:7]{index=7}
        merge_gap_ms_agent=0,
        merge_gap_ms_client=0,
        # keep your existing thresholds:
        min_overlap_ms=450,
        eps_ms=30,
        min_agent_segment_ms=200,
        min_client_segment_ms=800,
        min_words_agent=1,
        min_words_client=2,   # (often you can lower this if drop_backchannels=True)
        ignore_tail_ms_ag=1500,
        ignore_tail_ms_cl=100,
    )

    # overlaps_res = await asyncio.to_thread(
    #     analyze_turn_overlaps,
    #     diar_segs_turns,
    #     min_overlap_ms=450,
    #     eps_ms=30,
    #     min_agent_segment_ms=200,
    #     min_client_segment_ms=800,
    #     min_other_lead_ms=0,
    #     min_segment_ms_agent=200,
    #     min_segment_ms_client=200,
    #     min_words_agent=1,
    #     min_words_client=2,
    #     ignore_tail_ms_ag=1500,
    #     ignore_tail_ms_cl=100,
    # )

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


    res = {
        "id": "2_03_conversation_interrupts",
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
        }
    }

    log.info(f"\n\nFINAL RESULT - THE INTERRUPTS ANALYSIS:\n\n" + json.dumps(res, ensure_ascii=False, indent=2) + "\n")
    return res
    





async def async_evaluate_transcripted_scenario_pipeline(
    scenario: str,
    system_code: str,
    call_date: date,
    metadata: Any = None,
    call_info: Optional[Dict[str, Any]] = None,
    prev_result: Optional[Dict[str, Any]] = None,
    model_override: Optional[str] = None,
    interrupts_analysis: Optional[Dict[str, Any]] = None,
):      
    log.info("\n\n" + "="*20 + " Starting call scoring calculations " + "="*20)

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

    config_root = settings.TR_EVALUATION_CONFIGS_ROOT
    log.info(f"Loading current evaluation scheme for system: '{system_code}' from: '{config_root}' ")

    scheme = load_active_scheme(
        config_root=config_root,
        system_code=system_code,
        call_date=call_date,
        call_info=call_info,
        prev_result=prev_result,
    )


    log.info(f"Using scheme: '{scheme.system_code}' v{scheme.version} from config root: '{config_root}'")
    log.debug(
        f"Scoring will be called with following parameters:"
        f"\n\nConfiguration:\n{json.dumps(asdict(scheme), indent=2, ensure_ascii=False)}"
        f"\n\nCall info:\n{json.dumps(call_info, indent=2, ensure_ascii=False)}"
        f"\n\nMetadata:\n{metadata}"
        f"\n\nPrevResult:\n{json.dumps(prev_result, indent=2, ensure_ascii=False)}"
        f"\n\nScenario:\n{scenario}\n"
    )

    log.info(f"Starting to run evaluation scheme for: '{scheme.system_code}' v{scheme.version} from config root: '{config_root}'")
    result, success = await async_run_scheme(
        transcript_text=scenario,
        metadata=metadata_text,
        scheme=scheme,
        model_override=model_override,
        prev_result=prev_result,
        interrupts_analysis=interrupts_analysis
    )

    log.info(f"\n\nFINAL RESULT - THE SCORE CALCULATIONS:\n\nSuccess indicator: '{success}'\nResult data:" + json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    return result, success





async def async_run_analysis_of_the_transcription_pipeline(
    request_json: str | Dict[str, Any],
    model_override: str = "",
    parallel_requests: int = None,
    prev_result: Optional[Dict[str, Any]] = None,
    timeout: float = 120.0,
) -> Tuple[Dict[str, Any], bool]:
    
    log.info("\n\n" + "="*20 + " Starting scenario daily/night batch analysis " + "="*20)

    result, success = await async_analyze_transcription_questions(
        request_json=request_json,
        model=model_override,
        parallel_requests=parallel_requests,  
        prev_result=prev_result,
        timeout=timeout,
    )
    res_info_text = ""
    for a in result["answers"]:
        res_info_text += f'{a["questionId"]}: {a["status"]} -> {a.get("answer")}\n'
    log.info(f"\n\nFINAL RESULT - THE SCENARIO DAILY/NIGHT BATCH ANALYSIS:"
             f"\n\nSUCCESS indicator: '{success}"
             f"\n\nQuestions status:\n{res_info_text}"
             "\nResult json data:\n" + json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result, success
