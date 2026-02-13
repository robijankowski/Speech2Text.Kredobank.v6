# transcribe/utilities/transcribe_mono.py

from __future__ import annotations
from cmath import log
from cmath import log
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pydub import AudioSegment


from transcribe.core.tr_config import tr_settings

from transcribe.utilities.summary_tools import generate_crm_summary_o4
from transcribe.utilities.transcribe_mono import transcribe_mono_audio_file_to_scenario
from transcribe.utilities.transcribe_stereo import transcribe_stereo_audio_file_to_scenario





# -----------------------------
# Universal wrapper: AUTO stereo/mono -> scenario
# -----------------------------
def transcribe_audio_file_to_scenario(
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
        temp_root_dir = tr_settings.TR_TEMP_ROOT_DIR

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




def generate_scenario_summary(
    *,
    scenario: str, ) -> str:
      
    log.info("\n\n\n" + "="*30 + " Generating summary " + "="*30)
    summary = generate_crm_summary_o4(scenario)
    log.info("\n" + summary)
    return summary




def evaluate_transcripted_scenario(
    *,
    scenario: str, 
    metadata: Any = None,
    ):

    log.info("\n\n" + "="*30 + " Loading current evaluation scheme " + "="*30)
    scheme = load_active_scheme(tr_settings.TR_EVALUATION_CONFIGS_ROOT, 
                                "kcc", 
                                call_date=date(2026, 1,15),
                                call_info=CALL_INFO)
    log.info(
        f"\nUsing scheme: {scheme.system_code} v{scheme.version}\n"
    )
    # log.debug(
    #     f"\n{json.dumps(asdict(scheme), indent=2, ensure_ascii=False)}"
    # )

    log.info("\n\n" + "="*30 + " Running evaluation scheme " + "="*30)
    result, success = run_scheme(transcript_text=scenario, 
                                    metadata=metadata, 
                                    scheme=scheme)
    log.info("\n\n" + "="*30 + " Evaluation Results " + "="*30)
    log.info(f"\n\n{success}\n\n" + str(json.dumps(result, indent=2)))
