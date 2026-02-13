from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Tuple


from core.config import settings
from core.logger import log



from transcribe.utilities.scenario_tools import ( split_transcription_into_roles_4o, 
                                                 consolidate_dialogue, 
                                                 detect_speaker_roles, 
                                                 add_prefix_to_sentences )
from transcribe.utilities.audio_tools import prepare_audio_for_transcription
from transcribe.utilities.transcribe_stereo_tools import ( transcript_audio_file_verbose_o4_stereo, 
                                                          transcript_audio_file_verbose_o4_single_channel )



def transcribe_stereo_audio_file_to_scenario(
    *,
    source_file: str,
    temp_root_dir: str,
    metadata: Any = None,
) -> str:

    log.info("\n" + "=" * 60)
    log.info(f"\nStereo WAV File Splitter and Transcription Tool - FILE: {source_file}")

    l_file_cleaned, r_file_cleaned, s_file_cleaned = prepare_audio_for_transcription(source_file, temp_root_dir)

    if l_file_cleaned and r_file_cleaned:
        log.info("\n\n" + "="*30 + f" Transcribe O4 cleaned left channel wav " + "="*30)
        o4_left_trans = transcript_audio_file_verbose_o4_single_channel(l_file_cleaned, metadata)
        log.info(f"\n{o4_left_trans.text}")

        log.info("\n\n" + "="*30 + f" Transcribe O4 cleaned right channel wav as " + "="*30)
        o4_right_trans = transcript_audio_file_verbose_o4_single_channel(r_file_cleaned, metadata)
        log.info(f"\n{o4_right_trans.text}")

        log.info("\n\n" + "="*30 + f" Transcribe O4 cleaned stereo wav " + "="*30)
        o4_stereo_trans = transcript_audio_file_verbose_o4_stereo(s_file_cleaned, metadata)
        log.info(f"\n{o4_stereo_trans.text}")
    else:
        raise ValueError("The audio could not be prepared for transcription. Please ensure that the file is stereo and of sufficient quality.")

    log.info("\n\n" + "="*30 + " Detecting speaker roles in transcription " + "="*30)
    agent_text, client_text = detect_speaker_roles( o4_left_trans.text, o4_right_trans.text )
    agent_text = add_prefix_to_sentences(agent_text, "AG:")
    client_text = add_prefix_to_sentences(client_text, "CL:")

    log.info("\n\n" + "="*30 + " Modified AGENT for o4 " + "="*30)
    log.info("\n" + agent_text.replace("AG:", "\nAG:"))
    log.info("\n\n" + "="*30 + " Modified CLIENT for o4 " + "="*30)
    log.info("\n" + client_text.replace("CL:", "\nCL:"))


    log.info("\n\n\n" + "="*30 + " Generating roles/scenario with o4 " + "="*30)
    scenario_granular = split_transcription_into_roles_4o( agent_text = agent_text, 
                                                            client_text = client_text, 
                                                            stereo_text = o4_stereo_trans.text )
    scenario = consolidate_dialogue(scenario_granular)
    log.info("\n" + "="*30 + " o4 consolidated scenario "  + "="*30)
    log.info("\n" + scenario)

    return scenario
