import os
from datetime import datetime, date
import json
from dataclasses import asdict
import asyncio
from unittest import result

from core.config import settings
from openai_tools.openai_client_transcribe import Transcription

from core.logger import get_logger, shutdown_logger
log = get_logger(__name__)

from transcribe.utilities.scenario_tools import split_transcription_into_roles_4o, format_scenario_md, consolidate_dialogue, detect_speaker_roles, add_prefix_to_sentences
from transcribe.utilities.md_creator import create_documentation_md
from transcribe.utilities.summary_tools import generate_crm_summary_for_call_scenario, format_summary_md
from transcribe.utilities.evaluate_tools import evaluate_call, format_evaluation_results_md, format_evaluation_results
from transcribe.utilities.audio_tools import prepare_audio_for_transcription, stereo_to_mono
from transcribe.utilities.stats import get_stats, stats_json_to_csv_text, stats_save_json_to_csv, clean_file_names
from transcribe.utilities.transcribe_stereo_tools import transcript_audio_file_verbose_o4_stereo, transcript_audio_file_verbose_o4_single_channel
from transcribe.utilities.evaluation_engine import load_scheme, run_scheme
from transcribe.utilities.evaluation_engine_regs import load_active_scheme
from transcribe.utilities.evaluation_engine_qa import audit_evaluation_configs, format_audit_report_md, is_configuration_ok
from transcribe.utilities.call_analysis_engine import async_analyze_transcription_questions

AUDIO_FILES = [
    "AUTO-2025-06-30-09-05-380963799218-1096-1751263515.1528148-stereo1.wav",
    "AUTO-2025-06-30-10-08-380988442847-1087-1751267274.1529139-stereo1.wav",
    "AUTO-2025-06-30-10-17-380639093150-1006-1751267854.1529303-stereo1.wav",
    "AUTO-2025-06-30-12-05-380990805468-1098-1751274275.1530761-stereo1.wav",
    "OUT-2025-06-30-09-34-1099-0500814269-1751265256.1528626-stereo1.wav" 
]

O4_METADATA = [
    {"name": "Ivolo Olena Volodymyrivna", "agent": "Ulyana", "bank": "KredoBank Ukraine"},
    {"name": "Lukashchuk Serhii Mykolayivych", "agent": "Sviatoslav", "bank": "KredoBank Ukraine"},
    {},
    {"agent": "Ivanova", "bank": "KredoBank Ukraine"},
    {}
]

WHISPER_METADATA = [
    "Іволо Олена Володимирівна, Уляна, KredoBank, Україна",
    "'Лукашчук Сергій Миколаївич', 'Святослав', KredoBank, Україна",
    "",
    "'Іванна', KredoBank, Україна",
    ""
]



conversation_test_text = """
AG: Алло, добрий день.
CL: Алло, доброго дня. Скажіть, будь ласка, що це мені приходить, що в мене якась заборгованість?
AG: Зрозуміла. Це Іволо Олена Володимирівна, так?
CL: Так.
AG: Давайте я також представлюся. Мене звати Уляна, мій порядковий номер 1096, повідомляю, що наша розмова записується. По вашому рахунку виник від'ємний баланс на суму 8 гривень і 20 копійок.
CL: А я вже поповняла.
AG: Ви вже поповнили?
CL: Поповняла, так.
AG: Коли ви оплачували?
CL: Давненько трошки, точно не скажу. Ну, може місяць пройшов.
AG: Добре, давайте зараз я перегляну. Так, дякую за очікування. Я бачу, що у вас є арешт рахунки. Ви про це знаєте?
CL: Ні. Який арешт? На яку суму?
AG: У вас є арешт рахунки і потрібно звернутися у виконавчу службу, щоб...
CL: Мені приходить, що арешт знятий, бо в мене пару банків є мобільних додатків. Блокують, розблоковують, блокують, розблоковують.
AG: Я бачу, що в нас є арешт рахунків.
CL: На яку суму?
AG: Я не маю таких данів, не можу вам сказати. І тому у вас не виходить оплатити від'ємний баланс по рахунку, тому що є арешт рахунку.
CL: Добре. Дякую.
AG: Гарного вам дня, до побачення. Дякую за розмову.
CL: До побачення.
"""


conv_eval_res = {
  "system_code": "kcc",
  "scheme_name": "KredoBank Collection Calls (example scheme)",
  "scheme_version": "1.0.0",
  "model": "gpt-4o",
  "total_weighted_score": 11.0,
  "total_weighted_max": 30.0,
  "details": [
    {
      "id": "opening_and_verification",
      "desc": "Opening + recording disclosure + identity verification quality",
      "score": 5,
      "max_points": 15,
      "weight": 1.0,
      "weighted_score": 5.0,
      "weighted_max": 15.0,
      "model": "gpt-4o",
      "raw": {
        "score": 11
      }
    }
  ],
  "score_percent": 36.67
}


error_res = {
  "system_code": "kcc",
  "scheme_name": "KredoBank Collection Calls (example scheme)",
  "scheme_version": "1.0.0",
  "model": "gpt-4o",
  "total_weighted_score": 5.0,
  "total_weighted_max": 30.0,
  "details": [
    {
      "id": "opening_and_verification",
      "desc": "Opening + recording disclosure + identity verification quality",
      "score": 5,
      "max_points": 15,
      "weight": 1.0,
      "weighted_score": 5.0,
      "weighted_max": 15.0,
      "model": "gpt-4o",
      "raw": {
        "score": 11
      }
    },
    {
      "id": "clarity_and_accuracy",
      "desc": "Clarity and accuracy of communication (figures, dates, terms, logical flow)",
      "score": 0,
      "max_points": 15,
      "weight": 1.0,
      "weighted_score": 0.0,
      "weighted_max": 15.0,
      "model": "gpt-4o",
      "status": "error",
      "error": "Test exception for debugging"
    }
  ],
  "score_percent": 16.67
}

CALL_INFO = {"callType":"debt", "phoneType":"fin", "dpd":"dpd30"}

SAMPLE_REQUEST_JSON = r'''{
  "systemId": "CRM_TEST",
  "requestId": "REQ-90",
  "conversation": "AG: Алло, добрий день.\nCL: Алло, доброго дня. Скажіть, будь ласка, що це мені приходить, що в мене якась заборгованість?\nAG: Зрозуміла. Це Іволо Олена Володимирівна, так?\nCL: Так.\nAG: Давайте я також представлюся. Мене звати Уляна, мій порядковий номер 1096, повідомляю, що наша розмова записується. По вашому рахунку виник від'ємний баланс на суму 8 гривень і 20 копійок.\nCL: А я вже поповняла.\nAG: Ви вже поповнили?\nCL: Поповняла, так.\nAG: Коли оплачували?\nCL: Давненько трошки, точно не скажу. Ну, може місяць пройшов.\nAG: Добре, давайте зараз я пригляну. Так, дякую за очікування. Я бачу, що у вас є арешт рахунки. Ви про це знаєте?\nCL: Ні. Який арешт? На яку суму?\nAG: У вас є арешт рахунки і потрібно звернутися у виконавчу службу, щоб...\nCL: Мені приходять смс, що арешт знятий, бо в мене пару банків є мобільних додатків. Блокують, розблоковують, блокують, розблоковують.\nAG: Я бачу, що в нас є арешт рахунків.\nCL: На яку суму?\nAG: Я не маю таких данів, не можу вам сказати. І тому у вас не виходить оплатити від'ємний баланс по рахунку, тому що є арешт рахунків.\nCL: Добре. Дякую.\nAG: Гарного вам дня, до побачення.\nCL: Вам також до побачення.",
  "callbackEndpoint": "http://127.0.0.1:8002/api/questions/answers",
  "mode": "S",
  "conv_ext_metadata": "test-batch",
  "questions": [
    {"questionId": "Q1", "questionText": "Фахівець привітався?", "answerType": "BOOLEAN", "validChoices": null},
    {"questionId": "Q2", "questionText": "Фахівець назвав себе?", "answerType": "BOOLEAN", "validChoices": null},
    {"questionId": "Q3", "questionText": "Фахівець сказав що розмова записується?", "answerType": "BOOLEAN", "validChoices": null},
    {"questionId": "Q4", "questionText": "Фахівець був ввічливим?", "answerType": "BOOLEAN", "validChoices": null},
    {"questionId": "Q5", "questionText": "Як попрощався фахівець?", "answerType": "TEXT", "validChoices": null},
    {"questionId": "Q6", "questionText": "Чи була проблема вирішена?", "answerType": "CHOICE", "validChoices": ["Повністю", "Частково", "Не вирішена"]}
  ]
}'''


async def test_analysis():
    # First run (no prev_result)
    result, success = await async_analyze_transcription_questions(
        request_json=SAMPLE_REQUEST_JSON,
        model=settings.OPENAI_MODEL_CHAT_ANALYSIS_ENGINE,
        parallel_requests=settings.TR_ANALYSIS_PARALLEL_REQUESTS,  
        prev_result=None,
        timeout=120.0,
    )

    print("SUCCESS:", success)
    for a in result["answers"]:
        print(f'{a["questionId"]}: {a["status"]} -> {a.get("answer")}')

    print(json.dumps(result, ensure_ascii=False, indent=2))





def run_transcription(start_index=0, end_index=None):
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create transcriptions directory if it doesn't exist
    transcriptions_dir = "./test/transcriptions"
    os.makedirs(transcriptions_dir, exist_ok=True)
    
    try:

        reports = audit_evaluation_configs(settings.TR_EVALUATION_CONFIGS_ROOT)
        print(format_audit_report_md(reports))
        print(f"Is configuration OK? {is_configuration_ok(reports)} ")
    
        # log.info("\n\n" + "="*30 + " Loading current evaluation scheme " + "="*30)
        # scheme = load_active_scheme(tr_settings.TR_EVALUATION_CONFIGS_ROOT, 
        #                             "kcc", 
        #                             call_date=date(2026, 1, 21),
        #                             call_info=CALL_INFO)
        # log.info(
        #     f"\nUsing scheme: {scheme.system_code} v{scheme.version}\n"
        # )

        # log.debug(
        #     f"\n{json.dumps(asdict(scheme), indent=2, ensure_ascii=False)}"
        # )

        # metadata_json = O4_METADATA[0]
        # log.info("\n\n" + "="*30 + " Running evaluation scheme " + "="*30)
        # result, success = run_scheme(transcript_text=conversation_test_text, 
        #                              metadata=json.dumps(metadata_json), 
        #                              scheme=scheme)
        # log.info("\n\n" + "="*30 + " Evaluation Results " + "="*30)
        # log.info(f"\n\n{success}\n\n" + str(json.dumps(result, indent=2)))

        # return

        file_number = start_index
        if end_index is None:
            end_index = len(AUDIO_FILES) - 1

        while file_number <= end_index:
            audio_file = "./test/sources/" + AUDIO_FILES[file_number]
            metadata_text = json.dumps(O4_METADATA[file_number])

            log.info(f"\n\nStereo WAV File Splitter and Transcription Tool - FILE NUMBER: {file_number}")
            log.info("=" * 60)

            
            l_file_cleaned, r_file_cleaned, s_file_cleaned = prepare_audio_for_transcription(audio_file, 
                                                                                             settings.TR_TEMP_ROOT_DIR)

            if l_file_cleaned and r_file_cleaned:
                log.info("\n\n" + "="*30 + f" Transcribe O4 cleaned left channel wav " + "="*30)
                o4_left_trans = transcript_audio_file_verbose_o4_single_channel(l_file_cleaned, metadata_text)
                log.info(f"\n{o4_left_trans.text}")

                log.info("\n\n" + "="*30 + f" Transcribe O4 cleaned right channel wav as " + "="*30)
                o4_right_trans = transcript_audio_file_verbose_o4_single_channel(r_file_cleaned, metadata_text)
                log.info(f"\n{o4_right_trans.text}")

                log.info("\n\n" + "="*30 + f" Transcribe O4 cleaned stereo wav " + "="*30)
                o4_stereo_trans = transcript_audio_file_verbose_o4_stereo(s_file_cleaned, metadata_text)
                log.info(f"\n{o4_stereo_trans.text}")


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

            log.info("\n\n\n" + "="*30 + " Generating summary " + "="*30)
            summary = generate_crm_summary_for_call_scenario(scenario)
            log.info("\n" + summary)

            log.info("\n\n" + "="*30 + " Loading current evaluation scheme " + "="*30)
            scheme = load_active_scheme(settings.TR_EVALUATION_CONFIGS_ROOT, 
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
                                         metadata=metadata_text, 
                                         scheme=scheme)
            log.info("\n\n" + "="*30 + " Evaluation Results " + "="*30)
            log.info(f"\n\n{success}\n\n" + str(json.dumps(result, indent=2)))

            
            file_number += 1

    finally:
        # Always close the logger and restore stdout
        shutdown_logger()

# Run the main function
if __name__ == "__main__":
    # run_transcription(3,3)
    # audio_file = "./test/test_call.wav"
    # stereo_to_mono(audio_file, out_file=audio_file.replace(".wav", "_mono.wav"))

    run_transcription(0,0)  # Change indices to process specific files or ranges
    # asyncio.run(test_analysis())