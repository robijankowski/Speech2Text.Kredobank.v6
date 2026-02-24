from datetime import  date
import json
import asyncio

from app.core.config import settings
from app.core.logger import get_logger, shutdown_logger
log = get_logger(__name__)


from app.transcribe.utilities.evaluation_engine_qa import audit_evaluation_configs, format_audit_report_md, is_configuration_ok



AUDIO_FILES = [
    "./test/test_call_mono.wav",
    "./test/test_call.wav",
    "./test/sources/AUTO-2025-06-30-09-05-380963799218-1096-1751263515.1528148-stereo1.wav",
    "./test/sources/AUTO-2025-06-30-10-08-380988442847-1087-1751267274.1529139-stereo1.wav",
    "./test/sources/AUTO-2025-06-30-10-17-380639093150-1006-1751267854.1529303-stereo1.wav",
    "./test/sources/AUTO-2025-06-30-12-05-380990805468-1098-1751274275.1530761-stereo1.wav",
    "./test/sources/OUT-2025-06-30-09-34-1099-0500814269-1751265256.1528626-stereo1.wav" 
]

O4_METADATA = [
    {"name": "Ivolo Olena Volodymyrivna", "agentName": "Ulyana", "bank": "KredoBank Ukraine"},
    {"name": "Ivolo Olena Volodymyrivna", "agentName": "Ulyana", "bank": "KredoBank Ukraine"},
    {"name": "Ivolo Olena Volodymyrivna", "agentName": "Ulyana", "bank": "KredoBank Ukraine"},
    {"name": "Lukashchuk Serhii Mykolayivych", "agentName": "Sviatoslav", "bank": "KredoBank Ukraine"},
    {},
    {"agent": "Ivanova", "bank": "KredoBank Ukraine"},
    {}
]


TEST_SCENARIO = """
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


TEST_CONV_EVAL_RES = {
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


TEST_CONV_EVAL_RES_WITH_ERR = {
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





CALL_INFO = { "finphone": "FINPHONEYES_", 
             "finphone3": "COMPANYPHONE_", 
             "finphone2": "FINPHONENO_",
             "debtreason": "DEBTREASON_", 
             "outbdounf": "CALLOUT_", 
             "dpd151": "DPD15_", 
             "dpd152": "DPD1530_", 
             "dpd153": "DPD3060_", 
             "dpd154": "DPD30GTOR_", 
             "dpd155": "DPD6090_", 
             "dpd156": "DPD90PLUS_", 
             }
CALL_DATE = date(2026, 1,15)
SYSTEM_CODE = "kcc"


from app.transcribe.utilities.transcribe_pipeline_lr import ( async_transcribe_audio_file_to_scenario_pipeline,
                                                          async_generate_scenario_summary_pipeline,
                                                          async_evaluate_transcripted_scenario_pipeline,
                                                          async_run_analysis_of_the_transcription_pipeline,
                                                          async_evaluate_conversation_interrupts_pipeline,
                                                          )



async def async_run_transcription(start_index=0, end_index=None):
    try:
        reports = audit_evaluation_configs(settings.TR_EVALUATION_CONFIGS_ROOT)
        print(format_audit_report_md(reports))
        log.info(f"Is configuration OK? {is_configuration_ok(reports)} ")

        # settings.USE_AZURE_OPENAI = "Y"  # Change to "Y" to use Azure OpenAI for analysis

        file_number = start_index
        if end_index is None:
            end_index = len(AUDIO_FILES) - 1

        while file_number <= end_index:
            audio_file = AUDIO_FILES[file_number]
            metadata_json = O4_METADATA[file_number]


            
            log.info("\n\n" + "=" * 60 + f"\nRunning transcription for file name: '{audio_file}'\n")
            turns, scenario = await async_transcribe_audio_file_to_scenario_pipeline( source_file=audio_file,
                                                          temp_root_dir=settings.TR_TEMP_ROOT_DIR,
                                                          metadata=metadata_json 
                                                          )
            log.info(f"\n=== Final scenario for file: {audio_file} ===\n{str(scenario)}")

            return
            scenario = TEST_SCENARIO

            # log.info("\n\n" + "=" * 60 + f"\nRunning summary for file name: '{audio_file}'\n")
            # summary = await async_generate_scenario_summary_pipeline(scenario=scenario)
            # log.info(f"Summary for file number {audio_file} :\n{summary}")

            # return
            # log.info("\n\n" + "=" * 60 + f"\nRunning evaluation interrupts file name: '{audio_file}'\n")
            # res_interrupts = await async_evaluate_conversation_interrupts_pipeline(turns=turns, file_name=audio_file) # this is not async - pure calcs.
            # log.info(f"Evaluation interrupts result {audio_file} :\n{res_interrupts}")
            scenario = TEST_SCENARIO
            res_interrupts = {}
            log.info("\n\n" + "=" * 60 + f"\nRunning evaluation for file name: '{audio_file}'\n")
            res, success = await async_evaluate_transcripted_scenario_pipeline( scenario=scenario,
                                                                    metadata=metadata_json,
                                                                    system_code=SYSTEM_CODE,
                                                                    call_date=CALL_DATE,
                                                                    call_info=CALL_INFO,
                                                                    prev_result=None,
                                                                    interrupts_analysis=res_interrupts
                                                                    )
            log.info(f"Evaluation results {audio_file} :\n{success}\nResult: {json.dumps(res, ensure_ascii=False, indent=2)}")

            return
            log.info("\n\n" + "=" * 60 + f"\nRunning async analysis for  file name: '{audio_file}'\n")
            request = json.loads(SAMPLE_REQUEST_JSON) #simluation of phase 2 - analysis on free questions
            request["conversation"] = scenario
            res, success = await async_run_analysis_of_the_transcription_pipeline(request_json=request, prev_result=None)
            log.info(f"SUCCESS: {success}")
            for a in res["answers"]:
                log.info(f'{a["questionId"]}: {a["status"]} -> {a.get("answer")}')

            log.info(json.dumps(res, ensure_ascii=False, indent=2))
                  
            
            file_number += 1

    finally:
        # Always close the logger and restore stdout
        shutdown_logger()

# Run the main function
if __name__ == "__main__":
    asyncio.run(async_run_transcription(1,1))  # Change indices to process specific files or ranges
