import json
import os
import re
import sys
from datetime import datetime
from utilities.transcription_tools import process_with_o4
from utilities.scenario_tools import split_transcription_into_roles_4o, format_scenario_md, consolidate_dialogue
from utilities.md_creator import create_documentation_md
from utilities.merge_whisper_segs import concatenate_segments_to_phrases, combine_phrase_tables, format_conversation_text
from utilities.summary_tools import generate_crm_summary_o4, format_summary_md
from utilities.evaluate_tools import evaluate_call, format_evaluation_results_md, format_evaluation_results
from utilities.audio_tools import optimize_audio_files, remove_long_silences_in_audio, trim_audio_to_secs
from utilities.stats import get_stats, stats_json_to_csv_text, stats_save_json_to_csv, clean_file_names

from utilities.o4_api_diarize import transcript_audio_file_verbose_o4_diarize

AUDIO_FILES = [
    "AUTO-2025-06-30-09-05-380963799218-1096-1751263515.1528148-stereo1.wav",
    "AUTO-2025-06-30-10-08-380988442847-1087-1751267274.1529139-stereo1.wav",
    "AUTO-2025-06-30-10-17-380639093150-1006-1751267854.1529303-stereo1.wav",
    "AUTO-2025-06-30-12-05-380990805468-1098-1751274275.1530761-stereo1.wav",
    "OUT-2025-06-30-09-34-1099-0500814269-1751265256.1528626-stereo1.wav" 
]

O4_METADATA = [
    "Імена учасників: 'Іволо Олена Володимирівна', Ім'я агента: 'Уляна'; Назва банку: 'KredoBank Україна'",
    "Імена учасників: 'Лукашчук Сергій Миколаївич', Ім'я агента: 'Святослав'; Назва банку: 'KredoBank Україна'",
    "",
    "Ім'я агента: 'Іванна'; Назва банку: 'KredoBank Україна'",
    ""
]

WHISPER_METADATA = [
    "Іволо Олена Володимирівна, Уляна, KredoBank, Україна",
    "'Лукашчук Сергій Миколаївич', 'Святослав', KredoBank, Україна",
    "",
    "'Іванна', KredoBank, Україна",
    ""
]




class Logger:
    def __init__(self, log_file_path):
        self.log_file = open(log_file_path, 'w', encoding='utf-8')
        self.original_stdout = sys.stdout
        
    def write(self, message):
        # Write to both console and file
        self.original_stdout.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # Ensure immediate writing
        
    def flush(self):
        self.original_stdout.flush()
        self.log_file.flush()
        
    def close(self):
        self.log_file.close()
        sys.stdout = self.original_stdout




def add_prefix_to_sentences(text, prefix):   
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    prefixed_sentences = []
    for sentence in sentences:
        if sentence.strip():
            prefixed_sentences.append(f"{prefix} {sentence.strip()}")
    
    return ' '.join(prefixed_sentences)




def run_transcription(start_index=0, end_index=None):
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S___")
    
    # Create transcriptions directory if it doesn't exist
    transcriptions_dir = "./transcriptions"
    os.makedirs(transcriptions_dir, exist_ok=True)
    
    # Set up logging
    log_file_path = os.path.join(transcriptions_dir, f"{timestamp}log.txt")
    logger = Logger(log_file_path)
    sys.stdout = logger
    
    try:
        file_number = start_index
        if end_index is None:
            end_index = len(AUDIO_FILES) - 1

        while file_number <= end_index:
            audio_file = "./sources/" + AUDIO_FILES[file_number]

            print(f"\n\nStereo WAV File Splitter and Transcription Tool - FILE NUMBER: {file_number}")
            print("=" * 60)

            l_file, r_file, m_file, s_file = optimize_audio_files(audio_file)
            left_file_cleaned = remove_long_silences_in_audio(l_file)
            right_file_cleaned = remove_long_silences_in_audio(r_file)
            mono_file_cleaned = remove_long_silences_in_audio(m_file)
            stereo_file_cleaned = remove_long_silences_in_audio(s_file)

            # base_name = os.path.splitext(audio_file)[0]
            # lf_5sec = f"{base_name}_5s_left.wav"
            # rf_5sec = f"{base_name}_5s_right.wav"
            # trim_audio_to_secs(left_file_cleaned, lf_5sec, 5)
            # trim_audio_to_secs(right_file_cleaned, rf_5sec, 5)



            o4_left_trans = None
            o4_right_tran = None
            o4_mono_trans = None

            o4_diarize_trans_text = ""
            o4_diarize_trans_text = transcript_audio_file_verbose_o4_diarize(stereo_file_cleaned, 
                                                                        temperature=0.0, 
                                                                        language="uk", 
                                                                        chunking_strategy="auto" )

            print("\n\n\n" + "="*30 + " Diarized transcript text o4" + "="*30)
            print("\n" + o4_diarize_trans_text)
        
            o4_left_trans, o4_right_tran, o4_mono_trans = process_with_o4(  left_file_cleaned, 
                                                                            right_file_cleaned, 
                                                                            mono_file_cleaned, 
                                                                            file_number, 
                                                                            O4_METADATA[file_number])


            client_text = o4_right_tran.text
            agent_text = o4_left_trans.text

            if file_number == 4:
                client_text = o4_left_trans.text
                agent_text = o4_right_tran.text

            agent_text = add_prefix_to_sentences(agent_text, "AG:")
            client_text = add_prefix_to_sentences(client_text, "CL:")

            # print("\n\n\n" + "="*30 + " Modified AGENT for o4 " + "="*30)
            # print("\n" + agent_text)
            # print("\n\n\n" + "="*30 + " Modified CLIENT for o4 " + "="*30)
            # print("\n" + client_text)

            print("\n\n\n" + "="*30 + " Generating roles/scenario with o4 " + "="*30)
            scenario_granular = split_transcription_into_roles_4o( agent_text = agent_text, 
                                                                   client_text = client_text, 
                                                                   stereo_text = o4_mono_trans.text,
                                                                   file_name = audio_file )
            scenario = consolidate_dialogue(scenario_granular)
            scenario_md = format_scenario_md(scenario)
            print("\n" + "="*30 + " o4 original scenario " + "="*30)
            print("\n" + scenario_granular)
            print("\n" + "="*30 + " o4 consolidated scenario "  + "="*30)
            print("\n" + scenario)

            print("\n\n\n" + "="*30 + " Generating summary " + "="*30)
            # summary = generate_crm_summary_o4(scenario)
            # summary_md = format_summary_md(summary)
            # print("\n" + summary)

            print("\n\n\n" + "="*30 + " Evaluating call " + "="*30)
            # evaluation_results = evaluate_call(scenario)   
            # evaluation_text_md = format_evaluation_results_md(evaluation_results)
            # evaluation_text = format_evaluation_results(evaluation_results)
            # print("\n" + evaluation_text)

            print("\n\n\n" + "="*30 + " Documentation MD " + "="*30)
            # md_doc = create_documentation_md(
            #     file_name=audio_file,
            #     file_number=file_number,
            #     left_text=o4_left_trans,
            #     right_text=o4_right_tran,
            #     stereo_mono_text=o4_mono_trans,
            #     conversation_scenario=scenario_md,
            #     conversation_scenario_diarize=o4_diarize_trans_text,
            #     summary_record=summary_md,
            #     evaluation_text_md=evaluation_text_md
            # )
            md_doc = create_documentation_md(
                file_name=audio_file,
                file_number=file_number,
                left_text=o4_left_trans,
                right_text=o4_right_tran,
                stereo_mono_text=o4_mono_trans,
                conversation_scenario=scenario_md,
                conversation_scenario_diarize=o4_diarize_trans_text,
                summary_record="",
                evaluation_text_md=""
            )
            print("\nProcessing complete!")
            
            # Save transcriptions to files with timestamp prefix
            save_to_files = True
            if save_to_files:        
                if os.path.exists(md_doc):
                    import shutil
                    md_filename = os.path.basename(md_doc)
                    timestamped_md_path = os.path.join(transcriptions_dir, f"{timestamp}{md_filename}")
                    shutil.copy2(md_doc, timestamped_md_path)
                    print(f"Copied {md_doc} to {timestamped_md_path}")
                else:
                    print(f"Warning: MD documentation file not found at {md_doc}")

                print(f"All files saved to {transcriptions_dir}/ folder with timestamp {timestamp}\n\n\n")
            
            file_number += 1
        
        # print(json.dumps(get_stats(), indent=2))
        timestamped_csv_path = os.path.join(transcriptions_dir, f"{timestamp}stats.csv")
        csv_stats_text = stats_json_to_csv_text(get_stats())
        csv_stats_text = clean_file_names(AUDIO_FILES, csv_stats_text)
        stats_save_json_to_csv( csv_stats_text, timestamped_csv_path)

    finally:
        # Always close the logger and restore stdout
        logger.close()

# Run the main function
if __name__ == "__main__":
    # run_transcription(3,3)
    run_transcription(4,4)  # Change indices to process specific files or ranges