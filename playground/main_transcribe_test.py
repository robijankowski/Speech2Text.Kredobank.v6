import json
import os
from utilities.stereo_tools import process_stereo_file
from utilities.correctors_tools import split_transcription_into_roles_4o, format_scenario_md
from utilities.md_creator import create_documentation_md
from utilities.merge_whisper_segs import concatenate_segments_to_phrases, combine_phrase_tables, format_conversation_text
from utilities.summary_tools import generate_crm_summary_o4, format_summary_md
from utilities.evaluate_tools import evaluate_call, format_evaluation_results_md, format_evaluation_results

AUDIO_FILES = [
    "AUTO-2025-06-30-09-05-380963799218-1096-1751263515.1528148-stereo1.wav",
    "AUTO-2025-06-30-10-08-380988442847-1087-1751267274.1529139-stereo1.wav",
    "AUTO-2025-06-30-10-17-380639093150-1006-1751267854.1529303-stereo1.wav",
    "AUTO-2025-06-30-12-05-380990805468-1098-1751274275.1530761-stereo1.wav",
    "OUT-2025-06-30-09-34-1099-0500814269-1751265256.1528626-stereo1.wav" 
]

AUDIO_FILES_METADATA = [
    "Імена учасників: 'Іволо Олена Володимирівна', Ім'я агента: 'Уляна'; Назва банку: 'KredoBank Україна'",
    "Імена учасників: 'Лукашчук Сергій Миколаївич', Ім'я агента: 'Святослав'; Назва банку: 'KredoBank Україна'",
    "",
    "Ім'я агента: 'Іванна'; Назва банку: 'KredoBank Україна'",
    ""
]

WHISPER_FILES_METADATA = [
    "Іволо Олена Володимирівна, Уляна, KredoBank, Україна",
    "'Лукашчук Сергій Миколаївич', 'Святослав', KredoBank, Україна",
    "",
    "'Іванна', KredoBank, Україна",
    ""
]


AG = """
Алло. Доброго ранку. Так, Оксана Володимирівна, так? Ну, давайте детальніше, Оксана Володимирівна, вже обговоримо. Звати мене Шалаєв Юрій Тимофійович, повідомляю, що розмова записується. Це угода ЦЛ-304-012, це одна угода у вас. Від 22.03.2021. На 29.06. даніще відображається 6 днів протермінування, тобто до оплати 2988 гривень. Зараз подивимося. А в тому то і справа, якщо ви трошки більше оплачуєте, то воно вам перекриває тим самим. Я відразу зараз перегляну, може графік. Так, все вірно, так як ви кажете, 3136. Зрозуміло, буває таке там, зараз все що завгодно. З запасом, так що пересрахуватись, так. Добре, будете намагатись сьогодні, крайній термін я перше зазначу. Ви можете сплачувати так, як і по графіку у вас є, бо це буде у вас перекривати тим самим сума. Добре, домовились, очікуємо оплату, до побачення. На взаєм.
"""
CL = """
Доброго ранку. Ви мені телефонували, моє прізвище Водовожська, в мене у вас оформлений кредит. Але в мене щось вибиває, коли ви мені телефонуєте, то я вас сама набираю. Я знаю, у мене була стояла там дата, якусь трохи затримка, сьогодні до 18-ї години. Так. А чого так мало? У мене ж там, коли робили, коли війна почалася, зробили реструктуризацію, там у мене 3 206, щось там, 3 з чимось у мене. Я завжди просто захожу. У мене не може бути 2 з чимось. А, я просто завжди плачу, може, по 3 500, по 3 600, тому може... Так, так. А, просто перекриваю, я зрозуміла. Бо в нас 17-го числа... О, бачите, я ж кажу, що в мене, так, 3 з чимось. Бо в нас просто пострадав 17-го числа від вистрілів завод наш, і трохи пішла затримка, на жаль. Так, на жаль. Тому дякую вам. А, подивіться, я просто сьогодні в нас понеділок. А ви можете поставити краще до завтра, щоб я вже була впевнена, бо сьогодні має бути, якби, ну, пообіцяли, розумієте? Давайте тоді до завтра, завтрашній день, ключ. Так, так, так, так. Звичайно, так. Дякую вам велике. Я зрозуміла. Добре, дякую вам велике. Так, дякую, гарного мирного дня, до побачення.
"""
ST = """
Алло. Доброго ранку. Ви мені телефонували, моє прізвище Водовожська, в мене у вас оформлений кредит. Але в мене щось вибиває, коли ви мені телефонуєте, то я вас сама набираю. Я знаю, у мене була стояла там дата, якусь трохи затримка, сьогодні до 18-ї години. Давайте детальніше, Оксана Володимирівна, вже обговоримо. Звати мене Шолаєв Юрій Тимофійович, повідомляю, що розмова записується. Це угода ЦЛ 304012, це одна угода у вас. Так. Від 22.03.2021. На 29.06. дані ще відображаються 6 днів протермінування, тобто до оплати 2988 гривень. А чого так мало? У мене ж там, коли робили, коли війна почалася, зробили реструктуризацію, у мене 3206, 3 з чимось у мене. Я завжди просто враховую. У мене не може бути 2,6. А, я просто завжди плачу може по 3500, по 3600. А в тому то і справа, якщо ви трошки більше оплачуєте, то воно перекриває тим самим. А, просто перекриває, я зрозуміла. Я відразу зараз перегляну, може графік. У нас 3136. О, бачите, я ж кажу, що у мене 3 з чимось. У нас просто пострадав 17-го числа від вистрілів завод наш і трохи пішла затримка. Зрозуміло, буває таке. Так, на жаль. Тому дякую вам. А подивіться, я просто сьогодні у нас понеділок. А ви можете поставити краще до завтра, щоб я вже була впевнена, бо сьогодні має бути, як би, ну пообіцяли, розумієте. Давайте тоді до завтра. З запасом, так, щоб пересрахуватися. Так, так, так. Добре, будете намагатись сьогодні, крайній термін. Звичайно, так. Дякую вам велике. Ви можете сплачувати так, як і по графіку у вас є, бо це буде у вас перекривати тим самим суму. Я зрозуміла, добре, дякую вам велике. Добре, домовилися, чекаємо на оплату. Дякую, гарного мирного дня, до побачення.
"""

import re

def add_ag_prefix(text, prefix):   
    # Split text into sentences using regex
    # This pattern looks for sentence endings followed by whitespace or end of string
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    # Filter out empty strings and add prefix to each sentence
    prefixed_sentences = []
    for sentence in sentences:
        if sentence.strip():
            prefixed_sentences.append(f"{prefix} {sentence.strip()}")
    
    # Join sentences back together with spaces
    return ' '.join(prefixed_sentences)


def run_transcription(start_index=0, end_index=None):
    file_number = start_index
    if end_index is None:
        end_index = len(AUDIO_FILES) - 1

    while file_number <= end_index:
        file_name = "./sources/" + AUDIO_FILES[file_number]

        print(f"\n\nStereo WAV File Splitter and Transcription Tool - FILE NUMBER: {file_number}")
        print("=" * 60)
            
        # left_text_cln, right_text_cln, mono_text_cln, whisper_text_cln, wl, wr = process_stereo_file(file_name, 
        #                                                                                              file_number,
        #                                                                        AUDIO_FILES_METADATA[file_number],
                                                                            #    WHISPER_FILES_METADATA[file_number])
        left_text_cln = add_ag_prefix(AG, "AG:")
        right_text_cln = add_ag_prefix(CL, "CL:")
        mono_text_cln = ST

        print(left_text_cln)
        print(right_text_cln)
        print(mono_text_cln)
        
        # wlt = concatenate_segments_to_phrases(wl)
        # wrt = concatenate_segments_to_phrases(wr)
        # wt = combine_phrase_tables(wlt, wrt)
        # print("\n\n\n" + "="*30 + " WHISPER conversation text " + "="*30)
        # print(format_conversation_text(wt))
        # print("="*30)

        client_text = right_text_cln
        agent_text = left_text_cln

        if file_number == 4:
            client_text = left_text_cln
            agent_text = right_text_cln

        print("\n\n\n" + "="*30 + " Splitting into roles/scenario " + "="*30)
        scenario = split_transcription_into_roles_4o( agent_text=agent_text, client_text=client_text, stereo_text=mono_text_cln )
        scenario_md = format_scenario_md(scenario)
        print("" + "="*30 + " SCENARIO " + "="*30)
        print(scenario)
        return
    
        print("\n\n\n" + "="*30 + " Generating summary " + "="*30)
        summary = generate_crm_summary_o4(scenario)
        summary_md = format_summary_md(summary)
        print("" + "="*30 + " SUMMARY " + "="*30)
        print(summary)

        print("\n\n\n" + "="*30 + " Evaluating call " + "="*30)
        evaluation_results = evaluate_call(scenario)   
        evaluation_text_md = format_evaluation_results_md(evaluation_results)
        evaluation_text = format_evaluation_results(evaluation_results)
        print("" + "="*30 + " EVALUATION " + "="*30)
        print(evaluation_text)
        print("" + "="*60)

        print("\n\n\n" + "="*30 + " Documentation MD " + "="*30)
        md_doc = create_documentation_md(
            file_name=file_name,
            file_number=file_number,
            left_text=left_text_cln,
            right_text=right_text_cln,
            stereo_mono_text=mono_text_cln,
            conversation_scenario=scenario_md,
            summary_record=summary_md,
            evaluation_text_md=evaluation_text_md
        )
        print(f"Generated Markdown documentation for call #{file_number}:\n{md_doc}")
        print("" + "="*60)

        print("\nProcessing complete!")
        
        # Optional: Save transcriptions to files
        save_to_files = True
        if save_to_files:        
            base_name = os.path.splitext(os.path.basename(file_name))[0]

            with open(f"./transcriptions/{base_name}_left_transcription.txt", "w", encoding="utf-8") as f:
                f.write(left_text_cln)
            
            with open(f"./transcriptions/{base_name}_right_transcription.txt", "w", encoding="utf-8") as f:
                f.write(right_text_cln)

            with open(f"./transcriptions/{base_name}_stereo_mono_transcription.txt", "w", encoding="utf-8") as f:
                f.write(mono_text_cln)

            # with open(f"./transcriptions/{base_name}_whisper_transcription.txt", "w", encoding="utf-8") as f:
            #     f.write(whisper_text_cln)

            with open(f"./transcriptions/{base_name}_scenario.txt", "w", encoding="utf-8") as f:
                f.write(scenario)
            
            with open(f"./transcriptions/{base_name}_summary.txt", "w", encoding="utf-8") as f:
                f.write(summary)

            with open(f"./transcriptions/{base_name}_evaluation.txt", "w", encoding="utf-8") as f:
                f.write(evaluation_text)

            print(f"Transcriptions saved to ./transcriptions/ folder")
        file_number += 1



# Run the main function
if __name__ == "__main__":
    # Create transcriptions directory if it doesn't exist
    os.makedirs("./transcriptions", exist_ok=True)
    # run_transcription(3,3)
    run_transcription(2,2)  # Change indices to process specific files or ranges

    