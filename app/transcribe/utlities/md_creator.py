import os
import json
from pathlib import Path

def create_documentation_md(conversation_scenario, 
                            conversation_scenario_diarize,
                            summary_record, 
                            evaluation_text_md, 
                            left_text, 
                            right_text, 
                            stereo_mono_text,
                            file_number, 
                            file_name):
    """
    Creates a markdown file in the 'documentation' subdirectory with the provided content.
    
    Args:
        conversation_scenario (str): The conversation text
        summary_record (str): The summary record text
        evaluation_json (str or dict): The evaluation JSON (can be string or dict)
        file_number (int or str): The file number for naming (e.g., 1 becomes 1.md)
    
    Returns:
        str: Path to the created file
    """
    # Create documentation directory if it doesn't exist
    doc_dir = Path("documentation")
    doc_dir.mkdir(exist_ok=True)
    
    
    # Create the markdown content
    md_content = f"""
# {file_name}

## Текст розмови

{conversation_scenario}

---

## Текст розмови DIARIZE

{conversation_scenario_diarize}

---

## Зведений звіт

{summary_record}

---

## Оцінка

{evaluation_text_md}

"""
    
    # Create the file
    file_path = doc_dir / f"{file_number}.md"
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    return str(file_path)

