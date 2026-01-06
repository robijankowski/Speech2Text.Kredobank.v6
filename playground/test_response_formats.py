import json
from openai import OpenAI
import os
import requests
from typing import Dict, Any, Optional







def _chat_completions_json_schema_request(
    *,
    messages: list,
    schema_name: str,
    schema: Dict[str, Any],
    model: str,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    timeout_s: int = 60,
) -> Dict[str, Any]:
    """
    Low-level helper: calls OpenAI Chat Completions via HTTP and returns parsed JSON content
    that conforms to the provided JSON Schema.
    """
    api_key = api_key or OPENAI_API_KEY
    if not api_key:
        raise ValueError("Missing API key. Set OPENAI_API_KEY env var or pass api_key=...")


    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "temperature": temperature,
        "messages": messages,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": schema,
                "strict": True,
            },
        },
    }

    resp = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=timeout_s)
    data = resp.json() if resp.content else {}

    if resp.status_code >= 400:
        raise RuntimeError(f"OpenAI API error {resp.status_code}: {json.dumps(data, ensure_ascii=False)}")

    content = data["choices"][0]["message"]["content"]
    return json.loads(content)





def evaluate_yes_no_from_python_request(
    req: Dict[str, Any],
    *,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Expects req = {"text": "...", "question": "..."}
    Returns: "YES" or "NO"
    """
    text = (req.get("text") or "").strip()
    question = (req.get("question") or "").strip()

    system_prompt = (
        "You are a careful text analyst.\n"
        "Answer the user's question using ONLY the provided text as evidence.\n"
        "If the answer is not explicitly supported by the text, answer NO.\n"
        "Return ONLY valid JSON that matches the schema."
    )

    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "enum": ["YES", "NO"]},
        },
        "required": ["answer"],
        "additionalProperties": False,
    }

    result = _chat_completions_json_schema_request(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f'Text:\n"""{text}"""\n\nQuestion: {question}\n'},
        ],
        schema_name="yes_no_text_check",
        schema=schema,
        model=model,
        api_key=api_key,
        temperature=0.0,
    )

    print("Evaluation result:", result)
    return result["answer"]



transcript_text = """
Robert is a software engineer with over 10 years of experience in the tech industry. 
He has worked on various projects ranging from web development to machine learning. 
In his free time, Robert enjoys hiking and photography.                
He was born in Poland at 9th of May 1969.
Last year he visited Krakow, Warsaw, and Gdansk.
He is 56 years old.
"""

req = {"text": transcript_text, "question": 'Is the name of the person from text "Robert"?'}
answer = evaluate_yes_no_from_python_request(req)
print("Answer:", answer)

