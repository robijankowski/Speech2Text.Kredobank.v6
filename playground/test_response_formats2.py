import json
from openai import OpenAI
import os
import requests
from typing import Dict, Any, Optional



OPENAI_API_KEY = "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"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"


transcript_text = """
Robert is a software engineer with over 10 years of experience in the tech industry. 
He has worked on various projects ranging from web development to machine learning. 
In his free time, Robert enjoys hiking and photography.                
He was born in Poland at 9th of May 1969.
Last year he visited Krakow, Warsaw, and Gdansk.
He is 56 years old.
"""

openai_client = OpenAI(api_key="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")
openai_model = "gpt-4o"




def _chat_completions_json_schema_request_http(
    *,
    messages: list,
    schema_name: str,
    schema: Dict[str, Any],
    model: str,
    api_key: Optional[str] = OPENAI_API_KEY,
    temperature: float = 0.0,
    timeout_s: int = 60,
    openai_url: str = OPENAI_URL,
) -> Dict[str, Any]:

    if not api_key:
        raise ValueError("Missing API key. Set OPENAI_API_KEY or pass api_key=...")

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

    resp = requests.post(openai_url, headers=headers, json=payload, timeout=timeout_s)
    data = resp.json() if resp.content else {}

    if resp.status_code >= 400:
        raise RuntimeError(f"OpenAI API error {resp.status_code}: {json.dumps(data, ensure_ascii=False)}")

    content = data["choices"][0]["message"]["content"]
    return json.loads(content)


def evaluate_structured_from_text_request(
    *,
    text: str,
    question: str,
    schema_name: str,
    schema_json: str,
    api_key: Optional[str] = OPENAI_API_KEY,
    model: str = "gpt-4o-mini",
    system_prompt: Optional[str] = None,
    temperature: float = 0.0,
    timeout_s: int = 60,
    openai_url: str = OPENAI_URL,
) -> Dict[str, Any]:
    """
    Generic structured-output tool.

    - text, question: passed as parameters
    - schema_name: string
    - schema_json: string containing a JSON Schema object (json.loads-able)
    Returns: parsed dict that conforms to schema.
    """
    text = (text or "").strip()
    question = (question or "").strip()
    schema_name = (schema_name or "").strip()
    schema_json = (schema_json or "").strip()

    if not schema_name:
        raise ValueError("schema_name must be a non-empty string")
    if not schema_json:
        raise ValueError("schema_json must be a non-empty JSON string")

    try:
        schema = json.loads(schema_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"schema_json is not valid JSON: {e}") from e

    if system_prompt is None:
        system_prompt = (
            "You are a careful text analyst.\n"
            "Answer the user's question using ONLY the provided text as evidence.\n"
            # "If the answer is not explicitly supported by the text, answer NO.\n"
            # "Return ONLY valid JSON that matches the schema."
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f'Text:\n"""{text}"""\n\nQuestion: {question}\n'},
    ]

    return _chat_completions_json_schema_request_http(
        messages=messages,
        schema_name=schema_name,
        schema=schema,
        model=model,
        api_key=api_key,
        temperature=temperature,
        timeout_s=timeout_s,
        openai_url=openai_url,
    )





schema_json_txt = json.dumps({
    "type": "object",
    "properties": {"answer": {"type": "string", "enum": ["YES", "NO"]}},
    "required": ["answer"],
    "additionalProperties": False
})

result = evaluate_structured_from_text_request(
    text=transcript_text,
    question='Is the name of the person "Robert"?',
    schema_name="schema_name",
    schema_json=schema_json_txt,
)
print(f"\n\nSchema JSON for YES/NO:\n{schema_json_txt}")
print(f"\nResult: {result}")          




schema_json_txt = json.dumps({
    "type": "object",
    "properties": {
        "date": {
            "type": "string",
            "pattern": r"^\d{4}/\d{2}/\d{2}$"
        }
    },
    "required": ["date"],
    "additionalProperties": False
})

result = evaluate_structured_from_text_request(
    text=transcript_text,
    question="What is the person's birth date? Return as YYYY/MM/DD.",
    schema_name="schema_name",
    schema_json=schema_json_txt,
)
print(f"\n\nSchema JSON for date:\n{schema_json_txt}")
print(f"\nResult: {result}")          




schema_json_txt = json.dumps({
    "type": "object",
    "properties": {
        "text": {"type": "string", "minLength": 1, "maxLength": 200}
    },
    "required": ["text"],
    "additionalProperties": False
})

result = evaluate_structured_from_text_request(
    text=transcript_text,
    question="What does Robert enjoy in his free time? Answer as a short sentence.",
    schema_name="schema_name",
    schema_json=schema_json_txt,
)
print(f"\n\nSchema JSON for short text:\n{schema_json_txt}")
print(f"\nResult: {result}")          




schema_json_txt = json.dumps({
    "type": "object",
    "properties": {
        "cities": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "minItems": 0        }
    },
    "required": ["cities"],
    "additionalProperties": False
})

result = evaluate_structured_from_text_request(
    text=transcript_text,
    question="Which cities did he visit last year? Return just the city names.",
    schema_name="schema_name",
    schema_json=schema_json_txt,
)
print(f"\n\nSchema JSON for cities array:\n{schema_json_txt}")
print(f"\nResult: {result}")          




schema_json_txt = json.dumps({
    "type": "object",
    "properties": {"value": {"type": "integer"}},
    "required": ["value"],
    "additionalProperties": False
})

result = evaluate_structured_from_text_request(
    text=transcript_text,
    question="How old is he? Return the age as an integer.",
    schema_name="schema_name",
    schema_json=schema_json_txt,
)
print(f"\n\nSchema JSON for integer value:\n{schema_json_txt}")
print(f"\nResult: {result}")          




schema_json_txt = json.dumps({
    "type": "object",
    "properties": {"value": {"type": "number"}},
    "required": ["value"],
    "additionalProperties": False
})

sample_text = "The package weighs 1.75 kg."
result = evaluate_structured_from_text_request(
    text=sample_text,
    question="What is the package weight in kg? Return a number.",
    schema_name="schema_name",
    schema_json=schema_json_txt,
)
print(f"\n\nSchema JSON for number value:\n{schema_json_txt}")
print(f"\nResult: {result}")          



schema_json_txt = json.dumps({
    "type": "object",
    "properties": {
        "amount": {"type": "number"},
        "currency": {"type": "string", "pattern": r"^[A-Z]{3}$"}  # ISO-4217 style
    },
    "required": ["amount", "currency"],
    "additionalProperties": False
})

sample_text = "Total due is 19.99 USD."
result = evaluate_structured_from_text_request(
    text=sample_text,
    question="What is the total due? Return amount and currency.",
    schema_name="schema_name",
    schema_json=schema_json_txt,
)
print(f"\n\nSchema JSON for money amount:\n{schema_json_txt}")
print(f"\nResult: {result}")          



