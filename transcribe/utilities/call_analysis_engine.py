from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Uses your existing OpenAI async wrapper with retries + json_schema structured outputs
from openai_tools.openai_client_text import async_chat_completion_with_format  # :contentReference[oaicite:2]{index=2}
from core.config import settings


def _default_analysis_model() -> str:
    return settings.AZURE_MODEL_CHAT_ANALYSIS_ENGINE if settings.USE_AZURE_OPENAI == "Y" else settings.OPENAI_MODEL_CHAT_ANALYSIS_ENGINE

# -----------------------------
# Helpers: schemas + prompts
# -----------------------------

def _schema_for_question(q: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a strict JSON schema for ONE question answer, based on answerType + validChoices.
    The model MUST return only {"answer": ...}.
    """
    qtype = (q.get("answerType") or "").upper().strip()
    valid = q.get("validChoices")

    if qtype == "BOOLEAN":
        answer_schema: Dict[str, Any] = {"type": "boolean"}
    elif qtype == "CHOICE":
        if isinstance(valid, list) and all(isinstance(x, str) for x in valid) and valid:
            answer_schema = {"type": "string", "enum": valid}
        else:
            # Fallback: accept any string if config is missing/invalid
            answer_schema = {"type": "string"}
    else:  # TEXT (or unknown)
        answer_schema = {"type": "string"}

    return {
        "type": "object",
        "properties": {
            "answer": answer_schema,
        },
        "required": ["answer"],
        "additionalProperties": False,
    }


def _build_messages(conversation: str, q: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Single-question prompting (fast + robust).
    """
    qid = q.get("questionId", "")
    qtext = q.get("questionText", "")
    atype = q.get("answerType", "")
    valid = q.get("validChoices")

    valid_txt = ""
    if isinstance(valid, list) and valid:
        valid_txt = f"\nValid choices: {valid}"

    system = (
        "You are a QA analyst for a bank call.\n"
        "Answer exactly ONE question based ONLY on the provided transcript.\n"
        "If the transcript does not contain enough information, make your best inference "
        "but do not invent specific facts.\n"
        "Return ONLY valid JSON matching the required schema."
    )

    user = (
        f"Transcript:\n---\n{conversation}\n---\n\n"
        f"QuestionId: {qid}\n"
        f"QuestionText: {qtext}\n"
        f"AnswerType: {atype}"
        f"{valid_txt}\n\n"
        "Return JSON."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# -----------------------------
# Resume logic
# -----------------------------

def _index_prev_answers(prev_result: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Map questionId -> previous answer record (only if present).
    """
    if not isinstance(prev_result, dict):
        return {}
    answers = prev_result.get("answers")
    if not isinstance(answers, list):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for rec in answers:
        if isinstance(rec, dict):
            qid = (rec.get("questionId") or "").strip()
            if qid:
                out[qid] = rec
    return out


def _should_rerun(prev_rec: Optional[Dict[str, Any]]) -> bool:
    """
    Re-run if:
      - no previous record
      - status == "error"
      - missing/None answer
    """
    if not isinstance(prev_rec, dict):
        return True
    if (prev_rec.get("status") or "").lower() == "error":
        return True
    if "answer" not in prev_rec or prev_rec.get("answer") is None:
        return True
    return False


# -----------------------------
# Core runner
# -----------------------------

async def async_analyze_transcription_questions(
    request_json: str | Dict[str, Any],
    *,
    model: str = "",
    parallel_requests: int = None,
    prev_result: Optional[Dict[str, Any]] = None,
    timeout: float = 120.0,
) -> Tuple[Dict[str, Any], bool]:
    """
    Runs the provided questions against the transcript in bucketed parallelism.

    Args:
        request_json: incoming request dict or JSON string
        model: OpenAI model name
        parallel_requests: bucket size (max concurrent requests per wave)
        prev_result: result from previous run; only failed questions will be re-run
        timeout: per-request timeout forwarded to the OpenAI client

    Returns:
        (result_json, success_bool)
        success_bool == True only when ALL questions have status == "ok"
    """
    # Parse input
    req: Dict[str, Any] = json.loads(request_json) if isinstance(request_json, str) else dict(request_json)

    if not model:
        model = _default_analysis_model()
    if not parallel_requests or parallel_requests < 1:
        parallel_requests = settings.TR_ANALYSIS_PARALLEL_REQUESTS  # default value

    system_id = req.get("systemId")
    request_id = req.get("requestId")
    conversation = req.get("conversation") or ""
    questions = req.get("questions") or []

    if not isinstance(questions, list):
        raise ValueError("request_json.questions must be a list")

    prev_by_id = _index_prev_answers(prev_result)
    q_by_id = {str(q.get("questionId", "")).strip(): q for q in questions if isinstance(q, dict)}

    async def _run_one(q: Dict[str, Any]) -> Dict[str, Any]:
        qid = (q.get("questionId") or "").strip()
        schema = _schema_for_question(q)
        messages = _build_messages(conversation, q)

        started = time.time()
        try:
            completion = await async_chat_completion_with_format(
                messages=messages,
                format_schema=schema,
                schema_name=f"answer_{qid or 'q'}",
                model=model,
                temperature=0.0,
                timeout=timeout,
            )
            raw_text = completion.choices[0].message.content
            payload = json.loads(raw_text)
            return {
                "questionId": qid,
                "questionText": q.get("questionText"),
                "answerType": q.get("answerType"),
                "answer": payload.get("answer"),
                "status": "ok",
                "error": None,
                "model": model,
                "latency_ms": int((time.time() - started) * 1000),
                "raw": payload,
            }
        except Exception as e:
            return {
                "questionId": qid,
                "questionText": q.get("questionText"),
                "answerType": q.get("answerType"),
                "answer": None,
                "status": "error",
                "error": str(e),
                "model": model,
                "latency_ms": int((time.time() - started) * 1000),
            }

    # Build final answer list in the same order as request.questions
    final_answers: List[Dict[str, Any]] = []

    # Prepare “to run” list (only failed / missing)
    to_run: List[Dict[str, Any]] = []
    reused: Dict[str, Dict[str, Any]] = {}

    for q in questions:
        if not isinstance(q, dict):
            continue
        qid = (q.get("questionId") or "").strip()
        prev_rec = prev_by_id.get(qid)
        if _should_rerun(prev_rec):
            to_run.append(q)
        else:
            reused[qid] = prev_rec  # keep as-is

    # Run in buckets (waves) of parallel_requests
    parallel_requests = max(1, int(parallel_requests))
    results_by_id: Dict[str, Dict[str, Any]] = {}

    # preload reused
    results_by_id.update(reused)

    for i in range(0, len(to_run), parallel_requests):
        bucket = to_run[i : i + parallel_requests]
        bucket_results = await asyncio.gather(*(_run_one(q) for q in bucket))
        for rec in bucket_results:
            qid = (rec.get("questionId") or "").strip()
            if qid:
                results_by_id[qid] = rec

    # Emit answers in original order; ignore any prev answers for questions that no longer exist
    for q in questions:
        if not isinstance(q, dict):
            continue
        qid = (q.get("questionId") or "").strip()
        rec = results_by_id.get(qid)
        if rec is None:
            # Shouldn't happen, but keep it explicit
            rec = {
                "questionId": qid,
                "questionText": q.get("questionText"),
                "answerType": q.get("answerType"),
                "answer": None,
                "status": "error",
                "error": "No result produced for this question",
                "model": model,
            }
        final_answers.append(rec)

    success = all((a.get("status") == "ok") for a in final_answers)

    result_json: Dict[str, Any] = {
        "systemId": system_id,
        "requestId": request_id,
        "mode": req.get("mode"),
        "conv_ext_metadata": req.get("conv_ext_metadata"),
        "model": model,
        "success": success,
        "answers": final_answers,
        "meta": {
            "totalQuestions": len([q for q in questions if isinstance(q, dict)]),
            "rerunQuestions": len(to_run),
            "reusedQuestions": len(reused),
            "callbackEndpoint": req.get("callbackEndpoint"),
        },
    }

    return result_json, success
