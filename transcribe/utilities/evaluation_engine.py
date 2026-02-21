from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.config import settings
from core.logger import log


from openai_tools.openai_client_text import async_chat_completion_with_format  


@dataclass(frozen=True)
class CheckDef:
    id: str
    desc: str
    weight: float
    max_points: int
    temperature: float
    schema_name: str
    system_prompt_template: str
    user_prompt_template: str
    response_schema: Dict[str, Any]
    result_key: str = "score"


@dataclass(frozen=True)
class SchemeDef:
    system_code: str
    name: str
    version: str
    valid_from: str
    valid_to: Optional[str]
    default_model: str
    checks: List[CheckDef]
    normalized_percent: bool = True


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def _default_evaluation_model() -> str:
    # normal transcription model (NOT diarize)
    return settings.AZURE_MODEL_CHAT_SCORE if settings.USE_AZURE_OPENAI == "Y" else settings.OPENAI_MODEL_CHAT_SCORE


def load_scheme(
    scheme_json_path: str | Path,
    *,
    call_info: Optional[Dict[str, Any]] = None,
) -> SchemeDef:
    """
    Loads scheme.json and resolves relative prompt/schema/check file references.

    If call_info is provided, this function will also include checks configured
    under scheme["conditional_checks"] (or legacy "conditional_checks") whose
    keys match ANY value from call_info (case-insensitive). Matching conditional
    checks are appended to the base scheme["checks"] list (deduplicated).
    """
    scheme_path = Path(scheme_json_path).resolve()
    root = scheme_path.parent
    scheme_raw = _read_json(scheme_path)

    # Resolve base + conditional check file list (relative paths)
    check_files: List[str] = list(scheme_raw.get("checks", []))
    cond_map = scheme_raw.get("conditional_checks") or {}

    if call_info and isinstance(cond_map, dict):
        # Extract all values from call_info and normalize for matching
        call_values = {
            str(v).strip().upper()
            for v in call_info.values()
            if v is not None and str(v).strip() != ""
        }

        # Add conditional checks whose key matches any call_info value
        for cond_key, cond_checks in cond_map.items():
            key_norm = str(cond_key).strip().upper()
            if key_norm in call_values and isinstance(cond_checks, list):
                for rel_path in cond_checks:
                    if isinstance(rel_path, str) and rel_path not in check_files:
                        check_files.append(rel_path)

    checks: List[CheckDef] = []
    for rel_check_path in check_files:
        chk = _read_json((root / rel_check_path).resolve())

        system_prompt_template = _read_text(root / chk["system_prompt_file"])
        user_prompt_template = _read_text(root / chk["user_prompt_file"])
        response_schema = _read_json(root / chk["response_schema_file"])

        checks.append(
            CheckDef(
                id=chk["id"],
                desc=chk["desc"],
                weight=float(chk["weight"]),
                max_points=int(chk["max_points"]),
                temperature=float(chk.get("temperature", 0.0)),
                schema_name=chk.get("schema_name", f"{chk['id']}_schema"),
                system_prompt_template=system_prompt_template,
                user_prompt_template=user_prompt_template,
                response_schema=response_schema,
                result_key=chk.get("result_key", "score"),
            )
        )

    scoring = scheme_raw.get("scoring", {})

    return SchemeDef(
        system_code=scheme_raw["system_code"],
        name=scheme_raw.get("name", scheme_raw["system_code"]),
        version=scheme_raw.get("version", scheme_raw.get("ver", "0.0.0")),
        valid_from=scheme_raw.get("valid_from", ""),
        valid_to=scheme_raw.get("valid_to"),
        default_model=scheme_raw.get("default_model", "gpt-4o"),
        checks=checks,
        normalized_percent=bool(scoring.get("normalized_percent", True)),
    )


async def async_run_check(*, transcript_text: str, metadata: str, check: CheckDef, model: str) -> Dict[str, Any]:
    """
    Runs ONE check and returns a normalized record for reporting.
    """
    user_prompt = check.user_prompt_template.format(transcript=transcript_text.strip(), metadata=metadata.strip())
    system_prompt = check.system_prompt_template.format(transcript=transcript_text.strip(), metadata=metadata.strip())

    # log.level = logging.INFO
    # log.debug(f"User prompt for check '{check.id}': {user_prompt}") 
    # log.debug(f"System prompt for check '{check.id}': {system_prompt}")

    log.info(f"Started evaluation check '{check.id}' with model: '{model}'")
    completion = await async_chat_completion_with_format (
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=model,
        temperature=check.temperature,
        format_schema=check.response_schema,
        schema_name=check.schema_name,
    )
    log.info(f"Executed evaluation check '{check.id}' with model: '{model}'")
    # log.info("\n"+str(completion.usage))

    log.debug(f"Check '{check.id}' completion: {completion}")

    payload = json.loads(completion.choices[0].message.content)
    score = int(payload[check.result_key])

    # Guardrails: clamp score into allowed range
    score = max(0, min(score, check.max_points))

    return {
        "id": check.id,
        "desc": check.desc,
        "score": score,
        "max_points": check.max_points,
        "weight": check.weight,
        "weighted_score": score * check.weight,
        "weighted_max": check.max_points * check.weight,
        "model": model,
        "raw": payload,
    }






def _normalize_prev_detail(*, prev: Dict[str, Any], chk: CheckDef) -> Optional[Dict[str, Any]]:
    if not isinstance(prev, dict):
        return None

    if (prev.get("id") or "").strip() != chk.id:
        return None

    # If previously failed -> force recompute next run
    if (prev.get("status") or "").lower() == "error":
        return None

    if "score" not in prev or prev["score"] is None:
        return None
    try:
        score = int(prev["score"])
    except Exception:
        return None

    score = max(0, min(score, chk.max_points))

    return {
        "id": chk.id,
        "desc": chk.desc,
        "score": score,
        "max_points": chk.max_points,
        "weight": chk.weight,
        "weighted_score": float(score) * float(chk.weight),
        "weighted_max": float(chk.max_points) * float(chk.weight),
        "model": prev.get("model"),
        "raw": prev.get("raw") if isinstance(prev.get("raw"), dict) else None,
    }


def _error_detail(*, chk: CheckDef, model: str, err: Exception) -> Dict[str, Any]:
    return {
        "id": chk.id,
        "desc": chk.desc,
        "score": None,  # important: not computed
        "max_points": chk.max_points,
        "weight": chk.weight,
        "weighted_score": None,
        "weighted_max": None,
        "model": model,
        "status": "error",
        "error": str(err),
    }







async def async_run_scheme(
    *,
    transcript_text: str,
    metadata: str,
    scheme: SchemeDef,
    model_override: Optional[str] = None,
    prev_result: Optional[Dict[str, Any]] = None,
    interrupts_analysis: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, Any], bool]:
    """
    Returns: (result, success)

    success == True  -> all checks computed successfully, totals/percent calculated
    success == False -> at least one check errored, totals/percent are NOT calculated
                       (but already computed details are returned)
    """
    if not model_override:
        model = _default_evaluation_model()
    else:
        model = model_override

    scheme_checks: List[CheckDef] = list(scheme.checks)
    scheme_ids = {c.id for c in scheme_checks}

    # ingest prev_result details
    prev_details_by_id: Dict[str, Dict[str, Any]] = {}
    if isinstance(prev_result, dict):
        prev_details = prev_result.get("details")
        if isinstance(prev_details, list):
            for item in prev_details:
                if isinstance(item, dict):
                    item_id = (item.get("id") or "").strip()
                    if item_id in scheme_ids:
                        prev_details_by_id[item_id] = item

    details: List[Dict[str, Any]] = []
    had_error = False

    # (optional) keep partial totals for debugging/visibility, even if final score isn't computed
    partial_weighted = 0.0
    partial_weighted_max = 0.0

    if interrupts_analysis:
        details.append(interrupts_analysis)
        ws = interrupts_analysis.get("weighted_score")
        wm = interrupts_analysis.get("weighted_max")
        if isinstance(ws, (int, float)) and isinstance(wm, (int, float)):
            partial_weighted += float(ws)
            partial_weighted_max += float(wm)

    for chk in scheme_checks:
        reused: Optional[Dict[str, Any]] = None

        if chk.id in prev_details_by_id:
            reused = _normalize_prev_detail(prev=prev_details_by_id[chk.id], chk=chk)

        if reused is None:
            try:
                # raise Exception("Test exception for debugging")
                rec = await async_run_check(transcript_text=transcript_text, metadata=metadata, check=chk, model=model)
            except Exception as e:
                had_error = True
                log.exception(
                    "run_check failed: system=%s scheme=%s v%s check_id=%s model=%s",
                    scheme.system_code, scheme.name, scheme.version, chk.id, model
                )
                rec = _error_detail(chk=chk, model=model, err=e)
        else:
            rec = reused
            if not rec.get("model"):
                rec["model"] = model

        details.append(rec)

        # accumulate partial totals only if this record has numeric weighted fields
        ws = rec.get("weighted_score")
        wm = rec.get("weighted_max")
        if isinstance(ws, (int, float)) and isinstance(wm, (int, float)):
            partial_weighted += float(ws)
            partial_weighted_max += float(wm)



    result: Dict[str, Any] = {
        "system_code": scheme.system_code,
        "scheme_name": scheme.name,
        "scheme_version": scheme.version,
        "model": model,
        "details": details,

        # helpful diagnostics even on failure
        "had_error": had_error,
        "partial_weighted_score": partial_weighted,
        "partial_weighted_max": partial_weighted_max,
    }

    result["score_percent"] = None
    if had_error:
        # final score is NOT calculated
        result["total_weighted_score"] = None
        result["total_weighted_max"] = None
        result["score_percent"] = None
        return result, False

    # final score calculated only if everything succeeded
    result["total_weighted_score"] = partial_weighted
    result["total_weighted_max"] = partial_weighted_max

    if scheme.normalized_percent and partial_weighted_max > 0:
        result["score_percent"] = round(partial_weighted / partial_weighted_max * 100.0, 2)

    return result, True







