from app.core.config import settings
from app.core.logger import log

import json
from typing import Any, Dict

from app.openai_tools.openai_client_text import  async_chat_completion_with_format


def _default_summary_model() -> str:
    return settings.AZURE_MODEL_CHAT_SUMMARY if settings.USE_AZURE_OPENAI == "Y" else settings.OPENAI_MODEL_CHAT_SUMMARY    



SYSTEM_PROMPT_EXT = (
    "You extract CRM fields from Ukrainian call transcripts. "
    "Be strictly factual. Do not invent information. "
    "If a field is not present, return empty string. "
    'Never output "none", null, "N/A", or placeholders.'
)

USER_PROMPT_EXT = """
Інформація для передачі:
summary
причина прострочки платежу
домовленість про платіж/реструктуризацію
дата домовленості
сума домовленості

Класифікатор причин прострочення:
Втрата роботи або доходу
Затримка заробітньої плати або виплат
затримка платежів від дебіторів покупців
зменшення виробництва  зменшення виручки
Знаходится за кордоном або відрядженні
Не оплачує через сімейні обставини
Не розуміє звідки заборгованість
Отримує дохід в інший день
Технічні проблеми з внесенням платежу
Хвороба, лікарня
інше

ТРАНСКРИПТ:
{transcript_text}
""".strip()




def _crm_summary_ext_schema() -> Dict[str, Any]:
    # IMPORTANT: return ONLY the JSON Schema object.
    # The router (async_chat_completion_with_format) will wrap it into response_format itself.

    reasons = [
        "",
        "Втрата роботи або доходу",
        "Затримка заробітньої плати або виплат",
        "затримка платежів від дебіторів покупців",
        "зменшення виробництва  зменшення виручки",
        "Знаходится за кордоном або відрядженні",
        "Не оплачує через сімейні обставини",
        "Не розуміє звідки заборгованість",
        "Отримує дохід в інший день",
        "Технічні проблеми з внесенням платежу",
        "Хвороба, лікарня",
        "інше",
    ]

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "summary": {"type": "string"},
            "reason_overdue": {"type": "string", "enum": reasons},
            "agreement_type": {"type": "string", "enum": ["", "платіж", "реструктуризація"]},
            "agreement_date": {
                "type": "string",
                "pattern": r"^$|^\d{4}-\d{2}-\d{2}$"
            },
            "agreement_amount": {"type": "string"},
        },
        "required": [
            "summary",
            "reason_overdue",
            "agreement_type",
            "agreement_date",
            "agreement_amount",
        ],
    }




async def async_generate_crm_summary_for_call_scenario_ext(
    transcript_text: str,
    model: str = "",
) -> Dict[str, str]:
    """
    Returns structured CRM summary fields:
      summary, reason_overdue, agreement_type, agreement_date, agreement_amount
    Missing info => "".
    Uses async_chat_completion_with_format (json_schema).
    """

    if not model:
        model = _default_summary_model()

    user_prompt = USER_PROMPT_EXT.format(transcript_text=transcript_text or "")

    resp = await async_chat_completion_with_format(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_EXT},
            {"role": "user", "content": user_prompt},
        ],
        format_schema=_crm_summary_ext_schema(),   # now plain schema ✅
        schema_name="crm_summary_ext",
    )


    # Depending on SDK path, content can be already JSON or stringified JSON.
    content = (resp.choices[0].message.content or "").strip()

    try:
        data = json.loads(content) if content else {}
    except Exception as e:
        log.error(f"Exceptione {e} when running summary")
        data = {}

    # Final guardrails (in case model somehow slips through)
    out = {
        "summary": str(data.get("summary", "") or ""),
        "reason_overdue": str(data.get("reason_overdue", "") or ""),
        "agreement_type": str(data.get("agreement_type", "") or ""),
        "agreement_date": str(data.get("agreement_date", "") or ""),
        "agreement_amount": str(data.get("agreement_amount", "") or ""),
    }

    # Never allow placeholders
    for k, v in list(out.items()):
        if v.strip().lower() in {"none", "null", "n/a", "na", "-", "невідомо"}:
            out[k] = ""

    return out