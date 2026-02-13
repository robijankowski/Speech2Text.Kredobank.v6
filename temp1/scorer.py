import json
import logging

from datetime import date, datetime
from app.transcribe.core.tr_config import tr_settings
from app.transcribe.utilities.evaluation_engine import load_scheme, run_scheme
from app.transcribe.utilities.evaluation_engine_regs import load_active_scheme

log = logging.getLogger("conversation_scorer")

# ============================================================
#  class class Scorer
# ============================================================

class Scorer:

    # ============================================================
    #  main scoring function
    # ============================================================
    def score(self, systemId: str, requestId: str, transcript: str) -> tuple[float, dict]:
        log.info(f"{systemId}: {requestId} Calculate agent score..." )

        log.info("\n\n" + "="*30 + " Loading current evaluation scheme " + "="*30)
        scheme = load_active_scheme(tr_settings.TR_EVALUATION_CONFIGS_ROOT, "CRM_TEST", call_date=date(2026, 1, 5) )        
        log.info(f"\nUsing scheme: {scheme.system_code} v{scheme.version}\n")

        log.info("\n\n" + "="*30 + " Running evaluation scheme " + "="*30)
        result, success = run_scheme(transcript_text=transcript, scheme=scheme)
        log.info("\n\n" + "="*30 + " Evaluation Results " + "="*30)
        log.info(f"\n\n{success}\n\n" + str(json.dumps(result, indent=2)))        

        return result["score_percent"], result
