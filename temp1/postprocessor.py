import logging

from app.transcribe.utilities.summary_tools import generate_crm_summary_o4, format_summary_md

log = logging.getLogger("conversation_postprocessor")

# ============================================================
#  class class PostProcessor
# ============================================================

class PostProcessor:

    # ============================================================
    #  main postprocessing function
    # ============================================================
    def process(self, systemId: str, requestId: str, raw_text: str) -> tuple[str, str]:
        log.info(f"{systemId}: {requestId} Postprocess text..." )

        transcription = raw_text
        log.info("\n\n\n" + "="*30 + " Generating summary " + "="*30)
        summary = generate_crm_summary_o4(raw_text)
        log.info("\n" + summary)

        return transcription, summary
