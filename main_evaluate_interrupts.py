from __future__ import annotations

from core.config import settings
from transcribe.utilities.evaluation_interrupts import async_detect_agent_interruptions

from core.logger import get_logger, shutdown_logger
log = get_logger(__name__)

if __name__ == "__main__":
    files = [
        "./test/Одночасна розмова фахівця і клієнта.wav",
        # "./test/Святослав одночасна розмова з клієнтом , деколи перебиває.wav",
        "./test/test_call.wav",
        # "./test/sources/AUTO-2025-06-30-10-08-380988442847-1087-1751267274.1529139-stereo1.wav",
        # "./test/sources/AUTO-2025-06-30-09-05-380963799218-1096-1751263515.1528148-stereo1.wav",
        # "./test/sources/AUTO-2025-06-30-10-17-380639093150-1006-1751267854.1529303-stereo1.wav",
        # "./test/sources/AUTO-2025-06-30-12-05-380990805468-1098-1751274275.1530761-stereo1.wav",
        "./test/sources/OUT-2025-06-30-09-34-1099-0500814269-1751265256.1528626-stereo1.wav" 
    ]

    for f in files:
        settings.USE_AZURE_OPENAI = "Y"  # force Azure for testing
        print(f"\n\n\n#########################\nFILE: '{f}'\n#########################")
        res = async_detect_agent_interruptions( f )
        print("\nResult:", res)
    shutdown_logger()  # ensure all logs flushed and file handlers closed