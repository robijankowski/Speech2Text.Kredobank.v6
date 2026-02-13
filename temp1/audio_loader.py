import os
from pathlib import Path
import logging

from app.infrastructure.db.pool import transaction
from app.infrastructure.db.repositories.conversation_repo import ConversationRepository
from app.infrastructure.storage.audio_paths import get_audio_file_path

log = logging.getLogger("audio_loader")

# ============================================================
#  class class AudioLoader
# ============================================================

class AudioLoaderError(Exception):
    pass


class AudioLoader:

    # ============================================================
    #  main AudioLoader function
    # ============================================================

    async def load(self, systemId: str, requestId: str) -> Path:
        conv_repo = ConversationRepository()

        log.debug(f"{systemId}: {requestId} Audio file loading")

        # 1) Get filename + date from DB
        async with transaction() as conn:
            info = await conv_repo.get_audio_file_info(
                conn,
                systemId,
                requestId
            )

        if not info:
            log.error(f"{systemId}: {requestId} Conversation not found")
            raise AudioLoaderError(f"{systemId}: {requestId} Conversation not found")

        log.debug(f"{systemId}: {requestId} Audio file info: {info}")

        fileName = info["file_name"]
        conversationDate = info["conversation_date"]

        # 2) Build expected storage path
        dateStr = conversationDate.date().isoformat()
        audio_path = get_audio_file_path(systemId, requestId, dateStr, fileName)

        log.debug(f"{systemId}: {requestId} Audio file path: {audio_path}")

        # 3) Validate existence
        if not audio_path.exists():
            log.error(f"{systemId}: {requestId} Audio file not found on disk: {audio_path}")
            raise AudioLoaderError(
                f"Audio file not found: {audio_path}"
            )

        log.info(f"{systemId}: {requestId} Audio loaded: {audio_path}")

        return audio_path
