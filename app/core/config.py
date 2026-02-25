from dotenv import load_dotenv
import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

models_config_path = os.path.join(os.path.dirname(__file__), '../config', '.models')
keys_config_path = os.path.join(os.path.dirname(__file__), '../config', '.keys')
app_config_path = os.path.join(os.path.dirname(__file__), '../config', '.config')
env_config_path = os.path.join(os.path.dirname(__file__), '../config', '.env')
tr_config_path = os.path.join(os.path.dirname(__file__), '../config', '.tr_config')

# zostawiasz jak jest (ładujesz 3 pliki do ENV)
load_dotenv(dotenv_path=models_config_path)
load_dotenv(dotenv_path=keys_config_path)
load_dotenv(dotenv_path=app_config_path)
load_dotenv(dotenv_path=env_config_path)
load_dotenv(dotenv_path=tr_config_path) 
 
class AppSettings(BaseSettings):
    API_V1_PREFIX: str = "/api/v1"
    BASE_URL: str = "http://localhost:8000"


    # ----------------------------- .config
    TRANSCRIBE_LOGS_DIR: str
    TRANSCRIBE_LOGS_PREF: str
    TRANSCRIBE_LOGGER_NAME: str = "tlog"

    ANALYZER_LOGS_DIR: str
    ANALYZER_LOGS_PREF: str
    ANALYZER_LOGGER_NAME: str = "anlog"

    ADMIN_CONSOLE_LOGS_DIR: str
    ADMIN_CONSOLE_LOGS_PREF: str
    ADMIN_CONSOLE_LOGGER_NAME: str = "aclog"    

    # ----------------------------- .env
    CALLBACK_SECRET: str

    LOG_LEVEL: str

    AUDIO_STORAGE_DIR: str
 
    LLM_PROVIDER: str 
    LLM_CHAT_MODEL: str

    LLM_TIMEOUT: int 
    LLM_CONNECT_TIMEOUT: int 



    # ----------------------------- .keys
    OPENAI_API_KEY: str
    AZURE_OPENAI_API_KEY: str = ""



    # ----------------------------- .models
    USE_AZURE_OPENAI: str = "N"

    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_API_VERSION: str


    OPENAI_MODEL_TRANSCRIBE_STEREO: str
    OPENAI_MODEL_TRANSCRIBE_DIARIZE: str

    OPENAI_MODEL_CHAT_TRS_DETECT_PLAYER: str 
    OPENAI_MODEL_CHAT_TRS_SPLIT_INTO_ROLES: str

    OPENAI_MODEL_CHAT_SUMMARY: str 
    OPENAI_MODEL_CHAT_SCORE: str 
    OPENAI_MODEL_CHAT_ANALYSIS_ENGINE: str


    AZURE_MODEL_TRANSCRIBE_STEREO: str
    AZURE_MODEL_TRANSCRIBE_DIARIZE: str

    AZURE_MODEL_CHAT_TRS_DETECT_PLAYER: str 
    AZURE_MODEL_CHAT_TRS_SPLIT_INTO_ROLES: str
    
    AZURE_MODEL_CHAT_SUMMARY: str 
    AZURE_MODEL_CHAT_SCORE: str 
    AZURE_MODEL_CHAT_ANALYSIS_ENGINE: str

    
    # ----------------------------- .tr_config
    TR_TEMP_ROOT_DIR: str

    TR_EVALUATION_CONFIGS_ROOT: str

    TR_SCORE_PARALLEL_REQUESTS: int
    TR_ANALYSIS_PARALLEL_REQUESTS: int

    TR_EVALUATE_INTERRUPTS: str

    # Pydantic v2 replacement for class Config
    model_config = SettingsConfigDict(
        env_prefix="",
        extra="ignore",  # zgodnie z zaleceniem kompatybilności dla v1-style settings
    )



def _ensure_dirs_exist(s: "AppSettings") -> None:
    # Add/remove fields as you need
    dir_fields = [
        "TRANSCRIBE_LOGS_DIR",
        "ANALYZER_LOGS_DIR",
        "ADMIN_CONSOLE_LOGS_DIR",
        "AUDIO_STORAGE_DIR",
        "TR_TEMP_ROOT_DIR",
        "TR_EVALUATION_CONFIGS_ROOT",
    ]

    for field in dir_fields:
        path = getattr(s, field, None)
        if not path:
            continue

        p = Path(str(path)).expanduser()
        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Cannot create directory for {field}={p}: {e}") from e


settings = AppSettings()
_ensure_dirs_exist(settings)