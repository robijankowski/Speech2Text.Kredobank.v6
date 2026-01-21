from dotenv import load_dotenv
import os

from pydantic_settings import BaseSettings, SettingsConfigDict

models_config_path = os.path.join(os.path.dirname(__file__), '../config', '.models')
keys_config_path = os.path.join(os.path.dirname(__file__), '../config', '.keys')
app_config_path = os.path.join(os.path.dirname(__file__), '../config', '.config')

# zostawiasz jak jest (ładujesz 3 pliki do ENV)
load_dotenv(dotenv_path=models_config_path)
load_dotenv(dotenv_path=keys_config_path)
load_dotenv(dotenv_path=app_config_path)

class AppSettings(BaseSettings):
    API_V1_PREFIX: str = "/api/v1"
    BASE_URL: str = "http://localhost:8000"

    TRANSCRIBE_LOGS_DIR: str
    TRANSCRIBE_LOGS_PREF: str
    ANALYZER_LOGS_DIR: str
    ANALYZER_LOGS_PREF: str
    ADMIN_CONSOLE_LOGS_DIR: str
    ADMIN_CONSOLE_LOGS_PREF: str

    OPENAI_API_KEY: str

    OPENAI_MODEL_TRANSCRIBE_STEREO: str

    OPENAI_MODEL_CHAT_TRS_DETECT_PLAYER: str 
    OPENAI_MODEL_CHAT_TRS_SPLIT_INTO_ROLES: str

    OPENAI_MODEL_CHAT_SUMMARY: str 
    OPENAI_MODEL_CHAT_SCORE: str 
    OPENAI_MODEL_CHAT_ANALYSIS_ENGINE: str

    USE_AZURE_OPENAI: bool = "N"

    AZURE_OPENAI_API_KEY: str = ""
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_API_VERSION: str

    AZURE_MODEL_CHAT_SUMMARY: str 
    AZURE_MODEL_CHAT_SCORE: str 
    AZURE_MODEL_CHAT_ANALYSIS_ENGINE: str

    # Pydantic v2 replacement for class Config
    model_config = SettingsConfigDict(
        env_prefix="",
        extra="ignore",  # zgodnie z zaleceniem kompatybilności dla v1-style settings
    )

settings = AppSettings()
