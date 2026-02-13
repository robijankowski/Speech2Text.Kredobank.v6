from dotenv import load_dotenv
from pydantic_settings import BaseSettings
import os

tr_config_path = os.path.join(os.path.dirname(__file__), '../config', '.tr_config')

# Load the environment variables from both files
load_dotenv(dotenv_path=tr_config_path)


# Define your settings model with Pydantic's BaseSettings
class AppSettings(BaseSettings):
    
    TR_LOGGER_NAME: str

    TR_TEMP_ROOT_DIR: str

    TR_EVALUATION_CONFIGS_ROOT: str

    TR_ANALYSIS_PARALLEL_REQUESTS: int
    

    class Config:
        env_prefix = ""  # Since variable names are unique, we don't need a prefix
        # Pydantic automatically uses environment variables if available
        

# Instantiate your settings
settings = AppSettings()

