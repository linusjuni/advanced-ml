from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    # General settings
    RANDOM_SEED: int = Field(default=69)

    # Paths
    DATA_DIR: Path = Path("/work3/s225224/data/advanced-ml/miniproject2")


settings = Settings()
