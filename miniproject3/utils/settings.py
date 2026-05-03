from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    RANDOM_SEED: int = Field(default=42)
    DATA_DIR: Path = Path("/tmp/MUTAG")
    OUTPUT_DIR: Path = Path("outputs")


settings = Settings()
