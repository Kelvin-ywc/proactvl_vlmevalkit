import os
from functools import lru_cache

class Settings:
    USE_DUMMY: bool = os.getenv("USE_DUMMY", "0") != "0"
    ALLOWED_ORIGINS: list[str] = os.getenv("ALLOWED_ORIGINS", "*").split(",")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
