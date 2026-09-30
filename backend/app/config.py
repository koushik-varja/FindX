from __future__ import annotations

import os
from pathlib import Path


class Settings:
    repo_root = Path(__file__).resolve().parents[2]
    database_url = os.getenv("DATABASE_URL", "sqlite:///./findx.db")
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    search_mode = os.getenv("FINDX_MODE", "lightweight").strip().lower()
    admin_key = os.getenv("ADMIN_KEY", "replace-this-value")
    cors_origins = [
        value.strip()
        for value in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
        if value.strip()
    ]
    test_schema_create = os.getenv("FINDX_TEST_CREATE_SCHEMA", "0") == "1"


settings = Settings()
