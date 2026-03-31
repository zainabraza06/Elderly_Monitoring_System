from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(slots=True)
class Settings:
    app_name: str = "Elderly Monitoring Backend"
    api_prefix: str = "/api/v1"
    debug: bool = False
    allowed_origins: list[str] = field(default_factory=lambda: ["*"])
    recent_event_limit: int = 500
    offline_artifacts_dir: str = "results/artifacts"


def get_settings() -> Settings:
    raw_origins = os.getenv("EMS_ALLOWED_ORIGINS", "*")
    allowed_origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
    return Settings(
        app_name=os.getenv("EMS_APP_NAME", "Elderly Monitoring Backend"),
        api_prefix=os.getenv("EMS_API_PREFIX", "/api/v1"),
        debug=os.getenv("EMS_DEBUG", "false").lower() == "true",
        allowed_origins=allowed_origins or ["*"],
        recent_event_limit=int(os.getenv("EMS_RECENT_EVENT_LIMIT", "500")),
        offline_artifacts_dir=os.getenv("EMS_OFFLINE_ARTIFACTS_DIR", "results/artifacts"),
    )
