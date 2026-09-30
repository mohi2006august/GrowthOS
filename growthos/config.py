"""Runtime configuration. Every value comes from the environment (never from code)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Role = Literal["marketer", "cfo"]
CopyBackend = Literal["template", "claude", "vllm"]

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}


class ConfigError(ValueError):
    """Raised when an environment variable holds an unusable value."""


@dataclass(frozen=True)
class ApiUser:
    name: str
    role: Role
    token: str


@dataclass(frozen=True)
class Settings:
    database_path: Path = Path("var/growthos.db")

    # Agent
    agent_model: str = "claude-opus-5-5"
    agent_effort: str = "high"
    anthropic_api_key_present: bool = False

    # Guardrails. These are hard limits: the stored policy may tighten, never loosen them.
    spend_cap_daily_usd: float = 4000.0
    min_runway_months: float = 18.0
    max_cac_usd: float = 1500.0
    marketer_approval_limit_usd: float = 5000.0
    allow_auto_pause: bool = False

    # Experiments
    conversion_lag_days: int = 3

    # API auth
    api_users: tuple[ApiUser, ...] = field(default_factory=tuple)

    # Copy model
    copy_backend: CopyBackend = "template"
    vllm_base_url: str = "http://localhost:8001/v1"
    lora_adapter: str = "brand-voice-lora"
    lora_adapter_version: str = "v1"

    @property
    def copy_model_tag(self) -> str:
        return f"{self.lora_adapter}@{self.lora_adapter_version}"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        e = os.environ if env is None else env
        copy_backend = e.get("GROWTHOS_COPY_BACKEND", "template").strip().lower()
        if copy_backend not in ("template", "claude", "vllm"):
            raise ConfigError(f"GROWTHOS_COPY_BACKEND must be template|claude|vllm, got {copy_backend!r}")
        return cls(
            database_path=Path(e.get("GROWTHOS_DB", "var/growthos.db")),
            agent_model=e.get("GROWTHOS_AGENT_MODEL", "claude-opus-5-5"),
            agent_effort=e.get("GROWTHOS_AGENT_EFFORT", "high"),
            anthropic_api_key_present=bool(e.get("ANTHROPIC_API_KEY") or e.get("ANTHROPIC_AUTH_TOKEN")),
            spend_cap_daily_usd=_positive(e, "SPEND_CAP_DAILY_USD", 4000.0),
            min_runway_months=_positive(e, "MIN_RUNWAY_MONTHS", 18.0),
            max_cac_usd=_positive(e, "MAX_CAC_USD", 1500.0),
            marketer_approval_limit_usd=_positive(e, "MARKETER_APPROVAL_LIMIT_USD", 5000.0),
            allow_auto_pause=_bool(e, "ALLOW_AUTO_PAUSE", False),
            conversion_lag_days=int(_positive(e, "CONVERSION_LAG_DAYS", 3.0)),
            api_users=_api_users(e.get("GROWTHOS_API_TOKENS", "")),
            copy_backend=copy_backend,  # type: ignore[arg-type]
            vllm_base_url=e.get("VLLM_BASE_URL", "http://localhost:8001/v1"),
            lora_adapter=e.get("LORA_ADAPTER", "brand-voice-lora"),
            lora_adapter_version=e.get("LORA_ADAPTER_VERSION", "v1"),
        )


def _positive(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be a number, got {raw!r}") from exc
    if value <= 0:
        raise ConfigError(f"{key} must be positive, got {value}")
    return value


def _bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = env.get(key)
    if raw is None:
        return default
    lowered = raw.strip().lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    raise ConfigError(f"{key} must be a boolean, got {raw!r}")


def _api_users(raw: str) -> tuple[ApiUser, ...]:
    users: list[ApiUser] = []
    for item in filter(None, (part.strip() for part in raw.split(","))):
        parts = item.split(":")
        if len(parts) != 3 or not all(parts):
            raise ConfigError("GROWTHOS_API_TOKENS entries must look like name:role:token")
        name, role, token = parts
        if role not in ("marketer", "cfo"):
            raise ConfigError(f"GROWTHOS_API_TOKENS role must be marketer|cfo, got {role!r}")
        users.append(ApiUser(name=name, role=role, token=token))  # type: ignore[arg-type]
    return tuple(users)
