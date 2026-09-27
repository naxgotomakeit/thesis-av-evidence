from __future__ import annotations

import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any


SENSITIVE_KEYS = {
    "api_key", "apikey", "authorization", "credential", "secret",
    "access_token", "bearer_token", "refresh_token",
}


# This file is deliberately outside the thesis repository.  It is a runtime
# fallback only; provider/model/pricing remain in the versioned experiment
# configuration.
PRIVATE_PLANNER_DOTENV_PATH = Path(
    "${PROJECT_MSC_ROOT}/.env.planner_api"
)


def _is_sensitive_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return (
        normalized in SENSITIVE_KEYS
        or normalized.endswith("_api_key")
        or normalized.endswith("_secret")
        or normalized.endswith("_credential")
    )


class CredentialError(RuntimeError):
    """A credential is unavailable; messages contain only the environment name."""


class SecretCredential:
    """An in-memory, non-serializable credential with a permanently redacted repr."""

    __slots__ = ("_env_name", "_value")

    def __init__(self, env_name: str, value: str):
        self._env_name = env_name
        self._value = value

    @property
    def env_name(self) -> str:
        return self._env_name

    def reveal_for_request(self) -> str:
        return self._value

    def safe_record(self) -> dict[str, Any]:
        return {"credential_env": self._env_name, "credential_loaded": True}

    def __repr__(self) -> str:
        return f"SecretCredential(env_name={self._env_name!r}, value=<redacted>)"

    __str__ = __repr__

    def __reduce__(self) -> Any:
        raise TypeError("SecretCredential objects cannot be serialized")


def load_credential(
    env_name: str,
    environ: Mapping[str, str] | None = None,
    private_dotenv_path: Path | None = None,
) -> SecretCredential:
    if not isinstance(env_name, str) or not env_name.strip():
        raise CredentialError("credential environment variable name is missing")
    source = os.environ if environ is None else environ
    if env_name in source:
        value = source[env_name]
        if not isinstance(value, str) or not value.strip():
            raise CredentialError(f"required credential environment variable {env_name} is empty")
        return SecretCredential(env_name, value)

    # An explicitly injected mapping is used by deterministic unit tests and
    # must never consult a developer's local credential file.
    if environ is not None:
        raise CredentialError(f"required credential environment variable {env_name} is missing")

    dotenv_path = private_dotenv_path or PRIVATE_PLANNER_DOTENV_PATH
    value = _load_private_dotenv_value(env_name, dotenv_path)
    if value is None:
        raise CredentialError(f"required credential environment variable {env_name} is missing")
    return SecretCredential(env_name, value)


def _load_private_dotenv_value(env_name: str, path: Path) -> str | None:
    """Read one credential assignment without exporting or exposing its value."""
    try:
        file_stat = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise CredentialError(
            f"private credential file for {env_name} cannot be accessed"
        ) from exc

    if not stat.S_ISREG(file_stat.st_mode):
        raise CredentialError(f"private credential file for {env_name} is not a regular file")
    if stat.S_IMODE(file_stat.st_mode) != 0o600:
        raise CredentialError(f"private credential file for {env_name} must have mode 0600")
    if file_stat.st_uid != os.geteuid():
        raise CredentialError(f"private credential file for {env_name} has an unexpected owner")

    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise CredentialError(
            f"private credential file for {env_name} cannot be read"
        ) from exc

    lines = content.splitlines()
    prefix = f"{env_name}="
    if len(lines) != 1 or not lines[0].startswith(prefix):
        raise CredentialError(
            f"private credential file for {env_name} must contain only its assignment"
        )
    value = lines[0][len(prefix):]
    if not value.strip():
        raise CredentialError(f"private credential file for {env_name} is empty")
    return value


def sanitize_for_persistence(value: Any, secrets: tuple[str, ...] = ()) -> Any:
    """Return a JSON-safe copy with sensitive keys and known secret values redacted."""
    if isinstance(value, SecretCredential):
        return value.safe_record()
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if _is_sensitive_key(str(key)):
                result[str(key)] = "<redacted>"
            else:
                result[str(key)] = sanitize_for_persistence(item, secrets)
        return result
    if isinstance(value, (list, tuple)):
        return [sanitize_for_persistence(item, secrets) for item in value]
    if isinstance(value, str):
        redacted = value
        for secret in secrets:
            if secret:
                redacted = redacted.replace(secret, "<redacted>")
        return redacted
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return sanitize_for_persistence(repr(value), secrets)
