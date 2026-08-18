"""Application settings, loaded from environment variables and ``.env``.

Every tunable value in the project lives here. Nothing else should read
``os.environ`` directly or hardcode a path.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Self

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application configuration.

    Field names map case-insensitively to environment variables, so
    ``CHUNK_MAX_TOKENS=1200`` in ``.env`` populates ``chunk_max_tokens``.
    Values arrive from the environment as strings and are converted and
    validated here, so a bad value fails at startup instead of deep inside
    an ingestion loop.

    C#/Java equivalent: an ``IOptions<T>`` bound from configuration, except
    the type conversion and range checks are declared inline on the fields.
    """

    # `model_config` is pydantic-settings' own configuration hook. `extra="ignore"`
    # means unrelated variables already in your environment (PATH, etc.) are
    # skipped rather than raising.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Sources ---------------------------------------------------------
    google_drive_credentials_path: Path | None = None

    # --- Storage ---------------------------------------------------------
    vector_store_path: Path = Path("data/index")

    # --- Ingestion -------------------------------------------------------
    # `Field(gt=0)` attaches a validation constraint to the field. This is
    # closest to a DataAnnotations attribute like [Range] in C#, but unlike
    # plain type hints these ARE enforced at runtime.
    max_file_size_bytes: int = Field(default=50 * 1024 * 1024, gt=0)  # 50 MB

    # --- Chunking --------------------------------------------------------
    chunk_min_tokens: int = Field(default=500, gt=0)
    chunk_max_tokens: int = Field(default=800, gt=0)
    chunk_overlap_ratio: float = Field(default=0.15, ge=0.0, lt=1.0)

    @field_validator("google_drive_credentials_path", mode="before")
    @classmethod
    def _blank_path_is_none(cls, value: Any) -> Any:
        """Treat an empty env var as "not configured".

        ``.env.example`` ships ``GOOGLE_DRIVE_CREDENTIALS_PATH=`` with no value.
        Without this, pydantic would coerce that empty string into ``Path("")``,
        which silently equals ``Path(".")`` — the current directory — and the
        Drive connector would think it had been given a credentials file.

        ``mode="before"`` runs this ahead of type conversion, while the value is
        still the raw string from the environment.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _check_chunk_bounds(self) -> Self:
        """Reject a chunk window that can never be satisfied.

        Runs after all individual fields are validated, so it can compare them
        against each other.
        """
        if self.chunk_min_tokens > self.chunk_max_tokens:
            raise ValueError(
                f"chunk_min_tokens ({self.chunk_min_tokens}) must be <= "
                f"chunk_max_tokens ({self.chunk_max_tokens})"
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the shared Settings instance.

    ``@lru_cache`` memoises the result, so the ``.env`` file is parsed once and
    every caller gets the same object — effectively a lazily-constructed
    singleton, like registering ``AddSingleton<Settings>()`` in a DI container.

    Tests that need different values should construct ``Settings(...)`` directly
    rather than going through this function, or call
    ``get_settings.cache_clear()``.
    """
    return Settings()
