"""Application settings, loaded from environment variables and ``.env``.

Every tunable value in the project lives here. Nothing else should read
``os.environ`` directly or hardcode a path.
"""

from functools import lru_cache
from pathlib import Path
from typing import Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.modes import PERSONAL_MODE_MESSAGE, AppMode, PersonalModeUnavailableError


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

    # --- Mode ------------------------------------------------------------
    # StrEnum members are strings, so a plain `APP_MODE=demo` line in .env
    # converts with no custom parsing. `personal` is accepted by the type and
    # then rejected below, so the user gets the explanation rather than
    # "not a valid enumeration member".
    app_mode: AppMode = AppMode.DEMO

    # --- Sources ---------------------------------------------------------
    # The bundled demo corpus. This is the only content the app ever reads.
    sample_docs_path: Path = Path("data/sample_docs")

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

    @model_validator(mode="after")
    def _reject_personal_mode(self) -> Self:
        """Refuse to start in personal mode.

        Personal mode is advertised but not implemented. Failing at startup —
        rather than somewhere deep inside a scan — means the app never
        half-starts in a mode it cannot honour.

        Raises a plain ``RuntimeError`` subclass rather than ``ValueError``, so
        pydantic lets it through untouched instead of folding it into a
        ``ValidationError`` and burying the explanation.
        """
        if self.app_mode is AppMode.PERSONAL:
            raise PersonalModeUnavailableError(PERSONAL_MODE_MESSAGE)
        return self

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
