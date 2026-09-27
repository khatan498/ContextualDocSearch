"""Application settings, loaded from environment variables and ``.env``.

Every tunable value in the project lives here. Nothing else should read
``os.environ`` directly or hardcode a path.
"""

from functools import lru_cache
from pathlib import Path
from typing import Self

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.modes import PERSONAL_MODE_MESSAGE, AppMode, PersonalModeUnavailableError

# The repository root: this file is app/config.py, so two levels up.
# `__file__` is the path of the current module — the closest C# analogue is
# `Assembly.GetExecutingAssembly().Location`. Anchoring on it rather than on the
# current working directory is what makes every relative path below mean the
# same thing whether the process starts in the repo root, in scripts/, or
# wherever a web server decides to launch it.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


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
        # Absolute, for the same reason as the paths below: a bare ".env" is
        # looked up in the working directory, so running from anywhere else
        # silently ignored your settings and fell back to the defaults.
        env_file=PROJECT_ROOT / ".env",
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
    #
    # Declared relative, then anchored to PROJECT_ROOT by `_anchor_to_root`.
    # `validate_default=True` is required for that: pydantic trusts default
    # values and skips validators on them unless told otherwise.
    sample_docs_path: Path = Field(
        default=Path("data/sample_docs"), validate_default=True
    )

    # --- Embeddings ------------------------------------------------------
    # Downloaded from Hugging Face on first use and cached under
    # ~/.cache/huggingface. bge-base-en-v1.5 accepts 512 tokens and produces
    # 768-dimension vectors; the chunk window below is sized against that limit
    # and the pairing is enforced in EmbeddingModel, not here — reading the
    # model's limit means loading the model, which config must never do.
    embedding_model_name: str = "BAAI/bge-base-en-v1.5"

    # How many chunks are encoded per forward pass. Larger is faster but uses
    # more memory; 32 is comfortable on CPU.
    embedding_batch_size: int = Field(default=32, gt=0)

    # --- Storage ---------------------------------------------------------
    # Holds both the Chroma database and the BM25 file. Anchored like above.
    vector_store_path: Path = Field(default=Path("data/index"), validate_default=True)

    # --- Ingestion -------------------------------------------------------
    # `Field(gt=0)` attaches a validation constraint to the field. This is
    # closest to a DataAnnotations attribute like [Range] in C#, but unlike
    # plain type hints these ARE enforced at runtime.
    max_file_size_bytes: int = Field(default=50 * 1024 * 1024, gt=0)  # 50 MB

    # --- Chunking --------------------------------------------------------
    # Sized for the embedding model above: 480 leaves 32 tokens of headroom
    # under bge-base's 512 limit for the special tokens it adds. Raising these
    # past the model's limit is caught at index time by EmbeddingModel.
    chunk_min_tokens: int = Field(default=350, gt=0)
    chunk_max_tokens: int = Field(default=480, gt=0)
    chunk_overlap_ratio: float = Field(default=0.15, ge=0.0, lt=1.0)

    # A decorator stack: `@field_validator` registers the method with pydantic,
    # and `@classmethod` makes it receive the class rather than an instance,
    # because it runs while the instance is still being built. Decorators apply
    # bottom-up, so `@classmethod` must sit closest to the function.
    @field_validator("sample_docs_path", "vector_store_path", mode="after")
    @classmethod
    def _anchor_to_root(cls, value: Path) -> Path:
        """Resolve a relative path against the project root, not the CWD.

        Absolute paths are returned untouched, so an explicit override in
        ``.env`` or the environment always wins.

        Args:
            value: The path as declared or as read from the environment,
                already converted from a string by pydantic.

        Returns:
            An absolute path.
        """
        return value if value.is_absolute() else PROJECT_ROOT / value

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
