"""Tests for app.config."""

from pathlib import Path

import pytest

from app.config import Settings, get_settings
from app.modes import AppMode, PersonalModeUnavailableError


class TestDefaults:
    def test_defaults_to_demo_mode(self) -> None:
        assert Settings().app_mode is AppMode.DEMO

    def test_sample_docs_path(self) -> None:
        assert Settings().sample_docs_path == Path("data/sample_docs")

    def test_chunk_defaults(self) -> None:
        settings = Settings()
        assert settings.chunk_min_tokens == 500
        assert settings.chunk_max_tokens == 800
        assert settings.chunk_overlap_ratio == pytest.approx(0.15)


class TestScope:
    """The app is demo-only; no cloud credential settings should exist."""

    @pytest.mark.parametrize(
        "field_name",
        ["google_drive_credentials_path", "onedrive_credentials_path"],
    )
    def test_no_cloud_credential_fields(self, field_name: str) -> None:
        assert field_name not in Settings.model_fields


class TestPersonalMode:
    def test_rejected_when_passed_directly(self) -> None:
        with pytest.raises(PersonalModeUnavailableError):
            Settings(app_mode=AppMode.PERSONAL)

    def test_rejected_from_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # monkeypatch undoes the change after the test, so it cannot leak into
        # any other test in the session.
        monkeypatch.setenv("APP_MODE", "personal")
        with pytest.raises(PersonalModeUnavailableError):
            Settings()

    def test_error_explains_why(self) -> None:
        with pytest.raises(PersonalModeUnavailableError, match="demo-only"):
            Settings(app_mode="personal")

    def test_demo_mode_from_the_environment_is_fine(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("APP_MODE", "demo")
        assert Settings().app_mode is AppMode.DEMO


class TestValidation:
    def test_min_above_max_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="chunk_min_tokens"):
            Settings(chunk_min_tokens=900, chunk_max_tokens=800)

    def test_equal_bounds_are_allowed(self) -> None:
        assert Settings(chunk_min_tokens=800, chunk_max_tokens=800)

    @pytest.mark.parametrize(
        ("field_name", "value"),
        [
            ("chunk_min_tokens", 0),
            ("chunk_max_tokens", -1),
            ("max_file_size_bytes", 0),
            ("chunk_overlap_ratio", 1.0),
            ("chunk_overlap_ratio", -0.1),
        ],
    )
    def test_out_of_range_values_rejected(self, field_name: str, value: object) -> None:
        # Unlike bare type hints, Field(gt=...) constraints ARE enforced at
        # runtime — this is the pydantic equivalent of a [Range] attribute.
        with pytest.raises(ValueError):
            Settings(**{field_name: value})


class TestGetSettings:
    def test_is_cached(self) -> None:
        get_settings.cache_clear()
        assert get_settings() is get_settings()

    def test_cache_clear_gives_a_fresh_instance(self) -> None:
        first = get_settings()
        get_settings.cache_clear()
        assert get_settings() is not first
