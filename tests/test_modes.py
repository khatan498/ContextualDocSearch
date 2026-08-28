"""Tests for app.modes."""

import pytest

from app.modes import PERSONAL_MODE_MESSAGE, AppMode, PersonalModeUnavailableError


class TestAppMode:
    def test_members_are_plain_strings(self) -> None:
        # StrEnum members compare equal to their value, which is what lets a
        # bare `APP_MODE=demo` line in .env convert without a custom parser.
        assert AppMode.DEMO == "demo"
        assert AppMode.PERSONAL == "personal"

    def test_constructible_from_string(self) -> None:
        assert AppMode("demo") is AppMode.DEMO
        assert AppMode("personal") is AppMode.PERSONAL

    def test_unknown_mode_rejected(self) -> None:
        with pytest.raises(ValueError):
            AppMode("enterprise")


class TestPersonalModeMessage:
    def test_says_the_build_is_demo_only(self) -> None:
        assert "demo-only" in PERSONAL_MODE_MESSAGE

    def test_promises_a_future_release(self) -> None:
        assert "future release" in PERSONAL_MODE_MESSAGE

    @pytest.mark.parametrize("provider", ["Google Drive", "OneDrive"])
    def test_names_the_planned_providers(self, provider: str) -> None:
        # The message is the only place these are promised; if a provider is
        # dropped from the roadmap this test is the reminder to reword it.
        assert provider in PERSONAL_MODE_MESSAGE


class TestPersonalModeUnavailableError:
    def test_defaults_to_the_shared_message(self) -> None:
        assert str(PersonalModeUnavailableError()) == PERSONAL_MODE_MESSAGE

    def test_accepts_an_override(self) -> None:
        assert str(PersonalModeUnavailableError("nope")) == "nope"

    def test_is_a_runtime_error(self) -> None:
        # Deliberately not a ValueError: pydantic folds ValueError raised in a
        # validator into a ValidationError, which would bury the explanation.
        assert issubclass(PersonalModeUnavailableError, RuntimeError)
        assert not issubclass(PersonalModeUnavailableError, ValueError)
