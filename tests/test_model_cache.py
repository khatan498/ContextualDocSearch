"""Tests for app.model_cache."""

import pytest

from app.model_cache import load_cache_first


class FakeModel:
    """Records each construction attempt; "cached" decides whether a
    local-only load succeeds, the way the real libraries behave."""

    cached = True
    attempts: list[dict] = []

    def __init__(self, name: str, local_files_only: bool = False, **kwargs: object) -> None:
        type(self).attempts.append({"local_files_only": local_files_only, **kwargs})
        if local_files_only and not type(self).cached:
            raise OSError(f"{name} is not in the local cache")
        self.name = name
        self.kwargs = kwargs


@pytest.fixture(autouse=True)
def fresh_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    # `autouse=True` applies the fixture to every test in this file without
    # naming it as a parameter.
    monkeypatch.setattr(FakeModel, "attempts", [])
    monkeypatch.setattr(FakeModel, "cached", True)


def test_cached_model_loads_in_one_local_only_attempt() -> None:
    model = load_cache_first(FakeModel, "fake/model")

    assert model.name == "fake/model"
    assert FakeModel.attempts == [{"local_files_only": True}]


def test_missing_model_falls_back_to_a_download(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(FakeModel, "cached", False)

    load_cache_first(FakeModel, "fake/model")

    assert [a["local_files_only"] for a in FakeModel.attempts] == [True, False]


def test_extra_arguments_reach_both_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(FakeModel, "cached", False)

    model = load_cache_first(FakeModel, "fake/model", max_length=512)

    assert all(a["max_length"] == 512 for a in FakeModel.attempts)
    assert model.kwargs == {"max_length": 512}


def test_fallback_is_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A first-run download is slow and large; saying so is the difference
    # between "it's downloading" and "it's hung".
    monkeypatch.setattr(FakeModel, "cached", False)

    with caplog.at_level("INFO", logger="app.model_cache"):
        load_cache_first(FakeModel, "fake/model")

    assert "downloading" in caplog.text
    assert "fake/model" in caplog.text


def test_other_errors_are_not_swallowed() -> None:
    def broken(name: str, **kwargs: object) -> None:
        raise ValueError("bad config")

    with pytest.raises(ValueError, match="bad config"):
        load_cache_first(broken, "fake/model")
