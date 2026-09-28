"""Tests for app.cli."""

import argparse
import sys
import types

import pytest

from app.cli import port_number, positive_int, run_with_refusals, silence_model_loading_bars
from app.indexing.built_index import IndexUnavailableError
from app.indexing.embeddings import ChunkWindowTooLargeError
from app.modes import PERSONAL_MODE_MESSAGE, PersonalModeUnavailableError


class TestPositiveInt:
    @pytest.mark.parametrize(("text", "expected"), [("1", 1), ("5", 5), ("250", 250)])
    def test_accepts_whole_numbers_from_one(self, text: str, expected: int) -> None:
        assert positive_int(text) == expected

    @pytest.mark.parametrize("text", ["0", "-1", "-20"])
    def test_rejects_zero_and_negatives(self, text: str) -> None:
        with pytest.raises(argparse.ArgumentTypeError, match="at least 1"):
            positive_int(text)

    @pytest.mark.parametrize("text", ["three", "2.5", ""])
    def test_rejects_non_integers(self, text: str) -> None:
        # argparse reports a ValueError from a type function as
        # "invalid positive_int value", so this needs no custom message.
        with pytest.raises(ValueError):
            positive_int(text)

    def test_argparse_turns_a_rejection_into_exit_code_2(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("-k", type=positive_int)

        with pytest.raises(SystemExit) as exited:
            parser.parse_args(["-k", "0"])

        assert exited.value.code == 2


class TestSilenceModelLoadingBars:
    def test_disables_the_transformers_progress_bar(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A stub package, because importing the real transformers costs over
        # two seconds.
        calls: list[str] = []
        logging_module = types.ModuleType("transformers.utils.logging")
        logging_module.disable_progress_bar = lambda: calls.append("disabled")  # type: ignore[attr-defined]
        utils = types.ModuleType("transformers.utils")
        utils.logging = logging_module  # type: ignore[attr-defined]
        package = types.ModuleType("transformers")
        package.utils = utils  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "transformers", package)
        monkeypatch.setitem(sys.modules, "transformers.utils", utils)
        monkeypatch.setitem(sys.modules, "transformers.utils.logging", logging_module)

        silence_model_loading_bars()

        assert calls == ["disabled"]


def make_validation_error() -> Exception:
    """A real pydantic ValidationError, from an impossible chunk window."""
    from app.config import Settings

    try:
        Settings(chunk_min_tokens=900, chunk_max_tokens=100)
    except Exception as exc:  # noqa: BLE001 - captured to be re-raised below
        return exc
    raise AssertionError("Settings accepted an impossible chunk window")


class TestRunWithRefusals:
    """The one mapping from deliberate refusals to messages and exit codes,
    shared by every script."""

    def test_returns_the_actions_own_exit_code(self) -> None:
        assert run_with_refusals(lambda: 0) == 0
        assert run_with_refusals(lambda: 1) == 1

    @pytest.mark.parametrize(
        ("error", "code"),
        [
            (IndexUnavailableError("No search index. Run build_index.py first."), 1),
            (PersonalModeUnavailableError(PERSONAL_MODE_MESSAGE), 2),
            (ChunkWindowTooLargeError("CHUNK_MAX_TOKENS is 600"), 2),
        ],
    )
    def test_refusals_print_their_message_and_exit(
        self, error: Exception, code: int, capsys: pytest.CaptureFixture
    ) -> None:
        def refuse() -> int:
            raise error

        assert run_with_refusals(refuse) == code
        err = capsys.readouterr().err
        assert f"error: {error}" in err
        assert "Traceback" not in err

    def test_invalid_configuration_names_the_setting(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        error = make_validation_error()

        def refuse() -> int:
            raise error

        assert run_with_refusals(refuse) == 2
        err = capsys.readouterr().err
        assert "invalid configuration" in err
        assert "chunk_min_tokens" in err

    def test_unexpected_errors_keep_their_traceback(self) -> None:
        def broken() -> int:
            raise RuntimeError("a real bug")

        with pytest.raises(RuntimeError, match="a real bug"):
            run_with_refusals(broken)


class TestPortNumber:
    @pytest.mark.parametrize("text", ["1", "8000", "65535"])
    def test_accepts_the_port_range(self, text: str) -> None:
        assert port_number(text) == int(text)

    @pytest.mark.parametrize("text", ["0", "-1", "65536"])
    def test_rejects_outside_the_range(self, text: str) -> None:
        with pytest.raises(argparse.ArgumentTypeError, match="1 to 65535"):
            port_number(text)

    def test_rejects_non_numbers(self) -> None:
        with pytest.raises(ValueError):
            port_number("http")
