"""Tests for app.cli."""

import argparse
import sys
import types

import pytest

from app.cli import positive_int, silence_model_loading_bars


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
