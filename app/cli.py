"""Small helpers shared by the command-line scripts in scripts/."""

import argparse


def positive_int(value: str) -> int:
    """Parse a command-line value that must be a whole number of at least 1.

    Passed to argparse as ``type=positive_int``. argparse calls it with the
    raw string and turns either exception into a normal usage error, exit
    code 2 — the same as a missing argument.

    Args:
        value: The text typed on the command line.

    Returns:
        The number.

    Raises:
        argparse.ArgumentTypeError: If the number is below 1.
        ValueError: If the text is not a whole number at all.
    """
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {number}")
    return number


def silence_model_loading_bars() -> None:
    """Hide the "Loading weights" progress bars that transformers draws.

    Each model load draws one to stderr, so every search printed two bars
    before any result. Download progress bars come from huggingface_hub
    instead and are left alone: on a first run they are the only sign that a
    large download is still progressing.
    """
    # Deferred import, as elsewhere: transformers is heavy, and a script that
    # fails on its arguments should not pay for importing it.
    from transformers.utils import logging as transformers_logging

    transformers_logging.disable_progress_bar()
