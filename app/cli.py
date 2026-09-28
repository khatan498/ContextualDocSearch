"""Small helpers shared by the command-line scripts in scripts/."""

import argparse
import sys
from collections.abc import Callable

from pydantic import ValidationError

from app.indexing.built_index import IndexUnavailableError
from app.indexing.embeddings import ChunkWindowTooLargeError
from app.modes import PersonalModeUnavailableError


def run_with_refusals(action: Callable[[], int]) -> int:
    """Run a script's work, turning deliberate refusals into messages.

    The refusals below each carry a message written for the person running
    the command. A traceback would bury it under thirty lines of library
    internals, so it is printed on its own. Anything else still raises with a
    full traceback: an unexpected error is a bug, and the traceback is what
    finding it needs.

    Exit codes:
        1: the index is missing, unreadable, or stale. Rebuilding fixes it.
        2: the configuration is refused — invalid settings, personal mode, or
            a chunk window the embedding model cannot read. The same code
            argparse uses for a bad argument: the command was wrong as given.

    Args:
        action: The script's work. Takes no arguments and returns an exit
            code; callers usually pass a lambda that closes over parsed args.

    Returns:
        The action's own exit code, or the refusal's code above.
    """
    # A tuple after `except` catches any of the listed types — the equivalent
    # of C#'s `catch (Exception e) when (e is A || e is B)`.
    try:
        return action()
    except IndexUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (PersonalModeUnavailableError, ChunkWindowTooLargeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValidationError as exc:
        # pydantic's own formatting already names each bad setting and why.
        print(f"error: invalid configuration\n{exc}", file=sys.stderr)
        return 2


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


def port_number(value: str) -> int:
    """Parse a TCP port for argparse: a whole number from 1 to 65535.

    Args:
        value: The text typed on the command line.

    Returns:
        The port.

    Raises:
        argparse.ArgumentTypeError: If the number is outside the port range.
        ValueError: If the text is not a whole number at all.
    """
    number = int(value)
    if not 1 <= number <= 65535:  # Python allows chained comparisons: 1 <= n <= 65535
        raise argparse.ArgumentTypeError(f"must be a port from 1 to 65535, got {number}")
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
