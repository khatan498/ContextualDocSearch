"""Project-wide checks that no single module owns.

CLAUDE.md promises Python 3.11+, but development happens on 3.14, where newer
syntax — `def f[T](...)` generics, `type X = ...` aliases — parses fine and so
would never fail a test here. Parsing every file with the 3.11 grammar makes
the promise checkable. (Phase 5 introduced exactly such a line; this caught it
in review.)
"""

import ast
from pathlib import Path

import pytest

from app.config import PROJECT_ROOT

SOURCE_FILES = sorted(
    [p for folder in ("app", "scripts", "ui", "tests") for p in (PROJECT_ROOT / folder).rglob("*.py")]
    + [PROJECT_ROOT / "conftest.py"]
)


def test_the_check_can_see_newer_syntax() -> None:
    # Guards the guard: if ast stopped enforcing feature_version, every file
    # below would pass vacuously.
    with pytest.raises(SyntaxError):
        ast.parse("def f[T](x: T) -> T: ...", feature_version=(3, 11))


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda p: p.relative_to(PROJECT_ROOT).as_posix())
def test_source_parses_with_python_3_11_grammar(path: Path) -> None:
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 11))
