"""Application modes, and the single source of truth for what each one means.

The app ships demo-only: it searches a fixed set of sample documents committed
to the repository. Personal mode is recognised but deliberately unimplemented,
so that selecting it produces one clear, consistent explanation everywhere it
can be selected — startup, API, and eventually the UI.
"""

from enum import StrEnum


class AppMode(StrEnum):
    """Which document set the app operates over.

    ``StrEnum`` members *are* strings: ``AppMode.DEMO == "demo"`` is ``True``
    and ``f"{AppMode.DEMO}"`` renders as ``demo``. That is what lets a plain
    ``APP_MODE=demo`` line in ``.env`` convert cleanly without a custom parser.

    C#/Java note: closest to an enum plus an implicit string conversion, except
    comparison against a bare string just works rather than needing ``.ToString()``.
    """

    DEMO = "demo"
    PERSONAL = "personal"


#: Shown wherever a user tries to use personal mode. Kept as a module constant
#: so the startup guard, the API, and the future UI all say the same thing —
#: if this text needs to change, it changes in exactly one place.
PERSONAL_MODE_MESSAGE: str = (
    "This build is demo-only and searches a fixed set of sample documents "
    "bundled with the app. A future release will add personal mode, letting "
    "you connect your own cloud drives (Google Drive, OneDrive) and search "
    "your own files."
)


class PersonalModeUnavailableError(RuntimeError):
    """Raised when personal mode is requested from a demo-only build.

    Defaults to :data:`PERSONAL_MODE_MESSAGE` so every raise site is identical
    without repeating the text.
    """

    def __init__(self, message: str = PERSONAL_MODE_MESSAGE) -> None:
        super().__init__(message)
