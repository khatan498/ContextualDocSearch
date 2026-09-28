"""Run the whole app locally — the search API and the UI — with one command.

    python scripts/run_local.py

Starts the search API (or reuses one already running), waits until it can
answer, then opens the search page in your browser. Ctrl+C stops everything
this command started.

Both run on 127.0.0.1 only, and Streamlit's usage statistics are switched
off: nothing is reachable from your network and nothing leaves the machine.
"""

import functools
import socket
import subprocess
import sys
import time
from pathlib import Path

# Same reason as in the other scripts: make `import app` work from anywhere.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.cli import run_with_refusals  # noqa: E402
from app.config import Settings, get_settings  # noqa: E402
from app.ui.client import ApiError, ApiUnavailableError, SearchClient  # noqa: E402

# Progress messages are flushed as they are printed. Python buffers stdout when
# it is redirected (to a file, a pipe, an IDE's run window), and this script's
# lines would otherwise appear only after everything had stopped.
# functools.partial pre-fills an argument — a function with `flush=True` baked in.
say = functools.partial(print, flush=True)

SERVE_SCRIPT = REPO_ROOT / "scripts" / "serve.py"
UI_SCRIPT = REPO_ROOT / "ui" / "streamlit_app.py"

# Loading both models takes ~9 s; a first run that downloads them takes longer.
STARTUP_TIMEOUT_SECONDS = 120.0
POLL_SECONDS = 0.5


def port_is_free(port: int) -> bool:
    """Whether this machine's loopback port is free for a server to use.

    Checked up front so a clash is reported at once, instead of after the API
    has spent ten seconds loading its models (found end to end: Streamlit
    then failed with just "Port 8501 is not available").

    Two tests, because binding alone is not enough on Windows: a program
    listening on 0.0.0.0 (all interfaces) does not stop another from binding
    127.0.0.1 on the same port — measured — so the two would silently share it.
    """
    # `with` closes each probe socket however the block exits — C#'s `using`.
    # 1. Does anything answer on it? Catches a listener on any address.
    with socket.socket() as probe:
        probe.settimeout(0.5)
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            return False
    # 2. Can it be bound? Catches a port that is taken but not yet listening.
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def api_is_up(base_url: str) -> bool:
    """Whether a search API is answering at ``base_url`` and ready to search."""
    try:
        return SearchClient(base_url, timeout=2.0).health().status == "ready"
    except (ApiUnavailableError, ApiError):
        return False


def wait_for_api(process: subprocess.Popen, base_url: str) -> int | None:
    """Wait until the API answers, or until it exits or takes too long.

    Args:
        process: The serve.py process just started.
        base_url: Where it will answer.

    Returns:
        ``None`` once the API is ready; otherwise the exit code to finish with.
        serve.py prints its own refusal message (no index, personal mode...)
        to this same terminal, so its exit code is simply passed on.
    """
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return process.returncode
        if api_is_up(base_url):
            return None
        time.sleep(POLL_SECONDS)

    print(
        f"error: the search API did not become ready within "
        f"{STARTUP_TIMEOUT_SECONDS:.0f} seconds.",
        file=sys.stderr,
    )
    return 1


def streamlit_command(settings: Settings) -> list[str]:
    """The Streamlit command line, with the privacy overrides applied.

    Each flag overrides a Streamlit default that would break the project's
    promises. Flags, unlike .streamlit/config.toml, apply whatever directory
    this is started from.
    """
    return [
        sys.executable, "-m", "streamlit", "run", str(UI_SCRIPT),
        # Default: every network interface. Loopback only.
        "--server.address", "127.0.0.1",
        "--server.port", str(settings.ui_port),
        # Default: send usage statistics to Streamlit.
        "--browser.gatherUsageStats", "false",
        # Default: stop the first launch to ask for an email address.
        "--server.showEmailPrompt", "false",
    ]  # fmt: skip


def stop(process: subprocess.Popen) -> None:
    """Stop a child process if it is still running, forcefully if it hangs."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def run() -> int:
    """Start what is needed, then wait until the UI exits or Ctrl+C.

    Returns:
        Process exit code.
    """
    # Reading the settings first refuses personal mode or invalid settings
    # before anything is started.
    settings = get_settings()
    base_url = settings.api_base_url

    if not port_is_free(settings.ui_port):
        print(
            f"error: port {settings.ui_port} is already in use, so the search page "
            f"cannot start. Is it already open in another terminal? Close that, or "
            f"set UI_PORT in .env to a free port.",
            file=sys.stderr,
        )
        return 1

    api: subprocess.Popen | None = None
    if api_is_up(base_url):
        say(f"using the search API already running at {base_url}")
    elif not port_is_free(settings.api_port):
        # Something is listening, but it is not a ready search API.
        print(
            f"error: port {settings.api_port} is in use by another program, so the "
            f"search API cannot start. Close it, or set API_PORT in .env to a free port.",
            file=sys.stderr,
        )
        return 1
    else:
        say("starting the search API (loading models, about 10 seconds)...")
        api = subprocess.Popen([sys.executable, str(SERVE_SCRIPT)])
        failed = wait_for_api(api, base_url)
        if failed is not None:
            stop(api)
            return failed

    say(f"opening the search page at http://127.0.0.1:{settings.ui_port}  (Ctrl+C to stop)")
    ui = subprocess.Popen(streamlit_command(settings))
    try:
        return ui.wait()
    except KeyboardInterrupt:
        # Ctrl+C reaches every process attached to this console, so the
        # children are already shutting down; `finally` makes sure of it.
        return 0
    finally:
        stop(ui)
        # Only stop an API this command started — never one it merely found.
        if api is not None:
            stop(api)


def main() -> int:
    """Entry point."""
    return run_with_refusals(run)


if __name__ == "__main__":
    raise SystemExit(main())
