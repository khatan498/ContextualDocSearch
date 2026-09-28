"""Tests for scripts/run_local.py.

``subprocess.Popen`` is replaced with a fake that records each command and
plays back a scripted life story, and the health probe is scripted too, so
nothing is started and no port is touched.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_run_local():
    """Import scripts/run_local.py as a module."""
    path = REPO_ROOT / "scripts" / "run_local.py"
    spec = importlib.util.spec_from_file_location("run_local_script", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_local_script"] = module
    spec.loader.exec_module(module)
    return module


run_local = _load_run_local()


class FakeProcess:
    """A child process whose behaviour each test scripts in advance."""

    def __init__(self, command: list[str], *, exits_with: int | None = None,
                 wait_result: int | BaseException = 0) -> None:
        self.command = command
        self.returncode = exits_with
        self.wait_result = wait_result
        self.terminated = False

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if isinstance(self.wait_result, BaseException):
            # Raised once, like Ctrl+C interrupting the launcher's wait. A
            # second raise would hit the cleanup's wait too — and pytest reads
            # a KeyboardInterrupt as the user aborting the whole run.
            error, self.wait_result = self.wait_result, 0
            raise error
        self.returncode = self.returncode if self.returncode is not None else self.wait_result
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


class Harness:
    """Replaces Popen, the health probe and sleep; records what happened."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.processes: list[FakeProcess] = []
        self.health_answers: list[bool] = []
        self.api_exits_with: int | None = None
        self.ui_wait: int | BaseException = 0
        monkeypatch.setattr(run_local.subprocess, "Popen", self._popen)
        monkeypatch.setattr(run_local, "api_is_up", self._api_is_up)
        monkeypatch.setattr(run_local.time, "sleep", lambda seconds: None)
        self.busy_ports: set[int] = set()
        monkeypatch.setattr(run_local, "port_is_free", lambda port: port not in self.busy_ports)

    def _popen(self, command: list[str]) -> FakeProcess:
        if str(run_local.SERVE_SCRIPT) in command:
            process = FakeProcess(command, exits_with=self.api_exits_with)
        else:
            process = FakeProcess(command, wait_result=self.ui_wait)
        self.processes.append(process)
        return process

    def _api_is_up(self, base_url: str) -> bool:
        return self.health_answers.pop(0) if self.health_answers else False

    @property
    def api(self) -> FakeProcess | None:
        return next((p for p in self.processes if str(run_local.SERVE_SCRIPT) in p.command), None)

    @property
    def ui(self) -> FakeProcess | None:
        return next((p for p in self.processes if "streamlit" in p.command), None)


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch, fresh_settings: None) -> Harness:
    return Harness(monkeypatch)


class TestReusingARunningApi:
    def test_starts_only_the_ui(self, harness: Harness) -> None:
        harness.health_answers = [True]

        assert run_local.main() == 0

        assert harness.api is None
        assert harness.ui is not None

    def test_never_stops_an_api_it_did_not_start(self, harness: Harness) -> None:
        harness.health_answers = [True]

        run_local.main()

        # Only the UI existed, and only the UI was stopped.
        assert [p for p in harness.processes if p.terminated] in ([], [harness.ui])


class TestStartingTheApi:
    def test_waits_for_the_api_then_starts_the_ui(self, harness: Harness) -> None:
        # Not up at first; then two polls before it answers.
        harness.health_answers = [False, False, False, True]

        assert run_local.main() == 0

        assert harness.api is not None
        assert harness.processes.index(harness.api) < harness.processes.index(harness.ui)

    def test_stops_the_api_it_started_on_exit(self, harness: Harness) -> None:
        harness.health_answers = [False, True]

        run_local.main()

        assert harness.api.terminated

    def test_stops_both_on_ctrl_c(self, harness: Harness) -> None:
        harness.health_answers = [False, True]
        harness.ui_wait = KeyboardInterrupt()

        assert run_local.main() == 0

        assert harness.api.terminated
        assert harness.ui.terminated

    def test_api_refusal_passes_its_exit_code_through(self, harness: Harness) -> None:
        # serve.py exits 1 (e.g. no index) and prints its own message.
        harness.health_answers = [False]
        harness.api_exits_with = 1

        assert run_local.main() == 1

        assert harness.ui is None

    def test_startup_timeout_gives_up_cleanly(
        self, harness: Harness, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        monkeypatch.setattr(run_local, "STARTUP_TIMEOUT_SECONDS", 0.0)

        assert run_local.main() == 1

        assert "did not become ready" in capsys.readouterr().err
        assert harness.api.terminated
        assert harness.ui is None


class TestPortClashes:
    """Reported at once — before ten seconds of model loading, not after."""

    def test_page_port_taken_starts_nothing(
        self, harness: Harness, capsys: pytest.CaptureFixture
    ) -> None:
        harness.busy_ports = {8501}

        assert run_local.main() == 1

        assert harness.processes == []
        assert "port 8501 is already in use" in capsys.readouterr().err

    def test_api_port_held_by_another_program_starts_nothing(
        self, harness: Harness, capsys: pytest.CaptureFixture
    ) -> None:
        harness.health_answers = [False]  # listening, but not a search API
        harness.busy_ports = {8000}

        assert run_local.main() == 1

        assert harness.processes == []
        assert "port 8000 is in use by another program" in capsys.readouterr().err

    def test_a_running_search_api_on_its_port_is_reused_not_refused(
        self, harness: Harness
    ) -> None:
        # Its port is busy precisely because it is running: that is fine.
        harness.health_answers = [True]
        harness.busy_ports = {8000}

        assert run_local.main() == 0
        assert harness.api is None and harness.ui is not None


class TestPortIsFree:
    """The real probe, against real sockets."""

    def test_a_listening_port_is_not_free(self) -> None:
        import socket

        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))  # port 0: the OS picks a free one
            listener.listen()
            port = listener.getsockname()[1]

            assert run_local.port_is_free(port) is False

    def test_a_port_held_on_all_interfaces_is_not_free(self) -> None:
        # Found end to end: Windows lets 127.0.0.1 be bound while another
        # program listens on 0.0.0.0 with the same port, so a bind-only check
        # reported it free and the two programs silently shared the port.
        import socket

        with socket.socket() as listener:
            listener.bind(("0.0.0.0", 0))
            listener.listen()
            port = listener.getsockname()[1]

            assert run_local.port_is_free(port) is False

    def test_a_bound_but_not_listening_port_is_not_free(self) -> None:
        import socket

        with socket.socket() as holder:
            holder.bind(("127.0.0.1", 0))
            port = holder.getsockname()[1]

            assert run_local.port_is_free(port) is False

    def test_an_unused_port_is_free(self) -> None:
        import socket

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        # Released on leaving the block.
        assert run_local.port_is_free(port) is True


class TestStreamlitCommand:
    @pytest.fixture
    def command(self, harness: Harness) -> list[str]:
        harness.health_answers = [True]
        run_local.main()
        return harness.ui.command

    def flag(self, command: list[str], name: str) -> str:
        return command[command.index(name) + 1]

    def test_runs_the_page_script(self, command: list[str]) -> None:
        assert command[1:4] == ["-m", "streamlit", "run"]
        assert command[4] == str(run_local.UI_SCRIPT)

    def test_binds_loopback_only(self, command: list[str]) -> None:
        assert self.flag(command, "--server.address") == "127.0.0.1"

    def test_usage_statistics_are_off(self, command: list[str]) -> None:
        assert self.flag(command, "--browser.gatherUsageStats") == "false"

    def test_email_prompt_is_off(self, command: list[str]) -> None:
        assert self.flag(command, "--server.showEmailPrompt") == "false"

    def test_uses_the_configured_port(
        self, harness: Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("UI_PORT", "8600")
        harness.health_answers = [True]

        run_local.main()

        assert self.flag(harness.ui.command, "--server.port") == "8600"


class TestRefusals:
    def test_personal_mode_starts_nothing(
        self, harness: Harness, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        from app.modes import PERSONAL_MODE_MESSAGE

        monkeypatch.setenv("APP_MODE", "personal")

        assert run_local.main() == 2

        assert harness.processes == []
        assert PERSONAL_MODE_MESSAGE in capsys.readouterr().err


class TestApiIsUp:
    """The real probe, against a fake transport: only a ready API counts."""

    def test_ready_api(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class Ready:
            def __init__(self, *args, **kwargs) -> None: ...

            def health(self):
                from app.api.schemas import HealthResponse

                return HealthResponse(status="ready", chunks=1, embedding_model="e", reranker_model="r")

        monkeypatch.setattr(run_local, "SearchClient", Ready)

        assert run_local.api_is_up("http://x") is True

    @pytest.mark.parametrize("error", ["ApiUnavailableError", "ApiError"])
    def test_unreachable_or_wrong_service(self, monkeypatch: pytest.MonkeyPatch, error: str) -> None:
        exception = getattr(run_local, error)

        class Failing:
            def __init__(self, *args, **kwargs) -> None: ...

            def health(self):
                raise exception("no")

        monkeypatch.setattr(run_local, "SearchClient", Failing)

        assert run_local.api_is_up("http://x") is False


class TestStop:
    def test_already_exited_process_is_left_alone(self) -> None:
        process = FakeProcess(["x"], exits_with=0)

        run_local.stop(process)

        assert not process.terminated

    def test_hanging_process_is_killed(self) -> None:
        class Hangs(FakeProcess):
            def terminate(self) -> None:
                self.terminated = True  # but keeps running

            def wait(self, timeout: float | None = None) -> int:
                raise subprocess.TimeoutExpired("x", timeout)

        process = Hangs(["x"])

        run_local.stop(process)

        assert process.returncode == -9


def test_progress_messages_are_flushed() -> None:
    # Redirected stdout is block-buffered: found end to end, the launcher's
    # "starting the search API..." appeared only after everything had stopped.
    assert run_local.say.func is print
    assert run_local.say.keywords == {"flush": True}
