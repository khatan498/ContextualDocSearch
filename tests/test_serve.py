"""Tests for scripts/serve.py.

Loaded by path, like the other script tests. Both the retriever and
``uvicorn.run`` are replaced, so nothing loads a model or binds a port.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_serve():
    """Import scripts/serve.py as a module."""
    path = REPO_ROOT / "scripts" / "serve.py"
    spec = importlib.util.spec_from_file_location("serve_script", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["serve_script"] = module
    spec.loader.exec_module(module)
    return module


serve_script = _load_serve()


class StubRetriever:
    chunk_count = 69

    def search(self, query: str, top_k: int | None = None) -> list:
        return []


@pytest.fixture(autouse=True)
def no_transformers_import(monkeypatch: pytest.MonkeyPatch) -> None:
    # main() hides the transformers progress bars; importing transformers to do
    # so costs over two seconds. The helper itself is tested in test_cli.py.
    monkeypatch.setattr(serve_script, "silence_model_loading_bars", lambda: None)


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    """Record, in order, the retriever opening and the server starting."""
    log: list[tuple] = []

    def fake_open(cls) -> StubRetriever:
        log.append(("open",))
        return StubRetriever()

    def fake_run(app, host: str, port: int) -> None:
        log.append(("run", app, host, port))

    monkeypatch.setattr(serve_script.HybridRetriever, "open", classmethod(fake_open))
    monkeypatch.setattr(serve_script.uvicorn, "run", fake_run)
    return log


class TestStartup:
    def test_opens_the_retriever_before_serving(
        self, events: list[tuple], fresh_settings: None
    ) -> None:
        assert serve_script.main([]) == 0

        assert [event[0] for event in events] == ["open", "run"]

    def test_serves_an_app_holding_the_opened_retriever(
        self, events: list[tuple], fresh_settings: None
    ) -> None:
        serve_script.main([])

        app = events[1][1]
        assert isinstance(app.state.retriever, StubRetriever)

    def test_defaults_come_from_settings(
        self, events: list[tuple], fresh_settings: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("API_PORT", "9123")

        serve_script.main([])

        _, _, host, port = events[1]
        assert (host, port) == ("127.0.0.1", 9123)

    def test_arguments_override_settings(
        self, events: list[tuple], fresh_settings: None
    ) -> None:
        serve_script.main(["--host", "0.0.0.0", "--port", "8080"])

        _, _, host, port = events[1]
        assert (host, port) == ("0.0.0.0", 8080)

    @pytest.mark.parametrize("port", ["0", "70000", "http"])
    def test_invalid_port_exits_2(self, events: list[tuple], port: str) -> None:
        with pytest.raises(SystemExit) as exited:
            serve_script.main(["--port", port])

        assert exited.value.code == 2
        assert events == []


class TestRefusals:
    """Every refusal is reported before the port is bound."""

    def test_missing_index_exits_1_without_serving(
        self,
        tmp_path: Path,
        fresh_settings: None,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        served: list[bool] = []
        monkeypatch.setattr(serve_script.uvicorn, "run", lambda *a, **k: served.append(True))
        monkeypatch.setenv("VECTOR_STORE_PATH", str(tmp_path / "absent"))

        assert serve_script.main([]) == 1
        err = capsys.readouterr().err
        assert "build_index.py" in err
        assert "Traceback" not in err
        assert served == []

    def test_personal_mode_exits_2_without_serving(
        self,
        fresh_settings: None,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        from app.modes import PERSONAL_MODE_MESSAGE

        served: list[bool] = []
        monkeypatch.setattr(serve_script.uvicorn, "run", lambda *a, **k: served.append(True))
        monkeypatch.setenv("APP_MODE", "personal")

        assert serve_script.main([]) == 2
        assert PERSONAL_MODE_MESSAGE in capsys.readouterr().err
        assert served == []

    def test_invalid_configuration_exits_2_without_serving(
        self,
        fresh_settings: None,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        served: list[bool] = []
        monkeypatch.setattr(serve_script.uvicorn, "run", lambda *a, **k: served.append(True))
        monkeypatch.setenv("API_PORT", "70000")

        assert serve_script.main([]) == 2
        assert "api_port" in capsys.readouterr().err
        assert served == []

    def test_chunk_window_too_large_exits_2_without_serving(
        self,
        fresh_settings: None,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        from app.indexing.embeddings import ChunkWindowTooLargeError

        def refuse(cls):
            raise ChunkWindowTooLargeError("CHUNK_MAX_TOKENS is 600 but the model reads 512")

        served: list[bool] = []
        monkeypatch.setattr(serve_script.HybridRetriever, "open", classmethod(refuse))
        monkeypatch.setattr(serve_script.uvicorn, "run", lambda *a, **k: served.append(True))

        assert serve_script.main([]) == 2
        assert "reads 512" in capsys.readouterr().err
        assert served == []

    def test_unexpected_errors_still_raise(
        self, fresh_settings: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken(cls):
            raise RuntimeError("a real bug")

        monkeypatch.setattr(serve_script.HybridRetriever, "open", classmethod(broken))

        with pytest.raises(RuntimeError, match="a real bug"):
            serve_script.main([])
