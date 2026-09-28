"""Tests for app.config."""

from pathlib import Path

import pytest

from app.config import PROJECT_ROOT, Settings, get_settings
from app.modes import AppMode, PersonalModeUnavailableError


class TestDefaults:
    def test_defaults_to_demo_mode(self) -> None:
        assert Settings().app_mode is AppMode.DEMO

    def test_sample_docs_path(self) -> None:
        assert Settings().sample_docs_path == PROJECT_ROOT / "data" / "sample_docs"

    def test_chunk_defaults(self) -> None:
        settings = Settings()
        assert settings.chunk_min_tokens == 350
        assert settings.chunk_max_tokens == 480
        assert settings.chunk_overlap_ratio == pytest.approx(0.15)

    def test_embedding_defaults(self) -> None:
        settings = Settings()
        assert settings.embedding_model_name == "BAAI/bge-base-en-v1.5"
        assert settings.embedding_batch_size == 32

    def test_retrieval_defaults(self) -> None:
        settings = Settings()
        assert settings.reranker_model_name == "cross-encoder/ms-marco-MiniLM-L6-v2"
        assert settings.retrieval_candidates == 20
        assert settings.rrf_k == 60
        assert settings.search_top_k == 5

    def test_api_defaults_to_loopback(self) -> None:
        # Reachable only from this machine unless deliberately widened.
        settings = Settings()
        assert settings.api_host == "127.0.0.1"
        assert settings.api_port == 8000

    def test_ui_port_default(self) -> None:
        assert Settings().ui_port == 8501

    def test_keyword_index_sits_beside_the_vector_store(self, tmp_path: Path) -> None:
        settings = Settings(vector_store_path=tmp_path)
        assert settings.keyword_index_path == tmp_path / "bm25_index.json"

    def test_chunk_window_fits_the_default_model(self) -> None:
        # bge-base-en-v1.5 accepts 512 tokens and truncates silently past it.
        # This asserts the shipped defaults are a safe pairing; the same check
        # runs against the real model in EmbeddingModel at index time.
        assert Settings().chunk_max_tokens <= 512


class TestPathAnchoring:
    """Relative paths mean "relative to the repo", never "to the CWD".

    Before this, running a script from scripts/ looked for data/index under
    scripts/ and reported that no index existed — and a web server launched
    from anywhere else would have done the same.
    """

    def test_project_root_is_the_repository(self) -> None:
        assert (PROJECT_ROOT / "app" / "config.py").is_file()

    def test_default_paths_are_absolute(self) -> None:
        settings = Settings()
        assert settings.sample_docs_path.is_absolute()
        assert settings.vector_store_path.is_absolute()

    def test_relative_override_is_anchored_to_the_root(self) -> None:
        settings = Settings(vector_store_path=Path("data/elsewhere"))
        assert settings.vector_store_path == PROJECT_ROOT / "data" / "elsewhere"

    def test_absolute_override_is_left_alone(self, tmp_path: Path) -> None:
        assert Settings(vector_store_path=tmp_path).vector_store_path == tmp_path

    def test_environment_override_is_anchored_too(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SAMPLE_DOCS_PATH", "data/other_docs")
        assert Settings().sample_docs_path == PROJECT_ROOT / "data" / "other_docs"

    def test_independent_of_the_working_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The actual failure mode: a different CWD must not move the paths.
        from_root = Settings()
        monkeypatch.chdir(tmp_path)
        from_elsewhere = Settings()

        assert from_elsewhere.sample_docs_path == from_root.sample_docs_path
        assert from_elsewhere.vector_store_path == from_root.vector_store_path

    def test_env_file_is_read_from_the_root(self) -> None:
        assert Settings.model_config["env_file"] == PROJECT_ROOT / ".env"


class TestScope:
    """The app is demo-only; no cloud credential settings should exist."""

    @pytest.mark.parametrize(
        "field_name",
        ["google_drive_credentials_path", "onedrive_credentials_path"],
    )
    def test_no_cloud_credential_fields(self, field_name: str) -> None:
        assert field_name not in Settings.model_fields


class TestPersonalMode:
    def test_rejected_when_passed_directly(self) -> None:
        with pytest.raises(PersonalModeUnavailableError):
            Settings(app_mode=AppMode.PERSONAL)

    def test_rejected_from_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # monkeypatch undoes the change after the test, so it cannot leak into
        # any other test in the session.
        monkeypatch.setenv("APP_MODE", "personal")
        with pytest.raises(PersonalModeUnavailableError):
            Settings()

    def test_error_explains_why(self) -> None:
        with pytest.raises(PersonalModeUnavailableError, match="demo-only"):
            Settings(app_mode="personal")

    def test_demo_mode_from_the_environment_is_fine(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("APP_MODE", "demo")
        assert Settings().app_mode is AppMode.DEMO


class TestValidation:
    def test_min_above_max_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="chunk_min_tokens"):
            Settings(chunk_min_tokens=900, chunk_max_tokens=800)

    def test_equal_bounds_are_allowed(self) -> None:
        assert Settings(chunk_min_tokens=800, chunk_max_tokens=800)

    def test_more_results_than_candidates_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="search_top_k"):
            Settings(search_top_k=21, retrieval_candidates=20)

    def test_results_equal_to_candidates_are_allowed(self) -> None:
        assert Settings(search_top_k=20, retrieval_candidates=20)

    @pytest.mark.parametrize(
        ("field_name", "value"),
        [
            ("chunk_min_tokens", 0),
            ("chunk_max_tokens", -1),
            ("max_file_size_bytes", 0),
            ("chunk_overlap_ratio", 1.0),
            ("chunk_overlap_ratio", -0.1),
            ("retrieval_candidates", 0),
            ("rrf_k", 0),
            ("search_top_k", 0),
            ("api_port", 0),
            ("api_port", 65536),
            ("ui_port", 0),
        ],
    )
    def test_out_of_range_values_rejected(self, field_name: str, value: object) -> None:
        # Unlike bare type hints, Field(gt=...) constraints ARE enforced at
        # runtime — this is the pydantic equivalent of a [Range] attribute.
        with pytest.raises(ValueError):
            Settings(**{field_name: value})


class TestGetSettings:
    def test_is_cached(self) -> None:
        get_settings.cache_clear()
        assert get_settings() is get_settings()

    def test_cache_clear_gives_a_fresh_instance(self) -> None:
        first = get_settings()
        get_settings.cache_clear()
        assert get_settings() is not first


class TestApiBaseUrl:
    def test_default_is_loopback(self) -> None:
        assert Settings().api_base_url == "http://127.0.0.1:8000"

    @pytest.mark.parametrize("wildcard", ["0.0.0.0", "::"])
    def test_listen_everywhere_addresses_map_to_loopback(self, wildcard: str) -> None:
        # A server binds 0.0.0.0; a client cannot connect to it.
        settings = Settings(api_host=wildcard, api_port=9000)

        assert settings.api_base_url == "http://127.0.0.1:9000"

    def test_an_explicit_host_is_kept(self) -> None:
        assert Settings(api_host="192.168.1.20").api_base_url == "http://192.168.1.20:8000"


class TestStreamlitConfig:
    """.streamlit/config.toml is the backstop for a hand-run `streamlit run`."""

    def test_privacy_defaults_are_overridden(self) -> None:
        import tomllib

        config = tomllib.loads(
            (PROJECT_ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
        )

        assert config["browser"]["gatherUsageStats"] is False
        assert config["server"]["address"] == "127.0.0.1"
        assert config["server"]["showEmailPrompt"] is False
        assert config["client"]["toolbarMode"] == "minimal"
