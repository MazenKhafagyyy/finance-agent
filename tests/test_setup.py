"""Tests for installing and starting: the settings check, `fin-agent check`, and the command line.
No network: settings are patched in, and the checks that would call out are replaced."""
import pytest

from fin_agent import checks, cli, config


@pytest.fixture
def good_settings(monkeypatch, tmp_path):
    """A complete, valid set of settings (the key file exists but is never opened)."""
    key = tmp_path / "key.json"
    key.write_text("{}")
    monkeypatch.delenv("MODE", raising=False)
    monkeypatch.delenv("DATA_DIR", raising=False)
    for name, value in {"GOOGLE_SHEET_ID": "abc", "GOOGLE_SERVICE_ACCOUNT_FILE": str(key),
                        "TIMEZONE": "Africa/Cairo", "DEFAULT_CURRENCY": "EGP",
                        "DISCORD_BOT_TOKEN": "token", "DISCORD_OWNER_ID": 123, "_owner": "123"}.items():
        monkeypatch.setattr(config, name, value)


def test_good_settings_have_no_problems(good_settings):
    assert config.settings_problems() == []
    assert config.discord_problems() == []


@pytest.mark.parametrize("name, value, expected", [
    ("GOOGLE_SHEET_ID", "", "GOOGLE_SHEET_ID is empty"),
    ("GOOGLE_SERVICE_ACCOUNT_FILE", "", "GOOGLE_SERVICE_ACCOUNT_FILE is empty"),
    ("GOOGLE_SERVICE_ACCOUNT_FILE", "/no/such/key.json", "no file at /no/such/key.json"),
    ("TIMEZONE", "Cairo", "isn't a time zone name"),
    ("DEFAULT_CURRENCY", "USD", "DEFAULT_CURRENCY must be one of EGP, CAD"),
])
def test_each_bad_setting_is_named_with_its_fix(good_settings, monkeypatch, name, value, expected):
    monkeypatch.setattr(config, name, value)
    [problem] = config.settings_problems()
    assert expected in problem


def test_an_old_mode_setting_is_not_silently_ignored(good_settings, monkeypatch):
    # Ignoring MODE=real would start with an EMPTY history and memory.
    monkeypatch.setenv("MODE", "real")
    [problem] = config.settings_problems()
    assert "DATA_DIR=data/real" in problem
    monkeypatch.setenv("DATA_DIR", "data/real")     # once DATA_DIR is set, MODE is just a leftover
    assert config.settings_problems() == []


def test_owner_id_must_be_a_number(good_settings, monkeypatch):
    monkeypatch.setattr(config, "_owner", "mazen")
    monkeypatch.setattr(config, "DISCORD_OWNER_ID", 0)
    [problem] = config.discord_problems()
    assert "must be a number" in problem


def test_cloud_models_count_as_pulled_and_latest_is_optional():
    pulled = {"qwen3-embedding:0.6b", "lfm2.5:latest"}
    assert checks.is_pulled("gemma4:31b-cloud", pulled)     # cloud models are never in the local list
    assert checks.is_pulled("lfm2.5", pulled)
    assert checks.is_pulled("qwen3-embedding:0.6b", pulled)
    assert not checks.is_pulled("qwen3.5:4b", pulled)


def test_cloud_model_does_not_trigger_the_not_installed_warning(monkeypatch, capsys):
    embed_model = config.EMBED_MODEL

    class Tags:
        def json(self):
            # the cloud model is NOT in the local list; the embedding model is
            return {"models": [{"name": "qwen3.5:4b"}, {"name": embed_model}]}

    monkeypatch.setattr(checks.httpx, "get", lambda *a, **k: Tags())
    monkeypatch.setattr(config, "MODEL", "gemma4:31b-cloud")
    checks.check_ollama()
    assert capsys.readouterr().out == ""                    # no false warning
    monkeypatch.setattr(config, "MODEL", "missing-model:1b")
    checks.check_ollama()
    assert "'missing-model:1b' isn't in `ollama list`" in capsys.readouterr().out   # really missing still warns
    monkeypatch.setattr(config, "EMBED_MODEL", "missing-embed:1b")
    checks.check_ollama()
    assert "ollama pull missing-embed:1b" in capsys.readouterr().out   # so does the memory model


def test_check_reports_every_line_and_fails_when_one_check_fails(monkeypatch, capsys):
    def broken():
        raise checks.Problem("Download it: ollama pull x")

    def waiting():
        raise checks.Skip("needs Ollama")

    def crashed():
        raise RuntimeError("boom")

    monkeypatch.setattr(checks, "CHECKS", [("Fine", lambda: "Fine thing"), ("Broken", broken),
                                           ("Waiting", waiting), ("Crashed", crashed)])
    assert checks.run() == 1
    out = capsys.readouterr().out
    assert "✓ Fine thing" in out
    assert "✗ Broken\n      Download it: ollama pull x" in out
    assert "- Waiting: skipped, needs Ollama" in out
    assert "✗ Crashed\n      RuntimeError: boom" in out      # a line, not a traceback
    assert "2 problems" in out


def test_check_passes_when_every_check_passes(monkeypatch, capsys):
    monkeypatch.setattr(checks, "CHECKS", [("A", lambda: "a"), ("B", lambda: "b")])
    assert checks.run() == 0
    assert "All set" in capsys.readouterr().out


def test_cli_runs_the_chosen_command(monkeypatch):
    monkeypatch.setattr(checks, "run", lambda: 0)
    with pytest.raises(SystemExit) as stop:
        cli.main(["check"])
    assert stop.value.code == 0


def test_cli_without_a_command_shows_usage(capsys):
    with pytest.raises(SystemExit) as stop:
        cli.main([])
    assert stop.value.code == 2
    assert "check" in capsys.readouterr().err
