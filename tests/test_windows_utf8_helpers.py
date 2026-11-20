"""Tests for Windows UTF-8 decoding, isolated env, and timeout reaping in models and usage."""

from __future__ import annotations

import contextlib
import json
import os
import shlex
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

plugin_dir = Path(__file__).resolve().parent.parent
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

import models  # noqa: E402
import process  # noqa: E402
import usage  # noqa: E402


def _write_cli_shim(directory: Path, name: str, script_body: str) -> Path:
    """Create a cross-platform executable shim running *script_body* via Python."""
    helper_py = directory / f"{name}_helper.py"
    helper_py.write_text(script_body, encoding="utf-8")

    if os.name == "nt":
        shim = directory / f"{name}.cmd"
        shim.write_text(
            f'@echo off\r\n"{sys.executable}" -B "{helper_py}" %*\r\n',
            encoding="utf-8",
        )
        return shim

    shim = directory / name
    shim.write_text(
        f"#!/bin/sh\nexec {shlex.quote(sys.executable)} -B {shlex.quote(str(helper_py))} \"$@\"\n",
        encoding="utf-8",
    )
    shim.chmod(0o755)
    return shim


def _preserve_and_reset_caches():
    prev_catalog = models._catalog_cache
    models._catalog_cache = None
    with usage._cache_lock:
        prev_usage = usage._cached_usage
        prev_ts = usage._cached_timestamp
        usage._cached_usage = None
        usage._cached_timestamp = 0.0

    def _restore() -> None:
        models._catalog_cache = prev_catalog
        with usage._cache_lock:
            usage._cached_usage = prev_usage
            usage._cached_timestamp = prev_ts

    return _restore


@contextlib.contextmanager
def _track_mkdtemp(target: str, prefix: str):
    """Wrap *target* mkdtemp and record created paths matching *prefix*."""
    created: list[Path] = []
    real_mkdtemp = tempfile.mkdtemp

    def tracking(*args, **kwargs):
        path = real_mkdtemp(*args, **kwargs)
        if str(kwargs.get("prefix", "")).startswith(prefix):
            created.append(Path(path))
        return path

    with patch(target, side_effect=tracking):
        yield created


_MODELS_UTF8_SCRIPT = (
    "import os, sys\n"
    "assert sys.argv[1:] == ['models'], sys.argv\n"
    "assert 'OPENAI_API_KEY' not in os.environ\n"
    "assert os.environ.get('AGY_CLI_DISABLE_AUTO_UPDATE') == 'true'\n"
    # Includes UTF-8 braille spinners (e2 a0 8b, e2 a0 8f where 0x8f fails cp1252)
    # and invalid UTF-8 bytes (\xff\xfe) requiring errors='replace'.
    "payload = (\n"
    "    b'\\xe2\\xa0\\x8b Fetching available models...\\n'\n"
    "    b'\\xe2\\xa0\\x8f Loading...\\n'\n"
    "    b'gemini-3.8-flash-high\\n'\n"
    "    b'\\xff\\xfe\\n'\n"
    "    b'claude-sonnet-4-6\\n'\n"
    ")\n"
    "sys.stdout.buffer.write(payload)\n"
    "sys.stdout.buffer.flush()\n"
)

_USAGE_EVENT = {
    "event": "command_result",
    "command": {
        "name": "usage",
        "data": {
            "groups": [
                {
                    "name": "Gemini Models",
                    "description": "Spinner \u280b\u280f caf\u00e9",
                    "buckets": [
                        {
                            "id": "gemini-5h",
                            "window": "5h",
                            "remaining_fraction": 0.85,
                            "reset_time": "2026-10-03T16:27:09Z",
                        }
                    ],
                }
            ]
        },
    },
}


def _usage_utf8_script() -> str:
    payload_json_literal = repr(json.dumps(_USAGE_EVENT, ensure_ascii=False).encode("utf-8"))
    return (
        "import os, sys\n"
        "assert sys.argv[1:] == ['-p', '/usage', '--output-format', 'json'], sys.argv\n"
        "assert 'OPENAI_API_KEY' not in os.environ\n"
        f"json_line = {payload_json_literal}\n"
        # Emit UTF-8 braille spinner (0x8f) + invalid UTF-8 (\xff\xfe) before valid JSON line.
        "sys.stdout.buffer.write(b'\\xe2\\xa0\\x8f\\xff\\xfe\\n' + json_line + b'\\n')\n"
        "sys.stdout.buffer.flush()\n"
    )


class WindowsUtf8HelpersTests(unittest.TestCase):
    def setUp(self) -> None:
        self.addCleanup(_preserve_and_reset_caches())

        p_token = patch("process.resolve_real_token_path", return_value=None)
        p_keychain = patch("process._link_macos_keychains")
        p_token.start()
        p_keychain.start()
        self.addCleanup(p_token.stop)
        self.addCleanup(p_keychain.stop)

    def test_load_catalog_passes_utf8_scrubbed_env_proxy_and_cleans_temp_dir(self) -> None:
        captured: dict[str, object] = {}

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            captured["kwargs"] = kwargs
            cwd = Path(kwargs["cwd"])
            self.assertTrue(cwd.is_dir())
            self.assertTrue(cwd.name.startswith("hermes_agy_models_"))
            return SimpleNamespace(stdout="gemini-3.8-flash-high\n", stderr="", returncode=0)

        parent_env = {
            "PATH": os.environ.get("PATH", ""),
            "OPENAI_API_KEY": "sk-secret",
            "SLACK_BOT_TOKEN": "xoxb-secret",
            "ANTIGRAVITY_PROXY": "http://127.0.0.1:8888",
        }
        with patch.dict(os.environ, parent_env, clear=True), \
                patch("process.resolve_agy_command", return_value="agy"), \
                patch("models.subprocess.run", side_effect=fake_run):
            catalog = models.load_catalog(timeout=5.0)

        self.assertEqual(catalog, {"gemini-3.8-flash": ("high",)})
        kwargs = captured["kwargs"]
        assert isinstance(kwargs, dict)
        self.assertEqual(kwargs["encoding"], "utf-8")
        self.assertEqual(kwargs["errors"], "replace")
        self.assertTrue(kwargs["text"])
        self.assertTrue(kwargs["capture_output"])

        child_env = kwargs["env"]
        assert isinstance(child_env, dict)
        self.assertNotIn("OPENAI_API_KEY", child_env)
        self.assertNotIn("SLACK_BOT_TOKEN", child_env)
        self.assertEqual(child_env["AGY_CLI_DISABLE_AUTO_UPDATE"], "true")
        self.assertEqual(child_env["HTTPS_PROXY"], "http://127.0.0.1:8888")
        self.assertEqual(child_env["HTTP_PROXY"], "http://127.0.0.1:8888")
        self.assertEqual(child_env["ALL_PROXY"], "http://127.0.0.1:8888")
        self.assertEqual(Path(child_env["HOME"]).parent, Path(kwargs["cwd"]))
        self.assertFalse(Path(kwargs["cwd"]).exists())

    def test_load_catalog_cleans_temp_dir_with_readonly_files_on_failure(self) -> None:
        captured_cwds: list[Path] = []

        def failing_run(argv, **kwargs):
            cwd = Path(kwargs["cwd"])
            captured_cwds.append(cwd)
            ro_dir = cwd / "conversations"
            ro_dir.mkdir(parents=True, exist_ok=True)
            ro_file = ro_dir / "state.db"
            ro_file.write_text("locked", encoding="utf-8")
            ro_file.chmod(stat.S_IREAD)
            ro_dir.chmod(stat.S_IREAD | stat.S_IEXEC)
            raise RuntimeError("spawn failure")

        with patch("process.resolve_agy_command", return_value="agy"), \
                patch("models.subprocess.run", side_effect=failing_run), \
                patch("process.reap_orphaned_plugin_workers") as mock_reap:
            catalog = models.load_catalog()

        self.assertEqual(catalog, {})
        self.assertEqual(len(captured_cwds), 1)
        self.assertFalse(captured_cwds[0].exists())
        mock_reap.assert_not_called()

    def test_load_catalog_timeout_removes_temp_dir_before_reaping_orphans(self) -> None:
        captured_cwds: list[Path] = []
        events: list[tuple[str, bool]] = []

        def timeout_run(argv, **kwargs):
            captured_cwds.append(Path(kwargs["cwd"]))
            raise subprocess.TimeoutExpired(cmd=argv, timeout=kwargs["timeout"])

        def record_reap() -> int:
            self.assertEqual(len(captured_cwds), 1)
            events.append(("reap", captured_cwds[0].exists()))
            raise RuntimeError("reaper error should be suppressed")

        with patch("process.resolve_agy_command", return_value="agy"), \
                patch("models.subprocess.run", side_effect=timeout_run), \
                patch("process.reap_orphaned_plugin_workers", side_effect=record_reap) as mock_reap:
            catalog = models.load_catalog(timeout=1.5)

        self.assertEqual(catalog, {})
        self.assertEqual(len(captured_cwds), 1)
        self.assertFalse(captured_cwds[0].exists())
        mock_reap.assert_called_once_with()
        self.assertEqual(events, [("reap", False)])

    def test_query_agy_usage_passes_utf8_handles_none_streams_and_cleans_up(self) -> None:
        captured: dict[str, object] = {}

        def fake_run_none_error(argv, **kwargs):
            captured["kwargs"] = kwargs
            return SimpleNamespace(returncode=2, stdout=None, stderr=None)

        with patch("usage.resolve_agy_command", return_value="agy"), \
                patch("usage.subprocess.run", side_effect=fake_run_none_error), \
                patch("usage.reap_orphaned_plugin_workers") as mock_reap:
            result = usage._query_agy_usage(timeout=2.0)

        self.assertIsNone(result)
        kwargs = captured["kwargs"]
        assert isinstance(kwargs, dict)
        self.assertEqual(kwargs["encoding"], "utf-8")
        self.assertEqual(kwargs["errors"], "replace")
        self.assertTrue(Path(kwargs["cwd"]).name.startswith("hermes_agy_usage_"))
        self.assertFalse(Path(kwargs["cwd"]).exists())
        mock_reap.assert_not_called()

        def fake_run_none_ok(argv, **kwargs):
            return SimpleNamespace(returncode=0, stdout=None, stderr=None)

        with patch("usage.resolve_agy_command", return_value="agy"), \
                patch("usage.subprocess.run", side_effect=fake_run_none_ok):
            self.assertIsNone(usage._query_agy_usage(timeout=2.0))

    def test_query_agy_usage_timeout_removes_temp_dir_before_reaping_orphans(self) -> None:
        captured_cwds: list[Path] = []
        events: list[tuple[str, bool]] = []

        def timeout_run(argv, **kwargs):
            captured_cwds.append(Path(kwargs["cwd"]))
            raise subprocess.TimeoutExpired(cmd=argv, timeout=kwargs["timeout"])

        def record_reap() -> int:
            self.assertEqual(len(captured_cwds), 1)
            events.append(("reap", captured_cwds[0].exists()))
            raise RuntimeError("reaper error should be suppressed")

        with patch("usage.resolve_agy_command", return_value="agy"), \
                patch("usage.subprocess.run", side_effect=timeout_run), \
                patch("usage.reap_orphaned_plugin_workers", side_effect=record_reap) as mock_reap:
            result = usage._query_agy_usage(timeout=1.5)

        self.assertIsNone(result)
        self.assertEqual(len(captured_cwds), 1)
        self.assertFalse(captured_cwds[0].exists())
        mock_reap.assert_called_once_with()
        self.assertEqual(events, [("reap", False)])

    def test_real_child_load_catalog_decodes_utf8_spinner_and_invalid_bytes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="utf8_shim_") as shim_dir:
            shim = _write_cli_shim(Path(shim_dir), "fake_agy_models", _MODELS_UTF8_SCRIPT)
            env = {k: v for k, v in os.environ.items() if k not in ("PYTHONIOENCODING", "PYTHONUTF8")}
            env["OPENAI_API_KEY"] = "sk-must-not-leak"

            with patch.dict(os.environ, env, clear=True), \
                    patch("process.resolve_agy_command", return_value=str(shim)), \
                    _track_mkdtemp("models.tempfile.mkdtemp", "hermes_agy_models_") as created_dirs:
                catalog = models.load_catalog(timeout=10.0)

        self.assertEqual(
            catalog,
            {"gemini-3.8-flash": ("high",), "claude-sonnet-4-6": ()},
        )
        self.assertEqual(len(created_dirs), 1)
        self.assertFalse(created_dirs[0].exists())

    def test_real_child_query_agy_usage_decodes_utf8_and_invalid_bytes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="utf8_shim_") as shim_dir:
            shim = _write_cli_shim(Path(shim_dir), "fake_agy_usage", _usage_utf8_script())
            env = {k: v for k, v in os.environ.items() if k not in ("PYTHONIOENCODING", "PYTHONUTF8")}
            env["OPENAI_API_KEY"] = "sk-must-not-leak"

            with patch.dict(os.environ, env, clear=True), \
                    patch("usage.resolve_agy_command", return_value=str(shim)), \
                    _track_mkdtemp("usage.tempfile.mkdtemp", "hermes_agy_usage_") as created_dirs:
                result = usage._query_agy_usage(timeout=10.0)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(len(result.groups), 1)
        self.assertEqual(result.groups[0].name, "Gemini Models")
        self.assertEqual(result.groups[0].description, "Spinner \u280b\u280f caf\u00e9")
        self.assertAlmostEqual(result.groups[0].buckets[0].remaining_fraction, 0.85)
        self.assertEqual(len(created_dirs), 1)
        self.assertFalse(created_dirs[0].exists())

    def test_non_utf8_host_process_decodes_real_child_utf8_output(self) -> None:
        """Spawn a non-UTF-8 (-X utf8=0) host Python to prove cp1252/C locales never blow up."""
        with tempfile.TemporaryDirectory(prefix="utf8_host_") as base_dir:
            base = Path(base_dir)
            empty_cfg = base / "empty_cfg"
            empty_cfg.mkdir()
            models_shim = _write_cli_shim(base, "models_shim", _MODELS_UTF8_SCRIPT)
            usage_shim = _write_cli_shim(base, "usage_shim", _usage_utf8_script())

            driver = (
                "import os, sys\n"
                "assert sys.flags.utf8_mode == 0, sys.flags.utf8_mode\n"
                "import models, usage\n"
                f"os.environ['ANTIGRAVITY_COMMAND'] = {str(models_shim)!r}\n"
                "cat = models.load_catalog(timeout=10.0)\n"
                "assert cat == {'gemini-3.8-flash': ('high',), 'claude-sonnet-4-6': ()}, cat\n"
                f"os.environ['ANTIGRAVITY_COMMAND'] = {str(usage_shim)!r}\n"
                "u = usage._query_agy_usage(timeout=10.0)\n"
                "assert u is not None and u.groups[0].description == 'Spinner \\u280b\\u280f caf\\u00e9', u\n"
                "print('NON_UTF8_HOST_OK')\n"
            )

            env = {k: v for k, v in os.environ.items() if k not in ("PYTHONIOENCODING", "PYTHONUTF8")}
            env["PYTHONPATH"] = str(plugin_dir)
            env["HOME"] = str(empty_cfg)
            env["USERPROFILE"] = str(empty_cfg)
            env["ANTIGRAVITY_CONFIG_DIR"] = str(empty_cfg)
            env["OPENAI_API_KEY"] = "sk-must-not-leak"
            if os.name != "nt":
                env["LC_ALL"] = "C"

            proc = subprocess.run(
                [sys.executable, "-X", "utf8=0", "-c", driver],
                cwd=base_dir,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15.0,
            )

        self.assertEqual(proc.returncode, 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}")
        self.assertIn("NON_UTF8_HOST_OK", proc.stdout)


if __name__ == "__main__":
    unittest.main()
