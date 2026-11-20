"""Tests for build_child_env, Windows home/strict env isolation, and SSH scrubbing (issue #18).

agy 1.2.16+ (and agy.exe 1.5+ on Windows) detects SSH sessions via
SSH_CONNECTION, SSH_CLIENT, or SSH_TTY and switches to file-based token storage
("Using file-based token storage because ... detected"). When credentials live
in the OS keyring and no real token file exists, build_child_env must strip
those variables while leaving SSH_AUTH_SOCK intact and preserving parent
os.environ.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

plugin_dir = Path(__file__).resolve().parent.parent
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

from process import build_child_env

_MOCK_TOKEN_PATH = Path("/tok") if os.name != "nt" else Path(r"C:\tok")


class ChildEnvTests(unittest.TestCase):
    def test_posix_no_token_file_strips_ssh_session_vars_and_keeps_auth_sock(self):
        env_sample = {
            "HOME": "/original/home",
            "PATH": "/usr/bin:/bin",
            "SSH_CONNECTION": "192.168.1.50 54321 192.168.1.1 22",
            "SSH_CLIENT": "192.168.1.50 54321 22",
            "SSH_TTY": "/dev/pts/2",
            "SSH_AUTH_SOCK": "/tmp/ssh-agent.sock",
        }
        with patch.dict(os.environ, env_sample, clear=True):
            with patch("process.os.name", "posix"):
                with patch("process.resolve_real_token_path", return_value=None):
                    child_env = build_child_env("/isolated/home")

        self.assertEqual(child_env["HOME"], "/isolated/home")
        self.assertNotIn("SSH_CONNECTION", child_env)
        self.assertNotIn("SSH_CLIENT", child_env)
        self.assertNotIn("SSH_TTY", child_env)
        self.assertIn("SSH_AUTH_SOCK", child_env)
        self.assertEqual(child_env["SSH_AUTH_SOCK"], "/tmp/ssh-agent.sock")

    def test_posix_with_token_file_preserves_all_ssh_vars(self):
        token_path = Path("/path/to/jetski-standalone-oauth-token")
        env_sample = {
            "HOME": "/original/home",
            "SSH_CONNECTION": "192.168.1.50 54321 192.168.1.1 22",
            "SSH_CLIENT": "192.168.1.50 54321 22",
            "SSH_TTY": "/dev/pts/2",
            "SSH_AUTH_SOCK": "/tmp/ssh-agent.sock",
        }
        with patch.dict(os.environ, env_sample, clear=True):
            with patch("process.os.name", "posix"):
                with patch("process.resolve_real_token_path", return_value=token_path):
                    child_env = build_child_env("/isolated/home")

        self.assertEqual(child_env["HOME"], "/isolated/home")
        self.assertEqual(child_env["SSH_CONNECTION"], "192.168.1.50 54321 192.168.1.1 22")
        self.assertEqual(child_env["SSH_CLIENT"], "192.168.1.50 54321 22")
        self.assertEqual(child_env["SSH_TTY"], "/dev/pts/2")
        self.assertEqual(child_env["SSH_AUTH_SOCK"], "/tmp/ssh-agent.sock")

    def test_parent_os_environ_remains_intact(self):
        env_sample = {
            "HOME": "/original/home",
            "SSH_CONNECTION": "10.0.0.1 12345 10.0.0.2 22",
            "SSH_CLIENT": "10.0.0.1 12345 22",
            "SSH_TTY": "/dev/pts/0",
            "SSH_AUTH_SOCK": "/run/user/1000/keyring/ssh",
        }
        with patch.dict(os.environ, env_sample, clear=True):
            with patch("process.os.name", "posix"):
                with patch("process.resolve_real_token_path", return_value=None):
                    build_child_env("/isolated/home")

            # Parent os.environ must still contain all original keys and values
            self.assertEqual(os.environ["HOME"], "/original/home")
            self.assertEqual(os.environ["SSH_CONNECTION"], "10.0.0.1 12345 10.0.0.2 22")
            self.assertEqual(os.environ["SSH_CLIENT"], "10.0.0.1 12345 22")
            self.assertEqual(os.environ["SSH_TTY"], "/dev/pts/0")
            self.assertEqual(os.environ["SSH_AUTH_SOCK"], "/run/user/1000/keyring/ssh")

    def test_windows_nt_splits_homedrive_and_homepath(self):
        env_sample = {
            "USERPROFILE": r"C:\Users\tester",
            "HOMEDRIVE": "C:",
            "HOMEPATH": r"\Users\tester",
        }
        with patch.dict(os.environ, env_sample, clear=True):
            with patch("process.os.name", "nt"):
                with patch("process.resolve_real_token_path", return_value=_MOCK_TOKEN_PATH):
                    child_env = build_child_env(r"C:\isolated\home")

        self.assertEqual(child_env["HOME"], r"C:\isolated\home")
        self.assertEqual(child_env["USERPROFILE"], r"C:\isolated\home")
        self.assertEqual(child_env["HOMEDRIVE"], "C:")
        self.assertEqual(child_env["HOMEPATH"], r"\isolated\home")

    def test_windows_nt_overrides_cross_drive_and_mixed_case_home_vars(self):
        # Cross-drive isolated home must replace parent HOMEDRIVE/HOMEPATH; also
        # verify mixed-case parent keys are removed case-insensitively.
        mixed_env = {
            "UserProfile": r"C:\Users\tester",
            "homedrive": "C:",
            "HomePath": r"\Users\tester",
        }
        with patch("process._filtered_parent_env", return_value=dict(mixed_env)):
            with patch("process.os.name", "nt"):
                with patch("process.resolve_real_token_path", return_value=_MOCK_TOKEN_PATH):
                    child_env = build_child_env(r"D:\temp\iso\home")

        self.assertEqual(child_env["USERPROFILE"], r"D:\temp\iso\home")
        self.assertEqual(child_env["HOMEDRIVE"], "D:")
        self.assertEqual(child_env["HOMEPATH"], r"\temp\iso\home")
        self.assertNotIn("homedrive", child_env)
        self.assertNotIn("HomePath", child_env)

    def test_windows_nt_unc_path_leaves_homedrive_and_homepath_unset(self):
        env_sample = {
            "USERPROFILE": r"C:\Users\tester",
            "HOMEDRIVE": "C:",
            "HOMEPATH": r"\Users\tester",
        }
        with patch.dict(os.environ, env_sample, clear=True):
            with patch("process.os.name", "nt"):
                with patch("process.resolve_real_token_path", return_value=_MOCK_TOKEN_PATH):
                    child_env = build_child_env(r"\\server\share\home")

        self.assertEqual(child_env["USERPROFILE"], r"\\server\share\home")
        self.assertNotIn("HOMEDRIVE", child_env)
        self.assertNotIn("HOMEPATH", child_env)

    def test_windows_nt_no_token_file_strips_ssh_vars_case_insensitively_and_keeps_auth_sock(self):
        mixed_env = {
            "USERPROFILE": r"C:\Users\tester",
            "ssh_connection": "192.168.1.50 54321 192.168.1.1 22",
            "Ssh_Client": "192.168.1.50 54321 22",
            "SSH_TTY": "pty1",
            "SSH_AUTH_SOCK": r"\\.\pipe\openssh-ssh-agent",
        }
        with patch("process._filtered_parent_env", return_value=dict(mixed_env)):
            with patch("process.os.name", "nt"):
                with patch("process.resolve_real_token_path", return_value=None):
                    child_env = build_child_env(r"C:\isolated\home")

        self.assertEqual(child_env["USERPROFILE"], r"C:\isolated\home")
        self.assertFalse(any(k.upper() in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY") for k in child_env))
        self.assertEqual(child_env["SSH_AUTH_SOCK"], r"\\.\pipe\openssh-ssh-agent")

    def test_windows_nt_with_token_file_preserves_all_ssh_vars(self):
        token_path = Path(r"C:\Users\tester\.gemini\antigravity-cli\jetski-standalone-oauth-token")
        env_sample = {
            "USERPROFILE": r"C:\Users\tester",
            "SSH_CONNECTION": "192.168.1.50 54321 192.168.1.1 22",
            "SSH_CLIENT": "192.168.1.50 54321 22",
            "SSH_TTY": "pty1",
            "SSH_AUTH_SOCK": r"\\.\pipe\openssh-ssh-agent",
        }
        with patch.dict(os.environ, env_sample, clear=True):
            with patch("process.os.name", "nt"):
                with patch("process.resolve_real_token_path", return_value=token_path):
                    child_env = build_child_env(r"C:\isolated\home")

        self.assertEqual(child_env["SSH_CONNECTION"], "192.168.1.50 54321 192.168.1.1 22")
        self.assertEqual(child_env["SSH_CLIENT"], "192.168.1.50 54321 22")
        self.assertEqual(child_env["SSH_TTY"], "pty1")
        self.assertEqual(child_env["SSH_AUTH_SOCK"], r"\\.\pipe\openssh-ssh-agent")

    def test_windows_strict_mode_retains_appdata_and_sets_isolated_home(self):
        env_sample = {
            "ANTIGRAVITY_ENV_STRICT": "1",
            "PATH": r"C:\Windows\System32",
            "SYSTEMROOT": r"C:\Windows",
            "APPDATA": r"C:\Users\tester\AppData\Roaming",
            "LOCALAPPDATA": r"C:\Users\tester\AppData\Local",
            "PROGRAMDATA": r"C:\ProgramData",
            "USERPROFILE": r"C:\Users\tester",
            "HOMEDRIVE": "C:",
            "HOMEPATH": r"\Users\tester",
            "OPENAI_API_KEY": "sk-secret",
            "UNLISTED_VAR": "drop-me",
        }
        with patch.dict(os.environ, env_sample, clear=True):
            with patch("process.os.name", "nt"):
                with patch("process.resolve_real_token_path", return_value=None):
                    child_env = build_child_env(r"D:\isolated\home")

        self.assertEqual(child_env["APPDATA"], r"C:\Users\tester\AppData\Roaming")
        self.assertEqual(child_env["LOCALAPPDATA"], r"C:\Users\tester\AppData\Local")
        self.assertEqual(child_env["PROGRAMDATA"], r"C:\ProgramData")
        self.assertEqual(child_env["SYSTEMROOT"], r"C:\Windows")
        self.assertEqual(child_env["USERPROFILE"], r"D:\isolated\home")
        self.assertEqual(child_env["HOMEDRIVE"], "D:")
        self.assertEqual(child_env["HOMEPATH"], r"\isolated\home")
        self.assertNotIn("OPENAI_API_KEY", child_env)
        self.assertNotIn("UNLISTED_VAR", child_env)


class ChildEnvSecretScrubbingTests(unittest.TestCase):
    """Hermes runs with gateway and dashboard secrets in its environment; agy must not receive them."""

    SECRETS = {
        "SLACK_BOT_TOKEN": "xoxb-secret",
        "SLACK_APP_TOKEN": "xapp-secret",
        "TELEGRAM_BOT_TOKEN": "tg-secret",
        "HERMES_DASHBOARD_SECRET": "dash-secret",
        "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH": "hash",
        "OPENAI_API_KEY": "sk-secret",
        "GEMINI_API_KEY": "g-secret",
        "AWS_SECRET_ACCESS_KEY": "aws-secret",
        "GITHUB_TOKEN": "ghp-secret",
    }
    HARMLESS = {
        "PATH": "/usr/bin",
        "LANG": "C.UTF-8",
        "HTTPS_PROXY": "http://proxy:3128",
        "SSL_CERT_FILE": "/etc/ssl/cert.pem",
        "SSH_AUTH_SOCK": "/tmp/agent.sock",
        "AGY_CLI_DISABLE_AUTO_UPDATE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "MONKEY": "banana",
    }

    def _build(self, extra=None):
        env = {**self.SECRETS, **self.HARMLESS, **(extra or {})}
        with patch.dict(os.environ, env, clear=True):
            with patch("process.os.name", "posix"):
                with patch("process.resolve_real_token_path", return_value=_MOCK_TOKEN_PATH):
                    return build_child_env("/isolated/home")

    def test_credential_looking_variables_are_not_passed(self):
        child = self._build()
        for name in self.SECRETS:
            self.assertNotIn(name, child)

    def test_ordinary_variables_proxy_and_ssh_agent_are_kept(self):
        child = self._build()
        for name, value in self.HARMLESS.items():
            self.assertEqual(child[name], value)

    def test_auto_update_is_disabled_unless_the_parent_sets_it(self):
        # agy's background updater opens its own console on Windows; an
        # explicit parent value (kept above) still wins.
        harmless = {k: v for k, v in self.HARMLESS.items() if k != "AGY_CLI_DISABLE_AUTO_UPDATE"}
        with patch.dict(os.environ, harmless, clear=True):
            with patch("process.os.name", "posix"):
                with patch("process.resolve_real_token_path", return_value=_MOCK_TOKEN_PATH):
                    child = build_child_env("/isolated/home")
        self.assertEqual(child["AGY_CLI_DISABLE_AUTO_UPDATE"], "true")

    def test_passthrough_keeps_named_variables(self):
        child = self._build({"ANTIGRAVITY_ENV_PASSTHROUGH": "GEMINI_API_KEY, GITHUB_TOKEN"})
        self.assertEqual(child["GEMINI_API_KEY"], "g-secret")
        self.assertEqual(child["GITHUB_TOKEN"], "ghp-secret")
        self.assertNotIn("SLACK_BOT_TOKEN", child)

    def test_strict_mode_passes_only_baseline_and_allowlist(self):
        child = self._build({"ANTIGRAVITY_ENV_STRICT": "1", "ANTIGRAVITY_ENV_ALLOWLIST": "MONKEY"})
        self.assertEqual(child["PATH"], "/usr/bin")
        self.assertEqual(child["HTTPS_PROXY"], "http://proxy:3128")
        self.assertEqual(child["MONKEY"], "banana")
        self.assertNotIn("TOKENIZERS_PARALLELISM", child)
        for name in self.SECRETS:
            self.assertNotIn(name, child)
        self.assertEqual(child["HOME"], "/isolated/home")

    def test_value_secret_conventions_are_scrubbed(self):
        extra = {
            "SLACK_WEBHOOK_URL": "https://hooks.slack.com/services/T/B/x",
            "DISCORD_WEBHOOK": "https://discord.com/api/webhooks/1/x",
            "DATABASE_URL": "postgres://u:p@h/db",
            "REDIS_URL": "redis://:p@h",
            "MONGO_URI": "mongodb://u:p@h",
            "SENTRY_DSN": "https://k@o.ingest.sentry.io/1",
            "COOKIE": "session=abc",
        }
        child = self._build(extra)
        for name in extra:
            self.assertNotIn(name, child)

    def test_dbus_session_address_is_kept_for_keyring_signin(self):
        child = self._build({"DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1000/bus"})
        self.assertIn("DBUS_SESSION_BUS_ADDRESS", child)

    def test_strict_mode_accepts_usual_truthy_spellings(self):
        for value in ("1", "true", "True ", "YES", "on"):
            child = self._build({"ANTIGRAVITY_ENV_STRICT": value})
            self.assertNotIn("MONKEY", child, value)
            self.assertIn("PATH", child, value)

    def test_strict_mode_keeps_temp_dirs_and_all_locale_variables(self):
        child = self._build({
            "ANTIGRAVITY_ENV_STRICT": "1",
            "TMPDIR": "/var/tmp",
            "LANGUAGE": "en",
            "LC_TIME": "C",
            "LC_NUMERIC": "C",
        })
        for name in ("TMPDIR", "LANGUAGE", "LC_TIME", "LC_NUMERIC"):
            self.assertIn(name, child)

    def test_strict_mode_keeps_lowercase_proxy_variables(self):
        child = self._build({
            "ANTIGRAVITY_ENV_STRICT": "1",
            "http_proxy": "http://proxy:3128",
            "https_proxy": "http://proxy:3128",
            "no_proxy": "localhost",
        })
        expected = ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY") if os.name == "nt" else ("http_proxy", "https_proxy", "no_proxy")
        for name in expected:
            self.assertIn(name, child)

    def test_names_are_matched_case_insensitively_on_windows(self):
        env = {"Path": r"C:\Windows", "Ssh_Auth_Sock": "sock", "Slack_Bot_Token": "t", "Monkey": "m"}
        with patch.dict(os.environ, {**env, "ANTIGRAVITY_ENV_PASSTHROUGH": "monkey"}, clear=True):
            with patch("process.os.name", "nt"), \
                 patch("process.resolve_real_token_path", return_value=_MOCK_TOKEN_PATH):
                child = build_child_env(r"C:\isolated\home")
        child_lower = {k.lower(): v for k, v in child.items()}
        self.assertIn("ssh_auth_sock", child_lower)
        self.assertIn("monkey", child_lower)
        self.assertNotIn("slack_bot_token", child_lower)

    def test_parent_environment_is_not_modified(self):
        env = {**self.SECRETS, **self.HARMLESS}
        with patch.dict(os.environ, env, clear=True):
            with patch("process.os.name", "posix"):
                with patch("process.resolve_real_token_path", return_value=_MOCK_TOKEN_PATH):
                    build_child_env("/isolated/home")
            self.assertEqual(os.environ["SLACK_BOT_TOKEN"], "xoxb-secret")


if __name__ == "__main__":
    unittest.main()
