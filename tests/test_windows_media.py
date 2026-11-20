"""Cross-platform regression tests for Windows media paths, normcase allowlisting, and file:// URIs."""

from __future__ import annotations

import io
import json
import ntpath
import os
import sys
import tempfile
import threading
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

PLUGIN_DIR = Path(__file__).resolve().parent.parent
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

import prompt as agy_prompt  # noqa: E402
import stream as agy_stream  # noqa: E402


def _make_stream(
    messages: list[dict] | None = None,
    media_paths: set[str] | None = None,
    events: list[dict] | None = None,
    *,
    is_worker: bool = False,
) -> agy_stream.AntigravityStream:
    proc = MagicMock()
    proc.poll.return_value = None if is_worker else 0
    lines = "".join(json.dumps(ev) + "\n" for ev in (events or []))
    proc.stdout = io.StringIO(lines)
    client = SimpleNamespace(
        _cancelled=False,
        _lock=MagicMock(),
        _active_processes={proc},
        _worker_proc=proc if is_worker else None,
        _terminate_process=MagicMock(),
        _terminate_worker=MagicMock(),
        _update_worker_history=MagicMock(),
        _note_worker_activity=MagicMock(),
        _set_worker_conversation_id=MagicMock(),
    )
    worker_lock = threading.Lock() if is_worker else None
    worker_lock_held = False
    if worker_lock is not None:
        worker_lock.acquire()
        worker_lock_held = True
    st = agy_stream.AntigravityStream(
        proc=proc,
        client=client,
        model="gemini-3.8-flash",
        timeout=5.0,
        is_worker=is_worker,
        worker_lock=worker_lock,
        worker_lock_held=worker_lock_held if is_worker else None,
        messages=messages,
    )
    if media_paths is not None:
        st._media_paths = set(media_paths)
    return st


class TestWindowsFileURIMaterialization:
    @pytest.mark.parametrize(
        ("uri", "expected"),
        [
            ("file:///C:/Users/Jane/pic.png", (r"C:\Users\Jane\pic.png", "Image")),
            (
                "file:///C:/Users/Jane%20Doe/cat%20pic.png",
                (r"C:\Users\Jane Doe\cat pic.png", "Image"),
            ),
            ("file://localhost/C:/foo/bar.webp", (r"C:\foo\bar.webp", "Image")),
            ("file://C:/foo/bar.png", (r"C:\foo\bar.png", "Image")),
            (r"file://C:\foo\bar.png", (r"C:\foo\bar.png", "Image")),
            (
                r"file://C:\Users\Jane%20Doe\clip.mp4",
                (r"C:\Users\Jane Doe\clip.mp4", "Video"),
            ),
            ("FILE:///C:/Users/Jane/audio.mp3", (r"C:\Users\Jane\audio.mp3", "Audio")),
        ],
    )
    def test_valid_windows_file_uris(self, uri: str, expected: tuple[str, str]):
        sink: set[str] = set()
        assert agy_prompt._materialize_media(uri, sink) == expected
        assert sink == set()

    @pytest.mark.parametrize(
        "uri",
        [
            "file://evil.host/C:/foo/bar.png",
            "FILE://evil.host/C:/foo/bar.png",
            "file://127.0.0.1/C:/foo/bar.png",
            "file://c:8080/foo/bar.png",
            "file://c-host/share/bar.png",
            "file:///C:/Users/Jane/.hermes/auth.json",
            r"file://C:\Users\Jane\.ssh\id_rsa",
            "file:///C:/",
        ],
    )
    def test_refused_windows_file_uris(self, uri: str):
        assert agy_prompt._materialize_media(uri) == ("", "")

    def test_posix_file_uri_contracts_preserved(self):
        assert agy_prompt._materialize_media("file:///tmp/pic.png") == (
            "/tmp/pic.png",
            "Image",
        )
        assert agy_prompt._materialize_media("file://localhost/tmp/clip.mp4") == (
            "/tmp/clip.mp4",
            "Video",
        )


class TestPluginOwnedMediaPath:
    def test_accepts_inside_and_rejects_outside_and_prefix_sibling(
        self, tmp_path, monkeypatch
    ):
        cache_dir = tmp_path / "CacheRoot" / "agy_media"
        monkeypatch.setattr(agy_prompt, "_media_cache_dir", lambda: str(cache_dir))

        inside = cache_dir / "abc.png"
        sibling = tmp_path / "CacheRoot" / "agy_media_evil" / "abc.png"
        outside = tmp_path / "other" / "abc.png"

        assert agy_prompt._is_plugin_owned_media_path(str(inside)) is True
        assert agy_prompt._is_plugin_owned_media_path(str(sibling)) is False
        assert agy_prompt._is_plugin_owned_media_path(str(outside)) is False

        if os.name == "nt":
            flipped = str(inside)
            flipped = flipped[0].swapcase() + flipped[1:].replace("CacheRoot", "cacheroot")
            assert agy_prompt._is_plugin_owned_media_path(flipped) is True
            assert agy_prompt._is_plugin_owned_media_path(flipped.replace("\\", "/")) is True

    def test_mixed_case_windows_drive_and_dir(self, monkeypatch):
        cache_dir = r"C:\Users\Jane\.hermes\cache\agy_media"
        monkeypatch.setattr(agy_prompt, "_media_cache_dir", lambda: cache_dir)
        monkeypatch.setattr(agy_prompt.os.path, "normcase", ntpath.normcase)
        monkeypatch.setattr(agy_prompt.os.path, "realpath", ntpath.normpath)
        monkeypatch.setattr(agy_prompt.os, "sep", "\\")

        assert (
            agy_prompt._is_plugin_owned_media_path(
                r"c:\users\jane\.hermes\cache\agy_media\img.png"
            )
            is True
        )
        assert (
            agy_prompt._is_plugin_owned_media_path(
                r"c:\users\jane\.hermes\cache\agy_media_other\img.png"
            )
            is False
        )
        assert (
            agy_prompt._is_plugin_owned_media_path(r"C:\Users\Jane\.hermes\auth.json")
            is False
        )


class TestWindowsNativeMediaStepPermissions:
    @pytest.mark.parametrize("tool_name", ["read_file", "view_file"])
    @pytest.mark.parametrize(
        "tool_info",
        [
            r"read_file(C:\Users\Jane\pic.png)",
            r"read_file(c:\users\jane\pic.png)",
            "read_file(C:/Users/Jane/pic.png)",
            r'Reading "C:\Users\Jane\pic.png"',
            r'Reading "c:\\users\\jane\\pic.png"',
            r"Reading C:\Users\Jane\pic.png",
            r"Viewing C:\Users\Jane\pic.png, 24 KB",
            r"read_file(C:\Users\Jane\..\Jane\pic.png)",
        ],
    )
    def test_permits_windows_referenced_variants(self, tool_name: str, tool_info: str):
        st = _make_stream(media_paths={r"C:\Users\Jane\pic.png"})
        assert st._is_permitted_native_media_step(tool_name, tool_info) is True

    @pytest.mark.parametrize(
        "tool_info",
        [
            r'Reading "C:\Users\Jane Doe\My Pics\cat 1.png"',
            r'Reading "C:\\Users\\Jane Doe\\My Pics\\cat 1.png"',
            r"Reading 'c:\users\jane doe\my pics\cat 1.png'",
            r"read_file(C:\Users\Jane Doe\My Pics\cat 1.png)",
            r"Reading C:\Users\Jane Doe\My Pics\cat 1.png",
            r"Reading C:\Users\Jane Doe\My Pics\cat 1.png, 128 KB",
            r"Reading C:\Users\Jane Doe\My Pics\cat 1.png (image/png)",
        ],
    )
    def test_permits_windows_paths_with_spaces_quoted_and_unquoted(self, tool_info: str):
        st = _make_stream(media_paths={r"C:\Users\Jane Doe\My Pics\cat 1.png"})
        assert st._is_permitted_native_media_step("read_file", tool_info) is True

    def test_permits_windows_path_with_parentheses_and_apostrophe(self):
        ref = r"C:\Users\O'Connor\Photos (1)\cat (2).png"
        st = _make_stream(media_paths={ref})
        assert (
            st._is_permitted_native_media_step(
                "read_file", f"read_file({ref})"
            )
            is True
        )
        assert (
            st._is_permitted_native_media_step(
                "read_file", f"Reading {ref}"
            )
            is True
        )

    def test_permits_unreferenced_new_hermes_media_roots(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes_home"
        other_temp = tmp_path / "unrelated_temp"
        module = types.ModuleType("hermes_constants")
        setattr(module, "get_hermes_home", lambda: str(hermes_home))
        monkeypatch.setitem(sys.modules, "hermes_constants", module)
        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(other_temp))

        st = _make_stream(media_paths=set())
        for subdir, filename in (
            ("image_cache", "photo.png"),
            ("audio_cache", "voice.ogg"),
            ("hosted-room-attachments", "clip.mp4"),
        ):
            target = str(hermes_home / subdir / filename)
            assert st._is_permitted_native_media_step("read_file", f'Reading "{target}"') is True
            assert st._is_permitted_native_media_step("view_file", f"read_file({target})") is True

        # Sibling directories and auth files under hermes_home remain denied.
        assert (
            st._is_permitted_native_media_step(
                "read_file", f'Reading "{hermes_home / "auth.json"}"'
            )
            is False
        )
        assert (
            st._is_permitted_native_media_step(
                "read_file", f'Reading "{hermes_home / "image_cache_evil" / "x.png"}"'
            )
            is False
        )
        assert (
            st._is_permitted_native_media_step(
                "read_file",
                f'Reading "{hermes_home / "image_cache" / ".." / "auth.json"}"',
            )
            is False
        )

    @pytest.mark.parametrize("tool_name", ["run_command", "write_to_file", "edit_file"])
    def test_denies_non_media_tools_even_with_referenced_image(self, tool_name: str):
        ref = r"C:\Users\Jane\pic.png"
        st = _make_stream(media_paths={ref})
        assert st._is_permitted_native_media_step(tool_name, f"read_file({ref})") is False
        assert st._is_permitted_native_media_step(tool_name, f'Reading "{ref}"') is False

    @pytest.mark.parametrize("tool_name", ["read_file", "view_file"])
    @pytest.mark.parametrize(
        "target",
        [
            r"C:\Users\Jane\.hermes\auth.json",
            r"C:\Users\Jane\.ssh\id_rsa",
            r"C:\Users\Jane\.gemini\antigravity-cli\jetski-standalone-oauth-token",
            r"C:\Windows\System32\drivers\etc\hosts",
            "/etc/passwd",
        ],
    )
    def test_denies_sensitive_paths(self, tool_name: str, target: str):
        st = _make_stream(media_paths={r"C:\Users\Jane\pic.png"})
        assert st._is_permitted_native_media_step(tool_name, f"read_file({target})") is False
        assert st._is_permitted_native_media_step(tool_name, f'Reading "{target}"') is False
        assert st._is_permitted_native_media_step(tool_name, f"Reading {target}") is False

    @pytest.mark.parametrize(
        "tool_info",
        [
            r'Reading "C:\Users\Jane\pic.png (evil)"',
            r"Reading C:\Users\Jane\pic.png.exe",
            r"Reading C:\Users\Jane\pic.png2",
            r"Reading C:\Users\Jane\pic.png\..\..\.ssh\id_rsa",
            r"read_file(C:\Users\Jane\pic.png\secret.txt)",
            "",
        ],
    )
    def test_denies_boundary_and_traversal_bypasses(self, tool_info: str):
        st = _make_stream(media_paths={r"C:\Users\Jane\pic.png"})
        assert st._is_permitted_native_media_step("read_file", tool_info) is False

    def test_posix_drive_prefix_does_not_leak_under_tmp(self, monkeypatch):
        # Even if a POSIX test runs with cwd inside /tmp, C:/tmp/secret.txt must not
        # be mistaken for /tmp/secret.txt or resolved relative to /tmp.
        st = _make_stream(media_paths=set())
        for tool_info in (
            "read_file(C:/tmp/secret.txt)",
            "read_file(C:/foo bar/tmp/secret.txt)",
        ):
            assert st._is_permitted_native_media_step("read_file", tool_info) is False

    def test_end_to_end_file_uri_preserves_casing_and_allows_stream_step(self):
        uri = "file:///C:/Users/Jane%20Doe/My%20Pics/Cat%201.PNG"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is in this image?"},
                    {"type": "image_url", "image_url": {"url": uri}},
                ],
            }
        ]
        ref_paths = agy_prompt.media_reference_paths(messages)
        assert ref_paths == {r"C:\Users\Jane Doe\My Pics\Cat 1.PNG"}

        events = [
            {"event": "init", "conversation_id": "win-media-1"},
            {
                "event": "step_update",
                "step_update": {
                    "step_type": "tool",
                    "tool_name": "read_file",
                    "tool_info": 'Reading "c:\\\\users\\\\jane doe\\\\my pics\\\\cat 1.png"',
                },
            },
            {
                "event": "step_update",
                "step_update": {"text_delta": "A tabby cat."},
            },
            {
                "event": "result",
                "result": {"status": "SUCCESS", "response": "A tabby cat."},
            },
        ]
        st = _make_stream(messages=messages, events=events, is_worker=True)
        chunks = list(st)
        text = "".join(
            c.choices[0].delta.content
            for c in chunks
            if getattr(c, "choices", None) and c.choices[0].delta.content
        )
        assert text == "A tabby cat."
        st.client._terminate_worker.assert_not_called()
        st.client._terminate_process.assert_not_called()
        assert not st.worker_lock.locked()

    def test_end_to_end_stream_neutralizes_disallowed_windows_read(self):
        events = [
            {"event": "init", "conversation_id": "win-media-deny"},
            {
                "event": "step_update",
                "step_update": {
                    "step_type": "tool",
                    "tool_name": "read_file",
                    "tool_info": r'Reading "C:\\Users\\Jane\\.hermes\\auth.json"',
                },
            },
        ]
        st = _make_stream(messages=[{"role": "user", "content": "hi"}], events=events)
        chunks = list(st)
        contents = [
            c.choices[0].delta.content
            for c in chunks
            if getattr(c, "choices", None) and c.choices[0].delta.content
        ]
        assert len(contents) == 1
        assert "blocked by host security policy" in contents[0]
        st.client._terminate_process.assert_called_once_with(st.proc)
