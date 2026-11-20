"""Concurrency and lock-ownership tests for AntigravityClient._create_chat_completion.

Pins the two worker-lock fixes in ``client.py``:
1. Non-streaming turns keep ``_worker_lock`` owned by ``_create_chat_completion``
   and never probe ``_worker_lock.locked()`` on exit (preventing cross-thread
   lock theft when another thread acquires immediately after release).
2. Token-contention retries (``auth_retried``) hold ``_worker_lock`` across
   attempt 1 completion, ``sync_isolated_token``, worker respawn (preserving
   ``conversation_id``), retry write, and attempt 2 completion, and media
   permission errors release the lock without terminating a live worker.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import threading
import unittest
from collections import deque
from pathlib import Path
from typing import Any, Callable
from unittest.mock import MagicMock, patch

plugin_dir = Path(__file__).resolve().parent.parent
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

import client as client_module  # noqa: E402
from client import AntigravityClient  # noqa: E402


MODEL = "gemini-3.8-flash-high"
MESSAGES = [{"role": "user", "content": "hello"}]
AUTH_CONTENTION_STDERR = (
    "E errorreport error getting token source: You are not logged into Antigravity."
)


def _success_lines(conv_id: str, text: str) -> list[str]:
    events = [
        {"event": "init", "conversation_id": conv_id},
        {"event": "step_update", "step_update": {"text_delta": text}},
        {
            "event": "result",
            "result": {"status": "SUCCESS", "response": text, "usage": {}},
        },
    ]
    return [json.dumps(event) + "\n" for event in events] + [""]


def _error_lines(conv_id: str, message: str) -> list[str]:
    events = [
        {"event": "init", "conversation_id": conv_id},
        {"event": "result", "result": {"status": "ERROR", "error": message}},
    ]
    return [json.dumps(event) + "\n" for event in events] + [""]


class _FakeStderrDrainer:
    def __init__(self, tail: str) -> None:
        self._tail = tail

    def get_tail(self, timeout: float = 1.0) -> str:
        return self._tail


class _FakeStdin:
    def __init__(self, on_write: Callable[[str], None] | None = None) -> None:
        self.writes: list[str] = []
        self._on_write = on_write

    def write(self, data: str) -> int:
        if self._on_write is not None:
            self._on_write(data)
        self.writes.append(data)
        return len(data)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class _ScriptedStdout:
    def __init__(
        self,
        lines: list[str],
        *,
        on_read: Callable[[], None] | None = None,
    ) -> None:
        self._lines = deque(lines)
        self._on_read = on_read

    def readline(self) -> str:
        if self._on_read is not None:
            self._on_read()
        return self._lines.popleft() if self._lines else ""


class _LockSpy:
    """Real threading.Lock wrapper recording locked() probes and release hooks."""

    def __init__(self, on_release: Callable[[], None] | None = None) -> None:
        self._inner = threading.Lock()
        self._on_release = on_release
        self.locked_probe_count = 0
        self.release_count = 0

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        if timeout == -1:
            return self._inner.acquire(blocking=blocking)
        return self._inner.acquire(blocking=blocking, timeout=timeout)

    def release(self) -> None:
        self.release_count += 1
        self._inner.release()
        if self._on_release is not None:
            self._on_release()

    def locked(self) -> bool:
        self.locked_probe_count += 1
        return self._inner.locked()


def _make_proc(
    lines: list[str],
    *,
    alive: bool = True,
    exit_code: int = 0,
    stderr_tail: str = "",
    on_write: Callable[[str], None] | None = None,
    on_read: Callable[[], None] | None = None,
) -> MagicMock:
    proc = MagicMock()
    proc.stdin = _FakeStdin(on_write=on_write)
    proc.stdout = _ScriptedStdout(lines, on_read=on_read)
    proc.stderr = io.StringIO("")
    proc.poll.return_value = None if alive else exit_code
    proc.wait.return_value = exit_code
    if stderr_tail:
        proc._stderr_drainer = _FakeStderrDrainer(stderr_tail)
    return proc


class WorkerLockRaceTests(unittest.TestCase):
    def setUp(self) -> None:
        temp_dir = tempfile.TemporaryDirectory(prefix="hermes_agy_lock_test_")
        self.addCleanup(temp_dir.cleanup)
        self.temp_dir = Path(temp_dir.name)
        seams = {
            "is_authenticated": lambda: True,
            "resolve_agy_command": lambda: "synthetic-agy",
            "resolve_model_and_effort": lambda model, effort: (model or MODEL, effort),
            "sync_isolated_token": lambda home: True,
            "start_stderr_drainer": lambda proc: None,
            "dump_prompt_debug": lambda **kwargs: None,
            "_worker_cpu_seconds": lambda proc: None,
        }
        for name, replacement in seams.items():
            patcher = patch.object(client_module, name, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        token_patcher = patch("process.resolve_real_token_path", return_value=None)
        keychain_patcher = patch("process._link_macos_keychains")
        token_patcher.start()
        keychain_patcher.start()
        self.addCleanup(token_patcher.stop)
        self.addCleanup(keychain_patcher.stop)

    def _client(self) -> AntigravityClient:
        client = AntigravityClient(cwd=str(self.temp_dir))
        self.addCleanup(client.close)
        return client

    def _run_release_handoff_scenario(self, *, fail_turn: bool) -> None:
        client = self._client()
        handoff_lock = threading.Lock()
        handoff_started = False
        thread_b_acquired = threading.Event()
        thread_b_done = threading.Event()
        thread_b_release = threading.Event()
        thread_b_errors: list[BaseException] = []

        def _thread_b() -> None:
            try:
                if not client._worker_lock.acquire(blocking=False):
                    raise AssertionError("Thread B failed to acquire _worker_lock after release")
                thread_b_acquired.set()
                thread_b_release.wait(timeout=5.0)
            except BaseException as exc:  # pragma: no cover - captured for assertion
                thread_b_errors.append(exc)
                thread_b_acquired.set()
            finally:
                if client._worker_lock._inner.locked():
                    client._worker_lock._inner.release()
                thread_b_done.set()

        def _on_first_release() -> None:
            nonlocal handoff_started
            with handoff_lock:
                if handoff_started:
                    return
                handoff_started = True
            worker_b = threading.Thread(target=_thread_b, daemon=True)
            worker_b.start()
            # Block Thread A inside release() until Thread B holds the lock,
            # so any subsequent locked() + release() in Thread A collides
            # directly with Thread B's live lock.
            if not thread_b_acquired.wait(timeout=5.0):
                raise AssertionError("Timed out waiting for Thread B to acquire _worker_lock")

        spy_lock = _LockSpy(on_release=_on_first_release)
        client._worker_lock = spy_lock

        lines = (
            _error_lines("conv-err", "synthetic turn failure")
            if fail_turn
            else _success_lines("conv-ok", "turn answer")
        )
        proc = _make_proc(lines, alive=True)

        try:
            with patch.object(client_module.subprocess, "Popen", return_value=proc):
                if fail_turn:
                    with self.assertRaisesRegex(RuntimeError, "synthetic turn failure"):
                        client.chat.completions.create(
                            model=MODEL, messages=MESSAGES, stream=False
                        )
                else:
                    res = client.chat.completions.create(
                        model=MODEL, messages=MESSAGES, stream=False
                    )
                    self.assertEqual(res.choices[0].message.content, "turn answer")

            self.assertEqual(
                spy_lock.locked_probe_count,
                0,
                "_create_chat_completion must not probe _worker_lock.locked()",
            )
            self.assertEqual(spy_lock.release_count, 1)
            self.assertTrue(thread_b_acquired.is_set())
            self.assertEqual(thread_b_errors, [])
            self.assertTrue(
                spy_lock._inner.locked(),
                "Thread A stole Thread B's live _worker_lock after returning",
            )
            self.assertFalse(
                spy_lock.acquire(blocking=False),
                "Thread C acquired _worker_lock while Thread B still held it",
            )
        finally:
            thread_b_release.set()
            thread_b_done.wait(timeout=5.0)
        self.assertFalse(spy_lock._inner.locked())

    def test_non_streaming_return_never_releases_another_threads_lock(self) -> None:
        self._run_release_handoff_scenario(fail_turn=False)

    def test_non_streaming_error_never_releases_another_threads_lock(self) -> None:
        self._run_release_handoff_scenario(fail_turn=True)

    def test_auth_contention_retry_holds_worker_lock_throughout(self) -> None:
        client = self._client()
        observations: list[tuple[str, bool]] = []

        # Attempt 1 emits init and then dies without a result amid auth contention.
        proc1 = _make_proc(
            [json.dumps({"event": "init", "conversation_id": "conv-retry-1"}) + "\n", ""],
            alive=False,
            exit_code=1,
            stderr_tail=AUTH_CONTENTION_STDERR,
        )
        proc2 = _make_proc(
            _success_lines("conv-retry-1", "recovered answer"),
            alive=True,
            on_write=lambda _data: observations.append(
                ("retry_write", client._worker_lock.locked())
            ),
        )

        real_collect = client_module.collect_stream_completion
        real_spawn = client._get_or_spawn_worker
        spawn_records: list[tuple[int, str | None, bool]] = []

        def _spy_collect(stream_iter: Any) -> Any:
            attempt = sum(1 for tag, _ in observations if tag.startswith("collect_enter_")) + 1
            observations.append((f"collect_enter_{attempt}", client._worker_lock.locked()))
            try:
                result = real_collect(stream_iter)
            except Exception:
                observations.append((f"collect_error_{attempt}", client._worker_lock.locked()))
                raise
            observations.append((f"collect_done_{attempt}", client._worker_lock.locked()))
            return result

        def _spy_sync(home: Path) -> bool:
            observations.append(("sync_isolated_token", client._worker_lock.locked()))
            return True

        def _spy_spawn(
            model: str, effort: str | None, conversation_id: str | None = None
        ) -> Any:
            idx = len(spawn_records) + 1
            spawn_records.append((idx, conversation_id, client._worker_lock.locked()))
            observations.append((f"spawn_{idx}", client._worker_lock.locked()))
            return real_spawn(model, effort, conversation_id=conversation_id)

        with (
            patch.object(client_module.subprocess, "Popen", side_effect=[proc1, proc2]),
            patch.object(client_module, "collect_stream_completion", side_effect=_spy_collect),
            patch.object(client_module, "sync_isolated_token", side_effect=_spy_sync),
            patch.object(client, "_get_or_spawn_worker", side_effect=_spy_spawn),
        ):
            res = client.chat.completions.create(
                model=MODEL, messages=MESSAGES, stream=False
            )

        self.assertEqual(res.choices[0].message.content, "recovered answer")
        self.assertFalse(client._worker_lock.locked())
        self.assertTrue(client._worker_lock.acquire(blocking=False))
        client._worker_lock.release()

        self.assertEqual(
            [tag for tag, _ in observations],
            [
                "spawn_1",
                "sync_isolated_token",  # inside _get_or_spawn_worker for proc1
                "collect_enter_1",
                "collect_error_1",
                "sync_isolated_token",  # auth_retried re-sync
                "spawn_2",
                "sync_isolated_token",  # inside _get_or_spawn_worker for proc2
                "retry_write",
                "collect_enter_2",
                "collect_done_2",
            ],
        )
        self.assertTrue(
            all(held for _, held in observations),
            f"_worker_lock dropped during auth retry: {observations}",
        )
        # Attempt 1's init conversation_id was preserved across stream_iter.close()
        # into the retry spawn.
        self.assertEqual(spawn_records, [(1, None, True), (2, "conv-retry-1", True)])

    def test_auth_contention_retry_preserves_continuation_conversation_id(self) -> None:
        client = self._client()
        turn1 = [{"role": "user", "content": "first turn"}]
        turn2 = turn1 + [
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "second turn"},
        ]

        # Pre-seed an established conversation whose worker exited before turn 2.
        client._worker_conversation_id = "conv-existing"
        client._conversation_history = list(turn1)

        # First attempt of turn 2 dies before emitting any stdout (even init).
        proc1 = _make_proc(
            [""],
            alive=False,
            exit_code=1,
            stderr_tail=AUTH_CONTENTION_STDERR,
        )
        proc2 = _make_proc(_success_lines("conv-existing", "continued answer"), alive=True)

        with patch.object(client_module.subprocess, "Popen", side_effect=[proc1, proc2]) as popen:
            res = client.chat.completions.create(
                model=MODEL, messages=turn2, stream=False
            )

        self.assertEqual(res.choices[0].message.content, "continued answer")
        self.assertFalse(client._worker_lock.locked())
        self.assertEqual(popen.call_count, 2)
        retry_cmd = popen.call_args_list[1].args[0]
        self.assertIn("--conversation", retry_cmd)
        self.assertEqual(retry_cmd[retry_cmd.index("--conversation") + 1], "conv-existing")
        self.assertEqual(client._worker_conversation_id, "conv-existing")
        self.assertEqual(client._worker_history, turn2)
        self.assertEqual(client._conversation_history, turn2)

    def test_auth_contention_retry_keeps_concurrent_caller_on_oneshot(self) -> None:
        client = self._client()
        in_retry_sync = threading.Event()
        release_retry = threading.Event()
        oneshot_result: list[str] = []
        oneshot_errors: list[BaseException] = []

        proc1 = _make_proc(
            [""],
            alive=False,
            exit_code=1,
            stderr_tail=AUTH_CONTENTION_STDERR,
        )
        proc2 = _make_proc(_success_lines("conv-r", "retry worker answer"), alive=True)
        oneshot_proc = _make_proc(
            _success_lines("conv-o", "oneshot answer"),
            alive=False,
        )

        sync_calls = 0

        def _sync_with_pause(_home: Path) -> bool:
            nonlocal sync_calls
            sync_calls += 1
            # Call 1 is proc1 spawn; call 2 is auth_retried before proc2 spawn.
            if sync_calls == 2:
                self.assertTrue(client._worker_lock.locked())
                in_retry_sync.set()
                if not release_retry.wait(timeout=5.0):
                    raise AssertionError("Timed out waiting to release retry")
            return True

        def _run_concurrent_caller() -> None:
            try:
                res = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": "parallel turn"}],
                    stream=False,
                )
                oneshot_result.append(res.choices[0].message.content)
            except BaseException as exc:  # pragma: no cover - captured for assertion
                oneshot_errors.append(exc)

        worker_procs = iter([proc1, proc2])

        def _popen_router(cmd_args: list[str], *args: Any, **kwargs: Any) -> Any:
            if "--input-format" not in cmd_args:
                return oneshot_proc
            return next(worker_procs)

        with (
            patch.object(client_module.subprocess, "Popen", side_effect=_popen_router),
            patch.object(client_module, "sync_isolated_token", side_effect=_sync_with_pause),
        ):
            caller = threading.Thread(target=_run_concurrent_caller, daemon=True)
            worker_result: list[Any] = []
            worker_errors: list[BaseException] = []

            def _run_worker() -> None:
                try:
                    worker_result.append(
                        client.chat.completions.create(
                            model=MODEL, messages=MESSAGES, stream=False
                        )
                    )
                except BaseException as exc:  # pragma: no cover - captured for assertion
                    worker_errors.append(exc)

            worker_thread = threading.Thread(target=_run_worker, daemon=True)
            worker_thread.start()
            self.assertTrue(in_retry_sync.wait(timeout=5.0))
            caller.start()
            caller.join(timeout=5.0)
            self.assertFalse(caller.is_alive(), "Concurrent oneshot caller hung")
            release_retry.set()
            worker_thread.join(timeout=5.0)
            self.assertFalse(worker_thread.is_alive(), "Worker retry thread hung")

        self.assertEqual(oneshot_errors, [])
        self.assertEqual(worker_errors, [])
        self.assertEqual(oneshot_result, ["oneshot answer"])
        self.assertEqual(worker_result[0].choices[0].message.content, "retry worker answer")
        self.assertIs(client._worker_proc, proc2)
        self.assertFalse(client._worker_lock.locked())

    def test_auth_contention_retries_at_most_once_and_releases_lock(self) -> None:
        client = self._client()
        proc1 = _make_proc(
            [""],
            alive=False,
            exit_code=0,
            stderr_tail=AUTH_CONTENTION_STDERR,
        )
        proc2 = _make_proc(
            [""],
            alive=False,
            exit_code=1,
            stderr_tail=AUTH_CONTENTION_STDERR,
        )

        with patch.object(
            client_module.subprocess, "Popen", side_effect=[proc1, proc2]
        ) as popen:
            with self.assertRaisesRegex(RuntimeError, "not logged into Antigravity"):
                client.chat.completions.create(
                    model=MODEL, messages=MESSAGES, stream=False
                )

        self.assertEqual(popen.call_count, 2)
        self.assertIsNone(client._worker_proc)
        self.assertFalse(client._worker_lock.locked())
        self.assertTrue(client._worker_lock.acquire(blocking=False))
        client._worker_lock.release()

    def test_media_permission_error_releases_lock_without_terminating_worker(self) -> None:
        client = self._client()
        live_proc = MagicMock(name="live_worker")
        live_proc.poll.return_value = None
        client._worker_proc = live_proc
        client._worker_conversation_id = "conv-healthy"
        client._active_processes.add(live_proc)

        with (
            patch.object(
                client,
                "_ensure_worker_media_permissions",
                side_effect=OSError("permission settings write failed"),
            ),
            patch.object(
                client, "_terminate_worker", wraps=client._terminate_worker
            ) as terminate_spy,
        ):
            with self.assertRaisesRegex(OSError, "permission settings write failed"):
                client._create_chat_completion(
                    model=MODEL, messages=MESSAGES, stream=False
                )

        terminate_spy.assert_not_called()
        live_proc.terminate.assert_not_called()
        self.assertIs(client._worker_proc, live_proc)
        self.assertEqual(client._worker_conversation_id, "conv-healthy")
        self.assertFalse(client._worker_lock.locked())
        self.assertTrue(client._worker_lock.acquire(blocking=False))
        client._worker_lock.release()


if __name__ == "__main__":
    unittest.main()
