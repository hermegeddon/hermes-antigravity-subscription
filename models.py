"""Model catalog mapping and reasoning effort resolution for Antigravity."""

from __future__ import annotations

import contextlib
import logging
import re
import subprocess
import tempfile
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

# Efforts that `agy --effort` accepts, per base model. A model that agy lists
# as `<base>-low|medium|high` takes --effort; a model it lists by bare name
# does not (agy 1.2.x rejects --effort for those, and rejects a missing
# --effort for the others).
#
# `agy models` is the source of truth. This table only lets known models
# resolve without spawning a process, and covers the case where `agy models`
# fails. An empty tuple means "bare name, no --effort".
_KNOWN_EFFORTS: dict[str, tuple[str, ...]] = {
    "gemini-3.8-flash": ("low", "medium", "high"),
    "gemini-3.7-flash": ("low", "medium", "high"),
    "gemini-3.6-flash": ("low", "medium", "high"),
    "gemini-3.1-pro": ("low", "high"),
    "claude-sonnet-4-6": (),
    "claude-opus-4-6-thinking": (),
    "gpt-oss-120b": ("medium",),
    "gemini-4-neon": ("low", "medium", "high"),
    "gemini-4-carbon": ("low", "medium", "high"),
    "gemini-4-argon": ("low", "medium", "high"),
    "gemini-4-hydrogen": ("low", "medium", "high"),
    "gemini-pro-orchestrator": (),
    "gemini-pro-orchestrator-neon": (),
    "gemini-pro-orchestrator-argon": (),
}

_FALLBACK_MODELS = list(_KNOWN_EFFORTS)

_MODEL_ALIASES = {
    "default": "gemini-3.8-flash",
    "flash": "gemini-3.8-flash",
    "gemini-flash": "gemini-3.8-flash",
    "gemini-3.8": "gemini-3.8-flash",
    "pro": "gemini-3.1-pro",
    "gemini-pro": "gemini-3.1-pro",
    "gemini-3.1": "gemini-3.1-pro",
    "orchestrator": "gemini-pro-orchestrator",
    "gemini-orchestrator": "gemini-pro-orchestrator",
    "gemini-4": "gemini-4-neon",
    "neon": "gemini-4-neon",
    "hydrogen": "gemini-4-hydrogen",
    "argon": "gemini-4-argon",
    "carbon": "gemini-4-carbon",
    "sonnet": "claude-sonnet-4-6",
    "claude-sonnet": "claude-sonnet-4-6",
    "claude-sonnet-5-5": "claude-sonnet-4-6",
    "opus": "claude-opus-4-6-thinking",
    "claude-opus": "claude-opus-4-6-thinking",
    "claude-opus-5-5": "claude-opus-4-6-thinking",
}

# Effort suffixes agy appends to a base model id, or that callers use as
# effort aliases. `-xhigh` (issue #47 duplicate listing on several models)
# and `-max` normalize into the base model's `high` variant, matching
# `_normalize_effort`.
_EFFORT_SUFFIXES = (
    ("-xhigh", "high"),
    ("-max", "high"),
    ("-high", "high"),
    ("-medium", "medium"),
    ("-low", "low"),
)
_EFFORT_RANK = {"low": 0, "medium": 1, "high": 2}

_SPINNER_CHARS = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
_SKIP_LINE_PREFIXES = ("error:", "usage:", "available models:", "warning:", "--")

_CATALOG_TTL_SECONDS = 3600.0
_CATALOG_FAILURE_TTL_SECONDS = 60.0
_catalog_lock = threading.Lock()
_catalog_cache: tuple[float, dict[str, tuple[str, ...]]] | None = None


def _normalize_effort(effort: str | None) -> str | None:
    if not effort:
        return None
    e = str(effort).strip().lower()
    if e in ("none", "off", "minimal"):
        return "low"
    if e in ("xhigh", "max", "ultra"):
        return "high"
    if e in ("low", "medium", "high"):
        return e
    return "medium"


def split_effort_suffix(model: str) -> tuple[str, str | None]:
    """Split `gemini-3.8-flash-high` into (`gemini-3.8-flash`, `high`).

    A trailing `-xhigh` splits as an effort alias for `high` (issue #47):
    agy lists `gemini-4-hydrogen-xhigh` as a lesser duplicate of the real
    `gemini-4-hydrogen-{low,medium,high}` variants, and it must group into
    the base model instead of surfacing as a separate bare-name model.
    """
    for suffix, effort in _EFFORT_SUFFIXES:
        if model.endswith(suffix):
            return model[: -len(suffix)], effort
    return model, None


def parse_catalog(text: str) -> dict[str, tuple[str, ...]]:
    """Group `agy models` output as {base model: supported efforts}.

    `gemini-3.8-flash-{low,medium,high}` becomes one `gemini-3.8-flash` entry
    with three efforts; an id with no effort suffix maps to an empty tuple.
    A trailing `-xhigh` groups into its base as the `high` effort (issue #47),
    so `gemini-4-hydrogen-xhigh` never surfaces as a separate bare-name model.
    Insertion order follows the order agy printed the models in.
    """
    grouped: dict[str, list[str]] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or "fetching" in line.lower() or line.startswith(_SPINNER_CHARS):
            continue
        lower_line = line.lower()
        if any(lower_line.startswith(prefix) for prefix in _SKIP_LINE_PREFIXES):
            continue
        model_id = line.split()[0]
        if not re.match(r"^[a-zA-Z0-9_\-\.]+$", model_id):
            continue
        base, effort = split_effort_suffix(model_id)
        efforts = grouped.setdefault(base, [])
        if effort and effort not in efforts:
            efforts.append(effort)
    return {
        base: tuple(sorted(efforts, key=_EFFORT_RANK.__getitem__))
        for base, efforts in grouped.items()
    }


def load_catalog(
    timeout: float = 15.0, *, only_if_stale: bool = False
) -> dict[str, tuple[str, ...]]:
    """Run `agy models`, cache the parsed result, and return it.

    Returns an empty dict when agy is missing, slow, or prints nothing usable.
    Failures are cached briefly so an absent agy does not cost a process
    spawn per request. With `only_if_stale`, a thread that waited on the lock
    reuses the result the previous holder just cached instead of spawning agy
    again.
    """
    global _catalog_cache

    with _catalog_lock:
        if only_if_stale:
            fresh = _cached_catalog()
            if fresh is not None:
                return fresh

        # Imported per-call to avoid a circular top-level import with client.py
        # and to preserve testability when process.* attributes are patched.
        try:
            from .client import _force_rmtree
            from .process import (
                _no_window,
                build_child_env,
                reap_orphaned_plugin_workers,
                resolve_agy_command,
                setup_isolated_home,
            )
        except ImportError:
            from client import _force_rmtree
            from process import (
                _no_window,
                build_child_env,
                reap_orphaned_plugin_workers,
                resolve_agy_command,
                setup_isolated_home,
            )

        tmp_dir: str | None = None
        timed_out = False
        catalog: dict[str, tuple[str, ...]] = {}
        try:
            tmp_dir = tempfile.mkdtemp(prefix="hermes_agy_models_")
            isolated_home, _ = setup_isolated_home(tmp_dir)
            env = build_child_env(isolated_home)
            res = subprocess.run(
                [resolve_agy_command(), "models"],
                env=env,
                cwd=tmp_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                **_no_window(),
            )
            catalog = parse_catalog(res.stdout or "")
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            logger.debug("Antigravity model catalog timed out after %.1fs: %s", timeout, exc)
            catalog = {}
        except Exception as exc:
            logger.debug("Antigravity model catalog unavailable: %s", exc)
            catalog = {}
        finally:
            if tmp_dir is not None:
                _force_rmtree(tmp_dir)
            if timed_out:
                # Delete tmp_dir first so the reaper identifies any surviving child as orphaned.
                with contextlib.suppress(Exception):
                    reap_orphaned_plugin_workers()

        ttl = _CATALOG_TTL_SECONDS if catalog else _CATALOG_FAILURE_TTL_SECONDS
        _catalog_cache = (time.monotonic() + ttl, catalog)
        return catalog


def _cached_catalog() -> dict[str, tuple[str, ...]] | None:
    cache = _catalog_cache
    if cache is not None and time.monotonic() < cache[0]:
        return cache[1]
    return None


def model_efforts(base_model: str, *, refresh: bool = True) -> tuple[str, ...] | None:
    """Efforts agy accepts for `base_model`; () means bare name, None means unknown.

    Order: a fresh `agy models` result, then the built-in table, then (when
    `refresh` is true) a new `agy models` call. While a cached result is
    fresh, a model it does not list is not looked up again until the cache
    expires, so an unknown name costs no process spawn per request.
    """
    cached = _cached_catalog()
    if cached is not None and base_model in cached:
        return cached[base_model]
    if base_model in _KNOWN_EFFORTS:
        return _KNOWN_EFFORTS[base_model]
    if refresh and cached is None:
        return load_catalog(only_if_stale=True).get(base_model)
    return None


def _heuristic_efforts(base_model: str, has_suffix: bool) -> tuple[str, ...]:
    """Efforts for a model that neither agy nor the built-in table lists."""
    name = base_model.lower()
    # Orchestrators are bare-name models despite containing "pro"; check first.
    if "orchestrator" in name:
        return ("low", "medium", "high") if has_suffix else ()
    if name.startswith("gemini-4-"):
        return ("low", "medium", "high")
    if name.startswith("gemini-") and "pro" in name:
        return ("low", "high")
    if name.startswith("gemini-") and "flash" in name:
        return ("low", "medium", "high")
    # A suffixed id is an effort variant; a bare unknown id takes no --effort.
    return ("low", "medium", "high") if has_suffix else ()


def _nearest_effort(requested: str, supported: tuple[str, ...]) -> str:
    """Closest supported effort; on a tie the stronger one wins."""
    target = _EFFORT_RANK[requested]
    return min(
        supported,
        key=lambda e: (abs(_EFFORT_RANK[e] - target), -_EFFORT_RANK[e]),
    )


def _apply_suffix_if_supported(
    target: str,
    suffix: str | None,
    catalog: dict[str, tuple[str, ...]],
) -> str:
    """Append `-<suffix>` only when `catalog` defines effort levels for `target`."""
    if suffix and catalog.get(target):
        return f"{target}-{suffix}"
    return target


def _resolve_catalog_family(
    lookup_term: str,
    catalog: dict[str, tuple[str, ...]],
) -> str | None:
    """Match a family alias against `catalog`, returning the base model id or None."""
    if lookup_term in ("pro", "gemini-pro", "gemini-3.1"):
        non_orch_pro = [
            k for k in catalog
            if "pro" in k.lower() and "orchestrator" not in k.lower()
        ]
        if non_orch_pro:
            return sorted(non_orch_pro, reverse=True)[0]
        # agy 1.5+ replaced gemini-3.1-pro with gemini-4-* while adding bare
        # gemini-pro-orchestrator*; preserve `agy models` order for gemini-4-*.
        gemini4_matches = [k for k in catalog if k.lower().startswith("gemini-4-")]
        if gemini4_matches:
            return gemini4_matches[0]
        pro_matches = [k for k in catalog if "pro" in k.lower()]
        if pro_matches:
            return sorted(pro_matches, reverse=True)[0]
        return None

    if lookup_term in ("orchestrator", "gemini-orchestrator"):
        if "gemini-pro-orchestrator" in catalog:
            return "gemini-pro-orchestrator"
        orch_matches = [k for k in catalog if "orchestrator" in k.lower()]
        return orch_matches[0] if orch_matches else None

    family_keywords = {
        ("sonnet", "claude-sonnet"): "sonnet",
        ("opus", "claude-opus"): "opus",
        ("flash", "gemini-flash", "gemini-3.8"): "flash",
        ("gpt", "gpt-oss", "oss"): "gpt",
    }
    for triggers, kw in family_keywords.items():
        if lookup_term in triggers:
            matches = [k for k in catalog if kw in k.lower()]
            if matches:
                # Sort descending so newer versions win (5-5 before 4-6, 3.8 before 3.7).
                return sorted(matches, reverse=True)[0]
    return None


def resolve_model_alias(
    model: str, catalog: dict[str, tuple[str, ...]] | None = None
) -> str:
    """Resolve user alias to concrete base model, prioritizing the live agy catalog."""
    m = (model or "gemini-3.8-flash").strip()
    lower_m = m.lower()

    if catalog:
        if lower_m in catalog:
            return lower_m
        base, suffix = split_effort_suffix(lower_m)
        if base in catalog:
            return _apply_suffix_if_supported(base, suffix, catalog)

        lookup_term = "flash" if base == "default" else base
        family_match = _resolve_catalog_family(lookup_term, catalog)
        if family_match is not None:
            return _apply_suffix_if_supported(family_match, suffix, catalog)

        # Check explicit aliases before substring matching so codenames like
        # `neon` and `argon` map to `gemini-4-*` instead of `gemini-pro-orchestrator-*`.
        if lookup_term in _MODEL_ALIASES:
            aliased = _MODEL_ALIASES[lookup_term]
            if aliased in catalog:
                return _apply_suffix_if_supported(aliased, suffix, catalog)

        matches = [k for k in catalog if lookup_term in k.lower()]
        if matches:
            best = sorted(matches, reverse=True)[0]
            return _apply_suffix_if_supported(best, suffix, catalog)

    base, suffix = split_effort_suffix(lower_m)
    fallback = _MODEL_ALIASES.get(base, base)
    if fallback in _KNOWN_EFFORTS:
        return _apply_suffix_if_supported(fallback, suffix, _KNOWN_EFFORTS)
    if suffix and fallback != base:
        return f"{fallback}-{suffix}"
    return _MODEL_ALIASES.get(lower_m, m)


def resolve_model_and_effort(
    model: str | None,
    reasoning_effort: str | None = None,
) -> tuple[str, str | None]:
    """Map user/hermes model request to concrete CLI model ID and effort level.

    Returns (model id, effort). The effort is None for models that agy selects
    by bare name; the client then omits --effort.
    """
    raw_model = str(model or "gemini-3.8-flash").strip()
    catalog = _cached_catalog()
    if catalog is None:
        catalog = load_catalog(only_if_stale=True)

    m = resolve_model_alias(raw_model, catalog)

    base_model, suffix_effort = split_effort_suffix(m)
    supported = model_efforts(base_model, refresh=False)
    if supported is None:
        supported = _heuristic_efforts(base_model, suffix_effort is not None)
    if not supported:
        return base_model, None

    # Resolve effort: explicit argument > model suffix > config setting > default
    effort = _normalize_effort(reasoning_effort)
    if not effort and suffix_effort:
        effort = suffix_effort
    if not effort:
        try:
            from hermes_cli.config import load_config_readonly
            cfg_effort = load_config_readonly().get("agent", {}).get("reasoning_effort")
            effort = _normalize_effort(cfg_effort)
        except Exception:
            pass
    if not effort:
        effort = "medium"

    concrete_effort = _nearest_effort(effort, supported)
    return f"{base_model}-{concrete_effort}", concrete_effort
