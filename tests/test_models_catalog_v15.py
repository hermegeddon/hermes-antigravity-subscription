"""Tests for agy 1.5+ model catalog parsing, aliases, and effort resolution."""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

plugin_dir = Path(__file__).resolve().parent.parent
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

import models  # noqa: E402
from providers import get_provider_profile  # noqa: E402

AGY_15_MODELS_OUTPUT = (
    "Fetching available models...\n"
    "gemini-4-neon-high\tGemini 4 Neon (High)\n"
    "gemini-4-neon-xhigh\tGemini 4 Neon (XHigh)\n"
    "gemini-4-neon-medium\tGemini 4 Neon (Medium)\n"
    "gemini-4-neon-low\tGemini 4 Neon (Low)\n"
    "gemini-4-carbon-medium\tGemini 4 Carbon (Medium)\n"
    "gemini-4-carbon-high\tGemini 4 Carbon (High)\n"
    "gemini-4-carbon-low\tGemini 4 Carbon (Low)\n"
    "gemini-4-argon-medium\tGemini 4 Argon (Medium)\n"
    "gemini-4-argon-high\tGemini 4 Argon (High)\n"
    "gemini-4-argon-low\tGemini 4 Argon (Low)\n"
    "gemini-4-hydrogen-high\tGemini 4 Hydrogen (High)\n"
    "gemini-4-hydrogen-xhigh\tGemini 4 Hydrogen (XHigh)\n"
    "gemini-4-hydrogen-medium\tGemini 4 Hydrogen (Medium)\n"
    "gemini-4-hydrogen-low\tGemini 4 Hydrogen (Low)\n"
    "gemini-pro-orchestrator-neon\tOrchestrator (Neon)\n"
    "gemini-pro-orchestrator\tOrchestrator (Carbon)\n"
    "gemini-pro-orchestrator-argon\tOrchestrator (Argon)\n"
    "gemini-3.8-flash-high\tGemini 3.8 Flash (High)\n"
    "gemini-3.8-flash-medium\tGemini 3.8 Flash (Medium)\n"
    "gemini-3.8-flash-low\tGemini 3.8 Flash (Low)\n"
    "claude-opus-5-5-low\tClaude Opus 5.5 (Low)\n"
    "claude-opus-5-5-medium\tClaude Opus 5.5 (Medium)\n"
    "claude-opus-5-5-high\tClaude Opus 5.5 (High)\n"
    "claude-opus-5-5-max\tClaude Opus 5.5 (Max)\n"
    "claude-sonnet-5-5-low\tClaude Sonnet 5.5 (Low)\n"
    "claude-sonnet-5-5-medium\tClaude Sonnet 5.5 (Medium)\n"
    "claude-sonnet-5-5-high\tClaude Sonnet 5.5 (High)\n"
    "gpt-oss-120b-medium\tGPT-OSS 120B (Medium)\n"
)


class ModelsCatalogV15Tests(unittest.TestCase):
    def setUp(self):
        self.catalog = models.parse_catalog(AGY_15_MODELS_OUTPUT)
        self._prev_cache = models._catalog_cache
        models._catalog_cache = None
        self.addCleanup(lambda: setattr(models, "_catalog_cache", self._prev_cache))

    def _pin_catalog(self, catalog: dict[str, tuple[str, ...]]):
        return patch("models._catalog_cache", (time.monotonic() + 60.0, catalog))

    def test_parse_catalog_preserves_selective_xhigh_and_keeps_orchestrators_bare(self):
        self.assertEqual(
            self.catalog,
            {
                "gemini-4-neon": ("low", "medium", "high", "xhigh"),
                "gemini-4-carbon": ("low", "medium", "high"),
                "gemini-4-argon": ("low", "medium", "high"),
                "gemini-4-hydrogen": ("low", "medium", "high", "xhigh"),
                "gemini-pro-orchestrator-neon": (),
                "gemini-pro-orchestrator": (),
                "gemini-pro-orchestrator-argon": (),
                "gemini-3.8-flash": ("low", "medium", "high"),
                "claude-opus-5-5": ("low", "medium", "high"),
                "claude-sonnet-5-5": ("low", "medium", "high"),
                "gpt-oss-120b": ("medium",),
            },
        )
        for ghost in ("gemini-4-neon-xhigh", "gemini-4-hydrogen-xhigh", "claude-opus-5-5-max"):
            self.assertNotIn(ghost, self.catalog)

    def test_pro_aliases_resolve_to_gemini4_not_orchestrator(self):
        for alias in ("pro", "gemini-pro", "gemini-3.1"):
            self.assertEqual(models.resolve_model_alias(alias, self.catalog), "gemini-4-neon")
            self.assertEqual(models.resolve_model_alias(f"{alias}-high", self.catalog), "gemini-4-neon-high")

        with self._pin_catalog(self.catalog):
            self.assertEqual(
                models.resolve_model_and_effort("pro", "high"),
                ("gemini-4-neon-high", "high"),
            )

    def test_pro_alias_preserves_catalog_order_for_gemini4_and_prefers_31_pro(self):
        reordered = {
            "gemini-4-hydrogen": ("low", "medium", "high"),
            "gemini-4-neon": ("low", "medium", "high"),
            "gemini-pro-orchestrator": (),
        }
        self.assertEqual(models.resolve_model_alias("pro", reordered), "gemini-4-hydrogen")

        old_catalog = {
            "gemini-3.8-flash": ("low", "medium", "high"),
            "gemini-3.1-pro": ("low", "high"),
            "gemini-4-neon": ("low", "medium", "high"),
            "gemini-pro-orchestrator": (),
        }
        self.assertEqual(models.resolve_model_alias("pro", old_catalog), "gemini-3.1-pro")
        self.assertEqual(models.resolve_model_alias("pro-high", old_catalog), "gemini-3.1-pro-high")

        orch_only = {"gemini-pro-orchestrator": ()}
        self.assertEqual(models.resolve_model_alias("pro", orch_only), "gemini-pro-orchestrator")
        self.assertEqual(models.resolve_model_alias("pro-high", orch_only), "gemini-pro-orchestrator")

    def test_orchestrators_never_get_effort_flag_or_suffix(self):
        self.assertEqual(models.resolve_model_alias("orchestrator", self.catalog), "gemini-pro-orchestrator")
        self.assertEqual(models.resolve_model_alias("gemini-orchestrator", self.catalog), "gemini-pro-orchestrator")
        self.assertEqual(models.resolve_model_alias("orchestrator-high", self.catalog), "gemini-pro-orchestrator")
        self.assertEqual(models.resolve_model_alias("gemini-pro-orchestrator-high", self.catalog), "gemini-pro-orchestrator")

        no_carbon_orch = {
            "gemini-pro-orchestrator-neon": (),
            "gemini-pro-orchestrator-argon": (),
        }
        self.assertEqual(models.resolve_model_alias("orchestrator", no_carbon_orch), "gemini-pro-orchestrator-neon")

        with self._pin_catalog(self.catalog):
            self.assertEqual(
                models.resolve_model_and_effort("orchestrator", "high"),
                ("gemini-pro-orchestrator", None),
            )
            self.assertEqual(
                models.resolve_model_and_effort("gemini-pro-orchestrator", "high"),
                ("gemini-pro-orchestrator", None),
            )
            self.assertEqual(
                models.resolve_model_and_effort("gemini-pro-orchestrator-high", "high"),
                ("gemini-pro-orchestrator", None),
            )
            self.assertEqual(
                models.resolve_model_and_effort("gemini-pro-orchestrator-neon", "low"),
                ("gemini-pro-orchestrator-neon", None),
            )
            self.assertEqual(
                models.resolve_model_and_effort("gemini-pro-orchestrator-argon", "medium"),
                ("gemini-pro-orchestrator-argon", None),
            )

    def test_gemini4_codename_aliases_beat_orchestrator_substring_matches(self):
        self.assertEqual(models.resolve_model_alias("gemini-4", self.catalog), "gemini-4-neon")
        self.assertEqual(models.resolve_model_alias("neon", self.catalog), "gemini-4-neon")
        self.assertEqual(models.resolve_model_alias("argon", self.catalog), "gemini-4-argon")
        self.assertEqual(models.resolve_model_alias("carbon", self.catalog), "gemini-4-carbon")
        self.assertEqual(models.resolve_model_alias("hydrogen", self.catalog), "gemini-4-hydrogen")
        self.assertEqual(models.resolve_model_alias("neon-high", self.catalog), "gemini-4-neon-high")
        self.assertEqual(models.resolve_model_alias("argon-low", self.catalog), "gemini-4-argon-low")

    def test_bare_catalog_entries_never_receive_effort_suffix_across_families(self):
        bare_catalog = {
            "claude-sonnet-4-6": (),
            "claude-opus-4-6-thinking": (),
            "gemini-3.8-flash-custom": (),
            "gpt-oss-120b": (),
            "custom-bare": (),
        }
        for query, expected in (
            ("sonnet-high", "claude-sonnet-4-6"),
            ("opus-max", "claude-opus-4-6-thinking"),
            ("flash-low", "gemini-3.8-flash-custom"),
            ("gpt-high", "gpt-oss-120b"),
            ("custom-bare-high", "custom-bare"),
        ):
            self.assertEqual(models.resolve_model_alias(query, bare_catalog), expected)

    def test_cold_cache_heuristics_and_fallbacks(self):
        self.assertEqual(models._heuristic_efforts("gemini-pro-orchestrator-future", False), ())
        self.assertEqual(models._heuristic_efforts("gemini-pro-orchestrator-future", True), ("low", "medium", "high"))
        self.assertEqual(models._heuristic_efforts("gemini-4-future", False), ("low", "medium", "high"))

        with patch("models._catalog_cache", None), patch("models.load_catalog", return_value={}):
            self.assertEqual(
                models.resolve_model_and_effort("gemini-pro-orchestrator-future", "high"),
                ("gemini-pro-orchestrator-future", None),
            )
            self.assertEqual(
                models.resolve_model_and_effort("gemini-4-future", "high"),
                ("gemini-4-future-high", "high"),
            )
            self.assertEqual(
                models.resolve_model_and_effort("orchestrator", "high"),
                ("gemini-pro-orchestrator", None),
            )
            self.assertEqual(
                models.resolve_model_and_effort("orchestrator-high", "high"),
                ("gemini-pro-orchestrator", None),
            )
            self.assertEqual(
                models.resolve_model_and_effort("gemini-pro-orchestrator-neon", "low"),
                ("gemini-pro-orchestrator-neon", None),
            )
            self.assertEqual(
                models.resolve_model_and_effort("gemini-4-neon", "medium"),
                ("gemini-4-neon-medium", "medium"),
            )

    def test_selective_native_xhigh_and_clamping(self):
        self.assertEqual(models.split_effort_suffix("claude-opus-5-5-max"), ("claude-opus-5-5", "high"))
        self.assertEqual(models.split_effort_suffix("gemini-4-neon-xhigh"), ("gemini-4-neon", "xhigh"))
        self.assertEqual(models.resolve_model_alias("claude-opus-5-5-max", self.catalog), "claude-opus-5-5-high")
        self.assertEqual(models.resolve_model_alias("gemini-4-neon-xhigh", self.catalog), "gemini-4-neon-xhigh")
        self.assertEqual(models.resolve_model_alias("neon-xhigh", self.catalog), "gemini-4-neon-xhigh")
        self.assertEqual(models.resolve_model_alias("argon-xhigh", self.catalog), "gemini-4-argon-high")
        self.assertEqual(models.resolve_model_alias("opus-max", self.catalog), "claude-opus-5-5-high")

        with self._pin_catalog(self.catalog):
            # Native xhigh models accept xhigh directly (and map max/ultra -> xhigh)
            self.assertEqual(models.resolve_model_and_effort("gemini-4-neon", "xhigh"), ("gemini-4-neon-xhigh", "xhigh"))
            self.assertEqual(models.resolve_model_and_effort("gemini-4-neon-xhigh"), ("gemini-4-neon-xhigh", "xhigh"))
            self.assertEqual(models.resolve_model_and_effort("gemini-4-neon", "max"), ("gemini-4-neon-xhigh", "xhigh"))
            self.assertEqual(models.resolve_model_and_effort("gemini-4-hydrogen", "xhigh"), ("gemini-4-hydrogen-xhigh", "xhigh"))
            # Models limited to low/medium/high clamp xhigh and max down to high
            self.assertEqual(models.resolve_model_and_effort("gemini-4-argon", "xhigh"), ("gemini-4-argon-high", "high"))
            self.assertEqual(models.resolve_model_and_effort("gemini-4-carbon", "xhigh"), ("gemini-4-carbon-high", "high"))
            self.assertEqual(models.resolve_model_and_effort("claude-opus-5-5-max"), ("claude-opus-5-5-high", "high"))
            self.assertEqual(models.resolve_model_and_effort("claude-opus-5-5", "xhigh"), ("claude-opus-5-5-high", "high"))
            self.assertEqual(models.resolve_model_and_effort("claude-opus-5-5", "max"), ("claude-opus-5-5-high", "high"))

    def test_profile_supported_reasoning_efforts_on_v15_catalog(self):
        profile = get_provider_profile("antigravity-subscription-directsdk")
        prov_models = sys.modules.get(f"{type(profile).__module__}.models", models)
        with patch.object(prov_models, "_catalog_cache", (time.monotonic() + 60.0, self.catalog)), \
                patch("models._catalog_cache", (time.monotonic() + 60.0, self.catalog)):
            self.assertEqual(profile.supported_reasoning_efforts("pro"), ("low", "medium", "high", "xhigh"))
            self.assertEqual(profile.supported_reasoning_efforts("orchestrator"), ())
            self.assertEqual(profile.supported_reasoning_efforts("gemini-pro-orchestrator-neon"), ())
            self.assertEqual(profile.supported_reasoning_efforts("gemini-4-neon"), ("low", "medium", "high", "xhigh"))
            self.assertEqual(profile.supported_reasoning_efforts("gemini-4-argon"), ("low", "medium", "high"))
            self.assertEqual(profile.supported_reasoning_efforts("claude-opus-5-5-max"), ("low", "medium", "high"))

        with patch.object(prov_models, "_catalog_cache", None), \
                patch("models._catalog_cache", None):
            self.assertEqual(profile.supported_reasoning_efforts("gemini-pro-orchestrator-future"), ())
            self.assertEqual(profile.supported_reasoning_efforts("gemini-4-future"), ("low", "medium", "high"))


if __name__ == "__main__":
    unittest.main()
