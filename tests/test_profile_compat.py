"""Compatibility tests for optional ProviderProfile fields across Hermes versions.

Older Hermes hosts omit newer dataclass fields such as ``supports_streaming``,
``models_dev_id``, ``model_capabilities``, and ``unsupported_response_formats``
from ``providers.base.ProviderProfile``. Passing any missing field directly to
``AntigravitySubscriptionDirectSDKProfile(...)`` raises ``TypeError`` at plugin
import time.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

PLUGIN_DIR = Path(__file__).resolve().parent.parent
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

PROFILE_NAME = "antigravity-subscription-directsdk"


@dataclass
class _MinimalProviderProfile:
    """Core ProviderProfile fields present on older Hermes hosts."""

    name: str
    aliases: tuple = ()
    display_name: str = ""
    description: str = ""
    base_url: str = ""
    api_mode: str = "chat_completions"
    auth_type: str = "api_key"
    process_command: str = ""
    process_args: tuple = ()
    process_command_env_vars: tuple = ()
    process_args_env_var: str = ""
    default_aux_model: str = ""
    fallback_models: tuple = ()
    supports_vision: bool = False
    classify_api_error: Callable[..., Any] | None = None


@dataclass
class _FullProviderProfile(_MinimalProviderProfile):
    """Modern ProviderProfile including all optional fields the plugin opts into."""

    supports_streaming: bool | None = field(default=None, kw_only=True)
    models_dev_id: str = ""
    model_capabilities: dict[str, dict[str, Any]] = field(default_factory=dict)
    unsupported_response_formats: tuple = ()


@contextmanager
def _loaded_plugin_with_profile(profile_cls: type, module_name: str):
    """Load __init__.py against a synthetic providers.base.ProviderProfile, then restore."""
    import providers
    import providers.base

    real_base = sys.modules["providers.base"]
    had_base_attr = hasattr(providers, "base")
    real_base_attr = getattr(providers, "base", None)
    real_profile = providers.get_provider_profile(PROFILE_NAME)

    fake_base = ModuleType("providers.base")
    fake_base.ProviderProfile = profile_cls

    spec = importlib.util.spec_from_file_location(
        module_name,
        PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {PLUGIN_DIR / '__init__.py'}")
    module = importlib.util.module_from_spec(spec)

    sys.modules["providers.base"] = fake_base
    providers.base = fake_base
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        yield module, providers.get_provider_profile(PROFILE_NAME)
    finally:
        sys.modules["providers.base"] = real_base
        if had_base_attr:
            providers.base = real_base_attr
        elif hasattr(providers, "base"):
            delattr(providers, "base")

        for key in [
            name for name in sys.modules
            if name == module_name or name.startswith(f"{module_name}.")
        ]:
            sys.modules.pop(key, None)

        if real_profile is not None:
            providers.register_provider(real_profile)
        else:
            providers._REGISTRY.pop(PROFILE_NAME, None)
            for alias in [
                key for key, target in list(providers._ALIASES.items())
                if target == PROFILE_NAME
            ]:
                providers._ALIASES.pop(alias, None)
            providers._SOURCES.pop(PROFILE_NAME, None)
            providers._PROVIDER_LIST_CACHE = None


class ProviderProfileCompatTests(unittest.TestCase):
    def test_loads_and_registers_with_minimal_provider_profile(self):
        with _loaded_plugin_with_profile(
            _MinimalProviderProfile, "antigravity_plugin_compat_minimal"
        ) as (module, registered):
            profile = module.antigravity_profile
            self.assertIsInstance(profile, _MinimalProviderProfile)
            self.assertIs(registered, profile)
            self.assertFalse(hasattr(profile, "unsupported_response_formats"))
            self.assertNotIn("unsupported_response_formats", vars(profile))
            self.assertNotIn("supports_streaming", vars(profile))
            self.assertNotIn("models_dev_id", vars(profile))
            self.assertNotIn("model_capabilities", vars(profile))

    def test_sets_optional_fields_when_declared_on_provider_profile(self):
        with _loaded_plugin_with_profile(
            _FullProviderProfile, "antigravity_plugin_compat_full"
        ) as (module, registered):
            profile = module.antigravity_profile
            self.assertIsInstance(profile, _FullProviderProfile)
            self.assertIs(registered, profile)
            self.assertEqual(profile.unsupported_response_formats, ("json_schema",))
            self.assertIs(profile.supports_streaming, True)
            self.assertEqual(profile.models_dev_id, "google")
            self.assertIsInstance(
                profile.model_capabilities, module._LazyModelCapabilities
            )

    def test_restores_real_provider_base_and_registered_profile(self):
        import providers

        real_base = sys.modules["providers.base"]
        before_profile = providers.get_provider_profile(PROFILE_NAME)

        with _loaded_plugin_with_profile(
            _MinimalProviderProfile, "antigravity_plugin_compat_restore"
        ):
            pass

        self.assertIs(sys.modules["providers.base"], real_base)
        self.assertIs(providers.base, real_base)
        self.assertNotIn("antigravity_plugin_compat_restore", sys.modules)
        self.assertIs(providers.get_provider_profile(PROFILE_NAME), before_profile)


if __name__ == "__main__":
    unittest.main()
