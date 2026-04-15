from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "experiments" / "run_chunking_variant_experiment.py"
)
SPEC = importlib.util.spec_from_file_location("run_chunking_variant_experiment", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

resolve_variant_features = MODULE._resolve_variant_features
infer_language_hint = MODULE._infer_language_hint_from_text


def test_new_variant_aliases_resolve_to_expected_features() -> None:
    assert resolve_variant_features("idea6_coverage_first_repair") == {
        "dynamic_lexicon",
        "coverage_first_repair",
    }
    assert resolve_variant_features("idea7_tail_rescue_backward_pass") == {
        "dynamic_lexicon",
        "tail_rescue_backward_pass",
    }
    assert resolve_variant_features("idea8_language_aware_translator_mode") == {
        "dynamic_lexicon",
        "language_aware_translator_mode",
    }


def test_language_hint_detection_handles_mixed_text() -> None:
    text = "hola gracias por llamar and please verify the member id today gracias"
    hint = infer_language_hint(text, "english")
    assert hint in {"mixed (english + spanish)", "english"}
