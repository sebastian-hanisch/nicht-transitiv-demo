"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Adressparameter, Szenario und PDF."""

import numpy as np
import pytest

import nti_constants as C
import nti_evaluation as ev
import nti_pdf_export as X
import nti_presets as P
from nti_scenario import Settings, make_tournament

FIELDS = {"n": "n", "delta": "delta", "games": "games", "style": "style", "c": "c", "lam": "lam", "seed": "seed"}


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for name, p in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            if spec.lo is not None:
                assert spec.lo <= p[key] <= spec.hi, (name, key)
            if spec.choices is not None:
                assert p[key] in spec.choices, (name, key)


def test_default_preset_equals_the_default_settings():
    p = C.PRESETS["Ausgewogener Zyklus"]
    assert Settings(**{field: p[key] for key, field in FIELDS.items()}) == Settings()


def test_preset_keys_are_exactly_the_settings_fields():
    assert set(FIELDS.values()) == set(Settings.__dataclass_fields__)


def test_setting_specs_cover_every_preset_key_and_have_unique_parameters():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    params = [spec.url_param for spec in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params))


def test_presets_show_their_mechanism():
    assert C.PRESETS["Transitiv"]["c"] == 0.0
    assert C.PRESETS["Der Beste wird gekontert"]["style"] == "konter" and C.PRESETS["Der Beste wird gekontert"]["c"] > 0
    assert C.PRESETS["Wenige Partien"]["games"] == 1 and C.PRESETS["Ausgewogener Zyklus"]["style"] == "ausgewogen"


def test_sweep_instances_are_fixed_and_distinct():
    assert len(C.SWEEP_SEEDS) == len(set(C.SWEEP_SEEDS)) == 60


def test_tournament_follows_the_settings_and_is_deterministic():
    s = Settings(n=8, delta=0.2, games=4, style="konter", c=0.3, seed=9)
    a, b = make_tournament(s), make_tournament(s)
    assert a.n == 8 and (a.games == 4 * (1 - np.eye(8, dtype=int))).all() and (a.wins == b.wins).all()
    assert (make_tournament(Settings(n=8, seed=9)).wins != make_tournament(Settings(n=8, seed=10)).wins).any()


def test_pdf_is_built_for_every_preset():
    for name in C.PRESETS:
        p = C.PRESETS[name]
        s = Settings(n=6, delta=p["delta"], games=min(p["games"], 4), style=p["style"], c=p["c"], lam=p["lam"], seed=p["seed"])
        data = X.build_pdf(ev.analyse(s))
        assert data[:5] == b"%PDF-" and len(data) > 1500


def test_latin_filter_replaces_greek_and_math_symbols():
    out = X._latin("λ ≈ 0,5; σ → τ µ θ − **fett** „Zitat“")
    assert out == 'lambda ca. 0,5; sigma -> Tau mu theta - fett "Zitat"'
    assert out.encode("latin-1")


@pytest.mark.parametrize("bad", [0, -1])
def test_only_valid_game_counts_are_offered(bad):
    assert bad not in C.GAMES_CHOICES
