"""Maße von Hand, Analyse des gezeigten Turniers und die Verdikt-Zweige."""

import numpy as np
import pytest

import nti_constants as C
import nti_evaluation as ev
import nti_models as M
from conftest import counts
from nti_scenario import Settings


def test_german_number_formatting():
    assert ev.de(1.5, 1) == "1,5" and ev.de(0.666) == "0,67" and ev.pct(0.256) == "26" and ev.pct(0.256, 1) == "25,6"


def test_kendall_tau_values_and_ties():
    assert ev.kendall_tau([3, 2, 1], [30, 20, 10]) == 1.0
    assert ev.kendall_tau([1, 2, 3], [30, 20, 10]) == -1.0
    assert ev.kendall_tau([1, 1, 2], [1, 2, 3]) == pytest.approx(1.0)           # Gleichstände zählen nicht


def test_place_of_counts_better_values_only():
    v = [0.3, 0.9, 0.5, 0.9]
    assert ev.place_of(v, 1) == 1 and ev.place_of(v, 3) == 1 and ev.place_of(v, 2) == 3 and ev.place_of(v, 0) == 4


def test_mean_skips_nan():
    assert ev._mean([1.0, float("nan"), 3.0]) == 2.0 and np.isnan(ev._mean([float("nan")]))


def test_analysis_matches_direct_model_calls():
    s = Settings(n=6, games=8, style="konter", c=0.4, seed=3)
    a = ev.analyse(s)
    t = a.tournament
    Lbt = M.logit_matrix(M.fit_bt(t.wins, t.games))
    assert a.loss["Bradley-Terry"] == pytest.approx(M.log_loss(Lbt, t.P))
    assert a.bound == pytest.approx(M.entropy_bound(t.P)) and a.loss["BT + Zyklus"] > a.bound - 1e-9
    assert a.best == int(np.argmax(t.strengths)) and a.place["Tabelle"] == ev.place_of(M.win_table_rate(t.wins, t.games), a.best)
    assert set(a.rate) == {"Tabelle", *ev.MODELS} and 0.0 <= a.share_data <= 1.0 and 0.0 <= a.triples_data <= 1.0


def test_analysis_of_a_transitive_tournament_keeps_the_best_on_top():
    a = ev.analyse(Settings(n=8, games=32, c=0.0, seed=2))
    assert a.place["Bradley-Terry"] == 1 and a.share_true == pytest.approx(0.0, abs=1e-9) and a.tau_s["Bradley-Terry"] > 0.85


def _fake(share_data, base, loss=None):
    t = ev.make_tournament(Settings(n=6, games=2, seed=1))
    loss = loss or {"Bradley-Terry": 0.6, "BT + Zyklus": 0.5, "gesättigt": 0.55}
    place = {"Tabelle": 2, "Bradley-Terry": 1, "BT + Zyklus": 1, "gesättigt": 2}
    z = {k: 0.0 for k in place}
    return ev.Analysis(Settings(n=6, games=2), t, {}, loss, 0.4, z, 0, place, z, z, 0.02, share_data, 0.05, {"share": base, "triples": 0.03})


def test_verdict_names_the_best_model_and_places():
    level, text = ev.verdict_analysis(_fake(0.5, 0.2))
    assert level == "info" and "am besten **BT + Zyklus**" in text and "Platz **2**" in text and "Platz 1" in text and "Untergrenze 0,400" in text


def test_verdict_flags_data_that_do_not_exceed_the_noise_baseline():
    assert "nicht vom Rauschen" in ev.verdict_analysis(_fake(0.30, 0.30))[1]
    assert "nicht vom Rauschen" in ev.verdict_analysis(_fake(0.30 * 1.15 + 0.009, 0.30))[1]
    assert "nicht vom Rauschen" not in ev.verdict_analysis(_fake(0.30 * 1.15 + 0.011, 0.30))[1]


def test_verdict_picks_the_model_with_the_smallest_loss():
    assert "am besten **Bradley-Terry**" in ev.verdict_analysis(_fake(0.5, 0.2, {"Bradley-Terry": 0.3, "BT + Zyklus": 0.5, "gesättigt": 0.4}))[1]
    assert "am besten **gesättigt**" in ev.verdict_analysis(_fake(0.5, 0.2, {"Bradley-Terry": 0.5, "BT + Zyklus": 0.5, "gesättigt": 0.4}))[1]


def test_log_loss_of_the_table_relation_is_not_part_of_the_models():
    assert "Tabelle" not in ev.MODELS and set(ev.MODELS) == set(C.COLORS) - {"Tabelle"}


def test_counts_helper_builds_symmetric_games():
    W, G = counts(3, {(0, 1): 2, (1, 0): 3})
    assert G[0, 1] == 5 and G[1, 0] == 5 and G[0, 2] == 0
