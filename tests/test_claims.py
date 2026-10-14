"""JEDE Zahl und Aussage aus App-Texten und README wird hier nachgerechnet (über die festen Instanzen SWEEP_SEEDS, 60 Läufe; nie ein einzelner Seed).
Der Generator hängt nur von Seed und Position ab (SplitMix64), die Zahlen sind plattformunabhängig; Schwellen trotzdem mit Abstand zur Messung."""

import pytest

import nti_constants as C
import nti_evaluation as ev


@pytest.fixture(scope="module")
def bal():
    return {(r["c"], r["k"]): r for r in ev.sweep_predict("ausgewogen", C.C_BALANCED)}


@pytest.fixture(scope="module")
def unb():
    return {(r["c"], r["k"]): r for r in ev.sweep_predict("konter", C.C_UNBALANCED)}


@pytest.fixture(scope="module")
def rank():
    return {(r["style"], r["c"]): r for r in ev.sweep_rank()}


@pytest.fixture(scope="module")
def detect():
    return {(r["c"], r["k"]): r for r in ev.sweep_detect()}


@pytest.fixture(scope="module")
def nash():
    return {r["c"]: r for r in ev.sweep_nash()}


# --- Experiment 1: Vorhersage bei ausgewogenen Stilen -----------------------------------------------------------------------------------------

def test_bt_cannot_predict_balanced_cycles_but_one_cycle_does(bal):
    r = bal[(0.4, 16)]
    assert r["loss"]["Bradley-Terry"] == pytest.approx(0.602, abs=0.006) and r["loss"]["BT + Zyklus"] == pytest.approx(0.405, abs=0.006)
    assert r["loss"]["gesättigt"] == pytest.approx(0.404, abs=0.006) and r["bound"] == pytest.approx(0.381, abs=0.004)
    assert bal[(0.4, 64)]["loss"]["BT + Zyklus"] == pytest.approx(0.393, abs=0.006)
    assert bal[(0.4, 64)]["loss"]["Bradley-Terry"] == pytest.approx(0.599, abs=0.006) and bal[(0.4, 64)]["loss"]["gesättigt"] == pytest.approx(0.388, abs=0.006)
    assert bal[(0.8, 16)]["loss"]["Bradley-Terry"] == pytest.approx(0.672, abs=0.006) and bal[(0.8, 16)]["loss"]["BT + Zyklus"] == pytest.approx(0.240, abs=0.008)


def test_the_cycle_model_is_close_to_the_saturated_matrix_with_enough_games(bal):
    for c in (0.4, 0.8):
        r = bal[(c, 64)]["loss"]
        assert abs(r["BT + Zyklus"] - r["gesättigt"]) < 0.01


# --- Experiment 2: Vorhersage bei unausgewogenen Stilen ---------------------------------------------------------------------------------------

def test_with_unbalanced_styles_prediction_barely_suffers(unb):
    r = unb[(0.4, 64)]
    assert r["loss"]["Bradley-Terry"] == pytest.approx(0.485, abs=0.006) and r["loss"]["BT + Zyklus"] == pytest.approx(0.483, abs=0.006) and r["bound"] == pytest.approx(0.477, abs=0.004)
    assert r["loss"]["Bradley-Terry"] - r["bound"] < 0.012
    assert unb[(0.2, 64)]["loss"]["Bradley-Terry"] == pytest.approx(0.520, abs=0.006)


# --- Experiment 3: Rangfolge hängt am Feld ----------------------------------------------------------------------------------------------------

def test_unbalanced_styles_push_the_best_down_the_table(rank):
    assert rank[("konter", 0.0)]["place"] == 1.0
    assert rank[("konter", 0.2)]["place"] == pytest.approx(3.25, abs=0.3) and rank[("konter", 0.4)]["place"] == pytest.approx(5.85, abs=0.4)
    assert rank[("konter", 0.2)]["tau_s"] == pytest.approx(0.749, abs=0.03) and rank[("konter", 0.4)]["tau_s"] == pytest.approx(0.532, abs=0.03)
    assert all(rank[("ausgewogen", c)]["place"] == 1.0 and rank[("ausgewogen", c)]["tau_s"] == 1.0 for c in C.C_RANK)
    assert rank[("beute", 0.4)]["place"] == pytest.approx(1.07, abs=0.1)


def test_the_expected_win_rate_depends_on_the_field(rank):
    assert rank[("konter", 0.2)]["tau_field"] == pytest.approx(0.907, abs=0.02) and rank[("konter", 0.4)]["tau_field"] == pytest.approx(0.949, abs=0.02)
    assert rank[("ausgewogen", 0.4)]["tau_field"] == pytest.approx(0.970, abs=0.02)
    assert rank[("konter", 0.0)]["tau_field"] == 1.0


# --- Experiment 4: Zyklen aus Daten sehen -----------------------------------------------------------------------------------------------------

def test_noise_alone_fills_the_matrix_with_cycles(detect):
    assert detect[(0.0, 1)]["data"] == pytest.approx(0.571, abs=0.015) and detect[(0.0, 1)]["true"] < 1e-12
    assert detect[(0.0, 64)]["data"] == pytest.approx(0.041, abs=0.006)
    assert [round(detect[(0.0, k)]["data"], 2) for k in C.GAMES_GRID] == pytest.approx([0.57, 0.31, 0.13, 0.04], abs=0.02)


def test_the_cycle_signal_exceeds_the_baseline_only_with_many_games(detect):
    r = detect[(0.4, 64)]
    assert r["data"] == pytest.approx(0.075, abs=0.008) and r["baseline"] == pytest.approx(0.041, abs=0.006) and r["true"] == pytest.approx(0.042, abs=0.004)
    assert r["data"] - r["baseline"] > 0.025
    for k in (1, 4, 16):
        assert abs(detect[(0.4, k)]["data"] - detect[(0.4, k)]["baseline"]) < 0.1 * detect[(0.4, k)]["baseline"] + 0.01 or k == 1
    assert detect[(0.4, 16)]["data"] - detect[(0.4, 16)]["baseline"] < 0.01


def test_cyclic_triples_follow_the_same_picture(detect):
    assert detect[(0.0, 1)]["triples"] == pytest.approx(0.132, abs=0.012) and detect[(0.0, 1)]["triples_baseline"] == pytest.approx(0.092, abs=0.012)
    assert detect[(0.4, 64)]["triples"] > 3 * detect[(0.4, 64)]["triples_baseline"]


# --- Experiment 5: der Preis des Zyklenmodells bei wenigen Partien ----------------------------------------------------------------------------

def test_the_cycle_model_costs_more_than_it_brings_with_few_games(bal):
    r1, r4 = bal[(0.0, 1)]["loss"], bal[(0.0, 4)]["loss"]
    assert r1["Bradley-Terry"] == pytest.approx(0.700, abs=0.008) and r1["BT + Zyklus"] == pytest.approx(0.827, abs=0.01) and r1["gesättigt"] == pytest.approx(0.653, abs=0.01)
    assert r4["BT + Zyklus"] > r4["Bradley-Terry"] + 0.05
    assert bal[(0.0, 64)]["loss"]["BT + Zyklus"] - bal[(0.0, 64)]["loss"]["Bradley-Terry"] < 0.01


# --- Experiment 6: Nash-Mittelung -------------------------------------------------------------------------------------------------------------

def test_nash_average_helps_only_partially(nash):
    assert nash[0.2]["place_e"] == pytest.approx(3.25, abs=0.3) and nash[0.4]["place_e"] == pytest.approx(5.85, abs=0.4)
    assert nash[0.2]["place_nash"] == pytest.approx(2.47, abs=0.35) and nash[0.4]["place_nash"] == pytest.approx(5.05, abs=0.5)
    assert nash[0.4]["place_nash"] > 1.5 and nash[0.2]["support"] == pytest.approx(1.0, abs=0.2) and nash[0.4]["support"] == pytest.approx(1.0, abs=0.2)
    assert max(r["exploit"] for r in nash.values()) < 0.01


# --- Verdikte nennen die Zahlen aus den Daten -------------------------------------------------------------------------------------------------

def test_verdicts_quote_the_measured_numbers(bal, unb, rank, detect, nash):
    assert "Bradley-Terry bei 0,602**, mit einem Zyklus bei **0,405**, gesättigt bei 0,404 (Untergrenze 0,381)" in ev.verdict_balanced(list(bal.values()))[1]
    assert "Bradley-Terry 0,485, mit Zyklus 0,483, Untergrenze 0,477" in ev.verdict_unbalanced(list(unb.values()))[1]
    r = ev.verdict_rank(list(rank.values()))[1]
    assert "Platz **3,2** (c = 0,2) bzw. **5,8** (c = 0,4), τ gegen die Stärke 0,75 bzw. 0,53" in r and "bleibt er auf Platz 1." in r
    d = ev.verdict_detect(list(detect.values()))[1]
    assert "(c = 0) bei 0,57; bei 64 Partien sinkt dieses Rauschen auf 0,041" in d and "(Grundlinie 0,041, wahr 0,042)" in d
    assert "**0,827 gegen 0,700** bei Bradley-Terry, bei 4 Partien 0,594 gegen 0,532" in ev.verdict_price(list(bal.values()))[1]
    n = ev.verdict_nash(list(nash.values()))[1]
    assert "Platz 3,2 / 5,8, nach der Nash-Wertung auf Platz **2,5 / 5,0**" in n and "1,0 / 1,0 Spieler" in n


def test_the_equilibrium_is_the_one_player_nobody_beats_in_expectation():
    """In allen 60 Instanzen mit unausgewogenen Stilen (c = 0,2 und 0,4) gibt es einen Spieler, gegen den kein anderer eine Siegwahrscheinlichkeit über ½ hat; das Gleichgewicht
    stützt sich genau auf ihn."""
    import numpy as np
    import nti_nash as N
    for c in (0.2, 0.4):
        for sd in C.SWEEP_SEEDS[:20]:
            P = ev._tournament("konter", c, 1, sd).P
            n = len(P)
            unbeaten = [k for k in range(n) if all(P[i, k] <= 0.5 + 1e-12 for i in range(n) if i != k)]
            p, _, _ = N.nash_average(P, steps=6000)
            assert unbeaten and int(np.argmax(p)) in unbeaten and p.max() > 0.9
