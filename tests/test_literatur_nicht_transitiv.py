"""Literatur-Referenzwerte (ORACLE-PLAYBOOK §5.1): Werte und Sätze aus Quellen, die für diese Datei tatsächlich eingesehen wurden. Einsicht jeweils am 2026-10-17 (PDF von arxiv.org
geladen und mit pdftotext gelesen). Kein Wert stammt aus dem Code der Demo.

1. Balduzzi, Tuyls, Pérolat, Graepel (2018): „Re-evaluating evaluation“, arXiv:1806.02643v2 (Seitenzählung der arXiv-PDF; die README nennt NeurIPS 31):
   - S. 6, Beispiel 1 (Invarianz): Antisymmetrische logit-Matrix A mit Einträgen 4,6 für den Zyklus A→B→C→A. Maxent-Nash p = (1/3, 1/3, 1/3), Nash-Wertung n = 0. Mit einer
     redundanten Kopie von C (A' = 4×4, C1 = C2) ist (1/3, 1/3, λ/3, (1−λ)/3) Nash für jedes λ ∈ [0, 1], das Maxent-Nash ist (1/3, 1/3, 1/6, 1/6), die Nash-Wertung ist 0.
     Gleichmäßige Mittelung dagegen: div(A') = (−1,15, 1,15, 0, 0) (Zeilenmittel der logit-Matrix; 4,6/4 = 1,15). Die Demo rechnet mit der Auszahlung P − ½; P = σ(A) hat dieselbe
     Struktur (Gleichgewichte hängen nur von der Struktur ab, nicht vom Betrag der Einträge).
   - S. 7, Beispiel 2 (Stetigkeit): C = [[0,1,−1],[−1,0,1],[1,−1,0]], T = [[0,1,2],[−1,0,1],[−2,−1,0]] (T = grad(r) mit r = (1, 0, −1)). Maxent-Nash von C + εT:
     p = ((1+ε)/3, (1−2ε)/3, (1+ε)/3) mit Wertung (0, 0, 0) für 0 ≤ ε ≤ ½, und p = (1, 0, 0) mit Wertung (0, −1−ε, 1−2ε) für ε > ½. Positive Streckung der Auszahlung (hier mit 0,2 zu
     Wahrscheinlichkeiten 0,5 + 0,2 (C + εT)) ändert die Gleichgewichte nicht; die Wertung skaliert mit dem Faktor.
   - S. 3, Satz 1: Elo-Wertungen sind stationär unter Stapelaktualisierung genau dann, wenn die Zeilensummen der empirischen und der vorhergesagten Wahrscheinlichkeiten gleich sind.
   - S. 4, Satz 3: (i) Stammen die Wahrscheinlichkeiten aus Elo-Wertungen r (Σ r = 0), so ist div(logit P) = r (Zeilenmittel); (ii) es gibt eine Elo-Wertung, die P erzeugt, genau dann, wenn
     curl(logit P) = 0, d. h. log(p_ij/p_ji) + log(p_jk/p_kj) + log(p_ki/p_ik) = 0 für alle i, j, k.
   - S. 3 (Abschnitt 2.1): p̂_ij = σ(α(r_i − r_j)) mit α = ln(10)/400.
   - S. 4 (Abschnitt 3.1, mElo_2k): p̂_ij = σ(r_i − r_j + c_i Ω_{2k×2k} c_j) mit c_i ∈ R^{2k} und Ω = Σ_{l=1..k} (e_{2l−1} e_{2l}ᵀ − e_{2l} e_{2l−1}ᵀ); die Demo rechnet k = 1.
   - Abschnitt 4.1 (S. 6): Das Papier definiert die Nash-Mittelung auf der logit-Matrix A mit dem Maxent-Gleichgewicht (Satz 4); die Demo nimmt die Auszahlung P − ½ und irgendein Gleichgewicht
     (nti_nash.py); die Tests rechnen die Beispiele 1 und 2 daher mit P = σ(A) bzw. P = 0,5 + 0,2 (C + εT) nach, nicht mit der logit-Matrix der Quelle.
2. Jiang, Lim, Yao, Ye (2011): „Statistical ranking and combinatorial Hodge theory“, Math. Programming 127, 203–244; eingesehen als arXiv:0811.1067v2 (Seitenzählung der arXiv-PDF):
   - S. 21/22, Satz 5.1: (i) Normalgleichung Δ₀ s = −div Ȳ der Kleinste-Quadrate-Anpassung; (ii) das Residuum R = Ȳ − grad s ist divergenzfrei (div R = 0).
   - S. 22, Gleichung (26): Bei vollständigem Graphen und ungewichtetem Skalarprodukt ist die Minimum-Norm-Lösung s_i = −(1/n) div(Ȳ)(i) = −(1/n) Σ_j Ȳ_ij mit Σ s_i = 0 (Borda-Zählung).
     Die Orientierung des Papiers ist (grad s)(i,j) = s_j − s_i; mit der Orientierung der Demo (A_ij ≈ s_i − s_j) lautet (26): s_i = (1/n) Σ_j A_ij.
   - S. 35, Abschnitt 8.1: Zyklizitätsverhältnis (cyclicity ratio) C_p = |R|² / |Ȳ|² (Anteil des Residuums an der Gesamtnorm), das Maß des Zyklenanteils der Demo.

Nicht eingesehen bzw. nicht verwertbar (kein Test, siehe Bericht „offen“):
- Balduzzi et al., Anhang A (S. 12): Drei Agenten mit Wahrscheinlichkeiten 0,9/0,1 und ein zweites C: Elo −63 / +63 / 0 / 0. Die Demo-Anpassung (Bradley-Terry, 400/ln 10 Punkte je θ) ergibt für P exakt ±71,9
  Punkte; ±71,9 erfüllt Satz 1 der Quelle (Zeilensummen 1,1 und 1,9), ±63 erfüllt ihn nicht (1,146). Der Wert der Quelle ist daher nicht reproduzierbar (vermutlich ungenau angegeben oder mit anderem,
  nicht beschriebenem Verfahren gerechnet); kein Test gegen −63 (dazu stünde im Test der Befund, nicht der Code).
- Dangauthier et al. und Coulom betreffen diese Demo nicht.
"""

import numpy as np
import pytest

import nti_constants as C
import nti_cycles as Y
import nti_models as M
import nti_nash as N
import turnier_generator as T


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


A3 = np.array([[0.0, 4.6, -4.6], [-4.6, 0.0, 4.6], [4.6, -4.6, 0.0]])
A4 = np.array([[0.0, 4.6, -4.6, -4.6], [-4.6, 0.0, 4.6, 4.6], [4.6, -4.6, 0.0, 0.0], [4.6, -4.6, 0.0, 0.0]])


# --- Balduzzi et al. 2018, Beispiel 1 -------------------------------------------------------------------------------------------------------------------

def test_balduzzi_example_1_nash_average_is_zero_and_uniform_for_the_plain_cycle():
    p, rating, expl = N.nash_average(sig(A3), steps=5000)
    assert p == pytest.approx([1 / 3] * 3, abs=1e-3) and np.abs(rating).max() < 1e-3 and expl < 1e-3


def test_balduzzi_example_1_redundant_copy_splits_the_mass_and_keeps_the_nash_average_zero():
    p, rating, expl = N.nash_average(sig(A4), steps=10000)
    assert p == pytest.approx([1 / 3, 1 / 3, 1 / 6, 1 / 6], abs=0.02)
    assert np.abs(rating).max() < 0.01 and expl < 0.01                      # Nash-Wertung 0 in beiden Fällen
    A = N.payoff(sig(A4))
    for lam in (0.0, 0.3, 1.0):                                              # (1/3, 1/3, λ/3, (1−λ)/3) ist für jedes λ ∈ [0, 1] ein Gleichgewicht
        q = np.array([1 / 3, 1 / 3, lam / 3, (1 - lam) / 3])
        assert N.exploitability(A, q) == pytest.approx(0.0, abs=1e-12)
        assert A @ q == pytest.approx(np.zeros(4), abs=1e-12)


def test_balduzzi_example_1_uniform_averaging_is_fooled_by_the_copy_row_mean_is_1_15():
    s = Y.antisymmetric(A4).mean(axis=1)
    assert s == pytest.approx([-1.15, 1.15, 0.0, 0.0], abs=1e-12)
    Tm = Y.transitive_part(A4)                                               # die beste Rangstärke der Demo ist genau dieses Zeilenmittel
    assert Tm == pytest.approx(s[:, None] - s[None, :], abs=1e-12)
    assert Y.antisymmetric(A3).mean(axis=1) == pytest.approx(np.zeros(3), abs=1e-12)          # div(A) = 0
    assert Y.cycle_share(A3) == pytest.approx(1.0, abs=1e-12)                # div = 0 heißt rein zyklisch (Satz 1, P3 (i))


# --- Balduzzi et al. 2018, Beispiel 2 -------------------------------------------------------------------------------------------------------------------

CYC = np.array([[0.0, 1.0, -1.0], [-1.0, 0.0, 1.0], [1.0, -1.0, 0.0]])
TRA = np.array([[0.0, 1.0, 2.0], [-1.0, 0.0, 1.0], [-2.0, -1.0, 0.0]])


@pytest.mark.parametrize("eps", [0.0, 0.1, 0.2, 0.3, 0.4])
def test_balduzzi_example_2_maxent_nash_of_cycle_plus_eps_transitive(eps):
    kappa = 0.2
    P = 0.5 + kappa * (CYC + eps * TRA)                                       # Auszahlung P − ½ = κ (C + εT)
    p, rating, expl = N.nash_average(P, steps=10000)
    expected = np.array([(1 + eps) / 3, (1 - 2 * eps) / 3, (1 + eps) / 3])
    assert p == pytest.approx(expected, abs=0.02)
    assert np.abs(rating / kappa).max() < 0.03                               # Nash-Wertung (0, 0, 0)
    assert expl < 0.01


def test_balduzzi_example_2_at_one_half_the_equilibrium_is_not_unique_and_the_solver_returns_a_valid_one():
    """Bei ε = ½ ist jedes (t, 0, 1−t) mit t ≥ ½ ein Gleichgewicht (von Hand: zweite Komponente von A p: 1,5 (1 − 2t) κ ≤ 0); das Maxent-Nash der Quelle ist (½, 0, ½). Die Demo gibt ausdrücklich irgendein Gleichgewicht zurück (nti_nash.py: „liefert eines“),
    hier ein anderes. Geprüft wird, was die Demo verspricht: ein gültiges Gleichgewicht mit Wertung ≤ 0."""
    kappa = 0.2
    P = 0.5 + kappa * (CYC + 0.5 * TRA)
    p, rating, expl = N.nash_average(P, steps=10000)
    assert expl < 0.01 and rating.max() < 0.01 and p[1] < 0.01 and p.sum() == pytest.approx(1.0)
    for t in (0.5, 0.7, 1.0):
        assert N.exploitability(N.payoff(P), np.array([t, 0.0, 1.0 - t])) == pytest.approx(0.0, abs=1e-12)
    assert N.exploitability(N.payoff(P), np.array([0.3, 0.0, 0.7])) > 0.05                      # t < ½ ist keins


@pytest.mark.parametrize("eps", [0.75, 1.0])
def test_balduzzi_example_2_dominant_first_player_beyond_one_half(eps):
    kappa = 0.2
    P = 0.5 + kappa * (CYC + eps * TRA)
    p, rating, _ = N.nash_average(P, steps=10000)
    assert p[0] > 0.98
    expected_rating = np.array([0.0, -1.0 - eps, 1.0 - 2 * eps])              # (0, −1−ε, 1−2ε)
    assert rating / kappa == pytest.approx(expected_rating, abs=0.03)


# --- Balduzzi et al. 2018, Sätze 1 und 3 ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("k", range(10))
def test_balduzzi_proposition_1_stationary_elo_has_equal_row_sums(k):
    """Satz 1: Im stationären Punkt der Stapelaktualisierung stimmen die Zeilensummen der empirischen und der vorhergesagten Wahrscheinlichkeiten überein (vollständiger Plan, gleich viele Partien)."""
    t = T.build_turnier(6, 0.3, 16, "vollstaendig", 0, 1.0, 0.4, "ausgewogen", 900 + k)
    W, G = t.wins, t.games
    s = M.fit_bt(W, G, lam=1e-9)
    n = len(W)
    off = ~np.eye(n, dtype=bool)
    emp = np.where(off, W / np.maximum(G, 1), 0.0)
    pred = np.where(off, sig(s[:, None] - s[None, :]), 0.0)
    assert emp.sum(axis=1) == pytest.approx(pred.sum(axis=1), abs=1e-6)


def test_balduzzi_proposition_1_on_the_appendix_a_probabilities():
    P = np.array([[.5, .9, .1, .1], [.1, .5, .9, .9], [.9, .1, .5, .5], [.9, .1, .5, .5]])
    G = np.full((4, 4), 1000.0)
    np.fill_diagonal(G, 0.0)
    W = P * G
    np.fill_diagonal(W, 0.0)
    s = M.fit_bt(W, G, lam=1e-9)
    pred = sig(s[:, None] - s[None, :])
    np.fill_diagonal(pred, 0.0)
    off = P - 0.5 * np.eye(4)
    assert off.sum(axis=1) == pytest.approx(pred.sum(axis=1), abs=1e-6)       # Zeilensummen 1,1 / 1,9 / 1,1 / 1,1 nach Abzug der Diagonale
    assert s[2] == pytest.approx(s[3], abs=1e-9) and s[1] > s[2] > s[0]       # Kopien gleich bewertet; Reihenfolge wie im Papier (A < C < B)


@pytest.mark.parametrize("k", range(15))
def test_balduzzi_proposition_3_transitive_probabilities_have_zero_curl_and_divergence_equals_the_ratings(k):
    rng = np.random.default_rng(1100 + k)
    n = int(rng.integers(3, 9))
    r = rng.normal(0, 1.5, n)
    r -= r.mean()                                                             # Σ r = 0
    A = r[:, None] - r[None, :]                                               # logit P für P = σ(r_i − r_j)
    P = sig(A)
    L = np.log(P / (1.0 - P))
    assert L.mean(axis=1) == pytest.approx(r, abs=1e-9)                       # (i) div(logit P) = r
    for i in range(n):
        for j in range(n):
            for m in range(n):
                assert L[i, j] + L[j, m] + L[m, i] == pytest.approx(0.0, abs=1e-9)         # (ii) curl = 0
    assert Y.cycle_share(L) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("k", range(15))
def test_balduzzi_proposition_3_nonzero_curl_means_positive_cycle_share(k):
    rng = np.random.default_rng(1200 + k)
    n = int(rng.integers(3, 8))
    X = rng.normal(0, 1, (n, n))
    L = X - X.T
    max_curl = max(abs(L[i, j] + L[j, m] + L[m, i]) for i in range(n) for j in range(n) for m in range(n))
    assert max_curl > 1e-3 and Y.cycle_share(L) > 1e-9


def test_elo_scale_alpha_of_the_paper():
    """p̂_ij = σ(α(r_i − r_j)), α = ln(10)/400 (Abschnitt 2.1): 400 Punkte Unterschied = Faktor 10 der Gewinnchancen."""
    alpha = np.log(10.0) / 400.0
    assert alpha * C.SCALE == pytest.approx(1.0, abs=1e-12)
    assert sig(alpha * 400.0) == pytest.approx(10.0 / 11.0, abs=1e-12)


# --- Jiang, Lim, Yao, Ye 2011 ------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("k", range(20))
def test_jiang_theorem_5_1_normal_equation_divergence_free_residual_and_borda(k):
    rng = np.random.default_rng(1300 + k)
    n = int(rng.integers(3, 10))
    X = rng.normal(0, 1, (n, n))
    A = X - X.T                                                               # Ȳ (in der Orientierung der Demo: A_ij ≈ s_i − s_j)
    Tm = Y.transitive_part(A)
    s = Tm.mean(axis=1)                                                       # Rangstärken aus der transitiven Matrix zurückgewonnen (Σ s = 0)
    assert s.sum() == pytest.approx(0.0, abs=1e-12)
    laplacian = n * np.eye(n) - np.ones((n, n))                               # Graph-Laplace-Matrix des vollständigen Graphen, Einheitsgewichte
    assert laplacian @ s == pytest.approx(A.sum(axis=1), abs=1e-10)           # (i) Normalgleichung Δ₀ s = div (Vorzeichen je nach Orientierung)
    R = A - Tm
    assert R.sum(axis=1) == pytest.approx(np.zeros(n), abs=1e-10)             # (ii) div R = 0
    assert s == pytest.approx(A.sum(axis=1) / n, abs=1e-12)                   # (26) Borda-Zählung s_i = (1/n) Σ_j A_ij
    assert R == pytest.approx(-R.T, abs=1e-12)


@pytest.mark.parametrize("k", range(20))
def test_jiang_cyclicity_ratio_equals_the_cycle_share(k):
    rng = np.random.default_rng(1400 + k)
    n = int(rng.integers(3, 10))
    X = rng.normal(0, 1, (n, n))
    A = X - X.T
    s = A.sum(axis=1) / n                                                     # Gleichung (26) in der Orientierung der Demo
    R = A - (s[:, None] - s[None, :])
    iu = np.triu_indices(n, 1)                                                # Normen über die Kanten (i < j)
    cp = float((R[iu] ** 2).sum() / (A[iu] ** 2).sum())
    assert Y.cycle_share(A) == pytest.approx(cp, abs=1e-12)


# --- Balduzzi et al. 2018, Abschnitt 3.1: mElo_2 -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("k", range(10))
def test_balduzzi_melo2_prediction_is_the_cycle_logit_of_the_demo(k):
    """mElo_2: r_i − r_j + c_i Ω c_j mit c_i = (u_i, v_i) und Ω = e₁e₂ᵀ − e₂e₁ᵀ; das ist r_i − r_j + u_i v_j − v_i u_j (antisymmetrisch, wie P(i schlägt j) + P(j schlägt i) = 1 verlangt)."""
    rng = np.random.default_rng(1600 + k)
    n = int(rng.integers(3, 9))
    r, u, v = rng.normal(0, 1, n), rng.normal(0, 1, n), rng.normal(0, 1, n)
    omega = np.array([[0.0, 1.0], [-1.0, 0.0]])
    c = np.stack([u, v], axis=1)
    paper = r[:, None] - r[None, :] + c @ omega @ c.T
    L = M.cycle_logit_matrix(r, u, v)
    assert L == pytest.approx(paper, abs=1e-12)
    assert L == pytest.approx(-L.T, abs=1e-12)
    assert sig(L) + sig(L).T == pytest.approx(np.ones((n, n)), abs=1e-12)
