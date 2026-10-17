"""Metamorphe Eigenschaftstests für Zyklen, Modelle und Nash-Mittelung (ORACLE-PLAYBOOK §5.2): Man verändert die Eingabe auf eine Weise, deren Wirkung auf die Ausgabe bekannt ist,
und prüft diese Beziehung.

Belegt sind die Aussagen durch die Definitionen im Code und in der README: logit-Matrizen sind antisymmetrisch (P_ij + P_ji = 1), der Zyklenanteil ist der Anteil der quadratischen Norm
der antisymmetrischen Matrix, der nach Abzug der besten Rangstärke s_i − s_j übrig bleibt (Kleinste Quadrate, Hodge-Zerlegung: Rangstärke-Anteil und Rest sind orthogonal).

Eigenschaften:
  1. Zyklenanteil: invariant gegen Umnummerieren, Spiegeln (L → Lᵀ) und positive Skalierung; er ignoriert den symmetrischen Teil.
  2. Ein rein transitives Modell (L_ij = s_i − s_j, auch aus dem Generator mit c = 0) hat Zyklenanteil 0; ein reiner Dreierzyklus hat 1 (von Hand).
  3. Orthogonale Zerlegung: für L = T + C mit T = s_i − s_j und zeilensummenfreiem antisymmetrischem C gilt Anteil = |C|²/(|T|² + |C|²); ein Rangstärke-Anteil, der zu L addiert wird,
     ändert die Norm des Rests nicht.
  4. Antisymmetrie des Zyklenterms (C = −Cᵀ) und p(a,b) + p(b,a) = 1 für alle Modelle (Bradley-Terry, BT + Zyklus, gesättigt, Generator); Vertauschen von u und v negiert den Zyklenterm.
  5. Zyklenmodell mit u = v = 0 ist Bradley-Terry: logit-Matrix, Zielwert (ohne Ridge) und Gradient in s; die Gradienten in u und v verschwinden dort.
  6. Umnummerieren und Spiegeln der Siegmatrix: Bradley-Terry und gesättigte Matrix folgen (Permutation bzw. Vorzeichen), Zielwert des Zyklenmodells ist invariant.
  7. Zyklische Dreiergruppen: Zahl und Anteil sind invariant gegen Umnummerieren und Spiegeln; bei vollständig entschiedenen Turnieren gilt die Zählformel
     zyklisch = C(n,3) − Σ_i C(Siege_i, 2) (nur transitive Dreiergruppen haben einen Spieler mit zwei Siegen in der Gruppe).
  8. Nash-Mittelung: Wert des symmetrischen Nullsummenspiels ist 0 (pᵀAp = 0, Ausbeutbarkeit ≥ 0 und im Gleichgewicht ≈ 0, Wertung ≤ 0 und ≈ 0 auf dem Träger),
     Umnummerieren permutiert Gleichgewicht und Wertung, eine positive Streckung der Auszahlung P − ½ lässt das Gleichgewicht (Ausbeutbarkeit gegen die ursprüngliche Auszahlung) gelten.
"""

from math import comb

import numpy as np
import pytest

import nti_cycles as Y
import nti_models as M
import nti_nash as N
import turnier_generator as T
from conftest import counts, rps

N_INST = 30


def _antisym(k, n=None):
    rng = np.random.default_rng(3000 + k)
    n = int(rng.integers(3, 9)) if n is None else n
    X = rng.normal(0.0, 1.0, (n, n))
    return X - X.T


def _tournament(k, c=0.4, n=None, games=None):
    rng = np.random.default_rng(4000 + k)
    n = int(rng.integers(4, 8)) if n is None else n
    games = int(rng.choice([4, 8, 16])) if games is None else games
    t = T.build_turnier(n, 0.3, games, "vollstaendig", 0, 1.0, c, str(rng.choice(["ausgewogen", "konter", "beute"])), 600 + k)
    return t.wins, t.games, t.P


def _grad(s):
    return np.asarray(s)[:, None] - np.asarray(s)[None, :]


# --- 1. Zyklenanteil: Invarianzen ----------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("k", range(N_INST))
def test_cycle_share_is_invariant_under_relabeling_mirroring_scaling_and_symmetric_parts(k):
    L = _antisym(k)
    n = len(L)
    rng = np.random.default_rng(k)
    perm = rng.permutation(n)
    base = Y.cycle_share(L)
    assert 0.0 <= base <= 1.0
    assert Y.cycle_share(L[np.ix_(perm, perm)]) == pytest.approx(base, abs=1e-12)
    assert Y.cycle_share(L.T) == pytest.approx(base, abs=1e-12)
    assert Y.cycle_share(float(rng.uniform(0.1, 9.0)) * L) == pytest.approx(base, abs=1e-12)
    S = rng.normal(0, 1, (n, n))
    assert Y.cycle_share(L + (S + S.T)) == pytest.approx(base, abs=1e-12)           # nur der antisymmetrische Teil zählt


# --- 2. Transitiv gleich 0, reiner Zyklus gleich 1 ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("k", range(N_INST))
def test_transitive_models_have_zero_cycle_share(k):
    rng = np.random.default_rng(5000 + k)
    s = rng.normal(0, 1.5, int(rng.integers(3, 10)))
    assert Y.cycle_share(_grad(s)) == pytest.approx(0.0, abs=1e-12)
    W, G, P = _tournament(k, c=0.0)                                         # Generator ohne Zyklenstärke: P = σ(s_i − s_j)
    off = ~np.eye(len(P), dtype=bool)
    L = np.zeros_like(P)
    L[off] = np.log(P[off] / (1.0 - P[off]))                                # logit der wahren Matrix (Diagonale des Generators ist 0)
    assert Y.cycle_share(L) == pytest.approx(0.0, abs=1e-9)


def test_pure_three_cycle_has_cycle_share_one_by_hand():
    """A = (0→1→2→0) mit Einträgen ±1: jede Zeile summiert zu 0, also ist das Zeilenmittel (beste Rangstärke) 0 und der ganze Anteil bleibt übrig."""
    A = np.array([[0.0, 1.0, -1.0], [-1.0, 0.0, 1.0], [1.0, -1.0, 0.0]])
    assert Y.cycle_share(A) == pytest.approx(1.0, abs=1e-15)
    W, G = rps(7)
    assert Y.cycle_share(M.saturated_logit(W, G)) == pytest.approx(1.0, abs=1e-9)
    assert Y.cycle_share(np.zeros((4, 4))) == 0.0                           # Randfall: alles 0


# --- 3. Orthogonalität (Hodge) ----------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("k", range(N_INST))
def test_rank_part_and_remainder_are_orthogonal(k):
    X = _antisym(k)
    n = len(X)
    r = X.mean(axis=1)
    C = X - _grad(r)                                                        # antisymmetrisch, Zeilensummen 0
    assert np.abs(C.sum(axis=1)).max() < 1e-12
    rng = np.random.default_rng(k)
    s = rng.normal(0, 1.0, n)
    Tm = _grad(s)
    expected = (C ** 2).sum() / ((Tm ** 2).sum() + (C ** 2).sum())
    assert Y.cycle_share(Tm + C) == pytest.approx(expected, abs=1e-10)
    # Ein addierter Rangstärke-Anteil ändert die Norm des Rests nicht: Anteil · |A|² bleibt gleich.
    t = rng.normal(0, 2.0, n)
    A0, A1 = Y.antisymmetric(X), Y.antisymmetric(X + _grad(t))
    assert Y.cycle_share(X + _grad(t)) * (A1 ** 2).sum() == pytest.approx(Y.cycle_share(X) * (A0 ** 2).sum(), abs=1e-9)


# --- 4. Antisymmetrie, p(a,b) + p(b,a) = 1 ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("k", range(N_INST))
def test_cycle_term_is_antisymmetric_and_swapping_u_and_v_negates_it(k):
    rng = np.random.default_rng(6000 + k)
    n = int(rng.integers(3, 9))
    s, u, v = rng.normal(0, 1, n), rng.normal(0, 1, n), rng.normal(0, 1, n)
    Cm = M.cycle_logit_matrix(np.zeros(n), u, v)
    assert Cm == pytest.approx(-Cm.T, abs=1e-14)
    assert M.cycle_logit_matrix(np.zeros(n), v, u) == pytest.approx(-Cm, abs=1e-14)
    assert M.cycle_logit_matrix(s, u, u) == pytest.approx(M.logit_matrix(s), abs=1e-14)          # u = v: kein Zyklus (u uᵀ ist symmetrisch)
    L = M.cycle_logit_matrix(s, u, v)
    assert L == pytest.approx(-L.T, abs=1e-14)
    P = M.sigmoid(L)
    off = ~np.eye(n, dtype=bool)
    assert (P + P.T)[off] == pytest.approx(np.ones(off.sum()), abs=1e-12)


@pytest.mark.parametrize("k", range(10))
def test_all_models_give_complementary_probabilities(k):
    W, G, P = _tournament(k)
    n = len(W)
    off = ~np.eye(n, dtype=bool)
    models = {"BT": M.logit_matrix(M.fit_bt(W, G)), "gesättigt": M.saturated_logit(W, G), "Zyklus": M.fit_cycle(W, G, 0.3, restarts=1, seed=k)[0]}
    for name, L in models.items():
        assert L == pytest.approx(-L.T, abs=1e-12), name
        Q = M.sigmoid(L)
        assert (Q + Q.T)[off] == pytest.approx(np.ones(off.sum()), abs=1e-12), name
    assert (P + P.T)[off] == pytest.approx(np.ones(off.sum()), abs=1e-12)                      # wahre Matrix des Generators


# --- 5. Zyklenmodell mit u = v = 0 ist Bradley-Terry ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("k", range(N_INST))
def test_cycle_model_without_cycle_is_bradley_terry(k):
    W, G, _ = _tournament(k)
    n = len(W)
    rng = np.random.default_rng(k)
    s = rng.normal(0, 1, n)
    x = np.concatenate([s, np.zeros(2 * n)])
    assert M.cycle_logit_matrix(s, np.zeros(n), np.zeros(n)) == pytest.approx(M.logit_matrix(s), abs=1e-15)
    f, g = M.neg_loglik_cycle(x, W, G, lam=0.0)
    i, j = M.pairs(n)
    d = s[i] - s[j]
    nll = float(np.sum(W[i, j] * np.logaddexp(0, -d) + (G[i, j] - W[i, j]) * np.logaddexp(0, d)))          # Bradley-Terry ohne Ridge
    assert f == pytest.approx(nll, abs=1e-9)
    p = 1.0 / (1.0 + np.exp(-d))
    grad_bt = np.zeros(n)
    np.add.at(grad_bt, i, -(W[i, j] - G[i, j] * p))
    np.add.at(grad_bt, j, (W[i, j] - G[i, j] * p))
    assert g[:n] == pytest.approx(grad_bt, abs=1e-9)
    assert np.abs(g[n:]).max() < 1e-12                                      # bei u = v = 0 hängt der Zielwert in erster Ordnung nicht an u, v


# --- 6. Umnummerieren und Spiegeln der Siegmatrix ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("k", range(N_INST))
def test_relabeling_and_mirroring_the_win_matrix(k):
    W, G, _ = _tournament(k)
    n = len(W)
    perm = np.random.default_rng(k).permutation(n)
    Wp, Gp = W[np.ix_(perm, perm)], G[np.ix_(perm, perm)]
    s = M.fit_bt(W, G)
    assert M.fit_bt(Wp, Gp) == pytest.approx(s[perm], abs=1e-7)
    assert M.saturated_logit(Wp, Gp) == pytest.approx(M.saturated_logit(W, G)[np.ix_(perm, perm)], abs=1e-12)
    assert M.fit_bt(W.T, G) == pytest.approx(-s, abs=1e-7)                  # Spiegeln: jeder Sieg wird zur Niederlage
    assert M.saturated_logit(W.T, G) == pytest.approx(-M.saturated_logit(W, G), abs=1e-12)
    assert Y.cycle_share(M.saturated_logit(W.T, G)) == pytest.approx(Y.cycle_share(M.saturated_logit(W, G)), abs=1e-12)
    # Zielwert des Zyklenmodells an einem festen Parametersatz: invariant gegen Umnummerieren der Spieler (samt Parametern)
    rng = np.random.default_rng(100 + k)
    x = rng.normal(0, 0.7, 3 * n)
    xp = np.concatenate([x[:n][perm], x[n:2 * n][perm], x[2 * n:][perm]])
    assert M.neg_loglik_cycle(xp, Wp, Gp, 0.3)[0] == pytest.approx(M.neg_loglik_cycle(x, W, G, 0.3)[0], abs=1e-9)


# --- 7. Zyklische Dreiergruppen ----------------------------------------------------------------------------------------------------------------------

def _random_tournament(rng, n):
    """Vollständig entschieden: je Paar gewinnt eine Seite 3 : 1."""
    W = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                W[i, j], W[j, i] = 3, 1
            else:
                W[i, j], W[j, i] = 1, 3
    return W, W + W.T


@pytest.mark.parametrize("k", range(N_INST))
def test_triples_count_formula_and_invariance(k):
    rng = np.random.default_rng(7000 + k)
    n = int(rng.integers(3, 10))
    W, G = _random_tournament(rng, n)
    cyc, dec = Y.cyclic_triples(W, G)
    scores = (W > W.T).sum(axis=1)
    assert dec == comb(n, 3)                                                # ohne Gleichstand sind alle Dreiergruppen entschieden
    assert cyc == comb(n, 3) - int(sum(comb(int(w), 2) for w in scores))
    perm = rng.permutation(n)
    assert Y.cyclic_triples(W[np.ix_(perm, perm)], G[np.ix_(perm, perm)]) == (cyc, dec)
    assert Y.cyclic_triples(W.T, G) == (cyc, dec)
    assert Y.triple_share(W, G) == pytest.approx(cyc / dec)


@pytest.mark.parametrize("k", range(10))
def test_triples_with_ties_never_exceed_decided_and_ties_remove_triples(k):
    rng = np.random.default_rng(7500 + k)
    n = 7
    W, G = _random_tournament(rng, n)
    cyc, dec = Y.cyclic_triples(W, G)
    i, j = rng.choice(n, 2, replace=False)
    W2 = W.copy()
    W2[i, j] = W2[j, i] = 2                                                 # ein Gleichstand 2:2
    cyc2, dec2 = Y.cyclic_triples(W2, G)
    assert dec2 == dec - (n - 2) and cyc2 <= cyc and cyc2 <= dec2          # jede Dreiergruppe mit diesem Paar ist nicht mehr entschieden


# --- 8. Nash-Mittelung ---------------------------------------------------------------------------------------------------------------------------------

def _random_P(k):
    rng = np.random.default_rng(8000 + k)
    n = int(rng.integers(3, 8))
    X = rng.uniform(0.05, 0.95, (n, n))
    P = np.triu(X, 1)
    return P + np.tril(1 - P.T, -1) + 0.5 * np.eye(n)


@pytest.mark.parametrize("k", range(20))
def test_nash_average_value_of_the_symmetric_zero_sum_game_is_zero(k):
    P = _random_P(k)
    A = N.payoff(P)
    assert A == pytest.approx(-A.T, abs=1e-15)
    assert A == pytest.approx(np.where(np.eye(len(P), dtype=bool), 0.0, P - 0.5), abs=1e-15)      # Auszahlung P − ½
    rng = np.random.default_rng(k)
    q = rng.dirichlet(np.ones(len(P)))
    assert float(q @ A @ q) == pytest.approx(0.0, abs=1e-14)                 # jede gemischte Strategie gegen sich selbst: 0
    assert N.exploitability(A, q) >= -1e-12                                  # also ist die Ausbeutbarkeit nie negativ
    p, rating, expl = N.nash_average(P, steps=2000)
    assert p.sum() == pytest.approx(1.0, abs=1e-9) and p.min() >= 0.0
    assert 0.0 <= expl < 0.02                                                # Wert 0: kein reiner Gegner gewinnt gegen p im Mittel mehr als ≈ 0
    assert rating.max() < 0.02
    assert float(p @ rating) == pytest.approx(0.0, abs=1e-12)                # Σ p_i (A p)_i = pᵀA p = 0
    assert np.abs(rating[p > 0.02]).max() < 0.03                             # auf dem Träger Wertung ≈ 0


@pytest.mark.parametrize("k", range(8))
def test_nash_average_relabeling_and_positive_scaling(k):
    P = _random_P(k)
    n = len(P)
    perm = np.random.default_rng(k).permutation(n)
    p, rating, _ = N.nash_average(P, steps=2000)
    pp, rr, _ = N.nash_average(P[np.ix_(perm, perm)], steps=2000)
    assert pp == pytest.approx(p[perm], abs=1e-8) and rr == pytest.approx(rating[perm], abs=1e-8)
    c = 0.3 + 0.1 * k
    Pc = 0.5 + c * (P - 0.5)                                                 # Auszahlung c · A mit c > 0: gleiche Gleichgewichte
    pc, _, _ = N.nash_average(Pc, steps=2000)
    assert N.exploitability(N.payoff(P), pc) < 0.03
    assert N.exploitability(N.payoff(Pc), p) < 0.03 * c + 0.01


def test_nash_average_edge_cases():
    P = np.full((4, 4), 0.5)
    p, rating, expl = N.nash_average(P, steps=200)
    assert p == pytest.approx(np.full(4, 0.25), abs=1e-12) and np.abs(rating).max() < 1e-12 and abs(expl) < 1e-12      # alle gleich stark
    # ein dominanter Spieler: das Gleichgewicht ist (fast) rein auf ihn, Wert 0
    P2 = np.array([[0.5, 0.9, 0.9], [0.1, 0.5, 0.6], [0.1, 0.4, 0.5]])
    p2, r2, e2 = N.nash_average(P2, steps=3000)
    assert p2[0] > 0.97 and e2 < 0.02 and r2[0] == pytest.approx(0.0, abs=0.02) and r2[1] < 0 and r2[2] < 0
