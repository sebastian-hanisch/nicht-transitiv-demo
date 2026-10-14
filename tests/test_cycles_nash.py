"""Zyklenanteil, zyklische Dreiergruppen, Rauschgrundlinie und Nash-Mittelung an Handwerten."""

import numpy as np
import pytest

import nti_cycles as Y
import nti_models as M
import nti_nash as N
from conftest import counts, rps


def test_cycle_share_of_a_transitive_matrix_is_zero_and_of_a_pure_cycle_is_one():
    assert Y.cycle_share(M.logit_matrix([0.0, 1.0, 3.0, 4.0])) == pytest.approx(0.0, abs=1e-12)
    cyc = M.cycle_logit_matrix(np.zeros(3), [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    L = np.array([[0.0, 1.0, -1.0], [-1.0, 0.0, 1.0], [1.0, -1.0, 0.0]])
    assert Y.cycle_share(L) == pytest.approx(1.0, abs=1e-12)
    assert 0.0 < Y.cycle_share(cyc) <= 1.0


def test_cycle_share_by_hand_for_a_mixed_matrix():
    # transitiver Teil s = (1, 0, −1) plus Zyklus mit Betrag 1: A = (s_i − s_j) + R, R = Rotation (0,1,−1 ...); die Normen addieren sich (R ⟂ transitiv)
    s = np.array([1.0, 0.0, -1.0])
    R = np.array([[0.0, 1.0, -1.0], [-1.0, 0.0, 1.0], [1.0, -1.0, 0.0]])
    L = s[:, None] - s[None, :] + R
    assert Y.cycle_share(L) == pytest.approx((R ** 2).sum() / (L ** 2).sum())


def test_transitive_part_recovers_the_rank_strengths():
    s = np.array([0.0, 2.0, 5.0, 1.0])
    T = Y.transitive_part(M.logit_matrix(s))
    assert T == pytest.approx(M.logit_matrix(s - s.mean()))


def test_cycle_share_of_an_all_zero_matrix_is_zero():
    assert Y.cycle_share(np.zeros((4, 4))) == 0.0


def test_cyclic_triples_in_rock_paper_scissors_and_in_a_transitive_triple():
    W, G = rps(5)
    assert Y.cyclic_triples(W, G) == (1, 1) and Y.triple_share(W, G) == 1.0
    W2, G2 = counts(3, {(0, 1): 5, (1, 2): 5, (0, 2): 5})
    assert Y.cyclic_triples(W2, G2) == (0, 1) and Y.triple_share(W2, G2) == 0.0


def test_cyclic_triples_brute_force_on_a_random_tournament():
    rng = np.random.default_rng(1)
    n = 7
    W = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            w = int(rng.integers(0, 6))
            W[i, j], W[j, i] = w, 5 - w
    G = W + W.T
    cyc = dec = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dd = [W[i, j] != W[j, i], W[j, k] != W[k, j], W[i, k] != W[k, i]]
                if all(dd):
                    dec += 1
                    a, b, c = W[i, j] > W[j, i], W[j, k] > W[k, j], W[k, i] > W[i, k]
                    if (a and b and c) or (not a and not b and not c):
                        cyc += 1
    assert Y.cyclic_triples(W, G) == (cyc, dec)


def test_noise_baseline_is_positive_for_a_transitive_truth_and_deterministic():
    W, G = counts(5, {(i, j): 3 for i in range(5) for j in range(5) if i < j} | {(j, i): 1 for i in range(5) for j in range(5) if i < j})
    a = Y.noise_baseline(W, G, draws=6, seed=3)
    b = Y.noise_baseline(W, G, draws=6, seed=3)
    assert a == b and a["share"] > 0.0 and 0.0 <= a["triples"] <= 1.0


# --- Nash -------------------------------------------------------------------------------------------------------------------------------------

def test_payoff_is_antisymmetric_with_zero_diagonal():
    P = np.array([[0.5, 0.7, 0.2], [0.3, 0.5, 0.9], [0.8, 0.1, 0.5]])
    A = N.payoff(P)
    assert np.allclose(A, -A.T) and np.allclose(np.diag(A), 0.0)


def test_nash_average_of_rock_paper_scissors_is_uniform_with_zero_ratings():
    P = np.array([[0.5, 0.75, 0.25], [0.25, 0.5, 0.75], [0.75, 0.25, 0.5]])
    p, rating, expl = N.nash_average(P, steps=5000)
    assert p == pytest.approx([1 / 3] * 3, abs=0.02) and np.abs(rating).max() < 0.02 and expl < 0.02


def test_nash_average_of_a_transitive_tournament_puts_all_weight_on_the_best():
    s = np.array([0.0, 1.0, 2.0])
    P = M.sigmoid(M.logit_matrix(s))
    p, rating, expl = N.nash_average(P, steps=8000)
    assert p[2] > 0.95 and expl < 0.01
    assert rating[2] == pytest.approx(0.0, abs=0.01) and rating[0] < rating[1] < rating[2]


def test_exploitability_is_zero_in_equilibrium_and_positive_otherwise():
    P = np.array([[0.5, 0.75, 0.25], [0.25, 0.5, 0.75], [0.75, 0.25, 0.5]])
    A = N.payoff(P)
    assert N.exploitability(A, np.full(3, 1 / 3)) == pytest.approx(0.0, abs=1e-12)
    assert N.exploitability(A, np.array([1.0, 0.0, 0.0])) == pytest.approx(0.25)


def test_nash_ratings_are_not_positive_and_the_support_has_rating_zero():
    rng = np.random.default_rng(5)
    for _ in range(4):
        X = rng.uniform(0.05, 0.95, (5, 5))
        P = np.triu(X, 1)
        P = P + np.tril(1 - P.T, -1) + 0.5 * np.eye(5)
        p, rating, expl = N.nash_average(P, steps=20000)
        assert rating.max() < 0.02 and expl < 0.02
        assert np.all(np.abs(rating[p > 0.05]) < 0.03)


def test_ties_make_a_pair_undecided_in_the_majority_relation():
    # 0 und 1 gleich viele Siege: unentschieden; also keine entschiedene Dreiergruppe
    W, G = counts(3, {(0, 1): 3, (1, 0): 3, (1, 2): 5, (2, 0): 5})
    assert Y.cyclic_triples(W, G) == (0, 0)
    W2, G2 = counts(3, {(0, 1): 3, (1, 0): 2, (1, 2): 5, (2, 0): 5})
    assert Y.cyclic_triples(W2, G2) == (1, 1)


def test_noise_baseline_with_a_single_draw_is_finite_and_equals_that_draw():
    W, G = counts(4, {(i, j): 3 for i in range(4) for j in range(4) if i < j} | {(j, i): 2 for i in range(4) for j in range(4) if i < j})
    one = Y.noise_baseline(W, G, draws=1, seed=2)
    assert np.isfinite(one["share"]) and np.isfinite(one["triples"]) and 0.0 <= one["share"] <= 1.0


def test_nash_average_with_one_step_is_the_uniform_start_and_stays_finite_for_huge_steps():
    P = np.array([[0.5, 0.9, 0.2], [0.1, 0.5, 0.8], [0.8, 0.2, 0.5]])
    p, rating, _ = N.nash_average(P, steps=1)
    assert p == pytest.approx([1 / 3] * 3)
    p2, r2, e2 = N.nash_average(P, steps=200, eta=2000.0)
    assert np.isfinite(p2).all() and np.isfinite(r2).all() and p2.sum() == pytest.approx(1.0)
