"""Modelle: Bradley-Terry (Newton), Zyklenmodell (BFGS, Gradient), gesättigte Matrix, Log-Verlust, Entropie, erwartete Siegquote."""

import math

import numpy as np
import pytest

import nti_constants as C
import nti_models as M
from conftest import counts, rps


def test_sigmoid_and_log1pexp_are_stable():
    assert M.sigmoid(0.0) == 0.5 and M.sigmoid(-800.0) == 0.0 and M.sigmoid(800.0) == 1.0
    assert M.log1pexp(0.0) == pytest.approx(math.log(2.0)) and M.log1pexp(800.0) == pytest.approx(800.0) and M.log1pexp(-800.0) == pytest.approx(0.0, abs=1e-300)


def test_logit_matrix_is_antisymmetric():
    L = M.logit_matrix([0.0, 1.0, 3.0])
    assert L[0, 1] == -1.0 and L[2, 0] == 3.0 and np.allclose(L, -L.T)


def test_cycle_logit_matrix_adds_one_rotation_plane():
    L = M.cycle_logit_matrix([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    assert L[0, 1] == 1.0 and L[1, 0] == -1.0 and L[0, 2] == 0.0 and np.allclose(L, -L.T)


def test_bt_estimates_follow_the_win_counts_and_have_a_zero_mean_by_the_ridge():
    W, G = counts(3, {(0, 1): 8, (1, 0): 2, (1, 2): 7, (2, 1): 3, (0, 2): 9, (2, 0): 1})
    s = M.fit_bt(W, G)
    assert s[0] > s[1] > s[2] and abs(s.mean()) < 1e-6


def test_bt_gradient_is_zero_at_the_solution():
    W, G = counts(4, {(0, 1): 6, (1, 0): 2, (1, 2): 5, (2, 1): 3, (2, 3): 4, (3, 2): 4, (0, 3): 7, (3, 0): 1})
    s = M.fit_bt(W, G, lam=0.05)
    i, j = M.pairs(4)
    r = W[i, j] - G[i, j] * M.sigmoid(s[i] - s[j])
    grad = np.zeros(4)
    np.add.at(grad, i, r)
    np.add.at(grad, j, -r)
    assert np.abs(grad - 0.05 * s).max() < 1e-8


def test_bt_cannot_represent_rock_paper_scissors():
    W, G = rps()
    s = M.fit_bt(W, G)
    assert np.abs(s).max() < 1e-6                                                  # alle gleich stark: ein Kreis hat keine Rangstärke


def test_cycle_model_represents_rock_paper_scissors():
    W, G = rps(10)
    L, (s, u, v) = M.fit_cycle(W, G, lam=0.05)
    P = M.sigmoid(L)
    assert P[0, 1] > 0.9 and P[1, 2] > 0.9 and P[2, 0] > 0.9
    assert np.abs(s).max() < 0.05


def test_cycle_model_loglik_gradient_matches_finite_differences():
    W, G = counts(4, {(0, 1): 6, (1, 0): 2, (1, 2): 5, (2, 1): 3, (2, 3): 4, (3, 2): 4, (0, 3): 7, (3, 0): 1, (0, 2): 3, (2, 0): 5})
    rng = np.random.default_rng(2)
    x = rng.normal(0, 0.5, 12)
    f, g = M.neg_loglik_cycle(x, W, G, 0.3)
    num = np.zeros(12)
    for k in range(12):
        up, dn = x.copy(), x.copy()
        up[k] += 1e-6
        dn[k] -= 1e-6
        num[k] = (M.neg_loglik_cycle(up, W, G, 0.3)[0] - M.neg_loglik_cycle(dn, W, G, 0.3)[0]) / 2e-6
    assert g == pytest.approx(num, abs=1e-5)


def test_cycle_model_is_never_worse_than_bt_in_penalised_loglik_when_started_from_bt():
    W, G = counts(4, {(0, 1): 6, (1, 0): 2, (1, 2): 5, (2, 1): 3, (2, 3): 4, (3, 2): 4, (0, 3): 7, (3, 0): 1})
    s = M.fit_bt(W, G, lam=0.3)
    f_bt = M.neg_loglik_cycle(np.concatenate([s, np.zeros(8)]), W, G, 0.3)[0]
    _, (s2, u, v) = M.fit_cycle(W, G, lam=0.3)
    f_cyc = M.neg_loglik_cycle(np.concatenate([s2, u, v]), W, G, 0.3)[0]
    assert f_cyc <= f_bt + 1e-9


def test_bfgs_minimises_a_quadratic():
    x, f = M.bfgs(lambda x: (float((x - 3.0) @ (x - 3.0)), 2.0 * (x - 3.0)), np.zeros(4))
    assert x == pytest.approx([3.0] * 4, abs=1e-6) and f == pytest.approx(0.0, abs=1e-10)


def test_saturated_logit_by_hand_and_antisymmetric():
    W, G = counts(3, {(0, 1): 3, (1, 0): 1, (0, 2): 4})
    L = M.saturated_logit(W, G, prior=0.5)
    assert L[0, 1] == pytest.approx(math.log((3.5 / 5.0) / (1.5 / 5.0)))        # (3 + 0,5)/(4 + 1)
    assert L[1, 0] == pytest.approx(-L[0, 1]) and L[1, 2] == 0.0 and np.allclose(np.diag(L), 0.0)
    assert L[0, 2] == pytest.approx(math.log(4.5 / 0.5))


def test_log_loss_equals_the_entropy_for_the_true_logits_and_is_larger_otherwise():
    P = np.array([[0.5, 0.8, 0.6], [0.2, 0.5, 0.3], [0.4, 0.7, 0.5]])
    L = np.log(P / (1 - P))
    L = (L - L.T) / 2.0                                                          # P ist hier nicht antisymmetrisch genug: nur die Paare i < j zählen
    i, j = M.pairs(3)
    Pt = np.zeros((3, 3))
    Pt[i, j] = P[i, j]
    Pt[j, i] = 1 - P[i, j]
    Lt = np.log(Pt / (1 - Pt + 1e-300) + 1e-300)
    Lt = np.where(np.eye(3) == 1, 0.0, Lt)
    assert M.log_loss(Lt, Pt) == pytest.approx(M.entropy_bound(Pt), abs=1e-9)
    assert M.log_loss(np.zeros((3, 3)), Pt) > M.entropy_bound(Pt)
    assert M.log_loss(np.zeros((3, 3)), Pt) == pytest.approx(math.log(2.0))


def test_entropy_bound_by_hand():
    P = np.array([[0.5, 0.75], [0.25, 0.5]])
    h = -(0.75 * math.log(0.75) + 0.25 * math.log(0.25))
    assert M.entropy_bound(P) == pytest.approx(h)


def test_expected_win_rate_and_table_rate():
    L = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    assert M.expected_win_rate(L).tolist() == [0.5, 0.5, 0.5]
    W, G = counts(3, {(0, 1): 3, (1, 0): 1, (0, 2): 4})
    assert M.win_table_rate(W, G).tolist() == pytest.approx([7 / 8, 1 / 4, 0.0])
    assert C.LAMBDA_BT > 0


def test_iteration_limits_are_honoured_exactly():
    W, G = counts(3, {(0, 1): 8, (1, 0): 2, (1, 2): 7, (2, 1): 3, (0, 2): 9, (2, 0): 1})
    s1 = M.fit_bt(W, G, iters=1)
    assert np.abs(s1).max() > 0.1                                           # ein Newton-Schritt bewegt sich schon vom Start weg
    assert np.abs(M.fit_bt(W, G, iters=0)).max() == 0.0
    def rosen(x):
        f = (x[0] - 1.0) ** 2 + 10.0 * (x[1] - x[0] ** 2) ** 2
        g = np.array([2.0 * (x[0] - 1.0) - 40.0 * x[0] * (x[1] - x[0] ** 2), 20.0 * (x[1] - x[0] ** 2)])
        return float(f), g

    x1, _ = M.bfgs(rosen, np.array([-1.0, 2.0]), iters=1)
    x0, _ = M.bfgs(rosen, np.array([-1.0, 2.0]), iters=0)
    x50, _ = M.bfgs(rosen, np.array([-1.0, 2.0]), iters=150, gtol=1e-8)
    assert np.array_equal(x0, [-1.0, 2.0]) and not np.array_equal(x1, x0) and np.abs(x1 - 1.0).max() > 0.05 and x50 == pytest.approx([1.0, 1.0], abs=1e-4)


def test_a_single_restart_is_allowed_and_zero_restarts_are_not_silently_accepted():
    W, G = rps(10)
    L, (s, u, v) = M.fit_cycle(W, G, lam=0.05, restarts=1)
    assert np.allclose(L, -L.T) and M.sigmoid(L)[0, 1] > 0.8
    with pytest.raises(TypeError):
        M.fit_cycle(W, G, lam=0.05, restarts=0)                            # best bleibt None: kein gültiges Ergebnis
