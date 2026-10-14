"""Orakel für Zyklen und Rangfolge: Bradley-Terry und Zyklenmodell gegen scipy, Zyklenanteil gegen eine unabhängige Kleinste-Quadrate-Rechnung über die Paardifferenzen,
Nash-Mittelung gegen scipy.linprog (Gleichgewicht, Wert 0), Stein-Schere-Papier von Hand."""

import math

import numpy as np
import pytest

import nti_cycles as Y
import nti_models as M
import nti_nash as N
import turnier_generator as T
from conftest import rps


def _tournament(seed, n=6, k=8, c=0.4, style="ausgewogen"):
    t = T.build_turnier(n, 0.3, k, "vollstaendig", 0, 1.0, c, style, seed)
    return t.wins, t.games, t.P


def test_bt_equals_the_scipy_maximiser():
    optimize = pytest.importorskip("scipy.optimize")
    for seed in range(5):
        W, G, _ = _tournament(seed)
        n = len(W)
        i, j = M.pairs(n)
        lam = 0.2

        def neg(s, W=W, G=G, i=i, j=j, lam=lam):
            d = s[i] - s[j]
            return float(np.sum(W[i, j] * np.logaddexp(0, -d) + (G[i, j] - W[i, j]) * np.logaddexp(0, d)) + 0.5 * lam * s @ s)

        res = optimize.minimize(neg, np.zeros(n), method="BFGS", options={"gtol": 1e-10})
        assert M.fit_bt(W, G, lam=lam) == pytest.approx(res.x, abs=1e-5), seed


def test_cycle_model_reaches_at_least_the_scipy_optimum_from_the_same_start():
    optimize = pytest.importorskip("scipy.optimize")
    for seed in range(4):
        W, G, _ = _tournament(seed, n=5, k=10, c=0.6)
        n = len(W)
        rng = np.random.default_rng(seed)
        x0 = np.concatenate([M.fit_bt(W, G), rng.normal(0, 0.3, 2 * n)])
        ref = optimize.minimize(lambda x: M.neg_loglik_cycle(x, W, G, 0.3), x0, jac=True, method="BFGS", options={"gtol": 1e-9})
        ours, f = M.bfgs(lambda x: M.neg_loglik_cycle(x, W, G, 0.3), x0, iters=500, gtol=1e-9)
        # (u, v) sind nur bis auf eine Drehung der Ebene bestimmt: verglichen wird die logit-Matrix
        Ln = M.cycle_logit_matrix(ours[:n], ours[n:2 * n], ours[2 * n:])
        Lr = M.cycle_logit_matrix(ref.x[:n], ref.x[n:2 * n], ref.x[2 * n:])
        assert f <= ref.fun + 1e-6 and Ln == pytest.approx(Lr, abs=1e-3), seed


def test_cycle_share_equals_an_independent_least_squares_over_the_pair_differences():
    rng = np.random.default_rng(7)
    for _ in range(5):
        n = 6
        X = rng.normal(0, 1, (n, n))
        L = X - X.T
        # unabhängig: s minimiert Σ_{i<j} (L_ij − (s_i − s_j))² über lstsq mit Nebenbedingung Σ s = 0
        rows, rhs = [], []
        for i in range(n):
            for j in range(i + 1, n):
                r = np.zeros(n)
                r[i], r[j] = 1.0, -1.0
                rows.append(r)
                rhs.append(L[i, j])
        rows.append(np.ones(n) * 1e3)
        rhs.append(0.0)
        s = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]
        resid = sum((L[i, j] - (s[i] - s[j])) ** 2 for i in range(n) for j in range(i + 1, n))
        total = sum(L[i, j] ** 2 for i in range(n) for j in range(i + 1, n))
        assert Y.cycle_share(L) == pytest.approx(resid / total, abs=1e-9)


def test_nash_average_is_a_best_response_equilibrium_like_scipys_linprog():
    optimize = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(11)
    for _ in range(4):
        n = 6
        X = rng.uniform(0.1, 0.9, (n, n))
        P = np.triu(X, 1)
        P = P + np.tril(1 - P.T, -1) + 0.5 * np.eye(n)
        A = N.payoff(P)
        # Wert des Spiels über LP: max v s.t. A^T q >= v ... (symmetrisch: Wert 0)
        c = np.zeros(n + 1)
        c[-1] = -1.0
        res = optimize.linprog(c, A_ub=np.hstack([-A.T, np.ones((n, 1))]), b_ub=np.zeros(n), A_eq=np.hstack([np.ones((1, n)), np.zeros((1, 1))]), b_eq=[1.0],
                               bounds=[(0, None)] * n + [(None, None)])
        assert res.x[-1] == pytest.approx(0.0, abs=1e-8)                              # Wert eines symmetrischen Nullsummenspiels
        p, rating, expl = N.nash_average(P, steps=30000)
        assert expl < 0.01 and rating.max() < 0.01
        # kein reiner Gegner gewinnt im Mittel gegen p mehr als den LP-Wert 0 (bis auf die Löser-Toleranz)
        assert (A @ p).max() < 0.01


def test_mini_instance_rock_paper_scissors_by_hand():
    W, G = rps(10)
    assert Y.cyclic_triples(W, G) == (1, 1) and Y.cycle_share(M.saturated_logit(W, G)) == pytest.approx(1.0, abs=1e-9)
    P = np.array([[0.5, 0.75, 0.25], [0.25, 0.5, 0.75], [0.75, 0.25, 0.5]])
    p, rating, _ = N.nash_average(P, steps=5000)
    assert p == pytest.approx([1 / 3] * 3, abs=0.02)
    # Bradley-Terry sagt jede Partie mit 0,5 voraus: Log-Verlust ln 2 gegen die Entropie der wahren Matrix
    Lbt = M.logit_matrix(M.fit_bt(W, G))
    assert M.log_loss(Lbt, P) == pytest.approx(math.log(2.0), abs=1e-4)
    assert M.entropy_bound(P) == pytest.approx(-(0.75 * math.log(0.75) + 0.25 * math.log(0.25)))


def test_balanced_cycles_leave_the_ranking_and_break_bt_prediction():
    """Statistische Prüfung (Eigenschaft über feste Läufe): bei ausgewogenen Stilen bleibt der Beste oben, der Log-Verlust von BT liegt weit über dem des Zyklenmodells."""
    gaps, places = [], []
    for seed in range(8):
        t = T.build_turnier(12, 0.3, 16, "vollstaendig", 0, 1.0, 0.4, "ausgewogen", 700 + seed)
        W, G, P = t.wins, t.games, t.P
        gaps.append(M.log_loss(M.logit_matrix(M.fit_bt(W, G)), P) - M.log_loss(M.fit_cycle(W, G, 0.3, seed=seed)[0], P))
        e = P.sum(axis=1) / 11
        places.append(1 + int((e > e[int(np.argmax(t.strengths))]).sum()))
    assert np.mean(gaps) > 0.15 and max(places) <= 2
