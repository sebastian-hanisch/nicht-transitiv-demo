"""Modelle der Paarvergleiche, in logistischen Einheiten (θ), nur numpy.

Daten: Siegmatrix W[i, j] (Siege von i gegen j) und Partienmatrix G[i, j] = W[i, j] + W[j, i]. Jedes Modell liefert eine antisymmetrische logit-Matrix L mit P(i schlägt j) = σ(L[i, j]):

  Bradley-Terry   L_ij = s_i − s_j (eine Rangstärke; Ridge λ_BT fixiert nur das Niveau; Newton-Verfahren, strikt konkav)
  BT + Zyklus     L_ij = s_i − s_j + u_i v_j − u_j v_i (eine Rotationsebene; Ridge λ auf alle Parameter; BFGS mit mehreren Zufallsstarts, nicht konvex)
  gesättigt       L_ij = logit((W_ij + a)/(G_ij + 2a)) mit a Pseudo-Partien je Seite (ein Parameter je Paar, geglättet)"""

import numpy as np

import nti_constants as C
import turnier_generator as T


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=float)))


def log1pexp(x):
    return np.logaddexp(0.0, x)


def pairs(n: int):
    return np.triu_indices(n, 1)


def logit_matrix(s) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    return s[:, None] - s[None, :]


def cycle_logit_matrix(s, u, v) -> np.ndarray:
    s, u, v = (np.asarray(a, dtype=float) for a in (s, u, v))
    return s[:, None] - s[None, :] + np.outer(u, v) - np.outer(v, u)


def neg_loglik_cycle(x: np.ndarray, W: np.ndarray, G: np.ndarray, lam: float):
    """(Zielwert, Gradient) von −log L + λ/2 |x|² für x = (s, u, v)."""
    n = len(W)
    s, u, v = x[:n], x[n:2 * n], x[2 * n:]
    i, j = pairs(n)
    d = s[i] - s[j] + u[i] * v[j] - u[j] * v[i]
    w, g = W[i, j], G[i, j]
    f = float(np.sum(w * log1pexp(-d) + (g - w) * log1pexp(d)) + 0.5 * lam * x @ x)
    r = w - g * sigmoid(d)                                              # d(log L)/dd
    gs, gu, gv = np.zeros(n), np.zeros(n), np.zeros(n)
    np.add.at(gs, i, r)
    np.add.at(gs, j, -r)
    np.add.at(gu, i, r * v[j])
    np.add.at(gu, j, -r * v[i])
    np.add.at(gv, j, r * u[i])
    np.add.at(gv, i, -r * u[j])
    return f, -np.concatenate([gs, gu, gv]) + lam * x


def bfgs(fun, x0: np.ndarray, iters: int = 150, gtol: float = 1e-5) -> tuple:
    """BFGS mit Rückwärtsschritten (Armijo); `fun(x)` liefert (Wert, Gradient). Rückgabe: (x, Wert)."""
    x = np.asarray(x0, dtype=float).copy()
    f, g = fun(x)
    H = np.eye(len(x))
    eye = np.eye(len(x))
    for _ in range(iters):
        if np.abs(g).max() < gtol:
            break
        d = -H @ g
        if d @ g >= 0.0:
            H, d = eye.copy(), -g
        t = 1.0
        while True:
            xn = x + t * d
            fn, gn = fun(xn)
            if fn <= f + 1e-4 * t * (g @ d) or t < 1e-12:
                break
            t *= 0.5
        s, y = xn - x, gn - g
        sy = float(s @ y)
        if sy > 1e-12:
            rho = 1.0 / sy
            H = (eye - rho * np.outer(s, y)) @ H @ (eye - rho * np.outer(y, s)) + rho * np.outer(s, s)
        x, f, g = xn, fn, gn
    return x, f


def fit_bt(W: np.ndarray, G: np.ndarray, lam: float = C.LAMBDA_BT, iters: int = 60) -> np.ndarray:
    """Bradley-Terry mit Ridge; Newton-Verfahren."""
    n = len(W)
    i, j = pairs(n)
    w, g = W[i, j], G[i, j]
    s = np.zeros(n)
    for _ in range(iters):
        p = sigmoid(s[i] - s[j])
        r = w - g * p
        grad = np.zeros(n)
        np.add.at(grad, i, r)
        np.add.at(grad, j, -r)
        grad -= lam * s
        h = g * p * (1.0 - p)
        H = lam * np.eye(n)
        np.add.at(H, (i, i), h)
        np.add.at(H, (j, j), h)
        np.add.at(H, (i, j), -h)
        np.add.at(H, (j, i), -h)
        step = np.linalg.solve(H, grad)
        s += step
        if np.abs(step).max() < 1e-10:
            break
    return s


def _normals(rng: T.SplitMix64, size: int) -> np.ndarray:
    u1, u2 = 1.0 - rng.uniform(size), rng.uniform(size)
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)


def fit_cycle(W: np.ndarray, G: np.ndarray, lam: float = C.DEFAULT_LAM, restarts: int = 2, seed: int = 0) -> tuple:
    """BT + ein Zyklus. Rückgabe: (L, (s, u, v)); bester von `restarts` Starts (Start s aus BT, u und v klein zufällig)."""
    n = len(W)
    rng = T.SplitMix64(0x51C1 ^ int(seed))
    s0 = fit_bt(W, G)
    best = None
    for _ in range(restarts):
        x0 = np.concatenate([s0, 0.3 * _normals(rng, 2 * n)])
        x, f = bfgs(lambda x: neg_loglik_cycle(x, W, G, lam), x0)
        if best is None or f < best[1]:
            best = (x, f)
    x = best[0]
    s, u, v = x[:n], x[n:2 * n], x[2 * n:]
    return cycle_logit_matrix(s, u, v), (s, u, v)


def saturated_logit(W: np.ndarray, G: np.ndarray, prior: float = C.CYCLE_PRIOR) -> np.ndarray:
    p = (W + prior) / (G + 2.0 * prior)
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    L = np.log(p / (1.0 - p))
    np.fill_diagonal(L, 0.0)
    return (L - L.T) / 2.0                                              # antisymmetrisch (bei G = 0 gleich 0)


def log_loss(L: np.ndarray, P: np.ndarray) -> float:
    """Mittlerer Log-Verlust der Modellwahrscheinlichkeiten σ(L) gegen die wahre Matrix P über die Paare i < j (frische Partien aus P)."""
    i, j = pairs(len(P))
    d, p = L[i, j], P[i, j]
    return float(np.mean(p * log1pexp(-d) + (1.0 - p) * log1pexp(d)))


def entropy_bound(P: np.ndarray) -> float:
    """Kleinster erreichbarer Log-Verlust: die mittlere Entropie der wahren Wahrscheinlichkeiten."""
    i, j = pairs(len(P))
    p = np.clip(P[i, j], 1e-12, 1.0 - 1e-12)
    return float(np.mean(-p * np.log(p) - (1.0 - p) * np.log(1.0 - p)))


def expected_win_rate(L: np.ndarray) -> np.ndarray:
    """Mittlere Siegwahrscheinlichkeit gegen ein gleichverteiltes Feld der übrigen Spieler."""
    n = len(L)
    p = sigmoid(L)
    np.fill_diagonal(p, 0.0)
    return p.sum(axis=1) / (n - 1)


def win_table_rate(W: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Siegquote der Tabelle (Siege je gespielte Partie)."""
    return W.sum(axis=1) / np.maximum(G.sum(axis=1), 1)
