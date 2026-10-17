"""Zyklen sichtbar machen: Anteil der logit-Matrix, den keine Rangstärke erklärt (Rest der Kleinste-Quadrate-Anpassung s_i − s_j, im Sinn der Hodge-Zerlegung des Vergleichsgraphen),
zyklische Dreiergruppen der beobachteten Mehrheitsrelation und die **Rauschgrundlinie**: Rauschen erzeugt Zyklen, auch wenn die Wahrheit transitiv ist; die Grundlinie ist der Wert
derselben Maße auf Partien, die aus dem angepassten transitiven Modell gezogen sind (parametrischer Bootstrap mit derselben Partienmatrix)."""

import numpy as np

import nti_constants as C
import nti_models as M
import turnier_generator as T


def antisymmetric(L: np.ndarray) -> np.ndarray:
    return (L - L.T) / 2.0


def transitive_part(L: np.ndarray) -> np.ndarray:
    """Beste Rangstärken-Anpassung: s_i = Zeilenmittel der antisymmetrischen Matrix (Kleinste Quadrate für A_ij ≈ s_i − s_j)."""
    A = antisymmetric(L)
    s = A.mean(axis=1)
    return s[:, None] - s[None, :]


def cycle_share(L: np.ndarray) -> float:
    """Anteil der quadratischen Norm der antisymmetrischen Matrix, der nach Abzug der besten Rangstärke übrig bleibt (0 = transitiv, 1 = reiner Zyklus)."""
    A = antisymmetric(L)
    total = float((A ** 2).sum())
    if total <= 1e-18:
        return 0.0
    R = A - transitive_part(L)
    return float((R ** 2).sum() / total)


def cyclic_triples(W: np.ndarray, G: np.ndarray) -> tuple:
    """(zyklische Dreiergruppen, entschiedene Dreiergruppen): entschieden heißt, in allen drei Paaren hat eine Seite mehr Siege; zyklisch, wenn die Mehrheiten im Kreis laufen."""
    B = (W > W.T).astype(float)
    np.fill_diagonal(B, 0.0)
    D = B + B.T
    decided = float(np.trace(D @ D @ D)) / 6.0
    cyclic = float(np.trace(B @ B @ B)) / 3.0
    return int(round(cyclic)), int(round(decided))


def triple_share(W: np.ndarray, G: np.ndarray) -> float:
    cyc, dec = cyclic_triples(W, G)
    return cyc / dec if dec else 0.0


def noise_baseline(W: np.ndarray, G: np.ndarray, draws: int = 12, seed: int = 0) -> dict:
    """Mittel von Zyklenanteil und Anteil zyklischer Dreiergruppen auf Partien, die aus dem angepassten Bradley-Terry-Modell gezogen sind (gleiche Partienmatrix)."""
    n = len(W)
    s = M.fit_bt(W, G, lam=C.BASELINE_LAM)
    P = M.sigmoid(M.logit_matrix(s))
    rng = T.SplitMix64(0xB45E ^ int(seed))
    shares, triples = [], []
    for _ in range(draws):
        Wd = T.play(G.astype(int), P, rng)
        shares.append(cycle_share(M.saturated_logit(Wd, G)))
        triples.append(triple_share(Wd, G))
    return {"share": float(np.mean(shares)), "triples": float(np.mean(triples))}
