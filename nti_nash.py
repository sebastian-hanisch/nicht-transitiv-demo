"""Nash-Mittelung (Balduzzi et al. 2018) für ein symmetrisches Nullsummenspiel mit der antisymmetrischen Auszahlung A = P − ½: Ein Gleichgewicht p (gemischte Strategie) erfüllt
(A p)_i ≤ 0 für alle i, mit Gleichheit auf dem Träger. Die Nash-Wertung eines Spielers ist (A p)_i: 0 auf dem Träger, negativ außerhalb.

Löser: Exponentialgewichte (Hedge) mit Mittelung der Iterierten; ohne scipy. Das Gleichgewicht ist im Allgemeinen nicht eindeutig, der Löser liefert eines (die Mittelung beginnt bei der
Gleichverteilung und bevorzugt breite Träger)."""

import numpy as np


def payoff(P: np.ndarray) -> np.ndarray:
    A = np.asarray(P, dtype=float) - 0.5
    np.fill_diagonal(A, 0.0)
    return (A - A.T) / 2.0


def exploitability(A: np.ndarray, p: np.ndarray) -> float:
    """Wie viel ein bester reiner Gegner im Mittel gegen p gewinnt: max_i (A p)_i; 0 genau im Gleichgewicht."""
    return float(np.max(A @ p))


def nash_average(P: np.ndarray, steps: int = 20000, eta: float = 0.5) -> tuple:
    """Rückgabe: (Gleichgewicht p, Wertungen A p, Ausbeutbarkeit)."""
    A = payoff(P)
    n = len(A)
    p = np.full(n, 1.0 / n)
    avg = np.zeros(n)
    for _ in range(steps):
        avg += p
        z = np.exp(eta * (A @ p) - np.max(eta * (A @ p)))
        p = p * z
        p /= p.sum()
    avg /= steps
    return avg, A @ avg, exploitability(A, avg)
