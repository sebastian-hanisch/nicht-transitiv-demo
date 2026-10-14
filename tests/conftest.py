"""Gemeinsame Fixtures: von Hand gebaute Siegmatrizen, an denen jede Einheit nachgerechnet wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def counts(n, wins):
    """Siegmatrix aus {(i, j): Siege von i gegen j}; Partienmatrix ergibt sich aus der Summe beider Richtungen."""
    W = np.zeros((n, n), dtype=int)
    for (i, j), w in wins.items():
        W[i, j] = w
    return W, W + W.T


def rps(n_each=10):
    """Stein-Schere-Papier: 0 schlägt 1, 1 schlägt 2, 2 schlägt 0, jedes Mal n_each zu 0 (deterministisch)."""
    return counts(3, {(0, 1): n_each, (1, 2): n_each, (2, 0): n_each})
