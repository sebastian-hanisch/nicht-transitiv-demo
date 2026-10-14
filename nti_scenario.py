"""Szenario: Reglerwerte (Settings) und das daraus erzeugte Turnier (Generator Version 1, vollständiger Plan)."""

from dataclasses import dataclass

import nti_constants as C
import turnier_generator as T


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    delta: float = C.DEFAULT_DELTA
    games: int = C.DEFAULT_GAMES            # Partien je Paar
    style: str = C.DEFAULT_STYLE
    c: float = C.DEFAULT_C                  # Zyklenstärke
    lam: float = C.DEFAULT_LAM              # Ridge des Zyklenmodells
    seed: int = C.DEFAULT_SEED


def make_tournament(s: Settings) -> T.Turnier:
    """Deterministisch aus den Reglerwerten: gleiche Eingabe, gleiches Turnier."""
    return T.build_turnier(s.n, s.delta, s.games, "vollstaendig", 0, 1.0, s.c, s.style, s.seed)
