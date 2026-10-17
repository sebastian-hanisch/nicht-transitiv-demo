"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

import math

SCALE = 400.0 / math.log(10.0)              # Elo-Punkte je logistische Einheit θ
BASE = 1500.0
CYCLE_PRIOR = 0.5                           # Pseudo-Partien je Seite der gesättigten Paarmatrix
LAMBDA_BT = 1e-3                            # Ridge des Bradley-Terry-Modells (fixiert nur das Niveau)

N_MIN, N_MAX, DEFAULT_N = 6, 12, 12
DELTA_MIN, DELTA_MAX, DELTA_STEP, DEFAULT_DELTA = 0.05, 1.0, 0.05, 0.3
GAMES_CHOICES, DEFAULT_GAMES = (1, 2, 4, 8, 16, 32, 64), 16          # Partien je Paar
STYLES = ("ausgewogen", "konter", "beute")                            # Stilverteilungen des Generators
DEFAULT_STYLE = STYLES[0]
C_MIN, C_MAX, C_STEP, DEFAULT_C = 0.0, 1.2, 0.05, 0.4                # Zyklenstärke (Aufschlag auf die Siegwahrscheinlichkeit)
LAM_MIN, LAM_MAX, LAM_STEP, DEFAULT_LAM = 0.05, 3.0, 0.05, 0.3       # Ridge des Zyklenmodells
SEED_MAX, DEFAULT_SEED = 999999, 5

# Feste Instanzen für alle Experimente: nie ein einzelner Seed als Befund.
SWEEP_SEEDS = tuple(range(100000, 100060))
EXP_N, EXP_DELTA = 12, 0.3
GAMES_GRID = (1, 4, 16, 64)
C_BALANCED = (0.0, 0.4, 0.8)
C_UNBALANCED = (0.0, 0.2, 0.4)
C_RANK = (0.0, 0.2, 0.4)
C_DETECT = (0.0, 0.2, 0.4)
BASELINE_DRAWS = 12
EXP_LAM = 0.3

COLOR_BT = "#9aa0a6"
COLOR_CYCLE = "#d98c1f"
COLOR_SAT = "#2F6B65"
COLOR_TABLE = "#8e6bb0"
COLOR_BOUND = "#c0392b"
PALETTE_DETECT = ("#2F6B65", "#d98c1f", "#c0392b")
COLORS = {"Tabelle": COLOR_TABLE, "Bradley-Terry": COLOR_BT, "BT + Zyklus": COLOR_CYCLE, "gesättigt": COLOR_SAT}

_BASE = {"n": DEFAULT_N, "delta": DEFAULT_DELTA, "games": DEFAULT_GAMES, "style": DEFAULT_STYLE, "c": DEFAULT_C, "lam": DEFAULT_LAM, "seed": DEFAULT_SEED}
PRESETS = {
    "Transitiv": {**_BASE, "c": 0.0},
    "Ausgewogener Zyklus": dict(_BASE),
    "Der Beste wird gekontert": {**_BASE, "style": STYLES[1], "c": 0.4},
    "Wenige Partien": {**_BASE, "games": 1},
}
PRESET_HELP = {
    "Transitiv": "Keine Zyklen: Eine Rangstärke erklärt alles, das Zyklenmodell bringt nichts und überpasst höchstens.",
    "Ausgewogener Zyklus": "Stein-Schere-Papier unter gleich vielen Spielern: Die Rangfolge nach der erwarteten Siegquote bleibt, aber Bradley-Terry sagt einzelne Partien schlecht voraus.",
    "Der Beste wird gekontert": "Der Beste hat einen Stil, den die meisten anderen kontern: Die Tabelle ist in sich stimmig und doch gegen die Stärke falsch.",
    "Wenige Partien": "Eine Partie je Paar: Rauschen erzeugt Zyklen, und die gesättigte Matrix sieht nur das Rauschen.",
}
