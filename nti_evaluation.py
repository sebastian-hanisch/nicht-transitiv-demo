"""Maße, Analyse des gezeigten Turniers, sechs Experimente über die festen Läufe (SWEEP_SEEDS) und die Verdikte.

Zwei Ziele werden getrennt gemessen: **Vorhersage** (mittlerer Log-Verlust der Modellwahrscheinlichkeiten gegen die wahre Matrix P, also auf frischen Partien) und **Rangfolge** (Platz des
wahren Besten, Kendall τ gegen die Generator-Stärke bzw. gegen die erwartete Siegquote). Alle Modelle sehen dieselben Partien."""

import functools
import math
from dataclasses import dataclass

import numpy as np

import nti_constants as C
import nti_cycles as Y
import nti_models as M
import nti_nash as N
import turnier_generator as T
from nti_scenario import Settings, make_tournament

NAN = float("nan")


def de(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def pct(x: float, digits: int = 0) -> str:
    return de(100 * x, digits)


def _mean(values) -> float:
    v = [x for x in values if not (isinstance(x, float) and math.isnan(x))]
    return float(sum(v) / len(v)) if v else NAN


def kendall_tau(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    iu = np.triu_indices(len(a), 1)
    sa, sb = np.sign(a[iu[0]] - a[iu[1]]), np.sign(b[iu[0]] - b[iu[1]])
    counted = (sa != 0) & (sb != 0)
    return float((sa[counted] * sb[counted]).mean()) if counted.any() else NAN


def place_of(values, index: int) -> int:
    """Platz von `index` (1 = höchster Wert; Gleichstände werden zugunsten des Spielers gezählt)."""
    values = np.asarray(values, dtype=float)
    return 1 + int((values > values[index]).sum())


# --- Das gezeigte Turnier ---------------------------------------------------------------------------------------------------------------------

MODELS = ("Bradley-Terry", "BT + Zyklus", "gesättigt")


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    tournament: T.Turnier
    logits: dict               # Modell -> L
    loss: dict                 # Modell -> Log-Verlust gegen P
    bound: float               # Entropie der wahren Matrix
    rate: dict                 # "Tabelle" und Modelle -> erwartete Siegquote bzw. Tabellenquote
    best: int                  # Index des wahren Besten (höchste Generator-Stärke)
    place: dict                # Name -> Platz des Besten
    tau_s: dict                # Name -> τ gegen die Generator-Stärke
    tau_e: dict                # Name -> τ gegen die wahre erwartete Siegquote
    share_true: float          # Zyklenanteil der wahren logit-Matrix
    share_data: float          # aus den Daten (gesättigt)
    triples_data: float        # Anteil zyklischer Dreiergruppen der Mehrheitsrelation
    baseline: dict             # Rauschgrundlinie: share, triples


def analyse(s: Settings) -> Analysis:
    t = make_tournament(s)
    W, G, P = t.wins, t.games, t.P
    Lbt = M.logit_matrix(M.fit_bt(W, G))
    Lcyc, _ = M.fit_cycle(W, G, s.lam, seed=s.seed)
    Lsat = M.saturated_logit(W, G)
    logits = {"Bradley-Terry": Lbt, "BT + Zyklus": Lcyc, "gesättigt": Lsat}
    rate = {"Tabelle": M.win_table_rate(W, G), **{k: M.expected_win_rate(L) for k, L in logits.items()}}
    best = int(np.argmax(t.strengths))
    Ptrue = np.clip(P, 1e-6, 1 - 1e-6)
    return Analysis(s, t, logits, {k: M.log_loss(L, P) for k, L in logits.items()}, M.entropy_bound(P), rate, best, {k: place_of(v, best) for k, v in rate.items()},
                    {k: kendall_tau(v, t.strengths) for k, v in rate.items()}, {k: kendall_tau(v, t.expected) for k, v in rate.items()},
                    Y.cycle_share(np.log(Ptrue / (1 - Ptrue))), Y.cycle_share(Lsat), Y.triple_share(W, G), Y.noise_baseline(W, G, C.BASELINE_DRAWS, s.seed))


def verdict_analysis(a: Analysis):
    s = a.settings
    best_model = min(MODELS, key=lambda k: a.loss[k])
    text = (f"Vorhersage (Log-Verlust, kleiner ist besser; Untergrenze {de(a.bound, 3)}): Bradley-Terry {de(a.loss['Bradley-Terry'], 3)}, BT + Zyklus {de(a.loss['BT + Zyklus'], 3)}, "
            f"gesättigt {de(a.loss['gesättigt'], 3)}; am besten **{best_model}**. Rangfolge: Der wahre Beste liegt in der Tabelle auf Platz **{a.place['Tabelle']}**, nach "
            f"Bradley-Terry auf Platz {a.place['Bradley-Terry']}. Zyklenanteil aus den Daten {de(a.share_data)} gegen eine Rauschgrundlinie von {de(a.baseline['share'])} "
            f"(wahr: {de(a.share_true)}).")
    if a.share_data < a.baseline["share"] * 1.15 + 0.01:
        text += " **Die Daten unterscheiden die Zyklen nicht vom Rauschen.**"
    return "info", text


# --- Feste Läufe der Experimente --------------------------------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=2048)
def _tournament(style: str, c: float, k: int, seed: int) -> T.Turnier:
    return T.build_turnier(C.EXP_N, C.EXP_DELTA, k, "vollstaendig", 0, 1.0, c, style, seed)


def sweep_predict(style: str, cs, ks=C.GAMES_GRID) -> list:
    """Experimente 1, 2 und 5: Log-Verlust der drei Modelle je Zyklenstärke c und Partien je Paar k."""
    rows = []
    for c in cs:
        for k in ks:
            acc = {m: [] for m in MODELS}
            bound = []
            for sd in C.SWEEP_SEEDS:
                t = _tournament(style, c, k, sd)
                W, G, P = t.wins, t.games, t.P
                acc["Bradley-Terry"].append(M.log_loss(M.logit_matrix(M.fit_bt(W, G)), P))
                acc["BT + Zyklus"].append(M.log_loss(M.fit_cycle(W, G, C.EXP_LAM, seed=sd)[0], P))
                acc["gesättigt"].append(M.log_loss(M.saturated_logit(W, G), P))
                bound.append(M.entropy_bound(P))
            rows.append({"style": style, "c": c, "k": k, "loss": {m: _mean(v) for m, v in acc.items()}, "bound": _mean(bound)})
    return rows


def sweep_rank(styles=C.STYLES, cs=C.C_RANK) -> list:
    """Experiment 3: Platz des wahren Besten nach der erwarteten Siegquote gegen das ganze Feld und τ (aus der wahren Matrix, ohne Stichprobenrauschen); dazu τ zwischen dem
    ganzen Feld und der oberen Hälfte als Maß der Abhängigkeit vom Feld."""
    rows = []
    for style in styles:
        for c in cs:
            places, tau_s, tau_f = [], [], []
            for sd in C.SWEEP_SEEDS:
                t = _tournament(style, c, 1, sd)
                n = len(t.strengths)
                order = np.argsort(-t.strengths)
                top = order[: n // 2]
                e_all = t.P.sum(axis=1) / (n - 1)
                e_top = np.array([t.P[i, top].sum() / (len(top) - (1 if i in top else 0)) for i in range(n)])
                places.append(float(place_of(e_all, int(order[0]))))
                tau_s.append(kendall_tau(e_all, t.strengths))
                tau_f.append(kendall_tau(e_all, e_top))
            rows.append({"style": style, "c": c, "place": _mean(places), "tau_s": _mean(tau_s), "tau_field": _mean(tau_f)})
    return rows


def sweep_detect(style="konter", cs=C.C_DETECT, ks=C.GAMES_GRID) -> list:
    """Experiment 4: Zyklenanteil aus den Daten gegen die Rauschgrundlinie (parametrischer Bootstrap) und den wahren Anteil."""
    rows = []
    for c in cs:
        true = _mean([Y.cycle_share(_logit_true(_tournament(style, c, 1, sd).P)) for sd in C.SWEEP_SEEDS])
        for k in ks:
            data, base, trip, trip_base = [], [], [], []
            for sd in C.SWEEP_SEEDS:
                t = _tournament(style, c, k, sd)
                data.append(Y.cycle_share(M.saturated_logit(t.wins, t.games)))
                trip.append(Y.triple_share(t.wins, t.games))
                nb = Y.noise_baseline(t.wins, t.games, C.BASELINE_DRAWS, sd)
                base.append(nb["share"])
                trip_base.append(nb["triples"])
            rows.append({"style": style, "c": c, "k": k, "true": true, "data": _mean(data), "baseline": _mean(base), "triples": _mean(trip), "triples_baseline": _mean(trip_base)})
    return rows


def _logit_true(P: np.ndarray) -> np.ndarray:
    P = np.clip(P, 1e-6, 1 - 1e-6)
    return np.log(P / (1 - P))


def sweep_nash(style="konter", cs=(0.2, 0.4)) -> list:
    """Experiment 6: Platz des wahren Besten nach der erwarteten Siegquote und nach der Nash-Wertung (aus der wahren Matrix); Größe des Trägers."""
    rows = []
    for c in cs:
        pe, pn, sup, expl = [], [], [], []
        for sd in C.SWEEP_SEEDS:
            t = _tournament(style, c, 1, sd)
            n = len(t.strengths)
            best = int(np.argmax(t.strengths))
            e = t.P.sum(axis=1) / (n - 1)
            p, rating, ex = N.nash_average(t.P, steps=6000)
            pe.append(float(place_of(e, best)))
            pn.append(float(place_of(rating, best)))
            sup.append(float((p > 0.02).sum()))
            expl.append(ex)
        rows.append({"style": style, "c": c, "place_e": _mean(pe), "place_nash": _mean(pn), "support": _mean(sup), "exploit": _mean(expl)})
    return rows


# --- Verdikte ---------------------------------------------------------------------------------------------------------------------------------

def _loss_row(rows, c, k):
    return next(r for r in rows if r["c"] == c and r["k"] == k)


def verdict_balanced(rows):
    c = C.C_BALANCED[1]
    r16 = _loss_row(rows, c, 16)
    return "info", (f"Bei ausgewogenen Stilen (4/4/4) und Zyklenstärke c = {de(c, 1)} liegt der Log-Verlust bei {C.GAMES_GRID[2]} Partien je Paar für **Bradley-Terry bei "
                    f"{de(r16['loss']['Bradley-Terry'], 3)}**, mit einem Zyklus bei **{de(r16['loss']['BT + Zyklus'], 3)}**, gesättigt bei {de(r16['loss']['gesättigt'], 3)} "
                    f"(Untergrenze {de(r16['bound'], 3)}): Eine Rangstärke kann Stein-Schere-Papier nicht darstellen; ein Zyklus genügt.")


def verdict_unbalanced(rows):
    c = C.C_UNBALANCED[-1]
    r = _loss_row(rows, c, 64)
    return "info", (f"Bei unausgewogenen Stilen (der Beste hat den Stil, den die meisten kontern) liegt der Log-Verlust bei c = {de(c, 1)} und {C.GAMES_GRID[-1]} Partien je Paar bei "
                    f"Bradley-Terry {de(r['loss']['Bradley-Terry'], 3)}, mit Zyklus {de(r['loss']['BT + Zyklus'], 3)}, Untergrenze {de(r['bound'], 3)}: **Die Vorhersage leidet "
                    "kaum**, die Zyklen betreffen nur wenige Paare. Das Problem liegt in der Rangfolge (Experiment 3).")


def verdict_rank(rows):
    def at(style, c):
        return next(r for r in rows if r["style"] == style and r["c"] == c)
    lo, hi = at("konter", C.C_RANK[1]), at("konter", C.C_RANK[2])
    bal = at("ausgewogen", C.C_RANK[2])
    return "warning", (f"Bei unausgewogenen Stilen liegt der wahre Beste nach der erwarteten Siegquote auf Platz **{de(lo['place'], 1)}** (c = {de(C.C_RANK[1], 1)}) bzw. "
                       f"**{de(hi['place'], 1)}** (c = {de(C.C_RANK[2], 1)}), τ gegen die Stärke {de(lo['tau_s'])} bzw. {de(hi['tau_s'])}; bei ausgewogenen Stilen bleibt er auf "
                       f"Platz {de(bal['place'], 0)}. Die Rangfolge nach der Siegquote hängt am Feld (τ zwischen „alle“ und „obere Hälfte“ {de(hi['tau_field'])}): Es gibt keine "
                       "feldunabhängige Rangfolge; sie ändert sich für einige Paare, nicht insgesamt.")


def verdict_detect(rows):
    def at(c, k):
        return next(r for r in rows if r["c"] == c and r["k"] == k)
    zero, strong = at(C.C_DETECT[0], C.GAMES_GRID[-1]), at(C.C_DETECT[-1], C.GAMES_GRID[-1])
    one = at(C.C_DETECT[0], 1)
    return "warning", (f"Zyklen sind schwer zu sehen: Bei einer Partie je Paar liegt der aus Daten gemessene Zyklenanteil ohne jeden Zyklus (c = 0) bei {de(one['data'])}; bei "
                       f"{C.GAMES_GRID[-1]} Partien sinkt dieses Rauschen auf {de(zero['data'], 3)}, und für c = {de(C.C_DETECT[-1], 1)} liegt der Wert bei {de(strong['data'], 3)} "
                       f"(Grundlinie {de(strong['baseline'], 3)}, wahr {de(strong['true'], 3)}). Ohne die Grundlinie wirkte schon reines Rauschen „voller Zyklen“.")


def verdict_price(rows_bal):
    r0, r1 = _loss_row(rows_bal, C.C_BALANCED[0], 1), _loss_row(rows_bal, C.C_BALANCED[0], 4)
    return "warning", (f"Ohne Zyklus in der Wahrheit kostet das Zyklenmodell bei wenigen Partien mehr, als es bringt: Bei c = 0 liegt sein Log-Verlust bei einer Partie je Paar bei "
                       f"**{de(r0['loss']['BT + Zyklus'], 3)} gegen {de(r0['loss']['Bradley-Terry'], 3)}** bei Bradley-Terry, bei 4 Partien {de(r1['loss']['BT + Zyklus'], 3)} gegen "
                       f"{de(r1['loss']['Bradley-Terry'], 3)}. Es passt das Rauschen an; bei starken ausgewogenen Zyklen lohnt es sich dagegen schon bei einer bis vier Partien je Paar.")


def verdict_nash(rows):
    lo, hi = rows[0], rows[-1]
    return "info", (f"Nash-Mittelung (Forschungsmarke, kein etablierter Standard): Der wahre Beste liegt nach der erwarteten Siegquote auf Platz {de(lo['place_e'], 1)} / "
                    f"{de(hi['place_e'], 1)}, nach der Nash-Wertung auf Platz **{de(lo['place_nash'], 1)} / {de(hi['place_nash'], 1)}** (c = {de(lo['c'], 1)} / {de(hi['c'], 1)}); "
                    f"das Gleichgewicht stützt sich im Mittel auf {de(lo['support'], 1)} / {de(hi['support'], 1)} Spieler. Es verbessert den Platz des Besten nur um {de(lo['place_e'] - lo['place_nash'], 1)} / {de(hi['place_e'] - hi['place_nash'], 1)} Plätze und stellt die Rangfolge insgesamt nicht her: "
                    "Wer gekontert wird, hat auch im Gleichgewicht keinen Platz.")
