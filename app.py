"""Nicht-Transitivität – wenn A B schlägt, B C und C wieder A - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Elftes Stück der Ranking-Linie: Spielstile erzeugen Zyklen, die eine einzige Skala nicht abbildet. Sechs Experimente trennen zwei Ziele (Vorhersage einzelner Partien und Rangfolge),
messen, was ein Modell mit Zyklus bringt und kostet, wie schwer Zyklen aus Daten zu sehen sind (Rauschgrundlinie) und was die Nash-Mittelung leistet. Aufbau und Regeln:
DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import nti_constants as C
import nti_evaluation as ev
from nti_pdf_export import build_pdf
from nti_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from nti_scenario import Settings
from nti_visualization import (
    build_detect,
    build_loss,
    build_matrix,
    build_nash,
    build_places,
    build_predict,
    build_rank,
)

st.set_page_config(page_title="Nicht-Transitivität – Sebastian Hanisch", layout="wide")

SHOW = {"success": st.success, "info": st.info, "warning": st.warning}


def verdict(level, text):
    SHOW[level](text)


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return ev.analyse(settings)


@st.cache_data(show_spinner="Experiment rechnet über 60 feste Läufe …")
def _predict_balanced():
    return ev.sweep_predict("ausgewogen", C.C_BALANCED)


@st.cache_data(show_spinner="Experiment rechnet über 60 feste Läufe …")
def _predict_unbalanced():
    return ev.sweep_predict("konter", C.C_UNBALANCED)


@st.cache_data(show_spinner="Experiment rechnet über 60 feste Läufe …")
def _rank():
    return ev.sweep_rank()


@st.cache_data(show_spinner="Experiment rechnet über 60 feste Läufe …")
def _detect():
    return ev.sweep_detect()


@st.cache_data(show_spinner="Experiment rechnet über 60 feste Läufe …")
def _nash():
    return ev.sweep_nash()


st.title("🔄 Nicht-Transitivität: wenn A B schlägt, B C und C wieder A")
st.markdown(
    """
Eine Rangliste setzt voraus, dass Stärke eine **einzige Skala** ist: Wer A schlägt und von B geschlagen wird, steht zwischen den beiden. Spielstile brechen das: **A schlägt B, B
schlägt C, C schlägt A.** Diese Demo ist das **elfte Stück der Ranking-Linie** und trennt zwei Ziele, die man leicht vermischt: die **Vorhersage einzelner Partien** und die
**Rangfolge**. Gemessen wird, wann Zyklen die eine oder die andere zerstören, was ein **Modell mit Zyklus** bringt und kostet, und wie schwer Zyklen aus Daten zu sehen sind, weil
**Rauschen selbst Zyklen erzeugt**.

Weiter unten: sechs **🔬 Experimente** mit Verdikt, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("Eine Skala, ein Zyklus", expanded=True):
    st.markdown(
        """
1. **Transitiv:** P(i schlägt j) = σ(s_i − s_j) mit einer Rangstärke s (Bradley-Terry). Die Matrix der logit-Werte ist ein reines Differenzmuster s_i − s_j.
2. **Ein Zyklus:** Ergänzt man u_i v_j − u_j v_i, bekommt jeder Spieler zusätzlich eine Lage in einer Ebene: A, B, C liegen im Kreis, und jeder schlägt den nächsten. Das ist das
   einfachste Modell, das Stein-Schere-Papier darstellen kann.
3. **Zwei Ziele:** Für die **Vorhersage** zählt, wie gut das Modell einzelne Partien trifft (Log-Verlust). Für die **Rangfolge** zählt, ob der Beste oben steht. Beide hängen an den
   Zyklen, aber an verschiedenen Stellen.
4. **Das Feld zählt:** Mit Zyklen ist die „erwartete Siegquote“ keine Eigenschaft des Spielers allein, sondern hängt vom Feld ab, gegen das gespielt wird.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n = st.slider("Spieler n", *bounds("n_slider"), key="n_slider", help="Wie viele Spieler im Feld sind (jeder spielt gegen jeden).")
    delta = st.slider("Stärkeabstand zwischen Nachbarn", *bounds("delta_slider"), step=C.DELTA_STEP, key="delta_slider",
                      help="Stärkedifferenz zweier benachbarter Spieler (logistische Einheiten): die transitive Komponente.")
    games = st.select_slider("Partien je Paar", C.GAMES_CHOICES, key="games_select", help="Wie oft jedes Paar gegeneinander spielt. Wenige Partien: Rauschen erzeugt Zyklen.")
    style = st.selectbox("Stilverteilung", C.STYLES, key="style_select",
                         help="ausgewogen: gleich viele Spieler je Stil; konter: der Beste hat einen Stil, den die meisten anderen kontern; beute: er hat den Stil, der die meisten schlägt.")
    c = st.slider("Zyklenstärke c", *bounds("c_slider"), step=C.C_STEP, key="c_slider",
                  help="Aufschlag auf die Siegwahrscheinlichkeit durch den Stilzyklus (0 = transitiv); die Wahrscheinlichkeiten bleiben auf [0,02; 0,98] begrenzt.")
    lam = st.slider("Ridge des Zyklenmodells", *bounds("lam_slider"), step=C.LAM_STEP, key="lam_slider",
                    help="Regularisierung der Parameter des Zyklenmodells; klein = mehr Anpassung ans Rauschen, groß = Zyklus verschwindet.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neuen Lauf würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"n_slider": int(n), "delta_slider": round(float(delta), 2), "games_select": int(games), "style_select": style, "c_slider": round(float(c), 2),
                   "lam_slider": round(float(lam), 2), "seed_input": int(seed)})

settings = Settings(int(n), round(float(delta), 2), int(games), style, round(float(c), 2), round(float(lam), 2), int(seed))
a = _analysis(settings)
t = a.tournament

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Das Turnier")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Log-Verlust: Bradley-Terry", ev.de(a.loss["Bradley-Terry"], 3), delta=f"Untergrenze {ev.de(a.bound, 3)}", delta_color="off",
          help="Mittlerer Log-Verlust der Modellwahrscheinlichkeiten gegen die wahre Matrix P (frische Partien). Kleiner ist besser.")
m2.metric("Log-Verlust: BT + Zyklus", ev.de(a.loss["BT + Zyklus"], 3), delta=f"{ev.de(a.loss['BT + Zyklus'] - a.loss['Bradley-Terry'], 3)} gegen BT", delta_color="off")
m3.metric("Platz des wahren Besten (Tabelle)", str(a.place["Tabelle"]), delta=f"BT: Platz {a.place['Bradley-Terry']}", delta_color="off",
          help="Platz des Spielers mit der höchsten Generator-Stärke in der Tabelle der Siegquoten.")
m4.metric("Zyklenanteil aus den Daten", ev.de(a.share_data), delta=f"Rauschgrundlinie {ev.de(a.baseline['share'])}", delta_color="off",
          help="Anteil der logit-Matrix, den keine Rangstärke erklärt; die Grundlinie ist derselbe Wert auf Partien, die aus dem angepassten transitiven Modell gezogen sind.")

st.plotly_chart(build_matrix(a), width="stretch", key="matrix_chart")
st.caption("Beobachtete Siegquote der Zeile gegen die Spalte, nach der Generator-Stärke geordnet. Ohne Zyklen wächst die Farbe von rechts oben nach links unten gleichmäßig; ein Zyklus "
           "zeigt sich als Block, der der Ordnung widerspricht.")
cl, cr = st.columns(2)
with cl:
    st.plotly_chart(build_loss(a), width="stretch", key="loss_chart")
    st.caption("Vorhersage: Log-Verlust der Modelle gegen die wahre Matrix.")
with cr:
    st.plotly_chart(build_places(a), width="stretch", key="places_chart")
    st.caption("Rangfolge: Platz des wahren Besten nach der Tabelle und nach der erwarteten Siegquote der Modelle.")

rows = [{"Spieler": t.names[i], "wahre Stärke (Punkte)": round(C.BASE + C.SCALE * float(t.strengths[i] - t.strengths.mean())), "Stil": int(t.styles[i]),
         "Siege": int(t.wins[i].sum()), "Partien": int(t.games[i].sum()), "Tabelle": f"{100 * a.rate['Tabelle'][i]:.0f} %", "BT": f"{100 * a.rate['Bradley-Terry'][i]:.0f} %",
         "BT + Zyklus": f"{100 * a.rate['BT + Zyklus'][i]:.0f} %", "wahre Erwartung": f"{100 * t.expected[i]:.0f} %"} for i in range(t.n)]
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.caption(f"Zyklische Dreiergruppen (Mehrheitsrelation): {ev.pct(a.triples_data, 1)} % der entschiedenen Dreiergruppen, Rauschgrundlinie {ev.pct(a.baseline['triples'], 1)} %.")
st.download_button("📄 PDF des aktuellen Turniers", data=build_pdf(a), file_name="nicht-transitiv.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Skala oder Zyklus?")
verdict(*ev.verdict_analysis(a))

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Vorhersage bei ausgewogenen Stilen")
st.caption("12 Spieler (feste Instanz, unabhängig vom Regler), Stilverteilung 4/4/4, Zyklenstärke c = 0 / 0,4 / 0,8, 1 / 4 / 16 / 64 Partien je Paar; Log-Verlust gegen die wahre Matrix, "
           "60 feste Läufe, Zyklenmodell mit Ridge 0,3.")
rows1 = _predict_balanced()
st.plotly_chart(build_predict(rows1), width="stretch", key="exp1_chart")
verdict(*ev.verdict_balanced(rows1))
st.dataframe(pd.DataFrame([{"c": ev.de(r["c"], 1), "Partien je Paar": r["k"], **{m: round(r["loss"][m], 3) for m in ev.MODELS}, "Untergrenze": round(r["bound"], 3)} for r in rows1]),
             hide_index=True, width="stretch")

st.subheader("🔬 Experiment 2: Vorhersage bei unausgewogenen Stilen")
st.caption("Dieselbe Instanz mit der Stilverteilung „konter“ (der Beste hat einen Stil, den die meisten anderen kontern), c = 0 / 0,2 / 0,4.")
rows2 = _predict_unbalanced()
st.plotly_chart(build_predict(rows2), width="stretch", key="exp2_chart")
verdict(*ev.verdict_unbalanced(rows2))

st.subheader("🔬 Experiment 3: Die Rangfolge hängt am Feld")
st.caption("Platz des Spielers mit der höchsten Generator-Stärke nach der erwarteten Siegquote gegen das ganze Feld (aus der wahren Matrix, ohne Stichprobenrauschen), je Stilverteilung "
           "und Zyklenstärke; 60 feste Instanzen.")
rows3 = _rank()
st.plotly_chart(build_rank(rows3), width="stretch", key="exp3_chart")
verdict(*ev.verdict_rank(rows3))
st.dataframe(pd.DataFrame([{"Stile": r["style"], "c": ev.de(r["c"], 1), "Platz des Besten": round(r["place"], 2), "τ gegen die Stärke": round(r["tau_s"], 3),
                            "τ ganzes Feld gegen obere Hälfte": round(r["tau_field"], 3)} for r in rows3]), hide_index=True, width="stretch")

st.subheader("🔬 Experiment 4: Zyklen aus Daten sehen")
st.caption("Stilverteilung „konter“; Zyklenanteil der logit-Matrix aus den Daten (gesättigtes Modell) gegen die Rauschgrundlinie, die aus dem angepassten transitiven Modell gezogen wird "
           f"({C.BASELINE_DRAWS} Ziehungen je Instanz), 60 feste Läufe.")
rows4 = _detect()
st.plotly_chart(build_detect(rows4), width="stretch", key="exp4_chart")
verdict(*ev.verdict_detect(rows4))
st.dataframe(pd.DataFrame([{"c": ev.de(r["c"], 1), "Partien je Paar": r["k"], "wahr": round(r["true"], 3), "aus Daten": round(r["data"], 3), "Grundlinie": round(r["baseline"], 3),
                            "zyklische Dreiergruppen": f"{ev.pct(r['triples'], 1)} %", "Grundlinie Dreiergruppen": f"{ev.pct(r['triples_baseline'], 1)} %"} for r in rows4]),
             hide_index=True, width="stretch")

st.subheader("🔬 Experiment 5: Was ein Zyklenmodell bei wenigen Partien kostet")
st.caption("Aus Experiment 1 (c = 0, ausgewogen, transitive Wahrheit): Log-Verlust bei 1, 4, 16 und 64 Partien je Paar.")
verdict(*ev.verdict_price(rows1))

st.subheader("🔬 Experiment 6: Nash-Mittelung (Forschungsmarke)")
st.caption("Stilverteilung „konter“, Wertung aus der wahren Matrix: Gleichgewicht des symmetrischen Nullsummenspiels (Exponentialgewichte mit "
           "Mittelung); Platz des wahren Besten nach der Nash-Wertung gegen die erwartete Siegquote, 60 feste Instanzen. Das Gleichgewicht ist im Allgemeinen nicht eindeutig.")
rows6 = _nash()
st.plotly_chart(build_nash(rows6), width="stretch", key="exp6_chart")
verdict(*ev.verdict_nash(rows6))

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Der Zyklus ist ein Stilzyklus** | Der Generator erzeugt Zyklen aus drei Stilen mit begrenztem Aufschlag; echte Nicht-Transitivität (Strategien, Ligen) kann stärker und anders gebaut sein. | – |
| **Ein Zyklus genügt** | Das Modell hat eine Rotationsebene; mehrere unabhängige Zyklen würden mehr Ebenen brauchen (hier nicht gemessen). | – |
| **Vollständiger Plan** | Mit Lücken im Spielplan ist der Zyklenanteil noch schwerer zu sehen, und die Tabelle misst das Feld (Stück Massey und Colley). | Massey und Colley |
| **Die Rangfolge ist feldunabhängig** | Mit Zyklen gibt es nur eine Rangfolge je Feld; die erwartete Siegquote gegen das ganze Feld ist eine Wahl (Experiment 3). | – |
| **Das Zyklenmodell überpasst nicht** | Bei wenigen Partien passt es das Rauschen an und liegt hinter Bradley-Terry (Experiment 5). | – |
| **Die Nash-Mittelung ist eindeutig** | Das Gleichgewicht ist im Allgemeinen nicht eindeutig; der Löser liefert eines, das Ergebnis hängt an dieser Wahl (Experiment 6). | – |
| **Es zählt der Ausgang** | Stärke des Siegs, Remis und Heimvorteil gehören nicht zum Modell. | Tore, Remis, Heimvorteil |
| **Alles ist simuliert** | Stärken und Stile sind Konstruktionen des Generators; die Aussagen gelten für diese Konstruktion. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Daten.** $W_{ij}$ Siege von $i$ gegen $j$, $G_{ij} = W_{ij} + W_{ji}$ Partien; $P_{ij}$ die wahre Siegwahrscheinlichkeit (Generator: $\sigma(s_i - s_j) + K_{ij}$ mit dem
Stilzyklus $K$, auf $[0{,}02;\, 0{,}98]$ begrenzt).

**Modelle** (jeweils eine antisymmetrische Matrix $L$ mit $P(i \text{ schlägt } j) = \sigma(L_{ij})$):
- *Bradley-Terry:* $L_{ij} = s_i - s_j$, Ridge $10^{-3}$, Newton-Verfahren.
- *BT + Zyklus:* $L_{ij} = s_i - s_j + u_i v_j - u_j v_i$, Ridge $\lambda$ auf $(s, u, v)$, BFGS mit zwei Zufallsstarts (nicht konvex).
- *gesättigt:* $L_{ij} = \operatorname{logit}\bigl((W_{ij} + a)/(G_{ij} + 2a)\bigr)$ mit $a = 0{,}5$.

**Vorhersage.** Mittlerer Log-Verlust gegen die wahre Matrix, $-\tfrac{1}{|\mathcal P|}\sum_{i<j}\bigl[P_{ij}\ln\sigma(L_{ij}) + (1 - P_{ij})\ln\sigma(-L_{ij})\bigr]$; die Untergrenze ist die
mittlere Entropie von $P$.

**Zyklenanteil.** Mit $A = (L - L^\top)/2$ und der besten Rangstärke $s_i = \tfrac1n\sum_j A_{ij}$: $\;\lVert A - (s_i - s_j)\rVert_F^2 / \lVert A \rVert_F^2$ (0 = transitiv, 1 = reiner
Zyklus). Das ist der nicht durch eine Rangstärke erklärte Rest der Kleinste-Quadrate-Anpassung des Vergleichsgraphen (im Sinn der Hodge-Zerlegung).

**Zyklische Dreiergruppen.** Mehrheitsrelation $B_{ij} = [W_{ij} > W_{ji}]$; zyklisch sind die Dreiergruppen mit $B_{ij} = B_{jk} = B_{ki} = 1$ (Zahl $\operatorname{tr}(B^3)/3$), entschieden die
mit allen drei Paaren entschieden ($\operatorname{tr}((B + B^\top)^3)/6$).

**Rauschgrundlinie.** Anpassung von Bradley-Terry an $W$, Ziehung neuer Partien aus dem angepassten Modell mit derselben Matrix $G$, Mittel der Maße über die Ziehungen.

**Nash-Mittelung.** $A = P - \tfrac12$ antisymmetrisch; ein Gleichgewicht $p$ erfüllt $(A p)_i \le 0$ mit Gleichheit auf dem Träger; die Wertung ist $(A p)_i$. Löser: Exponentialgewichte
$p \leftarrow p \cdot \exp(\eta A p)$, normiert, Mittel der Iterierten.

**Zufall.** SplitMix64-Strom des Generators; Zufallsstarts des Zyklenmodells aus demselben Strom (Box-Muller).

**Literatur.**
- Balduzzi, D., Tuyls, K., Pérolat, J., Graepel, T. (2018): Re-evaluating evaluation. *Advances in Neural Information Processing Systems 31* (Nash-Mittelung, mElo; nicht eingesehen).
- Jiang, X., Lim, L.-H., Yao, Y., Ye, Y. (2011): Statistical ranking and combinatorial Hodge theory. *Mathematical Programming* 127, 203–244 (nicht eingesehen).
- Steele, G. L., Lea, D., Flood, C. H. (2014): Fast splittable pseudorandom number generators. *OOPSLA 2014*, 453–472 (Zufallsstrom des Generators).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
