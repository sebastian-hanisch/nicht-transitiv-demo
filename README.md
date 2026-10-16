# Nicht-Transitivität: wenn A B schlägt, B C und C wieder A – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-nicht-transitiv-demo.streamlit.app/)**

Elftes Stück der **Ranking-Linie** (Konzepte-Demo; Forschungsmarke für die Nash-Mittelung), Nachfolger von `bradley-terry-demo` und `massey-colley-demo`: Eine Rangliste setzt voraus,
dass Stärke eine einzige Skala ist. Spielstile brechen das (Stein-Schere-Papier). Gemessen wird, **wann Zyklen die Vorhersage einzelner Partien und wann die Rangfolge zerstören**, was ein
**Modell mit einem Zyklus** bringt und kostet, wie schwer sich Zyklen aus Daten sehen lassen (**Rauschen erzeugt selbst Zyklen**) und was die **Nash-Mittelung** leistet. Das Vehikel ist
das neutrale, simulierte Turnier der Wurzel (Stärken und Stilzyklus aus dem Generator Version 1, wortgleich); alles ist simuliert, es gibt keinen echten Datensatz.

## Kernfrage

Wann zerstören Zyklen eine Rangliste? **Zwei Dinge sind zu trennen.** Bei **ausgewogenen** Stilen (4/4/4) und Zyklenstärke c = 0,4 bleibt die Rangfolge unberührt (der Beste auf Platz 1), aber
**Bradley-Terry sagt einzelne Partien schlecht voraus:** Log-Verlust 0,602 bei 16 Partien je Paar, ein Modell mit **einem** Zyklus kommt auf 0,405 (gesättigte Matrix 0,404, Untergrenze 0,381).
Bei **unausgewogenen** Stilen (der Beste hat den Stil, den die meisten kontern) ist es umgekehrt: Die Vorhersage leidet kaum (0,485 gegen 0,477 der Untergrenze bei 64 Partien), aber der
wahre Beste liegt nach der erwarteten Siegquote auf Platz **3,2** (c = 0,2) bzw. **5,8** (c = 0,4), τ gegen die Stärke 0,75 bzw. 0,53. **Zyklen sind schwer zu sehen:** Aus Daten gemessen liegt
der Anteil der logit-Matrix, den keine Rangstärke erklärt, ohne jeden Zyklus bei einer Partie je Paar bei 0,57 und bei 64 Partien noch bei 0,041, gegen 0,075 für c = 0,4 (wahr 0,042).
Das Zyklenmodell **kostet bei wenigen Partien mehr, als es bringt** (0,827 gegen 0,700 bei einer Partie je Paar, ohne Zyklus in der Wahrheit). Die Nash-Mittelung stellt die Rangfolge
nur teilweise her (Platz 3,2 → 2,5 und 5,8 → 5,0).

## Modell

- **Daten:** Siegmatrix W (Siege von i gegen j) und Partienmatrix G (vollständiger Plan, k Partien je Paar); wahre Siegwahrscheinlichkeit P_ij = clip(σ(s_i − s_j) + K_ij, 0,02; 0,98) mit
  dem Stilzyklus K (drei Stile, a schlägt a − 1, Aufschlag ± c) und den Stilverteilungen `ausgewogen`, `konter`, `beute` des Generators.
- **Modelle** (jeweils eine antisymmetrische logit-Matrix L mit P(i schlägt j) = σ(L_ij)): *Bradley-Terry* L_ij = s_i − s_j (Ridge 10⁻³, Newton-Verfahren); *BT + Zyklus*
  L_ij = s_i − s_j + u_i v_j − u_j v_i (eine Rotationsebene, Ridge λ, BFGS mit zwei Zufallsstarts, nicht konvex); *gesättigt* L_ij = logit((W_ij + ½)/(G_ij + 1)).
- **Vorhersage:** mittlerer Log-Verlust der Modellwahrscheinlichkeiten gegen die wahre Matrix P (also auf frischen Partien); Untergrenze ist die mittlere Entropie von P.
- **Rangfolge:** Platz des Spielers mit der höchsten Generator-Stärke nach der erwarteten Siegquote (gegen ein gleichverteiltes Feld) und Kendall τ gegen die Stärke.
- **Zyklenanteil:** Anteil der quadratischen Norm der antisymmetrischen logit-Matrix, der nach Abzug der besten Rangstärke s_i − s_j übrig bleibt (Kleinste Quadrate auf dem Vergleichsgraphen,
  im Sinn der Hodge-Zerlegung); **zyklische Dreiergruppen** der Mehrheitsrelation; **Rauschgrundlinie** durch parametrischen Bootstrap aus dem angepassten transitiven Modell.
- **Nash-Mittelung:** Gleichgewicht des symmetrischen Nullsummenspiels mit der Auszahlung P − ½ (Exponentialgewichte mit Mittelung), Wertung (A p)_i.
- Alle Formeln stehen im 📐-Expander der App.

## Methodik

- **Experimente** über 60 feste Läufe (`SWEEP_SEEDS`, nie ein einzelner Seed), je mit einem live berechneten Verdikt; feste Instanz mit 12 Spielern, Stärkeabstand 0,3, vollständigem Plan,
  Ridge des Zyklenmodells 0,3, unabhängig vom Regler.
- **Zwei Ziele, zwei Maße:** Vorhersage und Rangfolge werden nie mit einem Maß gemessen; Zyklenanteil und zyklische Dreiergruppen immer mit Rauschgrundlinie.
- **Zufall:** `turnier_generator.py` mit SplitMix64-Strom (wortgleich mit den Repos der Linie); Zufallsstarts des Zyklenmodells aus demselben Strom.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: 60 feste Läufe, 12 Spieler, δ = 0,3, vollständiger Plan.

| Frage | Befund | Test |
|---|---|---|
| Vorhersage, ausgewogen (c = 0,4), 16 / 64 Partien je Paar | **BT 0,602 / 0,599**, BT + Zyklus **0,405 / 0,393**, gesättigt 0,404 / 0,388; Untergrenze 0,381. Bei c = 0,8: BT 0,672, Zyklus 0,240. | `test_bt_cannot_predict_balanced_cycles_but_one_cycle_does` |
| Zyklenmodell gegen gesättigt bei 64 Partien | Unterschied unter 0,01 (c = 0,4 und 0,8): ein Zyklus genügt. | `test_the_cycle_model_is_close_to_the_saturated_matrix_with_enough_games` |
| Vorhersage, unausgewogen (konter), c = 0,4, 64 Partien | BT **0,485**, Zyklus 0,483, Untergrenze 0,477; bei c = 0,2: BT 0,520. | `test_with_unbalanced_styles_prediction_barely_suffers` |
| Platz des wahren Besten nach der erwarteten Siegquote, konter, c = 0 / 0,2 / 0,4 | **1,0 / 3,25 / 5,85**, τ gegen die Stärke 1,0 / 0,75 / 0,53; ausgewogen: Platz 1,0, τ 1,0; beute c = 0,4: Platz 1,07. | `test_unbalanced_styles_push_the_best_down_the_table` |
| τ zwischen „ganzes Feld“ und „obere Hälfte“ | 0,91 (konter, c = 0,2), 0,95 (c = 0,4), 0,97 (ausgewogen, c = 0,4): die Rangfolge hängt am Feld. | `test_the_expected_win_rate_depends_on_the_field` |
| Zyklenanteil aus Daten ohne jeden Zyklus (c = 0), k = 1 / 4 / 16 / 64 | **0,57 / 0,31 / 0,13 / 0,04** (reines Rauschen). | `test_noise_alone_fills_the_matrix_with_cycles` |
| Zyklenanteil, konter, c = 0,4, k = 64 | aus Daten **0,075** gegen Grundlinie **0,041**, wahr 0,042; bei k = 16 liegt der Unterschied zur Grundlinie unter 0,01. | `test_the_cycle_signal_exceeds_the_baseline_only_with_many_games` |
| Zyklische Dreiergruppen | c = 0, k = 1: 13,2 % gegen Grundlinie 9,2 %; c = 0,4, k = 64: mehr als das Dreifache der Grundlinie. | `test_cyclic_triples_follow_the_same_picture` |
| Preis des Zyklenmodells (c = 0, ausgewogen), k = 1 / 4 | BT 0,700 / 0,532 gegen **Zyklus 0,827 / 0,594** und gesättigt 0,653 / 0,588. | `test_the_cycle_model_costs_more_than_it_brings_with_few_games` |
| Nash-Mittelung, konter, c = 0,2 / 0,4 | Platz des Besten **2,47 / 5,05** gegen 3,25 / 5,85 nach der Siegquote; Gleichgewicht stützt sich auf 1,0 Spieler. | `test_nash_average_helps_only_partially` |

## Befunde und Korrekturen gegenüber dem Plan

- **Die Vorab-Zahlen verschieben sich um wenige Hundertstel** (Log-Verlust BT bei c = 0,4: 0,602; Platz des Besten 3,17 / 6,03 gegen **3,25 / 5,85**; Zyklenanteil bei k = 64 0,043 gegen
  **0,041**): 60 statt 30 Läufe, eigener Löser (BFGS) statt scipy, zwei statt drei Zufallsstarts.
- **Die Nash-Mittelung hilft hier etwas mehr bei c = 0,2, weniger bei c = 0,4 als in der Vorab-Messung** (2,5 / 5,0 gegen 3,2 / 4,7), und der Träger ist ein einzelner Spieler: In allen
  60 Instanzen gibt es einen Spieler, gegen den kein anderer eine Siegwahrscheinlichkeit über ½ hat (`test_the_equilibrium_is_the_one_player_nobody_beats_in_expectation`); das Gleichgewicht ist **nicht eindeutig**, der Löser (Exponentialgewichte mit Mittelung) liefert eines, ein
  Simplex-Löser ein anderes (Träger 4,2 / 3,3 in der Vorab-Messung). Die Aussage „hilft nur teilweise“ trägt, die Zahl hängt an der Wahl des Gleichgewichts.
- **Das Modell mit mehreren Zyklen ist nicht umgesetzt** (nur eine Rotationsebene), ebenso alpha-Rank; beides steht nur beim Namen.
- **Zwei Löser statt scipy:** Zur Laufzeit gibt es nur numpy; scipy dient nur als Orakel in den Tests.

## Abgrenzung

- **Siegtabelle** (Wurzel): Die Tabelle schätzt die erwartete Siegquote; ihr Fehler bei Zyklen steht dort in E4b. Dieses Stück erklärt, was ein Modell mit Zyklen kann und was es kostet.
- **Bradley-Terry** (Stück 5) und **Massey und Colley** (Stück 4): die transitive Skala und das Residuum; hier die Frage, was im Rest steckt.
- **Nachfolger (nur beim Namen):** Aktives Ranking; Welches Verfahren, gemessen woran?

## Ehrliche Grenzen

- **Der Zyklus des Generators ist eine Konstruktion** (drei Stile, Aufschlag auf [0,02; 0,98] begrenzt); echte Nicht-Transitivität kann stärker und anders gebaut sein.
- **Ein Zyklus genügt hier;** mehrere unabhängige Zyklen sind nicht gemessen.
- **Vollständiger Plan:** Bei Lücken im Spielplan ist der Zyklenanteil schwerer zu sehen; nicht gemessen.
- **Die erwartete Siegquote ist eine Wahl:** gegen ein gleichverteiltes Feld; mit Zyklen hängt die Rangfolge am Feld.
- **Nash-Mittelung:** Gleichgewicht nicht eindeutig; 12 Spieler; Forschungsmarke, kein etablierter Standard.
- **Alles simuliert:** keine Aussage über Strategiespiele oder einen Sport.

## Quellen

- Balduzzi, D., Tuyls, K., Pérolat, J., Graepel, T. (2018): Re-evaluating evaluation. *Advances in Neural Information Processing Systems 31* (Nash-Mittelung, Zyklenmodell mElo;
  nicht eingesehen).
- Jiang, X., Lim, L.-H., Yao, Y., Ye, Y. (2011): Statistical ranking and combinatorial Hodge theory. *Mathematical Programming* 127, 203–244 (Zerlegung des Vergleichsgraphen; nicht
  eingesehen).
- Steele, G. L., Lea, D., Flood, C. H. (2014): Fast splittable pseudorandom number generators. *OOPSLA 2014*, 453–472 (Zufallsstrom des Generators).
- **Nicht herangezogen:** α-Rank (Omidshafiei et al. 2019) und Arbeiten zu Intransitivität in Paarvergleichsdaten (Chen, Joachims 2016); die Zahlen dieser Demo sind Simulation.

## Tests

95 Tests, gut zwei Minuten (die Experimente rechnen über 60 feste Läufe): Einheiten gegen Handwerte (`test_models.py`: Bradley-Terry, Zyklenmodell mit Gradient gegen finite Differenzen,
BFGS, gesättigte Matrix, Log-Verlust, Entropie; `test_cycles_nash.py`: Zyklenanteil, zyklische Dreiergruppen gegen Aufzählen, Rauschgrundlinie, Nash-Mittelung), Generator
(`test_generator.py`, wortgleich), **Orakel** (`test_oracle_nicht_transitiv.py`: Bradley-Terry und Zyklenmodell gegen den BFGS von scipy (die Rotation der Ebene (u, v) ist nur bis auf
eine Drehung bestimmt, verglichen wird die logit-Matrix); Zyklenanteil gegen eine unabhängige Kleinste-Quadrate-Rechnung über die Paardifferenzen; Nash-Mittelung gegen `scipy.linprog`
(Wert 0, keine bessere Antwort); Stein-Schere-Papier von Hand; Eigenschaftsprüfung ausgewogener Zyklen), Maße und Verdikte (`test_evaluation.py`), alle Zahlen dieses README
(`test_claims.py`), Presets, Szenario und PDF (`test_presets.py`), Laufzeit-Importe und AppTest-Rauchtests.

**Mutationstest** (`tools/oracle_mutation.py` der Website, eingeschränkt auf die fünf `nti_`-Module; schnelle Testdateien zuerst, Überlebende gegen die Claims-Tests): **130 Stellen,
85 getötet, 45 überlebt.** Von den 45 sind **7 echte Lücken** (die Grenzen der Iterationen im Newton-Verfahren, im BFGS und bei den Zufallsstarts, die Ziehungen der Rauschgrundlinie, die
Schritte der Nash-Mittelung, die Behandlung von Gleichständen in der Mehrheitsrelation und die Stabilisierung der Exponentialgewichte); sie sind mit Tests geschlossen und auf einer Kopie
einzeln getötet. **15** weitere sind gegen die Claims-Tests getötet (die Suche nach der Zeile im Experiment, die Ränge und der Träger), **23 sind äquivalent:** Grenzfälle bei stetigen
Größen (`<` gegen `<=`), Schutzgrenzen der Wahrscheinlichkeiten (`clip` bei [0,02; 0,98]), das Vorzeichen von ½ in der Auszahlung (die Antisymmetrisierung hebt es auf),
die Normierung der erwarteten Siegquote durch n − 1 (ein konstanter Faktor ändert keinen Platz) und die Vorzeichen im BFGS-Update und im Armijo-Test (nur die Konvergenzgeschwindigkeit).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `turnier_generator.py` | gemeinsamer Generator (wortgleich in allen Repos der Linie; Stärken, Stilzyklus, Spielplan, Partien) |
| `nti_models.py` | Bradley-Terry (Newton), BT + Zyklus (BFGS), gesättigte Matrix, Log-Verlust, Entropie |
| `nti_cycles.py` | Zyklenanteil, zyklische Dreiergruppen, Rauschgrundlinie |
| `nti_nash.py` | Nash-Mittelung (Exponentialgewichte) |
| `nti_scenario.py` | Reglerwerte (`Settings`) und das Turnier |
| `nti_evaluation.py` | Analyse, sechs Experimente über die festen Läufe, Verdikte |
| `nti_presets.py`, `nti_constants.py` | Permalink, Presets, Grenzen |
| `nti_visualization.py`, `nti_pdf_export.py` | Plotly-Abbildungen, PDF-Export |
| `tests/` | Einheiten-, Orakel-, Claims- und App-Tests |

## Bewusst nicht umgesetzt

- **Mehrere Zyklen** (höherer Rang der antisymmetrischen Ergänzung) und **α-Rank**.
- **Lücken im Spielplan** (Staffeln, zufällige Paare) für die Zyklenmaße.
- **Zeitachse** (Zyklen, die sich über die Zeit ändern).

## Lokal ausführen

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` (enthält scipy für das Orakel) und `python -m pytest tests -q`.

Gebaut mit Streamlit, Plotly, numpy, pandas und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Ranking: von der Siegtabelle bis Learning to Rank](https://sebastianhanisch.net/konzepte-ranking.html).
