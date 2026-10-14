"""PDF-Export der aktuellen Konfiguration und der Kennzahlen (fpdf2, Kernschrift: nur Zeichen aus Latin-1)."""

from fpdf import FPDF

import nti_constants as C
import nti_evaluation as ev


def _latin(text: str) -> str:
    for old, new in (("–", "-"), ("—", "-"), ("τ", "Tau"), ("σ", "sigma"), ("λ", "lambda"), ("µ", "mu"), ("θ", "theta"), ("≈", "ca."), ("≥", ">="), ("≤", "<="),
                     ("→", "->"), ("**", ""), ("€", "EUR"), ("−", "-"), ("„", '"'), ("“", '"')):
        text = text.replace(old, new)
    return text.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: ev.Analysis) -> bytes:
    s, t = a.settings, a.tournament
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", "B", 15)
    pdf.multi_cell(0, 8, _latin("Nicht-Transitivität: wenn A B schlägt, B C und C wieder A"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.ln(2)
    pdf.multi_cell(0, 5.5, _latin(f"Spieler: {s.n}, Stärkeabstand {ev.de(s.delta)}, {s.games} Partien je Paar, Stilverteilung: {s.style}, Zyklenstärke c = {ev.de(s.c)}, "
                                  f"Ridge des Zyklenmodells {ev.de(s.lam)}, Seed: {s.seed}."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    _level, text = ev.verdict_analysis(a)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Kernaussage", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5.5, _latin(text), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    headers = ("Spieler", "Staerke", "Siege", "Partien", "Tabelle", "BT", "BT+Zyklus")
    widths = (24, 24, 22, 22, 26, 26, 30)
    pdf.set_font("Helvetica", "B", 9)
    for h, w in zip(headers, widths):
        pdf.cell(w, 6, h, border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for i in range(len(t.strengths)):
        row = (t.names[i], f"{C.BASE + C.SCALE * (t.strengths[i] - t.strengths.mean()):.0f}", f"{int(t.wins[i].sum())}", f"{int(t.games[i].sum())}",
               f"{100 * a.rate['Tabelle'][i]:.0f} %", f"{100 * a.rate['Bradley-Terry'][i]:.0f} %", f"{100 * a.rate['BT + Zyklus'][i]:.0f} %")
        for value, w in zip(row, widths):
            pdf.cell(w, 6, _latin(value), border=1)
        pdf.ln()
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4.5, _latin(
        "Alles ist simuliert (Stärken und Stilzyklus aus dem Generator der Ranking-Linie); die wahre Stärke ist nur in der Simulation bekannt. Die Spalten BT und BT+Zyklus zeigen "
        "die erwartete Siegquote gegen ein gleichverteiltes Feld. Quelle: Demo von Sebastian Hanisch, sebastianhanisch.net."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
