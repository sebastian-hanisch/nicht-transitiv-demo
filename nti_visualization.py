"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import nti_constants as C
import nti_evaluation as ev


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_matrix(a: ev.Analysis):
    """Beobachtete Siegquote der Zeile gegen die Spalte; Spieler nach der Generator-Stärke geordnet (stärkster oben). Ein Zyklus zeigt sich als Block, der mit der Stärke nicht wächst."""
    t = a.tournament
    order = np.argsort(-t.strengths)
    G = np.maximum(t.games, 1)
    R = (t.wins / G)[np.ix_(order, order)]
    np.fill_diagonal(R, np.nan)
    labels = [t.names[i] for i in order]
    fig = go.Figure(go.Heatmap(z=R, x=labels, y=labels, zmin=0, zmax=1, colorscale="RdBu", colorbar=dict(title="Siegquote"),
                               hovertemplate="%{y} gegen %{x}: %{z:.2f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title_text="Zeile (nach Stärke geordnet)")
    fig.update_xaxes(title_text="Spalte")
    return _base(fig, height=420)


def build_loss(a: ev.Analysis):
    names = list(ev.MODELS)
    fig = go.Figure(go.Bar(x=names, y=[a.loss[k] for k in names], marker_color=[C.COLORS[k] for k in names], text=[ev.de(a.loss[k], 3) for k in names], textposition="auto",
                           hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    fig.add_hline(y=a.bound, line_dash="dash", line_color=C.COLOR_BOUND, annotation_text="Untergrenze (Entropie)", annotation_position="bottom right")
    fig.update_yaxes(title_text="Log-Verlust (kleiner ist besser)", range=[0, max(0.8, max(a.loss.values()) * 1.1)])
    return _base(fig, height=300)


def build_places(a: ev.Analysis):
    names = ["Tabelle", *ev.MODELS]
    fig = go.Figure(go.Bar(x=names, y=[a.place[k] for k in names], marker_color=[C.COLORS[k] for k in names], text=[str(a.place[k]) for k in names], textposition="auto",
                           hovertemplate="%{x}: Platz %{y}<extra></extra>"))
    fig.update_yaxes(title_text="Platz des wahren Besten (1 = ganz oben)", autorange="reversed")
    return _base(fig, height=300)


def build_predict(rows, title_c: str = "Zyklenstärke c"):
    """Log-Verlust je Partien je Paar k, drei Modelle, eine Facette je Zyklenstärke (als gruppierte Linien)."""
    cs = sorted({r["c"] for r in rows})
    fig = go.Figure()
    dashes = {cs[i]: d for i, d in enumerate(("solid", "dash", "dot"))}
    for name in ev.MODELS:
        for c in cs:
            sel = sorted((r for r in rows if r["c"] == c), key=lambda r: r["k"])
            fig.add_trace(go.Scatter(x=[str(r["k"]) for r in sel], y=[r["loss"][name] for r in sel], mode="lines+markers", name=f"{name}, c = {ev.de(c, 1)}",
                                     line=dict(color=C.COLORS[name], width=2.5, dash=dashes[c]), hovertemplate=f"{name}, c = {ev.de(c, 1)}, k = %{{x}}: %{{y:.3f}}<extra></extra>"))
    fig.update_xaxes(title_text="Partien je Paar k", type="category")
    fig.update_yaxes(title_text="Log-Verlust gegen die wahre Matrix")
    return _base(fig, height=420)


def build_rank(rows):
    styles = list(dict.fromkeys(r["style"] for r in rows))
    fig = go.Figure()
    shades = {"ausgewogen": "#9cc5bf", "konter": C.COLOR_BOUND, "beute": "#8e6bb0"}
    for style in styles:
        sel = [r for r in rows if r["style"] == style]
        fig.add_trace(go.Bar(x=[f"c = {ev.de(r['c'], 1)}" for r in sel], y=[r["place"] for r in sel], name=style, marker_color=shades.get(style, "#888888"),
                             text=[ev.de(r["place"], 1) for r in sel], textposition="auto", hovertemplate=style + ", %{x}: Platz %{y:.1f}<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Platz des wahren Besten nach der erwarteten Siegquote", autorange="reversed")
    return _base(fig)


def build_detect(rows):
    cs = sorted({r["c"] for r in rows})
    fig = go.Figure()
    for k_c, c in enumerate(cs):
        sel = sorted((r for r in rows if r["c"] == c), key=lambda r: r["k"])
        fig.add_trace(go.Scatter(x=[str(r["k"]) for r in sel], y=[r["data"] for r in sel], mode="lines+markers", name=f"aus Daten, c = {ev.de(c, 1)}",
                                 line=dict(color=C.PALETTE_DETECT[k_c], width=3), hovertemplate="k = %{x}: %{y:.3f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[str(r["k"]) for r in sel], y=[r["baseline"] for r in sel], mode="lines", name=f"Rauschgrundlinie, c = {ev.de(c, 1)}",
                                 line=dict(color=C.PALETTE_DETECT[k_c], width=1.5, dash="dot"), hovertemplate="k = %{x}: %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title_text="Partien je Paar k", type="category")
    fig.update_yaxes(title_text="Zyklenanteil der logit-Matrix")
    return _base(fig, height=420)


def build_nash(rows):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f"c = {ev.de(r['c'], 1)}" for r in rows], y=[r["place_e"] for r in rows], name="erwartete Siegquote", marker_color=C.COLOR_TABLE))
    fig.add_trace(go.Bar(x=[f"c = {ev.de(r['c'], 1)}" for r in rows], y=[r["place_nash"] for r in rows], name="Nash-Wertung", marker_color=C.COLOR_SAT))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Platz des wahren Besten (1 = ganz oben)", autorange="reversed")
    return _base(fig, height=320)
