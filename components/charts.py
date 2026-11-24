"""Dashboard / insight charts."""
from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from modules import config
from modules.viz_style import style


def mood_distribution(entries: list[dict]):
    counts: dict[str, int] = {}
    for e in entries:
        counts[e["mood"]] = counts.get(e["mood"], 0) + 1
    moods = [m for m in config.MOODS if m in counts]
    fig = go.Figure(go.Bar(x=moods, y=[counts[m] for m in moods],
                           marker_color=[config.MOOD_COLORS[m] for m in moods]))
    fig.update_yaxes(title="Entries", dtick=1)
    return style(fig, "Mood distribution (user-selected moods)", 300)


def timeline(entries: list[dict]):
    es = sorted(entries, key=lambda e: e["entry_date"])
    fig = go.Figure(go.Scatter(
        x=[e["entry_date"] for e in es], y=[config.MOODS.index(e["mood"]) for e in es],
        mode="markers+lines", line=dict(color="rgba(129,140,248,.35)", width=1),
        marker=dict(size=16, color=[config.MOOD_COLORS[e["mood"]] for e in es],
                    line=dict(color="#0b1020", width=2)),
        text=[f'{e["title"]}<br>{e["mood"]}' for e in es], hoverinfo="text"))
    fig.update_yaxes(tickmode="array", tickvals=list(range(len(config.MOODS))),
                     ticktext=config.MOODS, autorange="reversed")
    return style(fig, "Memory timeline", 330)


SPECTRUM = [[0.0, "#05060f"], [0.25, "#3b1d6e"], [0.5, "#b5367a"], [0.75, "#f58a4b"], [1.0, "#fcf2b8"]]
SCALES = {"Magma": SPECTRUM}


def image_fig(arr: np.ndarray, title: str, scale: str = "Magma", zmin=None, zmax=None,
              colorbar_title: str | None = None):
    fig = px.imshow(arr, color_continuous_scale=SCALES.get(scale, scale), zmin=zmin, zmax=zmax, aspect="equal")
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(coloraxis_colorbar=dict(title=colorbar_title, thickness=10))
    return style(fig, title, 330)


def bar_h(labels, values, title, color="#818cf8", height=320):
    fig = go.Figure(go.Bar(x=values[::-1], y=labels[::-1], orientation="h", marker_color=color))
    return style(fig, title, height)
