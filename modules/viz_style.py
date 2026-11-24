"""Shared Plotly styling (dark navy/indigo theme)."""
import plotly.graph_objects as go

ACCENT = "#818cf8"
ACCENT2 = "#38bdf8"
PINK = "#f472b6"
TEXT = "#e2e8f0"
GRID = "rgba(148,163,184,0.15)"


def style(fig: go.Figure, title: str | None = None, height: int = 300) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=TEXT)) if title else None,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.35)",
        font=dict(color=TEXT, family="Inter, Segoe UI, sans-serif", size=12),
        margin=dict(l=50, r=20, t=45 if title else 15, b=40), height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig
