import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go

# -----------------------------
# Load & prepare data
# -----------------------------
df = pd.read_csv("vaccination_minimal_filtered.csv", low_memory=False)
df["Date_parsed"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
# Keep the latest record per county (FIPS)
df_latest = df.sort_values("Date_parsed").groupby("FIPS", as_index=False).last()

# Split SVI at median (flip <=/>= if your SVI meaning is inverted)
median_svi = df_latest["Series_Complete_Pop_Pct_SVI"].median()
df_latest["SVI_group"] = np.where(
    df_latest["Series_Complete_Pop_Pct_SVI"] <= median_svi, "Low SVI", "High SVI"
)

# Build 4-group label
df_latest["group"] = df_latest["Metro_status"] + " / " + df_latest["SVI_group"]
GROUP_ORDER = ["Metro / Low SVI", "Metro / High SVI", "Non-metro / Low SVI", "Non-metro / High SVI"]

# -----------------------------
# App
# -----------------------------
app = dash.Dash(__name__)
app.title = "Ethical Visualization: White Hat vs Black Hat"

app.layout = html.Div(
    style={"fontFamily": "Arial, Helvetica, sans-serif", "maxWidth": "980px", "margin": "24px auto"},
    children=[
        html.H2("Ethical Visualization: COVID-19 Vaccination Disparities"),
        html.Div(
            style={"display": "flex", "gap": "18px", "alignItems": "center"},
            children=[
                html.Label("Mode:"),
                dcc.RadioItems(
                    id="mode",
                    options=[
                        {"label": " White Hat (transparent)", "value": "white"},
                        {"label": " Black Hat (obscured)", "value": "black"},
                    ],
                    value="white",
                    inline=True,
                ),
            ],
        ),
        dcc.Graph(id="viz", config={"displayModeBar": False}, style={"height": "560px"}),
        html.Div(
            id="notes",
            style={
                "marginTop": "10px",
                "padding": "10px",
                "border": "1px solid #e5e5e5",
                "borderRadius": "6px",
                "background": "#fafafa",
                "fontSize": "14px",
            },
        ),
    ],
)

# -----------------------------
# Helpers
# -----------------------------
def white_hat_figure(df_in: pd.DataFrame) -> go.Figure:
    """
    White hat with FOUR groups:
      - Violin distribution per group (Metro/Non × SVI High/Low)
      - Overlay mean ± SD bars
      - Overall average reference line
      - 0–100% y-axis
    """
    # Ensure consistent order
    df_in = df_in[df_in["group"].isin(GROUP_ORDER)].copy()
    df_in["group"] = pd.Categorical(df_in["group"], categories=GROUP_ORDER, ordered=True)

    grouped = df_in.groupby("group")["Series_Complete_Pop_Pct"]
    means = grouped.mean().reindex(GROUP_ORDER)
    stds = grouped.std().reindex(GROUP_ORDER)

    fig = go.Figure()

    # Distributions (one violin per group)
    palette = {
        "Metro / Low SVI": "#3b82f6",      # blue
        "Metro / High SVI": "#60a5fa",
        "Non-metro / Low SVI": "#f59e0b",  # orange
        "Non-metro / High SVI": "#fbbf24",
    }
    for label in GROUP_ORDER:
        sub = df_in[df_in["group"] == label]
        if sub.empty:
            continue
        fig.add_trace(
            go.Violin(
                x=[label] * len(sub),
                y=sub["Series_Complete_Pop_Pct"],
                name=label,
                box_visible=True,
                meanline_visible=True,
                spanmode="hard",
                opacity=0.55,
                fillcolor=palette.get(label, "#999"),
                line_color=palette.get(label, "#999"),
                hovertemplate="%{x}<br>%{y:.1f}% complete<extra></extra>",
            )
        )

    # Mean ± SD overlay bars
    fig.add_trace(
        go.Bar(
            x=means.index.tolist(),
            y=means.values,
            error_y=dict(type="data", array=stds.values, visible=True),
            name="Mean ± SD",
            marker=dict(color="#111"),
            opacity=0.70,
            hovertemplate="%{x}<br>Mean: %{y:.1f}%<extra></extra>",
        )
    )

    # Overall average reference line
    overall = float(df_in["Series_Complete_Pop_Pct"].mean())
    fig.add_hline(
        y=overall,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Overall average: {overall:.1f}%",
        annotation_position="top left",
        annotation_font_color="gray",
    )

    metro_mean = df_in.loc[df_in["group"].str.startswith("Metro"), "Series_Complete_Pop_Pct"].mean()
    nonmetro_mean = df_in.loc[df_in["group"].str.startswith("Non-metro"), "Series_Complete_Pop_Pct"].mean()
    if pd.notna(metro_mean) and pd.notna(nonmetro_mean):
        gap = metro_mean - nonmetro_mean
        fig.add_annotation(
            x=0.5, y=max(means.dropna().values) + 5,
            text=f"Gap (Metro − Non-metro): {gap:.1f} pp",
            showarrow=False,
            font=dict(size=13),
        )

    fig.update_layout(
        title="White Hat: Vaccination Completion by Metro Status and Vulnerability (distribution + uncertainty)",
        yaxis_title="Series Complete (%)",
        xaxis_title=None,
        violingap=0.25,
        template="simple_white",
        legend_title="Legend",
        margin=dict(l=40, r=20, t=70, b=80),
        yaxis=dict(range=[0, 100]),
    )
    return fig


def black_hat_figure(df_in: pd.DataFrame) -> go.Figure:
    """
    Obscured KPI view:
      - Single average bar
      - Tight y-axis band around mean
      - Positive framing
    """
    overall = float(df_in["Series_Complete_Pop_Pct"].mean())

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=["All Counties"],
            y=[overall],
            marker=dict(color="#5B84B1"),
            width=0.6,
            hovertemplate=f"Overall completion: {overall:.1f}%<extra></extra>",
            showlegend=False,
        )
    )

    pad = 1.5
    fig.update_yaxes(
        range=[overall - pad, overall + pad],
        title_text="Series Complete (%)",
        showgrid=False,
        ticks="outside",
        ticklen=4,
        tickfont=dict(size=10),
    )
    fig.update_layout(
        title="Black Hat: Vaccination Rates — Broad Consistency Across Regions",
        template="simple_white",
        margin=dict(l=40, r=20, t=70, b=60),
    )
    fig.add_annotation(
        x=0, y=overall, text=f"{overall:.1f}%",
        showarrow=False, yshift=18, font=dict(size=20, color="#2F3B4C")
    )
    return fig

@app.callback(
    Output("viz", "figure"),
    Output("notes", "children"),
    Input("mode", "value"),
)
def render(mode):
    if mode == "white":
        fig = white_hat_figure(df_latest)
        msg = [
            html.B("White Hat choices: "),
            "four-group breakdown (Metro/Non × SVI High/Low) with distribution per group and mean ± SD, ",
            "overall average reference line, and fixed 0–100% y-axis.",
        ]
        return fig, msg

    fig = black_hat_figure(df_latest)
    msg = [
        html.B("Black Hat tactics: "),
        "collapse all counties to a single KPI, compress the y-axis around the mean, remove subgroup context and uncertainty, ",
        "and use positive framing in the title to suggest parity.",
    ]
    return fig, msg

if __name__ == "__main__":
    app.run(debug=True)
