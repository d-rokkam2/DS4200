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
# Keep the latest record per county
df_latest = df.sort_values("Date_parsed").groupby("FIPS", as_index=False).last()

# SVI split (adjust direction if your SVI meaning is inverted)
median_svi = df_latest["Series_Complete_Pop_Pct_SVI"].median()
df_latest["SVI_group"] = np.where(
    df_latest["Series_Complete_Pop_Pct_SVI"] <= median_svi, "Low SVI", "High SVI"
)

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
            style={"display": "flex", "gap": "18px", "alignItems": "center", "flexWrap": "wrap"},
            children=[
                html.Div([
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
                ]),
                html.Div([
                    html.Label("SVI filter:"),
                    dcc.Dropdown(
                        id="svi",
                        options=[
                            {"label": "All", "value": "All"},
                            {"label": "Low SVI", "value": "Low SVI"},
                            {"label": "High SVI", "value": "High SVI"},
                        ],
                        value="All",
                        clearable=False,
                        style={"width": 220},
                    ),
                ]),
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
    Transparent design:
      - Show distributions per Metro_status
      - Overlay mean ± SD bars
      - Full y-axis (0–100)
      - Neutral title and labels
    """
    grouped = df_in.groupby("Metro_status")["Series_Complete_Pop_Pct"]
    means = grouped.mean()
    stds = grouped.std()

    fig = go.Figure()

    # Distributions (violin) for each group, with box & mean line
    for label in sorted(df_in["Metro_status"].dropna().unique()):
        sub = df_in[df_in["Metro_status"] == label]
        fig.add_trace(
            go.Violin(
                x=[label] * len(sub),
                y=sub["Series_Complete_Pop_Pct"],
                name=label,
                box_visible=True,
                meanline_visible=True,
                spanmode="hard",
                opacity=0.55,
                hovertemplate="%{x}<br>%{y:.1f}% complete<extra></extra>",
            )
        )

    # Mean ± SD overlay bars
    fig.add_trace(
        go.Bar(
            x=means.index,
            y=means.values,
            error_y=dict(type="data", array=stds.values, visible=True),
            name="Mean ± SD",
            marker=dict(color="black"),
            opacity=0.7,
            hovertemplate="%{x}<br>Mean: %{y:.1f}%<extra></extra>",
        )
    )

    # Overall average reference line
    overall = df_in["Series_Complete_Pop_Pct"].mean()
    fig.add_hline(
        y=overall,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Overall average: {overall:.1f}%",
        annotation_position="top left",
        annotation_font_color="gray",
    )

    # Optional gap text if both labels present
    if "Metro" in means.index and any(lbl != "Metro" for lbl in means.index):
        nonmetro = [lbl for lbl in means.index if lbl != "Metro"][0]
        gap = means["Metro"] - means[nonmetro]
        fig.add_annotation(
            x=0.5, y=max(means.values) + 4,
            text=f"Gap (Metro − Non-metro): {gap:.1f} pp",
            showarrow=False,
            font=dict(size=13),
        )

    fig.update_layout(
        title="White Hat: Vaccination Completion by Metro Status (distribution + uncertainty)",
        yaxis_title="Series Complete (%)",
        xaxis_title="Metro Status",
        violingap=0.3,
        template="simple_white",
        legend_title="Legend",
        margin=dict(l=40, r=20, t=60, b=60),
        yaxis=dict(range=[0, 100]),
    )
    return fig


def black_hat_figure(df_in: pd.DataFrame) -> go.Figure:
    """
    Obscured design:
      - Collapse all subgroups to a single average (no metro split)
      - KPI style label to anchor attention on one number
      - Very tight y-axis band around mean to minimize perceived variation
      - Positive framing title
      - Minimal ticks, no gridlines, one calm color
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

    pad = 1.5  # tighten to reinforce "parity"
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
        margin=dict(l=40, r=20, t=60, b=60),
    )

    # Big KPI label
    fig.add_annotation(
        x=0, y=overall, text=f"{overall:.1f}%",
        showarrow=False, yshift=18, font=dict(size=20, color="#2F3B4C")
    )
    return fig


# -----------------------------
# Callback
# -----------------------------
@app.callback(
    Output("viz", "figure"),
    Output("notes", "children"),
    Input("mode", "value"),
    Input("svi", "value"),
)
def render(mode, svi_choice):
    # Filter by SVI
    if svi_choice != "All":
        data_view = df_latest[df_latest["SVI_group"] == svi_choice].copy()
    else:
        data_view = df_latest.copy()

    if mode == "white":
        fig = white_hat_figure(data_view)
        msg = [
            html.B("White Hat choices: "),
            "subgroup breakdown (Metro vs Non-metro), distribution for each group, and mean ± SD show both central tendency and variability. ",
            "Axis runs 0–100 with an overall average reference line for honest context.",
        ]
        return fig, msg

    # black
    fig = black_hat_figure(data_view)
    msg = [
        html.B("Black Hat tactics: "),
        "collapse all counties to a single KPI, compress the y-axis around the mean, remove subgroup context and uncertainty, ",
        "and use positive framing in the title to suggest parity.",
    ]
    return fig, msg


if __name__ == "__main__":
    app.run(debug=True)
