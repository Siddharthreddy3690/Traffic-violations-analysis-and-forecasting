import os
import numpy as np
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from dash import dash_table

# ML
from prophet import Prophet

# ==========================
# Load Dataset
# ==========================
data_path = os.path.join("data", "Indian_Traffic_Violations.csv")
df = pd.read_csv(data_path)

# Robust date/time parsing
df["Date"] = pd.to_datetime(df.get("Date"), errors="coerce")
if "Time" in df.columns:
    df["Time"] = pd.to_datetime(df["Time"], format="%H:%M", errors="coerce").dt.hour
else:
    df["Time"] = pd.NA

# Ensure required columns exist with safe defaults
for col, default in [
    ("Violation_Type", "Unknown"),
    ("Vehicle_Type", "Unknown"),
    ("Location", "Unknown"),
    ("Fine_Amount", 0),
    ("Penalty_Points", 0),
]:
    if col not in df.columns:
        df[col] = default

# ==========================
# Dash App
# ==========================
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)
app.title = "Traffic Violations Dashboard"

# ==========================
# Global Styles
# ==========================
app.index_string = """
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
        * { font-family: "Times New Roman", Times, serif !important; }
        body { background-color: #f8f9fa; margin: 0; overflow: hidden; }

        /* Sidebar */
        .sidebar {
            background-color: #1d3f5c;
            color: white;
            padding: 14px 12px;
            height: 100vh;
            position: fixed;
            left: 0; top: 0; bottom: 0;
            width: 16.6667%;
            border-right: 1px solid rgba(255,255,255,0.08);
            overflow-y: auto;
        }
        .sidebar h6, .sidebar label, .sidebar .form-label { color: white !important; }
        .kpi-card {
            background-color: #254f73; color: #ecf0f1; border: none; margin-bottom: 10px;
            text-align: center; border-radius: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);
        }

        /* Dropdowns inside sidebar */
        .sidebar .Select,
        .sidebar .Select-control,
        .sidebar .Select-menu-outer,
        .sidebar .Select-menu,
        .sidebar .Select-option,
        .sidebar .Select-placeholder,
        .sidebar .Select--single > .Select-control .Select-value,
        .sidebar .Select-value,
        .sidebar .Select-value-label,
        .sidebar input {
            background-color: #254f73 !important;
            color: #ffffff !important;
            border-color: rgba(255,255,255,0.2) !important;
        }
        .sidebar .Select-option.is-focused { background-color: #2f5f8a !important; color: #fff !important; }
        .sidebar .Select-option.is-selected { background-color: #1d3f5c !important; color: #fff !important; }
        .sidebar .Select-arrow-zone .Select-arrow { border-top-color: #fff !important; }

        /* Right wrapper */
        .right-wrapper {
            margin-left: 16.6667%;
            height: 100vh;
            display: flex;
            flex-direction: column;
            background: #f8f9fa;
        }

        .dashboard-title {
            text-align: center;
            font-weight: bold;
            padding: 15px 10px;
            border-bottom: 2px solid #ddd;
            background: white;
        }

        /* Controls row */
        .top-controls {
            background: white;
            border-bottom: 1px solid #ddd;
            padding: 10px 12px;
            display: flex;
            align-items: center;
            position: relative;
            min-height: 52px;
        }
        .nav-container {
            margin: 0 auto;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            flex-wrap: nowrap;
            overflow-x: auto;
            scrollbar-width: none;
        }
        .nav-container::-webkit-scrollbar { display: none; }

        .toggle-wrap {
            position: absolute;
            right: 10px; top: 8px;
            transform: scale(0.85);
            transform-origin: top right;
        }

        .nav-btn {
            border-radius: 50px;
            background-color: white;
            color: #1d3f5c;
            border: 2px solid #1d3f5c;
            padding: 8px 16px;
            font-weight: bold;
            white-space: nowrap;
            transition: all 0.2s ease;
        }
        .nav-btn:hover { background-color: #e6eef5; }
        .active-btn { background-color: #1d3f5c !important; color: white !important; }

        /* Scrollable content area */
        #content-area {
            flex: 1;
            overflow-y: auto;
            padding: 16px;
            background: #f8f9fa;
        }

        .section-container { width: 100%; max-width: 1400px; margin: 0 auto; }
        .chart-section { padding: 10px 0; }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>
"""

# ==========================
# Helpers
# ==========================
def shorten_labels(text, max_len=12):
    s = str(text)
    return s if len(s) <= max_len else s[:max_len] + "..."

def empty_fig(title, template):
    fig = go.Figure()
    fig.update_layout(
        template=template,
        title=title,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[dict(text="No data for current filters", x=0.5, y=0.5, showarrow=False)]
    )
    return fig

def money_fmt(n):
    try:
        return f"₹{n:,.0f}"
    except Exception:
        return "₹0"

def int_fmt(n):
    try:
        return f"{int(n):,}"
    except Exception:
        return "0"

def apply_filters(_df, violation_types, vehicle_types, cities, start_date, end_date):
    dff = _df.copy()
    if violation_types:
        dff = dff[dff["Violation_Type"].isin(violation_types)]
    if vehicle_types:
        dff = dff[dff["Vehicle_Type"].isin(vehicle_types)]
    if cities:
        dff = dff[dff["Location"].isin(cities)]
    if start_date:
        dff = dff[dff["Date"] >= pd.to_datetime(start_date)]
    if end_date:
        dff = dff[dff["Date"] <= pd.to_datetime(end_date)]
    return dff

# ==========================
# Sidebar
# ==========================
sidebar = html.Div([
    html.H6("Overview", className="text-center fw-bold mb-3"),

    dbc.Card(dbc.CardBody([
        html.Div("Violations", className="mb-1"),
        html.H4(id="kpi-violations", className="fw-bold mb-0")
    ]), className="kpi-card"),

    dbc.Card(dbc.CardBody([
        html.Div("Total Fines", className="mb-1"),
        html.H4(id="kpi-fines", className="fw-bold mb-0")
    ]), className="kpi-card"),

    dbc.Card(dbc.CardBody([
        html.Div("Penalty Points", className="mb-1"),
        html.H4(id="kpi-penalty", className="fw-bold mb-0")
    ]), className="kpi-card"),

    html.Hr(),
    html.H6("Filters", className="text-center fw-bold mb-2"),

    dcc.Dropdown(
        id="violation-filter",
        options=[{"label": v, "value": v} for v in sorted(df["Violation_Type"].dropna().unique())],
        placeholder="Select Violation Type",
        multi=True,
        className="mb-2"
    ),
    dcc.Dropdown(
        id="vehicle-filter",
        options=[{"label": v, "value": v} for v in sorted(df["Vehicle_Type"].dropna().unique())],
        placeholder="Select Vehicle Type",
        multi=True,
        className="mb-2"
    ),
    dcc.Dropdown(
        id="city-filter",
        options=[{"label": v, "value": v} for v in sorted(df["Location"].dropna().unique())],
        placeholder="Select City/Location",
        multi=True,
        className="mb-2"
    ),
    dcc.Dropdown(
        id="start-date",
        options=[{"label": str(d.date()), "value": str(d.date())} for d in sorted(df["Date"].dropna().unique())],
        placeholder="Select Start Date",
        className="mb-2"
    ),
    dcc.Dropdown(
        id="end-date",
        options=[{"label": str(d.date()), "value": str(d.date())} for d in sorted(df["Date"].dropna().unique())],
        placeholder="Select End Date",
        className="mb-2"
    ),
], className="sidebar")

# ==========================
# Sections
# ==========================
overview_section = html.Div(
    id="section-overview",
    children=[
        html.Div([
            dbc.Row([
                dbc.Col(dcc.Graph(id="violation-bar"), width=6),
                dbc.Col(dcc.Graph(id="city-bar"), width=6),
            ], className="gy-3"),
        ], className="section-container"),

        html.Div([
            dbc.Row([
                dbc.Col(dcc.Graph(id="fine-bar"), width=6),
                dbc.Col(dcc.Graph(id="penalty-bar"), width=6),
            ], className="gy-3"),
        ], className="section-container"),

        html.Div([
            dbc.Row([
                dbc.Col(dcc.Graph(id="violation-pie"), width=6),
                dbc.Col(dcc.Graph(id="vehicle-pie"), width=6),
            ], className="gy-3"),
        ], className="section-container"),
    ],
    className="chart-section",
)

# === Trend Section ===
trend_section = html.Div(
    id="section-trend",
    children=[
        html.Div([
            dcc.RadioItems(
                id="trend-granularity",
                options=[
                    {"label": "Daily", "value": "D"},
                    {"label": "Weekly", "value": "W"},
                    {"label": "Monthly", "value": "M"},
                ],
                value="M",
                labelStyle={"display": "inline-block", "marginRight": "12px"}
            )
        ], style={"marginBottom": "12px"}),

        dbc.Row([
            dbc.Col(dcc.Graph(id="trend-violations-line"), width=12),
        ], className="gy-3"),

        dbc.Row([
            dbc.Col(dcc.Graph(id="trend-fp-line"), width=12),
        ], className="gy-3"),
    ],
    className="section-container",
    style={"display": "none"},
)

# === Compare Section ===
compare_section = html.Div(
    id="section-compare",
    children=[
        dbc.Row([
            dbc.Col(dcc.Graph(id="compare-grouped-bar"), width=6),
            dbc.Col(dcc.Graph(id="compare-box-plot"), width=6),
        ], className="gy-3"),
    ],
    className="section-container",
    style={"display": "none"},
)

# === Predict Section (ML + Charts/Table Toggle) ===
predict_section = html.Div(
    id="section-predict",
    children=[
        html.Div([
            dcc.Dropdown(
                id="predict-horizon",
                options=[
                    {"label": "30 Days", "value": 30},
                    {"label": "60 Days", "value": 60},
                    {"label": "90 Days", "value": 90},
                ],
                value=30,
                clearable=False,
                style={"width": "220px", "display": "inline-block", "marginRight": "12px"}
            ),
            dcc.RadioItems(
                id="predict-view-mode",
                options=[
                    {"label": "Charts", "value": "charts"},
                    {"label": "Table", "value": "table"},
                ],
                value="charts",
                labelStyle={"display": "inline-block", "marginRight": "12px"}
            ),
        ], style={"marginBottom": "20px"}),

        # Charts container
        html.Div([
            dbc.Row([
                dbc.Col(dcc.Graph(id="predict-violations-forecast"), width=12),
            ], className="gy-3"),

            dbc.Row([
                dbc.Col(dcc.Graph(id="predict-fines-forecast"), width=12),
            ], className="gy-3"),

            dbc.Row([
                dbc.Col(dcc.Graph(id="predict-risk-indicator"), width=12),
            ], className="gy-3"),
        ], id="predict-charts-wrap", className="section-container"),

        # Table container
        html.Div([
            dash_table.DataTable(
                id="predict-forecast-table",
                columns=[],
                data=[],
                style_table={"overflowX": "auto"},
                page_size=15,
                sort_action="native",
                filter_action="native",
                export_format="csv",
                export_headers="display",
            )
        ], id="predict-table-wrap", className="section-container", style={"display": "none"}),

        # Hidden store to share forecast data to table builder
        dcc.Store(id="predict-forecast-store"),
    ],
    style={"display": "none"},
)

# ==========================
# Right wrapper
# ==========================
right_content = html.Div([
    html.H2("🚦 Traffic Violations Analysis and Forecasting ", className="dashboard-title"),

    html.Div([
        html.Div([
            dbc.Button(" Overview 📊 ", id="btn-overview", n_clicks=0, className="nav-btn active-btn"),
            dbc.Button(" Trend Analysis 📈 ", id="btn-trend", n_clicks=0, className="nav-btn"),
            dbc.Button(" Comparative Analysis ⚖️ ", id="btn-compare", n_clicks=0, className="nav-btn"),
            dbc.Button(" Predictive Analysis 🤖 ", id="btn-predict", n_clicks=0, className="nav-btn"),
        ], className="nav-container"),

        html.Div(
            dbc.Checklist(
                id="theme-toggle",
                options=[{"label": "Dark charts", "value": "dark"}],
                value=[],
                switch=True,
                className="mb-0",
            ),
            className="toggle-wrap"
        ),
    ], className="top-controls"),

    html.Div(
        id="content-area",
        children=[overview_section, trend_section, compare_section, predict_section]
    )
], className="right-wrapper")

# ==========================
# Main layout
# ==========================
app.layout = html.Div([sidebar, right_content])

# Keep validation_layout strictly in sync with actual IDs
app.validation_layout = html.Div([
    app.layout,
    html.Div([
        dcc.Graph(id="violation-bar"),
        dcc.Graph(id="city-bar"),
        dcc.Graph(id="fine-bar"),
        dcc.Graph(id="penalty-bar"),
        dcc.Graph(id="violation-pie"),
        dcc.Graph(id="vehicle-pie"),
        dcc.Graph(id="trend-violations-line"),
        dcc.Graph(id="trend-fp-line"),
        dcc.Graph(id="compare-grouped-bar"),
        dcc.Graph(id="compare-box-plot"),
        # Predictive
        dcc.Dropdown(id="predict-horizon"),
        dcc.RadioItems(id="predict-view-mode"),
        html.Div(id="predict-charts-wrap"),
        html.Div(id="predict-table-wrap"),
        dash_table.DataTable(id="predict-forecast-table"),
        dcc.Store(id="predict-forecast-store"),
        dcc.Graph(id="predict-violations-forecast"),
        dcc.Graph(id="predict-fines-forecast"),
        dcc.Graph(id="predict-risk-indicator"),
    ])
])

# ==========================
# Navigation
# ==========================
@app.callback(
    [
        Output("section-overview", "style"),
        Output("section-trend", "style"),
        Output("section-compare", "style"),
        Output("section-predict", "style"),
        Output("btn-overview", "className"),
        Output("btn-trend", "className"),
        Output("btn-compare", "className"),
        Output("btn-predict", "className"),
        # Reset Predictive view toggle to "charts" whenever tabs change
        Output("predict-view-mode", "value"),
    ],
    [
        Input("btn-overview", "n_clicks"),
        Input("btn-trend", "n_clicks"),
        Input("btn-compare", "n_clicks"),
        Input("btn-predict", "n_clicks"),
    ],
    prevent_initial_call=False
)
def nav_toggle(n1, n2, n3, n4):
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = "btn-overview"
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    show = {"display": "block"}
    hide = {"display": "none"}

    inactive = "nav-btn"
    active = "nav-btn active-btn"

    if button_id == "btn-overview":
        return show, hide, hide, hide, active, inactive, inactive, inactive, "charts"
    elif button_id == "btn-trend":
        return hide, show, hide, hide, inactive, active, inactive, inactive, "charts"
    elif button_id == "btn-compare":
        return hide, hide, show, hide, inactive, inactive, active, inactive, "charts"
    else:
        return hide, hide, hide, show, inactive, inactive, inactive, active, "charts"

# ==========================
# Trend Callbacks
# ==========================
@app.callback(
    Output("trend-violations-line", "figure"),
    [
        Input("trend-granularity", "value"),
        Input("violation-filter", "value"),
        Input("vehicle-filter", "value"),
        Input("city-filter", "value"),
        Input("start-date", "value"),
        Input("end-date", "value"),
        Input("theme-toggle", "value"),
    ],
)
def trend_violations(gran, violation_types, vehicle_types, cities, start_date, end_date, theme_val):
    template = "plotly_dark" if ("dark" in (theme_val or [])) else "plotly_white"
    dff = apply_filters(df, violation_types, vehicle_types, cities, start_date, end_date).dropna(subset=["Date"])

    if dff.empty:
        return empty_fig("Violations over Time", template)

    ts = dff.set_index("Date").resample(gran or "M").size().reset_index(name="Violations")

    if gran == "D":
        ts["Rolling"] = ts["Violations"].rolling(7, min_periods=1).mean()
        roll_label = "7-Day Avg"
    elif gran == "W":
        ts["Rolling"] = ts["Violations"].rolling(4, min_periods=1).mean()
        roll_label = "4-Week Avg"
    else:
        ts["Rolling"] = ts["Violations"]
        roll_label = "Monthly Avg"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts["Date"], y=ts["Violations"], mode="lines+markers", name="Actual",
        hovertemplate="%{x|%b %d, %Y}<br>Violations: %{y:,}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=ts["Date"], y=ts["Rolling"], mode="lines", name=roll_label, line=dict(dash="dash"),
        hovertemplate="%{x|%b %d, %Y}<br>Avg: %{y:.1f}<extra></extra>"
    ))
    # Title added
    fig.update_layout(template=template, title="Violations over Time", xaxis_title=None, yaxis_title=None)
    return fig

@app.callback(
    Output("trend-fp-line", "figure"),
    [
        Input("trend-granularity", "value"),
        Input("violation-filter", "value"),
        Input("vehicle-filter", "value"),
        Input("city-filter", "value"),
        Input("start-date", "value"),
        Input("end-date", "value"),
        Input("theme-toggle", "value"),
    ],
)
def trend_fines_points(gran, violation_types, vehicle_types, cities, start_date, end_date, theme_val):
    template = "plotly_dark" if ("dark" in (theme_val or [])) else "plotly_white"
    dff = apply_filters(df, violation_types, vehicle_types, cities, start_date, end_date).dropna(subset=["Date"])

    if dff.empty:
        return empty_fig("Fines & Penalty Points over Time", template)

    ts = dff.set_index("Date").resample(gran or "M").agg({
        "Fine_Amount": "sum",
        "Penalty_Points": "sum"
    }).reset_index()

    if gran == "D":
        ts["Fine_Roll"] = ts["Fine_Amount"].rolling(7, min_periods=1).mean()
        ts["Penalty_Roll"] = ts["Penalty_Points"].rolling(7, min_periods=1).mean()
        roll_label = "7-Day Avg"
    elif gran == "W":
        ts["Fine_Roll"] = ts["Fine_Amount"].rolling(4, min_periods=1).mean()
        ts["Penalty_Roll"] = ts["Penalty_Points"].rolling(4, min_periods=1).mean()
        roll_label = "4-Week Avg"
    else:
        ts["Fine_Roll"] = ts["Fine_Amount"]
        ts["Penalty_Roll"] = ts["Penalty_Points"]
        roll_label = "Monthly Avg"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts["Date"], y=ts["Fine_Amount"], mode="lines+markers", name="Total Fines",
        hovertemplate="%{x|%b %d, %Y}<br>Fines: ₹%{y:,}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=ts["Date"], y=ts["Penalty_Points"], mode="lines+markers", name="Penalty Points",
        hovertemplate="%{x|%b %d, %Y}<br>Points: %{y:,}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=ts["Date"], y=ts["Fine_Roll"], mode="lines", name=f"Fines ({roll_label})", line=dict(dash="dash"),
        hovertemplate="%{x|%b %d, %Y}<br>Fines Avg: ₹%{y:,.0f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=ts["Date"], y=ts["Penalty_Roll"], mode="lines", name=f"Penalty ({roll_label})", line=dict(dash="dash"),
        hovertemplate="%{x|%b %d, %Y}<br>Points Avg: %{y:,.0f}<extra></extra>"
    ))

    # Title added
    fig.update_layout(template=template, title="Fines & Penalty Points over Time", xaxis_title=None, yaxis_title=None)
    return fig

# ==========================
# Compare Analysis Callbacks
# ==========================
@app.callback(
    Output("compare-grouped-bar", "figure"),
    [
        Input("violation-filter", "value"),
        Input("vehicle-filter", "value"),
        Input("city-filter", "value"),
        Input("start-date", "value"),
        Input("end-date", "value"),
        Input("theme-toggle", "value"),
    ]
)
def update_compare_bar(violation_types, vehicle_types, cities, start_date, end_date, theme_val):
    template = "plotly_dark" if ("dark" in (theme_val or [])) else "plotly_white"
    dff = apply_filters(df, violation_types, vehicle_types, cities, start_date, end_date)

    if dff.empty:
        return empty_fig("Violations by City & Vehicle Type", template)

    counts = dff.groupby(["Location", "Vehicle_Type"]).size().reset_index(name="Count")
    fig = px.bar(
        counts, x="Location", y="Count", color="Vehicle_Type", barmode="group",
        template=template, text="Count", title="Violations by City & Vehicle Type"
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(xaxis_title=None, yaxis_title=None, xaxis_tickangle=-45)
    return fig

@app.callback(
    Output("compare-box-plot", "figure"),
    [
        Input("violation-filter", "value"),
        Input("vehicle-filter", "value"),
        Input("city-filter", "value"),
        Input("start-date", "value"),
        Input("end-date", "value"),
        Input("theme-toggle", "value"),
    ]
)
def update_compare_box(violation_types, vehicle_types, cities, start_date, end_date, theme_val):
    template = "plotly_dark" if ("dark" in (theme_val or [])) else "plotly_white"
    dff = apply_filters(df, violation_types, vehicle_types, cities, start_date, end_date)

    if dff.empty or "Fine_Amount" not in dff.columns:
        return empty_fig("Fine Distribution by Vehicle Type", template)

    fig = px.box(
        dff, x="Vehicle_Type", y="Fine_Amount", template=template,
        points="all", title="Fine Distribution by Vehicle Type"
    )
    fig.update_layout(xaxis_title=None, yaxis_title=None, xaxis_tickangle=-45)
    return fig

# ==========================
# Predictive Analysis (robust Prophet + graceful fallbacks)
#  -> Also stores forecast data for the table view
# ==========================
@app.callback(
    [
        Output("predict-forecast-store", "data"),
        Output("predict-violations-forecast", "figure"),
        Output("predict-fines-forecast", "figure"),
        Output("predict-risk-indicator", "figure"),
    ],
    [
        Input("predict-horizon", "value"),
        Input("violation-filter", "value"),
        Input("vehicle-filter", "value"),
        Input("city-filter", "value"),
        Input("start-date", "value"),
        Input("end-date", "value"),
        Input("theme-toggle", "value"),
    ]
)
def update_predictive(horizon, violation_types, vehicle_types, cities, start_date, end_date, theme_val):
    template = "plotly_dark" if ("dark" in (theme_val or [])) else "plotly_white"
    horizon = int(horizon or 30)

    # 1) Filter data
    dff = apply_filters(df, violation_types, vehicle_types, cities, start_date, end_date).dropna(subset=["Date"])
    if dff.empty:
        empty = empty_fig("No data", template)
        return (
            None,
            empty_fig("Violations Forecast", template),
            empty_fig("Fines Forecast", template),
            empty_fig("Risk Indicator", template),
        )

    # 2) Build the time series with the most detailed feasible frequency
    def make_ts(_dff, freq):
        ts = _dff.set_index("Date").resample(freq).size().rename("y").reset_index()
        ts = ts.rename(columns={"Date": "ds"})
        ts = ts.dropna(subset=["ds"])
        return ts

    ts = make_ts(dff, "D")
    chosen_freq = "D"
    if ts["y"].sum() == 0 or ts.shape[0] < 14:
        ts = make_ts(dff, "W")
        chosen_freq = "W"
    if ts["y"].sum() == 0 or ts.shape[0] < 8:
        ts = make_ts(dff, "M")
        chosen_freq = "M"

    # map horizon (days) to periods in chosen frequency
    if chosen_freq == "D":
        periods = horizon
    elif chosen_freq == "W":
        periods = max(2, horizon // 7)
    else:
        periods = max(2, horizon // 30)

    # 3) Fit Prophet if we have enough signal; else fall back to simple MA forecast
    forecast = None
    used_prophet = False
    try:
        if ts["y"].sum() > 0 and ts.shape[0] >= 6:
            m = Prophet(
                seasonality_mode="additive",
                yearly_seasonality=True if chosen_freq in ("W", "M") else False,
                weekly_seasonality=True if chosen_freq == "D" else False,
                daily_seasonality=False,
            )
            m.fit(ts)
            future = m.make_future_dataframe(periods=periods, freq=chosen_freq)
            pred = m.predict(future)
            forecast = pred[["ds", "yhat"]].copy()
            used_prophet = True
    except Exception:
        forecast = None
        used_prophet = False

    if forecast is None or not used_prophet:
        # Fallback: simple rolling mean projection
        N = min(7, max(3, ts.shape[0] // 3))
        base_mean = float(ts["y"].tail(N).mean()) if ts.shape[0] else 0.0
        last_ds = ts["ds"].max()
        future_index = pd.date_range(last_ds, periods=periods+1, freq=chosen_freq)[1:]
        forecast = pd.DataFrame({"ds": future_index, "yhat": [base_mean] * len(future_index)})
        hist_part = ts[["ds", "y"]].copy()
        hist_part["yhat"] = hist_part["y"].rolling(N, min_periods=1).mean()
        forecast = pd.concat([hist_part[["ds", "yhat"]], forecast[["ds", "yhat"]]], ignore_index=True)

    # Split forecast into past vs future for plotting
    cutoff = ts["ds"].max()
    future_mask = forecast["ds"] > cutoff
    forecast_future = forecast[future_mask].copy()

    # 4) Violations Forecast chart
    fig_viol = go.Figure()
    fig_viol.add_trace(go.Scatter(
        x=ts["ds"], y=ts["y"], mode="lines+markers", name="Actual",
        hovertemplate="%{x|%b %d, %Y}<br>Violations: %{y:,}<extra></extra>"
    ))
    fig_viol.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Forecast",
        hovertemplate="%{x|%b %d, %Y}<br>Forecast: %{y:,.2f}<extra></extra>"
    ))
    fig_viol.update_layout(template=template, title="Violations Forecast", xaxis_title=None, yaxis_title=None)

    # 5) Fines Forecast: average fine × predicted violations
    avg_fine = dff["Fine_Amount"].mean() if "Fine_Amount" in dff.columns and not dff.empty else 0.0
    fines_ts = dff.set_index("Date").resample(chosen_freq)["Fine_Amount"].sum().reset_index()
    fig_fines = go.Figure()
    fig_fines.add_trace(go.Scatter(
        x=fines_ts["Date"], y=fines_ts["Fine_Amount"], mode="lines+markers", name="Actual Fines",
        hovertemplate="%{x|%b %d, %Y}<br>₹%{y:,.0f}<extra></extra>"
    ))
    fig_fines.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat"] * avg_fine, mode="lines", name="Forecasted Fines",
        hovertemplate="%{x|%b %d, %Y}<br>₹%{y:,.0f}<extra></extra>"
    ))
    fig_fines.update_layout(template=template, title="Fines Forecast", xaxis_title=None, yaxis_title=None)

    # 6) Risk Indicator (gauge)
    forecasted_total = float(forecast_future["yhat"].clip(lower=0).sum()) if not forecast_future.empty else 0.0
    baseline = float(ts["y"].sum()) if ts["y"].sum() > 0 else 1.0
    risk_level = max(0.0, min(1.0, forecasted_total / baseline))

    if risk_level > 0.6:
        risk_text, color = "High", "red"
    elif risk_level > 0.3:
        risk_text, color = "Medium", "orange"
    else:
        risk_text, color = "Low", "green"

    fig_risk = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_level * 100,
        number={"suffix": "%"},
        delta={"reference": 30, "increasing": {"color": "red"}, "decreasing": {"color": "green"}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 30], "color": "lightgreen"},
                {"range": [30, 60], "color": "yellow"},
                {"range": [60, 100], "color": "tomato"},
            ],
        },
        title={"text": f"Risk Indicator — {risk_text}"}
    ))
    fig_risk.update_layout(template=template)

    # 7) Prepare data for the table view (future periods only)
    if forecast_future.empty:
        store_data = None
    else:
        table_df = forecast_future.copy()
        table_df = table_df.assign(
            Date=table_df["ds"].dt.date.astype(str),
            Forecasted_Violations=table_df["yhat"].clip(lower=0).round(2),
            Forecasted_Fines=(table_df["yhat"].clip(lower=0) * avg_fine).round(0),
        )[["Date", "Forecasted_Violations", "Forecasted_Fines"]]
        store_data = {
            "records": table_df.to_dict("records"),
            "meta": {
                "freq": chosen_freq,
                "avg_fine": float(avg_fine)
            }
        }

    return store_data, fig_viol, fig_fines, fig_risk


# ==========================
# Predictive View Mode (Charts/Table) + Table Styling
# ==========================
@app.callback(
    [
        Output("predict-charts-wrap", "style"),
        Output("predict-table-wrap", "style"),
        Output("predict-forecast-table", "columns"),
        Output("predict-forecast-table", "data"),
        Output("predict-forecast-table", "style_header"),
        Output("predict-forecast-table", "style_cell"),
    ],
    [
        Input("predict-view-mode", "value"),
        Input("predict-forecast-store", "data"),
        Input("theme-toggle", "value"),
    ],
)
def toggle_predict_view(view_mode, store_data, theme_val):
    # Visibility
    show = {"display": "block"}
    hide = {"display": "none"}

    charts_style = show if view_mode == "charts" else hide
    table_style = show if view_mode == "table" else hide

    # Defaults
    columns = []
    data = []
    # Table styles for light/dark
    is_dark = "dark" in (theme_val or [])
    header_style = {
        "backgroundColor": "#1f2630" if is_dark else "#f1f3f5",
        "color": "white" if is_dark else "#111",
        "fontWeight": "bold",
        "border": "1px solid #444" if is_dark else "1px solid #ddd",
    }
    cell_style = {
        "backgroundColor": "#111522" if is_dark else "white",
        "color": "white" if is_dark else "black",
        "border": "1px solid #333" if is_dark else "1px solid #ddd",
        "padding": "8px",
        "minWidth": "120px",
        "width": "120px",
        "maxWidth": "220px",
        "whiteSpace": "normal",
        "textAlign": "center",
    }

    # Build table only if we have data and the table is visible
    if view_mode == "table" and store_data and "records" in store_data:
        columns = [
            {"name": "Date", "id": "Date"},
            {"name": "Forecasted Violations", "id": "Forecasted_Violations", "type": "numeric", "format": None},
            {"name": "Forecasted Fines (₹)", "id": "Forecasted_Fines", "type": "numeric", "format": None},
        ]
        data = store_data["records"]

    return charts_style, table_style, columns, data, header_style, cell_style


# ==========================
# Overview KPIs + Charts
# ==========================
@app.callback(
    [
        Output("violation-bar", "figure"),
        Output("city-bar", "figure"),
        Output("fine-bar", "figure"),
        Output("penalty-bar", "figure"),
        Output("violation-pie", "figure"),
        Output("vehicle-pie", "figure"),
        Output("kpi-violations", "children"),
        Output("kpi-fines", "children"),
        Output("kpi-penalty", "children"),
    ],
    [
        Input("violation-filter", "value"),
        Input("vehicle-filter", "value"),
        Input("city-filter", "value"),
        Input("start-date", "value"),
        Input("end-date", "value"),
        Input("theme-toggle", "value"),
    ]
)
def update_dashboard(violation_types, vehicle_types, cities, start_date, end_date, theme_val):
    template = "plotly_dark" if ("dark" in (theme_val or [])) else "plotly_white"
    dff = apply_filters(df, violation_types, vehicle_types, cities, start_date, end_date)

    kpi_violations = int_fmt(dff.shape[0])
    kpi_fines = money_fmt(dff["Fine_Amount"].sum() if not dff.empty else 0)
    kpi_penalty = int_fmt(dff["Penalty_Points"].sum() if not dff.empty else 0)

    if dff.empty:
        empty = empty_fig("No data", template)
        return empty, empty, empty, empty, empty, empty, kpi_violations, kpi_fines, kpi_penalty

    # ---------- Violations by Type (bar with labels) ----------
    violation_counts = dff["Violation_Type"].value_counts().reset_index()
    violation_counts.columns = ["Violation_Type", "Total_Violations"]
    violation_counts["LabelTxt"] = violation_counts["Total_Violations"].map(lambda v: f"{int(v):,}")
    violation_bar = px.bar(
        violation_counts, x="Violation_Type", y="Total_Violations",
        title="Violations by Type", template=template, text="LabelTxt"
    ).update_traces(hovertemplate="<b>%{x}</b><br>Count: %{y:,}<extra></extra>", textposition="outside", cliponaxis=False)
    violation_bar.update_layout(xaxis_title=None, yaxis_title=None, xaxis_tickangle=-45)

    # ---------- Violations by City (bar with labels) ----------
    city_counts = dff["Location"].value_counts().reset_index()
    city_counts.columns = ["Location", "Total_Violations"]
    city_counts["LabelTxt"] = city_counts["Total_Violations"].map(lambda v: f"{int(v):,}")
    city_bar = px.bar(
        city_counts, x="Location", y="Total_Violations",
        title="Violations by City", template=template, text="LabelTxt"
    ).update_traces(hovertemplate="<b>%{x}</b><br>Count: %{y:,}<extra></extra>", textposition="outside", cliponaxis=False)
    city_bar.update_layout(xaxis_title=None, yaxis_title=None, xaxis_tickangle=-45)

    # ---------- Fine Amount by Violation Type (bar with labels) ----------
    fine_amounts = dff.groupby("Violation_Type", dropna=False)["Fine_Amount"].sum().reset_index()
    fine_amounts["LabelTxt"] = fine_amounts["Fine_Amount"].map(lambda v: f"₹{int(v):,}")
    fine_bar = px.bar(
        fine_amounts, x="Violation_Type", y="Fine_Amount",
        title="Fine Amount by Violation Type", template=template, text="LabelTxt"
    ).update_traces(hovertemplate="<b>%{x}</b><br>Total Fine: ₹%{y:,}<extra></extra>", textposition="outside", cliponaxis=False)
    fine_bar.update_layout(xaxis_title=None, yaxis_title=None, xaxis_tickangle=-45)

    # ---------- Penalty Points by Violation Type (bar with labels) ----------
    penalty_points = dff.groupby("Violation_Type", dropna=False)["Penalty_Points"].sum().reset_index()
    penalty_points["LabelTxt"] = penalty_points["Penalty_Points"].map(lambda v: f"{int(v):,}")
    penalty_bar = px.bar(
        penalty_points, x="Violation_Type", y="Penalty_Points",
        title="Penalty Points by Violation Type", template=template, text="LabelTxt"
    ).update_traces(hovertemplate="<b>%{x}</b><br>Total Points: %{y:,}<extra></extra>", textposition="outside", cliponaxis=False)
    penalty_bar.update_layout(xaxis_title=None, yaxis_title=None, xaxis_tickangle=-45)

    # ---------- Pies (bright colors + numeric labels) ----------
    # Build counts again for pie inputs
    violation_counts_pie = dff["Violation_Type"].value_counts().reset_index()
    violation_counts_pie.columns = ["Violation_Type", "Total_Violations"]
    vehicle_counts = dff["Vehicle_Type"].value_counts().reset_index()
    vehicle_counts.columns = ["Vehicle_Type", "Total_Violations"]

    bright_seq = px.colors.qualitative.Bold

    violation_pie = px.pie(
        violation_counts_pie, names="Violation_Type", values="Total_Violations",
        title="Violation Share", template=template, hole=0, color_discrete_sequence=bright_seq
    )
    violation_pie.update_traces(
        textinfo="label+percent+value", textposition="outside",
        hovertemplate="<b>%{label}</b><br>Count: %{value:,} (%{percent})<extra></extra>",
        insidetextorientation="radial"
    )
    violation_pie.update_layout(showlegend=False, uniformtext_minsize=10, uniformtext_mode='hide')

    vehicle_pie = px.pie(
        vehicle_counts, names="Vehicle_Type", values="Total_Violations",
        title="Vehicle Share", template=template, hole=0, color_discrete_sequence=bright_seq
    )
    vehicle_pie.update_traces(
        textinfo="label+percent+value", textposition="outside",
        hovertemplate="<b>%{label}</b><br>Count: %{value:,} (%{percent})<extra></extra>",
        insidetextorientation="radial"
    )
    vehicle_pie.update_layout(showlegend=False, uniformtext_minsize=10, uniformtext_mode='hide')

    return (
        violation_bar, city_bar, fine_bar, penalty_bar,
        violation_pie, vehicle_pie,
        kpi_violations, kpi_fines, kpi_penalty
    )

# ==========================
# Run
# ==========================
if __name__ == "__main__":
    app.run(debug=True)
