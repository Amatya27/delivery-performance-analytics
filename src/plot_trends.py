"""Interactive Plotly charts, saved as HTML in reports/.

These are the quick charts I used to explore the data before building the
Tableau dashboard. Open the HTML files in any browser.

Usage:
    python src/plot_trends.py
"""

import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "delivery.db"
REPORTS = ROOT / "reports"


def main():
    REPORTS.mkdir(exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    monthly = pd.read_sql(
        "SELECT * FROM v_monthly_delivery_kpis WHERE delivered_orders >= 100", con
    )
    monitoring = pd.read_sql("SELECT * FROM model_monitoring", con)
    states = pd.read_sql("SELECT * FROM v_state_performance", con)
    con.close()

    # 1. on-time rate and delivery days by month, two y axes
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=monthly["purchase_month"], y=monthly["on_time_rate"], name="On-time rate",
                   mode="lines+markers"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(x=monthly["purchase_month"], y=monthly["avg_delivery_days"], name="Avg delivery days",
               opacity=0.35),
        secondary_y=True,
    )
    fig.update_yaxes(title_text="On-time rate", tickformat=".0%", secondary_y=False)
    fig.update_yaxes(title_text="Avg delivery days", secondary_y=True)
    fig.update_layout(title="On-time delivery by purchase month", hovermode="x unified")
    fig.write_html(REPORTS / "on_time_trend.html", include_plotlyjs="cdn")

    # 2. model monitoring: predicted vs actual late rate on the 2018 test months
    long = monitoring.melt(id_vars="month", value_vars=["actual_late_rate", "avg_predicted_risk"],
                           var_name="series", value_name="rate")
    fig2 = px.line(long, x="month", y="rate", color="series", markers=True,
                   title="Late delivery model: predicted vs actual late rate (2018 test months)")
    fig2.update_yaxes(tickformat=".0%")
    fig2.write_html(REPORTS / "model_monitoring.html", include_plotlyjs="cdn")

    # 3. late rate vs distance by destination state, bubble = order volume
    fig3 = px.scatter(states, x="avg_distance_km", y="late_rate", size="delivered_orders",
                      text="state", hover_name="state_name",
                      title="Late rate vs average shipping distance, by customer state")
    fig3.update_traces(textposition="top center")
    fig3.update_yaxes(tickformat=".0%")
    fig3.write_html(REPORTS / "late_rate_vs_distance.html", include_plotlyjs="cdn")

    print("saved on_time_trend.html, model_monitoring.html, late_rate_vs_distance.html in reports/")


if __name__ == "__main__":
    main()
