"""
Smart Building Energy Consumption Dashboard
Author: Douglas | Eagle Technologies LTD | MSc Data Analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Building Energy Dashboard",
    page_icon="⚡",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .kpi-box {
        background: #1e2a3a;
        border-radius: 12px;
        padding: 18px 24px;
        text-align: center;
        border-left: 4px solid #00c8ff;
    }
    .kpi-label { color: #8fa8c8; font-size: 13px; font-weight: 600; margin-bottom: 4px; }
    .kpi-value { color: #ffffff; font-size: 28px; font-weight: 700; }
    .kpi-sub   { color: #00c8ff; font-size: 12px; margin-top: 2px; }
    section[data-testid="stSidebar"] { background-color: #12212f; }
</style>
""", unsafe_allow_html=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_excel("Smart_building_energy_Consumption_Model.xlsx",
                       sheet_name="BuildingData_Clean")
    # Keep only the 16 real columns
    df = df.iloc[:, :16]
    df.columns = [
        "Timestamp", "BuildingID", "RoomID", "Occupancy", "RoomCapacity",
        "Temperature_C", "Temp_Bin", "Energy_kWh", "Year", "Month",
        "Day", "DayOfWeek", "Hour", "Occupancy_Rate", "Energy_Per_Person", "ShiftPeriod"
    ]
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.dropna(subset=["Energy_kWh"])
    return df

df = load_data()

# ── Sidebar Filters ───────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/energy-saving-bulb.png", width=60)
st.sidebar.title("⚡ Filters")

buildings = st.sidebar.multiselect(
    "Building", sorted(df["BuildingID"].unique()),
    default=sorted(df["BuildingID"].unique())
)
shifts = st.sidebar.multiselect(
    "Shift Period", df["ShiftPeriod"].unique().tolist(),
    default=df["ShiftPeriod"].unique().tolist()
)
months = st.sidebar.multiselect(
    "Month", sorted(df["Month"].unique()),
    default=sorted(df["Month"].unique()),
    format_func=lambda m: ["Jan","Feb","Mar","Apr","May","Jun",
                           "Jul","Aug","Sep","Oct","Nov","Dec"][m-1]
)

filtered = df[
    df["BuildingID"].isin(buildings) &
    df["ShiftPeriod"].isin(shifts) &
    df["Month"].isin(months)
]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## ⚡ Smart Building Energy Consumption Dashboard")
st.caption(f"Showing **{len(filtered):,}** records | Buildings: {', '.join(buildings)} | "
           f"Shifts: {', '.join(shifts)}")
st.markdown("---")

# ── KPI Cards ─────────────────────────────────────────────────────────────────
total_kwh   = filtered["Energy_kWh"].sum()
avg_kwh     = filtered["Energy_kWh"].mean()
avg_occ     = filtered["Occupancy_Rate"].mean() * 100
avg_epp     = filtered["Energy_Per_Person"].mean()
peak_kwh    = filtered["Energy_kWh"].max()

k1, k2, k3, k4, k5 = st.columns(5)
for col, label, val, sub in zip(
    [k1, k2, k3, k4, k5],
    ["Total Energy", "Avg Energy / Record", "Avg Occupancy Rate",
     "Energy Per Person", "Peak Energy"],
    [f"{total_kwh:,.1f} kWh", f"{avg_kwh:.2f} kWh",
     f"{avg_occ:.1f}%", f"{avg_epp:.2f} kWh", f"{peak_kwh:.2f} kWh"],
    ["across all records", "mean hourly consumption", "room utilisation",
     "kWh per occupant", "highest single reading"]
):
    col.markdown(f"""
    <div class="kpi-box">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{val}</div>
        <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── Row 1: Time series + Shift breakdown ─────────────────────────────────────
r1c1, r1c2 = st.columns([2, 1])

with r1c1:
    st.subheader("📈 Energy Consumption Over Time")
    ts = (filtered.groupby("Timestamp")["Energy_kWh"]
          .mean().reset_index().rename(columns={"Energy_kWh": "Avg Energy (kWh)"}))
    fig_ts = px.line(ts, x="Timestamp", y="Avg Energy (kWh)",
                     color_discrete_sequence=["#00c8ff"])
    fig_ts.update_layout(template="plotly_dark", height=300,
                         margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_ts, use_container_width=True)

with r1c2:
    st.subheader("🌙 Energy by Shift Period")
    shift_data = (filtered.groupby("ShiftPeriod")["Energy_kWh"]
                  .mean().reset_index().rename(columns={"Energy_kWh": "Avg kWh"}))
    fig_shift = px.bar(shift_data, x="ShiftPeriod", y="Avg kWh",
                       color="ShiftPeriod",
                       color_discrete_sequence=px.colors.qualitative.Bold)
    fig_shift.update_layout(template="plotly_dark", height=300,
                            showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_shift, use_container_width=True)

# ── Row 2: By Building + Hourly Heatmap ──────────────────────────────────────
r2c1, r2c2 = st.columns([1, 2])

with r2c1:
    st.subheader("🏢 Avg Energy by Building")
    bld_data = (filtered.groupby("BuildingID")["Energy_kWh"]
                .mean().reset_index().rename(columns={"Energy_kWh": "Avg kWh"})
                .sort_values("Avg kWh", ascending=True))
    fig_bld = px.bar(bld_data, x="Avg kWh", y="BuildingID", orientation="h",
                     color="Avg kWh", color_continuous_scale="Blues")
    fig_bld.update_layout(template="plotly_dark", height=300,
                          margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_bld, use_container_width=True)

with r2c2:
    st.subheader("🕐 Hourly Energy Heatmap (Hour × Day of Week)")
    heat_data = (filtered.groupby(["DayOfWeek", "Hour"])["Energy_kWh"]
                 .mean().reset_index())
    heat_pivot = heat_data.pivot(index="DayOfWeek", columns="Hour",
                                 values="Energy_kWh")
    day_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri",
                 5: "Sat", 6: "Sun"}
    heat_pivot.index = [day_names.get(i, i) for i in heat_pivot.index]
    fig_heat = px.imshow(heat_pivot, color_continuous_scale="Blues",
                         labels=dict(color="Avg kWh"),
                         aspect="auto")
    fig_heat.update_layout(template="plotly_dark", height=300,
                           margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_heat, use_container_width=True)

# ── Row 3: Occupancy vs Energy + Temperature vs Energy ───────────────────────
r3c1, r3c2 = st.columns(2)

with r3c1:
    st.subheader("👥 Occupancy Rate vs Energy")
    sample = filtered.sample(min(500, len(filtered)), random_state=42)
    fig_occ = px.scatter(sample, x="Occupancy_Rate", y="Energy_kWh",
                         color="ShiftPeriod", opacity=0.7,
                         labels={"Occupancy_Rate": "Occupancy Rate",
                                 "Energy_kWh": "Energy (kWh)"},
                         color_discrete_sequence=px.colors.qualitative.Bold)
    fig_occ.update_layout(template="plotly_dark", height=320,
                          margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_occ, use_container_width=True)

with r3c2:
    st.subheader("🌡️ Temperature vs Energy")
    fig_temp = px.scatter(sample, x="Temperature_C", y="Energy_kWh",
                          color="BuildingID", opacity=0.7,
                          labels={"Temperature_C": "Temperature (°C)",
                                  "Energy_kWh": "Energy (kWh)"},
                          color_discrete_sequence=px.colors.qualitative.Vivid)
    fig_temp.update_layout(template="plotly_dark", height=320,
                           margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_temp, use_container_width=True)

# ── Row 4: Regression Model ───────────────────────────────────────────────────
st.markdown("---")
st.subheader("📐 Regression Model: Predicting Energy Consumption")

model_df = filtered[["Occupancy", "Temperature_C", "Energy_kWh",
                      "Occupancy_Rate"]].dropna()
X = model_df[["Occupancy", "Temperature_C"]]
y = model_df["Energy_kWh"]

reg = LinearRegression().fit(X, y)
y_pred = reg.predict(X)
r2   = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("Multiple R",      f"{np.sqrt(r2):.4f}")
mc2.metric("R² Score",        f"{r2:.4f}")
mc3.metric("RMSE",            f"{rmse:.4f}")
mc4.metric("Observations",    f"{len(y):,}")

coeff_df = pd.DataFrame({
    "Variable":    ["Intercept", "Occupancy", "Temperature_C"],
    "Coefficient": [reg.intercept_, reg.coef_[0], reg.coef_[1]],
})
st.dataframe(coeff_df.style.format({"Coefficient": "{:.6f}"}),
             use_container_width=True, hide_index=True)

# Actual vs Predicted chart
rc1, rc2 = st.columns(2)
with rc1:
    fig_reg = go.Figure()
    fig_reg.add_trace(go.Scatter(x=y, y=y_pred, mode="markers",
                                 marker=dict(color="#00c8ff", opacity=0.5),
                                 name="Predictions"))
    fig_reg.add_trace(go.Scatter(x=[y.min(), y.max()],
                                 y=[y.min(), y.max()],
                                 mode="lines", line=dict(color="red", dash="dash"),
                                 name="Perfect Fit"))
    fig_reg.update_layout(template="plotly_dark", height=320,
                          title="Actual vs Predicted Energy (kWh)",
                          xaxis_title="Actual", yaxis_title="Predicted",
                          margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_reg, use_container_width=True)

with rc2:
    st.subheader("🔮 Predict Energy Consumption")
    st.caption("Enter building conditions to get a prediction")
    occ_input  = st.slider("Occupancy (number of people)", 1, 40, 10)
    temp_input = st.slider("Temperature (°C)", 18.0, 35.0, 24.0, 0.5)
    pred_val   = reg.predict([[occ_input, temp_input]])[0]
    st.success(f"**Predicted Energy: {pred_val:.2f} kWh**")
    st.info(f"Coefficients: Occupancy × {reg.coef_[0]:.4f} + "
            f"Temp × {reg.coef_[1]:.4f} + {reg.intercept_:.4f}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("© 2026 Eagle Technologies LTD · Douglas · MSc Data Analytics · Istanbul")
