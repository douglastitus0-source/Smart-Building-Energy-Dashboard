"""
Smart Building Energy Consumption Dashboard
Author : Douglas | Eagle Technologies LTD | MSc Data Analytics
Run    : streamlit run energy_dashboard.py
Data   : Smart_building_energy_Consumption_Model.xlsx  (same folder)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Building Energy Dashboard",
    page_icon="⚡",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .kpi-box {
        background: #1e2a3a;
        border-radius: 12px;
        padding: 18px 24px;
        text-align: center;
        border-left: 4px solid #00c8ff;
        margin-bottom: 6px;
    }
    .kpi-label { color: #8fa8c8; font-size: 13px; font-weight: 600; margin-bottom: 4px; }
    .kpi-value { color: #ffffff; font-size: 26px; font-weight: 700; }
    .kpi-sub   { color: #00c8ff; font-size: 12px; margin-top: 2px; }
    section[data-testid="stSidebar"] { background-color: #12212f; }
</style>
""", unsafe_allow_html=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_excel(
        "Smart_building_energy_Consumption_Model.xlsx",
        sheet_name="BuildingData_Clean"
    )
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
st.sidebar.image("https://img.icons8.com/fluency/96/energy-saving-bulb.png", width=56)
st.sidebar.title("⚡ Dashboard Filters")
st.sidebar.markdown("---")

buildings = st.sidebar.multiselect(
    "🏢 Building", sorted(df["BuildingID"].unique()),
    default=sorted(df["BuildingID"].unique())
)
shifts = st.sidebar.multiselect(
    "🌙 Shift Period", sorted(df["ShiftPeriod"].unique()),
    default=sorted(df["ShiftPeriod"].unique())
)
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
months = st.sidebar.multiselect(
    "📅 Month", sorted(df["Month"].unique()),
    default=sorted(df["Month"].unique()),
    format_func=lambda m: MONTH_NAMES[m - 1]
)

filtered = df[
    df["BuildingID"].isin(buildings) &
    df["ShiftPeriod"].isin(shifts) &
    df["Month"].isin(months)
]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## ⚡ Smart Building Energy Consumption Dashboard")
st.caption(
    f"**{len(filtered):,}** records · Buildings: {', '.join(buildings)} · "
    f"Shifts: {', '.join(shifts)}"
)
st.markdown("---")

if filtered.empty:
    st.warning("No data matches the current filters. Adjust the sidebar selections.")
    st.stop()

# ── KPI Cards ─────────────────────────────────────────────────────────────────
kpi_data = [
    ("Total Energy",        f"{filtered['Energy_kWh'].sum():,.1f} kWh",  "across all records"),
    ("Avg Energy / Record", f"{filtered['Energy_kWh'].mean():.2f} kWh",  "mean hourly consumption"),
    ("Avg Occupancy Rate",  f"{filtered['Occupancy_Rate'].mean()*100:.1f}%", "room utilisation"),
    ("Energy Per Person",   f"{filtered['Energy_Per_Person'].mean():.2f} kWh", "kWh per occupant"),
    ("Peak Energy",         f"{filtered['Energy_kWh'].max():.2f} kWh",   "highest single reading"),
]
for col, (label, val, sub) in zip(st.columns(5), kpi_data):
    col.markdown(f"""
    <div class="kpi-box">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{val}</div>
        <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── Row 1 : Time Series + Shift Breakdown ────────────────────────────────────
c1, c2 = st.columns([2, 1])

with c1:
    st.subheader("📈 Energy Consumption Over Time")
    ts = (filtered.groupby("Timestamp")["Energy_kWh"]
          .mean().reset_index()
          .rename(columns={"Energy_kWh": "Avg Energy (kWh)"}))
    fig = px.line(ts, x="Timestamp", y="Avg Energy (kWh)",
                  color_discrete_sequence=["#00c8ff"])
    fig.update_layout(template="plotly_dark", height=290,
                      margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("🌙 Energy by Shift Period")
    shift_df = (filtered.groupby("ShiftPeriod")["Energy_kWh"]
                .mean().reset_index()
                .rename(columns={"Energy_kWh": "Avg kWh"}))
    fig = px.bar(shift_df, x="ShiftPeriod", y="Avg kWh",
                 color="ShiftPeriod",
                 color_discrete_sequence=px.colors.qualitative.Bold)
    fig.update_layout(template="plotly_dark", height=290,
                      showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

# ── Row 2 : Building Comparison + Hourly Heatmap ──────────────────────────────
c3, c4 = st.columns([1, 2])

with c3:
    st.subheader("🏢 Avg Energy by Building")
    bld_df = (filtered.groupby("BuildingID")["Energy_kWh"]
              .mean().reset_index()
              .rename(columns={"Energy_kWh": "Avg kWh"})
              .sort_values("Avg kWh"))
    fig = px.bar(bld_df, x="Avg kWh", y="BuildingID", orientation="h",
                 color="Avg kWh", color_continuous_scale="Blues")
    fig.update_layout(template="plotly_dark", height=290,
                      margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

with c4:
    st.subheader("🕐 Hourly Energy Heatmap  (Hour × Day of Week)")
    heat = (filtered.groupby(["DayOfWeek", "Hour"])["Energy_kWh"]
            .mean().reset_index())
    pivot = heat.pivot(index="DayOfWeek", columns="Hour", values="Energy_kWh")
    day_map = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
    pivot.index = [day_map.get(i, i) for i in pivot.index]
    fig = px.imshow(pivot, color_continuous_scale="Blues",
                    labels=dict(color="Avg kWh"), aspect="auto")
    fig.update_layout(template="plotly_dark", height=290,
                      margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

# ── Row 3 : Scatter Plots ─────────────────────────────────────────────────────
c5, c6 = st.columns(2)
sample = filtered.sample(min(600, len(filtered)), random_state=42)

with c5:
    st.subheader("👥 Occupancy Rate vs Energy")
    fig = px.scatter(sample, x="Occupancy_Rate", y="Energy_kWh",
                     color="ShiftPeriod", opacity=0.65,
                     labels={"Occupancy_Rate": "Occupancy Rate",
                              "Energy_kWh": "Energy (kWh)"},
                     color_discrete_sequence=px.colors.qualitative.Bold)
    fig.update_layout(template="plotly_dark", height=320,
                      margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

with c6:
    st.subheader("🌡️ Temperature vs Energy")
    fig = px.scatter(sample, x="Temperature_C", y="Energy_kWh",
                     color="BuildingID", opacity=0.65,
                     labels={"Temperature_C": "Temperature (°C)",
                              "Energy_kWh": "Energy (kWh)"},
                     color_discrete_sequence=px.colors.qualitative.Vivid)
    fig.update_layout(template="plotly_dark", height=320,
                      margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

# ── Row 4 : Regression Model ──────────────────────────────────────────────────
st.markdown("---")
st.subheader("📐 Regression Model — Predicting Energy Consumption")
st.caption("Multiple linear regression using Occupancy and Temperature as predictors")

reg_df  = filtered[["Occupancy", "Temperature_C", "Energy_kWh"]].dropna()
X_reg   = reg_df[["Occupancy", "Temperature_C"]]
y_reg   = reg_df["Energy_kWh"]
reg     = LinearRegression().fit(X_reg, y_reg)
y_hat   = reg.predict(X_reg)
r2_val  = r2_score(y_reg, y_hat)
rmse_val= np.sqrt(mean_squared_error(y_reg, y_hat))

m1, m2, m3, m4 = st.columns(4)
m1.metric("Multiple R",   f"{np.sqrt(r2_val):.4f}")
m2.metric("R² Score",     f"{r2_val:.4f}")
m3.metric("RMSE",         f"{rmse_val:.4f} kWh")
m4.metric("Observations", f"{len(y_reg):,}")

coeff_df = pd.DataFrame({
    "Variable":    ["Intercept", "Occupancy", "Temperature_C"],
    "Coefficient": [reg.intercept_, reg.coef_[0], reg.coef_[1]],
    "Interpretation": [
        "Base energy when inputs are zero",
        "Energy added per extra occupant",
        "Energy added per °C temperature rise",
    ]
})
st.dataframe(coeff_df.style.format({"Coefficient": "{:.6f}"}),
             use_container_width=True, hide_index=True)

rc1, rc2 = st.columns(2)

with rc1:
    st.markdown("**Actual vs Predicted Energy (kWh)**")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_reg, y=y_hat, mode="markers",
                             marker=dict(color="#00c8ff", opacity=0.45, size=5),
                             name="Predicted"))
    fig.add_trace(go.Scatter(x=[y_reg.min(), y_reg.max()],
                             y=[y_reg.min(), y_reg.max()],
                             mode="lines", line=dict(color="red", dash="dash"),
                             name="Perfect Fit"))
    fig.update_layout(template="plotly_dark", height=320,
                      xaxis_title="Actual kWh", yaxis_title="Predicted kWh",
                      margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

with rc2:
    st.markdown("**🔮 Live Energy Predictor**")
    st.caption("Adjust the sliders to get a real-time energy prediction")
    occ_in  = st.slider("Occupancy — number of people in room", 1, 40, 10)
    temp_in = st.slider("Temperature (°C)", 18.0, 35.0, 24.0, 0.5)
    pred    = reg.predict([[occ_in, temp_in]])[0]
    st.success(f"**Predicted Energy: {pred:.2f} kWh**")
    st.info(
        f"Formula: ({reg.coef_[0]:.4f} × occupancy) + "
        f"({reg.coef_[1]:.4f} × temp) + {reg.intercept_:.4f}"
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("© 2026 Eagle Technologies LTD · Douglas · MSc Data Analytics · Istanbul")
