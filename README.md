# ⚡ Smart Building Energy Consumption Dashboard

An interactive web dashboard built with **Streamlit** and **Plotly** to analyse
smart building energy consumption patterns across 5 buildings and 1,498 hourly records.

🔗 **Live App:** [Click to open on Streamlit Cloud](https://your-app-url.streamlit.app)

---

## 📊 Features

| Section | Description |
|---------|-------------|
| KPI Cards | Total energy, avg kWh, occupancy rate, energy/person, peak usage |
| Time Series | Average energy over time |
| Shift Breakdown | Energy by Morning / Afternoon / Evening / Night |
| Building Comparison | Average kWh per building (B01–B05) |
| Hourly Heatmap | Energy intensity by hour and day of week |
| Scatter Plots | Occupancy Rate vs Energy · Temperature vs Energy |
| Regression Model | Multiple linear regression with live coefficients and R² |
| Live Predictor | Enter occupancy + temperature → get predicted kWh |

---

## 🗂️ Files

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit application |
| `requirements.txt` | Python dependencies |
| `Smart_building_energy_Consumption_Model.xlsx` | Source dataset (Excel) |

---

## ⚙️ How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/smart-building-dashboard.git
cd smart-building-dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

---

## 📐 Regression Results

| Metric | Value |
|--------|-------|
| Multiple R | 0.8848 |
| R² Score | 0.7830 |
| RMSE | 0.3491 |
| Observations | 1,000 |

**Key findings:**
- Occupancy is the **strongest predictor** of energy use
- Temperature has a positive relationship with energy consumption
- Night shift consistently shows lowest energy usage

---

## 🛠️ Tools Used

- Python · Streamlit · Plotly · Pandas · Scikit-learn
- Original analysis built in Microsoft Excel (Power Query, Pivot Tables, Data Analysis ToolPak)

---

## 👤 Author

**Douglas** — Founder, Eagle Technologies LTD  
MSc Data Analytics | Istanbul  
[LinkedIn](https://linkedin.com/in/YOUR_PROFILE)
