import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

# Constants
epsilon = 1e-6

# Streamlit Page Setup
st.set_page_config(page_title="Residual Chlorine EPA Model", layout="wide")
st.title("잔류 염소 농도 예측 및 이상진단 툴 (EPA 공식 기반)")

# Sidebar: Inputs
with st.sidebar:
    st.header("Field Sensor and Model Parameter Inputs")

    # Logo
    try:
        logo = Image.open("AI_Lab_logo.jpg")
        st.image(logo, use_column_width=True)
    except FileNotFoundError:
        uploaded = st.file_uploader("Upload Logo (jpg, png)", type=["jpg","png"])
        if uploaded:
            logo = Image.open(uploaded)
            st.image(logo, use_column_width=True)

    st.markdown("---")
    # Initial chlorine concentration
    Cl0 = st.number_input("Initial Residual Chlorine C₀ (mg/L)", min_value=0.01, value=1.5)
    # Temperature and pH
    Temp = st.slider("Temperature T (°C)", 0.0, 35.0, 20.0)
    pH    = st.slider("pH", 6.0, 9.0, 7.5)
    # Decay constant at 20°C and temperature coefficient
    k20   = st.number_input("Decay Constant k20 (1/hr at 20°C)", min_value=0.001, value=0.05, step=0.005)
    theta = st.number_input("Temperature Coefficient θ", min_value=1.00, max_value=1.10, value=1.035, step=0.005)

    st.markdown("---")
    # Operational sensors
    turbidity = st.slider("Turbidity (NTU)", 0.0, 20.0, 1.0)
    flow_rate = st.number_input("Flow Rate (m³/h)", min_value=0.0, value=100.0)
    ORP       = st.slider("ORP (mV)", -500, 1000, 300)

    st.markdown("---")
    # Observation time point
    max_time = st.slider("Maximum Prediction Time (hrs)", 1, 24, 8)
    obs_time = st.slider("Observation Time (hrs)", 0.0, float(max_time), float(max_time/2), step=0.1)

# Time axis (hrs)
time_range = np.linspace(0, max_time, 200)

# EPA first-order decay model
# k_T = k20 * θ^(T - 20)
# C(t) = C0 * exp(-k_T * t)
def compute_decay_constant(k20, theta, temp):
    return k20 * (theta ** (temp - 20))

def predict_chlorine(C0, k_T):
    return C0 * np.exp(-k_T * time_range)

# Compute decay constant and predictions
k_T = compute_decay_constant(k20, theta, Temp)
C_pred = predict_chlorine(Cl0, k_T)
# Simulated observation with ±10% noise
np.random.seed(42)
noise = np.random.uniform(-0.1, 0.1, size=C_pred.shape)
C_obs = C_pred * (1 + noise)

# Bounds: EPA guideline ± 20%
C_low  = C_pred * 0.8
C_high = C_pred * 1.2

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(time_range, C_pred, label='Theoretical Prediction', linestyle='-', linewidth=2)
ax.plot(time_range, C_low,  label='Lower Bound (80%)',    linestyle='--')
ax.plot(time_range, C_high, label='Upper Bound (120%)',   linestyle='--')
ax.scatter(obs_time, C_obs[np.argmin(np.abs(time_range-obs_time))], color='red', label='Observed Value', zorder=5)
ax.set_xlabel('Time (hrs)')
ax.set_ylabel('Residual Chlorine (mg/L)')
ax.set_title('Residual Chlorine Prediction based on EPA First-Order Decay Model')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Anomaly Diagnosis

def diagnose_epa(C_val, low, high, turb, flow, orp):
    if C_val > high:
        if turb < 1 and orp > 400:
            return "Excessive Dosage Suspected: Good Water Quality"  
        else:
            return "Adjust Dosage: Check Pretreatment"
    elif C_val < low:
        if turb > 5:
            return "Pretreatment Issue Suspected: High Turbidity"
        elif flow > 0.9 * flow_rate:
            return "Hydraulic Issue Suspected: Overload Flow"
        else:
            return "Insufficient Dosage or Reaction Time"
    else:
        return "Normal Operation"

# Evaluate at observation time
idx = np.argmin(np.abs(time_range - obs_time))
C_val   = C_obs[idx]
low_val = C_low[idx]
high_val= C_high[idx]
diag     = diagnose_epa(C_val, low_val, high_val, turbidity, flow_rate, ORP)

st.subheader(f"Observation Time: {obs_time:.1f} hrs")
if diag == "Normal Operation":
    st.success("Normal Operation")
else:
    st.warning(f"Anomaly Detected: {diag}")

# Logging and Download
log = {
    "Time (hrs)": obs_time,
    "Observed Cl (mg/L)": round(float(C_val), 3),
    "Lower Bound": round(float(low_val),3),
    "Upper Bound": round(float(high_val),3),
    "Temperature (°C)": Temp,
    "pH": pH,
    "Decay Constant k_T (1/hr)": round(k_T,4),
    "Turbidity (NTU)": turbidity,
    "Flow Rate (m³/h)": flow_rate,
    "ORP (mV)": ORP,
    "Diagnosis": diag
}
df_log = pd.DataFrame([log])
st.download_button(
    "Download Report", df_log.to_csv(index=False), "epa_diagnosis_report.csv", "text/csv"
)
