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
    st.header("현장 센서 및 모델 파라미터 입력")

    # Logo
    try:
        logo = Image.open("AI_Lab_logo.jpg")
        st.image(logo, use_column_width=True)
    except FileNotFoundError:
        uploaded = st.file_uploader("로고 업로드 (jpg, png)", type=["jpg","png"])
        if uploaded:
            logo = Image.open(uploaded)
            st.image(logo, use_column_width=True)

    st.markdown("---")
    # Chlorine residual
    Cl0 = st.number_input("초기 잔류 염소 농도 C₀ (mg/L)", min_value=0.01, value=1.5)
    # Temperature and pH
    Temp = st.slider("온도 T (°C)", 0.0, 35.0, 20.0)
    pH    = st.slider("pH", 6.0, 9.0, 7.5)
    # Decay constant at 20°C (k₀) and temperature coefficient (θ)
    k20   = st.number_input("k₂₀ (1/hr at 20°C)", min_value=0.001, value=0.05, step=0.005)
    theta = st.number_input("온도 계수 θ", min_value=1.00, max_value=1.10, value=1.035, step=0.005)

    st.markdown("---")
    # Operational sensors
    turbidity = st.slider("탁도 (NTU)", 0.0, 20.0, 1.0)
    flow_rate = st.number_input("유량 (m³/h)", min_value=0.0, value=100.0)
    ORP       = st.slider("ORP (mV)", -500, 1000, 300)

    st.markdown("---")
    # Observation time point
    max_time = st.slider("최대 예측 시간 (hrs)", 1, 24, 8)
    obs_time = st.slider("관찰 시점 (hrs)", 0.0, float(max_time), float(max_time/2), step=0.1)

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
ax.plot(time_range, C_pred, label='이론값 (모델 예측)', linestyle='-', linewidth=2)
ax.plot(time_range, C_low,  label='허용 하한 (80%)',    linestyle='--')
ax.plot(time_range, C_high, label='허용 상한 (120%)',   linestyle='--')
ax.scatter(obs_time, C_obs[np.argmin(np.abs(time_range-obs_time))], color='red', label='관찰값', zorder=5)
ax.set_xlabel('시간 (hrs)')
ax.set_ylabel('잔류 염소 농도 (mg/L)')
ax.set_title('EPA 1차 반응 모델 기반 잔류 염소 예측')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Anomaly Diagnosis

def diagnose_epa(C_val, low, high, turb, flow, orp):
    if C_val > high:
        if turb < 1 and orp > 400:
            return "과다 투입 의심: 수질 양호"  
        else:
            return "투입량 조절 필요: 전처리 확인"
    elif C_val < low:
        if turb > 5:
            return "전처리 부실 의심: 탁도 높음"
        elif flow > 0.9 * flow_rate:
            return "수리학적 문제 의심: 유량 과부하"
        else:
            return "주입량 부족 또는 반응시간 부족"
    else:
        return "정상"

# Evaluate at observation time
idx = np.argmin(np.abs(time_range - obs_time))
C_val  = C_obs[idx]
low_val= C_low[idx]
high_val= C_high[idx]
diag    = diagnose_epa(C_val, low_val, high_val, turbidity, flow_rate, ORP)

st.subheader(f"관찰 시점: {obs_time:.1f} hrs")
if diag == "정상":
    st.success("정상 운전 중")
else:
    st.warning(f"이상 징후: {diag}")

# Logging and Download
log = {
    "time(hr)": obs_time,
    "C_obs": round(float(C_val), 3),
    "Lower_bound": round(float(low_val),3),
    "Upper_bound": round(float(high_val),3),
    "Temp(°C)": Temp,
    "pH": pH,
    "k_T(1/hr)": round(k_T,4),
    "Turbidity(NTU)": turbidity,
    "Flow(m³/h)": flow_rate,
    "ORP(mV)": ORP,
    "Diagnosis": diag
}
df_log = pd.DataFrame([log])
st.download_button(
    "진단 리포트 다운로드", df_log.to_csv(index=False), "epa_diagnosis_report.csv", "text/csv"
)
