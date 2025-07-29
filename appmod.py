import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

# Constants
epsilon = 1e-3

# Streamlit Page Setup
st.set_page_config(page_title="Residual Chlorine Anomaly Detection Tool", layout="wide")
st.title("잔류 염소 농도 예측 및 이상진단 툴")

# Sidebar: Logo
with st.sidebar:
    st.header("모델 인풋 설정")
    try:
        logo = Image.open("AI_Lab_logo.jpg")
        st.image(logo, use_column_width=True)
    except FileNotFoundError:
        st.write("Logo image not found. 업로드해주세요.")
        st.file_uploader("로고 업로드", type=["jpg","png"], key="logo_uploader")

    # Seed for reproducibility
    seed = st.number_input("Random Seed (재현성)", min_value=0, value=42, step=1)
    np.random.seed(int(seed))

    # Water quality inputs
    DOC   = st.slider("DOC (mg/L)", 0.001, 10.0, 5.0)
    NH3   = st.slider("NH3 (mg/L)", 0.0, 5.0, 0.5)
    Cl0   = st.slider("현재 농도 Cl0 (mg/L)", 0.001, 5.0, 1.5)
    Temp  = st.slider("Temperature (°C)", 0.0, 35.0, 20.0)
    max_time = st.slider("최대예측시간 (hrs)", 1, 24, 5)

    st.markdown("---")
    st.header("EPA 모델 k1/k2 범위 설정")
    k1_min = st.slider("k1 최소값", 0.01, 5.0, 0.5)
    k1_max = st.slider("k1 최대값", 0.01, 5.0, 3.5)
    k2_min = st.slider("k2 최소값", 0.01, 5.0, 0.1)
    k2_max = st.slider("k2 최대값", 0.01, 5.0, 0.5)

    st.markdown("---")
    st.header("현장 센서 입력")
    turbidity = st.slider("Turbidity (NTU)", 0.0, 20.0, 1.0)
    flow_rate = st.number_input("Flow rate (m³/h)", 0.0, 2000.0, 100.0)
    ORP       = st.slider("ORP (mV)", -500, 1000, 300)

    st.markdown("---")
    obs_time = st.slider("관찰 시점 (hrs)", 0.0, float(max_time), float(max_time/2), step=0.1)

# Time axis
time_range = np.linspace(0, max_time, 200)

# Safe log
log = lambda x: np.log(np.maximum(x, epsilon))

# Model computations
def compute_epa(doc, nh3, cl0):
    k1 = np.exp(-0.442 + 0.889 * log(doc) + 0.345 * log(7.6 * nh3) - 1.082 * log(cl0) + 0.192 * log(cl0 / doc))
    k2 = np.exp(-4.817 + 1.187 * log(doc) + 0.102 * log(7.6 * nh3) - 0.821 * log(cl0) - 0.271 * log(cl0 / doc))
    return k1, k2

# Two-phase stub (활성화 시 사용)
def compute_two_phase(doc, nh3, cl0, temp):
    A  = np.exp(0.168 - 0.148 * log(cl0 / doc) + 0.0554 * log(nh3) + 0.185 * log(temp) - 0.41 * log(cl0))
    k1 = np.exp(5.41 - 0.38 * log(cl0 / doc) + 0.274 * log(nh3) - 1.12 * log(temp) - 0.854 * log(7))
    k2 = np.exp(-7.13 + 0.864 * log(cl0 / doc) + 2.63 * log(doc) - 2.55 * log(cl0) + 0.48 * log(nh3) + 1.03 * log(temp))
    return A, k1, k2

# EPA coefficients
k1_epa, k2_epa = compute_epa(DOC, NH3, Cl0)

# Concentration predictions
def predict_epa(cl0, k1, k2):
    C = np.where(
        time_range <= 5,
        cl0 * np.exp(-k1 * time_range),
        cl0 * np.exp(5 * (k2 - k1)) * np.exp(-k2 * time_range)
    )
    return C

C_pred = predict_epa(Cl0, k1_epa, k2_epa)
# Variation
variation = 1 + (time_range / max_time * 2) * np.random.uniform(-0.2, 0.4, size=C_pred.shape)
C_obs = C_pred * variation

# Low/High bounds
C_low  = predict_epa(Cl0, k1_max, k2_max)
C_high = predict_epa(Cl0, k1_min, k2_min)

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(time_range, C_obs, label='실측(가상)', linewidth=2)
ax.plot(time_range, C_low, label='EPA Low', linestyle='--')
ax.plot(time_range, C_high, label='EPA High', linestyle='--')
ax.set_xlabel('Time (hrs)')
ax.set_ylabel('Residual Chlorine (mg/L)')
ax.set_title('Residual Chlorine Prediction')
ax.legend()
st.pyplot(fig)

# Anomaly diagnosis

def diagnose_anomaly(C_val, low, high, turb, flow, orp):
    if C_val > high:
        if turb < 1 and orp > 400:
            return "과다 투입 → 수질 양호해 과량 투입 의심"
        else:
            return "투입량 조절 필요 → 이전 단계 수질 모니터링 확인"
    elif C_val < low:
        if turb > 5:
            return "탁도 높음 → 전처리 부실 의심"
        elif flow > 0.9 * flow_rate:
            return "유량 과부하 → 수리학적 문제 의심"
        else:
            return "주입량 부족 또는 반응시간 부족 의심"
    return "정상 범위"

# Get observed at obs_time
idx = np.argmin(np.abs(time_range - obs_time))
C_val = C_obs[idx]
low_val = C_low[idx]
high_val = C_high[idx]
diagnosis = diagnose_anomaly(C_val, low_val, high_val, turbidity, flow_rate, ORP)

# Display result
st.subheader(f"관찰 시점: {obs_time:.1f} hrs")
if diagnosis == "정상 범위":
    st.success("정상 운전 중")
else:
    st.warning(f"이상 징후: {diagnosis}")

# Logging & download
log = {
    "time": obs_time,
    "Cl_obs": float(C_val),
    "Cl_low": float(low_val),
    "Cl_high": float(high_val),
    "Turbidity": turbidity,
    "Flow_rate": flow_rate,
    "ORP": ORP,
    "Diagnosis": diagnosis
}
df_log = pd.DataFrame([log])
st.download_button(
    label="진단 리포트 다운로드", data=df_log.to_csv(index=False), file_name="diagnosis_report.csv", mime="text/csv"
)
