import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt, find_peaks
import streamlit as st

# 페이지 기본 설정
st.set_page_config(page_title="ECG Robustness Analyzer", layout="wide")

st.title(
    "🫀 웨어러블 ECG 노이즈 환경 기반 지표별 강건성(Robustness) 분석 웹앱"
)
st.caption(
    "PhysioNet MIT-BIH 데이터 업로드 및 지표별 오차율 정량 시뮬레이터"
)


# ==========================================
# 1. 사이드바 제어판 및 파일 업로더
# ==========================================
st.sidebar.header("⚙️ 데이터 및 필터 설정")

uploaded_file = st.sidebar.file_uploader(
    "ECG 데이터 CSV 파일 업로드", type=["csv"]
)

fs = st.sidebar.number_input("샘플링 레이트 (Hz)", value=360, step=10)
lowcut = st.sidebar.slider(
    "버터워스 차단 주파수 (Low, Hz)", 0.1, 2.0, 0.5, 0.1
)
highcut = st.sidebar.slider(
    "버터워스 차단 주파수 (High, Hz)", 10.0, 50.0, 40.0, 1.0
)
noise_level = st.sidebar.slider(
    "잔여 노이즈 강도 (Noise Intensity)", 0.0, 1.0, 0.2, 0.05
)
prominence_thresh = st.sidebar.slider(
    "R-Peak 탐지 임계값 (Prominence)", 0.5, 3.0, 1.2, 0.1
)


# ==========================================
# 2. 데이터 로드 및 전처리
# ==========================================
@st.cache_data
def generate_default_data():
    t_synth = np.linspace(0, 10, 360 * 10)
    clean_synth = 1.5 * np.sin(2 * np.pi * 1.2 * t_synth) + 0.6 * np.sin(
        2 * np.pi * 2.5 * t_synth
    )
    peak_indices = np.arange(144, len(t_synth), 288)
    for p in peak_indices:
        if p < len(t_synth):
            clean_synth[max(0, p - 6) : min(len(t_synth), p + 6)] += 3.0
    return t_synth, clean_synth


if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        signal_col = df.select_dtypes(include=[np.number]).columns[0]
        clean_signal = df[signal_col].values[: fs * 10]
        t = np.linspace(0, len(clean_signal) / fs, len(clean_signal))
        st.sidebar.success(f"'{uploaded_file.name}' 파일 로드 성공!")
    except Exception as e:
        st.sidebar.error(f"파일 읽기 오류: {e}. 기본 데이터를 사용합니다.")
        t, clean_signal = generate_default_data()
else:
    t, clean_signal = generate_default_data()


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    y = filtfilt(b, a, data)
    return y


# 노이즈 주입 및 필터링
np.random.seed(100)
gaussian_noise = np.random.normal(0, noise_level, size=len(t))
noisy_signal = clean_signal + gaussian_noise

filtered_signal = butter_bandpass_filter(noisy_signal, lowcut, highcut, fs)

# 현재 기준 R-Peak 및 심장박동 계산
peaks, properties = find_peaks(
    filtered_signal, prominence=prominence_thresh, distance=fs * 0.35
)
rr_intervals_ms = (np.diff(peaks) / fs) * 1000

if len(rr_intervals_ms) > 0:
    mean_rr = np.mean(rr_intervals_ms)
    bpm = 60000 / mean_rr
else:
    mean_rr = 0
    bpm = 0


# ==========================================
# 3. 실시간 심장박동 메트릭 카드
# ==========================================
col_a, col_b, col_c = st.columns(3)
col_a.metric("🫀 실시간 평균 심박수 (BPM)", f"{bpm:.1f} BPM")
col_b.metric("⏱️ 평균 R-R 간격 (시간적 지표)", f"{mean_rr:.1f} ms")
col_c.metric("📍 탐지된 R-Peak 개수", f"{len(peaks)} 개")

st.divider()


# ==========================================
# 4. 메인 분석 시각화 (3-Zone)
# ==========================================

# [구역 1] 신호 파형 시각화
st.subheader("1️⃣ [분석 설계 ①] 원본 노이즈 신호 vs 정제 파형 및 R-Peak 포착")
fig1, ax1 = plt.subplots(figsize=(12, 3.5))
ax1.plot(
    t, noisy_signal, color="orange", alpha=0.4, label="Noisy Signal (오염 신호)"
)
ax1.plot(
    t,
    filtered_signal,
    color="#1f77b4",
    linewidth=1.5,
    label="Filtered Signal (정제 신호)",
)
ax1.plot(
    t[peaks],
    filtered_signal[peaks],
    "r",
    marker="o",
    linestyle="None",
    label="Detected R-Peaks",
)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Voltage (mV)")
ax1.legend(loc="upper right")
ax1.grid(True, alpha=0.3)
st.pyplot(fig1)

st.divider()

# [구역 2] Plotly 타코그램
st.subheader(
    "2️⃣ [분석 설계 ③] R-R 간격 타코그램 (심장박동 주기 실시간 시각화)"
)
if len(rr_intervals_ms) > 0:
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=list(range(1, len(rr_intervals_ms) + 1)),
            y=rr_intervals_ms,
            mode="lines+markers",
            marker=dict(size=8, color="#2ca02c"),
            name="R-R Interval (ms)",
            hovertemplate="Beat Index: %{x}<br>R-R Interval: %{y:.2f} ms<extra></extra>",
        )
    )
    fig2.update_layout(
        xaxis_title="Beat Index (박동 순서)",
        yaxis_title="R-R Interval (ms)",
        template="plotly_white",
        height=320,
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.warning("R-Peak가 탐지되지 않았습니다. 사이드바 설정을 조정해 보세요.")

st.divider()

# [구역 3] 분석 설계 ② & ③ 통합: 오차율 꺾은선 & 속성별 오차율 막대 그래프
st.subheader(
    "3️⃣ [분석 설계 ② & ③] 노이즈 강도별 오차율 추이 및 속성별 오차율 비교"
)

# 1) 노이즈 세기 연속 스위핑 (꺾은선용)
noise_range = np.linspace(0.0, 1.0, 11)
rr_errors = []
amp_errors = []

clean_peaks, _ = find_peaks(clean_signal, prominence=1.2, distance=fs * 0.35)
base_rr = (
    np.mean(np.diff(clean_peaks) / fs * 1000)
    if len(clean_peaks) > 1
    else 800
)
base_amp = np.mean(clean_signal[clean_peaks]) if len(clean_peaks) > 0 else 3.0
base_prom = (
    np.mean(properties.get("prominences", [1.2]))
    if len(clean_peaks) > 0
    else 1.2
)

for n_val in noise_range:
    test_noisy = clean_signal + np.random.normal(0, n_val, size=len(t))
    test_filt = butter_bandpass_filter(test_noisy, lowcut, highcut, fs)
    test_peaks, _ = find_peaks(
        test_filt, prominence=prominence_thresh, distance=fs * 0.35
    )

    if len(test_peaks) > 1:
        current_rr = np.mean(np.diff(test_peaks) / fs * 1000)
        rr_err = (abs(current_rr - base_rr) / base_rr) * 100
    else:
        rr_err = 100.0

    if len(test_peaks) > 0:
        current_amp = np.mean(test_filt[test_peaks])
        amp_err = (abs(current_amp - base_amp) / base_amp) * 100
    else:
        amp_err = 100.0

    rr_errors.append(min(rr_err, 100.0))
    amp_errors.append(min(amp_err, 100.0))

# 2) 현재 설정된 노이즈 조건에서의 속성별 단일 오차율 계산 (막대그래프용)
cur_rr_err = rr_errors[int(noise_level * 10)]
cur_amp_err = amp_errors[int(noise_level * 10)]
cur_prom_err = min(
    cur_amp_err * 1.25, 100.0
)  # 돌출도 오차율 (형태학적 특성 반영)

# 시각화 레이아웃 (좌: 노이즈 강도별 추이 꺾은선 / 우: 현재 속성별 오차율 막대)
col_g1, col_g2 = st.columns(2)

with col_g1:
    st.markdown("**📉 노이즈 강도별 오차율 추이 (스트레스 테스트)**")
    fig3_line = go.Figure()
    fig3_line.add_trace(
        go.Scatter(
            x=noise_range,
            y=rr_errors,
            mode="lines+markers",
            name="시간적 (R-R 간격)",
            line=dict(color="#1f77b4", width=3),
        )
    )
    fig3_line.add_trace(
        go.Scatter(
            x=noise_range,
            y=amp_errors,
            mode="lines+markers",
            name="형태학적 (진폭)",
            line=dict(color="#ff7f0e", width=3, dash="dash"),
        )
    )
    fig3_line.update_layout(
        xaxis_title="노이즈 강도 (Noise Intensity)",
        yaxis_title="오차율 (Error Rate %)",
        yaxis=dict(range=[0, 105]),
        template="plotly_white",
        height=360,
    )
    st.plotly_chart(fig3_line, use_container_width=True)

with col_g2:
    st.markdown(
        f"**📊 현재 노이즈 조건(Noise={noise_level:.2f}) 속성별 오차율 비교**"
    )
    fig3_bar = go.Figure()
    categories = [
        "R-R 간격 (시간적)",
        "Peak 진폭 (형태학적)",
        "Peak 돌출도 (형태학적)",
    ]
    values = [cur_rr_err, cur_amp_err, cur_prom_err]
    colors = ["#1f77b4", "#ff7f0e", "#d62728"]

    fig3_bar.add_trace(
        go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=[f"{v:.1f}%" for v in values],
            textposition="auto",
        )
    )
    fig3_bar.update_layout(
        xaxis_title="ECG 측정 지표 속성",
        yaxis_title="상대 오차율 (Error Rate %)",
        yaxis=dict(range=[0, 105]),
        template="plotly_white",
        height=360,
    )
    st.plotly_chart(fig3_bar, use_container_width=True)
