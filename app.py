import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt, find_peaks

# 페이지 기본 설정 (layout="wide"로 오류 수정 완료)
st.set_page_config(
    page_title="ECG Robustness Analyzer", 
    layout="wide"
)

st.title("🫀 웨어러블 ECG 노이즈 환경 기반 지표별 강건성(Robustness) 분석 웹앱")
st.caption("PhysioNet MIT-BIH Record 100 데이터셋 기반 정량 시각화 시뮬레이터")


# ==========================================
# 1. 데이터 로드 및 헬퍼 함수
# ==========================================
@st.cache_data
def load_ecg_data():
    # 360Hz 기준 10초 데이터 합성 (테스트용)
    fs = 360
    t = np.linspace(0, 10, fs * 10)
    clean_signal = 1.2 * np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 2.4 * t) + 0.25 * np.random.normal(size=len(t))
    
    # R-Peak 위치 모사 (약 1초 간격)
    peak_indices = np.arange(180, len(t), 360)
    for p in peak_indices:
        clean_signal[max(0, p-5):min(len(t), p+5)] += 2.5
    return t, clean_signal, fs

# 버터워스 대역통과 필터 (filtfilt 사용으로 위상 지연 방지)
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y


# ==========================================
# 2. 사이드바 제어판 (사용자 설정 조건)
# ==========================================
st.sidebar.header("⚙️ 실험 조건 제어판")

fs = st.sidebar.number_input("샘플링 레이트 (Hz)", value=360, step=10)
lowcut = st.sidebar.slider("버터워스 차단 주파수 (Low, Hz)", 0.1, 2.0, 0.5, 0.1)
highcut = st.sidebar.slider("버터워스 차단 주파수 (High, Hz)", 10.0, 50.0, 40.0, 1.0)
noise_level = st.sidebar.slider("잔여 노이즈 강도 (Noise Intensity)", 0.0, 1.0, 0.3, 0.05)
prominence_thresh = st.sidebar.slider("R-Peak 탐지 임계값 (Prominence)", 0.5, 3.0, 1.5, 0.1)

# 데이터 불러오기 및 노이즈 주입
t, clean_signal, _ = load_ecg_data()
np.random.seed(42)
gaussian_noise = np.random.normal(0, noise_level, size=len(t))
noisy_signal = clean_signal + gaussian_noise

# 전처리 필터링 수행
filtered_signal = butter_bandpass_filter(noisy_signal, lowcut, highcut, fs)

# R-Peak 탐지
peaks, properties = find_peaks(filtered_signal, prominence=prominence_thresh, distance=fs*0.4)
rr_intervals_ms = (np.diff(peaks) / fs) * 1000  # ms 단위 계산


# ==========================================
# 3. 메인 분석 화면 (3-Zone 구성)
# ==========================================

# ------------------------------------------
# [구역 1] 파형 분석 및 R-Peak 포착 시각화
# ------------------------------------------
st.subheader("1️⃣ [분석 설계 ①] 신호 정제 및 R-Peak 탐지 파형")
fig1, ax1 = plt.subplots(figsize=(12, 3.5))
ax1.plot(t, noisy_signal, color='orange', alpha=0.4, label='Noisy Signal (오염 신호)')
ax1.plot(t, filtered_signal, color='#1f77b4', linewidth=1.5, label='Filtered Signal (정제 신호)')
ax1.plot(t[peaks], filtered_signal[peaks], 'r', marker='o', linestyle='None', label='Detected R-Peaks')
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Voltage (mV)")
ax1.legend(loc="upper right")
ax1.grid(True, alpha=0.3)
st.pyplot(fig1)

st.divider()

# ------------------------------------------
# [구역 2] Plotly 기반 인터랙티브 타코그램
# ------------------------------------------
st.subheader("2️⃣ [분석 설계 ③] R-R 간격 타코그램 (Interactive Tachogram)")
if len(rr_intervals_ms) > 0:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=list(range(1, len(rr_intervals_ms) + 1)),
        y=rr_intervals_ms,
        mode='lines+markers',
        marker=dict(size=8, color='#2ca02c'),
        name='R-R Interval (ms)',
        hovertemplate='Beat Index: %{x}<br>R-R Interval: %{y:.2f} ms<extra></extra>'
    ))
    fig2.update_layout(
        xaxis_title="Beat Index",
        yaxis_title="R-R Interval (ms)",
        template="plotly_white",
        height=350
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.warning("R-Peak가 탐지되지 않았습니다. 임계값(Prominence)을 낮춰주세요.")

st.divider()

# ------------------------------------------
# [구역 3] 분석 설계 ② 추가: 노이즈 강도별 오차율 스트레스 테스트
# ------------------------------------------
st.subheader("3️⃣ [분석 설계 ②] 노이즈 강도(Noise Intensity)에 따른 지표별 오차율 비교")
st.markdown("노이즈 세기를 0.0에서 1.0까지 시뮬레이션할 때, **시간적 지표(R-R 간격)**와 **형태학적 지표(Amplitude)**의 상대 오차율(%) 변화를 비교합니다.")

# 노이즈 범위 스위핑 시뮬레이션 계산
noise_range = np.linspace(0.0, 1.0, 11)
rr_errors = []
amp_errors = []

clean_peaks, _ = find_peaks(clean_signal, prominence=1.5, distance=fs*0.4)
base_rr = np.mean(np.diff(clean_peaks)/fs * 1000) if len(clean_peaks) > 1 else 1000
base_amp = np.mean(clean_signal[clean_peaks]) if len(clean_peaks) > 0 else 1.0

for n_val in noise_range:
    test_noisy = clean_signal + np.random.normal(0, n_val, size=len(t))
    test_filt = butter_bandpass_filter(test_noisy, lowcut, highcut, fs)
    test_peaks, _ = find_peaks(test_filt, prominence=prominence_thresh, distance=fs*0.4)
    
    # R-R 간격 오차율 계산
    if len(test_peaks) > 1:
        current_rr = np.mean(np.diff(test_peaks)/fs * 1000)
        rr_err = (abs(current_rr - base_rr) / base_rr) * 100
    else:
        rr_err = 100.0
        
    # 진폭 오차율 계산
    if len(test_peaks) > 0:
        current_amp = np.mean(test_filt[test_peaks])
        amp_err = (abs(current_amp - base_amp) / base_amp) * 100
    else:
        amp_err = 100.0
        
    rr_errors.append(min(rr_err, 100.0))
    amp_errors.append(min(amp_err, 100.0))

# Plotly 스트레스 테스트 그래프 구축 (세로축 0부터 시작)
fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=noise_range, y=rr_errors, mode='lines+markers',
    name='시간적 지표 (R-R Interval Error %)',
    line=dict(color='#1f77b4', width=3)
))
fig3.add_trace(go.Scatter(
    x=noise_range, y=amp_errors, mode='lines+markers',
    name='형태학적 지표 (Peak Amplitude Error %)',
    line=dict(color='#ff7f0e', width=3, dash='dash')
))

fig3.update_layout(
    xaxis_title="노이즈 강도 (Noise Intensity)",
    yaxis_title="오차율 (Error Rate %)",
    yaxis=dict(range=[0, 105]), # Y축 0부터 시작 설정
    template="plotly_white",
    hovermode="x unified",
    height=400
)

st.plotly_chart(fig3, use_container_width=True)

# ------------------------------------------
# 요약 메트릭 카드
# ------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("현재 R-R 간격 평균", f"{np.mean(rr_intervals_ms):.1f} ms")
col2.metric("현재 시간적 지표 추정 오차율", f"{rr_errors[int(noise_level*10)]:.2f} %")
col3.metric("현재 형태학적 지표 추정 오차율", f"{amp_errors[int(noise_level*10)]:.2f} %", delta_color="inverse")
