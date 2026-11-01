import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.signal import find_peaks, butter, filtfilt

# 웹페이지 기본 설정
st.set_page_config(page_title="실시간 ECG 및 R-R 간격 분석기", layout="wide")

st.title("🫀 웨어러블 ECG 노이즈 강건성 및 R-R 간격 분석 웹앱")
st.markdown("날것의 심전도(ECG) 데이터에 잔여 노이즈를 주입하고, 대역통과 필터 전후의 **파형, 인터랙티브 R-R 간격 타코그램, 지표별 오차율 비교**를 구역별로 나누어 정밀 분석합니다.")

# 사이드바 설정 (분석 파라미터 조절)
st.sidebar.header("⚙️ 분석 파라미터 및 노이즈 설정")
fs = st.sidebar.number_input("샘플링 레이트 (Hz)", value=360, help="예제 데이터 기준 360Hz")
lowcut = st.sidebar.number_input("High-pass 컷오프 (Hz)", value=0.5)
highcut = st.sidebar.number_input("Low-pass 컷오프 (Hz)", value=40.0)
peak_height = st.sidebar.slider("R-Peak 탐지 임계값 (Height)", 0.1, 2.0, 0.4, 0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("🧪 노이즈 강건성 실험")
noise_level = st.sidebar.slider("잔여 노이즈 강도 (Noise Intensity)", 0.0, 1.0, 0.2, 0.05, help="슬라이더를 올릴수록 ECG 신호에 고주파/기저선 잡음이 주입됩니다.")

# 버터워스 대역통과 필터 함수 정의
def butter_bandpass_filter(data, low, high, fs, order=4):
    nyq = 0.5 * fs
    low_norm = low / nyq
    high_norm = high / nyq
    b, a = butter(order, [low_norm, high_norm], btype='band')
    return filtfilt(b, a, data)

# 파일 업로드 기능
uploaded_file = st.file_uploader("📂 날것의 ECG 데이터 파일 업로드 (CSV 또는 TXT)", type=['csv', 'txt'])

clean_signal = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, header=None)
        clean_signal = df.iloc[:, 0].values.astype(float)
        st.success("✨ ECG 데이터가 성공적으로 업로드되었습니다!")
    except Exception as e:
        st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
else:
    st.info("💡 업로드할 파일이 없다면, 아래 버튼을 눌러 샘플 데이터로 테스트해 보세요!")
    if st.button("🚀 샘플 ECG 데이터로 테스트하기"):
        t_sample = np.linspace(0, 5, int(fs * 5))
        clean_signal = np.sin(2 * np.pi * 1.2 * t_sample) * 0.3
        for t in np.arange(0.5, 5, 0.8):
            idx = np.abs(t_sample - t).argmin()
            clean_signal[idx:idx+15] += np.linspace(0, 2.5, 15)
            clean_signal[idx+15:idx+30] += np.linspace(2.5, 0, 15)
        clean_signal += 0.2 * np.sin(2 * np.pi * 0.2 * t_sample)
        st.success("✨ 샘플 ECG 데이터가 로드되었습니다!")

# 신호가 존재할 경우 자동 처리 및 시각화 실행
if clean_signal is not None:
    time_axis = np.arange(len(clean_signal)) / fs

    # 1. 인위적 노이즈 주입 및 필터링
    noisy_signal = clean_signal + noise_level * np.random.normal(0, 0.25, len(clean_signal))
    filtered_signal = butter_bandpass_filter(noisy_signal, lowcut, highcut, fs)

    # 2. Peak 탐지
    peaks_clean, props_clean = find_peaks(clean_signal, height=peak_height, distance=int(fs * 0.4), prominence=0.2)
    peaks_noisy, props_noisy = find_peaks(filtered_signal, height=peak_height, distance=int(fs * 0.4), prominence=0.2)

    # 기본 R-R 간격 및 심박수 계산
    if len(peaks_noisy) > 1:
        rr_intervals_ms = np.diff(peaks_noisy) / fs * 1000
        mean_rr = np.mean(rr_intervals_ms)
        mean_hr = 60000 / mean_rr
    else:
        rr_intervals_ms = np.array([])
        mean_rr = 0
        mean_hr = 0

    # 지표 산출 함수 (오차율 계산용)
    def calc_metrics(peaks, signal, props):
        if len(peaks) > 1:
            rr = np.diff(peaks) / fs * 1000
            amp = signal[peaks]
            prom = props.get('prominences', np.ones(len(peaks)) * 0.5)
            return np.mean(rr), np.mean(amp), np.mean(prom)
        return 0, 0, 0

    rr_c, amp_c, prom_c = calc_metrics(peaks_clean, clean_signal, props_clean)
    rr_n, amp_n, prom_n = calc_metrics(peaks_noisy, filtered_signal, props_noisy)

    err_rr = abs(rr_n - rr_c) / (rr_c + 1e-6) * 100 if rr_c > 0 else 0
    err_amp = abs(amp_n - amp_c) / (amp_c + 1e-6) * 100 if amp_c > 0 else 0
    err_prom = abs(prom_n - prom_c) / (prom_c + 1e-6) * 100 if prom_c > 0 else 0

    # 종합 대시보드 요약 지표
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("평균 심박수 (HR)", f"{mean_hr:.1f} BPM" if mean_hr > 0 else "측정 불가")
    col2.metric("평균 R-R 간격", f"{mean_rr:.1f} ms" if mean_rr > 0 else "측정 불가")
    col3.metric("탐지된 R-Peak 수", f"{len(peaks_noisy)} 개")

    # ==========================================
    # [구역 1] 원본 vs 필터링 파형 및 R-Peak 탐지
    # ==========================================
    st.markdown("---")
    st.header("📊 [구역 1] 원본 노이즈 신호 vs 대역통과 필터 정제 파형")
    st.markdown("노이즈가 섞인 원본 신호에서 버터워스 필터가 기저선 흔들림을 잡고 R-Peak를 정확히 포착하는 과정을 보여줍니다.")
    
    fig1, ax1 = plt.subplots(figsize=(12, 4), dpi=300)
    ax1.plot(time_axis, noisy_signal, color='#ff7f0e', alpha=0.5, label='Noisy Raw ECG (노이즈 주입)', linewidth=1)
    ax1.plot(time_axis, filtered_signal, color='#1f77b4', label='Filtered ECG (필터 정제 완료)', linewidth=1.5)
    ax1.plot(time_axis[peaks_noisy], filtered_signal[peaks_noisy], "ro", markersize=6, label='Detected R-Peaks')
    ax1.set_xlabel("Time (seconds)", fontweight='bold')
    ax1.set_ylabel("Amplitude (mV)", fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right')
    st.pyplot(fig1)

    # ==========================================
    # [구역 2] R-R 간격 타코그램 (Plotly 인터랙티브)
    # ==========================================
    st.markdown("---")
    st.header("📈 [구역 2] R-R 간격 타코그램 (인터랙티브 심장박동 주기 시각화)")
    st.markdown("마우스를 그래프의 선이나 점 위에 올리면(Hover) 해당 박동의 **Beat Index와 정확한 R-R 간격(ms)** 수치가 실시간으로 팝업됩니다.")

    if len(peaks_noisy) > 1:
        beat_indices = list(range(1, len(rr_intervals_ms) + 1))
        
        # Plotly를 이용한 인터랙티브 선 그래프 생성
        fig_interactive = go.Figure()
        fig_interactive.add_trace(go.Scatter(
            x=beat_indices,
            y=rr_intervals_ms,
            mode='lines+markers',
            name='R-R Interval',
            line=dict(color='#2ca02c', width=2),
            marker=dict(size=8, color='#2ca02c'),
            hovertemplate='<b>Beat Index</b>: %{x}<br><b>R-R Interval</b>: %{y:.1f} ms<extra></extra>'
        ))
        
        fig_interactive.update_layout(
            xaxis_title="Beat Index",
            yaxis_title="R-R Interval (ms)",
            hovermode="closest",
            margin=dict(l=40, r=40, t=20, b=40),
            height=350,
            plot_bgcolor='rgba(240,242,246,0.5)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_interactive, use_container_width=True)
    else:
        st.warning("R-Peak가 2개 이상 감지되지 않아 타코그램을 그릴 수 없습니다. 임계값을 조절해 보세요.")

    # ==========================================
    # [구역 3] 지표별 오차율 비교 (강건성 증명 기능)
    # ==========================================
    st.markdown("---")
    st.header("🏆 [구역 3] 지표별 노이즈 영향 오차율 비교 (왜 R-R 간격이 최고인가?)")
    st.markdown("노이즈 강도가 높아질 때, 형태학적 특징(Amplitude, Prominence)은 크게 무너지지만 **시간적 특징인 R-R 간격은 왜곡이 적어 강건함(Robustness)**을 유지함을 증명합니다.")

    features = ['R-R Interval\n(시간적 지표)', 'Peak Amplitude\n(형태학적 지표)', 'Peak Prominence\n(형태학적 지표)']
    errors = [err_rr, err_amp, err_prom]
    colors = ['#2ca02c', '#d62728', '#9467bd']

    fig3, ax3 = plt.subplots(figsize=(9, 4), dpi=300)
    bars = ax3.bar(features, errors, color=colors, width=0.45)
    ax3.set_ylabel("Error Rate (%)", fontweight='bold')
    ax3.set_ylim(0, max(max(errors) * 1.3, 10))  # 평가기준 반영: 세로축 0부터 시작
    ax3.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.2f}%', ha='center', va='bottom', fontweight='bold')

    st.pyplot(fig3)
