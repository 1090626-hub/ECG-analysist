import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt

# 웹페이지 기본 설정
st.set_page_config(page_title="실시간 ECG 및 R-R 간격 분석기", layout="wide")

st.title("🫀 웨어러블 ECG 노이즈 강건성 실시간 검증 웹앱")
st.markdown("잔여 노이즈 환경에서 **형태학적 특징(Amplitude, Prominence)**과 **시간적 특징(R-R 간격)**의 강건성을 직접 비교 분석합니다.")

# 사이드바 설정 (분석 파라미터 조절)
st.sidebar.header("⚙️ 분석 파라미터 및 노이즈 설정")
fs = st.sidebar.number_input("샘플링 레이트 (Hz)", value=360, help="예제 데이터 기준 360Hz")
lowcut = st.sidebar.number_input("High-pass 컷오프 (Hz)", value=0.5)
highcut = st.sidebar.number_input("Low-pass 컷오프 (Hz)", value=40.0)
peak_height = st.sidebar.slider("R-Peak 탐지 임계값 (Height)", 0.1, 2.0, 0.4, 0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("🧪 노이즈 강건성 실험")
noise_level = st.sidebar.slider("잔여 노이즈 강도 (Noise Intensity)", 0.0, 1.0, 0.3, 0.05, help="슬라이더를 올릴수록 ECG 신호에 고주파/기저선 잡음이 주입됩니다.")

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

    # 1. 사이드바 설정에 맞춰 인위적 노이즈 주입
    noisy_signal = clean_signal + noise_level * np.random.normal(0, 0.25, len(clean_signal))

    # 2. 대역통과 필터 적용
    filtered_signal = butter_bandpass_filter(noisy_signal, lowcut, highcut, fs)

    # 3. R-Peak 탐지 (깨끗한 신호 vs 필터링된 노이즈 신호)
    peaks_clean, props_clean = find_peaks(clean_signal, height=peak_height, distance=int(fs * 0.4), prominence=0.2)
    peaks_noisy, props_noisy = find_peaks(filtered_signal, height=peak_height, distance=int(fs * 0.4), prominence=0.2)

    # 지표 산출 함수
    def calc_metrics(peaks, signal, props):
        if len(peaks) > 1:
            rr = np.diff(peaks) / fs * 1000
            amp = signal[peaks]
            prom = props.get('prominences', np.ones(len(peaks)) * 0.5)
            return np.mean(rr), np.mean(amp), np.mean(prom)
        return 0, 0, 0

    rr_c, amp_c, prom_c = calc_metrics(peaks_clean, clean_signal, props_clean)
    rr_n, amp_n, prom_n = calc_metrics(peaks_noisy, filtered_signal, props_noisy)

    # 오차율(%) 계산
    err_rr = abs(rr_n - rr_c) / (rr_c + 1e-6) * 100 if rr_c > 0 else 0
    err_amp = abs(amp_n - amp_c) / (amp_c + 1e-6) * 100 if amp_c > 0 else 0
    err_prom = abs(prom_n - prom_c) / (prom_c + 1e-6) * 100 if prom_c > 0 else 0

    # 지표 대시보드
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("R-R 간격 오차율 (시간적 지표)", f"{err_rr:.2f}%", delta_color="normal")
    col2.metric("Peak Amplitude 오차율 (형태 지표)", f"{err_amp:.2f}%", delta_color="inverse")
    col3.metric("Peak Prominence 오차율 (형태 지표)", f"{err_prom:.2f}%", delta_color="inverse")

    # 4. 파형 시각화
    st.subheader("📊 1. 노이즈 주입 및 대역통과 필터 정제 파형")
    fig, ax = plt.subplots(figsize=(12, 4), dpi=300)
    ax.plot(time_axis, noisy_signal, color='#ff7f0e', alpha=0.5, label='Noisy Raw ECG (노이즈 주입됨)', linewidth=1)
    ax.plot(time_axis, filtered_signal, color='#1f77b4', label='Filtered ECG (필터 후 정제됨)', linewidth=1.5)
    ax.plot(time_axis[peaks_noisy], filtered_signal[peaks_noisy], "ro", markersize=6, label='Detected R-Peaks')
    ax.set_xlabel("Time (seconds)", fontweight='bold')
    ax.set_ylabel("Amplitude (mV)", fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')
    st.pyplot(fig)

    # 5. 핵심: 왜 R-R 간격이 최적의 Feature인지 증명하는 막대그래프
    st.markdown("---")
    st.subheader("🏆 2. 지표별 노이즈 영향 오차율 비교 (Why R-R Interval is Best?)")
    st.markdown("노이즈 세기가 증가할수록 형태학적 지표(Amplitude, Prominence)는 크게 왜곡되지만, **R-R 간격(시간적 지표)은 매우 낮은 오차율을 유지하며 강력한 강건성(Robustness)**을 보입니다.")

    features = ['R-R Interval\n(시간적 지표)', 'Peak Amplitude\n(형태학적 지표)', 'Peak Prominence\n(형태학적 지표)']
    errors = [err_rr, err_amp, err_prom]
    colors = ['#2ca02c', '#d62728', '#9467bd']

    fig_bar, ax_bar = plt.subplots(figsize=(9, 4), dpi=300)
    bars = ax_bar.bar(features, errors, color=colors, width=0.45)
    ax_bar.set_ylabel("Error Rate (%)", fontweight='bold')
    ax_bar.set_ylim(0, max(max(errors) * 1.3, 10))  # 세로축 0부터 시작하도록 보장
    ax_bar.grid(axis='y', linestyle='--', alpha=0.5)

    # 수치 라벨 추가
    for bar in bars:
        height = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.2f}%', ha='center', va='bottom', fontweight='bold')

    st.pyplot(fig_bar)
