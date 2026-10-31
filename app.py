import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt

# 웹페이지 기본 설정
st.set_page_config(page_title="실시간 ECG 및 R-R 간격 분석기", layout="wide")

st.title("🫀 실시간 웨어러블 ECG 및 R-R 간격 분석 웹페이지")
st.markdown("날것의 심전도(ECG) 데이터 파일을 업로드하면 자동으로 대역통과 필터를 거쳐 R-R 간격을 분석하고 실시간 그래프로 보여줍니다.")

# 사이드바 설정 (분석 파라미터 조절)
st.sidebar.header("⚙️ 분석 파라미터 설정")
fs = st.sidebar.number_input("샘플링 레이트 (Hz)", value=360, help="예제 데이터 기준 360Hz")
lowcut = st.sidebar.number_input("High-pass 컷오프 (Hz)", value=0.5)
highcut = st.sidebar.number_input("Low-pass 컷오프 (Hz)", value=40.0)
peak_height = st.sidebar.slider("R-Peak 탐지 임계값 (Height)", 0.1, 2.0, 0.4, 0.05)

# 버터워스 대역통과 필터 함수 정의
def butter_bandpass_filter(data, low, high, fs, order=4):
    nyq = 0.5 * fs
    low_norm = low / nyq
    high_norm = high / nyq
    b, a = butter(order, [low_norm, high_norm], btype='band')
    return filtfilt(b, a, data)

# 파일 업로드 기능
uploaded_file = st.file_uploader("📂 날것의 ECG 데이터 파일 업로드 (CSV 또는 TXT)", type=['csv', 'txt'])

signal = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, header=None)
        signal = df.iloc[:, 0].values.astype(float)
        st.success("✨ ECG 데이터가 성공적으로 업로드되었습니다!")
    except Exception as e:
        st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
else:
    st.info("💡 업로드할 파일이 없다면, 아래 버튼을 눌러 샘플 데이터로 테스트해 보세요!")
    if st.button("🚀 샘플 ECG 데이터로 테스트하기"):
        t_sample = np.linspace(0, 5, int(fs * 5))
        signal = np.sin(2 * np.pi * 1.2 * t_sample) * 0.3
        for t in np.arange(0.5, 5, 0.8):
            idx = np.abs(t_sample - t).argmin()
            signal[idx:idx+15] += np.linspace(0, 2.5, 15)
            signal[idx+15:idx+30] += np.linspace(2.5, 0, 15)
        signal += 0.2 * np.sin(2 * np.pi * 0.2 * t_sample) + np.random.normal(0, 0.08, len(t_sample))
        st.success("✨ 샘플 ECG 데이터가 로드되었습니다!")

# 신호가 존재할 경우 자동 처리 및 시각화 실행
if signal is not None:
    time_axis = np.arange(len(signal)) / fs

    # 1. 대역통과 필터 적용
    filtered_signal = butter_bandpass_filter(signal, lowcut, highcut, fs)

    # 2. R-Peak 탐지
    peaks, _ = find_peaks(filtered_signal, height=peak_height, distance=int(fs * 0.4))

    # 3. R-R 간격 및 평균 심박수(HR) 계산
    if len(peaks) > 1:
        rr_intervals_ms = np.diff(peaks) / fs * 1000
        mean_rr = np.mean(rr_intervals_ms)
        mean_hr = 60000 / mean_rr
    else:
        rr_intervals_ms = np.array([])
        mean_rr = 0
        mean_hr = 0

    # 결과 지표 대시보드 출력
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("평균 심박수 (HR)", f"{mean_hr:.1f} BPM" if mean_hr > 0 else "측정 불가")
    col2.metric("평균 R-R 간격", f"{mean_rr:.1f} ms" if mean_rr > 0 else "측정 불가")
    col3.metric("탐지된 R-Peak 수", f"{len(peaks)} 개")

    # 4. 실시간 그래프 시각화 (원본 vs 필터링된 파형)
    st.subheader("📊 1. 원본 vs 필터링된 ECG 파형 & R-Peak 탐지 결과")
    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=300)
    ax.plot(time_axis, signal, color='#ff7f0e', alpha=0.6, label='Raw ECG (필터 전 원본)', linewidth=1)
    ax.plot(time_axis, filtered_signal, color='#1f77b4', label='Filtered ECG (필터 후 정제됨)', linewidth=1.5)
    ax.plot(time_axis[peaks], filtered_signal[peaks], "ro", markersize=6, label='Detected R-Peaks')
    ax.set_xlabel("Time (seconds)", fontweight='bold')
    ax.set_ylabel("Amplitude (mV)", fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')
    st.pyplot(fig)

    # 5. R-R 간격 타코그램
    if len(peaks) > 1:
        st.subheader("📈 2. R-R 간격 타코그램 (심장박동 주기 변화)")
        fig2, ax2 = plt.subplots(figsize=(12, 3.5), dpi=300)
        ax2.plot(range(1, len(rr_intervals_ms)+1), rr_intervals_ms, marker='o', color='#2ca02c', linestyle='-', linewidth=1.5, markersize=5)
        ax2.set_xlabel("Beat Index", fontweight='bold')
        ax2.set_ylabel("R-R Interval (ms)", fontweight='bold')
        ax2.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig2)
