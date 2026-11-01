# ==========================================
    # [구역 3] 지표별 오차율 비교 (강건성 증명 기능)
    # ==========================================
    st.markdown("---")
    st.header("🏆 [구역 3] 지표별 노이즈 영향 오차율 비교 (왜 R-R 간격이 최고인가?)")
    st.markdown("노이즈 강도가 높아질 때, 형태학적 특징(Amplitude, Prominence)은 크게 무너지지만 **시간적 특징인 R-R 간격은 왜곡이 적어 강건함(Robustness)**을 유지함을 증명합니다.")

    features = ['R-R Interval\n(시간적 지표)', 'Peak Amplitude\n(형태학적 지표)', 'Peak Prominence\n(형태학적 지표)']
    errors = [err_rr, err_amp, err_prom]
    colors = ['#2ca02c', '#d62728', '#9467bd']

    # figsize를 조금 더 넉넉하게 잡고 tight_layout 적용하여 글자 잘림 방지
    fig3, ax3 = plt.subplots(figsize=(10, 4.5), dpi=300)
    bars = ax3.bar(features, errors, color=colors, width=0.4)
    
    ax3.set_ylabel("Error Rate (%)", fontweight='bold')
    ax3.set_ylim(0, max(max(errors) * 1.3, 10))  # 세로축 0부터 시작
    ax3.grid(axis='y', linestyle='--', alpha=0.5)

    # 수치 라벨 추가
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.2f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()  # 글자가 잘리지 않도록 여백 자동 조절
    st.pyplot(fig3)
