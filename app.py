"""
齿轮磨损数据分析系统 - 主应用

基于Streamlit的Web应用，提供完整的齿轮振动信号分析功能。

运行方式:
    streamlit run app.py

功能页面:
    - 数据浏览: 查看原始信号波形
    - 信号分析: 时域/频域/时频分析
    - 特征对比: 多工况特征对比
    - 齿轮诊断: 基于特征的故障诊断
"""

import streamlit as st
import numpy as np
from pathlib import Path

# 页面配置
st.set_page_config(
    page_title="齿轮磨损分析系统",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 导入组件
from ui.sidebar import render_sidebar, render_comparison_sidebar, UIConfig
from data.loader import DataLoader
from processing.time_domain import TimeDomainAnalyzer
from processing.frequency_analyzer import FrequencyAnalyzer
from processing.gear_analyzer import GearAnalyzer, create_default_gear_params
from visualization.time_plots import TimeDomainPlotter
from visualization.freq_plots import FrequencyPlotter
from visualization.timefreq_plots import TimeFrequencyPlotter
from visualization.comparison_plots import ComparisonPlotter
from config.settings import GEAR_STATES, SENSORS, AXES


# ====================================
# 会话状态初始化
# ====================================

if 'data_loader' not in st.session_state:
    st.session_state.data_loader = DataLoader(validate=False)
    st.session_state.cache = {}


# ====================================
# 工具函数
# ====================================

@st.cache_data(ttl=600)
def load_signal_data(_loader, drive_state, driven_state, torque, sensor, axis):
    """加载信号数据（带缓存）"""
    return _loader.load(drive_state, driven_state, torque, sensor, axis)


# ====================================
# 页面: 数据浏览
# ====================================

def page_data_viewer():
    """数据浏览页面"""
    st.title("📊 数据浏览")

    # 渲染侧边栏
    config = render_sidebar()

    # 加载数据
    with st.spinner("加载数据中..."):
        signal_data = load_signal_data(
            st.session_state.data_loader,
            config.drive_state,
            config.driven_state,
            config.torque,
            config.sensor,
            config.axis
        )

    if signal_data is None:
        st.error("❌ 无法加载数据，请检查文件是否存在")
        return

    # 应用时间范围
    start_idx = int(config.time_start * signal_data.sampling_rate)
    end_idx = int(config.time_end * signal_data.sampling_rate)
    signal_segment = signal_data.time_series[start_idx:end_idx]

    # 显示数据信息
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("采样率", f"{signal_data.sampling_rate} Hz")
    with col2:
        st.metric("时长", f"{signal_data.duration:.2f} 秒")
    with col3:
        st.metric("数据点数", f"{len(signal_data):,}")
    with col4:
        st.metric("显示范围", f"{config.time_end - config.time_start:.1f} 秒")

    # 基础统计
    st.subheader("📈 基础统计")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("均值", f"{np.mean(signal_segment):.6f}")
    with col2:
        st.metric("标准差", f"{np.std(signal_segment):.6f}")
    with col3:
        st.metric("最大值", f"{np.max(signal_segment):.6f}")
    with col4:
        st.metric("最小值", f"{np.min(signal_segment):.6f}")

    # 绘制波形
    st.subheader("🌊 时域波形")
    plotter = TimeDomainPlotter()
    fig = plotter.plot_waveform(
        signal_segment,
        signal_data.sampling_rate,
        title=f"{GEAR_STATES[config.drive_state]}-{GEAR_STATES[config.driven_state]} | {SENSORS[config.sensor]}_{AXES[config.axis]}",
        show_envelope=config.show_envelope
    )
    st.plotly_chart(fig, use_container_width=True)


# ====================================
# 页面: 信号分析
# ====================================

def page_signal_analysis():
    """信号分析页面"""
    st.title("🔬 信号分析")

    # 渲染侧边栏
    config = render_sidebar()

    # 加载数据
    with st.spinner("加载数据中..."):
        signal_data = load_signal_data(
            st.session_state.data_loader,
            config.drive_state,
            config.driven_state,
            config.torque,
            config.sensor,
            config.axis
        )

    if signal_data is None:
        st.error("❌ 无法加载数据")
        return

    # 应用时间范围
    start_idx = int(config.time_start * signal_data.sampling_rate)
    end_idx = int(config.time_end * signal_data.sampling_rate)
    signal_segment = signal_data.time_series[start_idx:end_idx]

    # 分析选项
    analysis_type = st.radio(
        "选择分析类型",
        ["时域分析", "频域分析", "时频分析"],
        horizontal=True
    )

    if analysis_type == "时域分析":
        st.subheader("⏱️ 时域特征分析")

        # 提取时域特征
        with st.spinner("计算时域特征..."):
            analyzer = TimeDomainAnalyzer()
            features = analyzer.extract_features(signal_segment)

        # 显示特征
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**基础统计特征**")
            st.write(f"- RMS: {features.rms:.6f}")
            st.write(f"- 峰值: {features.peak:.6f}")
            st.write(f"- 峰峰值: {features.peak_to_peak:.6f}")
            st.write(f"- 标准差: {features.std:.6f}")

        with col2:
            st.markdown("**形状特征**")
            st.write(f"- 偏度: {features.skewness:.4f}")
            st.write(f"- 峰度: {features.kurtosis:.4f}")
            st.write(f"- 波峰因子: {features.crest_factor:.4f}")
            st.write(f"- 裕度因子: {features.clearance_factor:.4f}")

    elif analysis_type == "频域分析":
        st.subheader("📡 频域特征分析")

        # 提取频域特征
        with st.spinner("计算频域特征..."):
            analyzer = FrequencyAnalyzer()
            features = analyzer.extract_features(signal_segment, signal_data.sampling_rate)

        # 显示特征
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**频率特征**")
            st.write(f"- 主频率: {features.dominant_frequency:.2f} Hz")
            st.write(f"- 频谱质心: {features.spectral_centroid:.2f} Hz")
            st.write(f"- 频谱扩展度: {features.spectral_spread:.2f} Hz")

        with col2:
            st.markdown("**谐波特征**")
            st.write(f"- 谐波比: {features.harmonic_ratio:.4f}")
            st.write(f"- 总谐波失真: {features.thd:.2f}%")
            st.write(f"- 峰值数量: {features.peak_count}")

        # 绘制频谱
        freq_plotter = FrequencyPlotter()
        fig = freq_plotter.plot_spectrum(
            signal_segment,
            signal_data.sampling_rate,
            freq_range=(0, config.freq_range_max),
            title="FFT频谱"
        )
        st.plotly_chart(fig, use_container_width=True)

    else:  # 时频分析
        st.subheader("🎵 时频分析")

        # STFT声谱图
        with st.spinner("计算时频谱..."):
            tf_plotter = TimeFrequencyPlotter()
            fig = tf_plotter.plot_spectrogram(
                signal_segment,
                signal_data.sampling_rate,
                nperseg=config.nperseg,
                freq_range=(0, config.freq_range_max)
            )
        st.plotly_chart(fig, use_container_width=True)


# ====================================
# 页面: 齿轮诊断
# ====================================

def page_gear_diagnosis():
    """齿轮诊断页面"""
    st.title("🔧 齿轮诊断")

    # 渲染侧边栏
    config = render_sidebar()

    # 加载数据
    with st.spinner("加载数据中..."):
        signal_data = load_signal_data(
            st.session_state.data_loader,
            config.drive_state,
            config.driven_state,
            config.torque,
            config.sensor,
            config.axis
        )

    if signal_data is None:
        st.error("❌ 无法加载数据")
        return

    # 应用时间范围
    start_idx = int(config.time_start * signal_data.sampling_rate)
    end_idx = int(config.time_end * signal_data.sampling_rate)
    signal_segment = signal_data.time_series[start_idx:end_idx]

    # 齿轮参数
    gear_params = create_default_gear_params(shaft_speed=1000)

    st.subheader("⚙️ 齿轮参数")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("主动轮齿数", gear_params.drive_teeth)
    with col2:
        st.metric("从动轮齿数", gear_params.driven_teeth)
    with col3:
        st.metric("啮合频率", f"{gear_params.mesh_freq:.2f} Hz")

    # 提取齿轮特征
    with st.spinner("分析齿轮特征..."):
        gear_analyzer = GearAnalyzer(gear_params)
        features = gear_analyzer.extract_gear_features(signal_segment, signal_data.sampling_rate)
        diagnosis = gear_analyzer.diagnose_condition(features)

    # 诊断结果
    st.subheader("📋 诊断结果")

    # 状态显示
    status_color = {
        "正常": "🟢",
        "轻度磨损": "🟡",
        "中度磨损": "🟠",
        "严重磨损": "🔴"
    }

    st.markdown(f"### {status_color.get(diagnosis['condition'], '⚪')} {diagnosis['condition']}")
    st.write(f"**严重程度**: {diagnosis['severity']}")
    st.write(f"**故障因子**: {diagnosis['fault_factor']}")

    st.info(f"**GMF分析**: {diagnosis['gmf_indication']}")
    st.info(f"**边频带分析**: {diagnosis['sideband_indication']}")

    # 诊断特征
    st.subheader("🔍 诊断特征")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("GMF幅值", f"{features.gmf_amplitude:.6f}")
    with col2:
        st.metric("GMF能量比", f"{features.gmf_energy_ratio:.4f}")
    with col3:
        st.metric("边频带数量", features.sideband_count)
    with col4:
        st.metric("磨损指标", f"{features.wear_indicator:.4f}")

    # 绘制齿轮频谱
    st.subheader("📊 齿轮特征频谱")
    freq_plotter = FrequencyPlotter()
    fig = freq_plotter.plot_gear_spectrum(
        signal_segment,
        signal_data.sampling_rate,
        gear_params,
        freq_range=(0, 2000),
        num_harmonics=4
    )
    st.plotly_chart(fig, use_container_width=True)


# ====================================
# 页面: 特征对比
# ====================================

def page_comparison():
    """特征对比页面"""
    st.title("📊 特征对比")

    # 渲染对比侧边栏
    comp_config = render_comparison_sidebar()

    if comp_config['mode'] == "磨损状态对比":
        st.subheader(f"🔬 磨损状态对比 - {SENSORS[comp_config['sensor']]}_{AXES[comp_config['axis']]}")

        features_dict = {}

        for state in comp_config['states']:
            with st.spinner(f"加载 {GEAR_STATES[state]}..."):
                signal_data = load_signal_data(
                    st.session_state.data_loader,
                    state,
                    'normal',
                    comp_config['torque'],
                    comp_config['sensor'],
                    comp_config['axis']
                )

                if signal_data:
                    # 使用前10秒数据
                    test_samples = min(10 * signal_data.sampling_rate, len(signal_data))
                    signal_segment = signal_data.time_series[:test_samples]

                    # 提取时域特征
                    analyzer = TimeDomainAnalyzer()
                    td_features = analyzer.extract_features(signal_segment)

                    features_dict[GEAR_STATES[state]] = {
                        'RMS': td_features.rms,
                        '峰值': td_features.peak,
                        '峰度': td_features.kurtosis,
                        '偏度': td_features.skewness,
                        '波峰因子': td_features.crest_factor
                    }

        if features_dict:
            # 绘制对比图
            comp_plotter = ComparisonPlotter()

            col1, col2 = st.columns(2)
            with col1:
                fig1 = comp_plotter.plot_feature_comparison(
                    features_dict,
                    title="时域特征对比",
                    normalize=False
                )
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                fig2 = comp_plotter.plot_radar_chart(
                    features_dict,
                    title="特征雷达图"
                )
                st.plotly_chart(fig2, use_container_width=True)

        else:
            st.warning("⚠️ 没有可用的数据进行对比")

    elif comp_config['mode'] == "传感器位置对比":
        st.subheader(f"🔬 传感器位置对比 - {GEAR_STATES[comp_config['drive_state']]}-{GEAR_STATES[comp_config['driven_state']]}")

        # 添加传感器说明
        st.info(f"""
        **传感器位置对比分析**
        - 对比传感器: {', '.join([f"{SENSORS[s.split('_')[0]]}_{AXES[s.split('_')[1]]}" for s in comp_config['sensors']])}
        - 齿轮状态: {GEAR_STATES[comp_config['drive_state']]}-{GEAR_STATES[comp_config['driven_state']]}
        - 扭矩: {comp_config['torque']}Nm
        - 分析维度: 不同位置传感器的振动响应特性
        """)

        time_features_dict = {}
        freq_features_dict = {}

        for sensor_axis in comp_config['sensors']:
            sensor, axis = sensor_axis.split('_')
            sensor_label = f"{SENSORS[sensor]}_{AXES[axis]}"

            with st.spinner(f"加载 {sensor_label}..."):
                signal_data = load_signal_data(
                    st.session_state.data_loader,
                    comp_config['drive_state'],
                    comp_config['driven_state'],
                    comp_config['torque'],
                    sensor,
                    axis
                )

                if signal_data:
                    # 使用前10秒数据
                    test_samples = min(10 * signal_data.sampling_rate, len(signal_data))
                    signal_segment = signal_data.time_series[:test_samples]

                    # 提取时域特征
                    td_analyzer = TimeDomainAnalyzer()
                    td_features = td_analyzer.extract_features(signal_segment)

                    time_features_dict[sensor_label] = {
                        'RMS': td_features.rms,
                        '峰值': td_features.peak,
                        '峰度': td_features.kurtosis,
                        '偏度': td_features.skewness,
                        '波峰因子': td_features.crest_factor
                    }

                    # 提取频域特征
                    freq_analyzer = FrequencyAnalyzer(signal_data.sampling_rate)
                    freq_result = freq_analyzer.compute_fft(signal_segment)

                    freq_features_dict[sensor_label] = {
                        '主频幅值': freq_result.dominant_freq_magnitude,
                        '频谱能量': freq_result.total_power,
                        '频谱熵': freq_result.spectral_entropy,
                        '频率重心': freq_result.spectral_centroid,
                    }

        if time_features_dict:
            comp_plotter = ComparisonPlotter()

            # 时域特征对比
            st.markdown("### 📈 时域特征对比")
            col1, col2 = st.columns(2)
            with col1:
                fig1 = comp_plotter.plot_feature_comparison(
                    time_features_dict,
                    title="传感器时域特征对比",
                    normalize=False
                )
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                fig2 = comp_plotter.plot_radar_chart(
                    time_features_dict,
                    title="时域特征雷达图"
                )
                st.plotly_chart(fig2, use_container_width=True)

            # 频域特征对比
            if freq_features_dict:
                st.markdown("### 🌊 频域特征对比")
                col3, col4 = st.columns(2)
                with col3:
                    fig3 = comp_plotter.plot_feature_comparison(
                        freq_features_dict,
                        title="传感器频域特征对比",
                        normalize=False
                    )
                    st.plotly_chart(fig3, use_container_width=True)

                with col4:
                    fig4 = comp_plotter.plot_radar_chart(
                        freq_features_dict,
                        title="频域特征雷达图"
                    )
                    st.plotly_chart(fig4, use_container_width=True)

            # 传感器位置分析
            st.markdown("### 📍 传感器位置说明")
            col5, col6, col7 = st.columns(3)

            with col5:
                st.markdown("**传感器A**")
                st.markdown("📌 主动轴输入轴承处")
                st.caption("监测主动轴和输入轴承的振动特性")
                if "传感器A_X轴" in time_features_dict:
                    st.metric("RMS", f"{time_features_dict['传感器A_X轴']['RMS']:.4f}")

            with col6:
                st.markdown("**传感器B**")
                st.markdown("📌 从动轴输入处")
                st.caption("监测齿轮啮合区域的振动特性（最敏感）")
                if "传感器B_X轴" in time_features_dict:
                    st.metric("RMS", f"{time_features_dict['传感器B_X轴']['RMS']:.4f}")

            with col7:
                st.markdown("**传感器C**")
                st.markdown("📌 从动轴输出处")
                st.caption("监测从动轴输出端的振动特性")
                if "传感器C_X轴" in time_features_dict:
                    st.metric("RMS", f"{time_features_dict['传感器C_X轴']['RMS']:.4f}")

        else:
            st.warning("⚠️ 没有可用的数据进行对比")

    elif comp_config['mode'] == "工况参数对比":
        st.subheader(f"🔬 工况参数对比 - {GEAR_STATES[comp_config['drive_state']]}-{GEAR_STATES[comp_config['driven_state']]}")

        # 添加工况说明
        st.info(f"""
        **工况参数对比分析**
        - 对比扭矩: {', '.join([f'{t}Nm' for t in comp_config['torques']])}
        - 齿轮状态: {GEAR_STATES[comp_config['drive_state']]}-{GEAR_STATES[comp_config['driven_state']]}
        - 传感器: {SENSORS[comp_config['sensor']]}_{AXES[comp_config['axis']]}
        - 分析维度: 扭矩变化对振动特性的影响
        """)

        time_features_dict = {}
        freq_features_dict = {}
        signal_data_dict = {}

        for torque in comp_config['torques']:
            torque_label = f"{torque}Nm"

            with st.spinner(f"加载 {torque_label} 数据..."):
                signal_data = load_signal_data(
                    st.session_state.data_loader,
                    comp_config['drive_state'],
                    comp_config['driven_state'],
                    torque,
                    comp_config['sensor'],
                    comp_config['axis']
                )

                if signal_data:
                    signal_data_dict[torque_label] = signal_data

                    # 使用前10秒数据
                    test_samples = min(10 * signal_data.sampling_rate, len(signal_data))
                    signal_segment = signal_data.time_series[:test_samples]

                    # 提取时域特征
                    td_analyzer = TimeDomainAnalyzer()
                    td_features = td_analyzer.extract_features(signal_segment)

                    time_features_dict[torque_label] = {
                        'RMS': td_features.rms,
                        '峰值': td_features.peak,
                        '峰度': td_features.kurtosis,
                        '偏度': td_features.skewness,
                        '波峰因子': td_features.crest_factor
                    }

                    # 提取频域特征
                    freq_analyzer = FrequencyAnalyzer(signal_data.sampling_rate)
                    freq_result = freq_analyzer.compute_fft(signal_segment)

                    freq_features_dict[torque_label] = {
                        '主频幅值': freq_result.dominant_freq_magnitude,
                        '频谱能量': freq_result.total_power,
                        '频谱熵': freq_result.spectral_entropy,
                        '频率重心': freq_result.spectral_centroid,
                    }

        if time_features_dict:
            comp_plotter = ComparisonPlotter()

            # 时域特征对比
            st.markdown("### 📈 时域特征对比")
            col1, col2 = st.columns(2)
            with col1:
                fig1 = comp_plotter.plot_feature_comparison(
                    time_features_dict,
                    title="时域特征对比",
                    normalize=False
                )
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                fig2 = comp_plotter.plot_radar_chart(
                    time_features_dict,
                    title="时域特征雷达图"
                )
                st.plotly_chart(fig2, use_container_width=True)

            # 频域特征对比
            if freq_features_dict:
                st.markdown("### 🌊 频域特征对比")
                col3, col4 = st.columns(2)
                with col3:
                    fig3 = comp_plotter.plot_feature_comparison(
                        freq_features_dict,
                        title="频域特征对比",
                        normalize=False
                    )
                    st.plotly_chart(fig3, use_container_width=True)

                with col4:
                    fig4 = comp_plotter.plot_radar_chart(
                        freq_features_dict,
                        title="频域特征雷达图"
                    )
                    st.plotly_chart(fig4, use_container_width=True)

            # 工况对比分析结论
            st.markdown("### 📊 对比分析")
            col5, col6 = st.columns(2)

            with col5:
                st.markdown("**时域特征趋势**")
                if len(time_features_dict) >= 2:
                    torques_sorted = sorted(comp_config['torques'])
                    if len(torques_sorted) >= 2:
                        t1_label = f"{torques_sorted[0]}Nm"
                        t2_label = f"{torques_sorted[-1]}Nm"

                        rms_change = (time_features_dict[t2_label]['RMS'] / time_features_dict[t1_label]['RMS'] - 1) * 100
                        peak_change = (time_features_dict[t2_label]['峰值'] / time_features_dict[t1_label]['峰值'] - 1) * 100

                        st.metric(
                            label=f"RMS变化 ({t1_label}→{t2_label})",
                            value=f"{time_features_dict[t2_label]['RMS']:.4f}",
                            delta=f"{rms_change:+.1f}%"
                        )
                        st.metric(
                            label=f"峰值变化 ({t1_label}→{t2_label})",
                            value=f"{time_features_dict[t2_label]['峰值']:.4f}",
                            delta=f"{peak_change:+.1f}%"
                        )

            with col6:
                st.markdown("**频域特征趋势**")
                if len(freq_features_dict) >= 2 and len(torques_sorted) >= 2:
                    energy_change = (freq_features_dict[t2_label]['频谱能量'] / freq_features_dict[t1_label]['频谱能量'] - 1) * 100

                    st.metric(
                        label=f"频谱能量变化 ({t1_label}→{t2_label})",
                        value=f"{freq_features_dict[t2_label]['频谱能量']:.2e}",
                        delta=f"{energy_change:+.1f}%"
                    )
                    st.metric(
                        label="频率重心",
                        value=f"{freq_features_dict[t2_label]['频率重心']:.2f} Hz"
                    )

        else:
            st.warning("⚠️ 没有可用的数据进行对比")

    else:
        st.info("ℹ️ 其他对比模式正在开发中...")


# ====================================
# 主入口
# ====================================

def main():
    """主函数"""

    # 页面导航
    pages = {
        "数据浏览": page_data_viewer,
        "信号分析": page_signal_analysis,
        "齿轮诊断": page_gear_diagnosis,
        "特征对比": page_comparison
    }

    # 选择页面
    page = st.sidebar.radio(
        "导航",
        list(pages.keys()),
        key="navigation"
    )

    st.sidebar.markdown("---")

    # 显示页面
    pages[page]()

    # 页脚
    st.sidebar.markdown("---")
    st.sidebar.caption("齿轮磨损数据分析系统 v1.0")
    st.sidebar.caption("© 2024 马辉教授课题组")


if __name__ == "__main__":
    main()
