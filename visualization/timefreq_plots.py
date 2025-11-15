"""
时频图表可视化

实现时频域分析的可视化：
- STFT声谱图
- CWT小波谱
- 小波包能量分布

使用示例:
    >>> from data.loader import DataLoader
    >>> from visualization.timefreq_plots import plot_spectrogram
    >>>
    >>> loader = DataLoader()
    >>> signal_data = loader.load('light_wear', 'normal', 10, 'A', 'X')
    >>>
    >>> fig = plot_spectrogram(signal_data.time_series, signal_data.sampling_rate)
    >>> fig.show()
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Tuple
from dataclasses import dataclass

from config.settings import logger


@dataclass
class TimeFreqPlotConfig:
    """时频图表配置"""
    title: str = "时频图"
    width: int = 1200
    height: int = 700
    template: str = "plotly_white"
    colorscale: str = "Jet"


class TimeFrequencyPlotter:
    """
    时频图表绘制器

    提供各种时频域可视化功能。
    """

    def __init__(self, config: Optional[TimeFreqPlotConfig] = None):
        """
        初始化绘图器

        Args:
            config: 图表配置
        """
        self.config = config or TimeFreqPlotConfig()
        logger.debug("TimeFrequencyPlotter 初始化完成")

    def plot_spectrogram(
        self,
        signal_data: np.ndarray,
        sampling_rate: int,
        nperseg: int = 512,
        title: Optional[str] = None,
        freq_range: Optional[Tuple[float, float]] = None
    ) -> go.Figure:
        """
        绘制STFT声谱图

        Args:
            signal_data: 时域信号
            sampling_rate: 采样率
            nperseg: STFT窗长度
            title: 图表标题
            freq_range: 显示频率范围

        Returns:
            Plotly图表对象
        """
        from processing.timefreq_analyzer import TimeFreqAnalyzer

        # 计算STFT
        analyzer = TimeFreqAnalyzer(nperseg=nperseg)
        stft_result = analyzer.compute_stft(signal_data, sampling_rate)

        time = stft_result.time
        freq = stft_result.frequency
        power_db = 10 * np.log10(stft_result.power + 1e-10)  # 转换为dB

        # 应用频率范围
        if freq_range:
            freq_mask = (freq >= freq_range[0]) & (freq <= freq_range[1])
            freq = freq[freq_mask]
            power_db = power_db[freq_mask, :]

        # 创建热图
        fig = go.Figure()

        fig.add_trace(go.Heatmap(
            x=time,
            y=freq,
            z=power_db,
            colorscale=self.config.colorscale,
            colorbar=dict(title="功率 (dB)"),
            hovertemplate='时间: %{x:.3f}s<br>频率: %{y:.1f}Hz<br>功率: %{z:.1f}dB<extra></extra>'
        ))

        # 更新布局
        plot_title = title or "STFT声谱图"
        fig.update_layout(
            title=dict(
                text=plot_title,
                x=0.5,
                xanchor='center'
            ),
            xaxis_title="时间 (秒)",
            yaxis_title="频率 (Hz)",
            width=self.config.width,
            height=self.config.height,
            template=self.config.template
        )

        logger.debug(f"STFT声谱图绘制完成: {len(time)}×{len(freq)}点")

        return fig

    def plot_wavelet_scalogram(
        self,
        signal_data: np.ndarray,
        sampling_rate: int,
        wavelet: str = 'morl',
        freq_range: Tuple[float, float] = (10, 3000),
        title: Optional[str] = None
    ) -> go.Figure:
        """
        绘制CWT小波谱（Scalogram）

        Args:
            signal_data: 时域信号
            sampling_rate: 采样率
            wavelet: 小波类型
            freq_range: 频率范围
            title: 图表标题

        Returns:
            Plotly图表对象
        """
        from processing.timefreq_analyzer import TimeFreqAnalyzer

        # 计算CWT
        analyzer = TimeFreqAnalyzer()
        cwt_result = analyzer.compute_cwt(
            signal_data,
            sampling_rate,
            wavelet=wavelet,
            freq_range=freq_range
        )

        time = cwt_result.time
        freq = cwt_result.frequencies
        power_db = 10 * np.log10(cwt_result.power + 1e-10)

        # 创建热图
        fig = go.Figure()

        fig.add_trace(go.Heatmap(
            x=time,
            y=freq,
            z=power_db,
            colorscale=self.config.colorscale,
            colorbar=dict(title="功率 (dB)"),
            hovertemplate='时间: %{x:.3f}s<br>频率: %{y:.1f}Hz<br>功率: %{z:.1f}dB<extra></extra>'
        ))

        # 更新布局
        plot_title = title or f"CWT小波谱 ({wavelet})"
        fig.update_layout(
            title=dict(
                text=plot_title,
                x=0.5,
                xanchor='center'
            ),
            xaxis_title="时间 (秒)",
            yaxis_title="频率 (Hz)",
            width=self.config.width,
            height=self.config.height,
            template=self.config.template,
            yaxis=dict(type='log')  # 频率轴使用对数刻度
        )

        logger.debug(f"CWT小波谱绘制完成: {wavelet}小波")

        return fig

    def plot_combined_timefreq(
        self,
        signal_data: np.ndarray,
        sampling_rate: int,
        time_range: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        绘制时域+频域+时频联合视图

        Args:
            signal_data: 时域信号
            sampling_rate: 采样率
            time_range: 显示时间范围
            title: 图表标题

        Returns:
            Plotly图表对象
        """
        from processing.frequency_analyzer import FrequencyAnalyzer
        from processing.timefreq_analyzer import TimeFreqAnalyzer

        # 应用时间范围
        if time_range:
            start_idx = int(time_range[0] * sampling_rate)
            end_idx = int(time_range[1] * sampling_rate)
            signal_data = signal_data[start_idx:end_idx]

        # 创建3×1子图
        fig = make_subplots(
            rows=3,
            cols=1,
            row_heights=[0.25, 0.25, 0.5],
            subplot_titles=("时域波形", "频谱", "时频谱图"),
            vertical_spacing=0.08
        )

        # 1. 时域波形
        time = np.arange(len(signal_data)) / sampling_rate
        fig.add_trace(
            go.Scatter(
                x=time,
                y=signal_data,
                mode='lines',
                name='时域',
                line=dict(color='#1f77b4', width=1)
            ),
            row=1, col=1
        )

        # 2. 频谱
        freq_analyzer = FrequencyAnalyzer()
        freq, amplitude = freq_analyzer.compute_fft(signal_data, sampling_rate)
        freq_mask = freq <= 3000  # 只显示0-3000Hz
        fig.add_trace(
            go.Scatter(
                x=freq[freq_mask],
                y=amplitude[freq_mask],
                mode='lines',
                name='频谱',
                line=dict(color='#2ca02c', width=1)
            ),
            row=2, col=1
        )

        # 3. 时频谱图
        timefreq_analyzer = TimeFreqAnalyzer(nperseg=256)
        stft_result = timefreq_analyzer.compute_stft(signal_data, sampling_rate)
        power_db = 10 * np.log10(stft_result.power + 1e-10)

        # 限制频率范围
        freq_mask = stft_result.frequency <= 3000
        fig.add_trace(
            go.Heatmap(
                x=stft_result.time,
                y=stft_result.frequency[freq_mask],
                z=power_db[freq_mask, :],
                colorscale=self.config.colorscale,
                colorbar=dict(title="dB", len=0.4, y=0.25),
                showscale=True
            ),
            row=3, col=1
        )

        # 更新布局
        plot_title = title or "时域-频域-时频联合分析"
        fig.update_layout(
            title=dict(
                text=plot_title,
                x=0.5,
                xanchor='center'
            ),
            width=self.config.width,
            height=self.config.height + 200,
            template=self.config.template,
            showlegend=False
        )

        # 更新坐标轴
        fig.update_xaxes(title_text="时间 (秒)", row=1, col=1)
        fig.update_xaxes(title_text="频率 (Hz)", row=2, col=1)
        fig.update_xaxes(title_text="时间 (秒)", row=3, col=1)

        fig.update_yaxes(title_text="幅值", row=1, col=1)
        fig.update_yaxes(title_text="幅值", row=2, col=1)
        fig.update_yaxes(title_text="频率 (Hz)", row=3, col=1)

        logger.debug("联合视图绘制完成")

        return fig


# ====================================
# 便捷函数
# ====================================

def plot_spectrogram(
    signal_data: np.ndarray,
    sampling_rate: int,
    nperseg: int = 512,
    title: Optional[str] = None,
    config: Optional[TimeFreqPlotConfig] = None
) -> go.Figure:
    """
    便捷函数：绘制声谱图

    Args:
        signal_data: 时域信号
        sampling_rate: 采样率
        nperseg: STFT窗长度
        title: 图表标题
        config: 图表配置

    Returns:
        Plotly图表对象
    """
    plotter = TimeFrequencyPlotter(config)
    return plotter.plot_spectrogram(signal_data, sampling_rate, nperseg, title)


def plot_wavelet_scalogram(
    signal_data: np.ndarray,
    sampling_rate: int,
    wavelet: str = 'morl',
    title: Optional[str] = None,
    config: Optional[TimeFreqPlotConfig] = None
) -> go.Figure:
    """
    便捷函数：绘制小波谱

    Args:
        signal_data: 时域信号
        sampling_rate: 采样率
        wavelet: 小波类型
        title: 图表标题
        config: 图表配置

    Returns:
        Plotly图表对象
    """
    plotter = TimeFrequencyPlotter(config)
    return plotter.plot_wavelet_scalogram(signal_data, sampling_rate, wavelet, title=title)


# ====================================
# 测试代码
# ====================================

if __name__ == "__main__":
    print("=" * 60)
    print("时频图表测试")
    print("=" * 60)

    # 加载测试数据
    print("\n【准备】加载测试数据...")
    from data.loader import DataLoader

    loader = DataLoader(validate=False)
    signal_data = loader.load('light_wear', 'normal', 10, 'A', 'X')

    if signal_data is None:
        print("❌ 无法加载测试数据")
        exit(1)

    # 使用前5秒数据（时频分析计算量较大）
    test_duration = 5
    test_samples = test_duration * signal_data.sampling_rate
    test_signal = signal_data.time_series[:test_samples]

    print(f"   数据长度: {len(test_signal)} 采样点")
    print(f"   采样率: {signal_data.sampling_rate} Hz")

    # 创建绘图器
    plotter = TimeFrequencyPlotter()

    # 测试1: STFT声谱图
    print("\n【测试1】STFT声谱图:")
    fig1 = plotter.plot_spectrogram(
        test_signal,
        signal_data.sampling_rate,
        nperseg=512,
        freq_range=(0, 3000)
    )
    print(f"   ✅ 创建成功: {len(fig1.data)}个轨迹")

    # 测试2: CWT小波谱
    print("\n【测试2】CWT小波谱:")
    fig2 = plotter.plot_wavelet_scalogram(
        test_signal,
        signal_data.sampling_rate,
        wavelet='morl',
        freq_range=(10, 3000)
    )
    print(f"   ✅ 创建成功: {len(fig2.data)}个轨迹")

    # 测试3: 联合视图
    print("\n【测试3】时域-频域-时频联合视图:")
    fig3 = plotter.plot_combined_timefreq(
        test_signal,
        signal_data.sampling_rate
    )
    print(f"   ✅ 创建成功: {len(fig3.data)}个轨迹")

    # 测试4: 便捷函数
    print("\n【测试4】便捷函数测试:")
    fig4 = plot_spectrogram(
        test_signal,
        signal_data.sampling_rate,
        nperseg=256
    )
    print(f"   ✅ plot_spectrogram: {len(fig4.data)}个轨迹")

    fig5 = plot_wavelet_scalogram(
        test_signal,
        signal_data.sampling_rate,
        wavelet='morl'
    )
    print(f"   ✅ plot_wavelet_scalogram: {len(fig5.data)}个轨迹")

    print("\n" + "=" * 60)
    print("测试完成! 所有图表已创建（未显示）")
    print("提示: 在实际应用中使用 fig.show() 显示图表")
    print("=" * 60)
