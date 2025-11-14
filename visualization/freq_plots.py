"""
频域图表可视化

实现各种频域分析的可视化：
- FFT频谱图
- 功率谱密度（PSD）
- 频段能量分布
- 齿轮特征频率标注
- 谐波分析

使用示例:
    >>> from data.loader import DataLoader
    >>> from visualization.freq_plots import plot_spectrum
    >>>
    >>> loader = DataLoader()
    >>> signal_data = loader.load('light_wear', 'normal', 10, 'A', 'X')
    >>>
    >>> fig = plot_spectrum(signal_data.time_series, signal_data.sampling_rate)
    >>> fig.show()
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass

from config.settings import logger


@dataclass
class FreqPlotConfig:
    """频域图表配置"""
    title: str = "频谱图"
    width: int = 1200
    height: int = 600
    template: str = "plotly_white"
    show_grid: bool = True

    # 颜色配置
    spectrum_color: str = "#1f77b4"
    psd_color: str = "#2ca02c"
    peak_color: str = "#d62728"
    gmf_color: str = "#ff7f0e"
    harmonic_color: str = "#9467bd"


class FrequencyPlotter:
    """
    频域图表绘制器

    提供各种频域可视化功能。
    """

    def __init__(self, config: Optional[FreqPlotConfig] = None):
        """
        初始化绘图器

        Args:
            config: 图表配置
        """
        self.config = config or FreqPlotConfig()
        logger.debug("FrequencyPlotter 初始化完成")

    def plot_spectrum(
        self,
        signal_data: np.ndarray,
        sampling_rate: int,
        freq_range: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
        log_scale: bool = False,
        mark_peaks: bool = True
    ) -> go.Figure:
        """
        绘制FFT频谱

        Args:
            signal_data: 时域信号
            sampling_rate: 采样率
            freq_range: 显示频率范围 (Hz)
            title: 图表标题
            log_scale: 是否使用对数刻度
            mark_peaks: 是否标注峰值

        Returns:
            Plotly图表对象
        """
        from processing.frequency_analyzer import FrequencyAnalyzer

        # 计算频谱
        analyzer = FrequencyAnalyzer()
        freq, amplitude = analyzer.compute_fft(signal_data, sampling_rate, normalize=True)

        # 应用频率范围
        if freq_range:
            mask = (freq >= freq_range[0]) & (freq <= freq_range[1])
            freq = freq[mask]
            amplitude = amplitude[mask]

        # 创建图表
        fig = go.Figure()

        # 添加频谱
        fig.add_trace(go.Scatter(
            x=freq,
            y=amplitude,
            mode='lines',
            name='频谱',
            line=dict(color=self.config.spectrum_color, width=1),
            hovertemplate='频率: %{x:.2f} Hz<br>幅值: %{y:.6f}<extra></extra>'
        ))

        # 标注峰值
        if mark_peaks and len(freq) > 100:
            from scipy.signal import find_peaks

            # 找峰值
            prominence = np.max(amplitude) * 0.1
            peaks, properties = find_peaks(amplitude, prominence=prominence, distance=20)

            if len(peaks) > 0:
                # 只显示前10个最大峰值
                if len(peaks) > 10:
                    sorted_idx = np.argsort(amplitude[peaks])[::-1][:10]
                    peaks = peaks[sorted_idx]

                peak_freqs = freq[peaks]
                peak_amps = amplitude[peaks]

                fig.add_trace(go.Scatter(
                    x=peak_freqs,
                    y=peak_amps,
                    mode='markers+text',
                    name='峰值',
                    marker=dict(
                        color=self.config.peak_color,
                        size=10,
                        symbol='diamond'
                    ),
                    text=[f'{f:.1f}Hz' for f in peak_freqs],
                    textposition='top center',
                    textfont=dict(size=9),
                    hovertemplate='峰值频率: %{x:.2f} Hz<br>幅值: %{y:.6f}<extra></extra>'
                ))

        # 更新布局
        plot_title = title or self.config.title
        fig.update_layout(
            title=dict(
                text=plot_title,
                x=0.5,
                xanchor='center'
            ),
            xaxis_title="频率 (Hz)",
            yaxis_title="幅值",
            width=self.config.width,
            height=self.config.height,
            template=self.config.template,
            hovermode='x unified',
            showlegend=True
        )

        if log_scale:
            fig.update_yaxes(type='log')

        if self.config.show_grid:
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

        logger.debug(f"频谱图绘制完成: {len(freq)}个频率点")

        return fig

    def plot_psd(
        self,
        signal_data: np.ndarray,
        sampling_rate: int,
        freq_range: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
        method: str = 'welch'
    ) -> go.Figure:
        """
        绘制功率谱密度（PSD）

        Args:
            signal_data: 时域信号
            sampling_rate: 采样率
            freq_range: 显示频率范围
            title: 图表标题
            method: 计算方法 ('welch' 或 'periodogram')

        Returns:
            Plotly图表对象
        """
        from processing.frequency_analyzer import FrequencyAnalyzer

        # 计算PSD
        analyzer = FrequencyAnalyzer()
        freq, psd = analyzer.compute_psd(signal_data, sampling_rate, method=method)

        # 应用频率范围
        if freq_range:
            mask = (freq >= freq_range[0]) & (freq <= freq_range[1])
            freq = freq[mask]
            psd = psd[mask]

        # 创建图表
        fig = go.Figure()

        # 添加PSD曲线
        fig.add_trace(go.Scatter(
            x=freq,
            y=psd,
            mode='lines',
            name=f'PSD ({method})',
            line=dict(color=self.config.psd_color, width=1.5),
            fill='tozeroy',
            fillcolor='rgba(44, 160, 44, 0.2)',
            hovertemplate='频率: %{x:.2f} Hz<br>PSD: %{y:.6e}<extra></extra>'
        ))

        # 更新布局
        plot_title = title or f"功率谱密度 - {method.capitalize()}"
        fig.update_layout(
            title=dict(
                text=plot_title,
                x=0.5,
                xanchor='center'
            ),
            xaxis_title="频率 (Hz)",
            yaxis_title="功率谱密度 (V²/Hz)",
            width=self.config.width,
            height=self.config.height,
            template=self.config.template,
            hovermode='x unified'
        )

        fig.update_yaxes(type='log')

        if self.config.show_grid:
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

        logger.debug(f"PSD图绘制完成: {len(freq)}个频率点")

        return fig

    def plot_gear_spectrum(
        self,
        signal_data: np.ndarray,
        sampling_rate: int,
        gear_params: 'GearParameters',
        freq_range: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
        num_harmonics: int = 5
    ) -> go.Figure:
        """
        绘制带齿轮特征频率标注的频谱

        Args:
            signal_data: 时域信号
            sampling_rate: 采样率
            gear_params: 齿轮参数
            freq_range: 显示频率范围
            title: 图表标题
            num_harmonics: 显示谐波数量

        Returns:
            Plotly图表对象
        """
        from processing.frequency_analyzer import FrequencyAnalyzer

        # 计算频谱
        analyzer = FrequencyAnalyzer()
        freq, amplitude = analyzer.compute_fft(signal_data, sampling_rate, normalize=True)

        # 应用频率范围
        if freq_range:
            mask = (freq >= freq_range[0]) & (freq <= freq_range[1])
            freq = freq[mask]
            amplitude = amplitude[mask]

        # 创建图表
        fig = go.Figure()

        # 添加频谱
        fig.add_trace(go.Scatter(
            x=freq,
            y=amplitude,
            mode='lines',
            name='频谱',
            line=dict(color=self.config.spectrum_color, width=1),
            hovertemplate='频率: %{x:.2f} Hz<br>幅值: %{y:.6f}<extra></extra>'
        ))

        # 标注GMF和谐波
        gmf = gear_params.mesh_freq
        max_amp = np.max(amplitude)

        for n in range(1, num_harmonics + 1):
            harmonic_freq = gmf * n

            # 只标注在显示范围内的谐波
            if freq_range and (harmonic_freq < freq_range[0] or harmonic_freq > freq_range[1]):
                continue

            # 添加垂直线
            color = self.config.gmf_color if n == 1 else self.config.harmonic_color
            line_style = 'solid' if n == 1 else 'dash'

            fig.add_vline(
                x=harmonic_freq,
                line=dict(color=color, width=2, dash=line_style),
                annotation=dict(
                    text=f'GMF×{n}<br>{harmonic_freq:.1f}Hz',
                    font=dict(size=10, color=color)
                )
            )

        # 标注轴频
        shaft_freq = gear_params.drive_freq
        if not freq_range or (freq_range[0] <= shaft_freq <= freq_range[1]):
            fig.add_vline(
                x=shaft_freq,
                line=dict(color='gray', width=1, dash='dot'),
                annotation=dict(
                    text=f'轴频<br>{shaft_freq:.1f}Hz',
                    font=dict(size=9, color='gray')
                )
            )

        # 更新布局
        plot_title = title or "齿轮频谱 - 特征频率标注"
        fig.update_layout(
            title=dict(
                text=plot_title,
                x=0.5,
                xanchor='center'
            ),
            xaxis_title="频率 (Hz)",
            yaxis_title="幅值",
            width=self.config.width,
            height=self.config.height,
            template=self.config.template,
            hovermode='x unified'
        )

        if self.config.show_grid:
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

        logger.debug(f"齿轮频谱图绘制完成: GMF={gmf:.2f}Hz")

        return fig

    def plot_band_energy(
        self,
        signal_data: np.ndarray,
        sampling_rate: int,
        freq_bands: Optional[Dict[str, Tuple[float, float]]] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        绘制频段能量分布

        Args:
            signal_data: 时域信号
            sampling_rate: 采样率
            freq_bands: 频段定义
            title: 图表标题

        Returns:
            Plotly图表对象
        """
        from processing.frequency_analyzer import FrequencyAnalyzer

        # 计算频谱和PSD
        analyzer = FrequencyAnalyzer()
        psd_freq, psd = analyzer.compute_psd(signal_data, sampling_rate)

        # 默认频段
        if freq_bands is None:
            freq_bands = {
                '超低频 (0-10 Hz)': (0, 10),
                '低频 (10-100 Hz)': (10, 100),
                '中频 (100-1000 Hz)': (100, 1000),
                '高频 (1000-3000 Hz)': (1000, 3000),
                '超高频 (3000-7500 Hz)': (3000, 7500)
            }

        # 计算各频段能量
        band_names = []
        band_energies = []

        for band_name, (f_min, f_max) in freq_bands.items():
            mask = (psd_freq >= f_min) & (psd_freq < f_max)

            if np.any(mask):
                freq_step = psd_freq[1] - psd_freq[0] if len(psd_freq) > 1 else 1
                energy = np.trapz(psd[mask], dx=freq_step)
            else:
                energy = 0.0

            band_names.append(band_name)
            band_energies.append(energy)

        # 创建柱状图
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=band_names,
            y=band_energies,
            marker=dict(
                color=band_energies,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="能量")
            ),
            text=[f'{e:.2e}' for e in band_energies],
            textposition='outside',
            hovertemplate='频段: %{x}<br>能量: %{y:.6e}<extra></extra>'
        ))

        # 更新布局
        plot_title = title or "频段能量分布"
        fig.update_layout(
            title=dict(
                text=plot_title,
                x=0.5,
                xanchor='center'
            ),
            xaxis_title="频段",
            yaxis_title="能量",
            width=self.config.width,
            height=self.config.height,
            template=self.config.template
        )

        if self.config.show_grid:
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

        logger.debug(f"频段能量图绘制完成: {len(freq_bands)}个频段")

        return fig

    def plot_spectrum_comparison(
        self,
        signals: Dict[str, np.ndarray],
        sampling_rate: int,
        freq_range: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
        normalize: bool = True
    ) -> go.Figure:
        """
        绘制多信号频谱对比

        Args:
            signals: 信号字典 {标签: 信号数据}
            sampling_rate: 采样率
            freq_range: 显示频率范围
            title: 图表标题
            normalize: 是否归一化

        Returns:
            Plotly图表对象
        """
        from processing.frequency_analyzer import FrequencyAnalyzer

        # 创建图表
        fig = go.Figure()

        # 颜色列表
        colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]

        analyzer = FrequencyAnalyzer()

        for idx, (label, signal_data) in enumerate(signals.items()):
            # 计算频谱
            freq, amplitude = analyzer.compute_fft(signal_data, sampling_rate, normalize=True)

            # 应用频率范围
            if freq_range:
                mask = (freq >= freq_range[0]) & (freq <= freq_range[1])
                freq = freq[mask]
                amplitude = amplitude[mask]

            # 归一化
            if normalize:
                amplitude = amplitude / (np.max(amplitude) + 1e-10)

            # 添加轨迹
            fig.add_trace(go.Scatter(
                x=freq,
                y=amplitude,
                mode='lines',
                name=label,
                line=dict(color=colors[idx % len(colors)], width=1.5),
                hovertemplate=f'{label}<br>频率: %{{x:.2f}} Hz<br>幅值: %{{y:.6f}}<extra></extra>'
            ))

        # 更新布局
        plot_title = title or "频谱对比"
        fig.update_layout(
            title=dict(
                text=plot_title,
                x=0.5,
                xanchor='center'
            ),
            xaxis_title="频率 (Hz)",
            yaxis_title="归一化幅值" if normalize else "幅值",
            width=self.config.width,
            height=self.config.height,
            template=self.config.template,
            hovermode='x unified',
            showlegend=True
        )

        if self.config.show_grid:
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

        logger.debug(f"频谱对比图绘制完成: {len(signals)}个信号")

        return fig


# ====================================
# 便捷函数
# ====================================

def plot_spectrum(
    signal_data: np.ndarray,
    sampling_rate: int,
    freq_range: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    config: Optional[FreqPlotConfig] = None
) -> go.Figure:
    """
    便捷函数：绘制频谱

    Args:
        signal_data: 时域信号
        sampling_rate: 采样率
        freq_range: 显示频率范围
        title: 图表标题
        config: 图表配置

    Returns:
        Plotly图表对象
    """
    plotter = FrequencyPlotter(config)
    return plotter.plot_spectrum(signal_data, sampling_rate, freq_range, title)


def plot_psd(
    signal_data: np.ndarray,
    sampling_rate: int,
    freq_range: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    config: Optional[FreqPlotConfig] = None
) -> go.Figure:
    """
    便捷函数：绘制功率谱密度

    Args:
        signal_data: 时域信号
        sampling_rate: 采样率
        freq_range: 显示频率范围
        title: 图表标题
        config: 图表配置

    Returns:
        Plotly图表对象
    """
    plotter = FrequencyPlotter(config)
    return plotter.plot_psd(signal_data, sampling_rate, freq_range, title)


# ====================================
# 测试代码
# ====================================

if __name__ == "__main__":
    print("=" * 60)
    print("频域图表测试")
    print("=" * 60)

    # 加载测试数据
    print("\n【准备】加载测试数据...")
    from data.loader import DataLoader

    loader = DataLoader(validate=False)
    signal_data = loader.load('light_wear', 'normal', 10, 'A', 'X')

    if signal_data is None:
        print("❌ 无法加载测试数据")
        exit(1)

    # 使用前10秒数据
    test_duration = 10
    test_samples = test_duration * signal_data.sampling_rate
    test_signal = signal_data.time_series[:test_samples]

    print(f"   数据长度: {len(test_signal)} 采样点")
    print(f"   采样率: {signal_data.sampling_rate} Hz")

    # 创建绘图器
    plotter = FrequencyPlotter()

    # 测试1: 基本频谱图
    print("\n【测试1】基本频谱图:")
    fig1 = plotter.plot_spectrum(
        test_signal,
        signal_data.sampling_rate,
        freq_range=(0, 3000),
        title="FFT频谱 (0-3000 Hz)"
    )
    print(f"   ✅ 创建成功: {len(fig1.data)}个轨迹")

    # 测试2: PSD图
    print("\n【测试2】功率谱密度:")
    fig2 = plotter.plot_psd(
        test_signal,
        signal_data.sampling_rate,
        freq_range=(10, 5000),
        method='welch'
    )
    print(f"   ✅ 创建成功: {len(fig2.data)}个轨迹")

    # 测试3: 齿轮频谱
    print("\n【测试3】齿轮特征频率标注:")
    from processing.gear_analyzer import create_default_gear_params

    gear_params = create_default_gear_params(shaft_speed=1000)
    fig3 = plotter.plot_gear_spectrum(
        test_signal,
        signal_data.sampling_rate,
        gear_params,
        freq_range=(0, 3000),
        num_harmonics=4
    )
    print(f"   ✅ 创建成功: GMF={gear_params.mesh_freq:.2f}Hz")

    # 测试4: 频段能量分布
    print("\n【测试4】频段能量分布:")
    fig4 = plotter.plot_band_energy(
        test_signal,
        signal_data.sampling_rate
    )
    print(f"   ✅ 创建成功: {len(fig4.data)}个轨迹")

    # 测试5: 频谱对比
    print("\n【测试5】频谱对比:")
    # 加载不同磨损状态的数据
    signal_light = loader.load('light_wear', 'normal', 10, 'A', 'X')
    signal_heavy = loader.load('heavy_wear', 'normal', 10, 'A', 'X')

    if signal_light and signal_heavy:
        signals = {
            '轻磨损': signal_light.time_series[:test_samples],
            '重磨损': signal_heavy.time_series[:test_samples]
        }
        fig5 = plotter.plot_spectrum_comparison(
            signals,
            signal_data.sampling_rate,
            freq_range=(0, 3000),
            title="磨损状态频谱对比"
        )
        print(f"   ✅ 创建成功: 对比{len(signals)}个信号")
    else:
        print("   ⚠️  部分数据加载失败")

    # 测试6: 便捷函数
    print("\n【测试6】便捷函数测试:")
    fig6 = plot_spectrum(
        test_signal,
        signal_data.sampling_rate,
        freq_range=(0, 2000),
        title="便捷函数测试"
    )
    print(f"   ✅ plot_spectrum: {len(fig6.data)}个轨迹")

    fig7 = plot_psd(
        test_signal,
        signal_data.sampling_rate,
        freq_range=(10, 5000)
    )
    print(f"   ✅ plot_psd: {len(fig7.data)}个轨迹")

    print("\n" + "=" * 60)
    print("测试完成! 所有图表已创建（未显示）")
    print("提示: 在实际应用中使用 fig.show() 显示图表")
    print("=" * 60)
