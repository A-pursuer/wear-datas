"""
时域波形图可视化

实现各种时域信号的可视化：
- 基本时域波形
- 多通道对比
- 包络线显示
- 分段波形
- 冲击标注

使用示例:
    >>> from data.loader import DataLoader
    >>> from visualization.time_plots import plot_waveform
    >>>
    >>> loader = DataLoader()
    >>> signal_data = loader.load('light_wear', 'normal', 10, 'A', 'X')
    >>>
    >>> fig = plot_waveform(signal_data.time_series, signal_data.sampling_rate)
    >>> fig.show()
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Tuple, Dict, Union
from dataclasses import dataclass

from config.settings import logger


@dataclass
class PlotConfig:
    """图表配置"""
    title: str = "时域波形"
    width: int = 1200
    height: int = 600
    template: str = "plotly_white"
    show_grid: bool = True

    # 颜色配置
    line_color: str = "#1f77b4"
    envelope_color: str = "#ff7f0e"
    impact_color: str = "#d62728"


class TimeDomainPlotter:
    """
    时域信号绘图器

    提供各种时域可视化功能。
    """

    def __init__(self, config: Optional[PlotConfig] = None):
        """
        初始化绘图器

        Args:
            config: 图表配置
        """
        self.config = config or PlotConfig()
        logger.debug("TimeDomainPlotter 初始化完成")

    def plot_waveform(
        self,
        signal_data: np.ndarray,
        sampling_rate: int,
        time_range: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
        show_envelope: bool = False
    ) -> go.Figure:
        """
        绘制基本时域波形

        Args:
            signal_data: 时域信号
            sampling_rate: 采样率
            time_range: 显示时间范围 (start, end) 秒
            title: 图表标题
            show_envelope: 是否显示包络

        Returns:
            Plotly图表对象
        """
        # 生成时间轴
        duration = len(signal_data) / sampling_rate
        time = np.arange(len(signal_data)) / sampling_rate

        # 应用时间范围
        if time_range:
            start_idx = int(time_range[0] * sampling_rate)
            end_idx = int(time_range[1] * sampling_rate)
            time = time[start_idx:end_idx]
            signal_data = signal_data[start_idx:end_idx]

        # 创建图表
        fig = go.Figure()

        # 添加波形
        fig.add_trace(go.Scatter(
            x=time,
            y=signal_data,
            mode='lines',
            name='信号',
            line=dict(color=self.config.line_color, width=1),
            hovertemplate='时间: %{x:.4f}s<br>幅值: %{y:.6f}<extra></extra>'
        ))

        # 添加包络
        if show_envelope:
            from scipy.signal import hilbert
            analytic_signal = hilbert(signal_data)
            envelope = np.abs(analytic_signal)

            fig.add_trace(go.Scatter(
                x=time,
                y=envelope,
                mode='lines',
                name='包络',
                line=dict(color=self.config.envelope_color, width=2, dash='dash'),
                hovertemplate='时间: %{x:.4f}s<br>包络: %{y:.6f}<extra></extra>'
            ))

            fig.add_trace(go.Scatter(
                x=time,
                y=-envelope,
                mode='lines',
                name='包络 (负)',
                line=dict(color=self.config.envelope_color, width=2, dash='dash'),
                showlegend=False,
                hovertemplate='时间: %{x:.4f}s<br>包络: %{y:.6f}<extra></extra>'
            ))

        # 更新布局
        plot_title = title or self.config.title
        fig.update_layout(
            title=dict(
                text=plot_title,
                x=0.5,
                xanchor='center'
            ),
            xaxis_title="时间 (秒)",
            yaxis_title="幅值",
            width=self.config.width,
            height=self.config.height,
            template=self.config.template,
            hovermode='x unified',
            showlegend=True
        )

        if self.config.show_grid:
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

        logger.debug(f"波形图绘制完成: {len(signal_data)}个采样点")

        return fig

    def plot_multi_channel(
        self,
        signals: Dict[str, np.ndarray],
        sampling_rate: int,
        time_range: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
        shared_yaxis: bool = False
    ) -> go.Figure:
        """
        绘制多通道对比图

        Args:
            signals: 信号字典 {通道名: 信号数据}
            sampling_rate: 采样率
            time_range: 显示时间范围
            title: 图表标题
            shared_yaxis: 是否共享Y轴

        Returns:
            Plotly图表对象
        """
        num_channels = len(signals)

        # 创建子图
        fig = make_subplots(
            rows=num_channels,
            cols=1,
            shared_xaxes=True,
            shared_yaxes=shared_yaxis,
            subplot_titles=list(signals.keys()),
            vertical_spacing=0.05
        )

        # 颜色列表
        colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]

        # 为每个通道添加波形
        for idx, (channel_name, signal_data) in enumerate(signals.items()):
            # 生成时间轴
            time = np.arange(len(signal_data)) / sampling_rate

            # 应用时间范围
            if time_range:
                start_idx = int(time_range[0] * sampling_rate)
                end_idx = int(time_range[1] * sampling_rate)
                time = time[start_idx:end_idx]
                signal_data = signal_data[start_idx:end_idx]

            # 添加轨迹
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=signal_data,
                    mode='lines',
                    name=channel_name,
                    line=dict(color=colors[idx % len(colors)], width=1),
                    hovertemplate='时间: %{x:.4f}s<br>幅值: %{y:.6f}<extra></extra>'
                ),
                row=idx + 1,
                col=1
            )

        # 更新布局
        plot_title = title or "多通道时域波形对比"
        fig.update_layout(
            title=dict(
                text=plot_title,
                x=0.5,
                xanchor='center'
            ),
            width=self.config.width,
            height=self.config.height * num_channels / 2,
            template=self.config.template,
            hovermode='x unified',
            showlegend=False
        )

        # 更新X轴和Y轴
        fig.update_xaxes(title_text="时间 (秒)", row=num_channels, col=1)
        for i in range(num_channels):
            fig.update_yaxes(title_text="幅值", row=i + 1, col=1)
            if self.config.show_grid:
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=i + 1, col=1)
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=i + 1, col=1)

        logger.debug(f"多通道波形图绘制完成: {num_channels}个通道")

        return fig

    def plot_with_impacts(
        self,
        signal_data: np.ndarray,
        sampling_rate: int,
        impact_indices: np.ndarray,
        time_range: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        绘制带冲击标注的波形

        Args:
            signal_data: 时域信号
            sampling_rate: 采样率
            impact_indices: 冲击位置索引
            time_range: 显示时间范围
            title: 图表标题

        Returns:
            Plotly图表对象
        """
        # 生成时间轴
        time = np.arange(len(signal_data)) / sampling_rate

        # 应用时间范围
        if time_range:
            start_idx = int(time_range[0] * sampling_rate)
            end_idx = int(time_range[1] * sampling_rate)
            time = time[start_idx:end_idx]
            signal_data = signal_data[start_idx:end_idx]
            # 调整冲击索引
            impact_indices = impact_indices[(impact_indices >= start_idx) & (impact_indices < end_idx)] - start_idx

        # 创建图表
        fig = go.Figure()

        # 添加波形
        fig.add_trace(go.Scatter(
            x=time,
            y=signal_data,
            mode='lines',
            name='信号',
            line=dict(color=self.config.line_color, width=1),
            hovertemplate='时间: %{x:.4f}s<br>幅值: %{y:.6f}<extra></extra>'
        ))

        # 添加包络
        from scipy.signal import hilbert
        analytic_signal = hilbert(signal_data)
        envelope = np.abs(analytic_signal)

        fig.add_trace(go.Scatter(
            x=time,
            y=envelope,
            mode='lines',
            name='包络',
            line=dict(color=self.config.envelope_color, width=2, dash='dash'),
            hovertemplate='时间: %{x:.4f}s<br>包络: %{y:.6f}<extra></extra>'
        ))

        # 标注冲击点
        if len(impact_indices) > 0:
            impact_times = time[impact_indices]
            impact_values = signal_data[impact_indices]

            fig.add_trace(go.Scatter(
                x=impact_times,
                y=impact_values,
                mode='markers',
                name=f'冲击点 (n={len(impact_indices)})',
                marker=dict(
                    color=self.config.impact_color,
                    size=8,
                    symbol='x',
                    line=dict(width=2)
                ),
                hovertemplate='冲击时间: %{x:.4f}s<br>幅值: %{y:.6f}<extra></extra>'
            ))

        # 更新布局
        plot_title = title or "时域波形 - 冲击检测"
        fig.update_layout(
            title=dict(
                text=plot_title,
                x=0.5,
                xanchor='center'
            ),
            xaxis_title="时间 (秒)",
            yaxis_title="幅值",
            width=self.config.width,
            height=self.config.height,
            template=self.config.template,
            hovermode='x unified',
            showlegend=True
        )

        if self.config.show_grid:
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

        logger.debug(f"冲击标注波形图绘制完成: {len(impact_indices)}个冲击点")

        return fig

    def plot_segments(
        self,
        signal_data: np.ndarray,
        sampling_rate: int,
        segment_duration: float = 1.0,
        num_segments: int = 4,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        绘制分段波形对比

        Args:
            signal_data: 时域信号
            sampling_rate: 采样率
            segment_duration: 每段时长（秒）
            num_segments: 显示段数
            title: 图表标题

        Returns:
            Plotly图表对象
        """
        segment_samples = int(segment_duration * sampling_rate)
        total_duration = len(signal_data) / sampling_rate

        # 均匀分布采样段
        step = max(1, int(len(signal_data) / num_segments))

        # 创建子图
        fig = make_subplots(
            rows=num_segments,
            cols=1,
            shared_xaxes=True,
            shared_yaxes=True,
            vertical_spacing=0.05
        )

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        for i in range(num_segments):
            start_idx = i * step
            end_idx = min(start_idx + segment_samples, len(signal_data))

            if start_idx >= len(signal_data):
                break

            segment = signal_data[start_idx:end_idx]
            time = np.arange(len(segment)) / sampling_rate
            start_time = start_idx / sampling_rate

            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=segment,
                    mode='lines',
                    name=f'段 {i+1} ({start_time:.1f}s)',
                    line=dict(color=colors[i % len(colors)], width=1),
                    hovertemplate='时间: %{x:.4f}s<br>幅值: %{y:.6f}<extra></extra>'
                ),
                row=i + 1,
                col=1
            )

            # 添加段标题
            fig.add_annotation(
                text=f"段 {i+1}: {start_time:.1f}s - {(start_idx + len(segment)) / sampling_rate:.1f}s",
                xref=f"x{i+1}", yref=f"y{i+1}",
                x=0.02, y=0.95,
                xanchor='left', yanchor='top',
                showarrow=False,
                font=dict(size=10)
            )

        # 更新布局
        plot_title = title or f"分段波形对比 ({num_segments}段)"
        fig.update_layout(
            title=dict(
                text=plot_title,
                x=0.5,
                xanchor='center'
            ),
            width=self.config.width,
            height=self.config.height * num_segments / 2,
            template=self.config.template,
            showlegend=False
        )

        # 更新X轴和Y轴
        fig.update_xaxes(title_text="段内时间 (秒)", row=num_segments, col=1)
        for i in range(num_segments):
            fig.update_yaxes(title_text="幅值", row=i + 1, col=1)
            if self.config.show_grid:
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=i + 1, col=1)
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=i + 1, col=1)

        logger.debug(f"分段波形图绘制完成: {num_segments}段")

        return fig


# ====================================
# 便捷函数
# ====================================

def plot_waveform(
    signal_data: np.ndarray,
    sampling_rate: int,
    time_range: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    show_envelope: bool = False,
    config: Optional[PlotConfig] = None
) -> go.Figure:
    """
    便捷函数：绘制时域波形

    Args:
        signal_data: 时域信号
        sampling_rate: 采样率
        time_range: 显示时间范围
        title: 图表标题
        show_envelope: 是否显示包络
        config: 图表配置

    Returns:
        Plotly图表对象
    """
    plotter = TimeDomainPlotter(config)
    return plotter.plot_waveform(signal_data, sampling_rate, time_range, title, show_envelope)


def plot_multi_channel(
    signals: Dict[str, np.ndarray],
    sampling_rate: int,
    time_range: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    config: Optional[PlotConfig] = None
) -> go.Figure:
    """
    便捷函数：绘制多通道对比

    Args:
        signals: 信号字典
        sampling_rate: 采样率
        time_range: 显示时间范围
        title: 图表标题
        config: 图表配置

    Returns:
        Plotly图表对象
    """
    plotter = TimeDomainPlotter(config)
    return plotter.plot_multi_channel(signals, sampling_rate, time_range, title)


# ====================================
# 测试代码
# ====================================

if __name__ == "__main__":
    print("=" * 60)
    print("时域波形图测试")
    print("=" * 60)

    # 加载测试数据
    print("\n【准备】加载测试数据...")
    from data.loader import DataLoader

    loader = DataLoader(validate=False)
    signal_data = loader.load('light_wear', 'normal', 10, 'A', 'X')

    if signal_data is None:
        print("❌ 无法加载测试数据")
        exit(1)

    # 使用前5秒数据
    test_duration = 5
    test_samples = test_duration * signal_data.sampling_rate
    test_signal = signal_data.time_series[:test_samples]

    print(f"   数据长度: {len(test_signal)} 采样点")
    print(f"   采样率: {signal_data.sampling_rate} Hz")

    # 创建绘图器
    plotter = TimeDomainPlotter()

    # 测试1: 基本波形
    print("\n【测试1】基本时域波形:")
    fig1 = plotter.plot_waveform(
        test_signal,
        signal_data.sampling_rate,
        time_range=(0, 1),
        title="时域波形 - 前1秒"
    )
    print(f"   ✅ 创建成功: {len(fig1.data)}个轨迹")

    # 测试2: 带包络的波形
    print("\n【测试2】带包络的波形:")
    fig2 = plotter.plot_waveform(
        test_signal,
        signal_data.sampling_rate,
        time_range=(0, 1),
        title="时域波形 - 包络显示",
        show_envelope=True
    )
    print(f"   ✅ 创建成功: {len(fig2.data)}个轨迹（含包络）")

    # 测试3: 多通道对比
    print("\n【测试3】多通道对比:")
    # 加载3个通道
    signal_ax = loader.load('light_wear', 'normal', 10, 'A', 'X')
    signal_ay = loader.load('light_wear', 'normal', 10, 'A', 'Y')
    signal_az = loader.load('light_wear', 'normal', 10, 'A', 'Z')

    if signal_ax and signal_ay and signal_az:
        signals = {
            'A_X (轴向)': signal_ax.time_series[:test_samples],
            'A_Y (径向)': signal_ay.time_series[:test_samples],
            'A_Z (径向)': signal_az.time_series[:test_samples]
        }
        fig3 = plotter.plot_multi_channel(
            signals,
            signal_data.sampling_rate,
            time_range=(0, 1),
            title="传感器A - 三轴对比"
        )
        print(f"   ✅ 创建成功: {len(fig3.data)}个轨迹")
    else:
        print("   ⚠️  部分通道数据加载失败")

    # 测试4: 冲击检测标注
    print("\n【测试4】冲击检测标注:")
    from processing.time_domain import ImpactDetector

    detector = ImpactDetector(threshold_multiplier=2.0)
    impact_result = detector.detect_impacts(test_signal, signal_data.sampling_rate)

    fig4 = plotter.plot_with_impacts(
        test_signal,
        signal_data.sampling_rate,
        np.array(impact_result['impact_positions']),
        time_range=(0, 1),
        title=f"冲击检测 - 检测到{impact_result['impact_count']}个冲击"
    )
    print(f"   ✅ 创建成功: 检测到{impact_result['impact_count']}个冲击")

    # 测试5: 分段波形
    print("\n【测试5】分段波形对比:")
    fig5 = plotter.plot_segments(
        test_signal,
        signal_data.sampling_rate,
        segment_duration=1.0,
        num_segments=4,
        title="分段波形 - 4段对比"
    )
    print(f"   ✅ 创建成功: {len(fig5.data)}个轨迹")

    # 测试6: 便捷函数
    print("\n【测试6】便捷函数测试:")
    fig6 = plot_waveform(
        test_signal,
        signal_data.sampling_rate,
        time_range=(0, 0.5),
        title="便捷函数测试",
        show_envelope=True
    )
    print(f"   ✅ plot_waveform: {len(fig6.data)}个轨迹")

    print("\n" + "=" * 60)
    print("测试完成! 所有图表已创建（未显示）")
    print("提示: 在实际应用中使用 fig.show() 显示图表")
    print("=" * 60)
