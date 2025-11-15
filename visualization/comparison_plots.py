"""
特征对比图可视化

实现多工况/多状态的特征对比可视化：
- 特征柱状图对比
- 雷达图
- 箱线图
- 散点图矩阵

使用示例:
    >>> from visualization.comparison_plots import plot_feature_comparison
    >>>
    >>> features = {
    ...     '正常': {'rms': 10.5, 'peak': 50.2, 'kurtosis': 3.1},
    ...     '轻磨损': {'rms': 15.3, 'peak': 75.8, 'kurtosis': 4.5},
    ...     '重磨损': {'rms': 22.1, 'peak': 110.3, 'kurtosis': 6.2}
    ... }
    >>> fig = plot_feature_comparison(features)
    >>> fig.show()
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
from dataclasses import dataclass

from config.settings import logger


@dataclass
class ComparisonPlotConfig:
    """对比图配置"""
    width: int = 1200
    height: int = 600
    template: str = "plotly_white"
    colors: List[str] = None

    def __post_init__(self):
        if self.colors is None:
            self.colors = [
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
            ]


class ComparisonPlotter:
    """
    特征对比绘图器

    提供各种对比可视化功能。
    """

    def __init__(self, config: Optional[ComparisonPlotConfig] = None):
        """
        初始化绘图器

        Args:
            config: 图表配置
        """
        self.config = config or ComparisonPlotConfig()
        logger.debug("ComparisonPlotter 初始化完成")

    def plot_feature_comparison(
        self,
        features_dict: Dict[str, Dict[str, float]],
        title: Optional[str] = None,
        normalize: bool = False
    ) -> go.Figure:
        """
        绘制特征柱状图对比

        Args:
            features_dict: 特征字典 {条件名: {特征名: 值}}
            title: 图表标题
            normalize: 是否归一化

        Returns:
            Plotly图表对象
        """
        # 提取特征名和条件名
        conditions = list(features_dict.keys())
        if not conditions:
            raise ValueError("特征字典为空")

        feature_names = list(features_dict[conditions[0]].keys())

        # 创建图表
        fig = go.Figure()

        # 为每个条件创建柱状图
        for idx, condition in enumerate(conditions):
            values = [features_dict[condition].get(fname, 0) for fname in feature_names]

            # 归一化
            if normalize:
                values = np.array(values)
                values = (values - np.min(values)) / (np.max(values) - np.min(values) + 1e-10)

            fig.add_trace(go.Bar(
                name=condition,
                x=feature_names,
                y=values,
                marker=dict(color=self.config.colors[idx % len(self.config.colors)]),
                hovertemplate='%{x}<br>%{y:.4f}<extra></extra>'
            ))

        # 更新布局
        plot_title = title or "特征对比"
        fig.update_layout(
            title=dict(
                text=plot_title,
                x=0.5,
                xanchor='center'
            ),
            xaxis_title="特征",
            yaxis_title="归一化值" if normalize else "值",
            barmode='group',
            width=self.config.width,
            height=self.config.height,
            template=self.config.template,
            showlegend=True
        )

        logger.debug(f"特征对比图绘制完成: {len(conditions)}个条件")

        return fig

    def plot_radar_chart(
        self,
        features_dict: Dict[str, Dict[str, float]],
        title: Optional[str] = None,
        normalize: bool = True
    ) -> go.Figure:
        """
        绘制雷达图

        Args:
            features_dict: 特征字典
            title: 图表标题
            normalize: 是否归一化

        Returns:
            Plotly图表对象
        """
        conditions = list(features_dict.keys())
        feature_names = list(features_dict[conditions[0]].keys())

        # 创建图表
        fig = go.Figure()

        for idx, condition in enumerate(conditions):
            values = [features_dict[condition].get(fname, 0) for fname in feature_names]

            # 归一化到0-1
            if normalize:
                values = np.array(values)
                max_val = np.max(np.abs(values))
                if max_val > 0:
                    values = values / max_val

            # 闭合雷达图
            values = list(values) + [values[0]]
            theta = feature_names + [feature_names[0]]

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=theta,
                fill='toself',
                name=condition,
                line=dict(color=self.config.colors[idx % len(self.config.colors)])
            ))

        # 更新布局
        plot_title = title or "特征雷达图"
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1] if normalize else None
                )
            ),
            title=dict(
                text=plot_title,
                x=0.5,
                xanchor='center'
            ),
            width=self.config.width,
            height=self.config.height,
            template=self.config.template,
            showlegend=True
        )

        logger.debug(f"雷达图绘制完成: {len(conditions)}个条件")

        return fig

    def plot_scatter_matrix(
        self,
        features_dict: Dict[str, Dict[str, float]],
        title: Optional[str] = None
    ) -> go.Figure:
        """
        绘制散点图矩阵

        Args:
            features_dict: 特征字典
            title: 图表标题

        Returns:
            Plotly图表对象
        """
        from plotly.figure_factory import create_scatterplotmatrix

        # 转换为DataFrame格式
        conditions = list(features_dict.keys())
        feature_names = list(features_dict[conditions[0]].keys())

        # 构建数据
        data_dict = {fname: [] for fname in feature_names}
        data_dict['condition'] = []

        for condition in conditions:
            for fname in feature_names:
                data_dict[fname].append(features_dict[condition].get(fname, 0))
            data_dict['condition'].append(condition)

        # 创建散点图矩阵
        fig = create_scatterplotmatrix(
            data_dict,
            dimensions=feature_names,
            index='condition',
            title=title or "特征散点图矩阵",
            height=self.config.height + 200,
            width=self.config.width
        )

        logger.debug(f"散点图矩阵绘制完成: {len(feature_names)}个特征")

        return fig


# ====================================
# 便捷函数
# ====================================

def plot_feature_comparison(
    features_dict: Dict[str, Dict[str, float]],
    title: Optional[str] = None,
    config: Optional[ComparisonPlotConfig] = None
) -> go.Figure:
    """
    便捷函数：绘制特征对比

    Args:
        features_dict: 特征字典
        title: 图表标题
        config: 图表配置

    Returns:
        Plotly图表对象
    """
    plotter = ComparisonPlotter(config)
    return plotter.plot_feature_comparison(features_dict, title)


def plot_radar_chart(
    features_dict: Dict[str, Dict[str, float]],
    title: Optional[str] = None,
    config: Optional[ComparisonPlotConfig] = None
) -> go.Figure:
    """
    便捷函数：绘制雷达图

    Args:
        features_dict: 特征字典
        title: 图表标题
        config: 图表配置

    Returns:
        Plotly图表对象
    """
    plotter = ComparisonPlotter(config)
    return plotter.plot_radar_chart(features_dict, title)


# ====================================
# 测试代码
# ====================================

if __name__ == "__main__":
    print("=" * 60)
    print("特征对比图测试")
    print("=" * 60)

    # 准备测试数据
    print("\n【准备】生成测试特征数据...")

    # 模拟特征数据
    features = {
        '正常': {
            'RMS': 10.5,
            '峰值': 50.2,
            '峰度': 3.1,
            '偏度': 0.2,
            '波峰因子': 4.8
        },
        '轻磨损': {
            'RMS': 15.3,
            '峰值': 75.8,
            '峰度': 4.5,
            '偏度': 0.5,
            '波峰因子': 5.0
        },
        '重磨损': {
            'RMS': 22.1,
            '峰值': 110.3,
            '峰度': 6.2,
            '偏度': 1.2,
            '波峰因子': 5.3
        }
    }

    print(f"   特征数量: {len(list(features.values())[0])}")
    print(f"   条件数量: {len(features)}")

    # 创建绘图器
    plotter = ComparisonPlotter()

    # 测试1: 特征柱状图对比
    print("\n【测试1】特征柱状图对比:")
    fig1 = plotter.plot_feature_comparison(
        features,
        title="磨损状态特征对比"
    )
    print(f"   ✅ 创建成功: {len(fig1.data)}个轨迹")

    # 测试2: 归一化对比
    print("\n【测试2】归一化特征对比:")
    fig2 = plotter.plot_feature_comparison(
        features,
        title="归一化特征对比",
        normalize=True
    )
    print(f"   ✅ 创建成功: {len(fig2.data)}个轨迹")

    # 测试3: 雷达图
    print("\n【测试3】特征雷达图:")
    fig3 = plotter.plot_radar_chart(
        features,
        title="磨损状态雷达图"
    )
    print(f"   ✅ 创建成功: {len(fig3.data)}个轨迹")

    # 测试4: 便捷函数
    print("\n【测试4】便捷函数测试:")
    fig4 = plot_feature_comparison(
        features,
        title="便捷函数测试"
    )
    print(f"   ✅ plot_feature_comparison: {len(fig4.data)}个轨迹")

    fig5 = plot_radar_chart(
        features,
        title="便捷函数雷达图"
    )
    print(f"   ✅ plot_radar_chart: {len(fig5.data)}个轨迹")

    print("\n" + "=" * 60)
    print("测试完成! 所有图表已创建（未显示）")
    print("提示: 在实际应用中使用 fig.show() 显示图表")
    print("=" * 60)
