"""
图表工具模块

提供通用的图表辅助功能：
- 图表样式设置
- 导出功能
- 布局管理
- 颜色方案

使用示例:
    >>> from visualization.plot_utils import save_figure, apply_theme
    >>>
    >>> fig = create_some_plot()
    >>> fig = apply_theme(fig, theme='dark')
    >>> save_figure(fig, 'output.png', width=1920, height=1080)
"""

import plotly.graph_objects as go
from typing import Optional, List, Tuple
from pathlib import Path

from config.settings import logger, PROJECT_ROOT


# 预定义主题
THEMES = {
    'light': 'plotly_white',
    'dark': 'plotly_dark',
    'minimal': 'simple_white',
    'scientific': 'plotly'
}

# 颜色方案
COLOR_SCHEMES = {
    'default': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
    'vibrant': ['#E63946', '#F1FAEE', '#A8DADC', '#457B9D', '#1D3557'],
    'earth': ['#8D5B4C', '#C9ADA7', '#9A8C98', '#4A4E69', '#22223B'],
    'ocean': ['#03045E', '#023E8A', '#0077B6', '#0096C7', '#00B4D8', '#48CAE4'],
    'sunset': ['#FFBA08', '#FAA307', '#F48C06', '#E85D04', '#DC2F02', '#D00000']
}


def apply_theme(
    fig: go.Figure,
    theme: str = 'light',
    font_family: str = 'Arial',
    font_size: int = 12
) -> go.Figure:
    """
    应用主题到图表

    Args:
        fig: Plotly图表对象
        theme: 主题名称
        font_family: 字体
        font_size: 字体大小

    Returns:
        更新后的图表对象
    """
    template = THEMES.get(theme, 'plotly_white')

    fig.update_layout(
        template=template,
        font=dict(
            family=font_family,
            size=font_size
        )
    )

    logger.debug(f"应用主题: {theme}")

    return fig


def save_figure(
    fig: go.Figure,
    filename: str,
    output_dir: Optional[Path] = None,
    format: str = 'png',
    width: int = 1200,
    height: int = 600,
    scale: float = 2.0
) -> Path:
    """
    保存图表到文件

    Args:
        fig: Plotly图表对象
        filename: 文件名
        output_dir: 输出目录（None则使用项目根目录/output）
        format: 输出格式 ('png', 'jpg', 'svg', 'pdf', 'html')
        width: 图片宽度（像素）
        height: 图片高度（像素）
        scale: 缩放比例（用于提高分辨率）

    Returns:
        保存的文件路径
    """
    # 确定输出目录
    if output_dir is None:
        output_dir = PROJECT_ROOT / "output"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 确保文件扩展名正确
    if not filename.endswith(f'.{format}'):
        filename = f"{filename}.{format}"

    filepath = output_dir / filename

    # 保存
    if format == 'html':
        fig.write_html(str(filepath))
    else:
        fig.write_image(
            str(filepath),
            format=format,
            width=width,
            height=height,
            scale=scale
        )

    logger.info(f"图表已保存: {filepath}")

    return filepath


def set_axis_range(
    fig: go.Figure,
    xrange: Optional[Tuple[float, float]] = None,
    yrange: Optional[Tuple[float, float]] = None,
    row: Optional[int] = None,
    col: Optional[int] = None
) -> go.Figure:
    """
    设置坐标轴范围

    Args:
        fig: Plotly图表对象
        xrange: X轴范围
        yrange: Y轴范围
        row: 子图行号
        col: 子图列号

    Returns:
        更新后的图表对象
    """
    update_kwargs = {}

    if xrange:
        update_kwargs['xaxis_range'] = list(xrange)
    if yrange:
        update_kwargs['yaxis_range'] = list(yrange)

    if row and col:
        # 更新特定子图
        if xrange:
            fig.update_xaxes(range=list(xrange), row=row, col=col)
        if yrange:
            fig.update_yaxes(range=list(yrange), row=row, col=col)
    else:
        # 更新全局
        if update_kwargs:
            fig.update_layout(**update_kwargs)

    return fig


def add_watermark(
    fig: go.Figure,
    text: str = "GEAR WEAR ANALYSIS",
    opacity: float = 0.1,
    font_size: int = 50,
    color: str = "gray"
) -> go.Figure:
    """
    添加水印

    Args:
        fig: Plotly图表对象
        text: 水印文本
        opacity: 不透明度
        font_size: 字体大小
        color: 颜色

    Returns:
        更新后的图表对象
    """
    fig.add_annotation(
        text=text,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(
            size=font_size,
            color=color
        ),
        opacity=opacity,
        textangle=-30
    )

    return fig


def get_color_scheme(scheme_name: str = 'default') -> List[str]:
    """
    获取颜色方案

    Args:
        scheme_name: 方案名称

    Returns:
        颜色列表
    """
    return COLOR_SCHEMES.get(scheme_name, COLOR_SCHEMES['default'])


def create_subplot_grid(
    rows: int,
    cols: int,
    subplot_titles: Optional[List[str]] = None,
    shared_xaxes: bool = False,
    shared_yaxes: bool = False,
    vertical_spacing: float = 0.1,
    horizontal_spacing: float = 0.1
) -> go.Figure:
    """
    创建子图网格

    Args:
        rows: 行数
        cols: 列数
        subplot_titles: 子图标题列表
        shared_xaxes: 是否共享X轴
        shared_yaxes: 是否共享Y轴
        vertical_spacing: 垂直间距
        horizontal_spacing: 水平间距

    Returns:
        子图对象
    """
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        shared_xaxes=shared_xaxes,
        shared_yaxes=shared_yaxes,
        vertical_spacing=vertical_spacing,
        horizontal_spacing=horizontal_spacing
    )

    logger.debug(f"创建子图网格: {rows}x{cols}")

    return fig


# ====================================
# 测试代码
# ====================================

if __name__ == "__main__":
    print("=" * 60)
    print("图表工具测试")
    print("=" * 60)

    # 创建测试图表
    print("\n【准备】创建测试图表...")
    import numpy as np

    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='sin(x)'))
    fig.update_layout(title="测试图表", xaxis_title="X", yaxis_title="Y")

    print("   ✅ 测试图表创建成功")

    # 测试1: 应用主题
    print("\n【测试1】应用主题:")
    for theme_name in ['light', 'dark', 'minimal']:
        fig_themed = apply_theme(fig, theme=theme_name)
        print(f"   ✅ {theme_name}主题已应用")

    # 测试2: 设置坐标轴范围
    print("\n【测试2】设置坐标轴范围:")
    fig = set_axis_range(fig, xrange=(0, 5), yrange=(-1.5, 1.5))
    print("   ✅ 坐标轴范围已设置")

    # 测试3: 添加水印
    print("\n【测试3】添加水印:")
    fig = add_watermark(fig, text="TEST", opacity=0.1)
    print("   ✅ 水印已添加")

    # 测试4: 获取颜色方案
    print("\n【测试4】获取颜色方案:")
    for scheme_name in ['default', 'vibrant', 'earth', 'ocean', 'sunset']:
        colors = get_color_scheme(scheme_name)
        print(f"   ✅ {scheme_name}: {len(colors)}种颜色")

    # 测试5: 创建子图网格
    print("\n【测试5】创建子图网格:")
    subplot_fig = create_subplot_grid(
        rows=2,
        cols=2,
        subplot_titles=['子图1', '子图2', '子图3', '子图4']
    )
    print("   ✅ 2x2子图网格已创建")

    # 测试6: 保存图表 (HTML格式，不需要kaleido)
    print("\n【测试6】保存图表:")
    try:
        output_path = save_figure(
            fig,
            "test_plot",
            format='html',
            width=800,
            height=600
        )
        print(f"   ✅ 图表已保存: {output_path}")
    except Exception as e:
        print(f"   ⚠️  保存失败（可能缺少依赖）: {e}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
