"""
侧边栏组件

提供数据选择和参数配置界面：
- 文件选择
- 传感器通道选择
- 时间范围选择
- 分析参数配置

使用示例:
    >>> import streamlit as st
    >>> from ui.sidebar import render_sidebar
    >>>
    >>> config = render_sidebar()
    >>> st.write(f"Selected: {config['drive_state']}")
"""

import streamlit as st
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

from config.settings import GEAR_STATES, TORQUES, SPEEDS, SENSORS, AXES, VALID_COMBINATIONS


@dataclass
class UIConfig:
    """UI配置数据类"""
    # 数据选择
    drive_state: str
    driven_state: str
    torque: int
    speed: int
    sensor: str
    axis: str

    # 时间范围
    time_start: float
    time_end: float

    # 分析参数
    show_envelope: bool = False
    freq_range_max: int = 3000
    nperseg: int = 512

    def to_dict(self) -> Dict:
        """转换为字典"""
        return asdict(self)


def render_sidebar() -> UIConfig:
    """
    渲染侧边栏

    Returns:
        UIConfig: 用户配置
    """
    st.sidebar.title("⚙️ 齿轮磨损分析系统")
    st.sidebar.markdown("---")

    # 数据选择
    st.sidebar.header("📁 数据选择")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        drive_state = st.selectbox(
            "主动轮状态",
            options=list(GEAR_STATES.keys()),
            format_func=lambda x: GEAR_STATES[x],
            index=1,  # 默认选择 'light_wear' (轻磨损)
            key="drive_state"
        )

    with col2:
        # 根据主动轮状态动态限制从动轮选项
        available_driven_states = VALID_COMBINATIONS.get(drive_state, list(GEAR_STATES.keys()))
        driven_state = st.selectbox(
            "从动轮状态",
            options=available_driven_states,
            format_func=lambda x: GEAR_STATES[x],
            index=0,  # 默认选择第一个可用选项
            key="driven_state"
        )

    col3, col4 = st.sidebar.columns(2)
    with col3:
        torque = st.selectbox(
            "扭矩 (Nm)",
            options=TORQUES,
            key="torque"
        )

    with col4:
        speed = st.selectbox(
            "转速 (rpm)",
            options=SPEEDS,
            key="speed"
        )

    st.sidebar.caption("✅ 从动轮选项已根据主动轮状态自动过滤")

    # 传感器选择
    st.sidebar.header("📡 传感器选择")

    col5, col6 = st.sidebar.columns(2)
    with col5:
        sensor = st.selectbox(
            "传感器",
            options=list(SENSORS.keys()),
            format_func=lambda x: SENSORS[x],
            key="sensor"
        )

    with col6:
        axis = st.selectbox(
            "方向",
            options=list(AXES.keys()),
            format_func=lambda x: AXES[x],
            key="axis"
        )

    # 时间范围
    st.sidebar.header("⏱️ 时间范围")

    time_range = st.sidebar.slider(
        "选择时间段 (秒)",
        min_value=0.0,
        max_value=30.0,
        value=(0.0, 5.0),
        step=0.1,
        key="time_range"
    )

    # 分析参数
    st.sidebar.header("🔧 分析参数")

    show_envelope = st.sidebar.checkbox(
        "显示包络",
        value=False,
        key="show_envelope"
    )

    freq_range_max = st.sidebar.slider(
        "频率范围上限 (Hz)",
        min_value=500,
        max_value=7500,
        value=3000,
        step=100,
        key="freq_range_max"
    )

    nperseg = st.sidebar.select_slider(
        "STFT窗长度",
        options=[128, 256, 512, 1024, 2048],
        value=512,
        key="nperseg"
    )

    # 构建配置对象
    config = UIConfig(
        drive_state=drive_state,
        driven_state=driven_state,
        torque=torque,
        speed=speed,
        sensor=sensor,
        axis=axis,
        time_start=time_range[0],
        time_end=time_range[1],
        show_envelope=show_envelope,
        freq_range_max=freq_range_max,
        nperseg=nperseg
    )

    # 显示当前配置摘要
    st.sidebar.markdown("---")
    st.sidebar.caption("**当前配置**")
    st.sidebar.caption(f"数据: {GEAR_STATES[drive_state]}-{GEAR_STATES[driven_state]}")
    st.sidebar.caption(f"传感器: {SENSORS[sensor]}_{AXES[axis]}")
    st.sidebar.caption(f"时间: {time_range[0]:.1f}s - {time_range[1]:.1f}s")

    return config


def render_comparison_sidebar() -> Dict:
    """
    渲染对比页面的侧边栏

    Returns:
        Dict: 对比配置
    """
    st.sidebar.title("⚙️ 对比分析")
    st.sidebar.markdown("---")

    st.sidebar.header("📊 对比模式")

    comparison_mode = st.sidebar.radio(
        "选择对比维度",
        options=["磨损状态对比", "传感器位置对比", "工况参数对比"],
        key="comparison_mode"
    )

    # 固定参数
    st.sidebar.header("🔧 固定参数")

    if comparison_mode == "磨损状态对比":
        # 选择传感器和扭矩，对比不同磨损状态
        sensor = st.sidebar.selectbox(
            "传感器",
            options=list(SENSORS.keys()),
            format_func=lambda x: SENSORS[x],
            key="comp_sensor"
        )

        axis = st.sidebar.selectbox(
            "方向",
            options=list(AXES.keys()),
            format_func=lambda x: AXES[x],
            key="comp_axis"
        )

        torque = st.sidebar.selectbox(
            "扭矩 (Nm)",
            options=TORQUES,
            key="comp_torque"
        )

        # 选择要对比的磨损状态
        st.sidebar.header("📁 选择状态")
        selected_states = st.sidebar.multiselect(
            "磨损状态",
            options=list(GEAR_STATES.keys()),
            default=["light_wear", "heavy_wear"],  # 仅选择有效组合（与normal从动轮配对）
            format_func=lambda x: GEAR_STATES[x],
            key="selected_states"
        )
        st.sidebar.caption("💡 对比模式固定从动轮为'正常'状态")

        return {
            "mode": comparison_mode,
            "sensor": sensor,
            "axis": axis,
            "torque": torque,
            "states": selected_states
        }

    elif comparison_mode == "传感器位置对比":
        # 选择磨损状态，对比不同传感器
        drive_state = st.sidebar.selectbox(
            "主动轮状态",
            options=list(GEAR_STATES.keys()),
            format_func=lambda x: GEAR_STATES[x],
            index=1,  # 默认选择 'light_wear' (轻磨损)
            key="comp_drive_state"
        )

        # 根据主动轮状态动态限制从动轮选项
        available_driven_states = VALID_COMBINATIONS.get(drive_state, list(GEAR_STATES.keys()))
        driven_state = st.sidebar.selectbox(
            "从动轮状态",
            options=available_driven_states,
            format_func=lambda x: GEAR_STATES[x],
            index=0,  # 默认选择第一个可用选项
            key="comp_driven_state"
        )

        torque = st.sidebar.selectbox(
            "扭矩 (Nm)",
            options=TORQUES,
            key="comp_torque2"
        )

        st.sidebar.caption("✅ 从动轮选项已根据主动轮状态自动过滤")

        # 选择要对比的传感器
        st.sidebar.header("📡 选择传感器")
        selected_sensors = st.sidebar.multiselect(
            "传感器通道",
            options=[f"{s}_{a}" for s in SENSORS.keys() for a in AXES.keys()],
            default=["A_X", "A_Y", "A_Z"],
            format_func=lambda x: f"{SENSORS[x.split('_')[0]]}_{AXES[x.split('_')[1]]}",
            key="selected_sensors"
        )

        return {
            "mode": comparison_mode,
            "drive_state": drive_state,
            "driven_state": driven_state,
            "torque": torque,
            "sensors": selected_sensors
        }

    else:  # 工况参数对比
        # 选择磨损状态和传感器，对比不同扭矩
        drive_state = st.sidebar.selectbox(
            "主动轮状态",
            options=list(GEAR_STATES.keys()),
            format_func=lambda x: GEAR_STATES[x],
            index=1,  # 默认选择 'light_wear' (轻磨损)
            key="comp_drive_state3"
        )

        # 根据主动轮状态动态限制从动轮选项
        available_driven_states = VALID_COMBINATIONS.get(drive_state, list(GEAR_STATES.keys()))
        driven_state = st.sidebar.selectbox(
            "从动轮状态",
            options=available_driven_states,
            format_func=lambda x: GEAR_STATES[x],
            index=0,  # 默认选择第一个可用选项
            key="comp_driven_state3"
        )

        sensor = st.sidebar.selectbox(
            "传感器",
            options=list(SENSORS.keys()),
            format_func=lambda x: SENSORS[x],
            key="comp_sensor3"
        )

        axis = st.sidebar.selectbox(
            "方向",
            options=list(AXES.keys()),
            format_func=lambda x: AXES[x],
            key="comp_axis3"
        )

        st.sidebar.caption("✅ 从动轮选项已根据主动轮状态自动过滤")

        # 选择要对比的扭矩
        st.sidebar.header("⚡ 选择扭矩")
        selected_torques = st.sidebar.multiselect(
            "扭矩 (Nm)",
            options=TORQUES,
            default=TORQUES,
            key="selected_torques"
        )

        return {
            "mode": comparison_mode,
            "drive_state": drive_state,
            "driven_state": driven_state,
            "sensor": sensor,
            "axis": axis,
            "torques": selected_torques
        }


# ====================================
# 测试代码
# ====================================

if __name__ == "__main__":
    print("=" * 60)
    print("侧边栏组件测试")
    print("=" * 60)

    print("\n✅ UIConfig数据类定义完成")
    print("✅ render_sidebar函数定义完成")
    print("✅ render_comparison_sidebar函数定义完成")

    # 测试数据类
    config = UIConfig(
        drive_state="normal",
        driven_state="normal",
        torque=10,
        speed=1000,
        sensor="A",
        axis="X",
        time_start=0.0,
        time_end=5.0
    )

    print(f"\n配置对象创建成功:")
    print(f"  主动轮: {config.drive_state}")
    print(f"  从动轮: {config.driven_state}")
    print(f"  传感器: {config.sensor}_{config.axis}")
    print(f"  时间范围: {config.time_start}-{config.time_end}s")

    print("\n" + "=" * 60)
    print("组件定义完成！需在Streamlit环境中运行")
    print("=" * 60)
