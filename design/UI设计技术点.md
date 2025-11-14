# UI设计技术点详解

## 1. Streamlit界面架构设计

### 1.1 页面布局架构

```python
import streamlit as st
import pandas as pd
from typing import Dict, List, Any

class AppLayoutManager:
    """应用布局管理器 - 统一管理整体UI结构"""

    def __init__(self):
        self.sidebar_width = 300
        self.main_content_ratio = [3, 7]  # 侧边栏:主内容 = 3:7

    def setup_page_config(self):
        """配置页面基本设置"""
        st.set_page_config(
            page_title="齿轮磨损数据分析系统",
            page_icon="⚙️",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://docs.streamlit.io/',
                'Report a bug': None,
                'About': "# 齿轮磨损数据分析系统\n基于Python和Streamlit的研究工具"
            }
        )

    def create_main_layout(self):
        """创建主要布局结构"""
        # 顶部标题区域
        self._create_header()

        # 主体内容区域
        col1, col2 = st.columns([1, 4])

        with col1:
            # 参数控制面板
            self._create_control_panel()

        with col2:
            # 主要显示区域
            self._create_main_display_area()

        # 底部状态栏
        self._create_status_bar()

    def _create_header(self):
        """创建页面头部"""
        st.title("⚙️ 齿轮磨损数据分析系统")
        st.markdown("---")

        # 添加快速导航
        tabs = st.tabs(["时域分析", "频域分析", "对比分析", "统计特征", "导出结果"])
        return tabs

    def _create_control_panel(self):
        """创建控制面板容器"""
        with st.container():
            st.header("📊 参数设置")
            # 这里返回容器，供具体组件填充
            return st.container()

    def _create_main_display_area(self):
        """创建主显示区域容器"""
        return st.container()

    def _create_status_bar(self):
        """创建状态栏"""
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            if 'data_loaded' in st.session_state:
                st.success(f"✅ 已加载 {len(st.session_state.data_loaded)} 个数据集")

        with col2:
            if 'processing_time' in st.session_state:
                st.info(f"⏱️ 处理时间: {st.session_state.processing_time:.2f}s")

        with col3:
            st.info(f"📅 {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
```

### 1.2 状态管理系统

```python
class SessionStateManager:
    """会话状态管理器 - 管理应用状态和数据缓存"""

    def __init__(self):
        self.initialize_session_state()

    def initialize_session_state(self):
        """初始化会话状态"""
        default_state = {
            # 数据相关状态
            'selected_datasets': [],
            'loaded_data_cache': {},
            'analysis_results': {},

            # UI状态
            'current_tab': 'time_domain',
            'show_advanced_options': False,
            'chart_configs': {},

            # 参数状态
            'gear_configs': {
                'drive_gear_state': '正常',
                'driven_gear_state': '正常',
                'torque': 10,
                'sensor': 'A',
                'axis': 'X'
            },

            # 分析参数
            'time_range': [0, 30],
            'frequency_range': [0, 8000],
            'filter_settings': {
                'enable_filter': False,
                'filter_type': 'bandpass',
                'low_freq': 10,
                'high_freq': 1000
            },

            # 性能监控
            'processing_time': 0,
            'memory_usage': 0
        }

        for key, value in default_state.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def update_state(self, key: str, value: Any):
        """更新状态值"""
        st.session_state[key] = value

    def get_state(self, key: str, default=None):
        """获取状态值"""
        return st.session_state.get(key, default)

    def clear_cache(self, cache_type: str = 'all'):
        """清理缓存"""
        if cache_type == 'all':
            st.session_state.loaded_data_cache = {}
            st.session_state.analysis_results = {}
        elif cache_type == 'data':
            st.session_state.loaded_data_cache = {}
        elif cache_type == 'analysis':
            st.session_state.analysis_results = {}

        st.rerun()
```

## 2. 智能参数选择组件

### 2.1 自适应选择器

```python
class SmartParameterSelector:
    """智能参数选择器 - 根据数据可用性动态调整选项"""

    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.available_configs = self._get_available_configurations()

    def create_gear_state_selector(self):
        """创建齿轮状态选择器"""
        with st.expander("🔧 齿轮配置", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("主动轮状态")
                available_drive_states = self._get_available_drive_states()

                drive_state = st.selectbox(
                    "选择主动轮磨损状态",
                    options=available_drive_states,
                    index=0,
                    help="主动轮的磨损程度会影响啮合频率特征"
                )

                # 根据主动轮状态动态更新从动轮选项
                available_driven_states = self._get_available_driven_states(drive_state)

            with col2:
                st.subheader("从动轮状态")
                driven_state = st.selectbox(
                    "选择从动轮磨损状态",
                    options=available_driven_states,
                    index=0,
                    help="从动轮磨损主要影响输出端的振动特征"
                )

            # 显示配置有效性
            self._show_configuration_validity(drive_state, driven_state)

            return drive_state, driven_state

    def create_experiment_condition_selector(self, drive_state, driven_state):
        """创建实验工况选择器"""
        with st.expander("⚡ 实验工况", expanded=True):
            # 获取当前齿轮状态组合下的可用工况
            available_conditions = self._get_available_conditions(drive_state, driven_state)

            col1, col2 = st.columns(2)

            with col1:
                available_torques = list(set([c['torque'] for c in available_conditions]))
                torque = st.selectbox(
                    "扭矩 (Nm)",
                    options=available_torques,
                    format_func=lambda x: f"{x} Nm",
                    help="负载扭矩影响齿轮振动的幅值和频率特征"
                )

            with col2:
                # 转速固定为1000rpm，显示为信息
                st.metric(
                    label="转速",
                    value="1000 rpm",
                    help="所有实验数据均在1000rpm下采集"
                )

            return torque

    def create_sensor_configuration_selector(self):
        """创建传感器配置选择器"""
        with st.expander("📡 传感器配置", expanded=True):
            # 传感器位置选择
            sensor_info = {
                'A': '主动轴输入轴承处',
                'B': '从动轴输入处',
                'C': '从动轴输出处'
            }

            col1, col2 = st.columns(2)

            with col1:
                sensor = st.selectbox(
                    "传感器位置",
                    options=['A', 'B', 'C'],
                    format_func=lambda x: f"传感器{x} - {sensor_info[x]}",
                    help="不同位置的传感器反映不同的齿轮传动特征"
                )

                # 显示传感器示意图（可选）
                self._show_sensor_diagram(sensor)

            with col2:
                axis = st.selectbox(
                    "测量方向",
                    options=['X', 'Y', 'Z'],
                    format_func=lambda x: f"{x}方向 - {'轴向' if x == 'X' else '径向'}",
                    help="X方向为轴向，Y和Z方向为径向"
                )

                # 显示方向说明
                direction_info = {
                    'X': '轴向振动主要反映轴承和轴的状态',
                    'Y': '径向振动反映齿轮啮合和不平衡',
                    'Z': '径向振动反映齿轮啮合和不平衡'
                }
                st.info(direction_info[axis])

            return sensor, axis

    def create_analysis_parameters_selector(self):
        """创建分析参数选择器"""
        with st.expander("🔬 分析参数", expanded=False):
            # 时间范围选择
            st.subheader("时域参数")
            time_range = st.slider(
                "时间范围 (秒)",
                min_value=0.0,
                max_value=30.0,
                value=[0.0, 30.0],
                step=0.1,
                help="选择要分析的时间段，全程30秒"
            )

            # 频域参数
            st.subheader("频域参数")
            col1, col2 = st.columns(2)

            with col1:
                frequency_range = st.slider(
                    "频率范围 (Hz)",
                    min_value=0,
                    max_value=8000,
                    value=[0, 8000],
                    step=10,
                    help="分析的频率范围，最大8000Hz（奈奎斯特频率）"
                )

            with col2:
                fft_params = self._create_fft_parameter_selector()

            # 滤波参数
            st.subheader("滤波设置")
            filter_config = self._create_filter_parameter_selector()

            return {
                'time_range': time_range,
                'frequency_range': frequency_range,
                'fft_params': fft_params,
                'filter_config': filter_config
            }

    def _get_available_configurations(self):
        """获取所有可用的配置组合"""
        return self.data_loader.get_available_configs()

    def _get_available_drive_states(self):
        """获取可用的主动轮状态"""
        drive_states = list(set([c.drive_gear_state for c in self.available_configs]))
        return sorted(drive_states)

    def _get_available_driven_states(self, drive_state):
        """根据主动轮状态获取可用的从动轮状态"""
        driven_states = [
            c.driven_gear_state for c in self.available_configs
            if c.drive_gear_state == drive_state
        ]
        return sorted(list(set(driven_states)))

    def _show_configuration_validity(self, drive_state, driven_state):
        """显示配置组合的有效性"""
        matching_configs = [
            c for c in self.available_configs
            if c.drive_gear_state == drive_state and c.driven_gear_state == driven_state
        ]

        if matching_configs:
            st.success(f"✅ 找到 {len(matching_configs)} 个匹配的数据文件")

            # 显示可用的扭矩值
            available_torques = sorted(list(set([c.torque for c in matching_configs])))
            st.info(f"可用扭矩: {', '.join(map(str, available_torques))} Nm")
        else:
            st.error("❌ 未找到匹配的数据文件")

    def _create_fft_parameter_selector(self):
        """创建FFT参数选择器"""
        window_type = st.selectbox(
            "窗函数类型",
            options=['hann', 'hamming', 'blackman', 'kaiser'],
            help="不同窗函数影响频谱的分辨率和泄漏"
        )

        nperseg = st.selectbox(
            "FFT长度",
            options=[512, 1024, 2048, 4096],
            index=1,
            help="更大的FFT长度提供更好的频率分辨率但降低时间分辨率"
        )

        return {
            'window': window_type,
            'nperseg': nperseg
        }

    def _create_filter_parameter_selector(self):
        """创建滤波器参数选择器"""
        enable_filter = st.checkbox("启用数字滤波", value=False)

        if enable_filter:
            col1, col2, col3 = st.columns(3)

            with col1:
                filter_type = st.selectbox(
                    "滤波器类型",
                    options=['lowpass', 'highpass', 'bandpass', 'bandstop']
                )

            with col2:
                if filter_type in ['lowpass', 'highpass']:
                    cutoff_freq = st.number_input(
                        "截止频率 (Hz)",
                        min_value=1,
                        max_value=8000,
                        value=1000,
                        step=10
                    )
                    filter_params = {'cutoff': cutoff_freq}
                else:
                    low_freq = st.number_input(
                        "下截止频率 (Hz)",
                        min_value=1,
                        max_value=8000,
                        value=10,
                        step=10
                    )
                    filter_params = {'low_freq': low_freq}

            with col3:
                if filter_type in ['bandpass', 'bandstop']:
                    high_freq = st.number_input(
                        "上截止频率 (Hz)",
                        min_value=low_freq + 10,
                        max_value=8000,
                        value=1000,
                        step=10
                    )
                    filter_params['high_freq'] = high_freq

                filter_order = st.number_input(
                    "滤波器阶数",
                    min_value=1,
                    max_value=10,
                    value=4,
                    step=1
                )
                filter_params['order'] = filter_order

            return {
                'enable': True,
                'type': filter_type,
                'params': filter_params
            }
        else:
            return {'enable': False}
```

## 3. 数据集管理界面

### 3.1 多数据集对比管理器

```python
class DatasetComparisonManager:
    """数据集对比管理器 - 管理多个数据集的添加、删除和对比"""

    def __init__(self):
        if 'comparison_datasets' not in st.session_state:
            st.session_state.comparison_datasets = []

    def create_dataset_manager_ui(self):
        """创建数据集管理界面"""
        with st.sidebar:
            st.header("📊 数据集管理")

            # 当前配置显示
            current_config = self._get_current_config()
            self._show_current_config(current_config)

            # 添加数据集按钮
            if st.button("➕ 添加当前配置", use_container_width=True):
                self._add_dataset(current_config)

            # 显示已添加的数据集
            self._show_dataset_list()

            # 批量操作
            self._create_batch_operations()

    def _get_current_config(self):
        """获取当前的参数配置"""
        return {
            'drive_gear_state': st.session_state.get('drive_gear_state', '正常'),
            'driven_gear_state': st.session_state.get('driven_gear_state', '正常'),
            'torque': st.session_state.get('torque', 10),
            'sensor': st.session_state.get('sensor', 'A'),
            'axis': st.session_state.get('axis', 'X'),
            'time_range': st.session_state.get('time_range', [0, 30])
        }

    def _show_current_config(self, config):
        """显示当前配置"""
        st.subheader("🔧 当前配置")

        config_text = (
            f"**齿轮状态**: {config['drive_gear_state']} - {config['driven_gear_state']}\\n"
            f"**工况**: {config['torque']}Nm\\n"
            f"**传感器**: {config['sensor']}_{config['axis']}\\n"
            f"**时间**: {config['time_range'][0]:.1f}s - {config['time_range'][1]:.1f}s"
        )

        st.markdown(config_text)

    def _add_dataset(self, config):
        """添加数据集到对比列表"""
        # 生成唯一标识
        dataset_id = self._generate_dataset_id(config)

        # 检查是否已存在
        existing_ids = [ds['id'] for ds in st.session_state.comparison_datasets]
        if dataset_id in existing_ids:
            st.warning("⚠️ 该配置已存在于对比列表中")
            return

        # 生成显示标签
        label = self._generate_dataset_label(config)

        # 添加到列表
        dataset_info = {
            'id': dataset_id,
            'label': label,
            'config': config.copy(),
            'color': self._assign_color(len(st.session_state.comparison_datasets)),
            'visible': True,
            'added_time': pd.Timestamp.now()
        }

        st.session_state.comparison_datasets.append(dataset_info)
        st.success(f"✅ 已添加: {label}")
        st.rerun()

    def _show_dataset_list(self):
        """显示数据集列表"""
        if not st.session_state.comparison_datasets:
            st.info("📋 对比列表为空")
            return

        st.subheader(f"📋 对比列表 ({len(st.session_state.comparison_datasets)} 项)")

        for i, dataset in enumerate(st.session_state.comparison_datasets):
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])

                with col1:
                    # 显示数据集信息
                    color_indicator = f"🟦" if dataset['color'] == 'blue' else "🟥" if dataset['color'] == 'red' else "🟩"
                    visibility = "👁️" if dataset['visible'] else "🙈"

                    st.markdown(f"{color_indicator} {visibility} **{dataset['label']}**")

                    # 显示详细配置（可折叠）
                    with st.expander("详细信息", expanded=False):
                        for key, value in dataset['config'].items():
                            st.text(f"{key}: {value}")

                with col2:
                    # 可见性切换
                    if st.button("👁️" if dataset['visible'] else "🙈",
                               key=f"vis_{i}",
                               help="切换显示/隐藏"):
                        dataset['visible'] = not dataset['visible']
                        st.rerun()

                with col3:
                    # 删除按钮
                    if st.button("🗑️", key=f"del_{i}", help="删除数据集"):
                        st.session_state.comparison_datasets.pop(i)
                        st.rerun()

                st.markdown("---")

    def _create_batch_operations(self):
        """创建批量操作"""
        if not st.session_state.comparison_datasets:
            return

        st.subheader("🔧 批量操作")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("👁️ 全部显示", use_container_width=True):
                for dataset in st.session_state.comparison_datasets:
                    dataset['visible'] = True
                st.rerun()

            if st.button("🙈 全部隐藏", use_container_width=True):
                for dataset in st.session_state.comparison_datasets:
                    dataset['visible'] = False
                st.rerun()

        with col2:
            if st.button("🗑️ 清空列表", use_container_width=True):
                if st.session_state.get('confirm_clear', False):
                    st.session_state.comparison_datasets = []
                    st.session_state.confirm_clear = False
                    st.rerun()
                else:
                    st.session_state.confirm_clear = True
                    st.warning("再次点击确认清空")

    def _generate_dataset_id(self, config):
        """生成数据集唯一标识"""
        import hashlib
        config_str = str(sorted(config.items()))
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def _generate_dataset_label(self, config):
        """生成数据集显示标签"""
        return (
            f"{config['drive_gear_state']}-{config['driven_gear_state']}-"
            f"{config['torque']}Nm-{config['sensor']}{config['axis']}"
        )

    def _assign_color(self, index):
        """分配颜色"""
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        return colors[index % len(colors)]

    def get_visible_datasets(self):
        """获取可见的数据集"""
        return [ds for ds in st.session_state.comparison_datasets if ds['visible']]
```

## 4. 交互式图表容器

### 4.1 智能图表容器

```python
class InteractiveChartContainer:
    """交互式图表容器 - 管理图表显示和交互"""

    def __init__(self):
        self.chart_cache = {}
        self.interaction_state = {}

    def create_tabbed_chart_display(self, datasets):
        """创建标签页式图表显示"""
        if not datasets:
            st.info("📊 请先添加数据集到对比列表")
            return

        # 创建标签页
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 时域分析",
            "🔊 频域分析",
            "🔍 统计特征",
            "📋 数据表格"
        ])

        with tab1:
            self._create_time_domain_tab(datasets)

        with tab2:
            self._create_frequency_domain_tab(datasets)

        with tab3:
            self._create_statistical_analysis_tab(datasets)

        with tab4:
            self._create_data_table_tab(datasets)

    def _create_time_domain_tab(self, datasets):
        """创建时域分析标签页"""
        col1, col2 = st.columns([3, 1])

        with col1:
            # 主图表区域
            chart_container = st.container()

            # 图表控制选项
            show_envelope = st.checkbox("显示包络线", value=False)
            show_rms_line = st.checkbox("显示RMS水平线", value=False)

            with chart_container:
                # 这里会渲染实际的时域图表
                self._render_time_domain_chart(datasets, show_envelope, show_rms_line)

        with col2:
            # 侧边控制面板
            st.subheader("🎛️ 图表控制")

            # Y轴范围控制
            auto_scale = st.checkbox("自动缩放", value=True)
            if not auto_scale:
                y_min = st.number_input("Y轴最小值", value=-1.0, step=0.1, format="%.3f")
                y_max = st.number_input("Y轴最大值", value=1.0, step=0.1, format="%.3f")
            else:
                y_min, y_max = None, None

            # 降采样控制
            enable_downsample = st.checkbox("启用降采样", value=True)
            if enable_downsample:
                downsample_factor = st.slider(
                    "降采样因子",
                    min_value=1,
                    max_value=100,
                    value=10,
                    help="减少显示点数以提高性能"
                )
            else:
                downsample_factor = 1

            # 导出选项
            st.subheader("📤 导出选项")
            if st.button("导出图表", use_container_width=True):
                self._export_chart('time_domain')

            if st.button("导出数据", use_container_width=True):
                self._export_data(datasets)

    def _render_time_domain_chart(self, datasets, show_envelope, show_rms_line):
        """渲染时域图表（这里是占位符，实际实现会调用可视化模块）"""
        # 这里会调用之前实现的可视化模块来生成图表
        st.plotly_chart(
            self._generate_time_domain_figure(datasets, show_envelope, show_rms_line),
            use_container_width=True
        )

    def create_floating_chart_controls(self):
        """创建浮动图表控制面板"""
        with st.sidebar:
            st.header("🎛️ 图表控制")

            # 全局图表设置
            chart_theme = st.selectbox(
                "图表主题",
                options=['plotly_white', 'plotly_dark', 'ggplot2', 'seaborn'],
                help="选择图表的整体主题风格"
            )

            # 性能设置
            st.subheader("⚡ 性能设置")
            max_points = st.slider(
                "最大显示点数",
                min_value=1000,
                max_value=50000,
                value=10000,
                help="限制显示点数以提高渲染性能"
            )

            # 交互设置
            st.subheader("🖱️ 交互设置")
            enable_crossfilter = st.checkbox(
                "启用图表联动",
                value=True,
                help="在一个图表中选择区域时，其他图表自动聚焦到相同区域"
            )

            enable_sync_zoom = st.checkbox(
                "同步缩放",
                value=True,
                help="所有时域图表的缩放操作保持同步"
            )

            return {
                'theme': chart_theme,
                'max_points': max_points,
                'enable_crossfilter': enable_crossfilter,
                'enable_sync_zoom': enable_sync_zoom
            }
```

## 5. 响应式设计与性能优化

### 5.1 自适应界面组件

```python
class ResponsiveUIManager:
    """响应式UI管理器 - 根据屏幕尺寸和内容动态调整界面"""

    def __init__(self):
        self.screen_info = self._detect_screen_info()

    def _detect_screen_info(self):
        """检测屏幕信息（通过JavaScript注入）"""
        # 在Streamlit中获取屏幕信息的方法
        screen_detection_js = """
        <script>
        function getScreenInfo() {
            return {
                width: window.screen.width,
                height: window.screen.height,
                availWidth: window.screen.availWidth,
                availHeight: window.screen.availHeight
            };
        }

        window.parent.postMessage({
            type: 'streamlit:setFrameHeight',
            data: getScreenInfo()
        }, '*');
        </script>
        """

        st.components.v1.html(screen_detection_js, height=0)

        # 默认桌面尺寸
        return {
            'width': 1920,
            'height': 1080,
            'is_mobile': False,
            'is_tablet': False
        }

    def adapt_layout_for_screen(self):
        """根据屏幕尺寸调整布局"""
        if self.screen_info['width'] < 768:
            # 移动设备布局
            return self._create_mobile_layout()
        elif self.screen_info['width'] < 1024:
            # 平板布局
            return self._create_tablet_layout()
        else:
            # 桌面布局
            return self._create_desktop_layout()

    def _create_mobile_layout(self):
        """创建移动设备布局"""
        # 垂直堆叠，减少并列列
        st.warning("📱 检测到移动设备，已优化界面布局")

        # 折叠式参数面板
        with st.expander("⚙️ 参数设置", expanded=False):
            # 参数选择组件
            pass

        # 图表全宽显示
        chart_container = st.container()

        # 简化的控制面板
        with st.expander("🎛️ 图表控制", expanded=False):
            # 最小化的控制选项
            pass

        return {
            'layout_type': 'mobile',
            'chart_container': chart_container,
            'sidebar_width': None
        }

    def _create_tablet_layout(self):
        """创建平板布局"""
        st.info("📱 检测到平板设备，已调整界面布局")

        # 上下分栏布局
        param_container = st.container()
        chart_container = st.container()

        return {
            'layout_type': 'tablet',
            'param_container': param_container,
            'chart_container': chart_container
        }

    def _create_desktop_layout(self):
        """创建桌面布局"""
        # 标准的侧边栏+主内容布局
        return {
            'layout_type': 'desktop',
            'use_sidebar': True,
            'main_columns': [1, 4]
        }

    def optimize_component_for_screen(self, component_type, **kwargs):
        """根据屏幕尺寸优化组件"""
        optimizations = {
            'mobile': {
                'font_size': 'small',
                'button_size': 'large',
                'input_width': 'full',
                'chart_height': 300
            },
            'tablet': {
                'font_size': 'medium',
                'button_size': 'medium',
                'input_width': 'auto',
                'chart_height': 400
            },
            'desktop': {
                'font_size': 'normal',
                'button_size': 'normal',
                'input_width': 'auto',
                'chart_height': 500
            }
        }

        layout_type = self.screen_info.get('layout_type', 'desktop')
        return optimizations.get(layout_type, optimizations['desktop'])
```

### 5.2 性能监控与优化

```python
class UIPerformanceMonitor:
    """UI性能监控器 - 监控和优化界面性能"""

    def __init__(self):
        self.performance_metrics = {}
        self.optimization_enabled = True

    def monitor_rendering_performance(self, component_name):
        """监控组件渲染性能"""
        import time
        import psutil

        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        # 这里返回一个上下文管理器
        class PerformanceContext:
            def __init__(self, monitor, name, start_time, start_memory):
                self.monitor = monitor
                self.name = name
                self.start_time = start_time
                self.start_memory = start_memory

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024

                self.monitor.performance_metrics[self.name] = {
                    'render_time': end_time - self.start_time,
                    'memory_usage': end_memory - self.start_memory,
                    'timestamp': end_time
                }

                # 性能告警
                if end_time - self.start_time > 2.0:  # 超过2秒
                    st.warning(f"⚠️ {self.name} 渲染时间较长: {end_time - self.start_time:.2f}s")

        return PerformanceContext(self, component_name, start_time, start_memory)

    def create_performance_dashboard(self):
        """创建性能监控面板"""
        if not self.performance_metrics:
            return

        with st.sidebar:
            with st.expander("⚡ 性能监控", expanded=False):
                for component, metrics in self.performance_metrics.items():
                    st.metric(
                        label=component,
                        value=f"{metrics['render_time']:.2f}s",
                        delta=f"{metrics['memory_usage']:.1f}MB"
                    )

                # 性能优化建议
                self._show_optimization_suggestions()

    def _show_optimization_suggestions(self):
        """显示性能优化建议"""
        suggestions = []

        # 分析渲染时间
        slow_components = [
            name for name, metrics in self.performance_metrics.items()
            if metrics['render_time'] > 1.0
        ]

        if slow_components:
            suggestions.append(f"🐌 慢组件: {', '.join(slow_components)}")

        # 分析内存使用
        high_memory_components = [
            name for name, metrics in self.performance_metrics.items()
            if metrics['memory_usage'] > 100  # 100MB
        ]

        if high_memory_components:
            suggestions.append(f"🧠 高内存: {', '.join(high_memory_components)}")

        if suggestions:
            st.subheader("💡 优化建议")
            for suggestion in suggestions:
                st.text(suggestion)
```

这些UI设计技术点确保了系统具有良好的用户体验、高性能和跨设备兼容性，为齿轮磨损数据分析提供了直观、高效的操作界面。

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "\u7f16\u5199UI\u8bbe\u8ba1\u6280\u672f\u70b9\u6587\u6863", "status": "completed", "activeForm": "\u6b63\u5728\u7f16\u5199UI\u8bbe\u8ba1\u6280\u672f\u70b9\u6587\u6863"}, {"content": "\u521b\u5efadesign\u6587\u4ef6\u5939\u603b\u7ed3\u6587\u6863", "status": "in_progress", "activeForm": "\u6b63\u5728\u521b\u5efadesign\u6587\u4ef6\u5939\u603b\u7ed3\u6587\u6863"}]