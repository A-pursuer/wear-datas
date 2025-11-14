"""
全局配置管理模块

提供项目的所有配置参数，包括路径、数据参数、缓存设置、可视化配置等。
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List
import logging

# ====================================
# 项目基础配置
# ====================================

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
PROJECT_NAME = "齿轮磨损数据分析系统"
VERSION = "1.1.0"

# ====================================
# 路径配置
# ====================================

# 数据目录
DATA_DIR = PROJECT_ROOT
CSV_PATTERN = "*.csv"

# 缓存目录
CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# 日志目录
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# 测试数据目录
TEST_DATA_DIR = PROJECT_ROOT / "tests" / "test_data"

# ====================================
# 数据参数配置
# ====================================

# 采样参数
SAMPLING_RATE = 15360  # Hz
SAMPLING_DURATION = 30  # seconds
EXPECTED_DATA_POINTS = 460805  # 每个文件的预期数据行数

# CSV文件结构
CSV_METADATA_ROW = 0  # 第1行：元数据
CSV_CONFIG_ROWS = [1, 2, 3]  # 第2-4行：配置参数
CSV_DATA_START_ROW = 4  # 第5行开始：实际数据
CHANNEL_COUNT = 10  # 总通道数

# 通道映射关系
CHANNEL_MAP = {
    'A_X': 0, 'A_Y': 1, 'A_Z': 2,  # 传感器A三轴
    'B_X': 3, 'B_Y': 4, 'B_Z': 5,  # 传感器B三轴
    'C_X': 6, 'C_Y': 7, 'C_Z': 8,  # 传感器C三轴
    'SPEED': 9  # 转速通道
}

# 传感器配置
SENSORS = ['A', 'B', 'C']
AXES = ['X', 'Y', 'Z']

# 传感器位置信息
SENSOR_INFO = {
    'A': {
        'name': '传感器A',
        'location': '主动轴输入轴承处',
        'description': '监测主动轴输入端振动特征'
    },
    'B': {
        'name': '传感器B',
        'location': '从动轴输入处',
        'description': '监测齿轮啮合区振动特征'
    },
    'C': {
        'name': '传感器C',
        'location': '从动轴输出处',
        'description': '监测从动轴输出端振动特征'
    }
}

# 轴向信息
AXIS_INFO = {
    'X': '轴向 - 反映轴承和轴的状态',
    'Y': '径向 - 反映齿轮啮合和不平衡',
    'Z': '径向 - 反映齿轮啮合和不平衡'
}

# 磨损状态映射
WEAR_STATE_MAP = {
    '正常': 'normal',
    '轻磨': 'light_wear',
    '重磨': 'heavy_wear'
}

WEAR_STATE_REVERSE_MAP = {v: k for k, v in WEAR_STATE_MAP.items()}

# 实验工况
AVAILABLE_TORQUES = [10, 15]  # Nm
AVAILABLE_SPEEDS = [1000]  # rpm
GEAR_MESH_FREQUENCY_BASE = 333.33  # Hz (1000rpm下的基频)

# ====================================
# 缓存配置
# ====================================

# 缓存开关
ENABLE_CACHE = True

# 内存缓存配置
MAX_CACHE_SIZE_MB = 500  # L1缓存最大内存占用
L1_CACHE_MAX_ITEMS = 50  # L1缓存最大条目数

# 磁盘缓存配置
ENABLE_DISK_CACHE = True
DISK_CACHE_DIR = CACHE_DIR / "disk_cache"
DISK_CACHE_DIR.mkdir(exist_ok=True)

# 缓存策略
CACHE_STRATEGY = "LRU"  # LRU, LFU, FIFO
CACHE_TTL = 3600  # 缓存过期时间（秒），-1表示永不过期

# ====================================
# 信号处理配置
# ====================================

# 滤波器默认参数
DEFAULT_FILTER_ORDER = 4
DEFAULT_LOWPASS_CUTOFF = 1000  # Hz
DEFAULT_HIGHPASS_CUTOFF = 10  # Hz
DEFAULT_BANDPASS_LOW = 10  # Hz
DEFAULT_BANDPASS_HIGH = 1000  # Hz

# FFT参数
DEFAULT_FFT_WINDOW = 'hann'  # 窗函数类型
DEFAULT_NPERSEG = 1024  # 每段长度
DEFAULT_NOVERLAP = 512  # 重叠长度

# 小波变换参数
DEFAULT_WAVELET = 'cmor'  # 复Morlet小波
DEFAULT_WAVELET_SCALES = list(range(1, 128))  # 尺度范围

# 特征提取参数
IMPACT_THRESHOLD_MULTIPLIER = 3.0  # 冲击检测阈值倍数
PEAK_DETECTION_DISTANCE = 100  # 峰值检测最小距离（采样点）

# ====================================
# 可视化配置
# ====================================

# 图表主题
CHART_THEME = 'plotly_white'  # plotly_white, plotly_dark, ggplot2, seaborn

# 性能优化
MAX_DISPLAY_POINTS = 10000  # 最大显示点数
DOWNSAMPLE_METHOD = 'lttb'  # lttb, max_min, uniform
ENABLE_DOWNSAMPLE = True

# 图表尺寸
DEFAULT_CHART_HEIGHT = 500  # px
DEFAULT_CHART_WIDTH = None  # None表示自适应

# 颜色方案
COLOR_PALETTE = [
    '#1f77b4',  # 蓝色
    '#ff7f0e',  # 橙色
    '#2ca02c',  # 绿色
    '#d62728',  # 红色
    '#9467bd',  # 紫色
    '#8c564b',  # 棕色
    '#e377c2',  # 粉色
    '#7f7f7f',  # 灰色
    '#bcbd22',  # 黄绿色
    '#17becf'   # 青色
]

# 磨损状态颜色映射
WEAR_STATE_COLORS = {
    'normal': '#2ca02c',      # 绿色
    'light_wear': '#ff7f0e',  # 橙色
    'heavy_wear': '#d62728'   # 红色
}

# ====================================
# UI配置
# ====================================

# Streamlit页面配置
PAGE_TITLE = "齿轮磨损数据分析系统"
PAGE_ICON = "⚙️"
LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "expanded"

# 侧边栏宽度
SIDEBAR_WIDTH = 300

# 主内容区列比例
MAIN_CONTENT_RATIO = [1, 4]  # 控制面板:显示区域 = 1:4

# ====================================
# 日志配置
# ====================================

# 日志级别
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# 日志格式
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 日志文件
LOG_FILE = LOG_DIR / "gear_analysis.log"
ERROR_LOG_FILE = LOG_DIR / "error.log"

# 日志轮转
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5

# ====================================
# 性能监控配置
# ====================================

# 性能监控开关
ENABLE_PERFORMANCE_MONITORING = True

# 性能警告阈值
PERFORMANCE_THRESHOLDS = {
    'data_load_time': 5.0,      # 数据加载时间（秒）
    'processing_time': 3.0,      # 信号处理时间（秒）
    'rendering_time': 2.0,       # 图表渲染时间（秒）
    'memory_usage': 1024,        # 内存占用（MB）
}

# ====================================
# 开发配置
# ====================================

# 调试模式
DEBUG = False

# 开发模式（启用更详细的日志和错误信息）
DEV_MODE = False

# 测试模式
TEST_MODE = False

# ====================================
# 数据类定义
# ====================================

@dataclass
class FileConfig:
    """文件配置数据类"""
    drive_gear_state: str      # 主动轮状态
    driven_gear_state: str     # 从动轮状态
    torque: int                # 扭矩值（Nm）
    speed: int                 # 转速（rpm）
    filepath: str              # 文件完整路径

    def __str__(self) -> str:
        return (f"{self.drive_gear_state}-{self.driven_gear_state}-"
                f"{self.torque}Nm-{self.speed}rpm")


@dataclass
class GearConfig:
    """齿轮配置数据类"""
    drive_gear_state: str     # 主动轮状态
    driven_gear_state: str    # 从动轮状态
    torque: int               # 扭矩值
    sensor: str               # 传感器位置
    axis: str                 # 测量方向

    def __str__(self) -> str:
        return (f"{self.drive_gear_state}-{self.driven_gear_state}-"
                f"{self.torque}Nm-{self.sensor}{self.axis}")


@dataclass
class SignalData:
    """信号数据包装类"""
    time_series: 'np.ndarray'   # 时域数据
    sampling_rate: int          # 采样频率
    metadata: Dict              # 元数据信息
    config: GearConfig          # 采集配置

    def __len__(self) -> int:
        return len(self.time_series)

    @property
    def duration(self) -> float:
        """信号时长（秒）"""
        return len(self.time_series) / self.sampling_rate


# ====================================
# 日志设置
# ====================================

def setup_logging(level: str = LOG_LEVEL) -> logging.Logger:
    """
    配置日志系统

    Args:
        level: 日志级别

    Returns:
        logging.Logger: 配置好的logger
    """
    # 创建logger
    logger = logging.getLogger(PROJECT_NAME)
    logger.setLevel(getattr(logging, level.upper()))

    # 避免重复添加handler
    if logger.handlers:
        return logger

    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
    console_handler.setFormatter(console_formatter)

    # 文件handler
    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
    file_handler.setFormatter(file_formatter)

    # 错误日志handler
    error_handler = logging.FileHandler(ERROR_LOG_FILE, encoding='utf-8')
    error_handler.setLevel(logging.ERROR)
    error_formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
    error_handler.setFormatter(error_formatter)

    # 添加handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)

    return logger


# 创建全局logger
logger = setup_logging()


# ====================================
# 配置验证
# ====================================

def validate_config() -> bool:
    """
    验证配置的有效性

    Returns:
        bool: 配置是否有效
    """
    try:
        # 检查必要目录
        assert DATA_DIR.exists(), f"数据目录不存在: {DATA_DIR}"
        assert CACHE_DIR.exists(), f"缓存目录不存在: {CACHE_DIR}"
        assert LOG_DIR.exists(), f"日志目录不存在: {LOG_DIR}"

        # 检查参数合理性
        assert SAMPLING_RATE > 0, "采样率必须大于0"
        assert MAX_DISPLAY_POINTS > 0, "最大显示点数必须大于0"
        assert MAX_CACHE_SIZE_MB > 0, "缓存大小必须大于0"

        logger.info("配置验证通过")
        return True

    except AssertionError as e:
        logger.error(f"配置验证失败: {e}")
        return False


if __name__ == "__main__":
    # 验证配置
    if validate_config():
        print(f"✅ {PROJECT_NAME} v{VERSION} 配置验证成功")
        print(f"📁 项目根目录: {PROJECT_ROOT}")
        print(f"📁 数据目录: {DATA_DIR}")
        print(f"📁 缓存目录: {CACHE_DIR}")
        print(f"📁 日志目录: {LOG_DIR}")
    else:
        print("❌ 配置验证失败，请检查配置")
