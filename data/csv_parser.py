"""
CSV数据解析器

负责解析齿轮振动数据CSV文件的结构，包括：
- 元数据解析（采样频率、数据点数、扭矩等）
- 传感器数据解析
- 分块读取支持
- 内存优化读取

CSV文件结构:
- 第1行: 元数据 [采样频率, 采样点数, 扭矩, ...]
- 第2-4行: 配置参数（暂未使用）
- 第5行起: 10列传感器数据
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from pathlib import Path

from config.settings import (
    CSV_METADATA_ROW,
    CSV_DATA_START_ROW,
    CHANNEL_COUNT,
    CHANNEL_MAP,
    SAMPLING_RATE,
    EXPECTED_DATA_POINTS,
    logger
)


class CSVParser:
    """
    CSV数据解析器

    提供CSV文件解析功能，支持元数据提取和传感器数据读取。
    """

    def __init__(self, filepath: str):
        """
        初始化CSV解析器

        Args:
            filepath: CSV文件路径
        """
        self.filepath = Path(filepath)

        if not self.filepath.exists():
            raise FileNotFoundError(f"文件不存在: {filepath}")

        self._metadata: Optional[Dict] = None
        logger.debug(f"CSVParser 初始化: {self.filepath.name}")

    @property
    def metadata(self) -> Dict:
        """
        获取元数据（懒加载）

        Returns:
            dict: 元数据字典
        """
        if self._metadata is None:
            self._metadata = self.parse_metadata()
        return self._metadata

    def parse_metadata(self) -> Dict:
        """
        解析CSV文件的元数据（第1行）

        Returns:
            dict: 包含采样频率、数据点数、扭矩等信息

        Examples:
            >>> parser = CSVParser('data.csv')
            >>> metadata = parser.parse_metadata()
            >>> print(metadata['sampling_rate'])
            15360
        """
        try:
            # 只读取第一行
            metadata_row = pd.read_csv(
                self.filepath,
                nrows=1,
                header=None
            ).values[0]

            # 解析元数据
            # 格式: [采样频率, 采样点数, 扭矩, param1, param2, ...]
            metadata = {
                'sampling_rate': int(metadata_row[0]),
                'data_points': int(metadata_row[1]),
                'torque': int(metadata_row[2]),
                'param1': metadata_row[3] if len(metadata_row) > 3 else None,
                'param2': metadata_row[4] if len(metadata_row) > 4 else None,
            }

            logger.debug(f"元数据解析成功: {metadata}")
            return metadata

        except Exception as e:
            logger.error(f"元数据解析失败: {e}")
            # 返回默认值
            return {
                'sampling_rate': SAMPLING_RATE,
                'data_points': EXPECTED_DATA_POINTS,
                'torque': 0
            }

    def read_all_data(self) -> np.ndarray:
        """
        读取所有传感器数据

        Returns:
            np.ndarray: shape为(data_points, 10)的数组

        Examples:
            >>> parser = CSVParser('data.csv')
            >>> data = parser.read_all_data()
            >>> print(data.shape)
            (460800, 10)
        """
        try:
            # 跳过前4行（元数据+配置），读取所有传感器数据
            data = pd.read_csv(
                self.filepath,
                skiprows=CSV_DATA_START_ROW,
                header=None
            ).values

            logger.info(f"读取数据完成: shape={data.shape}")
            return data

        except Exception as e:
            logger.error(f"读取数据失败: {e}")
            raise

    def read_channel(
        self,
        channel_idx: int,
        start_row: int = 0,
        end_row: Optional[int] = None
    ) -> np.ndarray:
        """
        读取指定通道的数据

        Args:
            channel_idx: 通道索引 (0-9)
            start_row: 开始行号（相对于数据起始行）
            end_row: 结束行号（None表示读到末尾）

        Returns:
            np.ndarray: 通道数据

        Examples:
            >>> parser = CSVParser('data.csv')
            >>> # 读取传感器A的X轴数据
            >>> ax_data = parser.read_channel(0)
        """
        if not 0 <= channel_idx < CHANNEL_COUNT:
            raise ValueError(f"通道索引超出范围: {channel_idx}")

        try:
            # 计算实际的skiprows
            skip_start = CSV_DATA_START_ROW + start_row

            # 计算要读取的行数
            if end_row is not None:
                nrows = end_row - start_row
            else:
                nrows = None

            # 只读取指定列
            data = pd.read_csv(
                self.filepath,
                usecols=[channel_idx],
                skiprows=skip_start,
                nrows=nrows,
                header=None
            ).values.flatten()

            logger.debug(f"读取通道 {channel_idx}: {len(data)} 个采样点")
            return data

        except Exception as e:
            logger.error(f"读取通道 {channel_idx} 失败: {e}")
            raise

    def read_sensor_data(
        self,
        sensor: str,
        axis: str,
        start_row: int = 0,
        end_row: Optional[int] = None
    ) -> np.ndarray:
        """
        读取指定传感器和轴向的数据

        Args:
            sensor: 传感器标识 ('A', 'B', 'C')
            axis: 轴向标识 ('X', 'Y', 'Z')
            start_row: 开始行号
            end_row: 结束行号

        Returns:
            np.ndarray: 传感器数据

        Examples:
            >>> parser = CSVParser('data.csv')
            >>> # 读取传感器B的Y轴数据
            >>> by_data = parser.read_sensor_data('B', 'Y')
        """
        # 获取通道索引
        channel_key = f"{sensor}_{axis}"

        if channel_key not in CHANNEL_MAP:
            raise ValueError(f"无效的传感器-轴向组合: {channel_key}")

        channel_idx = CHANNEL_MAP[channel_key]

        return self.read_channel(channel_idx, start_row, end_row)

    def read_chunk(
        self,
        start_row: int,
        chunk_size: int
    ) -> np.ndarray:
        """
        分块读取数据

        Args:
            start_row: 开始行号（相对于数据起始行）
            chunk_size: 块大小（行数）

        Returns:
            np.ndarray: shape为(chunk_size, 10)的数据块
        """
        end_row = start_row + chunk_size

        try:
            data = pd.read_csv(
                self.filepath,
                skiprows=CSV_DATA_START_ROW + start_row,
                nrows=chunk_size,
                header=None
            ).values

            logger.debug(f"读取数据块: rows {start_row}-{end_row}")
            return data

        except Exception as e:
            logger.error(f"读取数据块失败: {e}")
            raise

    def iter_chunks(self, chunk_size: int = 10000):
        """
        迭代器：分块读取所有数据

        Args:
            chunk_size: 每块的大小（行数）

        Yields:
            np.ndarray: 数据块

        Examples:
            >>> parser = CSVParser('data.csv')
            >>> for chunk in parser.iter_chunks(10000):
            ...     # 处理每个数据块
            ...     process(chunk)
        """
        total_rows = self.metadata['data_points']
        current_row = 0

        while current_row < total_rows:
            # 计算本次读取的行数
            rows_to_read = min(chunk_size, total_rows - current_row)

            # 读取数据块
            chunk = self.read_chunk(current_row, rows_to_read)

            yield chunk

            current_row += rows_to_read

    def get_time_axis(self) -> np.ndarray:
        """
        生成时间轴

        Returns:
            np.ndarray: 时间轴（秒）
        """
        sampling_rate = self.metadata['sampling_rate']
        data_points = self.metadata['data_points']

        time_axis = np.arange(data_points) / sampling_rate

        return time_axis

    def read_time_range(
        self,
        start_time: float,
        end_time: float,
        channel_idx: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        读取指定时间范围的数据

        Args:
            start_time: 开始时间（秒）
            end_time: 结束时间（秒）
            channel_idx: 通道索引，None表示读取所有通道

        Returns:
            tuple: (时间轴, 数据)
        """
        sampling_rate = self.metadata['sampling_rate']

        # 转换时间到行号
        start_row = int(start_time * sampling_rate)
        end_row = int(end_time * sampling_rate)

        # 读取数据
        if channel_idx is not None:
            data = self.read_channel(channel_idx, start_row, end_row)
        else:
            data = pd.read_csv(
                self.filepath,
                skiprows=CSV_DATA_START_ROW + start_row,
                nrows=end_row - start_row,
                header=None
            ).values

        # 生成时间轴
        time_axis = np.arange(start_row, end_row) / sampling_rate

        return time_axis, data

    def get_data_info(self) -> Dict:
        """
        获取数据文件的基本信息

        Returns:
            dict: 包含文件信息、元数据、数据统计等
        """
        info = {
            'filename': self.filepath.name,
            'filepath': str(self.filepath),
            'file_size_mb': self.filepath.stat().st_size / 1024 / 1024,
            'metadata': self.metadata,
            'expected_duration': self.metadata['data_points'] / self.metadata['sampling_rate'],
            'channels': CHANNEL_COUNT,
            'channel_names': list(CHANNEL_MAP.keys())
        }

        return info


# ====================================
# 便捷函数
# ====================================

def quick_read(filepath: str, sensor: str, axis: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    快速读取指定传感器数据（便捷函数）

    Args:
        filepath: CSV文件路径
        sensor: 传感器 ('A', 'B', 'C')
        axis: 轴向 ('X', 'Y', 'Z')

    Returns:
        tuple: (时间轴, 数据)
    """
    parser = CSVParser(filepath)
    data = parser.read_sensor_data(sensor, axis)
    time_axis = parser.get_time_axis()

    return time_axis, data


def get_file_info(filepath: str) -> Dict:
    """
    快速获取文件信息（便捷函数）

    Args:
        filepath: CSV文件路径

    Returns:
        dict: 文件信息
    """
    parser = CSVParser(filepath)
    return parser.get_data_info()


# ====================================
# 测试代码
# ====================================

if __name__ == "__main__":
    # 查找测试文件
    from data.filename_parser import scan_data_files

    print("=" * 60)
    print("CSV解析器测试")
    print("=" * 60)

    # 获取第一个数据文件
    configs = scan_data_files()

    if not configs:
        print("❌ 未找到数据文件")
        exit(1)

    test_file = configs[0].filepath
    print(f"\n使用测试文件: {Path(test_file).name}")

    # 测试1: 解析元数据
    print("\n【测试1】解析元数据:")
    parser = CSVParser(test_file)
    metadata = parser.metadata
    print(f"   采样频率: {metadata['sampling_rate']} Hz")
    print(f"   数据点数: {metadata['data_points']}")
    print(f"   扭矩: {metadata['torque']} Nm")
    print(f"   预计时长: {metadata['data_points'] / metadata['sampling_rate']:.2f} 秒")

    # 测试2: 读取单个通道
    print("\n【测试2】读取传感器A的X轴数据:")
    ax_data = parser.read_sensor_data('A', 'X')
    print(f"   数据形状: {ax_data.shape}")
    print(f"   数据范围: [{ax_data.min():.6f}, {ax_data.max():.6f}]")
    print(f"   均值: {ax_data.mean():.6f}")
    print(f"   标准差: {ax_data.std():.6f}")

    # 测试3: 读取时间范围数据
    print("\n【测试3】读取前1秒的数据:")
    time_axis, data = parser.read_time_range(0, 1.0, channel_idx=0)
    print(f"   时间范围: {time_axis[0]:.3f} - {time_axis[-1]:.3f} 秒")
    print(f"   数据点数: {len(data)}")

    # 测试4: 分块读取
    print("\n【测试4】分块读取测试（10000行/块）:")
    chunk_count = 0
    for i, chunk in enumerate(parser.iter_chunks(10000)):
        chunk_count += 1
        if i < 3:  # 只显示前3块
            print(f"   块 {i+1}: shape={chunk.shape}")

    print(f"   总共 {chunk_count} 个数据块")

    # 测试5: 获取文件信息
    print("\n【测试5】文件信息:")
    info = parser.get_data_info()
    print(f"   文件名: {info['filename']}")
    print(f"   文件大小: {info['file_size_mb']:.2f} MB")
    print(f"   通道数: {info['channels']}")
    print(f"   预计时长: {info['expected_duration']:.2f} 秒")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
