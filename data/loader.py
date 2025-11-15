"""
数据加载器

提供统一的数据加载接口，整合：
- 文件名解析
- CSV解析
- 数据验证
- 懒加载
- 缓存支持

使用示例:
    >>> loader = DataLoader()
    >>> # 加载特定配置的数据
    >>> data = loader.load('light_wear', 'normal', 10, 'A', 'X')
    >>> # 或使用配置对象加载
    >>> config = GearConfig('light_wear', 'normal', 10, 'A', 'X')
    >>> data = loader.load_by_config(config)
"""

import numpy as np
from typing import Optional, Dict, Tuple, List
from pathlib import Path
from datetime import datetime

from config.settings import (
    GearConfig,
    SignalData,
    FileConfig,
    SAMPLING_RATE,
    logger
)

from data.filename_parser import FilenameParser, get_file_by_config
from data.csv_parser import CSVParser
from data.validator import DataValidator


class DataLoader:
    """
    数据加载器（基础版）

    提供基本的数据加载功能。
    """

    def __init__(self, validate: bool = True):
        """
        初始化数据加载器

        Args:
            validate: 是否在加载前验证数据文件
        """
        self.validate = validate
        self.filename_parser = FilenameParser()
        self.validator = DataValidator()

        logger.info("DataLoader 初始化完成")

    def load(
        self,
        drive_state: str,
        driven_state: str,
        torque: int,
        sensor: str,
        axis: str,
        speed: int = 1000,
        time_range: Optional[Tuple[float, float]] = None
    ) -> Optional[SignalData]:
        """
        加载数据（根据参数）

        Args:
            drive_state: 主动轮状态 ('normal', 'light_wear', 'heavy_wear')
            driven_state: 从动轮状态
            torque: 扭矩 (10, 15)
            sensor: 传感器 ('A', 'B', 'C')
            axis: 轴向 ('X', 'Y', 'Z')
            speed: 转速 (默认1000)
            time_range: 时间范围 (start, end) 秒，None表示全部

        Returns:
            SignalData: 包装的信号数据，未找到返回None

        Examples:
            >>> loader = DataLoader()
            >>> data = loader.load('light_wear', 'normal', 10, 'A', 'X')
            >>> print(f"采样点数: {len(data)}")
        """
        # 查找文件
        file_config = get_file_by_config(drive_state, driven_state, torque, speed)

        if file_config is None:
            logger.error(
                f"未找到匹配的文件: {drive_state}-{driven_state}-{torque}Nm-{speed}rpm"
            )
            return None

        # 创建齿轮配置
        gear_config = GearConfig(
            drive_gear_state=drive_state,
            driven_gear_state=driven_state,
            torque=torque,
            sensor=sensor,
            axis=axis
        )

        # 加载数据
        return self._load_from_file(file_config.filepath, gear_config, time_range)

    def load_by_config(
        self,
        config: GearConfig,
        time_range: Optional[Tuple[float, float]] = None
    ) -> Optional[SignalData]:
        """
        根据配置对象加载数据

        Args:
            config: 齿轮配置对象
            time_range: 时间范围

        Returns:
            SignalData: 信号数据
        """
        return self.load(
            config.drive_gear_state,
            config.driven_gear_state,
            config.torque,
            config.sensor,
            config.axis,
            time_range=time_range
        )

    def _load_from_file(
        self,
        filepath: str,
        gear_config: GearConfig,
        time_range: Optional[Tuple[float, float]] = None
    ) -> Optional[SignalData]:
        """
        从文件加载数据（内部方法）

        Args:
            filepath: 文件路径
            gear_config: 齿轮配置
            time_range: 时间范围

        Returns:
            SignalData: 信号数据
        """
        # 可选：数据验证
        if self.validate:
            validation_result = self.validator.validate_file_complete(filepath)
            if not validation_result.is_valid:
                logger.error(f"文件验证失败: {filepath}")
                logger.error(f"错误: {validation_result.errors}")
                return None

        # 解析CSV
        try:
            parser = CSVParser(filepath)

            # 读取数据
            if time_range:
                time_axis, data = parser.read_time_range(
                    time_range[0],
                    time_range[1],
                    channel_idx=None
                )
                # 从数据中提取指定通道
                from config.settings import CHANNEL_MAP
                channel_key = f"{gear_config.sensor}_{gear_config.axis}"
                channel_idx = CHANNEL_MAP[channel_key]
                time_series = data[:, channel_idx]
            else:
                time_series = parser.read_sensor_data(
                    gear_config.sensor,
                    gear_config.axis
                )

            # 创建SignalData对象
            metadata = parser.metadata
            metadata['loaded_at'] = datetime.now().isoformat()
            metadata['filepath'] = filepath

            signal_data = SignalData(
                time_series=time_series,
                sampling_rate=metadata['sampling_rate'],
                metadata=metadata,
                config=gear_config
            )

            logger.info(
                f"数据加载成功: {gear_config} - {len(time_series)} 采样点"
            )

            return signal_data

        except Exception as e:
            logger.error(f"加载数据时出错: {e}")
            return None

    def get_available_files(self) -> List[FileConfig]:
        """
        获取所有可用的数据文件配置

        Returns:
            List[FileConfig]: 文件配置列表
        """
        return self.filename_parser.scan_directory()

    def get_available_configs(self) -> Dict[str, List]:
        """
        获取所有可用的配置选项

        Returns:
            dict: 包含可用的状态、扭矩等选项
        """
        files = self.get_available_files()
        return FilenameParser.get_available_states(files)


class LazyDataLoader(DataLoader):
    """
    懒加载数据加载器

    支持懒加载和内部简单缓存。
    """

    def __init__(self, validate: bool = True, cache_enabled: bool = True):
        """
        初始化懒加载数据加载器

        Args:
            validate: 是否验证数据
            cache_enabled: 是否启用缓存
        """
        super().__init__(validate)
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, SignalData] = {}

        logger.info("LazyDataLoader 初始化完成")

    def load(
        self,
        drive_state: str,
        driven_state: str,
        torque: int,
        sensor: str,
        axis: str,
        speed: int = 1000,
        time_range: Optional[Tuple[float, float]] = None
    ) -> Optional[SignalData]:
        """
        懒加载数据（带缓存）

        Args:
            drive_state: 主动轮状态
            driven_state: 从动轮状态
            torque: 扭矩
            sensor: 传感器
            axis: 轴向
            speed: 转速
            time_range: 时间范围

        Returns:
            SignalData: 信号数据
        """
        # 生成缓存键
        cache_key = self._make_cache_key(
            drive_state, driven_state, torque, sensor, axis, speed, time_range
        )

        # 检查缓存
        if self.cache_enabled and cache_key in self._cache:
            logger.debug(f"从缓存获取数据: {cache_key}")
            return self._cache[cache_key]

        # 加载数据
        data = super().load(
            drive_state, driven_state, torque, sensor, axis, speed, time_range
        )

        # 缓存数据
        if self.cache_enabled and data is not None:
            self._cache[cache_key] = data
            logger.debug(f"缓存数据: {cache_key} ({len(self._cache)} 项)")

        return data

    @staticmethod
    def _make_cache_key(
        drive_state: str,
        driven_state: str,
        torque: int,
        sensor: str,
        axis: str,
        speed: int,
        time_range: Optional[Tuple[float, float]]
    ) -> str:
        """生成缓存键"""
        key = f"{drive_state}_{driven_state}_{torque}_{sensor}_{axis}_{speed}"

        if time_range:
            key += f"_{time_range[0]}_{time_range[1]}"

        return key

    def clear_cache(self):
        """清除缓存"""
        count = len(self._cache)
        self._cache.clear()
        logger.info(f"清除缓存: {count} 项")

    def get_cache_info(self) -> Dict:
        """
        获取缓存信息

        Returns:
            dict: 缓存统计信息
        """
        cache_size_bytes = 0
        for data in self._cache.values():
            cache_size_bytes += data.time_series.nbytes

        return {
            'cached_items': len(self._cache),
            'cache_size_mb': cache_size_bytes / 1024 / 1024,
            'cache_keys': list(self._cache.keys())
        }


class BatchDataLoader(LazyDataLoader):
    """
    批量数据加载器

    支持批量加载多个数据集。
    """

    def load_multiple(
        self,
        configs: List[GearConfig],
        time_range: Optional[Tuple[float, float]] = None
    ) -> List[SignalData]:
        """
        批量加载多个配置的数据

        Args:
            configs: 齿轮配置列表
            time_range: 时间范围

        Returns:
            List[SignalData]: 信号数据列表
        """
        results = []

        for config in configs:
            data = self.load_by_config(config, time_range)
            if data:
                results.append(data)

        logger.info(f"批量加载完成: {len(results)}/{len(configs)} 成功")
        return results

    def load_all_conditions(
        self,
        sensor: str,
        axis: str,
        torque: Optional[int] = None
    ) -> List[SignalData]:
        """
        加载所有磨损状态组合的数据

        Args:
            sensor: 传感器
            axis: 轴向
            torque: 扭矩（None表示所有扭矩）

        Returns:
            List[SignalData]: 所有数据
        """
        files = self.get_available_files()

        # 筛选
        if torque is not None:
            files = FilenameParser.filter_configs(files, torque=torque)

        # 为每个文件创建配置并加载
        results = []
        for file_config in files:
            gear_config = GearConfig(
                drive_gear_state=file_config.drive_gear_state,
                driven_gear_state=file_config.driven_gear_state,
                torque=file_config.torque,
                sensor=sensor,
                axis=axis
            )

            data = self.load_by_config(gear_config)
            if data:
                results.append(data)

        logger.info(
            f"加载所有条件数据: {sensor}_{axis} - {len(results)} 个数据集"
        )

        return results


# ====================================
# 便捷函数
# ====================================

def load_signal(
    drive_state: str,
    driven_state: str,
    torque: int,
    sensor: str,
    axis: str
) -> Optional[SignalData]:
    """
    便捷函数：加载单个信号

    Args:
        drive_state: 主动轮状态
        driven_state: 从动轮状态
        torque: 扭矩
        sensor: 传感器
        axis: 轴向

    Returns:
        SignalData: 信号数据
    """
    loader = DataLoader()
    return loader.load(drive_state, driven_state, torque, sensor, axis)


# ====================================
# 测试代码
# ====================================

if __name__ == "__main__":
    print("=" * 60)
    print("数据加载器测试")
    print("=" * 60)

    # 测试1: 基础数据加载
    print("\n【测试1】基础数据加载:")
    loader = DataLoader(validate=False)  # 跳过验证以加快测试

    print("   加载: light_wear-normal-10Nm, 传感器A-X轴")
    data = loader.load('light_wear', 'normal', 10, 'A', 'X')

    if data:
        print(f"   ✅ 加载成功!")
        print(f"   - 采样点数: {len(data)}")
        print(f"   - 采样率: {data.sampling_rate} Hz")
        print(f"   - 时长: {data.duration:.2f} 秒")
        print(f"   - 数据范围: [{data.time_series.min():.6f}, {data.time_series.max():.6f}]")
    else:
        print("   ❌ 加载失败")

    # 测试2: 懒加载和缓存
    print("\n【测试2】懒加载和缓存:")
    lazy_loader = LazyDataLoader(validate=False, cache_enabled=True)

    # 第一次加载
    print("   第一次加载...")
    data1 = lazy_loader.load('light_wear', 'normal', 10, 'A', 'Y')

    # 第二次加载（应该从缓存获取）
    print("   第二次加载（应该从缓存）...")
    data2 = lazy_loader.load('light_wear', 'normal', 10, 'A', 'Y')

    cache_info = lazy_loader.get_cache_info()
    print(f"   缓存项数: {cache_info['cached_items']}")
    print(f"   缓存大小: {cache_info['cache_size_mb']:.2f} MB")

    # 测试3: 时间范围加载
    print("\n【测试3】加载指定时间范围:")
    print("   加载前5秒数据...")
    data_partial = loader.load(
        'light_wear', 'normal', 10, 'A', 'Z',
        time_range=(0, 5.0)
    )

    if data_partial:
        print(f"   采样点数: {len(data_partial)}")
        print(f"   时长: {data_partial.duration:.2f} 秒")

    # 测试4: 批量加载
    print("\n【测试4】批量加载:")
    batch_loader = BatchDataLoader(validate=False)

    configs = [
        GearConfig('light_wear', 'normal', 10, 'A', 'X'),
        GearConfig('heavy_wear', 'normal', 10, 'A', 'X'),
        GearConfig('normal', 'heavy_wear', 10, 'A', 'X'),
    ]

    print(f"   批量加载 {len(configs)} 个配置...")
    datasets = batch_loader.load_multiple(configs)
    print(f"   成功加载: {len(datasets)} 个数据集")

    # 测试5: 加载所有条件
    print("\n【测试5】加载所有条件的数据:")
    print("   加载所有10Nm扭矩下，传感器B-Y轴的数据...")
    all_data = batch_loader.load_all_conditions('B', 'Y', torque=10)
    print(f"   共加载 {len(all_data)} 个数据集:")
    for d in all_data:
        print(f"   - {d.config}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
