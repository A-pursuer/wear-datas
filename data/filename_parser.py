"""
文件名解析器

从CSV文件名中解析齿轮磨损实验的配置信息，包括：
- 主动轮磨损状态
- 从动轮磨损状态
- 扭矩值
- 转速

文件名格式: 主动轮(磨损状态)-从动轮(磨损状态)-扭矩值-转速.csv
示例: 主动轮(轻磨)-从动轮(正常)-10Nm-1000r.csv
"""

import re
import os
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import asdict

from config.settings import (
    FileConfig,
    WEAR_STATE_MAP,
    DATA_DIR,
    CSV_PATTERN,
    logger
)


class FilenameParser:
    """
    文件名解析器

    负责解析齿轮数据文件名，提取实验配置信息。
    """

    # 文件名正则表达式
    # 匹配格式: 主动轮(状态)-从动轮(状态)-扭矩Nm-转速r.csv
    PATTERN = re.compile(
        r'主动轮\((.+?)\)-从动轮\((.+?)\)-(\d+)Nm-(\d+)r\.csv'
    )

    def __init__(self):
        """初始化文件名解析器"""
        self.cache: Dict[str, FileConfig] = {}
        logger.debug("FilenameParser 初始化完成")

    @classmethod
    def parse(cls, filepath: str) -> Optional[FileConfig]:
        """
        从文件路径解析配置信息

        Args:
            filepath: CSV文件路径（绝对路径或相对路径）

        Returns:
            FileConfig: 解析后的配置对象，解析失败返回None

        Examples:
            >>> parser = FilenameParser()
            >>> config = parser.parse('主动轮(轻磨)-从动轮(正常)-10Nm-1000r.csv')
            >>> print(config.drive_gear_state)
            'light_wear'
        """
        # 获取文件名（不含路径）
        filename = os.path.basename(filepath)

        # 正则匹配
        match = cls.PATTERN.match(filename)

        if not match:
            logger.warning(f"文件名格式不正确: {filename}")
            return None

        # 提取匹配组
        drive_state, driven_state, torque, speed = match.groups()

        # 转换磨损状态为英文标识
        drive_state_en = WEAR_STATE_MAP.get(drive_state, drive_state)
        driven_state_en = WEAR_STATE_MAP.get(driven_state, driven_state)

        # 创建FileConfig对象
        config = FileConfig(
            drive_gear_state=drive_state_en,
            driven_gear_state=driven_state_en,
            torque=int(torque),
            speed=int(speed),
            filepath=str(Path(filepath).resolve())  # 转换为绝对路径
        )

        logger.debug(f"解析成功: {filename} -> {config}")
        return config

    def parse_with_cache(self, filepath: str) -> Optional[FileConfig]:
        """
        带缓存的解析（避免重复解析同一文件）

        Args:
            filepath: CSV文件路径

        Returns:
            FileConfig: 解析后的配置对象
        """
        # 检查缓存
        if filepath in self.cache:
            logger.debug(f"从缓存获取: {filepath}")
            return self.cache[filepath]

        # 解析并缓存
        config = self.parse(filepath)
        if config:
            self.cache[filepath] = config

        return config

    @staticmethod
    def scan_directory(directory: str = None) -> List[FileConfig]:
        """
        扫描目录获取所有可用的配置组合

        Args:
            directory: 要扫描的目录路径，默认为DATA_DIR

        Returns:
            List[FileConfig]: 所有找到的配置列表

        Examples:
            >>> configs = FilenameParser.scan_directory()
            >>> print(f"找到 {len(configs)} 个数据文件")
        """
        if directory is None:
            directory = DATA_DIR

        directory_path = Path(directory)

        if not directory_path.exists():
            logger.error(f"目录不存在: {directory}")
            return []

        configs = []
        csv_files = list(directory_path.glob(CSV_PATTERN))

        logger.info(f"在 {directory} 中找到 {len(csv_files)} 个CSV文件")

        for csv_file in csv_files:
            config = FilenameParser.parse(str(csv_file))
            if config:
                configs.append(config)
            else:
                logger.warning(f"跳过无效文件: {csv_file.name}")

        logger.info(f"成功解析 {len(configs)} 个配置")
        return configs

    @staticmethod
    def filter_configs(
        configs: List[FileConfig],
        drive_state: str = None,
        driven_state: str = None,
        torque: int = None,
        speed: int = None
    ) -> List[FileConfig]:
        """
        根据条件筛选配置

        Args:
            configs: 配置列表
            drive_state: 主动轮状态筛选条件
            driven_state: 从动轮状态筛选条件
            torque: 扭矩筛选条件
            speed: 转速筛选条件

        Returns:
            List[FileConfig]: 筛选后的配置列表

        Examples:
            >>> all_configs = FilenameParser.scan_directory()
            >>> # 只获取10Nm扭矩的数据
            >>> configs_10nm = FilenameParser.filter_configs(all_configs, torque=10)
        """
        filtered = configs

        if drive_state:
            filtered = [c for c in filtered if c.drive_gear_state == drive_state]

        if driven_state:
            filtered = [c for c in filtered if c.driven_gear_state == driven_state]

        if torque is not None:
            filtered = [c for c in filtered if c.torque == torque]

        if speed is not None:
            filtered = [c for c in filtered if c.speed == speed]

        logger.debug(f"筛选: {len(configs)} -> {len(filtered)} 个配置")
        return filtered

    @staticmethod
    def get_available_states(configs: List[FileConfig]) -> Dict[str, List]:
        """
        获取可用的磨损状态组合

        Args:
            configs: 配置列表

        Returns:
            dict: 包含可用的主动轮状态、从动轮状态、扭矩和转速

        Examples:
            >>> configs = FilenameParser.scan_directory()
            >>> states = FilenameParser.get_available_states(configs)
            >>> print(states['drive_states'])
            ['normal', 'light_wear', 'heavy_wear']
        """
        drive_states = sorted(list(set(c.drive_gear_state for c in configs)))
        driven_states = sorted(list(set(c.driven_gear_state for c in configs)))
        torques = sorted(list(set(c.torque for c in configs)))
        speeds = sorted(list(set(c.speed for c in configs)))

        return {
            'drive_states': drive_states,
            'driven_states': driven_states,
            'torques': torques,
            'speeds': speeds
        }

    @staticmethod
    def validate_filename(filename: str) -> bool:
        """
        验证文件名格式是否正确

        Args:
            filename: 文件名

        Returns:
            bool: 文件名格式是否正确
        """
        return FilenameParser.PATTERN.match(filename) is not None

    def clear_cache(self):
        """清除解析缓存"""
        self.cache.clear()
        logger.debug("解析缓存已清除")


# ====================================
# 便捷函数
# ====================================

def parse_filename(filepath: str) -> Optional[FileConfig]:
    """
    便捷函数：解析单个文件名

    Args:
        filepath: 文件路径

    Returns:
        FileConfig: 配置对象
    """
    return FilenameParser.parse(filepath)


def scan_data_files(directory: str = None) -> List[FileConfig]:
    """
    便捷函数：扫描数据目录

    Args:
        directory: 目录路径

    Returns:
        List[FileConfig]: 配置列表
    """
    return FilenameParser.scan_directory(directory)


def get_file_by_config(
    drive_state: str,
    driven_state: str,
    torque: int,
    speed: int = 1000
) -> Optional[FileConfig]:
    """
    便捷函数：根据配置查找文件

    Args:
        drive_state: 主动轮状态
        driven_state: 从动轮状态
        torque: 扭矩
        speed: 转速（默认1000）

    Returns:
        FileConfig: 找到的第一个匹配配置，未找到返回None
    """
    configs = FilenameParser.scan_directory()
    filtered = FilenameParser.filter_configs(
        configs,
        drive_state=drive_state,
        driven_state=driven_state,
        torque=torque,
        speed=speed
    )

    return filtered[0] if filtered else None


# ====================================
# 测试代码
# ====================================

if __name__ == "__main__":
    # 测试文件名解析
    print("=" * 60)
    print("文件名解析器测试")
    print("=" * 60)

    # 测试1: 解析单个文件名
    print("\n【测试1】解析单个文件名:")
    test_filename = "主动轮(轻磨)-从动轮(正常)-10Nm-1000r.csv"
    config = parse_filename(test_filename)

    if config:
        print(f"✅ 解析成功!")
        print(f"   主动轮状态: {config.drive_gear_state}")
        print(f"   从动轮状态: {config.driven_gear_state}")
        print(f"   扭矩: {config.torque} Nm")
        print(f"   转速: {config.speed} rpm")
    else:
        print(f"❌ 解析失败")

    # 测试2: 扫描目录
    print("\n【测试2】扫描数据目录:")
    configs = scan_data_files()
    print(f"找到 {len(configs)} 个数据文件:")
    for i, cfg in enumerate(configs, 1):
        print(f"   {i}. {cfg}")

    # 测试3: 获取可用状态
    print("\n【测试3】可用的实验配置:")
    if configs:
        states = FilenameParser.get_available_states(configs)
        print(f"   主动轮状态: {states['drive_states']}")
        print(f"   从动轮状态: {states['driven_states']}")
        print(f"   扭矩值: {states['torques']} Nm")
        print(f"   转速: {states['speeds']} rpm")

    # 测试4: 筛选配置
    print("\n【测试4】筛选特定配置 (扭矩=10Nm):")
    filtered = FilenameParser.filter_configs(configs, torque=10)
    print(f"找到 {len(filtered)} 个匹配的配置:")
    for cfg in filtered:
        print(f"   - {cfg}")

    # 测试5: 根据配置查找文件
    print("\n【测试5】查找特定配置的文件:")
    config = get_file_by_config('light_wear', 'normal', 10)
    if config:
        print(f"✅ 找到文件: {config.filepath}")
    else:
        print(f"❌ 未找到匹配的文件")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
