"""
数据验证器

负责验证CSV数据文件的：
- 文件完整性
- 数据格式正确性
- 数值范围合理性
- NaN和异常值检测
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from config.settings import (
    EXPECTED_DATA_POINTS,
    SAMPLING_RATE,
    CHANNEL_COUNT,
    CHANNEL_MAP,
    SENSORS,
    AXES,
    logger
)


@dataclass
class ValidationResult:
    """验证结果数据类"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: Dict = field(default_factory=dict)

    def add_error(self, message: str):
        """添加错误信息"""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str):
        """添加警告信息"""
        self.warnings.append(message)

    def __str__(self) -> str:
        status = "✅ 通过" if self.is_valid else "❌ 失败"
        msg = [f"验证结果: {status}"]

        if self.errors:
            msg.append(f"\n错误 ({len(self.errors)}):")
            for err in self.errors:
                msg.append(f"  - {err}")

        if self.warnings:
            msg.append(f"\n警告 ({len(self.warnings)}):")
            for warn in self.warnings:
                msg.append(f"  - {warn}")

        return "\n".join(msg)


class DataValidator:
    """
    数据验证器

    提供多层次的数据验证功能。
    """

    def __init__(self):
        """初始化验证器"""
        logger.debug("DataValidator 初始化完成")

    @staticmethod
    def validate_file_exists(filepath: str) -> ValidationResult:
        """
        验证文件是否存在

        Args:
            filepath: 文件路径

        Returns:
            ValidationResult: 验证结果
        """
        result = ValidationResult(is_valid=True)
        path = Path(filepath)

        if not path.exists():
            result.add_error(f"文件不存在: {filepath}")
        elif not path.is_file():
            result.add_error(f"不是文件: {filepath}")
        elif path.suffix.lower() != '.csv':
            result.add_warning(f"文件扩展名不是.csv: {path.suffix}")

        result.info['filepath'] = str(path.resolve())
        result.info['exists'] = path.exists()

        return result

    @staticmethod
    def validate_file_size(filepath: str) -> ValidationResult:
        """
        验证文件大小

        Args:
            filepath: 文件路径

        Returns:
            ValidationResult: 验证结果
        """
        result = ValidationResult(is_valid=True)
        path = Path(filepath)

        if not path.exists():
            result.add_error("文件不存在")
            return result

        file_size = path.stat().st_size
        file_size_mb = file_size / 1024 / 1024

        # 预期文件大小约 47-49 MB
        if file_size_mb < 1:
            result.add_error(f"文件过小 ({file_size_mb:.2f} MB)，可能数据不完整")
        elif file_size_mb < 40:
            result.add_warning(f"文件偏小 ({file_size_mb:.2f} MB)，请检查数据完整性")
        elif file_size_mb > 60:
            result.add_warning(f"文件偏大 ({file_size_mb:.2f} MB)，可能包含额外数据")

        result.info['file_size_bytes'] = file_size
        result.info['file_size_mb'] = file_size_mb

        logger.debug(f"文件大小: {file_size_mb:.2f} MB")
        return result

    @staticmethod
    def validate_csv_structure(filepath: str) -> ValidationResult:
        """
        验证CSV文件结构

        Args:
            filepath: 文件路径

        Returns:
            ValidationResult: 验证结果
        """
        result = ValidationResult(is_valid=True)

        try:
            # 导入pandas进行验证（如果可用）
            import pandas as pd

            # 读取第一行检查元数据
            metadata = pd.read_csv(filepath, nrows=1, header=None)

            if metadata.shape[1] < 3:
                result.add_error(f"元数据列数不足: {metadata.shape[1]} < 3")

            result.info['metadata_columns'] = metadata.shape[1]

            # 读取数据行检查列数
            data_sample = pd.read_csv(
                filepath,
                skiprows=4,
                nrows=10,
                header=None
            )

            if data_sample.shape[1] != CHANNEL_COUNT:
                result.add_error(
                    f"数据列数不匹配: 期望{CHANNEL_COUNT}，实际{data_sample.shape[1]}"
                )

            result.info['data_columns'] = data_sample.shape[1]

            # 统计总行数
            with open(filepath, 'r') as f:
                line_count = sum(1 for line in f)

            # 数据行数 = 总行数 - 4（元数据+配置）
            data_rows = line_count - 4

            if data_rows != EXPECTED_DATA_POINTS:
                result.add_warning(
                    f"数据行数不匹配: 期望{EXPECTED_DATA_POINTS}，实际{data_rows}"
                )

            result.info['total_lines'] = line_count
            result.info['data_rows'] = data_rows

            logger.debug(f"CSV结构: {line_count}行, {data_sample.shape[1]}列")

        except ImportError:
            result.add_warning("pandas未安装，跳过CSV结构验证")
        except Exception as e:
            result.add_error(f"验证CSV结构时出错: {str(e)}")

        return result

    @staticmethod
    def validate_data_range(
        data: np.ndarray,
        sensor: str = None,
        axis: str = None
    ) -> ValidationResult:
        """
        验证数据数值范围的合理性

        Args:
            data: 数据数组
            sensor: 传感器标识（用于日志）
            axis: 轴向标识（用于日志）

        Returns:
            ValidationResult: 验证结果
        """
        result = ValidationResult(is_valid=True)

        if data is None or len(data) == 0:
            result.add_error("数据为空")
            return result

        # 基本统计
        data_min = np.min(data)
        data_max = np.max(data)
        data_mean = np.mean(data)
        data_std = np.std(data)

        result.info['min'] = float(data_min)
        result.info['max'] = float(data_max)
        result.info['mean'] = float(data_mean)
        result.info['std'] = float(data_std)
        result.info['range'] = float(data_max - data_min)

        # 检查NaN值
        nan_count = np.isnan(data).sum()
        if nan_count > 0:
            result.add_warning(f"包含 {nan_count} 个NaN值 ({nan_count/len(data)*100:.2f}%)")
            result.info['nan_count'] = int(nan_count)

        # 检查Inf值
        inf_count = np.isinf(data).sum()
        if inf_count > 0:
            result.add_error(f"包含 {inf_count} 个Inf值")
            result.info['inf_count'] = int(inf_count)

        # 检查异常大的值（加速度通常不会超过1000）
        if np.abs(data_max) > 1000 or np.abs(data_min) > 1000:
            sensor_info = f"{sensor}_{axis}" if sensor and axis else "未知"
            result.add_warning(
                f"传感器{sensor_info}数据异常大: "
                f"范围[{data_min:.2f}, {data_max:.2f}]"
            )

        # 检查数据是否全为常数
        if data_std < 1e-10:
            result.add_warning("数据标准差过小，可能为常数")

        logger.debug(f"数据范围: [{data_min:.6f}, {data_max:.6f}], "
                    f"均值: {data_mean:.6f}, 标准差: {data_std:.6f}")

        return result

    @staticmethod
    def validate_sampling_consistency(
        data: np.ndarray,
        expected_rate: int = SAMPLING_RATE
    ) -> ValidationResult:
        """
        验证采样的一致性

        Args:
            data: 数据数组
            expected_rate: 期望的采样率

        Returns:
            ValidationResult: 验证结果
        """
        result = ValidationResult(is_valid=True)

        # 检查数据点数是否符合预期时长
        expected_points = expected_rate * 30  # 30秒
        actual_points = len(data)

        if abs(actual_points - expected_points) > 100:
            result.add_warning(
                f"数据点数不匹配: 期望{expected_points}，实际{actual_points}"
            )

        result.info['expected_points'] = expected_points
        result.info['actual_points'] = actual_points
        result.info['duration'] = actual_points / expected_rate

        return result

    @staticmethod
    def detect_outliers(
        data: np.ndarray,
        method: str = 'iqr',
        threshold: float = 3.0
    ) -> Tuple[np.ndarray, int]:
        """
        检测异常值

        Args:
            data: 数据数组
            method: 检测方法 ('iqr', 'zscore')
            threshold: 阈值倍数

        Returns:
            tuple: (异常值索引, 异常值数量)
        """
        if method == 'iqr':
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            outliers = (data < lower_bound) | (data > upper_bound)

        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(data))
            outliers = z_scores > threshold

        else:
            raise ValueError(f"未知的异常值检测方法: {method}")

        outlier_count = np.sum(outliers)
        outlier_indices = np.where(outliers)[0]

        logger.debug(f"异常值检测 ({method}): 发现 {outlier_count} 个异常值")

        return outlier_indices, outlier_count

    @classmethod
    def validate_file_complete(cls, filepath: str) -> ValidationResult:
        """
        完整的文件验证（组合所有验证）

        Args:
            filepath: 文件路径

        Returns:
            ValidationResult: 综合验证结果
        """
        result = ValidationResult(is_valid=True)

        # 1. 文件存在性
        file_result = cls.validate_file_exists(filepath)
        result.errors.extend(file_result.errors)
        result.warnings.extend(file_result.warnings)
        result.info.update(file_result.info)

        if not file_result.is_valid:
            result.is_valid = False
            return result

        # 2. 文件大小
        size_result = cls.validate_file_size(filepath)
        result.warnings.extend(size_result.warnings)
        result.info.update(size_result.info)

        # 3. CSV结构
        structure_result = cls.validate_csv_structure(filepath)
        result.errors.extend(structure_result.errors)
        result.warnings.extend(structure_result.warnings)
        result.info.update(structure_result.info)

        if structure_result.errors:
            result.is_valid = False

        logger.info(f"文件验证完成: {Path(filepath).name} - "
                   f"{'通过' if result.is_valid else '失败'}")

        return result


# ====================================
# 便捷函数
# ====================================

def quick_validate(filepath: str) -> bool:
    """
    快速验证文件（便捷函数）

    Args:
        filepath: 文件路径

    Returns:
        bool: 是否有效
    """
    result = DataValidator.validate_file_complete(filepath)
    return result.is_valid


def validate_all_files(directory: str = None) -> Dict[str, ValidationResult]:
    """
    验证目录中的所有文件

    Args:
        directory: 目录路径

    Returns:
        dict: 文件路径 -> 验证结果的字典
    """
    from data.filename_parser import scan_data_files

    configs = scan_data_files(directory)
    results = {}

    for config in configs:
        result = DataValidator.validate_file_complete(config.filepath)
        results[config.filepath] = result

    return results


# ====================================
# 测试代码
# ====================================

if __name__ == "__main__":
    print("=" * 60)
    print("数据验证器测试")
    print("=" * 60)

    # 获取测试文件
    from data.filename_parser import scan_data_files

    configs = scan_data_files()

    if not configs:
        print("❌ 未找到数据文件")
        exit(1)

    test_file = configs[0].filepath
    print(f"\n使用测试文件: {Path(test_file).name}")

    # 测试1: 文件存在性验证
    print("\n【测试1】文件存在性验证:")
    result = DataValidator.validate_file_exists(test_file)
    print(result)

    # 测试2: 文件大小验证
    print("\n【测试2】文件大小验证:")
    result = DataValidator.validate_file_size(test_file)
    print(result)
    print(f"   文件大小: {result.info['file_size_mb']:.2f} MB")

    # 测试3: CSV结构验证
    print("\n【测试3】CSV结构验证:")
    result = DataValidator.validate_csv_structure(test_file)
    print(result)
    if 'data_rows' in result.info:
        print(f"   数据行数: {result.info['data_rows']}")

    # 测试4: 完整验证
    print("\n【测试4】完整文件验证:")
    result = DataValidator.validate_file_complete(test_file)
    print(result)

    # 测试5: 验证所有文件
    print("\n【测试5】验证所有数据文件:")
    all_results = validate_all_files()
    print(f"共验证 {len(all_results)} 个文件:")

    valid_count = sum(1 for r in all_results.values() if r.is_valid)
    print(f"   有效: {valid_count}")
    print(f"   无效: {len(all_results) - valid_count}")

    for filepath, result in all_results.items():
        status = "✅" if result.is_valid else "❌"
        print(f"   {status} {Path(filepath).name}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
