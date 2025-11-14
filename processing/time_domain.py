"""
时域特征提取模块

提供齿轮振动信号的时域分析功能：
- 基础统计特征 (均值、标准差、RMS、峰值等)
- 形状特征 (偏度、峭度、波峰因子等)
- 能量特征 (总能量、功率等)
- 幅值特征
- 冲击特征检测

这些特征对齿轮磨损状态识别至关重要。
"""

import numpy as np
from scipy import stats
from scipy.signal import hilbert, find_peaks
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, asdict

from config.settings import (
    SignalData,
    IMPACT_THRESHOLD_MULTIPLIER,
    PEAK_DETECTION_DISTANCE,
    logger
)


@dataclass
class TimeDomainFeatures:
    """时域特征数据类"""
    # 基础统计特征
    mean: float
    std: float
    var: float
    rms: float
    peak: float
    peak_to_peak: float

    # 形状特征
    skewness: float
    kurtosis: float
    crest_factor: float
    clearance_factor: float
    impulse_factor: float
    shape_factor: float

    # 能量特征
    total_energy: float
    average_power: float
    peak_power: float

    # 幅值特征
    mean_amplitude: float
    median_amplitude: float
    amplitude_variance: float

    def to_dict(self) -> Dict:
        """转换为字典"""
        return asdict(self)


class TimeDomainAnalyzer:
    """
    时域分析器

    提供完整的时域特征提取功能。
    """

    def __init__(self):
        """初始化时域分析器"""
        logger.debug("TimeDomainAnalyzer 初始化完成")

    @staticmethod
    def extract_features(signal: np.ndarray) -> TimeDomainFeatures:
        """
        提取信号的所有时域特征

        Args:
            signal: 时域信号数组

        Returns:
            TimeDomainFeatures: 时域特征对象

        Examples:
            >>> analyzer = TimeDomainAnalyzer()
            >>> features = analyzer.extract_features(signal_data)
            >>> print(f"RMS: {features.rms}")
        """
        # 基础统计特征
        mean_val = np.mean(signal)
        std_val = np.std(signal)
        var_val = np.var(signal)
        rms_val = np.sqrt(np.mean(signal**2))
        peak_val = np.max(np.abs(signal))
        peak_to_peak_val = np.ptp(signal)

        # 形状特征
        skewness_val = stats.skew(signal)
        kurtosis_val = stats.kurtosis(signal)

        # 波峰因子 = 峰值 / RMS
        crest_factor_val = peak_val / rms_val if rms_val != 0 else 0

        # 裕度因子 = 峰值 / 方根幅值的平方
        mean_sqrt_abs = np.mean(np.sqrt(np.abs(signal)))
        clearance_factor_val = peak_val / (mean_sqrt_abs**2) if mean_sqrt_abs != 0 else 0

        # 脉冲因子 = 峰值 / 平均绝对值
        mean_abs = np.mean(np.abs(signal))
        impulse_factor_val = peak_val / mean_abs if mean_abs != 0 else 0

        # 形状因子 = RMS / 平均绝对值
        shape_factor_val = rms_val / mean_abs if mean_abs != 0 else 0

        # 能量特征
        total_energy_val = np.sum(signal**2)
        average_power_val = total_energy_val / len(signal)
        peak_power_val = np.max(signal**2)

        # 幅值特征
        abs_signal = np.abs(signal)
        mean_amplitude_val = np.mean(abs_signal)
        median_amplitude_val = np.median(abs_signal)
        amplitude_variance_val = np.var(abs_signal)

        features = TimeDomainFeatures(
            mean=float(mean_val),
            std=float(std_val),
            var=float(var_val),
            rms=float(rms_val),
            peak=float(peak_val),
            peak_to_peak=float(peak_to_peak_val),
            skewness=float(skewness_val),
            kurtosis=float(kurtosis_val),
            crest_factor=float(crest_factor_val),
            clearance_factor=float(clearance_factor_val),
            impulse_factor=float(impulse_factor_val),
            shape_factor=float(shape_factor_val),
            total_energy=float(total_energy_val),
            average_power=float(average_power_val),
            peak_power=float(peak_power_val),
            mean_amplitude=float(mean_amplitude_val),
            median_amplitude=float(median_amplitude_val),
            amplitude_variance=float(amplitude_variance_val)
        )

        logger.debug(f"时域特征提取完成: RMS={rms_val:.6f}, Peak={peak_val:.6f}")
        return features

    @staticmethod
    def basic_statistics(signal: np.ndarray) -> Dict[str, float]:
        """
        提取基础统计特征

        Args:
            signal: 时域信号

        Returns:
            dict: 基础统计特征字典
        """
        return {
            'mean': float(np.mean(signal)),
            'std': float(np.std(signal)),
            'var': float(np.var(signal)),
            'rms': float(np.sqrt(np.mean(signal**2))),
            'peak': float(np.max(np.abs(signal))),
            'peak_to_peak': float(np.ptp(signal)),
            'min_val': float(np.min(signal)),
            'max_val': float(np.max(signal))
        }

    @staticmethod
    def shape_features(signal: np.ndarray) -> Dict[str, float]:
        """
        提取形状特征

        Args:
            signal: 时域信号

        Returns:
            dict: 形状特征字典
        """
        rms = np.sqrt(np.mean(signal**2))
        peak = np.max(np.abs(signal))
        mean_abs = np.mean(np.abs(signal))
        mean_sqrt_abs = np.mean(np.sqrt(np.abs(signal)))

        return {
            'skewness': float(stats.skew(signal)),
            'kurtosis': float(stats.kurtosis(signal)),
            'crest_factor': float(peak / rms) if rms != 0 else 0,
            'clearance_factor': float(peak / (mean_sqrt_abs**2)) if mean_sqrt_abs != 0 else 0,
            'impulse_factor': float(peak / mean_abs) if mean_abs != 0 else 0,
            'shape_factor': float(rms / mean_abs) if mean_abs != 0 else 0
        }

    @staticmethod
    def energy_features(signal: np.ndarray, sampling_rate: int) -> Dict[str, float]:
        """
        提取能量特征

        Args:
            signal: 时域信号
            sampling_rate: 采样频率

        Returns:
            dict: 能量特征字典
        """
        energy = np.sum(signal**2)
        power = energy / len(signal)
        instantaneous_power = signal**2

        return {
            'total_energy': float(energy),
            'average_power': float(power),
            'peak_power': float(np.max(instantaneous_power)),
            'power_std': float(np.std(instantaneous_power)),
            'energy_per_second': float(energy * sampling_rate / len(signal))
        }


class ImpactDetector:
    """
    冲击信号检测器

    用于检测齿轮磨损产生的冲击信号。
    """

    def __init__(self, threshold_multiplier: float = IMPACT_THRESHOLD_MULTIPLIER):
        """
        初始化冲击检测器

        Args:
            threshold_multiplier: 阈值倍数（相对于信号标准差）
        """
        self.threshold_multiplier = threshold_multiplier
        logger.debug(f"ImpactDetector 初始化: 阈值倍数={threshold_multiplier}")

    def detect_impacts(
        self,
        signal: np.ndarray,
        sampling_rate: int
    ) -> Dict[str, any]:
        """
        检测信号中的冲击事件

        Args:
            signal: 输入信号
            sampling_rate: 采样频率

        Returns:
            dict: 冲击特征信息
        """
        # 计算包络
        envelope = self._compute_envelope(signal)

        # 动态阈值
        threshold = np.mean(envelope) + self.threshold_multiplier * np.std(envelope)

        # 寻找峰值
        min_distance = max(PEAK_DETECTION_DISTANCE, sampling_rate // 1000)  # 至少1ms间隔
        peaks, properties = find_peaks(
            envelope,
            height=threshold,
            distance=min_distance
        )

        # 分析冲击特征
        impact_features = self._analyze_impacts(signal, peaks, envelope, sampling_rate)

        logger.debug(f"检测到 {len(peaks)} 个冲击事件")
        return impact_features

    @staticmethod
    def _compute_envelope(signal: np.ndarray) -> np.ndarray:
        """
        计算信号包络（使用Hilbert变换）

        Args:
            signal: 输入信号

        Returns:
            np.ndarray: 包络信号
        """
        analytic_signal = hilbert(signal)
        envelope = np.abs(analytic_signal)
        return envelope

    def _analyze_impacts(
        self,
        signal: np.ndarray,
        peaks: np.ndarray,
        envelope: np.ndarray,
        sampling_rate: int
    ) -> Dict[str, any]:
        """
        分析冲击特征

        Args:
            signal: 原始信号
            peaks: 冲击峰值位置
            envelope: 包络信号
            sampling_rate: 采样频率

        Returns:
            dict: 冲击特征
        """
        if len(peaks) == 0:
            return {
                'impact_count': 0,
                'impact_rate': 0.0,
                'max_impact_amplitude': 0.0,
                'mean_impact_amplitude': 0.0,
                'impact_energy_ratio': 0.0,
                'impact_positions': [],
                'impact_intervals': []
            }

        # 冲击幅值
        impact_amplitudes = envelope[peaks]

        # 冲击间隔
        impact_intervals = []
        impact_rate = 0.0
        if len(peaks) > 1:
            impact_intervals = np.diff(peaks) / sampling_rate
            mean_interval = np.mean(impact_intervals)
            impact_rate = 1.0 / mean_interval if mean_interval > 0 else 0

        # 冲击能量占比
        total_energy = np.sum(signal**2)
        impact_energy = np.sum(impact_amplitudes**2)
        energy_ratio = impact_energy / total_energy if total_energy > 0 else 0

        return {
            'impact_count': int(len(peaks)),
            'impact_rate': float(impact_rate),
            'max_impact_amplitude': float(np.max(impact_amplitudes)),
            'mean_impact_amplitude': float(np.mean(impact_amplitudes)),
            'std_impact_amplitude': float(np.std(impact_amplitudes)),
            'impact_energy_ratio': float(energy_ratio),
            'impact_positions': peaks.tolist(),
            'impact_intervals': impact_intervals.tolist() if len(impact_intervals) > 0 else []
        }


class SegmentAnalyzer:
    """
    分段分析器

    将信号分段并提取每段的特征，用于分析时变特征。
    """

    def __init__(self, segment_length: float = 1.0):
        """
        初始化分段分析器

        Args:
            segment_length: 每段长度（秒）
        """
        self.segment_length = segment_length
        logger.debug(f"SegmentAnalyzer 初始化: 段长={segment_length}秒")

    def analyze_segments(
        self,
        signal: np.ndarray,
        sampling_rate: int
    ) -> List[TimeDomainFeatures]:
        """
        分段提取特征

        Args:
            signal: 输入信号
            sampling_rate: 采样频率

        Returns:
            list: 每段的时域特征列表
        """
        segment_samples = int(self.segment_length * sampling_rate)
        num_segments = len(signal) // segment_samples

        features_list = []
        analyzer = TimeDomainAnalyzer()

        for i in range(num_segments):
            start_idx = i * segment_samples
            end_idx = start_idx + segment_samples
            segment = signal[start_idx:end_idx]

            features = analyzer.extract_features(segment)
            features_list.append(features)

        logger.debug(f"分段分析完成: {num_segments}段")
        return features_list

    def compute_feature_trends(
        self,
        features_list: List[TimeDomainFeatures]
    ) -> Dict[str, Dict[str, float]]:
        """
        计算特征的变化趋势

        Args:
            features_list: 特征列表

        Returns:
            dict: 每个特征的统计信息（均值、标准差、趋势等）
        """
        if not features_list:
            return {}

        # 提取所有特征的时间序列
        feature_names = asdict(features_list[0]).keys()
        trends = {}

        for feature_name in feature_names:
            values = [getattr(f, feature_name) for f in features_list]

            trends[feature_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'trend': float(np.polyfit(range(len(values)), values, 1)[0])  # 线性趋势
            }

        return trends


# ====================================
# 便捷函数
# ====================================

def quick_features(signal: np.ndarray) -> TimeDomainFeatures:
    """
    快速提取时域特征（便捷函数）

    Args:
        signal: 信号数组

    Returns:
        TimeDomainFeatures: 时域特征
    """
    analyzer = TimeDomainAnalyzer()
    return analyzer.extract_features(signal)


def detect_impacts(signal: np.ndarray, sampling_rate: int) -> Dict:
    """
    快速检测冲击（便捷函数）

    Args:
        signal: 信号数组
        sampling_rate: 采样频率

    Returns:
        dict: 冲击特征
    """
    detector = ImpactDetector()
    return detector.detect_impacts(signal, sampling_rate)


# ====================================
# 测试代码
# ====================================

if __name__ == "__main__":
    print("=" * 60)
    print("时域特征提取测试")
    print("=" * 60)

    # 加载测试数据
    from data.loader import DataLoader

    loader = DataLoader(validate=False)

    print("\n加载测试数据...")
    signal_data = loader.load('light_wear', 'normal', 10, 'A', 'X')

    if signal_data is None:
        print("❌ 无法加载数据")
        exit(1)

    signal = signal_data.time_series
    sampling_rate = signal_data.sampling_rate

    print(f"✅ 数据加载成功: {len(signal)} 采样点, {sampling_rate} Hz")

    # 测试1: 提取时域特征
    print("\n【测试1】提取时域特征:")
    analyzer = TimeDomainAnalyzer()
    features = analyzer.extract_features(signal)

    print(f"\n基础统计特征:")
    print(f"   均值 (Mean):      {features.mean:.6f}")
    print(f"   标准差 (Std):     {features.std:.6f}")
    print(f"   RMS:              {features.rms:.6f}")
    print(f"   峰值 (Peak):      {features.peak:.6f}")
    print(f"   峰峰值 (P-P):     {features.peak_to_peak:.6f}")

    print(f"\n形状特征:")
    print(f"   偏度 (Skewness):  {features.skewness:.6f}")
    print(f"   峭度 (Kurtosis):  {features.kurtosis:.6f}")
    print(f"   波峰因子:         {features.crest_factor:.6f}")
    print(f"   脉冲因子:         {features.impulse_factor:.6f}")

    print(f"\n能量特征:")
    print(f"   总能量:           {features.total_energy:.2e}")
    print(f"   平均功率:         {features.average_power:.6f}")

    # 测试2: 冲击检测
    print("\n【测试2】冲击检测:")
    detector = ImpactDetector(threshold_multiplier=3.0)
    impact_features = detector.detect_impacts(signal, sampling_rate)

    print(f"   冲击数量:         {impact_features['impact_count']}")
    print(f"   冲击频率:         {impact_features['impact_rate']:.2f} Hz")
    print(f"   最大冲击幅值:     {impact_features['max_impact_amplitude']:.6f}")
    print(f"   平均冲击幅值:     {impact_features['mean_impact_amplitude']:.6f}")
    print(f"   冲击能量占比:     {impact_features['impact_energy_ratio']*100:.2f}%")

    # 测试3: 分段分析
    print("\n【测试3】分段分析 (每段1秒):")
    segment_analyzer = SegmentAnalyzer(segment_length=1.0)

    # 只分析前5秒
    signal_5s = signal[:sampling_rate * 5]
    segments_features = segment_analyzer.analyze_segments(signal_5s, sampling_rate)

    print(f"   分段数量: {len(segments_features)}")
    print(f"\n   各段RMS值:")
    for i, seg_feat in enumerate(segments_features):
        print(f"   段 {i+1}: RMS = {seg_feat.rms:.6f}")

    # 测试4: 特征趋势分析
    print("\n【测试4】特征趋势分析:")
    trends = segment_analyzer.compute_feature_trends(segments_features)

    print(f"   RMS 趋势:")
    print(f"   - 均值: {trends['rms']['mean']:.6f}")
    print(f"   - 标准差: {trends['rms']['std']:.6f}")
    print(f"   - 变化趋势: {trends['rms']['trend']:.6e}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
