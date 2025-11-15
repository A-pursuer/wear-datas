"""
频域分析器

实现频域信号处理和特征提取：
- FFT频谱分析
- 功率谱密度（PSD）
- 频域特征提取
- 峰值检测
- 频段分析

使用示例:
    >>> from data.loader import DataLoader
    >>> loader = DataLoader()
    >>> signal_data = loader.load('light_wear', 'normal', 10, 'A', 'X')
    >>>
    >>> analyzer = FrequencyAnalyzer()
    >>> spectrum = analyzer.compute_fft(signal_data.time_series, signal_data.sampling_rate)
    >>> features = analyzer.extract_features(signal_data.time_series, signal_data.sampling_rate)
    >>> print(f"主频率: {features.dominant_frequency:.2f} Hz")
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, welch

from config.settings import (
    SAMPLING_RATE,
    logger
)


@dataclass
class FrequencyFeatures:
    """
    频域特征数据类

    包含频谱分析的各种特征指标。
    """
    # 主要频率特征
    dominant_frequency: float  # 主频率 (Hz)
    dominant_amplitude: float  # 主频率幅值

    # 频谱统计特征
    mean_frequency: float  # 平均频率
    median_frequency: float  # 中位频率
    frequency_variance: float  # 频率方差
    frequency_std: float  # 频率标准差

    # 频谱形状特征
    spectral_centroid: float  # 频谱质心
    spectral_spread: float  # 频谱扩展度
    spectral_skewness: float  # 频谱偏度
    spectral_kurtosis: float  # 频谱峰度
    spectral_rolloff: float  # 频谱滚降点
    spectral_flatness: float  # 频谱平坦度

    # 能量特征
    total_power: float  # 总功率
    band_power: Dict[str, float] = field(default_factory=dict)  # 各频段功率

    # 峰值特征
    peak_count: int = 0  # 峰值数量
    peak_frequencies: List[float] = field(default_factory=list)  # 峰值频率
    peak_amplitudes: List[float] = field(default_factory=list)  # 峰值幅值

    # 谐波特征
    harmonic_ratio: float = 0.0  # 谐波比
    thd: float = 0.0  # 总谐波失真

    def __str__(self) -> str:
        return (
            f"FrequencyFeatures(\n"
            f"  主频率: {self.dominant_frequency:.2f} Hz @ {self.dominant_amplitude:.6f}\n"
            f"  频谱质心: {self.spectral_centroid:.2f} Hz\n"
            f"  总功率: {self.total_power:.6e}\n"
            f"  峰值数量: {self.peak_count}\n"
            f")"
        )


class FrequencyAnalyzer:
    """
    频域分析器

    提供完整的频域分析功能。
    """

    def __init__(
        self,
        window: str = 'hann',
        nperseg: Optional[int] = None,
        noverlap: Optional[int] = None
    ):
        """
        初始化频域分析器

        Args:
            window: 窗函数类型 ('hann', 'hamming', 'blackman', etc.)
            nperseg: Welch方法的段长度（默认为采样率的1/8）
            noverlap: Welch方法的重叠长度（默认为nperseg的50%）
        """
        self.window = window
        self.nperseg = nperseg
        self.noverlap = noverlap

        logger.debug(f"FrequencyAnalyzer 初始化: window={window}")

    @staticmethod
    def compute_fft(
        signal_data: np.ndarray,
        sampling_rate: int,
        normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算FFT频谱

        Args:
            signal_data: 时域信号
            sampling_rate: 采样率
            normalize: 是否归一化

        Returns:
            tuple: (频率数组, 幅值谱)
        """
        N = len(signal_data)

        # 计算FFT
        fft_values = fft(signal_data)

        # 只取正频率部分
        freq = fftfreq(N, 1.0 / sampling_rate)
        positive_freq_idx = freq >= 0
        freq = freq[positive_freq_idx]
        fft_values = fft_values[positive_freq_idx]

        # 计算幅值谱
        amplitude = np.abs(fft_values)

        # 归一化
        if normalize:
            amplitude = amplitude / N  # 除以点数
            amplitude[1:-1] = 2 * amplitude[1:-1]  # 正频率部分乘以2（除了DC和Nyquist）

        logger.debug(f"FFT计算完成: {len(freq)} 个频率点, 最大频率 {freq[-1]:.2f} Hz")

        return freq, amplitude

    def compute_psd(
        self,
        signal_data: np.ndarray,
        sampling_rate: int,
        method: str = 'welch'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算功率谱密度（PSD）

        Args:
            signal_data: 时域信号
            sampling_rate: 采样率
            method: 方法 ('welch' 或 'periodogram')

        Returns:
            tuple: (频率数组, PSD)
        """
        # 设置参数
        nperseg = self.nperseg if self.nperseg else sampling_rate // 8
        noverlap = self.noverlap if self.noverlap else nperseg // 2

        if method == 'welch':
            freq, psd = welch(
                signal_data,
                fs=sampling_rate,
                window=self.window,
                nperseg=nperseg,
                noverlap=noverlap,
                scaling='density'
            )
        elif method == 'periodogram':
            freq, psd = signal.periodogram(
                signal_data,
                fs=sampling_rate,
                window=self.window,
                scaling='density'
            )
        else:
            raise ValueError(f"未知的PSD方法: {method}")

        logger.debug(f"PSD计算完成 ({method}): {len(freq)} 个频率点")

        return freq, psd

    def extract_features(
        self,
        signal_data: np.ndarray,
        sampling_rate: int,
        freq_bands: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> FrequencyFeatures:
        """
        提取频域特征

        Args:
            signal_data: 时域信号
            sampling_rate: 采样率
            freq_bands: 频段定义，例如 {'low': (0, 100), 'mid': (100, 1000)}

        Returns:
            FrequencyFeatures: 频域特征对象
        """
        # 计算FFT
        freq, amplitude = self.compute_fft(signal_data, sampling_rate, normalize=True)

        # 计算PSD
        psd_freq, psd = self.compute_psd(signal_data, sampling_rate, method='welch')

        # 主频率和幅值
        dominant_idx = np.argmax(amplitude)
        dominant_frequency = freq[dominant_idx]
        dominant_amplitude = amplitude[dominant_idx]

        # 频谱统计特征
        power = amplitude ** 2  # 功率
        total_power = np.sum(power)

        # 归一化功率（作为概率分布）
        power_norm = power / total_power if total_power > 0 else power

        mean_frequency = np.sum(freq * power_norm)
        median_frequency = self._compute_median_frequency(freq, power)
        frequency_variance = np.sum(((freq - mean_frequency) ** 2) * power_norm)
        frequency_std = np.sqrt(frequency_variance)

        # 频谱形状特征
        spectral_centroid = mean_frequency
        spectral_spread = frequency_std
        spectral_skewness = np.sum(((freq - mean_frequency) ** 3) * power_norm) / (frequency_std ** 3) if frequency_std > 0 else 0
        spectral_kurtosis = np.sum(((freq - mean_frequency) ** 4) * power_norm) / (frequency_std ** 4) if frequency_std > 0 else 0
        spectral_rolloff = self._compute_rolloff(freq, power, rolloff_ratio=0.85)
        spectral_flatness = self._compute_flatness(power)

        # 峰值检测
        peaks = self._find_spectral_peaks(freq, amplitude, sampling_rate)

        # 频段功率分析
        if freq_bands is None:
            freq_bands = self._get_default_frequency_bands()

        band_power = self._compute_band_power(psd_freq, psd, freq_bands)

        # 谐波分析
        harmonic_ratio = self._compute_harmonic_ratio(freq, amplitude, dominant_frequency)
        thd = self._compute_thd(freq, amplitude, dominant_frequency)

        # 构建特征对象
        features = FrequencyFeatures(
            dominant_frequency=dominant_frequency,
            dominant_amplitude=dominant_amplitude,
            mean_frequency=mean_frequency,
            median_frequency=median_frequency,
            frequency_variance=frequency_variance,
            frequency_std=frequency_std,
            spectral_centroid=spectral_centroid,
            spectral_spread=spectral_spread,
            spectral_skewness=spectral_skewness,
            spectral_kurtosis=spectral_kurtosis,
            spectral_rolloff=spectral_rolloff,
            spectral_flatness=spectral_flatness,
            total_power=total_power,
            band_power=band_power,
            peak_count=peaks['count'],
            peak_frequencies=peaks['frequencies'],
            peak_amplitudes=peaks['amplitudes'],
            harmonic_ratio=harmonic_ratio,
            thd=thd
        )

        logger.info(f"频域特征提取完成: 主频率={dominant_frequency:.2f}Hz, 峰值数={peaks['count']}")

        return features

    @staticmethod
    def _compute_median_frequency(freq: np.ndarray, power: np.ndarray) -> float:
        """计算中位频率"""
        cumsum_power = np.cumsum(power)
        total_power = cumsum_power[-1]

        # 找到累积功率达到50%的频率
        median_idx = np.argmin(np.abs(cumsum_power - total_power * 0.5))
        return freq[median_idx]

    @staticmethod
    def _compute_rolloff(freq: np.ndarray, power: np.ndarray, rolloff_ratio: float = 0.85) -> float:
        """计算频谱滚降点"""
        cumsum_power = np.cumsum(power)
        total_power = cumsum_power[-1]

        rolloff_idx = np.argmin(np.abs(cumsum_power - total_power * rolloff_ratio))
        return freq[rolloff_idx]

    @staticmethod
    def _compute_flatness(power: np.ndarray) -> float:
        """
        计算频谱平坦度（Spectral Flatness）

        几何平均 / 算术平均，值越接近1表示越平坦（类似白噪声）
        """
        # 避免log(0)
        power_safe = power + 1e-10

        geometric_mean = np.exp(np.mean(np.log(power_safe)))
        arithmetic_mean = np.mean(power)

        flatness = geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0
        return flatness

    def _find_spectral_peaks(
        self,
        freq: np.ndarray,
        amplitude: np.ndarray,
        sampling_rate: int,
        prominence_factor: float = 0.1,
        max_peaks: int = 10
    ) -> Dict:
        """
        在频谱中检测峰值

        Args:
            freq: 频率数组
            amplitude: 幅值数组
            sampling_rate: 采样率
            prominence_factor: 峰值显著性因子（相对于最大幅值）
            max_peaks: 最大峰值数量

        Returns:
            dict: 峰值信息
        """
        # 计算峰值检测参数
        max_amplitude = np.max(amplitude)
        prominence = max_amplitude * prominence_factor

        # 查找峰值
        peaks_idx, properties = find_peaks(
            amplitude,
            prominence=prominence,
            distance=sampling_rate // 100  # 最小间隔约10Hz
        )

        # 按幅值排序，取前N个
        if len(peaks_idx) > max_peaks:
            sorted_idx = np.argsort(amplitude[peaks_idx])[::-1][:max_peaks]
            peaks_idx = peaks_idx[sorted_idx]

        peak_frequencies = freq[peaks_idx].tolist()
        peak_amplitudes = amplitude[peaks_idx].tolist()

        return {
            'count': len(peaks_idx),
            'frequencies': peak_frequencies,
            'amplitudes': peak_amplitudes,
            'indices': peaks_idx
        }

    @staticmethod
    def _get_default_frequency_bands() -> Dict[str, Tuple[float, float]]:
        """获取默认频段划分"""
        return {
            'very_low': (0, 10),      # 超低频
            'low': (10, 100),         # 低频
            'mid': (100, 1000),       # 中频
            'high': (1000, 3000),     # 高频
            'very_high': (3000, 7500) # 超高频
        }

    @staticmethod
    def _compute_band_power(
        freq: np.ndarray,
        psd: np.ndarray,
        freq_bands: Dict[str, Tuple[float, float]]
    ) -> Dict[str, float]:
        """
        计算各频段功率

        Args:
            freq: 频率数组
            psd: 功率谱密度
            freq_bands: 频段定义

        Returns:
            dict: 各频段的功率
        """
        band_power = {}

        for band_name, (f_min, f_max) in freq_bands.items():
            # 选择频段范围
            band_mask = (freq >= f_min) & (freq < f_max)

            # 计算该频段的功率（积分）
            if np.any(band_mask):
                freq_step = freq[1] - freq[0] if len(freq) > 1 else 1
                power = np.trapz(psd[band_mask], dx=freq_step)
            else:
                power = 0.0

            band_power[band_name] = power

        return band_power

    @staticmethod
    def _compute_harmonic_ratio(
        freq: np.ndarray,
        amplitude: np.ndarray,
        fundamental_freq: float,
        n_harmonics: int = 5,
        tolerance: float = 2.0
    ) -> float:
        """
        计算谐波比（HNR - Harmonic-to-Noise Ratio）

        Args:
            freq: 频率数组
            amplitude: 幅值数组
            fundamental_freq: 基频
            n_harmonics: 谐波数量
            tolerance: 频率容差（Hz）

        Returns:
            float: 谐波比
        """
        if fundamental_freq < 1:
            return 0.0

        total_power = np.sum(amplitude ** 2)
        harmonic_power = 0.0

        # 累加各次谐波的能量
        for n in range(1, n_harmonics + 1):
            harmonic_freq = n * fundamental_freq

            # 查找谐波频率附近的峰值
            freq_mask = np.abs(freq - harmonic_freq) < tolerance
            if np.any(freq_mask):
                harmonic_power += np.max(amplitude[freq_mask] ** 2)

        # 计算谐波比
        noise_power = total_power - harmonic_power
        ratio = harmonic_power / noise_power if noise_power > 0 else 0

        return ratio

    @staticmethod
    def _compute_thd(
        freq: np.ndarray,
        amplitude: np.ndarray,
        fundamental_freq: float,
        n_harmonics: int = 5,
        tolerance: float = 2.0
    ) -> float:
        """
        计算总谐波失真（THD - Total Harmonic Distortion）

        THD = sqrt(sum(harmonics^2)) / fundamental

        Args:
            freq: 频率数组
            amplitude: 幅值数组
            fundamental_freq: 基频
            n_harmonics: 谐波数量
            tolerance: 频率容差

        Returns:
            float: THD百分比
        """
        if fundamental_freq < 1:
            return 0.0

        # 获取基波幅值
        fundamental_mask = np.abs(freq - fundamental_freq) < tolerance
        if not np.any(fundamental_mask):
            return 0.0

        fundamental_amplitude = np.max(amplitude[fundamental_mask])

        # 累加谐波幅值的平方
        harmonic_sum_sq = 0.0
        for n in range(2, n_harmonics + 1):  # 从第2次谐波开始
            harmonic_freq = n * fundamental_freq
            harmonic_mask = np.abs(freq - harmonic_freq) < tolerance

            if np.any(harmonic_mask):
                harmonic_amplitude = np.max(amplitude[harmonic_mask])
                harmonic_sum_sq += harmonic_amplitude ** 2

        # 计算THD
        thd = np.sqrt(harmonic_sum_sq) / fundamental_amplitude if fundamental_amplitude > 0 else 0

        return thd * 100  # 转换为百分比


# ====================================
# 便捷函数
# ====================================

def compute_spectrum(
    signal_data: np.ndarray,
    sampling_rate: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    便捷函数：计算频谱

    Args:
        signal_data: 时域信号
        sampling_rate: 采样率

    Returns:
        tuple: (频率数组, 幅值谱)
    """
    analyzer = FrequencyAnalyzer()
    return analyzer.compute_fft(signal_data, sampling_rate)


def extract_frequency_features(
    signal_data: np.ndarray,
    sampling_rate: int
) -> FrequencyFeatures:
    """
    便捷函数：提取频域特征

    Args:
        signal_data: 时域信号
        sampling_rate: 采样率

    Returns:
        FrequencyFeatures: 频域特征
    """
    analyzer = FrequencyAnalyzer()
    return analyzer.extract_features(signal_data, sampling_rate)


# ====================================
# 测试代码
# ====================================

if __name__ == "__main__":
    print("=" * 60)
    print("频域分析器测试")
    print("=" * 60)

    # 加载测试数据
    print("\n【准备】加载测试数据...")
    from data.loader import DataLoader

    loader = DataLoader(validate=False)
    signal_data = loader.load('light_wear', 'normal', 10, 'A', 'X')

    if signal_data is None:
        print("❌ 无法加载测试数据")
        exit(1)

    print(f"   数据长度: {len(signal_data)} 采样点")
    print(f"   采样率: {signal_data.sampling_rate} Hz")
    print(f"   时长: {signal_data.duration:.2f} 秒")

    # 创建分析器
    analyzer = FrequencyAnalyzer(window='hann')

    # 测试1: FFT频谱
    print("\n【测试1】FFT频谱计算:")
    freq, amplitude = analyzer.compute_fft(
        signal_data.time_series,
        signal_data.sampling_rate
    )
    print(f"   频率点数: {len(freq)}")
    print(f"   频率范围: 0 - {freq[-1]:.2f} Hz")
    print(f"   最大幅值: {np.max(amplitude):.6f}")

    # 找到主频率
    dominant_idx = np.argmax(amplitude)
    print(f"   主频率: {freq[dominant_idx]:.2f} Hz @ {amplitude[dominant_idx]:.6f}")

    # 测试2: 功率谱密度
    print("\n【测试2】功率谱密度 (Welch方法):")
    psd_freq, psd = analyzer.compute_psd(
        signal_data.time_series,
        signal_data.sampling_rate,
        method='welch'
    )
    print(f"   频率点数: {len(psd_freq)}")
    print(f"   总功率: {np.trapz(psd, psd_freq):.6e}")
    print(f"   最大PSD: {np.max(psd):.6e} @ {psd_freq[np.argmax(psd)]:.2f} Hz")

    # 测试3: 频域特征提取
    print("\n【测试3】频域特征提取:")
    features = analyzer.extract_features(
        signal_data.time_series,
        signal_data.sampling_rate
    )

    print(f"   主频率: {features.dominant_frequency:.2f} Hz")
    print(f"   主幅值: {features.dominant_amplitude:.6f}")
    print(f"   平均频率: {features.mean_frequency:.2f} Hz")
    print(f"   中位频率: {features.median_frequency:.2f} Hz")
    print(f"   频谱质心: {features.spectral_centroid:.2f} Hz")
    print(f"   频谱扩展度: {features.spectral_spread:.2f} Hz")
    print(f"   频谱偏度: {features.spectral_skewness:.4f}")
    print(f"   频谱峰度: {features.spectral_kurtosis:.4f}")
    print(f"   频谱滚降点: {features.spectral_rolloff:.2f} Hz")
    print(f"   频谱平坦度: {features.spectral_flatness:.4f}")
    print(f"   总功率: {features.total_power:.6e}")
    print(f"   谐波比: {features.harmonic_ratio:.4f}")
    print(f"   总谐波失真: {features.thd:.2f}%")

    # 测试4: 峰值检测
    print("\n【测试4】频谱峰值检测:")
    print(f"   检测到 {features.peak_count} 个峰值")
    if features.peak_count > 0:
        print("   前5个峰值:")
        for i, (f, a) in enumerate(zip(features.peak_frequencies[:5], features.peak_amplitudes[:5])):
            print(f"     {i+1}. {f:.2f} Hz @ {a:.6f}")

    # 测试5: 频段功率分析
    print("\n【测试5】频段功率分析:")
    print("   各频段功率分布:")
    total_band_power = sum(features.band_power.values())
    for band_name, power in features.band_power.items():
        percentage = (power / total_band_power * 100) if total_band_power > 0 else 0
        print(f"     {band_name:12s}: {power:.6e} ({percentage:.1f}%)")

    # 测试6: 便捷函数
    print("\n【测试6】便捷函数测试:")
    freq2, amp2 = compute_spectrum(signal_data.time_series, signal_data.sampling_rate)
    features2 = extract_frequency_features(signal_data.time_series, signal_data.sampling_rate)
    print(f"   ✅ compute_spectrum: {len(freq2)} 频率点")
    print(f"   ✅ extract_frequency_features: 主频率 {features2.dominant_frequency:.2f} Hz")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
