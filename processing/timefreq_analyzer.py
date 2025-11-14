"""
时频分析器

实现时频域信号分析：
- 短时傅里叶变换（STFT）
- 连续小波变换（CWT）
- 小波包分解（WPD）
- 时频特征提取

使用示例:
    >>> from data.loader import DataLoader
    >>> loader = DataLoader()
    >>> signal_data = loader.load('light_wear', 'normal', 10, 'A', 'X')
    >>>
    >>> analyzer = TimeFreqAnalyzer()
    >>> stft_result = analyzer.compute_stft(signal_data.time_series, signal_data.sampling_rate)
    >>> cwt_result = analyzer.compute_cwt(signal_data.time_series, signal_data.sampling_rate)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from scipy import signal
from scipy.signal import stft, istft
import pywt

from config.settings import (
    SAMPLING_RATE,
    logger
)


@dataclass
class STFTResult:
    """
    STFT结果数据类

    包含STFT的时间、频率和幅值信息。
    """
    time: np.ndarray  # 时间轴
    frequency: np.ndarray  # 频率轴
    magnitude: np.ndarray  # 幅值谱（2D: frequency x time）
    phase: np.ndarray  # 相位谱（2D）
    power: np.ndarray  # 功率谱（2D）

    def __str__(self) -> str:
        return (
            f"STFTResult(\n"
            f"  时间点数: {len(self.time)}\n"
            f"  频率点数: {len(self.frequency)}\n"
            f"  频率范围: {self.frequency[0]:.1f} - {self.frequency[-1]:.1f} Hz\n"
            f"  时间范围: {self.time[0]:.2f} - {self.time[-1]:.2f} 秒\n"
            f")"
        )


@dataclass
class CWTResult:
    """
    连续小波变换（CWT）结果数据类
    """
    time: np.ndarray  # 时间轴
    scales: np.ndarray  # 尺度
    frequencies: np.ndarray  # 对应的频率
    coefficients: np.ndarray  # 小波系数（2D: scales x time）
    power: np.ndarray  # 功率（2D）
    wavelet_name: str  # 小波名称

    def __str__(self) -> str:
        return (
            f"CWTResult(\n"
            f"  小波: {self.wavelet_name}\n"
            f"  时间点数: {len(self.time)}\n"
            f"  尺度数: {len(self.scales)}\n"
            f"  频率范围: {self.frequencies[-1]:.1f} - {self.frequencies[0]:.1f} Hz\n"
            f")"
        )


@dataclass
class WaveletPacketNode:
    """小波包节点"""
    level: int  # 分解层数
    index: int  # 节点索引
    coefficients: np.ndarray  # 系数
    energy: float  # 能量
    freq_band: Tuple[float, float]  # 频段范围


@dataclass
class TimeFreqFeatures:
    """
    时频域特征数据类

    包含从STFT/CWT提取的特征。
    """
    # 时频域统计特征
    mean_instantaneous_frequency: float  # 平均瞬时频率
    instantaneous_frequency_std: float  # 瞬时频率标准差
    spectral_entropy: float  # 频谱熵
    time_bandwidth_product: float  # 时间带宽积

    # 小波特征
    wavelet_energy: Dict[str, float] = field(default_factory=dict)  # 各尺度能量
    wavelet_entropy: float = 0.0  # 小波熵

    # 时变特征
    frequency_variation: float = 0.0  # 频率变化程度
    amplitude_modulation: float = 0.0  # 幅度调制指数

    def __str__(self) -> str:
        return (
            f"TimeFreqFeatures(\n"
            f"  平均瞬时频率: {self.mean_instantaneous_frequency:.2f} Hz\n"
            f"  频谱熵: {self.spectral_entropy:.4f}\n"
            f"  小波熵: {self.wavelet_entropy:.4f}\n"
            f")"
        )


class TimeFreqAnalyzer:
    """
    时频分析器

    提供完整的时频域分析功能。
    """

    def __init__(
        self,
        window: str = 'hann',
        nperseg: int = 256,
        noverlap: Optional[int] = None
    ):
        """
        初始化时频分析器

        Args:
            window: STFT窗函数
            nperseg: STFT窗长度
            noverlap: STFT重叠长度
        """
        self.window = window
        self.nperseg = nperseg
        self.noverlap = noverlap if noverlap is not None else nperseg // 2

        logger.debug(f"TimeFreqAnalyzer 初始化: window={window}, nperseg={nperseg}")

    def compute_stft(
        self,
        signal_data: np.ndarray,
        sampling_rate: int,
        return_complex: bool = False
    ) -> Union[STFTResult, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        计算短时傅里叶变换（STFT）

        Args:
            signal_data: 时域信号
            sampling_rate: 采样率
            return_complex: 是否返回复数结果

        Returns:
            STFTResult 或 (f, t, Zxx)
        """
        # 计算STFT
        f, t, Zxx = stft(
            signal_data,
            fs=sampling_rate,
            window=self.window,
            nperseg=self.nperseg,
            noverlap=self.noverlap
        )

        if return_complex:
            return f, t, Zxx

        # 计算幅值、相位和功率
        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)
        power = magnitude ** 2

        result = STFTResult(
            time=t,
            frequency=f,
            magnitude=magnitude,
            phase=phase,
            power=power
        )

        logger.debug(
            f"STFT计算完成: {len(t)}个时间点, {len(f)}个频率点, "
            f"频率范围 {f[0]:.1f}-{f[-1]:.1f} Hz"
        )

        return result

    def compute_inverse_stft(
        self,
        Zxx: np.ndarray,
        sampling_rate: int
    ) -> np.ndarray:
        """
        计算逆STFT

        Args:
            Zxx: STFT复数结果
            sampling_rate: 采样率

        Returns:
            重构的时域信号
        """
        _, reconstructed = istft(
            Zxx,
            fs=sampling_rate,
            window=self.window,
            nperseg=self.nperseg,
            noverlap=self.noverlap
        )

        logger.debug(f"逆STFT完成: 重构信号长度 {len(reconstructed)}")

        return reconstructed

    def compute_cwt(
        self,
        signal_data: np.ndarray,
        sampling_rate: int,
        wavelet: str = 'morl',
        scales: Optional[np.ndarray] = None,
        freq_range: Tuple[float, float] = (10, 5000)
    ) -> CWTResult:
        """
        计算连续小波变换（CWT）

        Args:
            signal_data: 时域信号
            sampling_rate: 采样率
            wavelet: 小波类型 ('morl', 'mexh', 'cgau5', etc.)
            scales: 尺度数组（None则自动生成）
            freq_range: 感兴趣的频率范围

        Returns:
            CWTResult: CWT结果
        """
        # 如果未指定尺度，自动生成
        if scales is None:
            scales = self._generate_scales(
                wavelet,
                sampling_rate,
                freq_range,
                num_scales=100
            )

        # 计算CWT
        coefficients, frequencies = pywt.cwt(
            signal_data,
            scales,
            wavelet,
            sampling_period=1.0 / sampling_rate
        )

        # 时间轴
        time = np.arange(len(signal_data)) / sampling_rate

        # 计算功率
        power = np.abs(coefficients) ** 2

        result = CWTResult(
            time=time,
            scales=scales,
            frequencies=frequencies,
            coefficients=coefficients,
            power=power,
            wavelet_name=wavelet
        )

        logger.debug(
            f"CWT计算完成: {wavelet}小波, {len(scales)}个尺度, "
            f"频率范围 {frequencies[-1]:.1f}-{frequencies[0]:.1f} Hz"
        )

        return result

    @staticmethod
    def _generate_scales(
        wavelet: str,
        sampling_rate: int,
        freq_range: Tuple[float, float],
        num_scales: int = 100
    ) -> np.ndarray:
        """
        生成CWT尺度数组

        Args:
            wavelet: 小波名称
            sampling_rate: 采样率
            freq_range: 频率范围
            num_scales: 尺度数量

        Returns:
            尺度数组
        """
        f_min, f_max = freq_range

        # 使用pywt.scale2frequency来计算尺度
        # 先生成候选尺度，然后反推频率，筛选在范围内的
        # 简化方法：直接使用经验公式

        # 对于大多数小波，中心频率约为1.0
        # 使用更稳健的方法
        dt = 1.0 / sampling_rate

        # 根据频率范围计算尺度范围
        # scale = center_freq / (freq * dt)
        # 使用近似值：对于morl wavelet, center_freq ≈ 1.0
        scale_min = 1.0 / (f_max * dt)
        scale_max = 1.0 / (f_min * dt)

        # 对数间隔生成尺度
        scales = np.logspace(
            np.log10(scale_min),
            np.log10(scale_max),
            num=num_scales
        )

        return scales

    def wavelet_packet_decompose(
        self,
        signal_data: np.ndarray,
        wavelet: str = 'db4',
        level: int = 4,
        mode: str = 'symmetric'
    ) -> Dict[str, WaveletPacketNode]:
        """
        小波包分解（WPD）

        Args:
            signal_data: 时域信号
            wavelet: 小波类型
            level: 分解层数
            mode: 边界处理模式

        Returns:
            dict: 节点路径 -> WaveletPacketNode
        """
        # 创建小波包树
        wp = pywt.WaveletPacket(
            data=signal_data,
            wavelet=wavelet,
            mode=mode,
            maxlevel=level
        )

        # 提取所有叶节点
        nodes = {}

        for i, node in enumerate(wp.get_level(level, 'freq')):
            path = node.path
            coeffs = node.data
            energy = np.sum(coeffs ** 2)

            # 计算频段（简化，实际频段取决于采样率）
            # 这里假设信号频率范围 [0, fs/2]，均匀分割
            num_bands = 2 ** level
            # 使用频率顺序索引（freq模式已排序）
            band_idx = i
            band_width = (SAMPLING_RATE / 2) / num_bands
            freq_band = (band_idx * band_width, (band_idx + 1) * band_width)

            nodes[path] = WaveletPacketNode(
                level=level,
                index=band_idx,
                coefficients=coeffs,
                energy=energy,
                freq_band=freq_band
            )

        logger.debug(
            f"小波包分解完成: {wavelet}小波, {level}层, {len(nodes)}个节点"
        )

        return nodes

    def extract_timefreq_features(
        self,
        signal_data: np.ndarray,
        sampling_rate: int,
        method: str = 'stft'
    ) -> TimeFreqFeatures:
        """
        提取时频域特征

        Args:
            signal_data: 时域信号
            sampling_rate: 采样率
            method: 方法 ('stft' 或 'cwt')

        Returns:
            TimeFreqFeatures: 时频特征
        """
        if method == 'stft':
            return self._extract_stft_features(signal_data, sampling_rate)
        elif method == 'cwt':
            return self._extract_cwt_features(signal_data, sampling_rate)
        else:
            raise ValueError(f"未知的方法: {method}")

    def _extract_stft_features(
        self,
        signal_data: np.ndarray,
        sampling_rate: int
    ) -> TimeFreqFeatures:
        """从STFT提取特征"""
        # 计算STFT
        stft_result = self.compute_stft(signal_data, sampling_rate)

        # 瞬时频率
        instantaneous_freq = self._compute_instantaneous_frequency(
            stft_result.frequency,
            stft_result.power
        )
        mean_inst_freq = np.mean(instantaneous_freq)
        inst_freq_std = np.std(instantaneous_freq)

        # 频谱熵
        spectral_entropy = self._compute_spectral_entropy(stft_result.power)

        # 时间带宽积
        time_bandwidth = self._compute_time_bandwidth_product(
            stft_result.time,
            stft_result.frequency,
            stft_result.power
        )

        # 频率变化
        frequency_variation = np.std(np.diff(instantaneous_freq))

        # 幅度调制
        amplitude_modulation = self._compute_amplitude_modulation(
            stft_result.magnitude
        )

        features = TimeFreqFeatures(
            mean_instantaneous_frequency=mean_inst_freq,
            instantaneous_frequency_std=inst_freq_std,
            spectral_entropy=spectral_entropy,
            time_bandwidth_product=time_bandwidth,
            frequency_variation=frequency_variation,
            amplitude_modulation=amplitude_modulation
        )

        logger.info(f"STFT特征提取完成: 平均瞬时频率={mean_inst_freq:.2f}Hz")

        return features

    def _extract_cwt_features(
        self,
        signal_data: np.ndarray,
        sampling_rate: int
    ) -> TimeFreqFeatures:
        """从CWT提取特征"""
        # 计算CWT
        cwt_result = self.compute_cwt(signal_data, sampling_rate)

        # 瞬时频率（从CWT ridges提取）
        instantaneous_freq = self._compute_instantaneous_frequency_cwt(
            cwt_result.frequencies,
            cwt_result.power
        )
        mean_inst_freq = np.mean(instantaneous_freq)
        inst_freq_std = np.std(instantaneous_freq)

        # 小波熵
        wavelet_entropy = self._compute_wavelet_entropy(cwt_result.power)

        # 各尺度能量
        wavelet_energy = {}
        for i, scale in enumerate(cwt_result.scales[:10]):  # 只取前10个尺度
            energy = np.sum(cwt_result.power[i, :])
            wavelet_energy[f'scale_{i}'] = energy

        # 频谱熵
        spectral_entropy = self._compute_spectral_entropy(cwt_result.power)

        # 时间带宽积
        time_bandwidth = self._compute_time_bandwidth_product(
            cwt_result.time,
            cwt_result.frequencies,
            cwt_result.power
        )

        features = TimeFreqFeatures(
            mean_instantaneous_frequency=mean_inst_freq,
            instantaneous_frequency_std=inst_freq_std,
            spectral_entropy=spectral_entropy,
            time_bandwidth_product=time_bandwidth,
            wavelet_energy=wavelet_energy,
            wavelet_entropy=wavelet_entropy
        )

        logger.info(f"CWT特征提取完成: 平均瞬时频率={mean_inst_freq:.2f}Hz")

        return features

    @staticmethod
    def _compute_instantaneous_frequency(
        freq_axis: np.ndarray,
        power: np.ndarray
    ) -> np.ndarray:
        """
        计算瞬时频率（从STFT）

        每个时间点的加权平均频率
        """
        # 归一化功率（每个时间点）
        power_norm = power / (np.sum(power, axis=0, keepdims=True) + 1e-10)

        # 加权平均
        instantaneous_freq = np.sum(
            freq_axis[:, np.newaxis] * power_norm,
            axis=0
        )

        return instantaneous_freq

    @staticmethod
    def _compute_instantaneous_frequency_cwt(
        freq_axis: np.ndarray,
        power: np.ndarray
    ) -> np.ndarray:
        """计算瞬时频率（从CWT）"""
        # 归一化功率（每个时间点）
        power_norm = power / (np.sum(power, axis=0, keepdims=True) + 1e-10)

        # 加权平均
        instantaneous_freq = np.sum(
            freq_axis[:, np.newaxis] * power_norm,
            axis=0
        )

        return instantaneous_freq

    @staticmethod
    def _compute_spectral_entropy(power: np.ndarray) -> float:
        """
        计算频谱熵

        衡量频谱的复杂度和不确定性
        """
        # 归一化功率
        power_flat = power.flatten()
        power_norm = power_flat / (np.sum(power_flat) + 1e-10)

        # 去除零值
        power_norm = power_norm[power_norm > 0]

        # 计算熵
        entropy = -np.sum(power_norm * np.log2(power_norm + 1e-10))

        return entropy

    @staticmethod
    def _compute_wavelet_entropy(power: np.ndarray) -> float:
        """计算小波熵"""
        # 每个尺度的能量
        scale_energy = np.sum(power, axis=1)
        total_energy = np.sum(scale_energy)

        # 归一化
        scale_energy_norm = scale_energy / (total_energy + 1e-10)
        scale_energy_norm = scale_energy_norm[scale_energy_norm > 0]

        # 熵
        entropy = -np.sum(scale_energy_norm * np.log2(scale_energy_norm + 1e-10))

        return entropy

    @staticmethod
    def _compute_time_bandwidth_product(
        time_axis: np.ndarray,
        freq_axis: np.ndarray,
        power: np.ndarray
    ) -> float:
        """
        计算时间带宽积

        衡量信号的时频集中度
        """
        # 归一化功率
        power_flat = power.flatten()
        power_norm = power_flat / (np.sum(power_flat) + 1e-10)

        # 时间和频率的标准差
        # 简化计算：使用功率加权的全局标准差
        time_std = np.std(time_axis)
        freq_std = np.std(freq_axis)

        # 时间带宽积
        tbp = time_std * freq_std

        return tbp

    @staticmethod
    def _compute_amplitude_modulation(magnitude: np.ndarray) -> float:
        """
        计算幅度调制指数

        衡量信号幅度随时间的变化程度
        """
        # 每个时间点的总幅度
        time_amplitude = np.sum(magnitude, axis=0)

        # 归一化
        time_amplitude_norm = time_amplitude / (np.mean(time_amplitude) + 1e-10)

        # 幅度调制指数（变异系数）
        am_index = np.std(time_amplitude_norm) / (np.mean(time_amplitude_norm) + 1e-10)

        return am_index


# ====================================
# 便捷函数
# ====================================

def compute_spectrogram(
    signal_data: np.ndarray,
    sampling_rate: int,
    nperseg: int = 256
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    便捷函数：计算时频谱图

    Args:
        signal_data: 时域信号
        sampling_rate: 采样率
        nperseg: 窗长度

    Returns:
        tuple: (频率, 时间, 功率谱)
    """
    analyzer = TimeFreqAnalyzer(nperseg=nperseg)
    stft_result = analyzer.compute_stft(signal_data, sampling_rate)
    return stft_result.frequency, stft_result.time, stft_result.power


# ====================================
# 测试代码
# ====================================

if __name__ == "__main__":
    print("=" * 60)
    print("时频分析器测试")
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

    # 使用前5秒数据以加快测试
    test_duration = 5  # 秒
    test_samples = test_duration * signal_data.sampling_rate
    test_signal = signal_data.time_series[:test_samples]
    print(f"   测试数据: 前{test_duration}秒 ({len(test_signal)} 采样点)")

    # 创建分析器
    analyzer = TimeFreqAnalyzer(nperseg=512, noverlap=256)

    # 测试1: STFT
    print("\n【测试1】短时傅里叶变换 (STFT):")
    stft_result = analyzer.compute_stft(test_signal, signal_data.sampling_rate)
    print(stft_result)
    print(f"   功率谱形状: {stft_result.power.shape}")
    print(f"   最大功率: {np.max(stft_result.power):.6e}")

    # 测试2: 逆STFT
    print("\n【测试2】逆STFT (重构信号):")
    f, t, Zxx = analyzer.compute_stft(test_signal, signal_data.sampling_rate, return_complex=True)
    reconstructed = analyzer.compute_inverse_stft(Zxx, signal_data.sampling_rate)
    print(f"   原始信号长度: {len(test_signal)}")
    print(f"   重构信号长度: {len(reconstructed)}")
    # 计算重构误差
    min_len = min(len(test_signal), len(reconstructed))
    error = np.mean((test_signal[:min_len] - reconstructed[:min_len]) ** 2)
    print(f"   重构误差 (MSE): {error:.6e}")

    # 测试3: CWT
    print("\n【测试3】连续小波变换 (CWT):")
    cwt_result = analyzer.compute_cwt(
        test_signal,
        signal_data.sampling_rate,
        wavelet='morl',
        freq_range=(10, 3000)
    )
    print(cwt_result)
    print(f"   系数形状: {cwt_result.coefficients.shape}")
    print(f"   功率范围: {np.min(cwt_result.power):.6e} - {np.max(cwt_result.power):.6e}")

    # 测试4: 小波包分解
    print("\n【测试4】小波包分解 (WPD):")
    wpd_nodes = analyzer.wavelet_packet_decompose(
        test_signal,
        wavelet='db4',
        level=4
    )
    print(f"   分解节点数: {len(wpd_nodes)}")
    print(f"   前3个节点:")
    for i, (path, node) in enumerate(list(wpd_nodes.items())[:3]):
        print(f"     节点 {path}: 能量={node.energy:.6e}, 频段={node.freq_band[0]:.1f}-{node.freq_band[1]:.1f} Hz")

    # 测试5: STFT特征提取
    print("\n【测试5】STFT时频特征提取:")
    stft_features = analyzer.extract_timefreq_features(
        test_signal,
        signal_data.sampling_rate,
        method='stft'
    )
    print(f"   平均瞬时频率: {stft_features.mean_instantaneous_frequency:.2f} Hz")
    print(f"   瞬时频率标准差: {stft_features.instantaneous_frequency_std:.2f} Hz")
    print(f"   频谱熵: {stft_features.spectral_entropy:.4f}")
    print(f"   时间带宽积: {stft_features.time_bandwidth_product:.4f}")
    print(f"   频率变化: {stft_features.frequency_variation:.4f}")
    print(f"   幅度调制: {stft_features.amplitude_modulation:.4f}")

    # 测试6: CWT特征提取
    print("\n【测试6】CWT时频特征提取:")
    cwt_features = analyzer.extract_timefreq_features(
        test_signal,
        signal_data.sampling_rate,
        method='cwt'
    )
    print(f"   平均瞬时频率: {cwt_features.mean_instantaneous_frequency:.2f} Hz")
    print(f"   瞬时频率标准差: {cwt_features.instantaneous_frequency_std:.2f} Hz")
    print(f"   频谱熵: {cwt_features.spectral_entropy:.4f}")
    print(f"   小波熵: {cwt_features.wavelet_entropy:.4f}")
    print(f"   前3个尺度能量:")
    for i, (scale_name, energy) in enumerate(list(cwt_features.wavelet_energy.items())[:3]):
        print(f"     {scale_name}: {energy:.6e}")

    # 测试7: 便捷函数
    print("\n【测试7】便捷函数测试:")
    freq, time, power = compute_spectrogram(test_signal, signal_data.sampling_rate, nperseg=512)
    print(f"   ✅ compute_spectrogram: {power.shape[0]} 频率 x {power.shape[1]} 时间")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
