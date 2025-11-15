"""
齿轮特征频率分析器

实现齿轮专用的频域分析：
- 齿轮啮合频率（GMF）计算
- 特征频率提取
- 边频带分析
- 齿轮故障诊断指标

使用示例:
    >>> from data.loader import DataLoader
    >>> loader = DataLoader()
    >>> signal_data = loader.load('light_wear', 'normal', 10, 'A', 'X')
    >>>
    >>> # 齿轮参数
    >>> gear_params = GearParameters(
    ...     drive_teeth=40,
    ...     driven_teeth=40,
    ...     shaft_speed=1000  # rpm
    ... )
    >>>
    >>> analyzer = GearAnalyzer(gear_params)
    >>> gear_features = analyzer.extract_gear_features(signal_data.time_series, signal_data.sampling_rate)
    >>> print(f"啮合频率能量比: {gear_features.gmf_energy_ratio:.4f}")
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy import signal as sp_signal
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

from config.settings import logger


@dataclass
class GearParameters:
    """
    齿轮参数配置

    定义齿轮传动系统的基本参数。
    """
    drive_teeth: int  # 主动轮齿数
    driven_teeth: int  # 从动轮齿数
    shaft_speed: float  # 轴转速 (rpm)

    # 可选参数
    drive_diameter: Optional[float] = None  # 主动轮节圆直径 (mm)
    driven_diameter: Optional[float] = None  # 从动轮节圆直径 (mm)
    center_distance: Optional[float] = None  # 中心距 (mm)

    def __post_init__(self):
        """计算派生参数"""
        # 齿轮传动比
        self.gear_ratio = self.driven_teeth / self.drive_teeth

        # 主动轮转频 (Hz)
        self.drive_freq = self.shaft_speed / 60

        # 从动轮转频 (Hz)
        self.driven_freq = self.drive_freq / self.gear_ratio

        # 啮合频率 (Hz)
        self.mesh_freq = self.drive_freq * self.drive_teeth

        logger.debug(
            f"齿轮参数: 主动{self.drive_teeth}齿, 从动{self.driven_teeth}齿, "
            f"转速{self.shaft_speed}rpm, GMF={self.mesh_freq:.2f}Hz"
        )

    def __str__(self) -> str:
        return (
            f"GearParameters(\n"
            f"  主动轮齿数: {self.drive_teeth}\n"
            f"  从动轮齿数: {self.driven_teeth}\n"
            f"  转速: {self.shaft_speed} rpm\n"
            f"  传动比: {self.gear_ratio:.4f}\n"
            f"  主动轮转频: {self.drive_freq:.2f} Hz\n"
            f"  从动轮转频: {self.driven_freq:.2f} Hz\n"
            f"  啮合频率: {self.mesh_freq:.2f} Hz\n"
            f")"
        )


@dataclass
class CharacteristicFrequencies:
    """
    齿轮特征频率

    存储齿轮系统的各种特征频率。
    """
    # 基本频率
    shaft_frequency: float  # 轴频
    mesh_frequency: float  # 啮合频率

    # 谐波频率
    mesh_harmonics: List[float] = field(default_factory=list)  # 啮合频率谐波

    # 边频带
    sidebands: List[Tuple[float, float]] = field(default_factory=list)  # (频率, 能量)

    # 故障特征频率
    fault_frequencies: Dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"CharacteristicFrequencies(\n"
            f"  轴频: {self.shaft_frequency:.2f} Hz\n"
            f"  啮合频率: {self.mesh_frequency:.2f} Hz\n"
            f"  谐波数量: {len(self.mesh_harmonics)}\n"
            f"  边频带数量: {len(self.sidebands)}\n"
            f")"
        )


@dataclass
class GearFeatures:
    """
    齿轮诊断特征

    包含用于齿轮磨损诊断的各种特征指标。
    """
    # 啮合频率特征
    gmf_amplitude: float  # GMF幅值
    gmf_energy: float  # GMF能量
    gmf_energy_ratio: float  # GMF能量占比

    # 谐波特征
    harmonic_energies: List[float] = field(default_factory=list)  # 各次谐波能量
    total_harmonic_energy: float = 0.0  # 总谐波能量

    # 边频带特征
    sideband_count: int = 0  # 边频带数量
    sideband_energy: float = 0.0  # 边频带总能量
    sideband_ratio: float = 0.0  # 边频带能量比

    # 调制特征
    amplitude_modulation_index: float = 0.0  # 幅度调制指数
    frequency_modulation_index: float = 0.0  # 频率调制指数

    # 故障指标
    fault_factor: float = 0.0  # 综合故障因子
    wear_indicator: float = 0.0  # 磨损指标

    def __str__(self) -> str:
        return (
            f"GearFeatures(\n"
            f"  GMF幅值: {self.gmf_amplitude:.6f}\n"
            f"  GMF能量比: {self.gmf_energy_ratio:.4f}\n"
            f"  边频带数量: {self.sideband_count}\n"
            f"  边频带能量比: {self.sideband_ratio:.4f}\n"
            f"  故障因子: {self.fault_factor:.4f}\n"
            f")"
        )


class GearAnalyzer:
    """
    齿轮分析器

    提供齿轮传动系统的专用频域分析。
    """

    def __init__(
        self,
        gear_params: GearParameters,
        num_harmonics: int = 5,
        sideband_range: float = 3.0
    ):
        """
        初始化齿轮分析器

        Args:
            gear_params: 齿轮参数
            num_harmonics: 谐波数量
            sideband_range: 边频带搜索范围（倍数×轴频）
        """
        self.gear_params = gear_params
        self.num_harmonics = num_harmonics
        self.sideband_range = sideband_range

        logger.info(f"GearAnalyzer 初始化: GMF={gear_params.mesh_freq:.2f}Hz")

    def extract_gear_features(
        self,
        signal_data: np.ndarray,
        sampling_rate: int
    ) -> GearFeatures:
        """
        提取齿轮特征

        Args:
            signal_data: 时域信号
            sampling_rate: 采样率

        Returns:
            GearFeatures: 齿轮特征
        """
        # 计算频谱
        freq, amplitude = self._compute_spectrum(signal_data, sampling_rate)

        # 提取GMF特征
        gmf_features = self._extract_gmf_features(freq, amplitude)

        # 提取谐波特征
        harmonic_features = self._extract_harmonic_features(freq, amplitude)

        # 提取边频带特征
        sideband_features = self._extract_sideband_features(freq, amplitude)

        # 计算调制指标
        modulation_features = self._compute_modulation_indices(signal_data, sampling_rate)

        # 计算故障指标
        fault_indicators = self._compute_fault_indicators(
            gmf_features,
            harmonic_features,
            sideband_features
        )

        # 组装特征
        features = GearFeatures(
            gmf_amplitude=gmf_features['amplitude'],
            gmf_energy=gmf_features['energy'],
            gmf_energy_ratio=gmf_features['energy_ratio'],
            harmonic_energies=harmonic_features['energies'],
            total_harmonic_energy=harmonic_features['total_energy'],
            sideband_count=sideband_features['count'],
            sideband_energy=sideband_features['energy'],
            sideband_ratio=sideband_features['ratio'],
            amplitude_modulation_index=modulation_features['am_index'],
            frequency_modulation_index=modulation_features['fm_index'],
            fault_factor=fault_indicators['fault_factor'],
            wear_indicator=fault_indicators['wear_indicator']
        )

        logger.info(
            f"齿轮特征提取完成: GMF能量比={features.gmf_energy_ratio:.4f}, "
            f"故障因子={features.fault_factor:.4f}"
        )

        return features

    def identify_characteristic_frequencies(
        self,
        signal_data: np.ndarray,
        sampling_rate: int
    ) -> CharacteristicFrequencies:
        """
        识别特征频率

        Args:
            signal_data: 时域信号
            sampling_rate: 采样率

        Returns:
            CharacteristicFrequencies: 特征频率
        """
        # 计算频谱
        freq, amplitude = self._compute_spectrum(signal_data, sampling_rate)

        # 啮合频率谐波
        mesh_harmonics = []
        for n in range(1, self.num_harmonics + 1):
            harmonic_freq = self.gear_params.mesh_freq * n
            mesh_harmonics.append(harmonic_freq)

        # 边频带
        sidebands = self._find_sidebands(freq, amplitude)

        char_freqs = CharacteristicFrequencies(
            shaft_frequency=self.gear_params.drive_freq,
            mesh_frequency=self.gear_params.mesh_freq,
            mesh_harmonics=mesh_harmonics,
            sidebands=sidebands
        )

        logger.info(
            f"特征频率识别完成: {len(mesh_harmonics)}个谐波, {len(sidebands)}个边频带"
        )

        return char_freqs

    @staticmethod
    def _compute_spectrum(
        signal_data: np.ndarray,
        sampling_rate: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """计算频谱"""
        N = len(signal_data)

        # FFT
        fft_values = fft(signal_data)
        freq = fftfreq(N, 1.0 / sampling_rate)

        # 只取正频率
        positive_idx = freq >= 0
        freq = freq[positive_idx]
        fft_values = fft_values[positive_idx]

        # 幅值谱
        amplitude = np.abs(fft_values) / N
        amplitude[1:-1] = 2 * amplitude[1:-1]  # 正频率部分乘以2

        return freq, amplitude

    def _extract_gmf_features(
        self,
        freq: np.ndarray,
        amplitude: np.ndarray
    ) -> Dict:
        """提取GMF特征"""
        gmf = self.gear_params.mesh_freq

        # 在GMF附近搜索峰值（±2Hz容差）
        tolerance = 2.0
        gmf_mask = np.abs(freq - gmf) < tolerance

        if np.any(gmf_mask):
            gmf_amplitude = np.max(amplitude[gmf_mask])
            gmf_energy = np.sum(amplitude[gmf_mask] ** 2)
        else:
            gmf_amplitude = 0.0
            gmf_energy = 0.0

        # 总能量
        total_energy = np.sum(amplitude ** 2)
        gmf_energy_ratio = gmf_energy / (total_energy + 1e-10)

        return {
            'amplitude': gmf_amplitude,
            'energy': gmf_energy,
            'energy_ratio': gmf_energy_ratio
        }

    def _extract_harmonic_features(
        self,
        freq: np.ndarray,
        amplitude: np.ndarray
    ) -> Dict:
        """提取谐波特征"""
        gmf = self.gear_params.mesh_freq
        tolerance = 2.0

        harmonic_energies = []
        total_harmonic_energy = 0.0

        for n in range(1, self.num_harmonics + 1):
            harmonic_freq = gmf * n
            harmonic_mask = np.abs(freq - harmonic_freq) < tolerance

            if np.any(harmonic_mask):
                harmonic_energy = np.sum(amplitude[harmonic_mask] ** 2)
            else:
                harmonic_energy = 0.0

            harmonic_energies.append(harmonic_energy)
            total_harmonic_energy += harmonic_energy

        return {
            'energies': harmonic_energies,
            'total_energy': total_harmonic_energy
        }

    def _extract_sideband_features(
        self,
        freq: np.ndarray,
        amplitude: np.ndarray
    ) -> Dict:
        """提取边频带特征"""
        sidebands = self._find_sidebands(freq, amplitude)

        sideband_count = len(sidebands)
        sideband_energy = sum(energy for _, energy in sidebands)

        # 总能量
        total_energy = np.sum(amplitude ** 2)
        sideband_ratio = sideband_energy / (total_energy + 1e-10)

        return {
            'count': sideband_count,
            'energy': sideband_energy,
            'ratio': sideband_ratio
        }

    def _find_sidebands(
        self,
        freq: np.ndarray,
        amplitude: np.ndarray
    ) -> List[Tuple[float, float]]:
        """
        查找边频带

        边频带定义为：GMF ± n×轴频
        """
        gmf = self.gear_params.mesh_freq
        shaft_freq = self.gear_params.drive_freq

        sidebands = []

        # 搜索范围：±sideband_range倍轴频
        for n in range(1, int(self.sideband_range) + 1):
            # 下边频带
            lower_sideband = gmf - n * shaft_freq
            if lower_sideband > 0:
                lower_mask = np.abs(freq - lower_sideband) < 1.0
                if np.any(lower_mask):
                    lower_energy = np.sum(amplitude[lower_mask] ** 2)
                    if lower_energy > 1e-6:  # 能量阈值
                        sidebands.append((lower_sideband, lower_energy))

            # 上边频带
            upper_sideband = gmf + n * shaft_freq
            upper_mask = np.abs(freq - upper_sideband) < 1.0
            if np.any(upper_mask):
                upper_energy = np.sum(amplitude[upper_mask] ** 2)
                if upper_energy > 1e-6:
                    sidebands.append((upper_sideband, upper_energy))

        return sidebands

    def _compute_modulation_indices(
        self,
        signal_data: np.ndarray,
        sampling_rate: int
    ) -> Dict:
        """计算调制指标"""
        # 幅度调制指数（包络变化）
        from scipy.signal import hilbert

        analytic_signal = hilbert(signal_data)
        envelope = np.abs(analytic_signal)

        # 包络的变异系数
        envelope_mean = np.mean(envelope)
        envelope_std = np.std(envelope)
        am_index = envelope_std / (envelope_mean + 1e-10)

        # 频率调制指数（瞬时频率变化）
        # 简化计算：使用相位差分
        phase = np.unwrap(np.angle(analytic_signal))
        inst_freq = np.diff(phase) * sampling_rate / (2 * np.pi)
        fm_index = np.std(inst_freq) / (np.mean(inst_freq) + 1e-10)

        return {
            'am_index': am_index,
            'fm_index': fm_index
        }

    def _compute_fault_indicators(
        self,
        gmf_features: Dict,
        harmonic_features: Dict,
        sideband_features: Dict
    ) -> Dict:
        """
        计算故障指标

        综合多个特征计算齿轮故障指标
        """
        # 故障因子：综合考虑GMF能量、边频带、谐波
        # FF = (1 - GMF_ratio) + sideband_ratio + harmonic_decay

        gmf_ratio = gmf_features['energy_ratio']
        sideband_ratio = sideband_features['ratio']

        # 谐波衰减指标（理想情况下谐波应递减）
        harmonic_energies = harmonic_features['energies']
        if len(harmonic_energies) > 1:
            # 计算相邻谐波的能量比
            energy_ratios = []
            for i in range(len(harmonic_energies) - 1):
                if harmonic_energies[i] > 0:
                    ratio = harmonic_energies[i+1] / (harmonic_energies[i] + 1e-10)
                    energy_ratios.append(ratio)

            # 谐波衰减指标（越大表示谐波能量不递减，可能有故障）
            if energy_ratios:
                harmonic_decay = 1.0 - np.mean(energy_ratios)
            else:
                harmonic_decay = 0.0
        else:
            harmonic_decay = 0.0

        # 综合故障因子（0-1，越大表示故障越严重）
        fault_factor = 0.5 * (1 - gmf_ratio) + 0.3 * sideband_ratio + 0.2 * max(0, 1 - harmonic_decay)
        fault_factor = np.clip(fault_factor, 0, 1)

        # 磨损指标（基于边频带和GMF能量）
        wear_indicator = sideband_ratio / (gmf_ratio + 1e-10)

        return {
            'fault_factor': fault_factor,
            'wear_indicator': wear_indicator
        }

    def diagnose_condition(
        self,
        features: GearFeatures
    ) -> Dict[str, str]:
        """
        齿轮状态诊断

        Args:
            features: 齿轮特征

        Returns:
            dict: 诊断结果
        """
        # 基于故障因子的诊断
        if features.fault_factor < 0.3:
            condition = "正常"
            severity = "无"
        elif features.fault_factor < 0.5:
            condition = "轻度磨损"
            severity = "轻微"
        elif features.fault_factor < 0.7:
            condition = "中度磨损"
            severity = "中等"
        else:
            condition = "严重磨损"
            severity = "严重"

        # 边频带分析
        if features.sideband_count > 5:
            sideband_indication = "边频带丰富，可能存在调制"
        elif features.sideband_count > 2:
            sideband_indication = "存在边频带，需关注"
        else:
            sideband_indication = "边频带较少"

        # GMF能量比分析
        if features.gmf_energy_ratio > 0.5:
            gmf_indication = "GMF能量占比高，啮合良好"
        elif features.gmf_energy_ratio > 0.3:
            gmf_indication = "GMF能量占比正常"
        else:
            gmf_indication = "GMF能量占比低，可能啮合不良"

        diagnosis = {
            'condition': condition,
            'severity': severity,
            'fault_factor': f"{features.fault_factor:.4f}",
            'sideband_indication': sideband_indication,
            'gmf_indication': gmf_indication,
            'wear_indicator': f"{features.wear_indicator:.4f}"
        }

        logger.info(f"诊断完成: {condition} (故障因子={features.fault_factor:.4f})")

        return diagnosis


# ====================================
# 便捷函数
# ====================================

def create_default_gear_params(shaft_speed: float = 1000) -> GearParameters:
    """
    创建默认齿轮参数（基于实验平台）

    Args:
        shaft_speed: 轴转速 (rpm)

    Returns:
        GearParameters: 齿轮参数
    """
    # 根据实验平台，假设为40齿对40齿（1:1传动）
    return GearParameters(
        drive_teeth=40,
        driven_teeth=40,
        shaft_speed=shaft_speed
    )


# ====================================
# 测试代码
# ====================================

if __name__ == "__main__":
    print("=" * 60)
    print("齿轮分析器测试")
    print("=" * 60)

    # 加载测试数据
    print("\n【准备】加载测试数据...")
    from data.loader import DataLoader

    loader = DataLoader(validate=False)

    # 加载不同磨损状态的数据
    test_cases = [
        ('normal', 'normal', '正常-正常'),
        ('light_wear', 'normal', '轻磨-正常'),
        ('heavy_wear', 'normal', '重磨-正常')
    ]

    # 创建齿轮参数
    gear_params = create_default_gear_params(shaft_speed=1000)
    print(gear_params)

    # 创建分析器
    analyzer = GearAnalyzer(gear_params, num_harmonics=5)

    print("\n" + "=" * 60)

    for drive_state, driven_state, label in test_cases:
        print(f"\n【测试】{label}:")

        signal_data = loader.load(drive_state, driven_state, 10, 'A', 'X')

        if signal_data is None:
            print(f"   ❌ 无法加载数据")
            continue

        # 使用前10秒数据
        test_duration = 10
        test_samples = test_duration * signal_data.sampling_rate
        test_signal = signal_data.time_series[:test_samples]

        # 提取特征
        features = analyzer.extract_gear_features(test_signal, signal_data.sampling_rate)

        print(f"   GMF幅值: {features.gmf_amplitude:.6f}")
        print(f"   GMF能量比: {features.gmf_energy_ratio:.4f}")
        print(f"   边频带数量: {features.sideband_count}")
        print(f"   边频带能量比: {features.sideband_ratio:.4f}")
        print(f"   幅度调制指数: {features.amplitude_modulation_index:.4f}")
        print(f"   故障因子: {features.fault_factor:.4f}")
        print(f"   磨损指标: {features.wear_indicator:.4f}")

        # 诊断
        diagnosis = analyzer.diagnose_condition(features)
        print(f"\n   诊断结果:")
        print(f"   - 状态: {diagnosis['condition']}")
        print(f"   - 严重程度: {diagnosis['severity']}")
        print(f"   - GMF指示: {diagnosis['gmf_indication']}")
        print(f"   - 边频带指示: {diagnosis['sideband_indication']}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
