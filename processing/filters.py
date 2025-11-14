"""
数字滤波器

实现各种数字滤波和信号预处理：
- IIR滤波器（Butterworth, Chebyshev, Bessel）
- FIR滤波器
- 噪声抑制（移动平均、中值滤波、Savitzky-Golay）
- 信号预处理（去趋势、归一化、包络提取）

使用示例:
    >>> from data.loader import DataLoader
    >>> loader = DataLoader()
    >>> signal_data = loader.load('light_wear', 'normal', 10, 'A', 'X')
    >>>
    >>> # 应用带通滤波器
    >>> filtered = apply_bandpass_filter(signal_data.time_series, 100, 3000, signal_data.sampling_rate)
    >>>
    >>> # 去噪
    >>> denoised = denoise_signal(signal_data.time_series, method='savgol')
"""

import numpy as np
from typing import Optional, Tuple, Union, Literal
from dataclasses import dataclass
from scipy import signal
from scipy.signal import butter, cheby1, cheby2, bessel, filtfilt, sosfiltfilt
from scipy.signal import firwin, lfilter, medfilt, savgol_filter
from scipy.signal import detrend, hilbert

from config.settings import (
    SAMPLING_RATE,
    logger
)


@dataclass
class FilterResponse:
    """
    滤波器响应数据类

    包含滤波器的频率响应信息。
    """
    frequencies: np.ndarray  # 频率点
    magnitude: np.ndarray  # 幅频响应
    phase: np.ndarray  # 相频响应
    group_delay: Optional[np.ndarray] = None  # 群延迟

    def __str__(self) -> str:
        return (
            f"FilterResponse(\n"
            f"  频率点数: {len(self.frequencies)}\n"
            f"  频率范围: {self.frequencies[0]:.1f} - {self.frequencies[-1]:.1f} Hz\n"
            f"  最大增益: {np.max(self.magnitude):.4f}\n"
            f")"
        )


class DigitalFilter:
    """
    数字滤波器基类

    提供常见的IIR和FIR滤波器设计和应用。
    """

    def __init__(self, sampling_rate: int = SAMPLING_RATE):
        """
        初始化数字滤波器

        Args:
            sampling_rate: 采样率
        """
        self.sampling_rate = sampling_rate
        logger.debug(f"DigitalFilter 初始化: fs={sampling_rate}Hz")

    def design_butterworth(
        self,
        cutoff: Union[float, Tuple[float, float]],
        filter_type: Literal['lowpass', 'highpass', 'bandpass', 'bandstop'],
        order: int = 4,
        output: str = 'sos'
    ) -> Union[np.ndarray, Tuple]:
        """
        设计Butterworth滤波器

        Args:
            cutoff: 截止频率（Hz）或频带 (low, high)
            filter_type: 滤波器类型
            order: 滤波器阶数
            output: 输出格式 ('sos' 或 'ba')

        Returns:
            滤波器系数 (sos 或 (b, a))
        """
        # 归一化截止频率
        nyquist = self.sampling_rate / 2
        if isinstance(cutoff, tuple):
            normalized_cutoff = (cutoff[0] / nyquist, cutoff[1] / nyquist)
        else:
            normalized_cutoff = cutoff / nyquist

        # 设计滤波器
        if output == 'sos':
            sos = butter(
                order,
                normalized_cutoff,
                btype=filter_type,
                analog=False,
                output='sos'
            )
            logger.debug(
                f"Butterworth滤波器设计完成: {filter_type}, "
                f"order={order}, cutoff={cutoff}"
            )
            return sos
        else:
            b, a = butter(
                order,
                normalized_cutoff,
                btype=filter_type,
                analog=False,
                output='ba'
            )
            return b, a

    def design_chebyshev(
        self,
        cutoff: Union[float, Tuple[float, float]],
        filter_type: Literal['lowpass', 'highpass', 'bandpass', 'bandstop'],
        order: int = 4,
        ripple: float = 0.5,
        cheby_type: int = 1,
        output: str = 'sos'
    ) -> Union[np.ndarray, Tuple]:
        """
        设计Chebyshev滤波器

        Args:
            cutoff: 截止频率
            filter_type: 滤波器类型
            order: 滤波器阶数
            ripple: 波纹（dB）
            cheby_type: Chebyshev类型 (1 或 2)
            output: 输出格式

        Returns:
            滤波器系数
        """
        nyquist = self.sampling_rate / 2
        if isinstance(cutoff, tuple):
            normalized_cutoff = (cutoff[0] / nyquist, cutoff[1] / nyquist)
        else:
            normalized_cutoff = cutoff / nyquist

        if cheby_type == 1:
            cheby_func = cheby1
        else:
            cheby_func = cheby2

        if output == 'sos':
            sos = cheby_func(
                order,
                ripple,
                normalized_cutoff,
                btype=filter_type,
                analog=False,
                output='sos'
            )
            logger.debug(
                f"Chebyshev Type-{cheby_type}滤波器设计完成: {filter_type}, "
                f"order={order}, ripple={ripple}dB"
            )
            return sos
        else:
            b, a = cheby_func(
                order,
                ripple,
                normalized_cutoff,
                btype=filter_type,
                analog=False,
                output='ba'
            )
            return b, a

    def design_bessel(
        self,
        cutoff: Union[float, Tuple[float, float]],
        filter_type: Literal['lowpass', 'highpass', 'bandpass', 'bandstop'],
        order: int = 4,
        output: str = 'sos'
    ) -> Union[np.ndarray, Tuple]:
        """
        设计Bessel滤波器

        Args:
            cutoff: 截止频率
            filter_type: 滤波器类型
            order: 滤波器阶数
            output: 输出格式

        Returns:
            滤波器系数
        """
        nyquist = self.sampling_rate / 2
        if isinstance(cutoff, tuple):
            normalized_cutoff = (cutoff[0] / nyquist, cutoff[1] / nyquist)
        else:
            normalized_cutoff = cutoff / nyquist

        if output == 'sos':
            sos = bessel(
                order,
                normalized_cutoff,
                btype=filter_type,
                analog=False,
                output='sos',
                norm='phase'
            )
            logger.debug(f"Bessel滤波器设计完成: {filter_type}, order={order}")
            return sos
        else:
            b, a = bessel(
                order,
                normalized_cutoff,
                btype=filter_type,
                analog=False,
                output='ba',
                norm='phase'
            )
            return b, a

    def design_fir(
        self,
        cutoff: Union[float, Tuple[float, float]],
        filter_type: Literal['lowpass', 'highpass', 'bandpass', 'bandstop'],
        numtaps: int = 101,
        window: str = 'hamming'
    ) -> np.ndarray:
        """
        设计FIR滤波器

        Args:
            cutoff: 截止频率
            filter_type: 滤波器类型
            numtaps: 滤波器长度（奇数）
            window: 窗函数类型

        Returns:
            FIR滤波器系数
        """
        nyquist = self.sampling_rate / 2

        if filter_type == 'lowpass':
            pass_zero = True
            cutoff_norm = cutoff / nyquist
        elif filter_type == 'highpass':
            pass_zero = False
            cutoff_norm = cutoff / nyquist
        elif filter_type == 'bandpass':
            pass_zero = False
            cutoff_norm = [cutoff[0] / nyquist, cutoff[1] / nyquist]
        elif filter_type == 'bandstop':
            pass_zero = True
            cutoff_norm = [cutoff[0] / nyquist, cutoff[1] / nyquist]
        else:
            raise ValueError(f"未知的滤波器类型: {filter_type}")

        # 确保numtaps是奇数
        if numtaps % 2 == 0:
            numtaps += 1

        fir_coeff = firwin(
            numtaps,
            cutoff_norm,
            window=window,
            pass_zero=pass_zero
        )

        logger.debug(
            f"FIR滤波器设计完成: {filter_type}, numtaps={numtaps}, window={window}"
        )

        return fir_coeff

    def apply_filter(
        self,
        signal_data: np.ndarray,
        filter_coeff: Union[np.ndarray, Tuple],
        filter_format: str = 'sos'
    ) -> np.ndarray:
        """
        应用滤波器到信号

        Args:
            signal_data: 输入信号
            filter_coeff: 滤波器系数
            filter_format: 系数格式 ('sos', 'ba', 'fir')

        Returns:
            滤波后的信号
        """
        if filter_format == 'sos':
            # 使用SOS格式（二阶段）
            filtered = sosfiltfilt(filter_coeff, signal_data)
        elif filter_format == 'ba':
            # 使用传递函数格式
            b, a = filter_coeff
            filtered = filtfilt(b, a, signal_data)
        elif filter_format == 'fir':
            # FIR滤波器
            filtered = filtfilt(filter_coeff, 1.0, signal_data)
        else:
            raise ValueError(f"未知的滤波器格式: {filter_format}")

        logger.debug(f"滤波完成: {len(signal_data)} 采样点")

        return filtered

    def get_filter_response(
        self,
        filter_coeff: Union[np.ndarray, Tuple],
        filter_format: str = 'sos',
        worN: int = 2048
    ) -> FilterResponse:
        """
        计算滤波器的频率响应

        Args:
            filter_coeff: 滤波器系数
            filter_format: 系数格式
            worN: 频率点数

        Returns:
            FilterResponse: 滤波器响应
        """
        if filter_format == 'sos':
            w, h = signal.sosfreqz(filter_coeff, worN=worN, fs=self.sampling_rate)
        elif filter_format == 'ba':
            b, a = filter_coeff
            w, h = signal.freqz(b, a, worN=worN, fs=self.sampling_rate)
        elif filter_format == 'fir':
            w, h = signal.freqz(filter_coeff, 1.0, worN=worN, fs=self.sampling_rate)
        else:
            raise ValueError(f"未知的滤波器格式: {filter_format}")

        magnitude = np.abs(h)
        phase = np.angle(h)

        return FilterResponse(
            frequencies=w,
            magnitude=magnitude,
            phase=phase
        )


class SignalProcessor:
    """
    信号预处理器

    提供各种信号预处理功能。
    """

    @staticmethod
    def detrend_signal(
        signal_data: np.ndarray,
        detrend_type: Literal['linear', 'constant'] = 'linear'
    ) -> np.ndarray:
        """
        去趋势

        Args:
            signal_data: 输入信号
            detrend_type: 去趋势类型

        Returns:
            去趋势后的信号
        """
        detrended = detrend(signal_data, type=detrend_type)
        logger.debug(f"去趋势完成: {detrend_type}")
        return detrended

    @staticmethod
    def normalize_signal(
        signal_data: np.ndarray,
        method: Literal['zscore', 'minmax', 'peak'] = 'zscore'
    ) -> np.ndarray:
        """
        归一化信号

        Args:
            signal_data: 输入信号
            method: 归一化方法
                - 'zscore': Z-score归一化 (均值0, 标准差1)
                - 'minmax': 最小-最大归一化 [0, 1]
                - 'peak': 峰值归一化 [-1, 1]

        Returns:
            归一化后的信号
        """
        if method == 'zscore':
            mean = np.mean(signal_data)
            std = np.std(signal_data)
            normalized = (signal_data - mean) / (std + 1e-10)
        elif method == 'minmax':
            min_val = np.min(signal_data)
            max_val = np.max(signal_data)
            normalized = (signal_data - min_val) / (max_val - min_val + 1e-10)
        elif method == 'peak':
            peak_val = np.max(np.abs(signal_data))
            normalized = signal_data / (peak_val + 1e-10)
        else:
            raise ValueError(f"未知的归一化方法: {method}")

        logger.debug(f"归一化完成: {method}")
        return normalized

    @staticmethod
    def extract_envelope(
        signal_data: np.ndarray,
        method: Literal['hilbert', 'peak'] = 'hilbert'
    ) -> np.ndarray:
        """
        提取信号包络

        Args:
            signal_data: 输入信号
            method: 提取方法
                - 'hilbert': 希尔伯特变换
                - 'peak': 峰值包络

        Returns:
            包络信号
        """
        if method == 'hilbert':
            analytic_signal = hilbert(signal_data)
            envelope = np.abs(analytic_signal)
        elif method == 'peak':
            # 简化的峰值包络（使用局部最大值插值）
            from scipy.signal import find_peaks
            from scipy.interpolate import interp1d

            peaks, _ = find_peaks(signal_data)
            if len(peaks) > 1:
                f = interp1d(
                    peaks,
                    signal_data[peaks],
                    kind='cubic',
                    fill_value='extrapolate'
                )
                envelope = f(np.arange(len(signal_data)))
                envelope = np.abs(envelope)
            else:
                envelope = np.abs(signal_data)
        else:
            raise ValueError(f"未知的包络提取方法: {method}")

        logger.debug(f"包络提取完成: {method}")
        return envelope

    @staticmethod
    def moving_average(
        signal_data: np.ndarray,
        window_size: int = 5
    ) -> np.ndarray:
        """
        移动平均滤波

        Args:
            signal_data: 输入信号
            window_size: 窗口大小

        Returns:
            平滑后的信号
        """
        # 使用numpy的卷积实现移动平均
        window = np.ones(window_size) / window_size
        smoothed = np.convolve(signal_data, window, mode='same')

        logger.debug(f"移动平均完成: window_size={window_size}")
        return smoothed

    @staticmethod
    def median_filter(
        signal_data: np.ndarray,
        kernel_size: int = 5
    ) -> np.ndarray:
        """
        中值滤波

        Args:
            signal_data: 输入信号
            kernel_size: 核大小（奇数）

        Returns:
            滤波后的信号
        """
        if kernel_size % 2 == 0:
            kernel_size += 1

        filtered = medfilt(signal_data, kernel_size=kernel_size)

        logger.debug(f"中值滤波完成: kernel_size={kernel_size}")
        return filtered

    @staticmethod
    def savitzky_golay_filter(
        signal_data: np.ndarray,
        window_length: int = 11,
        polyorder: int = 3
    ) -> np.ndarray:
        """
        Savitzky-Golay滤波

        Args:
            signal_data: 输入信号
            window_length: 窗口长度（奇数）
            polyorder: 多项式阶数

        Returns:
            滤波后的信号
        """
        if window_length % 2 == 0:
            window_length += 1

        filtered = savgol_filter(signal_data, window_length, polyorder)

        logger.debug(
            f"Savitzky-Golay滤波完成: window={window_length}, order={polyorder}"
        )
        return filtered


# ====================================
# 便捷函数
# ====================================

def apply_bandpass_filter(
    signal_data: np.ndarray,
    lowcut: float,
    highcut: float,
    sampling_rate: int = SAMPLING_RATE,
    order: int = 4
) -> np.ndarray:
    """
    便捷函数：应用带通滤波器

    Args:
        signal_data: 输入信号
        lowcut: 下限截止频率（Hz）
        highcut: 上限截止频率（Hz）
        sampling_rate: 采样率
        order: 滤波器阶数

    Returns:
        滤波后的信号
    """
    filter_obj = DigitalFilter(sampling_rate)
    sos = filter_obj.design_butterworth((lowcut, highcut), 'bandpass', order)
    filtered = filter_obj.apply_filter(signal_data, sos, 'sos')
    return filtered


def apply_lowpass_filter(
    signal_data: np.ndarray,
    cutoff: float,
    sampling_rate: int = SAMPLING_RATE,
    order: int = 4
) -> np.ndarray:
    """
    便捷函数：应用低通滤波器

    Args:
        signal_data: 输入信号
        cutoff: 截止频率（Hz）
        sampling_rate: 采样率
        order: 滤波器阶数

    Returns:
        滤波后的信号
    """
    filter_obj = DigitalFilter(sampling_rate)
    sos = filter_obj.design_butterworth(cutoff, 'lowpass', order)
    filtered = filter_obj.apply_filter(signal_data, sos, 'sos')
    return filtered


def apply_highpass_filter(
    signal_data: np.ndarray,
    cutoff: float,
    sampling_rate: int = SAMPLING_RATE,
    order: int = 4
) -> np.ndarray:
    """
    便捷函数：应用高通滤波器

    Args:
        signal_data: 输入信号
        cutoff: 截止频率（Hz）
        sampling_rate: 采样率
        order: 滤波器阶数

    Returns:
        滤波后的信号
    """
    filter_obj = DigitalFilter(sampling_rate)
    sos = filter_obj.design_butterworth(cutoff, 'highpass', order)
    filtered = filter_obj.apply_filter(signal_data, sos, 'sos')
    return filtered


def denoise_signal(
    signal_data: np.ndarray,
    method: Literal['savgol', 'median', 'moving_avg'] = 'savgol',
    **kwargs
) -> np.ndarray:
    """
    便捷函数：信号去噪

    Args:
        signal_data: 输入信号
        method: 去噪方法
        **kwargs: 方法相关参数

    Returns:
        去噪后的信号
    """
    processor = SignalProcessor()

    if method == 'savgol':
        return processor.savitzky_golay_filter(signal_data, **kwargs)
    elif method == 'median':
        return processor.median_filter(signal_data, **kwargs)
    elif method == 'moving_avg':
        return processor.moving_average(signal_data, **kwargs)
    else:
        raise ValueError(f"未知的去噪方法: {method}")


# ====================================
# 测试代码
# ====================================

if __name__ == "__main__":
    print("=" * 60)
    print("数字滤波器测试")
    print("=" * 60)

    # 加载测试数据
    print("\n【准备】加载测试数据...")
    from data.loader import DataLoader

    loader = DataLoader(validate=False)
    signal_obj = loader.load('light_wear', 'normal', 10, 'A', 'X')

    if signal_obj is None:
        print("❌ 无法加载测试数据")
        exit(1)

    # 使用前5秒数据
    test_duration = 5
    test_samples = test_duration * signal_obj.sampling_rate
    test_signal = signal_obj.time_series[:test_samples]

    print(f"   数据长度: {len(test_signal)} 采样点")
    print(f"   采样率: {signal_obj.sampling_rate} Hz")
    print(f"   原始信号: 均值={np.mean(test_signal):.6f}, 标准差={np.std(test_signal):.6f}")

    # 创建滤波器对象
    digital_filter = DigitalFilter(signal_obj.sampling_rate)

    # 测试1: Butterworth带通滤波器
    print("\n【测试1】Butterworth带通滤波器:")
    sos_bandpass = digital_filter.design_butterworth(
        cutoff=(100, 3000),
        filter_type='bandpass',
        order=4
    )
    filtered_bandpass = digital_filter.apply_filter(test_signal, sos_bandpass, 'sos')
    print(f"   滤波后: 均值={np.mean(filtered_bandpass):.6f}, 标准差={np.std(filtered_bandpass):.6f}")

    # 测试2: Butterworth低通滤波器
    print("\n【测试2】Butterworth低通滤波器:")
    sos_lowpass = digital_filter.design_butterworth(
        cutoff=1000,
        filter_type='lowpass',
        order=6
    )
    filtered_lowpass = digital_filter.apply_filter(test_signal, sos_lowpass, 'sos')
    print(f"   滤波后: 均值={np.mean(filtered_lowpass):.6f}, 标准差={np.std(filtered_lowpass):.6f}")

    # 测试3: FIR滤波器
    print("\n【测试3】FIR带通滤波器:")
    fir_coeff = digital_filter.design_fir(
        cutoff=(100, 3000),
        filter_type='bandpass',
        numtaps=101,
        window='hamming'
    )
    filtered_fir = digital_filter.apply_filter(test_signal, fir_coeff, 'fir')
    print(f"   滤波器长度: {len(fir_coeff)}")
    print(f"   滤波后: 均值={np.mean(filtered_fir):.6f}, 标准差={np.std(filtered_fir):.6f}")

    # 测试4: 滤波器频率响应
    print("\n【测试4】滤波器频率响应:")
    response = digital_filter.get_filter_response(sos_bandpass, 'sos', worN=2048)
    print(response)
    max_mag_idx = np.argmax(response.magnitude)
    print(f"   最大增益频率: {response.frequencies[max_mag_idx]:.2f} Hz")

    # 测试5: 信号预处理
    print("\n【测试5】信号预处理:")
    processor = SignalProcessor()

    # 去趋势
    detrended = processor.detrend_signal(test_signal, 'linear')
    print(f"   去趋势: 均值={np.mean(detrended):.6f}")

    # 归一化
    normalized_zscore = processor.normalize_signal(test_signal, 'zscore')
    normalized_minmax = processor.normalize_signal(test_signal, 'minmax')
    print(f"   Z-score归一化: 均值={np.mean(normalized_zscore):.6f}, 标准差={np.std(normalized_zscore):.6f}")
    print(f"   MinMax归一化: 范围=[{np.min(normalized_minmax):.6f}, {np.max(normalized_minmax):.6f}]")

    # 包络提取
    envelope = processor.extract_envelope(test_signal, 'hilbert')
    print(f"   包络提取: 最大值={np.max(envelope):.6f}")

    # 测试6: 去噪方法
    print("\n【测试6】去噪方法对比:")

    # 移动平均
    smoothed_ma = processor.moving_average(test_signal, window_size=11)
    print(f"   移动平均: 标准差={np.std(smoothed_ma):.6f}")

    # 中值滤波
    smoothed_median = processor.median_filter(test_signal, kernel_size=11)
    print(f"   中值滤波: 标准差={np.std(smoothed_median):.6f}")

    # Savitzky-Golay
    smoothed_savgol = processor.savitzky_golay_filter(
        test_signal,
        window_length=21,
        polyorder=3
    )
    print(f"   Savitzky-Golay: 标准差={np.std(smoothed_savgol):.6f}")

    # 测试7: 便捷函数
    print("\n【测试7】便捷函数测试:")
    filtered_bp = apply_bandpass_filter(test_signal, 100, 3000, signal_obj.sampling_rate)
    filtered_lp = apply_lowpass_filter(test_signal, 1000, signal_obj.sampling_rate)
    filtered_hp = apply_highpass_filter(test_signal, 50, signal_obj.sampling_rate)
    denoised = denoise_signal(test_signal, method='savgol', window_length=21, polyorder=3)

    print(f"   ✅ apply_bandpass_filter: 标准差={np.std(filtered_bp):.6f}")
    print(f"   ✅ apply_lowpass_filter: 标准差={np.std(filtered_lp):.6f}")
    print(f"   ✅ apply_highpass_filter: 标准差={np.std(filtered_hp):.6f}")
    print(f"   ✅ denoise_signal: 标准差={np.std(denoised):.6f}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
