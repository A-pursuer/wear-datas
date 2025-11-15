"""
缓存管理器

实现多级缓存系统：
- L1缓存: 内存缓存（LRU策略）
- L2缓存: 磁盘缓存
- 缓存统计和监控
"""

import os
import pickle
import time
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional, Dict, List
from dataclasses import dataclass
import hashlib

from config.settings import (
    ENABLE_CACHE,
    MAX_CACHE_SIZE_MB,
    L1_CACHE_MAX_ITEMS,
    ENABLE_DISK_CACHE,
    DISK_CACHE_DIR,
    CACHE_STRATEGY,
    logger
)


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    data: Any
    size: int  # 字节
    timestamp: float  # 创建时间
    access_count: int = 0  # 访问次数
    last_access: float = None  # 最后访问时间

    def __post_init__(self):
        if self.last_access is None:
            self.last_access = self.timestamp


class LRUCache:
    """
    LRU (Least Recently Used) 缓存

    使用OrderedDict实现，线程安全。
    """

    def __init__(self, max_size_mb: int = MAX_CACHE_SIZE_MB):
        """
        初始化LRU缓存

        Args:
            max_size_mb: 最大内存占用（MB）
        """
        self.max_size = max_size_mb * 1024 * 1024  # 转换为字节
        self.current_size = 0
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.Lock()

        # 统计信息
        self.hits = 0
        self.misses = 0
        self.evictions = 0

        logger.info(f"LRUCache 初始化: 最大{max_size_mb}MB")

    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存数据

        Args:
            key: 缓存键

        Returns:
            缓存的数据，未找到返回None
        """
        with self.lock:
            if key in self.cache:
                # 命中：移到最后（最近使用）
                entry = self.cache[key]
                self.cache.move_to_end(key)

                # 更新访问信息
                entry.access_count += 1
                entry.last_access = time.time()

                self.hits += 1
                logger.debug(f"缓存命中: {key}")
                return entry.data
            else:
                self.misses += 1
                logger.debug(f"缓存未命中: {key}")
                return None

    def put(self, key: str, data: Any) -> bool:
        """
        存入缓存

        Args:
            key: 缓存键
            data: 要缓存的数据

        Returns:
            bool: 是否成功缓存
        """
        with self.lock:
            # 计算数据大小
            data_size = self._get_data_size(data)

            # 如果数据太大，拒绝缓存
            if data_size > self.max_size * 0.5:
                logger.warning(
                    f"数据过大，拒绝缓存: {data_size/1024/1024:.2f}MB > "
                    f"{self.max_size/1024/1024*0.5:.2f}MB"
                )
                return False

            # 如果键已存在，先删除旧数据
            if key in self.cache:
                old_entry = self.cache[key]
                self.current_size -= old_entry.size
                del self.cache[key]

            # 清理空间
            while (self.current_size + data_size > self.max_size
                   and len(self.cache) > 0):
                self._evict_lru()

            # 添加新数据
            entry = CacheEntry(
                key=key,
                data=data,
                size=data_size,
                timestamp=time.time()
            )

            self.cache[key] = entry
            self.current_size += data_size

            logger.debug(
                f"缓存存入: {key} ({data_size/1024:.1f}KB), "
                f"总计: {len(self.cache)}项, {self.current_size/1024/1024:.1f}MB"
            )

            return True

    def _evict_lru(self):
        """淘汰最久未使用的数据"""
        if self.cache:
            key, entry = self.cache.popitem(last=False)  # FIFO: 删除最旧的
            self.current_size -= entry.size
            self.evictions += 1

            logger.debug(
                f"缓存淘汰: {key} ({entry.size/1024:.1f}KB), "
                f"访问{entry.access_count}次"
            )

    @staticmethod
    def _get_data_size(data: Any) -> int:
        """估算数据大小（字节）"""
        try:
            import sys
            import numpy as np

            if isinstance(data, np.ndarray):
                return data.nbytes
            elif hasattr(data, '__sizeof__'):
                return sys.getsizeof(data)
            else:
                # 使用pickle序列化估算
                return len(pickle.dumps(data))
        except Exception as e:
            logger.warning(f"无法估算数据大小: {e}")
            return 0

    def clear(self):
        """清空缓存"""
        with self.lock:
            count = len(self.cache)
            size_mb = self.current_size / 1024 / 1024

            self.cache.clear()
            self.current_size = 0

            logger.info(f"缓存已清空: {count}项, {size_mb:.2f}MB")

    def get_stats(self) -> Dict:
        """
        获取缓存统计信息

        Returns:
            dict: 统计信息
        """
        total_accesses = self.hits + self.misses
        hit_rate = self.hits / total_accesses if total_accesses > 0 else 0

        return {
            'items': len(self.cache),
            'size_mb': self.current_size / 1024 / 1024,
            'max_size_mb': self.max_size / 1024 / 1024,
            'usage_percent': (self.current_size / self.max_size * 100) if self.max_size > 0 else 0,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'evictions': self.evictions
        }

    def __len__(self) -> int:
        return len(self.cache)

    def __contains__(self, key: str) -> bool:
        return key in self.cache


class DiskCache:
    """
    磁盘缓存

    将数据序列化到磁盘以节省内存。
    """

    def __init__(self, cache_dir: Path = DISK_CACHE_DIR):
        """
        初始化磁盘缓存

        Args:
            cache_dir: 缓存目录
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"DiskCache 初始化: {self.cache_dir}")

    def get(self, key: str) -> Optional[Any]:
        """
        从磁盘读取缓存

        Args:
            key: 缓存键

        Returns:
            缓存的数据
        """
        cache_file = self._get_cache_file(key)

        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)

                logger.debug(f"磁盘缓存命中: {key}")
                return data
            except Exception as e:
                logger.error(f"读取磁盘缓存失败: {key}, {e}")
                return None
        else:
            logger.debug(f"磁盘缓存未命中: {key}")
            return None

    def put(self, key: str, data: Any) -> bool:
        """
        写入磁盘缓存

        Args:
            key: 缓存键
            data: 数据

        Returns:
            bool: 是否成功
        """
        cache_file = self._get_cache_file(key)

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)

            logger.debug(f"磁盘缓存存入: {key}")
            return True
        except Exception as e:
            logger.error(f"写入磁盘缓存失败: {key}, {e}")
            return False

    def _get_cache_file(self, key: str) -> Path:
        """生成缓存文件路径"""
        # 使用MD5哈希作为文件名（避免特殊字符）
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.pkl"

    def clear(self):
        """清空磁盘缓存"""
        count = 0
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
                count += 1
            except Exception as e:
                logger.error(f"删除缓存文件失败: {cache_file}, {e}")

        logger.info(f"磁盘缓存已清空: {count}个文件")

    def get_size_mb(self) -> float:
        """获取磁盘缓存大小（MB）"""
        total_size = sum(
            f.stat().st_size for f in self.cache_dir.glob("*.pkl")
        )
        return total_size / 1024 / 1024


class HierarchicalCache:
    """
    多级缓存系统

    整合内存缓存(L1)和磁盘缓存(L2)。
    """

    def __init__(
        self,
        enable_l1: bool = ENABLE_CACHE,
        enable_l2: bool = ENABLE_DISK_CACHE,
        max_l1_size_mb: int = MAX_CACHE_SIZE_MB
    ):
        """
        初始化多级缓存

        Args:
            enable_l1: 是否启用L1缓存（内存）
            enable_l2: 是否启用L2缓存（磁盘）
            max_l1_size_mb: L1缓存最大内存
        """
        self.enable_l1 = enable_l1
        self.enable_l2 = enable_l2

        self.l1_cache = LRUCache(max_l1_size_mb) if enable_l1 else None
        self.l2_cache = DiskCache() if enable_l2 else None

        logger.info(
            f"HierarchicalCache 初始化: "
            f"L1={'ON' if enable_l1 else 'OFF'}, "
            f"L2={'ON' if enable_l2 else 'OFF'}"
        )

    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存（三级查找）

        Args:
            key: 缓存键

        Returns:
            缓存的数据
        """
        # L1: 内存缓存
        if self.enable_l1 and self.l1_cache:
            data = self.l1_cache.get(key)
            if data is not None:
                return data

        # L2: 磁盘缓存
        if self.enable_l2 and self.l2_cache:
            data = self.l2_cache.get(key)
            if data is not None:
                # 提升到L1缓存
                if self.enable_l1 and self.l1_cache:
                    self.l1_cache.put(key, data)
                return data

        return None

    def put(self, key: str, data: Any):
        """
        存入缓存

        Args:
            key: 缓存键
            data: 数据
        """
        # 存入L1缓存
        if self.enable_l1 and self.l1_cache:
            self.l1_cache.put(key, data)

        # 同时备份到L2缓存
        if self.enable_l2 and self.l2_cache:
            self.l2_cache.put(key, data)

    def clear(self):
        """清空所有缓存"""
        if self.l1_cache:
            self.l1_cache.clear()
        if self.l2_cache:
            self.l2_cache.clear()

        logger.info("所有缓存已清空")

    def get_stats(self) -> Dict:
        """获取缓存统计"""
        stats = {}

        if self.l1_cache:
            stats['l1'] = self.l1_cache.get_stats()

        if self.l2_cache:
            stats['l2'] = {
                'size_mb': self.l2_cache.get_size_mb()
            }

        return stats


# ====================================
# 全局缓存实例
# ====================================

# 创建全局缓存管理器
_global_cache = HierarchicalCache()


def get_cache() -> HierarchicalCache:
    """获取全局缓存实例"""
    return _global_cache


def clear_all_caches():
    """清空所有缓存"""
    _global_cache.clear()


# ====================================
# 测试代码
# ====================================

if __name__ == "__main__":
    import numpy as np

    print("=" * 60)
    print("缓存管理器测试")
    print("=" * 60)

    # 测试1: LRU缓存
    print("\n【测试1】LRU缓存:")
    lru = LRUCache(max_size_mb=10)  # 10MB限制

    # 存入一些数据
    for i in range(5):
        key = f"data_{i}"
        data = np.random.rand(1000, 100)  # 约0.8MB
        lru.put(key, data)

    stats = lru.get_stats()
    print(f"   缓存项数: {stats['items']}")
    print(f"   内存占用: {stats['size_mb']:.2f} MB")
    print(f"   使用率: {stats['usage_percent']:.1f}%")

    # 测试缓存命中
    data = lru.get("data_0")
    print(f"   获取data_0: {'✅ 命中' if data is not None else '❌ 未命中'}")

    print(f"   命中率: {stats['hit_rate']*100:.1f}%")

    # 测试2: 磁盘缓存
    print("\n【测试2】磁盘缓存:")
    disk = DiskCache()

    test_data = np.random.rand(100, 100)
    disk.put("test_disk", test_data)

    loaded_data = disk.get("test_disk")
    print(f"   写入并读取: {'✅ 成功' if loaded_data is not None else '❌ 失败'}")
    print(f"   磁盘缓存大小: {disk.get_size_mb():.2f} MB")

    # 测试3: 多级缓存
    print("\n【测试3】多级缓存:")
    cache = HierarchicalCache(enable_l1=True, enable_l2=True, max_l1_size_mb=5)

    # 存入数据
    for i in range(3):
        data = np.random.rand(500, 100)
        cache.put(f"multi_{i}", data)

    # 获取数据
    data = cache.get("multi_1")
    print(f"   获取multi_1: {'✅ 成功' if data is not None else '❌ 失败'}")

    # 统计
    stats = cache.get_stats()
    if 'l1' in stats:
        print(f"   L1缓存: {stats['l1']['items']}项, {stats['l1']['size_mb']:.2f}MB")
    if 'l2' in stats:
        print(f"   L2缓存: {stats['l2']['size_mb']:.2f}MB")

    # 测试4: 缓存淘汰
    print("\n【测试4】缓存淘汰测试:")
    small_cache = LRUCache(max_size_mb=2)  # 2MB限制

    for i in range(10):
        data = np.random.rand(500, 100)  # 约0.4MB
        small_cache.put(f"evict_{i}", data)

    stats = small_cache.get_stats()
    print(f"   存入10项后，缓存项数: {stats['items']}")
    print(f"   淘汰次数: {stats['evictions']}")

    # 清理
    print("\n【清理】清空所有缓存...")
    cache.clear()
    disk.clear()

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
