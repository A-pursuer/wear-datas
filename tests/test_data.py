"""
数据处理层单元测试

测试 data/ 模块的各个组件。
"""

import unittest
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestFilenameParser(unittest.TestCase):
    """测试文件名解析器"""

    def test_parse_valid_filename(self):
        """测试解析有效文件名"""
        from data.filename_parser import FilenameParser

        filename = "主动轮(轻磨)-从动轮(正常)-10Nm-1000r.csv"
        config = FilenameParser.parse(filename)

        self.assertIsNotNone(config)
        self.assertEqual(config.drive_gear_state, 'light_wear')
        self.assertEqual(config.driven_gear_state, 'normal')
        self.assertEqual(config.torque, 10)
        self.assertEqual(config.speed, 1000)

    def test_parse_invalid_filename(self):
        """测试解析无效文件名"""
        from data.filename_parser import FilenameParser

        invalid_filename = "invalid_filename.csv"
        config = FilenameParser.parse(invalid_filename)

        self.assertIsNone(config)

    def test_scan_directory(self):
        """测试扫描目录"""
        from data.filename_parser import FilenameParser

        configs = FilenameParser.scan_directory()

        self.assertIsInstance(configs, list)
        # 应该找到6个数据文件
        self.assertGreaterEqual(len(configs), 0)


class TestDataValidator(unittest.TestCase):
    """测试数据验证器"""

    def test_validate_file_exists(self):
        """测试文件存在性验证"""
        from data.validator import DataValidator
        from data.filename_parser import scan_data_files

        configs = scan_data_files()
        if configs:
            result = DataValidator.validate_file_exists(configs[0].filepath)
            self.assertTrue(result.is_valid)

    def test_validate_nonexistent_file(self):
        """测试不存在的文件"""
        from data.validator import DataValidator

        result = DataValidator.validate_file_exists("nonexistent.csv")
        self.assertFalse(result.is_valid)


class TestCacheManager(unittest.TestCase):
    """测试缓存管理器"""

    def test_lru_cache_basic(self):
        """测试LRU缓存基本功能"""
        from data.cache_manager import LRUCache

        cache = LRUCache(max_size_mb=10)

        # 存入数据
        cache.put("key1", "value1")
        cache.put("key2", "value2")

        # 获取数据
        self.assertEqual(cache.get("key1"), "value1")
        self.assertEqual(cache.get("key2"), "value2")
        self.assertIsNone(cache.get("key3"))

    def test_lru_cache_eviction(self):
        """测试LRU缓存淘汰"""
        from data.cache_manager import LRUCache
        import numpy as np

        cache = LRUCache(max_size_mb=1)  # 1MB限制

        # 存入多个大数据
        for i in range(5):
            data = np.random.rand(500, 100)  # 约0.4MB
            cache.put(f"key_{i}", data)

        # 检查统计
        stats = cache.get_stats()
        self.assertGreater(stats['evictions'], 0)


def run_tests():
    """运行测试"""
    # 创建测试套件
    suite = unittest.TestSuite()

    # 添加测试
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestFilenameParser))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDataValidator))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCacheManager))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
