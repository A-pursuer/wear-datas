# 齿轮磨损数据分析系统 - 行动计划

## 📋 项目概述

**项目名称**: 齿轮磨损数据分析系统
**项目目标**: 开发基于Streamlit的Web应用，用于齿轮磨损振动信号的分析和可视化
**当前状态**: 设计完成 (v1.0.0)
**目标版本**: v1.1.0 (功能实现版)

---

## 🎯 开发策略

### 核心原则
1. **自底向上**: 先实现基础设施，再构建上层功能
2. **渐进式开发**: 每个阶段都有可运行的里程碑
3. **测试驱动**: 关键模块先写测试用例
4. **文档同步**: 代码和文档同步更新

### 技术栈
- **后端**: Python 3.8+
- **数据处理**: Pandas, NumPy
- **信号处理**: SciPy, PyWavelets
- **可视化**: Plotly
- **界面**: Streamlit
- **测试**: Pytest

---

## 📅 开发阶段

总计 **8个阶段**，**39个具体任务**

### Phase 1: 项目基础设施搭建 (优先级: 🔴 最高)
**目标**: 建立项目骨架和开发环境
**预计时间**: 2-3小时
**依赖**: 无

#### 任务清单

| ID | 任务 | 产出 | 估时 |
|---|---|---|---|
| 1.1 | 创建项目目录结构 | 完整的文件夹结构 | 30min |
| 1.2 | 配置requirements.txt | 依赖清单 | 30min |
| 1.3 | 创建配置管理模块 | config/settings.py | 1h |
| 1.4 | 创建项目启动脚本 | run.sh / run.bat | 30min |

#### 详细规划

**1.1 创建项目目录结构**
```
wear-datas/
├── config/              # 配置模块
│   ├── __init__.py
│   └── settings.py      # 全局配置
├── data/                # 数据处理模块
│   ├── __init__.py
│   ├── csv_parser.py    # CSV解析
│   ├── loader.py        # 数据加载
│   ├── cache_manager.py # 缓存管理
│   ├── filename_parser.py # 文件名解析
│   └── validator.py     # 数据验证
├── processing/          # 信号处理模块
│   ├── __init__.py
│   ├── time_domain.py   # 时域分析
│   ├── frequency_analyzer.py # 频域分析
│   ├── timefreq_analyzer.py  # 时频分析
│   ├── filters.py       # 数字滤波
│   └── gear_analyzer.py # 齿轮特征分析
├── visualization/       # 可视化模块
│   ├── __init__.py
│   ├── time_domain_viz.py    # 时域可视化
│   ├── frequency_viz.py      # 频域可视化
│   ├── comparison_viz.py     # 对比分析
│   └── performance_optimizer.py # 性能优化
├── ui/                  # 用户界面模块
│   ├── __init__.py
│   ├── layout_manager.py     # 布局管理
│   ├── state_manager.py      # 状态管理
│   ├── parameter_selector.py # 参数选择器
│   ├── dataset_manager.py    # 数据集管理
│   └── chart_container.py    # 图表容器
├── tests/               # 测试模块
│   ├── __init__.py
│   ├── test_data/       # 测试数据
│   ├── test_loader.py
│   ├── test_processor.py
│   └── test_visualizer.py
├── cache/               # 缓存目录（.gitignore）
├── logs/                # 日志目录（.gitignore）
├── main.py              # 主应用入口
├── requirements.txt     # 依赖清单
├── .gitignore
└── README.md
```

**1.2 requirements.txt 依赖清单**
```txt
# 核心框架
streamlit>=1.28.0
plotly>=5.17.0

# 数据处理
pandas>=2.0.0
numpy>=1.24.0

# 信号处理
scipy>=1.11.0
PyWavelets>=1.4.1

# 性能优化
numba>=0.58.0

# 工具库
python-dateutil>=2.8.2
pytz>=2023.3

# 开发工具
pytest>=7.4.0
black>=23.0.0
flake8>=6.1.0

# 系统监控
psutil>=5.9.0
```

**1.3 config/settings.py 示例**
```python
from pathlib import Path
from dataclasses import dataclass

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据目录
DATA_DIR = PROJECT_ROOT
CSV_PATTERN = "*.csv"

# 缓存配置
CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)
ENABLE_CACHE = True
MAX_CACHE_SIZE_MB = 500

# 采样参数
SAMPLING_RATE = 15360  # Hz
SAMPLING_DURATION = 30  # seconds
EXPECTED_DATA_POINTS = 460805

# 传感器配置
SENSORS = ['A', 'B', 'C']
AXES = ['X', 'Y', 'Z']
SENSOR_INFO = {
    'A': '主动轴输入轴承处',
    'B': '从动轴输入处',
    'C': '从动轴输出处'
}

# 磨损状态映射
WEAR_STATE_MAP = {
    '正常': 'normal',
    '轻磨': 'light_wear',
    '重磨': 'heavy_wear'
}

# 可视化配置
CHART_THEME = 'plotly_white'
MAX_DISPLAY_POINTS = 10000
DOWNSAMPLE_METHOD = 'lttb'

# 日志配置
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_LEVEL = "INFO"
```

---

### Phase 2: 数据处理层实现 (优先级: 🔴 最高)
**目标**: 实现数据加载、解析和缓存功能
**预计时间**: 8-10小时
**依赖**: Phase 1

#### 任务清单

| ID | 任务 | 关键类/函数 | 估时 |
|---|---|---|---|
| 2.1 | CSV解析器 | CSVParser, DataValidator | 2h |
| 2.2 | 数据加载器 | DataLoader, LazyLoader | 2h |
| 2.3 | 缓存管理器 | LRUCache, HierarchicalCache | 2h |
| 2.4 | 文件名解析器 | FilenameParser, FileConfig | 1.5h |
| 2.5 | 数据验证器 | DataValidator | 1.5h |
| 2.6 | 单元测试 | test_loader.py | 1h |

#### 实现优先级
1. **FilenameParser** (最基础) → 识别数据文件
2. **CSVParser** → 解析CSV结构
3. **DataLoader** → 加载数据
4. **CacheManager** → 优化性能
5. **DataValidator** → 保证质量

#### 验收标准
- ✅ 能够正确解析所有6个CSV文件
- ✅ 支持按通道、时间范围加载数据
- ✅ 缓存命中率 > 80%
- ✅ 数据验证通过率 100%

---

### Phase 3: 信号处理层实现 (优先级: 🟠 高)
**目标**: 实现时域、频域、时频分析功能
**预计时间**: 12-15小时
**依赖**: Phase 2

#### 任务清单

| ID | 任务 | 关键算法 | 估时 |
|---|---|---|---|
| 3.1 | 时域特征提取 | RMS, Peak, Skewness, Kurtosis | 2.5h |
| 3.2 | 频域分析器 | FFT, PSD, Welch | 3h |
| 3.3 | 时频分析器 | CWT, STFT, Spectrogram | 3h |
| 3.4 | 数字滤波器 | Butterworth, Bandpass | 2h |
| 3.5 | 齿轮特征分析 | 啮合频率、谐波检测 | 2.5h |
| 3.6 | 单元测试 | test_processor.py | 1h |

#### 技术难点
1. **FFT性能优化** - 使用NumPy FFT + 窗函数
2. **小波变换** - PyWavelets库集成
3. **齿轮频率识别** - 峰值检测 + 频率匹配

#### 验收标准
- ✅ 时域特征提取准确（与Matlab对比）
- ✅ FFT频谱正确（能识别主要频率成分）
- ✅ 齿轮啮合频率自动识别成功率 > 90%

---

### Phase 4: 可视化层实现 (优先级: 🟠 高)
**目标**: 实现交互式图表和性能优化
**预计时间**: 10-12小时
**依赖**: Phase 3

#### 任务清单

| ID | 任务 | 图表类型 | 估时 |
|---|---|---|---|
| 4.1 | 时域可视化器 | 时域波形、统计特征 | 2.5h |
| 4.2 | 频域可视化器 | 频谱图、时频谱 | 3h |
| 4.3 | 对比分析可视化 | 多工况对比、雷达图 | 3h |
| 4.4 | 性能优化器 | LTTB降采样、增量更新 | 2.5h |

#### 关键技术
- **Plotly图表**: go.Scatter, go.Heatmap, make_subplots
- **降采样算法**: LTTB保持视觉特征
- **交互功能**: 缩放同步、交叉筛选

#### 验收标准
- ✅ 图表渲染时间 < 2秒（46万点数据）
- ✅ 支持至少4个数据集同时对比
- ✅ 缩放、平移操作流畅（60fps）

---

### Phase 5: UI界面层实现 (优先级: 🟡 中)
**目标**: 实现用户交互界面
**预计时间**: 10-12小时
**依赖**: Phase 4

#### 任务清单

| ID | 任务 | 组件 | 估时 |
|---|---|---|---|
| 5.1 | 布局管理器 | 页面布局、标签页 | 2h |
| 5.2 | 状态管理器 | Session State管理 | 1.5h |
| 5.3 | 参数选择器 | 齿轮状态、工况选择 | 3h |
| 5.4 | 数据集管理器 | 多数据集对比管理 | 2.5h |
| 5.5 | 图表容器 | 图表展示、控制面板 | 2h |

#### UI流程设计
```
侧边栏
├── 齿轮配置选择
├── 实验工况选择
├── 传感器配置选择
├── 分析参数设置
└── 数据集管理

主内容区
├── Tab 1: 时域分析
├── Tab 2: 频域分析
├── Tab 3: 统计特征
└── Tab 4: 数据导出
```

#### 验收标准
- ✅ 所有参数选择器工作正常
- ✅ 状态在页面刷新后保持
- ✅ 支持添加、删除、隐藏数据集
- ✅ 界面响应时间 < 500ms

---

### Phase 6: 主应用集成 (优先级: 🟡 中)
**目标**: 集成所有模块，实现完整工作流
**预计时间**: 4-5小时
**依赖**: Phase 5

#### 任务清单

| ID | 任务 | 描述 | 估时 |
|---|---|---|---|
| 6.1 | 实现main.py | 主应用入口 | 2h |
| 6.2 | 模块集成 | 连接各层模块 | 1.5h |
| 6.3 | 数据流测试 | 端到端测试 | 1.5h |

#### 集成检查清单
- [ ] 数据加载 → 信号处理 → 可视化 完整流程
- [ ] 参数变化 → 自动更新图表
- [ ] 缓存机制正常工作
- [ ] 错误处理和用户提示

---

### Phase 7: 测试与优化 (优先级: 🟢 中低)
**目标**: 保证质量和性能
**预计时间**: 6-8小时
**依赖**: Phase 6

#### 任务清单

| ID | 任务 | 测试类型 | 估时 |
|---|---|---|---|
| 7.1 | 单元测试 | 关键函数测试 | 2h |
| 7.2 | 性能测试 | 加载速度、内存使用 | 2h |
| 7.3 | 用户体验测试 | 交互流畅度、易用性 | 2h |
| 7.4 | Bug修复 | 修复发现的问题 | 2h |

#### 性能指标
- **数据加载**: < 3秒（单文件）
- **FFT计算**: < 1秒（46万点）
- **图表渲染**: < 2秒
- **内存占用**: < 1GB（4个数据集）

---

### Phase 8: 文档完善 (优先级: 🟢 低)
**目标**: 编写用户和开发者文档
**预计时间**: 4-6小时
**依赖**: Phase 7

#### 任务清单

| ID | 任务 | 文档类型 | 估时 |
|---|---|---|---|
| 8.1 | 用户使用手册 | 操作指南、FAQ | 2h |
| 8.2 | 开发者文档 | API文档、架构说明 | 2h |
| 8.3 | 代码注释 | Docstring完善 | 2h |

---

## 🔄 依赖关系图

```
Phase 1 (基础设施)
    ↓
Phase 2 (数据层) ←─────┐
    ↓                  │
Phase 3 (处理层)       │
    ↓                  │
Phase 4 (可视化层)     │
    ↓                  │
Phase 5 (UI层) ────────┘
    ↓
Phase 6 (集成)
    ↓
Phase 7 (测试)
    ↓
Phase 8 (文档)
```

---

## 📊 里程碑定义

### Milestone 1: 数据可读 (M1)
**完成标志**: 能够加载和显示原始数据
**包含阶段**: Phase 1, 2
**交付物**:
- ✅ 项目结构完整
- ✅ 能加载所有CSV文件
- ✅ 基本的数据验证

### Milestone 2: 分析可用 (M2)
**完成标志**: 能够进行基本的信号分析
**包含阶段**: Phase 3
**交付物**:
- ✅ 时域特征提取
- ✅ FFT频域分析
- ✅ 基本滤波功能

### Milestone 3: 可视化完成 (M3)
**完成标志**: 能够生成交互式图表
**包含阶段**: Phase 4
**交付物**:
- ✅ 时域波形图
- ✅ 频谱图
- ✅ 对比分析图

### Milestone 4: 系统可用 (M4)
**完成标志**: 完整的Web应用可运行
**包含阶段**: Phase 5, 6
**交付物**:
- ✅ Streamlit界面
- ✅ 完整工作流
- ✅ 用户操作正常

### Milestone 5: 生产就绪 (M5)
**完成标志**: 系统经过测试和优化
**包含阶段**: Phase 7, 8
**交付物**:
- ✅ 测试覆盖率 > 80%
- ✅ 性能达标
- ✅ 文档完整

---

## ⏱️ 时间估算

| 阶段 | 预计时间 | 累计时间 |
|------|---------|---------|
| Phase 1 | 2-3h | 3h |
| Phase 2 | 8-10h | 13h |
| Phase 3 | 12-15h | 28h |
| Phase 4 | 10-12h | 40h |
| Phase 5 | 10-12h | 52h |
| Phase 6 | 4-5h | 57h |
| Phase 7 | 6-8h | 65h |
| Phase 8 | 4-6h | 71h |

**总计**: 约 **60-71小时** (7.5-9个工作日)

### 开发节奏建议
- **全职开发**: 2周完成
- **兼职开发**: 4-6周完成
- **每日建议**: 4-6小时深度工作

---

## ⚠️ 风险管理

### 技术风险

| 风险 | 影响 | 概率 | 应对策略 |
|------|------|------|---------|
| 大数据量性能问题 | 高 | 中 | 实现多级缓存+降采样 |
| FFT计算慢 | 中 | 低 | 使用NumPy优化版本 |
| Streamlit内存泄漏 | 高 | 低 | 实时监控+主动释放 |
| 小波变换复杂 | 中 | 中 | 使用成熟库PyWavelets |

### 项目风险

| 风险 | 影响 | 概率 | 应对策略 |
|------|------|------|---------|
| 需求变更 | 中 | 中 | 模块化设计便于调整 |
| 时间超期 | 中 | 中 | 分阶段交付，优先核心功能 |
| 依赖库兼容性 | 低 | 低 | 固定版本号 |

---

## 🎯 优先级策略

### MVP (最小可行产品) 包含:
1. ✅ 数据加载 (Phase 2)
2. ✅ 时域分析 (Phase 3.1)
3. ✅ 基本可视化 (Phase 4.1)
4. ✅ 简单UI (Phase 5部分)

**MVP交付时间**: ~30小时

### 完整版本包含:
- 所有8个阶段全部功能
- **完整版交付时间**: ~71小时

---

## 📝 开发规范

### 代码规范
- 遵循 **PEP 8**
- 使用 **类型提示** (Type Hints)
- 函数/类必须有 **Docstring**
- 变量命名清晰、有意义

### Git提交规范
```
feat: 添加时域特征提取功能
fix: 修复缓存管理器内存泄漏
docs: 更新API文档
test: 添加数据加载器单元测试
refactor: 重构文件名解析器
perf: 优化FFT计算性能
```

### 分支策略
- `main`: 稳定版本
- `develop`: 开发主分支
- `feature/xxx`: 功能分支
- `bugfix/xxx`: 修复分支

---

## 🚀 快速开始指南

### 立即开始开发

**第一步**: 搭建基础设施
```bash
# 1. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 2. 安装依赖（待创建requirements.txt）
pip install -r requirements.txt

# 3. 创建目录结构
mkdir -p config data processing visualization ui tests cache logs
```

**第二步**: 实现第一个模块
- 从 `config/settings.py` 开始
- 然后实现 `data/filename_parser.py`
- 验证能否识别6个CSV文件

**第三步**: 迭代开发
- 按照Phase顺序逐步实现
- 每完成一个模块就进行测试
- 定期提交代码

---

## 📞 问题反馈

开发过程中遇到问题：
1. 查看设计文档 `design/` 目录
2. 查看本行动计划
3. 在GitHub Issues中提问

---

**最后更新**: 2025-11-14
**计划版本**: v1.0
**目标版本**: wear-datas v1.1.0
