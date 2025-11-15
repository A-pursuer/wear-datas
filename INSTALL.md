# 安装指南

## 快速开始

### 1. 系统要求

- **Python**: >= 3.8
- **操作系统**: Linux, macOS, Windows
- **内存**: >= 4GB
- **硬盘空间**: >= 2GB

### 2. 安装步骤

#### Linux / macOS

```bash
# 1. 克隆或下载项目
cd wear-datas

# 2. 一键安装并启动
./run.sh

# 或者分步安装
./run.sh install  # 安装依赖
./run.sh start    # 启动应用
```

#### Windows

```cmd
# 1. 创建虚拟环境
python -m venv venv

# 2. 激活虚拟环境
venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 验证配置
python config\settings.py

# 5. 启动应用
streamlit run main.py
```

### 3. 验证安装

访问浏览器打开：http://localhost:8501

如果看到"齿轮磨损数据分析系统"界面，说明安装成功！

## 常见问题

### Q1: 提示找不到模块

**解决方案**:
```bash
# 确保在虚拟环境中
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 重新安装依赖
pip install -r requirements.txt
```

### Q2: Streamlit启动失败

**解决方案**:
```bash
# 检查端口是否被占用
lsof -i :8501  # Linux/Mac
netstat -ano | findstr :8501  # Windows

# 使用其他端口启动
streamlit run main.py --server.port 8502
```

### Q3: 内存不足

**解决方案**:
- 在 `config/settings.py` 中降低 `MAX_CACHE_SIZE_MB`
- 在 `config/settings.py` 中降低 `MAX_DISPLAY_POINTS`

### Q4: 数据加载慢

**解决方案**:
- 启用缓存: 设置 `ENABLE_CACHE = True`
- 使用降采样: 设置 `ENABLE_DOWNSAMPLE = True`

## 开发模式

```bash
# 启用调试模式
# 编辑 config/settings.py
DEBUG = True
DEV_MODE = True

# 运行测试
./run.sh test

# 清理缓存
./run.sh clean
```

## 更新依赖

```bash
# 更新所有依赖到最新版本
pip install --upgrade -r requirements.txt

# 或只更新特定包
pip install --upgrade streamlit plotly pandas
```

## 卸载

```bash
# 删除虚拟环境
rm -rf venv

# 删除缓存和日志
rm -rf cache logs

# 如需完全删除项目
cd ..
rm -rf wear-datas
```

## 获取帮助

- 查看文档: `design/` 目录
- 查看行动计划: `ACTION_PLAN.md`
- 查看进度: `PROGRESS_TRACKER.md`
- 报告问题: GitHub Issues

---

**版本**: v1.1.0
**最后更新**: 2025-11-14
