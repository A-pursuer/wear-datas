# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个直齿轮磨损状态实验数据集，专注于研究不同磨损程度齿轮的振动特征。数据来自马辉教授课题组的齿轮传动系统典型故障实验平台，包含6个CSV文件，记录了不同磨损状态组合下的振动信号。

## 数据集结构

### 文件命名规则
`主动轮(磨损状态)-从动轮(磨损状态)-扭矩值-转速.csv`

### 磨损状态分类
- **正常**: 齿轮处于健康状态，无明显磨损
- **轻磨**: 齿轮轻度磨损，表面有轻微磨损痕迹
- **重磨**: 齿轮重度磨损，表面磨损严重

### 实验工况
- **扭矩**: 10Nm 或 15Nm
- **转速**: 1000r/min
- **采样频率**: 15360 Hz
- **采样时长**: 30秒
- **数据点数**: 460,805行/文件

## 数据格式说明

### CSV文件结构
- **第1行**: 元数据 `[采样点数,采样频率,扭矩,其他参数]`
  - 例如: `15360,460800,10,0,3`
- **第2-460805行**: 10通道传感器数据
  - 前9列: A、B、C传感器的x、y、z方向加速度信号
  - 第10列: 转速/时间戳信息

### 传感器配置
- **传感器A**: 主动轴输入轴承处 (x轴向, y径向, z径向)
- **传感器B**: 从动轴输入处 (x轴向, y径向, z径向)
- **传感器C**: 从动轴输出处 (x轴向, y径向, z径向)

## 开发命令

### 推荐的Python环境设置
```bash
pip install pandas numpy matplotlib scipy scikit-learn
```

### 数据读取示例
```python
import pandas as pd
import numpy as np

# 内存优化读取方式
def load_gear_data(filename):
    # 读取元数据
    metadata = pd.read_csv(filename, nrows=1, header=None).values[0]

    # 读取传感器数据(跳过元数据行)
    data = pd.read_csv(filename, skiprows=1, header=None)

    return metadata, data.values

# 使用示例
metadata, sensor_data = load_gear_data('主动轮(轻磨)-从动轮(正常)-10Nm-1000r.csv')
print(f"采样参数: {metadata}")
print(f"数据形状: {sensor_data.shape}")
```

### 批处理所有文件
```python
import glob

files = glob.glob("*.csv")
for file in files:
    # 解析文件名获取实验条件
    parts = file.replace('.csv', '').split('-')
    drive_gear = parts[0].split('(')[1].replace(')', '')
    driven_gear = parts[1].split('(')[1].replace(')', '')
    torque = parts[2]

    print(f"文件: {file}")
    print(f"主动轮状态: {drive_gear}, 从动轮状态: {driven_gear}, 扭矩: {torque}")
```

## 研究应用

此数据集适用于以下研究任务:
- 齿轮磨损状态分类与识别
- 振动信号特征提取与分析
- 机械故障诊断算法验证
- 磨损程度定量评估
- 多传感器数据融合研究

## 技术规范

- **文件格式**: CSV (ASCII文本)
- **文件大小**: 约50MB/文件
- **总数据量**: ~300MB
- **编码**: Windows CRLF换行符
- **数据完整性**: 每文件460,805行数据