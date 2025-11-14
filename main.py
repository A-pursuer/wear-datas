"""
齿轮磨损数据分析系统 - 主应用入口

这是Streamlit应用的主入口文件。

使用方法:
    streamlit run main.py

或使用启动脚本:
    ./run.sh        (Linux/Mac)
    run.bat         (Windows)
"""

import streamlit as st
from pathlib import Path

# 导入配置
from config.settings import (
    PROJECT_NAME,
    VERSION,
    PAGE_TITLE,
    PAGE_ICON,
    LAYOUT,
    INITIAL_SIDEBAR_STATE,
    logger
)


def setup_page():
    """配置Streamlit页面"""
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout=LAYOUT,
        initial_sidebar_state=INITIAL_SIDEBAR_STATE,
        menu_items={
            'Get Help': 'https://github.com/A-pursuer/wear-datas',
            'Report a bug': 'https://github.com/A-pursuer/wear-datas/issues',
            'About': f"# {PROJECT_NAME}\n\n版本: {VERSION}\n\n"
                     "基于Python和Streamlit的齿轮磨损数据分析工具"
        }
    )


def show_welcome():
    """显示欢迎界面"""
    st.title(f"{PAGE_ICON} {PROJECT_NAME}")
    st.markdown(f"**版本**: {VERSION}")
    st.markdown("---")

    st.success("🎉 系统启动成功！")

    st.info("""
    ### 👋 欢迎使用齿轮磨损数据分析系统！

    这是一个基于Web的交互式数据分析工具，专门用于齿轮磨损状态的振动信号分析。

    #### 🚀 快速开始

    系统当前处于开发阶段，核心功能正在实现中...

    #### 📋 已完成功能

    - ✅ 项目基础设施搭建
    - ✅ 配置管理系统
    - ⏳ 数据处理层（开发中）
    - ⏳ 信号处理层（待开发）
    - ⏳ 可视化层（待开发）
    - ⏳ 用户界面层（待开发）

    #### 📖 查看文档

    - [设计文档](design/)
    - [行动计划](ACTION_PLAN.md)
    - [进度跟踪](PROGRESS_TRACKER.md)
    - [安装指南](INSTALL.md)
    """)

    # 显示项目结构
    with st.expander("📁 查看项目结构", expanded=False):
        st.code("""
wear-datas/
├── config/              ✅ 配置模块
├── data/                ⏳ 数据处理模块
├── processing/          ⏳ 信号处理模块
├── visualization/       ⏳ 可视化模块
├── ui/                  ⏳ 用户界面模块
├── tests/               ⏳ 测试模块
├── cache/               📦 缓存目录
├── logs/                📝 日志目录
├── main.py              ✅ 主应用入口
└── requirements.txt     ✅ 依赖清单
        """, language="text")

    # 显示系统信息
    with st.expander("ℹ️ 系统信息", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**项目信息**")
            st.text(f"名称: {PROJECT_NAME}")
            st.text(f"版本: {VERSION}")
            st.text(f"框架: Streamlit")

        with col2:
            st.markdown("**目录信息**")
            project_root = Path(__file__).parent
            st.text(f"根目录: {project_root}")
            st.text(f"数据文件: {len(list(project_root.glob('*.csv')))} 个")

    # 开发进度
    st.markdown("---")
    st.markdown("### 📊 开发进度")

    progress_data = {
        "Phase 1: 基础设施": 100,
        "Phase 2: 数据层": 0,
        "Phase 3: 处理层": 0,
        "Phase 4: 可视化层": 0,
        "Phase 5: UI层": 0,
        "Phase 6: 集成": 0,
        "Phase 7: 测试": 0,
        "Phase 8: 文档": 0,
    }

    for phase, progress in progress_data.items():
        st.progress(progress / 100, text=f"{phase}: {progress}%")

    overall_progress = sum(progress_data.values()) / len(progress_data)
    st.metric("总体进度", f"{overall_progress:.1f}%")


def main():
    """主函数"""
    # 配置页面
    setup_page()

    # 记录启动日志
    logger.info(f"启动 {PROJECT_NAME} {VERSION}")

    # 显示欢迎界面
    show_welcome()


if __name__ == "__main__":
    main()
