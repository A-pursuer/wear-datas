#!/bin/bash

# 齿轮磨损数据分析系统 - 启动脚本
# 用法: ./run.sh [选项]

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目信息
PROJECT_NAME="齿轮磨损数据分析系统"
VERSION="v1.1.0"

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 显示横幅
show_banner() {
    echo -e "${BLUE}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "⚙️  ${PROJECT_NAME} ${VERSION}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${NC}"
}

# 检查Python版本
check_python() {
    print_info "检查Python环境..."

    if ! command -v python3 &> /dev/null; then
        print_error "未找到Python3，请先安装Python 3.8或更高版本"
        exit 1
    fi

    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_success "Python版本: ${PYTHON_VERSION}"
}

# 检查并创建虚拟环境
check_venv() {
    if [ ! -d "venv" ]; then
        print_warning "未找到虚拟环境，正在创建..."
        python3 -m venv venv
        print_success "虚拟环境创建成功"
    else
        print_success "虚拟环境已存在"
    fi
}

# 激活虚拟环境
activate_venv() {
    print_info "激活虚拟环境..."
    source venv/bin/activate
    print_success "虚拟环境已激活"
}

# 安装依赖
install_dependencies() {
    print_info "检查依赖包..."

    if [ -f "requirements.txt" ]; then
        print_info "正在安装依赖包（这可能需要几分钟）..."
        pip install --upgrade pip -q
        pip install -r requirements.txt -q
        print_success "依赖包安装完成"
    else
        print_error "未找到requirements.txt文件"
        exit 1
    fi
}

# 验证配置
validate_config() {
    print_info "验证系统配置..."
    python3 config/settings.py
}

# 启动应用
start_app() {
    print_info "启动Streamlit应用..."
    echo ""
    streamlit run main.py
}

# 清理缓存
clean_cache() {
    print_info "清理缓存..."
    rm -rf cache/*
    rm -rf logs/*
    print_success "缓存已清理"
}

# 运行测试
run_tests() {
    print_info "运行测试..."
    pytest tests/ -v
}

# 显示帮助信息
show_help() {
    echo "用法: ./run.sh [选项]"
    echo ""
    echo "选项:"
    echo "  start         启动应用（默认）"
    echo "  install       只安装依赖"
    echo "  clean         清理缓存"
    echo "  test          运行测试"
    echo "  help          显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  ./run.sh              # 启动应用"
    echo "  ./run.sh install      # 只安装依赖"
    echo "  ./run.sh clean        # 清理缓存"
    echo "  ./run.sh test         # 运行测试"
}

# 主函数
main() {
    show_banner

    # 解析命令行参数
    ACTION=${1:-start}

    case $ACTION in
        start)
            check_python
            check_venv
            activate_venv

            # 检查是否需要安装依赖
            if ! python3 -c "import streamlit" 2>/dev/null; then
                install_dependencies
            fi

            validate_config
            start_app
            ;;
        install)
            check_python
            check_venv
            activate_venv
            install_dependencies
            print_success "依赖安装完成！运行 './run.sh start' 启动应用"
            ;;
        clean)
            clean_cache
            ;;
        test)
            check_python
            check_venv
            activate_venv
            run_tests
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "未知选项: $ACTION"
            show_help
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@"
