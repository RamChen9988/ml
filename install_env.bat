@echo off
echo ========================================
echo 机器学习线性回归项目环境安装脚本
echo ========================================
echo.

echo 检查Python是否已安装...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未检测到Python，请先安装Python 3.7+
    echo 可以从 https://www.python.org/downloads/ 下载
    pause
    exit /b 1
)

echo Python已安装，开始安装依赖包...
echo.

echo 1. 使用pip安装依赖包...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo 警告: pip安装过程中出现错误
    echo 尝试使用国内镜像源安装...
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
)

echo.
echo 2. 验证安装结果...
python -c "import pandas; import numpy; import matplotlib; import seaborn; import sklearn; print('所有依赖包安装成功！')"

if %errorlevel% eq 0 (
    echo.
    echo ========================================
    echo 环境安装完成！
    echo ========================================
    echo 现在可以运行: python linear.py
) else (
    echo.
    echo ========================================
    echo 环境安装可能存在问题，请检查错误信息
    echo ========================================
)

echo.
pause
