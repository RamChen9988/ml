#!/bin/bash

echo "========================================"
echo "机器学习线性回归项目环境安装脚本"
echo "========================================"
echo

echo "检查Python是否已安装..."
python3 --version >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "错误: 未检测到Python3，请先安装Python 3.7+"
    echo "Ubuntu/Debian: sudo apt install python3 python3-pip"
    echo "CentOS/RHEL: sudo yum install python3 python3-pip"
    echo "macOS: brew install python3"
    exit 1
fi

echo "Python已安装，开始安装依赖包..."
echo

echo "1. 使用pip安装依赖包..."
pip3 install -r requirements.txt

if [ $? -ne 0 ]; then
    echo
    echo "警告: pip安装过程中出现错误"
    echo "尝试使用国内镜像源安装..."
    pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
fi

echo
echo "2. 验证安装结果..."
python3 -c "import pandas; import numpy; import matplotlib; import seaborn; import sklearn; print('所有依赖包安装成功！')"

if [ $? -eq 0 ]; then
    echo
    echo "========================================"
    echo "环境安装完成！"
    echo "========================================"
    echo "现在可以运行: python3 linear.py"
else
    echo
    echo "========================================"
    echo "环境安装可能存在问题，请检查错误信息"
    echo "========================================"
fi

echo
