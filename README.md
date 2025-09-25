# 机器学习线性回归项目 - 环境安装指南

本项目是一个机器学习线性回归实战项目，用于预测房价并进行模型评估。

## 项目依赖

项目需要以下Python包：
- pandas (数据处理)
- numpy (数值计算)
- matplotlib (数据可视化)
- seaborn (高级数据可视化)
- scikit-learn (机器学习库)

## 环境安装方法

### 方法一：使用pip安装（推荐）

1. **Windows系统**：
   - 双击运行 `install_env.bat`
   - 或手动执行：`pip install -r requirements.txt`

2. **Linux/macOS系统**：
   - 给脚本执行权限：`chmod +x install_env.sh`
   - 运行脚本：`./install_env.sh`
   - 或手动执行：`pip3 install -r requirements.txt`

### 方法二：使用Conda环境

1. 创建Conda环境：
   ```bash
   conda env create -f environment.yml
   ```

2. 激活环境：
   ```bash
   conda activate ml-linear-regression
   ```

### 方法三：手动安装

如果上述方法遇到问题，可以手动安装：

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 国内用户加速安装

如果下载速度慢，可以使用国内镜像源：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

## 验证安装

安装完成后，运行以下命令验证：

```bash
python -c "import pandas; import numpy; import matplotlib; import seaborn; import sklearn; print('环境配置成功！')"
```

## 运行项目

环境安装完成后，运行主程序：

```bash
python linear.py
```

## 故障排除

### 常见问题

1. **Python未安装**：
   - 下载并安装Python 3.7+：https://www.python.org/downloads/

2. **权限问题（Linux/macOS）**：
   ```bash
   sudo pip3 install -r requirements.txt
   ```

3. **网络连接问题**：
   - 使用国内镜像源
   - 检查网络连接

4. **版本冲突**：
   - 建议使用虚拟环境
   - 或尝试更新pip：`pip install --upgrade pip`

### 使用虚拟环境（推荐）

```bash
# 创建虚拟环境
python -m venv ml_env

# 激活虚拟环境
# Windows:
ml_env\Scripts\activate
# Linux/macOS:
source ml_env/bin/activate

# 安装依赖
pip install -r requirements.txt
```

## 项目文件说明

- `linear.py` - 主程序文件
- `requirements.txt` - pip依赖包列表
- `environment.yml` - Conda环境配置文件
- `install_env.bat` - Windows安装脚本
- `install_env.sh` - Linux/macOS安装脚本

## 技术支持

如果遇到安装问题，请检查：
1. Python版本是否为3.7+
2. 网络连接是否正常
3. 系统权限是否足够

Enjoy your machine learning journey! 🚀
