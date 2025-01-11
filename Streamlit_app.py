# File: app.py

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib import font_manager

# 全局设置
np.set_printoptions(precision=4)
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = ['Arial', 'KaiTi', 'SimSun', 'SimHei']

# 输入数据
pumping_rate = 528 / 1440  # m³/min
obs_dist = 90              # m
obs_time = np.array([1, 2, 4, 6, 9, 20, 30, 40, 50, 60,
                     90, 120, 150, 360, 550, 720])  # min
drawdown = np.array([2.5, 3.9, 6.1, 8.0, 10.6, 16.8, 20.0, 22.6, 24.7, 26.4,
                     30.4, 33.0, 35.0, 42.6, 44.0, 44.5]) / 100  # m

# 定义计算函数
def calc_beta(obs_time, drawdown, i, j):
    """计算拟合系数和误差平方和"""
    X = np.vstack([np.ones(j - i), np.log10(obs_time[i:j])]).T
    beta = np.linalg.lstsq(X, drawdown[i:j], rcond=None)[0]
    residuals = np.sum((drawdown[i:j] - X @ beta) ** 2)
    return beta, residuals

# Streamlit 应用
st.title("最小二乘法-Jacob公式拟合")

st.sidebar.header("输入参数")
data_limits = st.sidebar.slider(
    "选择拟合数据范围:",
    0, len(obs_time), (0, len(obs_time)), step=1
)

# 绘制拟合曲线和结果
fig, ax = plt.subplots(figsize=(6, 4), dpi=150)

# 选定范围
i, j = data_limits
beta, residuals = calc_beta(obs_time, drawdown, i, j)

# 绘制观测值和拟合曲线
ax.set(xscale="log", xlabel=r'$\lg t$', ylabel=r'$s$', title="Jacob fit(Least Squares)")
ax.grid(True)
ax.grid(True, which="major", linestyle="-", linewidth=0.5)
ax.grid(True, which="minor", linestyle="-", linewidth=0.2)
ax.scatter(obs_time, drawdown, label="obs", color="r", marker="*")
ax.plot(
    obs_time,
    beta[0] + beta[1] * np.log10(obs_time),
    label="line fit", linestyle="--", color="g"
)
ax.legend(loc=4")

# 计算并显示结果
T = 0.183 * pumping_rate / beta[1]
S = 2.25 * T / obs_dist ** 2 / 10 ** (beta[0] / beta[1])
u = obs_dist ** 2 * S / (4 * obs_time * T)

st.pyplot(fig)

st.write(f"拟合结果: T = {T:.4f}, S = {S:.4e}, 残差平方和 = {residuals:.4e}")
#st.write(f"无因次时间参数 (u): {u}")
