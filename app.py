import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, integrate, exp, sin, cos

# 设置页面标题
st.title('微积分可视化平台')

# 输入函数
st.header('输入一个函数（例如：sin(x), x**2, exp(x)）')
function_input = st.text_input('函数:', 'sin(x)')

# 解析输入的函数
x = symbols('x')
try:
    func = eval(function_input)  # 使用eval将用户输入的字符串转为符号表达式
except Exception as e:
    st.error(f"输入的函数有误: {e}")
    func = None

# 绘制图像
if func:
    st.header('函数图像')

    # 生成x的值
    x_vals = np.linspace(-10, 10, 400)
    y_vals = np.array([float(func.subs(x, val)) for val in x_vals])

    # 绘图
    fig, ax = plt.subplots()
    ax.plot(x_vals, y_vals, label=f'函数: {function_input}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    st.pyplot(fig)

# 导数可视化
st.header('导数')
if func:
    try:
        # 计算导数
        derivative = diff(func, x)
        st.write(f'导数: {derivative}')

        # 绘制导数图像
        y_vals_derivative = np.array([float(derivative.subs(x, val)) for val in x_vals])

        fig, ax = plt.subplots()
        ax.plot(x_vals, y_vals_derivative, label=f'导数: {derivative}')
        ax.set_xlabel('x')
        ax.set_ylabel("f'(x)")
        ax.legend()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"计算导数时出错: {e}")

# 积分可视化
st.header('定积分')
if func:
    try:
        # 计算积分
        integral = integrate(func, x)
        st.write(f'不定积分: {integral}')

        # 定积分计算
        a = st.number_input('积分下限 a', -10, 10, -5)
        b = st.number_input('积分上限 b', -10, 10, 5)
        integral_value = float(integral.subs(x, b) - integral.subs(x, a))
        st.write(f'定积分结果：∫{function_input}dx from {a} to {b} = {integral_value}')

        # 绘制积分区域
        x_vals_integral = np.linspace(a, b, 400)
        y_vals_integral = np.array([float(func.subs(x, val)) for val in x_vals_integral])

        fig, ax = plt.subplots()
        ax.plot(x_vals_integral, y_vals_integral, label=f'积分: {function_input}')
        ax.fill_between(x_vals_integral, y_vals_integral, color='skyblue', alpha=0.4)
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.legend()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"计算积分时出错: {e}")
