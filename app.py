import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import time

matplotlib.rcParams['font.family'] = 'SimHei'  # 黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号

st.set_page_config(page_title="线性回归演示 Plus", layout="centered")
st.title("📈 线性回归原理演示（动画 + 损失曲面）")

# Sidebar - 模拟数据参数
st.sidebar.header("🛠️ 数据生成参数")
n_samples = st.sidebar.slider("样本数量", 10, 300, 50)
true_w = st.sidebar.slider("真实权重 w", -10.0, 10.0, 2.0)
true_b = st.sidebar.slider("真实偏置 b", -10.0, 10.0, 1.0)
noise = st.sidebar.slider("噪声强度", 0.0, 5.0, 1.0)

# 数据生成
np.random.seed(42)
X = np.linspace(-5, 5, n_samples)
y_true = true_w * X + true_b + np.random.normal(0, noise, n_samples)

# 初始模型参数
st.header("🔧 模型参数调整")
w = st.slider("初始模型权重 (w)", -10.0, 10.0, 1.0)
b = st.slider("初始模型偏置 (b)", -10.0, 10.0, 0.0)
y_pred = w * X + b
mse = np.mean((y_pred - y_true)**2)
st.metric("当前模型 MSE", f"{mse:.4f}")

# 拟合图像
fig, ax = plt.subplots()
ax.scatter(X, y_true, label="真实数据", color="blue")
ax.plot(X, y_pred, color="red", label="模型预测")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("拟合曲线")
ax.legend()
st.pyplot(fig)

# 梯度下降参数
st.header("📉 模拟训练过程（动画）")
lr = st.number_input("学习率", 0.001, 1.0, 0.01)
epochs = st.number_input("训练轮数", 1, 500, 100)

if st.button("开始动画训练"):
    w_train, b_train = w, b
    history = []
    plot_area = st.empty()
    time.sleep(0.5)

    for epoch in range(int(epochs)):
        y_hat = w_train * X + b_train
        loss = np.mean((y_hat - y_true) ** 2)
        grad_w = np.mean(2 * (y_hat - y_true) * X)
        grad_b = np.mean(2 * (y_hat - y_true))

        w_train -= lr * grad_w
        b_train -= lr * grad_b
        history.append(loss)

        # 绘图动画
        fig, ax = plt.subplots()
        ax.scatter(X, y_true, color="blue", label="真实数据")
        ax.plot(X, y_hat, color="orange", label=f"第 {epoch+1} 轮")
        ax.legend()
        ax.set_title(f"Epoch {epoch+1} | Loss: {loss:.4f}")
        plot_area.pyplot(fig)
        time.sleep(0.05)

    st.success(f"训练完成 🎉\n最终参数：w = {w_train:.4f}, b = {b_train:.4f}")
    st.line_chart(history, height=300, use_container_width=True)

# 3D 曲面图
st.header("🧊 损失函数曲面图（MSE）")

# 生成网格参数
W, B = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
Loss = np.zeros_like(W)

for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        Y_pred = W[i, j] * X + B[i, j]
        Loss[i, j] = np.mean((Y_pred - y_true) ** 2)

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W, B, Loss, cmap='viridis', alpha=0.8)
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('MSE')
ax.set_title('损失曲面')
st.pyplot(fig)
