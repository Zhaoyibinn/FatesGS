from matplotlib.ticker import FixedLocator, FixedFormatter
from matplotlib import transforms
import numpy as np
from matplotlib.scale import FuncScale
import matplotlib.pyplot as plt

# # 数据
x = [1.00, 100.0,200.00, 300,500.00, 1000.00]
y_ours = [2.950,2.533, 2.442, 2.410,2.397, 2.066]

x_fatesgs = [1.00, 100.0,200.0,300.00, 500.00, 1000.00,15000.0]
y_fatesgs = [7.640258062,7.365043836, 8.887946676, 5.402667321,4.34369312, 4.233,5.534]

x_3dgs = [1.00, 100.0,200.0,300.00, 500.00, 1000.00,15000.0]
y_3dgs = [11.28804492, 9.57708403,9.687701991,9.586931736,9.336082061, 10.86,9.431]




plt.style.use('seaborn')
# 绘制折线图
fig, ax = plt.subplots(figsize=(6, 4))


# 用虚线连接最后一个点
ax.plot(x[:-1], y_ours[:-1], marker='X', color='#228B22',label="FSFSplatter")
ax.plot(x[-2:], y_ours[-2:], marker='X', linestyle='--', color='#228B22')

ax.plot(x_fatesgs[:-2], y_fatesgs[:-2], marker='X', color='red', label='FatesGS')
ax.plot(x_fatesgs[-3:], y_fatesgs[-3:], marker='X', color='red',linestyle='--')

ax.plot(x_3dgs[:-2], y_3dgs[:-2], marker='X', color='blue', label='3DGS')
ax.plot(x_3dgs[-3:], y_3dgs[-3:], marker='X', color='blue',linestyle='--')

pass_xi = [100,300,500,1000,15000]
for xi, yi in zip(x, y_ours):
    if xi not in pass_xi:
        continue
    ax.annotate(f'{yi:.2f}', (xi, yi), textcoords="offset points", xytext=(0,6), ha='center', fontsize=10)
    
for xi, yi in zip(x_fatesgs, y_fatesgs):
    if xi not in pass_xi:
        continue
    ax.annotate(f'{yi:.2f}', (xi, yi), textcoords="offset points", xytext=(0,6), ha='center', fontsize=10)

for xi, yi in zip(x_3dgs, y_3dgs):
    if xi not in pass_xi:
        continue
    ax.annotate(f'{yi:.2f}', (xi, yi), textcoords="offset points", xytext=(0,6), ha='center', fontsize=10)
    
ax.legend(fontsize=12)
ax.set_xlabel('Iteration', fontsize=14)
ax.set_ylabel('CD', fontsize=14)
ax.set_title('Error during iterations', fontsize=16)
ax.tick_params(axis='both')
ax.grid(True)


# 设置自定义的x轴范围和比例
ax.set_xlim(0, 15000)


# 通过次坐标轴实现分段线性


def custom_x(x):
    x = np.asarray(x)
    return np.where(
        x <= 1000,
        x,
        1000 + (x - 1000) / (15000 - 1000) * 1500
    )

def inv_custom_x(x):
    x = np.asarray(x)
    return np.where(
        x <= 1000,
        x,
        1000 + (x - 1000) * (15000 - 1000) / 1500
    )

ax.set_xscale(FuncScale(ax, (custom_x, inv_custom_x)))
ax.xaxis.set_major_locator(FixedLocator([100, 300,500, 1000, 15000]))
ax.xaxis.set_major_formatter(FixedFormatter(['100',  '300', '500', '1000', '15000']))



# ax.set_xscale('function', functions=(custom_x, inv_custom_x))
plt.savefig('test.png', dpi=500)
plt.close()