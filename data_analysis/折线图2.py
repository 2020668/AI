import matplotlib
from matplotlib import font_manager, pyplot as plt
import random


# font = {
#     "family": "monospace",
#     "weight": "bold",
#     "size": "larger"
# }
#
# matplotlib.rc("font", **font)
matplotlib.rc("font", family="MicroSoft YaHei")     # weight="bold" 加粗
# matplotlib.rc("font", family="PingFang", weight="bold")

# my_fount = font_manager.FontProperties(fname="")

x = range(0, 120)
y = [random.randint(20, 35) for i in range(120)]

plt.plot(x, y)

# 调整x轴的刻度
_x = list(x)
_xtick_labels = ["10点{}分".format(i) for i in range(60)]
# _xtick_labels += ["11点{}分".format(i-60) for i in range(60, 120)]
_xtick_labels += ["11点{}分".format(i) for i in range(60)]
plt.xticks(_x[::5], _xtick_labels[::5], rotation=45)     # 旋转九十度
# plt.xticks(_x[::5], _xtick_labels[::5], rotation=45, fontproperties=my_fount)     # 旋转九十度

# 添加描述信息
plt.xlabel("时  间")
plt.ylabel("温度 单位（℃）")
plt.title("10点到12点每分钟的温度变化情况")

plt.show()

