from matplotlib import pyplot as plt
import matplotlib


matplotlib.rc("font", family="MicroSoft YaHei")

x = range(11, 31)
y = [1, 0, 1, 1, 2, 4, 3, 2, 3, 4, 4, 5, 6, 5, 4, 3, 3, 1, 1, 1]

# 设置图像大小
plt.figure(figsize=(20, 8), dpi=80)
plt.plot(x, y)

# 设置x轴刻度
_xtick_labels = ["{}岁".format(i) for i in x]
plt.xticks(x, _xtick_labels)

plt.xlabel("年 龄")
plt.ylabel("人数 单位（人）")
plt.title("11岁到30岁每年交女朋友的情况")

# 绘制网格
plt.grid(alpha=0.4)


# 展示
plt.show()
