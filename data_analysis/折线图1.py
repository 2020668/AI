from matplotlib import pyplot as plt


x = range(2, 25, 2)
y = [15, 13, 14.5, 17, 20, 25, 26, 26, 27, 22, 18, 15]

# 设置图片大小和清晰度
fig = plt.figure(figsize=(15, 6), dpi=80)

# 绘图
plt.plot(x, y)

# 保存 可保存为svg这种矢量图 放大后不会有锯齿
# plt.savefig("./t1.png")

# 设置x轴刻度
# _xtick_labels = [i/2 for i in range(4, 49)]
# plt.xticks(_xtick_labels)
plt.xticks(range(2, 25))

# 设置y轴
plt.yticks(range(min(y), max(y)+1))

# 描述信息 x y轴代表什么意思

# 调整刻度间距

# 添加水印

# 展示
plt.show()

