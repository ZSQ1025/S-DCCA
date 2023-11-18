import pandas as pd
import matplotlib.pyplot as plt

# 读取包含四组数据的 CSV 文件
data = pd.read_excel(r'C:\Users\123\Desktop\DeepCCA-master - 副本 - 副本\results\results_auc.xlsx')

# 提取四组数据的列名
column_names = ['BurakMHD', 'TCA+', 'CFIW‐TNB', 'S-DCCA']

# 在同一个坐标轴中绘制四个箱型图
plt.boxplot(data.values, vert=None, showfliers=False, labels=column_names)
# plt.boxplot(data.values, vert=None, labels=column_names)
plt.ylabel('AUC')
plt.title('Boxplots of AUC values')
plt.ylim(0.45, 0.75)  # 设置 y 轴刻度范围为 0 到 100

# 显示图形
plt.show()