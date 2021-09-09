import numpy as np
import matplotlib.pyplot as plt

path = r"C:\Users\...\DNN"  # alter the loading path!!!
CFmatrix = np.load(path + '\CfMatrix.npy')

title_head = "\n[Zero initialization][Layer:784 320 160 10],[lamda = 0.1],[with updating bias]"  # alter as the data you plot!!!

fig, ax = plt.subplots()
# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

col_labels = ['0','1','2','3','4','5','6','7','8','9']
row_labels = ['0','1','2','3','4','5','6','7','8','9']
table_vals = CFmatrix  # [[11,12,13],[21,22,23],[28,29,30]]
row_colors = ["firebrick","darkorange","gold","limegreen","blue","navy","darkviolet","aqua","salmon","teal"]
table = ax.table(cellText=table_vals, colWidths=[0.1]*10,rowLabels=row_labels, colLabels=col_labels, rowColours=row_colors, colColours=row_colors, loc='center')
table.set_fontsize(25)
table.scale(1, 2)
fig.tight_layout()
plt.show()
