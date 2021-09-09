# Plot scatter figure in 2 node layer (before output layer)

import numpy as np
import matplotlib.pyplot as plt

EP = [200, 360, 499]  # 0~299
pointSize = 1
path = r"C:\Users\...\DNN"  # alter the loading path!!!
latent_var = np.load(path + '\\latent_var.npy')
TrainData = np.load(r"C:\Users\...\train.npz")  # alter the loading path!!!

t_digits = list(map(int, TrainData.f.label))
title_head = "[Layer:784 512 256 2 10],[lamda = 0.1],[with updating bias]\n"  # alter as the data you plot!!!
colorTable = ["firebrick","darkorange","gold","limegreen","blue","navy","darkviolet","aqua","salmon","teal"]


plt.figure(figsize=(25,4))  # 2500 X 400 pixel
plt.subplot(1, 3, 1)
temp = latent_var[EP[0]][:].T
for Num in range(10):
    num_list = ([i for i in range(len(t_digits)) if t_digits[i] == Num])
    plt.scatter(temp[0][num_list], temp[1][num_list], color=colorTable[Num], s=pointSize, label="%d" %Num)
plt.title("\n2D feature in " + str(EP[0]) + "epoch")

plt.subplot(1, 3, 2)
temp = latent_var[EP[1]][:].T
for Num in range(10):
    num_list = ([i for i in range(len(t_digits)) if t_digits[i] == Num])
    plt.scatter(temp[0][num_list], temp[1][num_list], color=colorTable[Num], s=pointSize, label="%d" %Num)
plt.title(title_head + "2D feature in " + str(EP[1]) + "epoch")

plt.subplot(1, 3, 3)
temp = latent_var[EP[2]][:].T
for Num in range(10):
    num_list = ([i for i in range(len(t_digits)) if t_digits[i] == Num])
    plt.scatter(temp[0][num_list], temp[1][num_list], color=colorTable[Num], s=pointSize, label="%d" %Num)
plt.title("\n2D feature in " + str(EP[2]) + "epoch")

legend_x = 1
legend_y = 0.5
plt.legend(loc='center left', bbox_to_anchor=(legend_x, legend_y))
plt.show()