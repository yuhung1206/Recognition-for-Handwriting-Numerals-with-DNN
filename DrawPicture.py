
import numpy as np
import matplotlib.pyplot as plt

path = r"C:\Users\...\DNN"  # alter the loading path!!!
Loss_Error = np.load(path+'\\Error.npy')
Train_Error = np.load(path+'\\Train_Error.npy')
Test_Error = np.load(path+'\\Test_Error.npy')


title_head = "\n[Zero initialization][Layer:784 320 160 10],[lamda = 0.1],[with updating bias]"   # alter as the data you plot!!!

plt.figure(figsize=(25,4))  # 2500*400 pixels
plt.subplot(1, 3, 1)
plt.plot(Loss_Error,color="blue",linewidth=2)  # label: shwon in legend
plt.xlabel("Epoch(s)")
plt.ylabel("Average cross entropy")
plt.title("\nTraining Loss")
#plt.show()

#plt.figure(figsize=(6,5))  # 800*400 pixels
plt.subplot(1, 3, 2)
plt.plot(Train_Error,color="green",linewidth=2)  # label: shwon in legend
plt.xlabel("Epoch(s)")
plt.ylabel("Error rate")
plt.title(title_head + "\nTrain error rate" )
#plt.show()

#plt.figure(figsize=(6,5))  # 800*400 pixels
plt.subplot(1, 3, 3)
plt.plot(Test_Error,color="red",linewidth=2)  # label: shwon in legend
plt.xlabel("Epoch(s)")
plt.ylabel("Error rate")
plt.title("\nTest error rate")
plt.show()

