import csv
import numpy as np
import matplotlib.pyplot as plt

file = "resnet.csv"
file_name = file.split('.')[0]
data = []
with open(str(file),"r") as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        # next(reader) # skip heading line
        m = 0
        n = 0
        for l in reader:
            data.append(np.array([int(l[0]), float(l[1])+n, float(l[2])+m]))
            # m -=0.01
            # n += 0.01
data = np.array(data)

epochs = data[:,0]
train_loss = data[:,1]
val_loss = data[:,2]
# print(epochs)

plt.title("Loss for Transfer Learning on Resnet18 Encoder")
plt.xlabel("Number of Epochs")
plt.ylabel("Cross Entropy Loss")
plt.plot(epochs, train_loss, color='r', label='training loss')
plt.plot(epochs, val_loss, color = 'b', label = 'validation loss')
plt.grid(True)
plt.legend()
plt.savefig("../results/%s_loss.png"%file_name)
plt.show()
