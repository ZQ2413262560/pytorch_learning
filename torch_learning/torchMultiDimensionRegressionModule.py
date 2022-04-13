import torchvision
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class MultiDimensionRegression(torch.nn.Module):
    def __init__(self):
        super(MultiDimensionRegression, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = F.sigmoid(self.linear3(x))
        return x


dataset = np.loadtxt(fname='./dataset/diabetes.csv.gz', delimiter=',', encoding='utf-8', dtype=np.float32)
x = np.array(dataset[:, :-1])
y = np.array(dataset[:, [-1]])
x_t = torch.Tensor(x)
y_t = torch.Tensor(y)

model = MultiDimensionRegression()

# construct loss and optimizer
# criterion = torch.nn.BCELoss(size_average = True)
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

epoch_list = []
loss_list = []
# training cycle forward, backward, update
for epoch in range(100):
    y_pred = model(x_t)
    loss = criterion(y_pred, y_t)
    print(epoch, "loss: ", loss.item())
    epoch_list.append(epoch)
    loss_list.append(loss.item())

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

plt.plot(epoch_list, loss_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
