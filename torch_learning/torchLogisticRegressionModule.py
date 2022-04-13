import torchvision
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# 本问题是一个二分类问题，损失函数用BCE，多分类问题的话就是使用crossEntropy交叉熵。
# 补充知识，KL散度是是相对熵，分布p和分布q的KL散度 KL(p||q)可以化成分布p的熵H(p)以及p和q的交叉熵的和。 所以优化交叉熵就是优化KL散度。

# 定义网络
# ----------------------------------------------
class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred


# 定义损失函数、优化器
# --------------------------------------------
model = LogisticRegression()
MseLoss = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

x = torch.Tensor([[1.0], [2.0], [3.0]])
y = torch.Tensor([[0], [0], [1]])

# ----------------------------------------------
# 训练5步骤， 得到预测值、计算损失、损失反向传播、优化器更新、梯度清零。
for epo in range(100):
    # forward
    y_pred = model(x)
    loss = MseLoss(y_pred, y)
    # backward
    loss.backward()
    # update
    optimizer.step()
    optimizer.zero_grad()
# ----------------------------------------------
print('w=', model.linear.weight.item())
print('b=', model.linear.bias.item())

x = np.linspace(0, 10, 200)
x_t = torch.Tensor(x).view((200, 1))
y_t = model(x_t)
y = y_t.data.numpy()
plt.plot(x, y)
plt.xlabel("hours")
plt.ylabel("probability of pass")
plt.grid()
plt.show()
