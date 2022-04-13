import torch


# 定义网络

class Linear(torch.nn.Module):
    def __init__(self):
        super(Linear, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
        pass

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

# 定义损失函数、优化器
model = Linear()
MseLoss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

x = torch.Tensor([[1.0], [2.0], [3.0]])
y = torch.Tensor([[2.0], [4.0], [6.0]])

# 训练5步骤， 得到预测值、计算损失、损失反向传播、优化器更新、梯度清零。
for epo in range(100):
    y_pred = model(x)
    loss = MseLoss(y_pred, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

print('w=', model.linear.weight.item())
print('b=', model.linear.bias.item())
