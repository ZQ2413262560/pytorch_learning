import torch
import torch.nn as nn
import numpy as np 


# crossEntropy衡量了惊喜度,值越小说明这个动作的惊喜度越低，也就是这个动作的概率越大.
# 过程 softmax->取log -> 计算 -log()
a=[[0.1,0.2,0.5]]
b=[2]

a=torch.tensor(a)
b=torch.tensor(b)
m=nn.LogSoftmax()

d=-m(a)
print(d)

loss=nn.CrossEntropyLoss(reduce=False)

c=loss(a,b)
print(c)