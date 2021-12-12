import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from time import perf_counter
import numpy as np
import pickle

with open('./data/dataset/dataset_tmp.pkl', 'rb') as f:
    dataset = pickle.load(f)
dataset_tensor = torch.tensor(dataset)
x = dataset_tensor[:, :6].to(torch.float32)
y = torch.unsqueeze(dataset_tensor[:, 6], dim=1).to(torch.float32)
# x1 = torch.unsqueeze(torch.linspace(-8, 8, 10000), dim=1)
# x2 = torch.unsqueeze(torch.tensor(np.random.randint(1, 10, size=(10000, ))), dim=1)
# x3 = torch.unsqueeze(torch.tensor(np.random.randint(3, 8, size=(10000, ))), dim=1)
# x = torch.cat([x1,x2,x3], axis=1)
# y = 1.5 * x1.pow(2) + 3.8*x2 + 4.7*x3 + 0.3 * torch.rand(x1.size())
# print(y)
# plt.scatter(x1.numpy(), y.numpy(), s=0.01)
# plt.show()


class Net(nn.Module):
    def __init__(self, input_feature, num_hidden, outputs):
        super(Net, self).__init__()
        self.hidden = nn.Linear(input_feature, num_hidden)
        self.out = nn.Linear(num_hidden, outputs)

    def forward(self, x):
        x = torch.nn.functional.relu(self.hidden(x))
        x = self.out(x)
        return x

net = Net(input_feature=6, num_hidden=40, outputs=1)
inputs = x
target = y

# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# optimizer = optim.RMSprop(net.parameters(), lr=0.01)
optimizer = optim.Adam(net.parameters(), lr=0.01)
criterion = nn.MSELoss()

for name, param in net.named_parameters(): #查看可优化的参数有哪些
  if param.requires_grad:
    print(name)

def draw(output, loss):
    plt.cla()  # 清空画布
    # if CUDA:
    #     output = output.cpu()  # 还原为cpu类型才能进行绘图
    plt.scatter(x.numpy(), y.numpy(), s=0.001)
    plt.plot(x.numpy(), output.data.numpy(), 'r-', lw=5)
    plt.text(0.5, 0, 'loss=%s' % (loss.item()),
             fontdict={'size': 20, 'color': 'red'})
    plt.pause(0.005)


def train(model, criterion, optimizer, epochs):
    for epoch in range(epochs):
        outputs = model(inputs)
        loss = criterion(outputs, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            # draw(outputs, loss)
            print(epoch,"===", loss)
    return model, loss


start = perf_counter()
model, loss = train(net, criterion, optimizer, 10000)
finish = perf_counter()
time = finish - start
torch.save(model, "model1.pkl")
torch.save(net.state_dict(),"model2.pkl")
print("计算时间：%s" % time)
print("final loss:", loss.item())
print("weights:", list(model.parameters()))
