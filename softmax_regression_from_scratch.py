import torch
from IPython import display
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

num_inputs = 784    # 28 x 28
num_outputs = 10

W = torch.normal(mean=0, std=0.01, size=(num_inputs, num_outputs),
                 requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
