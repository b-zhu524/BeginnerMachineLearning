import torch
from torch.utils import data
import linear_regression
from torch import nn


def load_array(data_arrays, batch_size, is_train=True):
    # same thing as data_iter
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


if __name__ == "__main__":
    true_weight = torch.tensor([2, -3.4])
    true_bias = 40

    # hyper parameters
    batch_size = 10
    num_epochs = 1000
    learning_rate = 0.03

    features, labels = linear_regression.generate_data(true_weight, true_bias, 1000)
    data_iter = load_array((features, labels), batch_size)

    net = nn.Sequential(nn.Linear(2, 1))
    loss = nn.MSELoss()
    trainer = torch.optim.SGD(net.parameters(), lr=learning_rate)

    # initialize model
    net[0].weight.data.fill_(1)
    net[0].bias.data.fill_(0)

    for epoch in range(num_epochs):
        for X, y_hat in data_iter:
            l = loss(y_hat, net(X))
            l.backward()
            trainer.step()
            trainer.zero_grad()

        train_l = loss(labels, net(features))
        print(f"epoch {epoch+1} loss: {train_l:f}")

    print(net[0].weight.data)
    print(net[0].bias.data)
