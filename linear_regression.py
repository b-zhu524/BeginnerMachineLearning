import torch
from torch import tensor
from random import shuffle


def generate_data(w, b, num_examples):
    X = torch.normal(mean=0, std=1, size=(num_examples, len(w)))
    y = torch.matmul(X, w) + b
    noise = torch.normal(mean=0, std=0.01, size=y.shape)
    y += noise
    y_reshaped = torch.reshape(y, shape=(-1, 1))

    return X, y_reshaped


def data_iter(features, labels, batch_size):
    num_examples = len(features)
    indices = list(range(num_examples))
    shuffle(indices)    # make sure to remove patterns from certain mini-batches

    for i in range(0, num_examples, batch_size):
        batch_indices = tensor([
            indices[i: min(i + batch_size, num_examples)]])

        yield features[batch_indices], labels[batch_indices]


def sgd(params, lr):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad
            param.grad.zero_()  # reset


def linear_regression(X, w, b):
    y = torch.matmul(X, w) + b
    return y


def squared_loss(y_hat, y):
    l = (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
    return l


if __name__ == '__main__':
    true_weight = tensor([2, -3.4])
    true_bias = 4.2

    features, labels = generate_data(true_weight, true_bias, 1000)

    learning_rate = 0.005
    num_epochs = 10000
    batch_size = 100

    net = linear_regression
    loss = squared_loss

    weight = torch.normal(mean=0, std=0.01, size=(2, 1), requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)

    for epoch in range(num_epochs):
        for X, y_hat in data_iter(features, labels, batch_size):
            l = loss(y_hat, net(X, weight, bias)).mean()
            l.backward()
            sgd([weight, bias], learning_rate)

        with torch.no_grad():
            train_l = loss(labels, net(features, weight, bias))
            print(f"epoch {epoch + 1} loss: {float(train_l.mean()):f}")

    print(weight, bias)
