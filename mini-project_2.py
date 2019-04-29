import torch
torch.set_grad_enabled(False)

import math

from dlc_practical_prologue import convert_to_one_hot_labels

class Module(object):

    def forward(self, *input):
        raise NotImplementedError

    def backward(self , *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []

class Linear(Module):

    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.empty(out_features, in_features)
        if bias:
            self.bias = torch.empty(out_features)
        else:
            self.bias = None
        self.grad_weight, self.grad_bias = torch.empty(self.weight.size()), torch.empty(self.bias.size())
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.uniform_(-stdv, stdv)

    def forward(self, input):
        self.input = input
        output = input.matmul(self.weight.t())
        if self.bias is not None:
            output = output + self.bias
        return output

    def backward(self, grad_output):
        self.grad_input = grad_output.matmul(self.weight)
        self.grad_weight.add_(grad_output.t().matmul(self.input))
        if self.bias is not None:
            self.grad_bias.add_(grad_output.sum(0).squeeze(0))
        return self.grad_input

    def param(self):
        list_param = [(self.weight, self.grad_weight)]
        if self.bias is not None:
            list_param.append((self.bias, self.grad_bias))
        return list_param

class Sequential(Module):

    def __init__(self, modules):
        self.modules = modules

    def forward(self, input):
        for module in self.modules:
            input = module.forward(input)
        return input

    def backward(self, grad_output):
        grad = grad_output
        for module in reversed(self.modules):
            grad = module.backward(grad)
        return grad

    def param(self):
        res = []
        for module in self.modules:
            res.extend(module.param())
        return res

class Tanh(Module):

    def forward(self, input):
        self.input = input
        return input.tanh()

    def backward(self , grad_output):
        return grad_output * (1 - self.input.tanh()) ** 2

class ReLU(Module):

    def forward(self, input):
        self.input = input
        return input.clamp(min=0)

    def backward(self, grad_output):
        return grad_output * (self.input>0)

class MSELoss(Module):

    def forward(self, input, target):
        self.input = input
        self.target = target
        return torch.sum((input - target) ** 2)

    def backward(self):
        return 2 * (self.input - self.target)

class Optimizer(object):

    def step(self):
        raise NotImplementedError

class SGD(Optimizer):

    def __init__(self, param, lr):
        self.param = param
        self.lr = lr

    def step(self):
        for p, grad in self.param:
            if grad is not None:
                p.sub_(self.lr*grad)

        for _, grad in self.param:
            if grad is not None:
                grad.zero_()

def gen_disc_set(num_samples=1000):
    input = torch.Tensor(num_samples, 2).uniform_(0,1)
    target = input.pow(2).sum(1).sub(1 / (2*math.pi)).sign().sub(1).div(-2).long()
    return input, convert_to_one_hot_labels(input, target)

def plot_dataset(input, target):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    input = input.numpy()
    target = target.numpy()
    in_ = [tuple(t) for t, b in zip(input, target) if b == 1]
    out_ = [tuple(t) for t, b in zip(input, target) if b == 0]

    ax.scatter(dict(in_).keys(), dict(in_).values(), c='r', label='inside')
    ax.scatter(dict(out_).keys(), dict(out_).values(), c='b', label='outside')
    plt.legend(framealpha=1)
    plt.show()

def compute_nb_errors(model, data_input, data_target, mini_batch_size):

    nb_data_errors = 0
    for b in range(0, data_input.size(0), mini_batch_size):
        pred = model.forward(data_input.narrow(0, b, mini_batch_size))
        for k in range(mini_batch_size):
            # print(data_target.data[b + k], pred[k])
            if torch.max(data_target.data[b + k], 0)[1] != torch.max(pred[k], 0)[1]:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors

def train_model(model, optimizer, lr, criterion, nb_epochs, train_input, train_target, mini_batch_size):
    for e in range(0, nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model.forward(train_input.narrow(0, b, mini_batch_size))
            target = train_target.narrow(0, b, mini_batch_size)

            loss = criterion.forward(output, target)
            grad_output = criterion.backward()
            
            model.backward(grad_output)
            optimizer.step()

        print('criterion: {:>8}, optimizer: {:>5}, learning rate: {:6}, num epochs: {:3}, '
                    'mini batch size: {:3}, train error: {:5.2f}%, test error: {:5.2f}%'.format(
                    criterion.__class__.__name__,
                    optimizer.__class__.__name__,
                    lr,
                    e,
                    mini_batch_size,
                    compute_nb_errors(model, train_input, train_target, mini_batch_size) / train_input.size(0) * 100,
                    compute_nb_errors(model, test_input, test_target, mini_batch_size) / test_input.size(0) * 100
                    )
        )

if __name__ == '__main__':

    train_input, train_target = gen_disc_set()
    test_input, test_target = gen_disc_set()
    #plot_dataset(train_input, train_target)

    mean, std = train_input.mean(), train_input.std()

    train_input.sub_(mean).div_(std)
    test_input.sub_(mean).div_(std)

    model = Sequential([
        Linear(2, 25),
        Tanh(),
        Linear(25, 25),
        Tanh(),
        Linear(25, 25),
        Tanh(),
        Linear(25, 2)]
    )

    lr = 0.001
    optimizer = SGD(model.param(), lr=lr)
    criterion = MSELoss()

    nb_epochs = 300
    mini_batch_size = 100

    train_model(model, optimizer, lr, criterion, nb_epochs, train_input, train_target, mini_batch_size)