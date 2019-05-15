import matplotlib.pyplot as plt
import math
import numpy as np
import torch
torch.set_grad_enabled(False)
torch.manual_seed(2019)

from activation import Tanh, ReLU, LeakyReLU, PReLU

def convert_to_one_hot_labels(input, target):
    '''This function converts labels to one hot encoding.'''
    tmp = input.new_zeros(target.size(0), target.max() + 1)
    tmp.scatter_(1, target.view(-1, 1), 1.0)
    return tmp

def gen_disc_set(num_samples=1000):
    '''This function generates the toy dataset.'''
    input = torch.empty(num_samples, 2).uniform_(0,1)
    target = input.sub(0.5).pow(2).sum(1).sub(1 / (2*math.pi)).sign().sub(1).div(-2).long()
    return input, convert_to_one_hot_labels(input, target)

def standardise_input(train_input, valid_input, test_input):
    '''This function standardises all the inputs.'''
    mean, std = train_input.mean(), train_input.std()
    return train_input.sub(mean).div(std), valid_input.sub(mean).div(std), test_input.sub(mean).div(std)

def build_CV_sets(k_fold, num_samples=1000):
    '''This function builds all the configuration sets for K-fold Cross Validation.'''

    samples_per_set = int(num_samples/k_fold)
    fold_sets = []
    for i in range(k_fold):
        input, target = gen_disc_set(samples_per_set)
        fold_sets.append((input, target))

    k_fold_sets = []
    for i in range(k_fold):
        valid_input, valid_target = fold_sets[i]
        train_input, train_target = [], []
        for j in range(k_fold):
            if i is not j:
                # print(i,j)
                input, target = fold_sets[j]
                train_input.append(input)
                train_target.append(target)
        k_fold_sets.append((torch.cat(train_input), torch.cat(train_target), valid_input, valid_target))
        
    return k_fold_sets
    
def plot_dataset(input, target):
    '''This function plots the toy dataset.'''
    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    input = input.numpy()
    target = target.numpy()
    in_ = [tuple(t) for t, b in zip(input, target) if b == 1]
    out_ = [tuple(t) for t, b in zip(input, target) if b == 0]

    ax.scatter(dict(in_).keys(), dict(in_).values(), c='r', label='inside')
    ax.scatter(dict(out_).keys(), dict(out_).values(), c='b', label='outside')
    plt.legend(framealpha=1)
    plt.tight_layout()
    plt.savefig('figures/dataset.png', dpi=300)
    plt.show()

def plot_activations():
    '''This function plots all the activation functions implemented.'''
    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    x_range = np.arange(-3, 3, 0.01)
    x = torch.Tensor(x_range)
    tanh = Tanh()
    plt.plot(x_range, tanh.forward(x).numpy(), color='b', label='Tanh', alpha=0.5)
    plt.plot(x_range, tanh.backward(1).numpy(), color='b', label='Tanh derivative', alpha=0.5, linestyle=':')
    relu = ReLU()
    plt.plot(x_range, relu.forward(x).numpy(), color='g', label='ReLU (0)', alpha=0.5)
    plt.plot(x_range, relu.backward(1).numpy(), color='g', label='ReLU derivative', alpha=0.5, linestyle=':')
    leakyrelu = LeakyReLU()
    plt.plot(x_range, leakyrelu.forward(x).numpy(), color='m', label='LeakyReLU (0.01)', alpha=0.5)
    plt.plot(x_range, leakyrelu.backward(1).numpy(), color='m', label='LeakyReLU derivative', alpha=0.5, linestyle=':')
    prelu = PReLU(init=0.1)
    plt.plot(x_range, prelu.forward(x).numpy(), color='y', label='PReLU', alpha=0.5)
    plt.plot(x_range, prelu.backward(1).numpy(), color='y', label='PReLU derivative (0.1 - trainable)', alpha=0.5, linestyle=':')
    plt.legend(framealpha=1)
    plt.tight_layout()
    plt.savefig('figures/activations.png', dpi=300)
    plt.show()

def compute_nb_errors(model, data_input, data_target, mini_batch_size):
    '''This function computes the number of errors between the model prediction and the groundtruth.'''
    nb_data_errors = 0
    for b in range(0, data_input.size(0), mini_batch_size):
        pred = model.forward(data_input.narrow(0, b, mini_batch_size))
        for k in range(mini_batch_size):
            if torch.max(data_target.data[b + k], 0)[1] != torch.max(pred[k], 0)[1]:
                nb_data_errors = nb_data_errors + 1
    return nb_data_errors

def copy(param):
    '''This function makes a deep copy of parameters.'''
    copy_param = []
    for p, grad in param:
        copy_param.append((p.clone(), grad.clone()))
    return copy_param

def log(string):
    '''This functions prints some log information based on the string passed by argument.'''
    log_info=False
    if log_info:
        print(string)

def train(model, optimizer, lr, criterion,
                train_input, train_target, valid_input, valid_target, mini_batch_size):
    '''This function trains the model based on the parameters passed by argument.'''            

    patience = 20
    best = {'epoch': 0, 'error': math.inf, 'param': None}
    epoch = 0
    while True:
        acc_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model.forward(train_input.narrow(0, b, mini_batch_size))
            target = train_target.narrow(0, b, mini_batch_size)
            loss = criterion.forward(output, target)
            acc_loss += loss
            grad_output = criterion.backward()
            model.backward(grad_output)
            optimizer.step()

        train_error = compute_nb_errors(model, train_input, train_target, mini_batch_size) / train_input.size(0) * 100
        valid_error = compute_nb_errors(model, valid_input, valid_target, mini_batch_size) / valid_input.size(0) * 100

        log('epoch: {:3}, loss: {:.7f}, train error: {:5.2f}%, valid error: {:5.2f}%'.format(
                    epoch,
                    acc_loss/(train_input.size(0)/mini_batch_size),
                    train_error,
                    valid_error
                    )
        )

        if valid_error < best['error']:
            best['epoch'] = epoch
            best['error'] = valid_error
            best['param'] = copy(model.param())

        if epoch > best['epoch'] + patience:
            log('Reached patience threshold. ({})'.format(patience))
            break

        epoch += 1
    
    log('***')
    log('Best model found at epoch {} with valid error: {:5.2f}%'.format(best['epoch'], best['error']))
    log('***')

    return best

def test(model, test_input, test_target, mini_batch_size):
    '''This function tests the model based on the parameters passed by argument.'''

    test_error = compute_nb_errors(model, test_input, test_target, mini_batch_size) / test_input.size(0) * 100
    log('best test error: {:5.2f}%'.format(test_error))

    return test_error

if __name__ == '__main__':
    plot_activations()