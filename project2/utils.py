import matplotlib.pyplot as plt
import math
import torch
#torch.manual_seed(2019)
torch.set_grad_enabled(False)

def convert_to_one_hot_labels(input, target):
    tmp = input.new_zeros(target.size(0), target.max() + 1)
    tmp.scatter_(1, target.view(-1, 1), 1.0)
    return tmp

def standardise_input(train_input, valid_input, test_input):

    mean, std = train_input.mean(), train_input.std()

    train_input.sub_(mean).div_(std)
    valid_input.sub_(mean).div_(std)
    test_input.sub_(mean).div_(std)

def plot_dataset(input, target):
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

def compute_nb_errors(model, data_input, data_target, mini_batch_size):
    nb_data_errors = 0
    for b in range(0, data_input.size(0), mini_batch_size):
        pred = model.forward(data_input.narrow(0, b, mini_batch_size))
        for k in range(mini_batch_size):
            # print(data_target.data[b + k], pred[k])
            if torch.max(data_target.data[b + k], 0)[1] != torch.max(pred[k], 0)[1]:
                nb_data_errors = nb_data_errors + 1
    return nb_data_errors

def copy(param):
    copy_param = []
    for p, grad in param:
        copy_param.append((p.clone(), grad.clone()))
    return copy_param

def train(model, optimizer, lr, criterion, nb_epochs,
                train_input, train_target, valid_input, valid_target, mini_batch_size):

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

        print('epoch: {:3}, loss: {:.7f}, train error: {:5.2f}%, valid error: {:5.2f}%'.format(
                    epoch,
                    acc_loss/train_input.size(0),
                    train_error,
                    valid_error
                    )
        )

        if valid_error < best['error']:
            best['epoch'] = epoch
            best['error'] = valid_error
            best['param'] = copy(model.param())

        if epoch > best['epoch'] + patience:
            print('Reached patience threshold. ({})'.format(patience))
            break

        epoch += 1
    
    print('***')
    print('Best model found at epoch {} with valid error: {:5.2f}%'.format(best['epoch'], best['error']))
    print('***')

    return best

def test(model, test_input, test_target, mini_batch_size):

    test_error = compute_nb_errors(model, test_input, test_target, mini_batch_size) / test_input.size(0) * 100
    print('best test error: {:5.2f}%'.format(test_error))