import matplotlib.pyplot as plt
import torch
torch.set_grad_enabled(False)

def convert_to_one_hot_labels(input, target):
    tmp = input.new_zeros(target.size(0), target.max() + 1)
    tmp.scatter_(1, target.view(-1, 1), 1.0)
    return tmp

def plot_dataset(input, target):
    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    input = input.numpy()
    target = target.numpy()
    in_ = [tuple(t) for t, b in zip(input, target) if b == 1]
    out_ = [tuple(t) for t, b in zip(input, target) if b == 0]

    ax.scatter(dict(in_).keys(), dict(in_).values(), c='r', label='inside')
    ax.scatter(dict(out_).keys(), dict(out_).values(), c='b', label='outside')
    plt.legend(framealpha=1)
    plt.show()

def compute_nb_errors(model, data_input, data_target):
    nb_data_errors = 0
    for i in range(data_input.size(0)):
        pred = model.forward(data_input[i])
        if torch.max(data_target[i], 0)[1] != torch.max(pred, 0)[1]:
            nb_data_errors = nb_data_errors + 1

    return nb_data_errors

def train_model(model, optimizer, lr, criterion, nb_epochs, train_input, train_target, test_input, test_target):
    for epoch in range(nb_epochs):
        acc_loss = 0
        # if epoch==0:
        #     model.print_param()
        for i in range(train_input.size(0)):
            output = model.forward(train_input[i])
            target = train_target[i]
            loss = criterion.forward(output, target)
            acc_loss += loss
            grad_output = criterion.backward()
            model.backward(grad_output)
            optimizer.step()

        # print('--')
        # if epoch==0:
        #     model.print_param()
        print('epoch: {:3}, loss: {:.7f}, train error: {:5.2f}%, test error: {:5.2f}%'.format(
                    epoch,
                    acc_loss/train_input.size(0),
                    compute_nb_errors(model, train_input, train_target) / train_input.size(0) * 100,
                    compute_nb_errors(model, test_input, test_target) / test_input.size(0) * 100
                    )
        )