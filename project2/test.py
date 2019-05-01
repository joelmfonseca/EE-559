import torch
torch.set_grad_enabled(False)

from module import Linear, Sequential
from activation import Tanh, ReLU
from optimizer import SGD
from loss import MSELoss, CrossEntropyLoss
from loader import gen_disc_set
from utils import plot_dataset, convert_to_one_hot_labels, standardise_input, train, test

if __name__ == '__main__':

    validation_split = 0.2
    train_input, train_target = gen_disc_set(1-validation_split)
    valid_input, valid_target = gen_disc_set(validation_split)
    test_input, test_target = gen_disc_set()
    # plot_dataset(train_input, train_target)

    train_target = convert_to_one_hot_labels(train_input, train_target)
    valid_target = convert_to_one_hot_labels(valid_input, valid_target)
    test_target = convert_to_one_hot_labels(test_input, test_target)

    standardise_input(train_input, valid_input, test_input)

    model = Sequential([
        Linear(2, 25),
        Tanh(),
        Linear(25, 25),
        Tanh(),
        Linear(25, 25),
        Tanh(),
        Linear(25, 25),
        Tanh(),
        Linear(25, 2)]
    )

    lr = 0.01
    optimizer = SGD(model.param(), lr=lr)
    criterion = CrossEntropyLoss()
    # criterion = MSELoss()

    nb_epochs = 80
    mini_batch_size = 10

    best = train(model, optimizer, lr, criterion, nb_epochs,
                train_input, train_target, valid_input, valid_target, mini_batch_size)

    model.load_param(best['param'])

    test(model, test_input, test_target, mini_batch_size)