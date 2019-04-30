import torch
torch.set_grad_enabled(False)

from module import Linear, Sequential
from activation import Tanh, ReLU
from optimizer import SGD
from loss import MSELoss
from loader import gen_disc_set
from utils import convert_to_one_hot_labels, plot_dataset, train_model

if __name__ == '__main__':

    train_input, train_target = gen_disc_set()
    test_input, test_target = gen_disc_set()
    # plot_dataset(train_input, train_target)

    train_target = convert_to_one_hot_labels(train_input, train_target)
    test_target = convert_to_one_hot_labels(test_input, test_target)
    
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
        Linear(25, 25),
        Tanh(),
        Linear(25, 2)]
    )

    lr = 0.01
    optimizer = SGD(model.param(), lr=lr)
    criterion = MSELoss()

    nb_epochs = 80
    mini_batch_size = 10

    train_model(model, optimizer, lr, criterion, nb_epochs, train_input, train_target, test_input, test_target, mini_batch_size)