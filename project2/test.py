import time
from tqdm import tqdm
import numpy as np
import torch

from module import Linear, Sequential
from activation import Tanh, ReLU, LeakyReLU, PReLU
from optimizer import SGD
from loss import MSELoss, CrossEntropyLoss
from utils import gen_disc_set, plot_dataset, build_CV_sets, standardise_input, train, test

if __name__ == '__main__':

    lr = 0.01
    k_fold = 10
    CV_sets = build_CV_sets(k_fold, 1000)
    print('CV sets built.')
    test_input, test_target = gen_disc_set(1000)

    for criterion in [MSELoss(), CrossEntropyLoss()]:
        for mini_batch_size in [20]:
            for activation in [Tanh()]:
                
                print('***')
                print('Criterion: {}, mini_batch_size: {}, activation: {}.'.format(
                        criterion.name(),
                        mini_batch_size,
                        activation.name())
                )
                print('***')

                training_time_acc = []
                test_error_acc = []
                for i in tqdm(range(k_fold), leave=False):

                    torch.manual_seed(2019)
                    model = Sequential([
                        Linear(2, 25, activation.name()),
                        activation,
                        Linear(25, 25, activation.name()),
                        activation,
                        Linear(25, 25, activation.name()),
                        activation,
                        Linear(25, 25, activation.name()),
                        activation,
                        Linear(25, 2, activation.name())]
                    )

                    optimizer = SGD(model.param(), lr=lr)

                    train_input, train_target, valid_input, valid_target = CV_sets[i]

                    train_input, valid_input, test_input_ = standardise_input(train_input, valid_input, test_input)

                    start = time.time()
                    best = train(model, optimizer, lr, criterion,
                                train_input, train_target, valid_input, valid_target, mini_batch_size)
                    end = time.time()

                    training_time = end-start
                    training_time_acc.append(training_time)

                    model.load_param(best['param'])

                    test_error = test(model, test_input_, test_target, mini_batch_size)
                    
                    test_error_acc.append(test_error)
                
                
                print('Results average over {}-fold CV. Training time: {:3.2f}(+/-{:3.2f})s. Test error: {:3.2f}(+/-{:3.2f})%.'.format(
                        k_fold,
                        np.mean(training_time_acc),
                        np.std(training_time_acc),
                        np.mean(test_error_acc),
                        np.std(test_error_acc))
                )