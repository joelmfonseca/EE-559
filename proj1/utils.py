import time
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import optim

from train import *

def compute_nb_errors(model, data_input, data_target, mini_batch_size):
    '''This function computes the number of errors between the model prediction and the groundtruth.'''
    nb_data_errors = 0
    for b in range(0, data_input.size(0), mini_batch_size):
        pred = model.predict(data_input.narrow(0, b, mini_batch_size))
        for k in range(mini_batch_size):
            if data_target.data[b + k] != pred[k]:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors

def update_target_type(model, data_target, test_target):
    '''This function updates the target type to use depending on the model.'''
    type_ = model.target_type
    return data_target.type(type_), test_target.type(type_)

def plot_model_comparison(train_input, train_target, train_class, test_input, test_target, optimizer, learning_rate, nb_epochs, mini_batch_size):
    '''This function plots the accuracy on the test set during training for the three main classes of model we implemented: standard, weight sharing and auxiliary models.'''
    model_raw = Net1()
    train_target, test_target = update_target_type(model_raw, train_target, test_target)
    test_raw, train_raw = train_model(model_raw, optimizer(model_raw.parameters(), lr=learning_rate), nb_epochs, \
                        train_input, train_target, test_input, test_target, mini_batch_size, True)

    model_weight = NetSharing1()
    train_target, test_target = update_target_type(model_weight, train_target, test_target)
    test_weight, train_weight = train_model(model_weight, optimizer(model_weight.parameters(), lr=learning_rate), nb_epochs, \
                        train_input, train_target, test_input, test_target, mini_batch_size, True)

    model_aux = NetAux3()
    train_target, test_target = update_target_type(model_aux, train_target, test_target)
    test_aux, train_aux = train_model_aux_bin(model_aux, optimizer(model_aux.parameters(), lr=learning_rate, momentum = 0.7), nb_epochs, \
                         train_input, train_target, train_class, test_input, test_target, mini_batch_size, 0.2, 0.8, 1.5, True)

    plt.plot(test_raw, label = 'Standard model')
    plt.plot(test_weight, label = 'Weight sharing model')
    plt.plot(test_aux, label = 'Auxiliary model')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy on the test set (%)')
    plt.legend(loc = 'best')
    plt.tight_layout()
    plt.savefig('figures/model_comparison.png', dpi=300)
    plt.show()

def plot_netsharing_comparison(train_input, train_target, test_input, test_target, optimizer, learning_rate, nb_epochs, mini_batch_size):
    '''This function plots the accuracy on the test set during training for the three weight sharing models implemented.'''
    model_weight_1 = NetSharing1()
    train_target, test_target = update_target_type(model_weight_1, train_target, test_target)
    test_weight_1, train_weight_1 = train_model(model_weight_1, optimizer(model_weight_1.parameters(), lr=learning_rate), nb_epochs, \
                        train_input, train_target, test_input, test_target, mini_batch_size, True)

    model_weight_2 = NetSharing2()
    train_target, test_target = update_target_type(model_weight_2, train_target, test_target)
    test_weight_2, train_weight_2 = train_model(model_weight_2, optimizer(model_weight_2.parameters(), lr=learning_rate), nb_epochs, \
                        train_input, train_target, test_input, test_target, mini_batch_size, True)

    model_weight_3 = NetSharing3()
    train_target, test_target = update_target_type(model_weight_3, train_target, test_target)
    test_weight_3, train_weight_3 = train_model(model_weight_3, optimizer(model_weight_3.parameters(), lr=learning_rate), nb_epochs, \
                        train_input, train_target, test_input, test_target, mini_batch_size, True)

    plt.plot(test_weight_1,label = 'Weight sharing model 1')
    plt.plot(test_weight_2, label = 'Weight sharing model 2')
    plt.plot(test_weight_3, label = 'Weight sharing model 3')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy on the test set (%)')
    plt.legend(loc = 'best')
    plt.title('Comparison between the three weight sharing models')
    plt.tight_layout()
    plt.savefig('figures/weight_sharing_comparison.png', dpi=300)
    plt.show()

def grid_search(models, optimizers, learning_rates, train_input, train_target, train_class, test_input, test_target, nb_epochs, mini_batch_size):
    '''This function applies a grid search based on the models, optimizers and learning rates passed by argument.'''
    test_errors = []
    times = []
    # for every model, optimizer and learning rate, we apply the good training function (ie, we train the model), for example NetAux1 is trained with
    # train_model_aux function but NetAux2 with train_model_aux_bin due to their different architecture.
    for m in models:
        for optimizer in optimizers:
            for learning_rate in learning_rates:
                
                
                model = m()
                model.apply(model.weight_init)
                train_target, test_target = update_target_type(model, train_target, test_target)
                
                if model.__class__.__name__ == 'Net3':
                    train_input_Net3 = train_input.view(-1, 1, 14, 14)
                    train_target_Net3 = train_class.flatten()

                if model.__class__.__name__ == 'Net3':
                    training_time = train_model(model, optimizer(model.parameters(), lr=learning_rate), nb_epochs, \
                                                    train_input_Net3, train_target_Net3,test_input,test_target ,mini_batch_size)

                elif model.__class__.__name__ == 'NetAux1':
                    alphas = [0.8, 0.8, 1]

                    #optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, momentum = 0.85)

                    training_time = train_model_aux(model, optimizer(model.parameters(), lr=learning_rate), nb_epochs, \
                                                        train_input, train_target, train_class, test_input, test_target, mini_batch_size, 0.2, 0.8, 1.5)
                
                elif model.__class__.__name__ == 'NetAux2' or model.__class__.__name__ == "NetAux3":
                    alphas = [0.8, 0.8, 1]

                    #optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, momentum = 0.85)

                    if optimizer == optim.SGD:
                        training_time = train_model_aux_bin(model, optimizer(model.parameters(), lr=learning_rate, momentum = 0.7), nb_epochs, \
                                                                train_input, train_target, train_class, test_input, test_target, mini_batch_size, 0.2, 0.8, 1.5)
                    elif optimizer == optim.RMSprop:
                        training_time = train_model_aux_bin(model, optimizer(model.parameters(), lr=learning_rate, alpha = 0.9), nb_epochs, \
                                                                train_input, train_target, train_class, test_input, test_target, mini_batch_size, 0.2, 0.8, 1.5)
                    else:
                        training_time = train_model_aux_bin(model, optimizer(model.parameters(), lr=learning_rate, betas=(0.9,0.99)), nb_epochs, \
                                                                train_input, train_target, train_class, test_input, test_target, mini_batch_size, 0.2, 0.8, 1.5)
                else:
                    training_time = train_model(model, optimizer(model.parameters(), lr=learning_rate), nb_epochs, \
                        train_input, train_target, test_input, test_target, mini_batch_size)
                            
                errors_train = compute_nb_errors(model, test_input, test_target, mini_batch_size) / test_input.size(0) * 100
                test_errors.append(errors_train)
                times.append(training_time)

                print('model: {:>13}, criterion: {:>10}, optimizer: {:>10}, learning rate: {:6}, num epochs: {:3}, '
                    'mini batch size: {:3}, training time: {:5.2f}, train error: {:5.2f}%, test error: {:5.2f}%'.format(
                    model.__class__.__name__,
                    model.criterion.__class__.__name__,
                    optimizer.__name__,
                    learning_rate,
                    nb_epochs,
                    mini_batch_size,
                    training_time,
                    compute_nb_errors(model, train_input, train_target, mini_batch_size) / train_input.size(0) * 100,
                    errors_train
                    )
                )

    # take the mean and std for the report
    mean_test_errors = np.mean(test_errors)
    std_test_errors = np.std(test_errors)
    mean_training_time = np.mean(times)
    std_training_time = np.std(times)

    # print('--------------------------------')
    # print('model : {:>13}, Average training time : {:5.2f} +- {:5.2f}, Average test error: {:5.2f}% +- {:5.2f}'.format(
    #    model.__class__.__name__,
    #    mean_training_time, 
    #    std_training_time,
    #    mean_test_errors,
    #    std_test_errors
    #    )
    # )