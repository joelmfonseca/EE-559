#Importations
import time
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np

import dlc_practical_prologue as prologue

from models import *
from utils import *
from trainer import *

from settings import (NB_EPOCH, BATCH, LEARNING_RATE)



def main():
    # load the data
    train_input, train_target, train_classes, test_input, test_target, test_classes = \
        prologue.generate_pair_sets(nb=1000)

    # normalize it
    mean, std = train_input.mean(), train_input.std()

    train_input.sub_(mean).div_(std)
    test_input.sub_(mean).div_(std)

    train_input, train_target, train_class = Variable(train_input), Variable(train_target), Variable(train_classes)
    test_input, test_target = Variable(test_input), Variable(test_target)

    nb_epochs = NB_EPOCH
    mini_batch_size = BATCH
    learning_rates = LEARNING_RATE
    
    #all our models and optimizers we implement.
    #models= [Net1, Net2, NetSharing1, NetSharing2, NetSharing3 ,NetAux1, NetAux2, NetAux3, NetAux4, Net3]
    #optimizers = [optim.SGD, optim.Adam, optim.RMSprop]
    
    #We test for this learning rates.
    

    models = [NetAux3] 
    optimizers = [optim.SGD]

    #grid search function to train with a model, optimizer, learning rate, nb epochs and mini batch size we want.
    grid_search(models, optimizers, learning_rates, train_input, train_target, train_class, test_input, test_target, nb_epochs, mini_batch_size)


    #plot function, uncomment if you want it.
    #plot_model_comparison(train_input, train_target, train_class, test_input, test_target, optim.SGD, 1e-1, nb_epochs, mini_batch_size)
    # plot_netsharing_comparison(train_input, train_target, test_input, test_target, optim.SGD, 1e-1, nb_epochs, mini_batch_size)


#main function
if __name__ == '__main__':
    main()