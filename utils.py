
import torch
import time

import dlc_practical_prologue as prologue

from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F


#Functions that help processing data.

#prepare input data for vanilla model
def prep_input_vanilla(train_input):
    new_train_input = train_input.view(2000,-1,14,14)
    return new_train_input

#prepare target for vanilla model
def prep_target_vanilla(train_classes):
    
    train_classes = train_classes.flatten()
    
    stack = torch.tensor([0.0]*10)
    
    for i in train_classes:
        current = torch.tensor([0.0]*10)
        current[i.item()] = 1
        stack = torch.cat([stack.view(-1,10),current.view(-1,10)])
        
    return stack[1:]
        

def prepare_vanilla(train_input,train_classes):
    
    return prep_input_vanilla(train_input), prep_target_vanilla(train_classes)
