#!/usr/bin/env python

import torch

import dlc_practical_prologue as prologue

train_input, train_target, train_classes, test_input, test_target, test_classes = \
    prologue.generate_pair_sets(nb=1000)

if __name__ == '__main__':
    print(train_input.shape, train_target.shape, train_target[1], train_classes.shape, train_classes[1])