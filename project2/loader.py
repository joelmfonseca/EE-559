import math
import torch
torch.set_grad_enabled(False)

def gen_disc_set(num_samples=1000):
    input = torch.empty(num_samples, 2).uniform_(0,1)
    target = input.sub_(0.5).pow(2).sum(1).sub(1 / (2*math.pi)).sign().sub(1).div(-2).long()
    return input, target