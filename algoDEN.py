import torch
from torch.autograd import Variable
import random as rd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DEN(object):

    def __init__(self, paramList, tau, mu, gamma, k, sigma)
        self.paramList = paramList
        self.tau = tau
        self.mu = mu
        self.gamma = gamma
        self.k = k
        self.sigma = sigma
    
    def train(self):