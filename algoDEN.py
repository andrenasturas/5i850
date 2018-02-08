import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DEN(nn.Module):

    def __init__(self, model, tau = 1, mu = 0.5, gamma = 1, k = 5, sigma = 0.5, lamb = 0.5, optim = optim.SGD, lr=1e-3, loss = nn.MSELoss()) :

        super().__init__()

        self.model = model
        self.tau = tau
        self.mu = mu
        self.gamma = gamma
        self.k = k
        self.sigma = sigma
        self.lamb = lamb
        self.optim = optim
        self.loss = loss
        self.lr = lr

    def trainFirst(self, x, y):
        return self.trainL1(x, y)

    def trainNext(self, x, y):
        l = selectiveRetraining(x, y)
        if l > self.tau:
            dynamicExpansion(x, y)
        split(x, y)


    def forward(self, x):
        return self.model.forward(x)

    def trainL1(self, x, y):
        op = self.optim(self.model.parameters(), self.lr)
        l1_crit = nn.L1Loss(size_average=False)

        loss = self.loss(self(x), y)

        s = 0
        for n, p in self.model.named_parameters() :
            if n == "weight" :
                s += l1_crit(p, torch.zeros_like(p))
        loss += self.mu * s
        loss.backward()
        op.step()

        return loss

