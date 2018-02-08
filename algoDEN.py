import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DEN(nn.Module):

    def __init__(self, model, tau, mu, gamma, k, sigma, optim = optim.SGD, lr=1e-3, loss = nn.MSELoss()) :

        '''
        paramList de la forme nbCouche*(weightTensor,biasTensor)
        '''

        super().__init__()

        self.model = model
        self.tau = tau
        self.mu = mu
        self.gamma = gamma
        self.k = k
        self.sigma = sigma
        self.init = False
        self.optim = optim
        self.loss = loss
        self.lr = lr

    def trainNext(self, x, y):
        if not self.init :
            self.trainL1(x, y)
            self.init = True
        else :
            l = selectiveRetraining(self, x, y)
            if l > self.tau:
                dynamicExpansion(self, x, y)
            split(self, x, y)


    def forward(self, x):
        return self.model.forward(x)

    def trainL1(self, x, y, n = 100):
        op = self.optim(self.model.parameters(), self.lr)
        l1_crit = nn.L1Loss(size_average=False)

        for i in range(n):

            loss = self.loss(self(x), y)

            s = 0
            for p in self.model.parameters() :
                s += l1_crit(p, torch.zeros_like(p))
            loss += self.mu * s
            loss.backward()
            op.step()

