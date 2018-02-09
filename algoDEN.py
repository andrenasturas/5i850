import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DEN(nn.Module):

    def __init__(self, model, tau = 1, mu = 0.5, gamma = 1, k = 5, sigma = 0.5, lamb = 0.5, optim = optim.SGD, lr=1e-3, loss = nn.MSELoss(), nbEpoch = 1000) :

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
        self.nbEpoch = nbEpoch

    def trainNext(self, x, y):
        l = selectiveRetraining(x, y)
        if l > self.tau:
            dynamicExpansion(x, y)
        split(x, y)


    def forward(self, x):
        return self.model.forward(x)

    def trainFirst(self, loader):

        lossHisto = []

        op = self.optim(self.model.parameters(), self.lr)
        l1_crit = nn.L1Loss(size_average=False)

        for i,(x,y) in enumerate(itertools.cycle(mnist_variation_train_set_loader)):
            if i > self.nbEpoch :
                break
            if i%(ite/10) == 0:
                print("Iteration", i)

            loss = self.loss(self(x), y)

            s = 0
            for n, p in self.model.named_parameters() :
                if n in "weight" :
                    s += l1_crit(p, torch.zeros_like(p))
            loss += self.mu * s
            loss.backward()
            op.step()
            
            if i%(self.nbEpoch/100) == 0:
                lossHisto.append(torch.eq(ypred.data, y.data).float().mean())

        return lossHisto

    def selectiveRetraining(self, loader):

        lossHisto = []

        param = [p for n, p in self.model.named_parameters() if n in "weight"]
        op = self.optim(param[-1:], self.lr)
        l1_crit = nn.L1Loss(size_average=False)

        for i,(x,y) in enumerate(itertools.cycle(mnist_variation_train_set_loader)):
            if i > self.nbEpoch :
                break
            if i%(ite/10) == 0:
                print("Iteration", i)

            loss = self.loss(self(x), y)
            loss += self.mu * l1_crit(param[-1], torch.zeros_like(p))
            loss.backward()
            op.step()

            if i%(self.nbEpoch/100) == 0:
                lossHisto.append(torch.eq(ypred.data, y.data).float().mean())

        mask = list(range(len(param)))
        mask[-1] = (param[-1].data > 0).float()
        for i in reversed(range(len(param)-1)) :
            mask[i] = ((param[i].data > 0).float().t() * mask[i+1].max(0)[0]).t()

        handle = []
        cpt = 0
        for m in den.model.children():
            if [p for p in m.parameters()] != []:
                handle.append(m.register_backward_hook(lambda module, grad_input, grad_output : grad_input * mask[cpt]))
                cpt += 1

        lossHisto2 = []
        op = self.optim(param[-1:], self.lr, weight_decay = self.mu)

        for i,(x,y) in enumerate(itertools.cycle(mnist_variation_train_set_loader)):
            if i > self.nbEpoch :
                break
            if i%(ite/10) == 0:
                print("Iteration", i)

            loss = self.loss(self(x), y)
            loss.backward()
            op.step()

            if i%(self.nbEpoch/100) == 0:
                lossHisto2.append(torch.eq(ypred.data, y.data).float().mean())

        [h.remove()for h in handle]

        return lossHisto, lossHisto2
