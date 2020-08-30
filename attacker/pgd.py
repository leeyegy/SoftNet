import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import math
import  sys
sys.path.append("../")
from loss import  *

class LinfPGDAttack():
    def __init__(self,model,loss,epsilon,step_size,max_iter):
        self.model = model
        # perturbation range
        self.epsilon = epsilon
        # loss function
        self.loss = loss
        # PGD step size
        self.step_size = step_size
        # max iteration allowed
        self.max_iter = max_iter


    def guess_label(self,x):
        return self.model(x).max(1,keepdim=True)[1].view(x.size()[0])

    def perturb(self,x_natural,y=None):
        # data
        self.model.eval()
        x_natural = x_natural.clone().detach()
        x_adv = Variable(x_natural.clone().detach().data,requires_grad=True)
        if y is None:
            y = self.guess_label(x_natural)

        for _ in range(self.max_iter):
            # define optimizer
            opt = optim.SGD([x_adv], lr=1e-3)
            opt.zero_grad()
            with torch.enable_grad():
                loss = self.loss(self.model(x_adv),y)
            loss.backward()
            x_adv = x_adv + self.step_size*x_adv.grad.data.sign()

            # clamp
            x_adv = torch.min(torch.max(x_adv,x_natural-self.epsilon),x_natural+self.epsilon)
            x_adv = torch.clamp(x_adv,min=0.0,max=1.0)
            x_adv = Variable(x_adv.data,requires_grad=True)

        return x_adv.clone().detach()
