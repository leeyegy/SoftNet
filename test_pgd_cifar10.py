from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from models.wideresnet import *
from models.resnet import *
from loss import  *

from attacker.pgd import LinfPGDAttack
from train_softnet_cifar10 import  args,test_loader

# setting
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# load net
model = ResNet18().to(device)
model.load_state_dict(torch.load(args.test_model_path))

# define attacker
adversary = LinfPGDAttack(model=model,loss=Cosine_Similarity_Loss(),epsilon=args.epsilon,step_size=args.step_size,max_iter=args.max_iter)

# eval
def eval_test():
    model.eval()
    cln_loss = 0
    cln_correct = 0
    adv_loss = 0
    adv_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            cln_loss += Cosine_Similarity_Loss()(output, target).item()
            cln_correct += get_correct_num(output, target)

            adv_data = adversary.perturb(data,target)
            output = model(adv_data)
            adv_loss += Cosine_Similarity_Loss()(output, target).item()
            adv_correct += get_correct_num(output, target)

    cln_loss /= len(test_loader.dataset)
    adv_loss /= len(test_loader.dataset)

    print('Clean Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        cln_loss, cln_correct, len(test_loader.dataset),
        100. * cln_correct / len(test_loader.dataset)))
    print('PGD Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        adv_loss, adv_correct, len(test_loader.dataset),
        100. * adv_correct / len(test_loader.dataset)))

if __name__ == "__main__":
    eval_test()
