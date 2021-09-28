import random
import torch
import warnings
import numpy as np
import torchvision

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, TensorDataset
from parser_args import *

def NoiseProductor(args):
    args = parser.parse_args()
    e = args.noise_factor
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    if args.cifar_10or100 == '10':
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    elif args.cifar_10or100 == '100':
        train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    else:
        warnings.warn('Dataset is not listed')

    
    train_sampler = None
        
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    trueSum = 0
    noise_img = torch.zeros(1, 3, 32, 32).cuda()
    noise_label = torch.zeros(1).cuda()
    changed = torch.zeros(1).cuda()
    for i,data in enumerate(trainloader, 0):
        input, target = data
        input = input.cuda()
        target = target.cuda()
    
        for j in range(target.size(0)):
            label = target[j].cpu().numpy()
            L = label
            proba = np.array([e,1-e])
            reset = np.random.choice(['true','false'],p = proba.ravel())
            if reset=='true':
                trueSum += 1
                changed = torch.cat((changed,torch.tensor(1).cuda().float().reshape(1)),0)
                while L==label:
                    L = random.randint(0,9)
                label = L
            else:
                changed = torch.cat((changed,torch.tensor(0).cuda().float().reshape(1)),0)
            noise_img = torch.cat((noise_img,input[j].reshape(1,3, 32, 32)),0)
            noise_label = torch.cat((noise_label,torch.tensor(label).cuda().float().reshape(1)),0)

    print("total reset trainset",trueSum)
    noise_img = noise_img[1:]
    noise_label = noise_label[1:]
    changed = changed[1:]
    print("total length:",len(noise_label))
    Noise_trainset = TensorDataset(noise_img,noise_label,changed)
    del noise_img,noise_label,changed
    torch.cuda.empty_cache()
    return Noise_trainset


def IndexMatch(trainloader,indexList,args):

    img = torch.zeros(1, 3, 32, 32).cuda()
    label = torch.zeros(1).cuda()
    noise = torch.zeros(1).cuda()
    index = torch.zeros(1).cuda()
    for i,data in enumerate(trainloader,0):
        input, target,change = data
        input = input.cuda()
        target = target.cuda()
        change = change.cuda()

        for j in range(target.size(0)):
            img = torch.cat((img,input[j].reshape(1,3, 32, 32)),0)
            label = torch.cat((label,target[j].float().reshape(1)),0)
            noise = torch.cat((noise,change[j].float().reshape(1)),0)
            index = torch.cat((index,indexList[args.batch_size*i+j].cuda().float().reshape(1)),0)

    img = img[1:]
    label = label[1:]
    noise = noise[1:]
    index = index[1:]
    trainset_sorted = TensorDataset(img,label,noise,index)
    return trainset_sorted
    
def Drop(trainloader,type,percentage):

    drop_point = percentage*50000
    img = torch.zeros(1, 3, 32, 32).cuda()
    label = torch.zeros(1).cuda()
    if type == 'index':
        for i,data in enumerate(trainloader,0):
            input,target,change,ind = data
            input = input.cuda()
            target = target.cuda()
            ind = ind.cuda()

            for j in range(target.size(0)):
                if ind[j] <= drop_point:
                    img = torch.cat((img,input[j].reshape(1,3, 32, 32)),0)
                    label = torch.cat((label,target[j].float().reshape(1)),0)
        img = img[1:]
        label = label[1:]
        New_trainset = TensorDataset(img,label)
        print("The new trainset is of length:",len(label))

    elif type == 'noise':
        for i,data in enumerate(trainloader,0):
            input,target,change,ind = data
            input = input.cuda()
            target = target.cuda()
            change = change.cuda()

            for j in range(target.size(0)):
                if change[j]==0:
                    img = torch.cat((img,input[j].reshape(1,3, 32, 32)),0)
                    label = torch.cat((label,target[j].float().reshape(1)),0)
        img = img[1:]
        label = label[1:]
        New_trainset = TensorDataset(img,label)
        print("The new trainset is of length:",len(label))

    return New_trainset




