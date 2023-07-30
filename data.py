import math

import numpy as np

import torch

import torchvision
import torchvision.transforms as transforms



def get_data(option, train_size):

    if option == "mnist":
        def preprocess(sample):
            return sample.view((784,)).double() * 2 - 1

        dataset = torchvision.datasets.MNIST('./data', train=True,
                                             transform=transforms.Compose([
                                                 transforms.ToTensor(),
                                                 preprocess]),
                                             download=True)
        # import pdb; pdb.set_trace()
        idx = (dataset.targets > -1) #== 1) #(dataset.targets == 1) | (dataset.targets == 0) 
        dataset.targets = dataset.targets[idx]
        dataset.data = dataset.data[idx]
    
    elif option == "mnist1":
        def preprocess(sample):
            return sample.view((784,)).double() * 2 - 1

        dataset = torchvision.datasets.MNIST('./data', train=False,
                                             transform=transforms.Compose([
                                                 transforms.ToTensor(),
                                                 preprocess]),
                                             download=True)

        idx = (dataset.targets > -1) #== 1) | (dataset.targets == 0)
        dataset.targets = dataset.targets[idx]
        dataset.data = dataset.data[idx]

    elif option == "cifar-10":
        transform = transforms.Compose(
    		[transforms.ToTensor(),
     		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

    elif option == "cifar-10-1":
	transform = transforms.Compose(
    		[transforms.ToTensor(),
     		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
	dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)


    return dataset


if __name__ == "__main__":
    pass
