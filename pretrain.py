import os
import time
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from model import *
from data import *
from utils import *
from loss_function import *

# Pretrain on MNIST dataset 
torch.manual_seed(0)

print("==> start pretraining")
print("==> model will be saved in pretrained folder and diretly used for the initialization of an attack")

device = 'cuda:0'
dataset_0 = 'mnist'
dataset_1 = 'mnist1'
batch_size = 1000
train_size = 60000
test_size=10000
n_epochs=100

leader = Leader().to(device).double()
follower = Follower().to(device).double()

# Train dataloader
dataset_train = get_data(dataset_0,train_size)
batch_size = 1000
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

# Test dataloader
dataset_test = get_data(dataset_1,test_size)
batch_size_1 = len(dataset_test) #10000
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size_1, shuffle=True)

f_optim= 'gd'
f_step_size = 0.1

def get_f_update(follower_loss, follower,
                 f_optim, f_step_size):
    
    d_follower = autograd(follower_loss, follower.parameters())
    
    if f_optim == "gd":
        return [f_step_size * xx for xx in d_follower]
    
def train(epoch):
    total_loss, total_num, train_bar = 0.0,0.0,tqdm(train_loader)
    for data in train_bar:
        real_data = data[0].to(device)
        real_y = data[1].to(device).float()
        loss = -leader_loss(leader, follower, real_data, real_y)
        f_update = get_f_update(loss,follower,f_optim,f_step_size)
        for param, update in zip(follower.parameters(), f_update):
            param.data -= update
        torch.save(follower.state_dict(), './pretrained/classifier_mlp.pt')
        total_num += train_loader.batch_size
        total_loss += loss.item() * train_loader.batch_size
        train_bar.set_description('Train Epoch: [{}/{}], Loss: {:.4f}'.format(epoch, n_epochs, total_loss / total_num))
        #print("epoch:{},loss:{}".format(epoch, loss))
        
def test():
    follower.load_state_dict(torch.load('./pretrained/classifier_mlp.pt'))
    # Evaluate on test set
    total_class_samples = 0
    total_class_correct= 0
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            real_data = data[0].to(device)
            real_y = data[1].to(device).float()
            real_scores = follower(real_data)
            total_class_scores = torch.argmax(real_scores, dim=1)
            total_class_labels = real_y 

            total_class_samples += len(total_class_scores)
            x = total_class_labels - total_class_scores
            ids = (x == 0)
            total_class_correct += sum(ids)
            print("Total Class accuracy: {:.4f}".format(total_class_correct / total_class_samples))
            
for epoch in range(n_epochs):
    train(epoch)
    losses = test()
