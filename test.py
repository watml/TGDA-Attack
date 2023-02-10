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

print("==> start testing")
print("==> test model will be saved in test folder and diretly used for the evaluation of an attack")

device = 'cuda:0'
dataset_0 = 'mnist'
dataset_1 = 'mnist1'
batch_size = 1000
train_size = 60000
test_size=10000
n_epochs=100
epsilon=0.03

leader = Leader().to(device).double()
follower = Follower().to(device).double()

# Train dataloader
dataset_train = get_data(dataset_0,train_size)
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

# Test dataloader
dataset_test = get_data(dataset_1,test_size)
batch_size_1 = len(dataset_test) #10000
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size_1, shuffle=True)

epoch_num = 200
folder = 'tgd-0.01-gd-0.1-20'

# specify leader's route
d_file_name = "./checkpoints/mnist/"+folder+"/leader-epoch_"+str(epoch_num)+".tar"
g_file_name = "./checkpoints/mnist/"+folder+"/follower-epoch_"+str(epoch_num)+".tar"
leader.load_state_dict(torch.load(d_file_name)['model_state_dict'])

# generate poisoned points from training data and pretrained generator
noise = torch.load('noise_tensor.pt')
noise_y = torch.load('noise_y_tensor.pt').float()
num_poisoned = int(epsilon * len(dataset_train))
noise = noise[:num_poisoned]
noise_y = noise_y[:num_poisoned]

with torch.no_grad():
    noise_input = torch.cat((noise_y.reshape((len(noise_y), 1)).double(), noise.double()), dim=1)
    x=leader(noise_input)
    torch.save(x, 'poisoned_data.pt')

f_optim= 'gd'
f_step_size = 0.1

def get_f_update(follower_loss, follower,
                 f_optim, f_step_size):
    
    d_follower = autograd(follower_loss, follower.parameters())
    
    if f_optim == "gd":
        return [f_step_size * xx for xx in d_follower]


poisoned = torch.load('poisoned_data.pt')
poisoned_label = noise_y


def train(epoch):
    for idx, data in enumerate(train_loader):
        real_data = data[0].to(device).double()
        real_y = data[1].to(device).double()
        loss= follower_loss(leader, follower, real_data, real_y, 
                            poisoned[int(idx*batch_size*epsilon):int((idx+1)*batch_size*epsilon)], 
                            poisoned_label[int(idx*batch_size*epsilon):int((idx+1)*batch_size*epsilon)])
        f_update = get_f_update(loss,follower,f_optim,f_step_size)
        for param, update in zip(follower.parameters(), f_update):
            param.data -= update
        torch.save(follower.state_dict(), './test_models/classifier_{:d}.pt'.format(epoch))
        print("epoch:{}/{},loss:{}".format(epoch,epoch, loss))

        
def test(epoch):
    follower.load_state_dict(torch.load('./test_models/classifier_{:d}.pt'.format(epoch)))
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
    test(epoch)
