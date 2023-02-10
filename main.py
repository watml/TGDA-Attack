import os
import time
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from model import *
from data import *
from utils import *
from loss_function import *
import argparse


torch.manual_seed(0)



parser = argparse.ArgumentParser(description="Train TGDA on MLP/MNIST")


parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--dataset", type=str, default="mnist")
parser.add_argument("--train_size", type=int, default=10000)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--pretrain", type=int, default=0)
parser.add_argument("--f_optim", type=str, default="gd")
parser.add_argument("--f_step_size", type=float, default=0.01)
parser.add_argument("--f_num_step", type=int, default=1)
parser.add_argument("--l_num_step", type=int, default=20)

parser.add_argument("--l_optim", type=str, default="tgd")
parser.add_argument("--l_step_size", type=float, default=0.01)

parser.add_argument("--cg_maxiter", type=int, default=64)
parser.add_argument("--cg_maxiter_cn", type=int, default=0)
parser.add_argument("--cg_tol", type=float, default=0.01)
parser.add_argument("--cg_lam", type=float, default=0.0)
parser.add_argument("--cg_lam_cn", type=float, default=0.0)

parser.add_argument("--simultaneous", type=int, default=0)

parser.add_argument("--save_iter", type=int, default=1)
parser.add_argument("--print_iter", type=int, default=10)
parser.add_argument("--epsilon", type=float, default=0.03)

args = parser.parse_args()
print(args)


device = "cuda:0"


def eval(leader, follower, val_data, val_y):

    val_scores = follower(val_data)

    total_val_scores = torch.argmax(val_scores, dim=1)

    x = total_val_scores - val_y.float()

    # real points
    ids = (x == 0)
    print('validation accuracy: ' + str(float(sum(ids)) / len(val_y)))




def train(leader, follower, loader, device="cuda", epsilon=0.03, epoch=1,
          l_optim="tgd", l_step_size=0.014, f_num_step=1, l_num_step=1,
          f_optim="gd", f_step_size=0.01,
          cg_maxiter=None, cg_maxiter_cn=None, cg_tol=None, cg_lam=None, cg_lam_cn=None,
          simultaneous=False,
          save_folder=None, save_iter=2, print_iter=2):

    noise = None
    noise_y = None
    
    start_time = time.time()

    for i in range(epoch_start, epoch):
        if True: 
            print("{:f} seconds in {:d} epochs".format(time.time() - start_time, i)) 

        for idx, data in enumerate(loader):
            real_data = data[0].to(device)
            real_y = data[1].to(device).float()
            num_poisoned = int(epsilon * len(real_data))
            
            #splitting the training set and the validation set
            
            train_data = real_data[:(int(0.7*len(real_data)))]
            train_y = real_y[:int(0.7*len(real_y))]
            val_data = real_data[int((0.7*len(real_data))):]
            val_y = real_y[int((0.7*len(real_y))):]
            
            # noise and noise_y are predefined as part of the clean set, namely real_data[:num_poisoned],real_y[:num_poisoned]
            
            noise = torch.load('noise_tensor.pt')
            noise_y = torch.load('noise_y_tensor.pt')
                
            # define loss function
            def leader_loss_fn(): 
                return leader_loss(leader, follower, val_data, val_y)
            def follower_loss_fn(): 
                return follower_loss(leader, follower, real_data, real_y, noise[int(idx*args.batch_size*epsilon):int((idx+1)*args.batch_size*epsilon)], noise_y[int(idx*args.batch_size*epsilon):int((idx+1)*args.batch_size*epsilon)])
                     
            # Stackelberg game
            if simultaneous == 0:
                """Update leader"""
                
                for _ in range(l_num_step):
                 
                    l_update = get_l_update(leader_loss_fn, follower_loss_fn, leader, follower, l_optim, l_step_size,
                     cg_maxiter, cg_maxiter_cn, cg_tol, cg_lam, cg_lam_cn)

                    #print("epoch: {:4d}".format(i), "batch_idx: {:4d}".format(batch_idx), "leader update")

                    for param, update in zip(leader.parameters(), l_update):
                        param.data -= update #10**(3 - i/200)
                        #print(update)
                    
                """Update follower"""
               
                for _ in range(f_num_step):
                    f_update = get_f_update(leader_loss_fn, follower_loss_fn, leader, follower,
                    f_optim, f_step_size)

                    #print("epoch: {:4d}".format(i), "batch_idx: {:4d}".format(batch_idx), "follower update") 

                    for param, update in zip(follower.parameters(), f_update):
                        param.data -= update
                        #print(update)

                wandb.log({'epoch': i, 'loss_l':leader_loss_fn(), 'loss_f': follower_loss_fn()})

        if i % save_iter == 0 and save_folder is not None:

            eval(leader, follower, val_data, val_y)
            print('leader loss: ' + str(leader_loss_fn()))
            print('follower loss: ' + str(follower_loss_fn()))
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
               
            print('saving to: ',
                os.path.join(save_folder, "leader-epoch_{:d}.tar".format(i)),
                os.path.join(save_folder, "follower-epoch_{:d}.tar".format(i)))

            torch.save({'model_state_dict': leader.state_dict()},
                       os.path.join(save_folder, "leader-epoch_{:d}.tar".format(i)))
            torch.save({'model_state_dict': follower.state_dict()},
                       os.path.join(save_folder, "follower-epoch_{:d}.tar".format(i)))


def get_save_folder(dataset, l_optim, l_step_size, f_num_step, f_optim, f_step_size):
    return "./checkpoints/{}/{}-{}-{}-{}-{}".format(dataset, l_optim, l_step_size, f_optim, f_step_size, f_num_step)

print("==> start the attack using tgda:")

pre_train=False  

  
# specify which architecture to evaluate here
leader = Leader().to(device).double()
follower = Follower_mlp().to(device).double()


epoch_start = 0

follower.load_state_dict(torch.load('./pretrained/classifier_mlp.pt'))

dataset = get_data(args.dataset, args.train_size)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)



train(leader.train(), follower.train(), train_loader, device=device, epsilon=args.epsilon, epoch=args.epoch,
      l_optim=args.l_optim, l_step_size=args.l_step_size, f_num_step=args.f_num_step, l_num_step=args.l_num_step,
      f_optim=args.f_optim, f_step_size=args.f_step_size,
      cg_maxiter=args.cg_maxiter, cg_maxiter_cn=args.cg_maxiter_cn, cg_tol=args.cg_tol, cg_lam=args.cg_lam,                     cg_lam_cn=args.cg_lam_cn, simultaneous=args.simultaneous,
      save_folder=get_save_folder(dataset=args.dataset,
                                  l_optim=args.l_optim, l_step_size=args.l_step_size, f_num_step=args.f_num_step,
                                  f_optim=args.f_optim, f_step_size=args.f_step_size),
      save_iter=args.save_iter, print_iter=args.print_iter,
      )

