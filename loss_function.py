import os
import numpy as np

import torch
import torchvision
import torch.nn.functional as F
import torch.optim as optim

from utils import *




def follower_loss(leader, follower, real_data, real_y, noise, noise_y):
    real_scores = follower(real_data)
    
    # concatenate x and y to include label information
    noise_input = torch.cat((noise_y.reshape((len(noise_y), 1)).double(), noise.double()), dim=1)
    fake_scores = follower(leader(noise_input))
    
    # evaluating the follower on both the real and the fake data
    total_scores = torch.cat((real_scores, fake_scores))
    class_scores = total_scores

    #import pdb; pdb.set_trace()
    class_labels = torch.cat((real_y, noise_y.float()))
    loss = torch.nn.CrossEntropyLoss()
    return loss(class_scores, class_labels.long())
    # return loss(fake_scores, noise_y.long())


def leader_loss(leader, follower, real_data, real_y):
    
    # evaluating the follower on only clean data
    real_scores = follower(real_data)
    loss = torch.nn.CrossEntropyLoss()
    return -loss(real_scores, real_y.long())


def autograd(outputs, inputs, create_graph=False):
    """Compute gradient of outputs w.r.t. inputs, assuming outputs is a scalar."""
    inputs = tuple(inputs)
    grads = torch.autograd.grad(outputs, inputs, create_graph=create_graph, allow_unused=True)
    return [xx if xx is not None else yy.new_zeros(yy.size()) for xx, yy in zip(grads, inputs)]


def hxx_product(leader_loss, leader, follower, tensors):
    d_leader = autograd(leader_loss(), leader.parameters(), create_graph=True)
    return autograd(dot(d_leader, tensors), leader.parameters())


def hww_product(follower_loss, leader, follower,  tensors):
    d_follower = autograd(follower_loss(), follower.parameters(), create_graph=True)
    return autograd(dot(d_follower, tensors), follower.parameters())


def hwx_product(leader_loss, leader, follower, tensors):
    d_leader = autograd(leader_loss(), leader.parameters(), create_graph=True)
    return autograd(dot(d_leader, tensors), follower.parameters())


def hxw_product(follower_loss, leader, follower,  tensors):
    d_follower = autograd(follower_loss(), follower.parameters(), create_graph=True)
    return autograd(dot(d_follower, tensors), leader.parameters())


def get_l_update(leader_loss, follower_loss, leader, follower, l_optim, l_step_size,
                 cg_maxiter, cg_maxiter_cn, cg_tol, cg_lam, cg_lam_cn):
 
    d_leader = autograd(leader_loss(), leader.parameters())

    if l_optim == "tgd":
        inv_hww_dw = conjugate_gradient(lambda tensors: hww_product(follower_loss, leader, follower, tensors=tensors),
                                        autograd(leader_loss(), follower.parameters()),
                                        maxiter=cg_maxiter,
                                        tol=cg_tol,
                                        lam=cg_lam,
                                        )
        hxw_inv_hww_dw = hxw_product(follower_loss, leader, follower, inv_hww_dw)

        return [l_step_size * xx - l_step_size * ww for xx, ww in zip(d_leader, hxw_inv_hww_dw)]

    
def get_f_update(leader_loss, follower_loss, leader, follower,
                 f_optim, f_step_size):
    
    d_follower = autograd(follower_loss(), follower.parameters())
    
    if f_optim == "gd":
        return [f_step_size * xx for xx in d_follower]

  
   
