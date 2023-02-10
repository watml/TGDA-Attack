#!/bin/bash

# configuration for MNIST

epoch=201
dataset=mnist
train_size=60000
batch_size=1000

cg_maxiter=16
cg_maxiter_cn=16

save_iter=2
print_iter=2


# # total gradient descent ascent, a.k.a. stackelberg dynamics in Fiez et al 2019
l_optim=tgd
l_step_size=0.01 
f_num_step=20
l_num_step=1
f_optim=gd
f_step_size=0.1 
simultaneous=0


python pretrain.py
python main.py --epoch $epoch \
              --dataset $dataset \
              --train_size $train_size \
              --batch_size $batch_size \
              --pretrain 1 \
              --l_optim $l_optim \
              --l_step_size $l_step_size \
              --f_num_step $f_num_step \
              --l_num_step $l_num_step \
              --f_optim $f_optim \
              --f_step_size $f_step_size \
              --cg_maxiter $cg_maxiter \
              --cg_maxiter_cn $cg_maxiter_cn \
              --cg_tol 1e-50 \
              --cg_lam 0.0 \
              --simultaneous $simultaneous \
              --save_iter $save_iter \
              --print_iter $print_iter #> out.txt
python test.py
