## Total Gradient Ascent Descent (TGDA) attack

Official Implementation of Indiscriminate Data Poisoning Attacks on Neural Networks in TMLR (https://openreview.net/forum?id=x4hmIsWu7e).

The code is implemented with python 3.7 and torch 1.6.0. 

For MNIST
* Logistic Regression
* Neural Network
* Convolutional Neural Network

For CIFAR-10
* Convolutional Neural Network


An example of running the attack is:

$ bash bash_mnist.sh 

This script automatically runs the pretrain (pretrain.py), the attack (main.py) and the testing (test.py). Note that this script runs TGDA on neural network (mlp) and MNIST automatically. Changing target model requires specifying the leader and the follower defined in model.py. Pretrained models are stored in ./pretrain, attack models are stored in ./checkpoints and test models are stored in ./test_models.

noise_tensor.pt and noise_y_tensor.pt are initializations of D_p, which is a subset of D_tr, can be initialized using D_tr[0:epsilon*len(D_tr)].


Please contact Yiwei Lu (y485lu@uwaterloo.ca) for further questions.
  
