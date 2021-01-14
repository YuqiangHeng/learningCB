# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 12:13:53 2021

@author: ethan
"""

import sys
sys.path.append('D:\\Github Repositories\\learn2learn')
import numpy as np
import matplotlib.pyplot as plt
from ComplexLayers_Torch import PhaseShifter, PowerPooling, ComputePower
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from beam_utils import DFT_codebook, GaussianCenters

import random
import torch
from learn2learn.algorithms import MAML
from learn2learn.utils import clone_module,detach_module

seed = 7
np.random.seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# num_of_beams = [2, 4, 8, 16, 32, 64, 96, 128]
num_of_beams = [8,16,24,32,64,128]
num_antenna = 64
antenna_sel = np.arange(num_antenna)

# --------------------------
# MAML training parameters
# --------------------------
batch_size = 10
nepoch = 1000
shots = 50
update_step = 1
ntest = 50
nval = 10

fast_lr = 0.5
meta_lr = 0.5

# --------------------------
# UE distribution generator parameters (clusters)
# --------------------------
n_clusters = 10
arrival_rate = int(shots*2/n_clusters)
cluster_variance = 10

plot_training_loss_history = False
#-------------------------------------------#
# Load channel data and UE locations
# Scale channel data by max 1-norm
# Compute EGC gain
#-------------------------------------------#
h_real = np.load('D://Github Repositories/mmWave Beam Management/H_Matrices FineGrid/MISO_Static_FineGrid_Hmatrices_real.npy')[:,antenna_sel]
h_imag = np.load('D://Github Repositories/mmWave Beam Management/H_Matrices FineGrid/MISO_Static_FineGrid_Hmatrices_imag.npy')[:,antenna_sel]
loc = np.load('D://Github Repositories/mmWave Beam Management/H_Matrices FineGrid/MISO_Static_FineGrid_UE_location.npy')

BS_loc = [641,435,10]
num_samples = h_real.shape[0]
h = h_real + 1j*h_imag
#norm_factor = np.max(np.power(abs(h),2))
norm_factor = np.max(abs(h))
h_scaled = h/norm_factor
h_concat_scaled = np.concatenate((h_real/norm_factor,h_imag/norm_factor),axis=1)
# Compute EGC gain
egc_gain_scaled = np.power(np.sum(abs(h_scaled),axis=1),2)/num_antenna


class AnalogBeamformer(nn.Module):
    def __init__(self, n_antenna = 64, n_beam = 64, theta = None):
        super(AnalogBeamformer, self).__init__()
        self.codebook = PhaseShifter(in_features=2*n_antenna, out_features=n_beam, scale=np.sqrt(n_antenna), theta = theta)
        self.beam_selection = PowerPooling(2*n_beam)
    def forward(self, x):
        bf_signal = self.codebook(x)
        bf_power_sel = self.beam_selection(bf_signal)
        return bf_power_sel

class Self_Supervised_AnalogBeamformer(nn.Module):
    def __init__(self, n_antenna, n_beam):
        super(Self_Supervised_AnalogBeamformer, self).__init__()
        self.codebook = PhaseShifter(in_features=2*n_antenna, out_features=n_beam, scale=np.sqrt(n_antenna))
        self.compute_power = ComputePower(2*n_beam)
    def forward(self, x):
        bf_signal = self.codebook(x)
        bf_power = self.compute_power(bf_signal)
        return bf_power

def bf_gain_loss(y_pred, y_true):
    return -torch.mean(y_pred,dim=0)

def estimate_h(h_batch, model, n_antenna):
    h_batch_complex = h_batch[:,:n_antenna] + 1j*h_batch[:,n_antenna:]
    theta = model.codebook.theta.detach().clone().numpy()
    bf_codebook = np.exp(1j*theta)/np.sqrt(n_antenna)
    z = bf_codebook.conj().T @ h_batch_complex.T
    h_est = np.linalg.pinv(bf_codebook.conj().T) @ z
    h_est_cat = np.concatenate((h_est.real, h_est.imag),axis=0)
    return h_est_cat.T

def fast_adapt_est_h_self_supervised(batch, learner, loss_fn, adaptation_steps, shots):
    data, labels = batch
    # Separate data into adaptation/evalutation sets
    adaptation_indices = np.zeros(data.shape[0], dtype=bool)
    adaptation_indices[np.arange(shots) * 2] = True
    evaluation_indices = ~adaptation_indices
    adaptation_h = data[adaptation_indices]
    evaluation_h = data[evaluation_indices]

    # Adapt the model
    for step in range(adaptation_steps):
        adaptation_x = torch.from_numpy(estimate_h(adaptation_h, learner.module, num_antenna)).float() 
        bf_power = learner(adaptation_x)
        bf_argmax = torch.argmax(bf_power.detach().clone(),dim=1)
        loss = loss_fn(bf_power, bf_argmax)
        learner.adapt(loss)

    # Evaluate the adapted model
    evaluation_x = torch.from_numpy(estimate_h(evaluation_h, learner.module, num_antenna)).float()        
    bf_power = learner(evaluation_x)
    bf_argmax = torch.argmax(bf_power.detach().clone(),dim=1)
    valid_loss = loss_fn(bf_power, bf_argmax)
    return valid_loss
        
def train_est_h_self_supervised(train_batch, model, optimizer, loss_fn, train_steps):
    model.train()
    train_h,train_y = train_batch
    train_y = torch.from_numpy(train_y).float()
    train_loss = 0.0
    for step in range(train_steps):
        train_x = torch.from_numpy(estimate_h(train_h, model, num_antenna)).float() 
        optimizer.zero_grad()
        bf_power = model(train_x)
        bf_argmax = torch.argmax(bf_power.detach().clone(),dim=1)
        loss = loss_fn(bf_power, bf_argmax)
        loss.backward()
        optimizer.step()
        train_loss += loss.detach().item()
    train_loss /= train_steps
    return train_loss

def eval_est_h_self_supervised(val_batch, model, loss_fn):
    model.eval()
    val_h,val_y = val_batch
    val_x = torch.from_numpy(estimate_h(val_h, model, num_antenna)).float()
    bf_power = model(val_x)
    bf_argmax = torch.argmax(bf_power.detach().clone(),dim=1)    
    loss = loss_fn(bf_power,bf_argmax)
    return loss.item()
   
           
dataset = GaussianCenters(possible_loc=loc[:,:2],
                           n_clusters=n_clusters, arrival_rate = arrival_rate, cluster_variance = cluster_variance)

test_gains_maml = np.zeros((len(num_of_beams),ntest,dataset.n_clusters*dataset.arrival_rate))
test_gains_scratch = np.zeros((len(num_of_beams),ntest,dataset.n_clusters*dataset.arrival_rate))
test_gains_dft = np.zeros((len(num_of_beams),ntest,dataset.n_clusters*dataset.arrival_rate))

for i,N in enumerate(num_of_beams):
    print(str(N) + '-beams Codebook')
    
    # Model:
    # ------
    model = AnalogBeamformer(n_antenna = num_antenna, n_beam = N)
    maml = MAML(model, lr=fast_lr, first_order=False)
    # Training:
    # ---------
    optimizer = optim.Adam(model.parameters(),lr=meta_lr, betas=(0.9,0.999), amsgrad=False)
    loss_fn = bf_gain_loss

    for iteration in range(nepoch):
        optimizer.zero_grad()
        meta_train_error = 0.0
        meta_valid_error = 0.0
        for task in range(batch_size):
            dataset.change_cluster()
            # Compute meta-training loss
            learner = maml.clone()
            batch_idc = dataset.sample()
            batch = (h_concat_scaled[batch_idc,:],egc_gain_scaled[batch_idc])
            evaluation_error = fast_adapt_est_h_self_supervised(batch,
                                        learner,
                                        loss_fn,
                                        update_step,
                                        shots)
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
    
            # Compute meta-validation loss
            learner = maml.clone()
            batch_idc = dataset.sample()
            batch = (h_concat_scaled[batch_idc,:],egc_gain_scaled[batch_idc])
            evaluation_error = fast_adapt_est_h_self_supervised(batch,
                                        learner,
                                        loss_fn,
                                        update_step,
                                        shots)
            meta_valid_error += evaluation_error.item()
    
        # Print some metrics
        print('\n')
        print('Iteration', iteration)
        print('Meta Train Loss', meta_train_error / batch_size)
        print('Meta Valid Loss', meta_valid_error / batch_size)
    
        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            p.grad.data.mul_(1.0 / batch_size)
        optimizer.step()
    
        if iteration % 50 == 0:
            maml_bf_gains_val = []
            scratch_bf_gains_val = []
            dft_bf_gains_val = []
            
            for test_iter in range(nval):
                dataset.change_cluster()
                sample_idc_train = dataset.sample()
                x_train = h_concat_scaled[sample_idc_train,:]
                y_train = egc_gain_scaled[sample_idc_train]
            
                # model_maml = maml.module.clone()
                model_maml = AnalogBeamformer(n_antenna = num_antenna, n_beam = N, theta = torch.from_numpy(maml.module.codebook.theta.detach().clone().numpy()))
                opt_maml_model = optim.Adam(model_maml.parameters(),lr=fast_lr, betas=(0.9,0.999), amsgrad=False)
                train_loss_maml = train_est_h_self_supervised((x_train,y_train), model_maml, opt_maml_model, loss_fn, update_step)
                maml_theta = model_maml.codebook.theta.detach().clone().numpy()
                maml_codebook = np.exp(1j*maml_theta)/np.sqrt(num_antenna)       
                
                model_scratch = AnalogBeamformer(n_antenna = num_antenna, n_beam = N)
                opt_scratch_model = optim.Adam(model_scratch.parameters(),lr=fast_lr, betas=(0.9,0.999), amsgrad=False)
                train_loss_scratch = train_est_h_self_supervised((x_train,y_train), model_scratch, opt_scratch_model, loss_fn, update_step)
                scratch_theta = model_scratch.codebook.theta.detach().clone().numpy()
                scratch_codebook = np.exp(1j*scratch_theta)/np.sqrt(num_antenna)
            
                sample_idc_test = dataset.sample()
                x_test = h_concat_scaled[sample_idc_test,:]
                y_test = egc_gain_scaled[sample_idc_test]
                maml_bf_gains_val.extend(np.max(np.power(np.absolute(np.matmul(h[sample_idc_test,:], maml_codebook.conj())),2),axis=1))
                scratch_bf_gains_val.extend(np.max(np.power(np.absolute(np.matmul(h[sample_idc_test,:], scratch_codebook.conj())),2),axis=1))
                dft_bf_gains_val.extend(np.max(np.power(np.absolute(np.matmul(h[sample_idc_test,:], np.transpose(np.conj(DFT_codebook(N,num_antenna))))),2),axis=1))
              
            maml_bf_gains_val = 10*np.log10(maml_bf_gains_val)
            scratch_bf_gains_val = 10*np.log10(scratch_bf_gains_val)
            dft_bf_gains_val = 10*np.log10(dft_bf_gains_val)
            fig,ax = plt.subplots(figsize=(8,6))
            ax.hist(maml_bf_gains_val,bins=100,density=True,cumulative=True,histtype='step',label='MAML, {} beams'.format(num_of_beams[i]))    
            ax.hist(scratch_bf_gains_val,bins=100,density=True,cumulative=True,histtype='step',label='Learned from scratch, {} beams'.format(num_of_beams[i]))
            ax.hist(dft_bf_gains_val,bins=100,density=True,cumulative=True,histtype='step',label='DFT codebook,{} beams'.format(num_of_beams[i]))
            # tidy up the figure
            ax.grid(True)
            ax.legend(loc='upper left')
            #ax.set_title('Cumulative step histograms')
            ax.set_xlabel('BF Gain (dB)')
            ax.set_ylabel('Emperical CDF')
            ax.set_title('Codebook comparison with {} beams, Epoch {}.'.format(N, iteration))
            plt.show()
        
    for test_iter in range(ntest):
        dataset.change_cluster()
        sample_idc_train = dataset.sample()
        x_train = h_concat_scaled[sample_idc_train,:]
        y_train = egc_gain_scaled[sample_idc_train]
    
        # model_maml = maml.module.clone()
        model_maml = AnalogBeamformer(n_antenna = num_antenna, n_beam = N, theta = torch.from_numpy(maml.module.codebook.theta.clone().detach().numpy()))
        opt_maml_model = optim.Adam(model_maml.parameters(),lr=fast_lr, betas=(0.9,0.999), amsgrad=False)
        train_loss_maml = train_est_h_self_supervised((x_train,y_train), model_maml, opt_maml_model, loss_fn, update_step)
        maml_theta = model_maml.codebook.theta.clone().detach().numpy()
        maml_codebook = np.exp(1j*maml_theta)/np.sqrt(num_antenna)       
        
        model_scratch = AnalogBeamformer(n_antenna = num_antenna, n_beam = N)
        opt_scratch_model = optim.Adam(model_scratch.parameters(),lr=fast_lr, betas=(0.9,0.999), amsgrad=False)
        train_loss_scratch = train_est_h_self_supervised((x_train,y_train), model_scratch, opt_scratch_model, loss_fn, update_step)
        scratch_theta = model_scratch.codebook.theta.clone().detach().numpy()
        scratch_codebook = np.exp(1j*scratch_theta)/np.sqrt(num_antenna)
        
        sample_idc_test = dataset.sample()
        x_test = h_concat_scaled[sample_idc_test,:]
        y_test = egc_gain_scaled[sample_idc_test]
        test_gains_maml[i,test_iter,:] = np.max(np.power(np.absolute(np.matmul(h[sample_idc_test,:], maml_codebook.conj())),2),axis=1)
        test_gains_scratch[i,test_iter,:] = np.max(np.power(np.absolute(np.matmul(h[sample_idc_test,:], scratch_codebook.conj())),2),axis=1)
        test_gains_dft[i,test_iter,:] = np.max(np.power(np.absolute(np.matmul(h[sample_idc_test,:], np.transpose(np.conj(DFT_codebook(N,num_antenna))))),2),axis=1)

test_gains_maml = 10*np.log10(test_gains_maml)   
test_gains_scratch = 10*np.log10(test_gains_scratch)   
test_gains_dft = 10*np.log10(test_gains_dft) 

for i, N in enumerate(num_of_beams):
    fig,ax = plt.subplots(figsize=(8,6))
    ax.hist(test_gains_maml[i,:,:].flatten(),bins=100,density=True,cumulative=True,histtype='step',label='MAML, {} beams'.format(num_of_beams[i]))    
    ax.hist(test_gains_scratch[i,:,:].flatten(),bins=100,density=True,cumulative=True,histtype='step',label='Learned from scratch, {} beams'.format(num_of_beams[i]))
    ax.hist(test_gains_dft[i,:,:].flatten(),bins=100,density=True,cumulative=True,histtype='step',label='DFT codebook,{} beams'.format(num_of_beams[i]))
    # tidy up the figure
    ax.grid(True)
    ax.legend(loc='upper left')
    #ax.set_title('Cumulative step histograms')
    ax.set_xlabel('BF Gain (dB)')
    ax.set_ylabel('Emperical CDF')
    ax.set_title('Codebook comparison with {} beams, Epoch {}.'.format(N, iteration))
    plt.show()
    
# for i, N in enumerate(num_of_beams):
#     for test_iter in range(ntest):
#         fig,ax = plt.subplots(figsize=(8,6))
#         ax.hist(test_gains_maml[i,test_iter,:],bins=100,density=True,cumulative=True,histtype='step',label='MAML, {} beams'.format(num_of_beams[i]))    
#         ax.hist(test_gains_scratch[i,test_iter,:],bins=100,density=True,cumulative=True,histtype='step',label='Learned from scratch, {} beams'.format(num_of_beams[i]))
#         ax.hist(test_gains_dft[i,test_iter,:],bins=100,density=True,cumulative=True,histtype='step',label='DFT codebook,{} beams'.format(num_of_beams[i]))
#         # tidy up the figure
#         ax.grid(True)
#         ax.legend(loc='upper left')
#         #ax.set_title('Cumulative step histograms')
#         ax.set_xlabel('BF Gain (dB)')
#         ax.set_ylabel('Emperical CDF')
#         ax.set_title('Codebook comparison with {} beams, Epoch {}.'.format(N, iteration))
#         plt.show()
        
plt.figure(figsize=(8,6))
plt.plot(num_of_beams,[test_gains_maml[i,:,:].mean() for i in range(len(num_of_beams))], marker='+',label='MAML')    
plt.plot(num_of_beams,[test_gains_scratch[i,:,:].mean() for i in range(len(num_of_beams))],marker = 's', label='Learned from scratch')    
plt.plot(num_of_beams,[test_gains_dft[i,:,:].mean() for i in range(len(num_of_beams))],marker='o', label='DFT') 
plt.xticks(num_of_beams,num_of_beams)
plt.grid(True)
plt.legend(loc='lower right')   
plt.xlabel('num of beams')
plt.ylabel('Avg. BF Gain (dB)')
plt.show()