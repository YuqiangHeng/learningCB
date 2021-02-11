# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 16:14:35 2021

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
from beam_utils import GaussianCenters, DFT_codebook, DFT_codebook_blockmatrix, bf_gain_loss, plot_codebook_pattern, DFT_beam

import random
import torch
from learn2learn.algorithms import MAML
from learn2learn.utils import clone_module,detach_module

seed = 7
np.random.seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# n_narrow_beams = [128, 128, 128, 128, 128, 128]
n_nb = 128
n_wide_beams = [4, 6, 8, 10, 12, 16]
n_antenna = 64
antenna_sel = np.arange(n_antenna)

# --------------------------
# MAML training parameters
# --------------------------
batch_size = 200
nepoch = 1000
shots = 1000
update_step = 5
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

dft_nb_codebook = DFT_codebook(nseg=n_nb,n_antenna=n_antenna)
label = np.argmax(np.power(np.absolute(np.matmul(h_scaled, dft_nb_codebook.conj().T)),2),axis=1)
soft_label = np.power(np.absolute(np.matmul(h, dft_nb_codebook.conj().T)),2)
    
class Beam_Classifier(nn.Module):
    def __init__(self, n_antenna, n_wide_beam, n_narrow_beam, trainable_codebook = True, theta = None):
        super(Beam_Classifier, self).__init__()
        self.trainable_codebook = trainable_codebook
        self.n_antenna = n_antenna
        self.n_wide_beam = n_wide_beam
        self.n_narrow_beam = n_narrow_beam
        if trainable_codebook:
            self.codebook = PhaseShifter(in_features=2*n_antenna, out_features=n_wide_beam, scale=np.sqrt(n_antenna), theta=theta)
        else:
            dft_codebook = DFT_codebook_blockmatrix(n_antenna=n_antenna, nseg=n_wide_beam)
            self.codebook = torch.from_numpy(dft_codebook).float()
            self.codebook.requires_grad = False
        self.compute_power = ComputePower(2*n_wide_beam)
        self.relu = nn.ReLU()
        self.dense1 = nn.Linear(in_features=n_wide_beam, out_features=2*n_wide_beam)
        self.dense2 = nn.Linear(in_features=2*n_wide_beam, out_features=3*n_wide_beam)
        self.dense3 = nn.Linear(in_features=3*n_wide_beam, out_features=n_narrow_beam)
        self.softmax = nn.Softmax()
    def forward(self, x):
        if self.trainable_codebook:
            bf_signal = self.codebook(x)
        else:
            bf_signal = torch.matmul(x,self.codebook)
        bf_power = self.compute_power(bf_signal)
        out = self.relu(bf_power)
        out = self.relu(self.dense1(out))
        out = self.relu(self.dense2(out))
        out = self.dense3(out)
        return out
    def get_codebook(self) -> np.ndarray:
        if self.trainable_codebook:
            return self.codebook.get_weights().detach().clone().numpy()
        else:
            return DFT_codebook(nseg=self.n_wide_beam,n_antenna=self.n_antenna).T


def fast_adapt(batch, learner, loss, adaptation_steps, shots):
    data, labels = batch
    data, labels = torch.from_numpy(data).float(),torch.from_numpy(labels).long()

    # Separate data into adaptation/evalutation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    # Adapt the model
    for step in range(adaptation_steps):
        train_error = loss(learner(adaptation_data), adaptation_labels)
        learner.adapt(train_error)

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)
    return valid_error
        
dataset = GaussianCenters(possible_loc=loc[:,:2],
                           n_clusters=n_clusters, arrival_rate = arrival_rate, cluster_variance = cluster_variance)

# test_gains_maml = np.zeros((len(n_wide_beams),ntest,dataset.n_clusters*dataset.arrival_rate))
# test_gains_scratch = np.zeros((len(n_wide_beams),ntest,dataset.n_clusters*dataset.arrival_rate))
# test_gains_dft = np.zeros((len(n_wide_beams),ntest,dataset.n_clusters*dataset.arrival_rate))

test_snr_maml_all = []
test_snr_scratch_all = []
test_snr_optimal_all = []

for n_wb in n_wide_beams:
    print('{} Wide Beams, {} Narrow Beams.'.format(n_wb,n_nb))
    # Model:
    # ------
    model = Beam_Classifier(n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,trainable_codebook=True)
    maml = MAML(model, lr=fast_lr, first_order=False)
    # Training:
    # ---------
    optimizer = optim.Adam(model.parameters(),lr=meta_lr, betas=(0.9,0.999), amsgrad=False)
    loss_fn = nn.CrossEntropyLoss()

    for iteration in range(nepoch):
        optimizer.zero_grad()
        meta_train_error = 0.0
        meta_valid_error = 0.0
        for task in range(batch_size):
            dataset.change_cluster()
            # Compute meta-training loss
            learner = maml.clone()
            batch_idc = dataset.sample()
            batch = (h_concat_scaled[batch_idc,:],label[batch_idc])
            evaluation_error = fast_adapt(batch,
                                        learner,
                                        loss_fn,
                                        update_step,
                                        shots)
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
    
            # Compute meta-validation loss
            learner = maml.clone()
            batch_idc = dataset.sample()
            batch = (h_concat_scaled[batch_idc,:],label[batch_idc])
            evaluation_error = fast_adapt(batch,
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
            optimal_bf_gains_val = []
            
            for val_iter in range(nval):
                dataset.change_cluster()
                sample_idc = dataset.sample()
                sample_idc_train = sample_idc[:shots]
                x_train = torch.from_numpy(h_concat_scaled[sample_idc_train,:]).float()
                y_train = torch.from_numpy(label[sample_idc_train]).long()
            
                train_loss_maml = []        
                # model_maml = maml.module.clone()
                model_maml = Beam_Classifier(n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,trainable_codebook=True,
                                             theta = torch.from_numpy(maml.module.codebook.theta.detach().clone().numpy()))
                opt_maml_model = optim.Adam(model_maml.parameters(),lr=fast_lr, betas=(0.9,0.999), amsgrad=False)
                for step in range(update_step):
                    opt_maml_model.zero_grad()
                    output = model_maml(x_train)
                    loss = loss_fn(output, y_train)
                    loss.backward()
                    opt_maml_model.step()
                    train_loss_maml.append(loss.detach().item())
                maml_theta = model_maml.codebook.theta.clone().detach().numpy()
                maml_codebook = np.exp(1j*maml_theta)/np.sqrt(n_antenna)       
                
                train_loss_scratch = []        
                model_scratch = Beam_Classifier(n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,trainable_codebook=True)
                opt_scratch_model = optim.Adam(model_scratch.parameters(),lr=fast_lr, betas=(0.9,0.999), amsgrad=False)
                for step in range(update_step):
                    opt_scratch_model.zero_grad()
                    output = model_scratch(x_train)
                    loss = loss_fn(output, y_train)
                    loss.backward()
                    opt_scratch_model.step()
                    train_loss_scratch.append(loss.detach().item())
                scratch_theta = model_scratch.codebook.theta.clone().detach().numpy()
                scratch_codebook = np.exp(1j*scratch_theta)/np.sqrt(n_antenna)
                
                if plot_training_loss_history:
                    plt.figure(figsize=(8,6))
                    plt.plot(train_loss_maml,label='maml')
                    plt.plot(train_loss_scratch,label='scratch')
                    plt.legend()
                    plt.title('Validation training loss, epoch {}, val iter {}'.format(iteration,val_iter))
                    plt.show()
            
                sample_idc_test = sample_idc[shots:]
                x_test = h_concat_scaled[sample_idc_test,:]
                y_test = label[sample_idc_test]
                y_test_soft = soft_label[sample_idc_test,:]
                
                y_test_predict_maml = model_maml(torch.from_numpy(x_test).float()).detach().numpy()
                topk_sorted_test_maml = (-y_test_predict_maml).argsort()
                topk_bf_gain_maml = []
                for ue_bf_gain, pred_sort in zip(y_test_soft,topk_sorted_test_maml):
                    topk_gains = [ue_bf_gain[pred_sort[:k]].max() for k in range(1,11)]
                    topk_bf_gain_maml.append(topk_gains)
                topk_bf_gain_maml = np.array(topk_bf_gain_maml)

                y_test_predict_scratch = model_scratch(torch.from_numpy(x_test).float()).detach().numpy()
                topk_sorted_test_scratch = (-y_test_predict_scratch).argsort()
                topk_bf_gain_scratch = []
                for ue_bf_gain, pred_sort in zip(y_test_soft,topk_sorted_test_scratch):
                    topk_gains = [ue_bf_gain[pred_sort[:k]].max() for k in range(1,11)]
                    topk_bf_gain_scratch.append(topk_gains)
                topk_bf_gain_scratch = np.array(topk_bf_gain_scratch)
                            
                maml_bf_gains_val.extend(topk_bf_gain_maml)
                scratch_bf_gains_val.extend(topk_bf_gain_scratch)
                optimal_bf_gains_val.extend(y_test_soft.max(axis=1))
              
            maml_snr_val = 30 + 10*np.log10(maml_bf_gains_val) + 94 -13
            scratch_snr_val = 30 + 10*np.log10(scratch_bf_gains_val) + 94 -13
            optimal_snr_val = 30 + 10*np.log10(optimal_bf_gains_val) + 94 -13
            fig,ax = plt.subplots(figsize=(8,6))
            for k in [0,2]:
                ax.hist(maml_snr_val[:,k],bins=100,density=True,cumulative=True,histtype='step',label='MAML, k={}'.format(k+1))    
                ax.hist(scratch_snr_val[:,k],bins=100,density=True,cumulative=True,histtype='step',label='Learned from scratch, k={}'.format(k+1))
            ax.hist(optimal_snr_val,bins=100,density=True,cumulative=True,histtype='step',label='Optimal')
            # tidy up the figure
            ax.grid(True)
            ax.legend(loc='upper left')
            #ax.set_title('Cumulative step histograms')
            ax.set_xlabel('SNR (dB)')
            ax.set_ylabel('Emperical CDF')
            ax.set_title('SNR with {} beams, Epoch {}.'.format(n_wb, iteration))
            plt.show()
        
    maml_bf_gains_test = []
    scratch_bf_gains_test = []
    optimal_bf_gains_test = []
    
    for test_iter in range(ntest):
        dataset.change_cluster()
        sample_idc = dataset.sample()
        sample_idc_train = sample_idc[:shots]
        x_train = torch.from_numpy(h_concat_scaled[sample_idc_train,:]).float()
        y_train = torch.from_numpy(label[sample_idc_train]).long()
    
        train_loss_maml = []        
        # model_maml = maml.module.clone()
        model_maml = Beam_Classifier(n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,trainable_codebook=True,
                                     theta = torch.from_numpy(maml.module.codebook.theta.detach().clone().numpy()))
        opt_maml_model = optim.Adam(model_maml.parameters(),lr=fast_lr, betas=(0.9,0.999), amsgrad=False)
        for step in range(update_step):
            opt_maml_model.zero_grad()
            output = model_maml(x_train)
            loss = loss_fn(output, y_train)
            loss.backward()
            opt_maml_model.step()
            train_loss_maml.append(loss.detach().item())
        maml_theta = model_maml.codebook.theta.clone().detach().numpy()
        maml_codebook = np.exp(1j*maml_theta)/np.sqrt(n_antenna)       
        
        train_loss_scratch = []        
        model_scratch = Beam_Classifier(n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,trainable_codebook=True)
        opt_scratch_model = optim.Adam(model_scratch.parameters(),lr=fast_lr, betas=(0.9,0.999), amsgrad=False)
        for step in range(update_step):
            opt_scratch_model.zero_grad()
            output = model_scratch(x_train)
            loss = loss_fn(output, y_train)
            loss.backward()
            opt_scratch_model.step()
            train_loss_scratch.append(loss.detach().item())
        scratch_theta = model_scratch.codebook.theta.clone().detach().numpy()
        scratch_codebook = np.exp(1j*scratch_theta)/np.sqrt(n_antenna)
        
        if plot_training_loss_history:
            plt.figure(figsize=(8,6))
            plt.plot(train_loss_maml,label='maml')
            plt.plot(train_loss_scratch,label='scratch')
            plt.legend()
            plt.title('Training loss during testing, epoch {}, test iter {}'.format(iteration,test_iter))
            plt.show()
    
        sample_idc_test = sample_idc[shots:]
        x_test = h_concat_scaled[sample_idc_test,:]
        y_test = label[sample_idc_test]
        y_test_soft = soft_label[sample_idc_test,:]
        
        y_test_predict_maml = model_maml(torch.from_numpy(x_test).float()).detach().numpy()
        topk_sorted_test_maml = (-y_test_predict_maml).argsort()
        topk_bf_gain_maml = []
        for ue_bf_gain, pred_sort in zip(y_test_soft,topk_sorted_test_maml):
            topk_gains = [ue_bf_gain[pred_sort[:k]].max() for k in range(1,11)]
            topk_bf_gain_maml.append(topk_gains)
        topk_bf_gain_maml = np.array(topk_bf_gain_maml)

        y_test_predict_scratch = model_scratch(torch.from_numpy(x_test).float()).detach().numpy()
        topk_sorted_test_scratch = (-y_test_predict_scratch).argsort()
        topk_bf_gain_scratch = []
        for ue_bf_gain, pred_sort in zip(y_test_soft,topk_sorted_test_scratch):
            topk_gains = [ue_bf_gain[pred_sort[:k]].max() for k in range(1,11)]
            topk_bf_gain_scratch.append(topk_gains)
        topk_bf_gain_scratch = np.array(topk_bf_gain_scratch)
                    
        maml_bf_gains_test.extend(topk_bf_gain_maml)
        scratch_bf_gains_test.extend(topk_bf_gain_scratch)
        optimal_bf_gains_test.extend(y_test_soft.max(axis=1))
        
    maml_bf_gains_test = np.array(maml_bf_gains_test)  
    scratch_bf_gains_test = np.array(scratch_bf_gains_test)
    optimal_bf_gains_test = np.array(optimal_bf_gains_test)
    
    maml_snr_test = 30 + 10*np.log10(maml_bf_gains_test) + 94 -13
    scratch_snr_test = 30 + 10*np.log10(scratch_bf_gains_test) + 94 -13
    optimal_snr_test = 30 + 10*np.log10(optimal_bf_gains_test) + 94 -13
    
    test_snr_maml_all.append(maml_snr_test)
    test_snr_scratch_all.append(scratch_snr_test)
    test_snr_optimal_all.append(optimal_snr_test)
    
    fig,ax = plt.subplots(figsize=(8,6))
    for k in [0,2]:
        ax.hist(maml_snr_test[:,k],bins=100,density=True,cumulative=True,histtype='step',label='MAML, k={}'.format(k+1))    
        ax.hist(scratch_snr_test[:,k],bins=100,density=True,cumulative=True,histtype='step',label='Learned from scratch, k={}'.format(k+1))
    ax.hist(optimal_snr_test,bins=100,density=True,cumulative=True,histtype='step',label='Optimal')
    # tidy up the figure
    ax.grid(True)
    ax.legend(loc='upper left')
    #ax.set_title('Cumulative step histograms')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Emperical CDF')
    ax.set_title('SNR with {} beams, Epoch {}.'.format(n_wb, iteration))
    plt.show()

test_snr_maml_all = np.array(test_snr_maml_all)
test_snr_scratch_all = np.array(test_snr_scratch_all)
test_snr_optimal_all = np.array(test_snr_optimal_all)

    

        
plt.figure(figsize=(8,6))
for k in [0,2]:
    plt.plot(n_wide_beams,test_snr_maml_all[:,:,k].mean(axis=-1), marker='+',label='MAML, k={}'.format(k+1))    
    plt.plot(n_wide_beams,test_snr_scratch_all[:,:,k].mean(axis=-1),marker = 's', label='Learned from scratch, k={}'.format(k+1))    
plt.plot(n_wide_beams,test_snr_optimal_all.mean(axis=-1),marker='o', label='Optimal') 
plt.xticks(n_wide_beams,n_wide_beams)
plt.grid(True)
plt.legend(loc='lower right')   
plt.xlabel('num of beams')
plt.ylabel('Avg. SNR (dB)')
plt.show()