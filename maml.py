# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 18:56:13 2021

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

# use_cuda = torch.cuda.is_available()
use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")

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
n_clusters = 5
arrival_rate = int(shots*2/n_clusters)
cluster_variance = 20

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
# train_idc, test_idc = train_test_split(np.arange(h.shape[0]),test_size=0.4)
# val_idc, test_idc = train_test_split(test_idc,test_size=0.5)
# x_train,y_train = h_concat_scaled[train_idc,:],egc_gain_scaled[train_idc]
# x_val,y_val = h_concat_scaled[val_idc,:],egc_gain_scaled[val_idc]
# x_test,y_test = h_concat_scaled[test_idc,:],egc_gain_scaled[test_idc]

# # torch_x_train = torch.from_numpy(x_train).type(torch.LongTensor)
# # torch_y_train = torch.from_numpy(y_train).type(torch.LongTensor) # data type is long
# # torch_x_val = torch.from_numpy(x_val).type(torch.LongTensor)
# # torch_y_val = torch.from_numpy(y_val).type(torch.LongTensor)
# # torch_x_test = torch.from_numpy(x_test).type(torch.LongTensor)
# # torch_y_test = torch.from_numpy(y_test).type(torch.LongTensor)

# torch_x_train = torch.from_numpy(x_train)
# torch_y_train = torch.from_numpy(y_train)
# torch_x_val = torch.from_numpy(x_val)
# torch_y_val = torch.from_numpy(y_val)
# torch_x_test = torch.from_numpy(x_test)
# torch_y_test = torch.from_numpy(y_test)

# # Pytorch train and test sets
# train = torch.utils.data.TensorDataset(torch_x_train,torch_y_train)
# val = torch.utils.data.TensorDataset(torch_x_val,torch_y_val)
# test = torch.utils.data.TensorDataset(torch_x_test,torch_y_test)

# # data loader
# train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
# val_loader = torch.utils.data.DataLoader(val, batch_size = batch_size, shuffle = False)
# test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)

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

def fit(model, train_loader, val_loader, opt, loss_fn, EPOCHS):
    optimizer = opt
    train_loss_hist = []
    val_loss_hist = []
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            var_X_batch = X_batch.float()
            var_y_batch = y_batch.float()
            optimizer.zero_grad()
            output = model(var_X_batch)
            loss = loss_fn(output, var_y_batch.unsqueeze(dim=-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()
        train_loss /= batch_idx + 1
        model.eval()
        val_loss = 0
        for batch_idx, (X_batch, y_batch) in enumerate(val_loader):
            var_X_batch = X_batch.float()
            var_y_batch = y_batch.float()  
            output = model(var_X_batch)
            loss = loss_fn(output, var_y_batch.unsqueeze(dim=-1))
            val_loss += loss.detach().item()
        val_loss /= batch_idx + 1
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        if epoch % 10 == 0:
            print('Epoch : {} Training loss = {:.2f}, Validation loss = {:.2f}.'.format(epoch, train_loss, val_loss))
    return train_loss_hist, val_loss_hist

def fit_genius(model:AnalogBeamformer, train_loader, val_loader, opt, loss_fn, EPOCHS):
    optimizer = opt
    train_loss_hist = []
    val_loss_hist = []
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            theta = model.codebook.theta.detach().clone().numpy()
            learned_codebook = np.exp(1j*theta)/np.sqrt(num_antenna)
            x_batch_np = X_batch.detach().clone().numpy()
            x_batch_complex = x_batch_np[:,:num_antenna] + 1j*x_batch_np[:,num_antenna:]
            # z = np.matmul(x_batch_complex, learned_codebook.conj())
            # h_est = np.matmul(np.linalg.pinv(learned_codebook.conj().T),z)
            z = learned_codebook.conj().T @ x_batch_complex.T
            h_est = np.linalg.pinv(learned_codebook.conj().T) @ z
            h_est_cat = np.concatenate((h_est.real, h_est.imag),axis=0)
            var_X_batch = torch.from_numpy(h_est_cat.T).float()
            var_y_batch = y_batch.float()
            optimizer.zero_grad()
            output = model(var_X_batch)
            loss = loss_fn(output, var_y_batch.unsqueeze(dim=-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()
        train_loss /= batch_idx + 1
        model.eval()
        val_loss = 0
        for batch_idx, (X_batch, y_batch) in enumerate(val_loader):
            theta = model.codebook.theta.detach().clone().numpy()
            learned_codebook = np.exp(1j*theta)/np.sqrt(num_antenna)
            x_batch_np = X_batch.detach().clone().numpy()
            x_batch_complex = x_batch_np[:,:num_antenna] + 1j*x_batch_np[:,num_antenna:]
            # z = np.matmul(x_batch_complex, learned_codebook.conj())
            # h_est = np.matmul(np.linalg.pinv(learned_codebook.conj().T),z)
            z = learned_codebook.conj().T @ x_batch_complex.T
            h_est = np.linalg.pinv(learned_codebook.conj().T) @ z
            h_est_cat = np.concatenate((h_est.real, h_est.imag),axis=0)
            var_X_batch = torch.from_numpy(h_est_cat.T).float()
            var_y_batch = y_batch.float()
            output = model(var_X_batch)
            loss = loss_fn(output, var_y_batch.unsqueeze(dim=-1))
            val_loss += loss.detach().item()
        val_loss /= batch_idx + 1
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        if epoch % 10 == 0:
            print('Epoch : {} Training loss = {:.2f}, Validation loss = {:.2f}.'.format(epoch, train_loss, val_loss))
    return train_loss_hist, val_loss_hist

# def fit_self_supervised(model:Self_Supervised_AnalogBeamformer, train_loader, val_loader, opt, EPOCHS, loss_fn = nn.CrossEntropyLoss()):
#     optimizer = opt
#     train_loss_hist = []
#     val_loss_hist = []
#     for epoch in range(EPOCHS):
#         model.train()
#         train_loss = 0
#         for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
#             theta = model.codebook.theta.detach().clone().numpy()
#             learned_codebook = np.exp(1j*theta)/np.sqrt(num_antenna)
#             x_batch_np = X_batch.detach().clone().numpy()
#             x_batch_complex = x_batch_np[:,:num_antenna] + 1j*x_batch_np[:,num_antenna:]
#             z = learned_codebook.conj().T @ x_batch_complex.T
#             h_est = np.linalg.pinv(learned_codebook.conj().T) @ z
#             h_est_cat = np.concatenate((h_est.real, h_est.imag),axis=0)
#             var_X_batch = torch.from_numpy(h_est_cat.T).float()
#             optimizer.zero_grad()
#             bf_power =  model(var_X_batch)
#             bf_argmax = torch.argmax(bf_power.detach().clone(),dim=1)
#             loss = loss_fn(bf_power, bf_argmax)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.detach().item()
#         train_loss /= batch_idx + 1
#         model.eval()
#         val_loss = 0
#         for batch_idx, (X_batch, y_batch) in enumerate(val_loader):
#             theta = model.codebook.theta.detach().clone().numpy()
#             learned_codebook = np.exp(1j*theta)/np.sqrt(num_antenna)
#             x_batch_np = X_batch.detach().clone().numpy()
#             x_batch_complex = x_batch_np[:,:num_antenna] + 1j*x_batch_np[:,num_antenna:]
#             # z = np.matmul(x_batch_complex, learned_codebook.conj())
#             # h_est = np.matmul(np.linalg.pinv(learned_codebook.conj().T),z)
#             z = learned_codebook.conj().T @ x_batch_complex.T
#             h_est = np.linalg.pinv(learned_codebook.conj().T) @ z
#             h_est_cat = np.concatenate((h_est.real, h_est.imag),axis=0)
#             var_X_batch = torch.from_numpy(h_est_cat.T).float()
#             bf_power =  model(var_X_batch)
#             bf_argmax = torch.argmax(bf_power.detach().clone(),dim=1)
#             loss = loss_fn(bf_power, bf_argmax)
#             val_loss += loss.detach().item()
#         val_loss /= batch_idx + 1
#         train_loss_hist.append(train_loss)
#         val_loss_hist.append(val_loss)
#         if epoch % 10 == 0:
#             print('Epoch : {} Training loss = {:.2f}, Validation loss = {:.2f}.'.format(epoch, train_loss, val_loss))
#     return train_loss_hist, val_loss_hist

# # ------------------------------------------------------------------
# # Codebook learning using H estimate, GD on crossentropy of softmax and argmax of bf power
# # ------------------------------------------------------------------
# learned_codebook_gains_self_supervised = np.zeros((len(num_of_beams),len(test_idc)))
# learned_codebooks_self_supervised = []
# for i,N in enumerate(num_of_beams):
#     print(str(N) + '-beams Codebook')
    
#     # Model:
#     # ------
#     model = Self_Supervised_AnalogBeamformer(n_antenna = num_antenna, n_beam = N)
#     # Training:
#     # ---------
#     opt = optim.Adam(model.parameters(),lr=0.01, betas=(0.9,0.999), amsgrad=False)
    
#     train_hist, val_hist = fit_self_supervised(model, train_loader, val_loader, opt, 100)    


#     plt.figure()
#     plt.plot(np.array(train_hist),label='train loss')
#     plt.plot(np.array(val_hist),label='val loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('loss')
#     plt.title('Genius. {} beams'.format(N))
#     # Extract learned codebook:
#     # -------------------------
#     theta = model.codebook.theta.detach().clone().numpy()
#     print(theta.shape)

#     learned_codebook = np.exp(1j*theta)/np.sqrt(num_antenna)
#     learned_codebooks_self_supervised.append(learned_codebook)
#     learned_codebook_gains_self_supervised[i,:] = np.max(np.power(np.absolute(np.matmul(h[test_idc,:], learned_codebook.conj())),2),axis=1)
# learned_codebook_gains_self_supervised = 10*np.log10(learned_codebook_gains_self_supervised)

def fast_adapt(batch, learner, loss, adaptation_steps, shots, device):
    data, labels = batch
    data, labels = torch.from_numpy(data).float(),torch.from_numpy(labels).long()
    data, labels = data.to(device), labels.to(device)

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
        
#-------------------------------------------#
# Comppare between learned codebook and DFT codebook on test set
#-------------------------------------------#    
# dft_gains = np.zeros((len(num_of_beams),len(test_idc)))
# for i, nbeams in enumerate(num_of_beams):
#     dft_gains[i,:] = np.max(np.power(np.absolute(np.matmul(h[test_idc,:], np.transpose(np.conj(DFT_codebook(nbeams,num_antenna))))),2),axis=1)
# dft_gains = 10*np.log10(dft_gains)

# fig,ax = plt.subplots(figsize=(8,6))
# for i in range(len(num_of_beams)):
#     ax.hist(learned_codebook_gains[i,:],bins=100,density=True,cumulative=True,histtype='step',label='learned codebook unsupervised, {} beams'.format(num_of_beams[i]))    
#     ax.hist(learned_codebook_gains_supervised[i,:],bins=100,density=True,cumulative=True,histtype='step',label='learned codebook supervised, {} beams'.format(num_of_beams[i]))
#     ax.hist(dft_gains[i,:],bins=100,density=True,cumulative=True,histtype='step',label='DFT codebook,{} beams'.format(num_of_beams[i]))
# # tidy up the figure
# ax.grid(True)
# ax.legend(loc='upper left')
# #ax.set_title('Cumulative step histograms')
# ax.set_xlabel('BF Gain (dB)')
# ax.set_ylabel('Emperical CDF')
# plt.show()

# for i, N in enumerate(num_of_beams):
#     fig,ax = plt.subplots(figsize=(8,6))
#     ax.hist(dft_gains[i,:],bins=100,density=True,cumulative=True,histtype='step',label='DFT codebook,{} beams'.format(num_of_beams[i]))
#     # tidy up the figure
#     ax.grid(True)
#     ax.legend(loc='upper left')
#     #ax.set_title('Cumulative step histograms')
#     ax.set_xlabel('BF Gain (dB)')
#     ax.set_ylabel('Emperical CDF')
#     ax.set_title('Codebook comparison with {} beams.'.format(N))
#     plt.show()

# plt.figure()
# plt.plot(num_of_beams,np.mean(dft_gains,axis=1),label='DFT')
# plt.legend()
# plt.xlabel('number of beams')
# plt.ylabel('avg BF gain (dB)')
# plt.show()
dataset = GaussianCenters(possible_loc=loc[:,:2],
                           n_clusters=n_clusters, arrival_rate = arrival_rate, cluster_variance = cluster_variance)

test_gains_maml = np.zeros((len(num_of_beams),ntest,dataset.n_clusters*dataset.arrival_rate))
test_gains_scratch = np.zeros((len(num_of_beams),ntest,dataset.n_clusters*dataset.arrival_rate))
test_gains_dft = np.zeros((len(num_of_beams),ntest,dataset.n_clusters*dataset.arrival_rate))

for i,N in enumerate(num_of_beams):
    print(str(N) + '-beams Codebook')
    
    # Model:
    # ------
    model = AnalogBeamformer(n_antenna = num_antenna, n_beam = N).to(device)
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
            batch = (h_concat_scaled[batch_idc,:],h_concat_scaled[batch_idc,:])
            evaluation_error = fast_adapt(batch,
                                        learner,
                                        loss_fn,
                                        update_step,
                                        shots,
                                        device)
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
    
            # Compute meta-validation loss
            learner = maml.clone()
            batch_idc = dataset.sample()
            batch = (h_concat_scaled[batch_idc,:],h_concat_scaled[batch_idc,:])
            evaluation_error = fast_adapt(batch,
                                        learner,
                                        loss_fn,
                                        update_step,
                                        shots,
                                        device)
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
                x_train = torch.from_numpy(h_concat_scaled[sample_idc_train,:]).float()
                y_train = torch.from_numpy(egc_gain_scaled[sample_idc_train]).float()
            
                train_loss_maml = []        
                # model_maml = maml.module.clone()
                model_maml = AnalogBeamformer(n_antenna = num_antenna, n_beam = N, theta = torch.from_numpy(maml.module.codebook.theta.clone().detach().numpy()))
                opt_maml_model = optim.Adam(model_maml.parameters(),lr=fast_lr, betas=(0.9,0.999), amsgrad=False)
                for step in range(update_step):
                    opt_maml_model.zero_grad()
                    output = model_maml(x_train)
                    loss = loss_fn(output, y_train.unsqueeze(dim=-1))
                    loss.backward()
                    opt_maml_model.step()
                    train_loss_maml.append(loss.detach().item())
                maml_theta = model_maml.codebook.theta.clone().detach().numpy()
                maml_codebook = np.exp(1j*maml_theta)/np.sqrt(num_antenna)       
                
                train_loss_scratch = []        
                model_scratch = AnalogBeamformer(n_antenna = num_antenna, n_beam = N).to(device)
                opt_scratch_model = optim.Adam(model_scratch.parameters(),lr=fast_lr, betas=(0.9,0.999), amsgrad=False)
                for step in range(update_step):
                    opt_scratch_model.zero_grad()
                    output = model_scratch(x_train)
                    loss = loss_fn(output, y_train.unsqueeze(dim=-1))
                    loss.backward()
                    opt_scratch_model.step()
                    train_loss_scratch.append(loss.detach().item())
                scratch_theta = model_scratch.codebook.theta.clone().detach().numpy()
                scratch_codebook = np.exp(1j*scratch_theta)/np.sqrt(num_antenna)
                
                if plot_training_loss_history:
                    plt.figure(figsize=(8,6))
                    plt.plot(train_loss_maml,label='maml')
                    plt.plot(train_loss_scratch,label='scratch')
                    plt.legend()
                    plt.title('Training loss during testing, epoch {}, test iter {}'.format(iteration,test_iter))
                    plt.show()
            
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
        x_train = torch.from_numpy(h_concat_scaled[sample_idc_train,:]).float()
        y_train = torch.from_numpy(egc_gain_scaled[sample_idc_train]).float()
    
        train_loss_maml = []        
        # model_maml = maml.module.clone()
        model_maml = AnalogBeamformer(n_antenna = num_antenna, n_beam = N, theta = torch.from_numpy(maml.module.codebook.theta.clone().detach().numpy()))
        opt_maml_model = optim.Adam(model_maml.parameters(),lr=fast_lr, betas=(0.9,0.999), amsgrad=False)
        for step in range(update_step):
            opt_maml_model.zero_grad()
            output = model_maml(x_train)
            loss = loss_fn(output, y_train.unsqueeze(dim=-1))
            loss.backward()
            opt_maml_model.step()
            train_loss_maml.append(loss.detach().item())
        maml_theta = model_maml.codebook.theta.clone().detach().numpy()
        maml_codebook = np.exp(1j*maml_theta)/np.sqrt(num_antenna)       
        
        train_loss_scratch = []        
        model_scratch = AnalogBeamformer(n_antenna = num_antenna, n_beam = N).to(device)
        opt_scratch_model = optim.Adam(model_scratch.parameters(),lr=fast_lr, betas=(0.9,0.999), amsgrad=False)
        for step in range(update_step):
            opt_scratch_model.zero_grad()
            output = model_scratch(x_train)
            loss = loss_fn(output, y_train.unsqueeze(dim=-1))
            loss.backward()
            opt_scratch_model.step()
            train_loss_scratch.append(loss.detach().item())
        scratch_theta = model_scratch.codebook.theta.clone().detach().numpy()
        scratch_codebook = np.exp(1j*scratch_theta)/np.sqrt(num_antenna)
        
        if plot_training_loss_history:
            plt.figure(figsize=(8,6))
            plt.plot(train_loss_maml,label='maml')
            plt.plot(train_loss_scratch,label='scratch')
            plt.legend()
            plt.title('Training loss during testing, epoch {}, test iter {}'.format(iteration,test_iter))
            plt.show()
        
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
        