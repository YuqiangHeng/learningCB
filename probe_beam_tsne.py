# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 17:03:52 2021

@author: ethan
"""

import time
import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from ComplexLayers_Torch import PhaseShifter, PowerPooling, ComputePower
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
# from beam_utils import DFT_codebook, DFT_codebook_blockmatrix, plot_codebook_pattern, DFT_beam
from beam_utils import ULA_DFT_codebook as DFT_codebook
from beam_utils import ULA_DFT_codebook_blockmatrix as DFT_codebook_blockmatrix
from beam_utils import plot_codebook_pattern
import seaborn as sns
from sklearn.utils import resample

np.random.seed(7)
n_narrow_beams = 128
n_wide_beams = [4, 6, 8, 10, 12, 16]
n_antenna = 64
antenna_sel = np.arange(n_antenna)
tsne_perplexity = 20

dataset_name = 'Rosslyn_ULA' # 'Rosslyn_ULA' or 'O28B_ULA'

# Training and testing data:
# --------------------------
batch_size = 500
#-------------------------------------------#
# Here should be the data_preparing function
# It is expected to return:
# train_inp, train_out, val_inp, and val_out
#-------------------------------------------#
if dataset_name == 'Rosslyn_ULA':
    h_real = np.load('D://Github Repositories/mmWave Beam Management/H_Matrices FineGrid/MISO_Static_FineGrid_Hmatrices_real.npy')[:,antenna_sel]
    h_imag = np.load('D://Github Repositories/mmWave Beam Management/H_Matrices FineGrid/MISO_Static_FineGrid_Hmatrices_imag.npy')[:,antenna_sel]
    loc = np.load('D://Github Repositories/mmWave Beam Management/H_Matrices FineGrid/MISO_Static_FineGrid_UE_location.npy')
    # h_real = np.load('/Users/yh9277/Dropbox/ML Beam Alignment/Data/H_Matrices FineGrid/MISO_Static_FineGrid_Hmatrices_real.npy')
    # h_imag = np.load('/Users/yh9277/Dropbox/ML Beam Alignment/Data/H_Matrices FineGrid/MISO_Static_FineGrid_Hmatrices_imag.npy')
elif dataset_name == 'O28B_ULA':
    fname_h_real = 'D://Github Repositories/DeepMIMO-codes/DeepMIMO_Dataset_Generation_v1.1/DeepMIMO_Dataset_Generation_v1.1/DeepMIMO Dataset/O28B_1x64x1_ULA/h_real.mat'
    fname_h_imag = 'D://Github Repositories/DeepMIMO-codes/DeepMIMO_Dataset_Generation_v1.1/DeepMIMO_Dataset_Generation_v1.1/DeepMIMO Dataset/O28B_1x64x1_ULA/h_imag.mat'
    fname_loc = 'D://Github Repositories/DeepMIMO-codes/DeepMIMO_Dataset_Generation_v1.1/DeepMIMO_Dataset_Generation_v1.1/DeepMIMO Dataset/O28B_1x64x1_ULA/loc.mat'
    h_real = sio.loadmat(fname_h_real)['h_real']
    h_imag = sio.loadmat(fname_h_imag)['h_imag']
    loc = sio.loadmat(fname_loc)['loc']
else:
    raise NameError('Dataset Not Supported')

h = h_real + 1j*h_imag
#norm_factor = np.max(np.power(abs(h),2))
norm_factor = np.max(abs(h))
# norm_factor = 1
h_scaled = h/norm_factor
train_idc, test_idc = train_test_split(np.arange(h.shape[0]),test_size=0.4)
val_idc, test_idc = train_test_split(test_idc,test_size=0.5)
dft_nb_codebook = DFT_codebook(nseg=n_narrow_beams,n_antenna=n_antenna)
label = np.argmax(np.power(np.absolute(np.matmul(h_scaled, dft_nb_codebook.conj().T)),2),axis=1)

# dataset_per_class = {}
# for i in range(n_narrow_beams):
#     dataset_per_class[i] = h_scaled[label==i]
# nsample_per_class = np.array([dataset_per_class[i].shape[0] for i in range(n_narrow_beams)])
# nsample_minority = nsample_per_class.min()
# dataset_per_class_balanced = []
# label_balanced = []
# for i in range(n_narrow_beams):
#     balanced = resample(dataset_per_class[i],replace=False,n_samples=nsample_minority,random_state=7)
#     dataset_per_class_balanced.append(balanced)
#     label_balanced.extend([i for j in range(nsample_minority)])
# dataset_balanced = np.concatenate(tuple(dataset_per_class_balanced),axis=0)
# x_test,y_test = dataset_balanced,np.array(label_balanced)


x_train,y_train = h_scaled[train_idc,:],label[train_idc]
x_val,y_val = h_scaled[val_idc,:],label[val_idc]
x_test,y_test = h_scaled[test_idc,:],label[test_idc]

# for i,N in enumerate(n_wide_beams):  
#     trainable_codebook = np.load('probe_trainable_codebook_{}_beam.npy'.format(N))
#     dft_codebook = np.load('probe_DFT_codebook_{}_beam.npy'.format(N))
#     h_project_trainable = np.power(np.absolute(np.matmul(h_scaled, trainable_codebook.conj().T)),2)
#     trainable_silhouette_score = silhouette_score(h_project_trainable, label)    

#     h_project_dft = np.power(np.absolute(np.matmul(h_scaled, dft_codebook.conj().T)),2)
#     dft_silhouette_score = silhouette_score(h_project_dft, label) 
#     print('{} beam probing codebook, trainable silhouette score = {}, DFT silhouette score = {}'.format(N,trainable_silhouette_score,dft_silhouette_score))

trainable_codebook = np.load('./Saved Codebooks/{}_probe_trainable_codebook_{}_beam.npy'.format(dataset_name,2))
dft_codebook = np.load('./Saved Codebooks/{}_probe_DFT_codebook_{}_beam.npy'.format(dataset_name,2))
AMCF_codebook = np.load('./Saved Codebooks/{}_probe_AMCF_codebook_{}_beam.npy'.format(dataset_name,2))
# feat_cols = ['Beam {} SNR (dB)'.format(bi) for bi in range(1,3)]
feat_cols = ['Beam {}'.format(bi) for bi in range(1,3)]

h_project_trainable = np.power(np.absolute(np.matmul(x_test, trainable_codebook.conj().T)),2)
h_project_dft = np.power(np.absolute(np.matmul(x_test, dft_codebook.conj().T)),2)
h_project_AMCF = np.power(np.absolute(np.matmul(x_test, AMCF_codebook.conj().T)),2)
# h_project_trainable = 30 + 10*np.log10(h_project_trainable) + 94 - 13
# h_project_dft = 30 + 10*np.log10(h_project_dft) + 94 - 13
# h_project_AMCF = 30 + 10*np.log10(h_project_AMCF) + 94 - 13

df_trainable = pd.DataFrame(h_project_trainable,columns=feat_cols)
df_trainable['y'] = y_test
df_trainable['label'] = df_trainable['y'].apply(lambda i: str(i))
    

df_dft = pd.DataFrame(h_project_dft,columns=feat_cols)
df_dft['y'] = y_test
df_dft['label'] = df_dft['y'].apply(lambda i: str(i))
    
df_AMCF = pd.DataFrame(h_project_AMCF,columns=feat_cols)
df_AMCF['y'] = y_test
df_AMCF['label'] = df_AMCF['y'].apply(lambda i: str(i))
    
plt.figure(figsize=(18,5))

ax1 = plt.subplot(1, 3, 1)
sns.scatterplot(
    x=feat_cols[0], y=feat_cols[1],
    hue="y",
    palette=sns.color_palette("hls", len(df_trainable['y'].unique())),
    data=df_trainable,
    legend=False,
    alpha=0.3,
    ax=ax1
)    
ax1.set_title('Trainable Codebook')
ax2 = plt.subplot(1, 3, 2)
sns.scatterplot(
    x=feat_cols[0], y=feat_cols[1],
    hue="y",
    palette=sns.color_palette("hls", len(df_dft['y'].unique())),
    data=df_dft,
    legend=False,
    alpha=0.3,
    ax=ax2
)   
ax2.set_title('DFT Codebook')
ax3 = plt.subplot(1, 3, 3)
sns.scatterplot(
    x=feat_cols[0], y=feat_cols[1],
    hue="y",
    palette=sns.color_palette("hls", len(df_AMCF['y'].unique())),
    data=df_AMCF,
    legend=False,
    alpha=0.3,
    ax=ax3
)   
ax3.set_title('AMCF Codebook')
plt.suptitle('Visualization for {}-beam probing codebook'.format(2))
plt.show()
    
    
for i,N in enumerate(n_wide_beams):  
    trainable_codebook = np.load('./Saved Codebooks/{}_probe_trainable_codebook_{}_beam.npy'.format(dataset_name,N))
    dft_codebook = np.load('./Saved Codebooks/{}_probe_DFT_codebook_{}_beam.npy'.format(dataset_name,N))
    AMCF_codebook = np.load('./Saved Codebooks/{}_probe_AMCF_codebook_{}_beam.npy'.format(dataset_name,N))
    
    h_project_trainable = np.power(np.absolute(np.matmul(x_test, trainable_codebook.conj().T)),2)
    trainable_silhouette_score = silhouette_score(h_project_trainable, y_test)    

    h_project_dft = np.power(np.absolute(np.matmul(x_test, dft_codebook.conj().T)),2)
    dft_silhouette_score = silhouette_score(h_project_dft, y_test) 

    h_project_AMCF = np.power(np.absolute(np.matmul(x_test, AMCF_codebook.conj().T)),2)
    AMCF_silhouette_score = silhouette_score(h_project_AMCF, y_test) 
    
    print('{} beam probing codebook, trainable silhouette score = {:.4f}, DFT silhouette score = {:.4f}, AMCF silhouette score = {:.4f}'.format(N,trainable_silhouette_score,
                                                                                                                                                dft_silhouette_score,
                                                                                                                                                AMCF_silhouette_score))
    
for i,N in enumerate(n_wide_beams):  
    trainable_codebook = np.load('./Saved Codebooks/{}_probe_trainable_codebook_{}_beam.npy'.format(dataset_name,N))
    dft_codebook = np.load('./Saved Codebooks/{}_probe_DFT_codebook_{}_beam.npy'.format(dataset_name,N))
    AMCF_codebook = np.load('./Saved Codebooks/{}_probe_AMCF_codebook_{}_beam.npy'.format(dataset_name,N))
    
    feat_cols = ['beam_{}'.format(bi) for bi in range(N)]
    h_project_trainable = np.power(np.absolute(np.matmul(x_test, trainable_codebook.conj().T)),2)
    df_trainable = pd.DataFrame(h_project_trainable,columns=feat_cols)
    df_trainable['y'] = y_test
    df_trainable['label'] = df_trainable['y'].apply(lambda i: str(i))

    time_start = time.time()
    tsne_trainable = TSNE(n_components=2, verbose=0, perplexity=tsne_perplexity, n_iter=5000)
    trainable_tsne_results = tsne_trainable.fit_transform(df_trainable[feat_cols].values)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    df_trainable['tsne_2d_1'] = trainable_tsne_results[:,0]
    df_trainable['tsne_2d_2'] = trainable_tsne_results[:,1]
    

    h_project_dft = np.power(np.absolute(np.matmul(x_test, dft_codebook.conj().T)),2)
    df_dft = pd.DataFrame(h_project_dft,columns=feat_cols)
    df_dft['y'] = y_test
    df_dft['label'] = df_dft['y'].apply(lambda i: str(i))

    time_start = time.time()
    tsne_dft = TSNE(n_components=2, verbose=0, perplexity=tsne_perplexity, n_iter=5000)
    dft_tsne_results = tsne_dft.fit_transform(df_dft[feat_cols].values)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    df_dft['tsne_2d_1'] = dft_tsne_results[:,0]
    df_dft['tsne_2d_2'] = dft_tsne_results[:,1]    
    
    h_project_AMCF = np.power(np.absolute(np.matmul(x_test, AMCF_codebook.conj().T)),2)
    df_AMCF = pd.DataFrame(h_project_AMCF,columns=feat_cols)
    df_AMCF['y'] = y_test
    df_AMCF['label'] = df_AMCF['y'].apply(lambda i: str(i))

    time_start = time.time()
    tsne_AMCF = TSNE(n_components=2, verbose=0, perplexity=tsne_perplexity, n_iter=5000)
    AMCF_tsne_results = tsne_AMCF.fit_transform(df_AMCF[feat_cols].values)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    df_AMCF['tsne_2d_1'] = AMCF_tsne_results[:,0]
    df_AMCF['tsne_2d_2'] = AMCF_tsne_results[:,1] 

    plt.figure(figsize=(18,5))
    
    ax1 = plt.subplot(1, 3, 1)
    sns.scatterplot(
        x="tsne_2d_1", y="tsne_2d_2",
        hue="y",
        palette=sns.color_palette("hls", len(df_trainable['y'].unique())),
        data=df_trainable,
        legend=False,
        alpha=0.3,
        ax=ax1
    )    
    ax1.set_title('Trainable Codebook')
    ax2 = plt.subplot(1, 3, 2)
    sns.scatterplot(
        x="tsne_2d_1", y="tsne_2d_2",
        hue="y",
        palette=sns.color_palette("hls", len(df_dft['y'].unique())),
        data=df_dft,
        legend=False,
        alpha=0.3,
        ax=ax2
    )   
    ax2.set_title('DFT Codebook')
    ax3 = plt.subplot(1, 3, 3)
    sns.scatterplot(
        x="tsne_2d_1", y="tsne_2d_2",
        hue="y",
        palette=sns.color_palette("hls", len(df_AMCF['y'].unique())),
        data=df_AMCF,
        legend=False,
        alpha=0.3,
        ax=ax3
    )   
    ax3.set_title('AMCF Codebook')
    plt.suptitle('t-SNE visualization for {}-beam probing codebook'.format(N))
    plt.show()
    

# for n_dft_beam in [32,40,48,64]:    
#     dft_codebook = DFT_codebook(nseg=n_dft_beam,n_antenna=n_antenna)
#     feat_cols = ['beam_{}'.format(bi) for bi in range(n_dft_beam)]
#     h_project_dft = np.power(np.absolute(np.matmul(x_test, dft_codebook.conj().T)),2)
#     df_dft = pd.DataFrame(h_project_dft,columns=feat_cols)
#     df_dft['y'] = y_test
#     df_dft['label'] = df_dft['y'].apply(lambda i: str(i))
    
#     time_start = time.time()
#     tsne_dft = TSNE(n_components=2, verbose=1, perplexity=20, n_iter=5000)
#     dft_tsne_results = tsne_dft.fit_transform(df_dft[feat_cols].values)
#     print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
#     df_dft['tsne_2d_1'] = dft_tsne_results[:,0]
#     df_dft['tsne_2d_2'] = dft_tsne_results[:,1]    
    
#     plt.figure(figsize=(8,6))
    
#     ax1 = plt.subplot(1, 1, 1)
#     sns.scatterplot(
#         x="tsne_2d_1", y="tsne_2d_2",
#         hue="y",
#         palette=sns.color_palette("hls", len(df_dft['y'].unique())),
#         data=df_dft,
#         legend=False,
#         alpha=0.3,
#         ax=ax1
#     )    
#     ax1.set_title('t-SNE visualization for {}-beam DFT probing codebook'.format(n_dft_beam))
#     plt.show()