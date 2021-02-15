# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 11:59:58 2021

@author: ethan
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

fname_h_real = 'D://Github Repositories/DeepMIMO-codes/DeepMIMO_Dataset_Generation_v1.1/DeepMIMO_Dataset_Generation_v1.1/DeepMIMO Dataset/O28B_1x8x8_ULA/h_real.mat'
fname_h_imag = 'D://Github Repositories/DeepMIMO-codes/DeepMIMO_Dataset_Generation_v1.1/DeepMIMO_Dataset_Generation_v1.1/DeepMIMO Dataset/O28B_1x8x8_ULA/h_imag.mat'
fname_loc = 'D://Github Repositories/DeepMIMO-codes/DeepMIMO_Dataset_Generation_v1.1/DeepMIMO_Dataset_Generation_v1.1/DeepMIMO Dataset/O28B_1x8x8_ULA/loc.mat'

h_real = sio.loadmat(fname_h_real)['h_real']
h_imag = sio.loadmat(fname_h_imag)['h_imag']
loc = sio.loadmat(fname_loc)['loc']
n_antenna = 64
array_size = np.array([1,8,8])
oversampling_factor = 1


h = h_real + 1j*h_imag
norm_factor = np.max(abs(h))
h_scaled = h/norm_factor
        

def DFT_codebook(n_azimuth,n_elevation,array_size):
    # bw = np.pi/nseg
    # bfdirections = np.arccos(np.linspace(np.cos(0+bw/2),np.cos(np.pi-bw/2-1e-6),nseg))
    azimuth = np.linspace(0,np.pi,nseg)
    codebook_all = np.zeros((nseg,n_antenna),dtype=np.complex_)
    for i in range(nseg):
        phi = bfdirections[i]
        arr_response_vec = [-1j*np.pi*k*np.cos(phi) for k in range(n_antenna)]
        codebook_all[i,:] = np.exp(arr_response_vec)/np.sqrt(n_antenna)
    return codebook_all, bfdirections




def cart2sph(xyz,center):
    x = np.subtract(xyz[:,0],center[0])
    y = np.subtract(xyz[:,1],center[1])
    z = np.subtract(xyz[:,2],center[2])
    rtp = np.zeros(xyz.shape)
    r = np.sqrt(np.power(x,2)+np.power(y,2)+np.power(z,2))
    theta = np.arccos(np.divide(z,r))
    #phi = np.arctan(np.divide(y,x))
    phi = np.arctan2(y,x)
    rtp[:,0] = r
    rtp[:,1] = theta
    rtp[:,2] = phi
    return rtp

bfdirections = np.linspace(0,np.pi,128)
cb1,bfd1 = DFT_codebook_1(128,n_antenna)
cb2,bfd2 = DFT_codebook_2(128,n_antenna)

bf_gain_1 = np.absolute(np.matmul(h_scaled, cb1.conj().T))**2 #shape n_ue x codebook_size
bf_gain_2 = np.absolute(np.matmul(h_scaled, cb2.conj().T))**2 #shape n_ue x codebook_size




# max_beams = np.argmax(all_snr, axis = 1)
# max_beams_snr = np.max(all_snr, axis = 1)
# max_beam_dir = bfdirections[max_beams]
# plt.figure()
# plt.scatter(all_sph[:,2],max_beam_dir)

for plt_idx in np.random.choice(h.shape[0],5).astype(int):
    plt.figure()
    plt.plot(bfd1,bf_gain_1[plt_idx,:])
    plt.axvline(x = bfd1[np.argmax(bf_gain_1[plt_idx,:])],c='r')
    plt.title('Codebook 1, UE {}'.format(plt_idx))
    plt.figure()
    plt.plot(bfd2,bf_gain_2[plt_idx,:])
    plt.axvline(x = bfd2[np.argmax(bf_gain_2[plt_idx,:])],c='r')
    plt.title('Codebook 2, UE {}'.format(plt_idx))
        
# plt.figure()
# plt.hist(max_beams_snr)
