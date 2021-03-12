# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 23:56:12 2021

@author: ethan
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.io as sio


h_real = np.load('D://Github Repositories/mmWave Beam Management/H_Matrices FineGrid/MISO_Static_FineGrid_Hmatrices_real.npy')
h_imag = np.load('D://Github Repositories/mmWave Beam Management/H_Matrices FineGrid/MISO_Static_FineGrid_Hmatrices_imag.npy')
loc = np.load('D://Github Repositories/mmWave Beam Management/H_Matrices FineGrid/MISO_Static_FineGrid_UE_location.npy')

bs_loc = [641,435,10]
n_antenna = 64
oversampling_factor = 1
n_beam = 4

h = h_real + 1j*h_imag
norm_factor = np.max(abs(h))
h_scaled = h/norm_factor
        
def DFT_beam(n_antenna,azimuths):
    codebook_all = np.zeros((len(azimuths),n_antenna),dtype=np.complex_)
    for i,phi in enumerate(azimuths):
        # arr_response_vec = [-1j*np.pi*k*np.cos(phi) for k in range(n_antenna)]
        arr_response_vec = [-1j*np.pi*k*np.sin(phi) for k in range(n_antenna)]
        codebook_all[i,:] = np.exp(arr_response_vec)/np.sqrt(n_antenna)
    return codebook_all

def calc_beam_pattern(beam, resolution = int(1e3), n_antenna = 64, array_type='ULA', k=0.5):
    # phi_all = np.linspace(0,np.pi,resolution)
    phi_all = np.linspace(-np.pi/2,np.pi/2,resolution)
    array_response_vectors = np.tile(phi_all,(n_antenna,1)).T
    array_response_vectors = -1j*2*np.pi*k*np.sin(array_response_vectors)
    # array_response_vectors = -1j*2*np.pi*k*np.cos(array_response_vectors)
    array_response_vectors = array_response_vectors * np.arange(n_antenna)
    array_response_vectors = np.exp(array_response_vectors)/np.sqrt(n_antenna)
    gains = abs(array_response_vectors.conj() @ beam)**2
    return phi_all, gains

def plot_codebook_pattern(codebook, n_antenna = 64):
    fig = plt.figure()
    ax = fig.add_subplot(111,polar=True)
    for beam_i, beam in enumerate(codebook):
        phi, bf_gain = calc_beam_pattern(beam, n_antenna = n_antenna)
        # bf_gain = 10*np.log10(bf_gain)
        ax.plot(phi,bf_gain)
    ax.grid(True)
    ax.set_rlabel_position(-90)  # Move radial labels away from plotted line
    return fig, ax
    # fig.show()
    
def calc_superimposed_beam_pattern(beams, resolution = int(1e3), n_antenna = 64, array_type='ULA', k=0.5):
    # phi_all = np.linspace(0,np.pi,resolution)
    phi_all = np.linspace(-np.pi/2,np.pi/2,resolution)
    array_response_vectors = np.tile(phi_all,(n_antenna,1)).T
    array_response_vectors = -1j*2*np.pi*k*np.sin(array_response_vectors)
    # array_response_vectors = -1j*2*np.pi*k*np.cos(array_response_vectors)
    array_response_vectors = array_response_vectors * np.arange(n_antenna)
    array_response_vectors = np.exp(array_response_vectors)/np.sqrt(n_antenna)
    gains = abs((array_response_vectors.conj() @ beams.T).sum(axis=1))**2
    return phi_all, gains

def plot_superimposed_codebook_pattern(codebook, n_antenna = 64):
    fig = plt.figure()
    ax = fig.add_subplot(111,polar=True)
    for beam_i, beam in enumerate(codebook):
        phi, bf_gain = calc_superimposed_beam_pattern(beam, n_antenna = n_antenna)
        ax.plot(phi,bf_gain)
    ax.grid(True)
    ax.set_rlabel_position(-90)  # Move radial labels away from plotted line
    return fig, ax
    # fig.show()

def DFT_angles(n_beam):
    delta_theta = 1/n_beam
    if n_beam % 2 == 1:
        thetas = np.arange(0,1/2,delta_theta)
        # thetas = np.linspace(0,1/2,n_beam//2+1,endpoint=False)
        thetas = np.concatenate((-np.flip(thetas[1:]),thetas))
    else:
        thetas = np.arange(delta_theta/2,1/2,delta_theta) 
        thetas = np.concatenate((-np.flip(thetas),thetas))
    return thetas

def DFT_codebook_1(nseg,n_antenna,spacing=0.5):
    codebook_all = np.zeros((nseg,n_antenna),dtype=np.complex_)
    thetas = DFT_angles(nseg)
    azimuths = np.arcsin(1/spacing*thetas)
    for i,theta in enumerate(azimuths):
        #array response vector original
        arr_response_vec = [-1j*2*np.pi*k*spacing*np.sin(theta) for k in range(n_antenna)]
        codebook_all[i,:] = np.exp(arr_response_vec)/np.sqrt(n_antenna)
    return codebook_all,azimuths

# def DFT_codebook_1(nseg,n_antenna):
#     # bw = np.pi/nseg
#     # bfdirections = np.arccos(np.linspace(np.cos(0+bw/2),np.cos(np.pi-bw/2-1e-6),nseg))
#     bfdirections = np.linspace(0,np.pi,nseg)
#     codebook_all = np.zeros((nseg,n_antenna),dtype=np.complex_)
#     for i in range(nseg):
#         phi = bfdirections[i]
#         arr_response_vec = [-1j*np.pi*k*np.cos(phi) for k in range(n_antenna)]
#         codebook_all[i,:] = np.exp(arr_response_vec)/np.sqrt(n_antenna)
#     return codebook_all, bfdirections

# def DFT_codebook_2(nseg,n_antenna):
#     # bfdirections = np.arccos(np.linspace(np.cos(0),np.cos(np.pi-1e-6),nseg))
#     bfdirections = np.arcsin(2*np.linspace(0,1,nseg,endpoint=False))
#     # bfdirections = np.linspace(-np.pi/2,np.pi/2,nseg)
#     codebook_all = np.zeros((nseg,n_antenna),dtype=np.complex_)
#     for i in range(nseg):
#         phi = bfdirections[i]
#         #array response vector original
#         arr_response_vec = [-1j*np.pi*k*np.sin(phi) for k in range(n_antenna)]
#         codebook_all[i,:] = np.exp(arr_response_vec)/np.sqrt(n_antenna)
#     return codebook_all,bfdirections

def DFT_codebook_2(nseg,n_antenna,spacing=0.5):
    codebook_all = np.zeros((nseg,n_antenna),dtype=np.complex_)
    thetas = np.linspace(-1/2,1/2,nseg,endpoint=False)
    azimuths = np.arcsin(1/spacing*thetas)
    for i,theta in enumerate(azimuths):
        #array response vector original
        arr_response_vec = [-1j*2*np.pi*k*spacing*np.sin(theta) for k in range(n_antenna)]
        codebook_all[i,:] = np.exp(arr_response_vec)/np.sqrt(n_antenna)
    return codebook_all,azimuths

def DFT_codebook_3(nseg,n_antenna,spacing=0.5):
    codebook_all = np.zeros((nseg,n_antenna),dtype=np.complex_)
    thetas = np.array([-np.pi+(2*i-1)*np.pi/n_antenna for i in range(1,nseg+1)])
    # thetas = np.linspace(-1/2,1/2,nseg,endpoint=False)
    # azimuths = np.arcsin(1/spacing*thetas)
    for i,theta in enumerate(thetas):
        #array response vector original
        arr_response_vec = [(k-(n_antenna-1)/2)*1j*theta for k in range(n_antenna)]
        codebook_all[i,:] = np.exp(arr_response_vec)/np.sqrt(n_antenna)
    return codebook_all

def superimposed_DFT_codebook(nseg,n_antenna,spacing=0.5):
    nb_codebook,nb_azimuths = DFT_codebook_1(n_antenna,n_antenna,spacing=spacing)
    wb_azimuths = DFT_angles(nseg)
    nnb_per_wb = math.ceil(n_antenna/nseg)
    wb_codebook = []
    for i,az in enumerate(wb_azimuths):
        nb_idc = abs(nb_azimuths-az).argsort()[:nnb_per_wb]
        nb_idc.sort()
        print(i,nb_idc)
        wb_codebook.append(nb_codebook[nb_idc]*np.exp(-1j*(n_antenna-1)/2))
    return np.array(wb_codebook)

def ULA_DFT_codebook(nseg,n_antenna,spacing=0.5):
    codebook_all = np.zeros((nseg,n_antenna),dtype=np.complex_)
    thetas = DFT_angles(nseg)
    azimuths = np.arcsin(1/spacing*thetas)
    for i,theta in enumerate(azimuths):
        #array response vector original
        arr_response_vec = [-1j*2*np.pi*k*spacing*np.sin(theta) for k in range(n_antenna)]
        codebook_all[i,:] = np.exp(arr_response_vec)/np.sqrt(n_antenna)
    return codebook_all

def ULA_DFT_codebook_blockmatrix(nseg,n_antenna):
    codebook = ULA_DFT_codebook(nseg,n_antenna).T
    w_r = np.real(codebook)
    w_i = np.imag(codebook)
    w = np.concatenate((np.concatenate((w_r,-w_i),axis=1),np.concatenate((w_i,w_r),axis=1)),axis=0)
    return w

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

dft_cb, dft_az = DFT_codebook_1(nseg=2,n_antenna=64)
plot_codebook_pattern(np.expand_dims(dft_cb[0],0))


# wb_cb = superimposed_DFT_codebook(nseg=4,n_antenna=64,spacing=0.5)
nb_cb = DFT_codebook_3(64,64,spacing=0.5)
# fig,ax = plot_codebook_pattern(nb_cb,n_antenna=32)
wb_cb = nb_cb/np.sqrt(16)
wb_cb = wb_cb[np.arange(16),:]
wb_cb = np.expand_dims(wb_cb,axis=0)
fig,ax = plot_superimposed_codebook_pattern(wb_cb,n_antenna=64)


cb_fname = '4_beam_AMCF.mat'
cb = sio.loadmat(cb_fname)['V']
fig,ax = plot_codebook_pattern(cb.T,n_antenna=64)
    

# all_sph = cart2sph(loc,bs_loc)

# bfdirections = np.linspace(0,np.pi,n_beam)
# cb1,bfd1 = DFT_codebook_1(n_beam,4)
# fig1,ax1 = plot_codebook_pattern(cb1,n_antenna=4)
# ax1.set_title('Codebook1')

# cb2,bfd2 = DFT_codebook_2(n_beam,4)
# fig2,ax2 = plot_codebook_pattern(cb2,n_antenna=4)
# ax2.set_title('Codebook2')

# # cb2,bfd2 = DFT_codebook_3(4,n_antenna)
# # fig2,ax2 = plot_codebook_pattern(cb2)
# # ax2.set_title('Codebook3')
# bf_gain_1 = np.absolute(np.matmul(h_scaled, cb1.conj().T))**2 #shape n_ue x codebook_size
# bf_gain_2 = np.absolute(np.matmul(h_scaled, cb2.conj().T))**2 #shape n_ue x codebook_size


# # cb3 = DFT_beam(n_antenna, [np.pi/2])
# # fig3,ax3 = plot_codebook_pattern(cb3)
# # ax3.set_title('Codebook2')

# max_beams = np.argmax(bf_gain_2, axis = 1)
# max_beams_snr = np.max(bf_gain_2, axis = 1)
# max_beam_dir = bfd2[max_beams]
# plt.figure()
# plt.scatter(all_sph[:,2],max_beam_dir)

# for plt_idx in np.random.choice(h.shape[0],5).astype(int):
#     plt.figure()
#     plt.plot(bfd1,bf_gain_1[plt_idx,:])
#     plt.axvline(x = bfd1[np.argmax(bf_gain_1[plt_idx,:])],c='r')
#     plt.title('Codebook 1, UE {}'.format(plt_idx))
#     plt.figure()
#     plt.plot(bfd2,bf_gain_2[plt_idx,:])
#     plt.axvline(x = bfd2[np.argmax(bf_gain_2[plt_idx,:])],c='r')
#     plt.title('Codebook 2, UE {}'.format(plt_idx))
        
# # plt.figure()
# # plt.hist(max_beams_snr)
