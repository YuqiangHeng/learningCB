# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 11:59:58 2021

@author: ethan
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# fname_h_real = 'D://Github Repositories/DeepMIMO-codes/DeepMIMO_Dataset_Generation_v1.1/DeepMIMO_Dataset_Generation_v1.1/DeepMIMO Dataset/O28B_1x8x8_UPA/h_real.mat'
# fname_h_imag = 'D://Github Repositories/DeepMIMO-codes/DeepMIMO_Dataset_Generation_v1.1/DeepMIMO_Dataset_Generation_v1.1/DeepMIMO Dataset/O28B_1x8x8_UPA/h_imag.mat'
# fname_loc = 'D://Github Repositories/DeepMIMO-codes/DeepMIMO_Dataset_Generation_v1.1/DeepMIMO_Dataset_Generation_v1.1/DeepMIMO Dataset/O28B_1x8x8_UPA/loc.mat'

# h_real = sio.loadmat(fname_h_real)['h_real']
# h_imag = sio.loadmat(fname_h_imag)['h_imag']
# loc = sio.loadmat(fname_loc)['loc']
# n_antenna = 64
# array_size = np.array([1,8,8])
# oversampling_factor = 1


# h = h_real + 1j*h_imag
# norm_factor = np.max(abs(h))
# h_scaled = h/norm_factor
        

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

def plot_codebook_pattern(codebook):
    fig = plt.figure()
    ax = fig.add_subplot(111,polar=True)
    for beam_i, beam in enumerate(codebook):
        phi, bf_gain = calc_beam_pattern(beam)
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
            
    
def DFT_codebook_UPA(n_azimuth,n_elevation,n_antenna_azimuth, n_antenna_elevation,spacing=0.5):
    codebook_all = np.zeros((n_azimuth,n_elevation,n_antenna_azimuth*n_antenna_elevation),dtype=np.complex_)
    thetas = np.linspace(-1/2,1/2,n_azimuth,endpoint=False)
    azimuths = np.arcsin(1/spacing*thetas)
    phis = np.linspace(-1/2,1/2,n_elevation,endpoint=False)
    elevations = np.arcsin(1/spacing*phis)
    for theta_i,theta in enumerate(azimuths):
        for phi_i,phi in enumerate(elevations):
            a_azimuth = [-1j*2*np.pi*k*spacing*np.sin(theta) for k in range(n_antenna_azimuth)]
            a_azimuth = np.exp(a_azimuth)/np.sqrt(n_antenna_azimuth)
            a_elevation = [-1j*2*np.pi*k*spacing*np.sin(phi) for k in range(n_antenna_elevation)] 
            a_elevation = np.exp(a_elevation)/np.sqrt(n_antenna_elevation)
            a = np.kron(a_azimuth,a_elevation)
            codebook_all[theta_i,phi_i,:] = a
    return codebook_all,(azimuths,elevations)

def DFT_codebook_ULA(nseg,n_antenna,spacing=0.5):
    codebook_all = np.zeros((nseg,n_antenna),dtype=np.complex_)
    thetas = DFT_angles(nseg)
    # thetas = np.linspace(-1/2,1/2,nseg,endpoint=False)
    azimuths = np.arcsin(1/spacing*thetas)
    for i,theta in enumerate(azimuths):
        #array response vector original
        arr_response_vec = [-1j*2*np.pi*k*spacing*np.sin(theta) for k in range(n_antenna)]
        codebook_all[i,:] = np.exp(arr_response_vec)/np.sqrt(n_antenna)
    return codebook_all,azimuths


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

cb1,bfd1 = DFT_codebook_ULA(19,64)
fig1,ax1 = plot_codebook_pattern(cb1)
ax1.set_title('Codebook1')

cb2,bfd2 = DFT_codebook_ULA(20,64)
fig2,ax2 = plot_codebook_pattern(cb2)
ax2.set_title('Codebook2')

# bfdirections = np.linspace(0,np.pi,128)
# cb1,bfd1 = DFT_codebook_1(128,n_antenna)
# cb2,bfd2 = DFT_codebook_2(128,n_antenna)

# bf_gain_1 = np.absolute(np.matmul(h_scaled, cb1.conj().T))**2 #shape n_ue x codebook_size
# bf_gain_2 = np.absolute(np.matmul(h_scaled, cb2.conj().T))**2 #shape n_ue x codebook_size




# # max_beams = np.argmax(all_snr, axis = 1)
# # max_beams_snr = np.max(all_snr, axis = 1)
# # max_beam_dir = bfdirections[max_beams]
# # plt.figure()
# # plt.scatter(all_sph[:,2],max_beam_dir)

# for plt_idx in np.random.choice(h.shape[0],5).astype(int):
#     plt.figure()
#     plt.plot(bfd1,bf_gain_1[plt_idx,:])
#     plt.axvline(x = bfd1[np.argmax(bf_gain_1[plt_idx,:])],c='r')
#     plt.title('Codebook 1, UE {}'.format(plt_idx))
#     plt.figure()
#     plt.plot(bfd2,bf_gain_2[plt_idx,:])
#     plt.axvline(x = bfd2[np.argmax(bf_gain_2[plt_idx,:])],c='r')
#     plt.title('Codebook 2, UE {}'.format(plt_idx))
        
# plt.figure()
# plt.hist(max_beams_snr)
