# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 20:23:40 2021

@author: ethan
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import math
# from typing import Tuple


ue_loc_fname = 'D://Github Repositories/mmWave Beam Management/H_Matrices FineGrid/MISO_Static_FineGrid_UE_location.npy'
bs_loc = [641,435,10]
n_antenna = 64

class GaussianCenters():
    def __init__(self, possible_loc = np.load(ue_loc_fname)[:,:2],
#               means:np.array=default_means, #2-d array w/. shape lx2, l centers
#               covs:np.array=default_covs, #3-d array w/. shape lx2x2, covariance of each center
#               arrival_rates:np.array=default_arr_rates # 1-d lx1 array, arrival rates of UEs at each center
               n_clusters = 4, arrival_rate = 5, cluster_variance = 5, random_clusters = True, cluster_exclusion = False, seed = 0
               ):
        self.arrival_rate = arrival_rate
        self.cluster_variance = cluster_variance
#        assert means.shape[1] == covs.shape[1] == covs.shape[2] == 2
#        assert means.shape[0] == covs.shape[0] == arrival_rates.shape[0]
        self.default_means = np.array([[640,470],[600,460],[680,460],[640,400]])
        self.bs_loc = [641,435]
        self.default_covs = np.array([[[self.cluster_variance,0],[0,self.cluster_variance]] for i in self.default_means])
        self.default_arr_rates = np.array([arrival_rate for i in self.default_means])
        self.n_clusters = n_clusters
        self.random_clusters = random_clusters
        self.all_loc = possible_loc
        self.tot_num_pts = self.all_loc.shape[0]
        self.seed = seed
        self.Random = np.random.RandomState(seed = self.seed)
        self.cluster_exclusion = cluster_exclusion
        if self.random_clusters:
            self.current_cluster_centers = self.gen_new_clusters()
            self.covs =  np.array([[[self.cluster_variance,0],[0,self.cluster_variance]] for i in range(self.n_clusters)])
            self.arrival_rates = np.array([arrival_rate for i in range(self.n_clusters)])
        else: 
            self.current_cluster_centers = self.default_means
            self.covs = self.default_covs
            self.arrival_rates = self.default_arr_rates


    def change_cluster(self):
        """
        change in clusters (according to a time-varying UE arrival process)
        the arrival rates are constant (same distributions)
        """
        self.current_cluster_centers = self.gen_new_clusters()


    def gen_new_clusters(self):
        """
        generate new cluster centers:
            number of clusters is the same
            randomly sample ray-traced UE points as cluster centers
            use a repulsion mechanism so that cluster centers are seperated by at least n*covariance of each cluster
        return: n_cluter x 2 array (loc of new cluster centers)
        """
        new_cluster_centers = np.zeros((self.n_clusters,2))
        for cluster_idx in range(self.n_clusters):       
            if cluster_idx == 0:
                sample_loc_idx = self.Random.choice(self.tot_num_pts)
                sample_loc = self.all_loc[sample_loc_idx]     
                new_cluster_centers[cluster_idx,:] = sample_loc
            else:
                if self.cluster_exclusion:
                    while True:
                        sample_loc_idx = self.Random.choice(self.tot_num_pts)
                        sample_loc = self.all_loc[sample_loc_idx]    
                        min_dist = min(np.linalg.norm(new_cluster_centers[0:cluster_idx,:] - sample_loc, axis=1))
                        if min_dist > 2*self.cluster_variance:
                            new_cluster_centers[cluster_idx,:] = sample_loc
                            break
                else:
                    sample_loc_idx = self.Random.choice(self.tot_num_pts)
                    sample_loc = self.all_loc[sample_loc_idx]     
                    new_cluster_centers[cluster_idx,:] = sample_loc                    
        return new_cluster_centers 
    
    def find_closest_ue(self, ue_pos:np.array):
        """
        input: 
            ue_loc: lx2 array of x,y coordinates of ues generated from gaussian center
        output:
            lx1 vector of index of ues with ray-traced channels that are closest to the target ues
        """
        #currently calc. l2 distance of all ue data points, can be more efficient
        closest_idx = [np.argmin((self.all_loc[:,0]-ue_pos[i,0])**2 + (self.all_loc[:,1]-ue_pos[i,1])**2) for i in range(ue_pos.shape[0])]
        return np.array(closest_idx)
    
    def plot_sample(self, sample):
        plt.figure()
        plt.scatter(self.all_loc[sample,0],self.all_loc[sample,1],s=1,label='sampled UE')
        plt.scatter(self.bs_loc[0],self.bs_loc[1],s=12,marker='s',label='BS')
        plt.xlabel('x (meter)')
        plt.ylabel('y (meter)')

        
    def sample(self):
        """
        output:
            n x 2 array, coordinates of n UEs generated according to arrival rates and centers
            assuming poisson arrival at each center
        """
#        num_UEs = np.random.poisson(lam = self.arrival_rates).astype(int)
        # num_UEs = self.Random.randint(0,self.arrival_rate*2,len(self.arrival_rates)) #uniform arrival rate so that its bounded
        num_UEs = self.arrival_rate*np.ones(len(self.arrival_rates)).astype(int)
        total_num_UEs = sum(num_UEs)
        sampled_loc = np.zeros((total_num_UEs,2))
        for i in range(self.n_clusters):
            samples = self.Random.multivariate_normal(self.current_cluster_centers[i,:], self.covs[i,:,:], num_UEs[i])
            sampled_loc[sum(num_UEs[0:i]):sum(num_UEs[0:i+1]),:] = samples
        sampled_idc = self.find_closest_ue(sampled_loc)
        return sampled_idc
    
# def DFT_codebook(nseg,n_antenna):
#     # bw = np.pi/nseg
#     # bfdirections = np.arccos(np.linspace(np.cos(0+bw/2),np.cos(np.pi-bw/2-1e-6),nseg))
#     bfdirections = np.linspace(0,np.pi,nseg)
#     codebook_all = np.zeros((nseg,n_antenna),dtype=np.complex_)
#     for i in range(nseg):
#         phi = bfdirections[i]
#         #array response vector original
#         arr_response_vec = [-1j*np.pi*k*np.cos(phi) for k in range(n_antenna)]
#         #array response vector for rotated ULA
#         #arr_response_vec = [1j*np.pi*k*np.sin(phi+np.pi/2) for k in range(64)]
#         codebook_all[i,:] = np.exp(arr_response_vec)/np.sqrt(n_antenna)
#     return codebook_all

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

def ULA_DFT_codebook(nseg,n_antenna,spacing=0.5):
    codebook_all = np.zeros((nseg,n_antenna),dtype=np.complex_)
    thetas = DFT_angles(nseg)
    azimuths = np.arcsin(1/spacing*thetas)
    for i,theta in enumerate(azimuths):
        #array response vector original
        arr_response_vec = [-1j*2*np.pi*k*spacing*np.sin(theta) for k in range(n_antenna)]
        codebook_all[i,:] = np.exp(arr_response_vec)/np.sqrt(n_antenna)
    return codebook_all

# def DFT_codebook(nseg,n_antenna,spacing=0.5):
#     codebook_all = np.zeros((nseg,n_antenna),dtype=np.complex_)
#     thetas = np.linspace(-1/2,1/2,nseg,endpoint=False)
#     azimuths = np.arcsin(1/spacing*thetas)
#     for i,theta in enumerate(azimuths):
#         #array response vector original
#         arr_response_vec = [-1j*2*np.pi*k*spacing*np.sin(theta) for k in range(n_antenna)]
#         codebook_all[i,:] = np.exp(arr_response_vec)/np.sqrt(n_antenna)
#     return codebook_all

def UPA_DFT_codebook(n_azimuth,n_elevation,n_antenna_azimuth,n_antenna_elevation,spacing=0.5):
    codebook_all = np.zeros((n_azimuth,n_elevation,n_antenna_azimuth*n_antenna_elevation),dtype=np.complex_)
    thetas = DFT_angles(n_azimuth)
    azimuths = np.arcsin(1/spacing*thetas)
    a_azimuth = np.tile(azimuths,(n_antenna_azimuth,1)).T
    a_azimuth = -1j*2*np.pi*spacing*np.sin(a_azimuth)
    a_azimuth = a_azimuth * np.tile(np.arange(n_antenna_azimuth),(n_azimuth,1))
    a_azimuth = np.exp(a_azimuth)/np.sqrt(n_antenna_azimuth)  

    phis = DFT_angles(n_elevation)
    elevations = np.arcsin(1/spacing*phis)
    a_elevation = np.tile(elevations,(n_antenna_elevation,1)).T
    a_elevation = -1j*2*np.pi*spacing*np.sin(a_elevation)
    a_elevation = a_elevation * np.tile(np.arange(n_antenna_elevation),(n_elevation,1))
    a_elevation = np.exp(a_elevation)/np.sqrt(n_antenna_elevation)  
    
    codebook_all = np.kron(a_elevation,a_azimuth)
    return codebook_all

def UPA_DFT_codebook_blockmatrix(n_azimuth,n_elevation,n_antenna_azimuth,n_antenna_elevation):
    codebook = UPA_DFT_codebook(n_azimuth=n_azimuth,n_elevation=n_elevation,n_antenna_azimuth=n_antenna_azimuth,n_antenna_elevation=n_antenna_elevation).T
    w_r = np.real(codebook)
    w_i = np.imag(codebook)
    w = np.concatenate((np.concatenate((w_r,-w_i),axis=1),np.concatenate((w_i,w_r),axis=1)),axis=0)
    return w


def DFT_codebook_alt(nseg,n_antenna):
    bfdirections = np.arccos(np.linspace(np.cos(0),np.cos(np.pi-1e-6),nseg))
    codebook_all = np.zeros((nseg,n_antenna),dtype=np.complex_)
    for i in range(nseg):
        phi = bfdirections[i]
        #array response vector original
        arr_response_vec = [-1j*np.pi*k*np.cos(phi) for k in range(n_antenna)]
        #array response vector for rotated ULA
        #arr_response_vec = [1j*np.pi*k*np.sin(phi+np.pi/2) for k in range(64)]
        codebook_all[i,:] = np.exp(arr_response_vec)/np.sqrt(n_antenna)
    return codebook_all


def DFT_beam(n_antenna,azimuths):
    codebook_all = np.zeros((len(azimuths),n_antenna),dtype=np.complex_)
    for i,phi in enumerate(azimuths):
        #array response vector original
        arr_response_vec = [-1j*np.pi*k*np.cos(phi) for k in range(n_antenna)]
        #array response vector for rotated ULA
        #arr_response_vec = [1j*np.pi*k*np.sin(phi+np.pi/2) for k in range(64)]
        codebook_all[i,:] = np.exp(arr_response_vec)/np.sqrt(n_antenna)
    return codebook_all

def DFT_beam_blockmatrix(n_antenna,azimuths):
    codebook = DFT_beam(n_antenna,azimuths).T
    w_r = np.real(codebook)
    w_i = np.imag(codebook)
    w = np.concatenate((np.concatenate((w_r,-w_i),axis=1),np.concatenate((w_i,w_r),axis=1)),axis=0)
    return w

def ULA_DFT_codebook_blockmatrix(nseg,n_antenna):
    codebook = ULA_DFT_codebook(nseg,n_antenna).T
    w_r = np.real(codebook)
    w_i = np.imag(codebook)
    w = np.concatenate((np.concatenate((w_r,-w_i),axis=1),np.concatenate((w_i,w_r),axis=1)),axis=0)
    return w

def codebook_blockmatrix(codebook):
    # codebook has dimension n_antenna x n_beams
    w_r = np.real(codebook)
    w_i = np.imag(codebook)
    w = np.concatenate((np.concatenate((w_r,-w_i),axis=1),np.concatenate((w_i,w_r),axis=1)),axis=0)
    return w

def bf_gain_loss(y_pred, y_true):
    return -torch.mean(y_pred,dim=0)

# def calc_beam_pattern(beam, resolution = int(1e3), n_antenna = 64, array_type='ULA', k=0.5):
#     phi_all = np.linspace(0,np.pi,resolution)
#     array_response_vectors = np.tile(phi_all,(n_antenna,1)).T
#     # array_response_vectors = 1j*2*np.pi*k*np.sin(array_response_vectors)
#     array_response_vectors = -1j*2*np.pi*k*np.cos(array_response_vectors)
#     array_response_vectors = array_response_vectors * np.arange(n_antenna)
#     array_response_vectors = np.exp(array_response_vectors)/np.sqrt(n_antenna)
#     gains = abs(array_response_vectors.conj() @ beam)**2
#     return phi_all, gains

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
    
def plot_codebook_pattern_on_axe(codebook,ax):
    for beam_i, beam in enumerate(codebook):
        phi, bf_gain = calc_beam_pattern(beam)
        ax.plot(phi,bf_gain)
    ax.grid(True)
    ax.set_rlabel_position(-90)  # Move radial labels away from plotted line
    # fig.show()
    
    
def get_AMCF_beam(omega_min, omega_max, n_antenna=64,spacing=0.5,Q=2**10): 
    #omega_min and omega_max are the min and max beam coverage, Q is the resolution
    q = np.arange(Q)+1
    omega_q = -1 + (2*q-1)/Q
    #array response matrix
    A_phase = np.outer(np.arange(n_antenna),omega_q)
    A = np.exp(1j*np.pi*A_phase)
    #physical angles between +- 90 degree
    theta_min, theta_max = np.arcsin(omega_min),np.arcsin(omega_max)
    #beamwidth in spatial angle
    B = omega_max - omega_min
    mainlobe_idc = ((omega_q >= omega_min) & (omega_q <= omega_max)).nonzero()[0]
    sidelobe_idc = ((omega_q < omega_min) | (omega_q > omega_max)).nonzero()[0]
    #ideal beam amplitude pattern
    g = np.zeros(Q)
    g[mainlobe_idc] = np.sqrt(2/B)
    #g_eps = g in mainlobe and = eps in sidelobe to avoid log0=Nan
    eps = 2**(-52)
    g_eps = g
    g_eps[sidelobe_idc] = eps
    
    v0_phase = B*np.arange(n_antenna)*np.arange(1,n_antenna+1)/2/n_antenna + np.arange(n_antenna)*omega_min
    v0 = 1/np.sqrt(n_antenna)*np.exp(1j*np.pi*v0_phase.conj().T)
    v = v0
    ite = 1
    mse_history = []
    while True:
        mse = np.power(abs(A.conj().T @ v) - g,2).mean()
        mse_history.append(mse)
        if ite >= 10 and abs(mse_history[-1] - mse_history[-2]) < 0.01*mse_history[-1]:
            break
        else:
            ite += 1
        Theta = np.angle(A.conj().T @ v)
        r = g * np.exp(1j*Theta)
        v = 1/np.sqrt(n_antenna)*np.exp(1j*np.angle(A @ r))
    return v
        
def AMCF_boundaries(n_beams):
    beam_boundaries = np.zeros((n_beams,2))
    for k in range(n_beams):
        beam_boundaries[k,0] = -1 + k*2/n_beams
        beam_boundaries[k,1] = beam_boundaries[k,0] + 2/n_beams
    return beam_boundaries

class Node():
    def __init__(self, n_antenna:int, n_beam:int, codebook:np.ndarray, beam_index:np.ndarray, noise_power=0):
        super(Node, self).__init__()
        self.codebook = codebook
        self.n_antenna = n_antenna
        self.n_beam = n_beam
        self.beam_index = beam_index # indices of the beams (in the same layer) in the codebook
        self.noise_power = noise_power
        self.parent = None
        self.child = None
        
    def forward(self, h):
        bf_signal = np.matmul(h, self.codebook.conj().T)
        noise_real = np.random.normal(loc=0,scale=1,size=bf_signal.shape)*np.sqrt(self.noise_power/2)
        noise_imag = np.random.normal(loc=0,scale=1,size=bf_signal.shape)*np.sqrt(self.noise_power/2)
        bf_signal = bf_signal + noise_real + 1j*noise_imag
        bf_gain = np.power(np.absolute(bf_signal),2)
        # bf_gain = np.power(np.absolute(np.matmul(h, self.codebook.conj().T)),2)
        return bf_gain
    
    def get_beam_index(self):
        return self.beam_index

    def set_child(self, child):
        self.child = child
        
    def set_parent(self, parent):
        self.parent = parent
        
    def get_child(self):
        return self.child
    
    def get_parent(self):
        return self.parent
    
    def is_leaf(self):
        return self.get_child() is None
    
    def is_root(self):
        return self.get_parent() is None
    
class Beam_Search_Tree():
    def __init__(self, n_antenna, n_narrow_beam, k, noise_power):
        super(Beam_Search_Tree, self).__init__()
        assert math.log(n_narrow_beam,k).is_integer()
        self.n_antenna = n_antenna
        self.k = k #number of beams per branch per layer
        self.n_layer = int(math.log(n_narrow_beam,k))
        self.n_narrow_beam = n_narrow_beam
        self.noise_power = noise_power
        self.beam_search_candidates = []
        for l in range(self.n_layer):
            self.beam_search_candidates.append([])
        self.nodes = []
        for l in range(self.n_layer):
            n_nodes = k**l
            n_beams = k**(l+1)
            if l == self.n_layer-1:
                beams = ULA_DFT_codebook(nseg=n_beams,n_antenna=n_antenna)
            else:                    
                beam_boundaries = AMCF_boundaries(n_beams)
                beams = np.array([get_AMCF_beam(omega_min=beam_boundaries[i,0], omega_max=beam_boundaries[i,1], n_antenna = n_antenna) for i in range(n_beams)])
                beams = np.flipud(beams)
            beam_idx_per_codebook = [np.arange(i,i+k) for i in np.arange(0,n_beams,k)]
            codebooks = [beams[beam_idx_per_codebook[i]] for i in range(n_nodes)]
            nodes_cur_layer = []
            nodes_cur_layer = [Node(n_antenna=n_antenna,n_beam = k, codebook=codebooks[i], beam_index=beam_idx_per_codebook[i], noise_power=self.noise_power) for i in range(n_nodes)]
            self.nodes.append(nodes_cur_layer)
            if l > 0:
                parent_nodes = self.nodes[l-1]
                for p_i, p_n in enumerate(parent_nodes):
                    child_nodes = nodes_cur_layer[p_i*k:(p_i+1)*k]
                    p_n.set_child(child_nodes)
                    for c_n in child_nodes:
                        c_n.set_parent(p_n)
        self.root = self.nodes[0][0]
        
    def forward(self, h):
        cur_node = self.root
        while not cur_node.is_leaf():
            bf_gain = cur_node.forward(h)
            next_node_idx = bf_gain.argmax()
            cur_node = cur_node.get_child()[next_node_idx]
        nb_bf_gain = cur_node.forward(h)
        max_nb_bf_gain = nb_bf_gain.max()
        max_nb_idx_local = nb_bf_gain.argmax()
        max_nb_idx_global = cur_node.get_beam_index()[max_nb_idx_local]
        return max_nb_bf_gain, max_nb_idx_global        
        
    def forward_batch(self, hbatch):
        bsize, in_dim = hbatch.shape
        max_nb_idx_batch = np.zeros(bsize,dtype=np.int32)
        max_nb_bf_gain_batch = np.zeros(bsize)
        for b_idx in range(bsize):
            h = hbatch[b_idx]
            nb_gain,nb_idx = self.forward(h)
            max_nb_idx_batch[b_idx] = nb_idx
            max_nb_bf_gain_batch[b_idx] = nb_gain
        return max_nb_bf_gain_batch, max_nb_idx_batch
    