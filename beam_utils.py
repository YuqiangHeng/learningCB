# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 20:23:40 2021

@author: ethan
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
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
    
def DFT_codebook(nseg,n_antenna):
    bw = np.pi/nseg
    # bfdirections = np.arccos(np.linspace(np.cos(0+bw/2),np.cos(np.pi-bw/2-1e-6),nseg))
    bfdirections = np.linspace(0,np.pi,nseg)
    codebook_all = np.zeros((nseg,n_antenna),dtype=np.complex_)
    for i in range(nseg):
        phi = bfdirections[i]
        #array response vector original
        arr_response_vec = [-1j*np.pi*k*np.cos(phi) for k in range(n_antenna)]
        #array response vector for rotated ULA
        #arr_response_vec = [1j*np.pi*k*np.sin(phi+np.pi/2) for k in range(64)]
        codebook_all[i,:] = np.exp(arr_response_vec)/np.sqrt(n_antenna)
    return codebook_all

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

def DFT_codebook_blockmatrix(nseg,n_antenna):
    codebook = DFT_codebook(nseg,n_antenna).T
    w_r = np.real(codebook)
    w_i = np.imag(codebook)
    w = np.concatenate((np.concatenate((w_r,-w_i),axis=1),np.concatenate((w_i,w_r),axis=1)),axis=0)
    return w

def bf_gain_loss(y_pred, y_true):
    return -torch.mean(y_pred,dim=0)

def calc_beam_pattern(beam, resolution = int(1e3), n_antenna = 64, array_type='ULA', k=0.5):
    phi_all = np.linspace(0,np.pi,resolution)
    array_response_vectors = np.tile(phi_all,(n_antenna,1)).T
    # array_response_vectors = 1j*2*np.pi*k*np.sin(array_response_vectors)
    array_response_vectors = -1j*2*np.pi*k*np.cos(array_response_vectors)
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