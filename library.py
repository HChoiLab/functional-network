#%%
from unittest.mock import NonCallableMagicMock
import numpy as np
# import numpy.ma as ma
import pandas as pd
import os
import itertools
from itertools import cycle
# import scipy
from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import entropy
from scipy.spatial import distance
from scipy.special import softmax
from scipy.optimize import curve_fit
from tqdm import tqdm
import pickle
import time
import sys
import re
# import random
from mpl_toolkits.axes_grid1 import axes_grid
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib import colors
import statsmodels.stats.weightstats as ws
from statsmodels.stats.multitest import multipletests, fdrcorrection
import networkx as nx
import networkx.algorithms.community as nx_comm
from collections import defaultdict, deque
from networkx import NetworkXError
from networkx.algorithms.community.community_utils import is_partition
from networkx.utils import py_random_state
import community
import seaborn as sns
from numpy.lib.stride_tricks import as_strided
from scipy.ndimage.filters import uniform_filter1d
from scipy import signal
from scipy import stats
from scipy import sparse
# from plfit import plfit
import collections
from collections import defaultdict
# from sklearn.manifold import TSNE
from sklearn.metrics.cluster import adjusted_rand_score
import multiprocessing
import gurobipy as gp
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
np.seterr(divide='ignore', invalid='ignore')

customPalette = ['#630C3A', '#39C8C6', '#D3500C', '#FFB139', 'palegreen', 'darkblue', 'slategray', '#a6cee3', '#b2df8a', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#6a3d9a', '#b15928']
stimulus_colors = ['tab:blue', 'darkorange', 'bisque', 'limegreen', 'darkgreen', 'maroon', 'indianred', 'mistyrose']
region_colors = ['tab:green', 'lightcoral', 'steelblue', 'tab:orange', 'tab:purple', 'grey']
visual_regions = ['VISp', 'VISl', 'VISrl', 'VISal', 'VISpm', 'VISam'] #, 'LGd', 'LP'
# session_ids = [719161530, 750332458, 750749662, 754312389, 755434585, 756029989, 791319847, 797828357]
session_ids = ['719161530','750332458','750749662','754312389','755434585','756029989','791319847','797828357']
# stimulus_names = ['spontaneous', 'flashes', 'gabors',
#         'drifting_gratings', 'static_gratings',
#           'natural_scenes', 'natural_movie_one', 'natural_movie_three']
stimulus_names = ['spontaneous', 'flash_dark', 'flash_light',
        'drifting_gratings', 'static_gratings',
          'natural_scenes', 'natural_movie_one', 'natural_movie_three']

def safe_division(n, d):
    return n / d if d else 0

class pattern_jitter():
    def __init__(self, num_sample, sequences, L, R=None, memory=True):
        self.num_sample = num_sample
        self.sequences = np.array(sequences)
        if len(self.sequences.shape) > 1:
            self.N, self.T = self.sequences.shape
        else:
            self.T = len(self.sequences)
            self.N = None
        self.L = L
        self.memory = memory
        if self.memory:
            assert R is not None, 'R needs to be given if memory is True!'
            self.R = R
        else:
            self.R = None

    def spike_timing2train(self, spikeTrain):
        if len(spikeTrain.shape) == 1:
            spikeData = np.zeros(self.T)
            spikeData[spikeTrain.astype(int)] = 1
        else:
            spikeData = np.zeros((spikeTrain.shape[0], self.T))
            spikeData[np.repeat(np.arange(spikeTrain.shape[0]), spikeTrain.shape[1]), spikeTrain.ravel().astype(int)] = 1
        return spikeData

    def getSpikeTrain(self, spikeData):
        if len(spikeData.shape) == 1:
            spikeTrain = np.squeeze(np.where(spikeData>0)).ravel()
        else:
            spikeTrain = np.zeros((spikeData.shape[0], len(np.where(spikeData[0, :]>0)[0])))
            for i in range(spikeData.shape[0]):
                spikeTrain[i, :] = np.squeeze(np.where(spikeData[i, :]>0)).ravel()
        return spikeTrain

    def getInitDist(self):
        initDist = np.random.rand(self.L)
        return initDist/initDist.sum()

    def getTransitionMatrices(self, num_spike):
        tDistMatrices = np.zeros((num_spike - 1, self.L, self.L))
        for i in range(tDistMatrices.shape[0]):
            matrix = np.random.rand(self.L, self.L)
            stochMatrix = matrix/matrix.sum(axis=1)[:,None]
            tDistMatrices[i, :, :] = stochMatrix.astype('f')
        return tDistMatrices

    def getX1(self, jitter_window, initDist):
        
        randX = np.random.random()
        ind = np.where(randX <= np.cumsum(initDist))[0][0]
        return jitter_window[0][ind]

    def initializeX(self, initX, Prob):
        return initX + np.sum(Prob == 0)

    def getOmega(self, spikeTrain):
        Omega = []
        n = spikeTrain.size
        for i in range(n):
            temp = spikeTrain[i] - np.ceil(self.L/2) + 1
            temp = max(0, temp)
            temp = min(temp, self.T - self.L)
            Omega.append(np.arange(temp, temp + self.L, 1))
        return Omega

    # def getOmega(self, spikeTrain):
    #     Omega = []
    #     n = spikeTrain.size
    #     for i in range(n):
    #         temp = spikeTrain[i] - np.ceil(self.L/2) + 1
    #         lower_bound = max(0, temp)
    #         upper_bound = min(temp + self.L, self.T)
    #         Omega.append(np.arange(lower_bound, upper_bound, 1))
    #     return Omega

    def getGamma(self, spikeTrain):
        Gamma = []
        ks = [] # list of k_d
        ks.append(0)
        n = spikeTrain.size
        temp = int(spikeTrain[ks[-1]]/self.L)*self.L
        temp = max(0, temp)
        temp = min(temp, self.T - self.L)
        Gamma.append(np.arange(temp, temp + self.L, 1))
        for i in range(1, n):
            if spikeTrain[i] - spikeTrain[i-1] > self.R:
                ks.append(i)
            temp = int(spikeTrain[ks[-1]]/self.L)*self.L+spikeTrain[i]-spikeTrain[ks[-1]]
            temp = max(0, temp)
            temp = min(temp, self.T - self.L)
            Gamma.append(np.arange(temp, temp + self.L, 1))
        return Gamma

    def getSurrogate(self, spikeTrain, initDist, tDistMatrices):
        surrogate = []
        if self.memory:
            jitter_window = self.getGamma(spikeTrain)
        else:
            jitter_window = self.getOmega(spikeTrain)
        givenX = self.getX1(jitter_window, initDist)
        surrogate.append(givenX)
        for i, row in enumerate(tDistMatrices):
            if self.memory and spikeTrain[i+1] - spikeTrain[i] <= self.R:
                givenX = surrogate[-1] + spikeTrain[i+1] - spikeTrain[i]
            else:
                index = np.where(np.array(jitter_window[i]) == givenX)[0]
                p_i = np.squeeze(np.array(row[index]))
                initX = self.initializeX(jitter_window[i + 1][0], p_i)
                randX = np.random.random()
                # safe way to find the ind
                larger = np.where(randX <= np.cumsum(p_i))[0]
                if larger.shape[0]:
                    ind = larger[0]
                else:
                    ind = len(p_i) - 1
                givenX = initX + np.sum(p_i[:ind]!=0)
            givenX = min(self.T - 1, givenX) # possibly same location
            if givenX in surrogate:
                locs = jitter_window[i + 1]
                available_locs = [loc for loc in locs if loc not in surrogate]
                givenX = np.random.choice(available_locs)
            surrogate.append(givenX)
        return surrogate

    def sample_spiketrain(self, spikeTrain, initDist, tDistMatrices):
        spikeTrainMat = np.zeros((self.num_sample, spikeTrain.size))
        for i in tqdm(range(self.num_sample), disable=True):
            surrogate = self.getSurrogate(spikeTrain, initDist, tDistMatrices)
            spikeTrainMat[i, :] = surrogate
        return spikeTrainMat

    def jitter(self):
        # num_sample x N x T
        if self.N is not None:
            jittered_seq = np.zeros((self.num_sample, self.N, self.T))
            for n in range(self.N):
                spikeTrain = self.getSpikeTrain(self.sequences[n, :])
                num_spike = spikeTrain.size
                if num_spike:
                    initDist = self.getInitDist()
                    tDistMatrices = self.getTransitionMatrices(num_spike)
                    sampled_spiketrain = self.sample_spiketrain(spikeTrain, initDist, tDistMatrices)
                    jittered_seq[:, n, :] = self.spike_timing2train(sampled_spiketrain)
                else:
                    jittered_seq[:, n, :] = np.zeros((self.num_sample, self.T))
        else:
            spikeTrain = self.getSpikeTrain(self.sequences)
            num_spike = spikeTrain.size
            initDist = self.getInitDist()
            tDistMatrices = self.getTransitionMatrices(num_spike)
            sampled_spiketrain = self.sample_spiketrain(spikeTrain, initDist, tDistMatrices)
            jittered_seq = self.spike_timing2train(sampled_spiketrain).squeeze()
        return jittered_seq

def getSpikeTrain(spikeData):
    spikeTrain = np.squeeze(np.where(spikeData>0))
    return spikeTrain

def spike_timing2train(T, spikeTrain):
    if len(spikeTrain.shape) == 1:
        spikeData = np.zeros(T)
        spikeData[spikeTrain.astype(int)] = 1
    else:
        spikeData = np.zeros((spikeTrain.shape[0], T))
        spikeData[np.repeat(np.arange(spikeTrain.shape[0]), spikeTrain.shape[1]), spikeTrain.ravel().astype(int)] = 1
    return spikeData

class CommunityLayout():
  def __init__(self, comm_scale=3., node_scale=1.):
    self.comm_scale = comm_scale
    self.node_scale = node_scale
  def get_community_layout(self, g, partition):
    """
    Compute the layout for a modular graph.
    Arguments:
    ----------
    g -- networkx.Graph or networkx.DiGraph instance
        graph to plot

    partition -- dict mapping int node -> int community
        graph partitions
    Returns:
    --------
    pos -- dict mapping int node -> (float x, float y)
        node positions
    """
    pos_communities = self._position_communities(g, partition, scale=self.comm_scale)
    pos_nodes = self._position_nodes(g, partition, scale=self.node_scale)
    # combine positions
    pos = dict()
    for node in g.nodes():
        pos[node] = pos_communities[node] + pos_nodes[node]
    return pos

  def _position_communities(self, g, partition, **kwargs):
      # create a weighted graph, in which each node corresponds to a community,
      # and each edge weight to the number of edges between communities
      between_community_edges = self._find_between_community_edges(g, partition)
      communities = set(partition.values())
      hypergraph = nx.DiGraph()
      hypergraph.add_nodes_from(communities)
      for (ci, cj), edges in between_community_edges.items():
          hypergraph.add_edge(ci, cj, weight=len(edges))
      # find layout for communities
      pos_communities = nx.spring_layout(hypergraph, **kwargs)
      # set node positions to position of community
      pos = dict()
      for node, community in partition.items():
          pos[node] = pos_communities[community]
      return pos

  def _find_between_community_edges(self, g, partition):
      edges = dict()
      for (ni, nj) in g.edges():
          ci = partition[ni]
          cj = partition[nj]

          if ci != cj:
              try:
                  edges[(ci, cj)] += [(ni, nj)]
              except KeyError:
                  edges[(ci, cj)] = [(ni, nj)]
      return edges

  def _position_nodes(self, g, partition, **kwargs):
      """
      Positions nodes within communities.
      """
      communities = dict()
      for node, community in partition.items():
          try:
              communities[community] += [node]
          except KeyError:
              communities[community] = [node]
      pos = dict()
      for ci, nodes in communities.items():
          subgraph = g.subgraph(nodes)
          pos_subgraph = nx.spring_layout(subgraph, **kwargs)
          pos.update(pos_subgraph)
      return pos

def save_active_inds(min_FR, session_ids, stimulus_names):
  directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
  if not os.path.isdir(directory):
    os.mkdir(directory)
  inds_path = './data/ecephys_cache_dir/sessions/active_inds/'
  if not os.path.isdir(inds_path):
    os.mkdir(inds_path)
  for session_id in session_ids:
    print(session_id)
    active_inds = []
    for stimulus_name in stimulus_names:
      file = str(session_id) + '_' + stimulus_name + '.npz'
      print(file)
      sequences = load_npz_3d(os.path.join(directory, file))
      print('Spike train shape: {}'.format(sequences.shape))
      active_neuron_inds = sequences.mean(1).sum(1) > sequences.shape[2] * min_FR
      active_inds.append(active_neuron_inds)
      if stimulus_name == stimulus_names[-1]:
        active_inds = np.logical_and.reduce(active_inds)
        print('Number of active neurons {}'.format(active_inds.sum()))
        np.save(os.path.join(inds_path, str(session_id) + '.npy'), np.where(active_inds)[0])

def corr_mat(sequences, measure, maxlag=12, noprogressbar=True):
  if measure == 'pearson':
    adj_mat = np.corrcoef(sequences)
  elif measure == 'cosine':
    adj_mat = cosine_similarity(sequences)
  elif measure == 'correlation':
    adj_mat = squareform(pdist(sequences, 'correlation'))
  elif measure == 'MI':
    adj_mat = MI(sequences)
  elif measure == 'xcorr':
    adj_mat = n_cross_correlation6(sequences, maxlag=maxlag, disable=noprogressbar)
  elif measure == 'causality':
    adj_mat = granger_causality(sequences)
  else:
    sys.exit('Unrecognized measure value!!! Choices: pearson, cosin, correlation.')
  adj_mat = np.nan_to_num(adj_mat)
  np.fill_diagonal(adj_mat, 0)
  return adj_mat

def get_metric_names(G_dict):
  G = list(list(G_dict.items())[0][1].items())[0][1]
  if type(G) == list:
    G = G[0]
  if nx.is_directed(G):
    metric_names = ['assortativity', 'betweenness', 'closeness', 'clustering', 'modularity', 'transitivity']
  else:
    # metric_names = ['assortativity', 'betweenness', 'closeness', 'clustering', 'density', 'efficiency', 'modularity', 'small-worldness', 'transitivity']
    metric_names = ['assortativity', 'betweenness', 'closeness', 'clustering', 'efficiency', 'modularity', 'transitivity']
  return metric_names

def calculate_metric(G, metric_name, cc):
  if metric_name == 'density':
    metric = nx.density(G)
  else:
    if cc:
      if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = nx.subgraph(G, largest_cc)
    if metric_name == 'efficiency':
      metric = nx.global_efficiency(G)
    elif metric_name == 'clustering':
      metric = nx.average_clustering(G)
    elif metric_name == 'transitivity':
      metric = nx.transitivity(G)
    elif metric_name == 'betweenness':
      metric = np.mean(list(nx.betweenness_centrality(G).values()))
    elif metric_name == 'closeness':
      metric = np.mean(list(nx.closeness_centrality(G).values()))
    elif metric_name == 'modularity':
      try:
        if nx.is_directed(G):
          G = G.to_undirected()
        unweight = {(i, j):1 for i,j in G.edges()}
        nx.set_edge_attributes(G, unweight, 'weight')
        part = community.best_partition(G)
        metric = community.modularity(part,G)
      except:
        metric = 0
    elif metric_name == 'assortativity':
      metric = nx.degree_assortativity_coefficient(G)
    elif metric_name == 'small-worldness':
      if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = nx.subgraph(G, largest_cc)
      if nx.number_of_nodes(G) > 2 and nx.number_of_edges(G) > 2:
        metric = nx.sigma(G)
      else:
        metric = 0
  return metric

def calculate_weighted_metric(G, metric_name, cc):
  if metric_name == 'density':
    metric = nx.density(G)
  else:
    if cc:
      if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = nx.subgraph(G, largest_cc)
    if metric_name == 'efficiency':
      metric = nx.global_efficiency(G)
    elif metric_name == 'clustering':
      metric = nx.average_clustering(G, weight='weight')
    elif metric_name == 'transitivity':
      metric = nx.transitivity(G)
    elif metric_name == 'betweenness':
      metric = np.mean(list(nx.betweenness_centrality(G, weight='weight').values()))
    elif metric_name == 'closeness':
      metric = np.mean(list(nx.closeness_centrality(G).values()))
    elif metric_name == 'modularity':
      try:
        if nx.is_directed(G):
          G = G.to_undirected()
        # weight cannot be negative
        if sum([n<0 for n in nx.get_edge_attributes(G, "weight").values()]):
          print('Edge weight cannot be negative for weighted modularity, setting to unweighted...')
          unweight = {(i, j):1 for i,j in G.edges()}
          nx.set_edge_attributes(G, unweight, 'weight')
        part = community.best_partition(G, weight='weight')
        metric = community.modularity(part, G, weight='weight')
      except:
        metric = 0
    elif metric_name == 'assortativity':
      metric = nx.degree_assortativity_coefficient(G, weight='weight')
    elif metric_name == 'small-worldness':
      if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = nx.subgraph(G, largest_cc)
      if nx.number_of_nodes(G) > 2 and nx.number_of_edges(G) > 2:
        metric = nx.sigma(G)
      else:
        metric = 0
  return metric

def calculate_directed_metric(G, metric_name):
  if metric_name == 'in_degree':
    metric = G.in_degree()
  elif metric_name == 'out_degree':
    metric = G.out_degree()
    if metric_name == 'efficiency':
      metric = nx.global_efficiency(G)
    elif metric_name == 'clustering':
      metric = nx.average_clustering(G, weight='weight')
    elif metric_name == 'transitivity':
      metric = nx.transitivity(G)
    elif metric_name == 'betweenness':
      metric = np.mean(list(nx.betweenness_centrality(G, weight='weight').values()))
    elif metric_name == 'closeness':
      metric = np.mean(list(nx.closeness_centrality(G).values()))
    elif metric_name == 'modularity':
      try:
        part = community.best_partition(G, weight='weight')
        metric = community.modularity(part, G, weight='weight') 
      except:
        metric = 0
    elif metric_name == 'assortativity':
      metric = nx.degree_assortativity_coefficient(G, weight='weight')
    elif metric_name == 'small-worldness':
      if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = nx.subgraph(G, largest_cc)
      if nx.number_of_nodes(G) > 2 and nx.number_of_edges(G) > 2:
        metric = nx.sigma(G)
      else:
        metric = 0
  return metric

def metric_stimulus_individual(G_dict, sign, measure, n, weight, cc):
  rows, cols = get_rowcol(G_dict)
  metric_names = get_metric_names(G_dict)
  plots_shape = (3, 2) if len(metric_names) == 6 else (3, 3)
  metric = np.empty((len(rows), len(cols), len(metric_names)))
  metric[:] = np.nan
  fig = plt.figure(figsize=(5*plots_shape[1], 13))
  # fig = plt.figure(figsize=(20, 10))
  for metric_ind, metric_name in enumerate(metric_names):
    print(metric_name)
    for row_ind, row in enumerate(rows):
      print(row)
      for col_ind, col in enumerate(cols):
        G = G_dict[row][col] if col in G_dict[row] else nx.DiGraph()
        # print(nx.info(G))
        if G.number_of_nodes() > 2 and G.number_of_edges() > 0:
          if weight:
            m = calculate_weighted_metric(G, metric_name, cc)
          else:
            m = calculate_metric(G, metric_name, cc)
        metric[row_ind, col_ind, metric_ind] = m
    plt.subplot(*plots_shape, metric_ind + 1)
    for row_ind, row in enumerate(rows):
      plt.plot(cols, metric[row_ind, :, metric_ind], label=row, alpha=1)
    plt.gca().set_title(metric_name, fontsize=30, rotation=0)
    plt.xticks(rotation=90)
    if metric_ind // 2 < 2:
      plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
  plt.legend()
  plt.tight_layout()
  figname = './plots/metric_stimulus_individual_weighted_{}_{}_{}_fold.jpg'.format(sign, measure, n) if weight else './plots/metric_stimulus_individual_{}_{}_{}_fold.jpg'.format(sign, measure, n)
  plt.savefig(figname)
  return metric

def get_lcc(G):
  if nx.is_directed(G):
    Gcc = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
  else:
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
  return G.subgraph(Gcc[0])

def save_sparse_npz(matrix, filename):
  matrix_2d = matrix.reshape(matrix.shape[0], int(len(matrix.flatten())/matrix.shape[0]))
  sparse_matrix = sparse.csc_matrix(matrix_2d)
  with open(filename, 'wb') as outfile:
    pickle.dump([sparse_matrix, matrix.shape], outfile, pickle.HIGHEST_PROTOCOL)

def load_sparse_npz(filename):
  with open(filename, 'rb') as infile:
    [sparse_matrix, shape] = pickle.load(infile)
    matrix_2d = sparse_matrix.toarray()
  return matrix_2d.reshape(shape)

def save_npz(matrix, filename):
    matrix_2d = matrix.reshape(matrix.shape[0], int(len(matrix.flatten())/matrix.shape[0]))
    sparse_matrix = sparse.csc_matrix(matrix_2d)
    np.savez(filename, [sparse_matrix, matrix.shape])
    return 'npz file saved'

def load_npz(filename):
    """
    load npz files with sparse matrix and dimension
    output dense matrix with the correct dim
    """
    npzfile = np.load(filename, allow_pickle=True) 
    sparse_matrix = npzfile['arr_0'][0]
    ndim=npzfile['arr_0'][1]

    new_matrix_2d = np.array(sparse_matrix.todense())
    new_matrix = new_matrix_2d
    # new_matrix = new_matrix_2d.reshape(ndim)
    return new_matrix

def load_npz_3d(filename):
    """
    load npz files with sparse matrix and dimension
    output dense matrix with the correct dim
    """
    npzfile = np.load(filename, allow_pickle=True) 
    sparse_matrix = npzfile['arr_0'][0]
    ndim=npzfile['arr_0'][1]

    new_matrix_2d = np.array(sparse_matrix.todense())
    new_matrix = new_matrix_2d.reshape(ndim)
    return new_matrix

def get_rowcol(G_dict):
  rows = list(G_dict.keys())
  cols = []
  for row in rows:
    cols += list(G_dict[row].keys())
  cols = list(set(cols))
  if 'drifting_gratings_contrast' in cols:
    cols.remove('drifting_gratings_contrast')
  # sort stimulus
  # if measure == 'ccg_xiaoxuan':
  #   stimulus_rank = ['spon', 'spon_20', 'None', 'denoised', 'low', 'flash', 'flash_40', 'movie', 'movie_20']
  # else:
  # stimulus_rank = ['spontaneous', 'flashes', 'gabors',
  #     'drifting_gratings', 'static_gratings', 'drifting_gratings_contrast',
  #       'natural_scenes', 'natural_movie_one', 'natural_movie_three']
  stimulus_rank = ['spontaneous', 'flash_dark', 'flash_light', 'gabors',
        'drifting_gratings', 'static_gratings',
          'natural_scenes', 'natural_movie_one', 'natural_movie_three']
  stimulus_rank_dict = {i:stimulus_rank.index(i) for i in cols}
  stimulus_rank_dict = dict(sorted(stimulus_rank_dict.items(), key=lambda item: item[1]))
  cols = list(stimulus_rank_dict.keys())
  return rows, cols

def Z_score(r):
  return np.log((1+r)/(1-r)) / 2

############## correlation
def n_cross_correlation6(matrix, maxlag, disable): ### fastest, only causal correlation (A>B, only positive time lag on B), largest deviation from time window average
  N, M =matrix.shape
  xcorr=np.zeros((N,N))
  norm_mata = np.nan_to_num((matrix-np.mean(matrix, axis=1).reshape(-1, 1))/(np.std(matrix, axis=1).reshape(-1, 1)*np.sqrt(M)))
  norm_matb = np.nan_to_num((matrix-np.mean(matrix, axis=1).reshape(-1, 1))/(np.std(matrix, axis=1).reshape(-1, 1)*np.sqrt(M)))
  #### padding
  norm_mata = np.concatenate((norm_mata.conj(), np.zeros((N, maxlag))), axis=1)
  norm_matb = np.concatenate((np.zeros((N, maxlag)), norm_matb.conj(), np.zeros((N, maxlag))), axis=1) # must concat zeros to the left, why???????????
  total_len = len(list(itertools.permutations(range(N), 2)))
  for row_a, row_b in tqdm(itertools.permutations(range(N), 2), total=total_len , miniters=int(total_len/100), disable=disable): # , miniters=int(total_len/100)
  # for ind, (row_a, row_b) in enumerate(itertools.permutations(range(N), 2)): # , miniters=int(total_len/100)
    # faulthandler.enable()
    px, py = norm_mata[row_a, :], norm_matb[row_b, :]
    T = as_strided(py[maxlag:], shape=(maxlag+1, M + maxlag),
                    strides=(-py.strides[0], py.strides[0])) # must be py[maxlag:], why???????????
    # corr = np.dot(T, px)
    corr = T @ px
    corr = corr - uniform_filter1d(corr, maxlag, mode='mirror') # mirror for the boundary values
    xcorr[row_a, row_b] = corr[np.argmax(np.abs(corr))]
  return xcorr

def n_cross_correlation7(matrix, maxlag, disable): ### original correlation
  N, M =matrix.shape
  xcorr=np.zeros((N,N))
  norm_mata = np.nan_to_num((matrix-np.mean(matrix, axis=1).reshape(-1, 1))/(np.std(matrix, axis=1).reshape(-1, 1)*np.sqrt(M)))
  norm_matb = np.nan_to_num((matrix-np.mean(matrix, axis=1).reshape(-1, 1))/(np.std(matrix, axis=1).reshape(-1, 1)*np.sqrt(M)))
  #### padding
  norm_mata = np.concatenate((norm_mata.conj(), np.zeros((N, maxlag))), axis=1)
  norm_matb = np.concatenate((np.zeros((N, maxlag)), norm_matb.conj(), np.zeros((N, maxlag))), axis=1) # must concat zeros to the left, why???????????
  total_len = len(list(itertools.permutations(range(N), 2)))
  for row_a, row_b in tqdm(itertools.permutations(range(N), 2), total=total_len , miniters=int(total_len/100), disable=disable): # , miniters=int(total_len/100)
  # for ind, (row_a, row_b) in enumerate(itertools.permutations(range(N), 2)): # , miniters=int(total_len/100)
    # faulthandler.enable()
    px, py = norm_mata[row_a, :], norm_matb[row_b, :]
    T = as_strided(py[maxlag:], shape=(maxlag+1, M + maxlag),
                    strides=(-py.strides[0], py.strides[0])) # must be py[maxlag:], why???????????
    # corr = np.dot(T, px)
    corr = T @ px
    xcorr[row_a, row_b] = corr[np.argmax(np.abs(corr))]
  return xcorr

def n_cross_correlation8(matrix, maxlag=12, window=100, disable=True): ### fastest, only causal correlation (A>B, only positive time lag on B), largest deviation from flank
  N, M =matrix.shape
  # xcorr, peak_offset=np.zeros((N,N)), np.zeros((N,N))
  xcorr, peak_offset=np.empty((N,N)), np.empty((N,N))
  xcorr[:] = np.nan
  peak_offset[:] = np.nan
  norm_mata = np.nan_to_num((matrix-np.mean(matrix, axis=1).reshape(-1, 1))/(np.std(matrix, axis=1).reshape(-1, 1)*np.sqrt(M)))
  norm_matb = np.nan_to_num((matrix-np.mean(matrix, axis=1).reshape(-1, 1))/(np.std(matrix, axis=1).reshape(-1, 1)*np.sqrt(M)))
  #### padding
  norm_mata = np.concatenate((norm_mata.conj(), np.zeros((N, window))), axis=1)
  norm_matb = np.concatenate((np.zeros((N, window)), norm_matb.conj(), np.zeros((N, window))), axis=1) # must concat zeros to the left, why???????????
  total_len = len(list(itertools.permutations(range(N), 2)))
  for row_a, row_b in tqdm(itertools.permutations(range(N), 2), total=total_len , miniters=int(total_len/100), disable=disable): # , miniters=int(total_len/100)
  # for ind, (row_a, row_b) in enumerate(itertools.permutations(range(N), 2)): # , miniters=int(total_len/100)
    # faulthandler.enable()
    px, py = norm_mata[row_a, :], norm_matb[row_b, :]
    T = as_strided(py[window:], shape=(window+1, M + window),
                    strides=(-py.strides[0], py.strides[0])) # must be py[window:], why???????????
    # corr = np.dot(T, px)
    corr = T @ px
    corr = (corr - corr.mean())[:maxlag+1]
    max_offset = np.argmax(np.abs(corr))
    xcorr[row_a, row_b] = corr[max_offset]
    peak_offset[row_a, row_b] = max_offset
  return xcorr, peak_offset

def all_n_cross_correlation8(matrix, window=100, disable=True): ### fastest, only causal correlation (A>B, only positive time lag on B), largest deviation from flank
  N, M =matrix.shape
  # xcorr, peak_offset=np.zeros((N,N)), np.zeros((N,N))
  xcorr=np.empty((N,N,window+1))
  xcorr[:] = np.nan
  norm_mata = np.nan_to_num((matrix-np.mean(matrix, axis=1).reshape(-1, 1))/(np.std(matrix, axis=1).reshape(-1, 1)*np.sqrt(M)))
  norm_matb = np.nan_to_num((matrix-np.mean(matrix, axis=1).reshape(-1, 1))/(np.std(matrix, axis=1).reshape(-1, 1)*np.sqrt(M)))
  #### padding
  norm_mata = np.concatenate((norm_mata.conj(), np.zeros((N, window))), axis=1)
  norm_matb = np.concatenate((np.zeros((N, window)), norm_matb.conj(), np.zeros((N, window))), axis=1) # must concat zeros to the left, why???????????
  total_len = len(list(itertools.permutations(range(N), 2)))
  for row_a, row_b in tqdm(itertools.permutations(range(N), 2), total=total_len , miniters=int(total_len/100), disable=disable): # , miniters=int(total_len/100)
  # for ind, (row_a, row_b) in enumerate(itertools.permutations(range(N), 2)): # , miniters=int(total_len/100)
    # faulthandler.enable()
    px, py = norm_mata[row_a, :], norm_matb[row_b, :]
    T = as_strided(py[window:], shape=(window+1, M + window),
                    strides=(-py.strides[0], py.strides[0])) # must be py[window:], why???????????
    xcorr[row_a, row_b, :] = T @ px
  return xcorr

def n_cross_correlation8_2mat(matrix_a, matrix_b, maxlag=12, window=100, disable=True): ### CCG-mean of flank
  if len(matrix_a.shape) >= 2:
    N, M = matrix_a.shape
    if len(matrix_b.shape) < 2:
      matrix_b = np.tile(matrix_b, (N, 1))
  else:
    N, M = matrix_b.shape
    matrix_a = np.tile(matrix_a, (N, 1))
  xcorr=np.zeros((2,2,N))
  norm_mata = np.nan_to_num((matrix_a-np.mean(matrix_a, axis=1).reshape(-1, 1))/(np.std(matrix_a, axis=1).reshape(-1, 1)*np.sqrt(M)))
  norm_matb = np.nan_to_num((matrix_b-np.mean(matrix_b, axis=1).reshape(-1, 1))/(np.std(matrix_b, axis=1).reshape(-1, 1)*np.sqrt(M)))
  #### padding
  norm_mata_0 = np.concatenate((norm_mata.conj(), np.zeros((N, window))), axis=1)
  norm_mata_1 = np.concatenate((np.zeros((N, window)), norm_mata.conj(), np.zeros((N, window))), axis=1)
  norm_matb_0 = np.concatenate((norm_matb.conj(), np.zeros((N, window))), axis=1)
  norm_matb_1 = np.concatenate((np.zeros((N, window)), norm_matb.conj(), np.zeros((N, window))), axis=1) # must concat zeros to the left, why???????????
  for row in tqdm(range(N), total=N, miniters=int(N/100), disable=disable): # , miniters=int(total_len/100)
    # faulthandler.enable()
    px, py = norm_mata_0[row, :], norm_matb_1[row, :]
    T = as_strided(py[window:], shape=(window+1, M + window),
                    strides=(-py.strides[0], py.strides[0])) # must be py[window:], why???????????
    # corr = np.dot(T, px)
    corr = T @ px
    corr = (corr - corr.mean())[:maxlag+1]
    xcorr[0, 1, row] = corr[np.argmax(np.abs(corr))]
    px, py = norm_matb_0[row, :], norm_mata_1[row, :]
    T = as_strided(py[window:], shape=(window+1, M + window),
                    strides=(-py.strides[0], py.strides[0])) # must be py[window:], why???????????
    # corr = np.dot(T, px)
    corr = T @ px
    corr = (corr - corr.mean())[:maxlag+1]
    xcorr[1, 0, row] = corr[np.argmax(np.abs(corr))]
  return xcorr

def n_cross_correlation_2mat(matrix_a, matrix_b, maxlag, disable): ### fastest, only causal correlation (A>B, only positive time lag on B)
  if len(matrix_a.shape) >= 2:
    N, M = matrix_a.shape
    if len(matrix_b.shape) < 2:
      matrix_b = np.tile(matrix_b, (N, 1))
  else:
    N, M = matrix_b.shape
    matrix_a = np.tile(matrix_a, (N, 1))
  xcorr=np.zeros((2,2,N))
  norm_mata = np.nan_to_num((matrix_a-np.mean(matrix_a, axis=1).reshape(-1, 1))/(np.std(matrix_a, axis=1).reshape(-1, 1)*np.sqrt(M)))
  norm_matb = np.nan_to_num((matrix_b-np.mean(matrix_b, axis=1).reshape(-1, 1))/(np.std(matrix_b, axis=1).reshape(-1, 1)*np.sqrt(M)))
  #### padding
  norm_mata_0 = np.concatenate((norm_mata.conj(), np.zeros((N, maxlag))), axis=1)
  norm_mata_1 = np.concatenate((np.zeros((N, maxlag)), norm_mata.conj(), np.zeros((N, maxlag))), axis=1)
  norm_matb_0 = np.concatenate((norm_matb.conj(), np.zeros((N, maxlag))), axis=1)
  norm_matb_1 = np.concatenate((np.zeros((N, maxlag)), norm_matb.conj(), np.zeros((N, maxlag))), axis=1) # must concat zeros to the left, why???????????
  for row in tqdm(range(N), total=N, miniters=int(N/100), disable=disable): # , miniters=int(total_len/100)
    # faulthandler.enable()
    px, py = norm_mata_0[row, :], norm_matb_1[row, :]
    T = as_strided(py[maxlag:], shape=(maxlag+1, M + maxlag),
                    strides=(-py.strides[0], py.strides[0])) # must be py[maxlag:], why???????????
    # corr = np.dot(T, px)
    corr = T @ px
    corr = corr - uniform_filter1d(corr, maxlag, mode='mirror') # mirror for the boundary values
    xcorr[0, 1, row] = corr[np.argmax(np.abs(corr))]
    px, py = norm_matb_0[row, :], norm_mata_1[row, :]
    T = as_strided(py[maxlag:], shape=(maxlag+1, M + maxlag),
                    strides=(-py.strides[0], py.strides[0])) # must be py[maxlag:], why???????????
    # corr = np.dot(T, px)
    corr = T @ px
    corr = corr - uniform_filter1d(corr, maxlag, mode='mirror') # mirror for the boundary values
    xcorr[1, 0, row] = corr[np.argmax(np.abs(corr))]
  return xcorr

def cross_correlation_delta(matrix, maxlag, disable): ### fastest, only causal correlation (A>B, only positive time lag on B), largest deviation from time window average
  N, M =matrix.shape
  xcorr=np.zeros((N,N))
  #### padding
  norm_mata = np.concatenate((matrix.conj(), np.zeros((N, maxlag))), axis=1)
  norm_matb = np.concatenate((np.zeros((N, maxlag)), matrix.conj(), np.zeros((N, maxlag))), axis=1) # must concat zeros to the left, why???????????
  total_len = len(list(itertools.permutations(range(N), 2)))
  for row_a, row_b in tqdm(itertools.permutations(range(N), 2), total=total_len , miniters=int(total_len/100), disable=disable): # , miniters=int(total_len/100)
  # for ind, (row_a, row_b) in enumerate(itertools.permutations(range(N), 2)): # , miniters=int(total_len/100)
    # faulthandler.enable()
    px, py = norm_mata[row_a, :], norm_matb[row_b, :]
    T = as_strided(py[maxlag:], shape=(maxlag+1, M + maxlag),
                    strides=(-py.strides[0], py.strides[0])) # must be py[maxlag:], why???????????
    # corr = np.dot(T, px)
    corr = T @ px
    corr = corr - uniform_filter1d(corr, maxlag, mode='mirror') # mirror for the boundary values
    xcorr[row_a, row_b] = corr[np.argmax(np.abs(corr))]
  return xcorr

def cross_correlation(matrix, maxlag, disable): ### fastest, only causal correlation (A>B, only positive time lag on B), largest
  N, M =matrix.shape
  xcorr=np.zeros((N,N))
  #### padding
  norm_mata = np.concatenate((matrix.conj(), np.zeros((N, maxlag))), axis=1)
  norm_matb = np.concatenate((np.zeros((N, maxlag)), matrix.conj(), np.zeros((N, maxlag))), axis=1) # must concat zeros to the left, why???????????
  total_len = len(list(itertools.permutations(range(N), 2)))
  for row_a, row_b in tqdm(itertools.permutations(range(N), 2), total=total_len , miniters=int(total_len/100), disable=disable): # , miniters=int(total_len/100)
  # for ind, (row_a, row_b) in enumerate(itertools.permutations(range(N), 2)): # , miniters=int(total_len/100)
    # faulthandler.enable()
    px, py = norm_mata[row_a, :], norm_matb[row_b, :]
    T = as_strided(py[maxlag:], shape=(maxlag+1, M + maxlag),
                    strides=(-py.strides[0], py.strides[0])) # must be py[maxlag:], why???????????
    # corr = np.dot(T, px)
    corr = T @ px
    xcorr[row_a, row_b] = corr[np.argmax(np.abs(corr))]
  return xcorr

#################### significant correlation
def significant_xcorr(sequences, num_baseline, alpha=0.05, sign='all'):
  N = sequences.shape[0]
  xcorr = np.zeros((N, N))
  xcorr_bl = np.zeros((N, N, num_baseline))
  adj_mat, peaks = n_cross_correlation8(sequences, disable=False)
  xcorr = adj_mat
  for b in range(num_baseline):
    print(b)
    sample_seq = sequences.copy()
    np.random.shuffle(sample_seq) # rowwise for 2d array
    adj_mat_bl, peaks_bl = n_cross_correlation8(sample_seq, disable=False)
    xcorr_bl[:, :, b] = adj_mat_bl
  k = int(num_baseline * alpha) + 1 # allow int(N * alpha) random correlations larger
  significant_adj_mat, significant_peaks=np.zeros_like(xcorr), np.zeros_like(peaks)
  significant_adj_mat[:] = np.nan
  significant_peaks[:] = np.nan
  if sign == 'pos':
    indx = xcorr > np.clip(np.partition(xcorr_bl, -k, axis=-1)[:, :, -k], a_min=0, a_max=None)
  elif sign == 'neg':
    indx = xcorr < np.clip(np.partition(xcorr_bl, k-1, axis=-1)[:, :, k-1], a_min=None, a_max=0)
  elif sign == 'all':
    pos = xcorr > np.clip(np.partition(xcorr_bl, -k, axis=-1)[:, :, -k], a_min=0, a_max=None)
    neg = xcorr < np.clip(np.partition(xcorr_bl, k-1, axis=-1)[:, :, k-1], a_min=None, a_max=0)
    indx = np.logical_or(pos, neg)
  if np.sum(indx):
    significant_adj_mat[indx] = xcorr[indx]
    significant_peaks[indx] = peaks[indx]
  return significant_adj_mat, significant_peaks

def get_all_ccg(matrix, window=100, disable=True): ### fastest, only causal correlation (A>B, only positive time lag on B), largest deviation from flank
  N, M =matrix.shape
  # xcorr, peak_offset=np.zeros((N,N)), np.zeros((N,N))
  ccg=np.empty((N,N,window+1))
  ccg[:] = np.nan
  firing_rates = np.count_nonzero(matrix, axis=1) / (matrix.shape[1]/1000) # Hz instead of kHz
  #### add time lag on neuron B, should be causal correlation B -> A
  norm_mata = np.concatenate((matrix.conj(), np.zeros((N, window))), axis=1)
  norm_matb = np.concatenate((np.zeros((N, window)), matrix.conj(), np.zeros((N, window))), axis=1) # must concat zeros to the left, why???????????
  total_len = len(list(itertools.permutations(range(N), 2)))
  for row_a, row_b in tqdm(itertools.permutations(range(N), 2), total=total_len, miniters=int(total_len/50), maxinterval=200, disable=disable): # , miniters=int(total_len/100)
    if firing_rates[row_a] * firing_rates[row_b] > 0: # there could be no spike in a certain trial
        px, py = norm_mata[row_a, :], norm_matb[row_b, :]
        T = as_strided(py[window:], shape=(window+1, M + window),
                        strides=(-py.strides[0], py.strides[0])) # must be py[window:], why???????????
        # positive time lag is on neuron b, a->b
        ccg[row_a, row_b, :] = (T @ px) / ((M-np.arange(window+1))/1000 * np.sqrt(firing_rates[row_a] * firing_rates[row_b]))
    else:
        ccg[row_a, row_b, :] = np.zeros(window+1)
  return ccg

def save_ccg_corrected(sequences, fname, num_jitter=10, L=25, window=100, disable=True): ### fastest, only causal correlation (A>B, only positive time lag on B), largest deviation from flank
  ccg = get_all_ccg(sequences, window, disable=disable) # N x N x window
  save_sparse_npz(ccg, fname)
  N = sequences.shape[0]
  # jitter
  ccg_jittered = np.zeros((N, N, window+1)) # , num_jitter, to save memory
  pj = pattern_jitter(num_sample=num_jitter, sequences=sequences, L=L, memory=False)
  sampled_matrix = pj.jitter() # num_sample x N x T
  for i in range(num_jitter):
    print(i)
    ccg_jittered += get_all_ccg(sampled_matrix[i, :, :], window, disable=disable)
  ccg_jittered = ccg_jittered / num_jitter
  save_sparse_npz(ccg_jittered, fname.replace('.npz', '_bl.npz'))

def save_mean_ccg_corrected(sequences, fname, num_jitter=10, L=25, window=100, disable=True): ### fastest, only causal correlation (A>B, only positive time lag on B), largest deviation from flank
  num_neuron, num_trial, T = sequences.shape
  # num_trial = min(num_trial, 1000) # at most 1000 trials
  ccg, ccg_jittered = np.zeros((num_neuron, num_neuron, window + 1)), np.zeros((num_neuron, num_neuron, window + 1))
  pj = pattern_jitter(num_sample=num_jitter, sequences=sequences[:,0,:], L=L, memory=False)
  for m in range(num_trial):
    print('Trial {} / {}'.format(m+1, num_trial))
    ccg += get_all_ccg(sequences[:,m,:], window, disable=disable) # N x N x window
    pj.sequences = sequences[:,m,:]
    sampled_matrix = pj.jitter() # num_sample x N x T
    for i in range(num_jitter):
      ccg_jittered += get_all_ccg(sampled_matrix[i, :, :], window, disable=disable)
  ccg = ccg / num_trial
  ccg_jittered = ccg_jittered / (num_jitter * num_trial)
  save_sparse_npz(ccg, fname)
  save_sparse_npz(ccg_jittered, fname.replace('.npz', '_bl.npz'))

# def save_xcorr_shuffled(sequences, fname, num_baseline=10, disable=True):
#   N = sequences.shape[0]
#   xcorr = np.zeros((N, N))
#   xcorr_bl = np.zeros((N, N, num_baseline))
#   adj_mat, peaks = n_cross_correlation8(sequences, disable=disable)
#   xcorr = adj_mat
#   save_npz(xcorr, fname)
#   save_npz(peaks, fname.replace('.npz', '_peak.npz'))
#   for b in range(num_baseline):
#     print(b)
#     sample_seq = sequences.copy()
#     np.random.shuffle(sample_seq) # rowwise for 2d array
#     adj_mat_bl, peaks_bl = n_cross_correlation8(sample_seq, disable=disable)
#     xcorr_bl[:, :, b] = adj_mat_bl
#   save_npz(xcorr_bl, fname.replace('.npz', '_bl.npz'))

def save_xcorr_shuffled(sequences, fname, window=100, num_baseline=10, disable=True):
  N = sequences.shape[0]
  xcorr_bl = np.zeros((N, N, window+1))
  xcorr = all_n_cross_correlation8(sequences, disable=disable)
  save_npz(xcorr, fname)
  for b in range(num_baseline):
    print(b)
    sample_seq = sequences.copy()
    np.random.shuffle(sample_seq) # rowwise for 2d array
    xcorr_bl += all_n_cross_correlation8(sample_seq, disable=disable)
  xcorr_bl = xcorr_bl / num_baseline
  save_npz(xcorr_bl, fname.replace('.npz', '_bl.npz'))

def xcorr_n_fold(matrix, n=7, num_jitter=10, L=25, R=1, maxlag=12, window=100, disable=True): ### fastest, only causal correlation (A>B, only positive time lag on B), largest deviation from flank
  xcorr = all_xcorr(matrix, window, disable=disable) # N x N x window
  N = matrix.shape[0]
  significant_ccg, peak_offset=np.empty((N,N)), np.empty((N,N))
  significant_ccg[:] = np.nan
  peak_offset[:] = np.nan
  # jitter
  xcorr_jittered = np.zeros((N, N, window+1, num_jitter))
  pj = pattern_jitter(num_sample=num_jitter, sequences=matrix, L=L, memory=False)
  sampled_matrix = pj.jitter() # num_sample x N x T
  for i in range(num_jitter):
    print(i)
    xcorr_jittered[:, :, :, i] = all_xcorr(sampled_matrix[i, :, :], window, disable=disable)
  total_len = len(list(itertools.permutations(range(N), 2)))
  for row_a, row_b in tqdm(itertools.permutations(range(N), 2), total=total_len , miniters=int(total_len/100), disable=disable): # , miniters=int(total_len/100)
    ccg_corrected = (xcorr[row_a, row_b, :, None] - xcorr_jittered[row_a, row_b, :, :]).mean(-1)
    if ccg_corrected[:maxlag+1].max() > ccg_corrected.mean() + n * ccg_corrected.std():
    # if np.max(np.abs(corr))
      max_offset = np.argmax(ccg_corrected[:maxlag+1])
      significant_ccg[row_a, row_b] = ccg_corrected[:maxlag+1][max_offset]
      peak_offset[row_a, row_b] = max_offset
  return significant_ccg, peak_offset

########## save significant ccg for sharp peaks
def save_ccg_corrected_sharp_peak(directory, measure, maxlag=12, n=7):
  path = directory.replace(measure, measure+'_sharp_peak')
  if not os.path.exists(path):
    os.makedirs(path)
  files = os.listdir(directory)
  files.sort(key=lambda x:int(x[:9]))
  for file in files:
    if '_bl' not in file:
      print(file)
      try:
        ccg = load_npz_3d(os.path.join(directory, file))
      except:
        ccg = load_sparse_npz(os.path.join(directory, file))
      try:
        ccg_jittered = load_npz_3d(os.path.join(directory, file.replace('.npz', '_bl.npz')))
      except:
        ccg_jittered = load_sparse_npz(os.path.join(directory, file.replace('.npz', '_bl.npz')))
      num_nodes = ccg.shape[0]
      significant_ccg, significant_peaks=np.zeros((num_nodes,num_nodes)), np.zeros((num_nodes,num_nodes))
      significant_ccg[:] = np.nan
      significant_peaks[:] = np.nan
      ccg_corrected = ccg - ccg_jittered
      corr = (ccg_corrected - ccg_corrected.mean(-1)[:, :, None])[:, :, :maxlag+1]
      max_offset = np.argmax(np.abs(corr), -1)
      ccg_mat = np.choose(max_offset, np.moveaxis(corr, -1, 0))
      num_nodes = ccg.shape[0]
      significant_ccg, significant_peaks=np.zeros((num_nodes,num_nodes)), np.zeros((num_nodes,num_nodes))
      significant_ccg[:] = np.nan
      significant_peaks[:] = np.nan
      pos_fold = ccg_corrected[:, :, :maxlag+1].max(-1) > ccg_corrected.mean(-1) + n * ccg_corrected.std(-1)
      neg_fold = ccg_corrected[:, :, :maxlag+1].max(-1) < ccg_corrected.mean(-1) - n * ccg_corrected.std(-1)
      indx = np.logical_or(pos_fold, neg_fold)
      if np.sum(indx):
        significant_ccg[indx] = ccg_mat[indx]
        significant_peaks[indx] = max_offset[indx]

      # total_len = len(list(itertools.permutations(range(num_nodes), 2)))
      # for row_a, row_b in tqdm(itertools.permutations(range(num_nodes), 2), total=total_len , miniters=int(total_len/100), disable=disable): # , miniters=int(total_len/100)
      #   ccg_corrected = ccg[row_a, row_b, :] - ccg_jittered[row_a, row_b, :]
      #   if ccg_corrected[:maxlag+1].max() > ccg_corrected.mean() + n * ccg_corrected.std():
      #   # if np.max(np.abs(corr))
      #     max_offset = np.argmax(ccg_corrected[:maxlag+1])
      #     significant_ccg[row_a, row_b] = ccg_corrected[:maxlag+1][max_offset]
      #     significant_peaks[row_a, row_b] = max_offset
      print('{} significant edges'.format(np.sum(~np.isnan(significant_ccg))))
      # np.save(os.path.join(path, file), adj_mat)
      save_npz(significant_ccg, os.path.join(path, file))
      save_npz(significant_peaks, os.path.join(path, file.replace('.npz', '_peak.npz')))

def moving_average(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

########## save significant ccg for sharp integral
def save_ccg_corrected_sharp_integral(directory, measure, maxlag=12, n=3):
  path = directory.replace(measure, measure+'_sharp_integral')
  if not os.path.exists(path):
    os.makedirs(path)
  files = os.listdir(directory)
  files.sort(key=lambda x:int(x[:9]))
  for file in files:
    if '_bl' not in file:
      print(file)
      try: 
        ccg = load_npz_3d(os.path.join(directory, file))
      except:
        ccg = load_sparse_npz(os.path.join(directory, file))
      try:
        ccg_jittered = load_npz_3d(os.path.join(directory, file.replace('.npz', '_bl.npz')))
      except:
        ccg_jittered = load_sparse_npz(os.path.join(directory, file.replace('.npz', '_bl.npz')))
      num_nodes = ccg.shape[0]
      significant_ccg=np.zeros((num_nodes,num_nodes))
      significant_ccg[:] = np.nan
      ccg_corrected = ccg - ccg_jittered
      corr = (ccg_corrected - ccg_corrected.mean(-1)[:, :, None])
      filter = np.array([[[1/maxlag]]]).repeat(maxlag, axis=2)
      corr_integral = signal.convolve(corr, filter, mode='valid')
      ccg_mat = corr_integral[:, :, 0] # average of first maxlag window
      num_nodes = ccg.shape[0]
      pos_fold = ccg_mat > corr_integral.mean(-1) + n * corr_integral.std(-1)
      neg_fold = ccg_mat < corr_integral.mean(-1) - n * corr_integral.std(-1)
      indx = np.logical_or(pos_fold, neg_fold)
      if np.sum(indx):
        significant_ccg[indx] = ccg_mat[indx]
      print('{} significant edges'.format(np.sum(~np.isnan(significant_ccg))))
      save_npz(significant_ccg, os.path.join(path, file))

########## save significant ccg for highland
########## use mean of highland to compare with mean of rest
def find_highland_old(corr, min_spike=50,duration=6, maxlag=12, n=7):
  num_nodes = corr.shape[0]
  highland_ccg,offset=np.zeros((num_nodes,num_nodes)), np.zeros((num_nodes,num_nodes))
  highland_ccg[:] = np.nan
  offset[:] = np.nan
  filter = np.array([[[1/(duration+1)]]]).repeat(duration+1, axis=2)
  corr_integral = signal.convolve(corr, filter, mode='valid')
  # ccg_mat_max = corr_integral[:, :, :maxlag-duration+1].max(-1) # average of first maxlag window
  # ccg_mat_min = corr_integral[:, :, :maxlag-duration+1].min(-1) # average of first maxlag window
  max_offset = np.argmax(corr_integral[:, :, :maxlag-duration+1], -1)
  ccg_mat_max = np.choose(max_offset, np.moveaxis(corr_integral[:, :, :maxlag-duration+1], -1, 0))
  min_offset = np.argmin(corr_integral[:, :, :maxlag-duration+1], -1)
  ccg_mat_min = np.choose(min_offset, np.moveaxis(corr_integral[:, :, :maxlag-duration+1], -1, 0))
  pos_fold = ccg_mat_max > corr_integral.mean(-1) + n * corr_integral.std(-1)
  neg_fold = ccg_mat_min < corr_integral.mean(-1) - n * corr_integral.std(-1)
  fre_filter = np.count_nonzero(corr, axis=-1) > min_spike
  pos_fold = np.logical_and(pos_fold, fre_filter)
  neg_fold = np.logical_and(neg_fold, fre_filter)
  highland_ccg[pos_fold] = ccg_mat_max[pos_fold]
  highland_ccg[neg_fold] = ccg_mat_min[neg_fold]
  offset[pos_fold] = max_offset[pos_fold]
  offset[neg_fold] = min_offset[neg_fold]
  indx = np.logical_or(pos_fold, neg_fold)
  return highland_ccg, offset, indx 

########## use sum of highland to compare with the rest
def find_highland(corr, min_spike=50,duration=6, maxlag=12, n=7):
  num_nodes = corr.shape[0]
  highland_ccg,offset,confidence_level=np.zeros((num_nodes,num_nodes)),np.zeros((num_nodes,num_nodes)),np.zeros((num_nodes,num_nodes))
  highland_ccg[:] = np.nan
  offset[:] = np.nan
  filter = np.array([[[1]]]).repeat(duration+1, axis=2) # sum instead of mean
  corr_integral = signal.convolve(corr, filter, mode='valid', method='fft')
  mu, sigma = np.nanmean(corr_integral, -1), np.nanstd(corr_integral, -1)
  extreme_offset = np.argmax(np.abs(corr_integral[:, :, :maxlag-duration+1] - mu[:,:,None]), -1)
  ccg_mat_extreme = np.choose(extreme_offset, np.moveaxis(corr_integral[:, :, :maxlag-duration+1], -1, 0))
  pos_fold = ccg_mat_extreme > mu + n * sigma
  neg_fold = ccg_mat_extreme < mu - n * sigma
  c_level = (ccg_mat_extreme - mu) / sigma
  # pos_zero_fold = ccg_mat_extreme > n * sigma
  # neg_zero_fold = ccg_mat_extreme < n * sigma
  # pos_fold = np.logical_and(pos_fold, pos_zero_fold)
  # neg_fold = np.logical_and(neg_fold, neg_zero_fold)

  # corr_integral = signal.convolve(corr[:, :, :maxlag + 1], filter, mode='valid', method='fft')
  # extreme_offset = np.argmax(np.abs(corr_integral - mu[:,:,None]), -1)
  # ccg_mat_extreme = np.choose(extreme_offset, np.moveaxis(corr_integral, -1, 0))
  # pos_fold = ccg_mat_extreme > mu + n * sigma
  # neg_fold = ccg_mat_extreme < mu - n * sigma

  # fre_filter = np.count_nonzero(corr, axis=-1) > min_spike
  # pos_fold = np.logical_and(pos_fold, fre_filter)
  # neg_fold = np.logical_and(neg_fold, fre_filter)
  indx = np.logical_or(pos_fold, neg_fold)
  highland_ccg[indx] = ccg_mat_extreme[indx]
  confidence_level[indx] = c_level[indx]
  offset[indx] = extreme_offset[indx]
  return highland_ccg, confidence_level, offset, indx

########## remove double count of 0 time lag peak (compare 2nd peak)
def find_highland_new(corr, min_spike=50,duration=6, maxlag=12, n=7):
  num_nodes = corr.shape[0]
  highland_ccg,offset,confidence_level=np.zeros((num_nodes,num_nodes)),np.zeros((num_nodes,num_nodes)),np.zeros((num_nodes,num_nodes))
  highland_ccg[:] = np.nan
  offset[:] = np.nan
  filter = np.array([[[1]]]).repeat(duration+1, axis=2) # sum instead of mean
  corr_integral = signal.convolve(corr, filter, mode='valid', method='fft')
  mu, sigma = np.nanmean(corr_integral, -1), np.nanstd(corr_integral, -1)
  abs_deviation = np.abs(corr_integral[:, :, :maxlag-duration+1] - mu[:,:,None])
  extreme_offset = np.argmax(abs_deviation, -1)
  ccg_mat_extreme = np.choose(extreme_offset, np.moveaxis(corr_integral[:, :, :maxlag-duration+1], -1, 0))
  pos_fold = ccg_mat_extreme > mu + n * sigma
  neg_fold = ccg_mat_extreme < mu - n * sigma
  c_level = (ccg_mat_extreme - mu) / sigma
  indx = np.logical_or(pos_fold, neg_fold)
  highland_ccg[indx] = ccg_mat_extreme[indx]
  confidence_level[indx] = c_level[indx]
  offset[indx] = extreme_offset[indx]
  # find the double count 0 time lag in significant edges
  pos_double_0 = (extreme_offset==0) & (extreme_offset.T==0) & (pos_fold==pos_fold.T) & pos_fold
  neg_double_0 = (extreme_offset==0) & (extreme_offset.T==0) & (neg_fold==neg_fold.T) & neg_fold
  double_0 = np.logical_or(pos_double_0, neg_double_0)
  if np.sum(double_0):
    extreme_offset_2nd = np.argpartition(abs_deviation, -2, axis=-1)[:, :, -2]
    ccg_mat_extreme_2nd = np.choose(extreme_offset_2nd, np.moveaxis(corr_integral[:, :, :maxlag-duration+1], -1, 0))
    c_level_2nd = (ccg_mat_extreme_2nd - mu) / sigma
    
    # pos_keep_0 = (ccg_mat_extreme_2nd < ccg_mat_extreme_2nd.T) and pos_double_0
    # neg_keep_0 = (ccg_mat_extreme_2nd > ccg_mat_extreme_2nd.T) and neg_double_0
    # keep_0 = np.logical_or(pos_keep_0, neg_keep_0)
    
    pos_remove_0 = np.logical_and(ccg_mat_extreme_2nd >= ccg_mat_extreme_2nd.T, pos_double_0)
    neg_remove_0 = np.logical_and(ccg_mat_extreme_2nd <= ccg_mat_extreme_2nd.T, neg_double_0)
    remove_0 = np.logical_or(pos_remove_0, neg_remove_0)
    highland_ccg[remove_0], confidence_level[remove_0], offset[remove_0], indx[remove_0] = np.nan, 0, np.nan, False
    
    # for i, j in zip(*np.where(np.triu(double_0, 1))):
    #     if (ccg_mat_extreme_2nd[i, j] >= ccg_mat_extreme_2nd[j, i]) and  (ccg_mat_extreme_2nd[i, j] >= mu):
    pos_fold_2nd = np.logical_and(ccg_mat_extreme_2nd > mu + n * sigma, pos_remove_0)
    neg_fold_2nd = np.logical_and(ccg_mat_extreme_2nd < mu - n * sigma, neg_remove_0)
    indx_2nd = np.logical_or(pos_fold_2nd, neg_fold_2nd)
    indx_2nd = np.logical_and(indx_2nd, remove_0)
    highland_ccg[indx_2nd], confidence_level[indx_2nd], offset[indx_2nd] = ccg_mat_extreme_2nd[indx_2nd], c_level_2nd[indx_2nd], extreme_offset_2nd[indx_2nd]
    indx = np.logical_or(indx, indx_2nd)
  return highland_ccg, confidence_level, offset, indx

############### find 2nd largest highland for double-count edges to remove
def find_2nd_highland(corr, duration=6, maxlag=12, n=7):
  num_pairs = corr.shape[0]
  highland_ccg_2nd,offset_2nd,confidence_level_2nd=np.zeros(num_pairs),np.zeros(num_pairs),np.zeros(num_pairs)
  highland_ccg_2nd[:] = np.nan
  offset_2nd[:] = np.nan
  filter = np.array([[1]]).repeat(duration+1, axis=1) # sum instead of mean
  corr_integral = signal.convolve(corr, filter, mode='valid', method='fft')
  mu, sigma = np.nanmean(corr_integral, -1), np.nanstd(corr_integral, -1)
  abs_deviation = np.abs(corr_integral[:, :maxlag-duration+1] - mu[:,None])
  extreme_offset_2nd = np.argpartition(abs_deviation, -2, axis=-1)[:, -2]
  ccg_mat_extreme_2nd = np.choose(extreme_offset_2nd, np.moveaxis(corr_integral[:, :maxlag-duration+1], -1, 0))
  c_level_2nd = (ccg_mat_extreme_2nd - mu) / sigma
  pos_fold_2nd = ccg_mat_extreme_2nd > mu + n * sigma
  neg_fold_2nd = ccg_mat_extreme_2nd < mu - n * sigma
  c_level_2nd = (ccg_mat_extreme_2nd - mu) / sigma
  indx_2nd = np.logical_or(pos_fold_2nd, neg_fold_2nd)
  highland_ccg_2nd[indx_2nd], confidence_level_2nd[indx_2nd], offset_2nd[indx_2nd] = ccg_mat_extreme_2nd[indx_2nd], c_level_2nd[indx_2nd], extreme_offset_2nd[indx_2nd]

  return highland_ccg_2nd, confidence_level_2nd, offset_2nd, indx_2nd

def save_ccg_corrected_highland(directory, measure, min_spike=50, max_duration=6, maxlag=12, n=3):
  path = directory.replace(measure, measure+'_highland')
  if not os.path.exists(path):
    os.makedirs(path)
  files = os.listdir(directory)
  files.sort(key=lambda x:int(x[:9]))
  for file in files:
    if '_bl' not in file:
      print(file)
      try: 
        ccg = load_npz_3d(os.path.join(directory, file))
      except:
        ccg = load_sparse_npz(os.path.join(directory, file))
      try:
        ccg_jittered = load_npz_3d(os.path.join(directory, file.replace('.npz', '_bl.npz')))
      except:
        ccg_jittered = load_sparse_npz(os.path.join(directory, file.replace('.npz', '_bl.npz')))
      num_nodes = ccg.shape[0]
      significant_ccg,significant_confidence,significant_offset,significant_duration=np.zeros((num_nodes,num_nodes)),np.zeros((num_nodes,num_nodes)),np.zeros((num_nodes,num_nodes)),np.zeros((num_nodes,num_nodes))
      significant_ccg[:] = np.nan
      significant_confidence[:] = np.nan
      significant_offset[:] = np.nan
      significant_duration[:] = np.nan
      ccg_corrected = ccg - ccg_jittered
      # corr = (ccg_corrected - ccg_corrected.mean(-1)[:, :, None])
      for duration in np.arange(max_duration, -1, -1): # reverse order, so that sharp peaks can override highland
        print('duration {}'.format(duration))
        highland_ccg, confidence_level, offset, indx = find_highland(ccg_corrected, min_spike, duration, maxlag, n)
        # highland_ccg, confidence_level, offset, indx = find_highland(corr, min_spike, duration, maxlag, n)
        if np.sum(indx):
          significant_ccg[indx] = highland_ccg[indx]
          significant_confidence[indx] = confidence_level[indx]
          significant_offset[indx] = offset[indx]
          significant_duration[indx] = duration
      print('{} significant edges'.format(np.sum(~np.isnan(significant_ccg))))
      save_npz(significant_ccg, os.path.join(path, file))
      save_npz(significant_confidence, os.path.join(path, file.replace('.npz', '_confidence.npz')))
      save_npz(significant_offset, os.path.join(path, file.replace('.npz', '_offset.npz')))
      save_npz(significant_duration, os.path.join(path, file.replace('.npz', '_duration.npz')))

########## remove double count of 0 time lag peak (compare 2nd peak)
def save_ccg_corrected_highland_new(directory, measure, min_spike=50, max_duration=6, maxlag=12, n=3):
  path = directory.replace(measure, measure+'_highland')
  if not os.path.exists(path):
    os.makedirs(path)
  files = os.listdir(directory)
  files.sort(key=lambda x:int(x[:9]))
  for file in files:
    if '_bl' not in file:
      print(file)
      try: 
        ccg = load_npz_3d(os.path.join(directory, file))
      except:
        ccg = load_sparse_npz(os.path.join(directory, file))
      try:
        ccg_jittered = load_npz_3d(os.path.join(directory, file.replace('.npz', '_bl.npz')))
      except:
        ccg_jittered = load_sparse_npz(os.path.join(directory, file.replace('.npz', '_bl.npz')))
      num_nodes = ccg.shape[0]
      significant_ccg,significant_confidence,significant_offset,significant_duration=np.zeros((num_nodes,num_nodes)),np.zeros((num_nodes,num_nodes)),np.zeros((num_nodes,num_nodes)),np.zeros((num_nodes,num_nodes))
      significant_ccg[:] = np.nan
      significant_confidence[:] = np.nan
      significant_offset[:] = np.nan
      significant_duration[:] = np.nan
      ccg_corrected = ccg - ccg_jittered
      # corr = (ccg_corrected - ccg_corrected.mean(-1)[:, :, None])
      for duration in np.arange(max_duration, -1, -1): # reverse order, so that sharp peaks can override highland
        print('duration {}'.format(duration))
        highland_ccg, confidence_level, offset, indx = find_highland_new(ccg_corrected, min_spike, duration, maxlag, n)
        # highland_ccg, confidence_level, offset, indx = find_highland(corr, min_spike, duration, maxlag, n)
        if np.sum(indx):
          significant_ccg[indx] = highland_ccg[indx]
          significant_confidence[indx] = confidence_level[indx]
          significant_offset[indx] = offset[indx]
          significant_duration[indx] = duration
      double_0 = (significant_offset==0) & (significant_offset.T==0) & (~np.isnan(significant_ccg)) & (~np.isnan(significant_ccg.T))
      print('Number of cross duration double-count edges: {}'.format(np.sum(double_0)))
      if np.sum(double_0):
        remove_0 = (significant_duration >= significant_duration.T) & double_0
        significant_ccg[remove_0], significant_confidence[remove_0], significant_offset[remove_0], significant_duration[remove_0] = np.nan, np.nan, np.nan, np.nan
        for duration in np.arange(max_duration, -1, -1): # reverse order, so that sharp peaks can override highland
          highland_ccg_2nd, confidence_level_2nd, offset_2nd, indx_2nd =  find_2nd_highland(ccg_corrected[remove_0], duration, maxlag, n)
          if np.sum(indx_2nd):
            significant_ccg[remove_0][indx_2nd] = highland_ccg_2nd[indx_2nd]
            significant_confidence[remove_0][indx_2nd] = confidence_level_2nd[indx_2nd]
            significant_offset[remove_0][indx_2nd] = offset_2nd[indx_2nd]
            significant_duration[remove_0][indx_2nd] = duration

      print('{} significant edges'.format(np.sum(~np.isnan(significant_ccg))))
      save_npz(significant_ccg, os.path.join(path, file))
      save_npz(significant_confidence, os.path.join(path, file.replace('.npz', '_confidence.npz')))
      save_npz(significant_offset, os.path.join(path, file.replace('.npz', '_offset.npz')))
      save_npz(significant_duration, os.path.join(path, file.replace('.npz', '_duration.npz')))
      
############ save significant xcorr for sharp peaks
def save_xcorr_sharp_peak(directory, sign, measure, maxlag=12, alpha=0.01, n=3):
  path = directory.replace(measure, sign+'_'+measure+'_sharp_peak')
  if not os.path.exists(path):
    os.makedirs(path)
  files = os.listdir(directory)
  files.sort(key=lambda x:int(x[:9]))
  # adj_temp = load_npz_3d(os.path.join(directory, [f for f in files if not '_bl' in f][0]))
  # R = adj_temp.shape[2] # number of downsamples
  adj_bl_temp = load_npz_3d(os.path.join(directory, [f for f in files if '_bl' in f][0]))
  num_baseline = adj_bl_temp.shape[2] # number of shuffles
  k = int(num_baseline * alpha) + 1 # allow int(N * alpha) random correlations larger
  for file in files:
    if ('_bl' not in file) and ('_peak' not in file):
      print(file)
      # adj_mat_ds = np.load(os.path.join(directory, file))
      # adj_mat_bl = np.load(os.path.join(directory, file.replace('.npy', '_bl.npy')))
      all_xcorr = load_npz_3d(os.path.join(directory, file))
      # import pdb;pdb.set_trace()
      corr = (all_xcorr - all_xcorr.mean(-1)[:, :, None])[:, :, :maxlag+1]
      max_offset = np.argmax(np.abs(corr), -1)
      xcorr = np.choose(max_offset, np.moveaxis(corr, -1, 0))
      xcorr_bl = load_npz_3d(os.path.join(directory, file.replace('.npz', '_bl.npz')))
      all_xcorr = all_xcorr - xcorr_bl
      significant_adj_mat, significant_peaks=np.zeros_like(xcorr), np.zeros_like(xcorr)
      significant_adj_mat[:] = np.nan
      significant_peaks[:] = np.nan
      if sign == 'pos':
        # indx = xcorr > np.clip(np.partition(xcorr_bl, -k, axis=-1)[:, :, -k], a_min=0, a_max=None)
        fold = all_xcorr[:, :, :maxlag+1].max(-1) > all_xcorr.mean(-1) + n * all_xcorr.std(-1)
      elif sign == 'neg':
        # indx = xcorr < np.clip(np.partition(xcorr_bl, k-1, axis=-1)[:, :, k-1], a_min=None, a_max=0)
        fold = all_xcorr[:, :, :maxlag+1].max(-1) < all_xcorr.mean(-1) - n * all_xcorr.std(-1)
      elif sign == 'all':
        # pos = xcorr > np.clip(np.partition(xcorr_bl, -k, axis=-1)[:, :, -k], a_min=0, a_max=None)
        # neg = xcorr < np.clip(np.partition(xcorr_bl, k-1, axis=-1)[:, :, k-1], a_min=None, a_max=0)
        # indx = np.logical_or(pos, neg)
        pos_fold = all_xcorr[:, :, :maxlag+1].max(-1) > all_xcorr.mean(-1) + n * all_xcorr.std(-1)
        neg_fold = all_xcorr[:, :, :maxlag+1].max(-1) < all_xcorr.mean(-1) - n * all_xcorr.std(-1)
        fold = np.logical_or(pos_fold, neg_fold)
      # indx = np.logical_and(indx, fold)
      indx = fold
      if np.sum(indx):
        significant_adj_mat[indx] = xcorr[indx]
        significant_peaks[indx] = max_offset[indx]
      
      # np.save(os.path.join(path, file), adj_mat)
      save_npz(significant_adj_mat, os.path.join(path, file))
      save_npz(significant_peaks, os.path.join(path, file.replace('.npz', '_peak.npz')))

############ save significant xcorr for larger
def save_xcorr_larger_2d(directory, sign, measure, alpha):
  path = directory.replace(measure, sign+'_'+measure+'_larger')
  if not os.path.exists(path):
    os.makedirs(path)
  files = os.listdir(directory)
  files.sort(key=lambda x:int(x[:9]))
  # adj_temp = load_npz_3d(os.path.join(directory, [f for f in files if not '_bl' in f][0]))
  # R = adj_temp.shape[2] # number of downsamples
  adj_bl_temp = load_npz_3d(os.path.join(directory, [f for f in files if '_bl' in f][0]))
  N = adj_bl_temp.shape[2] # number of shuffles
  k = int(N * alpha) + 1 # allow int(N * alpha) random correlations larger
  for file in files:
    if ('_bl' not in file) and ('_peak' not in file) and ('719161530' in file):
      print(file)
      # adj_mat_ds = np.load(os.path.join(directory, file))
      # adj_mat_bl = np.load(os.path.join(directory, file.replace('.npy', '_bl.npy')))
      xcorr = load_npz_3d(os.path.join(directory, file))
      xcorr_bl = load_npz_3d(os.path.join(directory, file.replace('.npz', '_bl.npz')))
      peaks = load_npz_3d(os.path.join(directory, file.replace('.npz', '_peak.npz')))
      significant_adj_mat, significant_peaks=np.zeros_like(xcorr), np.zeros_like(peaks)
      significant_adj_mat[:] = np.nan
      significant_peaks[:] = np.nan
      if sign == 'pos':
        indx = xcorr > np.clip(np.partition(xcorr_bl, -k, axis=-1)[:, :, -k], a_min=0, a_max=None)
      elif sign == 'neg':
        indx = xcorr < np.clip(np.partition(xcorr_bl, k-1, axis=-1)[:, :, k-1], a_min=None, a_max=0)
      elif sign == 'all':
        pos = xcorr > np.clip(np.partition(xcorr_bl, -k, axis=-1)[:, :, -k], a_min=0, a_max=None)
        neg = xcorr < np.clip(np.partition(xcorr_bl, k-1, axis=-1)[:, :, k-1], a_min=None, a_max=0)
        indx = np.logical_or(pos, neg)
      if np.sum(indx):
        significant_adj_mat[indx] = xcorr[indx]
        significant_peaks[indx] = peaks[indx]
      
      # np.save(os.path.join(path, file), adj_mat)
      save_npz(significant_adj_mat, os.path.join(path, file))
      save_npz(significant_peaks, os.path.join(path, file.replace('.npz', '_peak.npz')))

############ save significant xcorr for larger with downsampling
def save_xcorr_larger_3d(directory, sign, measure, alpha):
  path = directory.replace(measure, sign+'_'+measure+'_larger')
  if not os.path.exists(path):
    os.makedirs(path)
  files = os.listdir(directory)
  files.sort(key=lambda x:int(x[:9]))
  # adj_temp = load_npz_3d(os.path.join(directory, [f for f in files if not '_bl' in f][0]))
  # R = adj_temp.shape[2] # number of downsamples
  adj_bl_temp = load_npz_3d(os.path.join(directory, [f for f in files if '_bl' in f][0]))
  N = adj_bl_temp.shape[2] # number of shuffles
  k = int(N * alpha) + 1 # allow int(N * alpha) random correlations larger
  for file in files:
    if '_bl' not in file:
      print(file)
      # adj_mat_ds = np.load(os.path.join(directory, file))
      # adj_mat_bl = np.load(os.path.join(directory, file.replace('.npy', '_bl.npy')))
      adj_mat_ds = load_npz_3d(os.path.join(directory, file))
      adj_mat_bl = load_npz_3d(os.path.join(directory, file.replace('.npz', '_bl.npz')))
      adj_mat = np.zeros_like(adj_mat_ds)
      if measure == 'xcorr':
        iterator = list(itertools.permutations(range(adj_mat_ds.shape[0]), 2))
      else:
        iterator = list(itertools.combinations(range(adj_mat_ds.shape[0]), 2))
      total_len = len(iterator)
      for row_a, row_b in tqdm(iterator, total=total_len):
        # if adj_mat_ds[row_a, row_b, r] > max(np.partition(adj_mat_bl[row_a, row_b, :], -k)[-k], 0): # only keep positive edges:
        # if adj_mat_ds[row_a, row_b, :].mean() > max(adj_mat_bl[row_a, row_b, :].max(), 0): # only keep positive edges:
          # adj_mat[row_a, row_b, r] = adj_mat_ds[row_a, row_b, r]
        if sign == 'pos':
          indx = adj_mat_ds[row_a, row_b] > max(np.partition(adj_mat_bl[row_a, row_b, :], -k)[-k], 0)
        elif sign == 'neg':
          indx = adj_mat_ds[row_a, row_b] < min(np.partition(adj_mat_bl[row_a, row_b, :], k-1)[k-1], 0)
        elif sign == 'all':
          pos = adj_mat_ds[row_a, row_b] > max(np.partition(adj_mat_bl[row_a, row_b, :], -k)[-k], 0)
          neg = adj_mat_ds[row_a, row_b] < min(np.partition(adj_mat_bl[row_a, row_b, :], k-1)[k-1], 0)
          indx = np.logical_or(pos, neg)
        if np.sum(indx):
          adj_mat[row_a, row_b, indx] = adj_mat_ds[row_a, row_b, indx]
      # np.save(os.path.join(path, file), adj_mat)
      save_npz(adj_mat, os.path.join(path, file))

def remove_outlier(array):
  mean = np.mean(array)
  standard_deviation = np.std(array)
  distance_from_mean = abs(array - mean)
  max_deviations = 2
  not_outlier = distance_from_mean < max_deviations * standard_deviation
  return array[not_outlier]

def get_regions_spiking_sequence(session_id, stimulus_name, regions, resolution):
  data_directory = './data/ecephys_cache_dir'
  manifest_path = os.path.join(data_directory, "manifest.json")
  cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
  session = cache.get_session_data(int(session_id),
                                  amplitude_cutoff_maximum=np.inf,
                                  presence_ratio_minimum=-np.inf,
                                  isi_violations_maximum=np.inf)
  df = session.units
  df = df.rename(columns={"channel_local_index": "channel_id", 
                          "ecephys_structure_acronym": "ccf", 
                          "probe_id":"probe_global_id", 
                          "probe_description":"probe_id",
                          'probe_vertical_position': "ypos"})
  df['unit_id']=df.index
  if stimulus_name!='invalid_presentation':
    if (stimulus_name=='flash_light') or (stimulus_name=='flash_dark'):
      stim_table = session.get_stimulus_table(['flashes'])
    else:
      stim_table = session.get_stimulus_table([stimulus_name])
    stim_table=stim_table.rename(columns={"start_time": "Start", "stop_time": "End"})
    if stimulus_name=='flash_light':
      stim_table = stim_table[stim_table['color']==1]
    elif stimulus_name=='flash_dark':
      stim_table = stim_table[stim_table['color']==-1]
    if 'natural_movie' in stimulus_name:
        frame_times = stim_table.End-stim_table.Start
        print('frame rate:', 1/np.mean(frame_times), 'Hz', np.mean(frame_times))
        # stim_table.to_csv(output_path+'stim_table_'+stimulus_name+'.csv')
        # chunch each movie clip
        stim_table = stim_table[stim_table.frame==0]
        stim_table = stim_table.drop(['End'], axis=1)
        duration = np.mean(remove_outlier(np.diff(stim_table.Start.values))[:10]) - 1e-4
        stimulus_presentation_ids = stim_table.index.values
    elif stimulus_name=='spontaneous':
        index = np.where(stim_table.duration>=20)[0]
        if len(index): # only keep the longest spontaneous; has to be longer than 20 sec
            duration=20
            stimulus_presentation_ids = stim_table.index[index]
    else:
        # ISI = np.mean(session.get_inter_presentation_intervals_for_stimulus([stimulus_name]).interval.values)
        duration = round(np.mean(stim_table.duration.values), 2)
    # each trial should have at least 250 ms
    try: stimulus_presentation_ids
    except NameError: stimulus_presentation_ids = stim_table.index[stim_table.duration >= 0.25].values
    time_bin_edges = np.linspace(0, duration, int(duration / resolution)+1)
    cortical_units_ids = np.array([idx for idx, ccf in enumerate(df.ccf.values) if ccf in regions])
    print('Number of units is {}, duration is {}'.format(len(cortical_units_ids), duration))
    # get binarized tensor
    df_cortex = df.iloc[cortical_units_ids]
    histograms = session.presentationwise_spike_counts(
        bin_edges=time_bin_edges,
        stimulus_presentation_ids=stimulus_presentation_ids,
        unit_ids=df_cortex.unit_id.values
    )
    return histograms

############# save area_dict and average speed data
def load_area_speed(session_ids, stimulus_names, regions):
  data_directory = './data/ecephys_cache_dir'
  manifest_path = os.path.join(data_directory, "manifest.json")
  cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
  area_dict = {}
  speed_dict = {}
  for mouseID in session_ids:
    print(mouseID)
    session = cache.get_session_data(int(mouseID),
                                amplitude_cutoff_maximum=np.inf,
                                presence_ratio_minimum=-np.inf,
                                isi_violations_maximum=np.inf)
    df = session.units
    df = df.rename(columns={"channel_local_index": "channel_id", 
                            "ecephys_structure_acronym": "ccf", 
                            "probe_id":"probe_global_id", 
                            "probe_description":"probe_id",
                            'probe_vertical_position': "ypos"})
    cortical_units_ids = np.array([idx for idx, ccf in enumerate(df.ccf.values) if ccf in regions])
    df_cortex = df.iloc[cortical_units_ids]
    instruction = df_cortex.ccf
    # if set(instruction.unique()) == set(regions): # if the mouse has all regions recorded
    #   speed_dict[mouseID] = {}
    instruction = instruction.reset_index()
    if not mouseID in area_dict:
      area_dict[mouseID] = {}
      speed_dict[mouseID] = {}
    for i in range(instruction.shape[0]):
      area_dict[mouseID][i] = instruction.ccf.iloc[i]
    for stimulus_name in stimulus_names:
      print(stimulus_name)
      stim_table = session.get_stimulus_table([stimulus_name])
      stim_table=stim_table.rename(columns={"start_time": "Start", "stop_time": "End"})
      if 'natural_movie' in stimulus_name:
        frame_times = stim_table.End-stim_table.Start
        print('frame rate:', 1/np.mean(frame_times), 'Hz', np.mean(frame_times))
        # stim_table.to_csv(output_path+'stim_table_'+stimulus_name+'.csv')
        # chunch each movie clip
        stim_table = stim_table[stim_table.frame==0]
      speed = session.running_speed[(session.running_speed['start_time']>=stim_table['Start'].min()) & (session.running_speed['end_time']<=stim_table['End'].max())]
      speed_dict[mouseID][stimulus_name] = speed['velocity'].mean()
  # switch speed_dict to dataframe
  mouseIDs = list(speed_dict.keys())
  stimuli = list(speed_dict[list(speed_dict.keys())[0]].keys())
  mean_speed_df = pd.DataFrame(columns=stimuli, index=mouseIDs)
  for k in speed_dict:
    for v in speed_dict[k]:
      mean_speed_df.loc[k][v] = speed_dict[k][v]
  return area_dict, mean_speed_df

def save_area_speed(session_ids, stimulus_names, visual_regions):
  area_dict, mean_speed_df = load_area_speed(session_ids, stimulus_names, visual_regions)
  a_file = open('./data/ecephys_cache_dir/sessions/area_dict.pkl', 'wb')
  pickle.dump(area_dict, a_file)
  a_file.close()
  mean_speed_df.to_pickle('./data/ecephys_cache_dir/sessions/mean_speed_df.pkl')

def save_active_area_dict(area_dict):
  active_area_dict = {}
  inds_path = './data/ecephys_cache_dir/sessions/active_inds/'
  files = os.listdir(inds_path)
  files.sort(key=lambda x:int(x[:9]))
  for file_order in range(len(files)):
    file = files[file_order]
    print(file)
    mouseID = file.split('.')[0]
    active_neuron_inds = np.load(os.path.join(inds_path, mouseID+'.npy'))
    active_area_dict[mouseID] = {key:area_dict[mouseID][key] for key in active_neuron_inds}
  a_file = open('./data/ecephys_cache_dir/sessions/active_area_dict.pkl', 'wb')
  pickle.dump(active_area_dict, a_file)
  a_file.close()

############# load area_dict
def load_area_dict(session_ids):
  a_file = open('./data/ecephys_cache_dir/sessions/area_dict.pkl', 'rb')
  area_dict = pickle.load(a_file)
  # change the keys of area_dict from int to string
  int_2_str = dict((session_id, str(session_id)) for session_id in session_ids)
  area_dict = dict((int_2_str[key], value) for (key, value) in area_dict.items())
  a_file.close()
  return area_dict

############# load area_dict and average speed dataframe #################
def load_other_data(session_ids):
  a_file = open('./data/ecephys_cache_dir/sessions/area_dict.pkl', 'rb')
  area_dict = pickle.load(a_file)
  # change the keys of area_dict from int to string
  int_2_str = dict((session_id, str(session_id)) for session_id in session_ids)
  area_dict = dict((int_2_str[key], value) for (key, value) in area_dict.items())
  a_file.close()
  a_file = open('./data/ecephys_cache_dir/sessions/active_area_dict.pkl', 'rb')
  active_area_dict = pickle.load(a_file)
  # change the keys of area_dict from int to string
  # int_2_str = dict((session_id, str(session_id)) for session_id in session_ids)
  # active_area_dict = dict((int_2_str[key], value) for (key, value) in active_area_dict.items())
  a_file.close()
  mean_speed_df = pd.read_pickle('./data/ecephys_cache_dir/sessions/mean_speed_df.pkl')
  return area_dict, active_area_dict, mean_speed_df

def generate_graph(adj_mat, confidence_level, active_area, cc=False, weight=False):
  if not weight:
    adj_mat[adj_mat.nonzero()] = 1
  G = nx.from_numpy_array(adj_mat, create_using=nx.DiGraph) # same as from_numpy_matrix
  node_idx = sorted(active_area.keys())
  mapping = {i:node_idx[i] for i in range(len(node_idx))}
  G = nx.relabel_nodes(G, mapping)
  assert set(G.nodes())==set(node_idx)
  nodes = sorted(G.nodes())
  cl = {(nodes[i],nodes[j]):confidence_level[i,j] for i,j in zip(*np.where(~np.isnan(confidence_level)))}
  nx.set_edge_attributes(G, cl, 'confidence')
  if cc: # extract the largest (strongly) connected components
    if np.allclose(adj_mat, adj_mat.T, rtol=1e-05, atol=1e-08): # if the matrix is symmetric
      largest_cc = max(nx.connected_components(G), key=len)
    else:
      largest_cc = max(nx.strongly_connected_components(G), key=len)
    G = nx.subgraph(G, largest_cc)
  return G

#################### load sharp peak corr mat
def load_sharp_peak_xcorr(directory, weight):
  G_dict, peak_dict = {}, {}
  files = os.listdir(directory)
  files.sort(key=lambda x:int(x[:9]))
  for file in files:
    if file.endswith(".npz") and ('_peak' not in file) and ('_bl' not in file):
      print(file)
      adj_mat = load_npz_3d(os.path.join(directory, file))
      # adj_mat = np.load(os.path.join(directory, file))
      mouseID = file.split('_')[0]
      stimulus_name = file.replace('.npz', '').replace(mouseID + '_', '')
      if not mouseID in G_dict:
        G_dict[mouseID], peak_dict[mouseID] = {}, {}
      G_dict[mouseID][stimulus_name] = generate_graph(adj_mat=np.nan_to_num(adj_mat), cc=False, weight=weight)
      peak_dict[mouseID][stimulus_name] = load_npz_3d(os.path.join(directory, file.replace('.npz', '_peak.npz')))
  return G_dict, peak_dict

#################### load sharp integral corr mat
def load_sharp_integral_xcorr(directory, weight):
  G_dict = {}
  files = os.listdir(directory)
  files.sort(key=lambda x:int(x[:9]))
  for file in files:
    if file.endswith(".npz") and ('_bl' not in file):
      print(file)
      adj_mat = load_npz_3d(os.path.join(directory, file))
      # adj_mat = np.load(os.path.join(directory, file))
      mouseID = file.split('_')[0]
      stimulus_name = file.replace('.npz', '').replace(mouseID + '_', '')
      if not mouseID in G_dict:
        G_dict[mouseID] = {}
      G_dict[mouseID][stimulus_name] = generate_graph(adj_mat=np.nan_to_num(adj_mat), cc=False, weight=weight)
  return G_dict

#################### load highland corr mat
def load_highland_xcorr(directory, active_area_dict, weight):
  G_dict, offset_dict, duration_dict = {}, {}, {}
  files = os.listdir(directory)
  files.sort(key=lambda x:int(x[:9]))
  for file in files:
    if file.endswith(".npz") and ('gabors' not in file) and ('flashes' not in file) and ('_offset' not in file) and ('_duration' not in file) and ('_bl' not in file) and ('confidence' not in file):
      print(file)
      adj_mat = load_npz_3d(os.path.join(directory, file))
      confidence_level = load_npz_3d(os.path.join(directory, file.replace('.npz', '_confidence.npz')))
      # adj_mat = np.load(os.path.join(directory, file))
      mouseID = file.split('_')[0]
      stimulus_name = file.replace('.npz', '').replace(mouseID + '_', '')
      if not mouseID in G_dict:
        G_dict[mouseID], offset_dict[mouseID], duration_dict[mouseID] = {}, {}, {}
      G_dict[mouseID][stimulus_name] = generate_graph(adj_mat=np.nan_to_num(adj_mat), confidence_level=confidence_level, active_area=active_area_dict[mouseID], cc=False, weight=weight)
      offset_dict[mouseID][stimulus_name] = load_npz_3d(os.path.join(directory, file.replace('.npz', '_offset.npz')))
      duration_dict[mouseID][stimulus_name] = load_npz_3d(os.path.join(directory, file.replace('.npz', '_duration.npz')))
  return G_dict, offset_dict, duration_dict

########### plot time offset distribution of reciprocal edge
def double0_edge_heatmap(data_dict, max_duration, name, measure, n):
  rows, cols = get_rowcol(data_dict)
  scale = np.zeros(len(rows))
  data_mat = np.zeros((len(rows), len(cols), max_duration+2, max_duration+2))
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      data = data_dict[row][col]
      unidirectional_edge0_mask = (~np.isnan(data)) & (np.isnan(data.T))
      if np.sum(unidirectional_edge0_mask) and not 'bidirectional' in name:
        for row_i, row_j in zip(*np.where(unidirectional_edge0_mask)):
          data_A, data_B = int(data[row_i, row_j])+1, 0
          data_mat[row_ind, col_ind, data_A, data_B] += 1
      unidirectional_edge1_mask = (np.isnan(data)) & (~np.isnan(data.T))
      if np.sum(unidirectional_edge1_mask) and not 'bidirectional' in name:
        for row_i, row_j in zip(*np.where(unidirectional_edge1_mask)):
          data_A, data_B = 0, int(data[row_j, row_i])+1
          data_mat[row_ind, col_ind, data_A, data_B] += 1
      bidirectional_edge_mask = (~np.isnan(data)) & (~np.isnan(data.T))
      if np.sum(bidirectional_edge_mask):
        for row_i, row_j in zip(*np.where(bidirectional_edge_mask)):
          data_A, data_B = int(data[row_i, row_j])+1, int(data[row_j, row_i])+1
          data_mat[row_ind, col_ind, data_A, data_B] += 1
          data_mat[row_ind, col_ind, data_B, data_A] += 1
      data_mat[row_ind, col_ind, :, :] = safe_division(data_mat[row_ind, col_ind, :, :], data_mat[row_ind, col_ind, :, :].sum())
    scale[row_ind] = data_mat[row_ind, :, :, :].max()
  ind = 1
  fig = plt.figure(figsize=(4*len(cols), 3*len(rows)))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      plt.subplot(len(rows), len(cols), ind)
      if row_ind == 0:
        plt.gca().set_title(cols[col_ind], fontsize=20, rotation=0)
      if col_ind == 0:
        plt.gca().text(0, 0.5 * (bottom + top), rows[row_ind],
        horizontalalignment='left',
        verticalalignment='center',
        # rotation='vertical',
        transform=plt.gca().transAxes, fontsize=20, rotation=90)
      plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
      ind += 1
      sns_plot = sns.heatmap(data_mat[row_ind, col_ind, :, :].astype(float), vmin=0, vmax=scale[row_ind],center=0,cmap="RdBu_r", norm=colors.LogNorm())# cmap="YlGnBu"
      sns_plot.set_xticks(np.arange(max_duration+2)+0.5)
      sns_plot.set_xticklabels(['N/A']+list(range(max_duration+1)), rotation=90)
      sns_plot.set_yticks(np.arange(max_duration+2)+0.5)
      sns_plot.set_yticklabels(['N/A']+list(range(max_duration+1)), rotation=0)
      sns_plot.invert_yaxis()
  plt.suptitle('{} distribution'.format(name), size=30)
  plt.tight_layout(rect=[0, 0.03, 1, 0.98])
  # plt.show()
  plt.savefig('./plots/{}_{}_{}fold.jpg'.format(name.replace(' ', '_'), measure, n))

def remove_gabor(G_dict):
  for key in G_dict:
    G_dict[key].pop('gabors', None)
  return G_dict

def remove_thalamic(G_dict, area_dict, regions):
  rows, cols = get_rowcol(G_dict)
  for row in rows:
    for col in cols:
      G = G_dict[row][col].copy()
      nodes = [n for n in G.nodes() if area_dict[row][n] in regions]
      G_dict[row][col] = G.subgraph(nodes)
  return G_dict

def remove_thalamic_mat(data_dict, active_area_dict, regions):
  rows, cols = get_rowcol(data_dict)
  for row in rows:
    for col in cols:
      mat = data_dict[row][col].copy()
      active_area = active_area_dict[row]
      all_node_idx = sorted(list(active_area.keys()))
      active_node_idx = sorted([n for n in active_area if active_area[n] in regions])
      mat_idx = np.array([all_node_idx.index(n) for n in active_node_idx])
      data_dict[row][col] = mat[mat_idx[:, None], mat_idx]
  return data_dict

def remove_thalamic_area(active_area_dict, regions):
  rows = list(active_area_dict.keys())
  for row in rows:
    active_area = active_area_dict[row].copy()
    new_active_area = {n:a for n,a in active_area.items() if a in regions}
    active_area_dict[row] = new_active_area
  return active_area_dict

############### regular network statistics
def split_pos_neg(G_dict, measure):
  pos_G_dict, neg_G_dict = {}, {}
  rows, cols = get_rowcol(G_dict)
  if isinstance(G_dict[rows[0]][cols[0]], list):
    num_sample = len(G_dict[rows[0]][cols[0]])
    no_sample = False
  else:
    no_sample = True
  for row in rows:
    pos_G_dict[row] = {}
    neg_G_dict[row] = {}
    # print(row)
    for col in cols:
      # print(col)
      if not no_sample:
        pos_G_dict[row][col] = []
        neg_G_dict[row][col] = []
        for s in range(num_sample):
          G = G_dict[row][col][s] if col in G_dict[row] else nx.Graph()
          pos_edges = [(u,v,w) for (u,v,w) in G.edges(data=True) if w['weight']>0]
          neg_edges = [(u,v,w) for (u,v,w) in G.edges(data=True) if w['weight']<0]
          Gpos = nx.DiGraph()
          Gneg = nx.DiGraph()
          Gpos.add_edges_from(pos_edges)
          Gneg.add_edges_from(neg_edges)
          pos_G_dict[row][col].append(Gpos)
          neg_G_dict[row][col].append(Gneg)
      else:
        G = G_dict[row][col] if col in G_dict[row] else nx.Graph()
        pos_edges = [(u,v,w) for (u,v,w) in G.edges(data=True) if w['weight']>0]
        neg_edges = [(u,v,w) for (u,v,w) in G.edges(data=True) if w['weight']<0]
        Gpos = nx.DiGraph()
        Gneg = nx.DiGraph()
        Gpos.add_edges_from(pos_edges)
        Gneg.add_edges_from(neg_edges)
        pos_G_dict[row][col] = Gpos
        neg_G_dict[row][col] = Gneg
  return pos_G_dict, neg_G_dict

def get_abs_weight(neg_G_dict):
  abs_neg_G_dict = {}
  rows, cols = get_rowcol(neg_G_dict)
  for row in rows:
    abs_neg_G_dict[row] = {}
    for col in cols:
      G = neg_G_dict[row][col].copy()
      weights = nx.get_edge_attributes(G, "weight")
      abs_weights = {(i, j):abs(weights[i, j]) for i, j in weights}
      nx.set_edge_attributes(G, abs_weights, 'weight')
      abs_neg_G_dict[row][col] = G
  return abs_neg_G_dict

def get_lcc(G_dict):
  G_lcc_dict = {}
  for row in G_dict:
    G_lcc_dict[row] = {}
    for col in G_dict[row]:
      G = G_dict[row][col]
      G_lcc_dict[row][col] = nx.DiGraph()
      if not nx.is_empty(G):
        if nx.is_directed(G):
          if not nx.is_weakly_connected(G):
            Gcc = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
            G_lcc_dict[row][col] = G.subgraph(Gcc[0])
        else:
          if not nx.is_connected(G):
            Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
            G_lcc_dict[row][col] = G.subgraph(Gcc[0])

          # largest_cc = max(nx.connected_components(G), key=len)
          # G_dict[row][col][i] = nx.subgraph(G, largest_cc)
      print(G_dict[row][col].number_of_nodes(), G_lcc_dict[row][col].number_of_nodes())
  return G_lcc_dict

def print_stat(G_dict):
  for row in G_dict:
      for col in G_dict[row]:
          nodes = nx.number_of_nodes(G_dict[row][col])
          edges = nx.number_of_edges(G_dict[row][col])
          print('Number of nodes for {} {} {}'.format(row, col, nodes))
          print('Number of edges for {} {} {}'.format(row, col, edges))
          print('Density for {} {} {}'.format(row, col, nx.density(G_dict[row][col])))

def plot_stat(pos_G_dict, n, neg_G_dict=None, measure='xcorr'):
  rows, cols = get_rowcol(pos_G_dict)
  pos_num_nodes, neg_num_nodes, pos_num_edges, neg_num_edges, pos_densities, neg_densities, pos_total_weight, neg_total_weight, pos_mean_weight, neg_mean_weight, pos_mean_confidence, neg_mean_confidence = [np.full([len(rows), len(cols)], np.nan) for _ in range(12)]
  num_col = 4 if neg_G_dict is not None else 2
  # fig = plt.figure(figsize=(5*num_col, 25))
  fig = plt.figure(figsize=(5*num_col, 13))
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      pos_G = pos_G_dict[row][col] if col in pos_G_dict[row] else nx.DiGraph()
      pos_densities[row_ind, col_ind] = nx.density(pos_G)
      pos_num_nodes[row_ind, col_ind] = nx.number_of_nodes(pos_G)
      pos_num_edges[row_ind, col_ind] = nx.number_of_edges(pos_G)
      pos_total_weight[row_ind, col_ind] = np.sum(list(nx.get_edge_attributes(pos_G, "weight").values()))
      pos_mean_weight[row_ind, col_ind] = np.mean(list(nx.get_edge_attributes(pos_G, "weight").values()))
      pos_mean_confidence[row_ind, col_ind] = np.mean(list(nx.get_edge_attributes(pos_G, "confidence").values()))
      if neg_G_dict is not None:
        neg_G = neg_G_dict[row][col] if col in neg_G_dict[row] else nx.DiGraph()
        neg_densities[row_ind, col_ind] = nx.density(neg_G)
        neg_num_nodes[row_ind, col_ind] = nx.number_of_nodes(neg_G)
        neg_num_edges[row_ind, col_ind] = nx.number_of_edges(neg_G)
        neg_total_weight[row_ind, col_ind] = np.sum(list(nx.get_edge_attributes(neg_G, "weight").values()))
        neg_mean_weight[row_ind, col_ind] = np.mean(list(nx.get_edge_attributes(neg_G, "weight").values()))
        neg_mean_confidence[row_ind, col_ind] = np.mean(list(nx.get_edge_attributes(neg_G, "confidence").values()))
  if neg_G_dict is not None:
    metrics = {'positive number of nodes':pos_num_nodes, 'negative number of nodes':neg_num_nodes, 
  'positive number of edges':pos_num_edges, 'negative number of edges':neg_num_edges, 
  'positive density':pos_densities, 'negative density':neg_densities,
  'positive total weights':pos_total_weight, 'negative total weights':neg_total_weight, 
  'positive average weights':pos_mean_weight, 'negative average weights':neg_mean_weight,
  'positive average confidence':pos_mean_confidence, 'negative average confidence':neg_mean_confidence}
  else:
    metrics = {'total number of nodes':pos_num_nodes, 'total number of edges':pos_num_edges,
    'total density':pos_densities, 'total total weights':pos_total_weight,
    'total average weights':pos_mean_weight, 'total average confidence':pos_mean_confidence}
  # distris = {'total weight distribution':pos_weight_distri, 'negative weight distribution':neg_weight_distri}
  
  for i, k in enumerate(metrics):
    plt.subplot(3, num_col, i+1)
    metric = metrics[k]
    for row_ind, row in enumerate(rows):
      mean = metric[row_ind, :]
      plt.plot(cols, mean, label=row, alpha=0.6)
    plt.gca().set_title(k, fontsize=20, rotation=0)
    plt.xticks(rotation=90)
    # plt.yscale('symlog')
    if i == len(metrics)-1:
      plt.legend()
    if i // num_col < 2:
      plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    plt.tight_layout()
  # plt.show()
  figname = './plots/stats_pos_neg_{}_{}fold.jpg' if neg_G_dict is not None else './plots/stats_total_{}_{}fold.jpg'
  plt.savefig(figname.format(measure, n))

def box_intra_link_ratio(G_dict, area_dict, regions, measure, n):
  df = pd.DataFrame()
  rows, cols = get_rowcol(G_dict)
  # metric = np.zeros((len(rows), len(cols), 3))
  region_connection = np.zeros((len(rows), len(cols), len(regions), len(regions)))
  for col_ind, col in enumerate(cols):
    print(col)
    data = []
    for row_ind, row in enumerate(rows):
      G = G_dict[row][col] if col in G_dict[row] else nx.Graph()
      if G.number_of_nodes() >= 2 and G.number_of_edges() > 0:
        nodes = list(G.nodes())
        node_area = {key: area_dict[row][key] for key in nodes}
        areas = list(node_area.values())
        area_size = [areas.count(r) for r in regions]
        A = nx.to_numpy_array(G)
        A[A.nonzero()] = 1
        for region_ind_i, region_i in enumerate(regions):
          for region_ind_j, region_j in enumerate(regions):
            region_indices_i = np.array([k for k, v in area_dict[row].items() if v==region_i])
            region_indices_j = np.array([k for k, v in area_dict[row].items() if v==region_j])
            region_indices_i = np.array([nodes.index(i) for i in list(set(region_indices_i) & set(nodes))]) # some nodes not in cc are removed 
            region_indices_j = np.array([nodes.index(i) for i in list(set(region_indices_j) & set(nodes))])
            if len(region_indices_i) and len(region_indices_j):
              region_connection[row_ind, col_ind, region_ind_i, region_ind_j] = np.sum(A[region_indices_i[:, None], region_indices_j])
              assert np.sum(A[region_indices_i[:, None], region_indices_j]) == len(A[region_indices_i[:, None], region_indices_j].nonzero()[0])
        diag_indx = np.eye(len(regions),dtype=bool)
        # metric[row_ind, col_ind, 0] =  np.sum(region_connection[row_ind, col_ind][diag_indx])
        # metric[row_ind, col_ind, 1] =  np.sum(region_connection[row_ind, col_ind][~diag_indx])
        data.append(np.sum(region_connection[row_ind, col_ind][diag_indx]) / np.sum(region_connection[row_ind, col_ind]))
    # data = remove_outliers(data, 3)
    df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(data)[:,None], np.array([col] * len(data))[:,None]), 1), columns=['ratio', 'stimulus'])], ignore_index=True)
    df['ratio'] = pd.to_numeric(df['ratio'])
  fig = plt.figure(figsize=(5, 5))
  # ax = sns.violinplot(x='stimulus', y='ratio', data=df, palette="muted", scale='count', cut=0)
  colors_transparency = [transparent_rgb(colors.to_rgb(color), [1,1,1], alpha=1.) for color in stimulus_colors]
  ax = sns.boxplot(x='stimulus', y='ratio', data=df, palette=colors_transparency, showfliers=False)
  plt.xticks(fontsize=10, rotation=90)
  plt.title('ratio of intra region links')
  # plt.yscale('log')
  ax.set(xlabel=None)
  plt.tight_layout()
  # plt.savefig('violin_intra_divide_inter_{}_{}fold.jpg'.format(measure, n))
  plt.savefig('./plots/box_intra_ratio_{}_{}fold.jpg'.format(measure, n))

def plot_intra_inter_box(G_dict, area_dict, regions):
  df = pd.DataFrame()
  rows, cols = get_rowcol(G_dict)
  # metric = np.zeros((len(rows), len(cols), 3))
  region_connection = np.zeros((len(rows), len(cols), len(regions), len(regions)))
  for col_ind, col in enumerate(cols):
    print(col)
    intra_data, inter_data = [], []
    for row_ind, row in enumerate(rows):
      G = G_dict[row][col] if col in G_dict[row] else nx.Graph()
      nodes = list(G.nodes())
      node_area = {key: area_dict[row][key] for key in nodes}
      areas = list(node_area.values())
      area_size = [areas.count(r) for r in regions]
      A = nx.to_numpy_array(G)
      A[A.nonzero()] = 1
      for region_ind_i, region_i in enumerate(regions):
        for region_ind_j, region_j in enumerate(regions):
          region_indices_i = np.array([k for k, v in area_dict[row].items() if v==region_i])
          region_indices_j = np.array([k for k, v in area_dict[row].items() if v==region_j])
          region_indices_i = np.array([nodes.index(i) for i in list(set(region_indices_i) & set(nodes))]) # some nodes not in cc are removed 
          region_indices_j = np.array([nodes.index(i) for i in list(set(region_indices_j) & set(nodes))])
          if len(region_indices_i) and len(region_indices_j):
            region_connection[row_ind, col_ind, region_ind_i, region_ind_j] = np.sum(A[region_indices_i[:, None], region_indices_j])
            assert np.sum(A[region_indices_i[:, None], region_indices_j]) == len(A[region_indices_i[:, None], region_indices_j].nonzero()[0])
      diag_indx = np.eye(len(regions),dtype=bool)
      # metric[row_ind, col_ind, 0] =  np.sum(region_connection[row_ind, col_ind][diag_indx])
      # metric[row_ind, col_ind, 1] =  np.sum(region_connection[row_ind, col_ind][~diag_indx])
      intra_data.append(np.sum(region_connection[row_ind, col_ind][diag_indx]))
      inter_data.append(np.sum(region_connection[row_ind, col_ind][~diag_indx]))
    df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(intra_data)[:,None], np.array(['intra-region'] * len(intra_data))[:,None], np.array([col] * len(intra_data))[:,None]), 1), columns=['number of connections', 'type', 'stimulus']), 
                pd.DataFrame(np.concatenate((np.array(inter_data)[:,None], np.array(['inter-region'] * len(inter_data))[:,None], np.array([col] * len(inter_data))[:,None]), 1), columns=['number of connections', 'type', 'stimulus'])], ignore_index=True)
    df['number of connections'] = pd.to_numeric(df['number of connections'])
  fig = plt.figure(figsize=(5, 5))
  # ax = sns.violinplot(x='stimulus', y='number of connections', hue="type", data=df, palette="muted", split=False)
  ax = sns.boxplot(x='stimulus', y='number of connections', hue="type", data=df, palette="muted",showfliers=False)
  plt.xticks(fontsize=10, rotation=90)
  ax.set(xlabel=None)
  plt.tight_layout()
  # plt.savefig('./plots/violin_intra_inter_{}_{}fold.jpg'.format(measure, n))
  plt.savefig('./plots/box_intra_inter_{}_{}fold.jpg'.format(measure, n))

def plot_pos_neg_box(G_dict, measure, n, density=False):
  df = pd.DataFrame()
  rows, cols = get_rowcol(G_dict)
  for col_ind, col in enumerate(cols):
    print(col)
    ex_data, in_data = [], []
    for row_ind, row in enumerate(rows):
      G = G_dict[row][col] if col in G_dict[row] else nx.Graph()
      signs = list(nx.get_edge_attributes(G, "sign").values())
      if density:
        ex_data.append(signs.count(1) / len(signs))
        in_data.append(signs.count(-1) / len(signs))
      else:
        ex_data.append(signs.count(1))
        in_data.append(signs.count(-1))
    df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(ex_data)[:,None], np.array(['excitatory'] * len(ex_data))[:,None], np.array([col] * len(ex_data))[:,None]), 1), columns=['number of connections', 'type', 'stimulus']), 
                pd.DataFrame(np.concatenate((np.array(in_data)[:,None], np.array(['inhibitory'] * len(in_data))[:,None], np.array([col] * len(in_data))[:,None]), 1), columns=['number of connections', 'type', 'stimulus'])], ignore_index=True)
    df['number of connections'] = pd.to_numeric(df['number of connections'])
  if density:
    y = 'density'
    df['density'] = df['number of connections']
  else:
    y = 'number of connections'
  fig = plt.figure(figsize=(5, 5))
  # ax = sns.violinplot(x='stimulus', y='number of connections', hue="type", data=df, palette="muted", split=False)
  ax = sns.boxplot(x='stimulus', y=y, hue="type", data=df, palette="muted", showfliers = False)
  plt.xticks(fontsize=10, rotation=90)
  ax.set(xlabel=None)
  plt.tight_layout()
  figname = './plots/box_ex_in_num_{}_{}fold.jpg' if not density else './plots/box_ex_in_density_{}_{}fold.jpg'
  # plt.savefig('./plots/violin_intra_inter_{}_{}fold.jpg'.format(measure, n))
  plt.savefig(figname.format(measure, n))

#############################################################################
# signed louvain algorithm with Hamiltonian for community detection in signed graphs
class NotAPartition(NetworkXError):
    """Raised if a given collection is not a partition."""

    def __init__(self, G, collection):
        msg = f"{G} is not a valid partition of the graph {collection}"
        super().__init__(msg)

def signed_degree(G, sign_type='pos', dir_type='out', weight='weight'):
  degree = {}
  for node in G.nodes():
    if dir_type == 'out':
      if sign_type == 'pos':
        weights = [G[node][n][weight] for n in G.successors(node) if G[node][n][weight] > 0]
      elif sign_type == 'neg':
        weights = [G[node][n][weight] for n in G.successors(node) if G[node][n][weight] < 0]
      elif sign_type == 'abs':
        weights = [abs(G[node][n][weight]) for n in G.successors(node)]
    elif dir_type == 'in':
      if sign_type == 'pos':
        weights = [G[n][node][weight] for n in G.predecessors(node) if G[n][node][weight] > 0]
      elif sign_type == 'neg':
        weights = [G[n][node][weight] for n in G.predecessors(node) if G[n][node][weight] < 0]
      elif sign_type == 'abs':
        weights = [abs(G[node][n][weight]) for n in G.predecessors(node)]
    elif dir_type == 'undirected':
      if sign_type == 'pos':
        weights = [G[n][node][weight] for n in G.neighbors(node) if G[n][node][weight] > 0]
      elif sign_type == 'neg':
        weights = [G[n][node][weight] for n in G.neighbors(node) if G[n][node][weight] < 0]
      elif sign_type == 'abs':
        weights = [abs(G[node][n][weight]) for n in G.neighbors(node)]
    degree[node] = sum(weights)
  return degree

def Hamiltonian(G, communities, weight="weight", pos_resolution=1, neg_resolution=1):
    if not isinstance(communities, list):
        communities = list(communities)
    if not is_partition(G, communities):
        raise NotAPartition(G, communities)

    directed = G.is_directed()
    if directed:
    #   out_degree = signed_degree(G, 'abs', 'out', weight)
      out_degree_pos = signed_degree(G, 'pos', 'out', weight)
      out_degree_neg = signed_degree(G, 'neg', 'out', weight)
      in_degree_pos = signed_degree(G, 'pos', 'in', weight)
      in_degree_neg = signed_degree(G, 'neg', 'in', weight)
      # m = sum(abs(out_degree.values()))
    else:
        out_degree_pos = in_degree_pos = signed_degree(G, 'pos', 'undirected', weight)
        out_degree_neg = in_degree_neg = signed_degree(G, 'neg', 'undirected', weight)
    pos_norm = safe_division(1, sum(out_degree_pos.values()))
    neg_norm = safe_division(1, sum(out_degree_neg.values()))

    def signed_community_contribution(community):
        comm = set(community)
        L_c = sum(wt for u, v, wt in G.edges(comm, data=weight, default=1) if v in comm)
        out_degree_pos_sum = sum(out_degree_pos[u] for u in comm)
        in_degree_pos_sum = sum(in_degree_pos[u] for u in comm) if directed else out_degree_pos_sum
        out_degree_neg_sum = sum(out_degree_neg[u] for u in comm)
        in_degree_neg_sum = sum(in_degree_neg[u] for u in comm) if directed else out_degree_neg_sum

        return - L_c + pos_resolution * out_degree_pos_sum * in_degree_pos_sum * pos_norm - neg_resolution * out_degree_neg_sum * in_degree_neg_sum * neg_norm

    return sum(map(signed_community_contribution, communities))

@py_random_state("seed")
def signed_louvain_communities(
    G, weight="weight", pos_resolution=1, neg_resolution=1, threshold=0.0000001, seed=None
):
    d = signed_louvain_partitions(G, weight, pos_resolution, neg_resolution, threshold, seed)
    q = deque(d, maxlen=1)
    return q.pop()


@py_random_state("seed")
def signed_louvain_partitions(
    G, weight="weight", pos_resolution=1, neg_resolution=1, threshold=0.0000001, seed=None
):
    partition = [{u} for u in G.nodes()]
    ham = Hamiltonian(G, partition, weight, pos_resolution, neg_resolution)
    is_directed = G.is_directed()
    if G.is_multigraph():
        graph = _convert_multigraph(G, weight, is_directed)
    else:
        graph = G.__class__()
        graph.add_nodes_from(G)
        graph.add_weighted_edges_from(G.edges(data=weight, default=1))
    
    out_degree_pos = signed_degree(G, 'pos', 'out', weight)
    out_degree_neg = signed_degree(G, 'neg', 'out', weight)
    pos_norm = safe_division(1, sum(out_degree_pos.values()))
    neg_norm = safe_division(1, sum(out_degree_neg.values()))
    # pos_norm = 1 / sum(out_degree_pos.values())
    # neg_norm = 1 / sum(out_degree_neg.values())
    partition, inner_partition, improvement = _one_level(
        graph, pos_norm, neg_norm, partition, weight, pos_resolution, neg_resolution, is_directed, seed
    )
    improvement = True
    while improvement:
        yield partition
        new_ham = Hamiltonian(graph, inner_partition, weight, pos_resolution, neg_resolution)
        if ham - new_ham <= threshold:
            return
        ham = new_ham
        graph = _gen_graph(graph, inner_partition)
        partition, inner_partition, improvement = _one_level(
            graph, pos_norm, neg_norm, partition, weight, pos_resolution, neg_resolution, is_directed, seed
        )

def _one_level(G, pos_norm, neg_norm, partition, weight='weight', pos_resolution=1, neg_resolution=1, is_directed=False, seed=None):
    node2com = {u: i for i, u in enumerate(G.nodes())}
    inner_partition = [{u} for u in G.nodes()]
    if is_directed:
      out_degrees_pos = signed_degree(G, 'pos', 'out', weight)
      out_degrees_neg = signed_degree(G, 'neg', 'out', weight)
      in_degrees_pos = signed_degree(G, 'pos', 'in', weight)
      in_degrees_neg = signed_degree(G, 'neg', 'in', weight)
      Stot_in_pos = [deg for deg in in_degrees_pos.values()]
      Stot_in_neg = [deg for deg in in_degrees_neg.values()]
      Stot_out_pos = [deg for deg in out_degrees_pos.values()]
      Stot_out_neg = [deg for deg in out_degrees_neg.values()]
      # Calculate weights for both in and out neighbours
      nbrs = {} # key is each node and its in and out neighbors, value is their weight
      for u in G:
        nbrs[u] = defaultdict(float)
        for _, n, wt in G.out_edges(u, data="weight"):
          nbrs[u][n] += wt
        for n, _, wt in G.in_edges(u, data="weight"):
          nbrs[u][n] += wt
    else:
        pos_degrees = signed_degree(G, 'pos', 'undirected', weight)
        neg_degrees = signed_degree(G, 'neg', 'undirected', weight)
        Stot_pos = [deg for deg in pos_degrees.values()]
        Stot_neg = [deg for deg in neg_degrees.values()]
        nbrs = {u: {v: data["weight"] for v, data in G[u].items() if v != u} for u in G}
    rand_nodes = list(G.nodes)
    seed.shuffle(rand_nodes)
    nb_moves = 1
    improvement = False
    while nb_moves > 0:
        nb_moves = 0
        for u in rand_nodes:
            best_gain = 0
            best_com = node2com[u]
            weights2com = _neighbor_weights(nbrs[u], node2com) # summed weights of in and out neighbors of u, key is their current community
            if is_directed:
                in_degree_pos = in_degrees_pos[u]
                out_degree_pos = out_degrees_pos[u]
                in_degree_neg = in_degrees_neg[u]
                out_degree_neg = out_degrees_neg[u]
                Stot_in_pos[best_com] -= in_degree_pos # pos in degree of other nodes in the same community
                Stot_out_pos[best_com] -= out_degree_pos
                Stot_in_neg[best_com] -= in_degree_neg
                Stot_out_neg[best_com] -= out_degree_neg
                remove_cost = (
                    + weights2com[best_com]
                    - pos_resolution * pos_norm * (out_degree_pos * Stot_in_pos[best_com] + in_degree_pos * Stot_out_pos[best_com])
                    + neg_resolution * neg_norm * (out_degree_neg * Stot_in_neg[best_com] + in_degree_neg * Stot_out_neg[best_com])
                )
            else:
                pos_degree = pos_degrees[u]
                neg_degree = neg_degrees[u]
                Stot_pos[best_com] -= pos_degree
                Stot_neg[best_com] -= neg_degree
                remove_cost = +weights2com[best_com] - pos_resolution * (
                    Stot_pos[best_com] * pos_degree) * 2 * pos_norm \
                    + neg_resolution * (
                    Stot_neg[best_com] * neg_degree) * 2 * neg_norm
            for nbr_com, wt in weights2com.items(): # compare every other community nbr_com node u can move to
                if wt > 0: # only move to a neighbor's community if weight is positive, unless will cause node u to consistently jump between negative neighbor's community
                    if is_directed:
                        gain = (
                            remove_cost
                            - wt
                            + pos_resolution * pos_norm * (out_degree_pos * Stot_in_pos[nbr_com] + in_degree_pos * Stot_out_pos[nbr_com])
                            - neg_resolution * neg_norm * (out_degree_neg * Stot_in_neg[nbr_com] + in_degree_neg * Stot_out_neg[nbr_com])
                        )
                    else:
                        gain = (
                            remove_cost
                            - wt
                            + pos_resolution * (Stot_pos[nbr_com] * pos_degree) * 2 * pos_norm
                            - neg_resolution * (Stot_neg[nbr_com] * neg_degree) * 2 * pos_norm
                        )
                    if gain < best_gain:
                        best_gain = gain
                        best_com = nbr_com
            if is_directed:
                Stot_in_pos[best_com] += in_degree_pos
                Stot_out_pos[best_com] += out_degree_pos
                Stot_in_neg[best_com] += in_degree_neg
                Stot_out_neg[best_com] += out_degree_neg
            else:
                Stot_pos[best_com] += pos_degree
                Stot_neg[best_com] += neg_degree
            if best_com != node2com[u]:
                com = G.nodes[u].get("nodes", {u})
                partition[node2com[u]].difference_update(com)
                inner_partition[node2com[u]].remove(u)
                partition[best_com].update(com)
                inner_partition[best_com].add(u)
                improvement = True
                nb_moves += 1
                node2com[u] = best_com
  
    partition = list(filter(len, partition))
    inner_partition = list(filter(len, inner_partition))
    return partition, inner_partition, improvement


def _neighbor_weights(nbrs, node2com):
    weights = defaultdict(float)
    for nbr, wt in nbrs.items():
        weights[node2com[nbr]] += wt
    return weights


def _gen_graph(G, partition):
    # generate a graph whose node is community, edge weight is the summation of weights between two communities
    H = G.__class__()
    node2com = {}
    for i, part in enumerate(partition):
        nodes = set()
        for node in part:
            node2com[node] = i
            nodes.update(G.nodes[node].get("nodes", {node}))
        H.add_node(i, nodes=nodes)

    for node1, node2, wt in G.edges(data=True):
        wt = wt["weight"]
        com1 = node2com[node1]
        com2 = node2com[node2]
        temp = H.get_edge_data(com1, com2, {"weight": 0})["weight"]
        H.add_edge(com1, com2, **{"weight": wt + temp})
    return H

def _convert_multigraph(G, weight, is_directed):
    if is_directed:
        H = nx.DiGraph()
    else:
        H = nx.Graph()
    H.add_nodes_from(G)
    for u, v, wt in G.edges(data=weight, default=1):
        if H.has_edge(u, v):
            H[u][v]["weight"] += wt
        else:
            H.add_edge(u, v, weight=wt)
    return H

def get_Hamiltonian(G, weight='weight', pos_resolution=1, neg_resolution=1, comms=None):
  if comms == None:
    comms = signed_louvain_communities(G, weight=weight, pos_resolution=pos_resolution, neg_resolution=neg_resolution)
  return Hamiltonian(G, comms, weight=weight, pos_resolution=pos_resolution, neg_resolution=neg_resolution)

def get_random_Hamiltonian(G, num_rewire, algorithm, weight='weight', pos_resolution=1, neg_resolution=1):
  random_Hamiltonian = np.zeros(num_rewire)
  random_Gs = random_graph_generator(G, num_rewire, algorithm=algorithm)
  for random_ind, random_G in enumerate(random_Gs):
    random_Hamiltonian[random_ind] = get_Hamiltonian(random_G, weight=weight, pos_resolution=pos_resolution, neg_resolution=neg_resolution)
  return random_Hamiltonian

def comms_Hamiltonian_resolution(G_dict, resolution_list, num_repeat, num_rewire, cc=False):
  rows, cols = get_rowcol(G_dict)
  comms_dict = {}
  Hamiltonian = np.full([len(rows), len(cols), len(resolution_list), num_repeat], np.nan)
  gnm_Hamiltonian, config_Hamiltonian = [np.full([len(rows), len(cols), len(resolution_list), num_rewire], np.nan) for _ in range(2)]
  for row_ind, row in enumerate(rows):
    comms_dict[row] = {}
    for col_ind, col in enumerate(cols):
      comms_dict[row][col] = {}
      G = G_dict[row][col].copy() if col in G_dict[row] else nx.DiGraph()
      if cc:
        G = get_lcc(G)
      num_pos = sum([w for i,j,w in G.edges.data('weight') if w > 0])
      num_neg = sum([w for i,j,w in G.edges.data('weight') if w < 0])
      for resolution_ind, resolution in enumerate(resolution_list):
        print(row, col, round(resolution, 2))
        pos_resolution = abs(resolution * num_neg / num_pos)
        comms_dict[row][col][resolution] = []
        for repeat in range(num_repeat):
          comms = signed_louvain_communities(G.copy(), weight='weight', pos_resolution=pos_resolution, neg_resolution=resolution)
          comms_dict[row][col][resolution].append(comms)
          Hamiltonian[row_ind, col_ind, resolution_ind, repeat] = get_Hamiltonian(G.copy(), weight='weight', pos_resolution=pos_resolution, neg_resolution=resolution, comms=comms)
        gnm_Hamiltonian[row_ind, col_ind, resolution_ind] = get_random_Hamiltonian(G.copy(), num_rewire, algorithm='Gnm', weight='weight', pos_resolution=pos_resolution, neg_resolution=resolution)
        config_Hamiltonian[row_ind, col_ind, resolution_ind] = get_random_Hamiltonian(G.copy(), num_rewire, algorithm='directed_configuration_model', weight='weight', pos_resolution=pos_resolution, neg_resolution=resolution)   
  metrics = {'Hamiltonian':Hamiltonian, 'Gnm Hamiltonian':gnm_Hamiltonian, 'configuration model Hamiltonian':config_Hamiltonian}
  return comms_dict, metrics

def get_max_dH_resolution(rows, cols, resolution_list, metrics): 
  max_reso_gnm, max_reso_config = np.zeros((len(rows), len(cols))), np.zeros((len(rows), len(cols)))
  Hamiltonian, gnm_Hamiltonian, config_Hamiltonian = metrics['Hamiltonian'], metrics['Gnm Hamiltonian'], metrics['configuration model Hamiltonian']
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      metric_mean = Hamiltonian[row_ind, col_ind].mean(-1)
      metric_gnm = gnm_Hamiltonian[row_ind, col_ind].mean(-1)
      metric_config = config_Hamiltonian[row_ind, col_ind].mean(-1)
      max_reso_gnm[row_ind, col_ind] = resolution_list[np.argmax(metric_gnm - metric_mean)]
      max_reso_config[row_ind, col_ind] = resolution_list[np.argmax(metric_config - metric_mean)]
  return max_reso_gnm, max_reso_config

def get_max_pos_reso(G_ccg_dict, max_neg_reso):
  rows, cols = get_rowcol(G_ccg_dict)
  max_pos_reso = np.zeros((len(rows), len(cols)))
  for row_ind, row in enumerate(rows):
    for col_ind, col in enumerate(cols):
      G = G_ccg_dict[row][col]
      num_pos = sum([w for i,j,w in G.edges.data('weight') if w > 0])
      num_neg = sum([w for i,j,w in G.edges.data('weight') if w < 0])
      max_pos_reso[row_ind, col_ind] = abs(max_neg_reso[row_ind, col_ind] * num_neg / num_pos)
  return max_pos_reso

def plot_Hamiltonian_resolution(rows, cols, resolution_list, metrics, measure, n): 
  num_row, num_col = len(rows), len(cols)
  fig = plt.figure(figsize=(5*num_col, 5*num_row))
  ind = 1
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      plt.subplot(num_row, num_col, ind)
      ind += 1
      metric = metrics['Hamiltonian'].mean(-1)
      plt.plot(resolution_list, metric[row_ind, col_ind], label=r'$H$', alpha=0.6)
      metric_config = metrics['configuration model Hamiltonian'][row_ind, col_ind].mean(-1)
      metric_gnm = metrics['Gnm Hamiltonian'][row_ind, col_ind].mean(-1)
      plt.plot(resolution_list, metric_config, color='r', label=r'$H_{config}$', alpha=0.6)
      plt.plot(resolution_list, metric_gnm, color='g', label=r'$H_{Gnm}$', alpha=0.6)
      plt.plot(resolution_list, metric[row_ind, col_ind] - metric_config, 'r--', label=r'$H-H_{config}$', alpha=0.8)
      plt.plot(resolution_list, metric[row_ind, col_ind] - metric_gnm, 'g--', label=r'$H-H_{Gnm}$', alpha=0.8)
      # plt.gca().set_title(k, fontsize=14, rotation=0)
      plt.xticks(rotation=0)
      
      # plt.yscale('symlog')
      if ind == num_row*num_col+1:
        plt.legend(fontsize=20)
      if row_ind == 0:
        plt.title(col, size=25)
      if row_ind < num_row -1 :
        plt.tick_params(
          axis='x',          # changes apply to the x-axis
          which='both',      # both major and minor ticks are affectedrandom_metrics
          bottom=False,      # ticks along the bottom edge are off
          top=False,         # ticks along the top edge are off
          labelbottom=False) # labels along the bottom edge are off
      else:
        plt.xlabel(r'$\gamma^-$', size=20)
  plt.suptitle('Hamiltonian', size=30)
  plt.tight_layout(rect=[0, 0.03, 1, 0.98])
  # plt.show()
  figname = './plots/{}_{}_{}fold.jpg'
  # plt.show()
  plt.savefig(figname.format('Hamiltonian', measure, n))

def stat_modular_structure_Hamiltonian_comms(G_dict, measure, n, resolution_list, max_neg_reso=None, comms_dict=None, metrics=None, max_method='none'):
  rows, cols = get_rowcol(G_dict)
  if max_neg_reso is None:
    max_neg_reso = np.ones((len(rows), len(cols)))
  num_comm, hamiltonian, num_lcomm, cov_lcomm = [np.full([len(rows), len(cols)], np.nan) for _ in range(4)]
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      G = G_dict[row][col].copy() if col in G_dict[row] else nx.DiGraph()
      max_reso = max_neg_reso[row_ind][col_ind]
      comms_list = comms_dict[row][col][max_reso]
      hamiltonian[row_ind, col_ind] = metrics['Hamiltonian'][row_ind, col_ind, np.where(resolution_list==max_reso)[0][0]].mean()
      nc, nl, cl = [], [], []
      for comms in comms_list:
        nc.append(len(comms))
        count = np.array([len(comm) for comm in comms])
        nl.append(sum(count >= 4))
        cl.append(count[count >=4].sum() / G.number_of_nodes())
      num_comm[row_ind, col_ind] = np.mean(nc)
      num_lcomm[row_ind, col_ind] = np.mean(nl)
      cov_lcomm[row_ind, col_ind] = np.mean(cl)
      
  metrics2plot = {'number of communities':num_comm, 'Hamiltonian':hamiltonian, 
  'number of large communities':num_lcomm, 'coverage of large comm':cov_lcomm}
  num_col = 2
  num_row = int(len(metrics2plot) / num_col)
  fig = plt.figure(figsize=(5*num_col, 5*num_row))
  for i, k in enumerate(metrics2plot):
    plt.subplot(num_row, num_col, i+1)
    metric = metrics2plot[k]
    for row_ind, row in enumerate(rows):
      mean = metric[row_ind, :]
      plt.plot(cols, mean, label=row, alpha=0.6)
    plt.gca().set_title(k, fontsize=14, rotation=0)
    plt.xticks(rotation=90)
    # plt.yscale('symlog')
    if i == len(metrics2plot)-1:
      plt.legend()
    if i // num_col < num_row - 1:
      plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
  plt.suptitle(max_method, size=25)
  plt.tight_layout(rect=[0, 0.03, 1, 0.98])
  # plt.show()
  figname = './plots/stat_modular_Hamiltonian_maxreso_{}_{}_{}fold.jpg'
  plt.savefig(figname.format(max_method, measure, n))

def stat_modular_structure_Hamiltonian(G_dict, measure, n, max_pos_reso=None, max_neg_reso=None, max_method='none'):
  rows, cols = get_rowcol(G_dict)
  if max_pos_reso is None:
    max_pos_reso = np.ones((len(rows), len(cols)))
  if max_neg_reso is None:
    max_neg_reso = np.ones((len(rows), len(cols)))
  num_comm, hamiltonian, num_lcomm, cov_lcomm = [np.full([len(rows), len(cols)], np.nan) for _ in range(4)]
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      G = G_dict[row][col].copy() if col in G_dict[row] else nx.DiGraph()
      comms = signed_louvain_communities(G, weight='weight', pos_resolution=max_pos_reso[row_ind, col_ind], neg_resolution=max_neg_reso[row_ind, col_ind])
      num_comm[row_ind, col_ind] = len(comms)
      hamiltonian[row_ind, col_ind] = get_Hamiltonian(G, weight='weight', pos_resolution=max_pos_reso[row_ind, col_ind], neg_resolution=max_neg_reso[row_ind, col_ind], comms=comms)
      count = np.array([len(comm) for comm in comms])
      num_lcomm[row_ind, col_ind] = sum(count >= 4)
      cov_lcomm[row_ind, col_ind] = count[count >=4].sum() / G.number_of_nodes()
      assert G.number_of_nodes() == count.sum()
      
  metrics = {'number of communities':num_comm, 'Hamiltonian':hamiltonian, 
  'number of large communities':num_lcomm, 'coverage of large comm':cov_lcomm}
  num_col = 2
  num_row = int(len(metrics) / num_col)
  fig = plt.figure(figsize=(5*num_col, 5*num_row))
  for i, k in enumerate(metrics):
    plt.subplot(num_row, num_col, i+1)
    metric = metrics[k]
    for row_ind, row in enumerate(rows):
      mean = metric[row_ind, :]
      plt.plot(cols, mean, label=row, alpha=0.6)
    plt.gca().set_title(k, fontsize=14, rotation=0)
    plt.xticks(rotation=90)
    # plt.yscale('symlog')
    if i == len(metrics)-1:
      plt.legend()
    if i // num_col < num_row - 1:
      plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
  plt.suptitle(max_method, size=25)
  plt.tight_layout(rect=[0, 0.03, 1, 0.98])
  # plt.show()
  figname = './plots/stat_modular_Hamiltonian_{}_{}_{}fold.jpg'
  plt.savefig(figname.format(max_method, measure, n))

def plot_Hcomm_size_purity(comms_dict, area_dict, measure, n, max_neg_reso=None, max_method='none'):
  ind = 1
  rows, cols = get_rowcol(comms_dict)
  if max_neg_reso is None:
    max_neg_reso = np.ones((len(rows), len(cols)))
  fig = plt.figure(figsize=(23, 9))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  for col_ind, col in enumerate(cols):
    print(col)
    plt.subplot(4, 2, ind)
    plt.gca().set_title(cols[col_ind], fontsize=20, rotation=0)
    plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ind += 1
    data = {}
    for row_ind, row in enumerate(rows):
      max_reso = max_neg_reso[row_ind][col_ind]
      comms_list = comms_dict[row][col][max_reso]
      for comms in comms_list:
        sizes = [len(comm) for comm in comms]
        # part = community.best_partition(G, weight='weight')
        # comms, sizes = np.unique(list(part.values()), return_counts=True)
        for comm, size in zip(comms, sizes):
          c_regions = [area_dict[row][node] for node in comm]
          _, counts = np.unique(c_regions, return_counts=True)
          assert len(c_regions) == size == counts.sum()
          purity = counts.max() / size
          if size in data:
            data[size].append(purity)
          else:
            data[size] = [purity]
    
    c_size, c_purity = [k for k,v in sorted(data.items())], [v for k,v in sorted(data.items())]
    plt.boxplot(c_purity)
    plt.xticks(list(range(1, len(c_purity)+1)), c_size, rotation=0)
    # plt.hist(data.flatten(), bins=12, density=True)
    # plt.axvline(x=np.nanmean(data), color='r', linestyle='--')
    # plt.xlabel('region')
    plt.xlabel('size')
    plt.ylabel('purity')
  plt.suptitle('{} Hamiltonian community purity VS size'.format(max_method), size=30)
  plt.tight_layout()
  image_name = './plots/Hcomm_size_purity_{}_{}_{}fold.jpg'.format(max_method, measure, n)
  # plt.show()
  plt.savefig(image_name)

def plot_top_Hcomm_purity(comms_dict, num_top, area_dict, measure, n, max_neg_reso=None, max_method='none'):
  ind = 1
  rows, cols = get_rowcol(comms_dict)
  if max_neg_reso is None:
    max_neg_reso = np.ones((len(rows), len(cols)))
  fig = plt.figure(figsize=(7, 4))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  top_purity = []
  for col_ind, col in enumerate(cols):
    print(col)
    top_purity_col = []
    for row_ind, row in enumerate(rows):
      max_reso = max_neg_reso[row_ind][col_ind]
      comms_list = comms_dict[row][col][max_reso]
      for comms in comms_list: # 100 repeats
        data = []
        sizes = [len(comm) for comm in comms]
        # part = community.best_partition(G, weight='weight')
        # comms, sizes = np.unique(list(part.values()), return_counts=True)
        for comm, size in zip(comms, sizes):
          c_regions = [area_dict[row][node] for node in comm]
          _, counts = np.unique(c_regions, return_counts=True)
          assert len(c_regions) == size == counts.sum()
          purity = counts.max() / size
          data.append((size, purity))
        c_purity = [v for k,v in sorted(data, key=lambda x:x[0], reverse=True) if k>=4][:num_top]
        # c_size, c_purity = [k for k,v in sorted(data.items(), reverse=True)][:num_top], [v for k,v in sorted(data.items(), reverse=True)][:num_top]
        # print(c_size, c_purity)
        # c_purity = [x for xs in c_purity for x in xs]
        # top_purity_col += [x for xs in c_purity for x in xs]
        top_purity_col += [x for x in c_purity]
    top_purity.append(top_purity_col)
  plt.boxplot(top_purity)
  plt.xticks(list(range(1, len(top_purity)+1)), cols, rotation=90)
  # plt.hist(data.flatten(), bins=12, density=True)
  # plt.axvline(x=np.nanmean(data), color='r', linestyle='--')
  # plt.xlabel('region')
  plt.ylabel('purity')
  plt.title(' top {} largest {} Hamiltonian community purity'.format(num_top, max_method), size=18)
  plt.tight_layout()
  image_name = './plots/top_{}_Hcomm_purity_{}_{}_{}fold.jpg'.format(num_top, max_method, measure, n)
  # plt.show()
  plt.savefig(image_name)
  return top_purity

def plot_scatter_purity_Hcommsize(comms_dict, area_dict, measure, n, max_neg_reso=None, max_method='none'):
  ind = 1
  rows, cols = get_rowcol(comms_dict)
  if max_neg_reso is None:
    max_neg_reso = np.ones((len(rows), len(cols)))
  fig = plt.figure(figsize=(7, 6))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  size_dict = {}
  purity_dict = {}
  for col_ind, col in enumerate(cols):
    print(col)
    size_col = []
    purity_col = []
    for row_ind, row in enumerate(rows):
      max_reso = max_neg_reso[row_ind][col_ind]
      comms_list = comms_dict[row][col][max_reso]
      for comms in comms_list: # 100 repeats
        data = []
        sizes = [len(comm) for comm in comms]
        # part = community.best_partition(G, weight='weight')
        # comms, sizes = np.unique(list(part.values()), return_counts=True)
        for comm, size in zip(comms, sizes):
          c_regions = [area_dict[row][node] for node in comm]
          _, counts = np.unique(c_regions, return_counts=True)
          assert len(c_regions) == size == counts.sum()
          purity = counts.max() / size
          data.append((size, purity))
        size_col += [k for k,v in data if k>=4]
        purity_col += [v for k,v in data if k>=4]
    size_dict[col] = size_col
    purity_dict[col] = purity_col
  color_list = ['tab:blue', 'darkorange', 'bisque', 'limegreen', 'darkgreen', 'maroon', 'indianred', 'mistyrose']
  for col_ind, col in enumerate(size_dict):
    plt.scatter(size_dict[col], purity_dict[col], color=color_list[col_ind], label=col, alpha=0.8)
  plt.legend()
  plt.xscale('log')
  plt.xlabel('community size')
  plt.ylabel('purity')
  plt.title('{} purity VS community size'.format(max_method), size=18)
  plt.tight_layout()
  image_name = './plots/Hcomm_purity_size_{}_{}_{}fold.jpg'.format(max_method, measure, n)
  # plt.show()
  plt.savefig(image_name)

def plot_dist_Hcommsize(comms_dict, measure, n, max_neg_reso=None, max_method='none'):
  ind = 1
  rows, cols = get_rowcol(comms_dict)
  if max_neg_reso is None:
    max_neg_reso = np.ones((len(rows), len(cols)))
  fig = plt.figure(figsize=(7, 6))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  df = pd.DataFrame()
  for col_ind, col in enumerate(cols):
    print(col)
    size_col = []
    for row_ind, row in enumerate(rows):
      max_reso = max_neg_reso[row_ind][col_ind]
      comms_list = comms_dict[row][col][max_reso]
      for comms in comms_list: # 100 repeats
        sizes = [len(comm) for comm in comms]
        size_col += [k for k in sizes if k>=4]
    df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(size_col)[:,None], np.array([col] * len(size_col))[:,None]), 1), columns=['community size', 'stimulus'])], ignore_index=True)
  color_list = ['tab:blue', 'tab:orange', 'limegreen', 'darkgreen', 'maroon', 'indianred', 'mistyrose']
  df['community size'] = pd.to_numeric(df['community size'])
  sns.kdeplot(data=df, x='community size', hue='stimulus', palette=color_list, cut=0)
  # plt.legend()
  plt.xscale('log')
  plt.yscale('log')
  # plt.xlabel('community size')
  # plt.ylabel('purity')
  plt.title('{} distribution of community size'.format(max_method), size=18)
  plt.tight_layout()
  image_name = './plots/dist_Hcomm_size_{}_{}_{}fold.jpg'.format(max_method, measure, n)
  # plt.show()
  plt.savefig(image_name)

def plot_2Ddist_Hcommsize(comms_dict, area_dict, measure, n, max_neg_reso=None, max_method='none', kind='scatter'):
  rows, cols = get_rowcol(comms_dict)
  if max_neg_reso is None:
    max_neg_reso = np.ones((len(rows), len(cols)))
  fig = plt.figure(figsize=(10, 10))
  df = pd.DataFrame()
  for col_ind, col in enumerate(cols):
    print(col)
    size_col = []
    purity_col = []
    for row_ind, row in enumerate(rows):
      max_reso = max_neg_reso[row_ind][col_ind]
      comms_list = comms_dict[row][col][max_reso]
      for comms in comms_list: # 100 repeats
        sizes = [len(comm) for comm in comms]
        data = []
        for comm, size in zip(comms, sizes):
          c_regions = [area_dict[row][node] for node in comm]
          _, counts = np.unique(c_regions, return_counts=True)
          assert len(c_regions) == size == counts.sum()
          purity = counts.max() / size
          data.append((size, purity))
        size_col += [s for s,p in data if s>=4]
        purity_col += [p for s,p in data if s>=4]
    df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(size_col)[:,None], np.array(purity_col)[:,None], np.array([col] * len(size_col))[:,None]), 1), columns=['community size', 'purity', 'stimulus'])], ignore_index=True)
  color_list = ['tab:blue', 'darkorange', 'bisque', 'limegreen', 'darkgreen', 'maroon', 'indianred', 'mistyrose']
  df['community size'] = pd.to_numeric(df['community size'])
  df['purity'] = pd.to_numeric(df['purity'])
  if kind == 'scatter':
    sns.jointplot(data=df, x='community size', y='purity', hue='stimulus', kind='scatter', palette=color_list, alpha=0.7)
  elif kind == 'kde':
    sns.jointplot(data=df, x='community size', y='purity', hue='stimulus', kind='kde', ylim=(0.13, 1.0), log_scale=[True, False], palette=color_list, alpha=0.7)
  # plt.xscale('log')
  # plt.yscale('log')
  # plt.xlabel('community size')
  # plt.ylabel('purity')
  # plt.xlim(-5, 250)
  # plt.ylim(0.12, 1.02)
  plt.suptitle('{} 2D distribution of community size'.format(max_method), size=18)
  plt.tight_layout()
  image_name = './plots/dist2D_Hcomm_size_{}_{}_{}_{}fold.jpg'.format(kind, max_method, measure, n)
  # plt.show()
  plt.savefig(image_name)

def plot_scatter_mean_purity_Hcommsize_col(comms_dict, area_dict, measure, n, max_neg_reso=None, max_method='none'):
  rows, cols = get_rowcol(comms_dict)
  if max_neg_reso is None:
    max_neg_reso = np.ones((len(rows), len(cols)))
  size_dict = {}
  purity_dict = {}
  for col_ind, col in enumerate(cols):
    print(col)
    size_col = []
    purity_col = []
    data = defaultdict(list)
    for row_ind, row in enumerate(rows):
      max_reso = max_neg_reso[row_ind][col_ind]
      comms_list = comms_dict[row][col][max_reso]
      for comms in comms_list: # 100 repeats
        sizes = [len(comm) for comm in comms]
        # part = community.best_partition(G, weight='weight')
        # comms, sizes = np.unique(list(part.values()), return_counts=True)
        for comm, size in zip(comms, sizes):
          c_regions = [area_dict[row][node] for node in comm]
          _, counts = np.unique(c_regions, return_counts=True)
          assert len(c_regions) == size == counts.sum()
          purity = counts.max() / size
          data[size].append(purity)
    size_col = [k for k,v in data.items() if k>=4]
    purity_col = [np.mean(v) for k,v in data.items() if k>=4]
    size_dict[col] = size_col
    purity_dict[col] = purity_col
  fig = plt.figure(figsize=(9*len(cols)/2, 6*2))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  for col_ind, col in enumerate(size_dict):
    ax=plt.subplot(2, int(np.ceil(len(cols)/2)), col_ind+1)
    plt.scatter(size_dict[col], purity_dict[col], color=stimulus_colors[col_ind], label=col, alpha=1)
    # popt, pcov = curve_fit(func_powerlaw, size_dict[col], purity_dict[col], p0=[1, 1]) #, bounds=[[1e-3, 1e-3], [1e20, 50]]
    # plt.plot(sorted(size_dict[col]), func_powerlaw(sorted(size_dict[col]), *popt), '-', color=stimulus_colors[col_ind], alpha=.8, linewidth=5)
    plt.title(col, size=30)
    coef = np.polyfit(size_dict[col], purity_dict[col], 1)
    poly1d_fn = np.poly1d(coef) 
    plt.plot(size_dict[col], poly1d_fn(size_dict[col]), '-', color=stimulus_colors[col_ind], alpha=.8, linewidth=5) #'--k'=black dashed line
    # plt.xscale('log')
    plt.xlabel('community size', fontsize=30)
    plt.ylabel('purity', fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
  plt.suptitle('{} average purity VS community size'.format(max_method), size=40)
  plt.tight_layout()
  image_name = './plots/Hcomm_mean_purity_size_col_{}_{}_{}fold.jpg'.format(max_method, measure, n)
  plt.savefig(image_name)
  # plt.show()

def plot_all_Hcomm_purity(G_dict, area_dict, measure, n, max_pos_reso=None, max_neg_reso=None, max_method='none'):
  ind = 1
  rows, cols = get_rowcol(G_dict)
  if max_pos_reso is None:
    max_pos_reso = np.ones((len(rows), len(cols)))
  if max_neg_reso is None:
    max_neg_reso = np.ones((len(rows), len(cols)))
  fig = plt.figure(figsize=(7, 4))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  all_purity = []
  for col_ind, col in enumerate(cols):
    print(col)
    data = {}
    for row_ind, row in enumerate(rows):
      G = G_dict[row][col].copy() if col in G_dict[row] else nx.DiGraph()
      comms = signed_louvain_communities(G, weight='weight', pos_resolution=max_pos_reso[row_ind, col_ind], neg_resolution=max_neg_reso[row_ind, col_ind])
      sizes = [len(comm) for comm in comms]
      # part = community.best_partition(G, weight='weight')
      # comms, sizes = np.unique(list(part.values()), return_counts=True)
      for comm, size in zip(comms, sizes):
        c_regions = [area_dict[row][node] for node in comm]
        _, counts = np.unique(c_regions, return_counts=True)
        assert len(c_regions) == size == counts.sum()
        purity = counts.max() / size
        if size in data:
          data[size].append(purity)
        else:
          data[size] = [purity]
    
    c_size, c_purity = [k for k,v in sorted(data.items(), reverse=True)], [v for k,v in sorted(data.items(), reverse=True)]
    # c_purity = [x for xs in c_purity for x in xs]
    all_purity.append([np.mean(xs) for xs in c_purity])
  plt.boxplot(all_purity)
  plt.xticks(list(range(1, len(all_purity)+1)), cols, rotation=90)
  # plt.hist(data.flatten(), bins=12, density=True)
  # plt.axvline(x=np.nanmean(data), color='r', linestyle='--')
  # plt.xlabel('region')
  plt.ylabel('purity')
  plt.title('all {} Hamiltonian community purity'.format(max_method), size=18)
  plt.tight_layout()
  image_name = './plots/all_Hcomm_purity_{}_{}_{}fold.jpg'.format(max_method, measure, n)
  # plt.show()
  plt.savefig(image_name)
  return all_purity

def plot_weighted_Hcomm_purity(G_dict, area_dict, measure, n, max_pos_reso=None, max_neg_reso=None, max_method='none'):
  ind = 1
  rows, cols = get_rowcol(G_dict)
  if max_pos_reso is None:
    max_pos_reso = np.ones((len(rows), len(cols)))
  if max_neg_reso is None:
    max_neg_reso = np.ones((len(rows), len(cols)))
  fig = plt.figure(figsize=(7, 4))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  weighted_purity = []
  for col_ind, col in enumerate(cols):
    print(col)
    w_purity_col = []
    for row_ind, row in enumerate(rows):
      data = {}
      G = G_dict[row][col].copy() if col in G_dict[row] else nx.DiGraph()
      comms = signed_louvain_communities(G, weight='weight', pos_resolution=max_pos_reso[row_ind, col_ind], neg_resolution=max_neg_reso[row_ind, col_ind])
      sizes = [len(comm) for comm in comms]
      # part = community.best_partition(G, weight='weight')
      # comms, sizes = np.unique(list(part.values()), return_counts=True)
      for comm, size in zip(comms, sizes):
        c_regions = [area_dict[row][node] for node in comm]
        _, counts = np.unique(c_regions, return_counts=True)
        assert len(c_regions) == size == counts.sum()
        purity = counts.max() / size
        if size in data:
          data[size].append(purity)
        else:
          data[size] = [purity]
    
      c_size, c_purity = [k for k,v in data.items() if k >= 4], [v for k,v in data.items() if k >= 4]
      c_size = np.array(c_size) / sum(c_size)
      w_purity_col.append(sum([cs * np.mean(cp) for cs, cp in zip(c_size, c_purity)]))
    weighted_purity.append(w_purity_col)
  plt.boxplot(weighted_purity, showfliers=False)
  plt.xticks(list(range(1, len(weighted_purity)+1)), cols, rotation=90)
  # plt.hist(data.flatten(), bins=12, density=True)
  # plt.axvline(x=np.nanmean(data), color='r', linestyle='--')
  # plt.xlabel('region')
  plt.ylabel('purity')
  plt.title('weighted {} Hamiltonian community purity'.format(max_method), size=18)
  plt.tight_layout()
  image_name = './plots/weighted_Hcomm_purity_{}_{}_{}fold.jpg'.format(max_method, measure, n)
  # plt.show()
  plt.savefig(image_name)
  return weighted_purity

#############################################################################

def stat_modular_structure(pos_G_dict, measure, n, neg_G_dict=None, max_reso=None, max_method='none'):
  rows, cols = get_rowcol(pos_G_dict)
  if max_reso is None:
    max_reso = np.ones((len(rows), len(cols)))
  pos_w_num_comm, neg_w_num_comm, pos_w_modularity, neg_w_modularity, \
  pos_uw_num_comm, neg_uw_num_comm, pos_uw_modularity, neg_uw_modularity, \
  pos_w_num_lcomm, neg_uw_num_lcomm, neg_w_num_lcomm, pos_uw_num_lcomm, \
  pos_w_cov_lcomm, neg_uw_cov_lcomm, neg_w_cov_lcomm, pos_uw_cov_lcomm = [np.full([len(rows), len(cols)], np.nan) for _ in range(16)]
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      pos_G = pos_G_dict[row][col].copy() if col in pos_G_dict[row] else nx.DiGraph()
      if neg_G_dict is not None:
        comms = nx_comm.louvain_communities(pos_G, weight='weight')
        pos_w_num_comm[row_ind, col_ind] = len(comms)
        pos_w_modularity[row_ind, col_ind] = get_modularity(pos_G, weight='weight', comms=comms)
        count = np.array([len(comm) for comm in comms])
        pos_w_num_lcomm[row_ind, col_ind] = sum(count >= 4) # size at least 4 is considered (large) community
        pos_w_cov_lcomm[row_ind, col_ind] = count[count >=4].sum() / pos_G.number_of_nodes()
        assert pos_G.number_of_nodes() == count.sum()
        neg_G = neg_G_dict[row][col].copy() if col in neg_G_dict[row] else nx.DiGraph()
        comms = nx_comm.louvain_communities(neg_G, weight='weight')
        neg_w_num_comm[row_ind, col_ind] = len(comms)
        neg_w_modularity[row_ind, col_ind] = get_modularity(neg_G, weight='weight', comms=comms)
        count = np.array([len(comm) for comm in comms])
        neg_w_num_lcomm[row_ind, col_ind] = sum(count >= 4)
        neg_w_cov_lcomm[row_ind, col_ind] = count[count >=4].sum() / neg_G.number_of_nodes()
        assert neg_G.number_of_nodes() == count.sum()
        unweight = {(i, j):1 for i,j in neg_G.edges()}
        nx.set_edge_attributes(neg_G, unweight, 'weight')
        comms = nx_comm.louvain_communities(neg_G, weight='weight')
        neg_uw_num_comm[row_ind, col_ind] = len(comms)
        neg_uw_modularity[row_ind, col_ind] = get_modularity(neg_G, weight='weight', comms=comms)
        count = np.array([len(comm) for comm in comms])
        neg_uw_num_lcomm[row_ind, col_ind] = sum(count >= 4)
        neg_uw_cov_lcomm[row_ind, col_ind] = count[count >=4].sum() / neg_G.number_of_nodes()
        assert neg_G.number_of_nodes() == count.sum()
      unweight = {(i, j):1 for i,j in pos_G.edges()}
      nx.set_edge_attributes(pos_G, unweight, 'weight')
      comms = nx_comm.louvain_communities(pos_G, weight='weight', resolution=max_reso[row_ind, col_ind])
      pos_uw_num_comm[row_ind, col_ind] = len(comms)
      pos_uw_modularity[row_ind, col_ind] = get_modularity(pos_G, weight='weight', comms=comms)
      count = np.array([len(comm) for comm in comms])
      pos_uw_num_lcomm[row_ind, col_ind] = sum(count >= 4)
      pos_uw_cov_lcomm[row_ind, col_ind] = count[count >=4].sum() / pos_G.number_of_nodes()
      assert pos_G.number_of_nodes() == count.sum()
      
  if neg_G_dict is not None:
    metrics = {'positive weighted number of communities':pos_w_num_comm, 'positive weighted modularity':pos_w_modularity, 
  'negative weighted number of communities':neg_w_num_comm, 'negative weighted modularity':neg_w_modularity, 
  'positive unweighted number of communities':pos_uw_num_comm, 'positive unweighted modularity':pos_uw_modularity, 
  'negative unweighted number of communities':neg_uw_num_comm, 'negative unweighted modularity':neg_uw_modularity,
  'positive weighted num of large comm':pos_w_num_lcomm, 'positive unweighted num of large comm':pos_uw_num_lcomm,
  'negative weighted num of large comm':neg_w_num_lcomm, 'negative unweighted num of large comm':neg_uw_num_lcomm,
  'positive weighted coverage of large comm':pos_w_cov_lcomm, 'positive unweighted coverage of large comm':pos_uw_cov_lcomm,
  'negative weighted coverage of large comm':neg_w_cov_lcomm, 'negative unweighted coverage of large comm':neg_uw_cov_lcomm}
  else:
    metrics = {'total unweighted number of communities':pos_uw_num_comm, 'total unweighted modularity':pos_uw_modularity, 
    'total unweighted number of large communities':pos_uw_num_lcomm, 'total unweighted coverage of large comm':pos_uw_cov_lcomm}
  num_col = 4 if neg_G_dict is not None else 2
  num_row = int(len(metrics) / num_col)
  fig = plt.figure(figsize=(5*num_col, 5*num_row))
  for i, k in enumerate(metrics):
    plt.subplot(num_row, num_col, i+1)
    metric = metrics[k]
    for row_ind, row in enumerate(rows):
      mean = metric[row_ind, :]
      plt.plot(cols, mean, label=row, alpha=0.6)
    plt.gca().set_title(k, fontsize=14, rotation=0)
    plt.xticks(rotation=90)
    # plt.yscale('symlog')
    if i == len(metrics)-1:
      plt.legend()
    if i // num_col < num_row - 1:
      plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    plt.tight_layout()
  # plt.show()
  figname = './plots/stat_modular_pos_neg_{}_{}_{}fold.jpg' if neg_G_dict is not None else './plots/stat_modular_total_{}_{}_{}fold.jpg'
  plt.savefig(figname.format(max_method, measure, n))

def comm2histogram(comms, density=True, binning=False):
    if type(comms) == dict:
      dataseq=[v for k, v in comms.items()]
    else:
      dataseq=[item for sublist in [[i]*len(comm) for i, comm in enumerate(comms)] for item in sublist]
    if not binning:
      # dmax=max(dataseq)+1
      # data_seq = np.arange(0, dmax)
      # freq= [ 0 for d in range(dmax) ]
      # for d in data_seq:
      #   freq[d] += 1
      data_seq, freq = np.unique(dataseq, return_counts=True)
      if density:
        freq = freq / freq.sum()
    else:
      freq, bin_edges = np.histogram(dataseq, density=density)
      data_seq = (bin_edges[:-1] + bin_edges[1:]) / 2
    return data_seq, freq

def size_of_each_community(G_dict, sign, measure, n, max_reso=None, max_method='none'):
  rows, cols = get_rowcol(G_dict)
  if max_reso is None:
    max_reso = np.ones((len(rows), len(cols)))
  fig = plt.figure(figsize=(4*len(cols), 3*len(rows)))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  ind = 1
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      plt.subplot(len(rows), len(cols), ind)
      ind += 1
      if row_ind == 0:
        plt.gca().set_title(cols[col_ind], fontsize=20, rotation=0)
      if col_ind == 0:
        plt.gca().text(0, 0.5 * (bottom + top), rows[row_ind],
        horizontalalignment='left',
        verticalalignment='center',
        # rotation='vertical',
        transform=plt.gca().transAxes, fontsize=20, rotation=90)
      plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
      G = G_dict[row][col].copy() if col in G_dict[row] else nx.DiGraph()
      if nx.is_directed(G):
        G = G.to_undirected()
      
      unweight = {(i, j):1 for i,j in G.edges()}
      nx.set_edge_attributes(G, unweight, 'weight')
      comms = nx_comm.louvain_communities(G, weight='weight', resolution=max_reso[row_ind, col_ind])
      # part = community.best_partition(G, weight='weight')
      # metric = community.modularity(part, G, weight='weight')
      number, freq = comm2histogram(comms, density=False, binning=False)
      # plt.plot(size, np.array(freq) / sum(freq),'go-', label='size', alpha=0.4)
      plt.plot(range(len(freq)), sorted(freq, reverse=True),'go-', label='number', alpha=0.4)
      # plt.legend(loc='upper right', fontsize=7)
      xlabel = 'index of community in descending order of size'
      plt.xlabel(xlabel)
      plt.ylabel('size')
      # plt.xscale('symlog')
      # plt.yscale('log')
  # plt.show()
  plt.suptitle('{} size of each community'.format(sign), size=25)
  plt.tight_layout(rect=[0, 0.03, 1, 0.98])
  image_name = './plots/size_of_each_community_{}_{}_{}_{}fold.jpg'.format(sign, max_method, measure, n)
  plt.savefig(image_name)

def distribution_community_size(comms_dict, measure, n, max_neg_reso=None, max_method='none'):
  rows, cols = get_rowcol(comms_dict)
  fig = plt.figure(figsize=(4*len(cols), 3*len(rows)))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  ind = 1
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      plt.subplot(len(rows), len(cols), ind)
      ind += 1
      if row_ind == 0:
        plt.gca().set_title(cols[col_ind], fontsize=20, rotation=0)
      if col_ind == 0:
        plt.gca().text(0, 0.5 * (bottom + top), rows[row_ind],
        horizontalalignment='left',
        verticalalignment='center',
        # rotation='vertical',
        transform=plt.gca().transAxes, fontsize=20, rotation=90)
      plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
      max_reso = max_neg_reso[row_ind][col_ind]
      comms_list = comms_dict[row][col][max_reso]
      joined_comms = []
      for comms in comms_list:
        joined_comms += comms
      _, freq = comm2histogram(joined_comms, density=False, binning=False)
      size, counts = np.unique(freq, return_counts=True)
      # plt.plot(size, np.array(freq) / sum(freq),'go-', label='size', alpha=0.4)
      plt.plot(size, counts / counts.sum(),'go-', label='size', alpha=0.4)
      # plt.legend(loc='upper right', fontsize=7)
      xlabel = 'size of community'
      plt.xlabel(xlabel)
      plt.ylabel('number of nodes')
      plt.xscale('symlog')
      plt.yscale('log')
  # plt.show()
  plt.suptitle('community size distribution', size=25)
  plt.tight_layout(rect=[0, 0.03, 1, 0.98])
  image_name = './plots/comm_distribution_size_{}_{}_{}fold.jpg'.format(max_method, measure, n)
  plt.savefig(image_name)

def distribution_community_size_mean(comms_dict, measure, n, max_neg_reso=None, max_method='none'):
  rows, cols = get_rowcol(comms_dict)
  fig = plt.figure(figsize=(4*len(cols), 2*len(cols)))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  ind = 1
  for col_ind, col in enumerate(cols):
    print(col)
    joined_comms = []
    for row_ind, row in enumerate(rows):
      max_reso = max_neg_reso[row_ind][col_ind]
      comms_list = comms_dict[row][col][max_reso]
      for comms in comms_list:
        joined_comms += comms
    _, freq = comm2histogram(joined_comms, density=False, binning=False)
    size, counts = np.unique(freq, return_counts=True)
    # plt.plot(size, np.array(freq) / sum(freq),'go-', label='size', alpha=0.4)
    plt.subplot(2, int(np.ceil(len(cols)/2)), ind)
    ind += 1
    plt.gca().set_title(cols[col_ind], fontsize=20, rotation=0)
    plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.plot(size, counts / counts.sum(),'go-', label='size', alpha=0.4)
    # plt.legend(loc='upper right', fontsize=7)
    xlabel = 'size of community'
    plt.xlabel(xlabel)
    plt.ylabel('number of nodes')
    plt.xscale('symlog')
    plt.yscale('log')
  # plt.show()
  plt.suptitle('mean community size distribution', size=25)
  plt.tight_layout(rect=[0, 0.03, 1, 0.98])
  image_name = './plots/comm_distribution_mean_size_{}_{}_{}fold.jpg'.format(max_method, measure, n)
  plt.savefig(image_name)

def random_graph_generator(input_G, num_rewire, algorithm, weight='weight', cc=False, Q=100):
  origin_G = input_G.copy()
  if cc:
    origin_G = get_lcc(origin_G)
    # largest_cc = max(nx.connected_components(origin_G), key=len)
    # origin_G = nx.subgraph(origin_G, largest_cc)
  # weights = np.squeeze(np.array(nx.adjacency_matrix(origin_G)[nx.adjacency_matrix(origin_G).nonzero()]))
  weights = list(nx.get_edge_attributes(origin_G, weight).values())
  np.random.shuffle(weights)
  # if nx.is_directed(origin_G):
  #   algorithm = 'directed_configuration_model'
  random_graphs = []
  for num in tqdm(range(num_rewire)):
    # print(G.number_of_nodes(), G.number_of_edges())
    if algorithm == 'Gnm':
      n, m = origin_G.number_of_nodes(), origin_G.number_of_edges()
      G = nx.gnm_random_graph(n, m, seed=None, directed=True)
    elif algorithm == 'configuration_model':
      degree_sequence = [d for n, d in origin_G.degree()]
      G = nx.configuration_model(degree_sequence)
      # remove parallel edges and self-loops
      G = nx.Graph(G)
      G.remove_edges_from(nx.selfloop_edges(G))
      # print(G.number_of_nodes(), G.number_of_edges())
    elif algorithm == 'directed_configuration_model':
      din = list(d for n, d in origin_G.in_degree())
      dout = list(d for n, d in origin_G.out_degree())
      G = nx.directed_configuration_model(din, dout)
      G = nx.DiGraph(G)
      G.remove_edges_from(nx.selfloop_edges(G))
    elif algorithm == 'double_edge_swap':
      # at least four nodes with edges
      G = origin_G.copy()
      degrees = dict(nx.degree(G))
      if len(np.nonzero(list(degrees.values()))[0]) >= 4:
        nx.double_edge_swap(G, nswap=Q*G.number_of_edges(), max_tries=1e75)
    elif algorithm == 'connected_double_edge_swap':
      G = origin_G.copy()
      swaps = nx.connected_double_edge_swap(G, nswap=Q*G.number_of_edges(), _window_threshold=3)
      print('Number of successful swaps: {}'.format(swaps))
    elif algorithm == 'directed_double_edge_swap':
      # swap u->v, x->y to u->y, x->v
      G = origin_G.copy()
      nswap = Q*G.number_of_edges()
      max_tries = 1e75
      n_tries = 0
      swapcount = 0
      keys, out_degrees = zip(*G.out_degree())  # keys, degree
      cdf = nx.utils.cumulative_distribution(out_degrees)  # cdf of degree
      discrete_sequence = nx.utils.discrete_sequence
      while swapcount < nswap:
        #        if random.random() < 0.5: continue # trick to avoid periodicities?
        # pick two random edges without creating edge list
        # choose source node indices from discrete distribution
        (ui, xi) = discrete_sequence(2, cdistribution=cdf, seed=None)
        if ui == xi:
          continue  # same source, skip
        u = keys[ui]  # convert index to label
        x = keys[xi]
        # choose target uniformly from neighbors
        v = np.random.choice(list(G[u]))
        y = np.random.choice(list(G[x]))
        if (v == y) or (u == y) or (x == v):
          continue  # same target or self loop, skip
        if (y not in G[u]) and (v not in G[x]):  # don't create existed edges
          G.add_edge(u, y)
          G.add_edge(x, v)
          G.remove_edge(u, v)
          G.remove_edge(x, y)
          swapcount += 1
        if n_tries >= max_tries:
          e = (
            f"Maximum number of swap attempts ({n_tries}) exceeded "
            f"before desired swaps achieved ({nswap})."
          )
          raise nx.NetworkXAlgorithmError(e)
        n_tries += 1
    # add link weights
    for ind, e in enumerate(G.edges()):
      G[e[0]][e[1]][weight] = weights[ind]
    random_graphs.append(G)
  return random_graphs

def get_modularity(G, weight='weight', resolution=1, comms=None):
  if comms == None:
    comms = nx_comm.louvain_communities(G, weight=weight, resolution=resolution)
  return nx_comm.modularity(G, comms, weight=weight, resolution=resolution)
  # part = community.best_partition(G, weight=weight, resolution=resolution)
  # return community.modularity(part, G, weight=weight)

def get_random_modularity(G, num_rewire, algorithm, weight='weight', resolution=1):
  random_modularity = np.zeros(num_rewire)
  random_Gs = random_graph_generator(G, num_rewire, algorithm=algorithm)
  for random_ind, random_G in enumerate(random_Gs):
    random_modularity[random_ind] = get_modularity(random_G, weight=weight, resolution=resolution)
  return random_modularity

def modular_resolution(pos_G_dict, resolution_list, num_rewire, neg_G_dict=None):
  rows, cols = get_rowcol(pos_G_dict)
  pos_w_modularity, neg_w_modularity, \
  pos_uw_modularity, neg_uw_modularity = [np.full([len(rows), len(cols), len(resolution_list)], np.nan) for _ in range(4)]
  pos_gnm_w_modularity, neg_gnm_w_modularity, pos_swap_w_modularity, neg_swap_w_modularity, \
  pos_gnm_uw_modularity, neg_gnm_uw_modularity, pos_swap_uw_modularity, neg_swap_uw_modularity = [np.full([len(rows), len(cols), len(resolution_list), num_rewire], np.nan) for _ in range(8)]
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      print(col)
      pos_G = pos_G_dict[row][col].copy() if col in pos_G_dict[row] else nx.DiGraph()
      if nx.is_directed(pos_G):
        pos_G = pos_G.to_undirected()
      for resolution_ind, resolution in enumerate(resolution_list):
        if neg_G_dict is not None:
          pos_w_modularity[row_ind, col_ind, resolution_ind] = get_modularity(pos_G, weight='weight', resolution=resolution)
          pos_gnm_w_modularity[row_ind, col_ind, resolution_ind] = get_random_modularity(pos_G, num_rewire, algorithm='Gnm', weight='weight', resolution=resolution)
          pos_swap_w_modularity[row_ind, col_ind, resolution_ind] = get_random_modularity(pos_G, num_rewire, algorithm='double_edge_swap', weight='weight', resolution=resolution)
          
          neg_G = neg_G_dict[row][col].copy() if col in neg_G_dict[row] else nx.DiGraph()
          if nx.is_directed(neg_G):
            neg_G = neg_G.to_undirected()
          neg_w_modularity[row_ind, col_ind, resolution_ind] = get_modularity(neg_G, weight='weight', resolution=resolution)
          neg_gnm_w_modularity[row_ind, col_ind, resolution_ind] = get_random_modularity(neg_G, num_rewire, algorithm='Gnm', weight='weight', resolution=resolution)
          neg_swap_w_modularity[row_ind, col_ind, resolution_ind] = get_random_modularity(neg_G, num_rewire, algorithm='double_edge_swap', weight='weight', resolution=resolution)

          unweight = {(i, j):1 for i,j in neg_G.edges()}
          nx.set_edge_attributes(neg_G, unweight, 'weight')
          neg_uw_modularity[row_ind, col_ind, resolution_ind] = get_modularity(neg_G, weight='weight', resolution=resolution)
          neg_gnm_uw_modularity[row_ind, col_ind, resolution_ind] = get_random_modularity(neg_G, num_rewire, algorithm='Gnm', weight='weight', resolution=resolution)
          neg_swap_uw_modularity[row_ind, col_ind, resolution_ind] = get_random_modularity(neg_G, num_rewire, algorithm='double_edge_swap', weight='weight', resolution=resolution)
      
        unweight = {(i, j):1 for i,j in pos_G.edges()}
        nx.set_edge_attributes(pos_G, unweight, 'weight')
        pos_uw_modularity[row_ind, col_ind, resolution_ind] = get_modularity(pos_G, weight='weight', resolution=resolution)
        pos_gnm_uw_modularity[row_ind, col_ind, resolution_ind] = get_random_modularity(pos_G, num_rewire, algorithm='Gnm', weight='weight', resolution=resolution)
        pos_swap_uw_modularity[row_ind, col_ind, resolution_ind] = get_random_modularity(pos_G, num_rewire, algorithm='double_edge_swap', weight='weight', resolution=resolution)
      
  if neg_G_dict is not None:
    metrics = {'positive weighted modularity':pos_w_modularity, 'negative weighted modularity':neg_w_modularity, 
  'positive unweighted modularity':pos_uw_modularity, 'negative unweighted modularity':neg_uw_modularity}
  else:
    metrics = {'total unweighted modularity':pos_uw_modularity}
  random_metrics = {'positive Gnm weighted modularity':pos_gnm_w_modularity, 'negative Gnm weighted modularity':neg_gnm_w_modularity, \
    'positive swap weighted modularity':pos_swap_w_modularity, 'negative swap weighted modularity':neg_swap_w_modularity, \
    'positive Gnm unweighted modularity':pos_gnm_uw_modularity, 'negative Gnm unweighted modularity':neg_gnm_uw_modularity, \
    'positive swap unweighted modularity':pos_swap_uw_modularity, 'negative swap unweighted modularity':neg_swap_uw_modularity}
  return metrics, random_metrics

def plot_modularity_resolution_rand(rows, cols, resolution_list, metrics, random_metrics, measure, n): 
  num_row, num_col = len(rows), len(cols)
  for i, k in enumerate(metrics):
    fig = plt.figure(figsize=(5*num_col, 5*num_row))
    ind = 1
    for row_ind, row in enumerate(rows):
      print(row)
      for col_ind, col in enumerate(cols):
        plt.subplot(num_row, num_col, ind)
        ind += 1
        plt.plot(resolution_list, metrics[k][row_ind, col_ind], label=r'$Q$', alpha=0.6)
        metric_swap = random_metrics[k.replace('total', 'positive').replace('tive ', 'tive swap ')][row_ind, col_ind].mean(-1)
        metric_gnm = random_metrics[k.replace('total', 'positive').replace('tive ', 'tive Gnm ')][row_ind, col_ind].mean(-1)
        plt.plot(resolution_list, metric_swap, color='r', label=r'$Q_{swap}$', alpha=0.6)
        plt.plot(resolution_list, metric_gnm, color='g', label=r'$Q_{Gnm}$', alpha=0.6)
        plt.plot(resolution_list, metrics[k][row_ind, col_ind] - metric_swap, 'r--', label=r'$Q-Q_{swap}$', alpha=0.8)
        plt.plot(resolution_list, metrics[k][row_ind, col_ind] - metric_gnm, 'g--', label=r'$Q-Q_{Gnm}$', alpha=0.8)
        # plt.gca().set_title(k, fontsize=14, rotation=0)
        plt.xticks(rotation=0)
        
        # plt.yscale('symlog')
        if ind == num_row*num_col+1:
          plt.legend(fontsize=20)
        if row_ind == 0:
          plt.title(col, size=25)
        if row_ind < num_row -1 :
          plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affectedrandom_metrics
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
        else:
          plt.xlabel(r'$\gamma$', size=20)
    plt.suptitle(k, size=30)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    # plt.show()
    figname = './plots/{}_{}_{}fold.jpg'
    # plt.show()
    plt.savefig(figname.format(k.replace(' ', '_'), measure, n))

def comms_modularity_resolution(G_dict, resolution_list, num_rewire, cc=False):
  rows, cols = get_rowcol(G_dict)
  comms_dict = {}
  uw_modularity = np.full([len(rows), len(cols), len(resolution_list)], np.nan)
  gnm_uw_modularity, swap_uw_modularity = [np.full([len(rows), len(cols), len(resolution_list), num_rewire], np.nan) for _ in range(2)]
  for row_ind, row in enumerate(rows):
    print(row)
    comms_dict[row] = {}
    for col_ind, col in enumerate(cols):
      print(col)
      comms_dict[row][col] = {}
      G = G_dict[row][col].copy() if col in G_dict[row] else nx.DiGraph()
      if nx.is_directed(G):
        G = G.to_undirected()
      if cc:
        Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
        G = G.subgraph(Gcc[0])
      for resolution_ind, resolution in enumerate(resolution_list):      
        unweight = {(i, j):1 for i,j in G.edges()}
        nx.set_edge_attributes(G, unweight, 'weight')
        comms = nx_comm.louvain_communities(G.copy(), weight='weight', resolution=resolution)
        comms_dict[row][col][resolution] = comms
        uw_modularity[row_ind, col_ind, resolution_ind] = get_modularity(G.copy(), weight='weight', resolution=resolution, comms=comms)
        gnm_uw_modularity[row_ind, col_ind, resolution_ind] = get_random_modularity(G.copy(), num_rewire, algorithm='Gnm', weight='weight', resolution=resolution)
        swap_uw_modularity[row_ind, col_ind, resolution_ind] = get_random_modularity(G.copy(), num_rewire, algorithm='double_edge_swap', weight='weight', resolution=resolution)   
  metrics = {'total unweighted modularity':uw_modularity, 'total Gnm unweighted modularity':gnm_uw_modularity, 'total swap unweighted modularity':swap_uw_modularity}
  return comms_dict, metrics

def plot_modularity_resolution(rows, cols, resolution_list, metrics, measure, n): 
  num_row, num_col = len(rows), len(cols)
  fig = plt.figure(figsize=(5*num_col, 5*num_row))
  ind = 1
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      plt.subplot(num_row, num_col, ind)
      ind += 1
      metric = metrics['total unweighted modularity']
      plt.plot(resolution_list, metric[row_ind, col_ind], label=r'$Q$', alpha=0.6)
      metric_swap = metrics['total swap unweighted modularity'][row_ind, col_ind].mean(-1)
      metric_gnm = metrics['total Gnm unweighted modularity'][row_ind, col_ind].mean(-1)
      plt.plot(resolution_list, metric_swap, color='r', label=r'$Q_{swap}$', alpha=0.6)
      plt.plot(resolution_list, metric_gnm, color='g', label=r'$Q_{Gnm}$', alpha=0.6)
      plt.plot(resolution_list, metric[row_ind, col_ind] - metric_swap, 'r--', label=r'$Q-Q_{swap}$', alpha=0.8)
      plt.plot(resolution_list, metric[row_ind, col_ind] - metric_gnm, 'g--', label=r'$Q-Q_{Gnm}$', alpha=0.8)
      # plt.gca().set_title(k, fontsize=14, rotation=0)
      plt.xticks(rotation=0)
      
      # plt.yscale('symlog')
      if ind == num_row*num_col+1:
        plt.legend(fontsize=20)
      if row_ind == 0:
        plt.title(col, size=25)
      if row_ind < num_row -1 :
        plt.tick_params(
          axis='x',          # changes apply to the x-axis
          which='both',      # both major and minor ticks are affectedrandom_metrics
          bottom=False,      # ticks along the bottom edge are off
          top=False,         # ticks along the top edge are off
          labelbottom=False) # labels along the bottom edge are off
      else:
        plt.xlabel(r'$\gamma$', size=20)
  plt.suptitle('total unweighted modularity', size=30)
  plt.tight_layout(rect=[0, 0.03, 1, 0.98])
  # plt.show()
  figname = './plots/{}_{}_{}fold.jpg'
  # plt.show()
  plt.savefig(figname.format('total_unweighted_modularity', measure, n))

def get_max_resolution(rows, cols, resolution_list, metrics): 
  max_reso_gnm, max_reso_swap = np.zeros((len(rows), len(cols))), np.zeros((len(rows), len(cols)))
  uw_modularity, gnm_uw_modularity, swap_uw_modularity = metrics['total unweighted modularity'], metrics['total Gnm unweighted modularity'], metrics['total swap unweighted modularity']
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      metric_gnm = gnm_uw_modularity[row_ind, col_ind].mean(-1)
      metric_swap = swap_uw_modularity[row_ind, col_ind].mean(-1)
      max_reso_gnm[row_ind, col_ind] = resolution_list[np.argmax(uw_modularity[row_ind, col_ind] - metric_gnm)]
      max_reso_swap[row_ind, col_ind] = resolution_list[np.argmax(uw_modularity[row_ind, col_ind] - metric_swap)]
  return max_reso_gnm, max_reso_swap

def comm2label(comms):
  return [p for n, p in sorted({n:comms.index(comm) for comm in comms for n in comm}.items(), key=lambda x:x[0])]
  
def comm2partition(comms):
  return dict(sorted({n:comms.index(comm) for comm in comms for n in comm}.items(), key=lambda x:x[0]))

def variation_of_information(X, Y):
  n = float(sum([len(x) for x in X]))
  sigma = 0.0
  for x in X:
    p = len(x) / n
    for y in Y:
      q = len(y) / n
      r = len(set(x) & set(y)) / n
      if r > 0.0:
        sigma += r * (np.log2(r / p) + np.log2(r / q))
  return abs(sigma)

def normalized_variation_of_information(X, Y):
  return variation_of_information(X, Y) / np.log2(len([x for xs in X for x in xs]))

def plot_comm_diff_resolution(rows, cols, resolution_list, comms_dict, metrics, measure, n): 
  num_row, num_col = len(rows), len(cols)
  fig = plt.figure(figsize=(7*num_col, 5*num_row))
  ind = 1
  uw_modularity, gnm_uw_modularity, swap_uw_modularity = metrics['total unweighted modularity'], metrics['total Gnm unweighted modularity'], metrics['total swap unweighted modularity']
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      ax1 = plt.subplot(num_row, num_col, ind)
      metric_swap = swap_uw_modularity[row_ind, col_ind].mean(-1)
      metric_gnm = gnm_uw_modularity[row_ind, col_ind].mean(-1)
      max_reso_swap = resolution_list[np.argmax(uw_modularity[row_ind, col_ind] - metric_swap)]
      max_reso_gnm = resolution_list[np.argmax(uw_modularity[row_ind, col_ind] - metric_gnm)]
      comm_best_swap = comms_dict[row][col][max_reso_swap]
      comm_best_gnm = comms_dict[row][col][max_reso_gnm]
      VI_swap, VI_gnm, ARS_swap, ARS_gnm = [], [], [], []
      for reso in resolution_list:
        comms = comms_dict[row][col][reso]
        VI_swap.append(variation_of_information(comm_best_swap, comms))
        VI_gnm.append(variation_of_information(comm_best_gnm, comms))
        ARS_swap.append(adjusted_rand_score(comm2label(comm_best_swap), comm2label(comms)))
        ARS_gnm.append(adjusted_rand_score(comm2label(comm_best_gnm), comm2label(comms)))
      ax2 = ax1.twinx()
      lns1 = ax1.plot(resolution_list, VI_swap, color='tab:blue', label='variation of information (swap)', alpha=0.6)
      lns2 = ax1.plot(resolution_list, VI_gnm, color='tab:red', label=r'variation of information ($G_{nm}$)', alpha=0.6)
      lns3 = ax2.plot(resolution_list, ARS_swap, color='tab:cyan', label='adjusted rand score (swap)', alpha=0.6)
      lns4 = ax2.plot(resolution_list, ARS_gnm, color='tab:orange', label=r'adjusted rand score ($G_{nm}$)', alpha=0.6)
      lns5 = ax1.axvline(x=max_reso_swap, color='tab:blue', label=r'resolution with max $Q - Q_{swap}$', linestyle='--', alpha=0.6)
      lns6 = ax1.axvline(x=max_reso_gnm, color='tab:red', label=r'resolution with max $Q - Q_{G_{nm}}$', linestyle='--', alpha=0.6)
      # plt.gca().set_title(k, fontsize=14, rotation=0)
      plt.xticks(rotation=0)
      ax1.set_ylabel('variation of information')
      ax2.set_ylabel('adjusted rand score')
      
      # plt.yscale('symlog')
      if ind == 1:
        # print(type(lns1), type(lns2),type(lns3),type(lns4),type(lns5),type(lns6))
        # print(lns5.get_label())
        # print(lns1.get_label())
        lns = lns1+lns2+lns3+lns4+[lns5,lns6]
        
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc=0, fontsize=16)
        # plt.legend(fontsize=16)
      if row_ind == 0:
        plt.title(col, size=25)
      if row_ind < num_row -1 :
        plt.tick_params(
          axis='x',          # changes apply to the x-axis
          which='both',      # both major and minor ticks are affectedrandom_metrics
          bottom=False,      # ticks along the bottom edge are off
          top=False,         # ticks along the top edge are off
          labelbottom=False) # labels along the bottom edge are off
      else:
        plt.xlabel(r'$\gamma$', size=20)
      ind += 1
  # plt.suptitle('community partition difference', size=30)
  # plt.tight_layout(rect=[0, 0.03, 1, 0.98])
  plt.tight_layout()
  # plt.show()
  figname = './plots/comm_diff_resolution_{}_{}fold.jpg'
  # plt.show()
  plt.savefig(figname.format(measure, n))

def plot_region_size(G_dict, area_dict, regions, measure, n, sign):
  rows, cols = get_rowcol(G_dict)
  region_size = np.zeros((len(rows), len(cols), len(regions)))
  # fig = plt.figure(figsize=(5*num_col, 25))
  fig = plt.figure(figsize=(15, 13))
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      G = G_dict[row][col] if col in G_dict[row] else nx.DiGraph()
      nodes = list(G.nodes())
      for r_ind, r in enumerate(regions):
        r_nodes = [n for n in nodes if area_dict[row][n] == r]
        region_size[row_ind, col_ind, r_ind] = len(r_nodes)
  for col_ind, col in enumerate(cols):
    plt.subplot(3, 3, col_ind + 1)
    for row_ind, row in enumerate(rows):
      plt.plot(regions, region_size[row_ind, col_ind], label=row, alpha=0.6)
    plt.gca().set_title(col, fontsize=20, rotation=0)
    plt.xticks(rotation=0)
    # plt.yscale('symlog')
    if col_ind == 1:
      plt.legend()
    if col_ind // 3 < 2:
      plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
  plt.tight_layout()
  figname = './plots/region_size_{}_{}_{}fold.jpg'
  plt.savefig(figname.format(sign, measure, n))

def plot_region_size_box(G_dict, area_dict, regions, measure, n, sign):
  rows, cols = get_rowcol(G_dict)
  region_size = np.zeros((len(rows), len(cols), len(regions)))
  # fig = plt.figure(figsize=(5*num_col, 25))
  fig = plt.figure(figsize=(24, 3))
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      G = G_dict[row][col] if col in G_dict[row] else nx.DiGraph()
      nodes = list(G.nodes())
      for r_ind, r in enumerate(regions):
        r_nodes = [n for n in nodes if area_dict[row][n] == r]
        region_size[row_ind, col_ind, r_ind] = len(r_nodes)
  for col_ind, col in enumerate(cols):
    plt.subplot(1, len(cols), col_ind + 1)
    data = []
    # df = pd.DataFrame(columns=regions)
    for r_ind, r in enumerate(regions):
      data.append([region_size[row_ind, col_ind, r_ind] for row_ind in range(len(rows))])
      # r_degree = np.mean([nx.degree(G, n) for n in r_nodes])
      # data.append(r_degree)
    # data = data_dict[row][col]
    plt.boxplot(data)
    plt.xticks([1, 2, 3, 4, 5, 6], regions, rotation=0)
    # plt.hist(data.flatten(), bins=12, density=True)
    # plt.axvline(x=np.nanmean(data), color='r', linestyle='--')
    # plt.xlabel('region')
    plt.ylabel('size')
    plt.title(col, size=20)
  plt.tight_layout()
  figname = './plots/region_size_box_{}_{}_{}fold.jpg'
  plt.savefig(figname.format(sign, measure, n))

def plot_region_degree(G_dict, area_dict, regions, measure, n, sign):
  ind = 1
  rows, cols = get_rowcol(G_dict)
  fig = plt.figure(figsize=(4*len(cols), 3*len(rows)))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      plt.subplot(len(rows), len(cols), ind)
      if row_ind == 0:
        plt.gca().set_title(cols[col_ind], fontsize=20, rotation=0)
      if col_ind == 0:
        plt.gca().text(0, 0.5 * (bottom + top), rows[row_ind],
        horizontalalignment='left',
        verticalalignment='center',
        # rotation='vertical',
        transform=plt.gca().transAxes, fontsize=20, rotation=90)
      plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
      ind += 1
      G = G_dict[row][col]
      nodes = list(G.nodes())
      data = []
      # df = pd.DataFrame(columns=regions)
      for r in regions:
        r_nodes = [n for n in nodes if area_dict[row][n] == r]
        data.append([nx.degree(G, n) for n in r_nodes])
        # r_degree = np.mean([nx.degree(G, n) for n in r_nodes])
        # data.append(r_degree)
      # data = data_dict[row][col]
      plt.boxplot(data)
      plt.xticks([1, 2, 3, 4, 5, 6], regions, rotation=0)
      # plt.hist(data.flatten(), bins=12, density=True)
      # plt.axvline(x=np.nanmean(data), color='r', linestyle='--')
      # plt.xlabel('region')
      plt.ylabel('degree')
      
  plt.tight_layout()
  image_name = './plots/region_degree_{}_{}_{}fold.jpg'.format(sign, measure, n)
  # plt.show()
  plt.savefig(image_name)

def region_large_comm(pos_G_dict, area_dict, regions, measure, n, neg_G_dict=None, max_reso=None, max_method='none'):
  rows, cols = get_rowcol(pos_G_dict)
  if max_reso is None:
    max_reso = np.ones((len(rows), len(cols)))
  pos_w_frac_lcomm, neg_uw_frac_lcomm, neg_w_frac_lcomm, pos_uw_frac_lcomm = [np.full([len(rows), len(cols), len(regions)], np.nan) for _ in range(4)]
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      pos_G = pos_G_dict[row][col].copy() if col in pos_G_dict[row] else nx.DiGraph()
      if nx.is_directed(pos_G):
        pos_G = pos_G.to_undirected()
      if neg_G_dict is not None:
        comms = nx_comm.louvain_communities(pos_G, weight='weight', resolution=max_reso[row_ind, col_ind])
        lcomm_nodes = [list(comm) for comm in comms if len(comm) >= 4]
        lcomm_nodes = [item for sublist in lcomm_nodes for item in sublist]
        nodes = list(pos_G.nodes())
        for r_ind, r in enumerate(regions):
          r_nodes = [n for n in nodes if area_dict[row][n] == r]
          lcomm_r_nodes = [n for n in r_nodes if n in lcomm_nodes]
          pos_w_frac_lcomm[row_ind, col_ind, r_ind] = len(lcomm_r_nodes) / len(r_nodes) if len(r_nodes) else 0
        neg_G = neg_G_dict[row][col].copy() if col in neg_G_dict[row] else nx.DiGraph()
        if nx.is_directed(neg_G):
          neg_G = neg_G.to_undirected()
        comms = nx_comm.louvain_communities(neg_G, weight='weight', resolution=max_reso[row_ind, col_ind])
        lcomm_nodes = [list(comm) for comm in comms if len(comm) >= 4]
        lcomm_nodes = [item for sublist in lcomm_nodes for item in sublist]
        nodes = list(neg_G.nodes())
        for r_ind, r in enumerate(regions):
          r_nodes = [n for n in nodes if area_dict[row][n] == r]
          lcomm_r_nodes = [n for n in r_nodes if n in lcomm_nodes]
          neg_w_frac_lcomm[row_ind, col_ind, r_ind] = len(lcomm_r_nodes) / len(r_nodes) if len(r_nodes) else 0
        unweight = {(i, j):1 for i,j in neg_G.edges()}
        nx.set_edge_attributes(neg_G, unweight, 'weight')
        comms = nx_comm.louvain_communities(neg_G, weight='weight', resolution=max_reso[row_ind, col_ind])
        lcomm_nodes = [list(comm) for comm in comms if len(comm) >= 4]
        lcomm_nodes = [item for sublist in lcomm_nodes for item in sublist]
        nodes = list(neg_G.nodes())
        for r_ind, r in enumerate(regions):
          r_nodes = [n for n in nodes if area_dict[row][n] == r]
          lcomm_r_nodes = [n for n in r_nodes if n in lcomm_nodes]
          neg_uw_frac_lcomm[row_ind, col_ind, r_ind] = len(lcomm_r_nodes) / len(r_nodes) if len(r_nodes) else 0
      unweight = {(i, j):1 for i,j in pos_G.edges()}
      nx.set_edge_attributes(pos_G, unweight, 'weight')
      comms = nx_comm.louvain_communities(pos_G, weight='weight', resolution=max_reso[row_ind, col_ind])
      lcomm_nodes = [list(comm) for comm in comms if len(comm) >= 4]
      lcomm_nodes = [item for sublist in lcomm_nodes for item in sublist]
      nodes = list(pos_G.nodes())
      for r_ind, r in enumerate(regions):
        r_nodes = [n for n in nodes if area_dict[row][n] == r]
        lcomm_r_nodes = [n for n in r_nodes if n in lcomm_nodes]
        pos_uw_frac_lcomm[row_ind, col_ind, r_ind] = len(lcomm_r_nodes) / len(r_nodes) if len(r_nodes) else 0
      
  if neg_G_dict is not None:
    metrics = {'pos w fraction of regions in large communities':pos_w_frac_lcomm, 'pos uw fraction of regions in large communities':pos_uw_frac_lcomm, 
  'neg w fraction of regions in large communities':neg_w_frac_lcomm, 'neg uw fraction of regions in large communities':neg_uw_frac_lcomm}
  else:
    metrics = {'total uw fraction of regions in large communities':pos_uw_frac_lcomm}
  num_col = len(metrics)
  num_row = len(rows)
  fig = plt.figure(figsize=(6*num_col, 4*num_row))
  for row_ind, row in enumerate(rows):
    for i, k in enumerate(metrics):
    
      plt.subplot(num_row, num_col, row_ind * num_col + i+1)
      metric = metrics[k][row_ind]
      A = metric[:, 0]
      B = metric[:, 1]
      C = metric[:, 2]
      D = metric[:, 3]
      E = metric[:, 4]
      F = metric[:, 5]
      # Plot stacked bar chart
          
      plt.bar(cols, A, label=regions[0]) #, color='cyan',
      plt.bar(cols, B, bottom=A, label=regions[1]) #, color='green'
      plt.bar(cols, C, bottom=A+B, label=regions[2]) #, color='red'
      plt.bar(cols, D, bottom=A+B+C, label=regions[3]) #, color='yellow'
      plt.bar(cols, E, bottom=A+B+C+D, label=regions[4]) #, color='yellow'
      plt.bar(cols, F, bottom=A+B+C+D+E, label=regions[5]) #, color='yellow'
      plt.xticks(rotation=90)
      plt.ylabel('stacked percentage')
      # plt.xticks(rotation=90)
      # plt.yscale('symlog')
      if row_ind == 0:
        plt.title(k)
        plt.legend()
      if row_ind < num_row - 1:
        plt.tick_params(
          axis='x',          # changes apply to the x-axis
          which='both',      # both major and minor ticks are affected
          bottom=False,      # ticks along the bottom edge are off
          top=False,         # ticks along the top edge are off
          labelbottom=False) # labels along the bottom edge are off
      plt.tight_layout()
  # plt.suptitle(k, fontsize=14, rotation=0)
  # plt.show()
  sign = 'pos_neg' if neg_G_dict is not None else 'total'
  figname = './plots/region_large_comm_{}_{}_{}_{}fold.jpg'
  plt.savefig(figname.format(sign, max_method, measure, n))

def region_larg_comm_box(G_dict, area_dict, regions, measure, n, sign, weight=False, max_reso=None, max_method='none'):
  rows, cols = get_rowcol(G_dict)
  if max_reso is None:
    max_reso = np.ones((len(rows), len(cols)))
  frac_lcomm = pd.DataFrame(columns=['stimulus', 'fraction', 'region'])
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      G = G_dict[row][col].copy() if col in G_dict[row] else nx.DiGraph()
      if nx.is_directed(G):
        G = G.to_undirected()
      if not weight:
        name = '{}_uw'.format(sign)
        unweight = {(i, j):1 for i,j in G.edges()}
        nx.set_edge_attributes(G, unweight, 'weight')
      else:
        name = '{}_w'.format(sign)
      comms = nx_comm.louvain_communities(G, weight='weight', resolution=max_reso[row_ind, col_ind])
      lcomm_nodes = [list(comm) for comm in comms if len(comm) >= 4]
      lcomm_nodes = [item for sublist in lcomm_nodes for item in sublist]
      nodes = list(G.nodes())
      for r_ind, r in enumerate(regions):
        r_nodes = [n for n in nodes if area_dict[row][n] == r]
        lcomm_r_nodes = [n for n in r_nodes if n in lcomm_nodes]
        frac = len(lcomm_r_nodes) / len(r_nodes) if len(r_nodes) else 0
        frac_lcomm.loc[len(frac_lcomm), frac_lcomm.columns] = col, frac, r
  plt.figure(figsize=(17, 7))
  ax = sns.boxplot(x="stimulus", y="fraction", hue="region", data=frac_lcomm, palette="Set3")
  ax.set(xlabel=None)
  plt.title(name + ' percentage of region in large communities', size=15)
  plt.savefig('./plots/region_large_comm_box_{}_{}_{}_{}fold.jpg'.format(name, max_method, measure, n))
  # plt.show()

def plot_comm_size_purity(G_dict, area_dict, measure, n, sign, weight=False, max_reso=None, max_method='none'):
  ind = 1
  rows, cols = get_rowcol(G_dict)
  if max_reso is None:
    max_reso = np.ones((len(rows), len(cols)))
  fig = plt.figure(figsize=(23, 9))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  for col_ind, col in enumerate(cols):
    print(col)
    plt.subplot(4, 2, ind)
    plt.gca().set_title(cols[col_ind], fontsize=20, rotation=0)
    plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ind += 1
    data = {}
    for row_ind, row in enumerate(rows):
      G = G_dict[row][col].copy() if col in G_dict[row] else nx.DiGraph()
      if nx.is_directed(G):
        G = G.to_undirected()
      if not weight:
        name = '{}_uw'.format(sign)
        unweight = {(i, j):1 for i,j in G.edges()}
        nx.set_edge_attributes(G, unweight, 'weight')
      else:
        name = '{}_w'.format(sign)
      comms = nx_comm.louvain_communities(G, weight='weight', resolution=max_reso[row_ind, col_ind])
      sizes = [len(comm) for comm in comms]
      # part = community.best_partition(G, weight='weight')
      # comms, sizes = np.unique(list(part.values()), return_counts=True)
      for comm, size in zip(comms, sizes):
        c_regions = [area_dict[row][node] for node in comm]
        _, counts = np.unique(c_regions, return_counts=True)
        assert len(c_regions) == size == counts.sum()
        purity = counts.max() / size
        if size in data:
          data[size].append(purity)
        else:
          data[size] = [purity]
    
    c_size, c_purity = [k for k,v in sorted(data.items())], [v for k,v in sorted(data.items())]
    plt.boxplot(c_purity)
    plt.xticks(list(range(1, len(c_purity)+1)), c_size, rotation=0)
    # plt.hist(data.flatten(), bins=12, density=True)
    # plt.axvline(x=np.nanmean(data), color='r', linestyle='--')
    # plt.xlabel('region')
    plt.xlabel('size')
    plt.ylabel('purity')
  plt.suptitle(name + ' community purity VS size', size=30)
  plt.tight_layout()
  image_name = './plots/comm_size_purity_{}_{}_{}_{}fold.jpg'.format(name, max_method, measure, n)
  # plt.show()
  plt.savefig(image_name)

def plot_top_comm_purity(G_dict, num_top, area_dict, measure, n, sign, weight=False, max_reso=None, max_method='none'):
  ind = 1
  rows, cols = get_rowcol(G_dict)
  if max_reso is None:
    max_reso = np.ones((len(rows), len(cols)))
  fig = plt.figure(figsize=(7, 4))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  all_purity = []
  for col_ind, col in enumerate(cols):
    print(col)
    data = {}
    for row_ind, row in enumerate(rows):
      G = G_dict[row][col].copy() if col in G_dict[row] else nx.DiGraph()
      if nx.is_directed(G):
        G = G.to_undirected()
      if not weight:
        name = '{}_uw'.format(sign)
        unweight = {(i, j):1 for i,j in G.edges()}
        nx.set_edge_attributes(G, unweight, 'weight')
      else:
        name = '{}_w'.format(sign)
      comms = nx_comm.louvain_communities(G, weight='weight', resolution=max_reso[row_ind, col_ind])
      sizes = [len(comm) for comm in comms]
      # part = community.best_partition(G, weight='weight')
      # comms, sizes = np.unique(list(part.values()), return_counts=True)
      for comm, size in zip(comms, sizes):
        c_regions = [area_dict[row][node] for node in comm]
        _, counts = np.unique(c_regions, return_counts=True)
        assert len(c_regions) == size == counts.sum()
        purity = counts.max() / size
        if size in data:
          data[size].append(purity)
        else:
          data[size] = [purity]
    
    c_size, c_purity = [k for k,v in sorted(data.items(), reverse=True)][:num_top], [v for k,v in sorted(data.items(), reverse=True)][:num_top]
    # c_purity = [x for xs in c_purity for x in xs]
    all_purity.append([x for xs in c_purity for x in xs])
  plt.boxplot(all_purity)
  plt.xticks(list(range(1, len(all_purity)+1)), cols, rotation=90)
  # plt.hist(data.flatten(), bins=12, density=True)
  # plt.axvline(x=np.nanmean(data), color='r', linestyle='--')
  # plt.xlabel('region')
  plt.ylabel('purity')
  plt.title(name + ' top {} largest community purity'.format(num_top), size=18)
  plt.tight_layout()
  image_name = './plots/top_{}_comm_purity_{}_{}_{}_{}fold.jpg'.format(num_top, name, max_method, measure, n)
  # plt.show()
  plt.savefig(image_name)

def plot_all_comm_purity(G_dict, area_dict, measure, n, sign, weight=False, max_reso=None, max_method='none'):
  ind = 1
  rows, cols = get_rowcol(G_dict)
  if max_reso is None:
    max_reso = np.ones((len(rows), len(cols)))
  fig = plt.figure(figsize=(7, 4))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  all_purity = []
  for col_ind, col in enumerate(cols):
    print(col)
    data = {}
    for row_ind, row in enumerate(rows):
      G = G_dict[row][col].copy() if col in G_dict[row] else nx.DiGraph()
      if nx.is_directed(G):
        G = G.to_undirected()
      if not weight:
        name = '{}_uw'.format(sign)
        unweight = {(i, j):1 for i,j in G.edges()}
        nx.set_edge_attributes(G, unweight, 'weight')
      else:
        name = '{}_w'.format(sign)
      comms = nx_comm.louvain_communities(G, weight='weight', resolution=max_reso[row_ind, col_ind])
      sizes = [len(comm) for comm in comms]
      # part = community.best_partition(G, weight='weight')
      # comms, sizes = np.unique(list(part.values()), return_counts=True)
      for comm, size in zip(comms, sizes):
        c_regions = [area_dict[row][node] for node in comm]
        _, counts = np.unique(c_regions, return_counts=True)
        assert len(c_regions) == size == counts.sum()
        purity = counts.max() / size
        if size in data:
          data[size].append(purity)
        else:
          data[size] = [purity]
    
    c_size, c_purity = [k for k,v in sorted(data.items(), reverse=True)], [v for k,v in sorted(data.items(), reverse=True)]
    # c_purity = [x for xs in c_purity for x in xs]
    all_purity.append([np.mean(xs) for xs in c_purity])
  plt.boxplot(all_purity)
  plt.xticks(list(range(1, len(all_purity)+1)), cols, rotation=90)
  # plt.hist(data.flatten(), bins=12, density=True)
  # plt.axvline(x=np.nanmean(data), color='r', linestyle='--')
  # plt.xlabel('region')
  plt.ylabel('purity')
  plt.title(name + ' all community purity', size=18)
  plt.tight_layout()
  image_name = './plots/all_comm_purity_{}_{}_{}_{}fold.jpg'.format(name, max_method, measure, n)
  # plt.show()
  plt.savefig(image_name)
  return all_purity

def plot_top_comm_purity_kstest(G_dict, num_top, area_dict, measure, n, sign, weight=False, max_reso=None, max_method='none'):
  ind = 1
  rows, cols = get_rowcol(G_dict)
  if max_reso is None:
    max_reso = np.ones((len(rows), len(cols)))
  fig = plt.figure(figsize=(7, 5.5))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  all_purity = {}
  stimulus_pvalue = np.zeros((len(cols), len(cols)))
  stimulus_pvalue[:] = np.nan
  for col_ind, col in enumerate(cols):
    print(col)
    data = {}
    for row_ind, row in enumerate(rows):
      G = G_dict[row][col].copy() if col in G_dict[row] else nx.DiGraph()
      if nx.is_directed(G):
        G = G.to_undirected()
      if not weight:
        name = '{}_uw'.format(sign)
        unweight = {(i, j):1 for i,j in G.edges()}
        nx.set_edge_attributes(G, unweight, 'weight')
      else:
        name = '{}_w'.format(sign)
      comms = nx_comm.louvain_communities(G, weight='weight', resolution=max_reso[row_ind, col_ind])
      sizes = [len(comm) for comm in comms]
      # part = community.best_partition(G, weight='weight')
      # comms, sizes = np.unique(list(part.values()), return_counts=True)
      for comm, size in zip(comms, sizes):
        c_regions = [area_dict[row][node] for node in comm]
        _, counts = np.unique(c_regions, return_counts=True)
        assert len(c_regions) == size == counts.sum()
        purity = counts.max() / size
        if size in data:
          data[size].append(purity)
        else:
          data[size] = [purity]
    
    c_size, c_purity = [k for k,v in sorted(data.items(), reverse=True)][:num_top], [v for k,v in sorted(data.items(), reverse=True)][:num_top]
    # c_purity = [x for xs in c_purity for x in xs]
    all_purity[col] = [x for xs in c_purity for x in xs]
  for col_ind1, col1 in enumerate(cols):
    for col_ind2, col2 in enumerate(cols):
      if not col1 == col2:
        p_less = stats.ks_2samp(all_purity[col1], all_purity[col2], alternative='less')[1]
        p_greater = stats.ks_2samp(all_purity[col1], all_purity[col2], alternative='greater')[1]
        stimulus_pvalue[col_ind1, col_ind2] = min(p_less, p_greater)
        # print(np.mean(all_purity[col1]), np.mean(all_purity[col2]))
  # print(stimulus_pvalue)
  sns_plot = sns.heatmap(stimulus_pvalue.astype(float),cmap="RdBu",norm=colors.LogNorm(5.668934240362814e-06, 1))# cmap="YlGnBu" (0.000001, 1) for 0.01 confidence level, (0.0025, 1) for 0.05
  # sns_plot = sns.heatmap(region_connection.astype(float), vmin=0, cmap="YlGnBu")
  sns_plot.set_xticks(np.arange(len(cols))+0.5)
  sns_plot.set_xticklabels(cols, rotation=90)
  sns_plot.set_yticks(np.arange(len(cols))+0.5)
  sns_plot.set_yticklabels(cols, rotation=0)
  sns_plot.invert_yaxis()
  plt.title('p-value ' + name + ' top {} largest community purity'.format(num_top), size=18)
  plt.tight_layout()
  image_name = './plots/top_{}_comm_purity_{}_{}_{}_{}fold.jpg'.format(num_top, name, max_method, measure, n)
  # plt.show()
  plt.savefig(image_name)

def find_close_elements(l, t, e):
  return [i for i in l if abs(i-t)<=e]

def plot_similar_purity(G_dict, area_dict, error, measure, n, sign, weight=False, max_reso=None, max_method='none'):
  ind = 1
  rows, cols = get_rowcol(G_dict)
  if max_reso is None:
    max_reso = np.ones((len(rows), len(cols)))
  fig = plt.figure(figsize=(7, 4))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  all_purity = {}
  min_size = 10000
  for col_ind, col in enumerate(cols):
    print(col)
    data = {}
    for row_ind, row in enumerate(rows):
      G = G_dict[row][col].copy() if col in G_dict[row] else nx.DiGraph()
      if nx.is_directed(G):
        G = G.to_undirected()
      if not weight:
        name = '{}_uw'.format(sign)
        unweight = {(i, j):1 for i,j in G.edges()}
        nx.set_edge_attributes(G, unweight, 'weight')
      else:
        name = '{}_w'.format(sign)
      comms = nx_comm.louvain_communities(G, weight='weight', resolution=max_reso[row_ind, col_ind])
      sizes = [len(comm) for comm in comms]
      # part = community.best_partition(G, weight='weight')
      # comms, sizes = np.unique(list(part.values()), return_counts=True)
      for comm, size in zip(comms, sizes):
        c_regions = [area_dict[row][node] for node in comm]
        _, counts = np.unique(c_regions, return_counts=True)
        assert len(c_regions) == size == counts.sum()
        purity = counts.max() / size
        if size in data:
          data[size].append(purity)
        else:
          data[size] = [purity]
    
    # c_size, c_purity = [k for k,v in sorted(data.items(), reverse=True)], [v for k,v in sorted(data.items(), reverse=True)]
    # c_purity = [x for xs in c_purity for x in xs]
    all_purity[col] = dict(sorted(data.items(), reverse=True))
    min_size = max(data.keys()) if max(data.keys()) < min_size else min_size
  print('Min size is {}'.format(min_size))
  similar_size = [find_close_elements(list(all_purity[col].keys()), min_size, error) for col in cols]
  print(similar_size)
  similar_purity = [[i for s in size for i in all_purity[cols[similar_size.index(size)]][s]] for size in similar_size]
  plt.boxplot(similar_purity)
  plt.xticks(list(range(1, len(cols)+1)), cols, rotation=90)
  # plt.hist(data.flatten(), bins=12, density=True)
  # plt.axvline(x=np.nanmean(data), color='r', linestyle='--')
  # plt.xlabel('region')
  plt.ylabel('purity')
  plt.title(name + r' similar community size purity (${}\pm{}$)'.format(min_size, error), size=18)
  plt.tight_layout()
  image_name = './plots/similar_size_error{}_purity_{}_{}_{}_{}fold.jpg'.format(error, name, max_method, measure, n)
  # plt.show()
  plt.savefig(image_name)

def plot_size_lcc(G_dict, G_lcc_dict):
  rows, cols = get_rowcol(G_lcc_dict)
  plt.figure(figsize=(7,6))
  for row in rows:
    n_nodes, n_nodes_lcc = [], []
    for col in cols:
      # n_nodes.append(G_dict[row][col].number_of_nodes())
      n_nodes_lcc.append(G_lcc_dict[row][col].number_of_nodes())
    # plt.scatter(n_nodes, n_nodes_lcc, label=row)
    plt.plot(cols, n_nodes_lcc, label=row)
  # plt.xlabel('number of nodes', size=15)
  plt.xticks(rotation=90)
  plt.ylabel('number of nodes', size=15)
  plt.title('size of LCC', size=20)
  plt.legend()
  plt.savefig('./plots/size_of_lcc_{}_{}_fold.jpg'.format(measure, n))
  # plt.show()

def get_all_active_areas(G_dict, area_dict):
  rows, cols = get_rowcol(G_dict)
  active_areas = np.array([])
  for row in rows:
    print(row)
    for col in cols:
      G = G_dict[row][col] if col in G_dict[row] else nx.Graph()
      if G.number_of_nodes() >= 2 and G.number_of_edges() > 0:
        nodes = list(G.nodes())
        active_areas_G = np.unique(list({key: area_dict[row][key] for key in nodes}.values()))
        active_areas = np.union1d(active_areas, active_areas_G)
  return active_areas

def region_connection_heatmap(G_dict, sign, area_dict, regions, measure, n):
  rows, cols = get_rowcol(G_dict)
  scale = np.zeros(len(rows))
  region_connection = np.zeros((len(rows), len(cols), len(regions), len(regions)))
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      G = G_dict[row][col] if col in G_dict[row] else nx.Graph()
      if G.number_of_nodes() >= 2 and G.number_of_edges() > 0:
        nodes = list(G.nodes())
        active_areas = np.unique(list({key: area_dict[row][key] for key in nodes}.values()))
        # active_areas = [i for i in regions if i in active_areas]
        A = nx.adjacency_matrix(G)
        A = A.todense()
        A[A.nonzero()] = 1
        for region_ind_i, region_i in enumerate(regions):
          if region_i in active_areas:
            for region_ind_j, region_j in enumerate(regions):
              if region_j in active_areas:
                region_indices_i = np.array([k for k, v in area_dict[row].items() if v==region_i])
                region_indices_j = np.array([k for k, v in area_dict[row].items() if v==region_j])
                region_indices_i = np.array([nodes.index(i) for i in list(set(region_indices_i) & set(nodes))]) # some nodes not in cc are removed 
                region_indices_j = np.array([nodes.index(i) for i in list(set(region_indices_j) & set(nodes))])
                region_connection[row_ind, col_ind, region_ind_i, region_ind_j] = np.sum(A[region_indices_i[:, None], region_indices_j])
                assert np.sum(A[region_indices_i[:, None], region_indices_j]) == len(A[region_indices_i[:, None], region_indices_j].nonzero()[0])
      region_connection[row_ind, col_ind, :, :] = region_connection[row_ind, col_ind, :, :] / region_connection[row_ind, col_ind, :, :].sum()
    scale[row_ind] = region_connection[row_ind, :, :, :].max()
  ind = 1
  fig = plt.figure(figsize=(4*len(cols), 3*len(rows)))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      plt.subplot(len(rows), len(cols), ind)
      if row_ind == 0:
        plt.gca().set_title(cols[col_ind], fontsize=20, rotation=0)
      if col_ind == 0:
        plt.gca().text(0, 0.5 * (bottom + top), rows[row_ind],
        horizontalalignment='left',
        verticalalignment='center',
        # rotation='vertical',
        transform=plt.gca().transAxes, fontsize=20, rotation=90)
      plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
      ind += 1
      sns_plot = sns.heatmap(region_connection[row_ind, col_ind, :, :].astype(float), vmin=0, vmax=scale[row_ind],center=0,cmap="RdBu_r")# cmap="YlGnBu"
      # sns_plot = sns.heatmap(region_connection.astype(float), vmin=0, cmap="YlGnBu")
      sns_plot.set_xticks(np.arange(len(regions))+0.5)
      sns_plot.set_xticklabels(regions, rotation=90)
      sns_plot.set_yticks(np.arange(len(regions))+0.5)
      sns_plot.set_yticklabels(regions, rotation=0)
      sns_plot.invert_yaxis()
  plt.tight_layout()
  # plt.show()
  plt.savefig('./plots/region_connection_scale_{}_{}_{}fold.jpg'.format(sign, measure, n))

def colors_from_values(values, palette_name):
  # normalize the values to range [0, 1]
  normalized = (values - min(values)) / (max(values) - min(values))
  # convert to indices
  indices = np.round(normalized * (len(values) - 1)).astype(np.int32)
  # use the indices to get the colors
  palette = sns.color_palette(palette_name, len(values))
  return np.array(palette).take(indices, axis=0)

def region_connection_seperate_diagonal(G_dict, sign, area_dict, regions, measure, n, weight):
  rows, cols = get_rowcol(G_dict)
  each_shape_row, each_shape_col = len(regions)+2, len(regions)+1
  full_shape = (len(rows)*each_shape_row, len(cols)*each_shape_col)
  scale = np.zeros(len(rows))
  region_connection = np.zeros((len(rows), len(cols), len(regions), len(regions)))
  scale_mask = np.ones((len(regions), len(regions)))
  scale_mask[np.diag_indices_from(scale_mask)] = 0
  scale_mask = scale_mask[None,:,:]
  heat_cm = "RdBu_r" if sign=='pos' or 'total' else "RdBu"
  bar_cm = "Reds" if sign=='pos' or 'total' else "Blues"
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      G = G_dict[row][col] if col in G_dict[row] else nx.Graph()
      if G.number_of_nodes() >= 2 and G.number_of_edges() > 0:
        nodes = list(G.nodes())
        active_areas = np.unique(list({key: area_dict[row][key] for key in nodes}.values()))
        # active_areas = [i for i in regions if i in active_areas]
        A = nx.adjacency_matrix(G)
        A = A.todense()
        if not weight:
          A[A.nonzero()] = 1
        for region_ind_i, region_i in enumerate(regions):
          if region_i in active_areas:
            for region_ind_j, region_j in enumerate(regions):
              if region_j in active_areas:
                region_indices_i = np.array([k for k, v in area_dict[row].items() if v==region_i])
                region_indices_j = np.array([k for k, v in area_dict[row].items() if v==region_j])
                region_indices_i = np.array([nodes.index(i) for i in list(set(region_indices_i) & set(nodes))]) # some nodes not in cc are removed 
                region_indices_j = np.array([nodes.index(i) for i in list(set(region_indices_j) & set(nodes))])
                region_connection[row_ind, col_ind, region_ind_i, region_ind_j] = np.sum(A[region_indices_i[:, None], region_indices_j])
                # assert np.sum(A[region_indices_i[:, None], region_indices_j]) == len(A[region_indices_i[:, None], region_indices_j].nonzero()[0])
      region_connection[row_ind, col_ind, :, :] = region_connection[row_ind, col_ind, :, :] / region_connection[row_ind, col_ind, :, :].sum()
    scale[row_ind] = (region_connection[row_ind, :, :, :]*scale_mask).max()
  fig = plt.figure(figsize=(4*len(cols), 4*len(rows)))
  # left, width = .25, .5
  # bottom, height = .25, .5
  # right = left + width
  # top = bottom + height
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      # plt.subplot(len(rows), len(cols), ind)
      # if col_ind == 0:
      #   plt.gca().text(0, 0.5 * (bottom + top), rows[row_ind],
      #   horizontalalignment='left',
      #   verticalalignment='center',
      #   # rotation='vertical',
      #   transform=plt.gca().transAxes, fontsize=20, rotation=90)
      plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
      
      ax1 = plt.subplot2grid(full_shape, (row_ind*each_shape_row,col_ind*each_shape_col), colspan=len(regions), rowspan=len(regions))
      ax2 = plt.subplot2grid(full_shape, (row_ind*each_shape_row+len(regions),col_ind*each_shape_col), colspan=len(regions), rowspan=1)
      if row_ind == 0:
        ax1.set_title(cols[col_ind], fontsize=20)
      if col_ind == 0:
        plt.text(-2.4, 0.15, row, fontsize=20, rotation=90)
      mask = np.zeros((len(regions), len(regions)))
      mask[np.diag_indices_from(mask)] = True
      data = region_connection[row_ind, col_ind, :, :].astype(float)
      cbar=True if col_ind == len(cols) - 1 else False
      # xticklabels=True if row_ind == len(rows) - 1 else False
      xticklabels = False
      yticklabels=True if col_ind == 0 else False
      sns_plot = sns.heatmap(data, ax=ax1, mask=mask, xticklabels=xticklabels, yticklabels=yticklabels, cbar=cbar, vmin=0, vmax=scale[row_ind],center=0,cmap=heat_cm)# cmap="YlGnBu"
      # sns_plot = sns.heatmap(region_connection.astype(float), vmin=0, cmap="YlGnBu")
      # if row_ind == 0:
      #   sns_plot.set_xticks(np.arange(len(regions))+0.5)
      #   sns_plot.set_xticklabels(regions, rotation=90)
      if col_ind == 0:
        sns_plot.set_yticks(np.arange(len(regions))+0.5)
        sns_plot.set_yticklabels(regions, rotation=0)
      sns_plot.invert_yaxis()

      bar_plot = sns.barplot(x=regions, y=data.diagonal(), ax=ax2, palette=colors_from_values(data.diagonal(), bar_cm))# "Reds"
      bar_plot.tick_params(bottom=False)  # remove the ticks
      if row_ind != len(rows) - 1:
        bar_plot.set(xticklabels=[])
  plt.tight_layout()
  # plt.show()
  figname = './plots/region_connection_seperate_weighted_{}_{}_{}fold.jpg' if weight else './plots/region_connection_seperate_unweighted_{}_{}_{}fold.jpg'
  plt.savefig(figname.format(sign, measure, n))

def region_connection_delta_heatmap(G_dict, sign, area_dict, regions, measure, n, weight):
  rows, cols = get_rowcol(G_dict)
  cols.remove('spontaneous')
  scale_min = np.zeros(len(rows))
  scale_max = np.zeros(len(rows))
  region_connection_bl = np.zeros((len(rows), len(regions), len(regions)))
  region_connection = np.zeros((len(rows), len(cols), len(regions), len(regions)))
  for row_ind, row in enumerate(rows):
    print(row)
    G = G_dict[row]['spontaneous']
    if G.number_of_nodes() >= 2 and G.number_of_edges() > 0:
      nodes = list(G.nodes())
      active_areas = np.unique(list({key: area_dict[row][key] for key in nodes}.values()))
      A = nx.adjacency_matrix(G)
      A = A.todense()
      if not weight:
        A[A.nonzero()] = 1
      for region_ind_i, region_i in enumerate(regions):
        if region_i in active_areas:
          for region_ind_j, region_j in enumerate(regions):
            if region_j in active_areas:
              region_indices_i = np.array([k for k, v in area_dict[row].items() if v==region_i])
              region_indices_j = np.array([k for k, v in area_dict[row].items() if v==region_j])
              region_indices_i = np.array([nodes.index(i) for i in list(set(region_indices_i) & set(nodes))]) # some nodes not in cc are removed 
              region_indices_j = np.array([nodes.index(i) for i in list(set(region_indices_j) & set(nodes))])
              region_connection_bl[row_ind, region_ind_i, region_ind_j] = np.sum(A[region_indices_i[:, None], region_indices_j])
              # assert np.sum(A[region_indices_i[:, None], region_indices_j]) == len(A[region_indices_i[:, None], region_indices_j].nonzero()[0])
    for col_ind, col in enumerate(cols):
      G = G_dict[row][col] if col in G_dict[row] else nx.DiGraph()
      if G.number_of_nodes() >= 2 and G.number_of_edges() > 0:
        nodes = list(G.nodes())
        active_areas = np.unique(list({key: area_dict[row][key] for key in nodes}.values()))
        A = nx.adjacency_matrix(G)
        A = A.todense()
        if not weight:
          A[A.nonzero()] = 1
        for region_ind_i, region_i in enumerate(regions):
          if region_i in active_areas:
            for region_ind_j, region_j in enumerate(regions):
              if region_j in active_areas:
                region_indices_i = np.array([k for k, v in area_dict[row].items() if v==region_i])
                region_indices_j = np.array([k for k, v in area_dict[row].items() if v==region_j])
                region_indices_i = np.array([nodes.index(i) for i in list(set(region_indices_i) & set(nodes))]) # some nodes not in cc are removed 
                region_indices_j = np.array([nodes.index(i) for i in list(set(region_indices_j) & set(nodes))])
                region_connection[row_ind, col_ind, region_ind_i, region_ind_j] = np.sum(A[region_indices_i[:, None], region_indices_j])
                # assert np.sum(A[region_indices_i[:, None], region_indices_j]) == len(A[region_indices_i[:, None], region_indices_j].nonzero()[0])
    scale_min[row_ind] = ((region_connection[row_ind, :, :, :]-region_connection_bl[row_ind][None, :, :])/region_connection_bl[row_ind].sum()).min()
    scale_max[row_ind] = ((region_connection[row_ind, :, :, :]-region_connection_bl[row_ind][None, :, :])/region_connection_bl[row_ind].sum()).max()
  ind = 1
  fig = plt.figure(figsize=(4*len(cols), 3*len(rows)))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  for row_ind, row in enumerate(rows):
    for col_ind, col in enumerate(cols):
      plt.subplot(len(rows), len(cols), ind)
      if row_ind == 0:
        plt.gca().set_title(cols[col_ind], fontsize=20, rotation=0)
      if col_ind == 0:
        plt.gca().text(0, 0.5 * (bottom + top), rows[row_ind],
        horizontalalignment='left',
        verticalalignment='center',
        # rotation='vertical',
        transform=plt.gca().transAxes, fontsize=20, rotation=90)
      plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
      ind += 1
      sns_plot = sns.heatmap((region_connection[row_ind, col_ind, :, :]-region_connection_bl[row_ind])/region_connection_bl[row_ind].sum(), vmin=scale_min[row_ind], vmax=scale_max[row_ind],center=0,cmap="RdBu_r") #  cmap="YlGnBu"
      # sns_plot = sns.heatmap((region_connection-region_connection_bl)/region_connection_bl.sum(), cmap="YlGnBu")
      sns_plot.set_xticks(np.arange(len(regions))+0.5)
      sns_plot.set_xticklabels(regions, rotation=90)
      sns_plot.set_yticks(np.arange(len(regions))+0.5)
      sns_plot.set_yticklabels(regions, rotation=0)
      sns_plot.invert_yaxis()
  plt.tight_layout()
  figname = './plots/region_connection_delta_weighted_{}_{}_{}fold.jpg'.format(sign, measure, n) if weight else './plots/region_connection_delta_{}_{}_{}.jpg'.format(sign, measure, n)
  plt.savefig(figname)
  # plt.savefig('./plots/region_connection_delta_scale_{}_{}.pdf'.format(measure, num), transparent=True)

def plot_multi_graphs_color(G_dict, max_reso, offset_dict, sign, area_dict, active_area_dict, measure, n, cc=False):
  com = CommunityLayout()
  ind = 1
  rows, cols = get_rowcol(G_dict)
  G_sample = G_dict[rows[0]][cols[0]]
  dire = True if nx.is_directed(G_sample) else False
  fig = plt.figure(figsize=(9*len(cols), 6*len(rows)))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  for row_ind, row in enumerate(rows):
    print(row)
    active_area = active_area_dict[row]
    for col_ind, col in enumerate(cols):
      offset_mat = offset_dict[row][col]
      plt.subplot(len(rows), len(cols), ind)
      if row_ind == 0:
        plt.gca().set_title(cols[col_ind], fontsize=30, rotation=0)
      if col_ind == 0:
        plt.gca().text(0, 0.5 * (bottom + top), rows[row_ind],
        horizontalalignment='left',
        verticalalignment='center',
        # rotation='vertical',
        transform=plt.gca().transAxes, fontsize=30, rotation=90)
      plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
      ind += 1
      G = G_dict[row][col].copy() if col in G_dict[row] else nx.DiGraph()
      nx.set_node_attributes(G, area_dict[row], "area")
      if G.number_of_nodes() > 2 and G.number_of_edges() > 0:
        if cc:
          if nx.is_directed(G):
            Gcc = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
            G = G.subgraph(Gcc[0])
          else:
            Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
            G = G.subgraph(Gcc[0])
        node_idx = sorted(active_area_dict[row].keys())
        reverse_mapping = {node_idx[i]:i for i in range(len(node_idx))}
        G = nx.relabel_nodes(G, reverse_mapping)
        try:
          edges, weights = zip(*nx.get_edge_attributes(G,'weight').items())
          weights = np.abs(weights)
        except:
          edges = nx.edges(G)
          weights = np.ones(len(edges))
        degrees = dict(G.degree)
        
        # use offset as edge weight (color)
        weights = [offset_mat[edge[0], edge[1]] for edge in edges]
        norm = mpl.colors.Normalize(vmin=-1, vmax=11)
        m= cm.ScalarMappable(norm=norm, cmap=cm.Greens)
        edge_colors = [m.to_rgba(w) for w in weights]
        # weights = [offset_mat[reverse_mapping[edge[0]], reverse_mapping[edge[1]]] for edge in edges]
        try:
          if nx.is_directed(G):
            G = G.to_undirected()
          # weight cannot be negative
          if sum([n<0 for n in nx.get_edge_attributes(G, "weight").values()]):
            print('Edge weight cannot be negative for weighted modularity, setting to unweighted...')
            unweight = {(i, j):1 for i,j in G.edges()}
            nx.set_edge_attributes(G, unweight, 'weight')
          
          unweight = {(i, j):1 for i,j in G.edges()}
          nx.set_edge_attributes(G, unweight, 'weight')
          comms = nx_comm.louvain_communities(G, weight='weight', resolution=max_reso[row_ind, col_ind])
          pos = com.get_community_layout(G, comm2partition(comms))
          metric = nx_comm.modularity(G, comms, weight='weight', resolution=max_reso[row_ind, col_ind])
          # partition = community.best_partition(G, weight='weight')
          # pos = com.get_community_layout(G, partition)
          # metric = community.modularity(partition, G, weight='weight')
          print('Modularity: {}'.format(metric))
        except:
          print('Community detection unsuccessful!')
          pos = nx.spring_layout(G)
        
        areas = [G.nodes[n]['area'] for n in G.nodes()]
        areas_uniq = list(set(areas))
        colors = [customPalette[areas_uniq.index(area)] for area in areas]
        # pos = nx.spring_layout(G, k=0.8, iterations=50) # make nodes as seperate as possible
        nx.draw_networkx_edges(G, pos, arrows=dire, edgelist=edges, edge_color=edge_colors, width=3.0, alpha=0.9) # , edge_cmap=plt.cm.Greens
        # nx.draw_networkx_nodes(G, pos, nodelist=degrees.keys(), node_size=[np.log(v + 2) * 20 for v in degrees.values()], 
        nx.draw_networkx_nodes(G, pos, nodelist=degrees.keys(), node_size=[5 * v for v in degrees.values()], 
        node_color=colors, alpha=0.4)
      if row_ind == col_ind == 0:
        areas = [G.nodes[n]['area'] for n in G.nodes()]
        areas_uniq = list(set(areas))
        for index, a in enumerate(areas_uniq):
          plt.scatter([],[], c=customPalette[index], label=a, s=30)
        legend = plt.legend(loc='upper center', fontsize=20)
  for handle in legend.legendHandles:
    handle.set_sizes([60.0])
  plt.tight_layout()
  image_name = './plots/graphs_region_color_cc_{}_{}_{}fold.jpg'.format(sign, measure, n) if cc else './plots/graphs_region_color_{}_{}_{}fold.jpg'.format(sign, measure, n)
  plt.savefig(image_name)
  # plt.savefig(image_name.replace('.jpg', '.pdf'), transparent=True)
  # plt.show()

# intra/inter region offset/duration
def plot_intra_inter_data(data_dict, G_dict, sign, name, density, active_area_dict, measure, n):
  rows, cols = get_rowcol(data_dict)
  num_row, num_col = len(rows), len(cols)
  fig = plt.figure(figsize=(5*num_col, 3*num_row))
  for row_ind, row in enumerate(rows):
    print(row)
    active_area = active_area_dict[row]
    for col_ind, col in enumerate(cols):
      intra_data, inter_data = [], []
      mat, G = data_dict[row][col].copy(), G_dict[row][col].copy()
      nodes = sorted(list(G.nodes()))
      for i, j in zip(*np.where(~np.isnan(mat))):
        if active_area[nodes[i]] == active_area[nodes[j]]:
          intra_data.append(mat[i, j])
        else:
          inter_data.append(mat[i, j])
      ax = plt.subplot(num_row, num_col, row_ind*num_col+col_ind+1)
      plt.hist(intra_data, 13, density=density, facecolor='g', alpha=0.3, label='intra-region {}'.format(name))
      plt.hist(inter_data, 13, density=density, facecolor='b', alpha=0.3, label='inter-region {}'.format(name))
      plt.axvline(x=np.nanmean(intra_data), color='g', linestyle='--', alpha=0.3)
      plt.axvline(x=np.nanmean(inter_data), color='b', linestyle='--', alpha=0.3)
      if density:
        plt.ylabel('Probability')
      else:
        plt.ylabel('Count')
      if row_ind == 0:
        plt.title(col, size=25)
      if row_ind == len(rows)-1:
        plt.xlabel(r'time lag $\tau$')
  plt.legend()
  plt.xticks(rotation=90)
  plt.tight_layout()
  # plt.show()
  figname = './plots/intra_inter_density_{}_{}_{}_{}fold.jpg' if density else './plots/intra_inter_count_{}_{}_{}_{}fold.jpg'
  plt.savefig(figname.format(name, sign, measure, n))

############ violin plot of intra/inter offset/duration
def plot_intra_inter_data_violin(data_dict, G_dict, sign, name, split, active_area_dict, measure, n):
  rows, cols = get_rowcol(data_dict)
  fig = plt.figure(figsize=(12, 6))
  df = pd.DataFrame()
  for col_ind, col in enumerate(cols):
    print(col)
    intra_data, inter_data = [], []
    for row_ind, row in enumerate(rows):
      active_area = active_area_dict[row]
      mat, G = data_dict[row][col].copy(), G_dict[row][col].copy()
      nodes = sorted(list(G.nodes()))
      for i, j in zip(*np.where(~np.isnan(mat))):
        if active_area[nodes[i]] == active_area[nodes[j]]:
          intra_data.append(mat[i, j])
        else:
          inter_data.append(mat[i, j])
    df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(intra_data)[:,None], np.array(['intra region'] * len(intra_data))[:,None], np.array([col] * len(intra_data))[:,None]), 1), columns=[name, 'type', 'stimulus']), pd.DataFrame(np.concatenate((np.array(inter_data)[:,None], np.array(['inter region'] * len(inter_data))[:,None], np.array([col] * len(inter_data))[:,None]), 1), columns=[name, 'type', 'stimulus'])], ignore_index=True)
  df[name] = pd.to_numeric(df[name])
  ax = sns.violinplot(x='stimulus', y=name, hue="type", data=df, palette="muted", split=split)
  # plt.legend()
  plt.xticks(rotation=90)
  # plt.tight_layout()
  # plt.show()
  figname = './plots/intra_inter_violin_split_{}_{}_{}_{}fold.jpg' if split else './plots/intra_inter_violin_{}_{}_{}_{}fold.jpg'
  plt.savefig(figname.format(name, sign, measure, n))

############ plot region distribution of offset/duration
def region_data_heatmap(data_dict, G_dict, sign, name, active_area_dict, regions, measure, n):
  rows, cols = get_rowcol(G_dict)
  scale = np.zeros(len(rows))
  region_data = np.zeros((len(rows), len(cols), len(regions), len(regions)))
  region_data[:] = np.nan
  for row_ind, row in enumerate(rows):
    print(row)
    active_area = active_area_dict[row]
    for col_ind, col in enumerate(cols):
      region_dict = {key: defaultdict(lambda: []) for key in regions}
      mat, G = data_dict[row][col].copy(), G_dict[row][col].copy()
      nodes = sorted(list(G.nodes()))
      for i, j in zip(*np.where(~np.isnan(mat))):
        region_dict[active_area[nodes[i]]][active_area[nodes[j]]].append(mat[i, j])
      for region_ind_i, region_i in enumerate(regions):
        for region_ind_j, region_j in enumerate(regions):
          region_data[row_ind, col_ind, region_ind_i, region_ind_j] = np.mean(region_dict[region_i][region_j]) if len(region_dict[region_i][region_j]) else np.nan
      
    scale[row_ind] = np.nanmax(region_data[row_ind, :, :, :])
  ind = 1
  fig = plt.figure(figsize=(4*len(cols), 3*len(rows)))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      plt.subplot(len(rows), len(cols), ind)
      if row_ind == 0:
        plt.gca().set_title(cols[col_ind], fontsize=20, rotation=0)
      if col_ind == 0:
        plt.gca().text(0, 0.5 * (bottom + top), rows[row_ind],
        horizontalalignment='left',
        verticalalignment='center',
        # rotation='vertical',
        transform=plt.gca().transAxes, fontsize=20, rotation=90)
      plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
      ind += 1
      sns_plot = sns.heatmap(region_data[row_ind, col_ind, :, :].astype(float), vmin=0, vmax=scale[row_ind],cmap="summer_r")# cmap="YlGnBu"
      # sns_plot = sns.heatmap(region_data.astype(float), vmin=0, cmap="YlGnBu")
      sns_plot.set_xticks(np.arange(len(regions))+0.5)
      sns_plot.set_xticklabels(regions, rotation=90)
      sns_plot.set_yticks(np.arange(len(regions))+0.5)
      sns_plot.set_yticklabels(regions, rotation=0)
      sns_plot.invert_yaxis()
  plt.tight_layout()
  # plt.show()
  plt.savefig('./plots/region_{}_scale_{}_{}_{}fold.jpg'.format(name, sign, measure, n))

# intra/inter region offset/duration
def plot_violin_intra_inter_data(data_dict, G_dict, sign, name, density, active_area_dict, measure, n):
  rows, cols = get_rowcol(data_dict)
  num_row, num_col = len(rows), len(cols)
  fig = plt.figure(figsize=(5*num_col, 3*num_row))
  for row_ind, row in enumerate(rows):
    print(row)
    active_area = active_area_dict[row]
    for col_ind, col in enumerate(cols):
      intra_data, inter_data = [], []
      mat, G = data_dict[row][col].copy(), G_dict[row][col].copy()
      nodes = sorted(list(G.nodes()))
      for i, j in zip(*np.where(~np.isnan(mat))):
        if active_area[nodes[i]] == active_area[nodes[j]]:
          intra_data.append(mat[i, j])
        else:
          inter_data.append(mat[i, j])
      ax = plt.subplot(num_row, num_col, row_ind*num_col+col_ind+1)
      plt.hist(intra_data, 13, density=density, facecolor='g', alpha=0.3, label='intra-region {}'.format(name))
      plt.hist(inter_data, 13, density=density, facecolor='b', alpha=0.3, label='inter-region {}'.format(name))
      plt.axvline(x=np.nanmean(intra_data), color='g', linestyle='--', alpha=0.3)
      plt.axvline(x=np.nanmean(inter_data), color='b', linestyle='--', alpha=0.3)
      if density:
        plt.ylabel('Probability')
      else:
        plt.ylabel('Count')
      if row_ind == 0:
        plt.title(col, size=25)
      if row_ind == len(rows)-1:
        plt.xlabel(r'time lag $\tau$')
  plt.legend()
  plt.xticks(rotation=90)
  plt.tight_layout()
  # plt.show()
  figname = './plots/intra_inter_density_{}_{}_{}_{}fold.jpg' if density else './plots/intra_inter_count_{}_{}_{}_{}fold.jpg'
  plt.savefig(figname.format(name, sign, measure, n))

def func_powerlaw(x, m, c):
  return x**m * c

def degree_histogram_directed(G, type='in_degree', weight=None):
    nodes = G.nodes()
    if type == 'in_degree':
        in_degree = dict(G.in_degree(weight=weight))
        degseq=[in_degree.get(k,0) for k in nodes]
    elif type == 'out_degree':
        out_degree = dict(G.out_degree(weight=weight))
        degseq=[out_degree.get(k,0) for k in nodes]
    else:
        degseq=[v for k, v in G.degree()]
    if weight == None:
      dmax=max(degseq)+1
      deg_seq = np.arange(0, dmax)
      freq= [ 0 for d in range(dmax) ]
      for d in degseq:
          freq[d] += 1
    else:
      freq, bin_edges = np.histogram(degseq, density=True)
      deg_seq = (bin_edges[:-1] + bin_edges[1:]) / 2
    return deg_seq, freq

def plot_directed_multi_degree_distributions(G_dict, sign, measure, n, weight=None, cc=False):
  ind = 1
  rows, cols = get_rowcol(G_dict)
  fig = plt.figure(figsize=(4*len(cols), 3*len(rows)))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      plt.subplot(len(rows), len(cols), ind)
      if row_ind == 0:
        plt.gca().set_title(cols[col_ind], fontsize=20, rotation=0)
      if col_ind == 0:
        plt.gca().text(0, 0.5 * (bottom + top), rows[row_ind],
        horizontalalignment='left',
        verticalalignment='center',
        # rotation='vertical',
        transform=plt.gca().transAxes, fontsize=20, rotation=90)
      plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
      ind += 1
      G = G_dict[row][col] if col in G_dict[row] else nx.DiGraph()
      if G.number_of_nodes() > 2 and G.number_of_edges() > 0:
        if cc:
          if nx.is_directed(G):
            Gcc = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
            G = G.subgraph(Gcc[0])
          else:
            Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
            G = G.subgraph(Gcc[0])
        in_degrees, in_degree_freq = degree_histogram_directed(G, type='in_degree', weight=weight)
        out_degrees, out_degree_freq = degree_histogram_directed(G, type='out_degree', weight=weight)
        if weight == None:
          in_degrees, in_degree_freq = in_degrees[1:], in_degree_freq[1:]
          out_degrees, out_degree_freq = out_degrees[1:], out_degree_freq[1:]
        plt.plot(in_degrees, np.array(in_degree_freq) / sum(in_degree_freq),'go-', label='in-degree', alpha=0.4)
        plt.plot(out_degrees, np.array(out_degree_freq) / sum(out_degree_freq),'bo-', label='out-degree', alpha=0.4)
        plt.legend(loc='upper right', fontsize=7)
        xlabel = 'Weighted Degree' if weight is not None else 'Degree'
        plt.xlabel(xlabel)
        plt.ylabel('Frequency')
        plt.xscale('symlog')
        plt.yscale('log')
      
  plt.tight_layout()
  image_name = './plots/directed_degree_distribution_weighted_{}_{}_{}fold.jpg'.format(sign, measure, n) if weight is not None else './plots/directed_degree_distribution_unweighted_{}_{}_{}fold.jpg'.format(sign, measure, n)
  # plt.show()
  plt.savefig(image_name, dpi=300)
  # plt.savefig(image_name.replace('jpg', 'pdf'), transparent=True)

def seq2histogram(degseq, weight=None):
    if weight == None:
      dmax=max(degseq)+1
      deg_seq = np.arange(0, dmax)
      freq= [ 0 for d in range(dmax) ]
      for d in degseq:
          freq[d] += 1
    else:
      freq, bin_edges = np.histogram(degseq, density=True)
      deg_seq = (bin_edges[:-1] + bin_edges[1:]) / 2
    return deg_seq, freq

def plot_directed_degree_distributions(G_dict, measure, n, weight=None, cc=False):
  ind = 1
  rows, cols = get_rowcol(G_dict)
  fig = plt.figure(figsize=(4*len(cols), 2*len(cols)))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  ind = 1
  for col_ind, col in enumerate(cols):
    print(col)
    in_degree_seq, out_degree_seq = [], []
    for row_ind, row in enumerate(rows):
      G = G_dict[row][col] if col in G_dict[row] else nx.DiGraph()
      if cc:
        if nx.is_directed(G):
          Gcc = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
          G = G.subgraph(Gcc[0])
        else:
          Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
          G = G.subgraph(Gcc[0])
      nodes = G.nodes()
      in_degree = dict(G.in_degree(weight=weight))
      in_degseq=[in_degree.get(k,0) for k in nodes]
      out_degree = dict(G.out_degree(weight=weight))
      out_degseq=[out_degree.get(k,0) for k in nodes]
      in_degree_seq += in_degseq
      out_degree_seq += out_degseq
    in_degrees, in_degree_freq = seq2histogram(in_degree_seq, weight=weight)
    out_degrees, out_degree_freq = seq2histogram(out_degree_seq, weight=weight)
    if weight == None:
      in_degrees, in_degree_freq = in_degrees[1:], in_degree_freq[1:]
      out_degrees, out_degree_freq = out_degrees[1:], out_degree_freq[1:]
    
    plt.subplot(2, int(np.ceil(len(cols)/2)), ind)
    plt.gca().set_title(cols[col_ind], fontsize=20, rotation=0)
    plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ind += 1
    plt.plot(in_degrees, np.array(in_degree_freq) / sum(in_degree_freq),'go-', label='in-degree', alpha=0.4)
    plt.plot(out_degrees, np.array(out_degree_freq) / sum(out_degree_freq),'bo-', label='out-degree', alpha=0.4)
    plt.legend(loc='upper right', fontsize=7)
    xlabel = 'Weighted Degree' if weight is not None else 'Degree'
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.xscale('symlog')
    plt.yscale('log')
      
  plt.tight_layout()
  image_name = './plots/directed_degree_distribution_weighted_{}_{}fold.jpg' if weight is not None else './plots/directed_degree_distribution_unweighted_{}_{}fold.jpg'
  # plt.show()
  plt.savefig(image_name.format(measure, n), dpi=300)

def plot_heatmap_xcorr_FR(corr, bins):
  divnorm=colors.TwoSlopeNorm(vcenter=0.)
  # pcolormesh(your_data, cmap="coolwarm", norm=divnorm)
  fig, ax = plt.subplots(figsize=(7, 6))
  im = ax.imshow(corr, norm=divnorm, cmap="RdBu_r")
  ax.set_xticks(ticks=np.arange(len(bins)))
  ax.set_xticklabels(bins)
  ax.set_yticks(ticks=np.arange(len(bins)))
  ax.set_yticklabels(bins)
  fmt = lambda x, position: '{:.1f}e-3'.format(bins[x]*1e3)
  ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
  ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
  for index, label in enumerate(ax.get_xticklabels()):
    if index % 15 == 0:
        label.set_visible(True)
    else:
        label.set_visible(False)
  for index, label in enumerate(ax.get_yticklabels()):
    if index % 15 == 0:
        label.set_visible(True)
    else:
        label.set_visible(False)
  fig.colorbar(im, ax=ax)
  plt.xlabel('firing rate of source neuron', size=15)
  plt.ylabel('firing rate of target neuron', size=15)
  plt.title('cross correlation VS firing rate', size=15)
  plt.savefig('./plots/xcorr_FR_heatmap.jpg')

def plot_multi_heatmap_xcorr_FR(session_ids, stimulus_names, xcorr_dict, bin_dict):
  ind = 1
  rows, cols = session_ids, stimulus_names
  divnorm=colors.TwoSlopeNorm(vcenter=0.)
  fig = plt.figure(figsize=(5*len(cols), 5*len(rows)))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      ax = plt.subplot(len(rows), len(cols), ind)
      if row_ind == 0:
        plt.gca().set_title(cols[col_ind], fontsize=30, rotation=0)
      if col_ind == 0:
        plt.gca().text(0, 0.5 * (bottom + top), rows[row_ind],
        horizontalalignment='right',
        verticalalignment='center',
        # rotation='vertical',
        transform=plt.gca().transAxes, fontsize=30, rotation=90)
      plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
      ind += 1
      corr, bins = xcorr_dict[row][col], bin_dict[row][col]
      # pcolormesh(your_data, cmap="coolwarm", norm=divnorm)
      im = ax.imshow(corr, norm=divnorm, cmap="RdBu_r")
      ax.set_xticks(ticks=np.arange(len(bins)))
      ax.set_xticklabels(bins)
      ax.set_yticks(ticks=np.arange(len(bins)))
      ax.set_yticklabels(bins)
      fmt = lambda x, position: '{:.1f}e-3'.format(bins[x]*1e3)
      ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
      ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
      for index, label in enumerate(ax.get_xticklabels()):
        if index % 15 == 0:
            label.set_visible(True)
        else:
            label.set_visible(False)
      for index, label in enumerate(ax.get_yticklabels()):
        if index % 15 == 0:
            label.set_visible(True)
        else:
            label.set_visible(False)
      if col_ind == 7:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
      # plt.xlabel('firing rate of source neuron', size=15)
      # plt.ylabel('firing rate of target neuron', size=15)
      # plt.title('cross correlation VS firing rate', size=15)
  plt.suptitle('cross correlation VS firing rate', size=40)
  plt.tight_layout()
  plt.savefig('./plots/xcorr_FR_multi_heatmap.jpg')

def concatenate_trial(sequences, min_duration=250, min_len=10000):
  num_neuron, num_trial, T = sequences.shape
  if num_trial < np.ceil(min_len / min_duration):
    duration = int(np.ceil(min_len / num_trial)) # > min_duration
  else:
    duration = T # enough trials, <= min_duration
  return sequences[:, :, :duration].reshape(num_neuron, -1)

################ hub node region distribution
def get_hub_region_count(G_dict, area_dict, regions, weight=None):
  rows, cols = get_rowcol(G_dict)
  region_counts = {}
  for row in rows:
    print(row)
    if row not in region_counts:
      region_counts[row] = {}
    for col in cols:
      areas = area_dict[row]
      G = G_dict[row][col]
      nodes = np.array(list(dict(nx.degree(G)).keys()))
      degrees = np.array(list(dict(nx.degree(G, weight=weight)).values()))
      hub_th = np.mean(degrees) + 3 * np.std(degrees)
      hub_nodes = nodes[np.where(degrees > hub_th)]
      region_hubs = [areas[n] for n in hub_nodes if areas[n] in regions]
      uniq, counts = np.unique(region_hubs, return_counts=True)
      region_counts[row][col] = {k: v for k, v in sorted(dict(zip(uniq, counts)).items(), key=lambda item: item[1], reverse=True)}
  return region_counts

################ maximal clique region distribution
def get_max_clique_region_count(G_dict, area_dict, regions):
  region_counts = {}
  rows, cols = get_rowcol(G_dict)
  max_cliq_size = pd.DataFrame(index=rows, columns=cols)
  for row in rows:
    print(row)
    if row not in region_counts:
      region_counts[row] = {}
    for col in cols:
      G = nx.to_undirected(G_dict[row][col])

      nodes = list(G.nodes())
      node_area = [area_dict[row][n] for n in nodes if area_dict[row][n] in regions]
      total_uniq, total_counts = np.unique(node_area, return_counts=True)
      r_counts = {k: v for k, v in dict(zip(total_uniq, total_counts)).items()}

      cliqs = np.array(list(nx.find_cliques(G)))
      cliq_size = np.array([len(l) for l in cliqs])
      
      # if type(cliqs[np.where(cliq_size==max(cliq_size))[0]][0]) == list: # multiple maximum cliques
      #   # cliq_nodes = []
      #   # for li in cliqs[np.where(cliq_size==max(cliq_size))[0]]:
      #   #   cliq_nodes += li
      #   cliq_nodes = cliqs[np.where(cliq_size==max(cliq_size))[0]][0]
      # else:
      cliq_nodes = cliqs[np.argmax(cliq_size)]
      cliq_area = [area_dict[row][n] for n in cliq_nodes if area_dict[row][n] in regions]
      uniq, counts = np.unique(cliq_area, return_counts=True)
      region_counts[row][col] = {k: v/r_counts[k] for k, v in sorted(dict(zip(uniq, counts)).items(), key=lambda item: item[1], reverse=True)}
      max_cliq_size.loc[row][col] = max(cliq_size)
  return region_counts, max_cliq_size

def plot_group_size_stimulus(pos_group_size, neg_group_size, name, measure, n):
  fig = plt.figure(figsize=(10, 6))
  plt.subplot(1, 2, 1)
  for row_ind, row in enumerate(pos_group_size.index):
    plt.plot(pos_group_size.columns, pos_group_size.loc[row], label=row, alpha=1)
  plt.gca().set_title('size of positive {}'.format(name), fontsize=20, rotation=0)
  plt.xticks(rotation=90)
  plt.legend()
  plt.subplot(1, 2, 2)
  for row_ind, row in enumerate(neg_group_size.index):
    plt.plot(neg_group_size.columns, neg_group_size.loc[row], label=row, alpha=1)
  plt.gca().set_title('size of negative {}'.format(name), fontsize=20, rotation=0)
  plt.xticks(rotation=90)
  plt.tight_layout()
  figname = './plots/{}_size_stimulus_{}_{}_fold.jpg'.format(name, measure, n)
  plt.savefig(figname)

def plot_total_group_size_stimulus(group_size, name, measure, n):
  fig = plt.figure(figsize=(7, 6))
  for row_ind, row in enumerate(group_size.index):
    plt.plot(group_size.columns, group_size.loc[row], label=row, alpha=1)
  plt.gca().set_title('size of {}'.format(name), fontsize=20, rotation=0)
  plt.xticks(rotation=90)
  plt.ylabel('percentage of {} nodes in all nodes'.format(name))
  plt.legend()
  plt.tight_layout()
  figname = './plots/{}_size_stimulus_{}_{}_fold.jpg'.format(name, measure, n)
  plt.savefig(figname)

################ LSCC region distribution (largest strongly connected component)
def get_lscc_region_count(G_dict, area_dict, regions):
  region_counts = {}
  rows, cols = get_rowcol(G_dict)
  lscc_size = pd.DataFrame(index=rows, columns=cols)
  for row in rows:
    print(row)
    if row not in region_counts:
      region_counts[row] = {}
    for col in cols:
      G = G_dict[row][col]
      nodes = list(G.nodes())
      lscc_nodes = max(nx.strongly_connected_components(G), key=len)
      lscc_size.loc[row][col] = len(lscc_nodes) / len(nodes)
      node_area = [area_dict[row][n] for n in nodes if area_dict[row][n] in regions]
      uniq, counts = np.unique(node_area, return_counts=True)
      r_counts = {k: v for k, v in dict(zip(uniq, counts)).items()}
      lscc_area = [area_dict[row][n] for n in lscc_nodes if area_dict[row][n] in regions]
      uniq, counts = np.unique(lscc_area, return_counts=True)
      region_counts[row][col] = {k: v/r_counts[k] for k, v in sorted(dict(zip(uniq, counts)).items(), key=lambda item: item[1], reverse=True)}
  return region_counts, lscc_size

def plot_hub_pie_chart(region_counts, sign, name, regions):
  ind = 1
  rows, cols = get_rowcol(region_counts)
  hub_num = np.zeros((len(rows), len(cols)))
  fig = plt.figure(figsize=(4*len(cols), 3*len(rows)))
  # fig.patch.set_facecolor('black')
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      ax = plt.subplot(len(rows), len(cols), ind)
      if row_ind == 0:
        plt.gca().set_title(cols[col_ind], fontsize=20, rotation=0)
      if col_ind == 0:
        plt.gca().text(0, 0.5 * (bottom + top), rows[row_ind],
        horizontalalignment='left',
        verticalalignment='center',
        # rotation='vertical',
        transform=plt.gca().transAxes, fontsize=20, rotation=90)
      plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
      ind += 1
      labels = region_counts[row][col].keys()
      sizes = np.array(list(region_counts[row][col].values()))
      sizes = sizes / sum(sizes)
      hub_num[row_ind][col_ind] = sum(sizes)
      explode = np.zeros(len(labels))  # only "explode" the 2nd slice (i.e. 'Hogs')
      colors = [customPalette[regions.index(l)] for l in labels]
      patches, texts, pcts = plt.pie(sizes, radius=sum(sizes), explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
              shadow=True, startangle=90, wedgeprops={'linewidth': 3.0, 'edgecolor': 'white'})
      for i, patch in enumerate(patches):
        texts[i].set_color(patch.get_facecolor())
      # for i in range(len(p[0])):
      #   p[0][i].set_alpha(0.6)
      ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
  plt.suptitle('{} region distribution'.format(name), size=30)
  plt.tight_layout()
  # plt.show()
  fname = './plots/pie_chart_{}_{}.jpg'
  plt.savefig(fname.format(name, sign))

def get_directed_hub_region_count(G_dict, area_dict, regions, weight=None):
  rows, cols = get_rowcol(G_dict)
  source_region_counts, target_region_counts = {}, {}
  for row in rows:
    print(row)
    if row not in source_region_counts:
      source_region_counts[row] = {}
      target_region_counts[row] = {}
    for col in cols:
      areas = area_dict[row]
      G = G_dict[row][col]
      nodes = np.array(list(dict(G.out_degree()).keys()))
      degrees = np.array(list(dict(G.out_degree(weight=weight)).values()))
      hub_th = np.mean(degrees) + 3 * np.std(degrees)
      hub_nodes = nodes[np.where(degrees > hub_th)]
      region_hubs = [areas[n] for n in hub_nodes if areas[n] in regions]
      uniq, counts = np.unique(region_hubs, return_counts=True)
      source_region_counts[row][col] = {k: v for k, v in sorted(dict(zip(uniq, counts)).items(), key=lambda item: item[1], reverse=True)}

      nodes = np.array(list(dict(G.in_degree()).keys()))
      degrees = np.array(list(dict(G.in_degree(weight=weight)).values()))
      hub_th = np.mean(degrees) + 3 * np.std(degrees)
      hub_nodes = nodes[np.where(degrees > hub_th)]
      region_hubs = [areas[n] for n in hub_nodes if areas[n] in regions]
      uniq, counts = np.unique(region_hubs, return_counts=True)
      target_region_counts[row][col] = {k: v for k, v in sorted(dict(zip(uniq, counts)).items(), key=lambda item: item[1], reverse=True)}
  return source_region_counts, target_region_counts

def plot_directed_hub_pie_chart(region_counts, sign, direction, regions, weight):
  ind = 1
  rows, cols = get_rowcol(region_counts)
  hub_num = np.zeros((len(rows), len(cols)))
  fig = plt.figure(figsize=(4*len(cols), 3*len(rows)))
  # fig.patch.set_facecolor('black')
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      ax = plt.subplot(len(rows), len(cols), ind)
      if row_ind == 0:
        plt.gca().set_title(cols[col_ind], fontsize=20, rotation=0)
      if col_ind == 0:
        plt.gca().text(0, 0.5 * (bottom + top), rows[row_ind],
        horizontalalignment='left',
        verticalalignment='center',
        # rotation='vertical',
        transform=plt.gca().transAxes, fontsize=20, rotation=90)
      plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
      ind += 1
      labels = region_counts[row][col].keys()
      sizes = region_counts[row][col].values()
      hub_num[row_ind][col_ind] = sum(sizes)
      explode = np.zeros(len(labels))  # only "explode" the 2nd slice (i.e. 'Hogs')
      colors = [customPalette[regions.index(l)] for l in labels]
      patches, texts, pcts = plt.pie(sizes, radius=sum(sizes), explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
              shadow=True, startangle=90, wedgeprops={'linewidth': 3.0, 'edgecolor': 'white'})
      for i, patch in enumerate(patches):
        texts[i].set_color(patch.get_facecolor())
      # for i in range(len(p[0])):
      #   p[0][i].set_alpha(0.6)
      ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
  plt.suptitle('Hub nodes distribution', size=30)
  plt.tight_layout()
  # plt.show()
  fname = './plots/pie_chart_strength_{}_{}.jpg' if weight is not None else './plots/pie_chart_degree_{}_{}.jpg'
  plt.savefig(fname.format(sign, direction))

def plot_intra_inter_density(G_dict, sign, area_dict, regions, measure):
  rows, cols = get_rowcol(G_dict)
  metric = np.zeros((len(rows), len(cols), 3))
  region_connection = np.zeros((len(rows), len(cols), len(regions), len(regions)))
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      G = G_dict[row][col] if col in G_dict[row] else nx.Graph()
      if G.number_of_nodes() >= 2 and G.number_of_edges() > 0:
        nodes = list(G.nodes())
        node_area = {key: area_dict[row][key] for key in nodes}
        areas = list(node_area.values())
        area_size = [areas.count(r) for r in regions]
        intra_num = sum(map(lambda x : x * (x-1), area_size)) 
        inter_num = len(nodes) * (len(nodes) - 1) - intra_num
        A = nx.to_numpy_array(G)
        A[A.nonzero()] = 1
        for region_ind_i, region_i in enumerate(regions):
          for region_ind_j, region_j in enumerate(regions):
            region_indices_i = np.array([k for k, v in area_dict[row].items() if v==region_i])
            region_indices_j = np.array([k for k, v in area_dict[row].items() if v==region_j])
            region_indices_i = np.array([nodes.index(i) for i in list(set(region_indices_i) & set(nodes))]) # some nodes not in cc are removed 
            region_indices_j = np.array([nodes.index(i) for i in list(set(region_indices_j) & set(nodes))])
            if len(region_indices_i) and len(region_indices_j):
              region_connection[row_ind, col_ind, region_ind_i, region_ind_j] = np.sum(A[region_indices_i[:, None], region_indices_j])
              assert np.sum(A[region_indices_i[:, None], region_indices_j]) == len(A[region_indices_i[:, None], region_indices_j].nonzero()[0])
        
        diag_indx = np.eye(len(regions),dtype=bool)
        metric[row_ind, col_ind, 0] =  np.sum(region_connection[row_ind, col_ind][diag_indx]) / intra_num
        metric[row_ind, col_ind, 1] =  np.sum(region_connection[row_ind, col_ind][~diag_indx]) / inter_num
        metric[row_ind, col_ind, 2] = nx.density(G)
  metric_names = ['intra-region density', 'inter-region density', 'density']
  metric_stimulus = pd.DataFrame(columns=['stimulus', 'metric', 'mean', 'std'])
  # for metric_ind, metric_name in enumerate(metric_names):
  #   df = pd.DataFrame(columns=['stimulus', 'metric', 'mean', 'std'])
  #   df['mean'] = np.nanmean(metric[:, :, metric_ind], axis=0)
  #   df['std'] = np.nanstd(metric[:, :, metric_ind], axis=0)
  #   df['metric'] = metric_name
  #   df['stimulus'] = cols
  #   metric_stimulus = metric_stimulus.append(df, ignore_index=True)
  # print(metric_stimulus)
  fig = plt.figure(figsize=[6, 6])
  for row_ind, row in enumerate(rows):
    plt.plot(cols, metric[row_ind, :, 0], color='#1f77b4', alpha=0.6)
    plt.plot(cols, metric[row_ind, :, 1], color='#ff7f0e', alpha=0.6)
    plt.plot(cols, metric[row_ind, :, 2], color='#2ca02c', alpha=0.6)
  # for i, m in metric_stimulus.groupby("metric"):
  #   plt.plot(m['stimulus'], m['mean'], alpha=0.6, label=m['metric'].iloc[0])
  #   print(m['mean'], m['mean'] - m['std'], m['mean'] + m['std'])
  #   plt.fill_between(m['stimulus'], m['mean'] - m['std'], m['mean'] + m['std'], alpha=0.2)
  plt.title('{} network'.format(sign), size=20)
  plt.plot([], [], color='#1f77b4', alpha=0.6, label=metric_names[0])
  plt.plot([], [], color='#ff7f0e', alpha=0.6, label=metric_names[1])
  plt.plot([], [], color='#2ca02c', alpha=0.6, label=metric_names[2])
  plt.legend()
  plt.ylabel('density')
  plt.xticks(rotation=90)
  plt.tight_layout()
  # plt.show()
  plt.savefig('./plots/intra_inter_density_{}_{}.jpg'.format(sign, measure))

def plot_intra_inter_connection(G_dict, sign, area_dict, regions, measure):
  rows, cols = get_rowcol(G_dict)
  metric = np.zeros((len(rows), len(cols), 3))
  region_connection = np.zeros((len(rows), len(cols), len(regions), len(regions)))
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      G = G_dict[row][col] if col in G_dict[row] else nx.Graph()
      if G.number_of_nodes() >= 2 and G.number_of_edges() > 0:
        nodes = list(G.nodes())
        node_area = {key: area_dict[row][key] for key in nodes}
        areas = list(node_area.values())
        area_size = [areas.count(r) for r in regions]
        A = nx.to_numpy_array(G)
        A[A.nonzero()] = 1
        for region_ind_i, region_i in enumerate(regions):
          for region_ind_j, region_j in enumerate(regions):
            region_indices_i = np.array([k for k, v in area_dict[row].items() if v==region_i])
            region_indices_j = np.array([k for k, v in area_dict[row].items() if v==region_j])
            region_indices_i = np.array([nodes.index(i) for i in list(set(region_indices_i) & set(nodes))]) # some nodes not in cc are removed 
            region_indices_j = np.array([nodes.index(i) for i in list(set(region_indices_j) & set(nodes))])
            if len(region_indices_i) and len(region_indices_j):
              region_connection[row_ind, col_ind, region_ind_i, region_ind_j] = np.sum(A[region_indices_i[:, None], region_indices_j])
              assert np.sum(A[region_indices_i[:, None], region_indices_j]) == len(A[region_indices_i[:, None], region_indices_j].nonzero()[0])
        
        diag_indx = np.eye(len(regions),dtype=bool)
        metric[row_ind, col_ind, 0] =  np.sum(region_connection[row_ind, col_ind][diag_indx])
        metric[row_ind, col_ind, 1] =  np.sum(region_connection[row_ind, col_ind][~diag_indx])
        metric[row_ind, col_ind, 2] = nx.density(G)
  metric_names = ['intra-region connection', 'inter-region connection']
  metric_stimulus = pd.DataFrame(columns=['stimulus', 'metric', 'mean', 'std'])
  fig = plt.figure(figsize=[6, 6])
  for row_ind, row in enumerate(rows):
    plt.plot(cols, metric[row_ind, :, 0], color='#1f77b4', alpha=0.6)
    plt.plot(cols, metric[row_ind, :, 1], color='#ff7f0e', alpha=0.6)
  plt.title('{} network'.format(sign), size=20)
  plt.plot([], [], color='#1f77b4', alpha=0.6, label=metric_names[0])
  plt.plot([], [], color='#ff7f0e', alpha=0.6, label=metric_names[1])
  plt.legend()
  plt.ylabel('number of connections')
  plt.xticks(rotation=90)
  plt.tight_layout()
  # plt.show()
  plt.savefig('./plots/intra_inter_connection_{}_{}.jpg'.format(sign, measure))

def plot_intra_inter_ratio(G_dict, sign, area_dict, regions, measure):
  rows, cols = get_rowcol(G_dict)
  metric = np.zeros((len(rows), len(cols)))
  region_connection = np.zeros((len(rows), len(cols), len(regions), len(regions)))
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      G = G_dict[row][col] if col in G_dict[row] else nx.Graph()
      if G.number_of_nodes() >= 2 and G.number_of_edges() > 0:
        nodes = list(G.nodes())
        node_area = {key: area_dict[row][key] for key in nodes}
        areas = list(node_area.values())
        area_size = [areas.count(r) for r in regions]
        A = nx.to_numpy_array(G)
        A[A.nonzero()] = 1
        for region_ind_i, region_i in enumerate(regions):
          for region_ind_j, region_j in enumerate(regions):
            region_indices_i = np.array([k for k, v in area_dict[row].items() if v==region_i])
            region_indices_j = np.array([k for k, v in area_dict[row].items() if v==region_j])
            region_indices_i = np.array([nodes.index(i) for i in list(set(region_indices_i) & set(nodes))]) # some nodes not in cc are removed 
            region_indices_j = np.array([nodes.index(i) for i in list(set(region_indices_j) & set(nodes))])
            if len(region_indices_i) and len(region_indices_j):
              region_connection[row_ind, col_ind, region_ind_i, region_ind_j] = np.sum(A[region_indices_i[:, None], region_indices_j])
              assert np.sum(A[region_indices_i[:, None], region_indices_j]) == len(A[region_indices_i[:, None], region_indices_j].nonzero()[0])
        
        diag_indx = np.eye(len(regions),dtype=bool)
        metric[row_ind, col_ind] =  np.sum(region_connection[row_ind, col_ind][diag_indx]) / np.sum(region_connection[row_ind, col_ind][~diag_indx])
  fig = plt.figure(figsize=[6, 6])
  for row_ind, row in enumerate(rows):
    plt.plot(cols, metric[row_ind, :], alpha=0.6, label=row)
  plt.title('{} network'.format(sign), size=20)
  plt.legend()
  plt.ylabel('intra/inter ratio')
  plt.xticks(rotation=90)
  plt.tight_layout()
  # plt.show()
  plt.savefig('./plots/intra_inter_ratio_{}_{}.jpg'.format(sign, measure))

def add_sign(G_dict):
  rows, cols = get_rowcol(G_dict)
  S_dict = {}
  for row in rows:
    S_dict[row] = {}
    for col in cols:
      G = G_dict[row][col]
      weights = nx.get_edge_attributes(G,'weight')
      A = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
      A[A.nonzero()] = 1
      S = nx.from_numpy_array(A, create_using=nx.DiGraph)
      node_idx = sorted(G.nodes())
      mapping = {i:node_idx[i] for i in range(len(node_idx))}
      S = nx.relabel_nodes(S, mapping)
      for (n1, n2, d) in S.edges(data=True):
        d.clear()
      signs = {}
      for e, w in weights.items():
        signs[e] = np.sign(w)
      nx.set_edge_attributes(S, signs, 'sign')
      S_dict[row][col] = S
  return S_dict

def add_offset(G_dict, offset_dict):
  rows, cols = get_rowcol(G_dict)
  S_dict = {}
  for row in rows:
    S_dict[row] = {}
    for col in cols:
      offset_mat = offset_dict[row][col]
      G = G_dict[row][col]
      nodes = sorted(list(G.nodes()))
      offset = {}
      for edge in G.edges():
        offset[edge] = offset_mat[nodes.index(edge[0]), nodes.index(edge[1])]
      S = G.copy()
      nx.set_edge_attributes(S, offset, 'offset')
      S_dict[row][col] = S
  return S_dict

def add_duration(G_dict, duration_dict):
  rows, cols = get_rowcol(G_dict)
  S_dict = {}
  for row in rows:
    S_dict[row] = {}
    for col in cols:
      duration_mat = duration_dict[row][col]
      G = G_dict[row][col]
      nodes = sorted(list(G.nodes()))
      duration = {}
      for edge in G.edges():
        duration[edge] = duration_mat[nodes.index(edge[0]), nodes.index(edge[1])]
      S = G.copy()
      nx.set_edge_attributes(S, duration, 'duration')
      S_dict[row][col] = S
  return S_dict

def add_delay(G_dict):
  rows, cols = get_rowcol(G_dict)
  S_dict = {}
  for row in rows:
    S_dict[row] = {}
    for col in cols:
      G = G_dict[row][col]
      nodes = sorted(list(G.nodes()))
      delay = {}
      for edge in G.edges():
        delay[edge] = G.get_edge_data(*edge)['offset'] + G.get_edge_data(*edge)['duration']
      S = G.copy()
      nx.set_edge_attributes(S, delay, 'delay')
      S_dict[row][col] = S
  return S_dict
# This section calculates balace in triads with respect to direction.
# We first extract the transitive triads, then we break the transitive triads to semi-cycles, and finally 
# calculate balance in each semicycle. The triad is balance only if all its semicycles are balance. 
# updated = 04/22/2020

# This algorithm is based on the model introduced in the following paper:
# XXX (will be updated)

# from networkx import *

## counting the number of instances in a list
def count_lists(mylist):            
    new_dict = {}
    for i in mylist:
        if i[1] not in new_dict:
            new_dict[i[1]] = 1
        else:
            new_dict[i[1]] += 1
    return (new_dict)            

## Get all triples in triads with respect to their census and edgelists (in edge_atts)
def get_directed_triads(triads):
    # Get all triplets of edges
    for candidate_edges in itertools.combinations(triads.items(), 3):
        # Get edges between unique pair of nodes
        unique_edges = set([tuple(sorted(k)) for k,v in candidate_edges])
        # Only consider triad in which the tree edges use a unique pair of nodes
        if len(unique_edges) == 3:
            yield dict(candidate_edges)
            
## searching through traids
def search_triangles(G, nodes = None):
    if nodes is None:
        nodes_nbrs = G.adj.items()
    else:
        nodes_nbrs = ((n, G[n]) for n in G.nbunch_iter(nodes))
    for v, v_nbrs in nodes_nbrs:
        vs = set(v_nbrs) - {v} # neighbors of v, remove selfloop
        for w in vs:
            #print(w)
            xx = vs & (set(G[w]) - {w})
            yield [ set(x) for x in list(zip(itertools.repeat(v), itertools.repeat(w), list(xx))) ]
            
#Calculate balance in traids (main function)
def calculate_traid_balance(G_new):
    triad_dict = {}
    triad_class = {}
    all_triads = []
    ## there are only 4 transistive census: 030T, 120D, 120U, and 300 
    non_transitive_census = ['003','012', '102', '021D', '021C', '021U', '021', '111U', '111D', '201', '030C', '120C', '210']
    all_triad = nx.triangles(G_new.to_undirected())
    iter_g = search_triangles(G_new)
    
    for iter_t in iter_g:
        for ta in list(iter_t):
            tt = ",".join([str(x) for x in sorted(set(ta))])
            triad_dict[tt] = True
            
    for val in triad_dict.keys():
        nodes = [int(x) for x in val.split(",")]
        census = [k for k, v in nx.triads.triadic_census(G_new.subgraph(nodes)).items() if v][0]
        if census not in non_transitive_census:
            sign = nx.get_edge_attributes(G_new.subgraph(nodes),'sign')
            triad_class[val] = [census, sign]
            #map_census_edges(G_new, val, triad_class)     
    for key, value in triad_class.items():
        all_directed_triads = list(get_directed_triads(value[1]))
        all_triads.append([all_directed_triads, value[0]])
            
    ## getting the balance vs. imbalance triads 
    balances = []
    imbalances = []
    for items in all_triads:
        balance_list = []
        
        ## removing two cycles from 300 and then calculate balance
        if items[1] == '300':
            for triangle in items[0]:
                node = []
                for edge in triangle:
                    if edge[0] not in node:
                        node.append(edge[0])
                if len(node) != 3:
                    balance = 1
                    for edge in triangle:
                        balance *= triangle[edge]
                    balance_list.append(balance)
        else:
            for item in items[0]:
                balance = 1
                for edge in item:
                    balance *= item[edge]
                balance_list.append(balance)
        neg = []
        for n in balance_list:
            if n <= 0 :
                neg.append(n)
        if neg:
            imbalances.append(items)
        else:
            balances.append(items)
    # print('Balance')
    # print(balances)
    # print('Imbalance')
    # print(imbalances)
    # print ('Triad Level Balance: ', (len(balances)/(len(balances) + len(imbalances))))        
    # print ('Number of balance and transitive triads: ', len(balances))
    # print ('Number of imbalance and transitive triads: ', len(imbalances))
    
    # print('Number of balance triads in each census', count_lists(balances))
    # print('Number of imbalance triads in each census', count_lists(imbalances))
    tb = len(balances)/(len(balances) + len(imbalances)) if len(balances) + len(imbalances) > 0 else 0
    return balances, imbalances, tb, len(balances), len(imbalances), count_lists(balances), count_lists(imbalances)

# Get census of all signed transitive triads
def get_all_signed_transitive_triads(G_dict):
  rows, cols = get_rowcol(G_dict)
  all_triads = {}
  for row in rows:
    all_triads[row] = {}
    for col in cols:
      triad_dict = {}
      triad_class = {}
      all_triads[row][col] = []
      ## there are only 4 transistive census: 030T, 120D, 120U, and 300 
      non_transitive_census = ['003','012', '102', '021D', '021C', '021U', '111U', '111D', '201', '030C', '120C', '210'] # , '021'
      G = G_dict[row][col]
      iter_g = search_triangles(G)
      for iter_t in iter_g:
        for ta in list(iter_t):
          tt = ",".join([str(x) for x in sorted(set(ta))])
          triad_dict[tt] = True     
      for val in triad_dict.keys():
        nodes = [int(x) for x in val.split(",")]
        census = [k for k, v in nx.triads.triadic_census(G.subgraph(nodes)).items() if v][0]
        if census not in non_transitive_census:
          sign = nx.get_edge_attributes(G.subgraph(nodes),'sign')
          triad_class[val] = [census, sign]
          #map_census_edges(G, val, triad_class)     
      for key, value in triad_class.items():
        all_directed_triads = list(get_directed_triads(value[1]))
        all_triads[row][col].append([all_directed_triads, value[0]])
  return all_triads

# Get census of all signed triads
def get_all_signed_triads(G_dict):
  rows, cols = get_rowcol(G_dict)
  all_triads = {}
  for row in rows:
    all_triads[row] = {}
    for col in cols:
      triad_dict = {}
      triad_class = {}
      all_triads[row][col] = []
      G = G_dict[row][col]
      iter_g = search_triangles(G)
      for iter_t in iter_g:
        for ta in list(iter_t):
          tt = ",".join([str(x) for x in sorted(set(ta))])
          triad_dict[tt] = True     
      for val in triad_dict.keys():
        nodes = [int(x) for x in val.split(",")]
        census = [k for k, v in nx.triads.triadic_census(G.subgraph(nodes)).items() if v][0]
        sign = nx.get_edge_attributes(G.subgraph(nodes),'sign')
        triad_class[val] = [census, sign]
        #map_census_edges(G, val, triad_class)     
      for key, value in triad_class.items():
        all_directed_triads = list(get_directed_triads(value[1]))
        all_triads[row][col].append([all_directed_triads, value[0]])
  return all_triads

def triadic_census(G_dict):
  rows, cols = get_rowcol(G_dict)
  triad_count = {}
  for row in rows:
    print(row)
    triad_count[row] = {}
    for  col in cols:
      G = G_dict[row][col]
      triad_count[row][col] = nx.triads.triadic_census(G)
      triad_count[row][col] = dict(sorted(triad_count[row][col].items(), key=lambda x:x[1], reverse=True))
  return triad_count
  
def count_triplet_connection_p(G):
  num0, num1, num2 = 0, 0, 0
  nodes = list(G.nodes())
  for node_i in range(len(nodes)):
    for node_j in range(len(nodes)):
      if node_i != node_j:
        if G.has_edge(nodes[node_i], nodes[node_j]) or G.has_edge(nodes[node_j], nodes[node_i]):
          if G.has_edge(nodes[node_i], nodes[node_j]) and G.has_edge(nodes[node_j], nodes[node_i]):
            num2 += 1
          else:
            num1 += 1
        else:
          num0 += 1
  assert num0 + num1 + num2 == len(nodes) * (len(nodes) - 1)
  assert num1 / 2 + num2 == G.number_of_edges()
  p0, p1, p2 = safe_division(num0, num0 + num1 + num2), safe_division(num1, num0 + num1 + num2), safe_division(num2, num0 + num1 + num2)
  return p0, p1, p2

def plot_pair_relative_count(G_dict, p_pair_func, measure, n, log=False, scale = True):
  ind = 1
  rows, cols = get_rowcol(G_dict)
  fig = plt.figure(figsize=(23, 4))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  ylim = 0
  for col in cols:
    for row in rows:
      G = G_dict[row][col].copy() if col in G_dict[row] else nx.DiGraph()
      p = nx.density(G)
      p0, p1, p2 = count_triplet_connection_p(G)
      ylim = max(ylim, p0 / p_pair_func['0'](p), p1 / p_pair_func['1'](p), p2 / p_pair_func['2'](p))
  for col in cols:
    print(col)
    plt.subplot(1, 7, ind)
    plt.gca().set_title(col, fontsize=20, rotation=0)
    plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ind += 1
    all_pair_count = defaultdict(lambda: [])
    for row in rows:
      G = G_dict[row][col].copy() if col in G_dict[row] else nx.DiGraph()
      p = nx.density(G)
      p0, p1, p2 = count_triplet_connection_p(G)
      all_pair_count['0'].append(p0 / p_pair_func['0'](p))
      all_pair_count['1'].append(p1 / p_pair_func['1'](p))
      all_pair_count['2'].append(p2 / p_pair_func['2'](p))
    
    triad_types, triad_counts = [k for k,v in all_pair_count.items()], [v for k,v in all_pair_count.items()]
    plt.boxplot(triad_counts, showfliers=False)
    plt.xticks(list(range(1, len(triad_counts)+1)), triad_types, rotation=0)
    left, right = plt.xlim()
    plt.hlines(1, xmin=left, xmax=right, color='r', linestyles='--', linewidth=0.5)
    # plt.hlines(1, color='r', linestyles='--')
    if scale:
      if not log:
        plt.ylim(top=ylim)
      else:
        plt.yscale('log')
        plt.ylim(top=ylim)
    # plt.hist(data.flatten(), bins=12, density=True)
    # plt.axvline(x=np.nanmean(data), color='r', linestyle='--')
    # plt.xlabel('region')
    # plt.xlabel('size')
    plt.ylabel('relative count')
  plt.suptitle('Relative count of all pairs', size=30)
  plt.tight_layout(rect=[0, 0.03, 1, 0.98])
  image_name = './plots/relative_count_allpair_scale_{}_{}fold.jpg' if scale else './plots/relative_count_allpair_{}_{}fold.jpg'
  # plt.show()
  plt.savefig(image_name.format(measure, n))

def plot_triad_relative_count(G_dict, p_triad_func, measure, n, log=True):
  ind = 1
  rows, cols = get_rowcol(G_dict)
  fig = plt.figure(figsize=(23, 15))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  for col in cols:
    print(col)
    plt.subplot(4, 2, ind)
    plt.gca().set_title(col, fontsize=20, rotation=0)
    plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ind += 1
    all_triad_count = defaultdict(lambda: [])
    for row in rows:
      G = G_dict[row][col].copy() if col in G_dict[row] else nx.DiGraph()
      G_triad_count = nx.triads.triadic_census(G)
      num_triplet = sum(G_triad_count.values())
      p0, p1, p2 = count_triplet_connection_p(G)
      for triad_type in G_triad_count:
        relative_c = G_triad_count[triad_type] / (num_triplet * p_triad_func[triad_type](p0, p1, p2)) if num_triplet * p_triad_func[triad_type](p0, p1, p2) else 0
        all_triad_count[triad_type].append(relative_c)
    
    triad_types, triad_counts = [k for k,v in all_triad_count.items()], [v for k,v in all_triad_count.items()]
    plt.boxplot(triad_counts)
    plt.xticks(list(range(1, len(triad_counts)+1)), triad_types, rotation=0)
    left, right = plt.xlim()
    plt.hlines(1, xmin=left, xmax=right, color='r', linestyles='--', linewidth=0.5, alpha=0.6)
    # plt.hist(data.flatten(), bins=12, density=True)
    # plt.axvline(x=np.nanmean(data), color='r', linestyle='--')
    # plt.xlabel('region')
    # plt.xlabel('size')
    if log:
      plt.yscale('log')
    plt.ylabel('relative count')
  plt.suptitle('Relative count of all triads', size=30)
  plt.tight_layout(rect=[0, 0.03, 1, 0.98])
  image_name = './plots/relative_count_alltriad_{}_{}fold.jpg'.format(measure, n)
  # plt.show()
  plt.savefig(image_name)

def plot_singleregion_pair_relative_count(G_dict, area_dict, area, p_pair_func, measure, n, scale = True):
  ind = 1
  rows, cols = get_rowcol(G_dict)
  fig = plt.figure(figsize=(23, 4))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  ylim = 0
  for col in cols:
    for row in rows:
      G = G_dict[row][col].copy() if col in G_dict[row] else nx.DiGraph()
      area_nodes = [k  for k, v in area_dict[row].items() if v == area]
      G = nx.subgraph(G, area_nodes)
      p = nx.density(G)
      p0, p1, p2 = count_triplet_connection_p(G)
      ylim = max(ylim, safe_division(p0, p_pair_func['0'](p)), safe_division(p1, p_pair_func['1'](p)), safe_division(p2, p_pair_func['2'](p)))
  for col in cols:
    print(col)
    plt.subplot(1, 7, ind)
    plt.gca().set_title(col, fontsize=20, rotation=0)
    plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ind += 1
    all_pair_count = defaultdict(lambda: [])
    for row in rows:
      G = G_dict[row][col].copy() if col in G_dict[row] else nx.DiGraph()
      area_nodes = [k  for k, v in area_dict[row].items() if v == area]
      G = nx.subgraph(G, area_nodes)
      p = nx.density(G)
      p0, p1, p2 = count_triplet_connection_p(G)
      all_pair_count['0'].append(safe_division(p0, p_pair_func['0'](p)))
      all_pair_count['1'].append(safe_division(p1, p_pair_func['1'](p)))
      all_pair_count['2'].append(safe_division(p2, p_pair_func['2'](p)))
    
    triad_types, triad_counts = [k for k,v in all_pair_count.items()], [v for k,v in all_pair_count.items()]
    plt.boxplot(triad_counts)
    plt.xticks(list(range(1, len(triad_counts)+1)), triad_types, rotation=0)
    left, right = plt.xlim()
    plt.hlines(1, xmin=left, xmax=right, color='r', linestyles='--', linewidth=0.5)
    # plt.hlines(1, color='r', linestyles='--')
    if scale:
      plt.ylim(0, ylim)
    # plt.hist(data.flatten(), bins=12, density=True)
    # plt.axvline(x=np.nanmean(data), color='r', linestyle='--')
    # plt.xlabel('region')
    # plt.xlabel('size')
    # plt.yscale('log')
    plt.ylabel('relative count')
  plt.suptitle('Relative count of all pairs in {}'.format(area), size=30)
  plt.tight_layout(rect=[0, 0.03, 1, 0.98])
  image_name = './plots/relative_count_{}_allpair_scale_{}_{}fold.jpg' if scale else './plots/relative_count_{}_allpair_{}_{}fold.jpg'
  # plt.show()
  plt.savefig(image_name.format(area, measure, n))

def plot_singleregion_triad_relative_count(G_dict, area_dict, area, p_triad_func, measure, n):
  ind = 1
  rows, cols = get_rowcol(G_dict)
  fig = plt.figure(figsize=(23, 15))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  for col in cols:
    print(col)
    plt.subplot(4, 2, ind)
    plt.gca().set_title(col, fontsize=20, rotation=0)
    plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ind += 1
    all_triad_count = defaultdict(lambda: [])
    for row in rows:
      G = G_dict[row][col].copy() if col in G_dict[row] else nx.DiGraph()
      area_nodes = [k  for k, v in area_dict[row].items() if v == area]
      G = nx.subgraph(G, area_nodes)
      G_triad_count = nx.triads.triadic_census(G)
      num_triplet = sum(G_triad_count.values())
      p0, p1, p2 = count_triplet_connection_p(G)
      for triad_type in G_triad_count:
        relative_c = safe_division(G_triad_count[triad_type], num_triplet * p_triad_func[triad_type](p0, p1, p2))
        all_triad_count[triad_type].append(relative_c)
    
    triad_types, triad_counts = [k for k,v in all_triad_count.items()], [v for k,v in all_triad_count.items()]
    plt.boxplot(triad_counts, showfliers=False)
    plt.xticks(list(range(1, len(triad_counts)+1)), triad_types, rotation=0)
    left, right = plt.xlim()
    plt.hlines(1, xmin=left, xmax=right, color='r', linestyles='--', linewidth=0.5, alpha=0.6)
    # plt.hist(data.flatten(), bins=12, density=True)
    # plt.axvline(x=np.nanmean(data), color='r', linestyle='--')
    # plt.xlabel('region')
    # plt.xlabel('size')
    # plt.yscale('log')
    plt.ylabel('relative count')
  plt.suptitle('Relative count of all triads in {}'.format(area), size=30)
  plt.tight_layout(rect=[0, 0.03, 1, 0.98])
  image_name = './plots/relative_count_{}_alltriad_{}_{}fold.jpg'.format(area, measure, n)
  # plt.show()
  plt.savefig(image_name)

def triad_census(all_triads):
  rows, cols = get_rowcol(all_triads)
  triad_count = {}
  for row in rows:
    print(row)
    triad_count[row] = {}
    for  col in cols:
      triad_count[row][col] = {}
      for triad in all_triads[row][col]:
        triad_type = triad[1]
        triad_count[row][col][triad_type] = triad_count[row][col].get(triad_type, 0) + 1 
      triad_count[row][col] = dict(sorted(triad_count[row][col].items(), key=lambda x:x[1], reverse=True))
  return triad_count

def signed_triad_census(all_triads):
  rows, cols = get_rowcol(all_triads)
  signed_triad_count = {}
  for row_ind, row in enumerate(rows):
    print(row)
    signed_triad_count[row] = {}
    for col_ind, col in enumerate(cols):
      signed_triad_count[row][col] = {}
      for triad in all_triads[row][col]:
        triad_type = triad[1]
        if triad_type == '030T':
          node_P = most_common([i for i,j in triad[0][0].keys()])
          node_X = most_common([j for i,j in triad[0][0].keys()])
          triplets = set([node for sub in triad[0][0].keys() for node in sub])
          triplets.remove(node_P)
          triplets.remove(node_X)
          node_O = list(triplets)[0]
          edge_order = [(node_P, node_X), (node_P, node_O), (node_O, node_X)]    
        elif triad_type == '120D' or triad_type == '120U':
          if triad_type == '120D':
            node_X = most_common([i for i,j in triad[0][0].keys()])
          else:
            node_X = most_common([j for i,j in triad[0][0].keys()])
          triplets = set([node for sub in triad[0][0].keys() for node in sub])
          triplets.remove(node_X)
          triplets = list(triplets)
          np.random.shuffle(triplets)
          node_P, node_O = triplets
          if triad_type == '120D':
            edge_order = [(node_X, node_P), (node_X, node_O), (node_P, node_O), (node_O, node_P)]
          else:
            edge_order = [(node_P, node_X), (node_O, node_X), (node_P, node_O), (node_O, node_P)]
        else:
          triplets = list(set([node for sub in triad[0][0].keys() for node in sub]))
          np.random.shuffle(triplets)
          node_P, node_X, node_O = triplets
          edge_order = [(node_X, node_P), (node_P, node_X), (node_X, node_O), (node_O, node_X), (node_P, node_O), (node_O, node_P)]
        triad_sign = {k: v for d in triad[0] for k, v in d.items()}
        sign = [triad_sign[edge] for edge in edge_order]
        sign = ''.join(map(lambda x:'+' if x==1 else '-', sign))
        if triad_type == '120D' or triad_type == '120U':
          sign = ''.join(sorted(sign[:2]) + sorted(sign[-2:])) # X->P/O, P<->O
        elif triad_type == '300':
          sign = ''.join([i for x in sorted([sorted(sign[:2]),sorted(sign[2:4]),sorted(sign[-2:])]) for i in x]) # X<->P, X<->O, P<->O
        signed_triad_count[row][col][triad_type + sign] = signed_triad_count[row][col].get(triad_type + sign, 0) + 1 
      signed_triad_count[row][col] = dict(sorted(signed_triad_count[row][col].items(), key=lambda x:x[1], reverse=True))
  return signed_triad_count

def summice_triad_census(all_triads):
  rows, cols = get_rowcol(all_triads)
  summice_triad_count = {}
  summice_triad_count['sum'] = {}
  for col in cols:
    print(col)
    summice_triad_count['sum'][col] = {}
    for row in rows:
      for triad in all_triads[row][col]:
        triad_type = triad[1]
        summice_triad_count['sum'][col][triad_type] = summice_triad_count['sum'][col].get(triad_type, 0) + 1 
      summice_triad_count['sum'][col] = dict(sorted(summice_triad_count['sum'][col].items(), key=lambda x:x[1], reverse=True))
  return summice_triad_count

def summice_signed_triad_census(all_triads):
  rows, cols = get_rowcol(all_triads)
  summice_signed_triad_count = {}
  summice_signed_triad_count['sum'] = {}
  for col in cols:
    print(col)
    summice_signed_triad_count['sum'][col] = {}
    for row in rows:
      for triad in all_triads[row][col]:
        triad_type = triad[1]
        if triad_type == '030T':
          node_P = most_common([i for i,j in triad[0][0].keys()])
          node_X = most_common([j for i,j in triad[0][0].keys()])
          triplets = set([node for sub in triad[0][0].keys() for node in sub])
          triplets.remove(node_P)
          triplets.remove(node_X)
          node_O = list(triplets)[0]
          edge_order = [(node_P, node_X), (node_P, node_O), (node_O, node_X)]    
        elif triad_type == '120D' or triad_type == '120U':
          if triad_type == '120D':
            node_X = most_common([i for i,j in triad[0][0].keys()])
          else:
            node_X = most_common([j for i,j in triad[0][0].keys()])
          triplets = set([node for sub in triad[0][0].keys() for node in sub])
          triplets.remove(node_X)
          triplets = list(triplets)
          np.random.shuffle(triplets)
          node_P, node_O = triplets
          if triad_type == '120D':
            edge_order = [(node_X, node_P), (node_X, node_O), (node_P, node_O), (node_O, node_P)]
          else:
            edge_order = [(node_P, node_X), (node_O, node_X), (node_P, node_O), (node_O, node_P)]
        else:
          triplets = list(set([node for sub in triad[0][0].keys() for node in sub]))
          np.random.shuffle(triplets)
          node_P, node_X, node_O = triplets
          edge_order = [(node_X, node_P), (node_P, node_X), (node_X, node_O), (node_O, node_X), (node_P, node_O), (node_O, node_P)]
        triad_sign = {k: v for d in triad[0] for k, v in d.items()}
        sign = [triad_sign[edge] for edge in edge_order]
        sign = ''.join(map(lambda x:'+' if x==1 else '-', sign))
        if triad_type == '120D' or triad_type == '120U':
          sign = ''.join(sorted(sign[:2]) + sorted(sign[-2:])) # X->P/O, P<->O
        elif triad_type == '300':
          sign = ''.join([i for x in sorted([sorted(sign[:2]),sorted(sign[2:4]),sorted(sign[-2:])]) for i in x]) # X<->P, X<->O, P<->O
        summice_signed_triad_count['sum'][col][triad_type + sign] = summice_signed_triad_count['sum'][col].get(triad_type + sign, 0) + 1 
    summice_signed_triad_count['sum'][col] = dict(sorted(summice_signed_triad_count['sum'][col].items(), key=lambda x:x[1], reverse=True))
  return summice_signed_triad_count

################# mean of percentage (each mouse has the same effect)
def meanmice_triad_census(all_triads):
  rows, cols = get_rowcol(all_triads)
  meanmice_triad_percent = {}
  meanmice_triad_percent['mean'] = {}
  for col in cols:
    print(col)
    meantriad_percent = {}
    num_nonzero = 0
    meanmice_triad_percent['mean'][col] = {}
    for row in rows:
      print(row)
      triad_percent = {}
      for triad in all_triads[row][col]:
        triad_type = triad[1]
        triad_percent[triad_type] = triad_percent.get(triad_type, 0) + 1
      triad_percent = {k:v/sum(triad_percent.values()) for k, v in triad_percent.items()}
      if len(triad_percent):
        num_nonzero += 1
        assert np.isclose(sum(triad_percent.values()), 1)
      meantriad_percent = {k:meantriad_percent.get(k, 0)+triad_percent.get(k, 0) for k in np.unique(list(triad_percent.keys()) + list(meantriad_percent.keys()))}
    meanmice_triad_percent['mean'][col] = {k:v/num_nonzero for k, v in meantriad_percent.items()}
    meanmice_triad_percent['mean'][col] = dict(sorted(meanmice_triad_percent['mean'][col].items(), key=lambda x:x[1], reverse=True))
  return meanmice_triad_percent

def meanmice_signed_triad_census(all_triads):
  rows, cols = get_rowcol(all_triads)
  meanmice_signed_triad_percent = {}
  meanmice_signed_triad_percent['mean'] = {}
  for col in cols:
    print(col)
    meantriad_percent = {}
    num_nonzero = 0
    meanmice_signed_triad_percent['mean'][col] = {}
    for row in rows:
      triad_percent = {}
      for triad in all_triads[row][col]:
        triad_type = triad[1]
        if triad_type == '030T':
          node_P = most_common([i for i,j in triad[0][0].keys()])
          node_X = most_common([j for i,j in triad[0][0].keys()])
          triplets = set([node for sub in triad[0][0].keys() for node in sub])
          triplets.remove(node_P)
          triplets.remove(node_X)
          node_O = list(triplets)[0]
          edge_order = [(node_P, node_X), (node_P, node_O), (node_O, node_X)]    
        elif triad_type == '120D' or triad_type == '120U':
          if triad_type == '120D':
            node_X = most_common([i for i,j in triad[0][0].keys()])
          else:
            node_X = most_common([j for i,j in triad[0][0].keys()])
          triplets = set([node for sub in triad[0][0].keys() for node in sub])
          triplets.remove(node_X)
          triplets = list(triplets)
          np.random.shuffle(triplets)
          node_P, node_O = triplets
          if triad_type == '120D':
            edge_order = [(node_X, node_P), (node_X, node_O), (node_P, node_O), (node_O, node_P)]
          else:
            edge_order = [(node_P, node_X), (node_O, node_X), (node_P, node_O), (node_O, node_P)]
        else:
          triplets = list(set([node for sub in triad[0][0].keys() for node in sub]))
          np.random.shuffle(triplets)
          node_P, node_X, node_O = triplets
          edge_order = [(node_X, node_P), (node_P, node_X), (node_X, node_O), (node_O, node_X), (node_P, node_O), (node_O, node_P)]
        triad_sign = {k: v for d in triad[0] for k, v in d.items()}
        sign = [triad_sign[edge] for edge in edge_order]
        sign = ''.join(map(lambda x:'+' if x==1 else '-', sign))
        if triad_type == '120D' or triad_type == '120U':
          sign = ''.join(sorted(sign[:2]) + sorted(sign[-2:])) # X->P/O, P<->O
        elif triad_type == '300':
          sign = ''.join([i for x in sorted([sorted(sign[:2]),sorted(sign[2:4]),sorted(sign[-2:])]) for i in x]) # X<->P, X<->O, P<->O
        triad_percent[triad_type + sign] = triad_percent.get(triad_type + sign, 0) + 1
      triad_percent = {k:v/sum(triad_percent.values()) for k, v in triad_percent.items()}
      if len(triad_percent):
        num_nonzero += 1
        assert np.isclose(sum(triad_percent.values()), 1)
      meantriad_percent = {k:meantriad_percent.get(k, 0)+triad_percent.get(k, 0) for k in np.unique(list(triad_percent.keys()) + list(meantriad_percent.keys()))}
    meanmice_signed_triad_percent['mean'][col] = {k:v/num_nonzero for k, v in meantriad_percent.items()}
    meanmice_signed_triad_percent['mean'][col] = dict(sorted(meanmice_signed_triad_percent['mean'][col].items(), key=lambda x:x[1], reverse=True))
  return meanmice_signed_triad_percent

########################### motif analysis for all 13 motifs ##########################
################## motif intensity and coherence
def get_motif_sign(motif, motif_type, weight='confidence'):
  edges = list(motif.edges())
  nodes = [node for sub in edges for node in sub]
  triplets = list(set(nodes))
  if motif_type == '021D':
    node_P = most_common([i for i,j in edges])
    node_X, node_O = [j for i,j in edges]
    edge_order = [(node_P, node_X), (node_P, node_O)]
  elif motif_type == '021U':
    node_P = most_common([j for i,j in edges])
    node_X, node_O = [i for i,j in edges]
    edge_order = [(node_X, node_P), (node_O, node_P)]
  elif motif_type == '021C':
    node_X = most_common(nodes)
    triplets.remove(node_X)
    if (triplets[0], node_X) in edges:
      node_P, node_O = triplets
    else:
      node_O, node_P = triplets
    edge_order = [(node_P, node_X), (node_X, node_O)]
  elif motif_type == '111D':
    node_X = most_common([j for i,j in edges])
    node_P = [j for i,j in edges if i == node_X][0]
    triplets.remove(node_X)
    triplets.remove(node_P)
    node_O = triplets[0]
    edge_order = [(node_P, node_X), (node_X, node_P), (node_O, node_X)]
  elif motif_type == '111U':
    node_X = most_common([i for i,j in edges])
    node_P = [i for i,j in edges if j == node_X][0]
    triplets.remove(node_X)
    triplets.remove(node_P)
    node_O = triplets[0]
    edge_order = [(node_P, node_X), (node_X, node_P), (node_X, node_O)]
  elif motif_type == '030T':
    node_P = most_common([i for i,j in edges])
    node_X = most_common([j for i,j in edges])
    triplets.remove(node_P)
    triplets.remove(node_X)
    node_O = triplets[0]
    edge_order = [(node_P, node_X), (node_P, node_O), (node_O, node_X)]
  elif motif_type == '030C':
    es = edges.copy()
    np.random.shuffle(es)
    node_P, node_O = es[0]
    triplets.remove(node_P)
    triplets.remove(node_O)
    node_X = triplets[0]
    edge_order = [(node_P, node_O), (node_O, node_X), (node_X, node_P)]
  elif motif_type == '201':
    node_P = most_common([i for i,j in edges])
    triplets.remove(node_P)
    np.random.shuffle(triplets)
    node_X, node_O = triplets
    edge_order = [(node_P, node_O), (node_O, node_P), (node_P, node_X), (node_X, node_P)]
  elif motif_type == '120D' or motif_type == '120U':
    if motif_type == '120D':
      node_X = most_common([i for i,j in edges])
    else:
      node_X = most_common([j for i,j in edges])
    triplets.remove(node_X)
    np.random.shuffle(triplets)
    node_P, node_O = triplets
    if motif_type == '120D':
      edge_order = [(node_X, node_P), (node_P, node_O), (node_X, node_O), (node_O, node_P)]
    else:
      edge_order = [(node_P, node_X), (node_P, node_O), (node_O, node_X), (node_O, node_P)]
  elif motif_type == '120C':
    node_P = most_common([i for i,j in edges])
    node_X = most_common([j for i,j in edges])
    triplets.remove(node_P)
    triplets.remove(node_X)
    node_O = triplets[0]
    edge_order = [(node_P, node_X), (node_X, node_P), (node_P, node_O), (node_O, node_X)]
  elif motif_type == '210':
    node_O = most_common([node for sub in edges for node in sub])
    triplets.remove(node_O)
    if tuple(triplets) in edges:
      node_P, node_X = triplets
    else:
      node_X, node_P = triplets
    edge_order = [(node_P, node_O), (node_O, node_P), (node_O, node_X), (node_X, node_O), (node_P, node_X)]
  elif motif_type == '300':
    np.random.shuffle(triplets)
    node_P, node_X, node_O = triplets
    edge_order = [(node_X, node_P), (node_P, node_X), (node_X, node_O), (node_O, node_X), (node_P, node_O), (node_O, node_P)]
  motif_sign = {edge:motif[edge[0]][edge[1]][weight] for edge in edges}
  sign = [motif_sign[edge] for edge in edge_order]
  sign = ''.join(map(lambda x:'+' if x > 0 else '-', sign))
  if motif_type in ['021D', '021U']:
    sign = ''.join(sorted(sign))
  # elif motif_type == '111D' or motif_type == '111U':
  #   sign = ''.join(sorted(sign[:2]) + sign[-2:]) # P<->X, O->X/X->O
  elif motif_type == '030C':
    sign = sorted([sign, sign[1:] + sign[:1], sign[2:] + sign[:2]])[0] # shift string
  elif motif_type in ['201', '120D', '120U']:
    sign = ''.join(sorted(sign[:2]) + sorted(sign[-2:])) # X->P/O, P<->O
  elif motif_type == '300':
    sign = ''.join([i for x in sorted([sorted(sign[:2]),sorted(sign[2:4]),sorted(sign[-2:])]) for i in x]) # X<->P, X<->O, P<->O
  return sign

def _tricode(G, v, u, w):
    """Returns the integer code of the given triad.

    This is some fancy magic that comes from Batagelj and Mrvar's paper. It
    treats each edge joining a pair of `v`, `u`, and `w` as a bit in
    the binary representation of an integer.

    """
    combos = ((v, u, 1), (u, v, 2), (v, w, 4), (w, v, 8), (u, w, 16),
              (w, u, 32))
    return sum(x for u, v, x in combos if v in G[u])

################### more efficient way of finding motifs
def find_triads(G):
  #: The integer codes representing each type of triad.
  #: Triads that are the same up to symmetry have the same code.
  TRICODES = (1, 2, 2, 3, 2, 4, 6, 8, 2, 6, 5, 7, 3, 8, 7, 11, 2, 6, 4, 8, 5, 9,
              9, 13, 6, 10, 9, 14, 7, 14, 12, 15, 2, 5, 6, 7, 6, 9, 10, 14, 4, 9,
              9, 12, 8, 13, 14, 15, 3, 7, 8, 11, 7, 12, 14, 15, 8, 14, 13, 15,
              11, 15, 15, 16)
  #: The names of each type of triad. The order of the elements is
  #: important: it corresponds to the tricodes given in :data:`TRICODES`.
  TRIAD_NAMES = ('003', '012', '102', '021D', '021U', '021C', '111D', '111U',
                '030T', '030C', '201', '120D', '120U', '120C', '210', '300')
  #: A dictionary mapping triad code to triad name.
  TRICODE_TO_NAME = {i: TRIAD_NAMES[code - 1] for i, code in enumerate(TRICODES)}
  triad_nodes = {name: set([]) for name in TRIAD_NAMES}
  m = {v: i for i, v in enumerate(G)}
  for v in G:
      vnbrs = set(G.pred[v]) | set(G.succ[v])
      for u in vnbrs:
          if m[u] > m[v]:
              unbrs = set(G.pred[u]) | set(G.succ[u])
              neighbors = (vnbrs | unbrs) - {u, v}
              not_neighbors = set(G.nodes()) - neighbors - {u, v}
              # Find dyadic triads
              for w in not_neighbors:
                  if v in G[u] and u in G[v]:
                      triad_nodes['102'].add(tuple(sorted([u, v, w])))
                  else:
                      triad_nodes['012'].add(tuple(sorted([u, v, w])))
              for w in neighbors:
                  if m[u] < m[w] or (m[v] < m[w] < m[u] and
                                    v not in G.pred[w] and
                                    v not in G.succ[w]):
                      code = _tricode(G, v, u, w)
                      triad_nodes[TRICODE_TO_NAME[code]].add(
                          tuple(sorted([u, v, w])))
  for triad_type in triad_nodes:
    if len(triad_nodes[triad_type]):
      G_list = []
      for triad in triad_nodes[triad_type]:
        G_list.append(G.subgraph(triad))
      triad_nodes[triad_type] = G_list
  # find null triads
  # all_tuples = set()
  # for s in triad_nodes.values():
  #     all_tuples = all_tuples.union(s)
  # triad_nodes['003'] = set(itertools.combinations(G.nodes(), 3)).difference(all_tuples)
  return triad_nodes

def get_motif_intensity_coherence(motif, weight='confidence'):
  edges = motif.edges()
  w_list = []
  for edge in edges:
    w_list.append(abs(motif[edge[0]][edge[1]][weight]))
  I = np.prod(w_list)**(1.0/len(w_list))
  intensity = I
  coherence = I / np.mean(w_list)
  return intensity, coherence

def get_signed_intensity_coherence(G_dict, motif_types):
  rows, cols = get_rowcol(G_dict)
  intensity_dict, coherence_dict = {}, {}
  for row_ind, row in enumerate(rows):
    print(row)
    intensity_dict[row], coherence_dict[row] = {}, {}
    for col_ind, col in enumerate(cols):
      print(col)
      intensity_dict[row][col], coherence_dict[row][col] = {}, {}
      G = G_dict[row][col]
      motifs_by_type = find_triads(G) # faster
      # motifs_by_type = nx.triads_by_type(G)
      for motif_type in motif_types:
        motifs = motifs_by_type[motif_type]
        # print(motif_type)
        for motif in motifs:
          intensity, coherence = get_motif_intensity_coherence(motif, weight='confidence')
          signed_motif_type = motif_type + get_motif_sign(motif, motif_type, weight='confidence')
          intensity_dict[row][col][signed_motif_type] = intensity_dict[row][col].get(signed_motif_type, 0) + intensity
          coherence_dict[row][col][signed_motif_type] = coherence_dict[row][col].get(signed_motif_type, 0) + coherence
      intensity_dict[row][col] = dict(sorted(intensity_dict[row][col].items(), key=lambda x:x[1], reverse=True))
      coherence_dict[row][col] = dict(sorted(coherence_dict[row][col].items(), key=lambda x:x[1], reverse=True))
  return intensity_dict, coherence_dict

def get_signed_intensity_coherence_baseline(G_dict, motif_types, algorithm='directed_double_edge_swap', num_baseline=10):
  rows, cols = get_rowcol(G_dict)
  intensity_dict, coherence_dict = {}, {}
  for row_ind, row in enumerate(rows):
    print(row)
    intensity_dict[row], coherence_dict[row] = {}, {}
    for col_ind, col in enumerate(cols):
      print(col)
      intensity_dict[row][col], coherence_dict[row][col] = defaultdict(lambda: np.zeros(num_baseline)), defaultdict(lambda: np.zeros(num_baseline))
      G = G_dict[row][col]
      random_graphs = random_graph_generator(G, num_rewire=num_baseline, algorithm=algorithm, weight='confidence', cc=False, Q=100)
      for g_ind, random_graph in tqdm(enumerate(random_graphs), total=num_baseline):
        # print(g_ind)
        motifs_by_type = find_triads(random_graph) # faster
        # motifs_by_type = nx.triads_by_type(random_graph)
        for motif_type in motif_types:
          motifs = motifs_by_type[motif_type]
          # print(motif_type)
          for motif in motifs:
            intensity, coherence = get_motif_intensity_coherence(motif, weight='confidence')
            signed_motif_type = motif_type + get_motif_sign(motif, motif_type, weight='confidence')
            intensity_dict[row][col][signed_motif_type][g_ind] += intensity
            coherence_dict[row][col][signed_motif_type][g_ind] += coherence
  return intensity_dict, coherence_dict

def defaultdict_to_dict(defaultdict):
  rows, cols = get_rowcol(defaultdict)
  out_dict = {}
  for row in rows:
    out_dict[row] = {}
    for col in cols:
      out_dict[row][col] = {}
      for k in defaultdict[row][col]:
        out_dict[row][col][k] = defaultdict[row][col][k]
  return out_dict

def tran2ffl(edge_order, triad_type):
  triads = []
  if triad_type == '030T':
    triads.append(edge_order)
  elif triad_type == '120D':
    triads.append([edge_order[1], edge_order[0], edge_order[2]])
    triads.append([edge_order[0], edge_order[1], edge_order[3]])
  elif triad_type == '120U':
    triads.append([edge_order[0], edge_order[2], edge_order[1]])
    triads.append([edge_order[1], edge_order[3], edge_order[0]])
  elif triad_type == '300':
    triads.append([edge_order[1], edge_order[4], edge_order[3]]) # P
    triads.append([edge_order[4], edge_order[1], edge_order[2]]) # P
    triads.append([edge_order[5], edge_order[3], edge_order[0]]) # O
    triads.append([edge_order[3], edge_order[5], edge_order[1]]) # O
    triads.append([edge_order[2], edge_order[0], edge_order[4]]) # X
    triads.append([edge_order[0], edge_order[2], edge_order[5]]) # X
  return triads

def signed_tran2ffl_census(G_dict, all_triads):
  rows, cols = get_rowcol(all_triads)
  signed_tran2ffl_count = {}
  for row_ind, row in enumerate(rows):
    print(row)
    signed_tran2ffl_count[row] = {}
    for col_ind, col in enumerate(cols):
      signed_tran2ffl_count[row][col] = {}
      G = G_dict[row][col]
      for triad in all_triads[row][col]:
        triad_type = triad[1]
        if triad_type == '030T':
          node_P = most_common([i for i,j in triad[0][0].keys()])
          node_X = most_common([j for i,j in triad[0][0].keys()])
          triplets = set([node for sub in triad[0][0].keys() for node in sub])
          triplets.remove(node_P)
          triplets.remove(node_X)
          node_O = list(triplets)[0]
          edge_order = [(node_P, node_X), (node_P, node_O), (node_O, node_X)]
        elif triad_type == '120D' or triad_type == '120U':
          if triad_type == '120D':
            node_X = most_common([i for i,j in triad[0][0].keys()])
          else:
            node_X = most_common([j for i,j in triad[0][0].keys()])
          triplets = set([node for sub in triad[0][0].keys() for node in sub])
          triplets.remove(node_X)
          triplets = list(triplets)
          np.random.shuffle(triplets)
          node_P, node_O = triplets
          if triad_type == '120D':
            edge_order = [(node_X, node_P), (node_X, node_O), (node_P, node_O), (node_O, node_P)]
          else:
            edge_order = [(node_P, node_X), (node_O, node_X), (node_P, node_O), (node_O, node_P)]
        else:
          triplets = list(set([node for sub in triad[0][0].keys() for node in sub]))
          np.random.shuffle(triplets)
          node_P, node_X, node_O = triplets
          edge_order = [(node_X, node_P), (node_P, node_X), (node_X, node_O), (node_O, node_X), (node_P, node_O), (node_O, node_P)]

        triads = tran2ffl(edge_order, triad_type)
        for e_order in triads:
          sign = [G.get_edge_data(*e)['sign'] for e in e_order]
          sign = ''.join(map(lambda x:'+' if x==1 else '-', sign))
          signed_tran2ffl_count[row][col]['030T' + sign] = signed_tran2ffl_count[row][col].get('030T' + sign, 0) + 1 
      signed_tran2ffl_count[row][col] = dict(sorted(signed_tran2ffl_count[row][col].items(), key=lambda x:x[1], reverse=True))
  return signed_tran2ffl_count

def signed_single_motif_census(all_triads, motif_type='030T'):
  rows, cols = get_rowcol(all_triads)
  signed_motif_count = {}
  for row_ind, row in enumerate(rows):
    print(row)
    signed_motif_count[row] = {}
    for col_ind, col in enumerate(cols):
      signed_motif_count[row][col] = {}
      for triad in all_triads[row][col]:
        triad_type = triad[1]
        if triad_type == motif_type == '030T':
          node_P = most_common([i for i,j in triad[0][0].keys()])
          node_X = most_common([j for i,j in triad[0][0].keys()])
          triplets = set([node for sub in triad[0][0].keys() for node in sub])
          triplets.remove(node_P)
          triplets.remove(node_X)
          node_O = list(triplets)[0]
          edge_order = [(node_P, node_X), (node_P, node_O), (node_O, node_X)]    
        elif triad_type == motif_type == '120D' or triad_type == motif_type == '120U':
          if triad_type == motif_type == '120D':
            node_X = most_common([i for i,j in triad[0][0].keys()])
          else:
            node_X = most_common([j for i,j in triad[0][0].keys()])
          triplets = set([node for sub in triad[0][0].keys() for node in sub])
          triplets.remove(node_X)
          triplets = list(triplets)
          np.random.shuffle(triplets)
          node_P, node_O = triplets
          if triad_type == motif_type == '120D':
            edge_order = [(node_X, node_P), (node_X, node_O), (node_P, node_O), (node_O, node_P)]
          else:
            edge_order = [(node_P, node_X), (node_O, node_X), (node_P, node_O), (node_O, node_P)]
        elif triad_type == motif_type == '300':
          triplets = list(set([node for sub in triad[0][0].keys() for node in sub]))
          np.random.shuffle(triplets)
          node_P, node_X, node_O = triplets
          edge_order = [(node_X, node_P), (node_P, node_X), (node_X, node_O), (node_O, node_X), (node_P, node_O), (node_O, node_P)]
        else:
          continue
        triad_sign = {k: v for d in triad[0] for k, v in d.items()}
        sign = [triad_sign[edge] for edge in edge_order]
        sign = ''.join(map(lambda x:'+' if x==1 else '-', sign))
        if triad_type == motif_type == '120D' or triad_type == motif_type == '120U':
          sign = ''.join(sorted(sign[:2]) + sorted(sign[-2:])) # X->P/O, P<->O
        elif triad_type == motif_type == '300':
          sign = ''.join([i for x in sorted([sorted(sign[:2]),sorted(sign[2:4]),sorted(sign[-2:])]) for i in x]) # X<->P, X<->O, P<->O
        signed_motif_count[row][col][triad_type + sign] = signed_motif_count[row][col].get(triad_type + sign, 0) + 1 
      signed_motif_count[row][col] = dict(sorted(signed_motif_count[row][col].items(), key=lambda x:x[1], reverse=True))
  return signed_motif_count

def right_temporal_order(edge_offset, triad_type):
  triads = []
  if triad_type == '030T':
    if edge_offset[1][2] < edge_offset[2][2]:
      triads.append([e[:-1] for e in edge_offset])
  elif triad_type == '120D':
    if edge_offset[0][2] < edge_offset[2][2]:
      triads.append([edge_offset[1][:-1], edge_offset[0][:-1], edge_offset[2][:-1]])
    if edge_offset[1][2] < edge_offset[3][2]:
      triads.append([edge_offset[0][:-1], edge_offset[1][:-1], edge_offset[3][:-1]])
  elif triad_type == '120U':
    if edge_offset[1][2] > edge_offset[2][2]:
      triads.append([edge_offset[0][:-1], edge_offset[2][:-1], edge_offset[1][:-1]])
    if edge_offset[0][2] > edge_offset[3][2]:
      triads.append([edge_offset[1][:-1], edge_offset[3][:-1], edge_offset[0][:-1]])
  elif triad_type == '300':
    if edge_offset[3][2] > edge_offset[4][2]: # P
      triads.append([edge_offset[1][:-1], edge_offset[4][:-1], edge_offset[3][:-1]])
    if edge_offset[2][2] > edge_offset[1][2]: # P
      triads.append([edge_offset[4][:-1], edge_offset[1][:-1], edge_offset[2][:-1]])
    if edge_offset[0][2] > edge_offset[3][2]: # O
      triads.append([edge_offset[5][:-1], edge_offset[3][:-1], edge_offset[0][:-1]])
    if edge_offset[1][2] > edge_offset[5][2]: # O
      triads.append([edge_offset[3][:-1], edge_offset[5][:-1], edge_offset[1][:-1]])
    if edge_offset[4][2] > edge_offset[0][2]: # X
      triads.append([edge_offset[2][:-1], edge_offset[0][:-1], edge_offset[4][:-1]])
    if edge_offset[5][2] > edge_offset[2][2]: # X
      triads.append([edge_offset[0][:-1], edge_offset[2][:-1], edge_offset[5][:-1]])
  return triads

def signed_temporal_030T_census(G_dict, all_triads):
  rows, cols = get_rowcol(all_triads)
  signed_temporal_030T_count = {}
  for row_ind, row in enumerate(rows):
    print(row)
    signed_temporal_030T_count[row] = {}
    for col_ind, col in enumerate(cols):
      signed_temporal_030T_count[row][col] = {}
      G = G_dict[row][col]
      for triad in all_triads[row][col]:
        triad_type = triad[1]
        if triad_type == '030T':
          node_P = most_common([i for i,j in triad[0][0].keys()])
          node_X = most_common([j for i,j in triad[0][0].keys()])
          triplets = set([node for sub in triad[0][0].keys() for node in sub])
          triplets.remove(node_P)
          triplets.remove(node_X)
          node_O = list(triplets)[0]
          edge_order = [(node_P, node_X), (node_P, node_O), (node_O, node_X)]
        elif triad_type == '120D' or triad_type == '120U':
          if triad_type == '120D':
            node_X = most_common([i for i,j in triad[0][0].keys()])
          else:
            node_X = most_common([j for i,j in triad[0][0].keys()])
          triplets = set([node for sub in triad[0][0].keys() for node in sub])
          triplets.remove(node_X)
          triplets = list(triplets)
          np.random.shuffle(triplets)
          node_P, node_O = triplets
          if triad_type == '120D':
            edge_order = [(node_X, node_P), (node_X, node_O), (node_P, node_O), (node_O, node_P)]
          else:
            edge_order = [(node_P, node_X), (node_O, node_X), (node_P, node_O), (node_O, node_P)]
        else:
          triplets = list(set([node for sub in triad[0][0].keys() for node in sub]))
          np.random.shuffle(triplets)
          node_P, node_X, node_O = triplets
          edge_order = [(node_X, node_P), (node_P, node_X), (node_X, node_O), (node_O, node_X), (node_P, node_O), (node_O, node_P)]
        edge_offset = []
        for e in edge_order:
          edge_offset.append((*e, G.get_edge_data(*e)['offset']))
        triads = right_temporal_order(edge_offset, triad_type)
        for e_order in triads:
          sign = [G.get_edge_data(*e)['sign'] for e in e_order]
          sign = ''.join(map(lambda x:'+' if x==1 else '-', sign))
          signed_temporal_030T_count[row][col]['030T' + sign] = signed_temporal_030T_count[row][col].get('030T' + sign, 0) + 1 
      signed_temporal_030T_count[row][col] = dict(sorted(signed_temporal_030T_count[row][col].items(), key=lambda x:x[1], reverse=True))
  return signed_temporal_030T_count

def triad_stimulus_error_region(triad_count, tran_triad_types, triad_color, measure, n, temporal=False):
  rows, cols = get_rowcol(triad_count)
  metric = np.zeros((len(rows), len(cols), len(tran_triad_types)))
  fig = plt.figure(figsize=(10, 8))
  # fig = plt.figure(figsize=(20, 10))
  for triad_ind, triad_type in enumerate(tran_triad_types):
    for row_ind, row in enumerate(rows):
      for col_ind, col in enumerate(cols):
        metric[row_ind, col_ind, triad_ind] = triad_count[row][col].get(triad_type, 0)
  metric_stimulus = pd.DataFrame(columns=['stimulus', 'type', 'mean', 'std'])
  for triad_ind, triad_type in enumerate(tran_triad_types):
    df = pd.DataFrame(columns=['stimulus', 'type', 'mean', 'std'])
    df['mean'] = np.nanmean(metric[:, :, triad_ind], axis=0)
    df['std'] = np.nanstd(metric[:, :, triad_ind], axis=0)
    df['type'] = triad_type
    df['stimulus'] = cols
    metric_stimulus = metric_stimulus.append(df, ignore_index=True)
  fig = plt.figure(figsize=[10, 6])
  axes1 = fig.add_subplot(111)
  # set props for left y-axis here
  axes2 = axes1.twinx()   # mirror them
  # axes2.set_ylabel()
  metric_stimulus['mean'] = metric_stimulus['mean'].astype('float')
  metric_stimulus['std'] = metric_stimulus['std'].astype('float')
  lns = []
  for i, m in metric_stimulus.groupby("type"):
    if m['type'].iloc[0] == '030T' or m['type'].iloc[0] == '030T+++':
      ln = axes2.plot(m['stimulus'], m['mean'], '--', alpha=0.6, label=m['type'].iloc[0], color=triad_color[m['type'].iloc[0]])
      axes2.fill_between(m['stimulus'], m['mean'] - m['std'], m['mean'] + m['std'], color=triad_color[m['type'].iloc[0]], alpha=0.2)
    else:
      ln = axes1.plot(m['stimulus'], m['mean'], alpha=0.6, label=m['type'].iloc[0], color=triad_color[m['type'].iloc[0]])
      axes1.fill_between(m['stimulus'], m['mean'] - m['std'], m['mean'] + m['std'], color=triad_color[m['type'].iloc[0]], alpha=0.2)
    lns += ln
  labs = [l.get_label() for l in lns]
  axes1.legend(lns, labs, loc=0)
  axes1.set_ylim(bottom=0)
  axes2.set_ylim(bottom=0)
  # plt.xticks(rotation=90)
  axes1.tick_params(axis='x', labelrotation=90)
  # plt.yscale('log')
  axes1.set_ylabel('number of triads')
  axes2.set_ylabel('number of 030T+++')
  
  if set(tran_triad_types) == set(['030T', '120D', '120U', '300']):
    triad_name = 'tran_triad'
  else:
    triad_name = '030T_triad'
  if temporal:
    triad_name = 'temporal_' + triad_name
  plt.title(triad_name + ' VS stimulus')
  plt.tight_layout()
  figname = './plots/num_{}_stimulus_error_region_{}_{}_fold.jpg'.format(triad_name, measure, n)
  plt.savefig(figname)

def plot_multi_bar_census(signed_triad_count, measure, n):
  rows, cols = get_rowcol(signed_triad_count)
  num_row, num_col = len(rows), len(cols)
  fig = plt.figure(figsize=(5*num_col, 3*num_row))
  for row_ind, row in enumerate(rows):
    for col_ind, col in enumerate(cols):
      plt.subplot(num_row, num_col, row_ind*num_col+col_ind+1)
      t_count = signed_triad_count[row][col]
      t_count = {k.replace(re.split('\+|\-',k)[0], re.split('\+|\-',k)[0]+'\n'): v/sum(t_count.values()) for k, v in t_count.items()} # add \n between triad type and signs
      t_count = {k: v/sum(t_count.values()) for k, v in t_count.items()} # normalize count as frequency
      t_count = {k: t_count[k] for k in list(t_count)[:5]} # only plot top 5
      plt.bar(range(len(t_count)), list(t_count.values()), align='center')
      plt.xticks(range(len(t_count)), list(t_count.keys()), fontsize=14)
      if row_ind == 0:
        plt.title(col, size=25)
  plt.tight_layout()
  # plt.suptitle(k, fontsize=14, rotation=0)
  # plt.show()
  figname = './plots/signed_triad_region_census_{}_{}fold.jpg'
  plt.savefig(figname.format(measure, n))

def plot_multi_pie_chart_census(triad_count, triad_types, triad_colormap, measure, n, sign=False):
  ind = 1
  rows, cols = get_rowcol(triad_count)
  hub_num = np.zeros((len(rows), len(cols)))
  fig = plt.figure(figsize=(4*len(cols), 3*len(rows)))
  # fig.patch.set_facecolor('black')
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      ax = plt.subplot(len(rows), len(cols), ind)
      if row_ind == 0:
        plt.gca().set_title(cols[col_ind], fontsize=20, rotation=0)
      if col_ind == 0:
        plt.gca().text(0, 0.5 * (bottom + top), rows[row_ind],
        horizontalalignment='left',
        verticalalignment='center',
        # rotation='vertical',
        transform=plt.gca().transAxes, fontsize=20, rotation=90)
      plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
      ind += 1
      t_count = triad_count[row][col].copy()
      t_count = dict(sorted(t_count.items(), key=lambda item: item[1], reverse=True))
      labels = t_count.keys()
      sizes = t_count.values()
      hub_num[row_ind][col_ind] = sum(sizes)
      explode = np.zeros(len(labels))  # only "explode" the 2nd slice (i.e. 'Hogs')
      vmax = 1 if len(triad_types) == 4 else 2.5
      norm = mpl.colors.Normalize(vmin=-1, vmax=vmax)
      specific_triad_colormap = {}
      for triad_type in triad_colormap:
        cmap = getattr(cm, triad_colormap[triad_type])
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        all_triad = [t for t in triad_types if triad_type in t]
        for st_ind in range(len(all_triad)):
          specific_triad_colormap[all_triad[st_ind]] = m.to_rgba(st_ind / (len(all_triad)-1)) if len(all_triad)-1 else m.to_rgba(0)
      colors = [specific_triad_colormap[l] for l in labels]
      # colors = [customPalette[triad_types.index(l)] for l in labels]
      patches, texts, pcts = plt.pie(sizes, radius=sum(sizes), explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
              shadow=True, startangle=90, wedgeprops={'linewidth': 3.0, 'edgecolor': 'white'})
      for i, patch in enumerate(patches):
        texts[i].set_color(patch.get_facecolor())
      # for i in range(len(p[0])):
      #   p[0][i].set_alpha(0.6)
      ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
  suptitle = 'triads distribution' if not sign else 'signed triads distribution'
  if set(triad_types) == set(['030T', '120D', '120U', '300']):
    suptitle = suptitle.replace('triads', 'transitive triads')
  plt.suptitle(suptitle, size=30)
  plt.tight_layout(rect=[0, 0.03, 1, 0.98])
  # plt.show()
  fname = './plots/pie_chart_all_triad_census_{}_{}fold.jpg' if not sign else './plots/pie_chart_signed_all_triad_census_{}_{}fold.jpg'
  if set(triad_types) == set(['030T', '120D', '120U', '300']):
    fname = fname.replace('all_triad', 'tran_triad')
  if len(rows) == 1:
    if rows[0] == 'sum':
      fname = fname.replace('triad', 'summice_triad')
    elif rows[0] == 'mean':
      fname = fname.replace('triad', 'meanmice_triad')
  plt.savefig(fname.format(measure, n))

def summice(triad_count):
  summice_triad_count = {'sum':{}}
  rows, cols = get_rowcol(triad_count)
  for col in cols:
    summice_triad_count['sum'][col] = {}
    for row in rows:
      for triad_type in triad_count[row][col]:
        summice_triad_count['sum'][col][triad_type] = summice_triad_count['sum'][col].get(triad_type, 0) + triad_count[row][col][triad_type]
  return summice_triad_count

def plot_multi_pie_chart_census_030T(triad_count, triad_types, measure, n, incoherent=False, temporal=False):
  ind = 1
  rows, cols = get_rowcol(triad_count)
  hub_num = np.zeros((len(rows), len(cols)))
  fig = plt.figure(figsize=(4*len(cols), 3*len(rows)))
  # fig.patch.set_facecolor('black')
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      ax = plt.subplot(len(rows), len(cols), ind)
      if row_ind == 0:
        plt.gca().set_title(cols[col_ind], fontsize=20, rotation=0)
      if col_ind == 0:
        plt.gca().text(0, 0.5 * (bottom + top), rows[row_ind],
        horizontalalignment='left',
        verticalalignment='center',
        # rotation='vertical',
        transform=plt.gca().transAxes, fontsize=20, rotation=90)
      plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
      ind += 1
      type_count = dict([(k, v) for k, v in triad_count[row][col].items() if '030T' in k])
      type_count = dict(sorted(type_count.items(), key=lambda x: x[1], reverse=True))
      if incoherent:
        type_count = dict([(k, v) for k, v in type_count.items() if '+++' not in k])
      labels = type_count.keys()
      sizes = [i / sum(type_count.values()) for i in type_count.values()]
      explode = np.zeros(len(labels))  # only "explode" the 2nd slice (i.e. 'Hogs')
      colors = [customPalette[triad_types.index(l)] for l in labels]
      patches, texts, pcts = plt.pie(sizes, radius=sum(sizes), explode=explode, colors=colors, labels=labels, autopct='%1.1f%%',
              shadow=True, startangle=90, wedgeprops={'linewidth': 3.0, 'edgecolor': 'white'})
      for i, patch in enumerate(patches):
        texts[i].set_color(patch.get_facecolor())
      # for i in range(len(p[0])):
      #   p[0][i].set_alpha(0.6)
      ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
  suptitle = 'signed 030T distribution' if not temporal else 'signed temporal 030T distribution'
  plt.suptitle(suptitle, size=30)
  plt.tight_layout(rect=[0, 0.03, 1, 0.98])
  # plt.show()
  fname = './plots/pie_chart_signed_030T_census_{}_{}fold.jpg' if not incoherent else './plots/pie_chart_signed_incoherent_030T_census_{}_{}fold.jpg'
  if temporal:
    fname = fname.replace('030T', 'temporal_030T')
  if len(rows) == 1:
    if rows[0] == 'sum':
      fname = fname.replace('030T', 'summice_030T')
    elif rows[0] == 'mean':
      fname = fname.replace('030T', 'meanmice_030T')
  plt.savefig(fname.format(measure, n))

def signed_triad_region_census(all_triads, area_dict):
  rows, cols = get_rowcol(all_triads)
  signed_triad_region_count = {}
  for row_ind, row in enumerate(rows):
    print(row)
    signed_triad_region_count[row] = {}
    node_area = area_dict[row]
    for col_ind, col in enumerate(cols):
      signed_triad_region_count[row][col] = {}
      for triad in all_triads[row][col]:
        triad_type = triad[1]
        if triad_type == '030T':
          node_P = most_common([i for i,j in triad[0][0].keys()])
          node_X = most_common([j for i,j in triad[0][0].keys()])
          triplets = set([node for sub in triad[0][0].keys() for node in sub])
          triplets.remove(node_P)
          triplets.remove(node_X)
          node_O = list(triplets)[0]
          edge_order = [(node_P, node_X), (node_P, node_O), (node_O, node_X)]    
        elif triad_type == '120D' or triad_type == '120U':
          if triad_type == '120D':
            node_X = most_common([i for i,j in triad[0][0].keys()])
          else:
            node_X = most_common([j for i,j in triad[0][0].keys()])
          triplets = set([node for sub in triad[0][0].keys() for node in sub])
          triplets.remove(node_X)
          triplets = list(triplets)
          np.random.shuffle(triplets)
          node_P, node_O = triplets
          if triad_type == '120D':
            edge_order = [(node_X, node_P), (node_X, node_O), (node_P, node_O), (node_O, node_P)]
          else:
            edge_order = [(node_P, node_X), (node_O, node_X), (node_P, node_O), (node_O, node_P)]
        else:
          triplets = list(set([node for sub in triad[0][0].keys() for node in sub]))
          np.random.shuffle(triplets)
          node_P, node_X, node_O = triplets
          edge_order = [(node_X, node_P), (node_P, node_X), (node_X, node_O), (node_O, node_X), (node_P, node_O), (node_O, node_P)]
        region = [node_area[n] for n in [node_P, node_O, node_X]]
        region = '_'.join(region)
        triad_sign = {k: v for d in triad[0] for k, v in d.items()}
        sign = [triad_sign[edge] for edge in edge_order]
        sign = ''.join(map(lambda x:'+' if x==1 else '-', sign))
        if triad_type == '120D' or triad_type == '120U':
          sign = ''.join(sorted(sign[:2]) + sorted(sign[-2:])) # X->P/O, P<->O
        elif triad_type == '300':
          sign = ''.join([i for x in sorted([sorted(sign[:2]),sorted(sign[2:4]),sorted(sign[-2:])]) for i in x]) # X<->P, X<->O, P<->O
        if triad_type + sign not in signed_triad_region_count[row][col]:
          signed_triad_region_count[row][col][triad_type + sign] = {}
        signed_triad_region_count[row][col][triad_type + sign][region] = signed_triad_region_count[row][col][triad_type + sign].get(region, 0) + 1 
      for st_type in signed_triad_region_count[row][col]:
        signed_triad_region_count[row][col][st_type] = dict(sorted(signed_triad_region_count[row][col][st_type].items(), key=lambda x:x[1], reverse=True))
  return signed_triad_region_count

def get_all_signed_transitive_triads_Glist(G_list):
  all_triads = [[] for _ in range(len(G_list))]
  for G_ind, G in enumerate(G_list):
    triad_dict = {}
    triad_class = {}
    ## there are only 4 transistive census: 030T, 120D, 120U, and 300 
    non_transitive_census = ['003','012', '102', '021D', '021C', '021U', '111U', '111D', '201', '030C', '120C', '210'] # , '021'
    iter_g = search_triangles(G)
    for iter_t in iter_g:
      for ta in list(iter_t):
        tt = ",".join([str(x) for x in sorted(set(ta))])
        triad_dict[tt] = True     
    for val in triad_dict.keys():
      nodes = [int(x) for x in val.split(",")]
      census = [k for k, v in nx.triads.triadic_census(G.subgraph(nodes)).items() if v][0]
      if census not in non_transitive_census:
        sign = nx.get_edge_attributes(G.subgraph(nodes),'sign')
        triad_class[val] = [census, sign]
        #map_census_edges(G, val, triad_class)     
    for key, value in triad_class.items():
      all_directed_triads = list(get_directed_triads(value[1]))
      all_triads[G_ind].append([all_directed_triads, value[0]])
  return all_triads

def signed_single_motif_census_Glist(all_triads, motif_type='030T'):
  signed_motif_count = {}
  for G_ind in range(len(all_triads)):
    for triad in all_triads[G_ind]:
      triad_type = triad[1]
      if triad_type == motif_type == '030T':
        node_P = most_common([i for i,j in triad[0][0].keys()])
        node_X = most_common([j for i,j in triad[0][0].keys()])
        triplets = set([node for sub in triad[0][0].keys() for node in sub])
        triplets.remove(node_P)
        triplets.remove(node_X)
        node_O = list(triplets)[0]
        edge_order = [(node_P, node_X), (node_P, node_O), (node_O, node_X)]    
      elif triad_type == motif_type == '120D' or triad_type == motif_type == '120U':
        if triad_type == motif_type == '120D':
          node_X = most_common([i for i,j in triad[0][0].keys()])
        else:
          node_X = most_common([j for i,j in triad[0][0].keys()])
        triplets = set([node for sub in triad[0][0].keys() for node in sub])
        triplets.remove(node_X)
        triplets = list(triplets)
        np.random.shuffle(triplets)
        node_P, node_O = triplets
        if triad_type == motif_type == '120D':
          edge_order = [(node_X, node_P), (node_X, node_O), (node_P, node_O), (node_O, node_P)]
        else:
          edge_order = [(node_P, node_X), (node_O, node_X), (node_P, node_O), (node_O, node_P)]
      elif triad_type == motif_type == '300':
        triplets = list(set([node for sub in triad[0][0].keys() for node in sub]))
        np.random.shuffle(triplets)
        node_P, node_X, node_O = triplets
        edge_order = [(node_X, node_P), (node_P, node_X), (node_X, node_O), (node_O, node_X), (node_P, node_O), (node_O, node_P)]
      else:
        continue
      triad_sign = {k: v for d in triad[0] for k, v in d.items()}
      sign = [triad_sign[edge] for edge in edge_order]
      sign = ''.join(map(lambda x:'+' if x==1 else '-', sign))
      if triad_type == motif_type == '120D' or triad_type == motif_type == '120U':
        sign = ''.join(sorted(sign[:2]) + sorted(sign[-2:])) # X->P/O, P<->O
      elif triad_type == motif_type == '300':
        sign = ''.join([i for x in sorted([sorted(sign[:2]),sorted(sign[2:4]),sorted(sign[-2:])]) for i in x]) # X<->P, X<->O, P<->O
      signed_motif_count[triad_type + sign] = signed_motif_count.get(triad_type + sign, 0) + 1/len(all_triads)
  signed_motif_count = dict(sorted(signed_motif_count.items(), key=lambda x:x[1], reverse=True))
  return signed_motif_count

def get_signed_motif_count_baseline(G_dict, num_rewire, motif_type='030T'):
  rows, cols = get_rowcol(G_dict)
  signed_motif_count_gnm, signed_motif_count_config = defaultdict(lambda: {}), defaultdict(lambda: {})
  for row in rows:
    print(row)
    for col in cols:
      print(col)
      G = G_dict[row][col]
      random_Gs = random_graph_generator(G, num_rewire, algorithm='Gnm', weight='sign')
      all_triads_baseline = get_all_signed_transitive_triads_Glist(random_Gs)
      signed_motif_count_gnm[row][col] = signed_single_motif_census_Glist(all_triads_baseline, motif_type)
      random_Gs = random_graph_generator(G, num_rewire, algorithm='directed_configuration_model', weight='sign')
      all_triads_baseline = get_all_signed_transitive_triads_Glist(random_Gs)
      signed_motif_count_config[row][col] = signed_single_motif_census_Glist(all_triads_baseline, motif_type)
  return signed_motif_count_gnm, signed_motif_count_config

def signed_tran2ffl_census_Glist(G_list, all_triads):
  signed_tran2ffl_count = {}
  for G_ind, G in enumerate(G_list):
    for triad in all_triads[G_ind]:
      triad_type = triad[1]
      if triad_type == '030T':
        node_P = most_common([i for i,j in triad[0][0].keys()])
        node_X = most_common([j for i,j in triad[0][0].keys()])
        triplets = set([node for sub in triad[0][0].keys() for node in sub])
        triplets.remove(node_P)
        triplets.remove(node_X)
        node_O = list(triplets)[0]
        edge_order = [(node_P, node_X), (node_P, node_O), (node_O, node_X)]
      elif triad_type == '120D' or triad_type == '120U':
        if triad_type == '120D':
          node_X = most_common([i for i,j in triad[0][0].keys()])
        else:
          node_X = most_common([j for i,j in triad[0][0].keys()])
        triplets = set([node for sub in triad[0][0].keys() for node in sub])
        triplets.remove(node_X)
        triplets = list(triplets)
        np.random.shuffle(triplets)
        node_P, node_O = triplets
        if triad_type == '120D':
          edge_order = [(node_X, node_P), (node_X, node_O), (node_P, node_O), (node_O, node_P)]
        else:
          edge_order = [(node_P, node_X), (node_O, node_X), (node_P, node_O), (node_O, node_P)]
      else:
        triplets = list(set([node for sub in triad[0][0].keys() for node in sub]))
        np.random.shuffle(triplets)
        node_P, node_X, node_O = triplets
        edge_order = [(node_X, node_P), (node_P, node_X), (node_X, node_O), (node_O, node_X), (node_P, node_O), (node_O, node_P)]

      triads = tran2ffl(edge_order, triad_type)
      for e_order in triads:
        sign = [G.get_edge_data(*e)['sign'] for e in e_order]
        sign = ''.join(map(lambda x:'+' if x==1 else '-', sign))
        signed_tran2ffl_count['030T' + sign] = signed_tran2ffl_count.get('030T' + sign, 0) + 1/len(G_list)
  signed_tran2ffl_count = dict(sorted(signed_tran2ffl_count.items(), key=lambda x:x[1], reverse=True))
  return signed_tran2ffl_count

def get_signed_tran2ffl_count_baseline(G_dict, num_rewire):
  rows, cols = get_rowcol(G_dict)
  signed_tran2ffl_count_gnm, signed_tran2ffl_count_config = defaultdict(lambda: {}), defaultdict(lambda: {})
  for row in rows:
    print(row)
    for col in cols:
      print(col)
      G = G_dict[row][col]
      random_Gs = random_graph_generator(G, num_rewire, algorithm='Gnm', weight='sign')
      all_triads_baseline = get_all_signed_transitive_triads_Glist(random_Gs)
      signed_tran2ffl_count_gnm[row][col] = signed_tran2ffl_census_Glist(random_Gs, all_triads_baseline)
      random_Gs = random_graph_generator(G, num_rewire, algorithm='directed_configuration_model', weight='sign')
      all_triads_baseline = get_all_signed_transitive_triads_Glist(random_Gs)
      signed_tran2ffl_count_config[row][col] = signed_tran2ffl_census_Glist(random_Gs, all_triads_baseline)
  return signed_tran2ffl_count_gnm, signed_tran2ffl_count_config

def plot_signed_motif_relative_count_baseline(signed_motif_count, signed_motif_count_gnm, signed_motif_count_config, signed_030T_triad_types, triad_type, measure, n):
  rows, cols = get_rowcol(signed_motif_count)
  xlabel = '{} type'.format(triad_type)
  fig = plt.figure(figsize=(23, 15))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  for col_ind, col in enumerate(cols):
    df = pd.DataFrame(columns=['motif type', 'relative count', 'baseline'])
    plt.subplot(4, 2, col_ind+1)
    plt.gca().set_title(col, fontsize=20, rotation=0)
    plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    for t_ind, t in enumerate(signed_030T_triad_types):
      rc_gnm, rc_config = [], []
      for row in rows:
        rc_gnm.append(safe_division(signed_motif_count[row][col].get(t, 0), signed_motif_count_gnm[row][col].get(t, 0)))
        rc_config.append(safe_division(signed_motif_count[row][col].get(t, 0), signed_motif_count_config[row][col].get(t, 0)))
      df = pd.concat([df, pd.DataFrame(np.concatenate((np.array([t] * len(rc_gnm))[:,None], np.array(rc_gnm)[:,None], np.array(['Gnm'] * len(rc_gnm))[:,None]), 1), columns=[xlabel, 'relative count', 'baseline'])], ignore_index=True)
      df = pd.concat([df, pd.DataFrame(np.concatenate((np.array([t] * len(rc_config))[:,None], np.array(rc_config)[:,None], np.array(['directed configuration model'] * len(rc_config))[:,None]), 1), columns=[xlabel, 'relative count', 'baseline'])], ignore_index=True)
    df['relative count'] = pd.to_numeric(df['relative count'])
    ax = sns.barplot(x=xlabel, y='relative count', hue='baseline', data=df)
    if triad_type == '030T':
      ax.set_xticks(range(len(signed_030T_triad_types)))
      ax.set_xticklabels(['C1', 'I1', 'I4', 'I3', 'C4', 'C3', 'C2', 'I2'])
    # ax.set_xticks(range(len(df)), labels=['C1', 'I1', 'I4', 'I3', 'C4', 'C3', 'C2', 'I2'])
    left, right = plt.xlim()
    plt.hlines(1, xmin=left, xmax=right, color='r', linestyles='--', linewidth=1, alpha=1)
    plt.yscale('log')
    if col_ind < len(cols) - 1:
      plt.legend([],[], frameon=False)
  plt.suptitle('Relative count of all signed {}'.format(triad_type), size=30)
  plt.tight_layout(rect=[0, 0.03, 1, 0.98])
  image_name = './plots/relative_count_{}_baseline_{}_{}fold.jpg'.format(triad_type, measure, n)
  # plt.show()
  plt.savefig(image_name)

def signed_temporal_030T_region_census(G_dict, all_triads, area_dict):
  rows, cols = get_rowcol(all_triads)
  signed_temporal_030T_count = {}
  for row_ind, row in enumerate(rows):
    print(row)
    signed_temporal_030T_count[row] = {}
    node_area = area_dict[row]
    for col_ind, col in enumerate(cols):
      signed_temporal_030T_count[row][col] = {}
      G = G_dict[row][col]
      for triad in all_triads[row][col]:
        triad_type = triad[1]
        if triad_type == '030T':
          node_P = most_common([i for i,j in triad[0][0].keys()])
          node_X = most_common([j for i,j in triad[0][0].keys()])
          triplets = set([node for sub in triad[0][0].keys() for node in sub])
          triplets.remove(node_P)
          triplets.remove(node_X)
          node_O = list(triplets)[0]
          edge_order = [(node_P, node_X), (node_P, node_O), (node_O, node_X)]
        elif triad_type == '120D' or triad_type == '120U':
          if triad_type == '120D':
            node_X = most_common([i for i,j in triad[0][0].keys()])
          else:
            node_X = most_common([j for i,j in triad[0][0].keys()])
          triplets = set([node for sub in triad[0][0].keys() for node in sub])
          triplets.remove(node_X)
          triplets = list(triplets)
          np.random.shuffle(triplets)
          node_P, node_O = triplets
          if triad_type == '120D':
            edge_order = [(node_X, node_P), (node_X, node_O), (node_P, node_O), (node_O, node_P)]
          else:
            edge_order = [(node_P, node_X), (node_O, node_X), (node_P, node_O), (node_O, node_P)]
        else:
          triplets = list(set([node for sub in triad[0][0].keys() for node in sub]))
          np.random.shuffle(triplets)
          node_P, node_X, node_O = triplets
          edge_order = [(node_X, node_P), (node_P, node_X), (node_X, node_O), (node_O, node_X), (node_P, node_O), (node_O, node_P)]
        edge_offset = []
        for e in edge_order:
          edge_offset.append((*e, G.get_edge_data(*e)['offset']))
        triads = right_temporal_order(edge_offset, triad_type)
        for e_order in triads:
          sign = [G.get_edge_data(*e)['sign'] for e in e_order]
          sign = ''.join(map(lambda x:'+' if x==1 else '-', sign))
          region = [node_area[n] for n in [e_order[0][0], e_order[1][1], e_order[0][1]]]
          region = '_'.join(region)
          if triad_type + sign not in signed_temporal_030T_count[row][col]:
            signed_temporal_030T_count[row][col]['030T' + sign] = {}
          signed_temporal_030T_count[row][col]['030T' + sign][region] = signed_temporal_030T_count[row][col]['030T' + sign].get(region, 0) + 1 
      for st_type in signed_temporal_030T_count[row][col]:
        signed_temporal_030T_count[row][col][st_type] = dict(sorted(signed_temporal_030T_count[row][col][st_type].items(), key=lambda x:x[1], reverse=True))
  return signed_temporal_030T_count

def plot_pie_chart_region_census_030T(signed_triad_region_count, triad_types, region_types, measure, n, temporal=False):
  rows, cols = get_rowcol(signed_triad_region_count)
  triad_region_dict = defaultdict(lambda: {})
  fig = plt.figure(figsize=(32,4))
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      for triad_type in signed_triad_region_count[row][col]:
        temp = signed_triad_region_count[row][col][triad_type]
        for region in temp:
          triad_region_dict[triad_type][region] = triad_region_dict[triad_type].get(region, 0) + temp[region]
  all_regions = np.unique([i for x in [list(triad_region_dict[t].keys()) for t in triad_region_dict] for i in x])
  region_rank_dict = {i:region_types.index(i) for i in all_regions}
  region_rank_dict = dict(sorted(region_rank_dict.items(), key=lambda item: item[1]))
  all_regions = list(region_rank_dict.keys())
  for t_ind, triad_type in enumerate(triad_types):
    ax = plt.subplot(1, len(triad_types), t_ind+1)
    triad_region = triad_region_dict[triad_type]
    triad_region = dict(sorted(triad_region.items(), key=lambda item: item[1], reverse=True))
    triad_region = {k:v for k,v in triad_region.items() if v >= 2} # remove region type that appears only once
    labels = triad_region.keys()
    sizes = [i / sum(triad_region.values()) for i in triad_region.values()]
    explode = np.zeros(len(labels))  # only "explode" the 2nd slice (i.e. 'Hogs')
    colors = [customPalette[all_regions.index(l)%len(customPalette)] for l in labels]
    patches, texts, pcts = plt.pie(sizes, radius=sum(sizes), explode=explode, colors=colors, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90, wedgeprops={'linewidth': 3.0, 'edgecolor': 'white'})
    for i, patch in enumerate(patches):
      texts[i].set_color(patch.get_facecolor())
    # for i in range(len(p[0])):
    #   p[0][i].set_alpha(0.6)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title(triad_type)
  suptitle = 'signed 030T distribution' if not temporal else 'signed temporal 030T distribution'
  plt.suptitle(suptitle, size=30)
  plt.tight_layout(rect=[0, 0.03, 1, 0.98])
  # plt.show()
  fname = './plots/pie_chart_signed_030T_region_census_{}_{}fold.jpg' if not temporal else './plots/pie_chart_signed_temporal_030T_region_census_{}_{}fold.jpg'
  plt.savefig(fname.format(measure, n))

def plot_stimulus_pie_chart_region_census_030T(signed_triad_region_count, triad_types, region_types, measure, n):
  rows, cols = get_rowcol(signed_triad_region_count)
  triad_region_dict = {}
  fig = plt.figure(figsize=(32,26))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  for col in cols:
    print(col)
    triad_region_dict[col] = defaultdict(lambda: {})
    for row in rows:
      for triad_type in signed_triad_region_count[row][col]:
        temp = signed_triad_region_count[row][col][triad_type]
        for region in temp:
          triad_region_dict[col][triad_type][region] = triad_region_dict[col][triad_type].get(region, 0) + temp[region]
  # return triad_region_dict
  all_regions = np.unique([i for x in [list(triad_region_dict[col][t].keys()) for col in triad_region_dict for t in triad_region_dict[col]] for i in x])
  region_rank_dict = {i:region_types.index(i) for i in all_regions}
  region_rank_dict = dict(sorted(region_rank_dict.items(), key=lambda item: item[1]))
  all_regions = list(region_rank_dict.keys())
  for col_ind, col in enumerate(cols):
    for t_ind, triad_type in enumerate(triad_types):
      ax = plt.subplot(len(cols), len(triad_types), col_ind*len(triad_types)+t_ind+1)
      triad_region = triad_region_dict[col][triad_type]
      triad_region = dict(sorted(triad_region.items(), key=lambda item: item[1], reverse=True))
      # triad_region = {k:v for k,v in triad_region.items() if v >= 6} # remove region type that appears only once
      labels = triad_region.keys()
      sizes = [i / sum(triad_region.values()) for i in triad_region.values()]
      explode = np.zeros(len(labels))  # only "explode" the 2nd slice (i.e. 'Hogs')
      colors = [customPalette[all_regions.index(l)%len(customPalette)] for l in labels]
      patches, texts, pcts = plt.pie(sizes, radius=sum(sizes), explode=explode, colors=colors, labels=labels, autopct='%1.1f%%',
              shadow=True, startangle=90, wedgeprops={'linewidth': 3.0, 'edgecolor': 'white'})
      for i, patch in enumerate(patches):
        texts[i].set_color(patch.get_facecolor())
      # for i in range(len(p[0])):
      #   p[0][i].set_alpha(0.6)
      ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
      plt.title(triad_type)
      if t_ind == 0:
        plt.gca().text(0, 0.5 * (bottom + top), col,
        horizontalalignment='left',
        verticalalignment='center',
        # rotation='vertical',
        transform=plt.gca().transAxes, fontsize=20, rotation=90)
  suptitle = 'signed 030T region distribution'
  plt.suptitle(suptitle, size=30)
  plt.tight_layout(rect=[0, 0.03, 1, 0.98])
  # plt.show()
  fname = './plots/pie_chart_stimulus_signed_030T_region_census_{}_{}fold.jpg'
  plt.savefig(fname.format(measure, n))

def count_sign_p(G):
  num_pos, num_neg = 0, 0
  edges = list(G.edges())
  for edge in edges:
    if G.get_edge_data(*edge)['sign'] == 1:
      num_pos += 1
    elif G.get_edge_data(*edge)['sign'] == -1:
      num_neg += 1
    else:
      sys.exit(G.get_edge_data(*edge)['sign'])
  assert num_pos + num_neg == len(edges)
  p_pos, p_neg = safe_division(num_pos, len(edges)), safe_division(num_neg, len(edges))
  return p_pos, p_neg

# p_pos and p_neg for 030T only
def count_030T_sign_p(G):
  all_signs = []
  G_all_triads = nx.all_triads(G)
  for triad in G_all_triads:
    if nx.triad_type(triad) == '030T':
      t_nodes = list(triad.nodes())
      edge_sign = nx.get_edge_attributes(G.subgraph(t_nodes),'sign')
      signs = list(edge_sign.values())
      assert len(signs) == 3
      all_signs += signs
  num_pos, num_neg = all_signs.count(1), all_signs.count(-1)
  assert num_pos + num_neg == len(all_signs)
  p_pos, p_neg = safe_division(num_pos, len(all_signs)), safe_division(num_neg, len(all_signs))
  return p_pos, p_neg

def plot_030T_relative_count(G_dict, signed_triad_count, signed_030T_triad_types, p_sign_func, measure, n):
  ind = 1
  rows, cols = get_rowcol(G_dict)
  fig = plt.figure(figsize=(23, 15))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  for col in cols:
    print(col)
    plt.subplot(4, 2, ind)
    plt.gca().set_title(col, fontsize=20, rotation=0)
    plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ind += 1
    all_triad_count = defaultdict(lambda: [])
    for row in rows:
      print(row)
      type_count = dict([(k, v) for k, v in signed_triad_count[row][col].items() if '030T' in k])
      type_count = dict([(k, v/sum(type_count.values())) for k, v in type_count.items()])
      G = G_dict[row][col].copy() if col in G_dict[row] else nx.DiGraph()
      p_pos, p_neg = count_030T_sign_p(G)
      for signed_triad_type in type_count:
        relative_c = safe_division(type_count[signed_triad_type], p_sign_func[signed_triad_type](p_pos, p_neg))
        all_triad_count[signed_triad_type].append(relative_c)
    
    avalaible_signed_030T_triad_types = [t for t in signed_030T_triad_types if t in all_triad_count]
    triad_types, triad_counts = avalaible_signed_030T_triad_types, [all_triad_count[k] for k in avalaible_signed_030T_triad_types]
    plt.boxplot(triad_counts)
    plt.xticks(list(range(1, len(triad_counts)+1)), triad_types, rotation=0)
    left, right = plt.xlim()
    plt.hlines(1, xmin=left, xmax=right, color='r', linestyles='--', linewidth=0.5, alpha=0.6)
    # plt.hist(data.flatten(), bins=12, density=True)
    # plt.axvline(x=np.nanmean(data), color='r', linestyle='--')
    # plt.xlabel('region')
    # plt.xlabel('size')
    if max([i for x in triad_counts for i in x]) > 20:
      plt.yscale('log')
    plt.ylabel('relative count')
  plt.suptitle('Relative count of all signed 030T', size=30)
  plt.tight_layout(rect=[0, 0.03, 1, 0.98])
  image_name = './plots/relative_count_030T_{}_{}fold.jpg'.format(measure, n)
  # plt.show()
  plt.savefig(image_name)

def triad300_bidirectional_edge_census(data_dict, G_dict, active_area_dict, max_duration):
  rows, cols = get_rowcol(data_dict)
  scale = np.zeros(len(rows))
  data_mat = np.zeros((len(rows), len(cols), max_duration+1, max_duration+1))
  for row_ind, row in enumerate(rows):
    print(row)
    active_area = active_area_dict[row]
    node_idx = sorted(active_area.keys())
    for col_ind, col in enumerate(cols):
      print(col)
      data, G = data_dict[row][col].copy(), G_dict[row][col].copy()
      nodes = sorted(list(G.nodes()))
      G_all_triads = nx.all_triads(G)
      for triad in G_all_triads:
        if nx.triad_type(triad) == '300':
          t_nodes = list(triad.nodes())
          active_idx = np.array([node_idx.index(n) for n in t_nodes])
          for node_i, node_j in itertools.combinations(active_idx, 2):
            data_A, data_B = int(data[node_i, node_j]), int(data[node_j, node_i])
            data_mat[row_ind, col_ind, data_A, data_B] += 1
            data_mat[row_ind, col_ind, data_B, data_A] += 1
      data_mat[row_ind, col_ind, :, :] = safe_division(data_mat[row_ind, col_ind, :, :], data_mat[row_ind, col_ind, :, :].sum())
    scale[row_ind] = data_mat[row_ind, :, :, :].max()
  return scale, data_mat

def plot_data_heatmap(scale, data_mat, max_duration, rows, cols, name, measure, n):
  ind = 1
  fig = plt.figure(figsize=(3.2*len(cols), 3*len(rows)))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      plt.subplot(len(rows), len(cols), ind)
      if row_ind == 0:
        plt.gca().set_title(cols[col_ind], fontsize=20, rotation=0)
      if col_ind == 0:
        plt.gca().text(0, 0.5 * (bottom + top), rows[row_ind],
        horizontalalignment='left',
        verticalalignment='center',
        # rotation='vertical',
        transform=plt.gca().transAxes, fontsize=20, rotation=90)
      cbar = True if col_ind == len(cols) - 1 else False
      plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
      ind += 1
      sns_plot = sns.heatmap(data_mat[row_ind, col_ind, :, :].astype(float), vmin=0, vmax=scale[row_ind],center=0,cmap="RdBu_r", cbar=cbar)# cmap="YlGnBu" , norm=colors.LogNorm()
      sns_plot.set_xticks(np.arange(max_duration+1)+0.5)
      sns_plot.set_xticklabels(list(range(max_duration+1)), rotation=90)
      sns_plot.set_yticks(np.arange(max_duration+1)+0.5)
      sns_plot.set_yticklabels(list(range(max_duration+1)), rotation=0)
      sns_plot.invert_yaxis()
  plt.suptitle('triad 300 {} distribution'.format(name), size=30)
  plt.tight_layout(rect=[0, 0.03, 1, 0.98])
  # plt.show()
  plt.savefig('./plots/triad300_{}_{}_{}fold.jpg'.format(name.replace(' ', '_'), measure, n))

#################### micro level structural balance
def plot_balance_stat(rows, cols, t_balance, num_balance, num_imbalance, n, measure='xcorr'):
  num_col = 3
  # fig = plt.figure(figsize=(5*num_col, 25))
  fig = plt.figure(figsize=(5*num_col, 5))
  metrics = {'Triad Level Balance':t_balance,
  'Number of balance and transitive triads':num_balance,
  'Number of imbalance and transitive triads':num_imbalance}
  
  for i, k in enumerate(metrics):
    plt.subplot(1, num_col, i+1)
    metric = metrics[k]
    for row_ind, row in enumerate(rows):
      mean = metric[row_ind, :]
      plt.plot(cols, mean, label=row, alpha=0.6)
    plt.gca().set_title(k, fontsize=15, rotation=0)
    plt.xticks(rotation=90)
    # plt.yscale('symlog')
    if i == len(metrics)-1:
      plt.legend()
    plt.tight_layout()
  # plt.show()
  figname = './plots/triad_balance_stats_{}_{}fold.jpg'.format(measure, n)
  plt.savefig(figname)

def plot_balance_pie_chart(balance_t_counts, name, tran_traid_census):
  ind = 1
  rows, cols = get_rowcol(balance_t_counts)
  hub_num = np.zeros((len(rows), len(cols)))
  fig = plt.figure(figsize=(4*len(cols), 3*len(rows)))
  # fig.patch.set_facecolor('black')
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      ax = plt.subplot(len(rows), len(cols), ind)
      if row_ind == 0:
        plt.gca().set_title(cols[col_ind], fontsize=20, rotation=0)
      if col_ind == 0:
        plt.gca().text(0, 0.5 * (bottom + top), rows[row_ind],
        horizontalalignment='left',
        verticalalignment='center',
        # rotation='vertical',
        transform=plt.gca().transAxes, fontsize=20, rotation=90)
      plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
      ind += 1
      labels = balance_t_counts[row][col].keys()
      sizes = balance_t_counts[row][col].values()
      hub_num[row_ind][col_ind] = sum(sizes)
      explode = np.zeros(len(labels))  # only "explode" the 2nd slice (i.e. 'Hogs')
      colors = [customPalette[tran_traid_census.index(l)] for l in labels]
      patches, texts, pcts = plt.pie(sizes, radius=sum(sizes), explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
              shadow=True, startangle=90, wedgeprops={'linewidth': 3.0, 'edgecolor': 'white'})
      for i, patch in enumerate(patches):
        texts[i].set_color(patch.get_facecolor())
      # for i in range(len(p[0])):
      #   p[0][i].set_alpha(0.6)
      ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
  plt.suptitle('{} and transitive triads distribution'.format(name), size=30)
  plt.tight_layout()
  # plt.show()
  fname = './plots/pie_chart_{}.jpg'
  plt.savefig(fname.format(name))

##################### triad region census
def most_common(lst):
    return max(set(lst), key=lst.count)

def triad_region_census(triads, triad_type, area_dict, regions, measure, n, name):
  rows, cols = get_rowcol(triads)
  num_row = len(rows)
  if triad_type == '300':
    num_col = 1
    pos = ['P/X/O']
  else:
    num_col = len(cols)
    if triad_type == '030T':
      pos = ['P', 'X', 'O']
    elif triad_type == '120D' or triad_type == '120U':
      pos = ['X', 'P/O']
  metric = np.zeros((len(rows), len(cols), len(pos), len(regions)))
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      region_list = []
      for _ in range(len(pos)):
        region_list.append([])
      for triad in triads[row][col]:
        if triad[1] == triad_type:
          if triad_type == '030T':
            node_P = most_common([i for i,j in triad[0][0].keys()])
            node_X = most_common([j for i,j in triad[0][0].keys()])
            triplets = set([node for sub in triad[0][0].keys() for node in sub])
            triplets.remove(node_P)
            triplets.remove(node_X)
            node_O = list(triplets)[0]
            region_list[0].append(area_dict[row][node_P])
            region_list[1].append(area_dict[row][node_X])
            region_list[2].append(area_dict[row][node_O])
          elif triad_type == '120D' or triad_type == '120U':
            if triad_type == '120D':
              node_X = most_common([i for i,j in triad[0][0].keys()])
            else:
              node_X = most_common([j for i,j in triad[0][0].keys()])
            triplets = set([node for sub in triad[0][0].keys() for node in sub])
            triplets.remove(node_X)
            node_PO = list(triplets)
            region_list[0].append(area_dict[row][node_X])
            for node in node_PO:
              region_list[1].append(area_dict[row][node])
          else:
            node_PXO = list(set([node for sub in triad[0][0].keys() for node in sub]))
            for node in node_PXO:
              region_list[0].append(area_dict[row][node])
      for p in range(len(region_list)):
        uniq, count = np.unique(region_list[p], return_counts=True)
        for a_ind, a in enumerate(uniq):
          r_ind = regions.index(a)
          metric[row_ind, col_ind, p, r_ind] = count[a_ind] / len(region_list[p])
  if triad_type == '300':
    fig = plt.figure(figsize=(6, 4*num_row))
  elif triad_type == '120D' or triad_type == '120U':
    fig = plt.figure(figsize=(3*num_col, 4*num_row))
  else:
    fig = plt.figure(figsize=(4*num_col, 4*num_row))
  for row_ind, row in enumerate(rows):
    if triad_type == '300':
      plt.subplot(num_row, 1, row_ind+1)
      m = metric[row_ind]
      A = m[:, 0, 0]
      B = m[:, 0, 1]
      C = m[:, 0, 2]
      D = m[:, 0, 3]
      E = m[:, 0, 4]
      F = m[:, 0, 5]
      # Plot stacked bar chart
          
      plt.bar(cols, A, label=regions[0]) #, color='cyan',
      plt.bar(cols, B, bottom=A, label=regions[1]) #, color='green'
      plt.bar(cols, C, bottom=A+B, label=regions[2]) #, color='red'
      plt.bar(cols, D, bottom=A+B+C, label=regions[3]) #, color='yellow'
      plt.bar(cols, E, bottom=A+B+C+D, label=regions[4]) #, color='yellow'
      plt.bar(cols, F, bottom=A+B+C+D+E, label=regions[5]) #, color='yellow'
      plt.xticks(rotation=90)
      plt.ylabel('stacked percentage')
    # plt.xticks(rotation=90)sign
      if row_ind == 0:
        plt.legend()
      if row_ind < num_row - 1:
        plt.tick_params(
          axis='x',          # changes apply to the x-axis
          which='both',      # both major and minor ticks are affected
          bottom=False,      # ticks along the bottom edge are off
          top=False,         # ticks along the top edge are off
          labelbottom=False) # labels along the bottom edge are off
    else:
      for col_ind, col in enumerate(cols):
        plt.subplot(num_row, num_col, row_ind * num_col + col_ind+1)
        m = metric[row_ind, col_ind]
        A = m[:, 0]
        B = m[:, 1]
        C = m[:, 2]
        D = m[:, 3]
        E = m[:, 4]
        F = m[:, 5]
        # Plot stacked bar chart
            
        plt.bar(pos, A, label=regions[0]) #, color='cyan',
        plt.bar(pos, B, bottom=A, label=regions[1]) #, color='green'
        plt.bar(pos, C, bottom=A+B, label=regions[2]) #, color='red'
        plt.bar(pos, D, bottom=A+B+C, label=regions[3]) #, color='yellow'
        plt.bar(pos, E, bottom=A+B+C+D, label=regions[4]) #, color='yellow'
        plt.bar(pos, F, bottom=A+B+C+D+E, label=regions[5]) #, color='yellow'
        plt.xticks(rotation=0, size=20)
        plt.ylabel('stacked percentage')
        if row_ind == 0:
          plt.title(col, size=20)
          plt.legend()
    
    plt.suptitle(name + '' + triad_type, size=30)
    plt.tight_layout()
  # plt.suptitle(k, fontsize=14, rotation=0)
  # plt.show()
  figname = './plots/triad_region_census_{}_{}_{}_{}fold.jpg'
  plt.savefig(figname.format(name, triad_type, measure, n))

###################### meso and macro level balance

# Binary linear programming model for computing the frustration index of 
# a directed signed graph (frustration-index-directed) as the optimal objective function

# This code solves graph optimization model(s) using "Gurobi solver"
# to compute the measure called, frustration index, for the input signed digraph(s)

# Note that you must have installed Gurobi into Jupyter and registered a Gurobi license
# in order to run this code

# This part of code requires the lower bound produced in the step above. If you intend to run this computation 
# without providing a lower bound, you should first comment out the line containing this command:
# model.addConstr(OFV >= int(LowerBounds[index]), 'LP lower bound')

#Setting parameters
#lazyParam=int(input("What is the lazy parameter for unbalanced triangle lazy cuts? (0/1/2/3)"))
# See "lazy" as a tunable parameter in linear constraint attributes in Gurobi optimizer reference manual below:
# https://www.gurobi.com/documentation/8.1/refman/lazy.html

def get_meso_macro_balance(S):
  lazyParam=int(3)
  #speedupParam=int(input("Do you want to use the speedups? (0=No, 1=Yes)"))
  speedupParam=int(1)
  S = S.copy()
  old_nodes = sorted(list(S.nodes()))
  mapping = {n:old_nodes.index(n) for n in old_nodes}
  S = nx.relabel_nodes(S, mapping)
  undirected_S = S.to_undirected(as_view=True)
  weighted_edges = nx.get_edge_attributes(S, 'sign')
  sorted_weighted_edges = {}
  for (u,v) in weighted_edges:
      (sorted_weighted_edges)[(u,v)] = weighted_edges[(u,v)]
  solveTime=[]
  effectiveBranchingFactors=[]
  undirected_sorted_weighted_edges = nx.get_edge_attributes(undirected_S, 'sign')
  ################ put nodes in ascending order
  new_undirected_sorted_weighted_edges = {}
  for edge in undirected_sorted_weighted_edges:
      if edge[0] > edge[1]:
          new_undirected_sorted_weighted_edges[(edge[1], edge[0])] = undirected_sorted_weighted_edges[edge]
      else:
          new_undirected_sorted_weighted_edges[edge] = undirected_sorted_weighted_edges[edge]
  undirected_sorted_weighted_edges = new_undirected_sorted_weighted_edges
  size = int(np.count_nonzero(nx.to_numpy_matrix(S)))
      
  type_of_edge=[]
  optimal_node_values=[]

  neighbors={}
  Degree=[]
  for u in sorted((undirected_S).nodes()):
      neighbors[u] = list((undirected_S)[u])
      Degree.append(len(neighbors[u]))
  # Note that reciprocated edges are counted as one and only contribute one to the degree of each endpoint
  
  #Finding the node with the highest unsigned degree
  maximum_degree = max(Degree)
  [node_to_fix]=[([i for i, j in enumerate(Degree) if j == maximum_degree]).pop()]

  # Model parameters
  model = gp.Model("Computing the frustration index of directed signed graphs")
  model.Params.LogToConsole = 0

  # There are different methods for solving optimization models:
  # (-1=automatic, 0=primal simplex, 1=dual simplex, 2=barrier, 3=concurrent, 4=deterministic concurrent)
  # model.setParam(GRB.param.Method, -1)
  
  # What is the time limit in second?
  # model.setParam('TimeLimit', 10*3600)
  
  # Do you want details of branching to be reported? (0=No, 1=Yes)
  model.setParam(gp.GRB.param.OutputFlag, 1) 
  
  # Do you want a non-zero Mixed integer programming tolerance (MIP Gap)?
  # Note that a non-zero MIP gap may prevent the model from computing the exact value of frustration index
  # model.setParam('MIPGap', 0.0001)  
  
  # How many threads to be used for exploring the feasible space in parallel?
  # Here, the minimum of 32 and the availbale CPUs is used
  model.setParam(gp.GRB.Param.Threads, min(32,multiprocessing.cpu_count()))
  
  #This chunk of code lists the graph triangles
  GraphTriangles=[]
  for n1 in sorted((undirected_S).nodes()):
      neighbors1 = set((undirected_S)[n1])
      for n2 in filter(lambda x: x>n1, neighbors1):
          neighbors2 = set((undirected_S)[n2])
          common = neighbors1 & neighbors2
          for n3 in filter(lambda x: x>n2, common):
              GraphTriangles.append([n1,n2,n3])
  #print("--- %Listed",len(GraphTriangles),"triangles for the graph")

  #This chunk of code lists the balanced and unbalanced triangles
  w=nx.get_edge_attributes(undirected_S, 'sign')  
  unbalanced_triangles = []
  balanced_triangles = []
  for triad in GraphTriangles: 
      if  (undirected_sorted_weighted_edges)[(triad[0],triad[1])]*\
      (undirected_sorted_weighted_edges)[(triad[0],triad[2])]*\
      (undirected_sorted_weighted_edges)[(triad[1],triad[2])] == -1:
          unbalanced_triangles.append(triad)
      elif (undirected_sorted_weighted_edges)[(triad[0],triad[1])]*\
      (undirected_sorted_weighted_edges)[(triad[0],triad[2])]*\
      (undirected_sorted_weighted_edges)[(triad[1],triad[2])] == 1:
          balanced_triangles.append(triad)  
  #print("--- %Listed",len(unbalanced_triangles),"unbalanced triangles for the graph")

  # Create decision variables and update model to integrate new variables
  x=[]
  for i in range(0,S.number_of_nodes()):
      x.append(model.addVar(vtype=gp.GRB.BINARY, name='x'+str(i))) # arguments by name
  model.update()
  
  f={}
  for (i,j) in (sorted_weighted_edges):
      f[(i,j)]=model.addVar(lb=0.0, ub=1, vtype=gp.GRB.CONTINUOUS, name='f'+str(i)+','+str(j))
  model.update()
  # print("--- %Binary variables are created")

  # Set the objective function
  OFV=0
  for (i,j) in (sorted_weighted_edges):
      OFV = OFV + f[(i,j)]                 
  model.setObjective(OFV, gp.GRB.MINIMIZE)

  # Add constraints to the model and update model to integrate new constraints
  
  ## ADD CORE CONSTRAINTS ##

  for (i,j) in (sorted_weighted_edges):
      model.addConstr( f[(i,j)] >= x[i] - ((sorted_weighted_edges)[(i,j)])*x[j] -\
                      (1-(sorted_weighted_edges)[(i,j)])/2
                          , '1st Edge'+','+str(i)+','+str(j))
      model.addConstr( f[(i,j)] >= -x[i] + ((sorted_weighted_edges)[(i,j)])*x[j] +\
                      (1-(sorted_weighted_edges)[(i,j)])/2
                          , '2nd Edge'+','+str(i)+','+str(j))                  
  model.update()
  # model.addConstr(OFV >= int(LowerBounds), 'LP lower bound')   # This line can be commented out
  model.update()
  # print("--- %Core constraints are added")
  ## ADD ADDITIONAL CONSTRAINTS (speed-ups) ##
  
  if speedupParam==1:
      # Triangle valid inequalities            

      triangleInequalityCount=len(unbalanced_triangles)
      for triangle in unbalanced_triangles:
          [i,j,k]=triangle
          b_ij=(i,j) in sorted_weighted_edges 
          b_ik=(i,k) in sorted_weighted_edges
          b_jk=(j,k) in sorted_weighted_edges
          if b_ij:
              if b_ik:
                  if b_jk:
                      model.addConstr(f[(i,j)] + f[(i,k)] + f[(j,k)] >= 1 ,\
                                      'UnbalancedTriangle'+','+str(i)+','+str(j)+','+str(k))
                  else:
                      model.addConstr(f[(i,j)] + f[(i,k)] + f[(k,j)] >= 1 ,\
                                      'UnbalancedTriangle'+','+str(i)+','+str(j)+','+str(k))
              elif b_jk:
                  model.addConstr(f[(i,j)] + f[(k,i)] + f[(j,k)] >= 1 ,\
                                      'UnbalancedTriangle'+','+str(i)+','+str(j)+','+str(k))
              else:
                  model.addConstr(f[(i,j)] + f[(k,i)] + f[(k,j)] >= 1 ,\
                                      'UnbalancedTriangle'+','+str(i)+','+str(j)+','+str(k))
          else:
              if b_ik:
                  if b_jk:
                      model.addConstr(f[(j,i)] + f[(i,k)] + f[(j,k)] >= 1 ,\
                                      'UnbalancedTriangle'+','+str(i)+','+str(j)+','+str(k))
                  else:
                      model.addConstr(f[(j,i)] + f[(i,k)] + f[(k,j)] >= 1 ,\
                                      'UnbalancedTriangle'+','+str(i)+','+str(j)+','+str(k))
              elif b_jk:
                  model.addConstr(f[(j,i)] + f[(k,i)] + f[(j,k)] >= 1 ,\
                                      'UnbalancedTriangle'+','+str(i)+','+str(j)+','+str(k))
              else:
                  model.addConstr(f[(j,i)] + f[(k,i)] + f[(k,j)] >= 1 ,\
                                      'UnbalancedTriangle'+','+str(i)+','+str(j)+','+str(k))
      model.update()
      model.setAttr('Lazy',model.getConstrs()[2*(size):2*(size)+triangleInequalityCount]\
                    ,[lazyParam]*triangleInequalityCount)
      model.update()
      # print("--- %Additional constraints are added")

      #branching priority is based on unsigned degree 
      model.setAttr('BranchPriority',model.getVars()[:S.number_of_nodes()],Degree)
      model.update() 
  # Solve
  start_time = time.time()
  try:
    model.optimize()
  except:
    print('Model too large for size-limited license; visit https://www.gurobi.com/free-trial for a full license')
    return {}, np.nan, np.nan, np.nan, np.nan
    # return {}, 0, 0, 0, 0
  solveTime.append(time.time() - start_time) 
  
  
  # Save optimal objective function values
  obj = model.getObjective()
  frustration_index = obj.getValue()
  
  # Compute the effective branching factors
  if (model.NodeCount)**(1/((size)+2*(S.number_of_nodes()))) >= 1:
      effectiveBranchingFactors.append((model.NodeCount)**(1/((size)+2*(S.number_of_nodes()))))
  # Printing the solution (optional)
  # print("Optimal values of the decision variables")
  inv_mapping = {v: k for k, v in mapping.items()}
  node_group = {}
  for v in model.getVars():
      if v.varName.startswith('x'):
          optimal_node_values.append(int(v.x))
          node_id = int(v.varName.replace('x', ''))
          node_group[inv_mapping[node_id]] = int(v.x)
          #if v.x!=1:
          # print (v.varName,":",int(v.x)) 
  
  # For printing types of the edges according to the optimal partition according to the four categories:
  # positive-internal, positive-external, negative-internal, negative-external
  
  for (u,v) in (sorted_weighted_edges):
      type_of_edge.append((2*(2*optimal_node_values[u]-1)*(2*optimal_node_values[v]-1)+((sorted_weighted_edges)[(u,v)])))
  counter=collections.Counter(type_of_edge)
  if (counter[1]+counter[3])>0:
      cohesiveness = counter[3]/(counter[1]+counter[3])
      # print("Cohesiveness:",(cohesiveness))
  else:
      print("Cohesiveness is undefined because there are no internal edges.")
  if (counter[-1]+counter[-3])>0:
      divisiveness = counter[-3]/(counter[-1]+counter[-3])
      # print("Divisiveness:",(divisiveness))
  else:
      print("Divisiveness is undefined because there are no external edges.")
  # print("-"*32,"***  EXPERIMENT STATS  ***","-"*32)
  # print("-"*92)
  # print("Frustration indices:",np.around(frustration_index))
  F = 1-2*frustration_index/S.number_of_edges()
  # print("Network level balance F: {}".format(F))
  #print("Average frustrarion index",np.mean(objectivevalue))
  #print("Frustration index Standard Deviation",np.std(objectivevalue))
  # print("Solve times (in seconds):",np.around(solveTime, decimals=2))
  #print("Average solve time",np.mean(solveTime))
  #print("Solve time Standard Deviation",np.std(solveTime))
  return node_group, cohesiveness, divisiveness, frustration_index, F

############### find the correlation window (percentage) for each significant edge
############### use offset, duration dict with thalamic regions, otherwise shape not matching
def get_correlation_window(directory, offset_dict, duration_dict):
  inds_path = './data/ecephys_cache_dir/sessions/active_inds/'
  rows, cols = get_rowcol(offset_dict)
  # cols.remove('flash_dark')
  # cols.remove('flash_light')
  c_window = {}
  for row in rows:
    c_window[row] = {}
    for col in cols:
      file = row + '_' + col + '.npz'
      print(file)
      adj_mat = load_npz_3d(os.path.join('./data/ecephys_cache_dir/sessions/adj_mat_ccg_highland_corrected/', file))
      c_window[row][col] = []
      offset_mat, duration_mat = offset_dict[row][col], duration_dict[row][col]   
      sequences = load_npz_3d(os.path.join(directory, file))
      active_neuron_inds = np.load(os.path.join(inds_path, row+'.npy'))
      sequences = sequences[active_neuron_inds]
      num_neuron, num_trial, T = sequences.shape
      # print(adj_mat.shape, offset_mat.shape, duration_mat.shape, (num_neuron, num_neuron))
      assert adj_mat.shape == offset_mat.shape == duration_mat.shape == (num_neuron, num_neuron) # remove thalamic region
      assert np.where(~np.isnan(adj_mat))[0].shape[0] == np.where(~np.isnan(offset_mat))[0].shape[0] == np.where(~np.isnan(duration_mat))[0].shape[0]
      for row_a, row_b in zip(*np.where(~np.isnan(adj_mat))): # each significant edge
        if adj_mat[row_a, row_b] > 0: # only for positive correlation
          offset, duration = int(offset_mat[row_a, row_b]), int(duration_mat[row_a, row_b])
          tfa, tla, tfa_ast, tla_ast, tfb, tlb, tfb_ast, tlb_ast = [[] for _ in range(8)]
          for m in range(num_trial):
            # print('Trial {} / {}'.format(m+1, num_trial))
            matrix = sequences[:,m,:]
            fr_rowa = np.count_nonzero(matrix[row_a]) / (matrix.shape[1]/1000) # Hz instead of kHz
            fr_rowb = np.count_nonzero(matrix[row_b]) / (matrix.shape[1]/1000)
            if fr_rowa * fr_rowb > 0: # there could be no spike in a certain trial
              for pad_len in range(offset, offset+duration+1): # min duration is 0
                s_a = np.pad(matrix[row_a], (0, pad_len), 'constant', constant_values=(0,0)) # false for incorrect CCG, should be opposite padding
                s_b = np.pad(matrix[row_b], (pad_len, 0), 'constant', constant_values=(0,0))
                prod = s_a * s_b
                perioda, periodb, periodp = np.where(s_a>0)[0], np.where(s_b>0)[0], np.where(prod>0)[0]
                if len(periodp):
                  tfa.append(perioda[0])
                  tla.append(perioda[-1])
                  tfa_ast.append(periodp[0])
                  tla_ast.append(periodp[-1])
                  tfb.append(periodb[0])
                  tlb.append(periodb[-1])
                  tfb_ast.append(periodp[0]-pad_len)
                  tlb_ast.append(periodp[-1]-pad_len)

          # if len(tla) and len(tfa) and len(tlb) and len(tfb) and len(tla_ast) and len(tfa_ast) and len(tlb_ast) and len(tfb_ast):
          Ta, Tb, Ta_ast, Tb_ast = max(tla)-min(tfa)+1, max(tlb)-min(tfb)+1, max(tla_ast)-min(tfa_ast)+1, max(tlb_ast)-min(tfb_ast)+1 # count start and end
          c_window[row][col].append(min(Ta_ast/Ta, Tb_ast/Tb))
        # print(min(Ta_ast/Ta, Tb_ast/Tb))
  return c_window

def plot_multi_correlation_window(c_window, measure, n):
  ind = 1
  rows, cols = get_rowcol(c_window)
  fig = plt.figure(figsize=(9*len(cols), 6*len(rows)))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      plt.subplot(len(rows), len(cols), ind)
      if row_ind == 0:
        plt.gca().set_title(cols[col_ind], fontsize=30, rotation=0)
      if col_ind == 0:
        plt.gca().text(0, 0.5 * (bottom + top), rows[row_ind],
        horizontalalignment='left',
        verticalalignment='center',
        # rotation='vertical',
        transform=plt.gca().transAxes, fontsize=30, rotation=90)
      plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
      ind += 1
      sns.histplot(data=c_window[row][col], stat='probability', kde=True, linewidth=0)
      plt.axvline(x=np.nanmean(c_window[row][col]), color='r', linestyle='--')
      plt.xticks(fontsize=20, rotation=0)
  plt.suptitle('correlation window', size=50)
  plt.tight_layout(rect=[0, 0.03, 1, 0.98])
  plt.savefig('./plots/multi_correlation_window{}_{}fold.jpg'.format(measure, n))

def plot_correlation_window_box(c_window, measure, n):
  rows, cols = get_rowcol(c_window)
  fig = plt.figure(figsize=(7, 5))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  cf = []
  for col_ind, col in enumerate(cols):
    print(col)
    cf_col = []
    for row_ind, row in enumerate(rows):
      cf_col += c_window[row][col]
    cf.append(cf_col)
  plt.boxplot(cf, showfliers=False)
  plt.xticks(list(range(1, len(cf)+1)), cols, rotation=90)
  # plt.hist(data.flatten(), bins=12, density=True)
  # plt.axvline(x=np.nanmean(data), color='r', linestyle='--')
  # plt.xlabel('region')
  plt.ylabel('CF')
  plt.suptitle('correlation window fraction', size=15)
  plt.tight_layout(rect=[0, 0.03, 1, 0.98])
  plt.savefig('./plots/box_correlation_window{}_{}fold.jpg'.format(measure, n))

def reorder_area(vector, current_order, areas_num, target_order):
  region_num = np.cumsum(areas_num)
  current_idx = [current_order.index(a) for a in target_order]
  new_vector = []
  for idx in current_idx:
    if idx > 0:
      new_vector.append(vector[region_num[idx-1]:region_num[idx]])
    else:
      new_vector.append(vector[0:region_num[idx]])
  return np.concatenate(new_vector)

########################### Message passing ##########################
##################### plot state change with area in hierarchical order
def plot_state(G_dict, row_ind, epsilon, active_area_dict, measure, n, timesteps=20):
  # fig, axs = plt.subplots(1, len(cols),figsize=(30,20))
  rows, cols = get_rowcol(G_dict)
  row = rows[row_ind]
  plt.figure(figsize=(40, 20))
  np.random.seed(1)
  # S_init = 2*np.random.rand(G.number_of_nodes())-1
  areas = [active_area_dict[row][node] for node in sorted(G_dict[row][cols[0]].nodes())]
  indexes = np.unique(areas, return_index=True)[1]
  uniq_areas = [areas[index] for index in sorted(indexes)]
  uniq_areas_num = [(np.array(areas)==a).sum() for a in uniq_areas]
  area_plot_order = ['VISam', 'VISpm', 'VISal', 'VISrl', 'VISl', 'VISp']
  areas_num = [(np.array(areas)==a).sum() for a in area_plot_order]
  areas_start_pos = list(np.insert(np.cumsum(areas_num)[:-1], 0, 0))
  text_pos = [s + (areas_num[areas_start_pos.index(s)] - 1) / 2 for s in areas_start_pos]
  S_init = []
  for a_ind, area in enumerate(uniq_areas):
    S_init += [round(np.linspace(-1, 1, len(uniq_areas))[area_plot_order.index(area)], 2)] * areas.count(area)
  S_init = np.array(S_init)
  for col_ind, col in enumerate(cols):
    G = G_dict[row][col]
    A = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
    A[A.nonzero()] = 1
    # A += 5*np.diag(A.sum(0)) # based on its degree
    A += (1+epsilon)*np.eye(A.shape[0]) # based on preset value
    no_neighbor = np.where(A.sum(0)==0)[0]
    A[no_neighbor, no_neighbor] = 1
    A = A.astype(float)
    A/=A.sum(0)
    T = A.T
    S = S_init.copy()
    # steps = 20
    state_variation= np.zeros((A.shape[0], timesteps))
    state_variation[:, 0] = reorder_area(S_init, uniq_areas, uniq_areas_num, area_plot_order)
    for ts in range(1, timesteps):
      S = T @ S
      state_variation[:, ts] = reorder_area(S, uniq_areas, uniq_areas_num, area_plot_order)
    plt.subplot(1, len(cols), col_ind+1)
    if col_ind == 0:
      cmap = cm.get_cmap('rainbow')
      norm = colors.Normalize(vmin=-1, vmax=1)
      for ind, t_pos in enumerate(text_pos):
        plt.text(-12., t_pos, area_plot_order[ind], size=20, color=cmap(norm(np.linspace(-1, 1, len(uniq_areas))[ind])))
    # plt.imshow(my_image1, vmin=0, vmax=10, cmap='jet', aspect='auto')
    plt.imshow(state_variation, aspect='auto', cmap='rainbow')
    plt.title(col)
    plt.xlabel('step')
    plt.ylabel('neuron number')
  plt.colorbar()
  plt.suptitle('{}, epsilon = {}'.format(row, epsilon), size=30)
  plt.tight_layout(rect=[0, 0.03, 1, 0.98])
  plt.savefig('./plots/state_vector_epislon_{}_{}_{}_{}fold.jpg'.format(epsilon, row, measure, n))
  # plt.show()

def inv_sigmoid(x):
  x = np.array(x)
  y = np.zeros_like(x)
  valid_inds = (x > 0) & (x < 1)
  large_inds = x >= 1
  small_inds = x <= 0
  y[valid_inds] = np.log(x[valid_inds]/(1-x[valid_inds]))
  y[large_inds] = 1000 # enough for softmax to be 1
  y[small_inds] = -1000 # enough for softmax to be 0
  return y



def plot_state_onehot(G_dict, row_ind, col_ind, epsilon, active_area_dict, measure, n, timesteps=20):
  rows, cols = get_rowcol(G_dict)
  row, col = rows[row_ind], cols[col_ind]
  np.random.seed(1) 
  area_plot_order = ['VISam', 'VISpm', 'VISal', 'VISrl', 'VISl', 'VISp']
  text_pos, state_variation = message_propagation(G_dict[row][col], epsilon, active_area_dict[row], area_plot_order, timesteps)
  colors_ = ['tab:green', 'lightcoral', 'steelblue', 'tab:orange', 'tab:purple', 'grey']
  cmap_list = cycle([colors.LinearSegmentedColormap.from_list("", ["white",c]) for c in colors_])
  steps = [0, 10, 15, 20, 199]
  nrows = 1
  ncols = len(steps)
  naxes = 6
  fig = plt.figure(figsize=(6, 10))
  for t_ind, ts in enumerate(steps):
    ag = axes_grid.Grid(fig, (nrows, ncols, t_ind+1), (1, naxes), axes_pad=0)
    ag.axes_all[2].set_title('step {}'.format(ts), size=10)
    for j in range(naxes):
      sns.heatmap(pd.DataFrame(state_variation[:,ts,j]), ax=ag[j], cbar=False, cmap=cmap_list.__next__(), xticklabels=False, yticklabels=False) # pad is the space between cbar and heatmap, aspect is the ratio of long to short dimensions   , cbar=True, cbar_kws={"orientation": "vertical", 'location':'left', "shrink": .8, 'pad': 0.01, 'aspect': 6}
    if t_ind == 0:
      for ind, t_pos in enumerate(text_pos):
        plt.text(-8., t_pos, area_plot_order[ind], size=10, color=colors_[ind])
  plt.subplots_adjust(wspace=0.5, hspace=0.5)
  plt.suptitle('{}, {}, epsilon = {}'.format(row, col, epsilon), size=13)
  plt.tight_layout(rect=[0, 0.03, 1, 0.98])
  plt.savefig('./plots/state_vector_epislon_{}_{}_{}_{}_{}fold.jpg'.format(epsilon, row, col, measure, n))
  # plt.show()

##################### plot state distance
def plot_state_jsdistance(G_dict, row_ind, epsilon, active_area_dict, measure, n, timesteps=20):
  rows, cols = get_rowcol(G_dict)
  row = rows[row_ind]
  np.random.seed(1)
  areas = [active_area_dict[row][node] for node in sorted(G_dict[row][cols[0]].nodes())]
  indexes = np.unique(areas, return_index=True)[1]
  uniq_areas = [areas[index] for index in sorted(indexes)]
  uniq_areas_num = [(np.array(areas)==a).sum() for a in uniq_areas]
  area_plot_order = ['VISam', 'VISpm', 'VISal', 'VISrl', 'VISl', 'VISp']
  one_hot = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
  S_init = []
  for a_ind, area in enumerate(uniq_areas):
    S_init += [one_hot[area_plot_order.index(area), :]] * areas.count(area)
  S_init = np.array(S_init)
  fig = plt.figure(figsize=(4*len(cols), 2*len(cols)))
  for col_ind, col in enumerate(cols):
    print(col)
    plt.subplot(2, int(np.ceil(len(cols)/2)), col_ind+1)
    plt.gca().set_title(col, fontsize=20, rotation=0)
    plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    G = G_dict[row][col]
    A = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
    A[A.nonzero()] = 1
    # A += 5*np.diag(A.sum(0)) # based on its degree
    A += (1+epsilon)*np.eye(A.shape[0]) # based on preset value
    no_neighbor = np.where(A.sum(0)==0)[0]
    A[no_neighbor, no_neighbor] = 1
    A = A.astype(float)
    A/=A.sum(0)
    T = A.T
    S = S_init.copy()
    state_variation= np.zeros((A.shape[0], timesteps, 6))
    state_variation[:, 0] = reorder_area(S_init, uniq_areas, uniq_areas_num, area_plot_order)
    for ts in range(1, timesteps):
      S = T @ S
      state_variation[:, ts] = reorder_area(S, uniq_areas, uniq_areas_num, area_plot_order)
    distances = np.zeros((A.shape[0], timesteps))
    for ts in range(timesteps):
      for neuron in range(A.shape[0]):
        distances[neuron, ts] = distance.jensenshannon(state_variation[neuron, 0], state_variation[neuron, ts])
    plot_areas_num = [(np.array(areas)==a).sum() for a in area_plot_order]
    area_inds = [0] + np.cumsum(plot_areas_num).tolist()
    colors_ = ['tab:green', 'lightcoral', 'steelblue', 'tab:orange', 'tab:purple', 'grey']
    for i in range(len(area_plot_order)):
      region_distance = distances[area_inds[i]:area_inds[i+1]]
      mean, std = region_distance.mean(0), region_distance.std(0)
      plt.plot(range(timesteps), mean, color=colors_[5-i], label=area_plot_order[i], alpha=0.8)
      # plt.fill_between(range(timesteps), mean-std, mean+std, color=colors_[5-i], alpha=0.2)
    plt.legend()
    plt.xlabel('step')
    plt.ylabel('JS distance')
  plt.suptitle('{}, epsilon = {}'.format(row, epsilon), size=30)
  plt.tight_layout(rect=[0, 0.03, 1, 0.98])
  plt.savefig('./plots/state_vector_jsdistance_epislon_{}_{}_{}_{}fold.jpg'.format(epsilon, row, measure, n))
  # plt.show()

##################### plot state region fraction
def plot_state_region_fraction(G_dict, epsilon, active_area_dict, measure, n, timesteps=20):
  rows, cols = get_rowcol(G_dict)
  np.random.seed(1)
  area_plot_order = ['VISam', 'VISpm', 'VISal', 'VISrl', 'VISl', 'VISp']
  one_hot = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
  fig = plt.figure(figsize=(4*len(cols), 2*len(cols)))
  for col_ind, col in enumerate(cols):
    print(col)
    plt.subplot(2, int(np.ceil(len(cols)/2)), col_ind+1)
    plt.gca().set_title(col, fontsize=20, rotation=0)
    plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    region_fraction = np.zeros((len(area_plot_order), len(rows), timesteps))
    for row_ind, row in enumerate(rows):
      if row_ind != 1:
        G = G_dict[row][col]
        areas = [active_area_dict[row][node] for node in sorted(G.nodes())]
        _, state_variation = message_propagation(G, epsilon, active_area_dict[row], area_plot_order, timesteps)
        plot_areas_num = [(np.array(areas)==a).sum() for a in area_plot_order]
        area_inds = [0] + np.cumsum(plot_areas_num).tolist()
        for region_ind, region in enumerate(area_plot_order):
          region_loc = np.where(one_hot[region_ind])[0][0]
          region_fraction[region_ind, row_ind] = state_variation[area_inds[region_ind]:area_inds[region_ind+1], :, region_loc].mean(0) 
    colors_ = ['tab:green', 'lightcoral', 'steelblue', 'tab:orange', 'tab:purple', 'grey']
    for i in range(len(area_plot_order)):
      mean, std = region_fraction[i,[0,2,3,4,5]].mean(0), region_fraction[i,[0,2,3,4,5]].std(0)
      plt.plot(range(timesteps), mean, color=colors_[i], label=area_plot_order[i], alpha=0.9)
      # plt.fill_between(range(timesteps), mean-std, mean+std, color=colors_[i], alpha=0.2)
    plt.legend()
    plt.ylim(0.2, 1.05)
    plt.xlabel('step')
    plt.ylabel('region fraction')
  plt.suptitle('epsilon = {}'.format(epsilon), size=30)
  plt.tight_layout(rect=[0, 0.03, 1, 0.98])
  plt.savefig('./plots/state_vector_region_fraction_epislon_{}_{}_{}fold.jpg'.format(epsilon, measure, n))
  # plt.show()

##################### plot state steady distribution for each region
def message_propagation(G, epsilon, active_area, area_plot_order, timesteps):
  areas = [active_area[node] for node in sorted(G.nodes())]
  indexes = np.unique(areas, return_index=True)[1]
  uniq_areas = [areas[index] for index in sorted(indexes)]
  uniq_areas_num = [(np.array(areas)==a).sum() for a in uniq_areas]
  areas_num = [(np.array(areas)==a).sum() for a in area_plot_order]
  areas_start_pos = list(np.insert(np.cumsum(areas_num)[:-1], 0, 0))
  text_pos = [s + (areas_num[areas_start_pos.index(s)] - 1) / 2 for s in areas_start_pos]
  # one_hot = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
  one_hot = np.zeros((len(uniq_areas), len(uniq_areas)))
  one_hot[np.arange(len(uniq_areas)), np.arange(len(uniq_areas))] = 1
  S_init = []
  for a_ind, area in enumerate(uniq_areas):
    S_init += [one_hot[area_plot_order.index(area), :]] * areas.count(area)
  S_init = np.array(S_init)
  A = nx.to_numpy_array(G, nodelist=sorted(G.nodes()),weight='sign')
  A[A<0] = 0 # only keep excitatory links
  A[A.nonzero()] = 1
  offset_mat = nx.to_numpy_array(G, nodelist=sorted(G.nodes()),weight='offset')
  A[offset_mat==0] = 0 # remove 0 time lag edges (common input)
  # disproportional to region size
  # area_inds = [0] + np.cumsum(uniq_areas_num).tolist()
  # for region_ind in range(len(uniq_areas)):
  #   nonzero = A[area_inds[region_ind]:area_inds[region_ind+1], :].nonzero()
  #   A[area_inds[region_ind]:area_inds[region_ind+1], :][nonzero] = 1 / uniq_areas_num[region_ind]

  # first normalize neighbors then add self loop
  # no_neighbor = np.where(A.sum(0)==0)[0]
  # A[no_neighbor, no_neighbor] = 1
  # A = A.astype(float)
  # A/=A.sum(0)
  # A += (1+epsilon)*np.eye(A.shape[0])
  # A/=A.sum(0)

  # neurons with more in neighbors are more vulnerable
  A += (1+epsilon)*np.eye(A.shape[0]) # based on preset value
  A = A.astype(float)
  A/=A.sum(0)

  T = A.T
  S = S_init.copy()
  state_variation= np.zeros((A.shape[0], timesteps, 6))
  state_variation[:, 0] = reorder_area(S_init, uniq_areas, uniq_areas_num, area_plot_order)
  for ts in range(1, timesteps):
    # S = T @ S
    S = softmax(inv_sigmoid((T @ T) @ S), axis=1)
    state_variation[:, ts] = reorder_area(S, uniq_areas, uniq_areas_num, area_plot_order)
  return text_pos, state_variation

def plot_steady_distribution(G_dict, epsilon, active_area_dict, measure, n, timesteps=20):
  rows, cols = get_rowcol(G_dict)
  np.random.seed(1)
  area_plot_order = ['VISam', 'VISpm', 'VISal', 'VISrl', 'VISl', 'VISp']
  # one_hot = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
  fig = plt.figure(figsize=(5*len(cols), 4*len(area_plot_order)))
  for col_ind, col in enumerate(cols):
    print(col)
    steady_distribution = np.zeros((len(area_plot_order), 2, len(area_plot_order)))
    s_distri = {r:[] for r in area_plot_order}
    for row_ind, row in enumerate(rows):
      if row_ind != 1:
        G = G_dict[row][col]
        areas = [active_area_dict[row][node] for node in sorted(G.nodes())]
        # indexes = np.unique(areas, return_index=True)[1]
        # uniq_areas = [areas[index] for index in sorted(indexes)]
        # uniq_areas_num = [(np.array(areas)==a).sum() for a in uniq_areas]
        _, state_variation = message_propagation(G, epsilon, active_area_dict[row], area_plot_order, timesteps)
        plot_areas_num = [(np.array(areas)==a).sum() for a in area_plot_order]
        area_inds = [0] + np.cumsum(plot_areas_num).tolist()
        for region_ind, region in enumerate(area_plot_order):
          s_distri[region].append(state_variation[area_inds[region_ind]:area_inds[region_ind+1], -1, :].mean(0))
    for region_ind, region in enumerate(area_plot_order):
      steady_distribution[region_ind, 0] = np.vstack((s_distri[region])).mean(0)
      steady_distribution[region_ind, 1] = np.vstack((s_distri[region])).std(0)
    colors_ = ['tab:green', 'lightcoral', 'steelblue', 'tab:orange', 'tab:purple', 'grey']
    for i in range(len(area_plot_order)):
      ax = plt.subplot(len(area_plot_order), len(cols), i*len(cols)+col_ind+1)
      if i == 0:
        plt.gca().set_title(col, fontsize=20, rotation=0)
      plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
      
      ax.bar(range(len(area_plot_order)), steady_distribution[i, 0], yerr=steady_distribution[i, 1], align='center', alpha=0.6, ecolor='black', color=colors_, capsize=10)
      ax.set_ylabel(area_plot_order[i], fontsize=20)
      ax.set_xticks(range(len(area_plot_order)))
      ax.set_xticklabels(area_plot_order, fontsize=15)
      plt.ylim(0, 1.)
  
  plt.suptitle('epsilon = {}'.format(epsilon), size=30)
  plt.tight_layout(rect=[0, 0.03, 1, 0.98])
  plt.savefig('./plots/state_vector_steady_distribution_epislon_{}_{}_{}fold.jpg'.format(epsilon, measure, n))
  # plt.show()

def plot_dominance_score(G_dict, epsilon, active_area_dict, measure, n, timesteps=20):
  rows, cols = get_rowcol(G_dict)
  np.random.seed(1)
  area_plot_order = ['VISam', 'VISpm', 'VISal', 'VISrl', 'VISl', 'VISp']
  # one_hot = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
  fig = plt.figure(figsize=(5*len(cols), 4))
  for col_ind, col in enumerate(cols):
    print(col)
    dominance_score = np.zeros((len(area_plot_order), 2))
    s_score = []
    for row_ind, row in enumerate(rows):
      if row_ind != 1:
        G = G_dict[row][col]
        areas = [active_area_dict[row][node] for node in sorted(G.nodes())]
        # indexes = np.unique(areas, return_index=True)[1]
        # uniq_areas = [areas[index] for index in sorted(indexes)]
        # uniq_areas_num = [(np.array(areas)==a).sum() for a in uniq_areas]
        _, state_variation = message_propagation(G, epsilon, active_area_dict[row], area_plot_order, timesteps)
        # plot_areas_num = [(np.array(areas)==a).sum() for a in area_plot_order]
        # area_inds = [0] + np.cumsum(plot_areas_num).tolist()
        s_score.append(state_variation[:, -1, :].mean(0) / state_variation[:, 0, :].mean(0))
    dominance_score[:, 0] = np.vstack((s_score)).mean(0)
    dominance_score[:, 1] = np.vstack((s_score)).std(0)
    colors_ = ['tab:green', 'lightcoral', 'steelblue', 'tab:orange', 'tab:purple', 'grey']
    ax = plt.subplot(1, len(cols), col_ind+1)
    plt.gca().set_title(col, fontsize=20, rotation=0)
    plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    
    ax.bar(range(len(area_plot_order)), dominance_score[:, 0], yerr=dominance_score[:, 1], align='center', alpha=0.6, ecolor='black', color=colors_, capsize=10)
    ax.set_xticks(range(len(area_plot_order)))
    ax.set_xticklabels(area_plot_order)
    if col_ind == 0:
      plt.ylabel('dominance score')
    plt.ylim(0, 3.2)

  plt.suptitle('epsilon = {}'.format(epsilon), size=30)
  plt.tight_layout(rect=[0, 0.03, 1, 0.98])
  plt.savefig('./plots/state_vector_dominance_score_scale_epislon_{}_{}_{}fold.jpg'.format(epsilon, measure, n))
  # plt.show()

def propagation2convergence(G, epsilon, area_plot_order, active_area, step2confirm=5, maxsteps=1000):
  areas = [active_area[node] for node in sorted(G.nodes())]
  indexes = np.unique(areas, return_index=True)[1]
  uniq_areas = [areas[index] for index in sorted(indexes)]
  uniq_areas_num = [(np.array(areas)==a).sum() for a in uniq_areas]
  # one_hot = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
  one_hot = np.zeros((len(uniq_areas), len(uniq_areas)))
  one_hot[np.arange(len(uniq_areas)), np.arange(len(uniq_areas))] = 1
  S_init = []
  for a_ind, area in enumerate(uniq_areas):
    S_init += [one_hot[a_ind, :]] * areas.count(area)
  S_init = np.array(S_init)
  A = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
  A[A.nonzero()] = 1
  A += (1+epsilon)*np.eye(A.shape[0]) # based on preset value
  no_neighbor = np.where(A.sum(0)==0)[0]
  A[no_neighbor, no_neighbor] = 1
  A = A.astype(float)
  A/=A.sum(0)
  T = A.T
  old_S = S_init.copy()
  step2conv = np.zeros(len(area_plot_order))
  step2conf = np.ones(len(area_plot_order)) * step2confirm
  area_inds = [0] + np.cumsum(uniq_areas_num).tolist()
  for ts in range(1, maxsteps):
    # S = T @ S
    new_S = softmax(inv_sigmoid((T @ T) @ old_S), axis=1)
    for area_ind, area in enumerate(uniq_areas):
      if step2conv[area_plot_order.index(area)] <= 0: # else it's already found
        if np.allclose(new_S[area_inds[area_ind]:area_inds[area_ind+1]], old_S[area_inds[area_ind]:area_inds[area_ind+1]]):
          step2conf[area_plot_order.index(area)] -= 1
        else:
          step2conf[area_plot_order.index(area)] = step2confirm
        if step2conf[area_plot_order.index(area)] <= 0:
          step2conv[area_plot_order.index(area)] = ts - step2confirm
    if (step2conv > 0).all():
      break
    else:
      old_S = new_S.copy()
  if (step2conv==0).sum() > 0:
    regions = [area_plot_order[ind] for ind in np.where(step2conv==0)[0]]
    print('Reach maximal step {} for region(s) {}'.format(maxsteps, regions))
    step2conv[step2conv==0] = maxsteps
  return step2conv

def plot_step2convergence(G_dict, epsilon_list, active_area_dict, measure, n, step2confirm=5, maxsteps=1000):
  rows, cols = get_rowcol(G_dict)
  np.random.seed(1)
  area_plot_order = ['VISam', 'VISpm', 'VISal', 'VISrl', 'VISl', 'VISp']
  # one_hot = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
  fig = plt.figure(figsize=(5*len(cols), 4))
  for col_ind, col in enumerate(cols):
    print(col)
    step2convergence = np.zeros((len(area_plot_order), len(rows), len(epsilon_list)))
    for row_ind, row in enumerate(rows):
      print(row)
      if row_ind != 1:
        G = G_dict[row][col]
        for e_ind, epsilon in enumerate(epsilon_list):
          step2convergence[:, row_ind, e_ind] = propagation2convergence(G, epsilon, area_plot_order, active_area_dict[row], step2confirm=step2confirm, maxsteps=maxsteps)
          
    colors_ = ['tab:green', 'lightcoral', 'steelblue', 'tab:orange', 'tab:purple', 'grey']
    ax = plt.subplot(1, len(cols), col_ind+1)
    plt.gca().set_title(col, fontsize=20, rotation=0)
    plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    
    for area_ind, area in enumerate(area_plot_order):
      ymean, yerr = step2convergence[area_ind].mean(0), 2 * step2convergence[area_ind].std(0)
      # plt.errorbar(epsilon_list, ymean, yerr=yerr, label=area, alpha=0.6, color=colors_[area_ind])
      plt.plot(epsilon_list, ymean, label=area, alpha=0.6, color=colors_[area_ind])
    plt.legend()
    plt.xlabel('epsilon')
    if col_ind == 0:
      plt.ylabel('steps to convergence')
    # plt.ylim(-.5, 4.5)

  plt.tight_layout()
  plt.savefig('./plots/state_vector_step2convergence_{}_{}fold.jpg'.format(measure, n))
  # plt.show()

def propagation_till_convergence(G, epsilon, area_plot_order, active_area, step2confirm=5, maxsteps=1000):
  areas = [active_area[node] for node in sorted(G.nodes())]
  indexes = np.unique(areas, return_index=True)[1]
  uniq_areas = [areas[index] for index in sorted(indexes)]
  uniq_areas_num = [(np.array(areas)==a).sum() for a in uniq_areas]
  # one_hot = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
  one_hot = np.zeros((len(uniq_areas), len(uniq_areas)))
  one_hot[np.arange(len(uniq_areas)), np.arange(len(uniq_areas))] = 1
  S_init = []
  for a_ind, area in enumerate(uniq_areas):
    S_init += [one_hot[area_plot_order.index(area), :]] * areas.count(area)
  S_init = np.array(S_init)
  A = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
  A[A.nonzero()] = 1
  A += (1+epsilon)*np.eye(A.shape[0]) # based on preset value
  no_neighbor = np.where(A.sum(0)==0)[0]
  A[no_neighbor, no_neighbor] = 1
  A = A.astype(float)
  A/=A.sum(0)
  T = A.T
  old_S = S_init.copy()
  state_variation= []
  state_variation.append(reorder_area(S_init, uniq_areas, uniq_areas_num, area_plot_order))
  step2conv = np.zeros(len(area_plot_order))
  step2conf = np.ones(len(area_plot_order)) * step2confirm
  area_inds = [0] + np.cumsum(uniq_areas_num).tolist()
  for ts in range(1, maxsteps):
    # S = T @ S
    new_S = softmax(inv_sigmoid((T @ T) @ old_S), axis=1)
    state_variation.append(reorder_area(new_S, uniq_areas, uniq_areas_num, area_plot_order))
    for area_ind, area in enumerate(uniq_areas):
      if step2conv[area_plot_order.index(area)] <= 0: # else it's already found
        if np.allclose(new_S[area_inds[area_ind]:area_inds[area_ind+1]], old_S[area_inds[area_ind]:area_inds[area_ind+1]]):
          step2conf[area_plot_order.index(area)] -= 1
        else:
          step2conf[area_plot_order.index(area)] = step2confirm
        if step2conf[area_plot_order.index(area)] <= 0:
          step2conv[area_plot_order.index(area)] = ts - step2confirm
    if (step2conv > 0).all():
      break
    else:
      old_S = new_S.copy()
  if (step2conv==0).sum() > 0:
    regions = [area_plot_order[ind] for ind in np.where(step2conv==0)[0]]
    print('Reach maximal step {} for region(s) {}'.format(maxsteps, regions))
    step2conv[step2conv==0] = maxsteps
  state_variation = np.stack(state_variation, axis=1)
  return state_variation

def plot_region_frac_epsilon(G_dict, epsilon_list, active_area_dict, measure, n, step2confirm=5, maxsteps=1000):
  rows, cols = get_rowcol(G_dict)
  np.random.seed(1)
  area_plot_order = ['VISam', 'VISpm', 'VISal', 'VISrl', 'VISl', 'VISp']
  one_hot = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
  fig = plt.figure(figsize=(5*len(cols), 4))
  for col_ind, col in enumerate(cols):
    print(col)
    region_frac = np.zeros((len(area_plot_order), len(rows), len(epsilon_list)))
    for row_ind, row in enumerate(rows):
      print(row)
      if row_ind != 1:
        G = G_dict[row][col]
        areas = [active_area_dict[row][node] for node in sorted(G.nodes())]
        for e_ind, epsilon in enumerate(epsilon_list):
          state_variation = propagation_till_convergence(G, epsilon, area_plot_order, active_area_dict[row], step2confirm=step2confirm, maxsteps=maxsteps)
          plot_areas_num = [(np.array(areas)==a).sum() for a in area_plot_order]
          area_inds = [0] + np.cumsum(plot_areas_num).tolist()
          for region_ind, region in enumerate(area_plot_order):
            region_loc = np.where(one_hot[region_ind])[0][0]
            region_frac[region_ind, row_ind, e_ind] = state_variation[area_inds[region_ind]:area_inds[region_ind+1], -1, region_loc].mean(0) 
    colors_ = ['tab:green', 'lightcoral', 'steelblue', 'tab:orange', 'tab:purple', 'grey']
    ax = plt.subplot(1, len(cols), col_ind+1)
    plt.gca().set_title(col, fontsize=20, rotation=0)
    plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    
    for area_ind, area in enumerate(area_plot_order):
      ymean, yerr = region_frac[area_ind].mean(0), 2 * region_frac[area_ind].std(0)
      # plt.errorbar(epsilon_list, ymean, yerr=yerr, label=area, alpha=0.6, color=colors_[area_ind])
      plt.plot(epsilon_list, ymean, label=area, alpha=0.6, color=colors_[area_ind])
    plt.legend()
    plt.xlabel('epsilon')
    if col_ind == 0:
      plt.ylabel('stable region fraction')
    plt.ylim(.2, .9)

  plt.tight_layout()
  plt.savefig('./plots/state_vector_region_frac_epsilon_{}_{}fold.jpg'.format(measure, n))
  # plt.savefig('./plots/state_vector_region_frac_epsilon_zoomin_{}_{}fold.jpg'.format(measure, n))
  # plt.show()

def plot_multi_connectivity_matrix(G_dict, active_area_dict, measure, n):
  ind = 1
  rows, cols = get_rowcol(G_dict)
  fig = plt.figure(figsize=(4*len(cols), 3*len(rows)))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  area_plot_order = ['VISam', 'VISpm', 'VISal', 'VISrl', 'VISl', 'VISp']
  colors_ = ['tab:green', 'lightcoral', 'steelblue', 'tab:orange', 'tab:purple', 'grey']
  for row_ind, row in enumerate(rows):
    print(row)
    active_area = active_area_dict[row]
    ordered_nodes = [] # order nodes based on hierarchical order 
    for area in area_plot_order:
      for node in active_area:
        if active_area[node] == area:
          ordered_nodes.append(node)
    for col_ind, col in enumerate(cols):
      plt.subplot(len(rows), len(cols), ind)
      if row_ind == 0:
        plt.gca().set_title(cols[col_ind], fontsize=20, rotation=0)
      if col_ind == 0:
        plt.gca().text(0, 0.5 * (bottom + top), rows[row_ind],
        horizontalalignment='left',
        verticalalignment='center',
        # rotation='vertical',
        transform=plt.gca().transAxes, fontsize=20, rotation=90)
      plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
      ind += 1
      G = G_dict[row][col] if col in G_dict[row] else nx.DiGraph()
      A = nx.to_numpy_array(G, nodelist=ordered_nodes)
      A[A>0] = 1
      A[A<0] = -1
      plt.imshow(A, cmap=plt.cm.RdBu_r)
      # A[A.nonzero()] = 1
      # plt.imshow(A, cmap=plt.cm.Greys)
      areas = [active_area[node] for node in ordered_nodes]
      indexes = np.unique(areas, return_index=True)[1]
      uniq_areas = [areas[index] for index in sorted(indexes)]
      uniq_areas_num = [(np.array(areas)==a).sum() for a in uniq_areas]
      area_inds = [0] + np.cumsum(uniq_areas_num).tolist()
      for region_ind in range(len(uniq_areas)):
        color_ind = area_plot_order.index(uniq_areas[region_ind])
        plt.gca().add_patch(Rectangle((area_inds[region_ind]-1,area_inds[region_ind]-1),uniq_areas_num[region_ind],uniq_areas_num[region_ind],linewidth=1,edgecolor=colors_[color_ind],facecolor='none'))
      plt.gca().invert_xaxis()
      plt.xticks([])
      plt.yticks([])
  # plt.colorbar()
  plt.tight_layout()
  plt.savefig('./plots/connectivity_matrix_{}_{}.jpg'.format(measure, n))
  # plt.show()

def region_FR(session_ids, stimulus_names, regions, active_area_dict):
  directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
  if not os.path.isdir(directory):
    os.mkdir(directory)
  FR = np.zeros((len(regions), len(stimulus_names), len(session_ids)))
  for se_ind, session_id in enumerate(session_ids):
    active_area = active_area_dict[session_id]
    node_idx = sorted(active_area.keys())
    for st_ind, stimulus_name in enumerate(stimulus_names):
      file = str(session_id) + '_' + stimulus_name + '.npz'
      print(file)
      sequences = load_npz_3d(os.path.join(directory, file))
      for r_ind, region in enumerate(regions):
        active_nodes = [node for node in node_idx if active_area[node]==region]
        if len(active_nodes):
          active_node_inds = np.array([node_idx.index(node) for node in active_nodes])
          # print(active_node_inds.shape)
          # print(sequences[active_node_inds].shape, sequences[active_node_inds].mean(0).mean(1).sum(1).shape, sequences.shape[2])
          FR[r_ind, st_ind, se_ind] = sequences[active_node_inds].mean(1).sum(1).mean(0) / sequences.shape[2]
  return FR

def transparent_rgb(rgb, bg_rgb, alpha):
  return [alpha * c1 + (1 - alpha) * c2 for (c1, c2) in zip(rgb, bg_rgb)]

def plot_FR_region(FR, stimulus_names, regions, measure, n):
  for se_ind in range(FR.shape[2]):
    if np.any(FR[:,:,se_ind] == 0):
      se_ind2remove = se_ind
      break
  s_inds = list(range(FR.shape[2]))
  s_inds.remove(se_ind2remove)
  name = 'firing rate (Hz)'
  df = pd.DataFrame()
  for r_ind, region in enumerate(regions):
    for se_ind, session_id in enumerate(session_ids):
      df = pd.concat([df, pd.DataFrame(np.concatenate((FR[r_ind, :, se_ind][:,None], np.array(stimulus_names)[:,None], np.array([region] * len(stimulus_names))[:,None]), 1), columns=[name, 'stimulus', 'region'])], ignore_index=True)
  df[name] = pd.to_numeric(df[name]) * 1000 # kHz to Hz
  plt.figure(figsize=(17, 7))
  hue_order = ['VISam', 'VISpm', 'VISal', 'VISrl', 'VISl', 'VISp']
  colors_ = ['tab:green', 'lightcoral', 'steelblue', 'tab:orange', 'tab:purple', 'grey']
  colors_transparency = [transparent_rgb(colors.to_rgb(color), [1,1,1], alpha=.5) for color in colors_]
  ax = sns.boxplot(x="stimulus", y=name, hue="region", hue_order=hue_order, data=df, palette=colors_transparency, showfliers=False) # , boxprops=dict(alpha=.6)
  ax.set(xlabel=None)
  plt.title('firing rate (Hz) of each region with stimulus', size=15)
  plt.savefig('./plots/FR_region_stimulus_{}_{}fold.jpg'.format(measure, n))

############################ one more significance test for CCG signal
def unique_with_tolerance(sequence, TOL=1e-3):
  seq = sequence.copy()
  seq.sort()
  diff = np.append(True, np.diff(seq))
  unique_values = seq[diff>TOL]
  return unique_values

def normalized_entropy_with_tolerance(sequence, TOL=1e-7):
  seq = sequence.copy()
  seq.sort()
  diff = np.append(True, np.diff(seq))
  unique_values = seq[diff>TOL]
  num_repeats = np.diff(np.append(np.where(diff>TOL)[0], len(diff)))
  discretized_seq = np.repeat(unique_values, num_repeats)
  _, counts = np.unique(discretized_seq, return_counts=True)
  return entropy(counts, base=2)  / entropy([1] * len(seq), base=2)

def test_portion_above_threshold(directory, threshold):
  files = os.listdir(directory)
  files.sort(key=lambda x:int(x[:9]))
  df_n, df_p = pd.DataFrame(index=session_ids, columns=stimulus_names), pd.DataFrame(index=session_ids, columns=stimulus_names)
  for file in files:
    if file.endswith(".npz") and ('gabors' not in file) and ('flashes' not in file) and ('_offset' not in file) and ('_duration' not in file) and ('_bl' not in file) and ('confidence' not in file):
      print(file)
      significant_ccg = load_npz_3d(os.path.join(directory, file))
      significant_duration = load_npz_3d(os.path.join(directory, file.replace('.npz', '_duration.npz')))
      ccg = load_sparse_npz(os.path.join(directory.replace('adj_mat_ccg_highland_corrected', 'adj_mat_ccg_corrected'), file))
      ccg_jittered = load_sparse_npz(os.path.join(directory.replace('adj_mat_ccg_highland_corrected', 'adj_mat_ccg_corrected'), file.replace('.npz', '_bl.npz')))
      ccg_corrected = ccg - ccg_jittered
      significant_inds = list(zip(*np.where(~np.isnan(significant_ccg))))
      mouseID = file.split('_')[0]
      stimulus_name = file.replace('.npz', '').replace(mouseID + '_', '')
      num = 0
      for row_a, row_b in significant_inds:
        filter = np.array([1]).repeat(significant_duration[row_a,row_b]+1) # sum instead of mean
        sig = signal.convolve(ccg_corrected[row_a, row_b], filter, mode='valid', method='fft')
        # uniq = unique_with_tolerance(sig, 1e-7)
        # if len(uniq)/len(sig) >= threshold:
        entropy = normalized_entropy_with_tolerance(sig, TOL=1e-7)
        if entropy >= threshold:
          num += 1
      df_n.loc[mouseID, stimulus_name] = num
      df_p.loc[mouseID, stimulus_name] = num / len(significant_inds)
  return df_n, df_p

def plot_uniq_comparison(directory, threshold, n):
  files = os.listdir(directory)
  files.sort(key=lambda x:int(x[:9]))
  for file in files:
    if file.endswith(".npz") and ('755434585' in file) and ('flash_light' in file) and ('_offset' not in file) and ('_duration' not in file) and ('_bl' not in file) and ('confidence' not in file):
      print(file)
      significant_ccg = load_npz_3d(os.path.join(directory, file))
      significant_duration = load_npz_3d(os.path.join(directory, file.replace('.npz', '_duration.npz')))
      significant_offset = load_npz_3d(os.path.join(directory, file.replace('.npz', '_offset.npz')))
      ccg = load_sparse_npz(os.path.join(directory.replace('adj_mat_ccg_highland_corrected', 'adj_mat_ccg_corrected'), file))
      ccg_jittered = load_sparse_npz(os.path.join(directory.replace('adj_mat_ccg_highland_corrected', 'adj_mat_ccg_corrected'), file.replace('.npz', '_bl.npz')))
      ccg_corrected = ccg - ccg_jittered
      significant_inds = list(zip(*np.where(~np.isnan(significant_ccg))))
      mouseID = file.split('_')[0]
      stimulus_name = file.replace('.npz', '').replace(mouseID + '_', '')
      inds_2plot = []
      for row_a, row_b in significant_inds:
        filter = np.array([1]).repeat(significant_duration[row_a,row_b]+1) # sum instead of mean
        sig = signal.convolve(ccg_corrected[row_a, row_b], filter, mode='valid', method='fft')
        # uniq = unique_with_tolerance(sig, 1e-7)
        # if len(uniq)/len(sig) >= threshold:
        entropy = normalized_entropy_with_tolerance(sig, TOL=1e-7)
        if entropy < threshold:
          inds_2plot.append((row_a, row_b))
  fig = plt.figure(figsize=(5*3, 5*3))
  for ind, (row_a, row_b) in enumerate(inds_2plot[:9]):
    ax = plt.subplot(3, 3, ind+1)
    filter = np.array([1]).repeat(significant_duration[row_a,row_b]+1) # sum instead of mean
    ccg_plot = signal.convolve(ccg_corrected[row_a, row_b], filter, mode='valid', method='fft')
    highland_lag = np.array([int(significant_offset[row_a,row_b])])
    plt.plot(np.arange(len(ccg_plot)), ccg_plot)
    plt.plot(highland_lag, ccg_plot[highland_lag], 'r.--', markersize=12, alpha=0.6)
    # uniq = unique_with_tolerance(ccg_plot, 1e-7)
    # plt.title(len(uniq)/len(ccg_plot))
    entropy = normalized_entropy_with_tolerance(ccg_plot, TOL=1e-7)
    plt.title(entropy)
    if ind % 3 == 0:
      plt.ylabel('signigicant CCG corrected', size=20)
    if ind // 3 == 3 - 1:
      plt.xlabel('time lag (ms)', size=20)
  # plt.suptitle('{} fold\n{}, {}'.format(n, mouseID, stimulus_name), size=25)
  plt.savefig('./plots/sample_significant_ccg_{}fold_below_threshold_entropy_{}_{}.jpg'.format(n, mouseID, stimulus_name))
  # plt.show()

def get_normalized_entropy_distri(directory):
  distri = defaultdict(lambda: [])
  files = os.listdir(directory)
  files.sort(key=lambda x:int(x[:9]))
  for file in files:
    if file.endswith(".npz") and ('gabor' not in file) and ('flashes' not in file) and ('_offset' not in file) and ('_duration' not in file) and ('_bl' not in file) and ('confidence' not in file):
      print(file)
      significant_ccg = load_npz_3d(os.path.join(directory, file))
      significant_duration = load_npz_3d(os.path.join(directory, file.replace('.npz', '_duration.npz')))
      ccg = load_sparse_npz(os.path.join(directory.replace('adj_mat_ccg_highland_corrected', 'adj_mat_ccg_corrected'), file))
      ccg_jittered = load_sparse_npz(os.path.join(directory.replace('adj_mat_ccg_highland_corrected', 'adj_mat_ccg_corrected'), file.replace('.npz', '_bl.npz')))
      ccg_corrected = ccg - ccg_jittered
      significant_inds = list(zip(*np.where(~np.isnan(significant_ccg))))
      mouseID = file.split('_')[0]
      stimulus_name = file.replace('.npz', '').replace(mouseID + '_', '')
      for row_a, row_b in significant_inds:
        filter = np.array([1]).repeat(significant_duration[row_a,row_b]+1) # sum instead of mean
        sig = signal.convolve(ccg_corrected[row_a, row_b], filter, mode='valid', method='fft')
        distri[stimulus_name].append(normalized_entropy_with_tolerance(sig, 1e-7))
  ordered_dict = {stimulus_name:distri[stimulus_name] for stimulus_name in stimulus_names}
  return ordered_dict

def plot_normalized_entropy(distri, measure, n):
  cols = list(distri.keys())
  fig = plt.figure(figsize=(9*len(cols)/2, 6*2))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  ind = 1
  for col in distri:
    print(col)
    ax=plt.subplot(2, int(np.ceil(len(cols)/2)), ind)
    plt.gca().set_title(col, fontsize=30, rotation=0)
    plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ind += 1
    sns.histplot(data=distri[col], linewidth=0, stat='probability')
    plt.axvline(x=np.nanmean(distri[col]), color='r', linestyle='--')
    # plt.xlim(0, 1)
    plt.xlabel('normalized entropy', size=20)
    plt.ylabel('probability', size=20)
  plt.tight_layout(rect=[0, 0.03, 1, 0.98])
  figname = './plots/normalized_entropy_{}_{}fold.jpg'.format(measure, n)
  plt.savefig(figname)

def keep_edges_above_threshold(directory, threshold):
  files = os.listdir(directory)
  files.sort(key=lambda x:int(x[:9]))
  for file in files:
    if file.endswith(".npz") and ('gabors' not in file) and ('flashes' not in file) and ('_offset' not in file) and ('_duration' not in file) and ('_bl' not in file) and ('confidence' not in file):
      print(file)
      significant_ccg = load_npz_3d(os.path.join(directory, file))
      significant_duration = load_npz_3d(os.path.join(directory, file.replace('.npz', '_duration.npz')))
      significant_offset = load_npz_3d(os.path.join(directory, file.replace('.npz', '_offset.npz')))
      significant_confidence = load_npz_3d(os.path.join(directory, file.replace('.npz', '_confidence.npz')))
      ccg = load_sparse_npz(os.path.join(directory.replace('adj_mat_ccg_highland_corrected', 'adj_mat_ccg_corrected'), file))
      ccg_jittered = load_sparse_npz(os.path.join(directory.replace('adj_mat_ccg_highland_corrected', 'adj_mat_ccg_corrected'), file.replace('.npz', '_bl.npz')))
      ccg_corrected = ccg - ccg_jittered
      significant_inds = list(zip(*np.where(~np.isnan(significant_ccg))))
      row_as, row_bs = [], []
      for row_a, row_b in significant_inds:
        filter = np.array([1]).repeat(significant_duration[row_a,row_b]+1) # sum instead of mean
        sig = signal.convolve(ccg_corrected[row_a, row_b], filter, mode='valid', method='fft')
        # uniq = unique_with_tolerance(sig, 1e-7)
        # if len(uniq)/len(sig) >= threshold:
        entropy = normalized_entropy_with_tolerance(sig, TOL=1e-7)
        if entropy >= threshold:
          row_as.append(row_a)
          row_bs.append(row_b)
      print(len(row_as)/len(significant_inds))
      inds = (np.array(row_as), np.array(row_bs))
      filtered_ccg,filtered_confidence,filtered_offset,filtered_duration= [np.zeros_like(significant_ccg), np.zeros_like(significant_ccg), np.zeros_like(significant_ccg), np.zeros_like(significant_ccg)]
      filtered_ccg[:] = np.nan
      filtered_confidence[:] = np.nan
      filtered_offset[:] = np.nan
      filtered_duration[:] = np.nan
      filtered_ccg[inds] = significant_ccg[inds]
      filtered_confidence[inds] = significant_confidence[inds]
      filtered_offset[inds] = significant_offset[inds]
      filtered_duration[inds] = significant_duration[inds]
      print(len(np.where(~np.isnan(significant_ccg))[0]), len(np.where(~np.isnan(significant_confidence))[0]), len(np.where(~np.isnan(significant_offset))[0]), len(np.where(~np.isnan(significant_duration))[0]))
      print(len(np.where(~np.isnan(filtered_ccg))[0]), len(np.where(~np.isnan(filtered_confidence))[0]), len(np.where(~np.isnan(filtered_offset))[0]), len(np.where(~np.isnan(filtered_duration))[0]))
      save_npz(filtered_ccg, os.path.join(directory, file))
      save_npz(filtered_confidence, os.path.join(directory, file.replace('.npz', '_confidence.npz')))
      save_npz(filtered_offset, os.path.join(directory, file.replace('.npz', '_offset.npz')))
      save_npz(filtered_duration, os.path.join(directory, file.replace('.npz', '_duration.npz')))
