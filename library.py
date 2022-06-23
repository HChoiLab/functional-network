#%%
import numpy as np
import numpy.ma as ma
import pandas as pd
import os
import itertools
import scipy
from scipy.stats import shapiro
from scipy.stats import normaltest
from tqdm import tqdm
import pickle
import time
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
import statsmodels.stats.weightstats as ws
import networkx as nx
import community
import seaborn as sns
from numpy.lib.stride_tricks import as_strided
from scipy.ndimage.filters import uniform_filter1d
from scipy import signal
from plfit import plfit
from scipy import sparse
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
np.seterr(divide='ignore', invalid='ignore')

customPalette = ['#630C3A', '#39C8C6', '#D3500C', '#FFB139', 'palegreen', 'darkblue', 'slategray', '#a6cee3', '#b2df8a', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#6a3d9a', '#b15928']

visual_regions = ['VISp', 'VISl', 'VISrl', 'VISal', 'VISpm', 'VISam'] #, 'LGd', 'LP'
session_ids = [719161530, 750332458, 750749662, 754312389, 755434585, 756029989, 791319847, 797828357]
stimulus_names = ['spontaneous', 'flashes', 'gabors',
        'drifting_gratings', 'static_gratings',
          'natural_scenes', 'natural_movie_one', 'natural_movie_three']
class pattern_jitter():
    def __init__(self, num_sample, sequences, L, R=None, memory=True):
        super(pattern_jitter,self).__init__()
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
  def __init__(self):
    super(CommunityLayout,self).__init__()
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
    pos_communities = self._position_communities(g, partition, scale=3.)
    pos_nodes = self._position_nodes(g, partition, scale=1.)
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
  stimulus_rank = ['spontaneous', 'flashes', 'gabors',
      'drifting_gratings', 'static_gratings', 'drifting_gratings_contrast',
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
        ccg[row_b, row_a, :] = (T @ px) / ((M-np.arange(window+1))/1000 * np.sqrt(firing_rates[row_a] * firing_rates[row_b]))
    else:
        ccg[row_b, row_a, :] = np.zeros(window+1)
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
    if file.endswith(".npz") and ('gabors' not in file) and ('_offset' not in file) and ('_duration' not in file) and ('_bl' not in file) and ('confidence' not in file):
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

def remove_gabor(G_dict):
  for key in G_dict:
    G_dict[key].pop('gabors', None)
  return G_dict

def remove_thalamic(G_dict, area_dict, regions):
  rows, cols = get_rowcol(G_dict)
  for row in rows:
    for col in cols:
      G = G_dict[row][col]
      nodes = [n for n in G.nodes() if area_dict[row][n] in regions]
      G_dict[row][col] = G.subgraph(nodes)
  return G_dict

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
    print(row)
    for col in cols:
      print(col)
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

def stat_modular_structure(pos_G_dict, measure, n, neg_G_dict=None):
  rows, cols = get_rowcol(pos_G_dict)
  pos_w_num_comm, neg_w_num_comm, pos_w_modularity, neg_w_modularity, \
  pos_uw_num_comm, neg_uw_num_comm, pos_uw_modularity, neg_uw_modularity, \
  pos_w_num_lcomm, neg_uw_num_lcomm, neg_w_num_lcomm, pos_uw_num_lcomm, \
  pos_w_cov_lcomm, neg_uw_cov_lcomm, neg_w_cov_lcomm, pos_uw_cov_lcomm = [np.full([len(rows), len(cols)], np.nan) for _ in range(16)]
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      pos_G = pos_G_dict[row][col].copy() if col in pos_G_dict[row] else nx.DiGraph()
      if nx.is_directed(pos_G):
        pos_G = pos_G.to_undirected()
      if neg_G_dict is not None:
        part = community.best_partition(pos_G, weight='weight')
        metric = community.modularity(part, pos_G, weight='weight')
        pos_w_num_comm[row_ind, col_ind] = len(set(list(part.values())))
        pos_w_modularity[row_ind, col_ind] = metric
        _, count = np.unique(list(part.values()), return_counts=True)
        pos_w_num_lcomm[row_ind, col_ind] = sum(count >= 4) # size at least 4 is considered (large) community
        pos_w_cov_lcomm[row_ind, col_ind] = count[count >=4].sum() / pos_G.number_of_nodes()
        assert pos_G.number_of_nodes() == count.sum()
        neg_G = neg_G_dict[row][col].copy() if col in neg_G_dict[row] else nx.DiGraph()
        if nx.is_directed(neg_G):
          neg_G = neg_G.to_undirected()
        part = community.best_partition(neg_G, weight='weight')
        metric = community.modularity(part, neg_G, weight='weight')
        neg_w_num_comm[row_ind, col_ind] = len(set(list(part.values())))
        neg_w_modularity[row_ind, col_ind] = metric
        _, count = np.unique(list(part.values()), return_counts=True)
        neg_w_num_lcomm[row_ind, col_ind] = sum(count >= 4)
        neg_w_cov_lcomm[row_ind, col_ind] = count[count >=4].sum() / neg_G.number_of_nodes()
        assert neg_G.number_of_nodes() == count.sum()
        unweight = {(i, j):1 for i,j in neg_G.edges()}
        nx.set_edge_attributes(neg_G, unweight, 'weight')
        part = community.best_partition(neg_G, weight='weight')
        metric = community.modularity(part, neg_G, weight='weight')
        neg_uw_num_comm[row_ind, col_ind] = len(set(list(part.values())))
        neg_uw_modularity[row_ind, col_ind] = metric
        _, count = np.unique(list(part.values()), return_counts=True)
        neg_uw_num_lcomm[row_ind, col_ind] = sum(count >= 4)
        neg_uw_cov_lcomm[row_ind, col_ind] = count[count >=4].sum() / neg_G.number_of_nodes()
        assert neg_G.number_of_nodes() == count.sum()
      unweight = {(i, j):1 for i,j in pos_G.edges()}
      nx.set_edge_attributes(pos_G, unweight, 'weight')
      part = community.best_partition(pos_G, weight='weight')
      metric = community.modularity(part, pos_G, weight='weight')
      pos_uw_num_comm[row_ind, col_ind] = len(set(list(part.values())))
      pos_uw_modularity[row_ind, col_ind] = metric
      _, count = np.unique(list(part.values()), return_counts=True)
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
  figname = './plots/stat_modular_pos_neg_{}_{}fold.jpg' if neg_G_dict is not None else './plots/stat_modular_total_{}_{}fold.jpg'
  plt.savefig(figname.format(measure, n))

def dict2histogram(dictionary, density=True, binning=False):
    dataseq=[v for k, v in dictionary.items()]
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

def size_of_each_community(G_dict, sign, measure, n):
  rows, cols = get_rowcol(G_dict)
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
      part = community.best_partition(G, weight='weight')
      # metric = community.modularity(part, G, weight='weight')
      number, freq = dict2histogram(part, density=False, binning=False)
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
  plt.tight_layout()
  image_name = './plots/size_of_each_community_{}_{}_{}fold.jpg'.format(sign, measure, n)
  plt.savefig(image_name)

def distribution_community_size(G_dict, sign, measure, n):
  rows, cols = get_rowcol(G_dict)
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
      part = community.best_partition(G, weight='weight')
      # metric = community.modularity(part, G, weight='weight')
      _, freq = dict2histogram(part, density=False, binning=False)
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
  plt.suptitle('{} community size distribution'.format(sign), size=25)
  plt.tight_layout()
  image_name = './plots/comm_distribution_size_{}_{}_{}fold.jpg'.format(sign, measure, n)
  plt.savefig(image_name)

def plot_region_degree(G_dict, area_dict, regions, measure, n, sign):
# def plot_multi_data_dist(data_dict, measure, n, name):
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

def region_large_comm(pos_G_dict, area_dict, regions, measure, n, neg_G_dict=None):
  rows, cols = get_rowcol(pos_G_dict)
  pos_w_frac_lcomm, neg_uw_frac_lcomm, neg_w_frac_lcomm, pos_uw_frac_lcomm = [np.full([len(rows), len(cols), len(regions)], np.nan) for _ in range(4)]
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      pos_G = pos_G_dict[row][col].copy() if col in pos_G_dict[row] else nx.DiGraph()
      if nx.is_directed(pos_G):
        pos_G = pos_G.to_undirected()
      if neg_G_dict is not None:
        part = community.best_partition(pos_G, weight='weight')
        uniq, count = np.unique(list(part.values()), return_counts=True)
        nodes = list(pos_G.nodes())
        large_comms = uniq[count >= 4]
        for r_ind, r in enumerate(regions):
          r_nodes = [n for n in nodes if area_dict[row][n] == r]
          lcomm_nodes = [n for n in r_nodes if part[n] in large_comms]
          pos_w_frac_lcomm[row_ind, col_ind, r_ind] = len(lcomm_nodes) / len(r_nodes) if len(r_nodes) else 0
        neg_G = neg_G_dict[row][col].copy() if col in neg_G_dict[row] else nx.DiGraph()
        if nx.is_directed(neg_G):
          neg_G = neg_G.to_undirected()
        part = community.best_partition(neg_G, weight='weight')
        uniq, count = np.unique(list(part.values()), return_counts=True)
        nodes = list(neg_G.nodes())
        large_comms = uniq[count >= 4]
        for r_ind, r in enumerate(regions):
          r_nodes = [n for n in nodes if area_dict[row][n] == r]
          lcomm_nodes = [n for n in r_nodes if part[n] in large_comms]
          neg_w_frac_lcomm[row_ind, col_ind, r_ind] = len(lcomm_nodes) / len(r_nodes) if len(r_nodes) else 0
        unweight = {(i, j):1 for i,j in neg_G.edges()}
        nx.set_edge_attributes(neg_G, unweight, 'weight')
        part = community.best_partition(neg_G, weight='weight')
        uniq, count = np.unique(list(part.values()), return_counts=True)
        nodes = list(neg_G.nodes())
        large_comms = uniq[count >= 4]
        for r_ind, r in enumerate(regions):
          r_nodes = [n for n in nodes if area_dict[row][n] == r]
          lcomm_nodes = [n for n in r_nodes if part[n] in large_comms]
          neg_uw_frac_lcomm[row_ind, col_ind, r_ind] = len(lcomm_nodes) / len(r_nodes) if len(r_nodes) else 0
      unweight = {(i, j):1 for i,j in pos_G.edges()}
      nx.set_edge_attributes(pos_G, unweight, 'weight')
      part = community.best_partition(pos_G, weight='weight')
      uniq, count = np.unique(list(part.values()), return_counts=True)
      nodes = list(pos_G.nodes())
      large_comms = uniq[count >= 4]
      for r_ind, r in enumerate(regions):
        r_nodes = [n for n in nodes if area_dict[row][n] == r]
        lcomm_nodes = [n for n in r_nodes if part[n] in large_comms]
        pos_uw_frac_lcomm[row_ind, col_ind, r_ind] = len(lcomm_nodes) / len(r_nodes) if len(r_nodes) else 0
      
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
  figname = './plots/region_large_comm_{}_{}_{}fold.jpg'
  plt.savefig(figname.format(sign, measure, n))

def region_larg_comm_box(G_dict, area_dict, regions, measure, n, sign, weight=False):
  rows, cols = get_rowcol(G_dict)
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
      part = community.best_partition(G, weight='weight')
      uniq, count = np.unique(list(part.values()), return_counts=True)
      nodes = list(G.nodes())
      large_comms = uniq[count >= 4]
      for r_ind, r in enumerate(regions):
        r_nodes = [n for n in nodes if area_dict[row][n] == r]
        lcomm_nodes = [n for n in r_nodes if part[n] in large_comms]
        frac = len(lcomm_nodes) / len(r_nodes) if len(r_nodes) else 0
        frac_lcomm.loc[len(frac_lcomm), frac_lcomm.columns] = col, frac, r
  plt.figure(figsize=(17, 7))
  ax = sns.boxplot(x="stimulus", y="fraction", hue="region", data=frac_lcomm, palette="Set3")
  ax.set(xlabel=None)
  plt.title(name + ' percentage of region in large communities', size=15)
  plt.savefig('./plots/region_large_comm_box_{}_{}_{}fold.jpg'.format(name, measure, n))
  # plt.show()

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

def plot_multi_graphs_color(G_dict, sign, area_dict, measure, n, cc=False):
  com = CommunityLayout()
  ind = 1
  rows, cols = get_rowcol(G_dict)
  G_sample = G_dict[rows[0]][cols[0]]
  dire = True if nx.is_directed(G_sample) else False
  fig = plt.figure(figsize=(6*len(cols), 6*len(rows)))
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
        try:
          edges, weights = zip(*nx.get_edge_attributes(G,'weight').items())
          weights = np.abs(weights)
        except:
          edges = nx.edges(G)
          weights = np.ones(len(edges))
        degrees = dict(G.degree)
        try:
          if nx.is_directed(G):
            G = G.to_undirected()
          # weight cannot be negative
          if sum([n<0 for n in nx.get_edge_attributes(G, "weight").values()]):
            print('Edge weight cannot be negative for weighted modularity, setting to unweighted...')
            unweight = {(i, j):1 for i,j in G.edges()}
            nx.set_edge_attributes(G, unweight, 'weight')
          partition = community.best_partition(G, weight='weight')
          pos = com.get_community_layout(G, partition)
          metric = community.modularity(partition, G, weight='weight')
          print('Modularity: {}'.format(metric))
        except:
          print('Community detection unsuccessful!')
          pos = nx.spring_layout(G)
        areas = [G.nodes[n]['area'] for n in G.nodes()]
        areas_uniq = list(set(areas))
        colors = [customPalette[areas_uniq.index(area)] for area in areas]
        # pos = nx.spring_layout(G, k=0.8, iterations=50) # make nodes as seperate as possible
        nx.draw_networkx_edges(G, pos, arrows=dire, edgelist=edges, edge_color=weights, width=3.0, edge_cmap=plt.cm.Greens, alpha=0.9)
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
  metric = np.zeros((len(rows), len(cols), 2))
  region_connection = np.zeros((len(rows), len(cols), len(regions), len(regions)))
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      G = G_dict[row][col] if col in G_dict[row] else nx.Graph()
      if G.number_of_nodes() >= 2 and G.number_of_edges() > 0:
        nodes = list(G.nodes())
        node_area = {key: area_dict[row][key] for key in nodes}
        areas = list(node_area.values())
        active_areas = np.unique(areas)
        area_size = [areas.count(r) for r in regions]
        intra_num = sum(map(lambda x : x * (x-1), area_size)) 
        inter_num = len(nodes) * (len(nodes) - 1) - intra_num
        A = nx.to_numpy_array(G)
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
        
        diag_indx = np.eye(len(regions),dtype=bool)
        metric[row_ind, col_ind, 0] =  np.sum(region_connection[row_ind, col_ind][diag_indx]) / intra_num
        metric[row_ind, col_ind, 1] =  np.sum(region_connection[row_ind, col_ind][~diag_indx]) / inter_num
  metric_names = ['intra-region density', 'inter-region density']
  metric_stimulus = pd.DataFrame(columns=['stimulus', 'metric', 'mean', 'std'])
  for metric_ind, metric_name in enumerate(metric_names):
    df = pd.DataFrame(columns=['stimulus', 'metric', 'mean', 'std'])
    df['mean'] = np.nanmean(metric[:, :, metric_ind], axis=0)
    df['std'] = np.nanstd(metric[:, :, metric_ind], axis=0)
    df['metric'] = metric_name
    df['stimulus'] = cols
    metric_stimulus = metric_stimulus.append(df, ignore_index=True)
  # print(metric_stimulus)
  fig = plt.figure(figsize=[6, 6])
  for i, m in metric_stimulus.groupby("metric"):
    plt.plot(m['stimulus'], m['mean'], alpha=0.6, label=m['metric'].iloc[0])
    plt.fill_between(m['stimulus'], m['mean'] - m['std'], m['mean'] + m['std'], alpha=0.2)
  plt.title('{} network'.format(sign), size=20)
  plt.legend()
  plt.xticks(rotation=90)
  plt.tight_layout()
  # plt.show()
  plt.savefig('./plots/intra_inter_density_{}_{}.jpg'.format(sign, measure))
