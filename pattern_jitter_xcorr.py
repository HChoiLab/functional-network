# %%
from turtle import window_height
import numpy as np
import numpy.ma as ma
import pandas as pd
import os
import itertools
from scipy.stats import shapiro
from scipy.stats import normaltest
from tqdm import tqdm
import pickle
from scipy import sparse
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
from plfit import plfit

customPalette = ['#630C3A', '#39C8C6', '#D3500C', '#FFB139', 'palegreen', 'darkblue', 'slategray', '#a6cee3', '#b2df8a', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#6a3d9a', '#b15928']

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
  stimulus_rank = ['spontaneous', 'flashes', 'gabors',
      'drifting_gratings', 'static_gratings', 'drifting_gratings_contrast',
        'natural_scenes', 'natural_movie_one', 'natural_movie_three']
  stimulus_rank_dict = {i:stimulus_rank.index(i) for i in cols}
  stimulus_rank_dict = dict(sorted(stimulus_rank_dict.items(), key=lambda item: item[1]))
  cols = list(stimulus_rank_dict.keys())
  return rows, cols

def getSpikeTrain(spikeData):
    spikeTrain = np.squeeze(np.where(spikeData>0)).ravel()
    return spikeTrain

def spike_timing2train(T, spikeTrain):
    if len(spikeTrain.shape) == 1:
      spikeData = np.zeros(T)
      spikeData[spikeTrain.astype(int)] = 1
    else:
      spikeData = np.zeros((spikeTrain.shape[0], T))
      spikeData[np.repeat(np.arange(spikeTrain.shape[0]), spikeTrain.shape[1]), spikeTrain.ravel().astype(int)] = 1
    return spikeData

def getInitDist(L):
    initDist = np.random.rand(L)
    return initDist/initDist.sum()

def getTransitionMatrices(L, N):
    tDistMatrices = np.zeros((N - 1, L, L))
    for i in range(tDistMatrices.shape[0]):
        matrix = np.random.rand(L, L)
        stochMatrix = matrix/matrix.sum(axis=1)[:,None]
        tDistMatrices[i, :, :] = stochMatrix.astype('f')
    return tDistMatrices

def getX1(initDist, L, R, T, spikeTrain):
    # Omega = getOmega(L, obsTar)
    Gamma = getGamma(L, R, T, spikeTrain)
    randX = np.random.random()
    ind = np.where(randX <= np.cumsum(initDist))[0][0]
    return Gamma[0][ind]

def initializeX(initX, Prob):
    return initX + np.sum(Prob == 0)

def getGamma(L, R, T, spikeTrain):
    Gamma = []
    ks = [] # list of k_d
    ks.append(0)
    n = spikeTrain.size
    temp = int(spikeTrain[ks[-1]]/L)*L
    temp = max(0, temp)
    temp = min(temp, T - L)
    Gamma.append(np.arange(temp, temp + L, 1))
    for i in range(1, n):
        if spikeTrain[i] - spikeTrain[i-1] > R:
            ks.append(i)
        temp = int(spikeTrain[ks[-1]]/L)*L+spikeTrain[i]-spikeTrain[ks[-1]]
        temp = max(0, temp)
        temp = min(temp, T - L)
        Gamma.append(np.arange(temp, temp + L, 1))
    return Gamma

def getSurrogate(spikeTrain, L, R, T, initDist, tDistMatrices):
    surrogate = []
    # Omega = getOmega(L, spikeTrain)
    Gamma = getGamma(L, R, T, spikeTrain)
    givenX = getX1(initDist, L, R, T, spikeTrain)
    surrogate.append(givenX)
    for i, row in enumerate(tDistMatrices):
        if spikeTrain[i+1] - spikeTrain[i] <= R:
            givenX = surrogate[-1] + spikeTrain[i+1] - spikeTrain[i]
        else:
            index = np.where(np.array(Gamma[i]) == givenX)[0]
            p_i = np.squeeze(np.array(row[index]))
            initX = initializeX(Gamma[i + 1][0], p_i)
            randX = np.random.random()
            # safe way to find the ind
            larger = np.where(randX <= np.cumsum(p_i))[0]
            if larger.shape[0]:
                ind = larger[0]
            else:
                ind = len(p_i) - 1
            givenX = initX + np.sum(p_i[:ind]!=0)
        givenX = min(T - 1, givenX)
        surrogate.append(givenX)
    return surrogate

def sample_spiketrain(L, R, T, spikeTrain, initDist, tDistMatrices, sample_size):
    spikeTrainMat = np.zeros((sample_size, spikeTrain.size))
    for i in tqdm(range(sample_size), disable=True):
        surrogate = getSurrogate(spikeTrain, L, R, T, initDist, tDistMatrices)
        spikeTrainMat[i, :] = surrogate
    return spikeTrainMat

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
    corr = (corr - corr.mean())[:maxlag]
    max_offset = np.argmax(np.abs(corr))
    xcorr[row_a, row_b] = corr[max_offset]
    peak_offset[row_a, row_b] = max_offset
  return xcorr, peak_offset

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
    corr = (corr - corr.mean())[:maxlag]
    xcorr[0, 1, row] = corr[np.argmax(np.abs(corr))]
    px, py = norm_matb_0[row, :], norm_mata_1[row, :]
    T = as_strided(py[window:], shape=(window+1, M + window),
                    strides=(-py.strides[0], py.strides[0])) # must be py[window:], why???????????
    # corr = np.dot(T, px)
    corr = T @ px
    corr = (corr - corr.mean())[:maxlag]
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

def get_lcc(G_dict):
  for row in G_dict:
    for col in G_dict[row]:
      G = G_dict[row][col]
      if nx.is_directed(G):
        if not nx.is_weakly_connected(G):
          Gcc = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
          G_dict[row][col] = G.subgraph(Gcc[0])
      else:
        if not nx.is_connected(G):
          Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
          G_dict[row][col] = G.subgraph(Gcc[0])

          # largest_cc = max(nx.connected_components(G), key=len)
          # G_dict[row][col][i] = nx.subgraph(G, largest_cc)
      print(G.number_of_nodes(), G_dict[row][col].number_of_nodes())
  return G_dict

def print_stat(G_dict):
  for row in G_dict:
      for col in G_dict[row]:
          nodes = nx.number_of_nodes(G_dict[row][col])
          edges = nx.number_of_edges(G_dict[row][col])
          print('Number of nodes for {} {} {}'.format(row, col, nodes))
          print('Number of edges for {} {} {}'.format(row, col, edges))
          print('Density for {} {} {}'.format(row, col, nx.density(G_dict[row][col])))

def plot_stat(pos_G_dict, neg_G_dict=None, measure='xcorr'):
  rows, cols = get_rowcol(pos_G_dict)
  pos_num_nodes, neg_num_nodes, pos_num_edges, neg_num_edges, pos_densities, neg_densities, pos_total_weight, neg_total_weight, pos_mean_weight, neg_mean_weight = [np.full([len(rows), len(cols)], np.nan) for _ in range(10)]
  num_col = 2 if neg_G_dict is not None else 1
  fig = plt.figure(figsize=(5*num_col, 25))
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      pos_G = pos_G_dict[row][col] if col in pos_G_dict[row] else nx.DiGraph()
      pos_densities[row_ind, col_ind] = nx.density(pos_G)
      pos_num_nodes[row_ind, col_ind] = nx.number_of_nodes(pos_G)
      pos_num_edges[row_ind, col_ind] = nx.number_of_edges(pos_G)
      pos_total_weight[row_ind, col_ind] = np.sum(list(nx.get_edge_attributes(pos_G, "weight").values()))
      pos_mean_weight[row_ind, col_ind] = np.mean(list(nx.get_edge_attributes(pos_G, "weight").values()))
      if neg_G_dict is not None:
        neg_G = neg_G_dict[row][col] if col in neg_G_dict[row] else nx.DiGraph()
        neg_densities[row_ind, col_ind] = nx.density(neg_G)
        neg_num_nodes[row_ind, col_ind] = nx.number_of_nodes(neg_G)
        neg_num_edges[row_ind, col_ind] = nx.number_of_edges(neg_G)
        neg_total_weight[row_ind, col_ind] = np.sum(list(nx.get_edge_attributes(neg_G, "weight").values()))
        neg_mean_weight[row_ind, col_ind] = np.mean(list(nx.get_edge_attributes(neg_G, "weight").values()))
  metrics = {'positive number of nodes':pos_num_nodes, 'negative number of nodes':neg_num_nodes, 
  'positive number of edges':pos_num_edges, 'negative number of edges':neg_num_edges, 
  'positive density':pos_densities, 'negative density':neg_densities,
  'positive total weights':pos_total_weight, 'negative total weights':neg_total_weight, 
  'positive average weights':pos_mean_weight, 'negative average weights':neg_mean_weight}
  if neg_G_dict is not None:
    metrics = {'positive number of nodes':pos_num_nodes, 'negative number of nodes':neg_num_nodes, 
    'positive number of edges':pos_num_edges, 'negative number of edges':neg_num_edges, 
    'positive density':pos_densities, 'negative density':neg_densities,
    'positive total weights':pos_total_weight, 'negative total weights':neg_total_weight, 
    'positive average weights':pos_mean_weight, 'negative average weights':neg_mean_weight}
  else:
    metrics = {'positive number of nodes':pos_num_nodes, 'positive number of edges':pos_num_edges,
    'positive density':pos_densities, 'positive total weights':pos_total_weight,
    'positive average weights':pos_mean_weight}
  # distris = {'positive weight distribution':pos_weight_distri, 'negative weight distribution':neg_weight_distri}
  
  for i, k in enumerate(metrics):
    plt.subplot(5, num_col, i+1)
    metric = metrics[k]
    for row_ind, row in enumerate(rows):
      mean = metric[row_ind, :]
      plt.plot(cols, mean, label=row, alpha=0.6)
    plt.gca().set_title(k, fontsize=20, rotation=0)
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
  # plt.show()
  figname = './plots/stats_{}.jpg'.format(measure)
  plt.savefig(figname)

def region_connection_heatmap(G_dict, sign, area_dict, regions, measure):
  rows, cols = get_rowcol(G_dict)
  scale = np.zeros(len(rows))
  region_connection = np.zeros((len(rows), len(cols), len(regions), len(regions)))
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      G = G_dict[row][col] if col in G_dict[row] else nx.Graph()
      if G.number_of_nodes() > 2 and G.number_of_edges() > 0:
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
  plt.savefig('./plots/region_connection_scale_{}_{}.jpg'.format(sign, measure))

def region_connection_delta_heatmap(G_dict, sign, area_dict, regions, measure, weight):
  rows, cols = get_rowcol(G_dict)
  cols.remove('spontaneous')
  scale_min = np.zeros(len(rows))
  scale_max = np.zeros(len(rows))
  region_connection_bl = np.zeros((len(rows), len(regions), len(regions)))
  region_connection = np.zeros((len(rows), len(cols), len(regions), len(regions)))
  for row_ind, row in enumerate(rows):
    print(row)
    G = G_dict[row]['spontaneous']
    if G.number_of_nodes() > 100 and G.number_of_edges() > 100:
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
      if G.number_of_nodes() > 100 and G.number_of_edges() > 100:
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
  figname = './plots/region_connection_delta_scale_weighted_{}_{}.jpg'.format(sign, measure) if weight else './plots/region_connection_delta_scale_{}_{}.jpg'.format(sign, measure)
  plt.savefig(figname)
  # plt.savefig('./plots/region_connection_delta_scale_{}_{}.pdf'.format(measure, num), transparent=True)

def plot_multi_graphs_color(G_dict, sign, area_dict, measure, cc=False):
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
      G = G_dict[row][col] if col in G_dict[row] else nx.DiGraph()
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
          partition = community.best_partition(G)
          pos = com.get_community_layout(G, partition)
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
  image_name = './plots/graphs_region_color_cc_{}_{}.jpg'.format(sign, measure) if cc else './plots/graphs_region_color_{}_{}.jpg'.format(sign, measure)
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

def plot_directed_multi_degree_distributions(G_dict, sign, measure, weight=None, cc=False):
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
  image_name = './plots/directed_degree_distribution_weighted_{}_{}.jpg'.format(sign, measure) if weight is not None else './plots/directed_degree_distribution_unweighted_{}_{}.jpg'.format(sign, measure)
  # plt.show()
  plt.savefig(image_name, dpi=300)
  # plt.savefig(image_name.replace('jpg', 'pdf'), transparent=True)
#%%
################### effect of pattern jitter on cross correlation
####### turn off warnings
np.seterr(divide='ignore', invalid='ignore')
############# save correlation matrices #################
# min_len, min_num = (260000, 739)
min_len, min_num = (10000, 29)
min_spikes = min_len * 0.002 # 2 Hz
# measure = 'pearson'
# measure = 'cosine'
# measure = 'correlation'
# measure = 'MI'
measure = 'xcorr'
# measure = 'causality'
directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
files = os.listdir(directory)
files = [f for f in files if f.endswith('.npz')]
files.sort(key=lambda x:int(x[:9]))
path = os.path.join(directory.replace('spiking_sequence', 'adj_mat_{}'.format(measure)))
if not os.path.exists(path):
  os.makedirs(path)
num_sample = 200
num_baseline = 1
# Ls = list(np.arange(2, 101))
Ls = list(np.arange(3, 51, 2)) # L should be larger than 1 and odd
# Ls = list(np.arange(3, 101, 2)) # L should be larger than 1 and odd
Rs = [1, 100, 200, 300, 400, 500]
file = files[0] # 0, 2, 7
start_time_mouse = time.time()
print(file)
mouseID = file.replace('.npz', '').split('_')[0]
stimulus = file.replace('.npz', '').replace(mouseID + '_', '')
sequences = load_npz(os.path.join(directory, file))
sequences = sequences[np.count_nonzero(sequences[:, :min_len], axis=1) > min_spikes, :min_len]
# sequences = sequences[np.count_nonzero(sequences[:, :min_len], axis=1) > 80, :min_len]
# active_inds = np.argpartition(np.count_nonzero(sequences, axis=1), -2)[-2:] # top 2 most active neurons
active_inds = np.argpartition(np.count_nonzero(sequences, axis=1), 2)[:2] # top 2 most inactive neurons
num_nodes = 2
############## Effect of pattern jitter on cross correlation
origin_adj_mat = np.zeros((2, 2))
origin_peak_off = np.zeros((2, 2))
origin_adj_mat_bl = np.zeros((2, 2, num_baseline))
all_adj_mat_A = np.zeros((2, 2, num_sample, len(Ls), len(Rs)))
all_peak_off_A = np.zeros((2, 2, num_sample, len(Ls), len(Rs)))
adj_mat_bl_A = np.zeros((2, 2, num_baseline, len(Ls), len(Rs)))
all_adj_mat_B = np.zeros((2, 2, num_sample, len(Ls), len(Rs)))
all_peak_off_B = np.zeros((2, 2, num_sample, len(Ls), len(Rs)))
adj_mat_bl_B = np.zeros((2, 2, num_baseline, len(Ls), len(Rs)))
all_adj_mat = np.zeros((2, 2, num_sample, len(Ls), len(Rs)))
all_peak_off = np.zeros((2, 2, num_sample, len(Ls), len(Rs)))
adj_mat_bl = np.zeros((2, 2, num_baseline, len(Ls), len(Rs)))
# complete_adj_mat = corr_mat(sequences, measure, maxlag=12)
T = min_len
origin_adj_mat, origin_peak_off = corr_mat(sequences[active_inds], measure, maxlag=12)
seq = sequences[active_inds].copy()
# for b in range(num_baseline):
#   for n in range(num_nodes):
#     np.random.shuffle(seq[n,:])
#   adj_mat = corr_mat(seq, measure)
#   origin_adj_mat_bl[:, :, b] = adj_mat
#%%
start_time = time.time()
start_time_A = start_time
print('Sampling neuron A...')
for L_ind, L in enumerate(Ls):
  for R_ind, R in enumerate(Rs):
    spikeTrain = getSpikeTrain(sequences[active_inds[0], :])
    N = len(spikeTrain)
    initDist = getInitDist(L)
    tDistMatrices = getTransitionMatrices(L, N)
    sampled_spiketrain = sample_spiketrain(L, R, T, spikeTrain, initDist, tDistMatrices, num_sample)
    all_adj_mat_A[:, :, :, L_ind, R_ind] = n_cross_correlation8_2mat(spike_timing2train(min_len, sampled_spiketrain), sequences[active_inds[1], :])
    for i in range(num_sample):
      sample_seq = np.concatenate((spike_timing2train(min_len, sampled_spiketrain[i, :])[None, :], sequences[active_inds[1], :][None, :]) ,axis=0)
      adj_mat, peak_offset = corr_mat(sample_seq, measure, maxlag=12)
      all_adj_mat_A[:, :, i, L_ind, R_ind] = adj_mat
      all_peak_off_A[:, :, i, L_ind, R_ind] = peak_offset
    # for b in range(num_baseline):
    #   sample_seq = np.concatenate((spike_timing2train(min_len, sampled_spiketrain[b, :])[None, :], sequences[active_inds[1], :][None, :]) ,axis=0)
    #   # print('Baseline {} out of {}'.format(b+1, num_baseline))
    #   for n in range(num_nodes):
    #     np.random.shuffle(sample_seq[n,:])
    #   adj_mat = corr_mat(sample_seq, measure, maxlag=12)
    #   adj_mat_bl_A[:, :, b, L_ind, R_ind] = adj_mat
print("--- %s minutes" % ((time.time() - start_time_A)/60))
start_time_B = time.time()
print('Sampling neuron B...')
for L_ind, L in enumerate(Ls):
  for R_ind, R in enumerate(Rs):
    spikeTrain = getSpikeTrain(sequences[active_inds[1], :])
    N = len(spikeTrain)
    initDist = getInitDist(L)
    tDistMatrices = getTransitionMatrices(L, N)
    sampled_spiketrain = sample_spiketrain(L, R, T, spikeTrain, initDist, tDistMatrices, num_sample)
    all_adj_mat_B[:, :, :, L_ind, R_ind] = n_cross_correlation8_2mat(sequences[active_inds[0], :], spike_timing2train(min_len, sampled_spiketrain))
    for i in range(num_sample):
      sample_seq = np.concatenate((sequences[active_inds[0], :][None, :], spike_timing2train(min_len, sampled_spiketrain[i, :])[None, :]) ,axis=0)
      adj_mat, peak_offset = corr_mat(sample_seq, measure, maxlag=12)
      all_adj_mat_B[:, :, i, L_ind, R_ind] = adj_mat
      all_peak_off_B[:, :, i, L_ind, R_ind] = peak_offset
    # for b in range(num_baseline):
    #   sample_seq = np.concatenate((sequences[active_inds[0], :][None, :], spike_timing2train(min_len, sampled_spiketrain[b, :])[None, :]) ,axis=0)
    #   # print('Baseline {} out of {}'.format(b+1, num_baseline))
    #   for n in range(num_nodes):
    #     np.random.shuffle(sample_seq[n,:])
    #   adj_mat = corr_mat(sample_seq, measure, maxlag=12)
    #   adj_mat_bl_B[:, :, b, L_ind, R_ind] = adj_mat
print("--- %s minutes" % ((time.time() - start_time_B)/60))
start_time_both = time.time()
print('Sampling neurons A and B...')
for L_ind, L in enumerate(Ls):
  for R_ind, R in enumerate(Rs):
    spikeTrain = getSpikeTrain(sequences[active_inds[0], :])
    N = len(spikeTrain)
    initDist = getInitDist(L)
    tDistMatrices = getTransitionMatrices(L, N)
    sampled_spiketrain1 = sample_spiketrain(L, R, T, spikeTrain, initDist, tDistMatrices, num_sample)
    spikeTrain = getSpikeTrain(sequences[active_inds[1], :])
    N = len(spikeTrain)
    initDist = getInitDist(L)
    tDistMatrices = getTransitionMatrices(L, N)
    sampled_spiketrain2 = sample_spiketrain(L, R, T, spikeTrain, initDist, tDistMatrices, num_sample)
    all_adj_mat[:, :, :, L_ind, R_ind] = n_cross_correlation8_2mat(spike_timing2train(min_len, sampled_spiketrain1), spike_timing2train(min_len, sampled_spiketrain2))
    for i in range(num_sample):
      sample_seq = np.concatenate((spike_timing2train(min_len, sampled_spiketrain1[i, :])[None, :], spike_timing2train(min_len, sampled_spiketrain2[i, :])[None, :]), axis=0)
      adj_mat, peak_offset = corr_mat(sample_seq, measure, maxlag=12)
      all_adj_mat[:, :, i, L_ind, R_ind] = adj_mat
      all_peak_off[:, :, i, L_ind, R_ind] = peak_offset
    # for b in range(num_baseline):
    #   sample_seq = np.concatenate((spike_timing2train(min_len, sampled_spiketrain1[b, :])[None, :], spike_timing2train(min_len, sampled_spiketrain2[b, :])[None, :]), axis=0)
    #   # print('Baseline {} out of {}'.format(b+1, num_baseline))
    #   for n in range(num_nodes):
    #     np.random.shuffle(sample_seq[n,:])
    #   adj_mat = corr_mat(sample_seq, measure, maxlag=12)
    #   adj_mat_bl[:, :, b, L_ind, R_ind] = adj_mat
print("--- %s minutes" % ((time.time() - start_time_both)/60))
print("--- %s minutes in total" % ((time.time() - start_time)/60))
# %%
############## one pair of neurons, significant xcorr vs L and R
def plot_xcorr_LR(origin_adj_mat, all_adj_mat_A, all_adj_mat_B, all_adj_mat, Ls, R, Rs, edge_type='active'):
  plt.figure(figsize=(20, 6))
  all_mat = [all_adj_mat_A, all_adj_mat_B, all_adj_mat]
  titles = ['Pattern jittering neuron A', 'Pattern jittering neuron B', 'Pattern jittering neurons A and B']
  for i in range(3):
    plt.subplot(1, 3, i + 1)
    mean = np.nanmean(all_mat[i][0, 1, :, :, Rs.index(R)], axis=0)
    std = np.nanstd(all_mat[i][0, 1, :, :, Rs.index(R)], axis=0)
    plt.plot(Ls, mean, alpha=0.6, label='A->B')
    plt.fill_between(Ls, (mean - std), (mean + std), alpha=0.2)
    mean = np.nanmean(all_mat[i][1, 0, :, :, Rs.index(R)], axis=0)
    std = np.nanstd(all_mat[i][1, 0, :, :, Rs.index(R)], axis=0)
    plt.plot(Ls, mean, alpha=0.6, label='B->A')
    plt.fill_between(Ls, (mean - std), (mean + std), alpha=0.2)
    plt.plot(Ls, len(Ls) * [origin_adj_mat[0, 1]], 'b--', alpha=0.6, label='original A->B')
    plt.plot(Ls, len(Ls) * [origin_adj_mat[1, 0]], 'r--', alpha=0.6, label='original B->A')
    plt.gca().set_title(titles[i] + ', R={}'.format(R), fontsize=20, rotation=0)
    plt.xscale('log')
    plt.xticks(rotation=90)
    plt.xlabel('Bin size L', fontsize=15)
    plt.ylabel('cross correlation', fontsize=15)
    plt.legend()
  plt.tight_layout()
  # plt.show()
  figname = './plots/xcorr_vs_L_R_{}_{}_{}.jpg'.format(R, edge_type, measure)
  plt.savefig(figname)

for R in Rs:
  plot_xcorr_LR(origin_adj_mat, all_adj_mat_A, all_adj_mat_B, all_adj_mat, Ls, R, Rs, 'inactive')

# %%
################ is cross correlation affected by firing rate?
# adj_mat = corr_mat(sequences, measure, maxlag=12, noprogressbar=False)
# np.fill_diagonal(adj_mat, np.nan)
# # adj_mat_flat = adj_mat[~np.eye(adj_mat.shape[0],dtype=bool)]
# firing_rates = np.count_nonzero(sequences, axis=1) / sequences.shape[1]
#%%
# source_FR = np.repeat(firing_rates[:, None], len(firing_rates), axis=1)
# source_FR_flat = source_FR[~np.eye(source_FR.shape[0],dtype=bool)]
# np.fill_diagonal(source_FR, np.nan)
# plt.figure()
# plt.scatter(source_FR_flat, adj_mat_flat, alpha=0.1)
# plt.scatter(np.nanmean(source_FR, axis=1), np.nanmean(adj_mat, axis=1), color='b', alpha = 0.5)
# plt.xlabel('FR of source neuron')
# plt.ylabel('cross correlation')
# plt.tight_layout()
# plt.savefig('./plots/xcorr_FR_source.jpg')
# # plt.show()
# # %%
# target_FR = np.repeat(firing_rates[None, :], len(firing_rates), axis=0)
# target_FR_flat = target_FR[~np.eye(target_FR.shape[0],dtype=bool)]
# np.fill_diagonal(target_FR, np.nan)
# plt.figure()
# plt.scatter(target_FR_flat, adj_mat_flat, alpha=0.1)
# plt.scatter(np.nanmean(target_FR, axis=0), np.nanmean(adj_mat, axis=0), color='b', alpha = 0.5)
# plt.xlabel('FR of target neuron')
# plt.ylabel('cross correlation')
# plt.tight_layout()
# plt.savefig('./plots/xcorr_FR_target.jpg')
# # plt.show()
# # %%
# avg_FR = ((firing_rates[None, :] + firing_rates[:, None]) / 2)
# avg_FR = avg_FR[~np.eye(avg_FR.shape[0],dtype=bool)]
# plt.figure()
# plt.scatter(avg_FR, adj_mat_flat, alpha=0.1)
# uniq_FR = np.unique(avg_FR)
# corr = np.zeros_like(uniq_FR)
# for i in range(len(uniq_FR)):
#   corr[i] = np.mean(adj_mat_flat[np.where(avg_FR==uniq_FR[i])])
# plt.scatter(uniq_FR, corr, color='b', alpha = 0.5)
# plt.xlabel('average FR of source and target neurons')
# plt.ylabel('cross correlation')
# plt.tight_layout()
# plt.savefig('./plots/xcorr_FR_avg.jpg')
#%%
# bins=np.logspace(start=np.log10(firing_rates.min()), stop=np.log10(firing_rates.max()+0.0001), num=50)
# bin_num = np.digitize(firing_rates, bins)
# # %%
# corr = np.zeros((len(bins), len(bins)))
# for i in range(1, len(bins)):
#   for j in range(1, len(bins)):
#     corr[i, j] = np.nanmean(adj_mat[np.where(bin_num==i)[0][:, None], np.where(bin_num==j)[0][None, :]])
# plt.xscale('log')
# %%
################### heatmap of xcrorr vs FR
# start_time = time.time()
# min_len, min_num = (10000, 29)
# min_spikes = min_len * 0.002 # 2 Hz
# # measure = 'pearson'
# # measure = 'cosine'
# # measure = 'correlation'
# # measure = 'MI'
# measure = 'xcorr'
# # measure = 'causality'
# directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
# stimulus_names = ['spontaneous', 'flashes', 'gabors',
#         'drifting_gratings', 'static_gratings',
#           'natural_scenes', 'natural_movie_one', 'natural_movie_three']
# session_ids = [719161530, 750749662, 755434585, 756029989, 791319847]
# xcorr_dict, bin_dict = {}, {}
# for session_id in session_ids:
#   print(session_id)
#   xcorr_dict[session_id], bin_dict[session_id] = {}, {}
#   for stimulus_name in stimulus_names:
#     print(stimulus_name)
#     sequences = load_npz(os.path.join(directory, str(session_id) + '_' + stimulus_name + '.npz'))
#     sequences = sequences[np.count_nonzero(sequences[:, :min_len], axis=1) > min_spikes, :min_len]
#     adj_mat = corr_mat(sequences, measure, maxlag=12, noprogressbar=False)
#     np.fill_diagonal(adj_mat, np.nan)
#     # adj_mat_flat = adj_mat[~np.eye(adj_mat.shape[0],dtype=bool)]
#     firing_rates = np.count_nonzero(sequences, axis=1) / sequences.shape[1]
#     bins=np.logspace(start=np.log10(firing_rates.min()), stop=np.log10(firing_rates.max()+0.0001), num=50)
#     bin_num = np.digitize(firing_rates, bins)
#     xcorr = np.zeros((len(bins), len(bins)))
#     for i in range(1, len(bins)):
#       for j in range(1, len(bins)):
#         xcorr[i, j] = np.nanmean(adj_mat[np.where(bin_num==i)[0][:, None], np.where(bin_num==j)[0][None, :]])
#     xcorr_dict[session_id][stimulus_name] = xcorr
#     bin_dict[session_id][stimulus_name] = bins
# print("--- %s minutes in total" % ((time.time() - start_time)/60))
#%%
# plot_multi_heatmap_xcorr_FR(session_ids, stimulus_names, xcorr_dict, bin_dict)
#%%
# min_len, min_num = (10000, 29)
# min_spikes = min_len * 0.002 # 2 Hz
# # measure = 'pearson'
# # measure = 'cosine'
# # measure = 'correlation'
# # measure = 'MI'
# measure = 'xcorr'
# # measure = 'causality'
# directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
# stimulus_names = ['spontaneous', 'flashes', 'gabors',
#         'drifting_gratings', 'static_gratings',
#           'natural_scenes', 'natural_movie_one', 'natural_movie_three']
# session_ids = [719161530, 750749662, 755434585, 756029989, 791319847]
# FR_dict = {}
# for session_id in session_ids:
#   print(session_id)
#   FR_dict[session_id] = {}
#   for stimulus_name in stimulus_names:
#     print(stimulus_name)
#     sequences = load_npz(os.path.join(directory, str(session_id) + '_' + stimulus_name + '.npz'))
#     sequences = sequences[np.count_nonzero(sequences[:, :min_len], axis=1) > min_spikes, :min_len]
#     firing_rates = np.count_nonzero(sequences, axis=1) / sequences.shape[1]
#     FR_dict[session_id][stimulus_name] = firing_rates
# %%
def func_powerlaw(x, m, c):
  return x**m * c
def plot_firing_rate_distributions(FR_dict, measure):
  alphas = pd.DataFrame(index=session_ids, columns=stimulus_names)
  xmins = pd.DataFrame(index=session_ids, columns=stimulus_names)
  loglikelihoods = pd.DataFrame(index=session_ids, columns=stimulus_names)
  proportions = pd.DataFrame(index=session_ids, columns=stimulus_names)
  ind = 1
  rows, cols = get_rowcol(FR_dict)
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
      firing_rates = FR_dict[row][col].tolist()
      hist, bin_edges = np.histogram(firing_rates, bins=50)
      bins = (bin_edges[:-1] + bin_edges[1:]) / 2
      plt.plot(bins, np.array(hist) / sum(hist),'go-')
      [alpha, xmin, L] = plfit(firing_rates, 'finite')
      proportion = np.sum(np.array(firing_rates)>=xmin)/len(firing_rates)
      alphas.loc[int(row)][col], xmins.loc[int(row)][col], loglikelihoods.loc[int(row)][col], proportions.loc[int(row)][col] = alpha, xmin, L, proportion
      C = (np.array(hist) / sum(hist))[bins>=xmin].sum() / np.power(bins[bins>=xmin], -alpha).sum()
      plt.scatter([],[], label='alpha={:.1f}'.format(alpha), s=20)
      plt.scatter([],[], label='xmin={}'.format(xmin), s=20)
      plt.scatter([],[], label='loglikelihood={:.1f}'.format(L), s=20)
      plt.plot(bins[bins>=xmin], func_powerlaw(bins[bins>=xmin], *np.array([-alpha, C])), linestyle='--', linewidth=2, color='black')
      
      plt.legend(loc='upper right', fontsize=7)
      plt.xlabel('firing rate')
      plt.ylabel('Frequency')
      plt.xscale('log')
      plt.yscale('log')
      
  plt.tight_layout()
  image_name = './plots/FR_distribution.jpg'
  # plt.show()
  plt.savefig(image_name)

# plot_firing_rate_distributions(FR_dict, measure)
# %%
# np.seterr(divide='ignore', invalid='ignore')
# ############# save correlation matrices #################
# # min_len, min_num = (260000, 739)
# min_len, min_num = (10000, 29)
# min_spikes = min_len * 0.002 # 2 Hz
# # measure = 'pearson'
# # measure = 'cosine'
# # measure = 'correlation'
# # measure = 'MI'
# measure = 'xcorr'
# # measure = 'causality'
# directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
# files = os.listdir(directory)
# files = [f for f in files if f.endswith('.npz')]
# files.sort(key=lambda x:int(x[:9]))
# path = os.path.join(directory.replace('spiking_sequence', 'adj_mat_{}'.format(measure)))
# if not os.path.exists(path):
#   os.makedirs(path)
# num_sample = 1000
# num_baseline = 2
# num = 10
# L = 10
# R_list = [1, 20, 50, 100, 500, 1000]
# T = min_len
# file = files[0]
# start_time_mouse = time.time()
# print(file)
# mouseID = file.replace('.npz', '').split('_')[0]
# stimulus = file.replace('.npz', '').replace(mouseID + '_', '')
# sequences = load_npz(os.path.join(directory, file))
# sequences = sequences[np.count_nonzero(sequences[:, :min_len], axis=1) > min_spikes, :min_len]
# # sequences = sequences[np.count_nonzero(sequences[:, :min_len], axis=1) > 80, :min_len]
# # active_inds = np.argpartition(np.count_nonzero(sequences, axis=1), -1)[-1:] # top 1 most active neurons
# active_inds = np.argpartition(np.count_nonzero(sequences, axis=1), 1)[:1] # top 1 most inactive neurons
# spikeTrain = getSpikeTrain(sequences[active_inds, :].squeeze())
# N = len(spikeTrain)
# initDist = getInitDist(L)
# tDistMatrices = getTransitionMatrices(L, N)
# Palette = np.array(['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

# colors = np.concatenate((np.array(['r']), np.repeat(Palette[:len(R_list)], num)))
# all_spiketrain = spikeTrain[None, :]
# for R in R_list:
#     print(R)
#     sampled_spiketrain = sample_spiketrain(L, R, T, spikeTrain, initDist, tDistMatrices, num)
#     all_spiketrain = np.concatenate((all_spiketrain, sampled_spiketrain[:num, :]), axis=0)
# ################ raster plot
# #%%
# text_pos = np.arange(8, 68, 10)
# fig = plt.figure(figsize=(10, 7))
# # plt.eventplot(spikeTrain, colors='b', lineoffsets=1, linewidths=1, linelengths=1)
# plt.eventplot(all_spiketrain, colors=colors, lineoffsets=1, linewidths=0.5, linelengths=0.4)
# for ind, t_pos in enumerate(text_pos):
#   plt.text(-700, t_pos, 'R={}'.format(R_list[ind]), size=10, color=Palette[ind], weight='bold')
# plt.axis('off')
# plt.gca().invert_yaxis()
# Gamma = getGamma(L, R, T, spikeTrain)
# # plt.vlines(np.concatenate((np.min(Gamma, axis=1), np.max(Gamma, axis=1))), ymin=0, ymax=num+1, colors='k', linewidth=0.2, linestyles='dashed')
# plt.tight_layout()
# plt.show()
# plt.savefig('../plots/sampled_spiketrain_L{}.jpg'.format(L))

#%%
# %%
########## whether pattern jitter changes number of spikes
# L = 5
# R = 300
# spikeTrain = getSpikeTrain(sequences[active_inds[0], :])
# N = len(spikeTrain)
# initDist = getInitDist(L)
# tDistMatrices = getTransitionMatrices(L, N)
# sampled_spiketrain1 = sample_spiketrain(L, R, T, spikeTrain, initDist, tDistMatrices, num_sample)
# spikeTrain = getSpikeTrain(sequences[active_inds[1], :])
# N = len(spikeTrain)
# initDist = getInitDist(L)
# tDistMatrices = getTransitionMatrices(L, N)
# sampled_spiketrain2 = sample_spiketrain(L, R, T, spikeTrain, initDist, tDistMatrices, num_sample)

# #%%
# st1 = spike_timing2train(min_len, sampled_spiketrain1)
# st2 = spike_timing2train(min_len, sampled_spiketrain2)
# print(np.count_nonzero(sequences[active_inds], axis=1))
# print(np.unique(np.count_nonzero(st1, axis=1)))
# print(np.unique(np.count_nonzero(st2, axis=1)))
# # %%
# adj = n_cross_correlation_2mat(spike_timing2train(min_len, sampled_spiketrain1), spike_timing2train(min_len, sampled_spiketrain2), maxlag=12, disable=True)
# adj.mean(-1)
# %%
################# normal test for xcorrs with adjacent Ls
# np.seterr(divide='ignore', invalid='ignore')
# min_len, min_num = (10000, 29)
# min_spikes = min_len * 0.002 # 2 Hz
# measure = 'xcorr'
# directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
# files = os.listdir(directory)
# files = [f for f in files if f.endswith('.npz')]
# files.sort(key=lambda x:int(x[:9]))
# path = os.path.join(directory.replace('spiking_sequence', 'adj_mat_{}'.format(measure)))
# if not os.path.exists(path):
#   os.makedirs(path)
# num_sample = 1000
# file = files[0] # 0, 2, 7
# print(file)
# sequences = load_npz(os.path.join(directory, file))
# sequences = sequences[np.count_nonzero(sequences[:, :min_len], axis=1) > min_spikes, :min_len]
# # sequences = sequences[np.count_nonzero(sequences[:, :min_len], axis=1) > 80, :min_len]
# # active_inds = np.argpartition(np.count_nonzero(sequences, axis=1), -2)[-2:] # top 2 most active neurons
# active_inds = np.argpartition(np.count_nonzero(sequences, axis=1), 2)[:2] # top 2 most inactive neurons
# print('Sampling neurons A and B...')
# start_time = time.time()
# all_xcorr = np.zeros((2, len(Ls), num_sample))
# R = 200
# for L_ind, L in enumerate(Ls):
#   spikeTrain = getSpikeTrain(sequences[active_inds[0], :])
#   N = len(spikeTrain)
#   initDist = getInitDist(L)
#   tDistMatrices = getTransitionMatrices(L, N)
#   sampled_spiketrain1 = sample_spiketrain(L, R, T, spikeTrain, initDist, tDistMatrices, num_sample)
#   spikeTrain = getSpikeTrain(sequences[active_inds[1], :])
#   N = len(spikeTrain)
#   initDist = getInitDist(L)
#   tDistMatrices = getTransitionMatrices(L, N)
#   sampled_spiketrain2 = sample_spiketrain(L, R, T, spikeTrain, initDist, tDistMatrices, num_sample)
#   adj = n_cross_correlation_2mat(spike_timing2train(min_len, sampled_spiketrain1), spike_timing2train(min_len, sampled_spiketrain2), maxlag=12, disable=True)
#   all_xcorr[0, L_ind, :] = adj[0, 1, :]
#   all_xcorr[1, L_ind, :] = adj[1, 0, :]
# print("--- %s minutes in total" % ((time.time() - start_time)/60))
# alpha = 0.05
# SW_p_A = []
# DA_p_A = []
# SW_p_B = []
# DA_p_B = []
# print('Shapiro-Wilk Test and D’Agostino’s K^2 Test for xcorr A->B...')
# for L_ind in range(len(Ls)):
#     _, p = shapiro(all_xcorr[0, L_ind, :])
#     SW_p_A.append(p)
#     _, p = normaltest(all_xcorr[0, L_ind, :])
#     DA_p_A.append(p)
# print('Shapiro-Wilk Test and D’Agostino’s K^2 Test for xcorr B->A...')
# for L_ind in range(len(Ls)):
#     _, p = shapiro(all_xcorr[1, L_ind, :])
#     SW_p_B.append(p)
#     _, p = normaltest(all_xcorr[1, L_ind, :])
#     DA_p_B.append(p)

# # %%
# ##################### plot percentage of links that follow normal distribution
# # %%
# # alpha = 0.05
# # SW = np.zeros(len(Ls))
# # DA = np.zeros(len(Ls))
# # for L_ind in range(len(Ls)):
# #   SW.loc[int(mouseID)][stimulus] = (np.array(SW_p_A) > alpha).sum() / len(SW_p)
# #   SW_bl.loc[int(mouseID)][stimulus] = (np.array(SW_p_bl) > alpha).sum() / len(SW_p_bl)
# #   DA.loc[int(mouseID)][stimulus] = (np.array(DA_p) > alpha).sum() / len(DA_p)
# #   DA_bl.loc[int(mouseID)][stimulus] = (np.array(DA_p_bl) > alpha).sum() / len(DA_p_bl)
# # %%
# plt.figure(figsize=(7, 6))
# plt.plot(Ls, SW_p_A, label='A->B', alpha=0.5)
# plt.plot(Ls, SW_p_B, label='B->A', alpha=0.5)
# plt.gca().set_title('p value of Shapiro-Wilk Test', fontsize=20, rotation=0)
# plt.xticks(rotation=90)
# plt.legend()
# plt.tight_layout()
# plt.show()
# # plt.savefig('./plots/SW_p.jpg')
# plt.figure(figsize=(7, 6))
# plt.plot(Ls, DA_p_A, label='A->B', alpha=0.5)
# plt.plot(Ls, DA_p_B, label='B->A', alpha=0.5)
# plt.gca().set_title('p value of D’Agostino’s K^2 Test', fontsize=20, rotation=0)
# plt.xticks(rotation=90)
# plt.legend()
# plt.tight_layout()
# plt.show()
# # plt.savefig('./plots/DA_p.jpg')

#%%
# #################### z test between adjacent xcorr
# pvals_A = []
# pvals_B = []
# print('Z test...')
# for L_ind in range(len(Ls) - 1):
#   xcorr_0 = ws.DescrStatsW(all_xcorr[0, L_ind, :])
#   xcorr_1 = ws.DescrStatsW(all_xcorr[0, L_ind + 1, :])
#   cm_obj = ws.CompareMeans(xcorr_0, xcorr_1)
#   zstat, z_pval = cm_obj.ztest_ind(alternative='two-sided', usevar='unequal', value=0)
#   pvals_A.append(z_pval)
#   xcorr_0 = ws.DescrStatsW(all_xcorr[1, L_ind, :])
#   xcorr_1 = ws.DescrStatsW(all_xcorr[1, L_ind + 1, :])
#   cm_obj = ws.CompareMeans(xcorr_0, xcorr_1)
#   zstat, z_pval = cm_obj.ztest_ind(alternative='two-sided', usevar='unequal', value=0)
#   pvals_B.append(z_pval)
# # %%
# alpha = 0.05
# plt.figure(figsize=(7, 6))
# plt.plot((np.array(Ls[1:]) + np.array(Ls[:-1])) / 2, pvals_A, label='A->B', alpha=0.5)
# plt.plot((np.array(Ls[1:]) + np.array(Ls[:-1])) / 2, pvals_B, label='B->A', alpha=0.5)
# plt.plot((np.array(Ls[1:]) + np.array(Ls[:-1])) / 2, [alpha]*(len(Ls)-1), 'k--', label='95%confidence level', alpha=0.5)
# plt.gca().set_title('Z test of adjacent cross correlations', fontsize=20, rotation=0)
# plt.xticks(rotation=90)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Bin size L', size=15)
# plt.ylabel('p value', size=15)
# plt.legend()
# plt.tight_layout()
# # plt.show()
# plt.savefig('./plots/z_test_adjacent_xcorr_L.jpg')
# # %%

# plt.figure(figsize=(7, 6))
# plt.plot((np.array(Ls[1:]) + np.array(Ls[:-1])) / 2, np.abs(all_xcorr[0, 1:, :].mean(-1)-all_xcorr[0, :-1, :].mean(-1)), label='A->B', alpha=0.5)
# plt.plot((np.array(Ls[1:]) + np.array(Ls[:-1])) / 2, np.abs(all_xcorr[1, 1:, :].mean(-1)-all_xcorr[1, :-1, :].mean(-1)), label='B->A', alpha=0.5)
# plt.gca().set_title('difference between adjacent cross correlations', fontsize=20, rotation=0)
# plt.xticks(rotation=90)
# plt.xscale('log')
# # plt.yscale('log')
# plt.xlabel('Bin size L', size=15)
# plt.ylabel('absolute difference', size=15)
# plt.legend()
# plt.tight_layout()
# # plt.show()
# plt.savefig('./plots/difference_adjacent_xcorr_L.jpg')
# %%
################ is cross correlation or correlation affected by firing rate?
# adj_mat = corr_mat(sequences, measure, maxlag=12, noprogressbar=False)
# np.fill_diagonal(adj_mat, np.nan)
# # adj_mat_flat = adj_mat[~np.eye(adj_mat.shape[0],dtype=bool)]
# firing_rates = np.count_nonzero(sequences, axis=1) / sequences.shape[1]
# bins=np.logspace(start=np.log10(firing_rates.min()), stop=np.log10(firing_rates.max()+0.0001), num=50)
# bin_num = np.digitize(firing_rates, bins)
# # %%
# corr = np.zeros((len(bins), len(bins)))
# for i in range(1, len(bins)):
#   for j in range(1, len(bins)):
#     corr[i, j] = np.nanmean(adj_mat[np.where(bin_num==i)[0][:, None], np.where(bin_num==j)[0][None, :]])
# plt.xscale('log')
#%%
################## heatmap of xcorr and pcrorr vs FR
start_time = time.time()
min_len, min_num = (10000, 29)
min_spikes = min_len * 0.002 # 2 Hz
# measure = 'pearson'
# measure = 'cosine'
# measure = 'correlation'
# measure = 'MI'
measure = 'xcorr'
# measure = 'causality'
directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
stimulus_names = ['spontaneous', 'flashes', 'gabors',
        'drifting_gratings', 'static_gratings',
          'natural_scenes', 'natural_movie_one', 'natural_movie_three']
session_ids = [719161530, 750749662, 755434585, 756029989, 791319847]
xcorr_dict, pcorr_dict, peak_dict, bin_dict = {}, {}, {}, {}
for session_id in session_ids:
  print(session_id)
  xcorr_dict[session_id], pcorr_dict[session_id], peak_dict[session_id], bin_dict[session_id] = {}, {}, {}, {}
  for stimulus_name in stimulus_names:
    print(stimulus_name)
    sequences = load_npz(os.path.join(directory, str(session_id) + '_' + stimulus_name + '.npz'))
    sequences = sequences[np.count_nonzero(sequences[:, :min_len], axis=1) > min_spikes, :min_len]
    adj_mat, peak_off = n_cross_correlation8(sequences, disable=False)
    np.fill_diagonal(adj_mat, np.nan)
    np.fill_diagonal(peak_off, np.nan)
    # p_adj_mat = corr_mat(sequences, measure='pearson', maxlag=12, noprogressbar=False)
    # np.fill_diagonal(p_adj_mat, np.nan)
    # adj_mat_flat = adj_mat[~np.eye(adj_mat.shape[0],dtype=bool)]
    firing_rates = np.count_nonzero(sequences, axis=1) / sequences.shape[1]
    bins=np.logspace(start=np.log10(firing_rates.min()), stop=np.log10(firing_rates.max()+0.0001), num=20)
    bin_num = np.digitize(firing_rates, bins)
    xcorr = np.zeros((len(bins), len(bins)))
    corr = np.zeros((len(bins), len(bins)))
    peaks = np.zeros((len(bins), len(bins)))
    for i in range(len(bins)):
      for j in range(len(bins)):
        xcorr[i, j] = np.nanmean(adj_mat[np.where(bin_num==i)[0][:, None], np.where(bin_num==j)[0][None, :]])
        # corr[i, j] = np.nanmean(p_adj_mat[np.where(bin_num==i)[0][:, None], np.where(bin_num==j)[0][None, :]])
        peaks[i, j] = np.nanmean(peak_off[np.where(bin_num==i)[0][:, None], np.where(bin_num==j)[0][None, :]])
    xcorr_dict[session_id][stimulus_name] = xcorr
    # pcorr_dict[session_id][stimulus_name] = corr
    peak_dict[session_id][stimulus_name] = peaks
    bin_dict[session_id][stimulus_name] = bins
print("--- %s minutes in total" % ((time.time() - start_time)/60))
#%%
def plot_multi_heatmap_corr_FR(session_ids, stimulus_names, corr_dict, bin_dict, name):
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
      corr, bins = corr_dict[row][col], bin_dict[row][col]
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
  plt.suptitle('{} correlation VS firing rate'.format(name), size=40)
  plt.tight_layout()
  # plt.show()
  plt.savefig('./plots/{}_FR_multi_heatmap.jpg'.format(name))
# plot_multi_heatmap_corr_FR(session_ids, stimulus_names, pcorr_dict, bin_dict, 'pearson')
plot_multi_heatmap_corr_FR(session_ids, stimulus_names, xcorr_dict, bin_dict, 'cross')
plot_multi_heatmap_corr_FR(session_ids, stimulus_names, peak_dict, bin_dict, 'peakoffset')
# %%
def plot_multi_corr_FR(session_ids, stimulus_names, corr_dict, bin_dict, name):
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
      corr, bins = corr_dict[row][col], bin_dict[row][col]
      gmean_FR = np.zeros(int(len(bins)/2))
      close_corr = np.zeros(int(len(bins)/2))
      for i in range(0, len(bins), 2):
        close_corr[int(i/2)] = corr[i, i+1]
        gmean_FR[int(i/2)] = np.sqrt(bins[i] * bins[i+1])
      ax.plot(gmean_FR, close_corr, 'o-')
      r = ma.corrcoef(ma.masked_invalid(gmean_FR), ma.masked_invalid(close_corr))
      ax.text(0.1, 0.9, 'r={:.2f}'.format(r[0, 1]), fontsize=15, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
      plt.xscale('log')
  plt.suptitle('{} correlation VS firing rate'.format(name), size=40)
  plt.tight_layout()
  # plt.show()
  plt.savefig('./plots/{}_FR_multi.jpg'.format(name))
# plot_multi_corr_FR(session_ids, stimulus_names, pcorr_dict, bin_dict, 'pearson')
plot_multi_corr_FR(session_ids, stimulus_names, xcorr_dict, bin_dict, 'cross')
plot_multi_corr_FR(session_ids, stimulus_names, peak_dict, bin_dict, 'peakoffset')
# %%
def plot_multi_peak_dist(peak_dict, measure):
  ind = 1
  rows, cols = get_rowcol(peak_dict)
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
      peaks = peak_dict[row][col]
      plt.hist(peaks.flatten(), bins=12, density=True)
      plt.axvline(x=peaks.mean(), color='r', linestyle='--')
      # plt.text(peaks.mean(), peaks.max()/2, "mean={}".format(peaks.mean()), rotation=0, verticalalignment='center')
      plt.plot()
      plt.xlabel('peak correlation offset (ms)')
      plt.ylabel('Probabilit')
      
  plt.tight_layout()
  image_name = './plots/peak_distribution.jpg'
  # plt.show()
  plt.savefig(image_name)
# plot_multi_corr_FR(session_ids, stimulus_names, pcorr_dict, bin_dict, 'pearson')
plot_multi_peak_dist(peak_dict, measure)
#%%
directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
stimulus_names = ['spontaneous', 'flashes', 'gabors',
        'drifting_gratings', 'static_gratings',
          'natural_scenes', 'natural_movie_one', 'natural_movie_three']
session_ids = [719161530, 750749662, 755434585, 756029989, 791319847]
peak_dict = {}
for session_id in session_ids:
  print(session_id)
  peak_dict[session_id] = {}
  for stimulus_name in stimulus_names:
    peak_dict[session_id][stimulus_name] = np.random.randint(low=0, high=12,size=1000)
# %%
####### if cross correlation at 0 time lag == pearson correlation
# a = np.random.random((5, 10))
# pcorr = np.corrcoef(a)
# dxcorr = n_cross_correlation6(a, 2, disable=True)
# xcorr = n_cross_correlation7(a, 0, disable=True)
# print(pcorr)
# print(xcorr)
# print(dxcorr)
# %%
################### heatmap of shuffled xcrorr vs FR
# start_time = time.time()
# min_len, min_num = (10000, 29)
# min_spikes = min_len * 0.002 # 2 Hz
# # measure = 'pearson'
# # measure = 'cosine'
# # measure = 'correlation'
# # measure = 'MI'
# measure = 'xcorr'
# # measure = 'causality'
# directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
# stimulus_names = ['spontaneous', 'flashes', 'gabors',
#         'drifting_gratings', 'static_gratings',
#           'natural_scenes', 'natural_movie_one', 'natural_movie_three']
# session_ids = [719161530, 750749662, 755434585, 756029989, 791319847]
# xcorr_dict, bin_dict = {}, {}
# for session_id in session_ids:
#   print(session_id)
#   xcorr_dict[session_id], bin_dict[session_id] = {}, {}
#   for stimulus_name in stimulus_names:
#     print(stimulus_name)
#     sequences = load_npz(os.path.join(directory, str(session_id) + '_' + stimulus_name + '.npz'))
#     sequences = sequences[np.count_nonzero(sequences[:, :min_len], axis=1) > min_spikes, :min_len]
#     for n in range(sequences.shape[0]):
#       np.random.shuffle(sequences[n,:])
#     adj_mat = corr_mat(sequences, measure, maxlag=12, noprogressbar=False)
#     np.fill_diagonal(adj_mat, np.nan)
#     # adj_mat_flat = adj_mat[~np.eye(adj_mat.shape[0],dtype=bool)]
#     firing_rates = np.count_nonzero(sequences, axis=1) / sequences.shape[1]
#     bins=np.logspace(start=np.log10(firing_rates.min()), stop=np.log10(firing_rates.max()+0.0001), num=50)
#     bin_num = np.digitize(firing_rates, bins)
#     xcorr = np.zeros((len(bins), len(bins)))
#     for i in range(1, len(bins)):
#       for j in range(1, len(bins)):
#         xcorr[i, j] = np.nanmean(adj_mat[np.where(bin_num==i)[0][:, None], np.where(bin_num==j)[0][None, :]])
#     xcorr_dict[session_id][stimulus_name] = xcorr
#     bin_dict[session_id][stimulus_name] = bins
# print("--- %s minutes in total" % ((time.time() - start_time)/60))
# #%%
# def plot_multi_heatmap_sxcorr_FR(session_ids, stimulus_names, xcorr_dict, bin_dict):
#   ind = 1
#   rows, cols = session_ids, stimulus_names
#   divnorm=colors.TwoSlopeNorm(vcenter=0.)
#   fig = plt.figure(figsize=(5*len(cols), 5*len(rows)))
#   left, width = .25, .5
#   bottom, height = .25, .5
#   right = left + width
#   top = bottom + height
#   for row_ind, row in enumerate(rows):
#     print(row)
#     for col_ind, col in enumerate(cols):
#       ax = plt.subplot(len(rows), len(cols), ind)
#       if row_ind == 0:
#         plt.gca().set_title(cols[col_ind], fontsize=30, rotation=0)
#       if col_ind == 0:
#         plt.gca().text(0, 0.5 * (bottom + top), rows[row_ind],
#         horizontalalignment='right',
#         verticalalignment='center',
#         # rotation='vertical',
#         transform=plt.gca().transAxes, fontsize=30, rotation=90)
#       plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
#       ind += 1
#       corr, bins = xcorr_dict[row][col], bin_dict[row][col]
#       # pcolormesh(your_data, cmap="coolwarm", norm=divnorm)
#       im = ax.imshow(corr, norm=divnorm, cmap="RdBu_r")
#       ax.set_xticks(ticks=np.arange(len(bins)))
#       ax.set_xticklabels(bins)
#       ax.set_yticks(ticks=np.arange(len(bins)))
#       ax.set_yticklabels(bins)
#       fmt = lambda x, position: '{:.1f}e-3'.format(bins[x]*1e3)
#       ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
#       ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
#       for index, label in enumerate(ax.get_xticklabels()):
#         if index % 15 == 0:
#             label.set_visible(True)
#         else:
#             label.set_visible(False)
#       for index, label in enumerate(ax.get_yticklabels()):
#         if index % 15 == 0:
#             label.set_visible(True)
#         else:
#             label.set_visible(False)
#       if col_ind == 7:
#         fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#       # plt.xlabel('firing rate of source neuron', size=15)
#       # plt.ylabel('firing rate of target neuron', size=15)
#       # plt.title('cross correlation VS firing rate', size=15)
#   plt.suptitle('shuffled cross correlation VS firing rate', size=40)
#   plt.tight_layout()
#   plt.savefig('./plots/xcorr_FR_shuffled_multi_heatmap.jpg')

# plot_multi_heatmap_sxcorr_FR(session_ids, stimulus_names, xcorr_dict, bin_dict)
# #%%
# def plot_multi_scorr_FR(session_ids, stimulus_names, corr_dict, bin_dict, name):
#   ind = 1
#   rows, cols = session_ids, stimulus_names
#   divnorm=colors.TwoSlopeNorm(vcenter=0.)
#   fig = plt.figure(figsize=(5*len(cols), 5*len(rows)))
#   left, width = .25, .5
#   bottom, height = .25, .5
#   right = left + width
#   top = bottom + height
#   for row_ind, row in enumerate(rows):
#     print(row)
#     for col_ind, col in enumerate(cols):
#       ax = plt.subplot(len(rows), len(cols), ind)
#       if row_ind == 0:
#         plt.gca().set_title(cols[col_ind], fontsize=30, rotation=0)
#       if col_ind == 0:
#         plt.gca().text(0, 0.5 * (bottom + top), rows[row_ind],
#         horizontalalignment='right',
#         verticalalignment='center',
#         # rotation='vertical',
#         transform=plt.gca().transAxes, fontsize=30, rotation=90)
#       plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
#       ind += 1
#       corr, bins = corr_dict[row][col], bin_dict[row][col]
#       gmean_FR = np.zeros(int(len(bins)/2))
#       close_corr = np.zeros(int(len(bins)/2))
#       for i in range(0, len(bins), 2):
#         close_corr[int(i/2)] = corr[i, i+1]
#         gmean_FR[int(i/2)] = np.sqrt(bins[i] * bins[i+1])
#       ax.plot(gmean_FR, close_corr, 'o-')
#       r = ma.corrcoef(ma.masked_invalid(gmean_FR), ma.masked_invalid(close_corr))
#       ax.text(0.1, 0.9, 'r={:.2f}'.format(r[0, 1]), fontsize=15, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
#       plt.xscale('log')
#   plt.suptitle('shuffled {} correlation VS firing rate'.format(name), size=40)
#   plt.tight_layout()
#   # plt.show()
#   plt.savefig('./plots/{}_corr_FR_shuffled_multi.jpg'.format(name))
# # plot_multi_corr_FR(session_ids, stimulus_names, pcorr_dict, bin_dict, 'pearson')
# plot_multi_scorr_FR(session_ids, stimulus_names, xcorr_dict, bin_dict, 'cross')
#%%
start_time = time.time()
start_time_A = start_time
for L_ind, L in enumerate(Ls):
  for R_ind, R in enumerate(Rs):
    print(L, R)
    for b in range(num_baseline):
      sample_seq = sequences[active_inds, :]
      # print('Baseline {} out of {}'.format(b+1, num_baseline))
      for n in range(num_nodes):
        np.random.shuffle(sample_seq[n,:])
      adj_mat = corr_mat(sample_seq, measure, maxlag=Ls[-1])
      adj_mat_bl_A[:, :, b, L_ind, R_ind] = adj_mat
print("--- %s minutes" % ((time.time() - start_time_A)/60))
# %%
############## one pair of neurons, significant xcorr vs L and R
def plot_sxcorr_LR(origin_adj_mat, adj_mat_bl_A, Ls, R, Rs, edge_type='active'):
  plt.figure(figsize=(6, 6))
  mean = np.nanmean(adj_mat_bl_A[0, 1, :, :, Rs.index(R)], axis=0)
  std = np.nanstd(adj_mat_bl_A[0, 1, :, :, Rs.index(R)], axis=0)
  plt.plot(Ls, mean, alpha=0.6, label='A->B')
  plt.fill_between(Ls, (mean - std), (mean + std), alpha=0.2)
  mean = np.nanmean(adj_mat_bl_A[1, 0, :, :, Rs.index(R)], axis=0)
  std = np.nanstd(adj_mat_bl_A[1, 0, :, :, Rs.index(R)], axis=0)
  plt.plot(Ls, mean, alpha=0.6, label='B->A')
  plt.fill_between(Ls, (mean - std), (mean + std), alpha=0.2)
  plt.plot(Ls, len(Ls) * [origin_adj_mat[0, 1]], 'b--', alpha=0.6, label='original A->B')
  plt.plot(Ls, len(Ls) * [origin_adj_mat[1, 0]], 'r--', alpha=0.6, label='original B->A')
  plt.gca().set_title('shuffled correlation, R={}'.format(R), fontsize=20, rotation=0)
  plt.xscale('log')
  plt.xticks(rotation=90)
  plt.xlabel('Bin size L', fontsize=15)
  plt.ylabel('cross correlation', fontsize=15)
  plt.legend()
  plt.tight_layout()
  # plt.show()
  figname = './plots/xcorr_shuffled_vs_L_R_{}_{}_{}.jpg'.format(R, edge_type, measure)
  plt.savefig(figname)

for R in Rs:
  plot_sxcorr_LR(origin_adj_mat, adj_mat_bl_A, Ls, R, Rs, 'inactive')
# %%
################### effect of maxlag on cross correlation
####### turn off warnings
np.seterr(divide='ignore', invalid='ignore')
############# save correlation matrices #################
# min_len, min_num = (260000, 739)
min_len, min_num = (10000, 29)
min_spikes = min_len * 0.002 # 2 Hz
# measure = 'pearson'
# measure = 'cosine'
# measure = 'correlation'
# measure = 'MI'
measure = 'xcorr'
# measure = 'causality'
directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
files = os.listdir(directory)
files = [f for f in files if f.endswith('.npz')]
files.sort(key=lambda x:int(x[:9]))
path = os.path.join(directory.replace('spiking_sequence', 'adj_mat_{}'.format(measure)))
if not os.path.exists(path):
  os.makedirs(path)
num_sample = 1
num_baseline = 500
lags = [1, 10, 100, 500, 1000, 5000, 7000, 9999]
file = files[0] # 0, 2, 7
start_time_mouse = time.time()
print(file)
mouseID = file.replace('.npz', '').split('_')[0]
stimulus = file.replace('.npz', '').replace(mouseID + '_', '')
sequences = load_npz(os.path.join(directory, file))
sequences = sequences[np.count_nonzero(sequences[:, :min_len], axis=1) > min_spikes, :min_len]
# sequences = sequences[np.count_nonzero(sequences[:, :min_len], axis=1) > 80, :min_len]
active_inds = np.argpartition(np.count_nonzero(sequences, axis=1), -2)[-2:] # top 2 most active neurons
inactive_inds = np.argpartition(np.count_nonzero(sequences, axis=1), 2)[:2] # top 2 most inactive neurons
num_nodes = 2
active_adj_mat_bl = np.zeros((2, 2, num_baseline, len(lags)))
inactive_adj_mat_bl = np.zeros((2, 2, num_baseline, len(lags)))
#%%
start_time = time.time()
for b in range(num_baseline):
  print(b)
  active_sample_seq = sequences[active_inds, :]
  for n in range(num_nodes):
    np.random.shuffle(active_sample_seq[n,:])
  inactive_sample_seq = sequences[inactive_inds, :]
  for n in range(num_nodes):
    np.random.shuffle(inactive_sample_seq[n,:])
  for ind, lag in enumerate(lags):
    adj_mat = cross_correlation_delta(active_sample_seq, maxlag=lag, disable=True)
    active_adj_mat_bl[:, :, b, ind] = adj_mat
    adj_mat = cross_correlation_delta(inactive_sample_seq, maxlag=lag, disable=True)
    inactive_adj_mat_bl[:, :, b, ind] = adj_mat
print("--- %s minutes" % ((time.time() - start_time)/60))
# %%
def plot_sxcorr_lag(adj_mat_bl, lags, edge_type='active'):
  plt.figure(figsize=(6, 6))
  mean = np.nanmean(adj_mat_bl[0, 1, :, :], axis=0)
  std = np.nanstd(adj_mat_bl[0, 1, :, :], axis=0)
  plt.plot(lags, mean, alpha=0.6, label='A->B')
  plt.fill_between(lags, (mean - std), (mean + std), alpha=0.2)
  mean = np.nanmean(adj_mat_bl[1, 0, :, :], axis=0)
  std = np.nanstd(adj_mat_bl[1, 0, :, :], axis=0)
  plt.plot(lags, mean, alpha=0.6, label='B->A')
  plt.fill_between(lags, (mean - std), (mean + std), alpha=0.2)
  # plt.gca().set_title('shuffled correlation', fontsize=20, rotation=0)
  plt.gca().set_title(r'$\Delta$ shuffled correlation', fontsize=20, rotation=0)
  plt.xscale('log')
  plt.xticks(rotation=90)
  plt.xlabel('maximal lag', fontsize=15)
  plt.ylabel('cross correlation', fontsize=15)
  plt.legend()
  plt.tight_layout()
  # plt.show()
  # figname = './plots/xcorr_shuffled_vs_lag_{}.jpg'.format(edge_type)
  figname = './plots/delta_xcorr_shuffled_vs_lag_{}.jpg'.format(edge_type)
  plt.savefig(figname)

plot_sxcorr_lag(active_adj_mat_bl, lags, 'active')
plot_sxcorr_lag(inactive_adj_mat_bl, lags, 'inactive')
# %%
################ CCG-mean of flank of 100ms
adj_mat, origin_peak_off = n_cross_correlation8(sequences, disable=False)
# adj_mat = np.nan_to_num(adj_mat)
# np.fill_diagonal(adj_mat, 0)
# np.fill_diagonal(origin_peak_off, np.nan)
firing_rates = np.count_nonzero(sequences, axis=1) / sequences.shape[1]
#%%
############## peak with binning
firing_rates = np.count_nonzero(sequences, axis=1) / sequences.shape[1]
bins=np.logspace(start=np.log10(firing_rates.min()), stop=np.log10(firing_rates.max()+0.0001), num=50)
bin_num = np.digitize(firing_rates, bins)
peaks = np.zeros((len(bins), len(bins)))
for i in range(1, len(bins)):
  for j in range(1, len(bins)):
    peaks[i, j] = np.nanmean(origin_peak_off[np.where(bin_num==i)[0][:, None], np.where(bin_num==j)[0][None, :]])
# %%
divnorm=colors.TwoSlopeNorm(vcenter=0.)
fig, ax = plt.subplots()
im = ax.imshow(peaks, norm=divnorm, cmap="RdBu_r")
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
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.title('Peak correlation offset (ms)')
plt.xlabel('firing rate of source neuron')
plt.ylabel('firing rate of target neuron')
# plt.show()
plt.savefig('./plots/peakoffset_FR.jpg')
# %%
fig = plt.figure()
plt.hist(origin_peak_off.flatten(), bins=12, density=True)
plt.xlabel('peak correlation offset (ms)')
plt.ylabel('probability')
# plt.show()
plt.savefig('./plots/peakoffset_dist.jpg')
# %%
#################### save correlation matrices
def save_ccg_corrected(sequences, fname, num_jitter=10, L=25, window=100, disable=True): ### fastest, only causal correlation (A>B, only positive time lag on B), largest deviation from flank
  xcorr = all_xcorr_ccg(sequences, window, disable=disable) # N x N x window
  save_npz(xcorr, fname)
  N = sequences.shape[0]
  # jitter
  xcorr_jittered = np.zeros((N, N, window+1, num_jitter))
  pj = pattern_jitter(num_sample=num_jitter, sequences=sequences, L=L, memory=False)
  sampled_matrix = pj.jitter() # num_sample x N x T
  for i in range(num_jitter):
    print(i)
    xcorr_jittered[:, :, :, i] = all_xcorr_ccg(sampled_matrix[i, :, :], window, disable=disable)
  save_npz(xcorr_jittered, fname.replace('.npz', '_bl.npz'))

def save_xcorr_shuffled(sequences, fname, num_baseline=10, disable=True):
  N = sequences.shape[0]
  xcorr = np.zeros((N, N))
  xcorr_bl = np.zeros((N, N, num_baseline))
  adj_mat, peaks = n_cross_correlation8(sequences, disable=disable)
  xcorr = adj_mat
  save_npz(xcorr, fname)
  save_npz(peaks, fname.replace('.npz', '_peak.npz'))
  for b in range(num_baseline):
    print(b)
    sample_seq = sequences.copy()
    np.random.shuffle(sample_seq) # rowwise for 2d array
    adj_mat_bl, peaks_bl = n_cross_correlation8(sample_seq, disable=disable)
    xcorr_bl[:, :, b] = adj_mat_bl
  save_npz(xcorr_bl, fname.replace('.npz', '_bl.npz'))
#%%
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
    # for n in range(sample_seq.shape[0]):
    #   np.random.shuffle(sample_seq[n,:])
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

def all_xcorr_ccg(matrix, window=100, disable=True): ### fastest, only causal correlation (A>B, only positive time lag on B), largest deviation from flank
  N, M =matrix.shape
  # xcorr, peak_offset=np.zeros((N,N)), np.zeros((N,N))
  xcorr=np.empty((N,N,window+1))
  xcorr[:] = np.nan
  firing_rates = np.count_nonzero(matrix, axis=1) / matrix.shape[1]
  #### padding
  norm_mata = np.concatenate((matrix.conj(), np.zeros((N, window))), axis=1)
  norm_matb = np.concatenate((np.zeros((N, window)), matrix.conj(), np.zeros((N, window))), axis=1) # must concat zeros to the left, why???????????
  total_len = len(list(itertools.permutations(range(N), 2)))
  for row_a, row_b in tqdm(itertools.permutations(range(N), 2), total=total_len , miniters=int(total_len/100), disable=disable): # , miniters=int(total_len/100)
    px, py = norm_mata[row_a, :], norm_matb[row_b, :]
    T = as_strided(py[window:], shape=(window+1, M + window),
                    strides=(-py.strides[0], py.strides[0])) # must be py[window:], why???????????
    xcorr[row_a, row_b, :] = (T @ px) / ((M-np.arange(window+1))/1000 * np.sqrt(firing_rates[row_a] * firing_rates[row_b]))
  return xcorr

def all_xcorr_xcorr(matrix, window=100, disable=True): ### fastest, only causal correlation (A>B, only positive time lag on B), largest deviation from flank
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
    # corr = np.dot(T, px)
    corr = T @ px
    xcorr[row_a, row_b, :] = corr = corr - corr.mean()
  return xcorr

def pattern_jitter(sequences, L, R, num_sample):
  if len(sequences.shape) > 1:
    N, T = sequences.shape
    jittered_seq = np.zeros((N, T, num_sample))
    for n in range(N):
      spikeTrain = getSpikeTrain(sequences[n, :])
      ns = spikeTrain.size
      if ns:
        initDist = getInitDist(L)
        tDistMatrices = getTransitionMatrices(L, ns)
        sampled_spiketrain = sample_spiketrain(L, R, T, spikeTrain, initDist, tDistMatrices, num_sample)
        jittered_seq[n, :, :] = spike_timing2train(T, sampled_spiketrain).T
      else:
        jittered_seq[n, :, :] = np.zeros((T, num_sample))
  else:
    T = len(sequences)
    spikeTrain = getSpikeTrain(sequences)
    ns = spikeTrain.size
    initDist = getInitDist(L)
    tDistMatrices = getTransitionMatrices(L, ns)
    sampled_spiketrain = sample_spiketrain(L, R, T, spikeTrain, initDist, tDistMatrices, num_sample)
    jittered_seq = spike_timing2train(T, sampled_spiketrain).T
  return jittered_seq

def xcorr_n_fold(matrix, n=7, num_jitter=10, L=25, R=1, maxlag=12, window=100, disable=True): ### fastest, only causal correlation (A>B, only positive time lag on B), largest deviation from flank
  xcorr = all_xcorr_ccg(matrix, window, disable=disable) # N x N x window
  N = matrix.shape[0]
  significant_ccg, peak_offset=np.empty((N,N)), np.empty((N,N))
  significant_ccg[:] = np.nan
  peak_offset[:] = np.nan
  # jitter
  xcorr_jittered = np.zeros((N, N, window+1, num_jitter))
  sampled_matrix = pattern_jitter(matrix, L, R, num_jitter) # N, T, num_jitter
  for i in range(num_jitter):
    print(i)
    xcorr_jittered[:, :, :, i] = all_xcorr_ccg(sampled_matrix[:, :, i], window, disable=disable)
  total_len = len(list(itertools.permutations(range(N), 2)))
  for row_a, row_b in tqdm(itertools.permutations(range(N), 2), total=total_len , miniters=int(total_len/100), disable=disable): # , miniters=int(total_len/100)
    ccg_corrected = (xcorr[row_a, row_b, :, None] - xcorr_jittered[row_a, row_b, :, :]).mean(-1)
    if ccg_corrected[:maxlag].max() > ccg_corrected.mean() + n * ccg_corrected.std():
    # if np.max(np.abs(corr))
      max_offset = np.argmax(ccg_corrected[:maxlag])
      significant_ccg[row_a, row_b] = ccg_corrected[:maxlag][max_offset]
      peak_offset[row_a, row_b] = max_offset
  return significant_ccg, peak_offset

np.seterr(divide='ignore', invalid='ignore')
############# save correlation matrices #################
# min_len, min_num = (260000, 739)
min_len, min_num = (10000, 29)
min_spikes = min_len * 0.002 # 2 Hz
# measure = 'pearson'
# measure = 'cosine'
# measure = 'correlation'
# measure = 'MI'
measure = 'xcorr'
# measure = 'causality'
directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
files = os.listdir(directory)
files = [f for f in files if f.endswith('.npz')]
files.sort(key=lambda x:int(x[:9]))
path = os.path.join(directory.replace('spiking_sequence', 'adj_mat_{}'.format(measure)))
if not os.path.exists(path):
  os.makedirs(path)
num_sample = 200
num_baseline = 1
# Ls = list(np.arange(2, 101))
Ls = list(np.arange(3, 51, 2)) # L should be larger than 1 and odd
# Ls = list(np.arange(3, 101, 2)) # L should be larger than 1 and odd
Rs = [1, 100, 200, 300, 400, 500]
file = files[0] # 0, 2, 7
start_time_mouse = time.time()
print(file)
mouseID = file.replace('.npz', '').split('_')[0]
stimulus = file.replace('.npz', '').replace(mouseID + '_', '')
sequences = load_npz(os.path.join(directory, file))
sequences = sequences[np.count_nonzero(sequences[:, :min_len], axis=1) > min_spikes, :min_len]
#%%
start_time = time.time()
significant_adj_mat, significant_peaks = significant_xcorr(sequences, num_baseline=2, alpha=0.01, sign='all')
print("--- %s minutes" % ((time.time() - start_time)/60))
#%%
start_time = time.time()
significant_ccg, peak_offset = xcorr_n_fold(sequences, n=7, num_jitter=2, L=25, R=1, maxlag=12, window=100, disable=False)
print("--- %s minutes" % ((time.time() - start_time)/60))
# %%
np.sum(~np.isnan(significant_adj_mat))
np.sum(~np.isnan(significant_ccg))

#%%
# %%
######### plot example significant sharp peaks
matrix = sequences
L=25; R=1; maxlag=12
window = 100
num_jitter = 10
disable = False
xcorr = all_xcorr_ccg(matrix, window, disable=disable) # N x N x window
N = matrix.shape[0]
significant_ccg, peak_offset=np.empty((N,N)), np.empty((N,N))
significant_ccg[:] = np.nan
peak_offset[:] = np.nan
# jitter
xcorr_jittered = np.zeros((N, N, window+1, num_jitter))
sampled_matrix = pattern_jitter(matrix, L, R, num_jitter) # N, T, num_jitter
for i in range(num_jitter):
  print(i)
  xcorr_jittered[:, :, :, i] = all_xcorr_ccg(sampled_matrix[:, :, i], window, disable=disable)
total_len = len(list(itertools.permutations(range(N), 2)))
for row_a, row_b in tqdm(itertools.permutations(range(N), 2), total=total_len , miniters=int(total_len/100), disable=disable): # , miniters=int(total_len/100)
  ccg_corrected = (xcorr[row_a, row_b, :, None] - xcorr_jittered[row_a, row_b, :, :]).mean(-1)
  if ccg_corrected[:maxlag].max() > ccg_corrected.mean() + 7 * ccg_corrected.std():
  # if np.max(np.abs(corr))
    max_offset = np.argmax(ccg_corrected[:maxlag])
    significant_ccg[row_a, row_b] = ccg_corrected[:maxlag][max_offset]
    peak_offset[row_a, row_b] = max_offset
#%%
cnt = 0
for row_a, row_b in list(zip(*np.where(~np.isnan(significant_ccg))))[:5]:
  ccg_corrected = (xcorr[row_a, row_b, :, None] - xcorr_jittered[row_a, row_b, :, :]).mean(-1)
  fig = plt.figure()
  plt.plot(np.arange(window+1), ccg_corrected)
  plt.xlabel('time lag (ms)')
  plt.ylabel('signigicant CCG corrected')
  plt.savefig('./plots/sample_significant_ccg_{}.jpg'.format(cnt))
  # plt.show()
  cnt += 1
#%%
xcorr = all_xcorr_xcorr(matrix, window=100, disable=False)
#%%
row = '719161530'
col = 'spontaneous'
pos_G, neg_G, peak = pos_G_dict[row][col], neg_G_dict[row][col], peak_dict[row][col]
pos_A = nx.adjacency_matrix(pos_G)
neg_A = nx.adjacency_matrix(neg_G)
print(pos_A.todense())
#%%
cnt = 0
for row_a, row_b in list(zip(*np.where(pos_A.todense())))[:5]:
  print(row_a, row_b)
  fig = plt.figure()
  plt.plot(np.arange(window+1), xcorr[row_a, row_b])
  plt.axvline(x=peak[row_a, row_b], color='r', alpha=0.2)
  plt.xlabel('time lag (ms)')
  plt.ylabel('signigicant cross correlation')
  plt.savefig('./plots/sample_pos_significant_xcorr_{}.jpg'.format(cnt))
  # plt.show()
  cnt += 1
#%%
cnt = 0
for row_a, row_b in list(zip(*np.where(~np.isnan(significant_ccg))))[:5]:
  ccg_corrected = (xcorr[row_a, row_b, :, None] - xcorr_jittered[row_a, row_b, :, :]).mean(-1)
  fig = plt.figure()
  plt.plot(np.arange(window+1), ccg_corrected)
  plt.xlabel('time lag (ms)')
  plt.ylabel('signigicant CCG corrected')
  plt.savefig('./plots/sample_significant_ccg_{}.jpg'.format(cnt))
  # plt.show()
  cnt += 1
# %%
#################### save correlation matrices
#%%
np.seterr(divide='ignore', invalid='ignore')
############# save correlation matrices #################
# min_len, min_num = (260000, 739)
min_len, min_num = (10000, 29)
min_spikes = min_len * 0.002 # 2 Hz
measure = 'xcorr'
directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
num_baseline = 100
file_order = int(sys.argv[1])
files = os.listdir(directory)
files = [f for f in files if f.endswith('.npz')]
files.sort(key=lambda x:int(x[:9]))
# path = os.path.join(directory.replace('spiking_sequence', 'adj_mat_ccg_corrected'))
path = os.path.join(directory.replace('spiking_sequence', 'adj_mat_xcorr_shuffled'))
if not os.path.exists(path):
  os.makedirs(path)
file = files[file_order] # 0, 2, 7 spontaneous, gabors, natural_movie_three
print(file)
sequences = load_npz(os.path.join(directory, file))
sequences = sequences[np.count_nonzero(sequences[:, :min_len], axis=1) > min_spikes, :min_len]
fname = os.path.join(path, file)
#%%
# start_time = time.time()
# save_ccg_corrected(sequences=sequences, fname=fname, num_jitter=num_baseline, L=25, window=100, disable=False)
# print("--- %s minutes" % ((time.time() - start_time)/60))
# %%
start_time = time.time()
save_xcorr_shuffled(sequences, fname, num_baseline=num_baseline, disable=False)
print("--- %s minutes" % ((time.time() - start_time)/60))
#%%
#################### load significant ccg_corrected
def generate_graph(adj_mat, cc=False, weight=False):
  if not weight:
    adj_mat[adj_mat.nonzero()] = 1
  G = nx.from_numpy_array(adj_mat, create_using=nx.DiGraph) # same as from_numpy_matrix
  if cc: # extract the largest (strongly) connected components
    if np.allclose(adj_mat, adj_mat.T, rtol=1e-05, atol=1e-08): # if the matrix is symmetric
      largest_cc = max(nx.connected_components(G), key=len)
    else:
      largest_cc = max(nx.strongly_connected_components(G), key=len)
    G = nx.subgraph(G, largest_cc)
  return G

def load_significant_xcorr(directory, weight):
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

directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
path = directory.replace('spiking_sequence', 'adj_mat_all_xcorr_larger_shuffled')
if not os.path.exists(path):
  os.makedirs(path)
G_shuffle_dict, peak_dict = load_significant_xcorr(path, weight=True)
measure = 'xcorr'
#%%
directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
path = directory.replace('spiking_sequence', 'adj_mat_ccg_significant_corrected')
if not os.path.exists(path):
  os.makedirs(path)
G_ccg_dict, peak_dict = load_significant_xcorr(path, weight=True)
measure = 'ccg'
#%%
############# load area_dict and average speed dataframe #################
visual_regions = ['VISp', 'VISl', 'VISrl', 'VISal', 'VISpm', 'VISam', 'LGd', 'LP']
# session_ids = [719161530, 750749662, 755434585, 756029989]
session_ids = [719161530, 750749662, 755434585, 756029989, 791319847]
stimulus_names = ['spontaneous', 'flashes', 'gabors',
        'drifting_gratings', 'static_gratings',
          'natural_scenes', 'natural_movie_one', 'natural_movie_three']
a_file = open('./data/ecephys_cache_dir/sessions/area_dict.pkl', 'rb')
area_dict = pickle.load(a_file)
# change the keys of area_dict from int to string
int_2_str = dict((session_id, str(session_id)) for session_id in session_ids)
area_dict = dict((int_2_str[key], value) for (key, value) in area_dict.items())
a_file.close()
mean_speed_df = pd.read_pickle('./data/ecephys_cache_dir/sessions/mean_speed_df.pkl')
# %%
######### split G_dict into pos and neg
pos_G_dict, neg_G_dict = split_pos_neg(G_shuffle_dict, measure=measure)
# %%
############# keep largest connected components for pos and neg G_dict
pos_G_dict = get_lcc(pos_G_dict)
neg_G_dict = get_lcc(neg_G_dict)
# %%
print_stat(pos_G_dict)
print_stat(neg_G_dict)
# %%
plot_stat(pos_G_dict, neg_G_dict, measure=measure)
# %%
region_connection_heatmap(pos_G_dict, 'pos', area_dict, visual_regions, measure)
region_connection_heatmap(neg_G_dict, 'neg', area_dict, visual_regions, measure)
# %%
weight = False
region_connection_delta_heatmap(pos_G_dict, 'pos', area_dict, visual_regions, measure, weight)
region_connection_delta_heatmap(neg_G_dict, 'neg', area_dict, visual_regions, measure, weight)
# %%
weight = True
region_connection_delta_heatmap(pos_G_dict, 'pos', area_dict, visual_regions, measure, weight)
region_connection_delta_heatmap(neg_G_dict, 'neg', area_dict, visual_regions, measure, weight)
# %%
############# plot all graphs with community layout and color as region #################
cc = True
plot_multi_graphs_color(pos_G_dict, 'pos', area_dict, measure, cc=cc)
plot_multi_graphs_color(neg_G_dict, 'neg', area_dict, measure, cc=cc)
# cc = True
# %%
plot_directed_multi_degree_distributions(pos_G_dict, 'pos', measure, weight=None, cc=False)
plot_directed_multi_degree_distributions(neg_G_dict, 'neg', measure, weight=None, cc=False)
# %%
plot_directed_multi_degree_distributions(pos_G_dict, 'pos', measure, weight='weight', cc=False)
plot_directed_multi_degree_distributions(neg_G_dict, 'neg', measure, weight='weight', cc=False)
# %%
G_ccg_dict = get_lcc(G_ccg_dict)
# %%
print_stat(G_ccg_dict)
# %%
plot_stat(G_ccg_dict, measure=measure)
# %%
region_connection_heatmap(G_ccg_dict, 'pos', area_dict, visual_regions, measure)
# %%
weight = False
region_connection_delta_heatmap(pos_G_dict, 'pos', area_dict, visual_regions, measure, weight)
region_connection_delta_heatmap(neg_G_dict, 'neg', area_dict, visual_regions, measure, weight)
# %%
weight = True
region_connection_delta_heatmap(pos_G_dict, 'pos', area_dict, visual_regions, measure, weight)
region_connection_delta_heatmap(neg_G_dict, 'neg', area_dict, visual_regions, measure, weight)
# %%
############# plot all graphs with community layout and color as region #################
cc = True
plot_multi_graphs_color(pos_G_dict, 'pos', area_dict, measure, cc=cc)
plot_multi_graphs_color(neg_G_dict, 'neg', area_dict, measure, cc=cc)
# cc = True
# %%
plot_directed_multi_degree_distributions(pos_G_dict, 'pos', measure, weight=None, cc=False)
plot_directed_multi_degree_distributions(neg_G_dict, 'neg', measure, weight=None, cc=False)
# %%
plot_directed_multi_degree_distributions(pos_G_dict, 'pos', measure, weight='weight', cc=False)
plot_directed_multi_degree_distributions(neg_G_dict, 'neg', measure, weight='weight', cc=False)