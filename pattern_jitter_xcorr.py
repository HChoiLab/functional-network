# %%
import numpy as np
import numpy.ma as ma
import pandas as pd
import os
import itertools
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
from plfit import plfit

customPalette = ['#630C3A', '#39C8C6', '#D3500C', '#FFB139', 'palegreen', 'darkblue', 'slategray', '#a6cee3', '#b2df8a', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#6a3d9a', '#b15928']

def get_rowcol(G_dict, measure):
  rows = list(G_dict.keys())
  cols = []
  for row in rows:
    cols += list(G_dict[row].keys())
  cols = list(set(cols))
  if 'drifting_gratings_contrast' in cols:
    cols.remove('drifting_gratings_contrast')
  # sort stimulus
  if measure == 'ccg':
    stimulus_rank = ['spon', 'spon_20', 'None', 'denoised', 'low', 'flash', 'flash_40', 'movie', 'movie_20']
  else:
    stimulus_rank = ['spontaneous', 'flashes', 'gabors',
        'drifting_gratings', 'static_gratings', 'drifting_gratings_contrast',
          'natural_scenes', 'natural_movie_one', 'natural_movie_three']
  stimulus_rank_dict = {i:stimulus_rank.index(i) for i in cols}
  stimulus_rank_dict = dict(sorted(stimulus_rank_dict.items(), key=lambda item: item[1]))
  cols = list(stimulus_rank_dict.keys())
  return rows, cols

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
    n = len(spikeTrain)
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
    spikeTrainMat = np.zeros((sample_size, len(spikeTrain)))
    for i in tqdm(range(sample_size), disable=True):
        surrogate = getSurrogate(spikeTrain, L, R, T, initDist, tDistMatrices)
        spikeTrainMat[i, :] = surrogate
    return spikeTrainMat

def n_cross_correlation6(matrix, maxlag, disable): ### fastest, only causal correlation (A>B, only positive time lag on B)
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
num_sample = 1000
num_baseline = 1
# Ls = list(np.arange(2, 101))
Ls = list(np.arange(3, 101, 2)) # L should be larger than 1 and odd
Rs = [1, 100, 200, 300, 400, 500, 1000]
file = files[2] # 0, 2, 7
start_time_mouse = time.time()
print(file)
mouseID = file.replace('.npz', '').split('_')[0]
stimulus = file.replace('.npz', '').replace(mouseID + '_', '')
sequences = load_npz(os.path.join(directory, file))
sequences = sequences[np.count_nonzero(sequences[:, :min_len], axis=1) > min_spikes, :min_len]
# sequences = sequences[np.count_nonzero(sequences[:, :min_len], axis=1) > 80, :min_len]
active_inds = np.argpartition(np.count_nonzero(sequences, axis=1), -2)[-2:] # top 2 most active neurons
# active_inds = np.argpartition(np.count_nonzero(sequences, axis=1), 2)[:2] # top 2 most inactive neurons
num_nodes = 2
############## Effect of pattern jitter on cross correlation
origin_adj_mat = np.zeros((2, 2))
origin_adj_mat_bl = np.zeros((2, 2, num_baseline))
all_adj_mat_A = np.zeros((2, 2, num_sample, len(Ls), len(Rs)))
adj_mat_bl_A = np.zeros((2, 2, num_baseline, len(Ls), len(Rs)))
all_adj_mat_B = np.zeros((2, 2, num_sample, len(Ls), len(Rs)))
adj_mat_bl_B = np.zeros((2, 2, num_baseline, len(Ls), len(Rs)))
all_adj_mat = np.zeros((2, 2, num_sample, len(Ls), len(Rs)))
adj_mat_bl = np.zeros((2, 2, num_baseline, len(Ls), len(Rs)))
# complete_adj_mat = corr_mat(sequences, measure, maxlag=12)
T = min_len
origin_adj_mat = corr_mat(sequences[active_inds], measure, maxlag=12)
seq = sequences[active_inds].copy()
for b in range(num_baseline):
  for n in range(num_nodes):
    np.random.shuffle(seq[n,:])
  adj_mat = corr_mat(seq, measure)
  origin_adj_mat_bl[:, :, b] = adj_mat
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
    all_adj_mat_A[:, :, :, L_ind, R_ind] = n_cross_correlation_2mat(spike_timing2train(min_len, sampled_spiketrain), sequences[active_inds[1], :], maxlag=12, disable=True)
    # for i in range(num_sample):
    #   sample_seq = np.concatenate((spike_timing2train(min_len, sampled_spiketrain[i, :])[None, :], sequences[active_inds[1], :][None, :]) ,axis=0)
    #   adj_mat = corr_mat(sample_seq, measure, maxlag=12)
    #   all_adj_mat_A[:, :, i, L_ind, R_ind] = adj_mat
    for b in range(num_baseline):
      sample_seq = np.concatenate((spike_timing2train(min_len, sampled_spiketrain[b, :])[None, :], sequences[active_inds[1], :][None, :]) ,axis=0)
      # print('Baseline {} out of {}'.format(b+1, num_baseline))
      for n in range(num_nodes):
        np.random.shuffle(sample_seq[n,:])
      adj_mat = corr_mat(sample_seq, measure)
      adj_mat_bl_A[:, :, b, L_ind, R_ind] = adj_mat
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
    all_adj_mat_B[:, :, :, L_ind, R_ind] = n_cross_correlation_2mat(sequences[active_inds[0], :], spike_timing2train(min_len, sampled_spiketrain), maxlag=12, disable=True)
    # for i in range(num_sample):
    #   sample_seq = np.concatenate((sequences[active_inds[0], :][None, :], spike_timing2train(min_len, sampled_spiketrain[i, :])[None, :]) ,axis=0)
    #   adj_mat = corr_mat(sample_seq, measure, maxlag=12)
    #   all_adj_mat_B[:, :, i, L_ind, R_ind] = adj_mat
    for b in range(num_baseline):
      sample_seq = np.concatenate((sequences[active_inds[0], :][None, :], spike_timing2train(min_len, sampled_spiketrain[b, :])[None, :]) ,axis=0)
      # print('Baseline {} out of {}'.format(b+1, num_baseline))
      for n in range(num_nodes):
        np.random.shuffle(sample_seq[n,:])
      adj_mat = corr_mat(sample_seq, measure)
      adj_mat_bl_B[:, :, b, L_ind, R_ind] = adj_mat
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
    all_adj_mat[:, :, :, L_ind, R_ind] = n_cross_correlation_2mat(spike_timing2train(min_len, sampled_spiketrain1), spike_timing2train(min_len, sampled_spiketrain2), maxlag=12, disable=True)
    # for i in range(num_sample):
    #   sample_seq = np.concatenate((spike_timing2train(min_len, sampled_spiketrain1[i, :])[None, :], spike_timing2train(min_len, sampled_spiketrain2[i, :])[None, :]), axis=0)
    #   adj_mat = corr_mat(sample_seq, measure, maxlag=12)
    #   all_adj_mat[:, :, i, L_ind, R_ind] = adj_mat
    for b in range(num_baseline):
      sample_seq = np.concatenate((spike_timing2train(min_len, sampled_spiketrain1[b, :])[None, :], spike_timing2train(min_len, sampled_spiketrain2[b, :])[None, :]), axis=0)
      # print('Baseline {} out of {}'.format(b+1, num_baseline))
      for n in range(num_nodes):
        np.random.shuffle(sample_seq[n,:])
      adj_mat = corr_mat(sample_seq, measure)
      adj_mat_bl[:, :, b, L_ind, R_ind] = adj_mat
print("--- %s minutes" % ((time.time() - start_time_both)/60))
print("--- %s minutes in total" % ((time.time() - start_time)/60))
# %%
############## one pair of neurons, significant xcorr vs L and R
def plot_xcorr_LR(origin_adj_mat, all_adj_mat_A, all_adj_mat_B, all_adj_mat, Ls, R, Rs, edge_type='active'):
  plt.figure(figsize=(20, 6))
  all_mat = [all_adj_mat_A, all_adj_mat_B, all_adj_mat]
  titles = ['Pattern jittering neuron A', 'Pattern jittering neuron B', 'Pattern jittering neuron A and B']
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
  plot_xcorr_LR(origin_adj_mat, all_adj_mat_A, all_adj_mat_B, all_adj_mat, Ls, R, Rs, 'active')
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
  rows, cols = get_rowcol(FR_dict, measure)
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
adj_mat = corr_mat(sequences, measure, maxlag=12, noprogressbar=False)
np.fill_diagonal(adj_mat, np.nan)
# adj_mat_flat = adj_mat[~np.eye(adj_mat.shape[0],dtype=bool)]
firing_rates = np.count_nonzero(sequences, axis=1) / sequences.shape[1]
bins=np.logspace(start=np.log10(firing_rates.min()), stop=np.log10(firing_rates.max()+0.0001), num=50)
bin_num = np.digitize(firing_rates, bins)
# %%
corr = np.zeros((len(bins), len(bins)))
for i in range(1, len(bins)):
  for j in range(1, len(bins)):
    corr[i, j] = np.nanmean(adj_mat[np.where(bin_num==i)[0][:, None], np.where(bin_num==j)[0][None, :]])
plt.xscale('log')
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
xcorr_dict, pcorr_dict, bin_dict = {}, {}, {}
for session_id in session_ids:
  print(session_id)
  xcorr_dict[session_id], pcorr_dict[session_id], bin_dict[session_id] = {}, {}, {}
  for stimulus_name in stimulus_names:
    print(stimulus_name)
    sequences = load_npz(os.path.join(directory, str(session_id) + '_' + stimulus_name + '.npz'))
    sequences = sequences[np.count_nonzero(sequences[:, :min_len], axis=1) > min_spikes, :min_len]
    adj_mat = corr_mat(sequences, measure, maxlag=12, noprogressbar=False)
    np.fill_diagonal(adj_mat, np.nan)
    p_adj_mat = corr_mat(sequences, measure='pearson', maxlag=12, noprogressbar=False)
    np.fill_diagonal(p_adj_mat, np.nan)
    # adj_mat_flat = adj_mat[~np.eye(adj_mat.shape[0],dtype=bool)]
    firing_rates = np.count_nonzero(sequences, axis=1) / sequences.shape[1]
    bins=np.logspace(start=np.log10(firing_rates.min()), stop=np.log10(firing_rates.max()+0.0001), num=20)
    bin_num = np.digitize(firing_rates, bins)
    xcorr = np.zeros((len(bins), len(bins)))
    corr = np.zeros((len(bins), len(bins)))
    for i in range(len(bins)):
      for j in range(len(bins)):
        xcorr[i, j] = np.nanmean(adj_mat[np.where(bin_num==i)[0][:, None], np.where(bin_num==j)[0][None, :]])
        corr[i, j] = np.nanmean(p_adj_mat[np.where(bin_num==i)[0][:, None], np.where(bin_num==j)[0][None, :]])
    xcorr_dict[session_id][stimulus_name] = xcorr
    pcorr_dict[session_id][stimulus_name] = corr
    bin_dict[session_id][stimulus_name] = bins
print("--- %s minutes in total" % ((time.time() - start_time)/60))
#%%
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
  plt.suptitle('pearson correlation VS firing rate', size=40)
  plt.tight_layout()
  # plt.show()
  plt.savefig('./plots/pcrorr_FR_multi.jpg')
plot_multi_heatmap_xcorr_FR(session_ids, stimulus_names, pcorr_dict, bin_dict)
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
  plt.savefig('./plots/{}_corr_FR_multi.jpg'.format(name))
plot_multi_corr_FR(session_ids, stimulus_names, pcorr_dict, bin_dict, 'pearson')
plot_multi_corr_FR(session_ids, stimulus_names, xcorr_dict, bin_dict, 'cross')
# %%
