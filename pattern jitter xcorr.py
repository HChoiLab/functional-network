# %%
import numpy as np
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
import statsmodels.stats.weightstats as ws
import networkx as nx
import community
import seaborn as sns
from numpy.lib.stride_tricks import as_strided
from scipy.ndimage.filters import uniform_filter1d
from plfit import plfit

customPalette = ['#630C3A', '#39C8C6', '#D3500C', '#FFB139', 'palegreen', 'darkblue', 'slategray', '#a6cee3', '#b2df8a', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#6a3d9a', '#b15928']

def getSpikeTrain(spikeData):
    spikeTrain = np.squeeze(np.where(spikeData>0))
    return spikeTrain

def spike_timing2train(T, spikeTrain):
    spikeData = np.zeros(T)
    spikeData[spikeTrain.astype(int)] = 1
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

def n_cross_correlation6(matrix, maxlag): ### fastest, only causal correlation (A>B, only positive time lag on B)
  N, M =matrix.shape
  xcorr=np.zeros((N,N))
  norm_mata = np.nan_to_num((matrix-np.mean(matrix, axis=1).reshape(-1, 1))/(np.std(matrix, axis=1).reshape(-1, 1)*M))
  norm_matb = np.nan_to_num((matrix-np.mean(matrix, axis=1).reshape(-1, 1))/(np.std(matrix, axis=1).reshape(-1, 1)))
  #### padding
  norm_mata = np.concatenate((norm_mata.conj(), np.zeros((N, maxlag))), axis=1)
  norm_matb = np.concatenate((np.zeros((N, maxlag)), norm_matb.conj(), np.zeros((N, maxlag))), axis=1) # must concat zeros to the left, why???????????
  total_len = len(list(itertools.permutations(range(N), 2)))
  for row_a, row_b in tqdm(itertools.permutations(range(N), 2), total=total_len , miniters=int(total_len/100), disable=True): # , miniters=int(total_len/100)
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

def corr_mat(sequences, measure, maxlag=12):
  if measure == 'pearson':
    adj_mat = np.corrcoef(sequences)
  elif measure == 'cosine':
    adj_mat = cosine_similarity(sequences)
  elif measure == 'correlation':
    adj_mat = squareform(pdist(sequences, 'correlation'))
  elif measure == 'MI':
    adj_mat = MI(sequences)
  elif measure == 'xcorr':
    adj_mat = n_cross_correlation6(sequences, maxlag=maxlag)
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
files.sort(key=lambda x:int(x[:9]))
path = os.path.join(directory.replace('spiking_sequence', 'adj_mat_{}'.format(measure)))
if not os.path.exists(path):
  os.makedirs(path)
num_sample = 100
num_baseline = 100
Ls = [2, 10, 20, 50, 100, 200] # L should be larger than 1
Rs = [1, 10, 50, 100, 500, 1000, 2000]
# %%
start_time = time.time()
for file in files:
  if file.endswith(".npz"):
    start_time_mouse = time.time()
    print(file)
    mouseID = file.replace('.npz', '').split('_')[0]
    stimulus = file.replace('.npz', '').replace(mouseID + '_', '')
    break
sequences = load_npz(os.path.join(directory, file))
sequences = sequences[np.count_nonzero(sequences[:, :min_len], axis=1) > 80, :min_len]
active_inds = np.argpartition(np.count_nonzero(sequences, axis=1), -2)[-2:] # top 2 most active neurons
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
print('Sampling neuron A...')
for L_ind, L in enumerate(Ls):
  for R_ind, R in enumerate(Rs):
    spikeTrain = getSpikeTrain(sequences[active_inds[0], :])
    N = len(spikeTrain)
    initDist = getInitDist(L)
    tDistMatrices = getTransitionMatrices(L, N)
    sampled_spiketrain = sample_spiketrain(L, R, T, spikeTrain, initDist, tDistMatrices, num_sample)
    for i in range(num_sample):
      sample_seq = np.concatenate((spike_timing2train(min_len, sampled_spiketrain[i, :])[None, :], sequences[active_inds[1], :][None, :]) ,axis=0)
      adj_mat = corr_mat(sample_seq, measure, maxlag=12)
      all_adj_mat_A[:, :, i, L_ind, R_ind] = adj_mat
    for b in range(num_baseline):
      # print('Baseline {} out of {}'.format(b+1, num_baseline))
      for n in range(num_nodes):
        np.random.shuffle(sample_seq[n,:])
      mask = np.where((sample_seq != 0).sum(axis=1) < min_spikes)[0]
      adj_mat = corr_mat(sample_seq, measure)
      adj_mat[mask[:, None], :] = 0
      adj_mat[:, mask] = 0
      adj_mat_bl_A[:, :, b, L_ind, R_ind] = adj_mat
print('Sampling neuron B...')
for L_ind, L in enumerate(Ls):
  for R_ind, R in enumerate(Rs):
    spikeTrain = getSpikeTrain(sequences[active_inds[1], :])
    N = len(spikeTrain)
    initDist = getInitDist(L)
    tDistMatrices = getTransitionMatrices(L, N)
    sampled_spiketrain = sample_spiketrain(L, R, T, spikeTrain, initDist, tDistMatrices, num_sample)
    for i in range(num_sample):
      sample_seq = np.concatenate((sequences[active_inds[0], :][None, :], spike_timing2train(min_len, sampled_spiketrain[i, :])[None, :]) ,axis=0)
      adj_mat = corr_mat(sample_seq, measure, maxlag=12)
      all_adj_mat_B[:, :, i, L_ind, R_ind] = adj_mat
    for b in range(num_baseline):
      # print('Baseline {} out of {}'.format(b+1, num_baseline))
      for n in range(num_nodes):
        np.random.shuffle(sample_seq[n,:])
      mask = np.where((sample_seq != 0).sum(axis=1) < min_spikes)[0]
      adj_mat = corr_mat(sample_seq, measure)
      adj_mat[mask[:, None], :] = 0
      adj_mat[:, mask] = 0
      adj_mat_bl_B[:, :, b, L_ind, R_ind] = adj_mat
print('Sampling neuron A and B...')
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
    for i in range(num_sample):
      sample_seq = np.concatenate((spike_timing2train(min_len, sampled_spiketrain1[i, :])[None, :], spike_timing2train(min_len, sampled_spiketrain2[i, :])[None, :]), axis=0)
      adj_mat = corr_mat(sample_seq, measure, maxlag=12)
      all_adj_mat[:, :, i, L_ind, R_ind] = adj_mat
    for b in range(num_baseline):
      # print('Baseline {} out of {}'.format(b+1, num_baseline))
      for n in range(num_nodes):
        np.random.shuffle(sample_seq[n,:])
      mask = np.where((sample_seq != 0).sum(axis=1) < min_spikes)[0]
      adj_mat = corr_mat(sample_seq, measure)
      adj_mat[mask[:, None], :] = 0
      adj_mat[:, mask] = 0
      adj_mat_bl[:, :, b, L_ind, R_ind] = adj_mat
print("--- %s minutes in total" % ((time.time() - start_time)/60))
# %%
############## one pair of neurons, significant xcorr vs L and R
def plot_xcorr_LR(origin_adj_mat, all_adj_mat_A, all_adj_mat_B, all_adj_mat, Ls, R, Rs):
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
    plt.gca().set_title(titles[i], fontsize=20, rotation=0)
    plt.xticks(rotation=90)
    plt.xlabel('Bin size L', fontsize=15)
    plt.ylabel('cross correlation', fontsize=15)
    plt.legend()
  plt.tight_layout()
  # plt.show()
  figname = './plots/xcorr_vs_L_R_{}_active_{}.jpg'.format(R, measure)
  plt.savefig(figname)

Rs = [1, 10, 50, 100, 500, 1000, 2000]
R = 1
plot_xcorr_LR(origin_adj_mat, all_adj_mat_A, all_adj_mat_B, all_adj_mat, Ls, R, Rs)
# %%
