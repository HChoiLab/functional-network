# %%
import numpy as np
import os
import matplotlib
from matplotlib import pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
import networkx as nx
import community
import time
import pickle
import random
from numpy.core.fromnumeric import size
import pandas as pd
import seaborn as sns
from plfit import plfit
from sklearn.metrics.cluster import mutual_info_score
from sklearn.metrics.cluster import normalized_mutual_info_score
import itertools
import sys
from scipy import signal
import pycorrelate as pyc
from scipy import sparse
from tqdm import tqdm
from numpy.lib.stride_tricks import as_strided
from scipy.ndimage.filters import uniform_filter1d
import faulthandler

customPalette = ['#630C3A', '#39C8C6', '#D3500C', '#FFB139', 'palegreen', 'darkblue', 'slategray', '#a6cee3', '#b2df8a', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#6a3d9a', '#b15928']

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

def down_sample(sequences, min_len, min_num):
  sequences = sequences[:, :min_len]
  i,j = np.nonzero(sequences)
  ix = np.random.choice(len(i), min_num * sequences.shape[0], replace=False)
  sample_seq = np.zeros_like(sequences)
  sample_seq[i[ix], j[ix]] = sequences[i[ix], j[ix]]
  return sample_seq

# def MI(matrix):
#   n=matrix.shape[0]
#   MI_score=np.zeros((n,n))
#   for i in range(n):
#       for j in range(i+1, n):
#           MI_score[i,j]=normalized_mutual_info_score(matrix[i,:], matrix[j,:])
#   return np.maximum(MI_score, MI_score.transpose())

def MI(matrix):
  N=matrix.shape[0]
  MI_score=np.zeros((N,N))
  for row_a, row_b in itertools.combinations(range(N), 2):
    c_xy = np.histogram2d(matrix[row_a,:], matrix[row_b,:], 100)[0]
    MI_score[row_a, row_b] = mutual_info_score(None, None, contingency=c_xy)
  return np.maximum(MI_score, MI_score.transpose())

def n_cross_correlation1(matrix):
  N, M =matrix.shape
  xcorr=np.zeros((N,N))
  norm_mata = np.nan_to_num((matrix-np.mean(matrix, axis=1).reshape(-1, 1))/(np.std(matrix, axis=1).reshape(-1, 1)*M))
  norm_matb = np.nan_to_num((matrix-np.mean(matrix, axis=1).reshape(-1, 1))/(np.std(matrix, axis=1).reshape(-1, 1)))
  total_len = len(list(itertools.permutations(range(N), 2)))
  for row_a, row_b in tqdm(itertools.permutations(range(N), 2), total=total_len):
    corr = np.correlate(norm_mata[row_a, :], norm_matb[row_b, :], mode='same')
    xcorr[row_a, row_b] = corr[np.argmax(np.abs(corr))]
  return xcorr

def n_cross_correlation2(matrix):
  N, M =matrix.shape
  xcorr=np.zeros((N,N))
  norm_mata = np.nan_to_num((matrix-np.mean(matrix, axis=1).reshape(-1, 1))/(np.std(matrix, axis=1).reshape(-1, 1)*M))
  norm_matb = np.nan_to_num((matrix-np.mean(matrix, axis=1).reshape(-1, 1))/(np.std(matrix, axis=1).reshape(-1, 1)))
  total_len = len(list(itertools.permutations(range(N), 2)))
  for row_a, row_b in tqdm(itertools.permutations(range(N), 2), total=total_len):
    corr = signal.correlate(norm_mata[row_a, :], norm_matb[row_b, :], mode='same', method='fft')
    xcorr[row_a, row_b] = corr[np.argmax(np.abs(corr))]
  return xcorr

def n_cross_correlation3(matrix):
  N, M =matrix.shape
  xcorr=np.zeros((N,N))
  norm_mata = np.nan_to_num((matrix-np.mean(matrix, axis=1).reshape(-1, 1))/(np.std(matrix, axis=1).reshape(-1, 1)*M))
  norm_matb = np.nan_to_num((matrix-np.mean(matrix, axis=1).reshape(-1, 1))/(np.std(matrix, axis=1).reshape(-1, 1)))
  total_len = len(list(itertools.permutations(range(N), 2)))
  b = np.zeros(2*M)
  for row_a, row_b in tqdm(itertools.permutations(range(N), 2), total=total_len):
    a = norm_mata[row_a, :]
    b[M//2:M//2+M] = norm_matb[row_b, :]
    corr = signal.fftconvolve(a[::-1], b, mode='valid')
    xcorr[row_a, row_b] = corr[np.argmax(np.abs(corr))]
  return xcorr

def n_cross_correlation4(matrix, maxlag): ### fastest
  N, M =matrix.shape
  xcorr=np.zeros((N,N))
  norm_mata = np.nan_to_num((matrix-np.mean(matrix, axis=1).reshape(-1, 1))/(np.std(matrix, axis=1).reshape(-1, 1)*M))
  norm_matb = np.nan_to_num((matrix-np.mean(matrix, axis=1).reshape(-1, 1))/(np.std(matrix, axis=1).reshape(-1, 1)))
  #### padding
  norm_mata = np.concatenate((np.zeros((N, maxlag)), norm_mata.conj(), np.zeros((N, maxlag))), axis=1)
  norm_matb = np.concatenate((np.zeros((N, 2*maxlag)), norm_matb.conj(), np.zeros((N, 2*maxlag))), axis=1)
  total_len = len(list(itertools.permutations(range(N), 2)))
  for row_a, row_b in tqdm(itertools.permutations(range(N), 2), total=total_len):
    px, py = norm_mata[row_a, :], norm_matb[row_b, :]
    T = as_strided(py[2*maxlag:], shape=(2*maxlag+1, M + 2*maxlag),
                   strides=(-py.strides[0], py.strides[0]))
    corr = T.dot(px)
    xcorr[row_a, row_b] = corr[np.argmax(np.abs(corr))]
  return xcorr

def n_cross_correlation5(matrix, maxlag): ### fastest, subtract sliding mean
  N, M =matrix.shape
  xcorr=np.zeros((N,N))
  norm_mata = np.nan_to_num((matrix-np.mean(matrix, axis=1).reshape(-1, 1))/(np.std(matrix, axis=1).reshape(-1, 1)*M))
  norm_matb = np.nan_to_num((matrix-np.mean(matrix, axis=1).reshape(-1, 1))/(np.std(matrix, axis=1).reshape(-1, 1)))
  #### padding
  norm_mata = np.concatenate((np.zeros((N, maxlag)), norm_mata.conj(), np.zeros((N, maxlag))), axis=1)
  norm_matb = np.concatenate((np.zeros((N, 2*maxlag)), norm_matb.conj(), np.zeros((N, 2*maxlag))), axis=1)
  total_len = len(list(itertools.permutations(range(N), 2)))
  for row_a, row_b in tqdm(itertools.permutations(range(N), 2), total=total_len):
  # for ind, (row_a, row_b) in enumerate(itertools.permutations(range(N), 2)): # , miniters=int(total_len/100)
    px, py = norm_mata[row_a, :], norm_matb[row_b, :]
    T = as_strided(py[2*maxlag:], shape=(2*maxlag+1, M + 2*maxlag),
                   strides=(-py.strides[0], py.strides[0]))
    # corr = T.dot(px)
    corr = T @ px
    corr = corr - uniform_filter1d(corr, maxlag, mode='mirror') # mirror for the boundary values
    xcorr[row_a, row_b] = corr[np.argmax(np.abs(corr))]
  return xcorr

def n_cross_correlation6(matrix, maxlag): ### fastest, only causal correlation (A>B, only positive time lag on B)
  N, M =matrix.shape
  xcorr=np.zeros((N,N))
  norm_mata = np.nan_to_num((matrix-np.mean(matrix, axis=1).reshape(-1, 1))/(np.std(matrix, axis=1).reshape(-1, 1)*M))
  norm_matb = np.nan_to_num((matrix-np.mean(matrix, axis=1).reshape(-1, 1))/(np.std(matrix, axis=1).reshape(-1, 1)))
  #### padding
  norm_mata = np.concatenate((norm_mata.conj(), np.zeros((N, maxlag))), axis=1)
  norm_matb = np.concatenate((np.zeros((N, maxlag)), norm_matb.conj(), np.zeros((N, maxlag))), axis=1) # must concat zeros to the left, why???????????
  total_len = len(list(itertools.permutations(range(N), 2)))
  for row_a, row_b in tqdm(itertools.permutations(range(N), 2), total=total_len , miniters=int(total_len/100)): # , miniters=int(total_len/100)
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

# def n_cross_correlation4(matrix, maxlag): ### fastest
#   N, M =matrix.shape
#   xcorr=np.zeros((N,N))
#   norm_mata = np.nan_to_num((matrix-np.mean(matrix, axis=1).reshape(-1, 1))/(np.std(matrix, axis=1).reshape(-1, 1)*M))
#   norm_matb = np.nan_to_num((matrix-np.mean(matrix, axis=1).reshape(-1, 1))/(np.std(matrix, axis=1).reshape(-1, 1)))
#   total_len = len(list(itertools.permutations(range(N), 2)))
#   for row_a, row_b in tqdm(itertools.permutations(range(N), 2), total=total_len):
#     a, b = norm_mata[row_a, :], norm_matb[row_b, :]
#     py = np.pad(b.conj(), 2*maxlag, mode='constant')
#     T = as_strided(py[2*maxlag:], shape=(2*maxlag+1, len(b) + 2*maxlag),
#                    strides=(-py.strides[0], py.strides[0]))
#     px = np.pad(a, maxlag, mode='constant')
#     corr = T.dot(px)
#     xcorr[row_a, row_b] = corr[np.argmax(np.abs(corr))]
#   return xcorr

def n_cross_correlation7(matrix, maxlag): ## always 0
  N, M =matrix.shape
  xcorr=np.zeros((N,N))
  norm_mata = np.nan_to_num((matrix-np.mean(matrix, axis=1).reshape(-1, 1))/(np.std(matrix, axis=1).reshape(-1, 1)*M))
  norm_matb = np.nan_to_num((matrix-np.mean(matrix, axis=1).reshape(-1, 1))/(np.std(matrix, axis=1).reshape(-1, 1)))
  total_len = len(list(itertools.permutations(range(N), 2)))
  for row_a, row_b in tqdm(itertools.permutations(range(N), 2), total=total_len):
    # a = np.nan_to_num((matrix[row_a, :]-np.mean(matrix[row_a, :]))/(np.std(matrix[row_a, :])*M))
    # b = np.nan_to_num((matrix[row_b, :]-np.mean(matrix[row_b, :]))/np.std(matrix[row_b, :]))
    corr = pyc.ucorrelate(norm_mata[row_a, :], norm_matb[row_b, :], maxlag=maxlag)
    xcorr[row_a, row_b] = corr[np.argmax(np.abs(corr))]
  return xcorr

def granger_causality(matrix):
  # calculate granger causality in time domain
  #x : array, 2d, (nobs,2)
  #data for test whether the time series in the second column Granger causes the time series in the first column
  #results : dictionary, keys arze the number of lags.
  #ssr-based F test is the "standard" granger causality test
  maxlag=100
  n=np.shape(matrix)[0]
  GC=np.zeros((n,n,maxlag))
  for i in range(n):
      for j in range(n):
          # bidirection interaction, but no auto
          if i!=j:
              # mean across orientation and repeats
              x=matrix[[i,j],:,:,20:].mean(1).mean(1).T
              G = grangercausalitytests(x, maxlag=maxlag, addconst=True, verbose=False)
              # index in list comprehension is also global, need to be careful
              fscore = [G[g][0]['ssr_ftest'][0] for g in np.arange(1,len(G)+1)]
              GC[i,j,:]=fscore

# def transfer_entropy(X, Y, lag):
    

#     # future of i
#     Fi = np.roll(X, -lag)
#     # past of i
#     Pi = X
#     # past of j
#     Pj = Y

#     #Transfer entropy
#     Inf_from_Pi_to_Fi = cond_entropy(Fi, Pi)

#     #same as cond_entropy(Fi, Pi_Pj)
#     Hy = entropy(Pi,Pj)
#     Hyx = entropy(Fi,Pj,Pi)
#     Inf_from_Pi_Pj_to_Fi = Hyx - Hy

#     TE_from_j_to_i = Inf_from_Pi_to_Fi-Inf_from_Pi_Pj_to_Fi

#     return TE_from_j_to_i

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

def save_npz(matrix, filename):
    matrix_2d = matrix.reshape(matrix.shape[0], int(len(matrix.flatten())/matrix.shape[0]))
    sparse_matrix = sparse.csc_matrix(matrix_2d)
    np.savez(filename, [sparse_matrix, matrix.shape])
    return 'npz file saved'

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
# %%
####### turn off warnings
np.seterr(divide='ignore', invalid='ignore')
start_time = time.time()
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
num_baseline = 10 # number of baseline correlation by random shuffling each sequence
num_sample = 10 # number of random sampling
# %%
for file in files:
  if file.endswith(".npz"):
    start_time_mouse = time.time()
    print(file)
    mouseID = file.replace('.npz', '').split('_')[0]
    stimulus = file.replace('.npz', '').replace(mouseID + '_', '')
    sequences = load_npz(os.path.join(directory, file))
    num_nodes = sequences.shape[0]
    all_adj_mat = np.zeros((num_nodes, num_nodes, num_sample))
    for i in range(num_sample):
      print('Sample {} out of {}'.format(i+1, num_sample))
      # if i == 1 :
      #   import pdb;pdb.set_trace()
      sample_seq = down_sample(sequences, min_len, min_num)
      # mask out neurons with firing rate under 2 Hz
      mask = np.where((sample_seq != 0).sum(axis=1) < min_spikes)[0]
      adj_mat = corr_mat(sample_seq, measure, maxlag=12)
      adj_mat[mask[:, None], :] = 0
      adj_mat[:, mask] = 0
      all_adj_mat[:, :, i] = adj_mat
    # np.save(os.path.join(path, file.replace('npz', 'npy')), all_adj_mat)
    save_npz(all_adj_mat, os.path.join(path, file))
    # sparse.save_npz(os.path.join(path, file), sparse.csr_matrix(all_adj_mat))
    adj_mat_bl = np.zeros((num_nodes, num_nodes, num_baseline))
    for b in range(num_baseline):
      print('Baseline {} out of {}'.format(b+1, num_baseline))
      for n in range(num_nodes):
        np.random.shuffle(sample_seq[n,:])
      mask = np.where((sample_seq != 0).sum(axis=1) < min_spikes)[0]
      adj_mat = corr_mat(sample_seq, measure)
      adj_mat[mask[:, None], :] = 0
      adj_mat[:, mask] = 0
      adj_mat_bl[:, :, b] = adj_mat
    # np.save(os.path.join(path, file.replace('.npz', '_bl.npy')), adj_mat_bl)
    save_npz(adj_mat_bl, os.path.join(path, file.replace('.npz', '_bl.npz')))
    # sparse.save_npz(os.path.join(path, file.replace('.npz', '_bl.npz')), sparse.csr_matrix(adj_mat_bl))
    print("--- %s minutes for %s %s" % ((time.time() - start_time_mouse)/60, mouseID, stimulus))
print("--- %s minutes in total" % ((time.time() - start_time)/60))

# %%
############ test run time
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
# files.sort(key=lambda x:int(x[:9]))
# path = os.path.join(directory.replace('spiking_sequence', 'adj_mat_{}'.format(measure)))
# if not os.path.exists(path):
#   os.makedirs(path)
# for file in files:
#   if file.endswith(".npz"):
#     start_time_mouse = time.time()
#     print(file)
#     break
# mouseID = file.replace('.npz', '').split('_')[0]
# stimulus = file.replace('.npz', '').replace(mouseID + '_', '')
# sequences = load_npz(os.path.join(directory, file))
# sample_seq = down_sample(sequences, min_len, min_num)
# %%
# start_time = time.time()
# adj_mat = n_cross_correlation5(sample_seq, 12)
# print("--- %s minutes in total" % ((time.time() - start_time)/60))
# # %%
# def running_mean_uniform_filter1d(x, N):
#     return uniform_filter1d(x, N, mode='constant', origin=-(N//2))[:-(N-1)]
# %%
