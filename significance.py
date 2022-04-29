# %%
from email import iterators
import numpy as np
from scipy import sparse
import pandas as pd
import os
import itertools
from scipy.stats import shapiro
from scipy.stats import normaltest
from tqdm import tqdm
import pickle
import time
import matplotlib.pyplot as plt
import statsmodels.stats.weightstats as ws
import networkx as nx

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

def Z_score(r):
  return np.log((1+r)/(1-r)) / 2
# %%
################# normal test for correlation distribution
# start_time = time.time()
# measure = 'pearson'
# alpha = 0.05
# ########## load networks with baseline
# directory = './data/ecephys_cache_dir/sessions/adj_mat_{}'.format(measure)
# # thresholds = list(np.arange(0, 0.014, 0.001))
# stimulus_names = ['spontaneous', 'flashes', 'gabors',
#         'drifting_gratings', 'static_gratings',
#           'natural_scenes', 'natural_movie_one', 'natural_movie_three']
# session_ids = [719161530, 750749662, 755434585, 756029989, 791319847]

# weight = True # weighted network
# SW_dict = {}
# DA_dict = {}
# SW_bl_dict = {}
# DA_bl_dict = {}
# files = os.listdir(directory)
# files.sort(key=lambda x:int(x[:9]))
# for file in files:
#   if '_bl' not in file:
#     print(file)
#     mouseID = file.replace('.npy', '').split('_')[0]
#     stimulus = file.replace('.npy', '').replace(mouseID + '_', '')
#     adj_mat = np.load(os.path.join(directory, file))
#     adj_mat_bl = np.load(os.path.join(directory, file.replace('.npy', '_bl.npy')))
#     SW_p = []
#     DA_p = []
#     SW_p_bl = []
#     DA_p_bl = []
#     print('Shapiro-Wilk Test and D’Agostino’s K^2 Test for adj_mat...')
#     total_len = len(list(itertools.combinations(range(adj_mat.shape[0]), 2)))
#     for row_a, row_b in tqdm(itertools.combinations(range(adj_mat.shape[0]), 2), total=total_len):
#         _, p = shapiro(adj_mat[row_a, row_b, :])
#         SW_p.append(p)
#         _, p = normaltest(adj_mat[row_a, row_b, :])
#         DA_p.append(p)
#     print('Shapiro-Wilk Test and D’Agostino’s K^2 Test for adj_mat_bl...')
#     total_len = len(list(itertools.combinations(range(adj_mat_bl.shape[0]), 2)))
#     for row_a, row_b in tqdm(itertools.combinations(range(adj_mat_bl.shape[0]), 2), total=total_len):
#         _, p = shapiro(adj_mat_bl[row_a, row_b, :])
#         SW_p_bl.append(p)
#         _, p = normaltest(adj_mat_bl[row_a, row_b, :])
#         DA_p_bl.append(p)
#     if not mouseID in SW_dict:
#       SW_dict[mouseID] = {}
#       DA_dict[mouseID] = {}
#       SW_bl_dict[mouseID] = {}
#       DA_bl_dict[mouseID] = {}
#     SW_dict[mouseID][stimulus] = SW_p
#     DA_dict[mouseID][stimulus] = DA_p
#     SW_bl_dict[mouseID][stimulus] = SW_p_bl
#     DA_bl_dict[mouseID][stimulus] = DA_p_bl
    
# dire = './data/ecephys_cache_dir/sessions/'
# a_file = open(os.path.join(dire, "{}_SW.pkl".format(measure)), "wb")
# pickle.dump(SW_dict, a_file)
# a_file.close()
# a_file = open(os.path.join(dire, "{}_SW_bl.pkl".format(measure)), "wb")
# pickle.dump(SW_bl_dict, a_file)
# a_file.close()
# a_file = open(os.path.join(dire, "{}_DA.pkl".format(measure)), "wb")
# pickle.dump(DA_dict, a_file)
# a_file.close()
# a_file = open(os.path.join(dire, "{}_DA_bl.pkl".format(measure)), "wb")
# pickle.dump(DA_bl_dict, a_file)
# a_file.close()
# print("--- %s minutes in total" % ((time.time() - start_time)/60))
# %%
################# normal test for correlation distribution fisher's r to z
# start_time = time.time()
# measure = 'pearson'
# alpha = 0.05
# ########## load networks with baseline
# directory = './data/ecephys_cache_dir/sessions/adj_mat_{}'.format(measure)
# # thresholds = list(np.arange(0, 0.014, 0.001))
# stimulus_names = ['spontaneous', 'flashes', 'gabors',
#         'drifting_gratings', 'static_gratings',
#           'natural_scenes', 'natural_movie_one', 'natural_movie_three']
# session_ids = [719161530, 750749662, 755434585, 756029989, 791319847]

# weight = True # weighted network
# SW_dict = {}
# DA_dict = {}
# SW_bl_dict = {}
# DA_bl_dict = {}
# files = os.listdir(directory)
# files.sort(key=lambda x:int(x[:9]))
# for file in files:
#   if '_bl' not in file:
#     print(file)
#     mouseID = file.replace('.npz', '').split('_')[0]
#     stimulus = file.replace('.npz', '').replace(mouseID + '_', '')
#     adj_mat = load_npz_3d(os.path.join(directory, file))
#     adj_mat_bl = load_npz_3d(os.path.join(directory, file.replace('.npz', '_bl.npz')))
#     SW_p = []
#     DA_p = []
#     SW_p_bl = []
#     DA_p_bl = []
#     print('Shapiro-Wilk Test and D’Agostino’s K^2 Test for adj_mat...')
#     total_len = len(list(itertools.combinations(range(adj_mat.shape[0]), 2)))
#     for row_a, row_b in tqdm(itertools.combinations(range(adj_mat.shape[0]), 2), total=total_len):
#         _, p = shapiro(Z_score(adj_mat[row_a, row_b, :]))
#         SW_p.append(p)
#         _, p = normaltest(Z_score(adj_mat[row_a, row_b, :]))
#         DA_p.append(p)
#     print('Shapiro-Wilk Test and D’Agostino’s K^2 Test for adj_mat_bl...')
#     total_len = len(list(itertools.combinations(range(adj_mat_bl.shape[0]), 2)))
#     for row_a, row_b in tqdm(itertools.combinations(range(adj_mat_bl.shape[0]), 2), total=total_len):
#         _, p = shapiro(Z_score(adj_mat_bl[row_a, row_b, :]))
#         SW_p_bl.append(p)
#         _, p = normaltest(Z_score(adj_mat_bl[row_a, row_b, :]))
#         DA_p_bl.append(p)
#     if not mouseID in SW_dict:
#       SW_dict[mouseID] = {}
#       DA_dict[mouseID] = {}
#       SW_bl_dict[mouseID] = {}
#       DA_bl_dict[mouseID] = {}
#     SW_dict[mouseID][stimulus] = SW_p
#     DA_dict[mouseID][stimulus] = DA_p
#     SW_bl_dict[mouseID][stimulus] = SW_p_bl
#     DA_bl_dict[mouseID][stimulus] = DA_p_bl
    
# dire = './data/ecephys_cache_dir/sessions/'
# a_file = open(os.path.join(dire, "{}_Z_SW.pkl".format(measure)), "wb")
# pickle.dump(SW_dict, a_file)
# a_file.close()
# a_file = open(os.path.join(dire, "{}_Z_SW_bl.pkl".format(measure)), "wb")
# pickle.dump(SW_bl_dict, a_file)
# a_file.close()
# a_file = open(os.path.join(dire, "{}_Z_DA.pkl".format(measure)), "wb")
# pickle.dump(DA_dict, a_file)
# a_file.close()
# a_file = open(os.path.join(dire, "{}_Z_DA_bl.pkl".format(measure)), "wb")
# pickle.dump(DA_bl_dict, a_file)
# a_file.close()
# print("--- %s minutes in total" % ((time.time() - start_time)/60))
# # %%
# ##################### plot percentage of links that follow normal distribution
# stimulus_names = ['spontaneous', 'flashes', 'gabors',
#         'drifting_gratings', 'static_gratings',
#           'natural_scenes', 'natural_movie_one', 'natural_movie_three']
# session_ids = [719161530, 750749662, 755434585, 756029989, 791319847]
# SW = pd.DataFrame(index=session_ids, columns=stimulus_names)
# DA = pd.DataFrame(index=session_ids, columns=stimulus_names)
# SW_bl = pd.DataFrame(index=session_ids, columns=stimulus_names)
# DA_bl = pd.DataFrame(index=session_ids, columns=stimulus_names)
# measure = 'pearson'
# alpha = 0.05
# dire = './data/ecephys_cache_dir/sessions/'
# a_file = open(os.path.join(dire, "{}_Z_SW.pkl".format(measure)), "rb")
# SW_dict = pickle.load(a_file)
# a_file.close()
# a_file = open(os.path.join(dire, "{}_Z_SW_bl.pkl".format(measure)), "rb")
# SW_bl_dict = pickle.load(a_file)
# a_file.close()
# a_file = open(os.path.join(dire, "{}_Z_DA.pkl".format(measure)), "rb")
# DA_dict = pickle.load(a_file)
# a_file.close()
# a_file = open(os.path.join(dire, "{}_Z_DA_bl.pkl".format(measure)), "rb")
# DA_bl_dict = pickle.load(a_file)
# a_file.close()
# # %%
# alpha = 0.05
# for mouseID in SW_dict:
#   print(mouseID)
#   for stimulus in SW_dict[mouseID]:
#     print(stimulus)
#     SW_p = SW_dict[mouseID][stimulus]
#     SW_p_bl = SW_bl_dict[mouseID][stimulus]
#     DA_p = DA_dict[mouseID][stimulus]
#     DA_p_bl = DA_bl_dict[mouseID][stimulus]
#     SW.loc[int(mouseID)][stimulus] = (np.array(SW_p) > alpha).sum() / len(SW_p)
#     SW_bl.loc[int(mouseID)][stimulus] = (np.array(SW_p_bl) > alpha).sum() / len(SW_p_bl)
#     DA.loc[int(mouseID)][stimulus] = (np.array(DA_p) > alpha).sum() / len(DA_p)
#     DA_bl.loc[int(mouseID)][stimulus] = (np.array(DA_p_bl) > alpha).sum() / len(DA_p_bl)
# # %%
# plt.figure(figsize=(7, 6))
# for session_id in session_ids:
#   plt.plot(stimulus_names, SW.loc[int(session_id)], label=session_id, alpha=1)
# plt.gca().set_title('percentage of normal distribution(Shapiro-Wilk Test)', fontsize=20, rotation=0)
# plt.xticks(rotation=90)
# plt.legend()
# plt.tight_layout()
# plt.savefig('./plots/SW_percentage_Z.jpg')
# plt.figure(figsize=(7, 6))
# for session_id in session_ids:
#   plt.plot(stimulus_names, SW_bl.loc[int(session_id)], label=session_id, alpha=1)
# plt.gca().set_title('percentage of shuffled normal distribution(Shapiro-Wilk Test)', fontsize=20, rotation=0)
# plt.xticks(rotation=90)
# plt.legend()
# plt.tight_layout()
# plt.savefig('./plots/SW_percentage_bl_Z.jpg')
# # %%
# ##################### scatter plot of p-value of sampled and shuffled
# stimulus_names = ['spontaneous', 'flashes', 'gabors',
#         'drifting_gratings', 'static_gratings',
#           'natural_scenes', 'natural_movie_one', 'natural_movie_three']
# session_ids = [719161530, 750749662, 755434585, 756029989, 791319847]
# measure = 'pearson'
# alpha = 0.05
# dire = './data/ecephys_cache_dir/sessions/'
# a_file = open(os.path.join(dire, "{}_SW.pkl".format(measure)), "rb")
# SW_dict = pickle.load(a_file)
# a_file.close()
# a_file = open(os.path.join(dire, "{}_SW_bl.pkl".format(measure)), "rb")
# SW_bl_dict = pickle.load(a_file)
# a_file.close()
# a_file = open(os.path.join(dire, "{}_DA.pkl".format(measure)), "rb")
# DA_dict = pickle.load(a_file)
# a_file.close()
# a_file = open(os.path.join(dire, "{}_DA_bl.pkl".format(measure)), "rb")
# DA_bl_dict = pickle.load(a_file)
# a_file.close()
# rows, cols = session_ids, stimulus_names
# # %%
# fig = plt.figure(figsize=(4*len(cols), 3*len(rows)))
# left, width = .25, .5
# bottom, height = .25, .5
# right = left + width
# top = bottom + height
# ind = 1
# for row_ind, row in enumerate(rows):
#   print(row)
#   for col_ind, col in enumerate(cols):
#     plt.subplot(len(rows), len(cols), ind)
#     if row_ind == 0:
#       plt.gca().set_title(cols[col_ind], fontsize=20, rotation=0)
#     if col_ind == 0:
#       plt.gca().text(0, 0.5 * (bottom + top), rows[row_ind],
#       horizontalalignment='left',
#       verticalalignment='center',
#       # rotation='vertical',
#       transform=plt.gca().transAxes, fontsize=20, rotation=90)
#     plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
#     ind += 1
#     plt.scatter(SW_bl_dict[str(row)][col], SW_dict[str(row)][col], s=5, alpha=0.2)
#     plt.xlabel('p-value for shuffled distribution')
#     plt.ylabel('p-value for downsampled distribution')
#     # plt.xscale('log')
#     # plt.yscale('log')
    
# plt.tight_layout()
# plt.show()
# # %%
# ######################## variance of distributions
# measure = 'pearson'
# ########## load networks with baseline
# directory = './data/ecephys_cache_dir/sessions/adj_mat_{}'.format(measure)
# # thresholds = list(np.arange(0, 0.014, 0.001))
# stimulus_names = ['spontaneous', 'flashes', 'gabors',
#         'drifting_gratings', 'static_gratings',
#           'natural_scenes', 'natural_movie_one', 'natural_movie_three']
# session_ids = [719161530, 750749662, 755434585, 756029989, 791319847]
# var_dict = {}
# var_bl_dict = {}
# files = os.listdir(directory)
# files.sort(key=lambda x:int(x[:9]))
# for file in files:
#   if '_bl' not in file:
#     print(file)
#     mouseID = file.replace('.npy', '').split('_')[0]
#     stimulus = file.replace('.npy', '').replace(mouseID + '_', '')
#     adj_mat = np.load(os.path.join(directory, file))
#     adj_mat_bl = np.load(os.path.join(directory, file.replace('.npy', '_bl.npy')))
#     var = []
#     var_bl = []
#     print('Variance for adj_mat...')
#     total_len = len(list(itertools.combinations(range(adj_mat.shape[0]), 2)))
#     for row_a, row_b in tqdm(itertools.combinations(range(adj_mat.shape[0]), 2), total=total_len):
#         var.append(np.var(adj_mat[row_a, row_b, :]))
#     print('Variance for adj_mat_bl...')
#     total_len = len(list(itertools.combinations(range(adj_mat_bl.shape[0]), 2)))
#     for row_a, row_b in tqdm(itertools.combinations(range(adj_mat_bl.shape[0]), 2), total=total_len):
#         var_bl.append(np.var(adj_mat_bl[row_a, row_b, :]))
#     if not mouseID in var_dict:
#       var_dict[mouseID] = {}
#       var_bl_dict[mouseID] = {}
#     var_dict[mouseID][stimulus] = var
#     var_bl_dict[mouseID][stimulus] = var_bl
# # %%
# rows, cols = session_ids, stimulus_names
# fig = plt.figure(figsize=(4*len(cols), 3*len(rows)))
# left, width = .25, .5
# bottom, height = .25, .5
# right = left + width
# top = bottom + height
# ind = 1
# for row_ind, row in enumerate(rows):
#   print(row)
#   for col_ind, col in enumerate(cols):
#     plt.subplot(len(rows), len(cols), ind)
#     if row_ind == 0:
#       plt.gca().set_title(cols[col_ind], fontsize=20, rotation=0)
#     if col_ind == 0:
#       plt.gca().text(0, 0.5 * (bottom + top), rows[row_ind],
#       horizontalalignment='left',
#       verticalalignment='center',
#       # rotation='vertical',
#       transform=plt.gca().transAxes, fontsize=20, rotation=90)
#     plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
#     ind += 1
#     plt.scatter(var_bl_dict[str(row)][col], var_dict[str(row)][col], s=5, alpha=0.2)
#     plt.xlabel('variance for shuffled distribution')
#     plt.ylabel('variance for downsampled distribution')
#     # plt.xscale('log')
#     # plt.yscale('log')
    
# plt.tight_layout()
# # plt.show()
# plt.savefig('./plots/variance_scatter.jpg')
# # %%
# ################### z test between downsample and shuffle
# measure = 'pearson'
# ########## load networks with baseline
# directory = './data/ecephys_cache_dir/sessions/adj_mat_{}'.format(measure)
# # thresholds = list(np.arange(0, 0.014, 0.001))
# stimulus_names = ['spontaneous', 'flashes', 'gabors',
#         'drifting_gratings', 'static_gratings',
#           'natural_scenes', 'natural_movie_one', 'natural_movie_three']
# session_ids = [719161530, 750749662, 755434585, 756029989, 791319847]
# z_stat_dict = {}
# p_dict = {}
# files = os.listdir(directory)
# files.sort(key=lambda x:int(x[:9]))
# for file in files:
#   if '_bl' not in file:
#     print(file)
#     mouseID = file.replace('.npy', '').split('_')[0]
#     stimulus = file.replace('.npy', '').replace(mouseID + '_', '')
#     adj_mat = np.load(os.path.join(directory, file))
#     adj_mat_bl = np.load(os.path.join(directory, file.replace('.npy', '_bl.npy')))
#     zstats = []
#     pvals = []
#     print('Z test...')
#     total_len = len(list(itertools.combinations(range(adj_mat.shape[0]), 2)))
#     for row_a, row_b in tqdm(itertools.combinations(range(adj_mat.shape[0]), 2), total=total_len):
#         sample = ws.DescrStatsW(adj_mat[row_a, row_b, :])
#         shuffle = ws.DescrStatsW(adj_mat_bl[row_a, row_b, :])
#         cm_obj = ws.CompareMeans(sample, shuffle)
#         zstat, z_pval = cm_obj.ztest_ind(alternative='larger', usevar='unequal', value=0)
#         zstats.append(zstat)
#         pvals.append(z_pval)
#     if not mouseID in z_stat_dict:
#       z_stat_dict[mouseID] = {}
#       p_dict[mouseID] = {}
#     z_stat_dict[mouseID][stimulus] = zstats
#     p_dict[mouseID][stimulus] = pvals
# # %%
# alpha = 0.01
# zs = pd.DataFrame(index=session_ids, columns=stimulus_names)
# pv = pd.DataFrame(index=session_ids, columns=stimulus_names)
# for mouseID in z_stat_dict:
#   print(mouseID)
#   for stimulus in z_stat_dict[mouseID]:
#     print(stimulus)
#     z_stat = z_stat_dict[mouseID][stimulus]
#     pval = p_dict[mouseID][stimulus]
#     zs.loc[int(mouseID)][stimulus] = (np.array(z_stat) < alpha).sum() / len(z_stat)
#     pv.loc[int(mouseID)][stimulus] = (np.array(pval) < alpha).sum() / len(pval)
# # %%
# plt.figure(figsize=(7, 6))
# for session_id in session_ids:
#   plt.plot(stimulus_names, pv.loc[int(session_id)], label=session_id, alpha=1)
# plt.gca().set_title('percentage of significant edges', fontsize=20, rotation=0)
# plt.xticks(rotation=90)
# plt.legend()
# plt.tight_layout()
# plt.savefig('./plots/t_test_percentage.jpg')
# plt.figure(figsize=(7, 6))
# %%
# def save_adj_ztest_normal(directory, measure, alpha, SW_dict, SW_bl_dict):
#   path = directory.replace(measure, measure+'_ztest')
#   if not os.path.exists(path):
#     os.makedirs(path)
#   files = os.listdir(directory)
#   files.sort(key=lambda x:int(x[:9]))
#   for file in files:
#     if '_bl' not in file:
#       print(file)
#       mouseID = file.replace('.npz', '').split('_')[0]
#       stimulus = file.replace('.npz', '').replace(mouseID + '_', '')
#       SW_p = SW_dict[mouseID][stimulus]
#       SW_p_bl = SW_bl_dict[mouseID][stimulus]
#       # adj_mat_ds = np.load(os.path.join(directory, file))
#       # adj_mat_bl = np.load(os.path.join(directory, file.replace('.npz', '_bl.npz')))
#       adj_mat_ds = load_npz_3d(os.path.join(directory, file))
#       adj_mat_bl = load_npz_3d(os.path.join(directory, file.replace('.npz', '_bl.npz')))
#       adj_mat = np.zeros_like(adj_mat_ds)
#       total_len = len(list(itertools.combinations(range(adj_mat_ds.shape[0]), 2)))
#       for ind, (row_a, row_b) in tqdm(enumerate(itertools.combinations(range(adj_mat_ds.shape[0]), 2)), total=total_len):
#         if SW_p[ind] > 0.05 and SW_p_bl[ind] > 0.05 and adj_mat_ds[row_a, row_b, :].mean() > 0: # only keep positive edges
#           sample = ws.DescrStatsW(adj_mat_ds[row_a, row_b, :])
#           shuffle = ws.DescrStatsW(adj_mat_bl[row_a, row_b, :])
#           cm_obj = ws.CompareMeans(sample, shuffle)
#           zstat, z_pval = cm_obj.ztest_ind(alternative='larger', usevar='unequal', value=0)
#           if z_pval < alpha:
#             adj_mat[row_a, row_b, :] = adj_mat_ds[row_a, row_b, :]
#       # np.save(os.path.join(path, file), adj_mat)
#       save_npz(adj_mat, os.path.join(path, file))

# start_time = time.time()
# alpha = 0.05
# measure = 'pearson'
# dire = './data/ecephys_cache_dir/sessions/'
# a_file = open(os.path.join(dire, "{}_Z_SW.pkl".format(measure)), "rb")
# SW_dict = pickle.load(a_file)
# a_file.close()
# a_file = open(os.path.join(dire, "{}_Z_SW_bl.pkl".format(measure)), "rb")
# SW_bl_dict = pickle.load(a_file)
# a_file.close()
# directory = './data/ecephys_cache_dir/sessions/adj_mat_{}/'.format(measure)
# save_adj_ztest_normal(directory, measure, alpha, SW_dict, SW_bl_dict)
# print("--- %s minutes in total" % ((time.time() - start_time)/60))


# %%
# directory = './data/ecephys_cache_dir/sessions/adj_mat_{}/'.format(measure)
# path = directory.replace(measure, measure+'_ztest')
# if not os.path.exists(path):
#   os.makedirs(path)
# files = os.listdir(directory)
# files.sort(key=lambda x:int(x[:9]))
# for file in files:
#   if '_bl' not in file:
#     print(file)
#     mouseID = file.replace('.npz', '').split('_')[0]
#     stimulus = file.replace('.npz', '').replace(mouseID + '_', '')
#     SW_p = SW_dict[mouseID][stimulus]
#     SW_p_bl = SW_bl_dict[mouseID][stimulus]
#     # adj_mat_ds = np.load(os.path.join(directory, file))
#     # adj_mat_bl = np.load(os.path.join(directory, file.replace('.npz', '_bl.npz')))
#     adj_mat_ds = load_npz_3d(os.path.join(directory, file))
#     adj_mat_bl = load_npz_3d(os.path.join(directory, file.replace('.npz', '_bl.npz')))
#     adj_mat = np.zeros_like(adj_mat_ds)
#     total_len = len(list(itertools.combinations(range(adj_mat_ds.shape[0]), 2)))
#     cnt = 0
#     for ind, (row_a, row_b) in tqdm(enumerate(itertools.combinations(range(adj_mat_ds.shape[0]), 2)), total=total_len):
#       if SW_p[ind] > 0.05 and SW_p_bl[ind] > 0.05: # only keep positive edges
#         cnt += 1
#     print(cnt / total_len)
# %%
def save_adj_larger_2d(directory, sign, measure, alpha):
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
    if ('_bl' not in file) and ('_peak' not in file):
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

def save_adj_larger_3d(directory, sign, measure, alpha):
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

def save_ccg_corrected_n_fold(directory, measure, maxlag=12, n=7, disable=False):
  path = directory.replace(measure, measure+'_significant')
  if not os.path.exists(path):
    os.makedirs(path)
  files = os.listdir(directory)
  files.sort(key=lambda x:int(x[:9]))
  # adj_temp = load_npz_3d(os.path.join(directory, [f for f in files if not '_bl' in f][0]))
  # R = adj_temp.shape[2] # number of downsamples
  adj_bl_temp = load_npz_3d(os.path.join(directory, [f for f in files if '_bl' in f][0]))
  N = adj_bl_temp.shape[2] # number of shuffles
  for file in files:
    if '_bl' not in file:
      print(file)
      # adj_mat_ds = np.load(os.path.join(directory, file))
      # adj_mat_bl = np.load(os.path.join(directory, file.replace('.npy', '_bl.npy')))
      try: 
        xcorr = load_npz_3d(os.path.join(directory, file))
      except:
        xcorr = load_sparse_npz(os.path.join(directory, file))
      try:
        xcorr_jittered = load_npz_3d(os.path.join(directory, file.replace('.npz', '_bl.npz')))
      except:
        xcorr_jittered = load_sparse_npz(os.path.join(directory, file.replace('.npz', '_bl.npz')))
      num_nodes = xcorr.shape[0]
      significant_ccg, significant_peaks=np.zeros((num_nodes,num_nodes)), np.zeros((num_nodes,num_nodes))
      significant_ccg[:] = np.nan
      significant_peaks[:] = np.nan
      total_len = len(list(itertools.permutations(range(N), 2)))
      for row_a, row_b in tqdm(itertools.permutations(range(N), 2), total=total_len , miniters=int(total_len/100), disable=disable): # , miniters=int(total_len/100)
        ccg_corrected = (xcorr[row_a, row_b, :, None] - xcorr_jittered[row_a, row_b, :, :]).mean(-1)
        if ccg_corrected[:maxlag].max() > ccg_corrected.mean() + n * ccg_corrected.std():
        # if np.max(np.abs(corr))
          max_offset = np.argmax(ccg_corrected[:maxlag])
          significant_ccg[row_a, row_b] = ccg_corrected[:maxlag][max_offset]
          significant_peaks[row_a, row_b] = max_offset
      
      # np.save(os.path.join(path, file), adj_mat)
      save_npz(significant_ccg, os.path.join(path, file))
      save_npz(significant_peaks, os.path.join(path, file.replace('.npz', '_peak.npz')))

# start_time = time.time()
# # measure = 'pearson'
# measure = 'xcorr'
# alpha = 0.01
# # sign = 'pos'
# # sign = 'neg'
# sign = 'all'
# directory = './data/ecephys_cache_dir/sessions/adj_mat_{}_shuffled/'.format(measure)
# save_adj_larger_2d(directory, sign, measure, alpha)
# print("--- %s minutes in total" % ((time.time() - start_time)/60))

#%%
start_time = time.time()
measure = 'ccg'
n = 7
directory = './data/ecephys_cache_dir/sessions/adj_mat_{}_corrected/'.format(measure)
save_ccg_corrected_n_fold(directory, measure, maxlag=12, n=n)
print("--- %s minutes in total" % ((time.time() - start_time)/60))
# %%
# for file in files:
#   if '_bl' not in file and '719161530' in file:
#     print(file)
#     adj_mat_ds = np.load(os.path.join(directory, file))
#     adj_mat_bl = np.load(os.path.join(directory, file.replace('.npy', '_bl.npy')))
#     adj_mat = np.zeros_like(adj_mat_ds)
#     total_len = len(list(itertools.combinations(range(adj_mat_ds.shape[0]), 2)))
#     for row_a, row_b in tqdm(itertools.combinations(range(adj_mat_ds.shape[0]), 2), total=total_len):
#       sample = ws.DescrStatsW(adj_mat_ds[row_a, row_b, :])
#       shuffle = ws.DescrStatsW(adj_mat_bl[row_a, row_b, :])
#       cm_obj = ws.CompareMeans(sample, shuffle)
#       zstat, z_pval = cm_obj.ztest_ind(alternative='larger', usevar='unequal', value=0)
#       if z_pval < alpha:
#         adj_mat[row_a, row_b, :] = adj_mat_ds[row_a, row_b, :]
#     mouseID = file.split('_')[0]
#     stimulus_name = file.replace('.npy', '').replace(mouseID + '_', '')
#     if not mouseID in G_dict:
#       G_dict[mouseID] = {}
#     G_dict[mouseID][stimulus_name] = []
#     for i in range(adj_mat.shape[2]):
#       G_dict[mouseID][stimulus_name].append(generate_graph(adj_mat=adj_mat[:, :, i], cc=False, weight=weight))
