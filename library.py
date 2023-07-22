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
from scipy.stats import spearmanr
from scipy.stats import ttest_ind
from scipy.stats import ranksums
from scipy.stats import kstest
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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle, PathPatch
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib import colors
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from statsmodels.stats.multitest import multipletests, fdrcorrection
from statsmodels.stats.weightstats import ztest as ztest
import networkx as nx
import networkx.algorithms.community as nx_comm
import networkx.algorithms.isomorphism as iso
from networkx.algorithms import isomorphism
from collections import defaultdict, deque
from networkx import NetworkXError
from networkx.algorithms.community.community_utils import is_partition
from networkx.utils import py_random_state
import community
from netgraph import Graph
import seaborn as sns
from numpy.lib.stride_tricks import as_strided
from scipy.ndimage.filters import uniform_filter1d
from scipy import signal
from scipy import stats
from scipy import sparse
# from plfit import plfit
import collections
from collections import defaultdict
from sklearn.manifold import TSNE
from sklearn.metrics.cluster import adjusted_rand_score
import multiprocessing
from upsetplot import from_contents, plot, UpSet
import gurobipy as gp
from bisect import bisect
import holoviews as hv
from holoviews import opts, dim
import multiprocessing
from itertools import repeat
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
np.seterr(divide='ignore', invalid='ignore')

plt.rcParams['font.family'] = 'serif'
plt.rcParams["font.serif"] = ["Times New Roman"]

session_ids = ['719161530','750749662','754312389','755434585','756029989','791319847','797828357']
stimulus_names = ['spontaneous', 'flash_dark', 'flash_light',
        'drifting_gratings', 'static_gratings',
          'natural_scenes', 'natural_movie_one', 'natural_movie_three']
visual_regions = ['VISam', 'VISpm', 'VISal', 'VISrl', 'VISl', 'VISp']
# region_colors = ['tab:green', 'lightcoral', 'steelblue', 'tab:orange', 'tab:purple', 'grey']
region_labels = ['AM', 'PM', 'AL', 'RL', 'LM', 'V1']
region_colors = ['#d9e9b5', '#c0d8e9', '#fed3a1', '#c3c3c3', '#fad3e4', '#cec5f2']
combined_stimuli = [['spontaneous'], ['flash_dark', 'flash_light'], ['drifting_gratings'], ['static_gratings'], ['natural_scenes'], ['natural_movie_one', 'natural_movie_three']]
combined_stimulus_names = ['Resting\nstate', 'Flashes', 'Drifting\ngratings', 'Static\ngratings', 'Natural\nscenes', 'Natural\nmovies']
combined_stimulus_colors = ['#8dd3c7', '#fee391', '#bebada', '#bebada', '#fb8072', '#fb8072']
stimulus2marker = {'Resting\nstate':'s', 'Flashes':'*', 'Drifting\ngratings':'X', 'Static\ngratings':'P', 'Natural\nscenes':r'$\clubsuit$', 'Natural\nmovies':'>'}
marker_size_dict = {'v':10, '*':22, 'P':13, 'X':13, 'o':11, 's':9.5, 'D':9, 'p':12, '>':10, r'$\clubsuit$':20}
scatter_size_dict = {'v':10, '*':17, 'P':13, 'X':13, 'o':11, 's':10, 'D':9, 'p':13, '>':12, r'$\clubsuit$':16}
error_size_dict = {'v':10, '*':24, 'P':16, 'X':16, 'o':11, 's':9., 'D':9, 'p':12, '>':13, r'$\clubsuit$':22}

def combine_stimulus(stimulus):
  t_ind = [i for i in range(len(combined_stimuli)) if stimulus in combined_stimuli[i]][0]
  return t_ind, combined_stimulus_names[t_ind]

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

def load_sparse_npz(filename):
  with open(filename, 'rb') as infile:
    [sparse_matrix, shape] = pickle.load(infile)
    matrix_2d = sparse_matrix.toarray()
  return matrix_2d.reshape(shape)

############# load area_dicts
def load_area_dicts(session_ids):
  a_file = open('./files/area_dict.pkl', 'rb')
  area_dict = pickle.load(a_file)
  int_2_str = dict((session_id, str(session_id)) for session_id in session_ids)
  area_dict = dict((int_2_str[key], value) for (key, value) in area_dict.items())
  a_file.close()
  a_file = open('./files/active_area_dict.pkl', 'rb')
  active_area_dict = pickle.load(a_file)
  a_file.close()
  return area_dict, active_area_dict

# build a graph from an adjacency matrix
def mat2graph(adj_mat, confidence_level, active_area, cc=False, weight=False):
  if not weight:
    adj_mat[adj_mat.nonzero()] = 1
  G = nx.from_numpy_array(adj_mat, create_using=nx.DiGraph) # same as from_numpy_matrix
  node_idx = sorted(active_area.keys())
  mapping = {i:node_idx[i] for i in range(len(node_idx))}
  G = nx.relabel_nodes(G, mapping)
  assert set(G.nodes())==set(node_idx), '{}, {}'.format(len(G.nodes()), len(node_idx))
  nodes = sorted(G.nodes())
  cl = {(nodes[i],nodes[j]):confidence_level[i,j] for i,j in zip(*np.where(~np.isnan(confidence_level)))}
  nx.set_edge_attributes(G, cl, 'confidence')
  if cc: # extract the largest (strongly) connected components
    if np.allclose(adj_mat, adj_mat.T, rtol=1e-05, atol=1e-08): # if the matrix is symmetric, which means undirected graph
      largest_cc = max(nx.connected_components(G), key=len)
    else:
      largest_cc = max(nx.strongly_connected_components(G), key=len)
    G = nx.subgraph(G, largest_cc)
  return G

# load all graphs into a dictionary
def load_graphs(directory, active_area_dict, weight):
  G_dict, offset_dict, duration_dict = {}, {}, {}
  files = os.listdir(directory)
  files.sort(key=lambda x:int(x[:9]))
  for file in files:
    if ('_offset' not in file) and ('_duration' not in file) and ('confidence' not in file):
      print(file)
      adj_mat = load_npz_3d(os.path.join(directory, file))
      confidence_level = load_npz_3d(os.path.join(directory, file.replace('.npz', '_confidence.npz')))
      session = file.split('_')[0]
      stimulus = file.replace('.npz', '').replace(session + '_', '')
      if not session in G_dict:
        G_dict[session], offset_dict[session], duration_dict[session] = {}, {}, {}
      G_dict[session][stimulus] = mat2graph(adj_mat=np.nan_to_num(adj_mat), confidence_level=confidence_level, active_area=active_area_dict[session], cc=False, weight=weight)
      offset_dict[session][stimulus] = load_npz_3d(os.path.join(directory, file.replace('.npz', '_offset.npz')))
      duration_dict[session][stimulus] = load_npz_3d(os.path.join(directory, file.replace('.npz', '_duration.npz')))
  return G_dict, offset_dict, duration_dict

# get sorted sessions and stimuli
def get_session_stimulus(G_dict):
  sessions = list(G_dict.keys())
  stimuli = []
  for session in sessions:
    stimuli += list(G_dict[session].keys())
  stimuli = list(set(stimuli))
  if 'drifting_gratings_contrast' in stimuli:
    stimuli.remove('drifting_gratings_contrast')
  # sort stimulus
  stimulus_rank = ['spontaneous', 'flash_dark', 'flash_light',
        'drifting_gratings', 'static_gratings',
          'natural_scenes', 'natural_movie_one', 'natural_movie_three']
  if set(stimuli).issubset(set(stimulus_rank)):
    stimulus_rank_dict = {i:stimulus_rank.index(i) for i in stimuli}
    stimulus_rank_dict = dict(sorted(stimulus_rank_dict.items(), key=lambda item: item[1]))
    stimuli = list(stimulus_rank_dict.keys())
  else:
    stimuli = sorted(stimuli)
  return sessions, stimuli

def add_sign(G_dict):
  sessions, stimuli = get_session_stimulus(G_dict)
  S_dict = {}
  for session in sessions:
    S_dict[session] = {}
    for stimulus in stimuli:
      G = G_dict[session][stimulus]
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
      S_dict[session][stimulus] = S
  return S_dict

def add_offset(G_dict, offset_dict):
  sessions, stimuli = get_session_stimulus(G_dict)
  S_dict = {}
  for session in sessions:
    S_dict[session] = {}
    for stimulus in stimuli:
      offset_mat = offset_dict[session][stimulus]
      G = G_dict[session][stimulus]
      nodes = sorted(list(G.nodes()))
      offset = {}
      for edge in G.edges():
        offset[edge] = offset_mat[nodes.index(edge[0]), nodes.index(edge[1])]
      S = G.copy()
      nx.set_edge_attributes(S, offset, 'offset')
      S_dict[session][stimulus] = S
  return S_dict

def add_duration(G_dict, duration_dict):
  sessions, stimuli = get_session_stimulus(G_dict)
  S_dict = {}
  for session in sessions:
    S_dict[session] = {}
    for stimulus in stimuli:
      duration_mat = duration_dict[session][stimulus]
      G = G_dict[session][stimulus]
      nodes = sorted(list(G.nodes()))
      duration = {}
      for edge in G.edges():
        duration[edge] = duration_mat[nodes.index(edge[0]), nodes.index(edge[1])]
      S = G.copy()
      nx.set_edge_attributes(S, duration, 'duration')
      S_dict[session][stimulus] = S
  return S_dict

def add_delay(G_dict):
  sessions, stimuli = get_session_stimulus(G_dict)
  S_dict = {}
  for session in sessions:
    S_dict[session] = {}
    for stimulus in stimuli:
      G = G_dict[session][stimulus]
      delay = {}
      for edge in G.edges():
        delay[edge] = G.get_edge_data(*edge)['offset'] + G.get_edge_data(*edge)['duration']
      S = G.copy()
      nx.set_edge_attributes(S, delay, 'delay')
      S_dict[session][stimulus] = S
  return S_dict

# get rgb values with certain transparency level
def transparent_rgb(rgb, bg_rgb, alpha):
  return [alpha * c1 + (1 - alpha) * c2 for (c1, c2) in zip(rgb, bg_rgb)]

def get_lcc(G):
  if nx.is_directed(G):
    Gcc = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
  else:
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
  return G.subgraph(Gcc[0])

def calculate_directed_metric(G, metric_name):
  if metric_name == 'in_degree':
    metric = G.in_degree()
  elif metric_name == 'out_degree':
    metric = G.out_degree()
  elif metric_name == 'diameter':
    metric = nx.diameter(G)
  elif metric_name == 'radius':
    metric = nx.radius(G)
  elif metric_name == 'efficiency':
    metric = nx.global_efficiency(G)
  elif metric_name == 'clustering':
    metric = nx.average_clustering(G, weight=None)
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
  elif metric_name == 'assortativity_oi':
    metric = nx.degree_assortativity_coefficient(G, x='out', y='in', weight=None)
  elif metric_name == 'assortativity_io':
    metric = nx.degree_assortativity_coefficient(G, x='in', y='out', weight=None)
  elif metric_name == 'assortativity_ii':
    metric = nx.degree_assortativity_coefficient(G, x='in', y='in', weight=None)
  elif metric_name == 'assortativity_oo':
    metric = nx.degree_assortativity_coefficient(G, x='out', y='out', weight=None)
  elif metric_name == 'num_cycles':
    metric = len(list(nx.simple_cycles(G)))
  elif metric_name == 'flow_hierarchy':
    metric = nx.flow_hierarchy(G)
  elif metric_name == 'overall_reciprocity':
    metric = nx.overall_reciprocity(G)
  elif metric_name == 'average_shortest_path_length':
    metric = nx.average_shortest_path_length(get_lcc(G))
  elif metric_name == 'global_reaching_centrality':
    metric = nx.global_reaching_centrality(G)
  elif metric_name == 'wiener_index':
    metric = nx.wiener_index(G)
  elif metric_name == 'small-worldness':
    if not nx.is_connected(G):
      largest_cc = max(nx.connected_components(G), key=len)
      G = nx.subgraph(G, largest_cc)
    if nx.number_of_nodes(G) > 2 and nx.number_of_edges(G) > 2:
      metric = nx.sigma(G)
    else:
      metric = 0
  return metric
