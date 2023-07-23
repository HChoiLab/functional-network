import numpy as np
import pandas as pd
import os
import itertools
from scipy.stats import shapiro
from scipy.stats import ttest_ind
from scipy.stats import ranksums
from tqdm import tqdm
import pickle
import time
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle #, PathPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
import matplotlib.gridspec as gridspec
from statsmodels.stats.weightstats import ztest as ztest
import networkx as nx
import networkx.algorithms.isomorphism as iso
import community
from netgraph import Graph
import seaborn as sns
from scipy import signal
from scipy import stats
from collections import defaultdict
from sklearn.metrics.cluster import adjusted_rand_score
from upsetplot import from_contents, plot, UpSet
from bisect import bisect
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
TRIAD_NAMES = ('003', '012', '102', '021D', '021U', '021C', '111D', '111U', '030T', '030C', '201', '120D', '120U', '120C', '210', '300')
model_names = [u'Erdős-Rényi model', 'Degree-preserving model', 'Pair-preserving model', 'Signed-pair-preserving model']

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

################### Figure 1B ###################
def get_raster_data(area_dict, session_index, s_lengths, blank_width=50):
  directory = './files/spike_trains/'
  total_sequence = np.zeros((len(area_dict[session_ids[session_index]]), 0))
  stimulus2plot = [stimulus_names[i] for i in [0, 1, 3, 4, 5, 6]]
  for s_ind, stimulus in enumerate(stimulus2plot):
    print(stimulus)
    sequences = load_npz_3d(os.path.join(directory, str(session_ids[session_index]) + '_' + stimulus + '.npz'))
    assert sequences.shape[0] == len(area_dict[str(session_ids[session_index])])
    total_sequence = np.concatenate((total_sequence, sequences[:, 0, :s_lengths[s_ind]]), 1)
    if s_ind < len(stimulus2plot) - 1:
      total_sequence = np.concatenate((total_sequence, np.zeros((total_sequence.shape[0], blank_width))), 1)
  node_area = area_dict[str(session_ids[session_index])]
  nodes, areas = list(node_area.keys()), list(node_area.values())
  areas_num = [(np.array(areas)==a).sum() for a in visual_regions]
  areas_start_pos = list(np.insert(np.cumsum(areas_num)[:-1], 0, 0))
  sequence_by_area = {area:[nodes.index(ind) for ind, a in node_area.items() if a == area] for area in visual_regions}
  return total_sequence, areas_num, areas_start_pos, sequence_by_area

def plot_raster(total_sequence, areas_num, areas_start_pos, sequence_by_area, s_lengths, blank_width):
  sorted_sample_seq = np.vstack([total_sequence[sequence_by_area[area], :] for area in visual_regions])
  spike_pos = [np.nonzero(t)[0] / 1000 for t in sorted_sample_seq[:, :]] # divided by 1000 cuz bin size is 1 ms
  colors1 = [region_colors[i] for i in sum([[visual_regions.index(a)] * areas_num[visual_regions.index(a)] for a in visual_regions], [])]
  text_pos = [s + (areas_num[areas_start_pos.index(s)] - 1) / 2 for s in areas_start_pos]
  lineoffsets2 = 1
  linelengths2 = 2.
  # create a horizontal plot
  fig = plt.figure(figsize=(16.5, 6))
  plt.eventplot(spike_pos, colors='k', lineoffsets=lineoffsets2,
                      linelengths=linelengths2) # colors=colors1
  for ind, t_pos in enumerate(text_pos):
    plt.text(-.1, t_pos, region_labels[ind], size=20, color='k', va='center', ha='center') #, color=region_colors[ind]
  plt.axis('off')
  plt.gca().invert_yaxis()
  s_loc1 = np.concatenate(([0],(np.cumsum(s_lengths)+np.arange(1,len(s_lengths)+1)*blank_width)))[:-1]
  s_loc2 = s_loc1 + np.array(s_lengths)
  stext_pos = (s_loc1 + s_loc2) / 2000
  loc_max = max(s_loc1.max(), s_loc2.max())
  s_loc_frac1, s_loc_frac2 = [loc/loc_max for loc in s_loc1], [loc/loc_max for loc in s_loc2]
  for ind, t_pos in enumerate(stext_pos):
    plt.text(t_pos, -60, combined_stimulus_names[ind], size=20, color='k', va='center',ha='center')
  #### add horizontal band
  band_pos = areas_start_pos + [areas_start_pos[-1]+areas_num[-1]]
  xgmin, xgmax=.045, .955
  alpha_list = [.4, .4, .4, .6, .5, .5]
  for loc1, loc2 in zip(band_pos[:-1], band_pos[1:]):
    for scale1, scale2 in zip(s_loc_frac1, s_loc_frac2):
      xmin, xmax = scale1 * (xgmax-xgmin) + xgmin, scale2 * (xgmax-xgmin) + xgmin
      plt.gca().axhspan(loc1, loc2, xmin=xmin, xmax=xmax, facecolor=region_colors[areas_start_pos.index(loc1)], alpha=alpha_list[areas_start_pos.index(loc1)])
  plt.savefig('./figures/figure1B.pdf', transparent=True)

################### Figure 1D ###################
def find_peak_zscore(corr,duration=6,maxlag=12):
  filter = np.array([[[1]]]).repeat(duration+1, axis=2) # sum instead of mean
  corr_integral = signal.convolve(corr, filter, mode='valid', method='fft')
  mu, sigma = np.nanmean(corr_integral, -1), np.nanstd(corr_integral, -1)
  abs_deviation = np.abs(corr_integral[:, :, :maxlag-duration+1] - mu[:,:,None])
  extreme_offset = np.argmax(abs_deviation, -1)
  ccg_mat_extreme = np.choose(extreme_offset, np.moveaxis(corr_integral[:, :, :maxlag-duration+1], -1, 0))
  zscore = (ccg_mat_extreme - mu) / sigma
  return zscore, ccg_mat_extreme

def ccg2zscore(ccg_corrected, max_duration=6, maxlag=12):
  all_zscore, all_ccg = np.zeros(ccg_corrected.shape[:2]), np.zeros(ccg_corrected.shape[:2])
  for duration in np.arange(max_duration, -1, -1): # reverse order, so that sharp peaks can override highland
    print('duration {}'.format(duration))
    zscore, ccg_mat_extreme = find_peak_zscore(ccg_corrected, duration, maxlag)
    indx = np.abs(zscore) > np.abs(all_zscore)
    # highland_ccg, confidence_level, offset, indx = find_highland(corr, min_spike, duration, maxlag, n)
    if np.sum(indx):
      all_zscore[indx] = zscore[indx]
      all_ccg[indx] = ccg_mat_extreme[indx]
  return all_zscore, all_ccg

def get_connectivity_data(G_dict, session_ind, stimulus_ind):
  sessions, stimuli = get_session_stimulus(G_dict)
  session, stimulus = sessions[session_ind], stimuli[stimulus_ind]
  directory = './data/ecephys_cache_dir/sessions/adj_mat_ccg_corrected/'
  file = session + '_' + stimulus + '.npz'
  ccg = load_sparse_npz(os.path.join(directory, file))
  ccg_jittered = load_sparse_npz(os.path.join(directory, file.replace('.npz', '_bl.npz')))
  ccg_corrected = ccg - ccg_jittered
  ccg_zscore, ccg_value = ccg2zscore(ccg_corrected, max_duration=11, maxlag=12)
  return ccg_zscore, ccg_value

def plot_connectivity_matrix_annotation(G_dict, active_area_dict, session_ind, stimulus_ind, ccg_zscore, ccg_value, weight=None, ratio=15):
  sessions, stimuli = get_session_stimulus(G_dict)
  session, stimulus = sessions[session_ind], stimuli[stimulus_ind]
  G = G_dict[session][stimulus]
  nsession = 2
  nstimulus = 2
  active_area = active_area_dict[session]
  ordered_nodes = [] # order nodes based on hierarchical order
  region_size = np.zeros(len(visual_regions))
  for r_ind, region in enumerate(visual_regions):
    for node in active_area:
      if active_area[node] == region:
        ordered_nodes.append(node)
        region_size[r_ind] += 1
  A = nx.to_numpy_array(G, nodelist=ordered_nodes, weight='weight').T # source on the bottom, target on the left
  areas = [active_area[node] for node in ordered_nodes]
  areas_num = [(np.array(areas)==a).sum() for a in visual_regions]
  rareas_num = [(np.array(areas)==a).sum() for a in visual_regions[::-1]]
  area_inds = [0] + np.cumsum(areas_num).tolist()
  r_area_inds = ([0]+np.cumsum(rareas_num)[:-1].tolist())[::-1] # low to high from left to right
  vareas_start_pos = list(np.insert(np.cumsum(areas_num)[:-1], 0, 0))
  vtext_pos = [s + (areas_num[vareas_start_pos.index(s)] - 1) / 2 for s in vareas_start_pos]
  hareas_start_pos = list(np.insert(np.cumsum(rareas_num)[:-1], 0, 0))
  htext_pos = [s + (rareas_num[hareas_start_pos.index(s)] - 1) / 2 for s in hareas_start_pos]
  region_bar = []
  for r_ind in range(len(visual_regions)):
    region_bar += [r_ind] * int(region_size[r_ind])
  cmap = colors.ListedColormap(region_colors)
  bounds = [-.5,.5,1.5,2.5,3.5,4.5,5.5]
  norm = colors.BoundaryNorm(bounds, cmap.N)
  alpha_list = [.6, .6, .6, .6, .6, .6]
  colors_transparency = colors.ListedColormap([transparent_rgb(colors.to_rgb(color), [1,1,1], alpha=alpha_list[c_ind]) for c_ind, color in enumerate(region_colors)])
  fig = plt.figure(figsize=(10, 10)) 
  gs = gridspec.GridSpec(nsession, nstimulus, width_ratios=[1, ratio-1], height_ratios=[1, ratio-1],
          wspace=0.0, hspace=0.0, top=1, bottom=0.001, left=0., right=.999)
  ax= plt.subplot(gs[0,1])
  ax.set_xticklabels([])
  ax.set_yticklabels([])
  # reverse order, low to high from left to right
  ax.imshow(np.repeat(np.array(region_bar[::-1])[None,:],len(region_bar)//ratio, 0), cmap=colors_transparency, norm=norm)
  ax.set_xticks([])
  ax.set_yticks([])
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  ax.spines['left'].set_visible(False)

  for ind, t_pos in enumerate(htext_pos):
    ax.text(t_pos, 6.7, region_labels[len(region_labels)-ind-1], va='center', ha='center', size=30, color='k')
  ax= plt.subplot(gs[1,0])
  ax.set_xticklabels([])
  ax.set_yticklabels([])
  ax.imshow(np.repeat(np.array(region_bar)[:,None],len(region_bar)//ratio, 1), cmap=colors_transparency, norm=norm)
  ax.set_xticks([])
  ax.set_yticks([])
  for ind, t_pos in enumerate(vtext_pos):
    ax.text(6.7, t_pos, region_labels[ind], va='center', ha='center', size=30, color='k', rotation=90)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  ax.spines['left'].set_visible(False)

  ax= plt.subplot(gs[1,1])
  ax.set_xticklabels([])
  ax.set_yticklabels([])
  nodes = sorted(active_area.keys())
  node2idx = {node:nodes.index(node) for node in nodes}
  if weight is None:
    mat = np.flip(A, 1) # from left to right
    vmin = np.percentile(mat[mat<0], 20)
    vmax = np.percentile(mat[mat>0], 85)
  elif weight=='confidence':
    mat = ccg_zscore
    reordered_nodes = np.array([node2idx[node] for node in ordered_nodes])
    mat = mat[reordered_nodes[:,None], reordered_nodes]
    mat = np.flip(mat, 1) # from left to right
    vmin = np.percentile(mat[mat<0], .5)
    vmax = np.percentile(mat[mat>0], 99.2)
  elif weight=='weight':
    mat = ccg_value
    reordered_nodes = np.array([node2idx[node] for node in ordered_nodes])
    mat = mat[reordered_nodes[:,None], reordered_nodes]
    mat = np.flip(mat, 1) # from left to right
    vmin = np.percentile(mat[mat<0], 2)
    vmax = np.percentile(mat[mat>0], 98)
  cmap = plt.cm.coolwarm
  cmap = plt.cm.Spectral
  cmap = plt.cm.RdBu_r
  norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
  im = ax.imshow(mat, cmap=cmap, norm=norm)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  ax.spines['left'].set_visible(False)
  if weight is not None:
    ec = '.4'
  else:
    ec = '.8'
  for region_ind in range(len(visual_regions)):
    ax.add_patch(Rectangle((r_area_inds[region_ind],area_inds[region_ind]),areas_num[region_ind]-1,areas_num[region_ind]-1,linewidth=5,edgecolor=ec,alpha=.6,facecolor='none')) # region_colors[region_ind]
  ax.set_xticks([])
  ax.set_yticks([])
  figname = './figures/figure1D_left.pdf' if weight is not None else './figures/figure1D_right.pdf'
  plt.savefig(figname.format(session, stimulus), transparent=True)

################### Figure 1E ###################
def plot_new_ex_in_bar(G_dict, density=False):
  df = pd.DataFrame()
  sessions, stimuli = get_session_stimulus(G_dict)
  for stimulus_ind, stimulus in enumerate(stimuli):
    print(stimulus)
    combined_stimulus_name = combine_stimulus(stimulus)[1]
    ex_data, in_data = [], []
    for session_ind, session in enumerate(sessions):
      G = G_dict[session][stimulus] if stimulus in G_dict[session] else nx.Graph()
      signs = list(nx.get_edge_attributes(G, "sign").values())
      num = G.number_of_nodes()
      if density:
        ex_data.append(signs.count(1) / (num * (num-1)))
        in_data.append(signs.count(-1) / (num * (num-1)))
      else:
        ex_data.append(signs.count(1))
        in_data.append(signs.count(-1))
    df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(ex_data)[:,None], np.array(['excitatory'] * len(ex_data))[:,None], np.array([combined_stimulus_name] * len(ex_data))[:,None]), 1), columns=['number of connections', 'type', 'stimulus']), 
                pd.DataFrame(np.concatenate((np.array(in_data)[:,None], np.array(['inhibitory'] * len(in_data))[:,None], np.array([combined_stimulus_name] * len(in_data))[:,None]), 1), columns=['number of connections', 'type', 'stimulus'])], ignore_index=True)
  df['number of connections'] = pd.to_numeric(df['number of connections'])
  if density:
    y = 'density'
    df['density'] = df['number of connections']
  else:
    y = 'number of connections'
  fig = plt.figure(figsize=(8, 5))
  barcolors = ['firebrick', 'navy']
  ax = sns.barplot(x="stimulus", y=y, hue="type",  data=df, palette=barcolors, errorbar="sd",  edgecolor="black", errcolor="black", errwidth=1.5, capsize = 0.1, alpha=0.5) #, width=.6
  sns.stripplot(x="stimulus", y=y, hue="type", palette=barcolors, data=df, dodge=True, alpha=0.6, ax=ax)
  # remove extra legend handles
  handles, labels = ax.get_legend_handles_labels()
  ax.legend(handles[2:], labels[2:], title='', bbox_to_anchor=(.7, 1.), loc='upper left', fontsize=28, frameon=False)
  plt.yticks(fontsize=25) #,  weight='bold'
  plt.ylabel(y.capitalize())
  ax.set_ylabel(ax.get_ylabel(), fontsize=30,color='k') #, weight='bold'
  for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2.5)
    ax.spines[axis].set_color('k')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.tick_params(width=2.5)
  ax.set(xlabel=None)
  plt.xticks([])
  plt.tight_layout()
  figname = './figures/figure1E.pdf'
  plt.savefig(figname, transparent=True)

################### Figure 1F ###################
def scatter_dataVSdensity(G_dict, area_dict, regions, name='intra'):
  sessions, stimuli = get_session_stimulus(G_dict)
  fig, ax = plt.subplots(figsize=(5, 5))
  X, Y = [], []
  df = pd.DataFrame()
  region_connection = np.zeros((len(sessions), len(stimuli), len(regions), len(regions)))
  for stimulus_ind, stimulus in enumerate(stimuli):
    intra_data, inter_data, density_data, ex_data, in_data, cluster_data = [], [], [], [], [], []
    for session_ind, session in enumerate(sessions):
      G = G_dict[session][stimulus]
      nodes = list(G.nodes())
      node_area = {key: area_dict[session][key] for key in nodes}
      A = nx.to_numpy_array(G)
      A[A.nonzero()] = 1
      for region_ind_i, region_i in enumerate(regions):
        for region_ind_j, region_j in enumerate(regions):
          region_indices_i = np.array([k for k, v in area_dict[session].items() if v==region_i])
          region_indices_j = np.array([k for k, v in area_dict[session].items() if v==region_j])
          region_indices_i = np.array([nodes.index(i) for i in list(set(region_indices_i) & set(nodes))]) # some nodes not in cc are removed 
          region_indices_j = np.array([nodes.index(i) for i in list(set(region_indices_j) & set(nodes))])
          if len(region_indices_i) and len(region_indices_j):
            region_connection[session_ind, stimulus_ind, region_ind_i, region_ind_j] = np.sum(A[region_indices_i[:, None], region_indices_j])
            assert np.sum(A[region_indices_i[:, None], region_indices_j]) == len(A[region_indices_i[:, None], region_indices_j].nonzero()[0])
      diag_indx = np.eye(len(regions),dtype=bool)
      intra_data.append(np.sum(region_connection[session_ind, stimulus_ind][diag_indx])/np.sum(region_connection[session_ind, stimulus_ind]))
      inter_data.append(np.sum(region_connection[session_ind, stimulus_ind][~diag_indx])/np.sum(region_connection[session_ind, stimulus_ind]))
      density_data.append(nx.density(G))
      signs = list(nx.get_edge_attributes(G, "sign").values())
      ex_data.append(signs.count(1) / len(signs))
      in_data.append(signs.count(-1) / len(signs))
      cluster_data.append(calculate_directed_metric(G, 'clustering'))
    X += density_data
    if name == 'intra':
      Y += intra_data
    elif name == 'ex':
      Y += ex_data
    elif name == 'cluster':
      Y += cluster_data
    df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(intra_data)[:,None], np.array(inter_data)[:,None], np.array(ex_data)[:,None], np.array(cluster_data)[:,None], np.array(density_data)[:,None], np.array([combine_stimulus(stimulus)[1]] * len(intra_data))[:,None]), 1), columns=['ratio of intra-region connections', 'ratio of inter-region connections', 'ratio of excitatory connections', 'cluster', 'density', 'stimulus'])], ignore_index=True)
  df['ratio of intra-region connections'] = pd.to_numeric(df['ratio of intra-region connections'])
  df['ratio of inter-region connections'] = pd.to_numeric(df['ratio of inter-region connections'])
  df['ratio of excitatory connections'] = pd.to_numeric(df['ratio of excitatory connections'])
  df['cluster'] = pd.to_numeric(df['cluster'])
  df['density'] = pd.to_numeric(df['density'])
  for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
    x = df[df['stimulus']==combined_stimulus_name]['density'].values
    if name == 'intra':
      y = df[df['stimulus']==combined_stimulus_name]['ratio of intra-region connections'].values
    elif name == 'ex':
      y = df[df['stimulus']==combined_stimulus_name]['ratio of excitatory connections'].values
    elif name == 'cluster':
      y = df[df['stimulus']==combined_stimulus_name]['cluster'].values
    ax.scatter(x, y, ec='.1', fc='none', marker=stimulus2marker[combined_stimulus_name], s=10*marker_size_dict[stimulus2marker[combined_stimulus_name]], alpha=.9, linewidths=1.5)
  X, Y = (list(t) for t in zip(*sorted(zip(X, Y))))
  X, Y = np.array(X), np.array(Y)
  if name in ['intra']:
    slope, intercept, r_value, p_value, std_err = stats.linregress(X,Y)
    line = slope*X+intercept
    locx, locy = .8, .9
    text = 'r={:.2f}, p={:.2f}'.format(r_value, p_value)
  elif name in ['ex', 'cluster']:
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(X),Y)
    line = slope*np.log10(X)+intercept
    locx, locy = .4, 1.
    text = 'r={:.2f}, p={:.1e}'.format(r_value, p_value)
  ax.plot(X, line, color='.2', linestyle=(5,(10,3)), alpha=.5)
  # ax.plot(X, line, color='.4', linestyle='-', alpha=.5) # (5,(10,3))
  # ax.scatter(X, Y, facecolors='none', edgecolors='.2', alpha=.6)
  ax.text(locx, locy, text, horizontalalignment='center',
     verticalalignment='center', transform=ax.transAxes, fontsize=22)
  plt.xticks(fontsize=22) #, weight='bold'
  plt.yticks(fontsize=22) # , weight='bold'
  plt.xlabel('Density')
  if name == 'intra':
    ylabel = 'Within-area fraction'
  elif name == 'ex':
    ylabel = 'Excitatory fraction'
    plt.xscale('log')
  elif name == 'cluster':
    ylabel = 'Clustering coefficient'
    plt.xscale('log')
  plt.ylabel(ylabel)
  ax.set_xlabel(ax.get_xlabel(), fontsize=28,color='k') #, weight='bold'
  ax.set_ylabel(ax.get_ylabel(), fontsize=28,color='k') #, weight='bold'
  for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(1.5)
    ax.spines[axis].set_color('0.2')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.tick_params(width=1.5)
  plt.tight_layout()
  # plt.show()
  if name == 'intra':
    fname = 'left'
  elif name == 'ex':
    fname = 'middle'
  elif name == 'cluster':
    fname = 'right'
  plt.savefig(f'./figures/figure1F_{fname}.pdf', transparent=True)

################### Figure 1G ###################
def get_pos_neg_p_signalcorr(G_dict, active_area_dict, signal_correlation_dict, pairtype='all'):
  sessions = signal_correlation_dict.keys()
  pos_connect_dict, neg_connect_dict, dis_connect_dict, signal_corr_dict = [{session:{csn:[] for csn in combined_stimulus_names[1:]} for session in sessions} for _ in range(4)]
  for session_ind, session in enumerate(sessions):
    print(session)
    active_area = active_area_dict[session]
    node_idx = sorted(active_area.keys())
    for combined_stimulus_name in combined_stimulus_names[2:]: # exclude spontaneous and flashes in signal correlation analysis
      cs_ind = combined_stimulus_names.index(combined_stimulus_name)
      signal_correlation = signal_correlation_dict[session][combined_stimulus_name]
      pos_connect, neg_connect, dis_connect, signal_corr = [], [], [], []
      for stimulus in combined_stimuli[cs_ind]:
        G = G_dict[session][stimulus].copy()
        nodes = sorted(G.nodes())
        for nodei, nodej in itertools.combinations(node_idx, 2):
          scorr = signal_correlation[nodes.index(nodei), nodes.index(nodej)] # abs(signal_correlation[nodei, nodej])
          if not np.isnan(scorr):
            if G.has_edge(nodei, nodej):
              signal_corr.append(scorr)
              w = G[nodei][nodej]['weight']
              if w > 0:
                pos_connect.append(1)
                neg_connect.append(0)
              elif w < 0:
                pos_connect.append(0)
                neg_connect.append(1)
            if G.has_edge(nodej, nodei):
              signal_corr.append(scorr)
              w = G[nodej][nodei]['weight']
              if w > 0:
                pos_connect.append(1)
                neg_connect.append(0)
              elif w < 0:
                pos_connect.append(0)
                neg_connect.append(1)
            if pairtype == 'all':
              if (not G.has_edge(nodei, nodej)) and (not G.has_edge(nodej, nodei)):
                dis_connect.append(scorr)
                signal_corr.append(scorr)
                pos_connect.append(0)
                neg_connect.append(0)

      signal_corr_dict[session][combined_stimulus_name] += signal_corr
      pos_connect_dict[session][combined_stimulus_name] += pos_connect
      neg_connect_dict[session][combined_stimulus_name] += neg_connect
      dis_connect_dict[session][combined_stimulus_name] += dis_connect

  pos_df, neg_df, dis_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
  for combined_stimulus_name in combined_stimulus_names[1:]:
    print(combined_stimulus_name)
    for session in sessions:
      pos_connect, neg_connect, dis_connect, signal_corr = pos_connect_dict[session][combined_stimulus_name], neg_connect_dict[session][combined_stimulus_name], dis_connect_dict[session][combined_stimulus_name], signal_corr_dict[session][combined_stimulus_name]
      # within_comm, across_comm = [e for e in within_comm if not np.isnan(e)], [e for e in across_comm if not np.isnan(e)] # remove nan values
      pos_df = pd.concat([pos_df, pd.DataFrame(np.concatenate((np.array(signal_corr)[:,None], np.array(pos_connect)[:,None], np.array([combined_stimulus_name] * len(pos_connect))[:,None], np.array([session] * len(pos_connect))[:,None]), 1), columns=['signal correlation', 'type', 'stimulus', 'session'])], ignore_index=True)
      neg_df = pd.concat([neg_df, pd.DataFrame(np.concatenate((np.array(signal_corr)[:,None], np.array(neg_connect)[:,None], np.array([combined_stimulus_name] * len(neg_connect))[:,None], np.array([session] * len(neg_connect))[:,None]), 1), columns=['signal correlation', 'type', 'stimulus', 'session'])], ignore_index=True)
      dis_df = pd.concat([dis_df, pd.DataFrame(np.concatenate((np.array(dis_connect)[:,None], np.array([combined_stimulus_name] * len(dis_connect))[:,None], np.array([session] * len(dis_connect))[:,None]), 1), columns=['signal correlation', 'stimulus', 'session'])], ignore_index=True)
  pos_df['signal correlation'] = pd.to_numeric(pos_df['signal correlation'])
  pos_df['type'] = pd.to_numeric(pos_df['type'])
  neg_df['signal correlation'] = pd.to_numeric(neg_df['signal correlation'])
  neg_df['type'] = pd.to_numeric(neg_df['type'])
  dis_df['signal correlation'] = pd.to_numeric(dis_df['signal correlation'])
  return pos_df, neg_df, dis_df

def plot_pos_neg_signal_correlation_distri(pos_df, neg_df, dis_df):
  fig, axes = plt.subplots(1,len(combined_stimulus_names)-2, figsize=(5*(len(combined_stimulus_names)-2), 3), sharex=True)
  for cs_ind in range(len(axes)):
    ax = axes[cs_ind]
    pos_data = pos_df[(pos_df.stimulus==combined_stimulus_names[cs_ind+2]) & (pos_df.type==1)].copy() #  & (df.session==session)
    neg_data = neg_df[(neg_df.stimulus==combined_stimulus_names[cs_ind+2]) & (neg_df.type==1)].copy() #  & (df.session==session)
    dis_data = dis_df[dis_df.stimulus==combined_stimulus_names[cs_ind+2]].copy()
    pos_x, neg_x, dis_x = pos_data['signal correlation'].values.flatten(), neg_data['signal correlation'].values.flatten(), dis_data['signal correlation'].values.flatten()
    df = pd.DataFrame()
    df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(pos_x)[:,None], np.array(['excitatory'] * len(pos_x))[:,None]), 1), columns=['signal correlation', 'type'])], ignore_index=True)
    df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(neg_x)[:,None], np.array(['inhibitory'] * len(neg_x))[:,None]), 1), columns=['signal correlation', 'type'])], ignore_index=True)
    df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(dis_x)[:,None], np.array(['disconnected'] * len(dis_x))[:,None]), 1), columns=['signal correlation', 'type'])], ignore_index=True)
    df['signal correlation'] = pd.to_numeric(df['signal correlation'])
    # sns.histplot(data=df, x='signal correlation', hue='type', stat='probability', common_norm=False, ax=ax, palette=['r', 'b', 'grey'], alpha=0.4)
    sns.kdeplot(data=df, x='signal correlation', hue='type', common_norm=False, ax=ax, palette=['r', 'b', 'grey'], alpha=.8)
    
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)
    for axis in ['bottom', 'left']:
      ax.spines[axis].set_linewidth(2.5)
      ax.spines[axis].set_color('k')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=2.5)
    ax.set_xlabel([], fontsize=0)
    if cs_ind == 0:
      ax.set_ylabel('KDE', fontsize=30)
    else:
      ax.set_ylabel([], fontsize=0)
    # ax.set_xlabel('Signal correlation', fontsize=25)
    # handles, labels = ax.get_legend_handles_labels()
    ax.legend().set_visible(False)

  plt.tight_layout(rect=[.01, 0, 1, 1])
  plt.savefig('./figures/figure1G.pdf', transparent=True)

################### Figure 2A ###################
def safe_division(n, d):
    return n / d if d else 0

def count_signed_triplet_connection_p(G):
  num0, num1, num2, num3, num4, num5 = 0, 0, 0, 0, 0, 0
  nodes = list(G.nodes())
  edge_sign = nx.get_edge_attributes(G,'sign')
  for node_i in range(len(nodes)):
    for node_j in range(len(nodes)):
      if node_i != node_j:
        edge_sum = edge_sign.get((nodes[node_i], nodes[node_j]), 0) + edge_sign.get((nodes[node_j], nodes[node_i]), 0)
        if edge_sum == 0:
          if G.has_edge(nodes[node_i], nodes[node_j]) and G.has_edge(nodes[node_j], nodes[node_i]):
            num4 += 1
          else:
            num0 += 1
        elif edge_sum == 1:
          num1 += 1
        elif edge_sum == 2:
          num3 += 1
        elif edge_sum == -1:
          num2 += 1
        elif edge_sum == -2:
          num5 += 1

  total_num = num0+num1+num2+num3+num4+num5
  assert total_num == len(nodes) * (len(nodes) - 1)
  assert (num1+num2)/2 + num3+num4+num5 == G.number_of_edges()
  p0, p1, p2, p3, p4, p5 = safe_division(num0, total_num), safe_division(num1, total_num), safe_division(num2, total_num), safe_division(num3, total_num), safe_division(num4, total_num), safe_division(num5, total_num)
  return p0, p1, p2, p3, p4, p5

def plot_signed_pair_relative_count(G_dict, p_signed_pair_func, log=False):
  sessions, stimuli = get_session_stimulus(G_dict)
  fig, axes = plt.subplots(len(combined_stimulus_names), 1, figsize=(8, 1.*len(combined_stimulus_names)), sharex=True, sharey=True)
  for cs_ind, stimulus_name in enumerate(combined_stimulus_names):
    ax = axes[len(axes)-1-cs_ind] # spontaneous in the bottom
    all_pair_count = defaultdict(lambda: [])
    for stimulus in combined_stimuli[cs_ind]:
      for session in sessions:
        G = G_dict[session][stimulus].copy()
        signs = list(nx.get_edge_attributes(G, "sign").values())
        p_pos, p_neg = signs.count(1)/(G.number_of_nodes()*(G.number_of_nodes()-1)), signs.count(-1)/(G.number_of_nodes()*(G.number_of_nodes()-1))
        p0, p1, p2, p3, p4, p5 = count_signed_triplet_connection_p(G)
        all_pair_count['0'].append(p0 / p_signed_pair_func['0'](p_pos, p_neg))
        all_pair_count['+'].append(p1 / p_signed_pair_func['1'](p_pos, p_neg))
        all_pair_count['-'].append(p2 / p_signed_pair_func['2'](p_pos, p_neg))
        all_pair_count['++'].append(p3 / p_signed_pair_func['3'](p_pos, p_neg))
        all_pair_count['+-'].append(p4 / p_signed_pair_func['4'](p_pos, p_neg))
        all_pair_count['--'].append(p5 / p_signed_pair_func['5'](p_pos, p_neg))
    # ax.set_ylabel('Relative count')
    triad_types, triad_counts = [k for k,v in all_pair_count.items()], [v for k,v in all_pair_count.items()]
    box_color = '.2'
    boxprops = dict(color=box_color,linewidth=1.5)
    medianprops = dict(color=box_color,linewidth=1.5)
    box_plot = ax.boxplot(triad_counts, showfliers=False, patch_artist=True, boxprops=boxprops,medianprops=medianprops)
    for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
      plt.setp(box_plot[item], color=box_color)
    for patch in box_plot['boxes']:
      patch.set_facecolor('none')
      ax.set_xticks([])
    left, right = plt.xlim()
    ax.hlines(1, xmin=left, xmax=right, color='.5', alpha=.6, linestyles='--', linewidth=2)
    if log:
      ax.set_yscale('log')
    ax.yaxis.set_tick_params(labelsize=18)
    ax.set_ylabel('', fontsize=20,color='k') #, weight='bold'
    for axis in ['bottom', 'left']:
      ax.spines[axis].set_linewidth(1.)
      ax.spines[axis].set_color('k')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=1.)
    relative_x = 0.1
    relative_y = 0.9
    ax.text(relative_x, relative_y, stimulus_name.replace('\n', ' '), transform=ax.transAxes,
            ha='left', va='center', fontsize=20, color='black')
  fig.subplots_adjust(hspace=.7) #wspace=0.2
  plt.savefig('./figures/figure2A.pdf', transparent=True)

p_signed_pair_func = {
  '0': lambda p_pos, p_neg: (1 - p_pos - p_neg)**2,
  '1': lambda p_pos, p_neg: 2 * p_pos * (1 - p_pos - p_neg),
  '2': lambda p_pos, p_neg: 2 * p_neg * (1 - p_pos - p_neg),
  '3': lambda p_pos, p_neg: p_pos ** 2,
  '4': lambda p_pos, p_neg: 2 * p_pos * p_neg,
  '5': lambda p_pos, p_neg: p_neg ** 2,
}

################## find significant signed motifs using z score for motif intensity and coherence
################## first Z score, then average
def get_intensity_zscore(intensity_dict, coherence_dict, baseline_intensity_dict, baseline_coherence_dict, num_baseline=100):
  sessions, stimuli = get_session_stimulus(intensity_dict)
  signed_motif_types = set()
  for session in sessions:
    for stimulus in stimuli:
      signed_motif_types = signed_motif_types.union(set(list(intensity_dict[session][stimulus].keys())).union(set(list(baseline_intensity_dict[session][stimulus].keys()))))
  signed_motif_types = list(signed_motif_types)
  pseudo_intensity = np.zeros(num_baseline)
  pseudo_intensity[0] = 5 # if a motif is not found in random graphs, assume it appeared once
  whole_df = pd.DataFrame()
  for stimulus in stimuli:
    for session in sessions:
      motif_list = []
      for motif_type in signed_motif_types:
        motif_list.append([motif_type, session, stimulus, intensity_dict[session][stimulus].get(motif_type, 0), baseline_intensity_dict[session][stimulus].get(motif_type, pseudo_intensity).mean(), 
                        baseline_intensity_dict[session][stimulus].get(motif_type, pseudo_intensity).std(), coherence_dict[session][stimulus].get(motif_type, 0), 
                        baseline_coherence_dict[session][stimulus].get(motif_type, np.zeros(10)).mean(), baseline_coherence_dict[session][stimulus].get(motif_type, np.zeros(10)).std()])
      df = pd.DataFrame(motif_list, columns =['signed motif type', 'session', 'stimulus', 'intensity', 'intensity mean', 'intensity std', 'coherence', 'coherence mean', 'coherence std']) 
      whole_df = pd.concat([whole_df, df], ignore_index=True, sort=False)
  whole_df['intensity z score'] = (whole_df['intensity']-whole_df['intensity mean'])/whole_df['intensity std']
  whole_df['coherence z score'] = (whole_df['coherence']-whole_df['coherence mean'])/whole_df['coherence std']
  mean_df = whole_df.groupby(['stimulus', 'signed motif type'], as_index=False).agg('mean') # average over session
  std_df = whole_df.groupby(['stimulus', 'signed motif type'], as_index=False).agg('std')
  mean_df['intensity z score std'] = std_df['intensity z score']
  return whole_df, mean_df, signed_motif_types

################### Figure 2C ###################
def plot_zscore_allmotif_lollipop(df, model_name):
  fig, axes = plt.subplots(len(combined_stimulus_names),1, sharex=True, sharey=True, figsize=(50, 3*len(combined_stimulus_names)))
  sorted_types = [sorted([smotif for smotif in df['signed motif type'].unique() if mt in smotif]) for mt in TRIAD_NAMES]
  sorted_types = [item for sublist in sorted_types for item in sublist]
  motif_types = TRIAD_NAMES[3:]
  motif_loc = [np.mean([i for i in range(len(sorted_types)) if mt in sorted_types[i]]) for mt in motif_types]
  palette = [[plt.cm.tab20b(i) for i in range(20)][i] for i in [0,2,3,4,6,8,10,12,16,18,19]] + [[plt.cm.tab20c(i) for i in range(20)][i] for i in [4,16]]
  for s_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
    print(combined_stimulus_name)
    data = df[df.apply(lambda x: combine_stimulus(x['stimulus'])[1], axis=1)==combined_stimulus_name]
    data = data.groupby('signed motif type').mean()
    ax = axes[len(axes)-1-s_ind] # spontaneous in the bottom
    # ax.set_title(combined_stimulus_names[s_ind].replace('\n', ' '), fontsize=35, rotation=0)
    for t, y in zip(sorted_types, data.loc[sorted_types, "intensity z score"]):
      color = palette[motif_types.index(t.replace('+', '').replace('\N{MINUS SIGN}', ''))]
      ax.plot([t,t], [0,y], color=color, marker="o", linewidth=7, markersize=20, markevery=(1,2))
    ax.set_xlim(-.5,len(sorted_types)+.5)
    ax.set_xticks([])
    ax.yaxis.set_tick_params(labelsize=45)
    for axis in ['bottom', 'left']:
      ax.spines[axis].set_linewidth(4.5)
      ax.spines[axis].set_color('k')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=4.5)
    ax.xaxis.set_tick_params(length=0)
    ax.set_ylabel('')
    # ax.set_ylabel('Z score', fontsize=40)
    if model_names.index(model_name) <= 1:
      ax.set_yscale('symlog')
    else:
      ax.set_ylim(-13, 21)
  plt.tight_layout()
  plt.savefig('./figures/figure2C.pdf', transparent=True)

################### Figure 3A ###################
def scatter_ZscoreVSdensity(origin_df, G_dict):
  df = origin_df.copy()
  df['density'] = 0
  sessions, stimuli = get_session_stimulus(G_dict)
  fig, ax = plt.subplots(figsize=(5, 5))
  for session_ind, session in enumerate(sessions):
    for stimulus in stimuli:
      G = G_dict[session][stimulus]
      df.loc[(df['session']==session) & (df['stimulus']==stimulus), 'density'] = nx.density(G)
  df['density'] = pd.to_numeric(df['density'])
  df['intensity z score'] = df['intensity z score'].abs()
  X, Y = [], []
  
  for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
    print(combined_stimulus_name)
    data = df[df.apply(lambda x: combine_stimulus(x['stimulus'])[1], axis=1)==combined_stimulus_name]
    data = data.groupby(['stimulus', 'session']).mean(numeric_only=True)
    # print(data['density'].values)
    x = data['density'].values.tolist()
    y = data['intensity z score'].values.tolist()
    X += x
    Y += y
    ax.scatter(x, y, ec='.1', fc='none', marker=stimulus2marker[combined_stimulus_name], s=10*marker_size_dict[stimulus2marker[combined_stimulus_name]], alpha=.9, linewidths=1.5)
  
  X, Y = (list(t) for t in zip(*sorted(zip(X, Y))))
  X, Y = np.array(X), np.array(Y)
  slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(X),Y)
  line = slope*np.log10(X)+intercept
  locx, locy = .8, .1
  text = 'r={:.2f}, p={:.1e}'.format(r_value, p_value)
  ax.plot(X, line, color='.2', linestyle=(5,(10,3)), alpha=.5)
  ax.text(locx, locy, text, horizontalalignment='center',
     verticalalignment='center', transform=ax.transAxes, fontsize=22)
  ax.xaxis.set_tick_params(labelsize=25)
  ax.yaxis.set_tick_params(labelsize=25)
  plt.xlabel('Density')
  ylabel = 'Absolute motif significance' # 'Absolute Z score'
  plt.xscale('log')
  plt.ylabel(ylabel)
  ax.set_xlabel(ax.get_xlabel(), fontsize=28,color='k') #, weight='bold'
  ax.set_ylabel(ax.get_ylabel(), fontsize=28,color='k') #, weight='bold'
  for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(1.5)
    ax.spines[axis].set_color('k')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.tick_params(width=1.5)
  plt.tight_layout(rect=[0, 0, 1, .9])
  plt.savefig(f'./figures/figure3A.pdf', transparent=True)

################### Figure 3B ###################
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
  new_cmap = colors.LinearSegmentedColormap.from_list(
    'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
    cmap(np.linspace(minval, maxval, n)))
  return new_cmap

######################## Heatmap of Pearson Correlation r of Z score
def plot_heatmap_correlation_zscore(df):
  fig, ax = plt.subplots(1,1, figsize=(5.6,5))
  sorted_types = [sorted([smotif for smotif in df['signed motif type'].unique() if mt in smotif]) for mt in TRIAD_NAMES]
  sorted_types = [item for sublist in sorted_types for item in sublist]
  data_mat = np.zeros((len(combined_stimulus_names), len(sorted_types)))
  for s_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
    print(combined_stimulus_name)
    data = df[df.apply(lambda x: combine_stimulus(x['stimulus'])[1], axis=1)==combined_stimulus_name]
    data = data.groupby('signed motif type').mean(numeric_only=True)
    data_mat[s_ind] = data.loc[sorted_types, "intensity z score"].values.flatten()
  hm_z = np.corrcoef(data_mat)
  np.fill_diagonal(hm_z, np.nan)
  colors = ['w', '.3'] # first color is black, last is red
  cm = LinearSegmentedColormap.from_list(
        "Custom", colors, N=20)
  hm = sns.heatmap(hm_z, ax=ax, cmap=cm, vmin=0, vmax=1, cbar=True, annot=True, annot_kws={'fontsize':20})#, mask=mask
  cbar = hm.collections[0].colorbar
  cbar.ax.tick_params(labelsize=24)
  ax.set_title('Motif significance correlation', fontsize=25)
  ax.set_xticks([])
  ax.set_yticks([])
  ax.invert_yaxis() # put spontaneous on the bottom
  hm.tick_params(left=False)  # remove the ticks
  hm.tick_params(bottom=False)
  hm.tick_params(top=False)
  fig.tight_layout()
  plt.savefig('./figures/figure3B.pdf', transparent=True)

################### Figure 3C ###################
def get_signalcorr_within_across_motif(G_dict, active_area_dict, mean_df, eFFLb_types, all_motif_types, signal_correlation_dict, pair_type='all'):
  sessions, stimuli = get_session_stimulus(G_dict)
  within_eFFLb_dict, within_motif_dict, across_motif_dict = {}, {}, {}
  motif_types = []
  motif_edges, motif_sms = {}, {}
  for signed_motif_type in all_motif_types:
    motif_types.append(signed_motif_type.replace('+', '').replace('-', ''))
  for motif_type in motif_types:
    motif_edges[motif_type], motif_sms[motif_type] = get_edges_sms(motif_type, weight='confidence')
  for session_ind, session in enumerate(sessions):
    print(session)
    active_area = active_area_dict[session]
    node_idx = sorted(active_area.keys())
    within_eFFLb_dict[session], within_motif_dict[session], across_motif_dict[session] = {}, {}, {}
    for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names[2:]):
      for stimulus in combined_stimuli[combined_stimulus_names.index(combined_stimulus_name)]:
        print(stimulus)
        df = mean_df[mean_df['stimulus']==stimulus]
        sig_motifs = df.loc[df['intensity z score'] > 2.576]['signed motif type'].tolist() # 99% 
        within_eFFLb_dict[session][stimulus], within_motif_dict[session][stimulus], across_motif_dict[session][stimulus] = [], [], []
        G = G_dict[session][stimulus]
        nodes = sorted(G.nodes())
        signal_corr = signal_correlation_dict[session][combined_stimulus_name]
        motifs_by_type = find_triads(G)
        if pair_type == 'all': # all neuron pairs
          neuron_pairs = list(itertools.combinations(node_idx, 2))
        elif pair_type == 'connected': # limited to connected pairs only
          neuron_pairs = list(G.to_undirected().edges())
        neuron_pairs = [tuple([nodes.index(node) for node in e]) for e in neuron_pairs]
        other_edges = set(neuron_pairs)
        for motif_type in motif_types:
          motifs = motifs_by_type[motif_type]
          for motif in motifs:
            smotif_type = motif_type + get_motif_sign(motif, motif_edges[motif_type], motif_sms[motif_type], weight='confidence')
            # smotif_type = motif_type + get_motif_sign(motif, motif_type, weight='weight')
            if pair_type == 'all': # all neuron pairs
              motif_pairs = list(itertools.combinations(motif.nodes(), 2))
            elif pair_type == 'connected': # limited to connected pairs only
              motif_pairs = list(motif.to_undirected().edges())
            motif_pairs = [tuple([nodes.index(node) for node in e]) for e in motif_pairs]
            within_signal_corr = [signal_corr[e] for e in motif_pairs if not np.isnan(signal_corr[e])]
            if len(within_signal_corr):
              if smotif_type in eFFLb_types:
                within_eFFLb_dict[session][stimulus] += within_signal_corr
                other_edges -= set(motif_pairs)
              # else: # if all motifs
              elif smotif_type in sig_motifs:
                within_motif_dict[session][stimulus] += within_signal_corr
                other_edges -= set(motif_pairs)
            
        for e in other_edges:
          if not np.isnan(signal_corr[e]):
            across_motif_dict[session][stimulus].append(signal_corr[e])
  df = pd.DataFrame()
  for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names[2:]):
    for stimulus in combined_stimuli[combined_stimulus_names.index(combined_stimulus_name)]:
      for session in sessions:
        within_eFFLb, within_motif, across_motif = within_eFFLb_dict[session][stimulus], within_motif_dict[session][stimulus], across_motif_dict[session][stimulus]
        # within_motif, across_motif = [e for e in within_motif if not np.isnan(e)], [e for e in across_motif if not np.isnan(e)] # remove nan values
        df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(within_eFFLb)[:,None], np.array(['within eFFLb'] * len(within_eFFLb))[:,None], np.array([combined_stimulus_name] * len(within_eFFLb))[:,None], np.array([session] * len(within_eFFLb))[:,None]), 1), columns=['signal_corr', 'type', 'stimulus', 'session'])], ignore_index=True)
        df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(within_motif)[:,None], np.array(['within other motif'] * len(within_motif))[:,None], np.array([combined_stimulus_name] * len(within_motif))[:,None], np.array([session] * len(within_motif))[:,None]), 1), columns=['signal_corr', 'type', 'stimulus', 'session'])], ignore_index=True)
        df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(across_motif)[:,None], np.array(['otherwise'] * len(across_motif))[:,None], np.array([combined_stimulus_name] * len(across_motif))[:,None], np.array([session] * len(across_motif))[:,None]), 1), columns=['signal_corr', 'type', 'stimulus', 'session'])], ignore_index=True)
  df['signal_corr'] = pd.to_numeric(df['signal_corr'])
  return df

def plot_signalcorr_within_across_motif_significance(origin_df, pair_type='all'):
  df = origin_df.copy()
  # df = df[df['stimulus']!='Flashes'] # remove flashes
  fig, ax = plt.subplots(1,1, figsize=(2*(len(combined_stimulus_names)-2), 5))
  df = df.set_index('stimulus')
  df = df.loc[combined_stimulus_names[2:]]
  df.reset_index(inplace=True)
  palette = ['k', 'grey','w']
  y = 'signal_corr'
  barplot = sns.barplot(x='stimulus', y=y, hue="type", hue_order=['within eFFLb', 'within other motif', 'otherwise'], palette=palette, ec='k', linewidth=2., data=df, ax=ax, capsize=.05, width=0.6)
  ax.yaxis.set_tick_params(labelsize=30)
  plt.setp(ax.get_legend().get_title(), fontsize='0') # for legend title
  plt.xticks([], []) # use markers to represent stimuli!
  ax.xaxis.set_tick_params(length=0)
  for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2)
    ax.spines[axis].set_color('k')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.tick_params(width=2)
  ax.set_xlabel('')
  ax.set_ylabel('Signal correlation', fontsize=40) #'Absolute ' + 
  handles, labels = ax.get_legend_handles_labels()
  ax.legend([], [], fontsize=0)
  # add significance annotation
  alpha_list = [.0001, .001, .01, .05]

  maxx = 0
  for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names[2:]):
    within_eFFLb, within_motif, across_motif = df[(df.stimulus==combined_stimulus_name)&(df.type=='within eFFLb')][y].values.flatten(), df[(df.stimulus==combined_stimulus_name)&(df.type=='within other motif')][y].values.flatten(), df[(df.stimulus==combined_stimulus_name)&(df.type=='otherwise')][y].values.flatten()
    eFFLb_sr, within_sr, across_sr = confidence_interval(within_eFFLb)[1], confidence_interval(within_motif)[1], confidence_interval(across_motif)[1]
    maxx = max(eFFLb_sr, within_sr, across_sr) if max(eFFLb_sr, within_sr, across_sr) > maxx else maxx
  h, l = .05 * maxx, .05 * maxx
  for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names[2:]):
    within_eFFLb, within_motif, across_motif = df[(df.stimulus==combined_stimulus_name)&(df.type=='within eFFLb')][y].values.flatten(), df[(df.stimulus==combined_stimulus_name)&(df.type=='within other motif')][y].values.flatten(), df[(df.stimulus==combined_stimulus_name)&(df.type=='otherwise')][y].values.flatten()
    if len(within_motif):
      _, p1 = ranksums(within_eFFLb, within_motif, alternative='greater')
      diff_star1 = '*' * (len(alpha_list) - bisect(alpha_list, p1)) if len(alpha_list) > bisect(alpha_list, p1) else 'ns'
    _, p2 = ranksums(within_eFFLb, across_motif, alternative='greater')
    diff_star2 = '*' * (len(alpha_list) - bisect(alpha_list, p2)) if len(alpha_list) > bisect(alpha_list, p2) else 'ns'
    # just for annotation location
    eFFLb_sr, within_sr, across_sr = confidence_interval(within_eFFLb)[1], confidence_interval(within_motif)[1], confidence_interval(across_motif)[1]
    eFFLb_sr += h
    within_sr += h
    across_sr += h
    if len(within_motif):
      annot_difference(diff_star1, -.18 + cs_ind, cs_ind, max(eFFLb_sr, within_sr), l, 2.5, 28, ax=ax)
    annot_difference(diff_star2, -.18 + cs_ind, .18 + cs_ind, max(eFFLb_sr, across_sr) + 3.5*h, l, 2.5, 28, ax=ax)
  plt.tight_layout(rect=[.02, -.03, 1, 1])
  plt.savefig('./figures/figure3C.pdf', transparent=True)

################### Figure 3D ###################
def most_common(lst):
    return max(set(lst), key=lst.count)

def get_motif_region(motif, node_area, motif_type):
  edges = list(motif.edges())
  nodes = [node for sub in edges for node in sub]
  triplets = list(set(nodes))
  if motif_type == '021D':
    node_P = most_common([i for i,j in edges])
    node_X, node_O = [j for i,j in edges]
  elif motif_type == '021U':
    node_P = most_common([j for i,j in edges])
    node_X, node_O = [i for i,j in edges]
  elif motif_type == '021C':
    node_X = most_common(nodes)
    triplets.remove(node_X)
    if (triplets[0], node_X) in edges:
      node_P, node_O = triplets
    else:
      node_O, node_P = triplets
  elif motif_type == '111D':
    node_X = most_common([j for i,j in edges])
    node_P = [j for i,j in edges if i == node_X][0]
    triplets.remove(node_X)
    triplets.remove(node_P)
    node_O = triplets[0]
  elif motif_type == '111U':
    node_X = most_common([i for i,j in edges])
    node_P = [i for i,j in edges if j == node_X][0]
    triplets.remove(node_X)
    triplets.remove(node_P)
    node_O = triplets[0]
  elif motif_type == '030T':
    node_P = most_common([i for i,j in edges])
    node_O = most_common([j for i,j in edges])
    triplets.remove(node_P)
    triplets.remove(node_O)
    node_X = triplets[0]
  elif motif_type == '030C':
    es = edges.copy()
    np.random.shuffle(es)
    node_P, node_O = es[0]
    triplets.remove(node_P)
    triplets.remove(node_O)
    node_X = triplets[0]
  elif motif_type == '201':
    node_P = most_common([i for i,j in edges])
    triplets.remove(node_P)
    np.random.shuffle(triplets)
    node_X, node_O = triplets
  elif motif_type == '120D':
    node_P = most_common([i for i,j in edges])
    triplets.remove(node_P)
    np.random.shuffle(triplets)
    node_X, node_O = triplets
  elif motif_type == '120U':
    node_O = most_common([j for i,j in edges])
    triplets.remove(node_O)
    np.random.shuffle(triplets)
    node_P, node_X = triplets
  elif motif_type == '120C':
    node_P = most_common([i for i,j in edges])
    node_O = most_common([j for i,j in edges])
    triplets.remove(node_P)
    triplets.remove(node_O)
    node_X = triplets[0]
  elif motif_type == '210':
    node_O = most_common([node for sub in edges for node in sub])
    triplets.remove(node_O)
    if tuple(triplets) in edges:
      node_P, node_X = triplets
    else:
      node_X, node_P = triplets
  elif motif_type == '300':
    np.random.shuffle(triplets)
    node_P, node_X, node_O = triplets
  node_order = [node_P, node_X, node_O]
  region = [node_area[node] for node in node_order]
  if motif_type in ['021D', '021U', '120D']: # P, X/O
    region = '_'.join([region[0], '_'.join(sorted(region[1:]))])
  elif motif_type == '030C':
    region = '_'.join(sorted([region, region[1:] + region[:1], region[2:] + region[:2]])[0]) # shift string
  elif motif_type in ['120U']:
    region = '_'.join(['_'.join(sorted(region[:2])), region[-1]]) # P/X, O
  elif motif_type == '300':
    region = '_'.join(sorted(region))
  else:
    region = '_'.join(region)
  return region

def get_motif_region_census(G_dict, area_dict, signed_motif_types):
  sessions, stimuli = get_session_stimulus(G_dict)
  region_count_dict = {}
  motif_types = []
  motif_edges, motif_sms = {}, {}
  for signed_motif_type in signed_motif_types:
    motif_types.append(signed_motif_type.replace('+', '').replace('-', ''))
  for motif_type in motif_types:
    motif_edges[motif_type], motif_sms[motif_type] = get_edges_sms(motif_type, weight='confidence')
  for session_ind, session in enumerate(sessions):
    print(session)
    node_area = area_dict[session]
    region_count_dict[session] = {}
    for stimulus_ind, stimulus in enumerate(stimuli):
      print(stimulus)
      region_count_dict[session][stimulus] = {}
      G = G_dict[session][stimulus]
      motifs_by_type = find_triads(G) # faster
      for signed_motif_type in signed_motif_types:
        motif_type = signed_motif_type.replace('+', '').replace('-', '')
        motifs = motifs_by_type[motif_type]
        for motif in motifs:
          smotif_type = motif_type + get_motif_sign(motif, motif_edges[motif_type], motif_sms[motif_type], weight='confidence')
          if smotif_type == signed_motif_type:
            region = get_motif_region(motif, node_area, motif_type)
            # print(smotif_type, region)
            region_count_dict[session][stimulus][smotif_type+region] = region_count_dict[session][stimulus].get(smotif_type+region, 0) + 1
      region_count_dict[session][stimulus] = dict(sorted(region_count_dict[session][stimulus].items(), key=lambda x:x[1], reverse=True))
  return region_count_dict

def plot_motif_region_error(whole_df, region_count_dict, signed_motif_types, mtype='all_V1'):
  fig, ax = plt.subplots(1,1, sharex=True, sharey=True, figsize=(7, 4))
  df = pd.DataFrame()
  palette = [[plt.cm.tab20b(i) for i in range(20)][i] for i in [8,16,18,19]] + [[plt.cm.tab20c(i) for i in range(20)][i] for i in [4,16]]
  for mt_ind, signed_motif_type in enumerate(signed_motif_types):
    print(signed_motif_type)
    for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
      for stimulus in combined_stimuli[cs_ind]:
        for session_ind, session in enumerate(session_ids):
          region_com = {}
          VISp_data, rest_data = [], []
          region_count = region_count_dict[session][stimulus]
          for k in region_count:
            if signed_motif_type in k:
              rs = k.replace(signed_motif_type, '')
              region_com[rs] = region_com.get(rs, 0) + region_count[k]
          if mtype == 'all_V1':
            VISp_data.append(region_com.get('VISp_VISp_VISp', 0))
            rest_data.append(sum([region_com[k] for k in region_com if k!= 'VISp_VISp_VISp']))
          elif mtype == 'one_V1':
            VISp_data.append(sum([region_com[k] for k in region_com if 'VISp' in k]))
            rest_data.append(sum([region_com[k] for k in region_com if 'VISp' not in k]))
          summ = sum(VISp_data) + sum(rest_data)
          if (summ >= 5) and (whole_df[(whole_df.session==session)&(whole_df['signed motif type']==signed_motif_type)&(whole_df.stimulus==stimulus)]['intensity z score'].item() > 1.96): # othewise flashes will disappear
            VISp_data = [sum(VISp_data)/summ]
            rest_data = [sum(rest_data)/summ]
            df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(VISp_data)[:,None], np.array([signed_motif_type] * len(VISp_data))[:,None], np.array([combined_stimulus_name] * len(VISp_data))[:,None]), 1), columns=['probability', 'type', 'stimulus'])], ignore_index=True)
  for mt_ind, signed_motif_type in enumerate(signed_motif_types):
    for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
      if not df[(df.type==signed_motif_type) & (df.stimulus==combined_stimulus_name)].shape[0]:
        df = pd.concat([df, pd.DataFrame(np.array([[0, signed_motif_type, combined_stimulus_name]]), columns=['probability', 'type', 'stimulus'])], ignore_index=True)
  df['probability'] = pd.to_numeric(df['probability'])
  for mt_ind, signed_motif_type in enumerate(signed_motif_types):
    data = df[df.type==signed_motif_type].groupby('stimulus')
    x = np.arange(len(combined_stimulus_names))
    y = data.mean(numeric_only=True).loc[combined_stimulus_names].values.flatten()
    err = data.std(numeric_only=True).loc[combined_stimulus_names].values.flatten()
    for ind, (xi, yi, erri) in enumerate(zip(x, y, err)):
      if yi:
        ax.errorbar(xi + .13 * mt_ind, yi, yerr=erri, fmt=' ', linewidth=2.,color=palette[mt_ind], zorder=1)
        ax.scatter(xi + .13 * mt_ind, yi, marker=stimulus2marker[combined_stimulus_names[ind]], s=10*error_size_dict[stimulus2marker[combined_stimulus_names[ind]]], linewidth=1.,color=palette[mt_ind], zorder=2)
  
  ax.set(xlabel=None)
  ax.xaxis.set_tick_params(length=0)
  ax.set_xticks([])
  ax.yaxis.set_tick_params(labelsize=25)
  ax.set_xlabel('')
  for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2.)
    ax.spines[axis].set_color('k')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.tick_params(width=2.)
  ax.set_ylim(bottom=0)
  ylabel = 'Fraction of at least\none V1 neuron' if mtype=='one_V1' else 'Fraction of three\nV1 neurons'
  ax.set_ylabel(ylabel, fontsize=25)
  plt.tight_layout(rect=[.02, -.03, 1, 1])
  fname = 'left' if mtype=='one_V1' else 'right'
  plt.savefig('./figures/figure3D_{}.pdf'.format(fname), transparent=True)

################### Figure 3E ###################
def get_motif_IDs(G_dict, area_dict, signed_motif_types):
  sessions, stimuli = get_session_stimulus(G_dict)
  motif_id_dict = {}
  motif_types = []
  motif_edges, motif_sms = {}, {}
  for signed_motif_type in signed_motif_types:
    motif_types.append(signed_motif_type.replace('+', '').replace('-', ''))
  for motif_type in motif_types:
    motif_edges[motif_type], motif_sms[motif_type] = get_edges_sms(motif_type, weight='confidence')
  for session_ind, session in enumerate(sessions):
    print(session)
    node_area = area_dict[session]
    motif_id_dict[session] = {}
    for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
      print(combined_stimulus_name)
      motif_id_dict[session][combined_stimulus_name] = {s_type:[] for s_type in signed_motif_types}
      for stimulus in combined_stimuli[cs_ind]:
        G = G_dict[session][stimulus]
        motifs_by_type = find_triads(G) # faster
        for signed_motif_type in signed_motif_types:
          motif_type = signed_motif_type.replace('+', '').replace('-', '')
          motifs = motifs_by_type[motif_type]
          for motif in motifs:
            smotif_type = motif_type + get_motif_sign(motif, motif_edges[motif_type], motif_sms[motif_type], weight='confidence')
            if (smotif_type == signed_motif_type) and (not tuple(list(motif.nodes())) in motif_id_dict[session][combined_stimulus_name][smotif_type]):
              motif_id_dict[session][combined_stimulus_name][smotif_type].append(tuple(list(motif.nodes())))
  return motif_id_dict

# get motif stimulus list dictionary for each mouse and each stimulus and each motif type
def get_motif_stim_list_dict(motif_id_dict, signed_motif_types):
  motif_stim_dict = {smt:{} for smt in signed_motif_types}
  for smt_ind, signed_motif_type in enumerate(signed_motif_types):
    motif_stim_dict[signed_motif_type] = {session_id:{} for session_id in session_ids}
    for s_ind, session_id in enumerate(session_ids):
      for combined_stimulus_name in combined_stimulus_names:
        motif_stim_dict[signed_motif_type][session_id][combined_stimulus_name] = motif_id_dict[session_id][combined_stimulus_name][signed_motif_type]
  return motif_stim_dict

# merge all individuals into one large set
def merge_mice(motif_stim_list_dict):
  merge_ind_motif_stim_list_dict = {smt:{csn:[] for csn in combined_stimulus_names} for smt in motif_stim_list_dict}
  for smt in motif_stim_list_dict:
    for csn in combined_stimulus_names:
      for s_ind, session_id in enumerate(session_ids):
        merge_ind_motif_stim_list_dict[smt][csn] += [tuple([s_ind]) + ele for ele in motif_stim_list_dict[smt][session_id][csn]] # s_ind to differentiate neurons
  return merge_ind_motif_stim_list_dict

def plot_certain_upsetplot_combined(merge_ind_motif_stim_list_dict, sig_motif_type, cutoff=10):
  motif_stim = from_contents(merge_ind_motif_stim_list_dict[sig_motif_type])
  motif_stim.index.rename(['',' ','  ','   ','    ','     '], inplace=True)
  fig = plt.figure(figsize=(10, 6))
  p = UpSet(data = motif_stim, 
      sort_categories_by='input', 
      sort_by='cardinality', 
      min_subset_size=cutoff, 
      totals_plot_elements=2, 
      intersection_plot_elements=5)
  p.plot(fig=fig)
  fig.savefig('./figures/figure3E.pdf', transparent=True)

################### Figure 3F ###################
def scale_to_interval(origin_x, low=1, high=100, logscale=False):
  if not logscale:
    x = np.abs(origin_x.copy())
  else:
    x = np.power(2, np.abs(origin_x.copy()))
    x = np.nan_to_num(x, neginf=0, posinf=0)
  y = ((x - x.min()) / (x.max() - x.min())) * (high - low) + low
  return y

def single_circular_lollipop(data, ind, sorted_types, COLORS, GREY, ax, lwv=.9, lw0=.7, neggrid=-5, posgrid=10, low=1, high=100, logscale=False):
  ANGLES = np.linspace(0, 2 * np.pi, len(data), endpoint=False)
  HEIGHTS = np.array(data)
  PLUS = 0
  ax.set_facecolor("white")
  ax.set_theta_offset(-np.pi / 2)
  ax.set_theta_direction(-1)
  ax.vlines(ANGLES, 0 + PLUS, HEIGHTS + PLUS, color=COLORS, lw=lwv)
  ax.scatter(ANGLES, HEIGHTS + PLUS, color=COLORS, s=scale_to_interval(HEIGHTS, low=low, high=high, logscale=logscale)) #
  ax.spines["start"].set_color("none")
  ax.spines["polar"].set_color("none")
  ax.grid(False)
  ax.set_xticks([])
  ax.set_yticklabels([])
  HANGLES = np.linspace(0, 2 * np.pi, 200)
  ls = (0, (5, 5))
  ax.plot(HANGLES, np.repeat(neggrid + PLUS, 200), color= GREY, lw=lw0, linestyle=(0, (5, 1)))
  ax.plot(HANGLES, np.repeat(0 + PLUS, 200), color= GREY, lw=lw0, linestyle=ls) # needs to be denser
  ax.plot(HANGLES, np.repeat(posgrid + PLUS, 200), color= GREY, lw=lw0, linestyle=ls)
  if ind == 0:
    ax.set_ylim([-5, 11]) # increase 
  elif ind == 1:
    ax.set_ylim(top = 8.5)
  elif ind == 2:
    ax.set_ylim([-7, 20])
  elif ind == 3:
    ax.set_ylim([-7, 21])

def multi_circular_lollipop(df1, df2, df3, df4, motif_palette, stimulus_name='natural_movie_three'):
  GREY = '.5'
  fig, axes = plt.subplots(1, 4, figsize=(20, 7), subplot_kw={"projection": "polar"})
  fig.patch.set_facecolor("white")
  sorted_types = [sorted([smotif for smotif in df1['signed motif type'].unique() if mt in smotif]) for mt in TRIAD_NAMES]
  sorted_types = [item for sublist in sorted_types for item in sublist]
  motif_types = TRIAD_NAMES[3:]
  COLORS = []
  for t in sorted_types:
    COLORS.append(motif_palette[motif_types.index(t.replace('+', '').replace('\N{MINUS SIGN}', '').replace('-', ''))])
  dfs = [df1, df2, df3, df4]
  for df_ind, df in enumerate(dfs):
    ax = axes[df_ind]
    data = df[df.stimulus==stimulus_name]
    data = data.groupby('signed motif type').mean()
    zscores = data.loc[sorted_types, "intensity z score"].values.tolist()
    neggrid, posgrid, low, high, logscale = -5, 10, 0, 200, False
    if df_ind == 0: # manual logscale
      signs = np.array([1 if zs >= 0 else -1 for zs in zscores])
      zscores = np.log2(np.abs(zscores)) * signs
      neggrid = - np.log2(10)
      posgrid = np.log2(100)
      # low=0
      high=600
      logscale = True
    elif df_ind == 1:
      signs = np.array([1 if zs >= 0 else -1 for zs in zscores])
      zscores = np.log2(np.abs(zscores)) * signs
      neggrid = - np.log2(10)
      posgrid = np.log2(20)
      # low=0
      high=300
      logscale = True
    # print(zscores)
    single_circular_lollipop(zscores, df_ind, sorted_types, COLORS, GREY, ax, lwv=2.5, lw0=3., neggrid=neggrid, posgrid=posgrid, low=low, high=high, logscale=logscale)
    ax.set_title(model_names[df_ind], fontsize=30)
  fig.subplots_adjust(wspace=0.) #
  plt.tight_layout()
  # plt.show()
  plt.savefig('./figures/figure3F.pdf', transparent=True)

def get_max_dH_pos_neg_resolution(sessions, stimuli, resolution_list, real_H, subs_H): 
  max_reso_subs = np.zeros((len(sessions), len(stimuli), 2)) # last dim 0 for pos, 1 for neg
  for session_ind, session in enumerate(sessions):
    print(session)
    for stimulus_ind, stimulus in enumerate(stimuli):
      metric_mean = real_H[session_ind, stimulus_ind].mean(-1)
      metric_subs = subs_H[session_ind, stimulus_ind].mean(-1)
      pos_ind, neg_ind = np.unravel_index(np.argmax(metric_subs - metric_mean), np.array(metric_mean).shape)
      max_reso_subs[session_ind, stimulus_ind] = resolution_list[pos_ind], resolution_list[neg_ind]
  return max_reso_subs

################### Figure 4A ###################
def comm2label(comms):
  return [p for n, p in sorted({n:comms.index(comm) for comm in comms for n in comm}.items(), key=lambda x:x[0])]

def comm2partition(comms):
  return dict(sorted({node:comms.index(comm) for comm in comms for node in comm}.items(), key=lambda x:x[0]))

############ find nodes and comms with at least one between community edge
def flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]

def get_unique_elements(nested_list):
    return list(set(flatten_list(nested_list)))

def _find_between_community_edges(edges, node_to_community):
  """Convert the graph into a weighted network of communities."""
  between_community_edges = dict()
  for (ni, nj) in edges:
      if (ni in node_to_community) and (nj in node_to_community):
          ci = node_to_community[ni]
          cj = node_to_community[nj]
          if ci != cj:
              if (ci, cj) in between_community_edges:
                  between_community_edges[(ci, cj)] += 1
              elif (cj, ci) in between_community_edges:
                  # only compute the undirected graph
                  between_community_edges[(cj, ci)] += 1
              else:
                  between_community_edges[(ci, cj)] = 1

  return between_community_edges

def plot_graph_community(G_dict, active_area_dict, session_ind, best_comms_dict):
  seed = 123
  np.random.seed(seed)
  sessions, stimuli = get_session_stimulus(G_dict)
  session = sessions[session_ind]
  fig, axes = plt.subplots(1, len(combined_stimuli), figsize=(12*len(combined_stimuli), 12))
  comm_inds = [49, 5, 102, 5, 13, 157] # randomly choose a realization for visualization
  th = 6 # only plot large communities for visualization
  for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
    comm_ind = comm_inds[cs_ind]
    stimulus = combined_stimuli[cs_ind][-1]
    ax = axes[cs_ind]
    stimulus_ind = stimuli.index(stimulus)
    ax.set_title(combined_stimulus_name.replace('\n', ' '), fontsize=80)
    print(stimulus)
    G = G_dict[session][stimulus]
    nx.set_node_attributes(G, active_area_dict[session], "area")
    comms_list = best_comms_dict[session][stimulus]
    comms = comms_list[comm_ind]
    comms = [comm for comm in comms if len(comm)>=th] # only plot large communities
    node_to_community = comm2partition(comms)
    between_community_edges = _find_between_community_edges(G.edges(), node_to_community)
    comms2plot = get_unique_elements(between_community_edges.keys())
    nodes2plot = [node for node in node_to_community if node_to_community[node] in comms2plot]
    node_color = {node:region_colors[visual_regions.index(G.nodes[node]['area'])] for node in nodes2plot}
    print('Number of communities {}, number of nodes: {}'.format(len(comms2plot), len(nodes2plot)))
    if len(nodes2plot):
      edge_colors = [transparent_rgb(colors.to_rgb('#2166ac'), [1,1,1], alpha=.4), transparent_rgb(colors.to_rgb('#b2182b'), [1,1,1], alpha=.3)] # blue and red
      # return G.subgraph(nodes2plot)
      G2plot = G.subgraph(nodes2plot).copy()
      for n1, n2, d in G2plot.edges(data=True):
        for att in ['confidence']:
          d.pop(att, None)
      # binary weight
      for edge in G2plot.edges():
        if G2plot[edge[0]][edge[1]]['weight'] > 0:
          G2plot[edge[0]][edge[1]]['weight'] = 1
        else:
          G2plot[edge[0]][edge[1]]['weight'] = -1
      # node_labels = {node:node_to_community[node] for node in nodes2plot}
      if cs_ind in [0]:
        origin, scale, node_origin, node_scale=(-1, -1), (.9, .9), np.array([-1., -1.]), np.array([2., 2.])
      elif cs_ind in [1]:
        origin, scale, node_origin, node_scale=(-1, -1), (.8, .8), np.array([.5, .5]), np.array([3.5, 3.5])
      elif cs_ind in [2]:
        origin, scale, node_origin, node_scale=(-1, -1), (.9, .9), np.array([-1., -1.]), np.array([2., 2.])
      elif cs_ind in [3, 4, 5]:
        origin, scale, node_origin, node_scale=(-1, -1), (.9, .9), np.array([-1., -1.]), np.array([2., 2.])
      Graph(G2plot, nodes=nodes2plot, edge_cmap=colors.LinearSegmentedColormap.from_list("", edge_colors),
            node_color=node_color, node_edge_width=0.5, node_alpha=1., edge_alpha=0.4,
            node_layout='community', node_layout_kwargs=dict(node_to_community={node: comm for node, comm in node_to_community.items() if node in nodes2plot}),
            edge_layout='straight', edge_layout_kwargs=dict(k=1), node_edge_color='k',
            origin=origin, scale=scale, node_origin=node_origin, node_scale=node_scale, ax=ax) # bundled, node_labels=node_labels
  plt.tight_layout()
  plt.savefig('./figures/figure4A_top.pdf', transparent=True)

def arrowed_spines(
        ax,
        x_width_fraction=0.2,
        x_height_fraction=0.02,
        lw=None,
        ohg=0.2,
        locations=('bottom right', 'left up'),
        **arrow_kwargs
):
    # set/override some default plotting parameters if required
    arrow_kwargs.setdefault('overhang', ohg)
    arrow_kwargs.setdefault('clip_on', False)
    arrow_kwargs.update({'length_includes_head': True})

    # axis line width
    if lw is None:
        # FIXME: does this still work if the left spine has been deleted?
        lw = ax.spines['left'].get_linewidth()
    annots = {}
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    # get width and height of axes object to compute
    # matching arrowhead length and width
    fig = ax.get_figure()
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height
    # manual arrowhead width and length
    hw = x_width_fraction * (ymax-ymin)
    hl = x_height_fraction * (xmax-xmin)
    # compute matching arrowhead length and width
    yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width
    yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height
    # draw x and y axis
    for loc_str in locations:
        side, direction = loc_str.split(' ')
        assert side in {'top', 'bottom', 'left', 'right'}, "Unsupported side"
        assert direction in {'up', 'down', 'left', 'right'}, "Unsupported direction"

        if side in {'bottom', 'top'}:
            if direction in {'up', 'down'}:
                raise ValueError("Only left/right arrows supported on the bottom and top")
            dy = 0
            head_width = hw
            head_length = hl
            y = ymin if side == 'bottom' else ymax

            if direction == 'right':
                x = xmin
                dx = xmax - xmin
            else:
                x = xmax
                dx = xmin - xmax
        else:
            if direction in {'left', 'right'}:
                raise ValueError("Only up/downarrows supported on the left and right")
            dx = 0
            head_width = yhw
            head_length = yhl
            x = xmin if side == 'left' else xmax
            if direction == 'up':
                y = ymin
                dy = ymax - ymin
            else:
                y = ymax
                dy = ymin - ymax
        annots[loc_str] = ax.arrow(x, y, dx, dy, fc='k', ec='k', lw = lw,
                 head_width=head_width, head_length=head_length, **arrow_kwargs)

    return annots

def plot_zscore_Hamiltonian2Q(G_dict, resolution_list, max_pos_neg_reso, real_H, subs_H, max_method='none', cc=False):
  sessions, stimuli = get_session_stimulus(G_dict)
  real_Q, subs_Q = np.zeros_like(real_H), np.zeros_like(subs_H)
  for session_ind, session in enumerate(sessions):
    for stimulus_ind, stimulus in enumerate(stimuli):
      G = G_dict[session][stimulus]
      tw = sum([abs(G.get_edge_data(*edge)['weight']) for edge in G.edges()])
      real_Q[session_ind, stimulus_ind] = - real_H[session_ind, stimulus_ind] / tw
      subs_Q[session_ind, stimulus_ind] = - subs_H[session_ind, stimulus_ind] / tw
  zscore_Hamiltonian2Q, ps = [[] for _ in range(len(combined_stimuli))], [[] for _ in range(len(combined_stimuli))]
  runs = real_H.shape[-1]
  for session_ind, session in enumerate(sessions):
    for stimulus_ind, stimulus in enumerate(stimuli):
      q, rq = [], []
      for run in range(runs):
        max_reso = max_pos_neg_reso[session_ind, stimulus_ind]
        q.append(real_Q[session_ind, stimulus_ind, resolution_list.index(max_reso[0]), resolution_list.index(max_reso[1]), run])
        rq.append(subs_Q[session_ind, stimulus_ind, resolution_list.index(max_reso[0]), resolution_list.index(max_reso[1]), run])
      zscore_Hamiltonian2Q[combine_stimulus(stimulus)[0]].append(ztest(q, rq)[0])
      ps[combine_stimulus(stimulus)[0]].append(ztest(q, rq)[1])
  fig, ax = plt.subplots(1, 1, figsize=(6*len(combined_stimuli), 1.2))
  ax.bar(range(len(combined_stimulus_names)), [np.mean(dH2Q) for dH2Q in zscore_Hamiltonian2Q], width=.07, align='center', alpha=1., linewidth=5, facecolor='w', edgecolor='black', capsize=10) #
  ax.set_xlim(-.4, 5.5)
  ax.set_ylim(0., 70)
  annots = arrowed_spines(ax, locations=['bottom right'], lw=4.)
  ax.set_xticks([])
  # ax.set_xticklabels(combined_stimulus_names, fontsize=25)
  for axis in ['top', 'left']:
    ax.spines[axis].set_linewidth(3.)
    ax.spines[axis].set_color('k')
  ax.xaxis.tick_top()
  ax.spines['right'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  ax.yaxis.set_tick_params(labelsize=30)
  ax.tick_params(width=3)
  ax.set_ylabel('$Z_Q$', size=40)
  ax.invert_yaxis()
  plt.tight_layout()
  plt.savefig('./figures/figure4A_bottom.pdf', transparent=True)

################### Figure4C ###################
def get_purity_coverage_ri(G_dict, area_dict, regions, best_comms_dict):
  sessions, stimuli = get_session_stimulus(G_dict)
  df = pd.DataFrame()
  for stimulus_ind, stimulus in enumerate(stimuli):
    print(stimulus)
    w_purity_col, w_coverage_col, ri_list = [], [], []
    for session_ind, session in enumerate(sessions):
      data = {}
      G = G_dict[session][stimulus].copy()
      node_area = area_dict[session]
      all_regions = [node_area[node] for node in sorted(G.nodes())]
      region_counts = np.array([all_regions.count(r) for r in regions])
      for run in range(len(best_comms_dict[session][stimulus])):
        comms_list = best_comms_dict[session][stimulus]
        comms = comms_list[run]
        large_comms = [comm for comm in comms if len(comm)>=4]
        for comm in large_comms:
          c_regions = [area_dict[session][node] for node in comm]
          counts = np.array([c_regions.count(r) for r in regions])
          size = counts.sum()
          purity = counts.max() / size
          coverage = (counts / region_counts).max()
          if size in data:
            data[size][0].append(purity)
            data[size][1].append(coverage)
          else:
            data[size] = [[purity], [coverage]]
        c_size, c_purity, c_coverage = [k for k,v in data.items()], [v[0] for k,v in data.items()], [v[1] for k,v in data.items()]
        all_size = np.repeat(c_size, [len(p) for p in c_purity])
        c_size = all_size / sum(all_size)
        w_purity_col.append(sum([cs * cp for cs, cp in zip(c_size, [p for ps in c_purity for p in ps])]))
        w_coverage_col.append(sum([cs * cc for cs, cc in zip(c_size, [c for css in c_coverage for c in css])]))
        assert len(all_regions) == len(comm2label(comms)), '{}, {}'.format(len(all_regions), len(comm2label(comms)))
        ri_list.append(adjusted_rand_score(all_regions, comm2label(comms)))
    df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(w_purity_col)[:,None], np.array(['weighted purity'] * len(w_purity_col))[:,None], np.array([combine_stimulus(stimulus)[1]] * len(w_purity_col))[:,None]), 1), columns=['data', 'type', 'combined stimulus'])], ignore_index=True)
    df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(w_coverage_col)[:,None], np.array(['weighted coverage'] * len(w_coverage_col))[:,None], np.array([combine_stimulus(stimulus)[1]] * len(w_coverage_col))[:,None]), 1), columns=['data', 'type', 'combined stimulus'])], ignore_index=True)
    df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(ri_list)[:,None], np.array(['rand index'] * len(ri_list))[:,None], np.array([combine_stimulus(stimulus)[1]] * len(ri_list))[:,None]), 1), columns=['data', 'type', 'combined stimulus'])], ignore_index=True)
  df['data'] = pd.to_numeric(df['data'])
  return df

def plot_weighted_coverage_purity_rand_index_markers(df, dname):
  fig, ax = plt.subplots(1, 1, figsize=(3, 2.5))
  x = np.arange(len(combined_stimulus_names))
  # use nonparametric confidence interval as error bar
  y, err = [], []
  for combined_stimulus_name in combined_stimulus_names:
    lb, ub = nonpara_confidence_interval(df[(df['combined stimulus']==combined_stimulus_name)&(df['type']==dname)]['data'].values, confidence_level=0.95)
    y.append((lb + ub) / 2)
    err.append((ub - lb) / 2)
  for ind, (xi, yi, erri) in enumerate(zip(x, y, err)):
    ax.errorbar(xi, yi, yerr=erri, fmt=' ', linewidth=2.,color='.1', zorder=1)
    ax.scatter(xi, yi, marker=stimulus2marker[combined_stimulus_names[ind]], s=15*error_size_dict[stimulus2marker[combined_stimulus_names[ind]]], linewidth=1.,color='k', facecolor='white', zorder=2)
  ax.set(xlabel=None)
  ax.xaxis.set_tick_params(length=0)
  ax.set_xlim(-.8, len(combined_stimulus_names)-.2)
  # ax.invert_yaxis()
  ax.set_xticks([])
  ax.yaxis.set_tick_params(labelsize=20)
  ax.set_xlabel('')
  for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(1.5)
    ax.spines[axis].set_color('k')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.tick_params(width=1.5)
  ymin, ymax = ax.get_ylim()
  ax.set_ylim(0.9*ymin, 1.08*ymax)
  plt.tight_layout()
  if dname == 'weighted coverage':
    fname = 'top'
    ylabel = 'WA coverage'
  elif dname == 'weighted purity':
    fname = 'middle'
    ylabel = 'WA purity'
  elif dname == 'rand index':
    fname = 'bottom'
    ylabel = 'ARI'
  ax.set_ylabel(ylabel, fontsize=28)
  plt.savefig('./figures/figure4C_{}.pdf'.format(fname), transparent=True)

################### Figure 5A ###################
def get_within_across_comm(G_dict, active_area_dict, signal_correlation_dict, comms_dict, pair_type='all'):
  sessions = list(signal_correlation_dict.keys())
  within_comm_dict, across_comm_dict = {}, {}
  runs = len(comms_dict[sessions[0]][stimulus_names[0]])
  for session_ind, session in enumerate(sessions):
    print(session)
    active_area = active_area_dict[session]
    node_idx = sorted(active_area.keys())
    within_comm_dict[session], across_comm_dict[session] = {}, {}
    for combined_stimulus_name in combined_stimulus_names[2:]:
      within_comm_dict[session][combined_stimulus_name], across_comm_dict[session][combined_stimulus_name] = [], []
      signal_correlation = signal_correlation_dict[session][combined_stimulus_name]
      for stimulus in combined_stimuli[combined_stimulus_names.index(combined_stimulus_name)]:
        stimulus_ind = stimulus_names.index(stimulus)
        G = G_dict[session][stimulus]
        nodes = sorted(G.nodes())
        comms_list = comms_dict[session][stimulus]
        for run in range(runs):
          comms = comms_list[run]
          within_comm, across_comm = [], []
          node_to_community = comm2partition(comms)
          if pair_type == 'all': # all neuron pairs
            neuron_pairs = list(itertools.combinations(node_idx, 2))
          elif pair_type == 'connected': # limited to connected pairs only
            neuron_pairs = G.to_undirected().edges()
          neuron_pairs = [tuple([nodes.index(node) for node in e]) for e in neuron_pairs]
          for nodei, nodej in neuron_pairs: # for all neurons
            scorr = signal_correlation[nodei, nodej] # abs(signal_correlation[nodei, nodej])
            if node_to_community[nodes[nodei]] == node_to_community[nodes[nodej]]:
              within_comm.append(scorr)
            else:
              across_comm.append(scorr)
          within_comm_dict[session][combined_stimulus_name] += within_comm
          across_comm_dict[session][combined_stimulus_name] += across_comm
  df = pd.DataFrame()
  for combined_stimulus_name in combined_stimulus_names[2:]:
    print(combined_stimulus_name)
    for session in sessions:
      within_comm, across_comm = within_comm_dict[session][combined_stimulus_name], across_comm_dict[session][combined_stimulus_name]
      within_comm, across_comm = [e for e in within_comm if not np.isnan(e)], [e for e in across_comm if not np.isnan(e)] # remove nan values
      df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(within_comm)[:,None], np.array(['within community'] * len(within_comm))[:,None], np.array([combined_stimulus_name] * len(within_comm))[:,None], np.array([session] * len(within_comm))[:,None]), 1), columns=['signal correlation', 'type', 'stimulus', 'session'])], ignore_index=True)
      df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(across_comm)[:,None], np.array(['across community'] * len(across_comm))[:,None], np.array([combined_stimulus_name] * len(across_comm))[:,None], np.array([session] * len(across_comm))[:,None]), 1), columns=['signal correlation', 'type', 'stimulus', 'session'])], ignore_index=True)
  df['signal correlation'] = pd.to_numeric(df['signal correlation'])
  return df

def plot_signal_correlation_within_across_comm_significance(origin_df, pair_type='all'):
  df = origin_df.copy()
  fig, ax = plt.subplots(1,1, sharex=True, sharey=True, figsize=(2*(len(combined_stimulus_names)-2), 5))
  palette = ['k','.6']
  barplot = sns.barplot(x='stimulus', y='signal correlation', hue="type", hue_order=['within community', 'across community'], palette=palette, ec='k', linewidth=2., data=df, ax=ax, capsize=.05, width=0.6, errorbar=('ci', 95))
  ax.yaxis.set_tick_params(labelsize=30)
  plt.setp(ax.get_legend().get_title(), fontsize='0') # for legend title
  plt.xticks([], []) # use markers to represent stimuli!
  ax.xaxis.set_tick_params(length=0)
  for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(1.5)
    ax.spines[axis].set_color('k')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.tick_params(width=1.5)
  ax.set_xlabel('')
  ax.set_ylabel('Signal correlation', fontsize=40)
  ax.legend([], [], fontsize=0)
  # add significance annotation
  alpha_list = [.0001, .001, .01, .05]
  maxx = 0
  for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names[2:]):
    within_comm, across_comm = df[(df.stimulus==combined_stimulus_name)&(df.type=='within community')]['signal correlation'].values.flatten(), df[(df.stimulus==combined_stimulus_name)&(df.type=='across community')]['signal correlation'].values.flatten()
    within_sr, across_sr = confidence_interval(within_comm)[1], confidence_interval(across_comm)[1]
    maxx = max(within_sr, across_sr) if max(within_sr, across_sr) > maxx else maxx
  h, l = .05 * maxx, .05 * maxx
  for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names[2:]):
    within_comm, across_comm = df[(df.stimulus==combined_stimulus_name)&(df.type=='within community')]['signal correlation'].values.flatten(), df[(df.stimulus==combined_stimulus_name)&(df.type=='across community')]['signal correlation'].values.flatten()
    # shapiro test for normality before using t test
    print(combined_stimulus_name.replace('\n', ' '), shapiro(within_comm)[1], shapiro(across_comm)[1])
    _, p = ttest_ind(within_comm, across_comm, alternative='greater')
    diff_star = '*' * (len(alpha_list) - bisect(alpha_list, p)) if len(alpha_list) > bisect(alpha_list, p) else 'ns'
    within_sr, across_sr = confidence_interval(within_comm)[1], confidence_interval(across_comm)[1]
    within_sr = within_sr + h
    across_sr = across_sr + h
    annot_difference(diff_star, -.15 + cs_ind, .15 + cs_ind, max(within_sr, across_sr), l, 2.5, 28, ax=ax)
  plt.tight_layout(rect=[.02, -.03, 1, 1])
  plt.savefig('./figures/figure5A.pdf', transparent=True)

################### Figure 5B ###################
def get_module_size_coverage_purity(G_dict, active_area_dict, regions, best_comms_dict):
  sessions, stimuli = get_session_stimulus(G_dict)
  runs = len(best_comms_dict[sessions[0]][stimuli[0]])
  size_dict, coverage_dict, purity_dict = [{session:{stimulus:{run:[] for run in range(runs)} for stimulus in stimuli} for session in sessions} for _ in range(3)]
  for session_ind, session in enumerate(sessions):
    print(session)
    active_area = active_area_dict[session]
    # size_dict[session], purity_dict[session] = {}, {}
    for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
      for stimulus in combined_stimuli[cs_ind]:
        G = G_dict[session][stimulus]
        all_regions = [active_area[node] for node in G.nodes()]
        region_counts = np.array([all_regions.count(r) for r in regions])
        # size_dict[session][stimulus], purity_dict[session][stimulus] = {}, {}
        stimulus_ind = stimulus_names.index(stimulus)
        for run in range(runs):
          comms_list = best_comms_dict[session][stimulus]
          comms = comms_list[run]
          comms = [comm for comm in comms if len(comm) >= 4]
          for comm in comms:
            c_regions = [active_area[node] for node in comm]
            counts = np.array([c_regions.count(r) for r in regions])
            if G.subgraph(comm).number_of_edges():
              size_dict[session][stimulus][run].append(len(comm) / G.number_of_nodes()) # relative size
              coverage_dict[session][stimulus][run].append((counts / region_counts).max())
              purity_dict[session][stimulus][run].append(counts.max() / counts.sum())
  return size_dict, coverage_dict, purity_dict

def plot_num_module_VSpurity_threshold(size_dict, coverage_dict, purity_dict, sth_list_lin, sth_list_log, cth_list_lin, cth_list_log, pth_list):
  sessions, stimuli = get_session_stimulus(purity_dict)
  runs = len(purity_dict[sessions[0]][stimuli[0]])
  fig, axes = plt.subplots(1, 3, figsize=(5*3, 4), sharex=True, sharey=True)
  num_module1_lin, num_module1_log, num_module2_lin, num_module2_log, num_module3 = np.zeros((len(combined_stimulus_names), len(sth_list_lin))), np.zeros((len(combined_stimulus_names), len(sth_list_log))), np.zeros((len(combined_stimulus_names), len(cth_list_lin))), np.zeros((len(combined_stimulus_names), len(cth_list_log))), np.zeros((len(combined_stimulus_names), len(pth_list)))
  for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
    size, coverage, purity = [], [], []
    for stimulus in combined_stimuli[cs_ind]:
      for session in sessions:
        for run in range(runs):
          # purity.append(np.mean(purity_dict[session][stimulus][run]))
          size += size_dict[session][stimulus][run]
          coverage += coverage_dict[session][stimulus][run]
          purity += purity_dict[session][stimulus][run]
    size, coverage, purity = np.array(size), np.array(coverage), np.array(purity)
    for sth_ind, sth in enumerate(sth_list_lin):
      inds = size>=sth
      num_module1_lin[cs_ind, sth_ind] = inds.sum() / (runs * len(sessions) * len(combined_stimuli[cs_ind]))
    for sth_ind, sth in enumerate(sth_list_log):
      inds = size>=sth
      num_module1_log[cs_ind, sth_ind] = inds.sum() / (runs * len(sessions) * len(combined_stimuli[cs_ind]))
    for cth_ind, cth in enumerate(cth_list_lin):
      inds = coverage>=cth
      num_module2_lin[cs_ind, cth_ind] = inds.sum() / (runs * len(sessions) * len(combined_stimuli[cs_ind]))
    for cth_ind, cth in enumerate(cth_list_log):
      inds = coverage>=cth
      num_module2_log[cs_ind, cth_ind] = inds.sum() / (runs * len(sessions) * len(combined_stimuli[cs_ind]))
    for pth_ind, pth in enumerate(pth_list):
      inds = purity>=pth
      num_module3[cs_ind, pth_ind] = inds.sum() / (runs * len(sessions) * len(combined_stimuli[cs_ind]))
  # dotted line
  for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
    axes[0].plot(sth_list_lin, num_module1_lin[cs_ind], label=combined_stimulus_name, color='.1', marker=stimulus2marker[combined_stimulus_name], markersize=scatter_size_dict[stimulus2marker[combined_stimulus_name]], alpha=1., markerfacecolor='w')
  for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
    axes[1].plot(cth_list_lin, num_module2_lin[cs_ind], label=combined_stimulus_name, color='.1', marker=stimulus2marker[combined_stimulus_name], markersize=scatter_size_dict[stimulus2marker[combined_stimulus_name]], alpha=1., markerfacecolor='w')
  for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
    axes[2].plot(pth_list, num_module3[cs_ind], label=combined_stimulus_name, color='.1', marker=stimulus2marker[combined_stimulus_name], markersize=scatter_size_dict[stimulus2marker[combined_stimulus_name]], alpha=1., markerfacecolor='w')
  axes[0].set_xlim(right=1)
  axes[1].set_xlim(right=1)
  axes[2].set_xlim(right=1)
  axins0 = inset_axes(axes[0], loc='upper right', width="70%", height="70%")
  for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
    axins0.plot(sth_list_log, num_module1_log[cs_ind], label=combined_stimulus_name, color='.1', marker=stimulus2marker[combined_stimulus_name], markersize=scatter_size_dict[stimulus2marker[combined_stimulus_name]]/1.54, alpha=1., markerfacecolor='w')
  axins1 = inset_axes(axes[1], loc='upper right', width="60%", height="60%")
  for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
    axins1.plot(cth_list_log, num_module2_log[cs_ind], label=combined_stimulus_name, color='.1', marker=stimulus2marker[combined_stimulus_name], markersize=scatter_size_dict[stimulus2marker[combined_stimulus_name]]/1.54, alpha=1., markerfacecolor='w')
  fontsize = 20
  for axins in [axins0, axins1]:
    axins.set_xscale('log')
    axins.set_yscale('log')
    axins.set_xlim(right=1.1)
    axins.xaxis.set_tick_params(labelsize=fontsize)
    axins.yaxis.set_tick_params(labelsize=fontsize)
    for axis in ['bottom', 'left']:
      axins.spines[axis].set_linewidth(1.5)
      axins.spines[axis].set_color('k')
    axins.spines['top'].set_visible(False)
    axins.spines['right'].set_visible(False)
    axins.tick_params(width=1.5)
  # axins1.set_xlim(left=np.power(10, -1.4))
  axes[0].set_ylabel('Number of modules', fontsize=fontsize)
  for ax in axes:
    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)
    for axis in ['bottom', 'left']:
      ax.spines[axis].set_linewidth(1.5)
      ax.spines[axis].set_color('k')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=1.5)
  axes[0].set_xlabel('Threshold on normalized size', fontsize=fontsize)
  axes[1].set_xlabel('Threshold on coverage', fontsize=fontsize)
  axes[2].set_xlabel('Threshold on purity', fontsize=fontsize)
  plt.tight_layout()
  plt.savefig('./figures/figure5B.pdf', transparent=True)

################### Figure 5C ###################
def get_module_size_coverage_purity_areawise(G_dict, active_area_dict, regions, comms_dict):
  sessions, stimuli = get_session_stimulus(G_dict)
  runs = len(comms_dict[sessions[0]][stimuli[0]])
  size_dict, coverage_dict, purity_dict = [{session:{stimulus:{region:{run:[] for run in range(runs)} for region in regions} for stimulus in stimuli} for session in sessions} for _ in range(3)]
  for session_ind, session in enumerate(sessions):
    print(session)
    active_area = active_area_dict[session]
    # size_dict[session], purity_dict[session] = {}, {}
    for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
      for stimulus in combined_stimuli[cs_ind]:
        G = G_dict[session][stimulus]
        all_regions = [active_area[node] for node in G.nodes()]
        region_counts = np.array([all_regions.count(r) for r in regions])
        # size_dict[session][stimulus], purity_dict[session][stimulus] = {}, {}
        stimulus_ind = stimulus_names.index(stimulus)
        for run in range(runs):
          comms_list = comms_dict[session][stimulus]
          comms = comms_list[run]
          comms = [comm for comm in comms if len(comm) >= 4]
          for comm in comms:
            c_regions = [active_area[node] for node in comm]
            counts = np.array([c_regions.count(r) for r in regions])
            dominant_area = regions[np.argmax(counts)]
            if G.subgraph(comm).number_of_edges():
              size_dict[session][stimulus][dominant_area][run].append(len(comm) / G.number_of_nodes()) # relative size
              coverage_dict[session][stimulus][dominant_area][run].append((counts / region_counts).max())
              purity_dict[session][stimulus][dominant_area][run].append(counts.max() / counts.sum())
  return size_dict, coverage_dict, purity_dict

def plot_num_module_VSpurity_threshold_areawise(purity_dict, pth_list, regions):
  region_colors = ['#b3de69', '#80b1d3', '#fdb462', '#c3c3c3', '#fccde5', '#cec5f2']
  sessions, stimuli = get_session_stimulus(purity_dict)
  runs = len(purity_dict[sessions[0]][stimuli[0]])
  fig, axes = plt.subplots(1, 6, figsize=(4.5*6, 5), sharex=True, sharey=True)
  num_module = np.zeros((len(combined_stimulus_names), len(pth_list)))
  for r_ind, region in enumerate(regions[::-1]):
    ax = axes[r_ind]
    for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
      purity = []
      for stimulus in combined_stimuli[cs_ind]:
        for session in sessions:
          for run in range(runs):
            purity += purity_dict[session][stimulus][region][run]
      purity = np.array(purity)
      for pth_ind, pth in enumerate(pth_list):
        inds = purity>=pth
        num_module[cs_ind, pth_ind] = inds.sum() / (runs * len(sessions) * len(combined_stimuli[cs_ind]))
    # dotted line
    for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
      ax.plot(pth_list, num_module[cs_ind], label=combined_stimulus_name, color=region_colors[5-r_ind], marker=stimulus2marker[combined_stimulus_name], markersize=scatter_size_dict[stimulus2marker[combined_stimulus_name]], alpha=1., markerfacecolor='w')
    ax.set_xlim(right=1)
    ax.set_title(region_labels[5-r_ind], fontsize=24)
  
  fontsize = 25
  axes[0].set_ylabel('Number of modules', fontsize=fontsize)
  for ax in axes:
    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)
    for axis in ['bottom', 'left']:
      ax.spines[axis].set_linewidth(1.5)
      ax.spines[axis].set_color('k')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=1.5)
    ax.set_xlabel('')
  plt.tight_layout(rect=[0., 0., 1., .97])
  plt.savefig('./figures/figure5C.pdf', transparent=True)

################### Figure 5D,E ###################
def get_data_module_per_area(G_dict, active_area_dict, signal_correlation_dict, comms_dict, regions, combined_stimulus_name):
  within_comm_dict, across_comm_dict, within_pos_density, across_pos_density, within_pos_frac, across_pos_frac, within_neg_frac, across_neg_frac = {r:[] for r in regions}, {r:[] for r in regions}, {r:[] for r in regions}, {r:[] for r in regions}, {r:[] for r in regions}, {r:[] for r in regions}, {r:[] for r in regions}, {r:[] for r in regions}
  rand_across_comm_list, rand_across_pos_density_list, rand_across_pos_frac_list, rand_across_neg_frac_list = [], [], [], []
  cs_ind = combined_stimulus_names.index(combined_stimulus_name)
  for stimulus in combined_stimuli[cs_ind]:
  # for stimulus in stimulus_by_type[3]: # only natural stimuli
    print(stimulus)
    stimulus_ind = stimulus_names.index(stimulus)
    for session_ind, session in enumerate(session_ids):
      signal_correlation = signal_correlation_dict[session][combine_stimulus(stimulus)[1]]
      active_area = active_area_dict[session]
      node_idx = sorted(active_area.keys())
      G = G_dict[session][stimulus].copy()
      nx.set_node_attributes(G, active_area_dict[session], "area")
      for run in range(len(comms_dict[session][stimulus][0])):
        comms_list = comms_dict[session][stimulus]
        comms = comms_list[run]
        comm_areas = []
        large_comms = [comm for comm in comms if len(comm)>=4]
        node_to_community = comm2partition(comms)
        for comm in large_comms:
          c_regions = [active_area_dict[session][node] for node in comm]
          counts = np.array([c_regions.count(r) for r in regions])
          dominant_area = regions[np.argmax(counts)]
          comm_areas.append(dominant_area)
        nodes2plot = large_comms
        pos_density_mat, pos_frac_mat, neg_frac_mat = np.zeros((len(large_comms), len(large_comms))), np.zeros((len(large_comms), len(large_comms))), np.zeros((len(large_comms), len(large_comms)))
        for nodes1, nodes2 in itertools.permutations(nodes2plot, 2):
          # pos_density_mat[nodes2plot.index(nodes1), nodes2plot.index(nodes2)] = (sum(1 for e in nx.edge_boundary(G, nodes1, nodes2) if G[e[0]][e[1]]['weight']>0) + sum(1 for e in nx.edge_boundary(G, nodes2, nodes1) if G[e[0]][e[1]]['weight']>0)) / (len(nodes1) * len(nodes2)) # only positive edges
          pos_density_mat[nodes2plot.index(nodes1), nodes2plot.index(nodes2)] = (sum(1 for e in nx.edge_boundary(G, nodes1, nodes2)) + sum(1 for e in nx.edge_boundary(G, nodes2, nodes1) if G[e[0]][e[1]]['weight']>0)) / (len(nodes1) * len(nodes2)) # all edges
          if (sum(1 for _ in nx.edge_boundary(G, nodes1, nodes2)) + sum(1 for _ in nx.edge_boundary(G, nodes2, nodes1))):
            pos_frac_mat[nodes2plot.index(nodes1), nodes2plot.index(nodes2)] = (sum(1 for e in nx.edge_boundary(G, nodes1, nodes2) if G[e[0]][e[1]]['weight']>0) + sum(1 for e in nx.edge_boundary(G, nodes2, nodes1) if G[e[0]][e[1]]['weight']>0)) / (sum(1 for _ in nx.edge_boundary(G, nodes1, nodes2)) + sum(1 for _ in nx.edge_boundary(G, nodes2, nodes1)))
            neg_frac_mat[nodes2plot.index(nodes1), nodes2plot.index(nodes2)] = (sum(1 for e in nx.edge_boundary(G, nodes1, nodes2) if G[e[0]][e[1]]['weight']<0) + sum(1 for e in nx.edge_boundary(G, nodes2, nodes1) if G[e[0]][e[1]]['weight']<0)) / (sum(1 for _ in nx.edge_boundary(G, nodes1, nodes2)) + sum(1 for _ in nx.edge_boundary(G, nodes2, nodes1)))
          else:
            pos_frac_mat[nodes2plot.index(nodes1), nodes2plot.index(nodes2)] = -1 # for modules with no connections
            neg_frac_mat[nodes2plot.index(nodes1), nodes2plot.index(nodes2)] = -1 # for modules with no connections
        for region in regions:
          if comm_areas.count(region) >= 1:
            indx = np.where(np.array(comm_areas)==region)[0]
            for ind in indx:
              inds = np.array(list(large_comms[ind]))
              node_inds = np.array([node_idx.index(node) for node in inds])
              within_comm_dict[region].append(np.nanmean(signal_correlation[node_inds[:,None], node_inds][np.triu_indices(len(node_inds), 1)].tolist()))
              # within_pos_density[region].append(sum(1 for _, _, w in G.subgraph(large_comms[ind]).edges(data='weight') if w>0) / (len(large_comms[ind]) * (len(large_comms[ind])-1))) # only positive edges
              within_pos_density[region].append(G.subgraph(large_comms[ind]).number_of_edges() / (len(large_comms[ind]) * (len(large_comms[ind])-1))) # all edges
              # if sum(1 for _ in G.subgraph(large_comms[ind]).edges()):
              within_pos_frac[region].append(sum(1 for _, _, w in G.subgraph(large_comms[ind]).edges(data='weight') if w>0) / sum(1 for _ in G.subgraph(large_comms[ind]).edges()))
              within_neg_frac[region].append(sum(1 for _, _, w in G.subgraph(large_comms[ind]).edges(data='weight') if w<0) / sum(1 for _ in G.subgraph(large_comms[ind]).edges()))
          if comm_areas.count(region) > 1:
            indx = np.where(np.array(comm_areas)==region)[0]
            for ind1, ind2 in itertools.combinations(indx, 2):
              inds1, inds2 = np.array(list(large_comms[ind1])), np.array(list(large_comms[ind2]))
              node_inds1, node_inds2 = np.array([node_idx.index(node) for node in inds1]), np.array([node_idx.index(node) for node in inds2])
              across_comm_dict[region].append(np.nanmean(signal_correlation[node_inds1[:,None], node_inds2].flatten().tolist()))
              across_pos_density[region].append(pos_density_mat[ind1, ind2])
              across_pos_frac[region].append(pos_frac_mat[ind1, ind2])
              across_neg_frac[region].append(neg_frac_mat[ind1, ind2]) # could be -1 for modules with no connections
        for nodei, nodej in itertools.combinations(node_idx, 2): # for all neurons
          if node_to_community[nodei] != node_to_community[nodej]:
            rand_across_comm_list.append(signal_correlation[node_idx.index(nodei), node_idx.index(nodej)])
        for commi, commj in itertools.combinations(large_comms, 2): # for all neurons
          rand_across_pos_density_list.append((sum(1 for e in nx.edge_boundary(G, commi, commj) if G[e[0]][e[1]]['weight']>0) + sum(1 for e in nx.edge_boundary(G, commj, commi) if G[e[0]][e[1]]['weight']>0)) / (len(commi) * len(commj)))
          summ = (sum(1 for _ in nx.edge_boundary(G, commi, commj)) + sum(1 for _ in nx.edge_boundary(G, commj, commi)))
          if summ:
            rand_across_pos_frac_list.append((sum(1 for e in nx.edge_boundary(G, commi, commj) if G[e[0]][e[1]]['weight']>0) + sum(1 for e in nx.edge_boundary(G, commj, commi) if G[e[0]][e[1]]['weight']>0)) / summ)
            rand_across_neg_frac_list.append((sum(1 for e in nx.edge_boundary(G, commi, commj) if G[e[0]][e[1]]['weight']<0) + sum(1 for e in nx.edge_boundary(G, commj, commi) if G[e[0]][e[1]]['weight']<0)) / summ)
          else:
            rand_across_pos_frac_list.append(-1)
            rand_across_neg_frac_list.append(-1) # for modules with no connections
  df = pd.DataFrame()
  for region in regions:
    print(region)
    within_comm, across_comm, within_posD, across_posD, within_posF, across_posF, within_negF, across_negF =  within_comm_dict[region], across_comm_dict[region], within_pos_density[region], across_pos_density[region], within_pos_frac[region], across_pos_frac[region], within_neg_frac[region], across_neg_frac[region]
    within_comm, across_comm = [e for e in within_comm if not np.isnan(e)], [e for e in across_comm if not np.isnan(e)] # remove nan values
    across_posF, across_negF = [e for e in across_posF if e != -1], [e for e in across_negF if e != -1] # remove nan values
    df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(within_comm)[:,None], np.array(['signal correlation'] * len(within_comm))[:,None], np.array(['within community'] * len(within_comm))[:,None], np.array([region] * len(within_comm))[:,None]), 1), columns=['data', 'dtype', 'type', 'region'])], ignore_index=True)
    df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(across_comm)[:,None], np.array(['signal correlation'] * len(across_comm))[:,None], np.array(['across community'] * len(across_comm))[:,None], np.array([region] * len(across_comm))[:,None]), 1), columns=['data', 'dtype', 'type', 'region'])], ignore_index=True)
    df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(within_posD)[:,None], np.array(['positive density'] * len(within_posD))[:,None], np.array(['within community'] * len(within_posD))[:,None], np.array([region] * len(within_posD))[:,None]), 1), columns=['data', 'dtype', 'type', 'region'])], ignore_index=True)
    df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(across_posD)[:,None], np.array(['positive density'] * len(across_posD))[:,None], np.array(['across community'] * len(across_posD))[:,None], np.array([region] * len(across_posD))[:,None]), 1), columns=['data', 'dtype', 'type', 'region'])], ignore_index=True)
    df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(within_posF)[:,None], np.array(['positive fraction'] * len(within_posF))[:,None], np.array(['within community'] * len(within_posF))[:,None], np.array([region] * len(within_posF))[:,None]), 1), columns=['data', 'dtype', 'type', 'region'])], ignore_index=True)
    df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(across_posF)[:,None], np.array(['positive fraction'] * len(across_posF))[:,None], np.array(['across community'] * len(across_posF))[:,None], np.array([region] * len(across_posF))[:,None]), 1), columns=['data', 'dtype', 'type', 'region'])], ignore_index=True)
    df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(within_negF)[:,None], np.array(['negative fraction'] * len(within_negF))[:,None], np.array(['within community'] * len(within_negF))[:,None], np.array([region] * len(within_negF))[:,None]), 1), columns=['data', 'dtype', 'type', 'region'])], ignore_index=True)
    df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(across_negF)[:,None], np.array(['negative fraction'] * len(across_negF))[:,None], np.array(['across community'] * len(across_negF))[:,None], np.array([region] * len(across_negF))[:,None]), 1), columns=['data', 'dtype', 'type', 'region'])], ignore_index=True)
  df['data'] = pd.to_numeric(df['data'])
  return df, rand_across_comm_list, rand_across_pos_density_list, rand_across_pos_frac_list, rand_across_neg_frac_list

def plot_data_within_across_comm_same_area_significance(df, data_list, regions, name, combined_stimulus_name):
  fig, ax = plt.subplots(1,1, sharex=True, sharey=True, figsize=(1.37*6, 4))
  # palette = [[plt.cm.tab20b(i) for i in range(4)][i] for i in [0, 3]]
  palette = ['w','k']
  data = df[df['dtype']==name]
  barplot = sns.barplot(x='region', y='data', hue="type", hue_order=['across community','within community'], palette=palette, ec='k', linewidth=2., data=data, ax=ax, capsize=.05, width=0.6)
  if 'fraction' in name:
    ax.axhline(y=np.nanmean([e for e in data_list if e != -1]), linestyle='--', linewidth=3, color='.2')
  else:
    ax.axhline(y=np.nanmean(data_list), linestyle='--', linewidth=3, color='.2')
  ax.yaxis.set_tick_params(labelsize=30)
  plt.setp(ax.get_legend().get_title(), fontsize='0') # for legend title
  plt.xticks(range(len(regions)), region_labels, fontsize=30)
  ax.xaxis.set_tick_params(length=0)
  for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2.5)
    ax.spines[axis].set_color('k')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.tick_params(width=2.5)
  ax.set_xlabel('')
  ax.set_ylabel('')
  handles, labels = ax.get_legend_handles_labels()
  ax.legend(handles, [], fontsize=0)
  if name == 'signal correlation':
    ylabel = name.capitalize()
  elif name == 'positive density':
    ylabel = 'Connection probability'
  elif name == 'positive fraction':
    ylabel = 'Excitatory fraction'
  elif name == 'negative fraction':
    ylabel = 'Inhibitory fraction'
  ax.set_ylabel(ylabel, fontsize=30)
  # add significance annotation
  alpha_list = [.0001, .001, .01, .05]
  maxx = 0
  for r_ind, region in enumerate(regions):
    within_comm, across_comm = data[(data.region==region)&(data.type=='within community')]['data'].values.flatten(), data[(data.region==region)&(data.type=='across community')]['data'].values.flatten()
    within_sr, across_sr = confidence_interval(within_comm)[1], confidence_interval(across_comm)[1]
    maxx = max(within_sr, across_sr) if max(within_sr, across_sr) > maxx else maxx
  h, l = .05 * maxx, .05 * maxx
  for r_ind, region in enumerate(regions):
    within_comm, across_comm = data[(data.region==region)&(data.type=='within community')]['data'].values.flatten(), data[(data.region==region)&(data.type=='across community')]['data'].values.flatten()
    # print p-values of normality test
    print(combined_stimulus_name.replace('\n', ' '), shapiro(within_comm)[1], shapiro(across_comm)[1])
    if name == 'negative fraction':
      _, p = ttest_ind(within_comm, across_comm, alternative='less')
    else:
      _, p = ttest_ind(within_comm, across_comm, alternative='greater')
    diff_star = '*' * (len(alpha_list) - bisect(alpha_list, p)) if len(alpha_list) > bisect(alpha_list, p) else 'ns'
    within_sr, across_sr = confidence_interval(within_comm)[1], confidence_interval(across_comm)[1]
    within_sr = within_sr + h
    across_sr = across_sr + h
    annot_difference(diff_star, -.15 + r_ind, .15 + r_ind, max(within_sr, across_sr), l, 2.5, 28, ax=ax)
  ax.invert_xaxis()
  plt.tight_layout()
  loc = 'D' if combined_stimulus_name == 'Natural\nscenes' else 'E'
  if name == 'signal correlation':
    fname = '1'
  elif name == 'positive density':
    fname = '2'
  elif name == 'positive fraction':
    fname = '3'
  elif name == 'negative fraction':
    fname = '4'
  plt.savefig('./figures/figure5{}_{}.pdf'.format(loc, fname), transparent=True)

# Other functions
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

################### an efficient way of finding motifs
def _tricode(G, v, u, w):
  """Returns the integer code of the given triad.

  This is some fancy magic that comes from Batagelj and Mrvar's paper. It
  treats each edge joining a pair of `v`, `u`, and `w` as a bit in
  the binary representation of an integer.

  """
  combos = ((v, u, 1), (u, v, 2), (v, w, 4), (w, v, 8), (u, w, 16),
            (w, u, 32))
  return sum(x for u, v, x in combos if v in G[u])

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
  return triad_nodes
  
# get the standard edges of a motif type
def find_all_unique_smotifs(edges, weight='weight'):
  em = iso.numerical_edge_match(weight, 1)
  G0 = nx.DiGraph()
  all_ws = list(itertools.product([1, -1], repeat=len(edges))) # either 1 or -1, all possible combinations
  edge2add = [(*edge, w) for edge, w in zip(edges, all_ws[0])]
  G0.add_weighted_edges_from((edge2add), weight=weight)
  unique_sms = [G0]
  for all_w in tqdm(all_ws[1:], total=len(all_ws)-1, disable=True):
    G = nx.DiGraph()
    edge2add = [(*edge, w) for edge, w in zip(edges, all_w)]
    G.add_weighted_edges_from((edge2add), weight=weight)
    is_unique = True
    for ex_G in unique_sms:
      if nx.is_isomorphic(G, ex_G, edge_match=em):  # match weight
        is_unique = False
        break
    if is_unique:
      unique_sms.append(G)
  return unique_sms

def get_edges_sms(motif_type, weight='weight'):
  if motif_type == '021D':
    edges = [(0, 1), (0, 2)]
  elif motif_type == '021U':
    edges = [(0, 1), (2, 1)]
  elif motif_type == '021C':
    edges = [(0, 1), (1, 2)]
  elif motif_type == '111D':
    edges = [(0, 1), (1, 2), (2, 1)]
  elif motif_type == '111U':
    edges = [(0, 1), (0, 2), (2, 0)]
  elif motif_type == '030T':
    edges = [(0, 1), (1, 2), (0, 2)]
  elif motif_type == '030C':
    edges = [(0, 1), (1, 2), (2, 0)]
  elif motif_type == '201':
    edges = [(0, 1), (1, 0), (0, 2), (2, 0)]
  elif motif_type == '120D':
    edges = [(0, 1), (0, 2), (1, 2), (2, 1)]
  elif motif_type == '120U':
    edges = [(0, 2), (1, 2), (0, 1), (1, 0)]
  elif motif_type == '120C':
    edges = [(0, 1), (1, 2), (0, 2), (2, 0)]
  elif motif_type == '210':
    edges = [(0, 1), (1, 2), (2, 1), (0, 2), (2, 0)]
  elif motif_type == '300':
    edges = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 0), (0, 2)]
  unique_sms = find_all_unique_smotifs(edges, weight=weight)
  return edges, unique_sms

# get sign of the motif
def to_unique_form(signed_motif, unique_sms, weight='weight'):
  em = iso.numerical_edge_match(weight, 1)
  for unique_sm in unique_sms:
    if nx.is_isomorphic(signed_motif, unique_sm, edge_match=em):  # match weight
      unique_form = unique_sm
      break
  return unique_form

def get_motif_sign(origin_motif, edges, unique_sms, weight='weight'):
  motif = origin_motif.copy()
  for u, v, w in motif.edges(data=weight):
    motif[u][v][weight] = 1 if w > 0 else -1
  unique_form = to_unique_form(motif, unique_sms, weight=weight)
  signs = [unique_form[edge[0]][edge[1]][weight] for edge in edges]
  signs = ''.join(['+' if sign > 0 else '-' for sign in signs])
  return signs

def confidence_interval(data, confidence=0.95):
  a = 1.0 * np.array(data)
  n = len(a)
  m, se = np.mean(a), stats.sem(a)
  h = se * stats.t.ppf((1 + confidence) / 2., n-1)
  return m-h, m+h

def nonpara_confidence_interval(data, confidence_level=0.95):
  # Choose the desired confidence level
  alpha = 1 - confidence_level
  # Generate 10,000 bootstrap samples and calculate the median for each one
  bootstrap_results = np.array([np.median(np.random.choice(data, size=len(data), replace=True)) for i in range(10000)])
  # Calculate the lower and upper bounds of the confidence interval
  lower_bound = np.percentile(bootstrap_results, alpha/2 * 100)
  upper_bound = np.percentile(bootstrap_results, (1 - alpha/2) * 100)
  return lower_bound, upper_bound

def annot_difference(star, x1, x2, y, h, lw=2.5, fontsize=14, col='k', ax=None):
  ax = plt.gca() if ax is None else ax
  ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=lw, c=col)
  ax.text((x1+x2)*.5, y+h, star, ha='center', va='bottom', color=col, fontsize=fontsize)
