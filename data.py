#%%
from collections import Counter
import os
import re
import sys
import seaborn as sns
# import shutil
from networkx.algorithms.efficiency_measures import efficiency
import numpy as np
import pandas as pd
import time
import community
from matplotlib import pyplot as plt
import networkx as nx
# from requests.sessions import session
from netgraph import Graph
import xarray as xr
from sklearn.metrics.cluster import normalized_mutual_info_score
from statsmodels.tsa.stattools import grangercausalitytests
# from CPAC.series_mod import cond_entropy
# from CPAC.series_mod import entropy
from scipy.spatial.distance import pdist, squareform, cosine
from sklearn.metrics.pairwise import cosine_similarity
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
data_directory = './data/ecephys_cache_dir'
manifest_path = os.path.join(data_directory, "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
sessions = cache.get_session_table()
# filtered_sessions = sessions[(sessions.sex == 'M') & \
#                              (sessions.full_genotype.str.find('Sst') > -1) & \
#                              (sessions.session_type == 'brain_observatory_1.1') & \
#                              (['VISl' in acronyms for acronyms in 
#                                sessions.ecephys_structure_acronyms])]
probes = cache.get_probes()
channels = cache.get_channels()
units = cache.get_units()
num_sessions = len(sessions)
num_neurons = len(units)
num_probes = len(units['name'].unique())

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

def get_spiking_sequence(session_id, stimulus_name, structure_acronym):
  session = cache.get_session_data(session_id)
  print(session.metadata)
  print(session.structurewise_unit_counts)
  presentations = session.get_stimulus_table(stimulus_name)
  units = session.units[session.units["ecephys_structure_acronym"]==structure_acronym]
  time_step = 0.001
  time_bins = np.arange(0, 2.0 + time_step, time_step)
  histograms = session.presentationwise_spike_counts(
      stimulus_presentation_ids=presentations.index.values,  
      bin_edges=time_bins,
      unit_ids=units.index.values)
  return histograms

def get_whole_spiking_sequence(session_id, stimulus_name):
  session = cache.get_session_data(session_id)
  # print(session.metadata)
  # print(session.structurewise_unit_counts)
  presentations = session.get_stimulus_table(stimulus_name)
  units = session.units
  time_step = 0.001
  time_bins = np.arange(0, 3.0 + time_step, time_step)
  histograms = session.presentationwise_spike_counts(
      stimulus_presentation_ids=presentations.index.values,  
      bin_edges=time_bins,
      unit_ids=units.index.values)
  return histograms

def remove_outlier(array):
  mean = np.mean(array)
  standard_deviation = np.std(array)
  distance_from_mean = abs(array - mean)
  max_deviations = 2
  not_outlier = distance_from_mean < max_deviations * standard_deviation
  return array[not_outlier]

def get_regions_spiking_sequence(session_id, stimulus_name, regions, resolution):
  session = cache.get_session_data(session_id,
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
    stim_table = session.get_stimulus_table([stimulus_name])
    stim_table=stim_table.rename(columns={"start_time": "Start", "stop_time": "End"})
    if 'natural_movie' in stimulus_name:
        frame_times = stim_table.End-stim_table.Start
        print('frame rate:', 1/np.mean(frame_times), 'Hz', np.mean(frame_times))
        # stim_table.to_csv(output_path+'stim_table_'+stimulus_name+'.csv')
        # chunch each movie clip
        stim_table = stim_table[stim_table.frame==0]
        stim_table = stim_table.drop(['End'], axis=1)
        duration = np.mean(remove_outlier(np.diff(stim_table.Start.values))[:10]) - 1e-4
    elif stimulus_name=='spontaneous':
        index = np.where(stim_table.duration>=20)[0]
        if len(index): # only keep the longest spontaneous; has to be longer than 20 sec
            duration=20
            stimulus_presentation_ids = stim_table.index[index]
    else:
        ISI = np.mean(session.get_inter_presentation_intervals_for_stimulus([stimulus_name]).interval.values)
        duration = round(np.mean(stim_table.duration.values), 2)+ISI
    if stimulus_name == 'gabors':
      duration -= 0.02
    try: stimulus_presentation_ids
    except NameError: stimulus_presentation_ids = stim_table.index.values
    #binarize tensor
    # binarize with 1 second bins
    time_bin_edges = np.linspace(0, duration, int(duration / resolution)+1)

    # and get a set of units with only decent snr
    #decent_snr_unit_ids = session.units[
    #    session.units['snr'] >= 1.5
    #].index.values
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

def MI(matrix):
  n=matrix.shape[0]
  MI=np.zeros((n,n))
  for i in range(n):
      for j in range(n):
          MI[i,j]=normalized_mutual_info_score(matrix[i,:], matrix[j,:])
  return MI

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

def compute_adj(path, measure):
  for file in os.listdir(path):
    print(file)
    if file.endswith(".nc"):
        data = xr.open_dataset(os.path.join(path, file)).to_array()
        # mouseID = file.replace('.nc', '').split('_')[0]
        # stimulus = file.replace('.nc', '').replace(mouseID + '_', '')
        if len(data.data.shape) > 2:
          sequences = data.data.squeeze().T
        else:
          sequences = data.data.T
        sequences = np.nan_to_num(sequences)
        if measure == 'pearson':
          adj_mat = np.corrcoef(sequences)
        elif measure == 'cosine':
          adj_mat = cosine_similarity(sequences)
        elif measure == 'correlation':
          adj_mat = squareform(pdist(sequences, 'correlation'))
        elif measure == 'MI':
          adj_mat = MI(sequences)
        elif measure == 'causality':
          adj_mat = granger_causality(sequences)
        else:
          sys.exit('Unrecognized measure value!!! Choices: pearson, cosin, correlation.')
        adj_mat = np.nan_to_num(adj_mat)
        np.fill_diagonal(adj_mat, 0)
        np.save(os.path.join(path.replace('spiking_sequence', '{}_matrix'.format(measure)), file.replace('.nc', '.npy')), adj_mat)

def corr_mat(sequences, measure, threshold=0.5, percentile=90):
  if measure == 'pearson':
    adj_mat = np.corrcoef(sequences)
  elif measure == 'cosine':
    adj_mat = cosine_similarity(sequences)
  elif measure == 'correlation':
    adj_mat = squareform(pdist(sequences, 'correlation'))
  elif measure == 'MI':
    adj_mat = MI(sequences)
  elif measure == 'causality':
    adj_mat = granger_causality(sequences)
  else:
    sys.exit('Unrecognized measure value!!! Choices: pearson, cosin, correlation.')
  adj_mat = np.nan_to_num(adj_mat)
  np.fill_diagonal(adj_mat, 0)
  if measure == 'pearson':
    adj_mat[adj_mat < threshold] = 0
  else:
    adj_mat[np.where(adj_mat<np.nanpercentile(np.abs(adj_mat), percentile))] = 0
  return adj_mat

def generate_graph(sequences, measure, cc=False, weight=False, threshold=0.5, percentile=90):
  adj_mat = corr_mat(sequences, measure, threshold, percentile)
  if not weight:
    adj_mat[adj_mat.nonzero()] = 1
  G = nx.from_numpy_array(adj_mat) # same as from_numpy_matrix
  if cc: # extract the largest (strongly) connected components
    if np.allclose(adj_mat, adj_mat.T, rtol=1e-05, atol=1e-08): # if the matrix is symmetric
      largest_cc = max(nx.connected_components(G), key=len)
    else:
      largest_cc = max(nx.strongly_connected_components(G), key=len)
    G = nx.subgraph(G, largest_cc)
  return G

def generate_region_graph(sequences, indices, measure, cc=True, weight=False, threshold=0.3, percentile=90):
  sequences = sequences[indices, :]
  # sequences = np.nan_to_num(sequences)
  adj_mat = corr_mat(sequences, measure, threshold, percentile)
  if not weight:
    adj_mat[adj_mat.nonzero()] = 1
  G = nx.from_numpy_array(adj_mat) # same as from_numpy_matrix
  if cc: # extract the largest (strongly) connected components
    if np.allclose(adj_mat, adj_mat.T, rtol=1e-05, atol=1e-08): # if the matrix is symmetric
      largest_cc = max(nx.connected_components(G), key=len)
    else:
      largest_cc = max(nx.strongly_connected_components(G), key=len)
    G = nx.subgraph(G, largest_cc)
  return G

def load_npy_as_graph(path):
  G_dict = {}
  for file in os.listdir(path):
    print(file)
    if file.endswith(".npy"):
        data = np.load(os.path.join(path, file))
        mouseID = file.split('_')[0].replace('mouse', '')
        stimulus = file.replace('mouse{}_adjacency_matrix_RF_'.format(mouseID), '').replace('mouse{}_adjacency_matrix_RF'.format(mouseID), '').replace('.npy', '')
        instruction = pd.read_csv(os.path.join(path, 'mouse{}_adjacency_matrix_RF.csv'.format(mouseID)), index_col=0)
        areas = np.unique(instruction['area'])
        adj_mat = [data[np.where(instruction['area']==area)[0][:,None], np.where(instruction['area']==area)[0]] for area in areas]
        if stimulus == '':
          stimulus = 'None' 
        if not mouseID in G_dict:
          G_dict[mouseID] = {}
        if not stimulus in G_dict[mouseID]:
          G_dict[mouseID][stimulus] = {}
        for ind, area in enumerate(areas):
          if adj_mat[ind].shape[0] > 10:
            adj_mat[ind][np.where(adj_mat[ind]<np.abs(adj_mat[ind]).mean())] = 0
            G_dict[mouseID][stimulus][area] = nx.from_numpy_array(adj_mat[ind])
  return G_dict

def load_npy_as_graph_whole(path, threshold=0.4, percentile=99, unweighted=False):
  G_dict = {}
  area_dict = {}
  for file in os.listdir(path):
    print(file)
    if file.endswith(".npy"):
        data = np.load(os.path.join(path, file))
        mouseID = file.split('_')[0].replace('mouse', '')
        stimulus = file.replace('mouse{}_adjacency_matrix_RF_'.format(mouseID), '').replace('mouse{}_adjacency_matrix_RF'.format(mouseID), '').replace('.npy', '')
        instruction = pd.read_csv(os.path.join(path, 'mouse{}_adjacency_matrix_RF.csv'.format(mouseID)), index_col=0)
        if not mouseID in area_dict:
          area_dict[mouseID] = {}
        for i in range(instruction.shape[0]):
          area_dict[mouseID][i] = instruction['area'].iloc[i]
        adj_mat = data
        if stimulus == '':
          stimulus = 'None' 
        if not mouseID in G_dict:
          G_dict[mouseID] = {}
        adj_mat[np.where(adj_mat<np.nanpercentile(np.abs(adj_mat), percentile))] = 0
        if unweighted:
          adj_mat[np.nonzero(adj_mat)] = 1
        G_dict[mouseID][stimulus] = nx.from_numpy_array(adj_mat, create_using=nx.DiGraph)
  return G_dict, area_dict, adj_mat

def load_allen_npy_as_graph_whole(path, filter_region=30, threshold=0.4, percentile=90, unweighted=True):
  G_dict = {}
  area_dict = {}
  for file in os.listdir(path):
    print(file)
    if file.endswith(".npy"):
      data = np.load(os.path.join(path, file))
      mouseID = int(file.split('_')[0])
      stimulus = file.replace('{}_'.format(mouseID), '').replace('.npy', '')
      instruction = units[units['ecephys_session_id']==mouseID]['ecephys_structure_acronym']
      unique, counts = np.unique(instruction, return_counts=True)
      cnt = dict(zip(unique, counts))
      big_region = {key: value for key, value in cnt.items() if value >= filter_region}
      instruction = instruction.reset_index()
      instruction = instruction[np.isin(instruction['ecephys_structure_acronym'], list(big_region))]
      mat_ind = np.array(instruction.index)
      data = data[mat_ind[:, None], mat_ind]
      if not mouseID in area_dict:
        area_dict[mouseID] = {}
      for i in range(instruction.shape[0]):
        area_dict[mouseID][i] = instruction['ecephys_structure_acronym'].iloc[i]
      if not mouseID in G_dict:
        G_dict[mouseID] = {}
      adj_mat = data
      if 'pearson' in path:
        adj_mat[adj_mat < threshold] = 0
      else:
        adj_mat[np.where(adj_mat<np.nanpercentile(np.abs(adj_mat), percentile))] = 0
      if unweighted:
        adj_mat[np.nonzero(adj_mat)] = 1
      G_dict[mouseID][stimulus] = nx.from_numpy_array(adj_mat, create_using=nx.Graph)
  return G_dict, area_dict

def load_nc_as_graph(path, measure, threshold):
  G_dict = {}
  for file in os.listdir(path):
    print(file)
    if file.endswith(".nc"):
        data = xr.open_dataset(os.path.join(path, file)).to_array()
        if len(data.data.shape) > 2:
          sequences = data.data.squeeze().T
        else:
          sequences = data.data.T
        structure_acronym = file.replace('.nc', '').split('_')[-1]
        stimulus_name = file.replace('.nc', '').replace('_' + structure_acronym, '')
        if not stimulus_name in G_dict:
          G_dict[stimulus_name] = {}
        G_dict[stimulus_name][structure_acronym] = generate_graph(sequences, measure=measure, cc=True, threshold=threshold)
  return G_dict

def load_nc_as_graph_whole(path, measure, threshold, percentile):
  G_dict = {}
  for file in os.listdir(path):
    print(file)
    if file.endswith(".nc"):
        data = xr.open_dataset(os.path.join(path, file)).to_array()
        if len(data.data.shape) > 2:
          sequences = data.data.squeeze().T
        else:
          sequences = data.data.T
        mouseID = file.split('_')[0]
        stimulus_name = file.replace('.nc', '').replace(mouseID + '_', '')
        if not mouseID in G_dict:
          G_dict[mouseID] = {}
        G_dict[mouseID][stimulus_name] = generate_graph(sequences, measure=measure, cc=True, threshold=threshold, percentile=percentile)
  return G_dict

def load_npy_regions_as_graph_whole(directory, regions, weight, measure, threshold, percentile):
  G_dict = {}
  area_dict = {}
  files = os.listdir(directory)
  files.sort(key=lambda x:int(x[:9]))
  for file in files:
    print(file)
    if file.endswith(".npy"):
      sequences = np.load(os.path.join(directory, file)).T
      # data = xr.open_dataset(os.path.join(directory, file)).to_array()
      # if len(data.data.shape) > 2:
      #   sequences = data.data.squeeze().T
      # else:
      #   sequences = data.data.T
      mouseID = file.split('_')[0]
      if not mouseID in G_dict:
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
        if set(instruction.unique()) == set(regions): # if the mouse has all regions recorded
          G_dict[mouseID] = {}
        instruction = instruction.reset_index()
      stimulus_name = file.replace('.npy', '').replace(mouseID + '_', '')
      G_dict[mouseID][stimulus_name] = generate_graph(sequences, measure=measure, cc=True, weight=weight, threshold=threshold, percentile=percentile)
      if not mouseID in area_dict:
        area_dict[mouseID] = {}
      for i in range(instruction.shape[0]):
        area_dict[mouseID][i] = instruction.ccf.iloc[i]
  return G_dict, area_dict

def load_npy_regions_nsteps_as_graph(directory, n, ind, regions, weight, measure, threshold, percentile):
  G_dict = {}
  area_dict = {}
  files = os.listdir(directory)
  files.sort(key=lambda x:int(x[:9]))
  for file in files:
    print(file)
    if file.endswith(".npy"):
      sequences = np.load(os.path.join(directory, file)).T
      sub_sequences = [sequences[:, x:x+n] for x in range(0, sequences.shape[1], n)]
      sub_sequence = sub_sequences[ind] if ind < len(sub_sequences) else sub_sequences[-1]
      print(ind if ind < len(sub_sequences) else len(sub_sequences)-1)
      # data = xr.open_dataset(os.path.join(directory, file)).to_array()
      # if len(data.data.shape) > 2:
      #   sequences = data.data.squeeze().T
      # else:
      #   sequences = data.data.T
      mouseID = file.split('_')[0]
      if not mouseID in G_dict:
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
        if set(instruction.unique()) == set(regions): # if the mouse has all regions recorded
          G_dict[mouseID] = {}
        instruction = instruction.reset_index()
      stimulus_name = file.replace('.npy', '').replace(mouseID + '_', '')
      G_dict[mouseID][stimulus_name] = generate_graph(sub_sequence, measure=measure, cc=False, weight=weight, threshold=threshold, percentile=percentile)
      if not mouseID in area_dict:
        area_dict[mouseID] = {}
      for i in range(instruction.shape[0]):
        area_dict[mouseID][i] = instruction.ccf.iloc[i]
  return G_dict, area_dict

def load_nc_regions_as_n_graphs(directory, n, regions, weight, measure, threshold, percentile):
  G_sub_dict = dict.fromkeys(range(n), {})
  area_dict = {}
  files = os.listdir(directory)
  files.sort(key=lambda x:int(x[:9]))
  for file in files:
    print(file)
    if file.endswith(".nc"):
      data = xr.open_dataset(os.path.join(directory, file)).to_array()
      if len(data.data.shape) > 2:
        sequences = data.data.squeeze().T
      else:
        sequences = data.data.T
      sub_sequences = np.array_split(sequences, n, axis=1)
      mouseID = file.split('_')[0]
      if not mouseID in G_sub_dict[0]:
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
        if set(instruction.unique()) == set(regions): # if the mouse has all regions recorded
          for i in range(n):
            G_sub_dict[i][mouseID] = {}
        instruction = instruction.reset_index()
      stimulus_name = file.replace('.nc', '').replace(mouseID + '_', '')
      for i in range(n):
        G_sub_dict[i][mouseID][stimulus_name] = generate_graph(sub_sequences[i], measure=measure, cc=True, weight=weight, threshold=threshold, percentile=percentile)
      if not mouseID in area_dict:
        area_dict[mouseID] = {}
      for unit_ind in range(instruction.shape[0]):
        area_dict[mouseID][unit_ind] = instruction.ccf.iloc[unit_ind]
  return G_sub_dict, area_dict

def load_nc_as_two_graphs(directory, origin_units, regions, weight, measure, threshold, percentile):
  units = origin_units.copy()
  G_dict1, G_dict2 = {}, {}
  area_dict = {}
  unit_regions = {}
  files = os.listdir(directory)
  files.sort(key=lambda x:int(x[:9]))
  for file in files:
    print(file)
    if file.endswith(".nc"):
        data = xr.open_dataset(os.path.join(directory, file)).to_array()
        if len(data.data.shape) > 2:
          sequences = data.data.squeeze().T
        else:
          sequences = data.data.T
        s1, s2 = np.array_split(sequences, 2, axis=1)
        mouseID = file.split('_')[0]
        if not mouseID in G_dict1:
          unit = units[units['ecephys_session_id']==int(mouseID)]['ecephys_structure_acronym'].reset_index()
          unit_regions[mouseID] = unit[np.isin(unit['ecephys_structure_acronym'], regions)]
          if set(unit_regions[mouseID]['ecephys_structure_acronym'].unique()) == set(regions): # if the mouse has all regions recorded
            G_dict1[mouseID], G_dict2[mouseID] = {}, {}
          else:
            continue
        stimulus_name = file.replace('.nc', '').replace(mouseID + '_', '')
        indices = np.array(unit_regions[mouseID].index)
        G_dict1[mouseID][stimulus_name] = generate_region_graph(s1, indices, measure=measure, cc=True, weight=weight, threshold=threshold, percentile=percentile)
        G_dict2[mouseID][stimulus_name] = generate_region_graph(s2, indices, measure=measure, cc=True, weight=weight, threshold=threshold, percentile=percentile)
        if not mouseID in area_dict:
          area_dict[mouseID] = {}
        instruction = unit_regions[mouseID].reset_index()
        for i in range(instruction.shape[0]):
          area_dict[mouseID][i] = instruction['ecephys_structure_acronym'].iloc[i]
  return G_dict1, G_dict2, area_dict

def compute_nc_as_graph_whole(path, measure, threshold):
  G_dict = {}
  for file in os.listdir(path):
    print(file)
    if file.endswith(".nc"):
        data = xr.open_dataset(os.path.join(path, file)).to_array()
        structure_acronym = file.replace('.nc', '').split('_')[-1]
        stimulus_name = file.replace('.nc', '').replace('_' + structure_acronym, '')
        if not stimulus_name in G_dict:
          G_dict[stimulus_name] = {}
        G_dict[stimulus_name][structure_acronym] = generate_graph(data, measure=measure, cc=True, threshold=threshold)
  return G_dict

def plot_graph(G):
  edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
  degrees = dict(G.degree)
  pos = nx.spring_layout(G)
  # pos = nx.spring_layout(G, k=0.8, iterations=50) # make nodes as seperate as possible
  nx.draw(G, pos, nodelist=degrees.keys(), node_size=[v * 0.5 for v in degrees.values()], 
  edgelist=edges, edge_color=weights, width=3.0, edge_cmap=plt.cm.Greens, alpha=0.4)
  plt.show()

def plot_graph_color(G, area_attr, cc):
  com = CommunityLayout()
  fig = plt.figure()
  nx.set_node_attributes(G, area_attr, "area")
  if G.number_of_nodes() > 2 and G.number_of_edges() > 0:
    if cc:
      if nx.is_directed(G):
        Gcc = sorted(nx.strongly_connected_components(G), key=len, reverse=True)
        G = G.subgraph(Gcc[0])
      else:
        Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
        G = G.subgraph(Gcc[0])
  try:
    edges, weights = zip(*nx.get_edge_attributes(G,'weight').items())
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
  nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=weights, width=3.0, edge_cmap=plt.cm.Greens, alpha=0.9)
  nx.draw_networkx_nodes(G, pos, nodelist=degrees.keys(), node_size=[np.log(v + 2) * 20 for v in degrees.values()], 
  node_color=colors, alpha=0.4)
  areas = [G.nodes[n]['area'] for n in G.nodes()]
  areas_uniq = list(set(areas))
  for index, a in enumerate(areas_uniq):
    plt.scatter([],[], c=customPalette[index], label=a, s=30)
  legend = plt.legend(loc='upper left', fontsize=5)
  for handle in legend.legendHandles:
    handle.set_sizes([6.0])
  plt.tight_layout()
  plt.show()
  

def plot_graph_community(G_dict, area_dict, mouseID, stimulus, measure):
  G = G_dict[mouseID][stimulus]
  print(nx.info(G))
  if G.number_of_nodes() > 2 and G.number_of_edges() > 0:
    node_area = area_dict[mouseID]
    # customPalette = ['#630C3A', '#39C8C6', '#D3500C', '#FFB139', 'palegreen', 'darkblue', 'slategray', '#a6cee3', '#b2df8a', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#6a3d9a', '#b15928']
    areas_uniq = list(set(node_area.values()))
    # print(areas_uniq)
    node_to_community = {node:areas_uniq.index(area) for node, area in node_area.items()}
    node_color = {node:customPalette[areas_uniq.index(area)] for node, area in node_area.items()}
    fig = plt.figure(figsize=(10, 8))
    Graph(G,
        node_color=node_color, node_edge_width=0, edge_alpha=0.1,
        node_layout='community', node_layout_kwargs=dict(node_to_community=node_to_community),
        edge_layout='bundled', edge_layout_kwargs=dict(k=2000))
    # plt.show()
    for ind, a in enumerate(areas_uniq):
      plt.scatter([],[], c=customPalette[ind], label=a)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('./plots/graph_region_{}_{}_{}.jpg'.format(measure, mouseID, stimulus))

def plot_multi_graphs(G_dict, measure, threshold, percentile, cc=False):
  # fig = plt.figure(figsize=(30, 40))
  ind = 1
  rows, cols = get_rowcol(G_dict, measure)
  fig = plt.figure(figsize=(4*len(cols), 4*len(rows)))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      plt.subplot(len(rows), len(cols), ind)
      if row_ind == 0:
        plt.gca().set_title(cols[col_ind], fontsize=30, rotation=45)
      if col_ind == 0:
        plt.gca().text(0, 0.5 * (bottom + top), rows[row_ind],
        horizontalalignment='right',
        verticalalignment='center',
        # rotation='vertical',
        transform=plt.gca().transAxes, fontsize=30, rotation=45)
      plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
      ind += 1
      G = G_dict[row][col] if col in G_dict[row] else nx.Graph()
      if G.number_of_nodes() > 2 and G.number_of_edges() > 0:
        if cc:
          if nx.is_directed(G):
            Gcc = sorted(nx.strongly_connected_components(G), key=len, reverse=True)
            G = G.subgraph(Gcc[0])
          else:
            Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
            G = G.subgraph(Gcc[0])
        edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
        degrees = dict(G.degree)
        pos = nx.spring_layout(G)
        # pos = nx.spring_layout(G, k=0.8, iterations=50) # make nodes as seperate as possible
        nx.draw(G, pos, nodelist=degrees.keys(), node_size=[v * 1 for v in degrees.values()], 
        edgelist=edges, edge_color=weights, width=3.0, edge_cmap=plt.cm.Greens, alpha=0.4)
      
  plt.tight_layout()
  th = threshold if measure == 'pearson' else percentile
  image_name = './plots/graphs_cc_{}_{}.jpg'.format(measure, th) if cc else './plots/graphs_{}_{}.jpg'.format(measure, th)
  plt.savefig(image_name)
  # plt.show()

def plot_multi_graphs_color(G_dict, area_dict, measure, cc=False):
  com = CommunityLayout()
  ind = 1
  rows, cols = get_rowcol(G_dict, measure)
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
      G = G_dict[row][col] if col in G_dict[row] else nx.Graph()
      nx.set_node_attributes(G, area_dict[row], "area")
      if G.number_of_nodes() > 2 and G.number_of_edges() > 0:
        if cc:
          if nx.is_directed(G):
            Gcc = sorted(nx.strongly_connected_components(G), key=len, reverse=True)
            G = G.subgraph(Gcc[0])
          else:
            Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
            G = G.subgraph(Gcc[0])
        try:
          edges, weights = zip(*nx.get_edge_attributes(G,'weight').items())
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
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=weights, width=3.0, edge_cmap=plt.cm.Greens, alpha=0.9)
        nx.draw_networkx_nodes(G, pos, nodelist=degrees.keys(), node_size=[np.log(v + 2) * 20 for v in degrees.values()], 
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
  th = threshold if measure == 'pearson' else percentile
  image_name = './plots/graphs_region_color_cc_{}_{}.jpg'.format(measure, th) if cc else './plots/graphs_region_color_{}_{}.jpg'.format(measure, th)
  plt.savefig(image_name)
  # plt.show()

def plot_degree_distribution(G):
  degree_freq = nx.degree_histogram(G)
  degrees = range(len(degree_freq))
  plt.figure(figsize=(12, 8)) 
  plt.loglog(degrees, degree_freq,'go-') 
  plt.xlabel('Degree')
  plt.ylabel('Frequency')

def plot_multi_degree_distributions(G_dict, measure, threshold, percentile, cc=False):
  ind = 1
  rows, cols = get_rowcol(G_dict, measure)
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
      G = G_dict[row][col] if col in G_dict[row] else nx.Graph()
      if G.number_of_nodes() > 2 and G.number_of_edges() > 0:
        if cc:
          if nx.is_directed(G):
            Gcc = sorted(nx.strongly_connected_components(G), key=len, reverse=True)
            G = G.subgraph(Gcc[0])
          else:
            Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
            G = G.subgraph(Gcc[0])
        degree_freq = nx.degree_histogram(G)
        degrees = range(len(degree_freq))
        plt.loglog(degrees, degree_freq,'go-') 
        plt.xlabel('Degree')
        plt.ylabel('Frequency')
      
  plt.tight_layout()
  th = threshold if measure == 'pearson' else percentile
  image_name = './plots/degree_distribution_cc_{}_{}.jpg'.format(measure, th) if cc else './plots/degree_distribution_{}_{}.jpg'.format(measure, th)
  plt.savefig(image_name)

def intra_inter_connection(G_dict, area_dict, percentile, measure):
  rows, cols = get_rowcol(G_dict, measure)
  metric = np.empty((len(rows), len(cols), 3))
  metric[:] = np.nan
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      print(col)
      G = G_dict[row][col] if col in G_dict[row] else nx.Graph()
      if G.number_of_nodes() > 2 and G.number_of_edges() > 0:
        node_area = area_dict[row]
        modularity = community.modularity(node_area,G.to_undirected())
        res = {}
        for i, v in node_area.items():
          res[v] = [i] if v not in res.keys() else res[v] + [i]
        comm = list(res.values())
        coverage = nx.community.coverage(G, comm)
        performance = nx.community.performance(G, comm)
        metric[row_ind, col_ind, :] = modularity, coverage, performance
  metric_names = ['modularity', 'coverage', 'performance']
  metric_stimulus = pd.DataFrame(columns=['stimulus', 'metric', 'mean', 'std'])
  for metric_ind, metric_name in enumerate(metric_names):
    df = pd.DataFrame(columns=['stimulus', 'metric', 'mean', 'std'])
    df['mean'] = np.nanmean(metric[:, :, metric_ind], axis=0)
    df['std'] = np.nanstd(metric[:, :, metric_ind], axis=0)
    df['metric'] = metric_name
    df['stimulus'] = cols
    metric_stimulus = metric_stimulus.append(df, ignore_index=True)
  print(metric_stimulus)
  fig = plt.figure(figsize=[10, 6])
  for i, m in metric_stimulus.groupby("metric"):
    plt.plot(m['stimulus'], m['mean'], alpha=0.6, label=m['metric'].iloc[0])
    plt.fill_between(m['stimulus'], m['mean'] - m['std'], m['mean'] + m['std'], alpha=0.2)
  plt.legend()
  plt.xticks(rotation=30)
  plt.tight_layout()
  # plt.show()
  num = threshold if measure=='pearson' else percentile
  plt.savefig('./plots/intra_inter_{}_{}.jpg'.format(measure, num))

def metric_stimulus_error_region(G_dict, percentile, measure):
  rows, cols = get_rowcol(G_dict, measure)
  metric_names = get_metric_names(G_dict)
  metric = np.empty((len(rows), len(cols), len(metric_names)))
  metric[:] = np.nan
  for metric_ind, metric_name in enumerate(metric_names):
    print(metric_name)
    for row_ind, row in enumerate(rows):
      print(row)
      for col_ind, col in enumerate(cols):
        G = G_dict[row][col] if col in G_dict[row] else nx.Graph()
        # print(nx.info(G))
        if G.number_of_nodes() > 2 and G.number_of_edges() > 0:
          metric[row_ind, col_ind, metric_ind] = calculate_metric(G, metric_name)
  metric_stimulus = pd.DataFrame(columns=['stimulus', 'metric', 'mean', 'std'])
  for metric_ind, metric_name in enumerate(metric_names):
    df = pd.DataFrame(columns=['stimulus', 'metric', 'mean', 'std'])
    df['mean'] = np.nanmean(metric[:, :, metric_ind], axis=0)
    df['std'] = np.nanstd(metric[:, :, metric_ind], axis=0)
    df['metric'] = metric_name
    df['stimulus'] = cols
    metric_stimulus = metric_stimulus.append(df, ignore_index=True)
  print(metric_stimulus)
  fig = plt.figure(figsize=[10, 6])
  for i, m in metric_stimulus.groupby("metric"):
    plt.plot(m['stimulus'], m['mean'], alpha=0.6, label=m['metric'].iloc[0])
    plt.fill_between(m['stimulus'], m['mean'] - m['std'], m['mean'] + m['std'], alpha=0.2)
  plt.legend()
  plt.xticks(rotation=30)
  plt.tight_layout()
  # plt.show()
  num = threshold if measure=='pearson' else percentile
  plt.savefig('./plots/metric_stimulus_{}_{}.jpg'.format(measure, num))

def metric_stimulus_individual(G_dict, threshold, percentile, measure, cc):
  rows, cols = get_rowcol(G_dict, measure)
  metric_names = get_metric_names(G_dict)
  plots_shape = (3, 3) if len(metric_names) == 9 else (2, 4)
  metric = np.empty((len(rows), len(cols), len(metric_names)))
  metric[:] = np.nan
  fig = plt.figure(figsize=(20, 10))
  for metric_ind, metric_name in enumerate(metric_names):
    print(metric_name)
    for row_ind, row in enumerate(rows):
      print(row)
      for col_ind, col in enumerate(cols):
        G = G_dict[row][col] if col in G_dict[row] else nx.Graph()
        # print(nx.info(G))
        if G.number_of_nodes() > 2 and G.number_of_edges() > 0:
          metric[row_ind, col_ind, metric_ind] = calculate_metric(G, metric_name, cc)
    plt.subplot(*plots_shape, metric_ind + 1)
    for row_ind, row in enumerate(rows):
      plt.plot(cols, metric[row_ind, :, metric_ind], label=row, alpha=1)
    plt.gca().set_title(metric_name, fontsize=30, rotation=0)
    plt.xticks(rotation=90)
  plt.legend()
  plt.tight_layout()
  # plt.show()
  num = threshold if measure=='pearson' else percentile
  plt.savefig('./plots/metric_stimulus_individual_{}_{}.jpg'.format(measure, num))

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

def delta_metric_stimulus_individual(G_dict, rewired_G_dict, algorithm, threshold, percentile, measure, cc):
  rows, cols = get_rowcol(G_dict, measure)
  metric_names = get_metric_names(G_dict)
  metric = np.empty((len(rows), len(cols), len(metric_names)))
  metric[:] = np.nan
  metric_base = np.empty((len(rows), len(cols), len(metric_names)))
  metric_base[:] = np.nan
  fig = plt.figure(figsize=(20, 10))
  for metric_ind, metric_name in enumerate(metric_names):
    print(metric_name)
    for row_ind, row in enumerate(rows):
      print(row)
      for col_ind, col in enumerate(cols):
        G = G_dict[row][col] if col in G_dict[row] else nx.Graph()
        G_base = rewired_G_dict[row][col] if col in rewired_G_dict[row] else nx.Graph()
        if G.number_of_nodes() > 2 and G.number_of_edges() > 0:
          metric[row_ind, col_ind, metric_ind] = calculate_metric(G, metric_name, cc=cc)
          metric_base[row_ind, col_ind, metric_ind] = calculate_metric(G_base, metric_name, cc=False)
    plt.subplot(2, 4, metric_ind + 1)
    for row_ind, row in enumerate(rows):
      plt.plot(cols, metric[row_ind, :, metric_ind] - metric_base[row_ind, :, metric_ind], label=row, alpha=1)
    plt.gca().set_title(metric_name, fontsize=30, rotation=0)
    plt.xticks(rotation=90)
  plt.legend()
  plt.tight_layout()
  # plt.show()
  num = threshold if measure=='pearson' else percentile
  plt.savefig('./plots/delta_metric_stimulus_individual_{}_{}_{}.jpg'.format(algorithm, measure, num))

def metric_stimulus_stat(G_dict, rows, cols, metric_names):
  metric = np.empty((len(rows), len(cols), len(metric_names)))
  metric[:] = np.nan
  for metric_ind, metric_name in enumerate(metric_names):
    print(metric_name)
    for row_ind, row in enumerate(rows):
      for col_ind, col in enumerate(cols):
        G = G_dict[row][col] if col in G_dict[row] else nx.Graph()
        # print(nx.info(G))
        if G.number_of_nodes() > 2 and G.number_of_edges() > 0:
          metric[row_ind, col_ind, metric_ind] = calculate_metric(G, metric_name)
  metric_stimulus = pd.DataFrame(columns=['stimulus', 'metric', 'mean', 'std'])
  for metric_ind, metric_name in enumerate(metric_names):
    df = pd.DataFrame(columns=['stimulus', 'metric', 'mean', 'std'])
    df['mean'] = np.nanmean(metric[:, :, metric_ind], axis=0)
    df['std'] = np.nanstd(metric[:, :, metric_ind], axis=0)
    df['metric'] = metric_name
    df['stimulus'] = cols
    df.fillna(0, inplace=True)
    metric_stimulus = metric_stimulus.append(df, ignore_index=True)
  return metric_stimulus

def delta_metric_stimulus_stat(G_dict, rewired_G_dict, rows, cols, metric_names):
  metric = np.empty((len(rows), len(cols), len(metric_names)))
  metric[:] = np.nan
  metric_base = np.empty((len(rows), len(cols), len(metric_names)))
  metric_base[:] = np.nan
  for metric_ind, metric_name in enumerate(metric_names):
    for row_ind, row in enumerate(rows):
      for col_ind, col in enumerate(cols):
        G = G_dict[row][col] if col in G_dict[row] else nx.Graph()
        G_base = rewired_G_dict[row][col] if col in rewired_G_dict[row] else nx.Graph()
        # print(nx.info(G))
        if G.number_of_nodes() > 2 and G.number_of_edges() > 0:
          metric[row_ind, col_ind, metric_ind] = calculate_metric(G, metric_name)
        if G_base.number_of_nodes() > 2 and G_base.number_of_edges() > 0:
          metric_base[row_ind, col_ind, metric_ind] = calculate_metric(G_base, metric_name)
  delta_metric_stimulus = pd.DataFrame(columns=['stimulus', 'metric', 'mean', 'std'])
  for metric_ind, metric_name in enumerate(metric_names):
    df = pd.DataFrame(columns=['stimulus', 'metric', 'mean', 'std'])
    df['mean'] = np.nanmean(metric[:, :, metric_ind] - metric_base[:, :, metric_ind], axis=0)
    df['std'] = np.nanstd(metric[:, :, metric_ind] - metric_base[:, :, metric_ind], axis=0)
    df['metric'] = metric_name
    df['stimulus'] = cols
    delta_metric_stimulus = delta_metric_stimulus.append(df, ignore_index=True)
  return delta_metric_stimulus

def metric_stimulus_half_graph(G_dict, G_dict1, G_dict2, threshold, percentile, measure):
  rows, cols = get_rowcol(G_dict, measure)
  customPalette = ['#630C3A', '#39C8C6', '#D3500C', '#FFB139', 'palegreen', 'darkblue', 'slategray']
  metric_names = get_metric_names(G_dict)
  metric_stimulus = metric_stimulus_stat(G_dict, rows, cols, metric_names)
  metric_stimulus1 = metric_stimulus_stat(G_dict1, rows, cols, metric_names)
  metric_stimulus2 = metric_stimulus_stat(G_dict2, rows, cols, metric_names)
  fig = plt.figure(figsize=[20, 10])
  ind = 1
  for i, m in metric_stimulus.groupby("metric"):
    plt.subplot(2, 4, ind)
    plt.gca().set_title(i, fontsize=20, rotation=0)
    plt.plot(m['stimulus'], m['mean'], alpha=0.6, label='whole graph', color=customPalette[6])
    plt.fill_between(m['stimulus'], m['mean'] - m['std'], m['mean'] + m['std'], alpha=0.2, color=customPalette[6])
    plt.xticks(rotation=90)
    ind += 1
  ind = 1
  for i, m in metric_stimulus1.groupby("metric"):
    plt.subplot(2, 4, ind)
    plt.plot(m['stimulus'], m['mean'], alpha=0.6, label='1st subgraph', color=customPalette[1])
    plt.fill_between(m['stimulus'], m['mean'] - m['std'], m['mean'] + m['std'], alpha=0.2, color=customPalette[1])
    plt.xticks(rotation=90)
    ind += 1
  ind = 1
  for i, m in metric_stimulus2.groupby("metric"):
    plt.subplot(2, 4, ind)
    plt.plot(m['stimulus'], m['mean'], alpha=0.6, label='2nd subgraph', color=customPalette[4])
    plt.fill_between(m['stimulus'], m['mean'] - m['std'], m['mean'] + m['std'], alpha=0.2, color=customPalette[4])
    plt.xticks(rotation=90)
    ind += 1
  plt.legend()
  plt.tight_layout()
  # plt.show()
  num = threshold if measure=='pearson' else percentile
  plt.savefig('./plots/metric_stimulus_half_graphs_{}_{}.jpg'.format(measure, num))

def metric_stimulus_subgraphs(G_dict, G_sub_dict, threshold, percentile, measure):
  rows, cols = get_rowcol(G_dict, measure)
  metric_names = get_metric_names(G_dict)
  metric_stimulus = metric_stimulus_stat(G_dict, rows, cols, metric_names)
  sub_metric_stimulus = {}
  for index in G_sub_dict.keys():
    print(index)
    sub_metric_stimulus[index] = metric_stimulus_stat(G_sub_dict[index], rows, cols, metric_names)
  fig = plt.figure(figsize=[20, 10])
  ind = 1
  for i, m in metric_stimulus.groupby("metric"):
    plt.subplot(2, 4, ind)
    plt.gca().set_title(i, fontsize=20, rotation=0)
    plt.plot(m['stimulus'], m['mean'], alpha=0.6, label='whole graph', color=customPalette[list(sub_metric_stimulus.keys())[-1]+1])
    plt.fill_between(m['stimulus'], m['mean'] - m['std'], m['mean'] + m['std'], alpha=0.2, color=customPalette[6])
    plt.xticks(rotation=90)
    ind += 1
  for index in sub_metric_stimulus.keys():
    ind = 1
    for i, m in sub_metric_stimulus[index].groupby("metric"):
      plt.subplot(2, 4, ind)
      plt.plot(m['stimulus'], m['mean'], alpha=0.6, label='subgraph {}'.format(index+1), color=customPalette[index])
      plt.fill_between(m['stimulus'], m['mean'] - m['std'], m['mean'] + m['std'], alpha=0.2, color=customPalette[index])
      plt.xticks(rotation=90)
      ind += 1
  plt.legend()
  plt.tight_layout()
  # plt.show()
  num = threshold if measure=='pearson' else percentile
  plt.savefig('./plots/metric_stimulus_subgraphs_{}_{}.jpg'.format(measure, num))

def delta_metric_stimulus_half_graph(G_dict, G_dict1, G_dict2, algorithm, threshold, percentile, measure):
  rows, cols = get_rowcol(G_dict, measure)
  customPalette = ['#630C3A', '#39C8C6', '#D3500C', '#FFB139', 'palegreen', 'darkblue', 'slategray']
  metric_names = get_metric_names(G_dict)
  print('Rewiring the whole graph...')
  rewired_G_dict = random_graph_baseline(G_dict, algorithm, measure)
  print('Rewiring the 1st half graph...')
  rewired_G_dict1 = random_graph_baseline(G_dict1, algorithm, measure)
  print('Rewiring the 2nd half graph...')
  rewired_G_dict2 = random_graph_baseline(G_dict2, algorithm, measure)
  print('Calculating delta metric stimulus for whole graph...')
  delta_metric_stimulus = delta_metric_stimulus_stat(G_dict, rewired_G_dict, rows, cols, metric_names)
  print('Calculating delta metric stimulus for the 1st half graph...')
  delta_metric_stimulus1 = delta_metric_stimulus_stat(G_dict1, rewired_G_dict1, rows, cols, metric_names)
  print('Calculating delta metric stimulus for the 2nd half graph...')
  delta_metric_stimulus2 = delta_metric_stimulus_stat(G_dict2, rewired_G_dict2, rows, cols, metric_names)
  fig = plt.figure(figsize=[20, 10])
  ind = 1
  for i, m in delta_metric_stimulus.groupby("metric"):
    plt.subplot(2, 4, ind)
    plt.gca().set_title(i, fontsize=20, rotation=0)
    plt.plot(m['stimulus'], m['mean'], alpha=0.6, label='whole graph', color=customPalette[6])
    plt.fill_between(m['stimulus'], m['mean'] - m['std'], m['mean'] + m['std'], alpha=0.2, color=customPalette[6])
    plt.xticks(rotation=90)
    ind += 1
  ind = 1
  for i, m in delta_metric_stimulus1.groupby("metric"):
    plt.subplot(2, 4, ind)
    plt.plot(m['stimulus'], m['mean'], alpha=0.6, label='1st subgraph', color=customPalette[1])
    plt.fill_between(m['stimulus'], m['mean'] - m['std'], m['mean'] + m['std'], alpha=0.2, color=customPalette[1])
    plt.xticks(rotation=90)
    ind += 1
  ind = 1
  for i, m in delta_metric_stimulus2.groupby("metric"):
    plt.subplot(2, 4, ind)
    plt.plot(m['stimulus'], m['mean'], alpha=0.6, label='2nd subgraph', color=customPalette[4])
    plt.fill_between(m['stimulus'], m['mean'] - m['std'], m['mean'] + m['std'], alpha=0.2, color=customPalette[4])
    plt.xticks(rotation=90)
    ind += 1
  plt.legend()
  plt.tight_layout()
  # plt.show()
  num = threshold if measure=='pearson' else percentile
  plt.savefig('./plots/delta_metric_stimulus_half_graphs_{}_{}_{}.jpg'.format(algorithm, measure, num))

def metric_heatmap(G_dict, measure):
  rows, cols = get_rowcol(G_dict, measure)
  metric_names = get_metric_names(G_dict)
  stimulus_mecs = pd.DataFrame(index=rows, columns=metric_names)
  for metric_name in metric_names:
    print(metric_name)
    metric = pd.DataFrame(np.zeros((len(rows), len(cols))), index=rows, columns=cols)
    for row in rows:
      print(row)
      for col in cols:
        print(col)
        G = G_dict[row][col] if col in G_dict[row] else nx.Graph()
        if G.number_of_nodes() > 2 and G.number_of_edges() > 0:
          metric.loc[row][col] = calculate_metric(G, metric_name)
    stimulus_mecs[metric_name] = metric.mean(axis=0)
    fig, axs = plt.subplots()
    sns_plot = sns.heatmap(metric.astype(float), cmap="YlGnBu")
    sns_plot.set(title=measure+'\n'+metric_name)
    # fig = sns_plot.get_figure()
    plt.tight_layout()
    plt.savefig('./plots/'+measure+'_'+metric_name+'.jpg')

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

def get_metric_names(G_dict):
  G = list(list(G_dict.items())[0][1].items())[0][1]
  if nx.is_directed(G):
    metric_names = ['assortativity', 'betweenness', 'closeness', 'clustering', 'density', 'modularity', 'transitivity']
  else:
    # metric_names = ['assortativity', 'betweenness', 'closeness', 'clustering', 'density', 'efficiency', 'modularity', 'small-worldness', 'transitivity']
    metric_names = ['assortativity', 'betweenness', 'closeness', 'clustering', 'density', 'efficiency', 'modularity', 'transitivity']
  return metric_names

def random_graph_baseline(G_dict, algorithm, measure, cc, Q=100):
  rewired_G_dict = {}
  rows, cols = get_rowcol(G_dict, measure)
  G = list(list(G_dict.items())[0][1].items())[0][1]
  if nx.is_directed(G):
    algorithm = 'directed_configuration_model'
  for row in rows:
      print(row)
      if not row in rewired_G_dict:
        rewired_G_dict[row] = {}
      for col in cols:
        print(col)
        G = G_dict[row][col].copy() if col in G_dict[row] else nx.Graph()
        if G.number_of_nodes() >= 2 and G.number_of_edges() >= 1:
          if cc:
            largest_cc = max(nx.connected_components(G), key=len)
            G = nx.subgraph(G, largest_cc)
          # print(G.number_of_nodes(), G.number_of_edges())
          if algorithm == 'configuration_model':
            degree_sequence = [d for n, d in G.degree()]
            G = nx.configuration_model(degree_sequence)
            # remove parallel edges and self-loops
            G = nx.Graph(G)
            G.remove_edges_from(nx.selfloop_edges(G))
            # print(G.number_of_nodes(), G.number_of_edges())
          elif algorithm == 'directed_configuration_model':
            din = list(d for n, d in G.in_degree())
            dout = list(d for n, d in G.out_degree())
            G = nx.directed_configuration_model(din, dout)
            G = nx.DiGraph(G)
            G.remove_edges_from(nx.selfloop_edges(G))
          elif algorithm == 'double_edge_swap':
            # at least four nodes with edges
            degrees = dict(nx.degree(G))
            if len(np.nonzero(list(degrees.values()))[0]) >= 4:
              nx.double_edge_swap(G, nswap=Q*G.number_of_edges(), max_tries=1e75)
        rewired_G_dict[row][col] = G
  return rewired_G_dict

def region_connection_heatmap(G_dict, area_dict, regions, measure, threshold, percentile):
  ind = 1
  rows, cols = get_rowcol(G_dict, measure)
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
      G = G_dict[row][col] if col in G_dict[row] else nx.Graph()
      region_connection = np.zeros((len(regions), len(regions)))
      if G.number_of_nodes() > 2 and G.number_of_edges() > 0:
        A = nx.adjacency_matrix(G)
        A = A.todense()
        for region_ind_i, region_i in enumerate(regions):
          for region_ind_j, region_j in enumerate(regions):
            region_indices_i = np.array([k for k, v in area_dict[row].items() if v==region_i])
            region_indices_j = np.array([k for k, v in area_dict[row].items() if v==region_j])
            region_connection[region_ind_i][region_ind_j] = np.sum(A[region_indices_i[:, None], region_indices_j])
            assert np.sum(A[region_indices_i[:, None], region_indices_j]) == len(A[region_indices_i[:, None], region_indices_j].nonzero()[0])
      sns_plot = sns.heatmap(region_connection.astype(float), vmin=0, cmap="YlGnBu")
      # sns_plot = sns.heatmap(region_connection.astype(float), vmin=0, vmax=1500, cmap="YlGnBu")
      sns_plot.set_xticks(np.arange(len(regions))+0.5)
      sns_plot.set_xticklabels(regions, rotation=90)
      sns_plot.set_yticks(np.arange(len(regions))+0.5)
      sns_plot.set_yticklabels(regions, rotation=0)
      sns_plot.invert_yaxis()
  plt.tight_layout()
  # plt.show()
  num = threshold if measure=='pearson' else percentile
  plt.savefig('./plots/region_connection_scale_{}_{}.jpg'.format(measure, num))
  # plt.savefig('./plots/region_connection_{}_{}.jpg'.format(measure, num))

def SBM_density(G_dict, area_dict, regions, measure):
  SBM_dict = {}
  rows, cols = get_rowcol(G_dict, measure)
  for row_ind, row in enumerate(rows):
    print(row)
    if row not in SBM_dict:
      SBM_dict[row] = {}
    sizes = list(dict(Counter(area_dict[row].values())).values())
    for col_ind, col in enumerate(cols):
      probs = np.zeros((len(regions), len(regions)))
      G = G_dict[row][col]
      A = nx.adjacency_matrix(G)
      A = A.todense()
      for region_ind_i, region_i in enumerate(regions):
        for region_ind_j, region_j in enumerate(regions):
          region_indices_i = np.array([k for k, v in area_dict[row].items() if v==region_i])
          region_indices_j = np.array([k for k, v in area_dict[row].items() if v==region_j])
          probs[region_ind_i][region_ind_j] = np.sum(A[region_indices_i[:, None], region_indices_j]) / A[region_indices_i[:, None], region_indices_j].size
      SBM_dict[row][col] = nx.stochastic_block_model(sizes, probs, seed=100)
  return SBM_dict

# %%
all_areas = units['ecephys_structure_acronym'].unique()
# %%
############# load data as a whole graph for each mouse and stimulus #################
measure = 'pearson'
# measure = 'cosine'
# measure = 'correlation'
# measure = 'MI'
# measure = 'causality'
threshold = 0.7
percentile = 99
directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
G_dict = load_nc_as_graph_whole(directory, measure, threshold, percentile)
# %%
############# load graph with only visual regions #################
measure = 'pearson'
# measure = 'cosine'
# measure = 'correlation'
# measure = 'MI'
# measure = 'causality'
threshold = 0.5
percentile = 99
weight = False # unweighted network
visual_regions = ['VISp', 'VISl', 'VISrl', 'VISal', 'VISpm', 'VISam', 'LGd', 'LP']
directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
G_dict, area_dict = load_npy_regions_as_graph_whole(directory, visual_regions, weight, measure, threshold, percentile)
# %%
############# load graph with only visual regions and n steps of the sequence #################
ind = 0
n = 100
measure = 'pearson'
# measure = 'cosine'
# measure = 'correlation'
# measure = 'MI'
# measure = 'causality'
threshold = 0.5
percentile = 99
weight = False # unweighted network
visual_regions = ['VISp', 'VISl', 'VISrl', 'VISal', 'VISpm', 'VISam', 'LGd', 'LP']
directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
G_dict, area_dict = load_npy_regions_nsteps_as_graph(directory, n, ind, visual_regions, weight, measure, threshold, percentile)
# %%
############# load graphs from n subsequences with only visual regions #################
n = 2
measure = 'pearson'
threshold = 0.5
percentile = 99
weight = False # unweighted network
visual_regions = ['VISp', 'VISl', 'VISrl', 'VISal', 'VISpm', 'VISam', 'LGd', 'LP']
directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
G_sub_dict, area_dict = load_nc_regions_as_n_graphs(directory, n, visual_regions, weight, measure, threshold, percentile)
# %%
metric_stimulus_subgraphs(G_dict, G_sub_dict, threshold, percentile, measure)
# %%
############# plot all graphs with community layout and color as region #################
cc = True
plot_multi_graphs_color(G_dict, area_dict, measure, cc=cc)
# %%
for row in G_dict:
    for col in G_dict[row]:
        print(row, col)
        print(nx.info(G_dict[row][col]))
# %%
measures = ['pearson', 'cosine']
thresholds = [0.4, 0.5, 0.6]
percentiles = [90, 95, 99]
ccs = [True, False]
measure = measures[0]
for threshold in thresholds:
  print(threshold)
  for cc in ccs:
    print(cc)
    plot_multi_graphs(G_dict, measure, threshold, percentile, cc)
    plot_multi_degree_distributions(G_dict, measure, threshold, percentile, cc)
measure = measures[1]
for percentile in percentiles:
  print(percentile)
  for cc in ccs:
    print(cc)
    plot_multi_graphs(G_dict, measure, threshold, percentile, cc)
    plot_multi_degree_distributions(G_dict, measure, threshold, percentile, cc)
# %%
# distribution: degree, closeness, clustering coefficient, betweenness
# %%
cc = False
plot_multi_graphs(G_dict, measure, threshold, percentile, cc)
# %%
cc = False
plot_multi_degree_distributions(G_dict, measure, threshold, percentile, cc)
# %%
directory = './data/connectivity_matrix/'
# G_dict = load_npy_as_graph(directory)
percentile = 90
G_dict, area_dict, adj_mat = load_npy_as_graph_whole(directory, percentile=percentile, unweighted=True)
measure = 'ccg'
# %%
############# plot metric_stimulus individually for each mouse #############
cc = True
metric_stimulus_individual(G_dict, threshold, percentile, measure, cc)
# %%
############# get rewired graphs #############
# algorithm = 'double_edge_swap'
cc = True
algorithm = 'configuration_model'
rewired_G_dict = random_graph_baseline(G_dict, algorithm, measure, cc, Q=100)
# %%
metric_stimulus_individual(rewired_G_dict, threshold, percentile, measure)
# %%
############# plot delta metric_stimulus individually for each mouse #############
cc = True # for real graphs, cc is false for rewired baselines
delta_metric_stimulus_individual(G_dict, rewired_G_dict, algorithm, threshold, percentile, measure, cc)
 # %%
############# load each half of sequence as one graph with only visual regions #################
measure = 'pearson'
# measure = 'cosine'
# measure = 'correlation'
# measure = 'MI'
# measure = 'causality'
threshold = 0.7
percentile = 99
weight = False # unweighted network
visual_regions = ['VISp', 'VISl', 'VISrl', 'VISal', 'VISpm', 'VISam', 'LGd', 'LP']
directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
G_dict1, G_dict2, area_dict = load_nc_as_two_graphs(directory, units, visual_regions, weight, measure, threshold, percentile)
# %%
############# plot metric_stimulus with error bar for whole, first and second half graphs
metric_stimulus_half_graph(G_dict, G_dict1, G_dict2, threshold, percentile, measure)
# %%
############# plot delta metric_stimulus for whole and half graphs #############
############# it takes a long time (15 mins) to execute double edge swap, configuration model is much faster #############
algorithm = 'double_edge_swap'
# algorithm = 'configuration_model'
start_time = time.time()
delta_metric_stimulus_half_graph(G_dict, G_dict1, G_dict2, algorithm, threshold, percentile, measure)
print("--- %s minutes in total" % ((time.time() - start_time)/60))
# %%
intra_inter_connection(G_dict, area_dict, percentile, measure)
# %%
metric_stimulus_error_region(G_dict, percentile, measure)
# %%
############# plot region connection heatmap #############
region_connection_heatmap(G_dict, area_dict, visual_regions, measure, threshold, percentile)
# %%

# %%
start_time = time.time()
mouseIDs, stimuli = get_rowcol(G_dict, measure)
mouseIDs = mouseIDs[::-1] # reverse order
stimuli = stimuli[::-1] # reverse order
for mouseID in mouseIDs:
  for stimulus in stimuli:
    print(mouseID, stimulus)
    plot_graph_community(G_dict, area_dict, mouseID, stimulus, measure)
print("--- %s minutes in total" % ((time.time() - start_time)/60))

# %%
############# generate SBM with same link density #############
SBM_dict = SBM_density(G_dict, area_dict, visual_regions, measure)
# %%
metric_stimulus_individual(SBM_dict, threshold, percentile, measure)
# %%
algorithm = 'configuration_model'
rewired_SBM_dict = random_graph_baseline(SBM_dict, algorithm, measure, Q=100)
# %%
############# plot delta metric_stimulus individually for each mouse #############
delta_metric_stimulus_individual(SBM_dict, rewired_SBM_dict, algorithm, threshold, percentile, measure)
# %%
############# plot all SBM graphs with community layout and color as region #################
cc = True
plot_multi_graphs_color(SBM_dict, area_dict, measure, cc=cc)
# %%
session_id = 719161530
stimulus_name = 'flashes'
session = cache.get_session_data(session_id)
# presentations = session.get_stimulus_table(stimulus_name)
# units = session.units
# time_step = 0.001
# time_bins = np.arange(0, 2.0 + time_step, time_step)
# a = session.presentationwise_spike_times(stimulus_presentation_ids=presentations.index.values,  
#       unit_ids=units.index.values)
# %%
stimulus_names = ['spontaneous', 'flashes', 'gabors',
        'drifting_gratings', 'static_gratings',
          'natural_scenes', 'natural_movie_one', 'natural_movie_three']
# session_ids = sessions[sessions.session_type=='brain_observatory_1.1'].index.values # another one is functional_connectivity
session_ids = [719161530, 750749662, 755434585, 756029989, 791319847]
for session_id in session_ids:
  for stimulus_name in stimulus_names:
    session = cache.get_session_data(session_id)
    presentation_ids = session.stimulus_presentations.loc[
        (session.stimulus_presentations['stimulus_name'] == 'drifting_gratings')
    ].index.values
    times = session.presentationwise_spike_times(stimulus_presentation_ids=presentation_ids)
    print('The maximum time of spike for session {} stimulus {} is {}'.format(session_id, stimulus_name, times.time_since_stimulus_presentation_onset.max()))
# %%
measure = 'pearson'
threshold = 0.5
percentile = 99
weight = False # unweighted network
regions = ['VISp', 'VISl', 'VISrl', 'VISal', 'VISpm', 'VISam', 'LGd', 'LP']
directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
G_dict = {}
area_dict = {}
files = os.listdir(directory)
files.sort(key=lambda x:int(x[:9]))
file = files[0]
if file.endswith(".nc"):
    data = xr.open_dataset(os.path.join(directory, file)).to_array()
    if len(data.data.shape) > 2:
      sequences = data.data.squeeze().T
    else:
      sequences = data.data.T
    mouseID = file.split('_')[0]
    if not mouseID in G_dict:
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
      print(df_cortex.ccf)
      instruction = df_cortex.ccf
      if set(instruction.unique()) == set(regions): # if the mouse has all regions recorded
        G_dict[mouseID] = {}
    stimulus_name = file.replace('.nc', '').replace(mouseID + '_', '')
    G_dict[mouseID][stimulus_name] = generate_graph(sequences, measure=measure, cc=True, weight=weight, threshold=threshold, percentile=percentile)
    if not mouseID in area_dict:
      area_dict[mouseID] = {}
    instruction = instruction.reset_index()
    for i in range(instruction.shape[0]):
      area_dict[mouseID][i] = instruction.ccf.iloc[i]
# %%
rows, cols = get_rowcol(G_dict, measure)
metric_names = get_metric_names(G_dict)
metric_stimulus = metric_stimulus_stat(G_dict, rows, cols, metric_names)
sub_metric_stimulus = {}
for index in G_sub_dict.keys():
  print(index)
  sub_metric_stimulus[index] = metric_stimulus_stat(G_sub_dict[index], rows, cols, metric_names)
# %%
fig = plt.figure(figsize=[20, 10])
for ind, (i, m) in enumerate(metric_stimulus.groupby("metric")):
  plt.subplot(2, 4, ind + 1)
  plt.gca().set_title(i, fontsize=20, rotation=0)
  plt.plot(m['stimulus'], m['mean'], alpha=0.6, label='whole graph', color=customPalette[list(sub_metric_stimulus.keys())[-1]+1])
  plt.fill_between(m['stimulus'], m['mean'] - m['std'], m['mean'] + m['std'], alpha=0.2, color=customPalette[list(sub_metric_stimulus.keys())[-1]+1])
  plt.xticks(rotation=90)
for index in sub_metric_stimulus.keys():
  for ind, (i, m) in enumerate(sub_metric_stimulus[index].groupby("metric")):
    plt.subplot(2, 4, ind + 1)
    plt.plot(m['stimulus'], m['mean'], alpha=0.6, label='subgraph {}'.format(index+1), color=customPalette[index])
    plt.fill_between(m['stimulus'], m['mean'] - m['std'], m['mean'] + m['std'], alpha=0.2, color=customPalette[index])
    plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()


# %%
n = 10
measure = 'pearson'
threshold = 0.5
percentile = 99
weight = False # unweighted network
visual_regions = ['VISp', 'VISl', 'VISrl', 'VISal', 'VISpm', 'VISam', 'LGd', 'LP']
directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
G_sub_dict = dict.fromkeys(range(n), {})
area_dict = {}
files = os.listdir(directory)
files.sort(key=lambda x:int(x[:9]))
file = files[-3]
print(file)
if file.endswith(".nc"):
  data = xr.open_dataset(os.path.join(directory, file)).to_array()
  if len(data.data.shape) > 2:
    sequences = data.data.squeeze().T
  else:
    sequences = data.data.T
  sub_sequences = np.array_split(sequences, n, axis=1)
# %%
adj_mat = {}
for i, sub_sequence in enumerate(sub_sequences):
  adj_mat[i] = corr_mat(sub_sequence, measure, threshold, percentile)
# %%
rows, cols = get_rowcol(G_dict, measure)
nodes = {'spontaneous':0, 'flashes':0, 'gabors':0, 'drifting_gratings':0, 'static_gratings':0, 'natural_scenes':0, 'natural_movie_one':0, 'natural_movie_three':0}
edges = {'spontaneous':0, 'flashes':0, 'gabors':0, 'drifting_gratings':0, 'static_gratings':0, 'natural_scenes':0, 'natural_movie_one':0, 'natural_movie_three':0}
for row in G_dict:
    for col in G_dict[row]:
        nodes[col] += nx.number_of_nodes(G_dict[row][col])
        edges[col] += nx.number_of_edges(G_dict[row][col])
for i in nodes:
    nodes[i] /= len(rows)
    edges[i] /= len(rows)
print(nodes)
print(edges)
# %%
############# load graph with only visual regions and n steps of the sequence #################
ind = 0
n = 100
measure = 'pearson'
# measure = 'cosine'
# measure = 'correlation'
# measure = 'MI'
# measure = 'causality'
threshold = 0.5
percentile = 99
weight = False # unweighted network
visual_regions = ['VISp', 'VISl', 'VISrl', 'VISal', 'VISpm', 'VISam', 'LGd', 'LP']
directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
G_dict, area_dict = load_npy_regions_nsteps_as_graph(directory, n, ind, visual_regions, weight, measure, threshold, percentile)
# %%
############# plot all graphs with community layout and color as region #################
cc = True
plot_multi_graphs_color(G_dict, area_dict, measure, cc=cc)
cc = False
plot_multi_degree_distributions(G_dict, measure, threshold, percentile, cc)
############# plot metric_stimulus individually for each mouse #############
cc = True
metric_stimulus_individual(G_dict, threshold, percentile, measure, cc)
############# get rewired graphs #############
# algorithm = 'double_edge_swap'
cc = True
algorithm = 'configuration_model'
rewired_G_dict = random_graph_baseline(G_dict, algorithm, measure, cc, Q=100)
############# plot delta metric_stimulus individually for each mouse #############
cc = True # for real graphs, cc is false for rewired baselines
delta_metric_stimulus_individual(G_dict, rewired_G_dict, algorithm, threshold, percentile, measure, cc)
# %%
