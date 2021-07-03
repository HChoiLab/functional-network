#%%
import os
import re
import sys
import seaborn as sns
# import shutil
from SimpleITK.SimpleITK import Threshold
from networkx.algorithms.efficiency_measures import efficiency
import numpy as np
import pandas as pd
import time
import community
from matplotlib import pyplot as plt
import networkx as nx
from requests.sessions import session
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

def get_spiking_sequence(session_id, stimulus_name, structure_acronym):
  session = cache.get_session_data(session_id)
  print(session.metadata)
  print(session.structurewise_unit_counts)
  presentations = session.get_stimulus_table(stimulus_name)
  units = session.units[session.units["ecephys_structure_acronym"]==structure_acronym]
  time_step = 0.01
  time_bins = np.arange(-0.1, 0.5 + time_step, time_step)
  histograms = session.presentationwise_spike_counts(
      stimulus_presentation_ids=presentations.index.values,  
      bin_edges=time_bins,
      unit_ids=units.index.values)
  return histograms
  # print(histograms.coords)

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

def generate_graph(mean_histograms, measure, lcc=True, threshold=0.3):
  if len(mean_histograms.data.shape) > 2:
    sequences = mean_histograms.data.squeeze().T
  else:
    sequences = mean_histograms.data.T
  if measure == 'pearson':
    adj_mat = np.corrcoef(sequences)
    adj_mat[adj_mat < threshold] = 0
  elif measure == 'cosine':
    adj_mat = cosine_similarity(sequences)
    adj_mat[adj_mat < adj_mat.mean()] = 0
  elif measure == 'correlation':
    adj_mat = squareform(pdist(sequences, 'correlation'))
    adj_mat[adj_mat < adj_mat.mean()] = 0
  elif measure == 'MI':
    adj_mat = MI(sequences)
    adj_mat[adj_mat < adj_mat.mean()] = 0
  elif measure == 'causality':
    adj_mat = granger_causality(sequences)
    adj_mat[adj_mat < adj_mat.mean()] = 0
  else:
    sys.exit('Unrecognized measure value!!! Choices: pearson, cosin, correlation.')
  adj_mat = np.nan_to_num(adj_mat)
  np.fill_diagonal(adj_mat, 0)
  G = nx.from_numpy_array(adj_mat) # same as from_numpy_matrix
  if lcc: # extract the largest connected components
    largest_cc = max(nx.connected_components(G), key=len)
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

def load_npy_as_graph_whole(path, percentile=99, unweighted=False):
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

def load_data_as_graph(path, measure, threshold):
  G_dict = {}
  for file in os.listdir(path):
    print(file)
    if file.endswith(".nc"):
        data = xr.open_dataset(os.path.join(path, file)).to_array()
        structure_acronym = file.replace('.nc', '').split('_')[-1]
        stimulus_name = file.replace('.nc', '').replace('_' + structure_acronym, '')
        if not stimulus_name in G_dict:
          G_dict[stimulus_name] = {}
        G_dict[stimulus_name][structure_acronym] = generate_graph(data, measure=measure, lcc=True, threshold=threshold)
  return G_dict

def plot_graph(G):
  edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
  degrees = dict(G.degree)
  pos = nx.spring_layout(G)
  # pos = nx.spring_layout(G, k=0.8, iterations=50) # make nodes as seperate as possible
  nx.draw(G, pos, nodelist=degrees.keys(), node_size=[v * 0.5 for v in degrees.values()], 
  edgelist=edges, edge_color=weights, width=3.0, edge_cmap=plt.cm.Greens, alpha=0.4)
  plt.show()

def plot_graph_color(G, area_dict, measure):
  customPalette = ['#630C3A', '#39C8C6', '#D3500C', '#FFB139', 'palegreen', 'darkblue', 'slategray']
  nx.set_node_attributes(G, area_dict, "area")
  edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
  degrees = dict(G.degree)
  pos = nx.spring_layout(G)
  # pos = nx.spring_layout(G, k=0.8, iterations=50) # make nodes as seperate as possible
  # pos = nx.spectral_layout(G, weight='weight')
  areas = [G.nodes[n]['area'] for n in G.nodes()]
  areas_uniq = list(set(areas))
  colors = [customPalette[areas_uniq.index(area)] for area in areas]
  
  if len(np.unique(weights)) <= 1:
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='green', width=3.0, edge_cmap=plt.cm.Greens, alpha=0.4)
  else:
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=weights, width=3.0, edge_cmap=plt.cm.Greens, alpha=0.4)
  nx.draw_networkx_nodes(G, pos, nodelist=degrees.keys(), node_size=[v * 10 for v in degrees.values()], 
  node_color=colors, alpha=0.4)
  plt.show()
  # plt.savefig('./plots/graphs_{}.jpg'.format(measure))

def plot_graph_community(G_dict, area_dict, mouseID, stimulus, measure):
  G = G_dict[mouseID][stimulus]
  print(nx.info(G))
  node_area = area_dict[mouseID]
  customPalette = ['#630C3A', '#39C8C6', '#D3500C', '#FFB139', 'palegreen', 'darkblue', 'slategray']
  areas_uniq = list(set(node_area.values()))
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
  plt.savefig('./plots/graph_region_{}_{}.jpg'.format(mouseID, stimulus))

def plot_multi_graphs(G_dict, measure):
  # fig = plt.figure(figsize=(30, 40))
  ind = 1
  rows = list(G_dict.keys())
  cols = []
  for row in rows:
    cols += list(G_dict[row].keys())
  cols = list(set(cols))
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
        plt.gca().set_title(cols[col_ind], fontsize=30)
      if col_ind == 0:
        plt.gca().text(0, 0.5 * (bottom + top), rows[row_ind],
        horizontalalignment='right',
        verticalalignment='center',
        # rotation='vertical',
        transform=plt.gca().transAxes, fontsize=30, rotation=45)
      plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
      ind += 1
      G = G_dict[row][col] if col in G_dict[row] else nx.Graph()
      if len(G.nodes()) > 1:
        edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
        degrees = dict(G.degree)
        pos = nx.spring_layout(G)
        # pos = nx.spring_layout(G, k=0.8, iterations=50) # make nodes as seperate as possible
        nx.draw(G, pos, nodelist=degrees.keys(), node_size=[v * 1 for v in degrees.values()], 
        edgelist=edges, edge_color=weights, width=3.0, edge_cmap=plt.cm.Greens, alpha=0.4)
      
  plt.tight_layout()
  plt.savefig('./plots/graphs_{}.jpg'.format(measure))
  # plt.show()

def plot_multi_graphs_color(G_dict, node_area, measure):
  # fig = plt.figure(figsize=(30, 40))
  customPalette = ['#630C3A', '#39C8C6', '#D3500C', '#FFB139', 'palegreen', 'darkblue', 'slategray']
  ind = 1
  rows = list(G_dict.keys())
  cols = []
  for row in rows:
    cols += list(G_dict[row].keys())
  cols = list(set(cols))
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
        plt.gca().set_title(cols[col_ind], fontsize=30)
      if col_ind == 0:
        plt.gca().text(0, 0.5 * (bottom + top), rows[row_ind],
        horizontalalignment='right',
        verticalalignment='center',
        # rotation='vertical',
        transform=plt.gca().transAxes, fontsize=30, rotation=45)
      plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
      ind += 1
      G = G_dict[row][col] if col in G_dict[row] else nx.Graph()
      nx.set_node_attributes(G, node_area[row], "area")
      if len(G.nodes()) > 1:
        edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
        degrees = dict(G.degree)
        pos = nx.spring_layout(G)
        areas = [G.nodes[n]['area'] for n in G.nodes()]
        areas_uniq = list(set(areas))
        colors = [customPalette[areas_uniq.index(area)] for area in areas]
        # pos = nx.spring_layout(G, k=0.8, iterations=50) # make nodes as seperate as possible
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=weights, width=3.0, edge_cmap=plt.cm.Greens, alpha=0.4)
        nx.draw_networkx_nodes(G, pos, nodelist=degrees.keys(), node_size=[v * 1 for v in degrees.values()], 
        node_color=colors, alpha=0.4)
      
  plt.tight_layout()
  plt.savefig('./plots/graphs_{}.jpg'.format(measure))

def plot_degree_distribution(G):
  degree_freq = nx.degree_histogram(G)
  degrees = range(len(degree_freq))
  plt.figure(figsize=(12, 8)) 
  plt.loglog(degrees, degree_freq,'go-') 
  plt.xlabel('Degree')
  plt.ylabel('Frequency')

def plot_multi_degree_distributions(G_dict, measure):
  ind = 1
  rows = list(G_dict.keys())
  cols = []
  for row in rows:
    cols += list(G_dict[row].keys())
  cols = list(set(cols))
  fig = plt.figure(figsize=(4*len(cols), 3*len(rows)))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  for row_ind, row in enumerate(rows):
    for col_ind, col in enumerate(cols):
      plt.subplot(len(rows), len(cols), ind)
      if row_ind == 0:
        plt.gca().set_title(cols[col_ind], fontsize=30)
      if col_ind == 0:
        plt.gca().text(0, 0.5 * (bottom + top), rows[row_ind],
        horizontalalignment='right',
        verticalalignment='center',
        # rotation='vertical',
        transform=plt.gca().transAxes, fontsize=30, rotation=45)
      plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
      ind += 1
      G = G_dict[row][col] if col in G_dict[row] else nx.Graph()
      if len(G.nodes()) > 1:
        degree_freq = nx.degree_histogram(G)
        degrees = range(len(degree_freq))
        plt.loglog(degrees, degree_freq,'go-') 
        plt.xlabel('Degree')
        plt.ylabel('Frequency')
      
  plt.tight_layout()
  plt.savefig('./plots/degree_distribution_{}.jpg'.format(measure))

def intra_inter_connection(G_dict, area_dict, percentile, measure):
  rows = list(G_dict.keys())
  cols = []
  for row in rows:
    cols += list(G_dict[row].keys())
  cols = list(set(cols))
  # sort stimulus
  stimulus_rank = ['spon', 'spon_20', 'None', 'denoised', 'low', 'flash', 'flash_40', 'movie', 'movie_20']
  stimulus_rank_dict = {i:stimulus_rank.index(i) for i in cols}
  stimulus_rank_dict = dict(sorted(stimulus_rank_dict.items(), key=lambda item: item[1]))
  cols = list(stimulus_rank_dict.keys())
  metric = np.zeros((len(rows), len(cols), 3))
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      print(col)
      G = G_dict[row][col] if col in G_dict[row] else nx.Graph()
      if len(G.nodes()) > 2:
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
  plt.tight_layout()
  # plt.show()
  plt.savefig('./plots/{}_{}_intra_inter.jpg'.format(measure, percentile))

def metric_stimulus_error_region(G_dict, percentile, measure):
  rows = list(G_dict.keys())
  cols = []
  for row in rows:
    cols += list(G_dict[row].keys())
  cols = list(set(cols))
  # sort stimulus
  stimulus_rank = ['spon', 'spon_20', 'None', 'denoised', 'low', 'flash', 'flash_40', 'movie', 'movie_20']
  stimulus_rank_dict = {i:stimulus_rank.index(i) for i in cols}
  stimulus_rank_dict = dict(sorted(stimulus_rank_dict.items(), key=lambda item: item[1]))
  cols = list(stimulus_rank_dict.keys())
  G = list(list(G_dict.items())[0][1].items())[0][1]
  if nx.is_directed(G):
    metric_names = ['clustering', 'transitivity', 'betweenness', 'closeness', 'modularity', 'assortativity', 'density']
  else:
    metric_names = ['efficiency', 'clustering', 'transitivity', 'betweenness', 'closeness', 'modularity', 'assortativity', 'density']
  metric = np.zeros((len(rows), len(cols), len(metric_names)))
  for metric_ind, metric_name in enumerate(metric_names):
    print(metric_name)
    for row_ind, row in enumerate(rows):
      print(row)
      for col_ind, col in enumerate(cols):
        print(col)
        G = G_dict[row][col] if col in G_dict[row] else nx.Graph()
        # print(nx.info(G))
        if len(G.nodes()) > 2:
          if metric_name == 'efficiency':
            met = nx.global_efficiency(G)
          elif metric_name == 'clustering':
            met = nx.average_clustering(G)
          elif metric_name == 'transitivity':
            met = nx.transitivity(G)
          elif metric_name == 'betweenness':
            met = np.mean(list(nx.betweenness_centrality(G).values()))
          elif metric_name == 'closeness':
            met = np.mean(list(nx.closeness_centrality(G).values()))
          elif metric_name == 'modularity':
            try:
              part = community.best_partition(G)
              met = community.modularity(part,G)
            except:
              met = 0
          elif metric_name == 'assortativity':
            met = nx.degree_assortativity_coefficient(G)
          elif metric_name == 'density':
            met = nx.density(G)
          metric[row_ind, col_ind, metric_ind] = met
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
  plt.tight_layout()
  # plt.show()
  plt.savefig('./plots/{}_{}_metric_stimulus.jpg'.format(measure, percentile))

def metric_heatmap(G_dict, measure):
  rows = list(G_dict.keys())
  cols = []
  for row in rows:
    cols += list(G_dict[row].keys())
  cols = list(set(cols))
  metric_names = ['efficiency', 'clustering', 'transitivity', 'betweenness', 'closeness', 'modularity', 'assortativity', 'density']
  stimulus_mecs = pd.DataFrame(index=rows, columns=metric_names)
  for metric_name in metric_names:
    print(metric_name)
    metric = pd.DataFrame(np.zeros((len(rows), len(cols))), index=rows, columns=cols)
    for row in rows:
      print(row)
      for col in cols:
        print(col)
        G = G_dict[row][col] if col in G_dict[row] else nx.Graph()
        if len(G.nodes()) > 2:
          if metric_name == 'efficiency':
            metric.loc[row][col] = nx.global_efficiency(G)
          elif metric_name == 'clustering':
            metric.loc[row][col] = nx.average_clustering(G)
          elif metric_name == 'transitivity':
            metric.loc[row][col] = nx.transitivity(G)
          elif metric_name == 'betweenness':
            metric.loc[row][col] = np.mean(list(nx.betweenness_centrality(G).values()))
          elif metric_name == 'closeness':
            metric.loc[row][col] = np.mean(list(nx.closeness_centrality(G).values()))
          elif metric_name == 'modularity':
            try:
              part = community.best_partition(G)
              metric.loc[row][col] = community.modularity(part,G)
            except:
              metric.loc[row][col] = 0
          elif metric_name == 'assortativity':
            metric.loc[row][col] = nx.degree_assortativity_coefficient(G)
          elif metric_name == 'density':
            metric.loc[row][col] = nx.density(G)
    stimulus_mecs[metric_name] = metric.mean(axis=0)
    fig, axs = plt.subplots()
    sns_plot = sns.heatmap(metric.astype(float), cmap="YlGnBu")
    sns_plot.set(title=measure+'\n'+metric_name)
    # fig = sns_plot.get_figure()
    plt.tight_layout()
    plt.savefig('./plots/'+measure+'_'+metric_name+'.jpg')
  return stimulus_mecs
# %%
# # mean histogram is a matrix
# mean_histograms = histograms.mean(dim="stimulus_presentation_id")
# fig, ax = plt.subplots(figsize=(8, 8))
# ax.pcolormesh(
#     mean_histograms["time_relative_to_stimulus_onset"], 
#     np.arange(mean_histograms["unit_id"].size),
#     mean_histograms.T, 
#     vmin=0,
#     vmax=1)
# ax.set_ylabel("unit", fontsize=24)
# ax.set_xlabel("time relative to stimulus onset (s)", fontsize=24)
# ax.set_title("peristimulus time histograms for VISp units on flash presentations", fontsize=24)
# plt.show()
# %%
stimulus_names = ['drifting_gratings', 'drifting_gratings_contrast', 'flashes',
       'gabors', 'natural_movie_one', 'natural_movie_three',
       'natural_scenes', 'spontaneous', 'static_gratings']
# structure_acronyms = ['VISp', 'CA1', 'VISrl', 'VISl']
structure_acronyms = ['VISal', 'VISpm', 'VISam']
session_id = 791319847
directory = './data/ecephys_cache_dir/session_{}/stimulus_structure/'.format(session_id)
ind = 1
for stimulus_name in stimulus_names:
  for structure_acronym in structure_acronyms:
    histograms = get_spiking_sequence(session_id, stimulus_name, structure_acronym)
    mean_histograms = histograms.mean(dim="stimulus_presentation_id")
    mean_histograms.to_netcdf(directory + stimulus_name + '_' + structure_acronym + '.nc')
    print('finished {} data'.format(ind))
    ind += 1
# %%
# G_dict = []
# G_dict.append([])
# G = generate_graph(mean_histograms, lcc=True, threshold=0.5)
# print(nx.info(G))
# G_dict[-1].append(G)
# plot_multi_graphs(G_dict)
# %%
# measure = 'pearson'
# measure = 'cosine'
# measure = 'correlation'
measure = 'MI'
# measure = 'causality'
session_id = 791319847
threshold = 0.4
directory = './data/ecephys_cache_dir/session_{}/stimulus_structure/'.format(session_id)
G_dict = load_data_as_graph(directory, measure, threshold)
stimulus_names = ['drifting_gratings', 'drifting_gratings_contrast', 'flashes',
       'gabors', 'natural_movie_one', 'natural_movie_three',
       'natural_scenes', 'spontaneous', 'static_gratings']
structure_acronyms = ['VISp', 'CA1', 'VISrl', 'VISl', 'VISal', 'VISpm', 'VISam']
# %%
# session_id = 791319847
# stimulus_names = ['drifting_gratings', 'drifting_gratings_contrast', 'flashes',
#        'gabors', 'natural_movie_one', 'natural_movie_three',
#        'natural_scenes', 'spontaneous', 'static_gratings']
# # structure_acronyms = ['VISp', 'CA1', 'VISrl', 'VISl', 'VISal', 'VISpm', 'VISam']
# structure_acronyms = ['VISal', 'VISpm', 'VISam']
# path = './data/ecephys_cache_dir/session_{}/stimulus_structure/'.format(session_id)
# dataset = xr.open_dataset(os.path.join(path, stimulus_names[0]+'_'+structure_acronyms[0]+'.nc')).to_array()
# data = dataset.data.squeeze().T
# %%
# distribution: degree, closeness, clustering coefficient, betweenness
# %%
plot_multi_graphs(G_dict, measure)
plot_multi_degree_distributions(G_dict, measure)
# %%
directory = './data/connectivity_matrix/'
# G_dict = load_npy_as_graph(directory)
percentile = 90
G_dict, area_dict, adj_mat = load_npy_as_graph_whole(directory, percentile=percentile, unweighted=True)
measure = 'ccg'
# %%
intra_inter_connection(G_dict, area_dict, percentile, measure)
# %%
metric_stimulus_error_region(G_dict, percentile, measure)
# %%

# %%
start_time = time.time()
for mouseID in G_dict:
  for stimulus in G_dict[mouseID]:
    print(mouseID, stimulus)
    plot_graph_community(G_dict, area_dict, mouseID, stimulus, measure)
print("--- %s minutes ---" % ((time.time() - start_time)/60))
# %%
# plot_multi_graphs_color(G_dict, area_dict, measure)
# %%
mouseID = '388523'
# mouseID = '306046'
# mouseID = '416357'
# mouseID = '419118'
plot_multi_graphs(G_dict[mouseID], measure + '_' + mouseID)
plot_multi_degree_distributions(G_dict[mouseID], measure + '_' + mouseID)

# %%
mouseID = '388523'
# mouseID = '306046'
# mouseID = '416357'
# mouseID = '419118'
stimulus_mecs = metric_heatmap(G_dict[mouseID], measure + '_' + mouseID)
# %%
# stimulus_mecs.sort_values('clustering', inplace=True)
stimulus_mecs = metric_heatmap(G_dict, measure)
stimulus_mecs = stimulus_mecs.reindex(['spontaneous', 'static_gratings', 'flashes', 'gabors', 'drifting_gratings_contrast', 'drifting_gratings', 
       'natural_movie_three', 'natural_movie_one', 'natural_scenes'])
stimulus_mecs.plot(figsize=(10, 8))
plt.xticks(rotation=20)
plt.title(measure)
plt.savefig('./plots/metrics_stimulus_{}.jpg'.format(measure))
# %%
# stimulus_mecs.sort_values('clustering', inplace=True)
stimulus_rank = ['spon', 'spon_20', 'None', 'denoised', 'low', 'flash', 'flash_40', 'movie', 'movie_20']
stimulus_rank_dict = {i:stimulus_rank.index(i) for i in stimulus_mecs.index}
stimulus_rank_dict = dict(sorted(stimulus_rank_dict.items(), key=lambda item: item[1]))
stimulus_mecs = stimulus_mecs.reindex(stimulus_rank_dict.keys())
stimulus_mecs.plot(figsize=(10, 8))
plt.xticks(rotation=20)
plt.title(measure + '_' + mouseID)
plt.savefig('./plots/metrics_stimulus_{}.jpg'.format(measure + '_' + mouseID))
# %%
ID = '419119'
stimulus = 'flash_40'
G = G_dict[ID][stimulus]
plot_graph(G)

# %%
mouseID = '306046'
stimulus = 'None'
G, node_area = G_dict[mouseID][stimulus], area_dict[mouseID]
customPalette = ['#630C3A', '#39C8C6', '#D3500C', '#FFB139', 'palegreen', 'darkblue', 'slategray']
areas_uniq = list(set(node_area.values()))
node_to_community = {node:areas_uniq.index(area) for node, area in node_area.items()}
node_color = {node:customPalette[areas_uniq.index(area)] for node, area in node_area.items()}
# %%
rows = list(G_dict.keys())
cols = []
for row in rows:
  cols += list(G_dict[row].keys())
cols = list(set(cols))
metric_names = ['efficiency', 'clustering', 'transitivity', 'betweenness', 'closeness', 'modularity', 'assortativity', 'density']
metric = np.zeros((len(rows), len(cols), len(metric_names)))
for metric_ind, metric_name in enumerate(metric_names):
  print(metric_name)
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      print(col)
      G = G_dict[row][col] if col in G_dict[row] else nx.Graph()
      # print(nx.info(G))
      if len(G.nodes()) > 2:
        if metric_name == 'efficiency':
          met = nx.global_efficiency(G)
        elif metric_name == 'clustering':
          met = nx.average_clustering(G)
        elif metric_name == 'transitivity':
          met = nx.transitivity(G)
        elif metric_name == 'betweenness':
          met = np.mean(list(nx.betweenness_centrality(G).values()))
        elif metric_name == 'closeness':
          met = np.mean(list(nx.closeness_centrality(G).values()))
        elif metric_name == 'modularity':
          try:
            part = community.best_partition(G)
            met = community.modularity(part,G)
          except:
            met = 0
        elif metric_name == 'assortativity':
          met = nx.degree_assortativity_coefficient(G)
        elif metric_name == 'density':
          met = nx.density(G)
        metric[row_ind, col_ind, metric_ind] = met
# %%
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
plt.tight_layout()
plt.show()
# %%
