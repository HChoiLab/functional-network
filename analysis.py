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
import matplotlib.pyplot as plt
import statsmodels.stats.weightstats as ws
import networkx as nx
import community
import seaborn as sns
from plfit import plfit
from scipy import stats
from sklearn.linear_model import LinearRegression
import random
from scipy.spatial import distance

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

def get_metric_names(G_dict):
  G = list(list(G_dict.items())[0][1].items())[0][1]
  if type(G) == list:
    G = G[0]
  if nx.is_directed(G):
    metric_names = ['assortativity', 'betweenness', 'closeness', 'clustering', 'density', 'modularity', 'transitivity']
  else:
    # metric_names = ['assortativity', 'betweenness', 'closeness', 'clustering', 'density', 'efficiency', 'modularity', 'small-worldness', 'transitivity']
    metric_names = ['assortativity', 'betweenness', 'closeness', 'clustering', 'density', 'efficiency', 'modularity', 'transitivity']
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

def generate_graph(adj_mat, cc=False, weight=False):
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

def load_significant_adj(directory, weight):
  G_dict = {}
  files = os.listdir(directory)
  files.sort(key=lambda x:int(x[:9]))
  for file in files:
    if file.endswith(".npz"):
      print(file)
      adj_mat = load_npz_3d(os.path.join(directory, file))
      mouseID = file.split('_')[0]
      stimulus_name = file.replace('.npz', '').replace(mouseID + '_', '')
      if not mouseID in G_dict:
        G_dict[mouseID] = {}
      G_dict[mouseID][stimulus_name] = []
      for i in range(adj_mat.shape[2]):
        G_dict[mouseID][stimulus_name].append(generate_graph(adj_mat=adj_mat[:, :, i], cc=False, weight=weight))
  return G_dict

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

def region_connection_heatmap(G_dict, sign, area_dict, regions, measure, threshold, percentile):
  rows, cols = get_rowcol(G_dict, measure)
  scale = np.zeros(len(rows))
  region_connection = np.zeros((len(rows), len(cols), len(regions), len(regions)))
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      G = G_dict[row][col][0] if col in G_dict[row] else nx.Graph()
      if G.number_of_nodes() > 2 and G.number_of_edges() > 0:
        nodes = list(G.nodes())
        A = nx.adjacency_matrix(G)
        A = A.todense()
        A[A.nonzero()] = 1
        for region_ind_i, region_i in enumerate(regions):
          for region_ind_j, region_j in enumerate(regions):
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
  num = threshold if measure=='pearson' else percentile
  plt.savefig('./plots/region_connection_scale_{}_{}_{}.jpg'.format(sign, measure, num))
  # plt.savefig('./plots/region_connection_{}_{}.jpg'.format(measure, num))

def region_connection_delta_heatmap(G_dict, sign, area_dict, regions, measure, threshold, percentile, index):
  rows, cols = get_rowcol(G_dict, measure)
  repeats = len(G_dict[rows[0]][cols[0]])
  # repeats = 10
  cols.remove('spontaneous')
  scale_min = np.zeros(len(rows))
  scale_max = np.zeros(len(rows))
  region_connection_bl = np.zeros((len(rows), repeats, len(regions), len(regions)))
  region_connection = np.zeros((len(rows), len(cols), repeats, len(regions), len(regions)))
  for row_ind, row in enumerate(rows):
    print(row)
    for r in range(repeats):
      print(r)
      G = G_dict[row]['spontaneous'][r]
      if G.number_of_nodes() > 100 and G.number_of_edges() > 100:
        nodes = list(G.nodes())
        A = nx.adjacency_matrix(G)
        A = A.todense()
        A[A.nonzero()] = 1
        for region_ind_i, region_i in enumerate(regions):
          for region_ind_j, region_j in enumerate(regions):
            region_indices_i = np.array([k for k, v in area_dict[row].items() if v==region_i])
            region_indices_j = np.array([k for k, v in area_dict[row].items() if v==region_j])
            region_indices_i = np.array([nodes.index(i) for i in list(set(region_indices_i) & set(nodes))]) # some nodes not in cc are removed 
            region_indices_j = np.array([nodes.index(i) for i in list(set(region_indices_j) & set(nodes))])
            region_connection_bl[row_ind, r, region_ind_i, region_ind_j] = np.sum(A[region_indices_i[:, None], region_indices_j])
            assert np.sum(A[region_indices_i[:, None], region_indices_j]) == len(A[region_indices_i[:, None], region_indices_j].nonzero()[0])
      for col_ind, col in enumerate(cols):
        G = G_dict[row][col][r] if col in G_dict[row] else nx.Graph()
        if G.number_of_nodes() > 100 and G.number_of_edges() > 100:
          nodes = list(G.nodes())
          A = nx.adjacency_matrix(G)
          A = A.todense()
          A[A.nonzero()] = 1
          for region_ind_i, region_i in enumerate(regions):
            for region_ind_j, region_j in enumerate(regions):
              region_indices_i = np.array([k for k, v in area_dict[row].items() if v==region_i])
              region_indices_j = np.array([k for k, v in area_dict[row].items() if v==region_j])
              region_indices_i = np.array([nodes.index(i) for i in list(set(region_indices_i) & set(nodes))]) # some nodes not in cc are removed 
              region_indices_j = np.array([nodes.index(i) for i in list(set(region_indices_j) & set(nodes))])
              region_connection[row_ind, col_ind, r, region_ind_i, region_ind_j] = np.sum(A[region_indices_i[:, None], region_indices_j])
              assert np.sum(A[region_indices_i[:, None], region_indices_j]) == len(A[region_indices_i[:, None], region_indices_j].nonzero()[0])
    scale_min[row_ind] = ((region_connection[row_ind, :, :, :, :]-region_connection_bl[row_ind][None, :, :, :]).mean(1)/region_connection_bl[row_ind].mean(1).sum()).min()
    scale_max[row_ind] = ((region_connection[row_ind, :, :, :, :]-region_connection_bl[row_ind][None, :, :, :]).mean(1)/region_connection_bl[row_ind].mean(1).sum()).max()
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
      sns_plot = sns.heatmap((region_connection[row_ind, col_ind, :, :, :]-region_connection_bl[row_ind]).mean(0)/region_connection_bl[row_ind].mean(0).sum(), vmin=scale_min[row_ind], vmax=scale_max[row_ind],center=0,cmap="RdBu_r") #  cmap="YlGnBu"
      # sns_plot = sns.heatmap((region_connection-region_connection_bl)/region_connection_bl.sum(), cmap="YlGnBu")
      sns_plot.set_xticks(np.arange(len(regions))+0.5)
      sns_plot.set_xticklabels(regions, rotation=90)
      sns_plot.set_yticks(np.arange(len(regions))+0.5)
      sns_plot.set_yticklabels(regions, rotation=0)
      sns_plot.invert_yaxis()
  plt.tight_layout()
  # plt.show()
  num = threshold if measure=='pearson' else percentile
  plt.savefig('./plots/region_connection_delta_scale_{}_{}_{}_{}.jpg'.format(sign, measure, num, index))
  # plt.savefig('./plots/region_connection_delta_scale_{}_{}.pdf'.format(measure, num), transparent=True)

def plot_multi_graphs_color(G_dict, sign, area_dict, measure, cc=False):
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
      G = G_dict[row][col][0] if col in G_dict[row] else nx.Graph()
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
  th = threshold if measure == 'pearson' else percentile
  image_name = './plots/graphs_region_color_cc_{}_{}_{}.jpg'.format(sign, measure, th) if cc else './plots/graphs_region_color_{}_{}_{}.jpg'.format(sign, measure, th)
  plt.savefig(image_name)
  # plt.savefig(image_name.replace('.jpg', '.pdf'), transparent=True)
  # plt.show()


def func_powerlaw(x, m, c):
  return x**m * c

def plot_multi_degree_distributions(G_dict, sign, measure, threshold, percentile, cc=False):
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
      G = G_dict[row][col][0] if col in G_dict[row] else nx.Graph()
      if G.number_of_nodes() > 2 and G.number_of_edges() > 0:
        if cc:
          if nx.is_directed(G):
            Gcc = sorted(nx.strongly_connected_components(G), key=len, reverse=True)
            G = G.subgraph(Gcc[0])
          else:
            Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
            G = G.subgraph(Gcc[0])
        degree_freq = nx.degree_histogram(G)[1:]
        degrees = np.array(range(1, len(degree_freq) + 1))
        g_degrees = list(dict(G.degree()).values())
        plt.plot(degrees, np.array(degree_freq) / sum(degree_freq),'go-')
        # plt.legend(loc='upper right', fontsize=7)
        plt.xlabel('Degree')
        plt.ylabel('Frequency')
        # plt.xscale('log')
        # plt.yscale('log')
      
  plt.tight_layout()
  th = threshold if measure == 'pearson' else percentile
  image_name = './plots/degree_distribution_cc_{}_{}_{}.jpg'.format(sign, measure, th) if cc else './plots/degree_distribution_{}_{}_{}.jpg'.format(sign, measure, th)
  # plt.show()
  plt.savefig(image_name, dpi=300)
  # plt.savefig(image_name.replace('jpg', 'pdf'), transparent=True)

def plot_multi_degree_distributions_baseline(rewired_G_dict, sign, algorithm,  measure, threshold, percentile, cc=False):
  ind = 1
  rows, cols = get_rowcol(rewired_G_dict, measure)
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
      G = rewired_G_dict[row][col][0] if col in rewired_G_dict[row] else nx.Graph()
      if G.number_of_nodes() > 2 and G.number_of_edges() > 0:
        if cc:
          if nx.is_directed(G):
            Gcc = sorted(nx.strongly_connected_components(G), key=len, reverse=True)
            G = G.subgraph(Gcc[0])
          else:
            Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
            G = G.subgraph(Gcc[0])
        degree_freq = nx.degree_histogram(G)[1:]
        degrees = np.array(range(1, len(degree_freq) + 1))
        g_degrees = list(dict(G.degree()).values())
        plt.plot(degrees, np.array(degree_freq) / sum(degree_freq),'go-')
        # plt.legend(loc='upper right', fontsize=7)
        plt.xlabel('Degree')
        plt.ylabel('Frequency')
        # plt.xscale('log')
        # plt.yscale('log')
      
  plt.tight_layout()
  th = threshold if measure == 'pearson' else percentile
  image_name = './plots/degree_distribution_{}_cc_{}_{}_{}.jpg'.format(algorithm, sign, measure, th) if cc else './plots/degree_distribution_{}_{}_{}_{}.jpg'.format(algorithm, sign, measure, th)
  # plt.show()
  plt.savefig(image_name, dpi=300)
  # plt.savefig(image_name.replace('jpg', 'pdf'), transparent=True)

def plot_multi_degree_distributions_scalefree(G_dict, measure, threshold, percentile, cc=False):
  alphas = pd.DataFrame(index=session_ids, columns=stimulus_names)
  xmins = pd.DataFrame(index=session_ids, columns=stimulus_names)
  loglikelihoods = pd.DataFrame(index=session_ids, columns=stimulus_names)
  proportions = pd.DataFrame(index=session_ids, columns=stimulus_names)
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
      G = G_dict[row][col][0] if col in G_dict[row] else nx.Graph()
      if G.number_of_nodes() > 2 and G.number_of_edges() > 0:
        if cc:
          if nx.is_directed(G):
            Gcc = sorted(nx.strongly_connected_components(G), key=len, reverse=True)
            G = G.subgraph(Gcc[0])
          else:
            Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
            G = G.subgraph(Gcc[0])
        degree_freq = nx.degree_histogram(G)[1:]
        degrees = np.array(range(1, len(degree_freq) + 1))
        g_degrees = list(dict(G.degree()).values())
        [alpha, xmin, L] = plfit(g_degrees, 'finite')
        proportion = np.sum(np.array(g_degrees)>=xmin)/len(g_degrees)
        alphas.loc[int(row)][col], xmins.loc[int(row)][col], loglikelihoods.loc[int(row)][col], proportions.loc[int(row)][col] = alpha, xmin, L, proportion
        # C is normalization constant that makes sure the sum is equal to real data points
        C = (np.array(degree_freq) / sum(degree_freq))[degrees>=xmin].sum() / np.power(degrees[degrees>=xmin], -alpha).sum()
        plt.scatter([],[], label='alpha={:.1f}'.format(alpha), s=20)
        plt.scatter([],[], label='xmin={}'.format(xmin), s=20)
        plt.scatter([],[], label='loglikelihood={:.1f}'.format(L), s=20)
        plt.plot(degrees, np.array(degree_freq) / sum(degree_freq),'go-')
        plt.plot(degrees[degrees>=xmin], func_powerlaw(degrees[degrees>=xmin], *np.array([-alpha, C])), linestyle='--', linewidth=2, color='black')
        
        plt.legend(loc='upper right', fontsize=7)
        plt.xlabel('Degree')
        plt.ylabel('Frequency')
        plt.xscale('log')
        plt.yscale('log')
      
  plt.tight_layout()
  th = threshold if measure == 'pearson' else percentile
  image_name = './plots/degree_distribution_sf_cc_{}_{}.jpg'.format(measure, th) if cc else './plots/degree_distribution_sf_{}_{}.jpg'.format(measure, th)
  # plt.show()
  plt.savefig(image_name, dpi=300)
  # plt.savefig(image_name.replace('jpg', 'pdf'), transparent=True)
  return alphas, xmins, loglikelihoods, proportions

def metric_stimulus_individual(G_dict, sign, num_sample, threshold, percentile, measure, weight, cc):
  rows, cols = get_rowcol(G_dict, measure)
  metric_names = get_metric_names(G_dict)
  plots_shape = (3, 3) if len(metric_names) == 9 else (2, 4)
  metric = np.empty((len(rows), len(cols), num_sample, len(metric_names)))
  metric[:] = np.nan
  fig = plt.figure(figsize=(20, 10))
  for metric_ind, metric_name in enumerate(metric_names):
    print(metric_name)
    for row_ind, row in enumerate(rows):
      print(row)
      for col_ind, col in enumerate(cols):
        for s in range(num_sample):
          print('Downsample {}/{}'.format(s, num_sample))
          G = G_dict[row][col][s] if col in G_dict[row] else nx.Graph()
          # print(nx.info(G))
          if G.number_of_nodes() > 2 and G.number_of_edges() > 0:
            if weight:
              metric[row_ind, col_ind, s, metric_ind] = calculate_weighted_metric(G, metric_name, cc)
            else:
              metric[row_ind, col_ind, s, metric_ind] = calculate_metric(G, metric_name, cc)
    plt.subplot(*plots_shape, metric_ind + 1)
    for row_ind, row in enumerate(rows):
      plt.plot(cols, metric[row_ind, :, :, metric_ind].mean(-1), label=row, alpha=1)
    plt.gca().set_title(metric_name, fontsize=30, rotation=0)
    plt.xticks(rotation=90)
  plt.legend()
  plt.tight_layout()
  # plt.show()
  num = threshold if measure=='pearson' else percentile
  figname = './plots/metric_stimulus_individual_weighted_{}_{}_{}.jpg'.format(sign, measure, num) if weight else './plots/metric_stimulus_individual_{}_{}_{}.jpg'.format(sign, measure, num)
  plt.savefig(figname)
  return metric

def random_graph_baseline(G_dict, num_rewire, algorithm, measure, cc, Q=100):
  rewired_G_dict = {}
  rows, cols = get_rowcol(G_dict, measure)
  G = list(list(G_dict.items())[0][1].items())[0][1][0]
  if nx.is_directed(G):
    algorithm = 'directed_configuration_model'
  for row in rows:
    print(row)
    if not row in rewired_G_dict:
      rewired_G_dict[row] = {}
    for col in cols:
      print(col)
      rewired_G_dict[row][col] = []
      for num in range(num_rewire):
        G = G_dict[row][col][num].copy() if col in G_dict[row] else nx.Graph()
        while not (nx.number_of_nodes(G) > 100 and nx.number_of_edges(G) > 100):
          num = num + num_rewire
          G = G_dict[row][col][num].copy()
        weights = np.squeeze(np.array(nx.adjacency_matrix(G)[nx.adjacency_matrix(G).nonzero()]))
        np.random.shuffle(weights)
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
        elif algorithm == 'connected_double_edge_swap':
          swaps = nx.connected_double_edge_swap(G, nswap=Q*G.number_of_edges(), _window_threshold=3)
          print('Number of successful swaps: {}'.format(swaps))
        elif algorithm == 'random':
          num_nodes, num_edges = G.number_of_nodes(), G.number_of_edges()
          G = nx.gnm_random_graph(num_nodes, num_edges)
        elif algorithm == 'rewire_hub_connection':
          degrees = dict(nx.degree(G))
          max_deg = np.max(list(degrees.values()))
          hub = [n for n, d in list(degrees.items()) if d >= max_deg/2]
          A = nx.adjacency_matrix(G).todense()
          # nonhub_pairs = np.array([[i, j] for i,j in zip(np.where(A==0)[0], np.where(A==0)[1]) if (i not in hub) and (j not in hub) and (i < j)])
          nonhub_pairs = np.array([[i, j] for i,j in itertools.combinations(G.nodes(), 2) if (i not in hub) and (j not in hub) and (i < j) and (not G.has_edge(i, j))])
          np.random.shuffle(nonhub_pairs)
          acc_rewire = []
          for h in hub:
            obj_deg = np.random.poisson(np.mean([d for n, d in list(degrees.items()) if d < max_deg/2]))
            # ns = [n for n in G.neighbors(h)]
            ns_deg = {neighbor:G.degree(neighbor) for neighbor in G.neighbors(h)}
            ns = [n for n, v in sorted(ns_deg.items(), key=lambda item: item[1], reverse=True)]
            if len(ns) > obj_deg:
              acc_rewire.append(len(ns) - obj_deg)
              i = 0
              actual_rewire = 0
              for p in range(int(np.sum(acc_rewire[:-1])), np.sum(acc_rewire)):
                num_edges = G.number_of_edges()
                assert not G.has_edge(*nonhub_pairs[p])
                weight=G.get_edge_data(h, ns[i])['weight']
                G.remove_edge(h, ns[i])
                # choose another edge if the removal disconnects the graph
                while not nx.is_connected(G):
                  G.add_edge(h, ns[i], weight=weight)
                  i += 1
                  if i >= len(ns):
                    print('Cannot remove edges from node {} anymore otherwise graph will become unconnected'.format(h))
                    break
                  weight=G.get_edge_data(h, ns[i])['weight']
                  G.remove_edge(h, ns[i])
                if i >= len(ns):
                  acc_rewire[-1] = actual_rewire
                  break
                else:
                  i += 1
                  actual_rewire += 1
                  G.add_edge(*nonhub_pairs[p], weight=weight)
                assert G.number_of_edges() == num_edges
          rewired_G_dict[row][col].append(G)
      
        # add link weights
        if not algorithm == 'rewire_hub_connection':
          for ind, e in enumerate(G.edges()):
            G[e[0]][e[1]]['weight'] = weights[ind]
        rewired_G_dict[row][col].append(G)
  return rewired_G_dict

def delta_metric_stimulus_individual(metric, sign, rewired_G_dict, algorithm, threshold, percentile, measure, weight, cc):
  rows, cols = get_rowcol(rewired_G_dict, measure)
  repeats = len(rewired_G_dict[rows[0]][cols[0]])
  metric_names = get_metric_names(rewired_G_dict)
  metric_base = np.empty((len(rows), len(cols), repeats, len(metric_names)))
  metric_base[:] = np.nan
  fig = plt.figure(figsize=(20, 10))
  for metric_ind, metric_name in enumerate(metric_names):
    print(metric_name)
    for row_ind, row in enumerate(rows):
      print(row)
      for col_ind, col in enumerate(cols):
        print(col)
        for r in range(repeats):
          print('Shuffle {}/{}'.format(r+1, repeats))
          G_base = rewired_G_dict[row][col][r] if col in rewired_G_dict[row] else nx.Graph()
          if G_base.number_of_nodes() > 2 and G_base.number_of_edges() > 0:
            if weight:
              metric_base[row_ind, col_ind, r, metric_ind] = calculate_weighted_metric(G_base, metric_name, cc=False)
            else:
              metric_base[row_ind, col_ind, r, metric_ind] = calculate_metric(G_base, metric_name, cc=False)
    plt.subplot(2, 4, metric_ind + 1)
    for row_ind, row in enumerate(rows):
      plt.plot(cols, (metric[row_ind, :, :, metric_ind] - metric_base[row_ind, :, :, metric_ind]).mean(-1), label=row, alpha=1)
    plt.gca().set_title(r'$\Delta$' + metric_name, fontsize=30, rotation=0)
    plt.xticks(rotation=90)
  plt.legend()
  plt.tight_layout()
  # plt.show()
  num = threshold if measure=='pearson' else percentile
  figname = './plots/delta_metric_stimulus_individual_weighted_{}_{}_{}_{}.jpg'.format(algorithm, sign, measure, num) if weight else './plots/delta_metric_stimulus_individual_{}_{}_{}_{}.jpg'.format(algorithm, sign, measure, num)
  plt.savefig(figname)
  return metric - metric_base

def get_full_degrees(G_dict, num_sample, measure):
  rows, cols = get_rowcol(G_dict, measure)
  max_len = 0
  for row in rows:
    for col in cols:
      for s in range(num_sample):
        G = G_dict[row][col][s] if col in G_dict[row] else nx.Graph()
        deg_fre = nx.degree_histogram(G)
        max_len = len(deg_fre) if len(deg_fre) > max_len else max_len
  print('Maximum length of degree sequence is {}'.format(max_len))
  full_degrees = np.zeros((len(rows), len(cols), num_sample, max_len))
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      print(col)
      for s in range(num_sample):
        G = G_dict[row][col][s] if col in G_dict[row] else nx.Graph()
        deg_fre = nx.degree_histogram(G)
        full_degrees[row_ind, col_ind, s, :] = deg_fre + [0] * (max_len - len(deg_fre))
  return full_degrees
# %%
start_time = time.time()
############# load area_dict and average speed dataframe #################
visual_regions = ['VISp', 'VISl', 'VISrl', 'VISal', 'VISpm', 'VISam', 'LGd', 'LP']
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
#################### load graph with significant edges (larger)
measure = 'pearson'
weight = True
cc = True
sign = 'pos'
# sign = 'neg'
directory = './data/ecephys_cache_dir/sessions/adj_mat_{}_{}_larger/'.format(sign, measure)
G_dict = load_significant_adj(directory, weight)
threshold = 0
percentile = 0
# %%
############# keep largest connected components
if cc:
  for row in G_dict:
    for col in G_dict[row]:
      for i, G in enumerate(G_dict[row][col]):
        if not nx.is_connected(G):
          largest_cc = max(nx.connected_components(G), key=len)
          G_dict[row][col][i] = nx.subgraph(G, largest_cc)
          print(G.number_of_nodes(), G_dict[row][col][i].number_of_nodes())
# %%
rows, cols = get_rowcol(G_dict, measure)
for row in G_dict:
    for col in G_dict[row]:
        nodes = nx.number_of_nodes(G_dict[row][col][0])
        edges = nx.number_of_edges(G_dict[row][col][0])
        print('Number of nodes for {} {} {}'.format(row, col, nodes))
        print('Number of edges for {} {} {}'.format(row, col, edges))
        print('Density for {} {} {}'.format(row, col, 2 * edges / nodes ** 2))
# %%
# region_connection_heatmap(G_dict, sign, area_dict, visual_regions, measure, threshold, percentile)
# # %%
# region_connection_delta_heatmap(G_dict, sign, area_dict, visual_regions, measure, threshold, percentile, 'static')
# # %%
# ############# plot all graphs with community layout and color as region #################
# cc = True
# plot_multi_graphs_color(G_dict, sign, area_dict, measure, cc=cc)
# cc = True
# plot_multi_degree_distributions(G_dict, sign, measure, threshold, percentile, cc)
# # %%
# ############# plot metric_stimulus individually for each mouse #############
# cc = True
# num_sample = 2
# metric = metric_stimulus_individual(G_dict, sign, num_sample, threshold, percentile, measure, weight, cc)
# ############# get rewired graphs #############
# # algorithm = 'double_edge_swap'
# # %%
# cc = False
# num_rewire = 10
# # algorithm = 'configuration_model'
# # algorithm = 'double_edge_swap'

# algorithm = 'connected_double_edge_swap'
# # algorithm = 'random'
# # algorithm = 'rewire_hub_connection'
# rewired_G_dict = random_graph_baseline(G_dict, num_rewire, algorithm, measure, cc, Q=10)
# # %%
# plot_multi_degree_distributions_baseline(rewired_G_dict, sign, algorithm,  measure, threshold, percentile, cc=False)
# # %%
# ############# if rewired graphs are connected
# # for row in rewired_G_dict:
# #   for col in rewired_G_dict[row]:
# #     for i in range(len(rewired_G_dict[row][col])):
# #       print(nx.is_connected(rewired_G_dict[row][col][0]))
# # %%
# ############# comparison between original and rewired graphs
# # for row in G_dict:
# #   for col in G_dict[row]:
# #     for i in range(len(rewired_G_dict[row][col])):
# #       print(G_dict[row][col][0].number_of_nodes(), rewired_G_dict[row][col][i].number_of_nodes())
# #       print(G_dict[row][col][0].number_of_edges(), rewired_G_dict[row][col][i].number_of_edges())
# # %%
# ############# plot delta metric_stimulus individually for each mouse #############
# cc = True # for real graphs, cc is false for rewired baselines
# delta_metric = delta_metric_stimulus_individual(metric, sign, rewired_G_dict, algorithm, threshold, percentile, measure, weight, cc)
# # %%
# print("--- %s minutes in total" % ((time.time() - start_time)/60))
# %%
# core_node_percent = pd.DataFrame(index=rows, columns=cols)
# core_degree_percent = pd.DataFrame(index=rows, columns=cols)
# core_top_overlap = pd.DataFrame(index=rows, columns=cols)
# core_hub_overlap = pd.DataFrame(index=rows, columns=cols)
# core_num = pd.DataFrame(index=rows, columns=cols)
# for row in rows:
#   print(row)
#   for col in cols:
#     print(col)
#     cn, cnum, cd, ct, ch = [], [], [], [], []
#     for i in range(len(G_dict[row][col])):
#       G = G_dict[row][col][i]
#       degrees = dict(nx.degree(G))
#       sorted_degree = {k: v for k, v in sorted(degrees.items(), key=lambda item: item[1], reverse=True)}
#       main_core = nx.algorithms.k_core(G)
#       cnum.append(max(nx.algorithms.core_number(G).values()))
#       top = [n for n, d in list(sorted_degree.items())[:main_core.number_of_nodes()]]
#       hub = [n for n, d in list(degrees.items()) if d >= np.max(list(degrees.values()))/2]
#       cn.append(main_core.number_of_nodes() / G.number_of_nodes())
#       cd.append(sum([degrees[n] for n in main_core.nodes()]) / sum(degrees.values()))
#       ct.append(len(set(list(main_core.nodes())) & set(top)) / main_core.number_of_nodes())
#       ch.append(len(set(list(main_core.nodes())) & set(hub)) / min(main_core.number_of_nodes(), len(hub)))
#     core_node_percent.loc[row][col] = np.mean(cn)
#     core_degree_percent.loc[row][col] = np.mean(cd)
#     core_top_overlap.loc[row][col] = np.mean(ct)
#     core_hub_overlap.loc[row][col] = np.mean(ch)
#     core_num.loc[row][col] = np.mean(cnum)
# # %%
# rows, cols = get_rowcol(G_dict, measure)
# fig = plt.figure(figsize=(14, 14))
# plt.subplot(2, 2, 1)
# for row_ind, row in enumerate(rows):
#   plt.plot(cols, core_node_percent.loc[row, :], label=row, alpha=1)
# plt.gca().set_title('fraction of core node', fontsize=20, rotation=0)
# plt.xticks(rotation=90, fontsize=20)
# plt.yticks(fontsize=20)
# plt.subplot(2, 2, 2)
# for row_ind, row in enumerate(rows):
#   plt.plot(cols, core_degree_percent.loc[row, :], label=row, alpha=1)
# plt.gca().set_title('fraction of core degree', fontsize=20, rotation=0)
# plt.xticks(rotation=90, fontsize=20)
# plt.yticks(fontsize=20)
# plt.subplot(2, 2, 3)
# for row_ind, row in enumerate(rows):
#   plt.plot(cols, core_top_overlap.loc[row, :], label=row, alpha=1)
# plt.gca().set_title(r'$\frac{|\{core\}\cap\{top\}|}{|core|}$', fontsize=30, rotation=0)
# plt.xticks(rotation=90, fontsize=20)
# plt.yticks(fontsize=20)
# plt.subplot(2, 2, 4)
# for row_ind, row in enumerate(rows):
#   plt.plot(cols, core_hub_overlap.loc[row, :], label=row, alpha=1)
# plt.gca().set_title(r'$\frac{|\{core\}\cap\{hub\}|}{\min(|core|,|hub|)}$', fontsize=30, rotation=0)
# plt.xticks(rotation=90, fontsize=20)
# plt.yticks(fontsize=20)
# plt.legend()
# plt.tight_layout()
# # plt.show()
# # sig = 'ztest'
# sig = 'larger'
# figname = './plots/core_percent_{}.jpg'.format(sig)
# plt.savefig(figname)
# # %%
# region_counts = {}
# repeats = len(G_dict[rows[0]][cols[0]])
# for row in rows:
#   print(row)
#   if row not in region_counts:
#     region_counts[row] = {}
#   areas = area_dict[row]
#   uni, cnts = np.unique(list(areas.values()), return_counts=True)
#   area_sizes = dict(zip(uni, cnts))
#   for col in cols:
#     print(col)
#     for r in range(len(G_dict[row][col])):
#       G = G_dict[row][col][r]
#       num_nodes = G.number_of_nodes()
#       degrees = dict(nx.degree(G))
#       hub = [n for n, d in list(degrees.items()) if d >= np.max(list(degrees.values()))/2]
#       region_hub = [areas[n] for n in hub]
#       uniq, counts = np.unique(region_hub, return_counts=True)
#       if not col in region_counts[row]:              # num_nodes
#         region_counts[row][col] = {k: v / (repeats * area_sizes[k]) for k, v in sorted(dict(zip(uniq, counts)).items(), key=lambda item: item[1], reverse=True)}
#       else:
#         for k, v in dict(zip(uniq, counts)).items():
#           if k in region_counts[row][col]:
#             region_counts[row][col][k] += v / (repeats * area_sizes[k])
#           else:
#             region_counts[row][col][k] = v / (repeats * area_sizes[k])
# # %%
# ind = 1
# hub_num = np.zeros((len(rows), len(cols)))
# rows, cols = get_rowcol(G_dict, measure)
# fig = plt.figure(figsize=(4*len(cols), 3*len(rows)))
# # fig.patch.set_facecolor('black')
# left, width = .25, .5
# bottom, height = .25, .5
# right = left + width
# top = bottom + height
# for row_ind, row in enumerate(rows):
#   print(row)
#   for col_ind, col in enumerate(cols):
#     ax = plt.subplot(len(rows), len(cols), ind)
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
#     labels = region_counts[row][col].keys()
#     sizes = region_counts[row][col].values()
#     hub_num[row_ind][col_ind] = sum(sizes)
#     explode = np.zeros(len(labels))  # only "explode" the 2nd slice (i.e. 'Hogs')
#     areas_uniq = ['VISam', 'VISpm', 'LGd', 'VISp', 'VISl', 'VISal', 'LP', 'VISrl']
#     colors = [customPalette[areas_uniq.index(l)] for l in labels]
#     patches, texts, pcts = plt.pie(sizes, radius=sum(sizes), explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
#             shadow=True, startangle=90, wedgeprops={'linewidth': 3.0, 'edgecolor': 'white'})
#     for i, patch in enumerate(patches):
#       texts[i].set_color(patch.get_facecolor())
#     # for i in range(len(p[0])):
#     #   p[0][i].set_alpha(0.6)
#     ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
# plt.suptitle('Hub nodes distribution', size=30)
# plt.tight_layout()
# th = threshold if measure == 'pearson' else percentile
# # plt.show()
# plt.savefig('./plots/pie_chart_hub_node.jpg', dpi=300)
# # %%
# plt.figure(figsize=(8, 8))
# for row_ind, row in enumerate(rows):
#   plt.plot(cols, core_num.loc[row, :], label=row, alpha=1)
# plt.gca().set_title('k of main core', fontsize=20, rotation=0)
# plt.xticks(rotation=90, fontsize=20)
# plt.yticks(fontsize=20)
# plt.legend()
# plt.tight_layout()
# # plt.show()
# # sig = 'ztest'
# sig = 'larger'
# figname = './plots/core_num_{}.jpg'.format(sig)
# plt.savefig(figname)
# # %%
# def plot_stats(G_dict, measure, stat_dict):
#   rows, cols = get_rowcol(G_dict, measure)
#   plots_shape = (1, 3)
#   fig = plt.figure(figsize=(20, 7))
#   for metric_ind, metric_name in enumerate(stat_dict):
#     print(metric_name)
#     plt.subplot(*plots_shape, metric_ind + 1)
#     for row_ind, row in enumerate(rows):
#       plt.plot(cols, stat_dict[metric_name][row_ind, :], label=row, alpha=1)
#     plt.gca().set_title(metric_name, fontsize=30, rotation=0)
#     plt.xticks(rotation=90, fontsize=20)
#     plt.yticks(fontsize=20)
#   plt.legend()
#   plt.tight_layout()
#   # plt.show()
#   # sig = 'ztest'
#   sig = 'larger'
#   figname = './plots/stats_{}.jpg'.format(sig)
#   plt.savefig(figname)

# rows, cols = get_rowcol(G_dict, measure)
# num_nodes = np.zeros((len(rows), len(cols)))
# num_edges = np.zeros((len(rows), len(cols)))
# densities = np.zeros((len(rows), len(cols)))
# for row_ind, row in enumerate(rows):
#   print(row)
#   for col_ind, col in enumerate(cols):
#     print(col)
#     d, nn, ne = [], [], []
#     for i in range(len(G_dict[row][col])):
#       G = G_dict[row][col][i]
#       d.append(nx.density(G))
#       nn.append(G.number_of_nodes())
#       ne.append(G.number_of_edges())
#     densities[row_ind, col_ind] = np.mean(d)
#     num_nodes[row_ind, col_ind] = np.mean(nn)
#     num_edges[row_ind, col_ind] = np.mean(ne)
# stat_dict = {'number of nodes':num_nodes, 'number of edges':num_edges, 'density':densities}
# # %%
# plot_stats(G_dict, measure, stat_dict)
# # %%
# plot_multi_degree_distributions_random(rewired_G_dict, measure, threshold, percentile, cc=False)
# # %%
# hierarchy_order = ['LGd', 'VISp', 'VISl', 'VISrl', 'LP', 'VISal', 'VISpm', 'VISam'] # ['LGN', 'V1', 'LM', 'RL', 'LP', 'AL', 'PM', 'AM']
# ind = 1
# hub_num = np.zeros((len(rows), len(cols)))
# rows, cols = get_rowcol(G_dict, measure)
# fig = plt.figure(figsize=(4*len(cols), 3*len(rows)))
# # fig.patch.set_facecolor('black')
# left, width = .25, .5
# bottom, height = .25, .5
# right = left + width
# top = bottom + height
# maxc, maxs = 0, 0
# for row_ind, row in enumerate(rows):
#   for col_ind, col in enumerate(cols):
#     maxc = max(region_counts[row][col].values()) if max(region_counts[row][col].values()) > maxc else maxc
#     maxs = sum(region_counts[row][col].values()) if sum(region_counts[row][col].values()) > maxs else maxs
# maxc += 0.005
# maxs += 0.01
# for row_ind, row in enumerate(rows):
#   print(row)
#   for col_ind, col in enumerate(cols):
#     ax = plt.subplot(len(rows), len(cols), ind)
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
#     region_count = region_counts[row][col]
#     dist = [region_count[r] if r in region_count else 0 for r in hierarchy_order]
#     l1 = ax.plot(hierarchy_order, dist, label='distribution')
#     ax2 = ax.twinx()
#     l2 = ax2.axhline(y=sum(region_count.values()), color='r', linestyle='--', label='sum')
#     # ax2.yaxis.label.set_color('r')
#     ax2.spines["right"].set_edgecolor('r')
#     ax2.tick_params(axis='y', colors='r')
#     ax.set_ylim(0, maxc)
#     ax2.set_ylim(0, maxs)
#     l1.append(l2)
#     labs = [l.get_label() for l in l1]
#     ax.legend(l1, labs, loc=0)
# plt.suptitle('Hub nodes distribution', size=30)
# plt.tight_layout()
# th = threshold if measure == 'pearson' else percentile
# # plt.show()
# plt.savefig('./plots/hierarchy_dist_hub_node.jpg', dpi=300)
# # %%
# num_sample = 200
# rows, cols = get_rowcol(G_dict, measure)
# fraction = np.empty((len(rows), len(cols)))
# fraction[:] = np.nan
# fig = plt.figure(figsize=(8, 6))
# for row_ind, row in enumerate(rows):
#   print(row)
#   for col_ind, col in enumerate(cols):
#     m = np.zeros(num_sample)
#     for s in range(num_sample):
#       m[s] = sum(region_counts[row][col].values())
#     fraction[row_ind, col_ind] = m.mean()
# for row_ind, row in enumerate(rows):
#   plt.plot(cols, fraction[row_ind, :], label=row, alpha=1)
# plt.gca().set_title('fraction of hub nodes', fontsize=20, rotation=0)
# plt.xticks(rotation=90)
# plt.legend()
# plt.tight_layout()
# # plt.show()
# num = threshold if measure=='pearson' else percentile
# figname = './plots/fraction_hubs.jpg'
# plt.savefig(figname)
# # %%
# ################# scatter plot of density and fraction of hub nodes
# repeats = len(G_dict[rows[0]][cols[0]])
# densities = np.zeros((len(rows), len(cols), repeats))
# fractions = np.zeros((len(rows), len(cols), repeats))
# for row_ind, row in enumerate(rows):
#   print(row)
#   for col_ind, col in enumerate(cols):
#     print(col)
#     for r in range(repeats):
#       G = G_dict[row][col][r]
#       if G.number_of_nodes() > 100 and G.number_of_edges() > 100:
#         densities[row_ind, col_ind, r] = nx.density(G)
#         num_nodes = G.number_of_nodes()
#         degrees = dict(nx.degree(G))
#         hub = [n for n, d in list(degrees.items()) if d >= np.max(list(degrees.values()))/2]
#         fractions[row_ind, col_ind, r] = len(hub) / num_nodes
# inds = densities>0
# densities = densities[inds]
# fractions = fractions[inds]
# # %%
# plt.figure()
# plt.scatter(densities, fractions, alpha=0.4)
# slope, intercept, r_value, p_value, std_err = stats.linregress(densities, fractions)
# line = slope*densities+intercept
# plt.plot(densities, line, 'k--', label='y={:.2f}x+{:.2f}'.format(slope,intercept))
# plt.scatter([],[], label='r={:.2f}'.format(r_value), s=20)
# plt.scatter([],[], label='p value={}'.format(p_value), s=20)
# plt.scatter([],[], label='standard error={}'.format(std_err), s=20)
# #end
# plt.legend(fontsize=9)
# plt.xlabel('density')
# plt.ylabel('fraction of hub nodes')
# plt.savefig('./plots/hubfrac_density.jpg')
# %%
# mean_degree = np.zeros((len(rows), len(cols)))
# mean_degree_nh = np.zeros((len(rows), len(cols)))
# mean_degree_rh = np.zeros((len(rows), len(cols)))
# mean_degree_df = np.zeros((len(rows), len(cols)))
# repeats = len(G_dict[rows[0]][cols[0]])
# for row_ind, row in enumerate(rows):
#   print(row)
#   for col_ind, col in enumerate(cols):
#     m = np.zeros(repeats)
#     m_nh = np.zeros(repeats)
#     m_rh = np.zeros(repeats)
#     for r in range(repeats):
#       G = G_dict[row][col][r].copy()
#       m[r] = np.mean(list(dict(G.degree()).values()))
#       degrees = dict(nx.degree(G))
#       max_deg = np.max(list(degrees.values()))
#       hub = [n for n, d in list(degrees.items()) if d >= max_deg/2]
#       m_nh[r] = np.mean([d for n, d in list(degrees.items()) if d < max_deg/2])
#       G.remove_nodes_from(hub)
#       m_rh[r] = np.mean(list(dict(G.degree()).values()))
#     mean_degree[row_ind, col_ind] = np.nanmean(m)
#     mean_degree_nh[row_ind, col_ind] = np.nanmean(m_nh)
#     mean_degree_rh[row_ind, col_ind] = np.nanmean(m_rh)
#     mean_degree_df[row_ind, col_ind] = mean_degree_nh[row_ind, col_ind] - mean_degree_rh[row_ind, col_ind]
# # %%
# fig = plt.figure(figsize=(7, 8))
# plt.subplot(2, 2, 1)
# for row_ind, row in enumerate(rows):
#   plt.plot(cols, mean_degree[row_ind, :], label=row, alpha=1)
# plt.xticks(rotation=90)
# plt.title('mean degree')
# plt.legend()
# plt.tight_layout()
# plt.subplot(2, 2, 2)
# for row_ind, row in enumerate(rows):
#   plt.plot(cols, mean_degree_nh[row_ind, :], label=row, alpha=1)
# plt.xticks(rotation=90)
# plt.title('mean degree of non-hub nodes')
# plt.tight_layout()
# plt.subplot(2, 2, 3)
# for row_ind, row in enumerate(rows):
#   plt.plot(cols, mean_degree_rh[row_ind, :], label=row, alpha=1)
# plt.xticks(rotation=90)
# plt.title('mean degree after removing hub nodes')
# plt.tight_layout()
# plt.subplot(2, 2, 4)
# for row_ind, row in enumerate(rows):
#   plt.plot(cols, mean_degree_df[row_ind, :], label=row, alpha=1)
# plt.xticks(rotation=90)
# plt.title('decrease in mean degree \n after removing hub nodes')
# plt.tight_layout()
# # plt.show()
# num = threshold if measure=='pearson' else percentile
# figname = './plots/mean_degree.jpg'
# plt.savefig(figname)
# %%
##################### rewire hub nodes to make them normal
# algorithm = 'rewire_hub_connection'
# rewired_G_dict = {}
# # repeats = len(G_dict[rows[0]][cols[0]])
# repeats = 10
# for row_ind, row in enumerate(rows):
#   print(row)
#   if not row in rewired_G_dict:
#       rewired_G_dict[row] = {}
#   for col_ind, col in enumerate(cols):
#     print(col)
#     rewired_G_dict[row][col] = []
#     for r in range(repeats):
#       G = G_dict[row][col][r].copy()
#       while not (nx.number_of_nodes(G) > 100 and nx.number_of_edges(G) > 100):
#         r = r + repeats
#         G = G_dict[row][col][r].copy()
#       degrees = dict(nx.degree(G))
#       max_deg = np.max(list(degrees.values()))
#       hub = [n for n, d in list(degrees.items()) if d >= max_deg/2]
#       A = nx.adjacency_matrix(G).todense()
#       # nonhub_pairs = np.array([[i, j] for i,j in zip(np.where(A==0)[0], np.where(A==0)[1]) if (i not in hub) and (j not in hub) and (i < j)])
#       nonhub_pairs = np.array([[i, j] for i,j in itertools.combinations(G.nodes(), 2) if (i not in hub) and (j not in hub) and (i < j) and (not G.has_edge(i, j))])
#       np.random.shuffle(nonhub_pairs)
#       acc_rewire = []
#       for h in hub:
#         obj_deg = np.random.poisson(np.mean([d for n, d in list(degrees.items()) if d < max_deg/2]))
#         # ns = [n for n in G.neighbors(h)]
#         ns_deg = {neighbor:G.degree(neighbor) for neighbor in G.neighbors(h)}
#         ns = [n for n, v in sorted(ns_deg.items(), key=lambda item: item[1], reverse=True)]
#         if len(ns) > obj_deg:
#           acc_rewire.append(len(ns) - obj_deg)
#           i = 0
#           actual_rewire = 0
#           for p in range(int(np.sum(acc_rewire[:-1])), np.sum(acc_rewire)):
#             num_edges = G.number_of_edges()
#             assert not G.has_edge(*nonhub_pairs[p])
#             weight=G.get_edge_data(h, ns[i])['weight']
#             G.remove_edge(h, ns[i])
#             # choose another edge if the removal disconnects the graph
#             while not nx.is_connected(G):
#               G.add_edge(h, ns[i], weight=weight)
#               i += 1
#               if i >= len(ns):
#                 print('Cannot remove edges from node {} anymore otherwise graph will become unconnected'.format(h))
#                 break
#               weight=G.get_edge_data(h, ns[i])['weight']
#               G.remove_edge(h, ns[i])
#             if i >= len(ns):
#               acc_rewire[-1] = actual_rewire
#               break
#             else:
#               i += 1
#               actual_rewire += 1
#               G.add_edge(*nonhub_pairs[p], weight=weight)
#             assert G.number_of_edges() == num_edges
#       rewired_G_dict[row][col].append(G)
# %%
cc = True
num_sample = 10
metric = metric_stimulus_individual(G_dict, sign, num_sample, threshold, percentile, measure, weight, cc)
# %%
full_degrees = get_full_degrees(G_dict, num_sample, measure)
# %%
def pairwise_JS_divergence_met(G_dict, full_degrees, metric):
  metric_names = get_metric_names(G_dict)
  JS_divergence = []
  met_diff = {met:[] for met in metric_names}
  full_degrees_flat = full_degrees.reshape(-1, full_degrees.shape[-1])
  metric_flat = metric.reshape(-1, metric.shape[-1])
  shape = full_degrees_flat.shape
  for i in range(shape[0]):
    for j in range(i + 1, shape[0]):
      JS_divergence.append(distance.jensenshannon(full_degrees_flat[i, :], full_degrees_flat[j, :]))
      for met_ind, met in enumerate(metric_names):
        met_diff[met].append(abs(metric_flat[i, met_ind] - metric_flat[j, met_ind]))
  return JS_divergence, met_diff

JS_divergence, met_diff = pairwise_JS_divergence_met(full_degrees, metric)
# %%
def plot_divergence_met_diff(JS_divergence, met_diff, G_dict, measure, weight):
  metric_names = get_metric_names(G_dict)
  plots_shape = (3, 3) if len(metric_names) == 9 else (2, 4)
  fig = plt.figure(figsize=(20, 10))
  for metric_ind, metric_name in enumerate(metric_names):
    print(metric_name)
    plt.subplot(*plots_shape, metric_ind + 1)
    plt.scatter(JS_divergence, met_diff[metric_name], alpha=0.4)
    plt.gca().set_title(r'$\Delta$ in '+metric_name, fontsize=30, rotation=0)
    plt.xticks(rotation=90)
    plt.xlabel('Jenson Shannon divergence')
  plt.tight_layout()
  # plt.show()
  num = threshold if measure=='pearson' else percentile
  figname = './plots/divergence_met_diff_weighted_{}_{}_{}.jpg'.format(sign, measure, num) if weight else './plots/divergence_met_diff_{}_{}_{}.jpg'.format(sign, measure, num)
  plt.savefig(figname)

plot_divergence_met_diff(JS_divergence, met_diff, G_dict, measure, weight)
# %%
print("--- %s minutes in total" % ((time.time() - start_time)/60))
