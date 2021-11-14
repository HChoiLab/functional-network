# %%
import numpy as np
import os
import matplotlib
from matplotlib import pyplot as plt
import networkx as nx
import community
import pickle
import random

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
  ix = np.random.choice(len(i), min_num, replace=False)
  sample_seq = np.zeros_like(sequences)
  sample_seq[i[ix], j[ix]] = sequences[i[ix], j[ix]]
  return sample_seq

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

def load_adj_regions_downsample_as_graph_whole(directory, weight, measure, threshold, percentile):
  G_dict = {}
  files = os.listdir(directory)
  files.sort(key=lambda x:int(x[:9]))
  for file in files:
    if file.endswith(".npy"):
      print(file)
      adj_mat = np.load(os.path.join(directory, file))
      if measure == 'pearson':
        adj_mat[adj_mat < threshold] = 0
      else:
        adj_mat[np.where(adj_mat<np.nanpercentile(np.abs(adj_mat), percentile))] = 0
      mouseID = file.split('_')[0]
      stimulus_name = file.replace('.npy', '').replace(mouseID + '_', '')
      if not mouseID in G_dict:
        G_dict[mouseID] = {}
      G_dict[mouseID][stimulus_name] = []
      for i in range(adj_mat.shape[2]):
        G_dict[mouseID][stimulus_name].append(generate_graph(adj_mat=adj_mat[:, :, i], cc=False, weight=weight))
  return G_dict

def plot_example_graph(G_dict, area_dict, session_ids, stimulus_names, mouse_id, stimulus_id, baseline=False):
  fig = plt.figure(figsize=(10, 8))
  G = G_dict[str(session_ids[mouse_id])][stimulus_names[stimulus_id]][0]
  nx.set_node_attributes(G, area_dict[str(session_ids[mouse_id])], "area")
  if nx.is_directed(G):
      Gcc = sorted(nx.strongly_connected_components(G), key=len, reverse=True)
      G = G.subgraph(Gcc[0])
  else:
      Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
      G = G.subgraph(Gcc[0])
  G = nx.k_core(G)
  try:
      edges, weights = zip(*nx.get_edge_attributes(G,'weight').items())
  except:
      edges = nx.edges(G)
      weights = np.ones(len(edges))
  degrees = dict(G.degree)
  pos = nx.spring_layout(G)
  areas = [G.nodes[n]['area'] for n in G.nodes()]
  areas_uniq = unique(areas)
  colors = [customPalette[areas_uniq.index(area)] for area in areas]
  # pos = nx.spring_layout(G, k=0.8, iterations=50) # make nodes as seperate as possible
  # nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=np.log(np.array(weights)*100), width=4 * np.log(np.array(weights)*100), edge_cmap=plt.cm.Greens, alpha=0.9)
  nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='green', width=4 * np.log(np.array(weights)*100), alpha=0.08)
  nx.draw_networkx_nodes(G, pos, nodelist=degrees.keys(), node_size=[2.5 * v ** 2 for v in degrees.values()], 
  node_color=colors, alpha=1)
  for index, a in enumerate(areas_uniq):
      plt.scatter([],[], c=customPalette[index], label=a, s=20, alpha=1)
  legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
  for handle in legend.legendHandles:
      handle.set_sizes([40.0])
  plt.axis('off')
  plt.tight_layout()
  image_name = './plots/example_graph_baseline_{}_{}.jpg'.format(mouse_id, stimulus_id) if baseline else './plots/example_graph_{}_{}.jpg'.format(mouse_id, stimulus_id)
  plt.savefig(image_name)
  # plt.savefig(image_name.replace('.jpg', '.pdf'), transparent=True)
  # plt.show()

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
        G = G_dict[row][col][0].copy() if col in G_dict[row] else nx.Graph()
        weights = list(nx.get_edge_attributes(G,'weight').values())
        random.shuffle(weights)
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
        for ind, (_, _, d) in enumerate(G.edges(data=True)):
          d['weight'] = weights[ind]
        rewired_G_dict[row][col].append(G)
  return rewired_G_dict

def unique(l):
  u, ind = np.unique(l, return_index=True)
  return list(u[np.argsort(ind)])
# %%
########## raster plot
mouse_id, stimulus_id = 1, 7
session_ids = [719161530, 750749662, 755434585, 756029989, 791319847]
stimulus_names = ['spontaneous', 'flashes', 'gabors',
        'drifting_gratings', 'static_gratings',
          'natural_scenes', 'natural_movie_one', 'natural_movie_three']
min_len, min_num = (260000, 578082)
measure = 'pearson'
# measure = 'cosine'
# measure = 'correlation'
# measure = 'MI'
# measure = 'causality'
directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
sequences = load_npz(os.path.join(directory, str(session_ids[mouse_id]) + '_' + stimulus_names[stimulus_id] + '.npz'))
sample_seq = down_sample(sequences, min_len, min_num)
a_file = open('./data/ecephys_cache_dir/sessions/area_dict.pkl', 'rb')
area_dict = pickle.load(a_file)
# change the keys of area_dict from int to string
int_2_str = dict((session_id, str(session_id)) for session_id in session_ids)
area_dict = dict((int_2_str[key], value) for (key, value) in area_dict.items())
a_file.close()
areas = list(area_dict[str(session_ids[mouse_id])].values())
areas_uniq = unique(areas)
areas_num = [(np.array(areas)==a).sum() for a in areas_uniq]
areas_start_pos = list(np.insert(np.cumsum(areas_num)[:-1], 0, 0))
sequence_by_area = {a:[name for name, age in area_dict[str(session_ids[mouse_id])].items() if age == a] for a in areas_uniq}
# %%
sorted_sample_seq = np.vstack([sample_seq[sequence_by_area[a], :10000] for a in areas_uniq])
spike_pos = [np.nonzero(t)[0] / 1000 for t in sorted_sample_seq[:, :10000]] # divided by 1000 cuz bin size is 1 ms
colors1 = [customPalette[i] for i in sum([[areas_uniq.index(a)] * areas_num[areas_uniq.index(a)] for a in areas_uniq], [])]
uniq_colors = unique(colors1)
text_pos = [s + (areas_num[areas_start_pos.index(s)] - 1) / 2 for s in areas_start_pos]
colors2 = 'black'
lineoffsets2 = 1
linelengths2 = 1
# create a horizontal plot
fig = plt.figure(figsize=(6, 8))
plt.eventplot(spike_pos, colors=colors1, lineoffsets=lineoffsets2,
                    linelengths=linelengths2)
for ind, t_pos in enumerate(text_pos):
  plt.text(-1, t_pos, areas_uniq[ind], size=10, color=uniq_colors[ind])
plt.axis('off')
plt.gca().invert_yaxis()
plt.tight_layout()
# plt.show()
plt.savefig('./plots/raster.jpg')

# %%
############ plot functional grah
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
measure = 'pearson'
########## load networks with multiple realizations of downsampling
directory = './data/ecephys_cache_dir/sessions/adj_mat_{}/'.format(measure)
threshold = 0.012
# thresholds = list(np.arange(0, 0.014, 0.001))
percentile = 99.8
weight = True # weighted network
G_dict = load_adj_regions_downsample_as_graph_whole(directory, weight, measure, threshold, percentile)
# %%
com = CommunityLayout()
# mouse_id = 0
# stimulus_id = 1
for mouse_id in range(5):
  print(mouse_id)
  for stimulus_id in range(8):
    print(stimulus_id)
    plot_example_graph(G_dict, area_dict, session_ids, stimulus_names, mouse_id, stimulus_id)
# %%
mouse_id, stimulus_id = 1, 7
plot_example_graph(G_dict, area_dict, session_ids, stimulus_names, mouse_id, stimulus_id, baseline=False)
# %%
num_rewire = 1
algorithm = 'configuration_model'
cc = False
rewired_G_dict = random_graph_baseline(G_dict, num_rewire, algorithm, measure, cc, Q=100)
mouse_id, stimulus_id = 1, 7
plot_example_graph(rewired_G_dict, area_dict, session_ids, stimulus_names, mouse_id, stimulus_id, baseline=True)
# %%
[name for name, age in area_dict[str(session_ids[mouse_id])].items() if age == 'LP']
