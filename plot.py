# %%
import numpy as np
import os
import matplotlib
from matplotlib import pyplot as plt
import networkx as nx
import community
import pickle

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

# %%
########## raster plot
min_len, min_num = (260000, 578082)
measure = 'pearson'
# measure = 'cosine'
# measure = 'correlation'
# measure = 'MI'
# measure = 'causality'
directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
files = os.listdir(directory)
files.sort(key=lambda x:int(x[:9]))
for file in files:
  if file.endswith(".npz"):
    print(file)
sequences = load_npz(os.path.join(directory, file))
sample_seq = down_sample(sequences, min_len, min_num)
# %%
spike_pos = [np.nonzero(t)[0] / 1000 for t in sample_seq[:, :10000]]
colors2 = 'black'
lineoffsets2 = 1
linelengths2 = 1
# create a horizontal plot
plt.eventplot(spike_pos, colors=colors2, lineoffsets=lineoffsets2,
                    linelengths=linelengths2)
plt.axis('off')
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
mouse_id = 0
fig = plt.figure(figsize=(8, 8))
G = G_dict[str(session_ids[mouse_id])][stimulus_names[2]][0]
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
nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=np.log(np.array(weights)*100), width=4 * np.log(np.array(weights)*100), edge_cmap=plt.cm.Greens, alpha=0.9)
nx.draw_networkx_nodes(G, pos, nodelist=degrees.keys(), node_size=[2.5 * v ** 2 for v in degrees.values()], 
node_color=colors, alpha=1)
areas = [G.nodes[n]['area'] for n in G.nodes()]
areas_uniq = list(set(areas))
for index, a in enumerate(areas_uniq):
    plt.scatter([],[], c=customPalette[index], label=a, s=20, alpha=1)
legend = plt.legend(loc='upper right', fontsize=16)
for handle in legend.legendHandles:
    handle.set_sizes([60.0])
plt.tight_layout()
image_name = './plots/example_graph.jpg'
plt.savefig(image_name)
# plt.savefig(image_name.replace('.jpg', '.pdf'), transparent=True)
# plt.show()
# %%
