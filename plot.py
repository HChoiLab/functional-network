# %%
from library import *
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

area_dict, active_area_dict, mean_speed_df = load_other_data(session_ids)
directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
path = directory.replace('spiking_sequence', 'adj_mat_ccg_highland_corrected')
if not os.path.exists(path):
  os.makedirs(path)
G_ccg_dict, offset_dict, duration_dict = load_highland_xcorr(path, active_area_dict, weight=True)
measure = 'ccg'

G_ccg_dict = remove_gabor(G_ccg_dict)
G_ccg_dict = remove_thalamic(G_ccg_dict, area_dict, visual_regions)
offset_dict = remove_thalamic_mat(offset_dict, active_area_dict, visual_regions)
duration_dict = remove_thalamic_mat(duration_dict, active_area_dict, visual_regions)
active_area_dict = remove_thalamic_area(active_area_dict, visual_regions)
n = 4
S_ccg_dict = add_sign(G_ccg_dict)
S_ccg_dict = add_offset(S_ccg_dict, offset_dict)
# customPalette = ['#630C3A', '#39C8C6', '#D3500C', '#FFB139', 'palegreen', 'darkblue', 'slategray', '#a6cee3', '#b2df8a', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#6a3d9a', '#b15928']

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
  # for index, a in enumerate(areas_uniq):
  #     plt.scatter([],[], c=customPalette[index], label=a, s=20, alpha=1)
  # legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
  # for handle in legend.legendHandles:
  #     handle.set_sizes([40.0])
  plt.axis('off')
  plt.tight_layout()
  image_name = './plots/example_graph_baseline_{}_{}.jpg'.format(mouse_id, stimulus_id) if baseline else './plots/example_graph_{}_{}.jpg'.format(mouse_id, stimulus_id)
  # plt.savefig(image_name)
  plt.savefig(image_name.replace('.jpg', '.pdf'), transparent=True)
  # plt.show()


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

def func_powerlaw(x, m, c):
  return x**m * c

def plot_degree_distributions(G_dict, mouse_id, measure, threshold, percentile, cc=False):
  stimulus_names = ['spontaneous', 'flashes', 'gabors',
          'drifting gratings', 'static gratings',
            'natural images', 'natural movies']
  ind = 1
  rows, cols = get_rowcol(G_dict, measure)
  fig = plt.figure(figsize=(4*len(cols[:-1]), 4))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  for col_ind, col in enumerate(cols[:-1]):
    plt.subplot(1, len(cols[:-1]), ind)
    plt.gca().set_title(stimulus_names[col_ind], fontsize=30, rotation=0)
    plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ind += 1
    G = G_dict[rows[mouse_id]][col][0] if col in G_dict[rows[mouse_id]] else nx.Graph()
    if G.number_of_nodes() > 2 and G.number_of_edges() > 0:
      if cc:
        if nx.is_directed(G):
          Gcc = sorted(nx.strongly_connected_components(G), key=len, reverse=True)
          G = G.subgraph(Gcc[0])
        else:
          Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
          G = G.subgraph(Gcc[0])
      G.remove_nodes_from(list(nx.isolates(G))) # remove isolated nodes
      degree_freq = nx.degree_histogram(G)[1:]
      degrees = np.array(range(1, len(degree_freq) + 1))
      g_degrees = list(dict(G.degree()).values())
      [alpha, xmin, L] = plfit(g_degrees, 'finite')
      # C is normalization constant that makes sure the sum is equal to real data points
      C = (np.array(degree_freq) / sum(degree_freq))[degrees>=xmin].sum() / np.power(degrees[degrees>=xmin], -alpha).sum()
      plt.scatter([],[], label=r'$\alpha$={:.1f}'.format(alpha), s=20)
      plt.scatter([],[], label=r'$x_{min}$=' + '{}'.format(xmin), s=20)
      plt.scatter([],[], label=r'$ll$={:.1f}'.format(L), s=20)
      plt.plot(degrees, np.array(degree_freq) / sum(degree_freq),'go-')
      plt.plot(degrees[degrees>=xmin], func_powerlaw(degrees[degrees>=xmin], *np.array([-alpha, C])), linestyle='--', linewidth=2, color='black')
      
      plt.legend(loc='lower left', fontsize=15)
      plt.xlabel('Degree', size=25)
      if col_ind == 0:
        plt.ylabel('Frequency', size=25)
      plt.xscale('log')
      plt.yscale('log')
      
  plt.tight_layout()
  th = threshold if measure == 'pearson' else percentile
  image_name = './plots/example_degree_distribution_cc_{}_{}.jpg'.format(measure, th) if cc else './plots/example_degree_distribution_{}_{}.jpg'.format(measure, th)
  # plt.show()
  # plt.savefig(image_name, dpi=300)
  plt.savefig(image_name.replace('jpg', 'pdf'), transparent=True)

def region_connection_delta_heatmap(G_dict, area_dict, mouse_id, regions, measure, threshold, percentile):
  
  rows, cols = get_rowcol(G_dict, measure)
  cols.remove('spontaneous')
  row = rows[mouse_id]
  stimulus_names = ['spontaneous', 'flashes', 'gabors',
            'drifting gratings', 'static gratings',
              'natural images', 'natural movies']
  region_connection_bl = np.zeros((len(regions), len(regions)))
  region_connection = np.zeros((len(cols), len(regions), len(regions)))
  G = G_dict[row]['spontaneous'][0]
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
        region_connection_bl[region_ind_i, region_ind_j] = np.sum(A[region_indices_i[:, None], region_indices_j])
        assert np.sum(A[region_indices_i[:, None], region_indices_j]) == len(A[region_indices_i[:, None], region_indices_j].nonzero()[0])
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
          region_connection[col_ind, region_ind_i, region_ind_j] = np.sum(A[region_indices_i[:, None], region_indices_j])
          assert np.sum(A[region_indices_i[:, None], region_indices_j]) == len(A[region_indices_i[:, None], region_indices_j].nonzero()[0])
  scale_min = ((region_connection[:, :, :]-region_connection_bl[None, :, :])/region_connection_bl.sum()).min()
  scale_max = ((region_connection[:, :, :]-region_connection_bl[None, :, :])/region_connection_bl.sum()).max()
  ind = 1
  fig = plt.figure(figsize=(4*len(cols[:-1]), 5))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  for col_ind, col in enumerate(cols[:-1]):
    plt.subplot(1, len(cols[:-1]), ind)
    plt.gca().set_title(stimulus_names[col_ind + 1], fontsize=30, rotation=0)
    plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ind += 1
    cbar = True if col_ind == len(cols[:-1]) - 1 else False
    # ytick = True if col_ind == 0 - 1 else False
    sns_plot = sns.heatmap((region_connection[col_ind, :, :]-region_connection_bl)/region_connection_bl.sum(), vmin=scale_min, vmax=scale_max,center=0, cmap="RdBu_r", cbar=cbar, yticklabels=False) #  cmap="YlGnBu"
    sns_plot.tick_params(axis='both', which='both', length=0)
    # sns_plot = sns.heatmap((region_connection-region_connection_bl)/region_connection_bl.sum(), cmap="YlGnBu")
    sns_plot.set_xticks(np.arange(len(regions))+0.5)
    sns_plot.set_xticklabels(regions, rotation=90)
    sns_plot.set_yticks(np.arange(len(regions))+0.5)
    if col_ind == 0:
      sns_plot.set_yticklabels(regions, rotation=0)
    sns_plot.invert_yaxis()
  plt.tight_layout()
  # plt.show()
  num = threshold if measure=='pearson' else percentile
  # plt.savefig('./plots/region_connection_delta_scale_{}_{}.jpg'.format(measure, num))
  plt.savefig('./plots/example_region_connection_delta_scale_{}_{}.pdf'.format(measure, num), transparent=True)

def flat(array):
  return array.reshape(array.size//array.shape[-1], array.shape[-1])

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
fig = plt.figure(figsize=(10, 16))
plt.eventplot(spike_pos, colors=colors1, lineoffsets=lineoffsets2,
                    linelengths=linelengths2)
for ind, t_pos in enumerate(text_pos):
  plt.text(-1.2, t_pos, areas_uniq[ind], size=20, color=uniq_colors[ind])
plt.axis('off')
plt.gca().invert_yaxis()
plt.tight_layout()
# plt.show()
# plt.savefig('./plots/raster.jpg')
plt.savefig('./plots/raster.pdf', transparent=True)

# %%
############ load G_dict
visual_regions = ['VISp', 'VISl', 'VISrl', 'VISal', 'VISpm', 'VISam', 'LGd', 'LP']
stimulus_names = ['spontaneous', 'flashes', 'gabors',
        'drifting_gratings', 'static_gratings',
          'natural_scenes', 'natural_movie_one', 'natural_movie_three']
threshold = 0.012
session_ids = [719161530, 750749662, 755434585, 756029989, 791319847]
a_file = open('./data/ecephys_cache_dir/sessions/area_dict.pkl', 'rb')
area_dict = pickle.load(a_file)
# change the keys of area_dict from int to string
int_2_str = dict((session_id, str(session_id)) for session_id in session_ids)
area_dict = dict((int_2_str[key], value) for (key, value) in area_dict.items())
a_file.close()
measure = 'pearson'
########## load networks with multiple realizations of downsampling
directory = './data/ecephys_cache_dir/sessions/adj_mat_{}/'.format(measure)
# thresholds = list(np.arange(0, 0.014, 0.001))
percentile = 99.8
weight = True # weighted network
G_dict = load_adj_regions_downsample_as_graph_whole(directory, weight, measure, threshold, percentile)
# %%
############ plot functional grah
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
# violin plot for 6 metrics
session_ids = [719161530, 750749662, 755434585, 756029989, 791319847]
stimulus_names = ['spontaneous', 'flashes', 'gabors',
        'drifting_gratings', 'static_gratings',
          'natural_scenes', 'natural_movie_one', 'natural_movie_three']
rows, cols = session_ids, stimulus_names
num_sample = 10
density = np.zeros((len(rows), len(cols), num_sample))
for row_ind, row in enumerate(rows):
  for col_ind, col in enumerate(cols):
    for i in range(num_sample):
      density[row_ind, col_ind, i] = nx.density(G_dict[str(row)][col][i])
a_file = open('./data/ecephys_cache_dir/sessions/0.012_100.pkl', 'rb')
stat_dict = pickle.load(a_file)
a_file.close()
stimulus_names = ['spontaneous', 'flashes', 'gabors',
        'drifting gratings', 'static gratings',
          'natural images', 'natural movies']
metric_names = ['assortativity', 'betweenness', 'closeness', 'clustering', 'density', 'efficiency', 'modularity', 'transitivity']
metric_inds = [0, 2, 5, 6, 7]
new_me_names = ['density'] + [metric_names[i] for i in metric_inds]
delta_metrics = stat_dict['delta metrics'].squeeze()[:, :, :, metric_inds] # (len(rows), len(cols), num_rewire)
stimuli_inds = {s:stimulus_names.index(s) for s in stimulus_names}
# combine natural movie 1 and 3
stimuli_inds[stimulus_names[-1]] = [stimuli_inds[stimulus_names[-1]], stimuli_inds[stimulus_names[-1]] + 1]
fig = plt.figure(figsize=(20, 10))
for metric_ind, metric_name in enumerate(new_me_names):
  print(metric_name)
  plt.subplot(2, 3, metric_ind + 1)
  if metric_ind == 0:
    metric = pd.concat([pd.DataFrame(np.concatenate((density[:, stimuli_inds[s_type], :].flatten()[:, None], np.array([s_type] * density[:, stimuli_inds[s_type], :].flatten().size)[:, None]), 1), columns=[metric_name, 'type']) for s_type in stimuli_inds], ignore_index=True)
  else:
    metric = pd.concat([pd.DataFrame(np.concatenate((delta_metrics[:, stimuli_inds[s_type], :, metric_ind-1].flatten()[:, None], np.array([s_type] * delta_metrics[:, stimuli_inds[s_type], :, metric_ind-1].flatten().size)[:, None]), 1), columns=[metric_name, 'type']) for s_type in stimuli_inds], ignore_index=True)
  metric[metric_name] = pd.to_numeric(metric[metric_name])
  ax = sns.violinplot(x='type', y=metric_name, data=metric, color=sns.color_palette("Set2")[0])
  ax.set(xlabel=None)
  ax.set(ylabel=None)
  if metric_ind < 3:
    plt.xticks([])
  else:
    plt.xticks(fontsize=20, rotation=90)
  if metric_ind > 0:
    metric_name = r'$\Delta$' + metric_name
  plt.gca().set_title(metric_name, fontsize=30, rotation=0)
plt.legend()
plt.tight_layout()
# plt.show()
num = threshold if measure=='pearson' else percentile
figname = './plots/violin_delta_metric_stimulus_weighted_{}_{}.jpg'.format(measure, num) if weight else './plots/violin_delta_metric_stimulus_{}_{}.jpg'.format(measure, num)
# plt.savefig(figname)
plt.savefig(figname.replace('.jpg', '.pdf'), transparent=True)
# %%
########## plot degree distribution of an example mouse
mouse_id = 0
cc = False
plot_degree_distributions(G_dict, mouse_id, measure, threshold, percentile, cc=cc)
# %%
########### plot example connection heatmap
mouse_id = 3
sns.set(font_scale=1.8)
region_connection_delta_heatmap(G_dict, area_dict, mouse_id, visual_regions, measure, threshold, percentile)
# %%
######### plot delta metrics VS threshold
measure = 'pearson'
########## load networks with multiple realizations of downsampling
directory = './data/ecephys_cache_dir/sessions/adj_mat_{}/'.format(measure)
# thresholds = list(np.arange(0, 0.014, 0.001))
percentile = 99.8
weight = True # weighted network
a_file = open('./data/ecephys_cache_dir/sessions/stat_dict_2.pkl', 'rb')
stat_dict = pickle.load(a_file)
a_file.close()
thresholds = [0, 0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.014]
session_ids = [719161530, 750749662, 755434585, 756029989, 791319847]
stimulus_names = ['spontaneous', 'flashes', 'gabors',
        'drifting_gratings', 'static_gratings',
          'natural_scenes', 'natural_movie_one', 'natural_movie_three']
rows, cols = session_ids, stimulus_names
density = np.load('./data/ecephys_cache_dir/sessions/density.npy') # len(rows), len(cols), num_sample, len(thresholds)
# num_sample = 1
# density = np.zeros((len(rows), len(cols), num_sample, len(thresholds)))
# for th_ind, threshold in enumerate(thresholds):
#   print(threshold)
#   G_dict = load_adj_regions_downsample_as_graph_whole(directory, weight, measure, threshold, percentile)
#   for row_ind, row in enumerate(rows):
#     for col_ind, col in enumerate(cols):
#       for i in range(num_sample):
#         density[row_ind, col_ind, i, th_ind] = nx.density(G_dict[str(row)][col][i])
stimulus_names = ['spontaneous', 'flashes', 'gabors',
        'drifting gratings', 'static gratings',
          'natural images', 'natural movies']
metric_names = ['assortativity', 'betweenness', 'closeness', 'clustering', 'density', 'efficiency', 'modularity', 'transitivity']
metric_inds = [0, 2, 5, 6, 7]
new_me_names = ['density'] + [metric_names[i] for i in metric_inds]
# %%
sns.set(font_scale=1.8)
delta_metrics = stat_dict['delta metrics'].squeeze()[:, :, :, metric_inds, :] # (len(rows), len(cols), num_rewire, len(new_me_names), len(thresholds))
stimuli_inds = {s:stimulus_names.index(s) for s in stimulus_names}
# combine natural movie 1 and 3
stimuli_inds[stimulus_names[-1]] = [stimuli_inds[stimulus_names[-1]], stimuli_inds[stimulus_names[-1]] + 1]
fig = plt.figure(figsize=(20, 10))
for metric_ind, metric_name in enumerate(new_me_names):
  print(metric_name)
  plt.subplot(2, 3, metric_ind + 1)
  title = r'$\Delta$'+metric_name if metric_ind > 0 else metric_name
  plt.title(title, fontsize=30, rotation=0)
  # plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
  if metric_ind == 0:
    sns_plot = sns.heatmap(flat(density.mean(axis=2)).astype(float), cmap="YlGnBu", norm=LogNorm())
  else:
    sns_plot = sns.heatmap(flat(delta_metrics[:, :, :, metric_ind - 1, :].mean(axis=2)).astype(float), cmap="YlGnBu")
  sns_plot.set_xticks(np.arange(len(thresholds))+0.5)
  sns_plot.tick_params(axis='both', which='both', length=0)
  sns_plot.set(xticklabels=[], yticklabels=[])
  if metric_ind == 0 or metric_ind == 3:
    sns_plot.set(ylabel='trial')
  if metric_ind >= 3:
    sns_plot.set_xticklabels(thresholds, rotation=90)
    sns_plot.set(xlabel='threshold')
# plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.tight_layout()
plt.savefig('./plots/delta_metrics_threshold_{}.pdf'.format(measure), transparent=True)
# %%
def plot_kstest(data_dict, name, measure, n):
  fig = plt.figure(figsize=(7, 5.5))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  keys = data_dict.keys()
  stimulus_pvalue = np.zeros((len(keys), len(keys)))
  stimulus_pvalue[:] = np.nan
  for key_ind1, key1 in enumerate(keys):
    for key_ind2, key2 in enumerate(keys):
      if not key1 == key2:
        p_less = stats.ks_2samp(data_dict[key1], data_dict[key2], alternative='less')[1]
        p_greater = stats.ks_2samp(data_dict[key1], data_dict[key2], alternative='greater')[1]
        stimulus_pvalue[key_ind1, key_ind2] = min(p_less, p_greater)
        # print(np.mean(all_purity[col1]), np.mean(all_purity[col2]))
  # print(stimulus_pvalue)
  sns_plot = sns.heatmap(stimulus_pvalue.astype(float), annot=True,cmap="RdBu",norm=colors.LogNorm(5.668934240362814e-06, 1))# cmap="YlGnBu" (0.000001, 1) for 0.01 confidence level, (0.0025, 1) for 0.05
  # sns_plot = sns.heatmap(region_connection.astype(float), vmin=0, cmap="YlGnBu")
  sns_plot.set_xticks(np.arange(len(cols))+0.5)
  sns_plot.set_xticklabels(cols, rotation=90)
  sns_plot.set_yticks(np.arange(len(cols))+0.5)
  sns_plot.set_yticklabels(cols, rotation=0)
  sns_plot.invert_yaxis()
  plt.title('p-value of {}'.format(name), size=18)
  plt.tight_layout()
  image_name = './plots/pvalue_{}_{}_{}fold.jpg'.format(name, measure, n)
  # plt.show()
  plt.savefig(image_name)

density_dict = {}
rows, cols = get_rowcol(G_ccg_dict)
for col in cols:
  density_dict[col] = []
  for row in rows:
    density_dict[col].append(nx.density(G_ccg_dict[row][col]))
plot_kstest(density_dict, 'density', measure, n)
# %%
# violin plot for 6 metrics
def violin_data(data_dict):
  session_ids = [719161530, 750749662, 755434585, 756029989, 791319847]
  stimulus_names = ['spontaneous', 'flashes', 'gabors',
          'drifting_gratings', 'static_gratings',
            'natural_scenes', 'natural_movie_one', 'natural_movie_three']
  rows, cols = session_ids, stimulus_names
  num_sample = 10
  density = np.zeros((len(rows), len(cols), num_sample))
  for row_ind, row in enumerate(rows):
    for col_ind, col in enumerate(cols):
      for i in range(num_sample):
        density[row_ind, col_ind, i] = nx.density(G_dict[str(row)][col][i])
  a_file = open('./data/ecephys_cache_dir/sessions/0.012_100.pkl', 'rb')
  stat_dict = pickle.load(a_file)
  a_file.close()
  stimulus_names = ['spontaneous', 'flashes', 'gabors',
          'drifting gratings', 'static gratings',
            'natural images', 'natural movies']
  metric_names = ['assortativity', 'betweenness', 'closeness', 'clustering', 'density', 'efficiency', 'modularity', 'transitivity']
  metric_inds = [0, 2, 5, 6, 7]
  new_me_names = ['density'] + [metric_names[i] for i in metric_inds]
  delta_metrics = stat_dict['delta metrics'].squeeze()[:, :, :, metric_inds] # (len(rows), len(cols), num_rewire)
  stimuli_inds = {s:stimulus_names.index(s) for s in stimulus_names}
  # combine natural movie 1 and 3
  stimuli_inds[stimulus_names[-1]] = [stimuli_inds[stimulus_names[-1]], stimuli_inds[stimulus_names[-1]] + 1]
  fig = plt.figure(figsize=(20, 10))
  for metric_ind, metric_name in enumerate(new_me_names):
    print(metric_name)
    plt.subplot(2, 3, metric_ind + 1)
    if metric_ind == 0:
      metric = pd.concat([pd.DataFrame(np.concatenate((density[:, stimuli_inds[s_type], :].flatten()[:, None], np.array([s_type] * density[:, stimuli_inds[s_type], :].flatten().size)[:, None]), 1), columns=[metric_name, 'type']) for s_type in stimuli_inds], ignore_index=True)
    else:
      metric = pd.concat([pd.DataFrame(np.concatenate((delta_metrics[:, stimuli_inds[s_type], :, metric_ind-1].flatten()[:, None], np.array([s_type] * delta_metrics[:, stimuli_inds[s_type], :, metric_ind-1].flatten().size)[:, None]), 1), columns=[metric_name, 'type']) for s_type in stimuli_inds], ignore_index=True)
    metric[metric_name] = pd.to_numeric(metric[metric_name])
    ax = sns.violinplot(x='type', y=metric_name, data=metric, color=sns.color_palette("Set2")[0])
    ax.set(xlabel=None)
    ax.set(ylabel=None)
    if metric_ind < 3:
      plt.xticks([])
    else:
      plt.xticks(fontsize=20, rotation=90)
    if metric_ind > 0:
      metric_name = r'$\Delta$' + metric_name
    plt.gca().set_title(metric_name, fontsize=30, rotation=0)
  plt.legend()
  plt.tight_layout()
  # plt.show()
  num = threshold if measure=='pearson' else percentile
  figname = './plots/violin_delta_metric_stimulus_weighted_{}_{}.jpg'.format(measure, num) if weight else './plots/violin_delta_metric_stimulus_{}_{}.jpg'.format(measure, num)
  # plt.savefig(figname)
  plt.savefig(figname.replace('.jpg', '.pdf'), transparent=True)
