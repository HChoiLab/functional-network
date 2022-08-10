# %%
from library import *
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

def remove_outliers(data, m=2.):
  data = np.array(data)
  d = np.abs(data - np.median(data))
  mdev = np.median(d)
  s = d / (mdev if mdev else 1.)
  return data[s < m]

# violin plot
def violin_data(data_dict, name, measure, n):
  keys = list(data_dict.keys())
  fig = plt.figure(figsize=(5, 5))
  data = [data_dict[key] for key in keys]
  ax = sns.violinplot(data=data, color=sns.color_palette("Set2")[0], scale='width')
  ax.set(xlabel=None)
  ax.set(ylabel=None)
  plt.xticks(range(len(keys)), keys, fontsize=10, rotation=90)
  plt.gca().set_title(name, fontsize=30, rotation=0)
  # plt.legend()
  plt.tight_layout()
  # plt.show()
  figname = './plots/violin_{}_{}_{}.jpg'.format(name, measure, n)
  plt.savefig(figname)
  # plt.savefig(figname.replace('.jpg', '.pdf'), transparent=True)

def box_data(data_dict, name, measure, n):
  keys = list(data_dict.keys())
  fig = plt.figure(figsize=(5, 5))
  data = [remove_outliers(data_dict[key], 3).tolist() for key in keys]
  ax = sns.boxplot(data=data, color=sns.color_palette("Set2")[0])
  ax.set(xlabel=None)
  ax.set(ylabel=None)
  plt.xticks(range(len(keys)), keys, fontsize=10, rotation=90)
  plt.gca().set_title(name, fontsize=30, rotation=0)
  # plt.legend()
  plt.tight_layout()
  # plt.show()
  figname = './plots/box_{}_{}_{}.jpg'.format(name, measure, n)
  plt.savefig(figname)

def plot_p_value(data_dict, name, measure, n, method='ks_test', logscale='True'):
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
        if method == 'ks_test':
          p_less = stats.ks_2samp(data_dict[key1], data_dict[key2], alternative='less')[1]
          p_greater = stats.ks_2samp(data_dict[key1], data_dict[key2], alternative='greater')[1]
        elif method == 'mwu_test':
          p_less = stats.mannwhitneyu(ratio_dict['natural_movie_three'], ratio_dict['natural_movie_one'], alternative='less', method="asymptotic")[1]
          p_greater = stats.mannwhitneyu(ratio_dict['natural_movie_three'], ratio_dict['natural_movie_one'], alternative='greater', method="asymptotic")[1]
        stimulus_pvalue[key_ind1, key_ind2] = min(p_less, p_greater)
        # print(np.mean(all_purity[col1]), np.mean(all_purity[col2]))
  # print(stimulus_pvalue)
  if logscale:
    norm = colors.LogNorm(5.668934240362814e-06, 1)# cmap="YlGnBu" (0.000001, 1) for 0.01 confidence level, (0.0025, 1) for 0.05
  else:
    norm = colors.Normalize(0.0, 1.0)
  sns_plot = sns.heatmap(stimulus_pvalue.astype(float), annot=True,cmap="RdBu",norm=norm)
  # sns_plot = sns.heatmap(region_connection.astype(float), vmin=0, cmap="YlGnBu")
  sns_plot.set_xticks(np.arange(len(cols))+0.5)
  sns_plot.set_xticklabels(cols, rotation=90)
  sns_plot.set_yticks(np.arange(len(cols))+0.5)
  sns_plot.set_yticklabels(cols, rotation=0)
  sns_plot.invert_yaxis()
  plt.title('{} p-value of {}'.format(method, name), size=18)
  plt.tight_layout()
  image_name = './plots/{}_pvalue_{}_{}_{}fold.jpg'.format(method, name, measure, n)
  # plt.show()
  plt.savefig(image_name)

def plot_intra_inter_data_2D(offset_dict, duration_dict, G_dict, sign, active_area_dict, measure, n):
  rows, cols = get_rowcol(offset_dict)
  num_row, num_col = len(rows), len(cols)
  fig = plt.figure(figsize=(5*num_col, 5))
  for col_ind, col in enumerate(cols):
    print(col)
    intra_offset, inter_offset, intra_duration, inter_duration = [], [], [], []
    df = pd.DataFrame()
    for row_ind, row in enumerate(rows):
      active_area = active_area_dict[row]
      offset_mat, duration_mat, G = offset_dict[row][col].copy(), duration_dict[row][col].copy(), G_dict[row][col].copy()
      nodes = sorted(list(G.nodes()))
      for i, j in zip(*np.where(~np.isnan(offset_mat))):
        if active_area[nodes[i]] == active_area[nodes[j]]:
          intra_offset.append(offset_mat[i, j])
          intra_duration.append(duration_mat[i, j])
        else:
          inter_offset.append(offset_mat[i, j])
          inter_duration.append(duration_mat[i, j])
    
    df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(intra_offset)[:,None], np.array(intra_duration)[:,None], np.array(['intra-region'] * len(intra_offset))[:,None]), 1), columns=[r'time lag $\tau$', r'duration $D$', 'type']), 
              pd.DataFrame(np.concatenate((np.array(inter_offset)[:,None], np.array(inter_duration)[:,None], np.array(['inter-region'] * len(inter_offset))[:,None]), 1), columns=[r'time lag $\tau$', r'duration $D$', 'type'])], ignore_index=True)
    df[r'duration $D$'] = pd.to_numeric(df[r'duration $D$'])
    df[r'time lag $\tau$'] = pd.to_numeric(df[r'time lag $\tau$'])
    ax = plt.subplot(1, num_col, col_ind+1)
    
    sns.kdeplot(data=df, x=r'duration $D$', y=r'time lag $\tau$', hue='type')
    plt.title(col, size=25)
    # if row_ind == len(rows)-1:
    #   plt.xlabel(r'time lag $\tau$')
  plt.legend()
  # plt.xticks(rotation=90)
  plt.tight_layout()
  # plt.show()
  figname = './plots/intra_inter_offset_duration_{}_{}_{}fold.jpg'
  plt.savefig(figname.format(sign, measure, n))

def plot_intra_inter_kde(data_dict, G_dict, sign, name, density, active_area_dict, measure, n):
  rows, cols = get_rowcol(data_dict)
  num_row, num_col = len(rows), len(cols)
  fig = plt.figure(figsize=(5*num_col, 5))
  for col_ind, col in enumerate(cols):
    print(col)
    intra_data, inter_data = [], []
    for row_ind, row in enumerate(rows):
      active_area = active_area_dict[row]
      mat, G = data_dict[row][col].copy(), G_dict[row][col].copy()
      nodes = sorted(list(G.nodes()))
      for i, j in zip(*np.where(~np.isnan(mat))):
        if active_area[nodes[i]] == active_area[nodes[j]]:
          intra_data.append(mat[i, j])
        else:
          inter_data.append(mat[i, j])
    df = pd.concat([pd.DataFrame(np.concatenate((np.array(intra_data)[:,None], np.array(['intra-region'] * len(intra_data))[:,None]), 1), columns=['data', 'type']), 
              pd.DataFrame(np.concatenate((np.array(inter_data)[:,None], np.array(['inter-region'] * len(inter_data))[:,None]), 1), columns=['data', 'type'])], ignore_index=True)
    df['data'] = pd.to_numeric(df['data'])
    ax = plt.subplot(1, num_col, col_ind+1)
    sns.kdeplot(data=df, x='data', hue='type', bw_adjust=1.5, cut=0, common_norm=not density)
    if density:
      plt.ylabel('Probability')
    else:
      plt.ylabel('Count')
    plt.title(col, size=25)
    # plt.xlim(0, 12)
    plt.xlabel(name)
  # plt.legend()
  plt.xticks(rotation=90)
  plt.tight_layout()
  # plt.show()
  figname = './plots/intra_inter_density_kde_{}_{}_{}_{}fold.jpg' if density else './plots/intra_inter_count_kde_{}_{}_{}_{}fold.jpg'
  plt.savefig(figname.format(name, sign, measure, n))

def plot_intra_inter_scatter(data_dict, G_dict, sign, name, active_area_dict, measure, n, color_pa):
  rows, cols = get_rowcol(data_dict)
  num_row, num_col = len(rows), len(cols)
  fig = plt.figure(figsize=(5, 5))
  df = pd.DataFrame(columns=['data', 'type', 'col'])
  for col_ind, col in enumerate(cols):
    print(col)
    intra_data, inter_data = [], []
    for row_ind, row in enumerate(rows):
      active_area = active_area_dict[row]
      mat, G = data_dict[row][col].copy(), G_dict[row][col].copy()
      nodes = sorted(list(G.nodes()))
      for i, j in zip(*np.where(~np.isnan(mat))):
        if active_area[nodes[i]] == active_area[nodes[j]]:
          intra_data.append(mat[i, j])
        else:
          inter_data.append(mat[i, j])
    df = pd.concat([df, pd.DataFrame([[np.mean(intra_data), np.mean(inter_data), col]],
                   columns=['intra-region', 'inter-region', 'col'])])
  df['intra-region'] = pd.to_numeric(df['intra-region'])
  df['inter-region'] = pd.to_numeric(df['inter-region'])
  ax = sns.scatterplot(data=df, x='intra-region', y='inter-region', hue='col', palette=color_pa, s=100)
  xliml, xlimu = ax.get_xlim()
  plt.plot(np.arange(xliml, xlimu, 0.1), np.arange(xliml, xlimu, 0.1), 'k--', alpha=0.4)
  plt.title(name, size=25)
  # plt.xlim(0, 12)
  # plt.legend()
  # plt.xticks(rotation=90)
  plt.tight_layout()
  # plt.show()
  figname = './plots/intra_inter_scatter_{}_{}_{}_{}fold.jpg'
  plt.savefig(figname.format(name, sign, measure, n))

def plot_intra_inter_ratio_box(data_dict, G_dict, name, active_area_dict, measure, n, color_pa):
  df = pd.DataFrame()
  rows, cols = get_rowcol(data_dict)
  for col_ind, col in enumerate(cols):
    print(col)
    intra_data, inter_data = [], []
    for row_ind, row in enumerate(rows):
      active_area = active_area_dict[row]
      mat, G = data_dict[row][col].copy(), G_dict[row][col].copy()
      nodes = sorted(list(G.nodes()))
      for i, j in zip(*np.where(~np.isnan(mat))):
        if active_area[nodes[i]] == active_area[nodes[j]]:
          intra_data.append(mat[i, j])
        else:
          inter_data.append(mat[i, j])
    # print(np.isnan(intra_data).sum(), np.isnan(inter_data).sum())
    # data = [d_i / d_j for d_i in np.random.choice(intra_data, 1000) for d_j in np.random.choice(inter_data, 1000)]
    data = [d_j / d_i for d_i in intra_data for d_j in inter_data]
    data = [x for x in data if not np.isnan(x)]
    data = [x for x in data if not np.isinf(x)]
    data = [x for x in data if not x==0]
    # data = remove_outliers(data, 3)
    df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(data)[:,None], np.array([col] * len(data))[:,None]), 1), columns=['ratio', 'stimulus'])], ignore_index=True)
    df['ratio'] = pd.to_numeric(df['ratio'])
  fig = plt.figure(figsize=(5, 5))
  # ax = sns.violinplot(x='stimulus', y='ratio', data=df, palette="muted", scale='count', cut=0)
  ax = sns.boxplot(x='stimulus', y='ratio', data=df, showfliers=False, palette=color_pa)
  plt.xticks(fontsize=10, rotation=90)
  plt.title('inter-region / intra-region {} ratio'.format(name))
  # plt.yscale('log')
  ax.set(xlabel=None)
  plt.tight_layout()
  # plt.savefig('violin_intra_divide_inter_{}_{}fold.jpg'.format(measure, n))
  plt.savefig('./plots/box_inter_intra_{}_ratio_{}_{}fold.jpg'.format(name, measure, n))

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

def down_sample(sequences, min_len, min_num):
  sequences = sequences[:, :min_len]
  i,j = np.nonzero(sequences)
  ix = np.random.choice(len(i), min_num, replace=False)
  sample_seq = np.zeros_like(sequences)
  sample_seq[i[ix], j[ix]] = sequences[i[ix], j[ix]]
  return sample_seq

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
density_dict = {}
rows, cols = get_rowcol(G_ccg_dict)
for col in cols:
  density_dict[col] = []
  for row in rows:
    density_dict[col].append(nx.density(G_ccg_dict[row][col]))
plot_kstest(density_dict, 'density', measure, n)
# %%
violin_data(density_dict, 'density', measure, n)
# %%
# box plot
box_data(density_dict, 'density', measure, n)
# %%
################ violin/box plot of intra/inter region links
df = pd.DataFrame()
rows, cols = get_rowcol(G_ccg_dict)
# metric = np.zeros((len(rows), len(cols), 3))
region_connection = np.zeros((len(rows), len(cols), len(visual_regions), len(visual_regions)))
for col_ind, col in enumerate(cols):
  print(col)
  intra_data, inter_data = [], []
  for row_ind, row in enumerate(rows):
    G = G_ccg_dict[row][col] if col in G_ccg_dict[row] else nx.Graph()
    if G.number_of_nodes() >= 2 and G.number_of_edges() > 0:
      nodes = list(G.nodes())
      node_area = {key: area_dict[row][key] for key in nodes}
      areas = list(node_area.values())
      area_size = [areas.count(r) for r in visual_regions]
      A = nx.to_numpy_array(G)
      A[A.nonzero()] = 1
      for region_ind_i, region_i in enumerate(visual_regions):
        for region_ind_j, region_j in enumerate(visual_regions):
          region_indices_i = np.array([k for k, v in area_dict[row].items() if v==region_i])
          region_indices_j = np.array([k for k, v in area_dict[row].items() if v==region_j])
          region_indices_i = np.array([nodes.index(i) for i in list(set(region_indices_i) & set(nodes))]) # some nodes not in cc are removed 
          region_indices_j = np.array([nodes.index(i) for i in list(set(region_indices_j) & set(nodes))])
          if len(region_indices_i) and len(region_indices_j):
            region_connection[row_ind, col_ind, region_ind_i, region_ind_j] = np.sum(A[region_indices_i[:, None], region_indices_j])
            assert np.sum(A[region_indices_i[:, None], region_indices_j]) == len(A[region_indices_i[:, None], region_indices_j].nonzero()[0])
      diag_indx = np.eye(len(visual_regions),dtype=bool)
      # metric[row_ind, col_ind, 0] =  np.sum(region_connection[row_ind, col_ind][diag_indx])
      # metric[row_ind, col_ind, 1] =  np.sum(region_connection[row_ind, col_ind][~diag_indx])
      intra_data.append(np.sum(region_connection[row_ind, col_ind][diag_indx]))
      inter_data.append(np.sum(region_connection[row_ind, col_ind][~diag_indx]))
  df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(intra_data)[:,None], np.array(['intra-region'] * len(intra_data))[:,None], np.array([col] * len(intra_data))[:,None]), 1), columns=['number of connections', 'type', 'stimulus']), 
              pd.DataFrame(np.concatenate((np.array(inter_data)[:,None], np.array(['inter-region'] * len(inter_data))[:,None], np.array([col] * len(inter_data))[:,None]), 1), columns=['number of connections', 'type', 'stimulus'])], ignore_index=True)
  df['number of connections'] = pd.to_numeric(df['number of connections'])
fig = plt.figure(figsize=(5, 5))
# ax = sns.violinplot(x='stimulus', y='number of connections', hue="type", data=df, palette="muted", split=False)
ax = sns.boxplot(x='stimulus', y='number of connections', hue="type", data=df, palette="muted")
plt.xticks(fontsize=10, rotation=90)
ax.set(xlabel=None)
plt.tight_layout()
# plt.savefig('./plots/violin_intra_inter_{}_{}fold.jpg'.format(measure, n))
plt.savefig('./plots/box_intra_inter_{}_{}fold.jpg'.format(measure, n))
# %%
def remove_outliers(data, m=2.):
  data = np.array(data)
  d = np.abs(data - np.median(data))
  mdev = np.median(d)
  s = d / (mdev if mdev else 1.)
  return data[s < m]

################ violin/box plot of intra-inter region links
df = pd.DataFrame()
rows, cols = get_rowcol(G_ccg_dict)
# metric = np.zeros((len(rows), len(cols), 3))
region_connection = np.zeros((len(rows), len(cols), len(visual_regions), len(visual_regions)))
for col_ind, col in enumerate(cols):
  print(col)
  data = []
  for row_ind, row in enumerate(rows):
    G = G_ccg_dict[row][col] if col in G_ccg_dict[row] else nx.Graph()
    if G.number_of_nodes() >= 2 and G.number_of_edges() > 0:
      nodes = list(G.nodes())
      node_area = {key: area_dict[row][key] for key in nodes}
      areas = list(node_area.values())
      area_size = [areas.count(r) for r in visual_regions]
      A = nx.to_numpy_array(G)
      A[A.nonzero()] = 1
      for region_ind_i, region_i in enumerate(visual_regions):
        for region_ind_j, region_j in enumerate(visual_regions):
          region_indices_i = np.array([k for k, v in area_dict[row].items() if v==region_i])
          region_indices_j = np.array([k for k, v in area_dict[row].items() if v==region_j])
          region_indices_i = np.array([nodes.index(i) for i in list(set(region_indices_i) & set(nodes))]) # some nodes not in cc are removed 
          region_indices_j = np.array([nodes.index(i) for i in list(set(region_indices_j) & set(nodes))])
          if len(region_indices_i) and len(region_indices_j):
            region_connection[row_ind, col_ind, region_ind_i, region_ind_j] = np.sum(A[region_indices_i[:, None], region_indices_j])
            assert np.sum(A[region_indices_i[:, None], region_indices_j]) == len(A[region_indices_i[:, None], region_indices_j].nonzero()[0])
      diag_indx = np.eye(len(visual_regions),dtype=bool)
      # metric[row_ind, col_ind, 0] =  np.sum(region_connection[row_ind, col_ind][diag_indx])
      # metric[row_ind, col_ind, 1] =  np.sum(region_connection[row_ind, col_ind][~diag_indx])
      data.append(np.sum(region_connection[row_ind, col_ind][diag_indx]) / np.sum(region_connection[row_ind, col_ind][~diag_indx]))
  # data = remove_outliers(data, 3)
  df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(data)[:,None], np.array([col] * len(data))[:,None]), 1), columns=['ratio', 'stimulus'])], ignore_index=True)
  df['ratio'] = pd.to_numeric(df['ratio'])
fig = plt.figure(figsize=(5, 5))
# ax = sns.violinplot(x='stimulus', y='ratio', data=df, palette="muted", scale='count', cut=0)
ax = sns.boxplot(x='stimulus', y='ratio', data=df, palette="muted")
plt.xticks(fontsize=10, rotation=90)
plt.title('intra-region / inter-region links')
plt.yscale('log')
ax.set(xlabel=None)
plt.tight_layout()
# plt.savefig('violin_intra_divide_inter_{}_{}fold.jpg'.format(measure, n))
plt.savefig('./plots/box_intra_divide_inter_{}_{}fold.jpg'.format(measure, n))
#%%
################ box plot of intra region links ratio
df = pd.DataFrame()
rows, cols = get_rowcol(G_ccg_dict)
# metric = np.zeros((len(rows), len(cols), 3))
region_connection = np.zeros((len(rows), len(cols), len(visual_regions), len(visual_regions)))
for col_ind, col in enumerate(cols):
  print(col)
  data = []
  for row_ind, row in enumerate(rows):
    G = G_ccg_dict[row][col] if col in G_ccg_dict[row] else nx.Graph()
    if G.number_of_nodes() >= 2 and G.number_of_edges() > 0:
      nodes = list(G.nodes())
      node_area = {key: area_dict[row][key] for key in nodes}
      areas = list(node_area.values())
      area_size = [areas.count(r) for r in visual_regions]
      A = nx.to_numpy_array(G)
      A[A.nonzero()] = 1
      for region_ind_i, region_i in enumerate(visual_regions):
        for region_ind_j, region_j in enumerate(visual_regions):
          region_indices_i = np.array([k for k, v in area_dict[row].items() if v==region_i])
          region_indices_j = np.array([k for k, v in area_dict[row].items() if v==region_j])
          region_indices_i = np.array([nodes.index(i) for i in list(set(region_indices_i) & set(nodes))]) # some nodes not in cc are removed 
          region_indices_j = np.array([nodes.index(i) for i in list(set(region_indices_j) & set(nodes))])
          if len(region_indices_i) and len(region_indices_j):
            region_connection[row_ind, col_ind, region_ind_i, region_ind_j] = np.sum(A[region_indices_i[:, None], region_indices_j])
            assert np.sum(A[region_indices_i[:, None], region_indices_j]) == len(A[region_indices_i[:, None], region_indices_j].nonzero()[0])
      diag_indx = np.eye(len(visual_regions),dtype=bool)
      # metric[row_ind, col_ind, 0] =  np.sum(region_connection[row_ind, col_ind][diag_indx])
      # metric[row_ind, col_ind, 1] =  np.sum(region_connection[row_ind, col_ind][~diag_indx])
      data.append(np.sum(region_connection[row_ind, col_ind][diag_indx]) / np.sum(region_connection[row_ind, col_ind]))
  # data = remove_outliers(data, 3)
  df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(data)[:,None], np.array([col] * len(data))[:,None]), 1), columns=['ratio', 'stimulus'])], ignore_index=True)
  df['ratio'] = pd.to_numeric(df['ratio'])
fig = plt.figure(figsize=(5, 5))
# ax = sns.violinplot(x='stimulus', y='ratio', data=df, palette="muted", scale='count', cut=0)
ax = sns.boxplot(x='stimulus', y='ratio', data=df, palette="muted")
plt.xticks(fontsize=10, rotation=90)
plt.title('intra-region link ratio')
# plt.yscale('log')
ax.set(xlabel=None)
plt.tight_layout()
# plt.savefig('violin_intra_divide_inter_{}_{}fold.jpg'.format(measure, n))
plt.savefig('./plots/box_intra_ratio_{}_{}fold.jpg'.format(measure, n))
# %%
################## p-value of intra/inter link ratio across stimulus
rows, cols = get_rowcol(G_ccg_dict)
intra_inter_ratio_dict = {}
region_connection = np.zeros((len(rows), len(cols), len(visual_regions), len(visual_regions)))
for col_ind, col in enumerate(cols):
  print(col)
  data = []
  intra_inter_ratio_dict[col] = []
  for row_ind, row in enumerate(rows):
    G = G_ccg_dict[row][col] if col in G_ccg_dict[row] else nx.Graph()
    if G.number_of_nodes() >= 2 and G.number_of_edges() > 0:
      nodes = list(G.nodes())
      node_area = {key: area_dict[row][key] for key in nodes}
      areas = list(node_area.values())
      area_size = [areas.count(r) for r in visual_regions]
      A = nx.to_numpy_array(G)
      A[A.nonzero()] = 1
      for region_ind_i, region_i in enumerate(visual_regions):
        for region_ind_j, region_j in enumerate(visual_regions):
          region_indices_i = np.array([k for k, v in area_dict[row].items() if v==region_i])
          region_indices_j = np.array([k for k, v in area_dict[row].items() if v==region_j])
          region_indices_i = np.array([nodes.index(i) for i in list(set(region_indices_i) & set(nodes))]) # some nodes not in cc are removed 
          region_indices_j = np.array([nodes.index(i) for i in list(set(region_indices_j) & set(nodes))])
          if len(region_indices_i) and len(region_indices_j):
            region_connection[row_ind, col_ind, region_ind_i, region_ind_j] = np.sum(A[region_indices_i[:, None], region_indices_j])
            assert np.sum(A[region_indices_i[:, None], region_indices_j]) == len(A[region_indices_i[:, None], region_indices_j].nonzero()[0])
      diag_indx = np.eye(len(visual_regions),dtype=bool)
      # metric[row_ind, col_ind, 0] =  np.sum(region_connection[row_ind, col_ind][diag_indx])
      # metric[row_ind, col_ind, 1] =  np.sum(region_connection[row_ind, col_ind][~diag_indx])
      data.append(np.sum(region_connection[row_ind, col_ind][diag_indx]) / np.sum(region_connection[row_ind, col_ind][~diag_indx]))
  intra_inter_ratio_dict[col] = data

plot_kstest(intra_inter_ratio_dict, 'intra_inter_ratio', measure, n)
# %%
inds_path = './data/ecephys_cache_dir/sessions/active_inds/'
directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
FR_dict = {}
for stimulus_name in stimulus_names:
  print(stimulus_name)
  FR_dict[stimulus_name] = []
  for session_id in session_ids:
    print(session_id)
    active_neuron_inds = np.load(os.path.join(inds_path, str(session_id)+'.npy'))
    sequences = load_npz_3d(os.path.join(directory, str(session_id) + '_' + stimulus_name + '.npz'))
    sequences = sequences[active_neuron_inds]
    firing_rates = np.count_nonzero(sequences, axis=-1) / (sequences.shape[-1]/1000) # Hz instead of kHz
    FR_dict[stimulus_name] += firing_rates.mean(-1).tolist()
# %%
box_data(FR_dict, 'firing_rate_(Hz)', measure, n)
# %%
# intra/inter region offset/duration 2D
plot_intra_inter_data_2D(offset_dict, duration_dict, G_ccg_dict, 'total', active_area_dict, measure, n)
# %%
# intra/inter region offset/duration kde
plot_intra_inter_kde(offset_dict, G_ccg_dict, 'total', 'offset', True, active_area_dict, measure, n)
plot_intra_inter_kde(offset_dict, G_ccg_dict, 'total', 'offset', False, active_area_dict, measure, n)
plot_intra_inter_kde(duration_dict, G_ccg_dict, 'total', 'duration', True, active_area_dict, measure, n)
plot_intra_inter_kde(duration_dict, G_ccg_dict, 'total', 'duration', False, active_area_dict, measure, n)
# %%
# scatter plot of intra/inter duration over stimulus
color_pa = ['tab:blue', 'tab:orange', 'limegreen', 'darkgreen', 'maroon', 'indianred', 'mistyrose']
plot_intra_inter_scatter(offset_dict, G_ccg_dict, 'total', 'offset', active_area_dict, measure, n, color_pa)
plot_intra_inter_scatter(duration_dict, G_ccg_dict, 'total', 'duration', active_area_dict, measure, n, color_pa)
# %%
################ box plot of inter intra region offset ratio
color_pa = ['tab:blue', 'tab:orange', 'limegreen', 'darkgreen', 'maroon', 'indianred', 'mistyrose']
plot_intra_inter_ratio_box(offset_dict, G_ccg_dict, 'offset', active_area_dict, measure, n, color_pa)
plot_intra_inter_ratio_box(duration_dict, G_ccg_dict, 'duration', active_area_dict, measure, n, color_pa)
# %%
################ p value of pairwise inter intra region offset ratio across stimulus
data_dict = offset_dict
# data_dict = duration_dict
ratio_dict = {}
rows, cols = get_rowcol(G_ccg_dict)
for col in cols:
  print(col)
  intra_data, inter_data = [], []
  for row_ind, row in enumerate(rows):
    active_area = active_area_dict[row]
    mat, G = data_dict[row][col].copy(), G_ccg_dict[row][col].copy()
    nodes = sorted(list(G.nodes()))
    for i, j in zip(*np.where(~np.isnan(mat))):
      if active_area[nodes[i]] == active_area[nodes[j]]:
        intra_data.append(mat[i, j])
      else:
        inter_data.append(mat[i, j])
  # print(np.isnan(intra_data).sum(), np.isnan(inter_data).sum())
  # data = [d_i / d_j for d_i in np.random.choice(intra_data, 1000) for d_j in np.random.choice(inter_data, 1000)]
  data = [d_j / d_i for d_i in intra_data for d_j in inter_data]
  data = [x for x in data if not np.isnan(x)]
  data = [x for x in data if not np.isinf(x)]
  data = [x for x in data if not x==0]
  ratio_dict[col] = data

# plot_p_value(ratio_dict, 'inter_intra_offset_ratio', measure, n, 'ks_test', False)
# plot_p_value(ratio_dict, 'inter_intra_offset_ratio', measure, n, 'mwu_test', True)
# %%
cols = list(ratio_dict.keys())
mean_data = [{k:np.mean(ratio_dict[k]) for k in ratio_dict}]
ratio_df = pd.DataFrame(mean_data)
# %%
################ average inter intra region offset ratio across stimulus
data_dict = offset_dict
# data_dict = duration_dict
mean_ratio_dict = {}
rows, cols = get_rowcol(G_ccg_dict)
for col in cols:
  print(col)
  intra_data, inter_data = [], []
  for row_ind, row in enumerate(rows):
    active_area = active_area_dict[row]
    mat, G = data_dict[row][col].copy(), G_ccg_dict[row][col].copy()
    nodes = sorted(list(G.nodes()))
    for i, j in zip(*np.where(~np.isnan(mat))):
      if active_area[nodes[i]] == active_area[nodes[j]]:
        intra_data.append(mat[i, j])
      else:
        inter_data.append(mat[i, j])
  mean_ratio_dict[col] = np.mean(inter_data) / np.mean(intra_data)

mean_ratio_df = pd.DataFrame([mean_ratio_dict])
# %%
def plot_example_graph(G_dict, area_dict, session_ids, stimulus_names, mouse_id, stimulus_id, baseline=False):
  fig = plt.figure(figsize=(10, 8))
  G = G_dict[str(session_ids[mouse_id])][stimulus_names[stimulus_id]][0]
  nx.set_node_attributes(G, area_dict[str(session_ids[mouse_id])], "area")
  if nx.is_directed(G):
      Gcc = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
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

def plot_example_graphs_offset(G_dict, max_reso, offset_dict, sign, area_dict, active_area_dict, session_ids, stimulus_names, mouse_id, stimulus_id, measure, n, cc=False):
  com = CommunityLayout()
  rows, cols = get_rowcol(G_dict)
  G_sample = G_dict[rows[0]][cols[0]]
  dire = True if nx.is_directed(G_sample) else False
  fig = plt.figure(figsize=(8, 7))
  row, col = str(session_ids[mouse_id]), stimulus_names[stimulus_id]
  print(row, col)
  offset_mat = offset_dict[row][col]
  G = G_dict[row][col].copy() if col in G_dict[row] else nx.DiGraph()
  nx.set_node_attributes(G, area_dict[row], "area")
  if G.number_of_nodes() > 2 and G.number_of_edges() > 0:
    if cc:
      if nx.is_directed(G):
        Gcc = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
        G = G.subgraph(Gcc[0])
      else:
        Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
        G = G.subgraph(Gcc[0])
    node_idx = sorted(active_area_dict[row].keys())
    reverse_mapping = {node_idx[i]:i for i in range(len(node_idx))}
    G = nx.relabel_nodes(G, reverse_mapping)
    edges = nx.edges(G)
    # weights = [offset_mat[reverse_mapping[edge[0]], reverse_mapping[edge[1]]] for edge in edges]
    try:
      if nx.is_directed(G):
        G = G.to_undirected()
      offset = {edge:offset_mat[edge[0], edge[1]] for edge in edges}
      nx.set_edge_attributes(G, offset, 'offset')
      comms = nx_comm.louvain_communities(G, weight='offset', resolution=max_reso)
      comms_tuple = [[[i for i in comm], len(comm)] for comm in comms]
      large_comms = [e[0] for e in sorted(comms_tuple, key=lambda x:x[1], reverse=True)][:6]
      nodes2plot = [j for i in comms if len(i) > 2 for j in i]
      comms2plot = [i for i in comms if len(i) > 2]
      pos = com.get_community_layout(G.subgraph(nodes2plot), comm2partition(comms2plot))
      metric = nx_comm.modularity(G.subgraph(nodes2plot), comms2plot, weight='offset', resolution=max_reso)
      # partition = community.best_partition(G, weight='weight')
      # pos = com.get_community_layout(G, partition)
      # metric = community.modularity(partition, G, weight='weight')
      print('Modularity: {}'.format(metric))
    except:
      print('Community detection unsuccessful!')
      pos = nx.spring_layout(G)
    edges = nx.edges(G.subgraph(nodes2plot))
    degrees = dict(G.degree(nodes2plot))
    # use offset as edge weight (color)
    weights = [offset[edge] for edge in offset if (edge[0] in nodes2plot) and (edge[1] in nodes2plot)]
    # weights = [offset_mat[edge[0], edge[1]] for edge in edges]
    norm = mpl.colors.Normalize(vmin=-1, vmax=11)
    m= cm.ScalarMappable(norm=norm, cmap=cm.Greens)
    edge_colors = [m.to_rgba(w) for w in weights]
    areas = [G.nodes[n]['area'] for n in nodes2plot]
    areas_uniq = list(set(areas))
    colors = [customPalette[areas_uniq.index(area)] for area in areas]
    # pos = nx.spring_layout(G, k=0.8, iterations=50) # make nodes as seperate as possible
    nx.draw_networkx_edges(G, pos, arrows=dire, edgelist=edges, edge_color=edge_colors, width=3.0, alpha=0.9) # , edge_cmap=plt.cm.Greens
    # nx.draw_networkx_nodes(G, pos, nodelist=degrees.keys(), node_size=[np.log(v + 2) * 20 for v in degrees.values()], 
    nx.draw_networkx_nodes(G, pos, nodelist=degrees.keys(), node_size=[5 * v for v in degrees.values()], 
    node_color=colors, alpha=0.4)

  areas = [G.nodes[n]['area'] for n in G.nodes()]
  areas_uniq = list(set(areas))
  for index, a in enumerate(areas_uniq):
    plt.scatter([],[], c=customPalette[index], label=a, s=30)
  legend = plt.legend(loc='upper left', fontsize=20)
  for handle in legend.legendHandles:
    handle.set_sizes([60.0])
  plt.tight_layout()
  image_name = './plots/example_graphs_region_color_cc_{}_{}_{}fold.jpg'.format(sign, measure, n) if cc else './plots/example_graphs_region_color_{}_{}_{}fold.jpg'.format(sign, measure, n)
  plt.savefig(image_name)
  # plt.savefig(image_name.replace('.jpg', '.pdf'), transparent=True)
  # plt.show()
  return weights, comms

mouse_id, stimulus_id = 4, 6
weights, comms = plot_example_graphs_offset(G_ccg_dict, max_reso_gnm[mouse_id][2], offset_dict, 'total_offset_gnm', area_dict, active_area_dict, session_ids, stimulus_names, mouse_id, stimulus_id, measure, n, True)
#%%
comms2plot = [i for i in comms if len(i) > 5]
print([len(i) for i in comms], len(comms2plot))
#%%
################# get optimal resolution that maximizes delta Q
rows, cols = get_rowcol(G_ccg_dict)
with open('metrics.pkl', 'rb') as f:
  metrics = pickle.load(f)
resolution_list = np.arange(0, 2.1, 0.1)
max_reso_gnm, max_reso_swap = get_max_resolution(rows, cols, resolution_list, metrics)
#%%

# %%
