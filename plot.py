# %%
from turtle import ycor
from library import *
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

session2keep = ['719161530','750749662','754312389','755434585','756029989','791319847','797828357']
stimulus_by_type = [['spontaneous'], ['flash_dark', 'flash_light'], ['drifting_gratings', 'static_gratings'], ['natural_scenes', 'natural_movie_one', 'natural_movie_three']]
stimulus_types = ['resting state', 'flashes', 'gratings', 'natural stimuli']
stimulus_type_color = ['tab:blue', 'darkorange', 'darkgreen', 'maroon']
paper_label = ['resting\nstate', 'dark\nflash', 'light\nflash',
          'drifting\ngrating', 'static\ngrating', 'natural\nscenes', 'natural\nmovie 1', 'natural\nmovie 3']

def stimulus2stype(stimulus):
  t_ind = [i for i in range(len(stimulus_by_type)) if stimulus in stimulus_by_type[i]][0]
  return t_ind, stimulus_types[t_ind]

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
          p_less = stats.mannwhitneyu(data_dict[key1], data_dict[key2], alternative='less', method="asymptotic")[1]
          p_greater = stats.mannwhitneyu(data_dict[key1], data_dict[key2], alternative='greater', method="asymptotic")[1]
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

def plot_corrected_p_value(data_dict, name, measure, n, method='ks_test', logscale='True'):
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
          p_less = stats.mannwhitneyu(data_dict[key1], data_dict[key2], alternative='less', method="asymptotic")[1]
          p_greater = stats.mannwhitneyu(data_dict[key1], data_dict[key2], alternative='greater', method="asymptotic")[1]
        stimulus_pvalue[key_ind1, key_ind2] = min(p_less, p_greater)
        # print(np.mean(all_purity[col1]), np.mean(all_purity[col2]))
  # print(stimulus_pvalue)
  is_rejected, corrected_pvalue = np.zeros_like(stimulus_pvalue), np.zeros_like(stimulus_pvalue)
  off_diag_idx = ~np.eye(stimulus_pvalue.shape[0],dtype=bool)
  # is_rejected[off_diag_idx], corrected_pvalue[off_diag_idx] = fdrcorrection(stimulus_pvalue[off_diag_idx])[:2]
  is_rejected[off_diag_idx], corrected_pvalue[off_diag_idx] = multipletests(stimulus_pvalue[off_diag_idx], method='fdr_bh')[:2]
  if logscale:
    norm = colors.LogNorm(0.0025, 1)# cmap="YlGnBu" (0.000001, 1) for 0.01 confidence level, (0.0025, 1) for 0.05 (5.668934240362814e-06, 1) for 0.05/21
  else:
    norm = colors.Normalize(0.0, 1.0)
  sns_plot = sns.heatmap(corrected_pvalue.astype(float), annot=True,cmap="RdBu",norm=norm)
  # sns_plot = sns.heatmap(region_connection.astype(float), vmin=0, cmap="YlGnBu")
  sns_plot.set_xticks(np.arange(len(cols))+0.5)
  sns_plot.set_xticklabels(cols, rotation=90)
  sns_plot.set_yticks(np.arange(len(cols))+0.5)
  sns_plot.set_yticklabels(cols, rotation=0)
  sns_plot.invert_yaxis()
  plt.title('BH corrected {} p-value of {}'.format(method, name), size=18)
  plt.tight_layout()
  image_name = './plots/{}_corrected_pvalue_{}_{}_{}fold.jpg'.format(method, name, measure, n)
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

def plot_intra_inter_kde_G(G_dict, name, density, active_area_dict, measure, n):
  rows, cols = get_rowcol(G_dict)
  num_row, num_col = len(rows), len(cols)
  fig = plt.figure(figsize=(5*num_col, 5))
  for col_ind, col in enumerate(cols):
    print(col)
    intra_data, inter_data = [], []
    for row_ind, row in enumerate(rows):
      active_area = active_area_dict[row]
      G = G_dict[row][col].copy()
      for edge in G.edges():
        if active_area[edge[0]] == active_area[edge[1]]:
          intra_data.append(G[edge[0]][edge[1]][name])
        else:
          inter_data.append(G[edge[0]][edge[1]][name])
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
  figname = './plots/intra_inter_density_kde_{}_{}_{}fold.jpg' if density else './plots/intra_inter_count_kde_{}_{}_{}fold.jpg'
  plt.savefig(figname.format(name, measure, n))

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

def plot_existed_edge_offset_stimulus(offset_dict, G_dict, active_area_dict, measure, n):
  rows, cols = get_rowcol(offset_dict)
  num_row, num_col = len(rows), len(cols)
  intra_offset, inter_offset = [[] for _ in range(num_col)], [[] for _ in range(num_col)]
  for row_ind, row in enumerate(rows):
    active_area = active_area_dict[row]
    print(row)
    df = pd.DataFrame()
    existed_edges = []
    for col_ind, col in enumerate(cols):
      col_edges = np.transpose(np.where(~np.isnan(offset_dict[row][col])))
      col_edges = [tuple(edge) for edge in col_edges]
      existed_edges.append(col_edges)
    existed_edges = set.intersection(*map(set, existed_edges))
    nodes = sorted(list(G_dict[row][col].nodes()))
    for i, j in existed_edges:
      if active_area[nodes[i]] == active_area[nodes[j]]:
        for col_ind, col in enumerate(cols):
          intra_offset[col_ind].append(offset_dict[row][col][i, j])
      else:
        for col_ind, col in enumerate(cols):
          inter_offset[col_ind].append(offset_dict[row][col][i, j])
  for col_ind, col in enumerate(cols):
    df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(intra_offset[col_ind])[:,None], np.array(['intra-region'] * len(intra_offset[col_ind]))[:,None], np.array([col] * len(intra_offset[col_ind]))[:,None]), 1), columns=[r'time lag $\tau$', 'type', 'stimulus']), 
              pd.DataFrame(np.concatenate((np.array(inter_offset[col_ind])[:,None], np.array(['inter-region'] * len(inter_offset[col_ind]))[:,None], np.array([col] * len(inter_offset[col_ind]))[:,None]), 1), columns=[r'time lag $\tau$', 'type', 'stimulus'])], ignore_index=True)
    df[r'time lag $\tau$'] = pd.to_numeric(df[r'time lag $\tau$'])
  plt.figure(figsize=(12, 8))
  ax = sns.violinplot(x='stimulus', y=r'time lag $\tau$', hue="type", data=df, palette="Set3", split=False, cut=1)
  # ax = sns.boxplot(x='stimulus', y=r'time lag $\tau$', hue="type", data=df, palette="Set3", showfliers=False)
  # ax = sns.swarmplot(x='stimulus', y=r'time lag $\tau$', data=df, color=".25")
  plt.xticks(fontsize=10, rotation=90)
  ax.set(xlabel=None)
  # plt.xticks(rotation=90)
  plt.tight_layout()
  # plt.show()
  figname = './plots/existed_edges_offset_stimulus_{}_{}fold.jpg'
  plt.savefig(figname.format(measure, n))
  return intra_offset, inter_offset
  
def plot_pairwise_existed_edge_offset_stimulus(offset_dict, G_dict, active_area_dict, measure, n):
  rows, cols = get_rowcol(offset_dict)
  intra_offset, inter_offset = defaultdict(list), defaultdict(list)
  for row_ind, row in enumerate(rows):
    active_area = active_area_dict[row]
    print(row)
    nodes = sorted(list(G_dict[row][cols[0]].nodes()))
    for col1, col2 in itertools.combinations(cols, 2):
      existed_edges = []
      col_edges1 = np.transpose(np.where(~np.isnan(offset_dict[row][col1])))
      col_edges1 = [tuple(edge) for edge in col_edges1]
      col_edges2 = np.transpose(np.where(~np.isnan(offset_dict[row][col2])))
      col_edges2 = [tuple(edge) for edge in col_edges2]
      existed_edges.append(col_edges1)
      existed_edges.append(col_edges2)
      existed_edges = set.intersection(*map(set, existed_edges))
      for i, j in existed_edges:
        if active_area[nodes[i]] == active_area[nodes[j]]:
          intra_offset[(col1, col2)].append(offset_dict[row][col1][i, j]-offset_dict[row][col2][i, j])
        else:
          inter_offset[(col1, col2)].append(offset_dict[row][col1][i, j]-offset_dict[row][col2][i, j])
  intra_df, inter_df = pd.DataFrame(data=np.zeros((len(cols), len(cols))), index=cols, columns=cols), pd.DataFrame(data=np.zeros((len(cols), len(cols))), index=cols, columns=cols)
  for col1, col2 in itertools.combinations(cols, 2):
    intra_df.loc[col1, col2] = np.mean(intra_offset[(col1, col2)])
    inter_df.loc[col1, col2] = np.mean(inter_offset[(col1, col2)])
  mask = np.ones((len(cols), len(cols)))
  mask[np.triu_indices_from(mask, 1)] = False
  rdgn = sns.diverging_palette(h_neg=130, h_pos=10, s=99, l=55, sep=3, as_cmap=True)
  plt.figure(figsize=(10, 8))
  sns_plot = sns.heatmap(intra_df, mask=mask, cmap=rdgn, center=0.00, cbar_kws={"orientation": "vertical","label": r'$\tau_{column}-$\tau_{row}$'})
  sns_plot.invert_yaxis()
  # plt.xticks(fontsize=10, rotation=90)
  # ax.set(xlabel=None)
  plt.title('intra-region')
  plt.tight_layout()
  # plt.show()
  figname = './plots/pairwise_existed_edges_intra_offset_{}_{}fold.jpg'
  plt.savefig(figname.format(measure, n))
  plt.figure(figsize=(10, 8))
  sns_plot = sns.heatmap(inter_df, mask=mask, cmap=rdgn, center=0.00, cbar_kws={"orientation": "vertical","label": r'$\tau_{column}-$\tau_{row}$'})
  sns_plot.invert_yaxis()
  # plt.xticks(fontsize=10, rotation=90)
  # ax.set(xlabel=None)
  plt.title('inter-region')
  plt.tight_layout()
  # plt.show()
  figname = './plots/pairwise_existed_edges_inter_offset_{}_{}fold.jpg'
  plt.savefig(figname.format(measure, n))
  return intra_df, inter_df

def plot_example_graphs_offset(G_dict, max_reso, offset_dict, sign, area_dict, active_area_dict, session_ids, stimulus_names, mouse_id, stimulus_id, measure, n, cc=False):
  com = CommunityLayout(30, 3)
  rows, cols = get_rowcol(G_dict)
  G_sample = G_dict[rows[0]][cols[0]]
  dire = True if nx.is_directed(G_sample) else False
  fig = plt.figure(figsize=(8, 7))
  row, col = str(session_ids[mouse_id]), stimulus_names[stimulus_id]
  print(row, col)
  offset_mat = offset_dict[row][col]
  G = G_dict[row][col].copy() if col in G_dict[row] else nx.DiGraph()
  nx.set_node_attributes(G, area_dict[row], "area")
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
    ordered_comms = [e[0] for e in sorted(comms_tuple, key=lambda x:x[1], reverse=True)]
    ordered_comm_regions = [[G.nodes[n]['area'] for n in comm] for comm in ordered_comms]
    ordered_most_common_regions = [most_common(comm_r) for comm_r in ordered_comm_regions]
    inds = [ordered_most_common_regions.index(r) for r in np.unique(ordered_most_common_regions)]
    comms2plot = [ordered_comms[i] for i in inds]
    nodes2plot = [j for i in comms2plot for j in i]
    # nodes2plot = [j for i in comms if len(i) > 2 for j in i]
    # comms2plot = [i for i in comms if len(i) > 2]
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
  nx.draw_networkx_nodes(G, pos, nodelist=degrees.keys(), node_size=[15 * v for v in degrees.values()], 
  node_color=colors, alpha=0.6)

  areas = [G.nodes[n]['area'] for n in G.nodes()]
  areas_uniq = list(set(areas))
  for index, a in enumerate(areas_uniq):
    plt.scatter([],[], c=customPalette[index], label=a, s=30, alpha=0.6)
  legend = plt.legend(fontsize=15) # loc='lower right', 
  for handle in legend.legendHandles:
    handle.set_sizes([60.0])
  plt.tight_layout()
  image_name = './plots/example_graphs_region_color_cc_{}_{}_{}_{}_{}fold.jpg' if cc else './plots/example_graphs_region_color_{}_{}_{}_{}_{}fold.jpg'
  plt.savefig(image_name.format(mouse_id, stimulus_id, sign, measure, n))
  # plt.savefig(image_name.replace('.jpg', '.pdf'), transparent=True)
  # plt.show()
  return weights, comms

def plot_example_graphs_each_region(G_dict, hamiltonian, comms, area_dict, active_area_dict, row, col, measure, n, cc=False):
  com = CommunityLayout(30, 3)
  G_sample = G_dict[row][col]
  dire = True if nx.is_directed(G_sample) else False
  fig = plt.figure(figsize=(8, 7))
  print(row, col)
  G = G_dict[row][col].copy() if col in G_dict[row] else nx.DiGraph()
  nx.set_node_attributes(G, area_dict[row], "area")
  if cc:
    G = get_lcc(G)
  node_idx = sorted(active_area_dict[row].keys())
  reverse_mapping = {node_idx[i]:i for i in range(len(node_idx))}
  G = nx.relabel_nodes(G, reverse_mapping)
  comms = [[reverse_mapping[n] for n in comm] for comm in comms]
  comms_tuple = [[[i for i in comm], len(comm)] for comm in comms]
  ordered_comms = [e[0] for e in sorted(comms_tuple, key=lambda x:x[1], reverse=True)]
  ordered_comm_regions = [[G.nodes[n]['area'] for n in comm] for comm in ordered_comms]
  ordered_most_common_regions = [most_common(comm_r) for comm_r in ordered_comm_regions]
  inds = [ordered_most_common_regions.index(r) for r in np.unique(ordered_most_common_regions)]
  comms2plot = [ordered_comms[i] for i in inds]
  nodes2plot = [j for i in comms2plot for j in i]
  # nodes2plot = [j for i in comms if len(i) > 2 for j in i]
  # comms2plot = [i for i in comms if len(i) > 2]
  pos = com.get_community_layout(G.subgraph(nodes2plot), comm2partition(comms2plot))
  # partition = community.best_partition(G, weight='weight')
  # pos = com.get_community_layout(G, partition)
  # metric = community.modularity(partition, G, weight='weight')
  print('Hamiltonian: {}'.format(hamiltonian))
  edges = nx.edges(G.subgraph(nodes2plot))
  degrees = dict(G.degree(nodes2plot))
  # use offset as edge weight (color)
  weights = [w for i,j,w in nx.edges(G.subgraph(nodes2plot)).data('weight')]
  # weights = [offset_mat[edge[0], edge[1]] for edge in edges]
  norm = mpl.colors.Normalize(vmin=-0.02, vmax=0.05)
  m= cm.ScalarMappable(norm=norm, cmap=cm.RdBu)
  edge_colors = [m.to_rgba(w) for w in weights]
  areas = [G.nodes[n]['area'] for n in nodes2plot]
  areas_uniq = list(set(areas))
  colors = [customPalette[areas_uniq.index(area)] for area in areas]
  # pos = nx.spring_layout(G, k=0.8, iterations=50) # make nodes as seperate as possible
  nx.draw_networkx_edges(G, pos, arrows=dire, edgelist=edges, edge_color=edge_colors, width=3.0, alpha=0.9) # , edge_cmap=plt.cm.Greens
  # nx.draw_networkx_nodes(G, pos, nodelist=degrees.keys(), node_size=[np.log(v + 2) * 20 for v in degrees.values()], 
  # print(len(set(nodes2plot)), len(nodes2plot), len(areas), len(degrees), len(colors))
  nx.draw_networkx_nodes(G, pos, nodelist=degrees.keys(), node_size=[15 * v for v in degrees.values()], 
  node_color=colors, alpha=0.6)
  for index, a in enumerate(areas_uniq):
    plt.scatter([],[], c=customPalette[index], label=a, s=30, alpha=0.6)
  legend = plt.legend(fontsize=15) # loc='lower right', 
  for handle in legend.legendHandles:
    handle.set_sizes([60.0])
  plt.tight_layout()
  image_name = './plots/example_graphs_each_region_color_cc_{}_{}_{}_{}fold.jpg' if cc else './plots/example_graphs_each_region_color_{}_{}_{}_{}fold.jpg'
  plt.savefig(image_name.format(row, col, measure, n))
  return weights, edge_colors



def plot_covering_comm_purity(G_dict, cover_p, area_dict, measure, n, sign, weight=False, max_reso=None, max_method='none'):
  rows, cols = get_rowcol(G_dict)
  if max_reso is None:
    max_reso = np.ones((len(rows), len(cols)))
  fig = plt.figure(figsize=(7, 4))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  covering_purity = []
  for col_ind, col in enumerate(cols):
    print(col)
    data = {}
    for row_ind, row in enumerate(rows):
      G = G_dict[row][col].copy() if col in G_dict[row] else nx.DiGraph()
      if nx.is_directed(G):
        G = G.to_undirected()
      if not weight:
        name = '{}_uw'.format(sign)
        unweight = {(i, j):1 for i,j in G.edges()}
        nx.set_edge_attributes(G, unweight, 'weight')
      else:
        name = '{}_w'.format(sign)
      comms = nx_comm.louvain_communities(G, weight='weight', resolution=max_reso[row_ind, col_ind])
      sizes = [len(comm) for comm in comms]
      # part = community.best_partition(G, weight='weight')
      # comms, sizes = np.unique(list(part.values()), return_counts=True)
      for comm, size in zip(comms, sizes):
        c_regions = [area_dict[row][node] for node in comm]
        _, counts = np.unique(c_regions, return_counts=True)
        assert len(c_regions) == size == counts.sum()
        purity = counts.max() / size
        if size in data:
          data[size].append(purity)
        else:
          data[size] = [purity]
    
    c_size, c_purity = [k for k,v in sorted(data.items(), reverse=True)], [v for k,v in sorted(data.items(), reverse=True)]
    ind = np.where(np.cumsum(c_size) >= cover_p * np.sum(c_size))[0][0]
    print(ind)
    # c_purity = [x for xs in c_purity for x in xs]
    covering_purity.append([x for xs in c_purity[:ind] for x in xs])
  plt.boxplot(covering_purity, showfliers=False)
  plt.xticks(list(range(1, len(covering_purity)+1)), cols, rotation=90)
  # plt.hist(data.flatten(), bins=12, density=True)
  # plt.axvline(x=np.nanmean(data), color='r', linestyle='--')
  # plt.xlabel('region')
  plt.ylabel('purity')
  plt.title(name + 'purity of largest communities covering {} nodes'.format(cover_p), size=18)
  plt.tight_layout()
  image_name = './plots/covering_{}_comm_purity_{}_{}_{}_{}fold.jpg'.format(cover_p, name, max_method, measure, n)
  # plt.show()
  plt.savefig(image_name)
  return covering_purity

def pos_neg_link_individual(G_dict, measure, n, density=False):
  rows, cols = get_rowcol(G_dict)
  num_row, num_col = len(rows), len(cols)
  links = np.zeros((num_row, num_col, 2))
  fig = plt.figure(figsize=(14, 7))
  # fig = plt.figure(figsize=(20, 10))
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      G = G_dict[row][col] if col in G_dict[row] else nx.DiGraph()
      signs = list(nx.get_edge_attributes(G, "sign").values())
      links[row_ind, col_ind, 0] = signs.count(1)
      links[row_ind, col_ind, 1] = signs.count(-1)
      if density:
        links[row_ind, col_ind] = links[row_ind, col_ind] / len(signs)
  ax = plt.subplot(1, 2, 1)
  for row_ind, row in enumerate(rows):
    plt.plot(cols, links[row_ind, :, 0], label=row, alpha=1)
  plt.gca().set_title('excitatory links', fontsize=20, rotation=0)
  ylabel = 'density' if density else 'number'
  plt.ylabel(ylabel)
  plt.xticks(rotation=90)
  plt.legend()
  ax = plt.subplot(1, 2, 2)
  for row_ind, row in enumerate(rows):
    plt.plot(cols, links[row_ind, :, 1], label=row, alpha=1)
  plt.gca().set_title('inhibitory links', fontsize=20, rotation=0)
  ylabel = 'density' if density else 'number'
  plt.ylabel(ylabel)
  plt.xticks(rotation=90)

  plt.legend()
  plt.tight_layout()
  figname = './plots/pos_neg_links_density_{}_{}_fold.jpg'.format(measure, n) if density else './plots/pos_neg_links_number_{}_{}_fold.jpg'.format(measure, n)
  plt.savefig(figname)

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

def plot_triad_fraction_relative_count(G_dict, triad_count, triad_types, p_triad_func, measure, n):
  rows, cols = get_rowcol(triad_count)
  frac_triad, relative_count = defaultdict(lambda: {}), defaultdict(lambda: {})
  for triad_type in triad_types:
    frac_triad[triad_type], relative_count[triad_type] = defaultdict(lambda: []), defaultdict(lambda: [])
  for col_ind, col in enumerate(cols):
    print(col)
    for row_ind, row in enumerate(rows):
      G = G_dict[row][col].copy() if col in G_dict[row] else nx.DiGraph()
      G_triad_count = nx.triads.triadic_census(G)
      num_triplet = sum(G_triad_count.values())
      p0, p1, p2 = count_pair_connection_p(G)
      for triad_type in triad_types:
        relative_c = G_triad_count[triad_type] / (num_triplet * p_triad_func[triad_type](p0, p1, p2)) if num_triplet * p_triad_func[triad_type](p0, p1, p2) else 0
        relative_count[triad_type][col].append(relative_c)
      
      t_count = triad_count[row][col].copy()
      safe_t_count = defaultdict(lambda: 0)
      for t_type in t_count:
        safe_t_count[t_type] = t_count[t_type]
      total_num = sum(t_count.values())
      for triad_type in triad_types:
        frac_triad[triad_type][col].append(safe_division(safe_t_count[triad_type], total_num))
  
  fig = plt.figure(figsize=(4*2, 3*2))
  # fig.patch.set_facecolor('black')
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  for ind, triad_type in enumerate(triad_types):
    ax = plt.subplot(2, 2, ind + 1)
    plt.gca().set_title(triad_type, fontsize=20, rotation=0)
    plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    for col in cols:
      plt.scatter(np.mean(frac_triad[triad_type][col]), np.mean(relative_count[triad_type][col]), label=col, marker='^', alpha=0.6)
    plt.xlabel('fraction of triad')
    plt.ylabel('relative count of triad')
  plt.legend()
  plt.tight_layout()
  # plt.tight_layout(rect=[0, 0.03, 1, 0.98])
  # plt.show()
  fname = './plots/triad_fraction_relative_count_{}_{}fold.jpg'
  plt.savefig(fname.format(measure, n))

def unique(l):
  u, ind = np.unique(l, return_index=True)
  return list(u[np.argsort(ind)])

def flat(array):
  return array.reshape(array.size//array.shape[-1], array.shape[-1])

from library import *
import importlib
import library
importlib.reload(library)

area_dict, active_area_dict, mean_speed_df = load_other_data(session_ids)
directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
path = directory.replace('spiking_sequence', 'adj_mat_ccg_highland_corrected')
if not os.path.exists(path):
  os.makedirs(path)
G_ccg_dict, offset_dict, duration_dict = load_highland_xcorr(path, active_area_dict, weight=True)
measure = 'ccg'
G_ccg_dict = remove_gabor(G_ccg_dict)
######### removed neurons from thalamic region
G_ccg_dict = remove_thalamic(G_ccg_dict, area_dict, visual_regions)
offset_dict = remove_thalamic_mat(offset_dict, active_area_dict, visual_regions)
duration_dict = remove_thalamic_mat(duration_dict, active_area_dict, visual_regions)
active_area_dict = remove_thalamic_area(active_area_dict, visual_regions)
n = 4
S_ccg_dict = add_sign(G_ccg_dict)
S_ccg_dict = add_offset(S_ccg_dict, offset_dict)
S_ccg_dict = add_duration(S_ccg_dict, duration_dict)
S_ccg_dict = add_delay(S_ccg_dict)
######### split G_dict into pos and neg
pos_G_dict, neg_G_dict = split_pos_neg(G_ccg_dict, measure=measure)
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
rewired_G_dict = random_graph_generator(G_dict, num_rewire, algorithm, measure, cc, Q=100)
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
# %%
violin_data(density_dict, 'density', measure, n)
# %%
# box plot
box_data(density_dict, 'density', measure, n)
# %%
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
# plot_intra_inter_kde(offset_dict, G_ccg_dict, 'total', 'offset', False, active_area_dict, measure, n)
plot_intra_inter_kde(duration_dict, G_ccg_dict, 'total', 'duration', True, active_area_dict, measure, n)
# plot_intra_inter_kde(duration_dict, G_ccg_dict, 'total', 'duration', False, active_area_dict, measure, n)
#%%
plot_intra_inter_kde_G(S_ccg_dict, 'delay', True, active_area_dict, measure, n)
#%%
def plot_intra_inter_scatter_G(G_dict, name, active_area_dict, measure, n, color_pa):
  rows, cols = get_rowcol(G_dict)
  num_row, num_col = len(rows), len(cols)
  fig = plt.figure(figsize=(5, 5))
  df = pd.DataFrame(columns=['data', 'type', 'col'])
  for col_ind, col in enumerate(cols):
    print(col)
    intra_data, inter_data = [], []
    for row_ind, row in enumerate(rows):
      active_area = active_area_dict[row]
      G = G_dict[row][col].copy()
      for edge in G.edges():
        if active_area[edge[0]] == active_area[edge[1]]:
          intra_data.append(G[edge[0]][edge[1]][name])
        else:
          inter_data.append(G[edge[0]][edge[1]][name])
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
  figname = './plots/intra_inter_scatter_{}_{}_{}fold.jpg'
  plt.savefig(figname.format(name, measure, n))
color_pa = ['tab:blue', 'darkorange', 'bisque', 'limegreen', 'darkgreen', 'maroon', 'indianred', 'mistyrose']
plot_intra_inter_scatter_G(S_ccg_dict, 'delay', active_area_dict, measure, n, color_pa)
# %%
# scatter plot of intra/inter duration over stimulus
color_pa = ['tab:blue', 'darkorange', 'bisque', 'limegreen', 'darkgreen', 'maroon', 'indianred', 'mistyrose']
plot_intra_inter_scatter(offset_dict, G_ccg_dict, 'total', 'offset', active_area_dict, measure, n, color_pa)
plot_intra_inter_scatter(duration_dict, G_ccg_dict, 'total', 'duration', active_area_dict, measure, n, color_pa)
# %%
################ box plot of inter intra region offset ratio
color_pa = ['tab:blue', 'darkorange', 'bisque', 'limegreen', 'darkgreen', 'maroon', 'indianred', 'mistyrose']
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
#%%
################# get optimal resolution that maximizes delta Q
rows, cols = get_rowcol(G_ccg_dict)
with open('metrics.pkl', 'rb') as f:
  metrics = pickle.load(f)
resolution_list = np.arange(0, 2.1, 0.1)
max_reso_gnm, max_reso_swap = get_max_resolution(rows, cols, resolution_list, metrics)
#%%
rows, cols = get_rowcol(G_ccg_dict)
# with open('comms_dict.pkl', 'rb') as f:
#     comms_dict = pickle.load(f)
with open('metrics.pkl', 'rb') as f:
    metrics = pickle.load(f)
plot_modularity_resolution(rows, cols, resolution_list, metrics, measure, n)
# plot_comm_diff_resolution(rows, cols, resolution_list, comms_dict, metrics, measure, n)
# %%
mouse_id, stimulus_id = 6, 7
weights, comms = plot_example_graphs_offset(G_ccg_dict, max_reso_gnm[mouse_id][2], offset_dict, 'total_offset_gnm', area_dict, active_area_dict, session_ids, stimulus_names, mouse_id, stimulus_id, measure, n, False)
# for mouse_id in range(len(session_ids)):
#   for stimulus_id in range(len(stimulus_names)):
#     if stimulus_id != 2:
#       print(session_ids[mouse_id], stimulus_names[stimulus_id])
#       # mouse_id, stimulus_id = 7, 3
#       weights, comms = plot_example_graphs_offset(G_ccg_dict, max_reso_gnm[mouse_id][2], offset_dict, 'total_offset_gnm', area_dict, active_area_dict, session_ids, stimulus_names, mouse_id, stimulus_id, measure, n, False)
#%%
################# purity of all communities
all_purity = plot_all_comm_purity(G_ccg_dict, area_dict, measure, n, 'total', weight=False, max_reso=max_reso_gnm, max_method='gnm')
purity_dict = {col:all_purity[cols.index(col)] for col in cols}
plot_p_value(purity_dict, 'all_purity', measure, n, 'ks_test', True)
plot_p_value(purity_dict, 'all_purity', measure, n, 'mwu_test', True)
# %%
################# purity of top communities covering cover_p nodes    NOT GOOD
covering_purity = plot_covering_comm_purity(G_ccg_dict, 0.7, area_dict, measure, n, 'total', weight=False, max_reso=max_reso_gnm, max_method='gnm')
purity_dict = {col:covering_purity[cols.index(col)] for col in cols}
plot_p_value(purity_dict, 'covering_purity', measure, n, 'ks_test', True)
plot_p_value(purity_dict, 'covering_purity', measure, n, 'mwu_test', True)
#%%
################# get optimal resolution that maximizes delta H
rows, cols = get_rowcol(G_ccg_dict)
with open('comms_dict.pkl', 'rb') as f:
  comms_dict = pickle.load(f)
with open('metrics.pkl', 'rb') as f:
  metrics = pickle.load(f)
#%%
resolution_list = np.arange(0, 2.1, 0.1)
max_reso_gnm, max_reso_config = get_max_dH_resolution(rows, cols, resolution_list, metrics)
############### community with Hamiltonian
max_pos_reso_gnm = get_max_pos_reso(G_ccg_dict, max_reso_gnm)
max_pos_reso_config = get_max_pos_reso(G_ccg_dict, max_reso_config)
#%%
plot_Hcomm_size_purity(comms_dict, area_dict, measure, n, max_neg_reso=max_reso_gnm, max_method='gnm')
plot_Hcomm_size_purity(comms_dict, area_dict, measure, n, max_neg_reso=max_reso_config, max_method='config')
#%%
top_purity_gnm = plot_top_Hcomm_purity(comms_dict, 1, area_dict, measure, n, max_neg_reso=max_reso_gnm, max_method='gnm')
top_purity_config = plot_top_Hcomm_purity(comms_dict, 1, area_dict, measure, n, max_neg_reso=max_reso_config, max_method='config')
# top_purity_config = plot_top_Hcomm_purity(comms_dict, 10, area_dict, measure, n, max_neg_reso=max_reso_config, max_method='config')
#%%
def plot_scatter_purity_Hcommsize(comms_dict, area_dict, measure, n, max_neg_reso=None, max_method='none'):
  ind = 1
  rows, cols = get_rowcol(comms_dict)
  if max_neg_reso is None:
    max_neg_reso = np.ones((len(rows), len(cols)))
  fig = plt.figure(figsize=(7, 6))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  size_dict = {}
  purity_dict = {}
  for col_ind, col in enumerate(cols):
    print(col)
    size_col = []
    purity_col = []
    for row_ind, row in enumerate(rows):
      max_reso = max_neg_reso[row_ind][col_ind]
      comms_list = comms_dict[row][col][max_reso]
      for comms in comms_list: # 100 repeats
        data = []
        sizes = [len(comm) for comm in comms]
        # part = community.best_partition(G, weight='weight')
        # comms, sizes = np.unique(list(part.values()), return_counts=True)
        for comm, size in zip(comms, sizes):
          c_regions = [area_dict[row][node] for node in comm]
          _, counts = np.unique(c_regions, return_counts=True)
          assert len(c_regions) == size == counts.sum()
          purity = counts.max() / size
          data.append((size, purity))
        size_col += [k for k,v in data if k>=4]
        purity_col += [v for k,v in data if k>=4]
    size_dict[col] = size_col
    purity_dict[col] = purity_col
  color_list = ['tab:blue', 'darkorange', 'bisque', 'limegreen', 'darkgreen', 'maroon', 'indianred', 'mistyrose']
  for col_ind, col in enumerate(size_dict):
    plt.scatter(size_dict[col], purity_dict[col], color=color_list[col_ind], label=col, alpha=0.8)
  plt.legend()
  # plt.xscale('log')
  plt.xlabel('community size')
  plt.ylabel('purity')
  plt.title('{} purity VS community size'.format(max_method), size=18)
  plt.tight_layout()
  image_name = './plots/Hcomm_purity_size_{}_{}_{}fold.jpg'.format(max_method, measure, n)
  # plt.show()
  plt.savefig(image_name)
#################### scatter of purity VS community size
plot_scatter_purity_Hcommsize(comms_dict, area_dict, measure, n, max_neg_reso=max_reso_gnm, max_method='gnm')
plot_scatter_purity_Hcommsize(comms_dict, area_dict, measure, n, max_neg_reso=max_reso_config, max_method='config')
#%%
#################### scatter of average purity VS community size
from scipy.optimize import curve_fit
def plot_scatter_mean_purity_Hcommsize(comms_dict, area_dict, measure, n, max_neg_reso=None, max_method='none'):
  ind = 1
  rows, cols = get_rowcol(comms_dict)
  if max_neg_reso is None:
    max_neg_reso = np.ones((len(rows), len(cols)))
  fig = plt.figure(figsize=(7, 6))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  size_dict = {}
  purity_dict = {}
  for col_ind, col in enumerate(cols):
    print(col)
    size_col = []
    purity_col = []
    data = defaultdict(list)
    for row_ind, row in enumerate(rows):
      max_reso = max_neg_reso[row_ind][col_ind]
      comms_list = comms_dict[row][col][max_reso]
      for comms in comms_list: # 100 repeats
        sizes = [len(comm) for comm in comms]
        # part = community.best_partition(G, weight='weight')
        # comms, sizes = np.unique(list(part.values()), return_counts=True)
        for comm, size in zip(comms, sizes):
          c_regions = [area_dict[row][node] for node in comm]
          _, counts = np.unique(c_regions, return_counts=True)
          assert len(c_regions) == size == counts.sum()
          purity = counts.max() / size
          data[size].append(purity)
    size_col = [k for k,v in data.items() if k>=4]
    purity_col = [np.mean(v) for k,v in data.items() if k>=4]
    size_dict[col] = size_col
    purity_dict[col] = purity_col
  for col_ind, col in enumerate(size_dict):
    plt.scatter(size_dict[col], purity_dict[col], color=stimulus_colors[col_ind], label=col, alpha=0.6)
    popt, pcov = curve_fit(func_powerlaw, size_dict[col], purity_dict[col], p0=[1, 1]) #, bounds=[[1e-3, 1e-3], [1e20, 50]]
    plt.plot(size_dict[col], func_powerlaw(size_dict[col], *popt), '--', color=stimulus_colors[col_ind], alpha=.4)
    
    # coef = np.polyfit(size_dict[col], purity_dict[col], 1)
    # poly1d_fn = np.poly1d(coef) 
    # plt.plot(size_dict[col], poly1d_fn(size_dict[col]), '--', color=stimulus_colors[col_ind], alpha=.4) #'--k'=black dashed line
  plt.legend()
  plt.xscale('log')
  plt.xlabel('community size')
  plt.ylabel('purity')
  plt.title('{} average purity VS community size'.format(max_method), size=18)
  plt.tight_layout()
  # image_name = './plots/Hcomm_mean_purity_size_{}_{}_{}fold.jpg'.format(max_method, measure, n)
  # plt.savefig(image_name)
  plt.show()

# plot_scatter_mean_purity_Hcommsize(comms_dict, area_dict, measure, n, max_neg_reso=max_reso_gnm, max_method='gnm')
plot_scatter_mean_purity_Hcommsize(comms_dict, area_dict, measure, n, max_neg_reso=max_reso_config, max_method='config')
#%%
#################### distribution of community size
plot_dist_Hcommsize(comms_dict, measure, n, max_neg_reso=max_reso_gnm, max_method='gnm')
plot_dist_Hcommsize(comms_dict, measure, n, max_neg_reso=max_reso_config, max_method='config')
#%%

#################### example topology of graphs, select one largest community for each region
rows, cols = get_rowcol(G_ccg_dict)
# for row_ind, row in enumerate(rows):
#   for col_ind, col in enumerate(cols):
#     print(row, col)
#     break
row_ind, col_ind = 4, 7
row, col = rows[row_ind], cols[col_ind]
max_reso = max_reso_gnm[row_ind][col_ind]
hamiltonian = metrics['Hamiltonian'][row_ind, col_ind, np.where(resolution_list==max_reso)[0][0], 0]
comms_list = comms_dict[row][col][max_reso]
weights, edge_colors = plot_example_graphs_each_region(G_ccg_dict, hamiltonian, comms_list[0], area_dict, active_area_dict, row, col, measure, n, False)
#%%
purity_dict = {col:top_purity_gnm[cols.index(col)] for col in cols}
plot_corrected_p_value(purity_dict, 'top_purity_1', measure, n, 'ks_test', True)
# plot_corrected_p_value(purity_dict, 'top_purity_1', measure, n, 'mwu_test', True)
#%%
# all_purity_gnm = plot_all_Hcomm_purity(G_ccg_dict, area_dict, measure, n, max_pos_reso=max_pos_reso_gnm, max_neg_reso=max_reso_gnm, max_method='gnm')
all_purity_config = plot_all_Hcomm_purity(G_ccg_dict, area_dict, measure, n, max_pos_reso=max_pos_reso_config, max_neg_reso=max_reso_config, max_method='config')
purity_dict = {col:all_purity_config[cols.index(col)] for col in cols}
plot_corrected_p_value(purity_dict, 'all_purity', measure, n, 'ks_test', True)
plot_corrected_p_value(purity_dict, 'all_purity', measure, n, 'mwu_test', True)
#%%
############### weighted purity by community size
# weighted_purity_gnm = plot_weighted_Hcomm_purity(G_ccg_dict, area_dict, measure, n, max_pos_reso=max_pos_reso_gnm, max_neg_reso=max_reso_gnm, max_method='gnm')
# weighted_purity_config = plot_weighted_Hcomm_purity(G_ccg_dict, area_dict, measure, n, max_pos_reso=max_pos_reso_config, max_neg_reso=max_reso_config, max_method='config')
purity_dict = {col:weighted_purity_config[cols.index(col)] for col in cols}
plot_corrected_p_value(purity_dict, 'weighted_purity', measure, n, 'ks_test', True)
plot_corrected_p_value(purity_dict, 'weighted_purity', measure, n, 'mwu_test', True)
#%%
method = 'ks_test'
data_dict = purity_dict
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
        p_less = stats.mannwhitneyu(data_dict[key1], data_dict[key2], alternative='less', method="asymptotic")[1]
        p_greater = stats.mannwhitneyu(data_dict[key1], data_dict[key2], alternative='greater', method="asymptotic")[1]
      stimulus_pvalue[key_ind1, key_ind2] = min(p_less, p_greater)

# %%
def plot_modularity_num_comms(G_dict, measure, n, max_reso=None, max_method='none'):
  rows, cols = get_rowcol(G_dict)
  if max_reso is None:
    max_reso = np.ones((len(rows), len(cols)))
  uw_modularity, uw_num_lcomm = [np.full([len(rows), len(cols)], np.nan) for _ in range(2)]
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      G = G_dict[row][col].copy() if col in G_dict[row] else nx.DiGraph()
      if nx.is_directed(G):
        G = G.to_undirected()
      unweight = {(i, j):1 for i,j in G.edges()}
      nx.set_edge_attributes(G, unweight, 'weight')
      comms = nx_comm.louvain_communities(G, weight='weight', resolution=max_reso[row_ind, col_ind], seed=1111)
      uw_modularity[row_ind, col_ind] = get_modularity(G, weight='weight', resolution=max_reso[row_ind, col_ind], comms=comms) # 
      count = np.array([len(comm) for comm in comms])
      uw_num_lcomm[row_ind, col_ind] = sum(count >= 4)
      assert G.number_of_nodes() == count.sum()
  num_col = 2
  num_row = 1
  fig = plt.figure(figsize=(5*num_col, 5*num_row))
  uw_modularity_list = [uw_modularity[:, col_ind] for col_ind in range(len(cols))]
  uw_num_lcomm_list = [uw_num_lcomm[:, col_ind] for col_ind in range(len(cols))]
  plt.subplot(num_row, num_col, 1)
  plt.boxplot(uw_modularity_list, showfliers=False)
  plt.xticks(list(range(1, len(uw_modularity_list)+1)), cols, rotation=90)
  plt.ylabel('modularity')
  plt.title('modularity', size=18)
  plt.subplot(num_row, num_col, 2)
  plt.boxplot(uw_num_lcomm_list, showfliers=False)
  plt.xticks(list(range(1, len(uw_num_lcomm_list)+1)), cols, rotation=90)
  plt.ylabel('number')
  plt.title('community (at least 4 nodes)', size=18)
  plt.tight_layout()
  # plt.show()
  figname = './plots/modularity_num_comms_{}_{}_{}fold.jpg'
  plt.savefig(figname.format(max_method, measure, n))
  return uw_modularity_list, uw_num_lcomm_list

uw_modularity_list, uw_num_lcomm_list = plot_modularity_num_comms(G_ccg_dict, measure, n, max_reso=max_reso_gnm, max_method='gnm')
# %%
def plot_modularity_num_comms_(G_dict, max_reso=None):
  rows, cols = get_rowcol(G_dict)
  if max_reso is None:
    max_reso = np.ones((len(rows), len(cols)))
  uw_modularity, uw_num_lcomm = [np.full([len(rows), len(cols)], np.nan) for _ in range(2)]
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      G = G_dict[row][col].copy() if col in G_dict[row] else nx.DiGraph()
      if nx.is_directed(G):
        G = G.to_undirected()
      unweight = {(i, j):1 for i,j in G.edges()}
      nx.set_edge_attributes(G, unweight, 'weight')
      comms = nx_comm.louvain_communities(G, weight='weight', resolution=max_reso[row_ind, col_ind], seed=4321)
      uw_modularity[row_ind, col_ind] = get_modularity(G, weight='weight', resolution=max_reso[row_ind, col_ind], comms=comms)
      count = np.array([len(comm) for comm in comms])
      uw_num_lcomm[row_ind, col_ind] = sum(count >= 4)
      assert G.number_of_nodes() == count.sum()
  metrics = {'total unweighted modularity':uw_modularity, 'total unweighted number of large communities':uw_num_lcomm}
  num_col = 1
  num_row = 2
  fig = plt.figure(figsize=(5*num_col, 5*num_row))
  for i, k in enumerate(metrics):
    plt.subplot(num_row, num_col, i+1)
    metric = metrics[k]
    for row_ind, row in enumerate(rows):
      mean = metric[row_ind, :]
      plt.plot(cols, mean, label=row, alpha=0.6)
    plt.gca().set_title(k, fontsize=14, rotation=0)
    plt.xticks(rotation=90)
    # plt.yscale('symlog')
    if i == len(metrics)-1:
      plt.legend()
    if i // num_col < num_row - 1:
      plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    plt.tight_layout()
  plt.show()

plot_modularity_num_comms_(G_ccg_dict, max_reso=max_reso_gnm)
#%%
######################## excitaroty link VS inhibitory link individual
# pos_neg_link_individual(S_ccg_dict, measure, n, density=False)
pos_neg_link_individual(S_ccg_dict, measure, n, density=True)
#%%
######################## excitatory link VS inhibitoray link with intra/inter region
def plot_bar_pos_neg_intra_inter(G_dict, active_area_dict, measure, n):
  df = pd.DataFrame(columns=['fraction of connections', 'type', 'region'])
  rows, cols = get_rowcol(G_dict)
  fig = plt.figure(figsize=(7*len(cols)/2, 6*2))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  ind = 1
  for col_ind, col in enumerate(cols):
    print(col)
    intra_exci, inter_exci, intra_inhi, inter_inhi = 0, 0, 0, 0
    for row_ind, row in enumerate(rows):
      G = G_dict[row][col] if col in G_dict[row] else nx.Graph()
      signs = nx.get_edge_attributes(G, "sign")
      active_area = active_area_dict[row]
      edges = list(G.edges())
      for nodei, nodej in edges:
        if active_area[nodei] == active_area[nodej]:
          if signs[(nodei, nodej)] > 0:
            intra_exci += 1
          else:
            intra_inhi += 1
        else:
          if signs[(nodei, nodej)] > 0:
            inter_exci += 1
          else:
            inter_inhi += 1
    intra_sum, inter_sum = intra_exci+intra_inhi, inter_exci+inter_inhi
    intra_exci /= intra_sum
    intra_inhi /= intra_sum
    inter_exci /= inter_sum
    inter_inhi /= inter_sum
    ax=plt.subplot(2, int(np.ceil(len(cols)/2)), ind)
    plt.gca().set_title(col, fontsize=30, rotation=0)
    plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ind += 1
    df = pd.concat([df, pd.DataFrame([[intra_exci, 'excitatory', 'intra']], columns=['fraction of connections', 'type', 'region']), 
                pd.DataFrame([[inter_exci, 'excitatory', 'inter']], columns=['fraction of connections', 'type', 'region']),
                pd.DataFrame([[intra_inhi, 'inhibitory', 'intra']], columns=['fraction of connections', 'type', 'region']),
                pd.DataFrame([[inter_inhi, 'inhibitory', 'inter']], columns=['fraction of connections', 'type', 'region'])], ignore_index=True)
    df['fraction of connections'] = pd.to_numeric(df['fraction of connections'])

    palette ={"excitatory": "lightcoral", "inhibitory": "lightsteelblue"}
    ax = sns.barplot(x='region', y='fraction of connections', hue='type', data=df, palette=palette)
    ax.set(xlabel=None)
  plt.tight_layout()
  figname = './plots/ex_in_intra_inter_{}_{}fold.jpg'
  # plt.savefig('./plots/violin_intra_inter_{}_{}fold.jpg'.format(measure, n))
  plt.savefig(figname.format(measure, n))

plot_bar_pos_neg_intra_inter(S_ccg_dict, active_area_dict, measure, n)
#%%
def plot_scatter_pos_neg_ratio_intra_inter(G_dict, active_area_dict, measure, n):
  color_list = ['tab:blue', 'darkorange', 'bisque', 'limegreen', 'darkgreen', 'maroon', 'indianred', 'mistyrose']
  rows, cols = get_rowcol(G_dict)
  for col_ind, col in enumerate(cols):
    intra_ratio, inter_ratio = [], []
    for row_ind, row in enumerate(rows):
      intra_exci, inter_exci, intra_inhi, inter_inhi = 0, 0, 0, 0
      G = G_dict[row][col] if col in G_dict[row] else nx.Graph()
      signs = nx.get_edge_attributes(G, "sign")
      active_area = active_area_dict[row]
      edges = list(G.edges())
      for nodei, nodej in edges:
        if active_area[nodei] == active_area[nodej]:
          if signs[(nodei, nodej)] > 0:
            intra_exci += 1
          else:
            intra_inhi += 1
        else:
          if signs[(nodei, nodej)] > 0:
            inter_exci += 1
          else:
            inter_inhi += 1
      intra_sum, inter_sum = intra_exci+intra_inhi, inter_exci+inter_inhi
      intra_exci /= intra_sum
      intra_inhi /= intra_sum
      inter_exci /= inter_sum
      inter_inhi /= inter_sum
      intra_ratio.append(intra_exci/intra_inhi)
      inter_ratio.append(inter_exci/inter_inhi)
    plt.scatter(intra_ratio, inter_ratio, label=col, color=color_list[col_ind], alpha=0.8)
  plt.legend()
  plt.title('excitatory/inhibitory ratio')
  plt.xlabel('ratio for intra-region links')
  plt.ylabel('ratio for inter-region links')
  plt.tight_layout()
  figname = './plots/ex_in_ratio_intra_inter_{}_{}fold.jpg'
  # plt.savefig('./plots/violin_intra_inter_{}_{}fold.jpg'.format(measure, n))
  plt.savefig(figname.format(measure, n))

plot_scatter_pos_neg_ratio_intra_inter(S_ccg_dict, active_area_dict, measure, n)
#%%
################# relative count of neuron pair
p_pair_func = {
  '0': lambda p: (1 - p)**2,
  '1': lambda p: 2 * (p * (1-p)),
  '2': lambda p: p**2,
}
# plot_pair_relative_count(S_ccg_dict, p_pair_func, measure, n, scale=False)
plot_pair_relative_count(S_ccg_dict, p_pair_func, measure, n, log=True, scale=True)
#%%
################# relative count of triad
p_triad_func = {
  '003': lambda p0, p1, p2: p0**3,
  '012': lambda p0, p1, p2: 6 * (p0**2 * p1),
  '102': lambda p0, p1, p2: 3 * (p0**2 * p2),
  '021D': lambda p0, p1, p2: 3 * (p0 * p1**2),
  '021U': lambda p0, p1, p2: 3 * (p0 * p1**2),
  '021C': lambda p0, p1, p2: 6 * (p0 * p1**2),
  '111D': lambda p0, p1, p2: 6 * (p0 * p1 * p2),
  '111U': lambda p0, p1, p2: 6 * (p0 * p1 * p2),
  '030T': lambda p0, p1, p2: 6 * (p1**3),
  '030C': lambda p0, p1, p2: 2 * (p1**3),
  '201': lambda p0, p1, p2: 3 * (p0 * p2**2),
  '120D': lambda p0, p1, p2: 3 * (p1**2 * p2),
  '120U': lambda p0, p1, p2: 3 * (p1**2 * p2),
  '120C': lambda p0, p1, p2: 6 * (p1**2 * p2),
  '210': lambda p0, p1, p2: 6 * (p1 * p2**2),
  '300': lambda p0, p1, p2: p2**3,
}
plot_triad_relative_count(S_ccg_dict, p_triad_func, measure, n, log=True)
#%%
################# p value of relative count of certain neuron pair/triad
def plot_singlepair_relative_count(G_dict, p_pair_func, pair_type, measure, n, log=False):
  ind = 1
  rows, cols = get_rowcol(G_dict)
  fig = plt.figure(figsize=(6, 6))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  pair_count = {}
  for col in cols:
    pair_count[col] = []
    print(col)
    ind += 1
    all_pair_count = defaultdict(lambda: [])
    for row in rows:
      G = G_dict[row][col].copy() if col in G_dict[row] else nx.DiGraph()
      p = nx.density(G)
      p0, p1, p2 = count_pair_connection_p(G)
      all_pair_count['0'].append(p0 / p_pair_func['0'](p))
      all_pair_count['1'].append(p1 / p_pair_func['1'](p))
      all_pair_count['2'].append(p2 / p_pair_func['2'](p))
    
    pair_count[col] = all_pair_count[pair_type]
  plt.boxplot([pair_count[col] for col in pair_count], showfliers=False)
  plt.xticks(list(range(1, len(pair_count)+1)), cols, rotation=90)
  left, right = plt.xlim()
  plt.hlines(1, xmin=left, xmax=right, color='r', linestyles='--', linewidth=0.5)
  # plt.hlines(1, color='r', linestyles='--')
  if log:
    plt.yscale('log')
  # plt.hist(data.flatten(), bins=12, density=True)
  # plt.axvline(x=np.nanmean(data), color='r', linestyle='--')
  # plt.xlabel('region')
  # plt.xlabel('size')
  plt.ylabel('relative count')
  plt.suptitle('Relative count of pair {}'.format(pair_type), size=20)
  plt.tight_layout(rect=[0, 0.03, 1, 0.98])
  image_name = './plots/relative_count_singlepair_{}_{}_{}fold.jpg'
  # plt.show()
  plt.savefig(image_name.format(pair_type, measure, n))
  return pair_count

pair_count = plot_singlepair_relative_count(S_ccg_dict, p_pair_func, '2', measure, n, log=False)
#%%
plot_p_value(pair_count, 'pair2', measure, n, 'ks_test', True)
#%%
################# p value of relative count of certain neuron pair/triad
def plot_singletriad_relative_count(G_dict, p_triad_func, ftriad_type, measure, n, log=False):
  ind = 1
  rows, cols = get_rowcol(G_dict)
  fig = plt.figure(figsize=(6, 6))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  triad_count = {}
  for col in cols:
    triad_count[col] = []
    print(col)
    ind += 1
    all_triad_count = defaultdict(lambda: [])
    for row in rows:
      G = G_dict[row][col].copy() if col in G_dict[row] else nx.DiGraph()
      G_triad_count = nx.triads.triadic_census(G)
      num_triplet = sum(G_triad_count.values())
      p0, p1, p2 = count_pair_connection_p(G)
      for triad_type in G_triad_count:
        relative_c = G_triad_count[triad_type] / (num_triplet * p_triad_func[triad_type](p0, p1, p2)) if num_triplet * p_triad_func[triad_type](p0, p1, p2) else 0
        all_triad_count[triad_type].append(relative_c)
    
    triad_count[col] = all_triad_count[ftriad_type]
  plt.boxplot([triad_count[col] for col in triad_count], showfliers=False)
  plt.xticks(list(range(1, len(triad_count)+1)), cols, rotation=90)
  left, right = plt.xlim()
  plt.hlines(1, xmin=left, xmax=right, color='r', linestyles='--', linewidth=0.5)
  # plt.hlines(1, color='r', linestyles='--')
  if log:
    plt.yscale('log')
  # plt.hist(data.flatten(), bins=12, density=True)
  # plt.axvline(x=np.nanmean(data), color='r', linestyle='--')
  # plt.xlabel('region')
  # plt.xlabel('size')
  plt.ylabel('relative count')
  plt.suptitle('Relative count of triad {}'.format(ftriad_type), size=20)
  plt.tight_layout(rect=[0, 0.03, 1, 0.98])
  image_name = './plots/relative_count_singletriad_{}_{}_{}fold.jpg'
  # plt.show()
  plt.savefig(image_name.format(ftriad_type, measure, n))
  return triad_count

triad_count = plot_singletriad_relative_count(S_ccg_dict, p_triad_func, '300', measure, n, log=False)
#%%
plot_p_value(triad_count, '300', measure, n, 'ks_test', True)
#%%
all_triads = get_all_signed_transitive_triads(S_ccg_dict)
triad_count = triad_census(all_triads)
summice_triad_count = summice_triad_census(all_triads)
meanmice_triad_percent = meanmice_triad_census(all_triads)
signed_triad_count = signed_triad_census(all_triads)
summice_signed_triad_count = summice_signed_triad_census(all_triads)
meanmice_signed_triad_percent = meanmice_signed_triad_census(all_triads)
#%%
####################### transitive triad distribution pie chart
tran_triad_types = ['030T', '120D', '120U', '300']
triad_colormap = {'030T':'Greens', '120D':'Blues', '120U':'Reds', '300':'Purples'}
plot_multi_pie_chart_census(summice_triad_count, tran_triad_types, triad_colormap, measure, n, False)
#%%
####################### fraction of motif VS its relative count
p_triad_func = {
  '003': lambda p0, p1, p2: p0**3,
  '012': lambda p0, p1, p2: 6 * (p0**2 * p1),
  '102': lambda p0, p1, p2: 3 * (p0**2 * p2),
  '021D': lambda p0, p1, p2: 3 * (p0 * p1**2),
  '021U': lambda p0, p1, p2: 3 * (p0 * p1**2),
  '021C': lambda p0, p1, p2: 6 * (p0 * p1**2),
  '111D': lambda p0, p1, p2: 6 * (p0 * p1 * p2),
  '111U': lambda p0, p1, p2: 6 * (p0 * p1 * p2),
  '030T': lambda p0, p1, p2: 6 * (p1**3),
  '030C': lambda p0, p1, p2: 2 * (p1**3),
  '201': lambda p0, p1, p2: 3 * (p0 * p2**2),
  '120D': lambda p0, p1, p2: 3 * (p1**2 * p2),
  '120U': lambda p0, p1, p2: 3 * (p1**2 * p2),
  '120C': lambda p0, p1, p2: 6 * (p1**2 * p2),
  '210': lambda p0, p1, p2: 6 * (p1 * p2**2),
  '300': lambda p0, p1, p2: p2**3,
}
tran_triad_types = ['030T', '120D', '120U', '300']
plot_triad_fraction_relative_count(G_ccg_dict, triad_count, tran_triad_types, p_triad_func, measure, n)
#%%
######################## compare time lag for edges that exist in all stimuli
intra_offset, inter_offset = plot_existed_edge_offset_stimulus(offset_dict, G_ccg_dict, active_area_dict, measure, n)
# %%
######################## compare time lag pairwise for stimuli for edges that exist in both
intra_df, inter_df = plot_pairwise_existed_edge_offset_stimulus(offset_dict, G_ccg_dict, active_area_dict, measure, n)
# %%
############################## figure for publication ##############################
##############################                        ##############################

#%%
def alt_bands(ax=None):
  ax = ax or plt.gca()
  x_left, x_right = ax.get_xlim()
  locs = ax.get_xticks().astype(float)
  locs -= .5
  locs = np.concatenate((locs, [x_right]))
  
  type_loc1, type_loc2 = locs[[0, 1, 3, 5]], locs[[1, 3, 5, 8]]
  for loc1, loc2 in zip(type_loc1, type_loc2):
    ax.axvspan(loc1, loc2, facecolor=stimulus_type_color[type_loc1.tolist().index(loc1)], alpha=0.2)
  ax.set_xlim(x_left, x_right)

def plot_ex_in_bar(G_dict, measure, n, density=False):
  df = pd.DataFrame()
  rows, cols = get_rowcol(G_dict)
  for col_ind, col in enumerate(cols):
    print(col)
    ex_data, in_data = [], []
    for row_ind, row in enumerate(rows):
      G = G_dict[row][col] if col in G_dict[row] else nx.Graph()
      signs = list(nx.get_edge_attributes(G, "sign").values())
      if density:
        ex_data.append(signs.count(1) / len(signs))
        in_data.append(signs.count(-1) / len(signs))
      else:
        ex_data.append(signs.count(1))
        in_data.append(signs.count(-1))
    df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(ex_data)[:,None], np.array(['excitatory'] * len(ex_data))[:,None], np.array([col] * len(ex_data))[:,None]), 1), columns=['number of connections', 'type', 'stimulus']), 
                pd.DataFrame(np.concatenate((np.array(in_data)[:,None], np.array(['inhibitory'] * len(in_data))[:,None], np.array([col] * len(in_data))[:,None]), 1), columns=['number of connections', 'type', 'stimulus'])], ignore_index=True)
  df['number of connections'] = pd.to_numeric(df['number of connections'])
  if density:
    y = 'density'
    df['density'] = df['number of connections']
  else:
    y = 'number of connections'
  fig = plt.figure(figsize=(10, 5))
  # ax = sns.violinplot(x='stimulus', y='number of connections', hue="type", data=df, palette="muted", split=False)
  # ax = sns.barplot(x='stimulus', y=y, hue="type", data=df, palette=['white', 'grey'], edgecolor=".5")
  barcolors = ['0', '.6']
  ax = sns.barplot(
      x="stimulus", 
      y=y, 
      hue="type",  
      data=df,
      palette=barcolors,
      ci="sd", 
      edgecolor="black",
      errcolor="black",
      errwidth=1.5,
      capsize = 0.1,
      alpha=0.7)
  sns.stripplot(
      x="stimulus", 
      y=y, 
      hue="type",
      palette=barcolors,
      data=df, dodge=True, alpha=0.6, ax=ax
  )
  # remove extra legend handles
  handles, labels = ax.get_legend_handles_labels()
  ax.legend(handles[2:], labels[2:], title='', bbox_to_anchor=(.75, 1.), loc='upper left', fontsize=14)
  plt.setp(ax.get_legend().get_texts(), weight='bold')
  plt.xticks(ticks=range(len(cols)), labels=paper_label, fontsize=14, weight='bold')
  plt.yticks(fontsize=14,  weight='bold')
  plt.ylabel(y)
  ax.set_ylabel(ax.get_ylabel(), fontsize=14, weight='bold',color='0.2')
  for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2.5)
    ax.spines[axis].set_color('0.2')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.tick_params(width=2.5)

  # plt.yscale('log')
  ax.set(xlabel=None)
  alt_bands(ax)
  mean_links = df.groupby(['stimulus', 'type']).mean().reset_index().groupby('stimulus').sum()
  for i, col in enumerate(cols):
    plt.hlines(y=mean_links.loc[col], xmin=(i - .18), xmax=(i + .18), color='white', linewidth=3) # , linestyles=(0, (1,1))
  plt.tight_layout()
  figname = './plots/box_ex_in_num_{}_{}fold.pdf' if not density else './plots/box_ex_in_density_{}_{}fold.pdf'
  # plt.savefig('./plots/violin_intra_inter_{}_{}fold.jpg'.format(measure, n))
  plt.savefig(figname.format(measure, n), transparent=True)
######################## excitaroty link VS inhibitory link box
plot_ex_in_bar(S_ccg_dict, measure, n, density=False)
#%%
def scatter_dataVSdensity(G_dict, area_dict, regions, name='intra'):
  rows, cols = get_rowcol(G_dict)
  fig, ax = plt.subplots(figsize=(5, 5))
  X, Y = [], []
  df = pd.DataFrame()
  region_connection = np.zeros((len(rows), len(cols), len(regions), len(regions)))
  for col_ind, col in enumerate(cols):
    # print(col)
    intra_data, inter_data, density_data, ex_data, in_data = [], [], [], [], []
    for row_ind, row in enumerate(rows):
      G = G_dict[row][col] if col in G_dict[row] else nx.Graph()
      nodes = list(G.nodes())
      node_area = {key: area_dict[row][key] for key in nodes}
      areas = list(node_area.values())
      A = nx.to_numpy_array(G)
      A[A.nonzero()] = 1
      for region_ind_i, region_i in enumerate(regions):
        for region_ind_j, region_j in enumerate(regions):
          region_indices_i = np.array([k for k, v in area_dict[row].items() if v==region_i])
          region_indices_j = np.array([k for k, v in area_dict[row].items() if v==region_j])
          region_indices_i = np.array([nodes.index(i) for i in list(set(region_indices_i) & set(nodes))]) # some nodes not in cc are removed 
          region_indices_j = np.array([nodes.index(i) for i in list(set(region_indices_j) & set(nodes))])
          if len(region_indices_i) and len(region_indices_j):
            region_connection[row_ind, col_ind, region_ind_i, region_ind_j] = np.sum(A[region_indices_i[:, None], region_indices_j])
            assert np.sum(A[region_indices_i[:, None], region_indices_j]) == len(A[region_indices_i[:, None], region_indices_j].nonzero()[0])
      diag_indx = np.eye(len(regions),dtype=bool)
      # metric[row_ind, col_ind, 0] =  np.sum(region_connection[row_ind, col_ind][diag_indx])
      # metric[row_ind, col_ind, 1] =  np.sum(region_connection[row_ind, col_ind][~diag_indx])
      intra_data.append(np.sum(region_connection[row_ind, col_ind][diag_indx])/np.sum(region_connection[row_ind, col_ind]))
      inter_data.append(np.sum(region_connection[row_ind, col_ind][~diag_indx])/np.sum(region_connection[row_ind, col_ind]))
      density_data.append(nx.density(G))
      signs = list(nx.get_edge_attributes(G, "sign").values())
      ex_data.append(signs.count(1) / len(signs))
      in_data.append(signs.count(-1) / len(signs))
    X += density_data
    if name == 'intra':
      Y += intra_data
    elif name == 'ex':
      Y += ex_data
    df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(intra_data)[:,None], np.array(inter_data)[:,None], np.array(ex_data)[:,None], np.array(density_data)[:,None], np.array([col] * len(intra_data))[:,None]), 1), columns=['ratio of intra-region connections', 'ratio of inter-region connections', 'ratio of excitatory connections', 'density', 'stimulus'])], ignore_index=True)
  df['ratio of intra-region connections'] = pd.to_numeric(df['ratio of intra-region connections'])
  df['ratio of inter-region connections'] = pd.to_numeric(df['ratio of inter-region connections'])
  df['ratio of excitatory connections'] = pd.to_numeric(df['ratio of excitatory connections'])
  df['density'] = pd.to_numeric(df['density'])
  stimulus_by_type = [['spontaneous'], ['flash_dark', 'flash_light'], ['drifting_gratings', 'static_gratings'], ['natural_scenes', 'natural_movie_one', 'natural_movie_three']]
  labels = ['resting state', 'flashes', 'gratings', 'natural stimuli']
  for st_ind, stype in enumerate(stimulus_by_type):
    x = df[df['stimulus'].isin(stype)]['density'].values
    if name == 'intra':
      y = df[df['stimulus'].isin(stype)]['ratio of intra-region connections'].values
    elif name == 'ex':
      y = df[df['stimulus'].isin(stype)]['ratio of excitatory connections'].values
    ax.scatter(x, y, facecolors='none', edgecolors=stimulus_type_color[st_ind], label=labels[st_ind], alpha=.6)
  X, Y = (list(t) for t in zip(*sorted(zip(X, Y))))
  X, Y = np.array(X), np.array(Y)
  if name == 'intra':
    slope, intercept, r_value, p_value, std_err = stats.linregress(X,Y)
    line = slope*X+intercept
    locx, locy = .8, .8
    text = 'r={:.2f}, p={:.2f}'.format(r_value, p_value)
  elif name == 'ex':
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(X),Y)
    line = slope*np.log10(X)+intercept
    locx, locy = .8, .3
    text = 'r={:.2f}, p={:.1e}'.format(r_value, p_value)
  ax.plot(X, line, color='.2', linestyle=(5,(10,3)), alpha=.5)
  # ax.scatter(X, Y, facecolors='none', edgecolors='.2', alpha=.6)
  ax.text(locx, locy, text, horizontalalignment='center',
     verticalalignment='center', transform=ax.transAxes, fontsize=16)
  if name == 'intra':
    plt.legend(loc='lower right', fontsize=14, frameon=False)
  plt.xticks(fontsize=14) #, weight='bold'
  plt.yticks(fontsize=14) # , weight='bold'
  plt.xlabel('density')
  if name == 'intra':
    ylabel = 'ratio of intra-region connections'
  elif name == 'ex':
    ylabel = 'ratio of excitatory connections'
    plt.xscale('log')
  plt.ylabel(ylabel)
  ax.set_xlabel(ax.get_xlabel(), fontsize=14,color='k') #, weight='bold'
  ax.set_ylabel(ax.get_ylabel(), fontsize=14,color='k') #, weight='bold'
  for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(1.5)
    ax.spines[axis].set_color('0.2')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.tick_params(width=1.5)
  plt.tight_layout()
  # plt.show()
  plt.savefig(f'./plots/{name}_density.pdf', transparent=True)

scatter_dataVSdensity(S_ccg_dict, area_dict, visual_regions, name='intra')
scatter_dataVSdensity(S_ccg_dict, area_dict, visual_regions, name='ex')
#%%
# def get_region_FR(session_ids, stimulus_names, regions, active_area_dict):
#   directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
#   if not os.path.isdir(directory):
#     os.mkdir(directory)
#   FR = np.zeros((len(regions), len(stimulus_names), len(session_ids)))
#   for se_ind, session_id in enumerate(session_ids):
#     active_area = active_area_dict[session_id]
#     node_idx = sorted(active_area.keys())
#     for st_ind, stimulus_name in enumerate(stimulus_names):
#       file = str(session_id) + '_' + stimulus_name + '.npz'
#       print(file)
#       sequences = load_npz_3d(os.path.join(directory, file))
#       for r_ind, region in enumerate(regions):
#         active_nodes = [node for node in node_idx if active_area[node]==region]
#         if len(active_nodes):
#           FR[r_ind, st_ind, se_ind] = 1000 * sequences[active_nodes].mean(1).sum(1).mean(0) / sequences.shape[2] # firing rate in Hz
#   return FR

def get_region_FR(session_ids, stimulus_names, regions, active_area_dict):
  directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
  if not os.path.isdir(directory):
    os.mkdir(directory)
  FR = {}
  for se_ind, session_id in enumerate(session_ids):
    FR[session_id] = {}
    active_area = active_area_dict[session_id]
    node_idx = sorted(active_area.keys())
    for st_ind, stimulus_name in enumerate(stimulus_names):
      FR[session_id][stimulus_name] = {}
      file = str(session_id) + '_' + stimulus_name + '.npz'
      print(file)
      sequences = load_npz_3d(os.path.join(directory, file))
      for region in regions:
        active_nodes = [node for node in node_idx if active_area[node]==region]
        if len(active_nodes):
          FR[session_id][stimulus_name][region] = FR[session_id][stimulus_name].get(region, []) + (1000 * sequences[active_nodes].mean(1).sum(1) / sequences.shape[2]).tolist() # firing rate in Hz
  return FR

FR = get_region_FR(session_ids, stimulus_names, visual_regions, active_area_dict)
#%%
# def get_region_links(G_dict, regions, area_dict):
  # rows, cols = get_rowcol(G_dict)
  # links = np.zeros((len(regions), len(cols), len(rows)))
  # for col_ind, col in enumerate(cols):
  #   print(col)
  #   for row_ind, row in enumerate(rows):
  #     G = G_dict[row][col] if col in G_dict[row] else nx.Graph()
  #     nodes = sorted(G.nodes())
  #     node_area = {key: area_dict[row][key] for key in nodes}
  #     areas = list(node_area.values())
  #     A = nx.to_numpy_array(G, nodelist=nodes)
  #     A[A.nonzero()] = 1
  #     for region_ind_i, region_i in enumerate(regions):
  #       region_indices_i = np.array([k for k, v in area_dict[row].items() if v==region_i])
  #       region_indices_i = np.array([nodes.index(i) for i in list(set(region_indices_i) & set(nodes))]) # some nodes not in cc are removed
  #       if len(region_indices_i):
  #         links[region_ind_i, col_ind, row_ind] = np.sum(A[region_indices_i[:, None], :]) + np.sum(A[:, region_indices_i]) # both out and in links
  # return links

def get_region_links(G_dict, regions, active_area_dict):
  rows, cols = get_rowcol(G_dict)
  intra_links, inter_links = {}, {}
  for row_ind, row in enumerate(rows):
    intra_links[row], inter_links[row] = {}, {}
    print(row)
    active_area = active_area_dict[row]
    for col_ind, col in enumerate(cols):
      intra_links[row][col], inter_links[row][col] = {}, {}
      G = G_dict[row][col]
      # nodes = sorted(G.nodes())
      node_idx = sorted(active_area.keys())
      A = nx.to_numpy_array(G, nodelist=node_idx)
      # A[A < 0] = 0 # only excitatory links!!!
      A[A.nonzero()] = 1
      for region in regions:
        active_nodes = [node for node in node_idx if active_area[node]==region]
        region_indices = np.array([node_idx.index(i) for i in active_nodes])
        if len(region_indices):
          # intra_links[row][col][region] = intra_links[row][col].get(region, []) + (np.sum(A[region_indices[:,None], region_indices], 1)).tolist() # only out links, within region links
          # inter_links[row][col][region] = inter_links[row][col].get(region, []) + (np.sum(A[region_indices, :], 1) - np.sum(A[region_indices[:,None], region_indices], 1)).tolist() # only out links, cross region links
          # intra_links[row][col][region] = intra_links[row][col].get(region, []) + (np.sum(A[region_indices[:,None], region_indices], 0)).tolist() # only in links, within region links
          # inter_links[row][col][region] = inter_links[row][col].get(region, []) + (np.sum(A[:, region_indices], 0) - np.sum(A[region_indices[:,None], region_indices], 0)).tolist() # only in links, cross region links
          intra_links[row][col][region] = intra_links[row][col].get(region, []) + (np.sum(A[region_indices[:,None], region_indices], 1) + np.sum(A[region_indices[:,None], region_indices], 0)).tolist() # both out and in links, within region links
          inter_links[row][col][region] = inter_links[row][col].get(region, []) + (np.sum(A[region_indices, :], 1) - np.sum(A[region_indices[:,None], region_indices], 1) + np.sum(A[:, region_indices], 0) - np.sum(A[region_indices[:,None], region_indices], 0)).tolist() # both out and in links, cross region links
  return intra_links, inter_links

intra_links, inter_links = get_region_links(G_ccg_dict, visual_regions, active_area_dict)
#%%
def plot_FR_links_region(data, regions, dataname):
  if dataname == 'FR':
    name = 'firing rate (Hz)'
  elif dataname == 'link':
    name = 'degree'
  df = pd.DataFrame()
  for region in regions:
    for stimulus_name in stimulus_names:
      for se_ind in session2keep:
        sub_data = np.array(data[se_ind][stimulus_name][region])
        df = pd.concat([df, pd.DataFrame(np.concatenate((sub_data[:,None], np.array([stimulus_name] * len(sub_data))[:,None], np.array([region] * len(sub_data))[:,None]), 1), columns=[name, 'stimulus', 'region'])], ignore_index=True)
  df[name] = pd.to_numeric(df[name])
  # return df
  plt.figure(figsize=(10, 14))
  colors_transparency = [transparent_rgb(colors.to_rgb(color), [1,1,1], alpha=.8) for color in region_colors]
  ax = sns.boxplot(y="stimulus", x=name, hue="region", hue_order=regions, data=df[(np.abs(stats.zscore(df[name])) < 2)], orient='h', palette=colors_transparency, showfliers=False) # , boxprops=dict(alpha=.6)
  # ax = sns.violinplot(x="stimulus", y=name, inner='box', cut=0, hue="region", scale="count", hue_order=regions, data=df[(np.abs(stats.zscore(df[name])) < 2)], palette=colors_transparency) # , boxprops=dict(alpha=.6)
  
  sns.stripplot(
      y="stimulus", 
      x=name, 
      hue="region",
      palette=colors_transparency,
      data=df[(np.abs(stats.zscore(df[name])) < 2)], dodge=True, alpha=0.1
  )
  ax.set(ylabel=None)
  handles, labels = ax.get_legend_handles_labels()
  ax.legend(handles[:6], labels[:6], title='', bbox_to_anchor=(.9, 1.), loc='upper left', fontsize=12)
  # plt.setp(ax.get_legend().get_texts())
  plt.yticks(ticks=range(len(cols)), labels=paper_label, fontsize=14)
  plt.xticks(fontsize=14)
  # plt.ylabel(y)
  ax.set_xlabel(ax.get_xlabel(), fontsize=14,color='0.2')
  for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(1.5)
    ax.spines[axis].set_color('0.2')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.tick_params(width=2.5)

  box_patches = [patch for patch in ax.patches if type(patch) == mpl.patches.PathPatch]
  if len(box_patches) == 0:  # in matplotlib older than 3.5, the boxes are stored in ax2.artists
    box_patches = ax.artists
  num_patches = len(box_patches)
  lines_per_boxplot = len(ax.lines) // num_patches
  for i, patch in enumerate(box_patches):
    # Set the linecolor on the patch to the facecolor, and set the facecolor to None
    col = patch.get_facecolor()
    patch.set_edgecolor(col)
    patch.set_facecolor('None')
    # Each box has associated Line2D objects (to make the whiskers, fliers, etc.)
    # Loop over them here, and use the same color as above
    for line in ax.lines[i * lines_per_boxplot: (i + 1) * lines_per_boxplot]:
      line.set_color(col)
      line.set_mfc(col)  # facecolor of fliers
      line.set_mec(col)  # edgecolor of fliers
  # Also fix the legend
  for legpatch in ax.legend_.get_patches():
    col = legpatch.get_facecolor()
    legpatch.set_edgecolor(col)
    legpatch.set_facecolor('None')
  # plt.show()
  plt.tight_layout()
  plt.savefig(f'./plots/{dataname}_region_stimulus.pdf', transparent=True)

plot_FR_links_region(FR, visual_regions, 'FR')
# plot_FR_links_region(links, visual_regions, 'link')
#%%
def get_all_FR(session_ids, stimulus_names, active_area_dict):
  directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
  all_FR = []
  for se_ind, session_id in enumerate(session_ids):
    active_area = active_area_dict[session_id]
    node_idx = sorted(active_area.keys())
    for st_ind, stimulus_name in enumerate(stimulus_names):
      file = str(session_id) + '_' + stimulus_name + '.npz'
      print(file)
      sequences = load_npz_3d(os.path.join(directory, file))
      for node_ind in node_idx:
        all_FR.append(1000 * sequences[node_ind].mean(0).sum() / sequences.shape[2])
  return all_FR

all_FR = get_all_FR(session_ids, stimulus_names, active_area_dict)
#%%
def get_all_links(G_dict, regions, active_area_dict):
  rows, cols = get_rowcol(G_dict)
  all_links = []
  for row in rows:
    active_area = active_area_dict[row]
    for col in cols:
      G = G_dict[row][col]
      node_idx = sorted(active_area.keys())
      A = nx.to_numpy_array(G, nodelist=node_idx)
      A[A.nonzero()] = 1
      region_links = np.zeros(len(node_idx))
      for region in regions:
        active_nodes = [node for node in node_idx if active_area[node]==region]
        region_indices = np.array([node_idx.index(i) for i in active_nodes])
        if len(region_indices):
          region_links[region_indices] = np.sum(A[region_indices[:,None], region_indices], 1) + np.sum(A[region_indices[:,None], region_indices], 0) # within region links
      all_links += region_links.tolist() # total degree, not in or out alone
  return all_links
all_links = get_all_links(G_ccg_dict, visual_regions, active_area_dict)
#%%
def scatter_linkVSFR(all_FR, all_links):
  X, Y = all_FR, all_links
  fig, ax = plt.subplots(figsize=(5, 5))
  ax.scatter(X, Y, facecolors='none', edgecolors='.1', alpha=.5)
  X, Y = (list(t) for t in zip(*sorted(zip(X, Y))))
  X, Y = np.array(X), np.array(Y)
  slope, intercept, r_value, p_value, std_err = stats.linregress(X,Y)
  line = slope*X+intercept
  locx, locy = .8, .8
  text = 'r={:.2f}, p={:.0e}'.format(r_value, p_value)
  ax.plot(X, line, color='.2', linestyle=(5,(10,3)), alpha=.5)
  ax.text(locx, locy, text, horizontalalignment='center',
     verticalalignment='center', transform=ax.transAxes, fontsize=16)
  # plt.legend(loc='lower right', fontsize=14, frameon=False)
  plt.xticks(fontsize=14) #, weight='bold'
  plt.yticks(fontsize=14) # , weight='bold'
  plt.xlabel('firing rate (Hz)')
  ylabel = 'degree'
  plt.ylabel(ylabel)
  plt.xscale('log')
  ax.set_xlabel(ax.get_xlabel(), fontsize=14,color='k') #, weight='bold'
  ax.set_ylabel(ax.get_ylabel(), fontsize=14,color='k') #, weight='bold'
  for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(1.5)
    ax.spines[axis].set_color('0.2')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.tick_params(width=1.5)
  plt.tight_layout()
  plt.show()
  # plt.savefig(f'./plots/link_FR.pdf', transparent=True)

scatter_linkVSFR(all_FR, all_links)
#%%
################### scatter for each region
def scatter_linkVSFR_region(FR, links, regions):

  fig, axes = plt.subplots(2, 3, figsize=(15, 10))
  for i, j in [(f,s) for f in range(axes.shape[0]) for s in range(axes.shape[1])]:
    r_ind = i * axes.shape[1] + j
    ax = axes[i, j]
    X, Y = [], []
    for row in session2keep:
      for col in cols:
        X += FR[row][col][regions[r_ind]]
        Y += links[row][col][regions[r_ind]]
    ax.scatter(X, Y, facecolors='none', edgecolors='.1', alpha=.5)
    X, Y = (list(t) for t in zip(*sorted(zip(X, Y))))
    X, Y = np.array(X), np.array(Y)
    slope, intercept, r_value, p_value, std_err = stats.linregress(X,Y)
    line = slope*X+intercept
    locx, locy = .8, .8
    text = 'r={:.2f}, p={:.0e}'.format(r_value, p_value)
    ax.plot(X, line, color='.2', linestyle=(5,(10,3)), alpha=.5)
    ax.text(locx, locy, text, horizontalalignment='center',
      verticalalignment='center', transform=ax.transAxes, fontsize=16)
    # plt.legend(loc='lower right', fontsize=14, frameon=False)
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.set_xlabel('firing rate (Hz)')
    ax.set_title(regions[r_ind])
    ylabel = 'degree'
    ax.set_ylabel(ylabel)
    # ax.set_xscale('log')
    ax.set_xlabel(ax.get_xlabel(), fontsize=14,color='k') #, weight='bold'
    ax.set_ylabel(ax.get_ylabel(), fontsize=14,color='k') #, weight='bold'
    for axis in ['bottom', 'left']:
      ax.spines[axis].set_linewidth(1.5)
      ax.spines[axis].set_color('0.2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=1.5)
  plt.tight_layout()
  # plt.show()
  plt.savefig(f'./plots/degree_FR_region.pdf', transparent=True)

scatter_linkVSFR_region(FR, links, visual_regions)
#%%
################### scatter for each stimulus
def scatter_linkVSFR_stimulus(FR, links, regions, degree_type='intra'):
  rows, cols = get_rowcol(FR)
  fig, axes = plt.subplots(2, 4, figsize=(20, 10))
  for i, j in [(f,s) for f in range(axes.shape[0]) for s in range(axes.shape[1])]:
    s_ind = i * axes.shape[1] + j
    ax = axes[i, j]
    X, Y = [], []
    for r_ind, region in enumerate(regions):
      for row in session2keep:
        X += FR[row][cols[s_ind]][region]
        Y += links[row][cols[s_ind]][region]
      # ax.scatter(X, Y, facecolors='none', edgecolors=region_colors[r_ind], alpha=.5, label=region)

    ax.scatter(X, Y, facecolors='none', edgecolors='.2', alpha=.5, label=region)
    X, Y = (list(t) for t in zip(*sorted(zip(X, Y))))
    X, Y = np.array(X), np.array(Y)

    slope, intercept, r_value, p_value, std_err = stats.linregress(X,Y)
    line = slope*X+intercept
    locx, locy = .8, .8
    text = 'r={:.2f}, p={:.0e}'.format(r_value, p_value)
    ax.plot(X, line, alpha=.5, linewidth=4, color=stimulus_type_color[stimulus2stype(stimulus_names[s_ind])[0]]) #, linestyle=(5,(10,3))
    # # with confidence interval
    # lin = IntervalRegressor(LinearRegression())
    # lin.fit(X_train, y_train)
    # sorted_X = np.array(list(sorted(X[:,None])))
    # pred = lin.predict(sorted_X)
    # bootstrapped_pred = lin.predict_sorted(sorted_X)
    # min_pred = bootstrapped_pred[:, 0]
    # max_pred = bootstrapped_pred[:, bootstrapped_pred.shape[1]-1]
    # ax.plot(sorted_X, pred, alpha=.5, linewidth=4, color=stimulus_type_color[stimulus2stype(stimulus_names[s_ind])[0]]) #, linestyle=(5,(10,3))
    # ax.fill_between(sorted_X.flatten(), min_pred, max_pred, alpha=.2, color=stimulus_type_color[stimulus2stype(stimulus_names[s_ind])[0]])

    ax.text(locx, locy, text, horizontalalignment='center',
      verticalalignment='center', transform=ax.transAxes, fontsize=16)
    # plt.legend(loc='lower right', fontsize=14, frameon=False)
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.set_xlabel('firing rate (Hz)')
    ax.set_title(paper_label[s_ind].replace('\n', ' '), fontsize=16, color=stimulus_type_color[stimulus2stype(stimulus_names[s_ind])[0]])
    ylabel = 'degree'
    ax.set_ylabel(ylabel)
    # ax.set_xscale('log')
    ax.set_xlabel(ax.get_xlabel(), fontsize=14,color='k') #, weight='bold'
    ax.set_ylabel(ax.get_ylabel(), fontsize=14,color='k') #, weight='bold'
    for axis in ['bottom', 'left']:
      ax.spines[axis].set_linewidth(1.5)
      ax.spines[axis].set_color('0.2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=1.5)
  plt.suptitle(f'{degree_type} region', fontsize=25)
  plt.tight_layout(rect=[0, 0.03, 1, 0.98])
  # plt.show()
  plt.savefig(f'./plots/{degree_type}_degree_FR_stimulus.pdf', transparent=True)

scatter_linkVSFR_stimulus(FR, intra_links, visual_regions, degree_type='intra')
scatter_linkVSFR_stimulus(FR, inter_links, visual_regions, degree_type='inter')
#%%
################### scatter for each stimulus
def difference_intra_inter_r_stimulus_type(FR, intra_links, inter_links, regions, alpha):
  rows, cols = get_rowcol(FR)
  df = pd.DataFrame()
  # fig, axes = plt.subplots(1, 4, figsize=(20, 5))
  # for i in range(len(axes)):
  #   ax = axes[i]
  for s_type in stimulus_types:
    for r_ind, region in enumerate(regions):
      X, Y1, Y2 = [], [], []
      for row in session2keep:
        for col in cols:
          if stimulus2stype(col)[1] == s_type:
            X += FR[row][col][region]
            Y1 += intra_links[row][col][region]
            Y2 += inter_links[row][col][region]
      # ax.scatter(X, Y, facecolors='none', edgecolors='.2', alpha=.5, label=region)
      X, Y1, Y2 = (list(t) for t in zip(*sorted(zip(X, Y1, Y2))))
      X, Y1, Y2 = np.array(X), np.array(Y1), np.array(Y2)
      _, _, r1, _, _ = stats.linregress(X,Y1)
      _, _, r2, _, _ = stats.linregress(X,Y2)
      _, _, r3, _, _ = stats.linregress(Y1,Y2)
      if r1 > r2:
        rh, rl = r1, r2
      else:
        rh, rl = r2, r1
      # L, U = MA_method(rh, rl, r3, len(X), alpha)
      # sig = 'True' if L > 0 else 'False'
      # p = confidence_interval2pvalue(L, U, alpha)
      # l1, u1 = r_confidence_interval(r1, len(X), alpha)
      # l2, u2 = r_confidence_interval(r2, len(X), alpha)
      # p1 = confidence_interval2pvalue(l1, u1, alpha)
      # p2 = confidence_interval2pvalue(l2, u2, alpha)
      # intra_sig = 'True' if (l1 > 0) or (u1 < 0) else 'False'
      # inter_sig = 'True' if (l2 > 0) or (u2 < 0) else 'False'
      # df = pd.concat([df, pd.DataFrame([[s_type, region, r1, r2, sig, intra_sig, inter_sig, p, p1, p2]], columns=['stimulus type', 'region', 'intra r', 'inter r', f'{alpha} significance', f'{alpha} intra significance', f'{alpha} inter significance', 'p value', 'p value1', 'p value2'])], ignore_index=True)
      Ls, l1s, l2s, u1s, u2s = [], [], [], [], []
      alpha_list = [.0001, .001, .01, .05]
      for alpha in alpha_list:
        L, _ = MA_method(rh, rl, r3, len(X), alpha)
        l1, u1 = r_confidence_interval(r1, len(X), alpha)
        l2, u2 = r_confidence_interval(r2, len(X), alpha)
        Ls.append(L)
        l1s.append(l1)
        l2s.append(l2)
        u1s.append(u1)
        u2s.append(u2)
      Ls, l1s, l2s, u1s, u2s = np.array(Ls), np.array(l1s), np.array(l2s), np.array(u1s), np.array(u2s)
      loc = np.where(Ls > 0)[0]
      asterisk = '*' * (len(alpha_list) - loc[0]) if len(loc) else 'ns'
      locl1, locu1 = np.where(l1s > 0)[0], np.where(u1s < 0)[0]
      asterisk1 = '*' * (len(alpha_list) - (locl1 if len(locl1) else locu1)[0]) if len(locl1) or len(locu1) else 'ns'
      locl2, locu2 = np.where(l2s > 0)[0], np.where(u2s < 0)[0]
      asterisk2 = '*' * (len(alpha_list) - (locl2 if len(locl2) else locu2)[0]) if len(locl2) or len(locu2) else 'ns'
      df = pd.concat([df, pd.DataFrame([[s_type, region, r1, r2, asterisk, asterisk1, asterisk2]], columns=['stimulus type', 'region', 'intra r', 'inter r', 'significance', 'intra significance', 'inter significance'])], ignore_index=True)
        # intra_sig = 'True' if (l1 > 0) or (u1 < 0) else 'False'
        # inter_sig = 'True' if (l2 > 0) or (u2 < 0) else 'False'
        # df = pd.concat([df, pd.DataFrame([[s_type, region, r1, r2, sig, intra_sig, inter_sig, p, p1, p2]], columns=['stimulus type', 'region', 'intra r', 'inter r', f'{alpha} significance', f'{alpha} intra significance', f'{alpha} inter significance', 'p value', 'p value1', 'p value2'])], ignore_index=True)
  return df

df = difference_intra_inter_r_stimulus_type(FR, intra_links, inter_links, visual_regions, alpha=.01)
df
# df[(df['significance'].isin(['*', '**', '***', '****'])) & (df['intra significance'].isin(['*', '**', '***', '****']))]
#%%
def annot_significance(star, x1, x2, y, col='k', ax=None):
  ax = plt.gca() if ax is None else ax
  ax.text((x1+x2)*.5, y, star, ha='center', va='bottom', color=col, fontsize=14)

def annot_difference(star, x1, x2, y, h, col='k', ax=None):
  ax = plt.gca() if ax is None else ax
  ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
  ax.text((x1+x2)*.5, y+h, star, ha='center', va='bottom', color=col, fontsize=14)

def plot_intra_inter_r_bar_significance(df, regions):
  fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharey=True)
  palette = [[plt.cm.tab20b(i) for i in range(4)][i] for i in [0, 3]]
  for i, j in [(f,s) for f in range(axes.shape[0]) for s in range(axes.shape[1])]:
    s_ind = i * axes.shape[1] + j
    stimulus_type = stimulus_types[s_ind]
    ax = axes[i, j]
  # for s_ind, stimulus_type in enumerate(stimulus_types):
  #   ax = axes[s_ind]
    data = df[df['stimulus type']==stimulus_type]
    new_df = pd.DataFrame()
    for region in regions:
      new_df = pd.concat([new_df, pd.DataFrame([[region, 'intra-region'] + data[data['region']==region][['intra r', 'significance', 'intra significance']].values[0].tolist()], columns=['region', 'type', 'r', 'significance', 'own significance'])], ignore_index=True)
      new_df = pd.concat([new_df, pd.DataFrame([[region, 'inter-region'] + data[data['region']==region][['inter r', 'significance', 'inter significance']].values[0].tolist()], columns=['region', 'type', 'r', 'significance', 'own significance'])], ignore_index=True)
    new_df['r'] = pd.to_numeric(new_df['r'])
    bar_plot = sns.barplot(data=new_df, x='region', y='r', hue='type', palette=palette, ax=ax) #, facecolor='white'
    # for patch0, patch1 in zip(ax.patches[:6], ax.patches[6:]):
    #   patch0.set_edgecolor('.2')
    #   patch1.set_edgecolor('.8')
    handles, labels = ax.get_legend_handles_labels()
    if s_ind == 0:
      ax.legend(handles, labels, title='', bbox_to_anchor=(.7, 1.1), loc='upper left', fontsize=14, frameon=False)
    else:
      ax.legend().set_visible(False)
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    ax.set_xlabel('')
    ax.set_title(stimulus_type, fontsize=20, color=stimulus_type_color[s_ind])
    ylabel = r'$r$'
    ax.set_ylabel(ylabel, rotation=90)
    # ax.set_xscale('log')
    ax.set_xlabel(ax.get_xlabel(), fontsize=16,color='k') #, weight='bold'
    ax.set_ylabel(ax.get_ylabel(), fontsize=16, color='k') #, weight='bold'
    for axis in ['bottom', 'left']:
      ax.spines[axis].set_linewidth(1.5)
      ax.spines[axis].set_color('0.2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=1.5)
    for r_ind, region in enumerate(regions):
      intra_r, intra_star  = new_df[(new_df['region']==region) & (new_df['type']=='intra-region')][['r', 'own significance']].values[0]
      inter_r, inter_star = new_df[(new_df['region']==region) & (new_df['type']=='inter-region')][['r', 'own significance']].values[0]
      diff_star = new_df[(new_df['region']==region) & (new_df['type']=='intra-region')].significance.values[0]
      intra_r = intra_r + .01 if intra_r > 0 else intra_r - .01
      inter_r = inter_r + .01 if inter_r > 0 else inter_r - .08
      annot_significance(intra_star, -.45 + r_ind, 0.05 + r_ind, intra_r, ax=ax)
      annot_significance(inter_star, -.05 + r_ind, .45 + r_ind, inter_r, ax=ax)
      annot_difference(diff_star, -.2 + r_ind, .2 + r_ind, max(intra_r, inter_r) + .1, .03, ax=ax)
    liml, limu = ax.get_ylim()
    ax.set_ylim([liml - .02, limu + .05])
  # plt.show()
  plt.tight_layout()
  plt.savefig('./plots/intra_inter_r_bar_significance.pdf', transparent=True)

plot_intra_inter_r_bar_significance(df, visual_regions)
#%%
################### confidence level for two correlations
def r_confidence_interval(r, n, alpha):
  z = np.log((1 + r) / (1 - r)) / 2.0
  se = 1.0 / np.sqrt(n - 3)
  z_crit = stats.norm.ppf(1 - alpha/2)  # 2-tailed z critical value
  lo = z - z_crit * se
  hi = z + z_crit * se
  l = ((np.exp(2 * lo) - 1) / (np.exp(2 * lo) + 1))
  u = ((np.exp(2 * hi) - 1) / (np.exp(2 * hi) + 1))
  # Return a sequence
  return l, u

def confidence_interval2pvalue(l, u, alpha):
  se = (u - l) / (2 * stats.norm.ppf(1 - alpha/2))
  z = (l + u) / (2 * se)
  return np.exp(-.717 * z - .416 * z ** 2)

def MA_method(r1, r2, r3, n, alpha):
  l1, u1 = r_confidence_interval(r1, n, alpha)
  l2, u2 = r_confidence_interval(r2, n, alpha)
  cov = ((r3 - r1 * r2 / 2) * (1 - r1 ** 2 - r2 ** 2 - r3 ** 2) + r3 ** 3) / n
  var1, var2 = (1 - r1 ** 2) ** 2 / n, (1 - r2 ** 2) ** 2 / n
  corr = cov / np.sqrt(var1 * var2)
  L = r1 - r2 - np.sqrt((r1 - l1) ** 2 + (u2 - r2) ** 2 - 2 * corr * (r1 - l1) * (u2 - r2))
  U = r1 - r2 + np.sqrt((u1 - r1) ** 2 + (r2 - l2) ** 2 - 2 * corr * (u1 - r1) * (r2 - l2))
  return L, U

r1, r2, r3, n, alpha = .48, .08, .15, 1000, .05
MA_method(r1, r2, r3, n, alpha)
#%%
r_confidence_interval(.5, 10, .99)
#%%
def get_certain_region_links(G_dict, regions, active_area_dict, certain_regions=['VISp']):
  rows, cols = get_rowcol(G_dict)
  certain_links, other_links = {}, {}
  for row_ind, row in enumerate(rows):
    certain_links[row], other_links[row] = {}, {}
    # print(row)
    active_area = active_area_dict[row]
    for col_ind, col in enumerate(cols):
      certain_links[row][col], other_links[row][col] = {}, {}
      G = G_dict[row][col]
      # nodes = sorted(G.nodes())
      node_idx = sorted(active_area.keys())
      A = nx.to_numpy_array(G, nodelist=node_idx)
      A[A.nonzero()] = 1
      certain_nodes = [node for node in node_idx if active_area[node] in certain_regions]
      certain_indices = np.array([node_idx.index(i) for i in certain_nodes])
      if len(certain_indices):
        for region in regions:
          active_nodes = [node for node in node_idx if active_area[node]==region]
          region_indices = np.array([node_idx.index(i) for i in active_nodes])
          if len(region_indices):
            # certain_links[row][col][region] = certain_links[row][col].get(region, []) + (np.sum(A[:, region_indices], 0)).tolist() # out certain_links np.sum(A[region_indices, :], 1)
            # certain_links[row][col][region] = certain_links[row][col].get(region, []) + (np.sum(A[region_indices[:,None], region_indices], 1) + np.sum(A[region_indices[:,None], region_indices], 0)).tolist() # both out and in links, within region links
            certain_links[row][col][region] = certain_links[row][col].get(region, []) + (np.sum(A[region_indices[:,None], certain_indices], 1) + np.sum(A[certain_indices[:,None], region_indices], 0)).tolist() # both out and in links, within region links
            other_links[row][col][region] = other_links[row][col].get(region, []) + (np.sum(A[region_indices, :], 1) - np.sum(A[region_indices[:,None], certain_indices], 1) + np.sum(A[:, region_indices], 0) - np.sum(A[certain_indices[:,None], region_indices], 0)).tolist() # both out and in links, cross region links
  return certain_links, other_links

all_region_combs = []
for L in range(1, 4):
    for subset in itertools.combinations(visual_regions, L):
        if (set(visual_regions) - set(subset)) not in all_region_combs:
          all_region_combs.append(set(subset))
for certain_regions in all_region_combs:
  print(certain_regions)
  certain_links, other_links = get_certain_region_links(G_ccg_dict, visual_regions, active_area_dict, certain_regions=certain_regions)
  scatter_certain_linkVSFR_region(FR, certain_links, other_links, visual_regions, certain_regions=certain_regions)
#%%
################### scatter for each stimulus
def scatter_certain_linkVSFR_region(FR, certain_links, other_links, regions, certain_regions):
  rows, cols = get_rowcol(FR)
  fig, axes = plt.subplots(2, 3, figsize=(18, 10))
  for i, j in [(f,s) for f in range(axes.shape[0]) for s in range(axes.shape[1])]:
    r_ind = i * axes.shape[1] + j
    ax = axes[i, j]
    for col_ind, col in enumerate(cols):
      r1, r2 = [], []
      for row in session2keep:
        X = FR[row][col][regions[r_ind]]
        Y1 = certain_links[row][col][regions[r_ind]]
        Y2 = other_links[row][col][regions[r_ind]]
      # ax.scatter(X, Y, facecolors='none', edgecolors=region_colors[r_ind], alpha=.5, label=region)
        X, Y1, Y2 = (list(t) for t in zip(*sorted(zip(X, Y1, Y2))))
        X, Y1, Y2 = np.array(X), np.array(Y1), np.array(Y2)
        _, _, r_value1, _, _ = stats.linregress(X,Y1)
        _, _, r_value2, _, _ = stats.linregress(X,Y2)
        r1.append(r_value1)
        r2.append(r_value2)

      ax.scatter(r1, r2, facecolors='none', edgecolors=stimulus_type_color[stimulus2stype(col)[0]], alpha=.5, label=stimulus2stype(col)[1])
      # ax.plot(X, line, alpha=.5, linewidth=4, color=stimulus_type_color[stimulus2stype(stimulus_names[s_ind])[0]]) #, linestyle=(5,(10,3))
    
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    # now plot both limits against eachother
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.plot(lims, [0, 0], 'k--', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.set_xlabel('r of {}'.format("_".join(certain_regions)))
    ax.set_title(regions[r_ind], fontsize=16, color=region_colors[r_ind])
    ylabel = 'r of the rest'
    ax.set_ylabel(ylabel)
    # ax.set_xscale('log')
    ax.set_xlabel(ax.get_xlabel(), fontsize=14,color='k') #, weight='bold'
    ax.set_ylabel(ax.get_ylabel(), fontsize=14,color='k') #, weight='bold'
    for axis in ['bottom', 'left']:
      ax.spines[axis].set_linewidth(1.5)
      ax.spines[axis].set_color('0.2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=1.5)
  plt.tight_layout()
  # plt.show()
  plt.savefig('./plots/{}_links_FR_region.jpg'.format("_".join(certain_regions)))

# scatter_intra_inter_linkVSFR_region(FR, intra_links, inter_links, visual_regions)
# scatter_intra_inter_linkVSFR_region(FR, certain_links, other_links, visual_regions)
#%%
################### scatter for each region
def scatter_linkVSFR_stimulus_region(FR, links, regions, stimulus_name='natural_movie_three'):

  fig, axes = plt.subplots(2, 3, figsize=(15, 10))
  for i, j in [(f,s) for f in range(axes.shape[0]) for s in range(axes.shape[1])]:
    r_ind = i * axes.shape[1] + j
    ax = axes[i, j]
    locx = .8
    locys = np.arange(.1, .9, .1)
    for row in session2keep:
      X, Y = [], []
      X += FR[row][stimulus_name][regions[r_ind]]
      Y += links[row][stimulus_name][regions[r_ind]]
      X, Y = (list(t) for t in zip(*sorted(zip(X, Y))))
      X, Y = np.array(X), np.array(Y)

      slope, intercept, r_value, p_value, std_err = stats.linregress(X,Y)
      line = slope*X+intercept
      text = 'r={:.2f}, p={:.0e}'.format(r_value, p_value)
      ax.plot(X, line, linestyle=(5,(10,3)), alpha=.5)
      ax.text(locx, locys[session2keep.index(row)], text, horizontalalignment='center',
        verticalalignment='center', transform=ax.transAxes, fontsize=16)

      ax.scatter(X, Y, label=text, alpha=.5) # , label=row, facecolors='none', edgecolors='.2'
    
    # plt.legend(loc='lower right', fontsize=14, frameon=False)
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.set_xlabel('firing rate (Hz)')
    ax.set_title(regions[r_ind])
    ylabel = 'degree'
    ax.set_ylabel(ylabel)
    # ax.set_xscale('log')
    ax.set_xlabel(ax.get_xlabel(), fontsize=14,color='k') #, weight='bold'
    ax.set_ylabel(ax.get_ylabel(), fontsize=14,color='k') #, weight='bold'
    for axis in ['bottom', 'left']:
      ax.spines[axis].set_linewidth(1.5)
      ax.spines[axis].set_color('0.2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=1.5)
  plt.suptitle(stimulus_name, fontsize=25)
  plt.tight_layout()
  plt.show()
  # plt.savefig(f'./plots/degree_FR_stimulus.pdf', transparent=True)

stimulus_types = ['resting state', 'flashes', 'gratings', 'natural stimuli']
for stimulus_name in stimulus_names:
  scatter_linkVSFR_stimulus_region(FR, links, visual_regions, stimulus_name=stimulus_name)
#%%
def scatter_linkVSFR_stimulus_type_region(FR, links, regions, stimulus_type='natural stimuli'):

  fig, axes = plt.subplots(2, 3, figsize=(15, 10))
  for i, j in [(f,s) for f in range(axes.shape[0]) for s in range(axes.shape[1])]:
    r_ind = i * axes.shape[1] + j
    ax = axes[i, j]
    X, Y = [], []
    for s_ind, stimulus in enumerate(stimulus_names):
      # if stimulus == 'natural_scenes':
      if stimulus2stype(stimulus)[1] == stimulus_type:
        for row in session2keep:
          X += FR[row][stimulus][regions[r_ind]]
          Y += links[row][stimulus][regions[r_ind]]

    ax.scatter(X, Y, facecolors='none', edgecolors='.2', alpha=.5)
    X, Y = (list(t) for t in zip(*sorted(zip(X, Y))))
    X, Y = np.array(X), np.array(Y)

    slope, intercept, r_value, p_value, std_err = stats.linregress(X,Y)
    line = slope*X+intercept
    locx, locy = .8, .8
    text = 'r={:.2f}, p={:.0e}'.format(r_value, p_value)
    ax.plot(X, line, color='.2', linestyle=(5,(10,3)), alpha=.5)
    ax.text(locx, locy, text, horizontalalignment='center',
      verticalalignment='center', transform=ax.transAxes, fontsize=16)
    # plt.legend(loc='lower right', fontsize=14, frameon=False)
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.set_xlabel('firing rate (Hz)')
    ax.set_title(regions[r_ind])
    ylabel = 'degree'
    ax.set_ylabel(ylabel)
    # ax.set_xscale('log')
    ax.set_xlabel(ax.get_xlabel(), fontsize=14,color='k') #, weight='bold'
    ax.set_ylabel(ax.get_ylabel(), fontsize=14,color='k') #, weight='bold'
    for axis in ['bottom', 'left']:
      ax.spines[axis].set_linewidth(1.5)
      ax.spines[axis].set_color('0.2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=1.5)
  plt.suptitle(stimulus_type, fontsize=25)
  plt.tight_layout()
  plt.show()
  # plt.savefig(f'./plots/degree_FR_stimulus.pdf', transparent=True)

stimulus_types = ['resting state', 'flashes', 'gratings', 'natural stimuli']
for stimulus_type in stimulus_types:
  scatter_linkVSFR_stimulus_type_region(FR, links, visual_regions, stimulus_type=stimulus_type)
#%%
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

def plot_metrics(G_dict):
  rows, cols = get_rowcol(G_dict)
  metric_names = ['clustering', 'overall_reciprocity', 'flow_hierarchy', 'global_reaching_centrality']
  # metric_names = ['wiener_index']
  plots_shape = (2, 2)
  metric = np.empty((len(rows), len(cols), len(metric_names)))
  metric[:] = np.nan
  # fig = plt.figure(figsize=(5*plots_shape[1], 8))
  # fig = plt.figure(figsize=(20, 10))
  df = pd.DataFrame()
  for metric_ind, metric_name in enumerate(metric_names):
    print(metric_name)
    for col_ind, col in enumerate(cols):
      print(col)
      for row_ind, row in enumerate(rows):
        G = G_dict[row][col]
        m = calculate_directed_metric(G, metric_name)
        metric[row_ind, col_ind, metric_ind] = m
      df = pd.concat([df, pd.DataFrame(np.concatenate((metric[:,col_ind,metric_ind][:,None], np.array([col] * len(rows))[:,None], np.array([metric_name] * len(rows))[:,None]), 1), columns=['metric', 'stimulus', 'metric name'])], ignore_index=True)
  #   plt.subplot(*plots_shape, metric_ind + 1)
  #   for row_ind, row in enumerate(rows):
  #     plt.plot(cols, metric[row_ind, :, metric_ind], label=row, alpha=1)
  #   plt.gca().set_title(metric_name, fontsize=30, rotation=0)
  #   plt.xticks(rotation=90)
  # plt.legend()
  # plt.tight_layout()
  # figname = './plots/metric_stimulus_individual_weighted_{}_{}_{}_fold.jpg'.format(sign, measure, n) if weight else './plots/metric_stimulus_individual_{}_{}_{}_fold.jpg'.format(sign, measure, n)
  # plt.savefig(figname)
  df.metric = pd.to_numeric(df.metric)
  fig = plt.figure(figsize=(14, 10))
  for metric_ind, metric_name in enumerate(metric_names):
    plt.subplot(2,2, metric_ind + 1)
    ax = sns.boxplot(x="stimulus", y='metric', color='white', hue_order=metric_names, data=df[df['metric name']==metric_name], showfliers=True) # , boxprops=dict(alpha=.6)
    plt.xticks(rotation=90)
    plt.title(metric_name)
    plt.xlabel(None)
    if metric_ind < 2:
      plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)
  plt.savefig('./plots/metrics.jpg')
  return df

metric_names = ['clustering', 'overall_reciprocity', 'flow_hierarchy', 'global_reaching_centrality']
df = plot_metrics(G_ccg_dict)
#%%
def chord_diagram_region_connection(G_dict, area_dict, regions):
  hv.extension('matplotlib')
  hv.output(size=200)
  region_connection = np.zeros((len(rows), len(cols), len(regions), len(regions)))
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      G = G_dict[row][col] if col in G_dict[row] else nx.Graph()
      nodes = sorted(G.nodes())
      active_areas = np.unique(list({key: area_dict[row][key] for key in nodes}.values()))
      # active_areas = [i for i in regions if i in active_areas]
      A = nx.to_numpy_array(G, nodelist=nodes,weight='weight')
      A[A.nonzero()] = 1
      for region_ind_i, region_i in enumerate(regions):
        if region_i in active_areas:
          for region_ind_j, region_j in enumerate(regions):
            if region_j in active_areas:
              region_indices_i = np.array([k for k, v in area_dict[row].items() if v==region_i])
              region_indices_j = np.array([k for k, v in area_dict[row].items() if v==region_j])
              region_indices_i = np.array([nodes.index(i) for i in list(set(region_indices_i) & set(nodes))]) # some nodes not in cc are removed 
              region_indices_j = np.array([nodes.index(i) for i in list(set(region_indices_j) & set(nodes))])
              region_connection[row_ind, col_ind, region_ind_i, region_ind_j] = np.sum(A[region_indices_i[:, None], region_indices_j])
              assert np.sum(A[region_indices_i[:, None], region_indices_j]) == len(A[region_indices_i[:, None], region_indices_j].nonzero()[0])
      region_connection[row_ind, col_ind, :, :] = region_connection[row_ind, col_ind, :, :]
  mean_region_connection = region_connection.mean(0)
  hv_li = []
  for col_ind in range(len(cols)):
    print(col_ind)
    # row_ind, col_ind = 7, 3
    # A = nx.to_numpy_array(G_lcc, nodelist=sorted(G_lcc.nodes()),weight='weight')
    # links = pd.DataFrame([(i, j, region_connection[row_ind, col_ind,i,j]) for i,j in zip(*region_connection[row_ind, col_ind].nonzero())], columns=['source', 'target', 'value'])
    links = pd.DataFrame([(i, j, mean_region_connection[col_ind,i,j]) for i,j in zip(*mean_region_connection[col_ind].nonzero())], columns=['source', 'target', 'value'])
    links.value = links.value.abs()
    # nodes = pd.DataFrame([[node, area_dict[row][node]] for node in sorted(G.nodes())], columns=['index', 'region'])
    nodes = hv.Dataset(pd.DataFrame([[i, regions[i]] for i in range(len(regions))], columns=['index', 'region']), 'index')
    # links = pd.DataFrame(data['links'])
    # nodes = hv.Dataset(pd.DataFrame(data['nodes']).sort_values('group'), 'index')
    # nodes.data.head()
    chord = hv.Chord((links, nodes), group=cols[col_ind])
    ch = chord.opts(cmap=region_colors, edge_cmap=region_colors, edge_color=dim('source').str(), 
                  labels='region', node_color=dim('index').str()) # , hooks=[rotate_label]
    hv_li.append(ch)
  hv.save(hv.Layout(hv_li), f'./plots/chord_diagram.pdf', fmt='pdf')

chord_diagram_region_connection(G_ccg_dict, area_dict, visual_regions)
#%%
#%%
################# get optimal resolution that maximizes delta H
rows, cols = get_rowcol(G_ccg_dict)
with open('comms_dict.pkl', 'rb') as f:
  comms_dict = pickle.load(f)
with open('metrics.pkl', 'rb') as f:
  metrics = pickle.load(f)
resolution_list = np.arange(0, 2.1, 0.1)
max_reso_gnm, max_reso_config = get_max_dH_resolution(rows, cols, resolution_list, metrics)
################# community with Hamiltonian
max_pos_reso_gnm = get_max_pos_reso(G_ccg_dict, max_reso_gnm)
max_pos_reso_config = get_max_pos_reso(G_ccg_dict, max_reso_config)
#%%
def plot_example_graphs_largest_comms(G_dict, hamiltonian, comms, area_dict, active_area_dict, regions, row, col, measure, n, cc=False):
  np.random.seed(1111)
  com = CommunityLayout(comm_scale=50, node_scale=6, k=.2)
  G_sample = G_dict[row][col]
  dire = True if nx.is_directed(G_sample) else False
  fig = plt.figure(figsize=(8, 7))
  print(row, col)
  G = G_dict[row][col].copy() if col in G_dict[row] else nx.DiGraph()
  nx.set_node_attributes(G, area_dict[row], "area")
  if cc:
    G = get_lcc(G)
  node_idx = sorted(active_area_dict[row].keys())
  reverse_mapping = {node_idx[i]:i for i in range(len(node_idx))}
  G = nx.relabel_nodes(G, reverse_mapping)
  comms = [[reverse_mapping[n] for n in comm] for comm in comms]
  comms_tuple = [[[i for i in comm], len(comm)] for comm in comms]
  ordered_comms = [e[0] for e in sorted(comms_tuple, key=lambda x:x[1], reverse=True)]
  comms2plot = ordered_comms[:6]
  nodes2plot = [j for i in comms2plot for j in i]
  # nodes2plot = [j for i in comms if len(i) > 2 for j in i]
  # comms2plot = [i for i in comms if len(i) > 2]
  pos = com.get_community_layout(G.subgraph(nodes2plot), comm2partition(comms2plot))
  # partition = community.best_partition(G, weight='weight')
  # pos = com.get_community_layout(G, partition)
  # metric = community.modularity(partition, G, weight='weight')
  print('Hamiltonian: {}'.format(hamiltonian))
  edges = nx.edges(G.subgraph(nodes2plot))
  degrees = dict(G.degree(nodes2plot))
  # use offset as edge weight (color)
  weights = [w for i,j,w in nx.edges(G.subgraph(nodes2plot)).data('weight')]
  # weights = [offset_mat[edge[0], edge[1]] for edge in edges]
  norm = mpl.colors.Normalize(vmin=-1, vmax=1)
  m= cm.ScalarMappable(norm=norm, cmap=cm.RdBu_r)
  edge_colors = [m.to_rgba(1) if w > 0 else m.to_rgba(-1) for w in weights]
  # edge_colors = [m.to_rgba(w) for w in weights]
  areas = [G.nodes[n]['area'] for n in nodes2plot]
  colors = [region_colors[regions.index(area)] for area in areas]
  # pos = nx.spring_layout(G, k=0.8, iterations=50) # make nodes as seperate as possible
  nx.draw_networkx_edges(G, pos, arrows=dire, edgelist=edges, edge_color=edge_colors, width=3.0, alpha=0.2) # , edge_cmap=plt.cm.Greens
  # nx.draw_networkx_nodes(G, pos, nodelist=degrees.keys(), node_size=[np.log(v + 2) * 20 for v in degrees.values()], 
  # print(len(set(nodes2plot)), len(nodes2plot), len(areas), len(degrees), len(colors))
  nx.draw_networkx_nodes(G, pos, nodelist=degrees.keys(), node_size=[15 * v for v in degrees.values()], 
  node_color=colors, alpha=0.6)
  for index, a in enumerate(regions):
    plt.scatter([],[], c=region_colors[index], label=a, s=30, alpha=0.6)
  legend = plt.legend(fontsize=15) # loc='lower right', 
  for handle in legend.legendHandles:
    handle.set_sizes([60.0])
  plt.tight_layout()
  image_name = './plots/example_graphs_top_comms_color_cc_{}_{}_{}_{}fold.pdf' if cc else './plots/example_graphs_top_comms_color_{}_{}_{}_{}fold.pdf'
  plt.savefig(image_name.format(row, col, measure, n), transparent=True)
#################### example topology of graphs, select the top 6 communities
rows, cols = get_rowcol(G_ccg_dict)
# for row_ind, row in enumerate(rows):
#   for col_ind, col in enumerate(cols):
#     print(row, col)
#     break
row_ind, col_ind = 3, 7
row, col = rows[row_ind], cols[col_ind]
max_reso = max_reso_gnm[row_ind][col_ind]
hamiltonian = metrics['Hamiltonian'][row_ind, col_ind, np.where(resolution_list==max_reso)[0][0], 0]
comms_list = comms_dict[row][col][max_reso]
plot_example_graphs_largest_comms(G_ccg_dict, hamiltonian, comms_list[2], area_dict, active_area_dict, visual_regions, row, col, measure, n, False)
#%%
def plot_weighted_Hcomm_purity(G_dict, area_dict, measure, n, max_pos_reso=None, max_neg_reso=None, max_method='none'):
  rows, cols = get_rowcol(G_dict)
  if max_pos_reso is None:
    max_pos_reso = np.ones((len(rows), len(cols)))
  if max_neg_reso is None:
    max_neg_reso = np.ones((len(rows), len(cols)))
  fig, ax = plt.subplots(figsize=(6, 4))
  weighted_purity = []
  for col_ind, col in enumerate(cols):
    print(col)
    w_purity_col = []
    for row_ind, row in enumerate(rows):
      data = {}
      G = G_dict[row][col].copy() if col in G_dict[row] else nx.DiGraph()
      comms = signed_louvain_communities(G, weight='weight', pos_resolution=max_pos_reso[row_ind, col_ind], neg_resolution=max_neg_reso[row_ind, col_ind])
      sizes = [len(comm) for comm in comms]
      for comm, size in zip(comms, sizes):
        c_regions = [area_dict[row][node] for node in comm]
        _, counts = np.unique(c_regions, return_counts=True)
        assert len(c_regions) == size == counts.sum()
        purity = counts.max() / size
        if size in data:
          data[size].append(purity)
        else:
          data[size] = [purity]
      c_size, c_purity = [k for k,v in data.items() if k >= 4], [v for k,v in data.items() if k >= 4]
      c_size = np.array(c_size) / sum(c_size)
      w_purity_col.append(sum([cs * np.mean(cp) for cs, cp in zip(c_size, c_purity)]))
    weighted_purity.append(w_purity_col)
  boxprops = dict(facecolor='white', edgecolor='.2')
  medianprops = dict(linestyle='-', linewidth=1.5, color='k')
  ax.boxplot(weighted_purity, showfliers=False, patch_artist=True, boxprops=boxprops, medianprops=medianprops)
  plt.xticks(list(range(1, len(weighted_purity)+1)), paper_label)
  ax.xaxis.set_tick_params(labelsize=14)
  ax.yaxis.set_tick_params(labelsize=14)
  ylabel = 'weighted purity'
  ax.set_ylabel(ylabel)
  # ax.set_xscale('log')
  ax.set_xlabel(ax.get_xlabel(), fontsize=14,color='k') #, weight='bold'
  ax.set_ylabel(ax.get_ylabel(), fontsize=14,color='k') #, weight='bold'
  for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(1.5)
    ax.spines[axis].set_color('0.2')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.tick_params(width=1.5)
  plt.tight_layout()
  image_name = './plots/weighted_Hcomm_purity_{}.pdf'.format(max_method)
  # plt.show()
  plt.savefig(image_name)
  return weighted_purity

weighted_purity_config = plot_weighted_Hcomm_purity(G_ccg_dict, area_dict, measure, n, max_pos_reso=max_pos_reso_config, max_neg_reso=max_reso_config, max_method='config')
# %%
def plot_2Ddist_Hcommsize(G_dict, comms_dict, area_dict, measure, n, max_neg_reso=None, max_method='none', kind='scatter'):
  rows, cols = get_rowcol(comms_dict)
  if max_neg_reso is None:
    max_neg_reso = np.ones((len(rows), len(cols)))
  fig, ax = plt.subplots(figsize=(3, 3))
  df = pd.DataFrame()
  for col_ind, col in enumerate(cols):
    print(col)
    size_col = []
    purity_col = []
    for row_ind, row in enumerate(rows):
      max_reso = max_neg_reso[row_ind][col_ind]
      comms_list = comms_dict[row][col][max_reso]
      G = G_dict[row][col]
      all_regions = [area_dict[row][node] for node in G.nodes()]
      _, counts = np.unique(all_regions, return_counts=True)
      lr_size = counts.max()
      for comms in comms_list: # 100 repeats
        sizes = [len(comm) for comm in comms]
        data = []
        for comm, size in zip(comms, sizes):
          c_regions = [area_dict[row][node] for node in comm]
          _, counts = np.unique(c_regions, return_counts=True)
          assert len(c_regions) == size == counts.sum()
          # purity = counts.max() / size
          purity = counts.max() / min(lr_size, size)
          data.append((size, purity))
        size_col += [s/lr_size for s,p in data if s>=4]
        purity_col += [p for s,p in data if s>=4]
    df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(size_col)[:,None], np.array(purity_col)[:,None], np.array([stimulus2stype(col)[1]] * len(size_col))[:,None]), 1), columns=['community size', 'purity', 'stimulus type'])], ignore_index=True)
  df['community size'] = pd.to_numeric(df['community size'])
  df['purity'] = pd.to_numeric(df['purity'])
  return df
  
  palette = {st:sc for st, sc in zip(stimulus_types, stimulus_type_color)}
  kws = {"facecolor": "0", "linewidth": 1.5}
  if kind == 'scatter':
    plot = sns.jointplot(data=df, x='community size', y='purity', hue='stimulus type', kind='scatter', edgecolor=df["stimulus type"].map(palette), alpha=0.7, **kws)
  elif kind == 'kde':
    plot = sns.jointplot(data=df, x='community size', y='purity', hue='stimulus type', kind='kde', ylim=(0.13, 1.0), log_scale=[True, False], alpha=0.7)
  plot.ax_marg_x.set_xlim(-5, 250)
  plot.ax_marg_y.set_ylim(0.1, 1.05)
  ax.xaxis.set_tick_params(labelsize=14)
  ax.yaxis.set_tick_params(labelsize=14)
  ylabel = 'purity'
  ax.set_ylabel(ylabel)
  # ax.set_xscale('log')
  ax.set_xlabel(ax.get_xlabel(), fontsize=14,color='k') #, weight='bold'
  ax.set_ylabel(ax.get_ylabel(), fontsize=14,color='k') #, weight='bold'
  for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(1.5)
    ax.spines[axis].set_color('0.2')
  # ax.spines['top'].set_visible(False)
  # ax.spines['right'].set_visible(False)
  ax.tick_params(width=1.5)

  plt.suptitle('{} 2D distribution of community size'.format(max_method), size=18)
  plt.tight_layout()
  image_name = './plots/dist2D_Hcomm_size_{}_{}.pdf'.format(kind, max_method)
  # plt.show()
  plt.savefig(image_name)

df = plot_2Ddist_Hcommsize(G_ccg_dict, comms_dict, area_dict, measure, n, max_neg_reso=max_reso_config, max_method='config', kind='scatter')
# %%
# fig, ax = plt.subplots(figsize=(3, 3))
palette = {st:sc for st, sc in zip(stimulus_types, stimulus_type_color)}
joint_kws = {"linewidth": 1., 'fc':'none',
      'edgecolor':df["stimulus type"].map(palette),
      'alpha':0.3,
      'height':10}
plot = sns.jointplot(data=df, x='community size', y='purity', hue='stimulus type', kind='scatter', legend = False, **joint_kws)

scatter_kws = {"fc": "none", "linewidth": 1.}
handles, labels = zip(*[
    (plt.scatter([], [], ec=color, **scatter_kws), key) for key, color in palette.items()
])
# ax = plt.gca()
# ax.legend(handles, labels, title="", loc='upper left', frameon=False)
# plot.ax_marg_x.set_xlim(-5, 250)
plot.ax_marg_y.set_ylim(0.1, 1.05)

plt.xlabel('relative community size', fontsize=18)
plt.ylabel('purity', fontsize=18)
plt.tick_params(axis="both", labelsize=18)
plt.legend(handles, labels, title="", loc='upper left', frameon=False, fontsize=20)

# ax.xaxis.set_tick_params(labelsize=14)
# ax.yaxis.set_tick_params(labelsize=14)
# ylabel = 'purity'
# ax.set_ylabel(ylabel)
# # ax.set_xscale('log')
# ax.set_xlabel(ax.get_xlabel(), fontsize=20,color='k') #, weight='bold'
# ax.set_ylabel(ax.get_ylabel(), fontsize=20,color='k') #, weight='bold'
# for axis in ['bottom', 'left']:
#   ax.spines[axis].set_linewidth(1.5)
#   ax.spines[axis].set_color('0.2')
# # ax.spines['top'].set_visible(False)
# # ax.spines['right'].set_visible(False)
# ax.tick_params(width=1.5)

# plt.suptitle('{} 2D distribution of community size'.format(max_method), size=18)
plt.tight_layout()
# image_name = './plots/dist2D_Hcomm_size_{}_{}.pdf'.format(kind, max_method)
plt.show()
# plt.savefig(image_name)
# %%
def rand_index_community_region_allnodes(comms_dict, area_dict, max_neg_reso=None):
  fig, ax = plt.subplots(figsize=(6, 4))
  rows, cols = get_rowcol(comms_dict)
  if max_neg_reso is None:
    max_neg_reso = np.ones((len(rows), len(cols)))
  all_ri_list = []
  for col_ind, col in enumerate(cols):
    print(col)
    ri_list = []
    for row_ind, row in enumerate(rows):
      # print(row)
      G = G_ccg_dict[row][col]
      node_area = area_dict[row]
      max_reso = max_neg_reso[row_ind][col_ind]
      comms_list = comms_dict[row][col][max_reso]
      regions = [node_area[node] for node in sorted(G.nodes())]   
      for comms in comms_list: # 100 repeats
        assert len(regions) == len(comm2label(comms)), '{}, {}'.format(len(regions), len(comm2label(comms)))
        # print(len(regions), len(comm2label(comms)))
        ri_list.append(adjusted_rand_score(regions, comm2label(comms)))
    all_ri_list.append(ri_list)
  boxprops = dict(facecolor='white', edgecolor='.2')
  medianprops = dict(linestyle='-', linewidth=1.5, color='k')
  ax.boxplot(all_ri_list, showfliers=False, patch_artist=True, boxprops=boxprops, medianprops=medianprops)
  plt.xticks(list(range(1, len(all_ri_list)+1)), paper_label)
  ax.xaxis.set_tick_params(labelsize=14)
  ax.yaxis.set_tick_params(labelsize=14)
  ylabel = 'Rand index'
  ax.set_ylabel(ylabel)
  # ax.set_xscale('log')
  ax.set_xlabel(ax.get_xlabel(), fontsize=14,color='k') #, weight='bold'
  ax.set_ylabel(ax.get_ylabel(), fontsize=14,color='k') #, weight='bold'
  for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(1.5)
    ax.spines[axis].set_color('0.2')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.tick_params(width=1.5)
  plt.tight_layout()
  plt.savefig('./plots/rand_index_allnodes.pdf', transparent=True)
  # plt.show()
      
rand_index_community_region_allnodes(comms_dict, area_dict, max_neg_reso=max_reso_config)
# %%
def rand_index_community_region_activenodes(comms_dict, area_dict, max_neg_reso=None):
  fig, ax = plt.subplots(figsize=(6, 4))
  rows, cols = get_rowcol(comms_dict)
  if max_neg_reso is None:
    max_neg_reso = np.ones((len(rows), len(cols)))
  all_ri_list = []
  for col_ind, col in enumerate(cols):
    print(col)
    ri_list = []
    for row_ind, row in enumerate(rows):
      # print(row)
      G = G_ccg_dict[row][col]
      nodes = sorted(G.nodes())
      active_nodes = [node for node in nodes if G.degree(node) > 0]
      active_node_indx = [nodes.index(node) for node in nodes if G.degree(node) > 0]
      node_area = area_dict[row]
      max_reso = max_neg_reso[row_ind][col_ind]
      comms_list = comms_dict[row][col][max_reso]
      regions = [node_area[node] for node in active_nodes]   
      for comms in comms_list: # 100 repeats
        comms_par = comm2label(comms)
        comms_par = [comms_par[i] for i in active_node_indx]
        assert len(regions) == len(comms_par), '{}, {}'.format(len(regions), len(comms_par))
        # print(len(regions), len(comm2label(comms)))
        ri_list.append(adjusted_rand_score(regions, comms_par))
    all_ri_list.append(ri_list)
  boxprops = dict(facecolor='white', edgecolor='.2')
  medianprops = dict(linestyle='-', linewidth=1.5, color='k')
  ax.boxplot(all_ri_list, showfliers=False, patch_artist=True, boxprops=boxprops, medianprops=medianprops)
  plt.xticks(list(range(1, len(all_ri_list)+1)), paper_label)
  ax.xaxis.set_tick_params(labelsize=14)
  ax.yaxis.set_tick_params(labelsize=14)
  ylabel = 'Rand index'
  ax.set_ylabel(ylabel)
  # ax.set_xscale('log')
  ax.set_xlabel(ax.get_xlabel(), fontsize=14,color='k') #, weight='bold'
  ax.set_ylabel(ax.get_ylabel(), fontsize=14,color='k') #, weight='bold'
  for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(1.5)
    ax.spines[axis].set_color('0.2')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.tick_params(width=1.5)
  plt.tight_layout()
  plt.savefig('./plots/rand_index_activenodes.pdf', transparent=True)
  # plt.show()
      
rand_index_community_region_activenodes(comms_dict, area_dict, max_neg_reso=max_reso_config)
# %%
def rand_index_community_region_lscc(comms_dict, area_dict, max_neg_reso=None):
  fig, ax = plt.subplots(figsize=(6, 4))
  rows, cols = get_rowcol(comms_dict)
  if max_neg_reso is None:
    max_neg_reso = np.ones((len(rows), len(cols)))
  all_ri_list = []
  for col_ind, col in enumerate(cols):
    print(col)
    ri_list = []
    for row_ind, row in enumerate(rows):
      # print(row)
      G = G_ccg_dict[row][col]
      Gcc = sorted(nx.strongly_connected_components(G), key=len, reverse=True)
      lscc = G.subgraph(Gcc[0])
      nodes = sorted(lscc.nodes())
      lscc_node_indx = [sorted(G.nodes()).index(node) for node in nodes]
      node_area = area_dict[row]
      max_reso = max_neg_reso[row_ind][col_ind]
      comms_list = comms_dict[row][col][max_reso]
      regions = [node_area[node] for node in nodes]   
      for comms in comms_list: # 100 repeats
        comms_par = comm2label(comms)
        comms_par = [comms_par[i] for i in lscc_node_indx]
        assert len(regions) == len(comms_par), '{}, {}'.format(len(regions), len(comms_par))
        # print(len(regions), len(comm2label(comms)))
        ri_list.append(adjusted_rand_score(regions, comms_par))
    all_ri_list.append(ri_list)
  boxprops = dict(facecolor='white', edgecolor='.2')
  medianprops = dict(linestyle='-', linewidth=1.5, color='k')
  ax.boxplot(all_ri_list, showfliers=False, patch_artist=True, boxprops=boxprops, medianprops=medianprops)
  plt.xticks(list(range(1, len(all_ri_list)+1)), paper_label)
  ax.xaxis.set_tick_params(labelsize=14)
  ax.yaxis.set_tick_params(labelsize=14)
  ylabel = 'Rand index'
  ax.set_ylabel(ylabel)
  # ax.set_xscale('log')
  ax.set_xlabel(ax.get_xlabel(), fontsize=14,color='k') #, weight='bold'
  ax.set_ylabel(ax.get_ylabel(), fontsize=14,color='k') #, weight='bold'
  for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(1.5)
    ax.spines[axis].set_color('0.2')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.tick_params(width=1.5)
  plt.tight_layout()
  plt.savefig('./plots/rand_index_lscc.pdf', transparent=True)
  # plt.show()
      
rand_index_community_region_lscc(comms_dict, area_dict, max_neg_reso=max_reso_config)
# %%
################# weighted average of community metrics
def occupancy_community_perregion(comms_dict, area_dict, regions, max_neg_reso=None):
  rows, cols = get_rowcol(comms_dict)
  if max_neg_reso is None:
    max_neg_reso = np.ones((len(rows), len(cols)))
  df = pd.DataFrame()
  for col_ind, col in enumerate(cols):
    print(col)
    for row_ind, row in enumerate(session2keep):
      G = G_ccg_dict[row][col]
      nodes = sorted(G.nodes())
      node_area = area_dict[row]
      max_reso = max_neg_reso[row_ind][col_ind]
      comms_list = comms_dict[row][col][max_reso]
      for region in regions:
        region_nodes = [node for node in nodes if node_area[node]==region]
        num_comm_list, comm_size_list, occupancy_list, leadership_list, purity_list, entropy_list = [], [], [], [], [], []
        for comms in comms_list: # 50 repeats
          purities, comm_size, occupancy, num_regions, region_distri = [], [], [], [], []
          for comm in comms:
            c_regions = [node_area[node] for node in comm]
            _, counts = np.unique(c_regions, return_counts=True)
            assert len(c_regions) == counts.sum()
            purities.append(counts.max() / counts.sum())
            comm_size.append(len(comm))
            occupancy.append(len(comm & set(region_nodes)) / len(comm))
            num_regions.append(len(counts))
            region_distri.append(len(comm & set(region_nodes)))
          # comm_size = [len(comm) for comm in comms]
          # occupancy = [len(comm & set(region_nodes)) / len(comm) for comm in comms]
          large_comm_indx = [i for i in range(len(comm_size)) if comm_size[i] >= 4]
          region_comm_indx = [i for i in large_comm_indx if (occupancy[i] > 0)] # large communities
          # assert len(region_comm_indx) > 0, (row, col, region)
          if len(region_comm_indx):
            # num_comm_list.append(len(region_comm_indx)) # relative num of comms
            num_comm_list.append(len(region_comm_indx) / len(large_comm_indx)) # relative num of comms
            comm_size_list.append(np.mean([comm_size[i] for i in region_comm_indx]))
            # max_comm_size = max(comm_size)
            # comm_size_list.append(np.mean([comm_size[i] for i in region_comm_indx]) / max_comm_size) # relative comm size
            total_comm_size = np.sum([comm_size[i] for i in region_comm_indx])
            occupancy_list.append(np.sum([occupancy[i] * comm_size[i] / total_comm_size for i in region_comm_indx])) # weighted
            purity_list.append(np.sum([purities[i] * comm_size[i] / total_comm_size for i in region_comm_indx])) # weighted
            # leadership_list.append(np.sum([num_regions[i] * occupancy[i] * comm_size[i] / total_comm_size for i in region_comm_indx]))
            # leadership_list.append(purity_list[-1]-occupancy_list[-1])
            leadership_list.append(safe_division(occupancy_list[-1], purity_list[-1]))
            region_distri = [region_distri[i] for i in large_comm_indx if region_distri[i] > 0]
            entropy_list.append(safe_division(entropy(region_distri, base=2), entropy([1] * len(region_distri), base=2))) # normalized entropy
        if len(num_comm_list):
          df = pd.concat([df, pd.DataFrame(np.concatenate((np.array([col] * len(num_comm_list))[:,None], np.array([region] * len(num_comm_list))[:,None], np.array(num_comm_list)[:,None], np.array(comm_size_list)[:,None], np.array(occupancy_list)[:,None], np.array(leadership_list)[:,None], np.array(purity_list)[:,None], np.array(entropy_list)[:,None]), 1), columns=['stimulus', 'region', 'number of communities', 'community size', 'occupancy', 'leadership', 'purity', 'entropy'])], ignore_index=True)

        # df = pd.concat([df, pd.DataFrame(np.concatenate((np.array([col] * len(num_comm_list))[:,None], np.array([region] * len(num_comm_list))[:,None], np.array(['number of communities'] * len(num_comm_list))[:,None], np.array(num_comm_list)[:,None]), 1), columns=['stimulus', 'region', 'type', 'metric'])], ignore_index=True)
        # df = pd.concat([df, pd.DataFrame(np.concatenate((np.array([col] * len(comm_size_list))[:,None], np.array([region] * len(comm_size_list))[:,None], np.array(['community size'] * len(comm_size_list))[:,None], np.array(comm_size_list)[:,None]), 1), columns=['stimulus', 'region', 'type', 'metric'])], ignore_index=True)
        # df = pd.concat([df, pd.DataFrame(np.concatenate((np.array([col] * len(comm_size_list))[:,None], np.array([region] * len(comm_size_list))[:,None], np.array(['occupancy'] * len(comm_size_list))[:,None], np.array(occupancy_list)[:,None]), 1), columns=['stimulus', 'region', 'type', 'metric'])], ignore_index=True)
        # df = pd.concat([df, pd.DataFrame(np.concatenate((np.array([col] * len(comm_size_list))[:,None], np.array([region] * len(comm_size_list))[:,None], np.array(['purity'] * len(comm_size_list))[:,None], np.array(purity_list)[:,None]), 1), columns=['stimulus', 'region', 'type', 'metric'])], ignore_index=True)
  # df['metric'] = pd.to_numeric(df['metric'])

  df['number of communities'] = pd.to_numeric(df['number of communities'])
  df['community size'] = pd.to_numeric(df['community size'])
  df['occupancy'] = pd.to_numeric(df['occupancy'])
  df['leadership'] = pd.to_numeric(df['leadership'])
  df['purity'] = pd.to_numeric(df['purity'])
  df['entropy'] = pd.to_numeric(df['entropy'])
  return df

df = occupancy_community_perregion(comms_dict, area_dict, visual_regions, max_neg_reso=max_reso_config)
# %%
def plot_data_per_region(df, regions):
  names = ['community size', 'occupancy', 'leadership', 'purity', 'entropy']
  for name1, name2 in itertools.combinations(names, 2):
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for s_ind, stimulus_name in enumerate(stimulus_names):
      i, j = s_ind // 4, s_ind % 4
      ax = axes[i, j]
      for r_ind, region in enumerate(regions):
        mat = df[(df['stimulus']==stimulus_name) & (df['region']==region)][[name1, name2]].values
        if mat.size:
          x, y = mat[:,0], mat[:,1]
          ax.scatter(x, y, facecolors='none', linewidth=.2, edgecolors=region_colors[r_ind], alpha=.5)
          ax.scatter(np.mean(x), np.mean(y), facecolors='none', linewidth=3., marker='^', s=200, edgecolors=region_colors[r_ind])
      ax.set_title(stimulus_name)
      ax.set_xlabel(name1)
      ax.set_ylabel(name2)
      ax.xaxis.set_tick_params(labelsize=14)
      ax.yaxis.set_tick_params(labelsize=14)
      ax.set_xlabel(ax.get_xlabel(), fontsize=14,color='k') #, weight='bold'
      ax.set_ylabel(ax.get_ylabel(), fontsize=14,color='k') #, weight='bold'
      for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_color('0.2')
      ax.spines['top'].set_visible(False)
      ax.spines['right'].set_visible(False)
      ax.tick_params(width=1.5)
    for r_ind, region in enumerate(regions):
      ax.scatter([], [], facecolors='none', linewidth=2, edgecolors=region_colors[r_ind], label=region)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='', bbox_to_anchor=(.9, 1.), loc='upper left', fontsize=12, frameon=False)
    plt.tight_layout()
    plt.savefig('./plots/{}_{}.pdf'.format(name1.replace(' ', '_'), name2.replace(' ', '_')), transparent=True)
    plt.show()

plot_data_per_region(df, visual_regions)
# %%
def plot_data_per_region_by_type(df, regions):
  names = ['community size', 'occupancy', 'leadership', 'purity', 'entropy']
  for name1, name2 in itertools.combinations(names, 2):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for s_ind, stimulus_type in enumerate(stimulus_types):
      ax = axes[s_ind]
      for r_ind, region in enumerate(regions):
        mat = df[(df.apply(lambda x: stimulus2stype(x['stimulus'])[1], axis=1)==stimulus_type) & (df['region']==region)][[name1, name2]].values
        if mat.size:
          x, y = mat[:,0], mat[:,1]
          ax.scatter(x, y, facecolors='none', linewidth=.2, edgecolors=region_colors[r_ind], alpha=.5)
          ax.scatter(np.mean(x), np.mean(y), facecolors='none', linewidth=3., marker='^', s=200, edgecolors=region_colors[r_ind])
      ax.set_title(stimulus_type)
      ax.set_xlabel(name1)
      ax.set_ylabel(name2)
      ax.xaxis.set_tick_params(labelsize=14)
      ax.yaxis.set_tick_params(labelsize=14)
      ax.set_xlabel(ax.get_xlabel(), fontsize=14,color='k') #, weight='bold'
      ax.set_ylabel(ax.get_ylabel(), fontsize=14,color='k') #, weight='bold'
      for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_color('0.2')
      ax.spines['top'].set_visible(False)
      ax.spines['right'].set_visible(False)
      ax.tick_params(width=1.5)
    for r_ind, region in enumerate(regions):
      ax.scatter([], [], facecolors='none', linewidth=2, edgecolors=region_colors[r_ind], label=region)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='', bbox_to_anchor=(.9, 1.), loc='upper left', fontsize=12, frameon=False)
    plt.tight_layout()
    plt.savefig('./plots/{}_{}_type.pdf'.format(name1.replace(' ', '_'), name2.replace(' ', '_')), transparent=True)
    # plt.show()

plot_data_per_region_by_type(df, visual_regions)
# %%
def plot_heatmap_region_community(comms_dict, area_dict, regions, max_neg_reso):
  rows, cols = get_rowcol(comms_dict)
  if max_neg_reso is None:
    max_neg_reso = np.ones((len(rows), len(cols)))
  all_region_counts = {}
  fig, axes = plt.subplots(1, 8, figsize=(30, 6))
  cbar_ax = fig.add_axes([.91, .3, .01, .6]) # (locx, locy, width, height)
  scolors = [stimulus_type_color[stimulus2stype(col)[0]] for col in cols]
  cmap_list = cycle([colors.LinearSegmentedColormap.from_list("", ["white",c]) for c in scolors])
  for col_ind, col in enumerate(cols):
    print(col)
    ax = axes[col_ind]
    region_counts = []
    for row_ind, row in enumerate(session2keep):
      node_area = area_dict[row]
      max_reso = max_neg_reso[row_ind][col_ind]
      comms_list = comms_dict[row][col][max_reso]
      for comms in comms_list: # 50 repeats
        for comm in comms:
          if len(comm) >= 4:
            c_regions = [node_area[node] for node in comm]
            region_counts.append([c_regions.count(region)/len(c_regions) for region in regions])
    region_counts = np.array(region_counts)
    all_region_counts[col] = region_counts[np.lexsort((region_counts[:,0],region_counts[:,1],region_counts[:,2],region_counts[:,3],region_counts[:,4],region_counts[:,5]))]
    hm = sns.heatmap(all_region_counts[col], ax=ax, cmap=cmap_list.__next__(), cbar=col_ind == 0, cbar_ax=None if col_ind else cbar_ax)
    ax.set_xticks(0.5 + np.arange(len(regions)))
    ax.set_xticklabels(labels=regions)
    hm.set(yticklabels=[])  
    hm.set(ylabel=None)
    hm.tick_params(left=False)  # remove the ticks
    hm.tick_params(bottom=False)
    ax.set_title(paper_label[col_ind].replace('\n', ' '))
  fig.tight_layout(rect=[0, 0, .9, 1])
  plt.savefig('./plots/heatmap_region_community.pdf', transparent=True)
  plt.show()

plot_heatmap_region_community(comms_dict, area_dict, visual_regions, max_neg_reso=max_reso_config)
# %%
