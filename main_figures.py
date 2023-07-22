# %%
from library import *
# # session_ids = 
# stimulus2marker = {'Resting\nstate':'s', 'Flashes':'*', 'Drifting\ngratings':'X', 'Static\ngratings':'P', 'Natural\nscenes':r'$\clubsuit$', 'Natural\nmovies':'>'}
# marker_size_dict = {'v':10, '*':22, 'P':13, 'X':13, 'o':11, 's':9.5, 'D':9, 'p':12, '>':10, r'$\clubsuit$':20}
# scatter_size_dict = {'v':10, '*':17, 'P':13, 'X':13, 'o':11, 's':10, 'D':9, 'p':13, '>':12, r'$\clubsuit$':16}
# error_size_dict = {'v':10, '*':24, 'P':16, 'X':16, 'o':11, 's':9., 'D':9, 'p':12, '>':13, r'$\clubsuit$':22}

# import statsmodels.api as sm
# def export_legend(legend, filename):
#     fig  = legend.figure
#     fig.canvas.draw()
#     bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#     fig.savefig(filename, transparent=True, bbox_inches=bbox)

# def confidence_interval(data, confidence=0.95):
#     a = 1.0 * np.array(data)
#     n = len(a)
#     m, se = np.mean(a), stats.sem(a)
#     h = se * stats.t.ppf((1 + confidence) / 2., n-1)
#     return m-h, m+h

# def get_confidence_interval(data, confidence=0.95):
#     a = 1.0 * np.array(data)
#     n = len(a)
#     se = stats.sem(a)
#     ci = se * stats.t.ppf((1 + confidence) / 2., n-1)
#     return ci

# def annot_significance(star, x1, x2, y, fontsize=14, stimulus='k', ax=None):
#   ax = plt.gca() if ax is None else ax
#   ax.text((x1+x2)*.5, y, star, ha='center', va='bottom', color=stimulus, fontsize=fontsize)

# def annot_difference(star, x1, x2, y, h, lw=2.5, fontsize=14, stimulus='k', ax=None):
#   ax = plt.gca() if ax is None else ax
#   ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=lw, c=stimulus)
#   ax.text((x1+x2)*.5, y+h, star, ha='center', va='bottom', color=stimulus, fontsize=fontsize)

# def double_binning(x, y, numbin=20, log=False):
#   if log:
#     bins = np.logspace(np.log10(x.min()), np.log10(x.max()), numbin) # log binning
#   else:
#     bins = np.linspace(x.min(), x.max(), numbin) # linear binning
#   digitized = np.digitize(x, bins) # binning based on community size
#   binned_x = [x[digitized == i].mean() for i in range(1, len(bins))]
#   binned_y = [y[digitized == i].mean() for i in range(1, len(bins))]
#   return binned_x, binned_y

# def double_equal_binning(x, y, numbin=20, log=False):
#   if log:
#     bins = np.logspace(np.log10(x.min()), np.log10(x.max()), numbin) # log binning
#   else:
#     bins = np.linspace(x.min(), x.max(), numbin) # linear binning
#   digitized = np.digitize(x, bins) # binning based on community size
#   binned_x = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
#   binned_y = [y[digitized == i].mean() for i in range(1, len(bins))]
#   return binned_x, binned_y

# def double_equal_binning_counts(x, y, numbin=20, log=False):
#   if log:
#     bins = np.logspace(np.log10(x.min()), np.log10(x.max()), numbin) # log binning
#   else:
#     bins = np.linspace(x.min(), x.max(), numbin) # linear binning
#   digitized = np.digitize(x, bins) # binning based on community size
#   connect = [(y[digitized == i]==1).sum() for i in range(1, len(bins))]
#   disconnect = [(y[digitized == i]==0).sum() for i in range(1, len(bins))]
#   return connect, disconnect

# ############ find nodes and comms with at least one between community edge
# def get_unique_elements(nested_list):
#     return list(set(flatten_list(nested_list)))

# def flatten_list(nested_list):
#     return [item for sublist in nested_list for item in sublist]

# def _find_between_community_edges(edges, node_to_community):
#   """Convert the graph into a weighted network of communities."""
#   between_community_edges = dict()
#   for (ni, nj) in edges:
#       if (ni in node_to_community) and (nj in node_to_community):
#           ci = node_to_community[ni]
#           cj = node_to_community[nj]
#           if ci != cj:
#               if (ci, cj) in between_community_edges:
#                   between_community_edges[(ci, cj)] += 1
#               elif (cj, ci) in between_community_edges:
#                   # only compute the undirected graph
#                   between_community_edges[(cj, ci)] += 1
#               else:
#                   between_community_edges[(ci, cj)] = 1

#   return between_community_edges

# plt.rcParams['font.family'] = 'serif'
# plt.rcParams["font.serif"] = ["Times New Roman"]

# combined_stimuli = [['spontaneous'], ['flash_dark', 'flash_light'], ['drifting_gratings'], ['static_gratings'], ['natural_scenes'], ['natural_movie_one', 'natural_movie_three']]
# combined_stimulus_names = ['Resting\nstate', 'Flashes', 'Drifting\ngratings', 'Static\ngratings', 'Natural\nscenes', 'Natural\nmovies']
# combined_stimulus_colors = ['#8dd3c7', '#fee391', '#bc80bd', '#bc80bd', '#fb8072', '#fb8072']
# # plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
# session2keep = ['719161530','750749662','754312389','755434585','756029989','791319847','797828357']
# stimulus_by_type = [['spontaneous'], ['flash_dark', 'flash_light'], ['drifting_gratings', 'static_gratings'], ['natural_scenes', 'natural_movie_one', 'natural_movie_three']]
# stimulus_types = ['Resting state', 'Flashes', 'Gratings', 'Natural stimuli']
# # stimulus_type_color = ['tab:blue', 'darkorange', 'darkgreen', 'maroon']
# stimulus_type_color = ['#8dd3c7', '#fee391', '#bc80bd', '#fb8072']
# stimulus_labels = ['Resting\nstate', 'Dark\nflash', 'Light\nflash', 'Drifting\ngrating', 
#               'Static\ngrating', 'Natural\nscenes', 'Natural\nmovie 1', 'Natural\nmovie 3']
# region_labels = ['AM', 'PM', 'AL', 'RL', 'LM', 'V1']
# # region_colors = ['#b3de69', '#80b1d3', '#fdb462', '#d9d9d9', '#fccde5', '#bebada']
# region_colors = ['#d9e9b5', '#c0d8e9', '#fed3a1', '#c3c3c3', '#fad3e4', '#cec5f2']
# TRIAD_NAMES = ('003', '012', '102', '021D', '021U', '021C', '111D', '111U', '030T', '030C', '201', '120D', '120U', '120C', '210', '300')
# model_names = [u'Erdős-Rényi model', 'Degree-preserving model', 'Pair-preserving model', 'Signed-pair-preserving model']

area_dict, active_area_dict = load_area_dicts(session_ids)
path = './files/adjacency_matrices'
G_ccg_dict, offset_dict, duration_dict = load_graphs(path, active_area_dict, weight=True)
measure = 'ccg'
n = 4
S_ccg_dict = add_sign(G_ccg_dict)
S_ccg_dict = add_offset(S_ccg_dict, offset_dict)
S_ccg_dict = add_duration(S_ccg_dict, duration_dict)
S_ccg_dict = add_delay(S_ccg_dict)
# # %%
# # remove flashes from spike_trains and adjacency_matrices  DONE!!!!!!!!!!!!!
# directory = './files/spike_trains'
# files = os.listdir(directory)
# files.sort(key=lambda x:int(x[:9]))
# for file in files:
#   if 'flashes' in file:
#     os.remove(os.path.join(directory, file))

# directory = './files/adjacency_matrices'
# files = os.listdir(directory)
# files.sort(key=lambda x:int(x[:9]))
# for file in files:
#   if 'flashes' in file:
#     os.remove(os.path.join(directory, file))
# # %%
# # remove thalamic areas from adj_mat.        DONE!!!!!!!!!!!!
# directory = './files/adjacency_matrices/'
# files = os.listdir(directory)
# files.sort(key=lambda x:int(x[:9]))
# for file in files:
#   if file.endswith(".npz") and ('gabors' not in file) and ('flashes' not in file) and ('_offset' not in file) and ('_duration' not in file) and ('_bl' not in file) and ('confidence' not in file):
#     print(file)
#     adj_path, conf_path, off_path, dur_path = os.path.join(directory, file), os.path.join(directory, file.replace('.npz', '_confidence.npz')), os.path.join(directory, file.replace('.npz', '_offset.npz')), os.path.join(directory, file.replace('.npz', '_duration.npz'))
#     adj_mat, confidence_mat, offset_mat, duration_mat = load_npz_3d(adj_path), load_npz_3d(conf_path), load_npz_3d(off_path), load_npz_3d(dur_path)
#     session = file.split('_')[0]
#     nodes = list(active_area_dict[session].keys())
#     cor_mat_inds = np.array([nodes.index(node) for node, area in active_area_dict[session].items() if area in visual_regions])
#     save_npz(adj_mat[cor_mat_inds[:,None], cor_mat_inds], adj_path)
#     save_npz(confidence_mat[cor_mat_inds[:,None], cor_mat_inds], conf_path)
#     save_npz(offset_mat[cor_mat_inds[:,None], cor_mat_inds], off_path)
#     save_npz(duration_mat[cor_mat_inds[:,None], cor_mat_inds], dur_path)
# # %%
# # remove thalamic areas from spike trains      DONE!!!!!!!!!!
# directory = './files/spike_trains/'
# files = os.listdir(directory)
# files.sort(key=lambda x:int(x[:9]))
# for file in files:
#   # print(file)
#   st_path = os.path.join(directory, file)
#   sequences = load_npz_3d(st_path)
#   print(file, sequences.shape)
  # session = file.split('_')[0]
  # nodes = list(area_dict[session].keys())
  # cor_mat_inds = np.array([nodes.index(node) for node, area in area_dict[session].items() if area in visual_regions])
  # save_npz(sequences[cor_mat_inds], st_path)

# # %%
# # remove second session and gabors      DONE!!!!!!!!!!
# directory = './files/spike_trains/'
# # directory = './files/adjacency_matrices/'
# files = os.listdir(directory)
# files.sort(key=lambda x:int(x[:9]))
# for file in files:
#   if ('gabors' in file) or ('750332458' in file):
#     path = os.path.join(directory, file)
#     os.remove(path)
# # %%
# # remove second session and thalamic area from area_dict & active_area_dict      DONE!!!!!!!!!!
# a_file = open('../functional_network/data/ecephys_cache_dir/sessions/area_dict.pkl', 'rb')
# area_dict = pickle.load(a_file)
# area_dict = {n:a for n,a in area_dict.items() if n in session2keep}
# area_dict = {s:{n:a for n,a in area_dict[s].items() if a in visual_regions} for s in area_dict}
# a_file = open('./files/area_dict.pkl', 'wb')
# pickle.dump(area_dict, a_file)
# a_file.close()

# a_file = open('../functional_network/data/ecephys_cache_dir/sessions/active_area_dict.pkl', 'rb')
# active_area_dict = pickle.load(a_file)
# active_area_dict = {n:a for n,a in active_area_dict.items() if n in session2keep}
# active_area_dict = {s:{n:a for n,a in active_area_dict[s].items() if a in visual_regions} for s in active_area_dict}
# a_file = open('./files/active_area_dict.pkl', 'wb')
# pickle.dump(active_area_dict, a_file)
# a_file.close()
# %%
# Figure 1B
####################### raster plot
def get_raster_data(session_index, s_lengths, blank_width=50):
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

def plot_raster(total_sequence, areas_num, areas_start_pos, sequence_by_area):
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

session_index = 1 # select an example session to plot
s_lengths  = [800, 250, 370, 250, 250, 1000] # length of spike trains for each stimulus to plot respectively
blank_width = 50 # add blank space between different stimuli
total_sequence, areas_num, areas_start_pos, sequence_by_area = get_raster_data(session_index, s_lengths, blank_width)
plot_raster(total_sequence, areas_num, areas_start_pos, sequence_by_area)
# %%
# Figure 1D
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

session_ind, stimulus_ind = 4, 7 # example session and stimulus to plot
ccg_zscore, ccg_value = get_connectivity_data(G_ccg_dict, session_ind, stimulus_ind)
#%%
# Figure 1D
###################### plot connectivity matrix from left to right, from bottom to up
def plot_connectivity_matrix_annotation(G_dict, session_ind, stimulus_ind, ccg_zscore, ccg_value, weight=None, ratio=15):
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
  # divider = make_axes_locatable(ax)
  # cax = divider.append_axes("right", size="3%", pad=0.2)
  # cax.axis('off')
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
  # divider = make_axes_locatable(ax)
  # cax = divider.append_axes("top", size="3%", pad=0.2)
  # cax.axis('off')
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
  # divider = make_axes_locatable(ax)
  # cax = divider.append_axes("right", size="3%", pad=0.2)
  # plt.colorbar(im, cax=cax, orientation='vertical')
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
  # plt.tight_layout()
  figname = './figures/figure1D_left.pdf' if weight is not None else './figures/figure1D_right.pdf'
  plt.savefig(figname.format(session, stimulus), transparent=True)
  # plt.show()

ratio = 15
plot_connectivity_matrix_annotation(G_ccg_dict, session_ind, stimulus_ind, ccg_zscore, ccg_value, weight='weight', ratio=ratio)
plot_connectivity_matrix_annotation(G_ccg_dict, session_ind, stimulus_ind, ccg_zscore, ccg_value, weight=None, ratio=ratio)
# %%
# Figure 1E
############################ new excitatory VS inhibitory connections
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

plot_new_ex_in_bar(S_ccg_dict, density=True)
# %%
# Figure 1F
def scatter_dataVSdensity_new(G_dict, area_dict, regions, name='intra'):
  sessions, stimuli = get_session_stimulus(G_dict)
  fig, ax = plt.subplots(figsize=(5, 5))
  X, Y = [], []
  df = pd.DataFrame()
  region_connection = np.zeros((len(sessions), len(stimuli), len(regions), len(regions)))
  for stimulus_ind, stimulus in enumerate(stimuli):
    # print(stimulus)
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
      # metric[session_ind, stimulus_ind, 0] =  np.sum(region_connection[session_ind, stimulus_ind][diag_indx])
      # metric[session_ind, stimulus_ind, 1] =  np.sum(region_connection[session_ind, stimulus_ind][~diag_indx])
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

scatter_dataVSdensity_new(S_ccg_dict, area_dict, visual_regions, name='intra')
scatter_dataVSdensity_new(S_ccg_dict, area_dict, visual_regions, name='ex')
scatter_dataVSdensity_new(S_ccg_dict, area_dict, visual_regions, name='cluster')
#%%
# Figure 1G
def get_pos_neg_p_signalcorr(G_dict, signal_correlation_dict, pairtype='all'):
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

start_time = time.time()
with open('./files/signal_correlation_dict.pkl', 'rb') as f:
  signal_correlation_dict = pickle.load(f)
pairtype = 'all'
# pairtype = 'connected'
pos_connectionp_signalcorr_df, neg_connectionp_signalcorr_df, dis_connected_signalcorr_df = get_pos_neg_p_signalcorr(G_ccg_dict, signal_correlation_dict, pairtype=pairtype)
print("--- %s minutes" % ((time.time() - start_time)/60))
# # %%
# # remove inactive neurons and thalamic neurons from signal_correlation_dict   DONE!!!!!!!!!!
# with open('../functional_network/files/signal_correlation_dict.pkl', 'rb') as f:
#   osignal_correlation_dict = pickle.load(f)
# sessions, stimuli = get_session_stimulus(G_ccg_dict)
# signal_correlation_dict = {}
# for session in sessions:
#   active_inds = np.array(list(active_area_dict[session].keys()))
#   signal_correlation_dict[session] = {}
#   for combined_stimulus_name in combined_stimulus_names[2:]:
#     signal_correlation_dict[session][combined_stimulus_name] = osignal_correlation_dict[session][combined_stimulus_name][active_inds[:,None], active_inds]
# a_file = open('./files/signal_correlation_dict.pkl', 'wb')
# pickle.dump(signal_correlation_dict, a_file)
# a_file.close()
# %%
# Figure 1G
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

plot_pos_neg_signal_correlation_distri(pos_connectionp_signalcorr_df, neg_connectionp_signalcorr_df, dis_connected_signalcorr_df)
# %%
# Figure 2A
################## relative count of signed node pairs
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
# plot_signed_pair_relative_count(S_ccg_dict, p_signed_pair_func, measure, n, scale=False)
plot_signed_pair_relative_count(S_ccg_dict, p_signed_pair_func, log=True)
# %%
# Load data for Figures 2-3
######################## signed motif detection
with open('./files/intensity_dict.pkl', 'rb') as f:
  intensity_dict = pickle.load(f)
with open('./files/coherence_dict.pkl', 'rb') as f:
  coherence_dict = pickle.load(f)
with open('./files/gnm_baseline_intensity_dict.pkl', 'rb') as f:
  gnm_baseline_intensity_dict = pickle.load(f)
with open('./files/gnm_baseline_coherence_dict.pkl', 'rb') as f:
  gnm_baseline_coherence_dict = pickle.load(f)
with open('./files/baseline_intensity_dict.pkl', 'rb') as f:
  baseline_intensity_dict = pickle.load(f)
with open('./files/baseline_coherence_dict.pkl', 'rb') as f:
  baseline_coherence_dict = pickle.load(f)
with open('./files/unibi_baseline_intensity_dict.pkl', 'rb') as f:
  unibi_baseline_intensity_dict = pickle.load(f)
with open('./files/unibi_baseline_coherence_dict.pkl', 'rb') as f:
  unibi_baseline_coherence_dict = pickle.load(f)
with open('./files/sunibi_baseline_intensity_dict.pkl', 'rb') as f:
  sunibi_baseline_intensity_dict = pickle.load(f)
with open('./files/sunibi_baseline_coherence_dict.pkl', 'rb') as f:
  sunibi_baseline_coherence_dict = pickle.load(f)
# # %%
# # remove second session from all dictionaries        DONE!!!!!!!!!!!!!!
# def remove_2nd(dic, fname):
#   dic = {n:a for n,a in dic.items() if n in sessions}
#   a_file = open('./files/{}'.format(fname), 'wb')
#   pickle.dump(dic, a_file)
#   a_file.close()

# remove_2nd(intensity_dict, 'intensity_dict.pkl')
# remove_2nd(coherence_dict, 'coherence_dict.pkl')
# remove_2nd(gnm_baseline_intensity_dict, 'gnm_baseline_intensity_dict.pkl')
# remove_2nd(gnm_baseline_coherence_dict, 'gnm_baseline_coherence_dict.pkl')
# remove_2nd(baseline_intensity_dict, 'baseline_intensity_dict.pkl')
# remove_2nd(baseline_coherence_dict, 'baseline_coherence_dict.pkl')
# remove_2nd(unibi_baseline_intensity_dict, 'unibi_baseline_intensity_dict.pkl')
# remove_2nd(unibi_baseline_coherence_dict, 'unibi_baseline_coherence_dict.pkl')
# remove_2nd(sunibi_baseline_intensity_dict, 'sunibi_baseline_intensity_dict.pkl')
# remove_2nd(sunibi_baseline_coherence_dict, 'sunibi_baseline_coherence_dict.pkl')
# %%
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

################## average intensity across session
################## first Z score, then average
num_baseline = 200
whole_df1, mean_df1, signed_motif_types1 = get_intensity_zscore(intensity_dict, coherence_dict, gnm_baseline_intensity_dict, gnm_baseline_coherence_dict, num_baseline=num_baseline) # Gnm
whole_df2, mean_df2, signed_motif_types2 = get_intensity_zscore(intensity_dict, coherence_dict, baseline_intensity_dict, baseline_coherence_dict, num_baseline=num_baseline) # directed double edge swap
whole_df3, mean_df3, signed_motif_types3 = get_intensity_zscore(intensity_dict, coherence_dict, unibi_baseline_intensity_dict, unibi_baseline_coherence_dict, num_baseline=num_baseline) # uni bi edge preserved
whole_df4, mean_df4, signed_motif_types4 = get_intensity_zscore(intensity_dict, coherence_dict, sunibi_baseline_intensity_dict, sunibi_baseline_coherence_dict, num_baseline=num_baseline) # signed uni bi edge preserved
whole_df1['signed motif type'] = whole_df1['signed motif type'].str.replace('-', '\N{MINUS SIGN}') # change minus sign to match width of plus
whole_df2['signed motif type'] = whole_df2['signed motif type'].str.replace('-', '\N{MINUS SIGN}') # change minus sign to match width of plus
whole_df3['signed motif type'] = whole_df3['signed motif type'].str.replace('-', '\N{MINUS SIGN}') # change minus sign to match width of plus
whole_df4['signed motif type'] = whole_df4['signed motif type'].str.replace('-', '\N{MINUS SIGN}') # change minus sign to match width of plus
signed_motif_types1 = [mt.replace('-', '\N{MINUS SIGN}') for mt in signed_motif_types1]
signed_motif_types2 = [mt.replace('-', '\N{MINUS SIGN}') for mt in signed_motif_types2]
signed_motif_types3 = [mt.replace('-', '\N{MINUS SIGN}') for mt in signed_motif_types3]
signed_motif_types4 = [mt.replace('-', '\N{MINUS SIGN}') for mt in signed_motif_types4]
# %%
# Figure 2C
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

dfs = [whole_df1, whole_df2, whole_df3, whole_df4]
df_ind = 3 # plot results obtained with Signed-pair-preserving model
plot_zscore_allmotif_lollipop(dfs[df_ind], model_names[df_ind])
# %%
# Figure 3A
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

scatter_ZscoreVSdensity(whole_df4, G_ccg_dict)
# %%
# Figure 3B
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

plot_heatmap_correlation_zscore(whole_df4)
# %%
# Figure 3C
def get_signalcorr_within_across_motif(G_dict, eFFLb_types, all_motif_types, signal_correlation_dict, pair_type='all'):
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
        df = mean_df4[mean_df4['stimulus']==stimulus]
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

# pair_type = 'all'
pair_type = 'connected'
sig_motif_types = ['030T+++', '120D++++', '120U++++', '120C++++', '210+++++', '300++++++']
motif_types = ['021D', '021U', '021C', '111D', '111U', '030T', '030C', '201', '120D', '120U', '120C', '210', '300']
signal_corr_within_across_motif_df = get_signalcorr_within_across_motif(G_ccg_dict, sig_motif_types, motif_types, signal_correlation_dict, pair_type=pair_type)
# %%
# Figure 3C
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

plot_signalcorr_within_across_motif_significance(signal_corr_within_across_motif_df, pair_type=pair_type)
# %%
# Figure 3D
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

################## regional distribution of significant motifs 030T+,120D+,120U+,120C+,210+,300+
sig_motif_types = ['030T+++', '120D++++', '120U++++', '120C++++', '210+++++', '300++++++']
region_count_dict = get_motif_region_census(G_ccg_dict, area_dict, sig_motif_types)
#%%
# Figure 3D
################## plot errorplot for motif region
def plot_motif_region_error(whole_df, region_count_dict, signed_motif_types, mtype='all_V1'):
  fig, ax = plt.subplots(1,1, sharex=True, sharey=True, figsize=(7, 4))
  df = pd.DataFrame()
  palette = [[plt.cm.tab20b(i) for i in range(20)][i] for i in [8,16,18,19]] + [[plt.cm.tab20c(i) for i in range(20)][i] for i in [4,16]]
  for mt_ind, signed_motif_type in enumerate(signed_motif_types):
    print(signed_motif_type)
    for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
      for stimulus in combined_stimuli[cs_ind]:
        for session_ind, session in enumerate(sessions):
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

plot_motif_region_error(whole_df4, region_count_dict, sig_motif_types, mtype='one_V1')
plot_motif_region_error(whole_df4, region_count_dict, sig_motif_types, mtype='all_V1')
