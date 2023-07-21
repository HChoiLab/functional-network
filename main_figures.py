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
# from mpl_toolkits.axes_grid1 import make_axes_locatable

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
  # try: 
  # ccg = load_npz_3d(os.path.join(directory, file))
  # except:
  ccg = load_sparse_npz(os.path.join(directory, file))
  # try:
  # ccg_jittered = load_npz_3d(os.path.join(directory, file.replace('.npz', '_bl.npz')))
  # except:
  ccg_jittered = load_sparse_npz(os.path.join(directory, file.replace('.npz', '_bl.npz')))
  ccg_corrected = ccg - ccg_jittered
  ccg_zscore, ccg_value = ccg2zscore(ccg_corrected, max_duration=11, maxlag=12)
  return ccg_zscore, ccg_value

session_ind, stimulus_ind = 4, 7
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
  rows, cols = get_rowcol(G_dict)
  for col_ind, col in enumerate(cols):
    print(col)
    combined_stimulus_name = combine_stimulus(col)[1]
    ex_data, in_data = [], []
    for row_ind, row in enumerate(rows):
      G = G_dict[row][col] if col in G_dict[row] else nx.Graph()
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
  # ax = sns.violinplot(x='stimulus', y='number of connections', hue="type", data=df, palette="muted", split=False)
  # ax = sns.barplot(x='stimulus', y=y, hue="type", data=df, palette=['white', 'grey'], edgecolor=".5")
  barcolors = ['firebrick', 'navy']
  ax = sns.barplot(x="stimulus", y=y, hue="type",  data=df, palette=barcolors, errorbar="sd",  edgecolor="black", errcolor="black", errwidth=1.5, capsize = 0.1, alpha=0.5) #, width=.6
  sns.stripplot(x="stimulus", y=y, hue="type", palette=barcolors, data=df, dodge=True, alpha=0.6, ax=ax)
  # remove extra legend handles
  handles, labels = ax.get_legend_handles_labels()
  ax.legend(handles[2:], labels[2:], title='', bbox_to_anchor=(.7, 1.), loc='upper left', fontsize=28, frameon=False)
  # plt.setp(ax.get_legend().get_texts()) #, weight='bold'
  
  plt.yticks(fontsize=25) #,  weight='bold'
  plt.ylabel(y.capitalize())
  ax.set_ylabel(ax.get_ylabel(), fontsize=30,color='k') #, weight='bold'
  for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2.5)
    ax.spines[axis].set_color('k')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.tick_params(width=2.5)
  # plt.yscale('log')
  ax.set(xlabel=None)
  plt.xticks([])
  plt.tight_layout()
  figname = './plots/box_ex_in_num.pdf' if not density else './plots/box_ex_in_density.pdf'
  plt.savefig(figname, transparent=True)
  # plt.show()
######################## excitaroty link VS inhibitory link box
plot_new_ex_in_bar(S_ccg_dict, density=True)
# %%
