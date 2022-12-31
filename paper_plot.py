# %%
############################## Figure plots for publication ##############################
##############################                        ##############################
from library import *
stimulus2marker = {'Resting\nstate':'s', 'Flashes':'*', 'Drifting\ngratings':'X', 'Static\ngratings':'P', 'Natural\nscenes':r'$\clubsuit$', 'Natural\nmovies':'>'}
marker_size_dict = {'v':10, '*':22, 'P':13, 'X':13, 'o':11, 's':9.5, 'D':9, 'p':12, '>':10, r'$\clubsuit$':20}
scatter_size_dict = {'v':10, '*':17, 'P':13, 'X':13, 'o':11, 's':10, 'D':9, 'p':13, '>':12, r'$\clubsuit$':16}
error_size_dict = {'v':10, '*':24, 'P':16, 'X':16, 'o':11, 's':9., 'D':9, 'p':12, '>':13, r'$\clubsuit$':22}

import statsmodels.api as sm
def export_legend(legend, filename):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, transparent=True, bbox_inches=bbox)

def confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m-h, m+h

def get_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    se = stats.sem(a)
    ci = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return ci

def annot_significance(star, x1, x2, y, fontsize=14, col='k', ax=None):
  ax = plt.gca() if ax is None else ax
  ax.text((x1+x2)*.5, y, star, ha='center', va='bottom', color=col, fontsize=fontsize)

def annot_difference(star, x1, x2, y, h, lw=2.5, fontsize=14, col='k', ax=None):
  ax = plt.gca() if ax is None else ax
  ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=lw, c=col)
  ax.text((x1+x2)*.5, y+h, star, ha='center', va='bottom', color=col, fontsize=fontsize)

def double_binning(x, y, numbin=20, log=False):
  if log:
    bins = np.logspace(np.log10(x.min()), np.log10(x.max()), numbin) # log binning
  else:
    bins = np.linspace(x.min(), x.max(), numbin) # linear binning
  digitized = np.digitize(x, bins) # binning based on community size
  binned_x = [x[digitized == i].mean() for i in range(1, len(bins))]
  binned_y = [y[digitized == i].mean() for i in range(1, len(bins))]
  return binned_x, binned_y

def double_equal_binning(x, y, numbin=20, log=False):
  if log:
    bins = np.logspace(np.log10(x.min()), np.log10(x.max()), numbin) # log binning
  else:
    bins = np.linspace(x.min(), x.max(), numbin) # linear binning
  digitized = np.digitize(x, bins) # binning based on community size
  binned_x = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
  binned_y = [y[digitized == i].mean() for i in range(1, len(bins))]
  return binned_x, binned_y

def double_equal_binning_counts(x, y, numbin=20, log=False):
  if log:
    bins = np.logspace(np.log10(x.min()), np.log10(x.max()), numbin) # log binning
  else:
    bins = np.linspace(x.min(), x.max(), numbin) # linear binning
  digitized = np.digitize(x, bins) # binning based on community size
  connect = [(y[digitized == i]==1).sum() for i in range(1, len(bins))]
  disconnect = [(y[digitized == i]==0).sum() for i in range(1, len(bins))]
  return connect, disconnect

############ find nodes and comms with at least one between community edge
def get_unique_elements(nested_list):
    return list(set(flatten_list(nested_list)))

def flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]

def _find_between_community_edges(edges, node_to_community):
  """Convert the graph into a weighted network of communities."""
  between_community_edges = dict()
  for (ni, nj) in edges:
      if (ni in node_to_community) and (nj in node_to_community):
          ci = node_to_community[ni]
          cj = node_to_community[nj]
          if ci != cj:
              if (ci, cj) in between_community_edges:
                  between_community_edges[(ci, cj)] += 1
              elif (cj, ci) in between_community_edges:
                  # only compute the undirected graph
                  between_community_edges[(cj, ci)] += 1
              else:
                  between_community_edges[(ci, cj)] = 1

  return between_community_edges

plt.rcParams['font.family'] = 'serif'
plt.rcParams["font.serif"] = ["Times New Roman"]

combined_stimuli = [['spontaneous'], ['flash_dark', 'flash_light'], ['drifting_gratings'], ['static_gratings'], ['natural_scenes'], ['natural_movie_one', 'natural_movie_three']]
combined_stimulus_names = ['Resting\nstate', 'Flashes', 'Drifting\ngratings', 'Static\ngratings', 'Natural\nscenes', 'Natural\nmovies']
combined_stimulus_colors = ['#8dd3c7', '#fee391', '#bc80bd', '#bc80bd', '#fb8072', '#fb8072']
# plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
session2keep = ['719161530','750749662','754312389','755434585','756029989','791319847','797828357']
stimulus_by_type = [['spontaneous'], ['flash_dark', 'flash_light'], ['drifting_gratings', 'static_gratings'], ['natural_scenes', 'natural_movie_one', 'natural_movie_three']]
stimulus_types = ['Resting state', 'Flashes', 'Gratings', 'Natural stimuli']
# stimulus_type_color = ['tab:blue', 'darkorange', 'darkgreen', 'maroon']
stimulus_type_color = ['#8dd3c7', '#fee391', '#bc80bd', '#fb8072']
stimulus_labels = ['Resting\nstate', 'Dark\nflash', 'Light\nflash', 'Drifting\ngrating', 
              'Static\ngrating', 'Natural\nscenes', 'Natural\nmovie 1', 'Natural\nmovie 3']
region_labels = ['AM', 'PM', 'AL', 'RL', 'LM', 'V1']
# region_colors = ['#b3de69', '#80b1d3', '#fdb462', '#d9d9d9', '#fccde5', '#bebada']
region_colors = ['#d9e9b5', '#c0d8e9', '#fed3a1', '#c3c3c3', '#fad3e4', '#cec5f2']
TRIAD_NAMES = ('003', '012', '102', '021D', '021U', '021C', '111D', '111U', '030T', '030C', '201', '120D', '120U', '120C', '210', '300')
model_names = [u'Erdős-Rényi model', 'Degree-preserving model', 'Pair-preserving model', 'Signed-pair-preserving model']

area_dict, active_area_dict, mean_speed_df = load_other_data(session_ids)
cortical_inds = get_cortical_inds(active_area_dict, visual_regions)
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
#%%
####################### Figure 1 #######################
########################################################
#%%
####################### Plot stimulus
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
data_directory = './data/ecephys_cache_dir'
manifest_path = os.path.join(data_directory, "manifest.json")
# cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
boc =  BrainObservatoryCache(manifest_file=manifest_path)
data_set = boc.get_ophys_experiment_data(501498760)

# read in the natural movie one clip
movie = data_set.get_stimulus_template('natural_movie_one')
#%%
# display a random frame for reference
frame = 430
plt.imshow(movie[frame,:,:], cmap='gray')
plt.axis('off')
plt.title('frame %d' % frame)
plt.show()
frame = 430
plt.imsave('./plots/natural_movie_one_frame_{}.jpg'.format(frame), movie[frame,:,:], cmap='gray')
# %%
scenes = data_set.get_stimulus_template('natural_scenes')
ind = 30
plt.imsave('./plots/natural_scene_ind_{}.jpg'.format(ind), scenes[ind,:,:], cmap='gray')
#%%
####################### raster plot
def get_raster_data(mouse_id, s_lengths, blank_width=50):
  # stimulus_id = 6
  directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
  total_sequence = np.zeros((len(area_dict[session_ids[mouse_id]]), 0))
  # s_lengths = [800, 500, 500, 250, 500, 250, 1000, 2000]
  # s_lengths = [250] * 8
  stimulus2plot = [stimulus_names[i] for i in [0, 1, 3, 4, 5, 6]]
  for s_ind, stimulus_name in enumerate(stimulus2plot):
    print(stimulus_name)
    sequences = load_npz_3d(os.path.join(directory, str(session_ids[mouse_id]) + '_' + stimulus_name + '.npz'))
    total_sequence = np.concatenate((total_sequence, sequences[:, 0, :s_lengths[s_ind]]), 1)
    if s_ind < len(stimulus2plot) - 1:
      total_sequence = np.concatenate((total_sequence, np.zeros((total_sequence.shape[0], blank_width))), 1)
  # node_area = active_area_dict[str(session_ids[mouse_id])]
  node_area = area_dict[str(session_ids[mouse_id])]
  areas = list(node_area.values())
  areas_num = [(np.array(areas)==a).sum() for a in visual_regions]
  areas_start_pos = list(np.insert(np.cumsum(areas_num)[:-1], 0, 0))
  sequence_by_area = {area:[ind for ind, a in node_area.items() if a == area] for area in visual_regions}
  return total_sequence, areas_num, areas_start_pos, sequence_by_area

mouse_id = 6
s_lengths  = [800, 250, 370, 250, 250, 1000]
blank_width = 50
total_sequence, areas_num, areas_start_pos, sequence_by_area = get_raster_data(mouse_id, s_lengths, blank_width)
# %%
def plot_raster(mouse_id, total_sequence, areas_num, areas_start_pos, sequence_by_area):
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
    # for i in range(len(s_locs_frac)-1):
    #   xmin, xmax = s_locs_frac[i] * (xgmax-xgmin) + xgmin, s_locs_frac[i+1] * (xgmax-xgmin) + xgmin
    for scale1, scale2 in zip(s_loc_frac1, s_loc_frac2):
      xmin, xmax = scale1 * (xgmax-xgmin) + xgmin, scale2 * (xgmax-xgmin) + xgmin
      # plt.gca().axhspan(loc1, loc2, xmin=xmin, xmax=xmax, facecolor=region_colors[areas_start_pos.index(loc1)], alpha=alpha)
      plt.gca().axhspan(loc1, loc2, xmin=xmin, xmax=xmax, facecolor=region_colors[areas_start_pos.index(loc1)], alpha=alpha_list[areas_start_pos.index(loc1)])
  #### add box
  # for stimulus_ind in range(len(stimulus2plot)):
  #   plt.gca().add_patch(Rectangle((s_loc1[stimulus_ind]/1000,0),(s_lengths[stimulus_ind]-1)/1000,sorted_sample_seq.shape[0],linewidth=4,edgecolor='k',alpha=.3,facecolor='none')) # region_colors[region_ind]
  plt.savefig('./plots/raster_{}.pdf'.format(session_ids[mouse_id]), transparent=True)
  # plt.show()

plot_raster(mouse_id, total_sequence, areas_num, areas_start_pos, sequence_by_area)
#%%
##################### plot best CCG sequence
################ plot example significant ccg for highland
################ find highest confidence for each duration
def get_best_ccg(directory, n=7, sign='pos'):
  files = os.listdir(directory)
  files.sort(key=lambda x:int(x[:9]))
  file2plot, inds_2plot, duration2plot, conf2plot = [], [], [], []
  for file in files:
    if ('_bl' not in file) and ('gabors' not in file) and ('flashes' not in file): #   and ('drifting_gratings' in file) and ('719161530' in file) and '719161530' in file and ('static_gratings' in file or 'gabors' in file) or 'flashes' in file
      print(file)
      mouseID = file.split('_')[0]
      stimulus_name = file.replace('.npz', '').replace(mouseID + '_', '')
      try:
        ccg = load_npz_3d(os.path.join(directory, file))
      except:
        ccg = load_sparse_npz(os.path.join(directory, file))
      try:
        ccg_jittered = load_npz_3d(os.path.join(directory, file.replace('.npz', '_bl.npz')))
      except:
        ccg_jittered = load_sparse_npz(os.path.join(directory, file.replace('.npz', '_bl.npz')))
      ccg_corrected = ccg - ccg_jittered
      sig_dir = './data/ecephys_cache_dir/sessions/adj_mat_ccg_highland_corrected/'
      significant_ccg = load_npz_3d(os.path.join(sig_dir, file))
      confidence_level = load_npz_3d(os.path.join(sig_dir, file.replace('.npz', '_confidence.npz')))
      significant_offset = load_npz_3d(os.path.join(sig_dir, file.replace('.npz', '_offset.npz')))
      significant_duration = load_npz_3d(os.path.join(sig_dir, file.replace('.npz', '_duration.npz')))
      significant_inds = list(zip(*np.where(~np.isnan(significant_ccg))))
      np.random.shuffle(significant_inds)
      
      for duration in np.arange(0,12,1):
        duration_inds = np.where((significant_duration==duration) & (significant_offset>0))
        if len(duration_inds[0]):
          if sign == 'pos':
            best = np.max(confidence_level[duration_inds])
            if best >= n:
              ind = np.argmax(confidence_level[duration_inds])
              file2plot.append(file)
              inds_2plot.append([duration_inds[0][ind], duration_inds[1][ind]])
              duration2plot.append(significant_duration[(duration_inds[0][ind], duration_inds[1][ind])])
              conf2plot.append(best)
          elif sign == 'neg':
            best = np.min(confidence_level[duration_inds])
            if best <= -n:
              ind = np.argmin(confidence_level[duration_inds])
              file2plot.append(file)
              inds_2plot.append([duration_inds[0][ind], duration_inds[1][ind]])
              duration2plot.append(significant_duration[(duration_inds[0][ind], duration_inds[1][ind])])
              conf2plot.append(best)
            # inds_2plot.append([duration_inds[0][indx[1]], duration_inds[1][indx[1]]])
  file2plot, inds_2plot, duration2plot, conf2plot = np.array(file2plot), np.array(inds_2plot), np.array(duration2plot), np.array(conf2plot)
  uniq_durations = np.unique(duration2plot)
  h_indx, h_duration, h_file, h_confidence = [], [], [], []
  for duration in sorted(uniq_durations):
    locs = np.where(np.array(duration2plot)==duration)[0]
    if sign == 'pos':
      if duration == 0:
        ind = np.argpartition(conf2plot[locs], -40)[-40]
      else:
        ind = np.argmax(conf2plot[locs])
    elif sign == 'neg':
      ind = np.argmin(conf2plot[locs])
    h_indx.append(inds_2plot[locs[ind]])
    h_duration.append(duration)
    h_file.append(file2plot[locs[ind]])
    h_confidence.append(conf2plot[locs[ind]])
  # return file2plot, inds_2plot, duration2plot, conf2plot
  return h_indx, h_file

directory = './data/ecephys_cache_dir/sessions/adj_mat_ccg_corrected/'
n = 4
pos_h_indx, pos_h_file = get_best_ccg(directory, n=n, sign='pos')
neg_h_indx, neg_h_file = get_best_ccg(directory, n=n, sign='neg')
#%%
def plot_multi_best_ccg(h_indx, h_file, sign='pos', window=100, length=100):
  fig = plt.figure(figsize=(8*4, 5*3))
  for ind, (row_a, row_b) in enumerate(h_indx):
    ax = plt.subplot(3, 4, ind+1)
    file = h_file[ind]
    try: 
      ccg = load_npz_3d(os.path.join(directory, file))
    except:
      ccg = load_sparse_npz(os.path.join(directory, file))
    try:
      ccg_jittered = load_npz_3d(os.path.join(directory, file.replace('.npz', '_bl.npz')))
    except:
      ccg_jittered = load_sparse_npz(os.path.join(directory, file.replace('.npz', '_bl.npz')))
    ccg_corrected = ccg - ccg_jittered
    sig_dir = './data/ecephys_cache_dir/sessions/adj_mat_ccg_highland_corrected/'
    significant_ccg = load_npz_3d(os.path.join(sig_dir, file))
    significant_offset = load_npz_3d(os.path.join(sig_dir, file.replace('.npz', '_offset.npz')))
    significant_duration = load_npz_3d(os.path.join(sig_dir, file.replace('.npz', '_duration.npz')))
    highland_lag = range(int(significant_offset[row_a,row_b]), int(significant_offset[row_a,row_b]+significant_duration[row_a,row_b]+1))
    plt.plot(np.arange(window+1)[:length], ccg_corrected[row_a, row_b][:length], linewidth=3, color='k')
    plt.plot(highland_lag, ccg_corrected[row_a, row_b, highland_lag], color='firebrick', marker='o', linewidth=3, markersize=10, alpha=0.8)
    if ind % 4 == 0:
      plt.ylabel('CCG corrected', size=25)
    if ind // 4 == 3 - 1:
      plt.xlabel('time lag (ms)', size=25)
    plt.title(significant_duration[row_a,row_b], fontsize=25)
    plt.xticks(fontsize=22) #, weight='bold'
    plt.yticks(fontsize=22) # , weight='bold'
    for axis in ['bottom', 'left']:
      ax.spines[axis].set_linewidth(1.5)
      ax.spines[axis].set_color('0.2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=1.5)
  plt.savefig('./plots/best_ccg_{}.jpg'.format(sign))
  # plt.show()

plot_multi_best_ccg(pos_h_indx, pos_h_file, sign='pos', window=100, length=100)
plot_multi_best_ccg(neg_h_indx, neg_h_file, sign='neg', window=100, length=100)
#%%
def plot_multi_best_ccg_smoothed(h_indx, h_file, sign='pos'):
  fig = plt.figure(figsize=(8*4, 5*3))
  for ind, (row_a, row_b) in enumerate(h_indx):
    ax = plt.subplot(3, 4, ind+1)
    file = h_file[ind]
    try: 
      ccg = load_npz_3d(os.path.join(directory, file))
    except:
      ccg = load_sparse_npz(os.path.join(directory, file))
    try:
      ccg_jittered = load_npz_3d(os.path.join(directory, file.replace('.npz', '_bl.npz')))
    except:
      ccg_jittered = load_sparse_npz(os.path.join(directory, file.replace('.npz', '_bl.npz')))
    ccg_corrected = ccg - ccg_jittered
    sig_dir = './data/ecephys_cache_dir/sessions/adj_mat_ccg_highland_corrected/'
    significant_ccg = load_npz_3d(os.path.join(sig_dir, file))
    significant_offset = load_npz_3d(os.path.join(sig_dir, file.replace('.npz', '_offset.npz')))
    significant_duration = load_npz_3d(os.path.join(sig_dir, file.replace('.npz', '_duration.npz')))
    
    filter = np.array([1]).repeat(significant_duration[row_a,row_b]+1) # sum instead of mean
    ccg_plot = signal.convolve(ccg_corrected[row_a, row_b], filter, mode='valid', method='fft')
    highland_lag = np.array([int(significant_offset[row_a,row_b])])
    plt.plot(np.arange(len(ccg_plot)), ccg_plot, linewidth=3, color='k')
    plt.plot(highland_lag, ccg_plot[highland_lag], color='firebrick', marker='o', linewidth=3, markersize=10, alpha=0.8)
    if ind % 4 == 0:
      plt.ylabel('signigicant CCG corrected', size=20)
    if ind // 4 == 3 - 1:
      plt.xlabel('time lag (ms)', size=20)
    plt.title(significant_duration[row_a,row_b], fontsize=25)
    plt.xticks(fontsize=22) #, weight='bold'
    plt.yticks(fontsize=22) # , weight='bold'
    for axis in ['bottom', 'left']:
      ax.spines[axis].set_linewidth(1.5)
      ax.spines[axis].set_color('0.2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=1.5)
  plt.savefig('./plots/best_ccg_smoothed_{}.jpg'.format(sign))
  # plt.show()

plot_multi_best_ccg_smoothed(pos_h_indx, pos_h_file, sign='pos')
plot_multi_best_ccg_smoothed(neg_h_indx, neg_h_file, sign='neg')
#%%
def plot_best_ccg(h_indx, h_file, ind, sign='pos', window=100, scalebar=False):
  pcolor = 'firebrick' if sign == 'pos' else 'blue'
  linewidth = 4.
  fig, axes = plt.subplots(1, 2, figsize=(16*2, 4), sharex=True, sharey=True)
  row_a, row_b = h_indx[ind]
  file = h_file[ind]
  try: 
    ccg = load_npz_3d(os.path.join(directory, file))
  except:
    ccg = load_sparse_npz(os.path.join(directory, file))
  try:
    ccg_jittered = load_npz_3d(os.path.join(directory, file.replace('.npz', '_bl.npz')))
  except:
    ccg_jittered = load_sparse_npz(os.path.join(directory, file.replace('.npz', '_bl.npz')))
  ccg_corrected = ccg - ccg_jittered
  sig_dir = './data/ecephys_cache_dir/sessions/adj_mat_ccg_highland_corrected/'
  significant_offset = load_npz_3d(os.path.join(sig_dir, file.replace('.npz', '_offset.npz')))
  significant_duration = load_npz_3d(os.path.join(sig_dir, file.replace('.npz', '_duration.npz')))
  highland_lag = range(int(significant_offset[row_a,row_b]), int(significant_offset[row_a,row_b]+significant_duration[row_a,row_b]+1))
  axes[0].plot(np.arange(window+1), ccg_corrected[row_a, row_b], linewidth=linewidth, color='k')
  if ind:
    axes[0].plot(highland_lag, ccg_corrected[row_a, row_b, highland_lag], color=pcolor, marker=' ', linewidth=linewidth+2, markersize=10, alpha=0.8)
  filter = np.array([1/(significant_duration[row_a,row_b]+1)]).repeat(significant_duration[row_a,row_b]+1) # mean instead of sum
  ccg_plot = signal.convolve(ccg_corrected[row_a, row_b], filter, mode='valid', method='fft')
  highland_lag = np.array([int(significant_offset[row_a,row_b])])
  axes[1].plot(np.arange(len(ccg_plot)), ccg_plot, linewidth=linewidth, color='k')
  if ind:
    axes[1].plot(highland_lag, ccg_plot[highland_lag], color=pcolor, marker=' ', linewidth=linewidth+2, markersize=10, alpha=1.)
  # plt.ylabel(r'$CCG_{corrected}$', size=25)
  # plt.xlabel('time lag (ms)', size=25)
  # axes[0].xaxis.set_tick_params(labelsize=30)
  # axes[0].yaxis.set_tick_params(labelsize=30)
  # axes[0].set_xticks([0, 100])
  axes[0].set_yticks([])
  axes[0].set_xlim([0, 100])
  if scalebar:
    fontprops = fm.FontProperties(size=40)
    size_v = (ccg_corrected[row_a, row_b].max()-ccg_corrected[row_a, row_b].min())/30
    scalebar = AnchoredSizeBar(axes[0].transData,
                              100, '100 ms', 'lower center',
                              borderpad=0,
                              pad=-1.4,
                              sep=5,
                              color='k',
                              frameon=False,
                              size_vertical=size_v,
                              fontproperties=fontprops)

    axes[0].add_artist(scalebar)
  axes[0].set_axis_off()
  axes[1].set_axis_off()
  plt.tight_layout()
  plt.subplots_adjust(left=0.,
                    bottom=0.,
                    right=1.,
                    top=1.,
                    wspace=1.2)
  plt.savefig('./plots/best_ccg_{}_{}.pdf'.format(sign, ind), transparent=True)
  # plt.show()

plot_best_ccg(pos_h_indx, pos_h_file, 0, sign='pos', window=100, scalebar=False)
plot_best_ccg(neg_h_indx, neg_h_file, 0, sign='neg', window=100, scalebar=False)
plot_best_ccg(pos_h_indx, pos_h_file, 9, sign='pos', window=100, scalebar=False)
plot_best_ccg(neg_h_indx, neg_h_file, 3, sign='neg', window=100, scalebar=False)
#%%
# Figure 1D
############### plot connectivity matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

def get_connectivity_data(G_dict, row_ind, col_ind):
  rows, cols = get_rowcol(G_dict)
  row, col = rows[row_ind], cols[col_ind]
  directory = './data/ecephys_cache_dir/sessions/adj_mat_ccg_corrected/'
  file = row + '_' + col + '.npz'
  try: 
    ccg = load_npz_3d(os.path.join(directory, file))
  except:
    ccg = load_sparse_npz(os.path.join(directory, file))
  try:
    ccg_jittered = load_npz_3d(os.path.join(directory, file.replace('.npz', '_bl.npz')))
  except:
    ccg_jittered = load_sparse_npz(os.path.join(directory, file.replace('.npz', '_bl.npz')))
  ccg_corrected = ccg - ccg_jittered
  ccg_zscore, ccg_value = ccg2zscore(ccg_corrected, max_duration=11, maxlag=12)
  return ccg_zscore, ccg_value

row_ind, col_ind = 5, 7
ccg_zscore, ccg_value = get_connectivity_data(G_ccg_dict, row_ind, col_ind)
#%%
# Figure 1D
###################### plot connectivity matrix from left to right, from bottom to up
def plot_connectivity_matrix_annotation(G_dict, row_ind, col_ind, ccg_zscore, ccg_value, weight=None, ratio=15):
  rows, cols = get_rowcol(G_dict)
  row, col = rows[row_ind], cols[col_ind]
  G = G_dict[row][col]
  nrow = 2
  ncol = 2
  active_area = active_area_dict[row]
  ordered_nodes = [] # order nodes based on hierarchical order
  region_size = np.zeros(len(visual_regions))
  for r_ind, region in enumerate(visual_regions):
    for node in active_area:
      if active_area[node] == region:
        ordered_nodes.append(node)
        region_size[r_ind] += 1
  A = nx.to_numpy_array(G, nodelist=ordered_nodes, weight='weight').T
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
  gs = gridspec.GridSpec(nrow, ncol, width_ratios=[1, ratio-1], height_ratios=[1, ratio-1],
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
    indx = np.array(cortical_inds[row])
    mat = ccg_zscore[indx[:,None], indx]
    reordered_nodes = np.array([node2idx[node] for node in ordered_nodes])
    mat = mat[reordered_nodes[:,None], reordered_nodes]
    mat = np.flip(mat, 1) # from left to right
    vmin = np.percentile(mat[mat<0], .5)
    vmax = np.percentile(mat[mat>0], 99.2)
  elif weight=='weight':
    indx = np.array(cortical_inds[row])
    mat = ccg_value[indx[:,None], indx]
    reordered_nodes = np.array([node2idx[node] for node in ordered_nodes])
    mat = mat[reordered_nodes[:,None], reordered_nodes]
    mat = np.flip(mat, 1) # from left to right
    vmin = np.percentile(mat[mat<0], 2)
    vmax = np.percentile(mat[mat>0], 98)
  cmap = plt.cm.coolwarm
  cmap = plt.cm.Spectral
  cmap = plt.cm.RdBu_r
  #   vmin, vmax = -5, 5
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
  figname = './plots/connectivity_matrix_{}_{}.pdf' if weight is None else './plots/connectivity_matrix_{}'.format(weight) + '_{}_{}.pdf'
  plt.savefig(figname.format(row, col), transparent=True)
  # plt.show()

weight=None
ratio = 15
plot_connectivity_matrix_annotation(G_ccg_dict, row_ind, col_ind, ccg_zscore, ccg_value, weight=None, ratio=ratio)
# plot_connectivity_matrix_annotation(G_ccg_dict, row_ind, col_ind, ccg_zscore, ccg_value, weight='confidence', ratio=ratio)
# plot_connectivity_matrix_annotation(G_ccg_dict, row_ind, col_ind, ccg_zscore, ccg_value, weight='weight', ratio=ratio)
#%%
# Figure S1A
row_ind = 5
col_inds = [0, 2, 3, 4, 5, 7]
weight=None
ratio = 15
for col_ind in col_inds:
  ccg_zscore, ccg_value = get_connectivity_data(G_ccg_dict, row_ind, col_ind)
  plot_connectivity_matrix_annotation(G_ccg_dict, row_ind, col_ind, ccg_zscore, ccg_value, weight=None, ratio=ratio)
#%%
# Figure S1C
def plot_intra_inter_scatter_G(G_dict, name, active_area_dict, remove_0=False):
  rows, cols = get_rowcol(G_dict)
  num_row, num_col = len(session2keep), len(cols)
  fig = plt.figure(figsize=(5, 5))
  df = pd.DataFrame(columns=['data', 'type', 'col'])
  for col_ind, col in enumerate(cols):
    print(col)
    for row_ind, row in enumerate(session2keep):
      intra_data, inter_data = [], []
      active_area = active_area_dict[row]
      G = G_dict[row][col].copy()
      for edge in G.edges():
        if active_area[edge[0]] == active_area[edge[1]]:
          intra_data.append(G[edge[0]][edge[1]][name])
        else:
          inter_data.append(G[edge[0]][edge[1]][name])
      if remove_0 is True:
        intra_data = [i for i in intra_data if i > 0]
        inter_data = [i for i in inter_data if i > 0]
      df = pd.concat([df, pd.DataFrame([[np.mean(intra_data), np.mean(inter_data), combine_stimulus(col)[1]]],
                    columns=['intra-region', 'inter-region', 'stimulus'])])
      # df = pd.concat([df, pd.DataFrame([[np.mean(intra_data), np.mean(inter_data), col]],
      #               columns=['intra-region', 'inter-region', 'col'])])
  df['intra-region'] = pd.to_numeric(df['intra-region'])
  df['inter-region'] = pd.to_numeric(df['inter-region'])
  # ax = sns.scatterplot(data=df, x='intra-region', y='inter-region', hue='stimulus type', palette=stimulus_type_color, s=100, alpha=.5)
  ax = plt.gca()
  for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
    x = df[df['stimulus']==combined_stimulus_name]['intra-region']
    y = df[df['stimulus']==combined_stimulus_name]['inter-region']
    ax.scatter(x, y, ec='.1', fc='none', marker=stimulus2marker[combined_stimulus_name], s=10*marker_size_dict[stimulus2marker[combined_stimulus_name]], alpha=.9, linewidths=1.5)
    # ax.scatter(x, y, facecolors=stimulus_type_color[st_ind], edgecolors=stimulus_type_color[st_ind], label=stimulus_type, alpha=.8)
  xliml, xlimu = ax.get_xlim()
  plt.plot(np.arange(xliml, xlimu, 0.1), np.arange(xliml, xlimu, 0.1), 'k--', alpha=0.4)
  plt.xticks(fontsize=18) #, weight='bold'
  plt.yticks(fontsize=18) # , weight='bold'
  name2label = {'offset':r'$\tau$', 'duration':r'$D$', 'delay':r'$\tau+D$'}
  plt.xlabel('Within-area {} (ms)'.format(name2label[name]))
  ylabel = 'Cross-area {} (ms)'.format(name2label[name])
  plt.ylabel(ylabel)
  # plt.title(name, fontsize=25)
  ax.set_xlabel(ax.get_xlabel(), fontsize=20, color='k') #, weight='bold'
  ax.set_ylabel(ax.get_ylabel(), fontsize=20, color='k') #, weight='bold'
  for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(1.5)
    ax.spines[axis].set_color('0.2')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.tick_params(width=1.5)
  # plt.xlim(0, 12)
  # plt.legend([])
  # plt.xticks(rotation=90)
  plt.tight_layout()
  # plt.show()
  figname = './plots/intra_inter_scatter_{}.pdf' if not remove_0 else './plots/intra_inter_no0_scatter_{}.pdf'
  plt.savefig(figname.format(name), transparent=True)

plot_intra_inter_scatter_G(S_ccg_dict, 'delay', active_area_dict, False)
# plot_intra_inter_scatter_G(S_ccg_dict, 'delay', active_area_dict, True)
plot_intra_inter_scatter_G(S_ccg_dict, 'offset', active_area_dict, False)
# plot_intra_inter_scatter_G(S_ccg_dict, 'offset', active_area_dict, True)
plot_intra_inter_scatter_G(S_ccg_dict, 'duration', active_area_dict, False)
# plot_intra_inter_scatter_G(S_ccg_dict, 'duration', active_area_dict, True)
#%%
# Figure 1 E
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
#%%
############################ old version of excitatory VS inhibitory connections
def alt_bands(ax=None):
  ax = ax or plt.gca()
  x_left, x_right = ax.get_xlim()
  locs = ax.get_xticks().astype(float)
  locs -= .5
  locs = np.concatenate((locs, [x_right]))
  
  # type_loc1, type_loc2 = locs[[0, 1, 3, 5]], locs[[1, 3, 5, 8]] # for all stimuli
  type_loc1, type_loc2 = locs[[0, 1, 2, 4]], locs[[1, 2, 4, 6]] # for combined stimuli
  for loc1, loc2 in zip(type_loc1, type_loc2):
    ax.axvspan(loc1, loc2, facecolor=stimulus_type_color[type_loc1.tolist().index(loc1)], alpha=0.2)
  ax.set_xlim(x_left, x_right)

def plot_ex_in_bar(G_dict, measure, n, density=False):
  df = pd.DataFrame()
  rows, cols = get_rowcol(G_dict)
  for col_ind, col in enumerate(cols):
    print(col)
    combined_stimulus_name = combine_stimulus(col)[1]
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
    df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(ex_data)[:,None], np.array(['excitatory'] * len(ex_data))[:,None], np.array([combined_stimulus_name] * len(ex_data))[:,None]), 1), columns=['number of connections', 'type', 'stimulus']), 
                pd.DataFrame(np.concatenate((np.array(in_data)[:,None], np.array(['inhibitory'] * len(in_data))[:,None], np.array([combined_stimulus_name] * len(in_data))[:,None]), 1), columns=['number of connections', 'type', 'stimulus'])], ignore_index=True)
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
  ax = sns.barplot(x="stimulus", y=y, hue="type",  data=df, palette=barcolors, errorbar="sd",  edgecolor="black", errcolor="black", errwidth=1.5, capsize = 0.1, alpha=0.5)
  sns.stripplot(x="stimulus", y=y, hue="type", palette=barcolors, data=df, dodge=True, alpha=0.6, ax=ax)
  # remove extra legend handles
  handles, labels = ax.get_legend_handles_labels()
  ax.legend(handles[2:], labels[2:], title='', bbox_to_anchor=(.7, 1.), loc='upper left', fontsize=20, frameon=False)
  # plt.setp(ax.get_legend().get_texts()) #, weight='bold'
  
  plt.yticks(fontsize=18) #,  weight='bold'
  plt.ylabel(y.capitalize())
  ax.set_ylabel(ax.get_ylabel(), fontsize=20,color='k') #, weight='bold'
  for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2.5)
    ax.spines[axis].set_color('0.2')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.tick_params(width=2.5)
  # plt.yscale('log')
  ax.set(xlabel=None)
  alt_bands(ax)
  plt.xticks([])
  # mean_links = df.groupby(['stimulus', 'type']).mean().reset_index().groupby('stimulus').sum()
  # for i, combined_stimulus_name in enumerate(combined_stimulus_names):
  #   plt.hlines(y=mean_links.loc[combined_stimulus_name], xmin=(i - .18), xmax=(i + .18), color='white', linewidth=3) # , linestyles=(0, (1,1))
  plt.tight_layout()
  figname = './plots/box_ex_in_num_{}_{}fold.pdf' if not density else './plots/box_ex_in_density_{}_{}fold.pdf'
  # plt.savefig('./plots/violin_intra_inter_{}_{}fold.jpg'.format(measure, n))
  plt.savefig(figname.format(measure, n), transparent=True)
######################## excitaroty link VS inhibitory link box
plot_ex_in_bar(S_ccg_dict, measure, n, density=False)
#%%
# Figure 1 F
######################### new scatter plots
def scatter_dataVSdensity_new(G_dict, area_dict, regions, name='intra'):
  rows, cols = get_rowcol(G_dict)
  fig, ax = plt.subplots(figsize=(5, 5))
  X, Y = [], []
  df = pd.DataFrame()
  region_connection = np.zeros((len(session2keep), len(cols), len(regions), len(regions)))
  for col_ind, col in enumerate(cols):
    # print(col)
    intra_data, inter_data, density_data, ex_data, in_data, cluster_data = [], [], [], [], [], []
    for row_ind, row in enumerate(session2keep):
      G = G_dict[row][col]
      nodes = list(G.nodes())
      node_area = {key: area_dict[row][key] for key in nodes}
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
      cluster_data.append(calculate_directed_metric(G, 'clustering'))
    X += density_data
    if name == 'intra':
      Y += intra_data
    elif name == 'ex':
      Y += ex_data
    elif name == 'cluster':
      Y += cluster_data
    df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(intra_data)[:,None], np.array(inter_data)[:,None], np.array(ex_data)[:,None], np.array(cluster_data)[:,None], np.array(density_data)[:,None], np.array([combine_stimulus(col)[1]] * len(intra_data))[:,None]), 1), columns=['ratio of intra-region connections', 'ratio of inter-region connections', 'ratio of excitatory connections', 'cluster', 'density', 'stimulus'])], ignore_index=True)
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
  plt.savefig(f'./plots/{name}_density.pdf', transparent=True)

scatter_dataVSdensity_new(S_ccg_dict, area_dict, visual_regions, name='intra')
scatter_dataVSdensity_new(S_ccg_dict, area_dict, visual_regions, name='ex')
scatter_dataVSdensity_new(S_ccg_dict, area_dict, visual_regions, name='cluster')
#%%
######################### old version of scatterplot
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
  for st_ind, stype in enumerate(stimulus_by_type):
    x = df[df['stimulus'].isin(stype)]['density'].values
    if name == 'intra':
      y = df[df['stimulus'].isin(stype)]['ratio of intra-region connections'].values
    elif name == 'ex':
      y = df[df['stimulus'].isin(stype)]['ratio of excitatory connections'].values
    ax.scatter(x, y, facecolors=stimulus_type_color[st_ind], edgecolors=stimulus_type_color[st_ind], label=stimulus_types[st_ind], alpha=.8)
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
  ax.plot(X, line, color='.4', linestyle='-', alpha=.5) # (5,(10,3))
  # ax.scatter(X, Y, facecolors='none', edgecolors='.2', alpha=.6)
  ax.text(locx, locy, text, horizontalalignment='center',
     verticalalignment='center', transform=ax.transAxes, fontsize=18)
  if name == 'intra':
    plt.legend(loc='lower right', fontsize=16, frameon=False)
  plt.xticks(fontsize=20) #, weight='bold'
  plt.yticks(fontsize=20) # , weight='bold'
  plt.xlabel('Density')
  if name == 'intra':
    ylabel = 'Intra-region fraction'
  elif name == 'ex':
    ylabel = 'Excitatory fraction'
    plt.xscale('log')
  plt.ylabel(ylabel)
  ax.set_xlabel(ax.get_xlabel(), fontsize=20,color='k') #, weight='bold'
  ax.set_ylabel(ax.get_ylabel(), fontsize=20,color='k') #, weight='bold'
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
# Figure S3
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
# Figure S3
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
# Figure S3
def plot_FR_links_region(data, regions, dataname):
  if dataname == 'FR':
    name = 'Firing rate (Hz)'
  elif dataname == 'link':
    name = 'degree'
  df = pd.DataFrame()
  for region in regions:
    for stimulus_name in stimulus_names:
      for se_ind in session2keep:
        sub_data = np.array(data[se_ind][stimulus_name][region])
        df = pd.concat([df, pd.DataFrame(np.concatenate((sub_data[:,None], np.array([combine_stimulus(stimulus_name)[1]] * len(sub_data))[:,None], np.array([region] * len(sub_data))[:,None]), 1), columns=[name, 'stimulus', 'region'])], ignore_index=True)
  df[name] = pd.to_numeric(df[name])
  # return df
  fig, ax = plt.subplots(1,1,figsize=(8, 10))
  colors_transparency = [transparent_rgb(colors.to_rgb(color), [1,1,1], alpha=.6) if c_ind <=2 else transparent_rgb(colors.to_rgb(color), [1,1,1], alpha=1.) for c_ind, color in enumerate(region_colors)]
  medianprops = dict(linestyle='-', linewidth=2.5)
  boxprops = dict(linewidth=3)
  boxplot = sns.boxplot(y="stimulus", x=name, hue="region", hue_order=regions, data=df[(np.abs(stats.zscore(df[name])) < 2)], orient='h', palette=colors_transparency, showfliers=False, boxprops=boxprops, medianprops=medianprops) # , boxprops=dict(alpha=.6)
  # ax = sns.violinplot(x="stimulus", y=name, inner='box', cut=0, hue="region", scale="count", hue_order=regions, data=df[(np.abs(stats.zscore(df[name])) < 2)], palette=colors_transparency) # , boxprops=dict(alpha=.6)
  
  sns.stripplot(
      y="stimulus", 
      x=name, 
      hue="region",
      palette=colors_transparency,
      data=df[(np.abs(stats.zscore(df[name])) < 2)], dodge=True, alpha=0.15
  )
  ax.set(ylabel=None)
  handles, labels = ax.get_legend_handles_labels()
  ax.legend(handles[:6], region_labels, title='', bbox_to_anchor=(.9, 1.), loc='upper left', fontsize=15, frameon=False)
  # plt.setp(ax.get_legend().get_texts())
  # nolinebreak = [name.replace('\n', ' ') for name in combined_stimulus_names]
  plt.yticks(ticks=range(len(combined_stimulus_names)), labels=combined_stimulus_names, fontsize=25, rotation=90, va='center')
  plt.xticks(fontsize=20)
  # plt.ylabel(y)
  ax.set_xlabel(ax.get_xlabel(), fontsize=25,color='k')
  for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(1.5)
    ax.spines[axis].set_color('0.2')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.tick_params(width=2.5)
  ax.yaxis.set_tick_params(length=0)
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
    # patch.set_linewidth(3)
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
    legpatch.set_linewidth(3)
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
    ax.set_title(stimulus_labels[s_ind].replace('\n', ' '), fontsize=16, color=stimulus_type_color[stimulus2stype(stimulus_names[s_ind])[0]])
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
# Figure S3
################### confidence level for difference between two correlations
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

def get_difference_intra_inter_r_stimulus(FR, intra_links, inter_links, regions, alpha):
  rows, cols = get_rowcol(FR)
  df = pd.DataFrame()
  # fig, axes = plt.subplots(1, 4, figsize=(20, 5))
  # for i in range(len(axes)):
  #   ax = axes[i]
  for combined_stimulus_name in combined_stimulus_names:
    for r_ind, region in enumerate(regions):
      X, Y1, Y2 = [], [], []
      for row in session2keep:
        for col in cols:
          if combine_stimulus(col)[1] == combined_stimulus_name:
            X += FR[row][col][region]
            Y1 += intra_links[row][col][region]
            Y2 += inter_links[row][col][region]
      # ax.scatter(X, Y, facecolors='none', edgecolors='.2', alpha=.5, label=region)
      X, Y1, Y2 = (list(t) for t in zip(*sorted(zip(X, Y1, Y2))))
      X, Y1, Y2 = np.array(X), np.array(Y1), np.array(Y2)
      _, _, r1, p1, _ = stats.linregress(X,Y1)
      _, _, r2, p2, _ = stats.linregress(X,Y2)
      _, _, r3, _, _ = stats.linregress(Y1,Y2)
      if r1 > r2:
        rh, rl = r1, r2
      else:
        rh, rl = r2, r1
      Ls = [] #, l1s, l2s, u1s, u2s , [], [], [], []
      alpha_list = [.0001, .001, .01, .05]
      for alpha in alpha_list:
        L, _ = MA_method(rh, rl, r3, len(X), alpha)
        # l1, u1 = r_confidence_interval(r1, len(X), alpha)
        # l2, u2 = r_confidence_interval(r2, len(X), alpha)
        Ls.append(L)
        # l1s.append(l1)
        # l2s.append(l2)
        # u1s.append(u1)
        # u2s.append(u2)
      Ls = np.array(Ls) #, l1s, l2s, u1s, u2s , np.array(l1s), np.array(l2s), np.array(u1s), np.array(u2s)
      loc = np.where(Ls > 0)[0]
      asterisk = '*' * (len(alpha_list) - loc[0]) if len(loc) else 'ns'

      # locl1, locu1 = np.where(l1s > 0)[0], np.where(u1s < 0)[0]
      # asterisk1 = '*' * (len(alpha_list) - (locl1 if len(locl1) else locu1)[0]) if len(locl1) or len(locu1) else 'ns'
      # locl2, locu2 = np.where(l2s > 0)[0], np.where(u2s < 0)[0]
      # asterisk2 = '*' * (len(alpha_list) - (locl2 if len(locl2) else locu2)[0]) if len(locl2) or len(locu2) else 'ns'
      
      asterisk1 = '*' * (len(alpha_list) - bisect(alpha_list, p1)) if len(alpha_list) > bisect(alpha_list, p1) else 'ns'
      asterisk2 = '*' * (len(alpha_list) - bisect(alpha_list, p2)) if len(alpha_list) > bisect(alpha_list, p2) else 'ns'

      df = pd.concat([df, pd.DataFrame([[combined_stimulus_name, region, r1, r2, asterisk, asterisk1, asterisk2]], columns=['stimulus', 'region', 'intra r', 'inter r', 'significance', 'intra significance', 'inter significance'])], ignore_index=True)
  return df

df = get_difference_intra_inter_r_stimulus(FR, intra_links, inter_links, visual_regions, alpha=.01)
df
# df[(df['significance'].isin(['*', '**', '***', '****'])) & (df['intra significance'].isin(['*', '**', '***', '****']))]
#%%
# Figure S3
################### barplot of correlation for intra/inter links VS FR with significance annotation
def plot_intra_inter_r_bar_significance(df, regions):
  fig, axes = plt.subplots(3, 2, figsize=(7*2, 5*3), sharey=True)
  palette = [[plt.cm.tab20b(i) for i in range(4)][i] for i in [0, 3]]
  for i, j in [(f,s) for f in range(axes.shape[0]) for s in range(axes.shape[1])]:
    cs_ind = i * axes.shape[1] + j
    combined_stimulus_name = combined_stimulus_names[cs_ind]
    ax = axes[i, j]
  # for s_ind, stimulus_type in enumerate(stimulus_types):
  #   ax = axes[s_ind]
    data = df[df['stimulus']==combined_stimulus_name]
    new_df = pd.DataFrame()
    for region in regions[::-1]:
      new_df = pd.concat([new_df, pd.DataFrame([[region, 'intra-region'] + data[data['region']==region][['intra r', 'significance', 'intra significance']].values[0].tolist()], columns=['region', 'type', 'r', 'significance', 'own significance'])], ignore_index=True)
      new_df = pd.concat([new_df, pd.DataFrame([[region, 'inter-region'] + data[data['region']==region][['inter r', 'significance', 'inter significance']].values[0].tolist()], columns=['region', 'type', 'r', 'significance', 'own significance'])], ignore_index=True)
    new_df['r'] = pd.to_numeric(new_df['r'])
    bar_plot = sns.barplot(data=new_df, x='region', y='r', hue='type', palette=palette, ax=ax) #, facecolor='white'
    # for patch0, patch1 in zip(ax.patches[:6], ax.patches[6:]):
    #   patch0.set_edgecolor('.2')
    #   patch1.set_edgecolor('.8')
    ax.legend().set_visible(False)
    # ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=25)
    ax.set_xticks(np.arange(0, len(regions), 1))
    ax.set_xticklabels(labels=region_labels[::-1], fontsize=25)
    ax.set_xlabel('')
    ax.set_title(combined_stimulus_name.replace('\n', ' '), fontsize=30)
    ylabel = r'$r$'
    ax.set_ylabel(ylabel, rotation=90)
    # ax.set_xscale('log')
    ax.set_xlabel(ax.get_xlabel(), fontsize=16,color='k') #, weight='bold'
    ax.set_ylabel(ax.get_ylabel(), fontsize=28, color='k') #, weight='bold'
    for axis in ['bottom', 'left']:
      ax.spines[axis].set_linewidth(1.5)
      ax.spines[axis].set_color('0.2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=1.5)
    
    for r_ind, region in enumerate(regions[::-1]):
      intra_r, intra_star  = new_df[(new_df['region']==region) & (new_df['type']=='intra-region')][['r', 'own significance']].values[0]
      inter_r, inter_star = new_df[(new_df['region']==region) & (new_df['type']=='inter-region')][['r', 'own significance']].values[0]
      diff_star = new_df[(new_df['region']==region) & (new_df['type']=='intra-region')].significance.values[0]
      intra_r = intra_r + .01 if intra_r > 0 else intra_r - .01
      inter_r = inter_r + .01 if inter_r > 0 else inter_r - .15
      annot_significance(intra_star, -.45 + r_ind, 0.05 + r_ind, intra_r, 22, ax=ax)
      annot_significance(inter_star, -.05 + r_ind, .45 + r_ind, inter_r, 22, ax=ax)
      annot_difference(diff_star, -.2 + r_ind, .2 + r_ind, max(intra_r, inter_r) + .2, .03, 1.5, 22, ax=ax)
    liml, limu = ax.get_ylim()
    ax.set_ylim([liml - .02, limu + .05])
  # plt.show()
  plt.tight_layout()
  plt.savefig('./plots/intra_inter_r_bar_significance.pdf', transparent=True)

plot_intra_inter_r_bar_significance(df, visual_regions)
#%%
def save_within_cross_region_legend():
  fig, ax = plt.subplots(1,1, figsize=(.4*(len(combined_stimulus_names)-1), .5))
  palette = [[plt.cm.tab20b(i) for i in range(4)][i] for i in [0, 3]]
  df = pd.DataFrame([[0,0,0],[0,0,1]], columns=['x', 'y', 'type'])
  barplot = sns.barplot(x='x', y='y', hue="type", hue_order=[0, 1], palette=palette, ec='none', linewidth=2., data=df, ax=ax, capsize=.05, width=0.6)
  handles, labels = ax.get_legend_handles_labels()
  legend = ax.legend(handles, ['within-area', 'cross-area'], title='', bbox_to_anchor=(0.1, -1.), handletextpad=0.3, loc='upper left', fontsize=25, frameon=False) # change legend order to within/cross similar module then cross random
  plt.axis('off')
  export_legend(legend, './plots/within_cross_region_legend.pdf')
  plt.tight_layout()
  # plt.savefig('./plots/stimuli_markers.pdf', transparent=True)
  plt.show()

save_within_cross_region_legend()
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
def plot_metrics(G_dict, metric_names):
  rows, cols = get_rowcol(G_dict)
  # metric_names = ['clustering', 'overall_reciprocity', 'flow_hierarchy', 'global_reaching_centrality']
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

# metric_names = ['clustering', 'overall_reciprocity', 'flow_hierarchy', 'global_reaching_centrality']
# df = plot_metrics(G_ccg_dict, metric_names)
#%%
################### plot clustering coefficient
def plot_metric(G_dict, metric_name):
  rows, cols = get_rowcol(G_dict)
  metric = np.empty((len(rows), len(cols)))
  metric[:] = np.nan
  df = pd.DataFrame()
  for col_ind, col in enumerate(cols):
    print(col)
    combined_stimulus_name = combine_stimulus(col)[1]
    for row_ind, row in enumerate(rows):
      G = G_dict[row][col]
      m = calculate_directed_metric(G, metric_name)
      metric[row_ind, col_ind] = m
    df = pd.concat([df, pd.DataFrame(np.concatenate((metric[:,col_ind][:,None], np.array([combined_stimulus_name] * len(rows))[:,None]), 1), columns=['metric', 'stimulus'])], ignore_index=True)
  df.metric = pd.to_numeric(df.metric)
  fig, ax = plt.subplots(1, 1, figsize=(10, 3))
  boxprops = dict(edgecolor='k')
  medianprops = dict(linestyle='-', linewidth=2, color='k')
  box = sns.boxplot(x="stimulus", y='metric', width=0.15, color='white', palette=combined_stimulus_colors, data=df, showfliers=True, boxprops=boxprops, medianprops=medianprops) # , boxprops=dict(alpha=.6)
  # Adjust boxplot and whisker line properties
  for p, artist in enumerate(ax.artists):
    # artist.set_edgecolor('blue')
    for q in range(p*6, p*6+6):
      line = ax.lines[q]
      # line.set_color('pink')
    ax.lines[p*6+2].set_xdata(ax.lines[p*6+4].get_xdata())
    ax.lines[p*6+3].set_xdata(ax.lines[p*6+4].get_xdata())
  box_patches = [patch for patch in ax.patches if type(patch) == mpl.patches.PathPatch]
  if len(box_patches) == 0:  # in matplotlib older than 3.5, the boxes are stored in ax.artists
    box_patches = ax.artists
  num_patches = len(box_patches)
  lines_per_boxplot = len(ax.lines) // num_patches
  for i, patch in enumerate(box_patches):
    # Set the linecolor on the patch to the facecolor, and set the facecolor to None
    col = patch.get_facecolor()
    patch.set_edgecolor(col)
    patch.set_facecolor('None')
    patch.set_linewidth(2)
    # Each box has associated Line2D objects (to make the whiskers, fliers, etc.)
    # Loop over them here, and use the same color as above
    for line in ax.lines[i * lines_per_boxplot: (i + 1) * lines_per_boxplot]:
      line.set_color(col)
      line.set_mfc(col)  # facecolor of fliers
      line.set_mec(col)  # edgecolor of fliers
  
  sns.stripplot(
      x="stimulus", 
      y='metric', 
      palette=combined_stimulus_colors,
      data=df, dodge=True, alpha=0.6
  )
  # ax.boxplot(weighted_purity, showfliers=False, patch_artist=True, boxprops=boxprops, medianprops=medianprops)
  # nolinebreak = [name.replace('\n', ' ') for name in combined_stimulus_names]
  # plt.xticks([])
  plt.xticks(ticks=range(len(combined_stimulus_names)), labels=combined_stimulus_names, fontsize=22) #, weight='bold'
  # plt.xticks(list(range(len(nolinebreak))), nolinebreak, rotation=90)
  # ax.xaxis.set_tick_params(labelsize=18)
  ax.yaxis.set_tick_params(labelsize=18)
  xlabel = ''
  ax.set_xlabel(xlabel)
  ylabel = 'Clustering coefficient'
  ax.set_ylabel(ylabel)
  # ax.set_xscale('log')
  # ax.set_xlabel(ax.get_xlabel(), fontsize=20,color='k') #, weight='bold'
  ax.set_ylabel(ax.get_ylabel(), fontsize=19,color='k') #, weight='bold'
  for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(1.5)
    ax.spines[axis].set_color('k')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.tick_params(width=1.5)
  ax.xaxis.set_tick_params(length=0)
  ax.yaxis.set_tick_params(length=0)
  plt.tight_layout()
  plt.savefig('./plots/{}.pdf'.format(metric_name))
  return df

metric_name = 'clustering'
df = plot_metric(G_ccg_dict, metric_name)
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
####################### Figure 2 #######################
########################################################
################# get optimal resolution that maximizes delta H

# def get_max_dH_resolution(rows, cols, resolution_list, metrics): 
#   max_reso_subs = np.zeros((len(rows), len(cols)))
#   Hamiltonian, subs_Hamiltonian = metrics['Hamiltonian'], metrics['subs Hamiltonian']
#   for row_ind, row in enumerate(rows):
#     print(row)
#     for col_ind, col in enumerate(cols):
#       metric_mean = Hamiltonian[row_ind, col_ind].mean(-1)
#       metric_subs = subs_Hamiltonian[row_ind, col_ind].mean(-1)
#       max_reso_subs[row_ind, col_ind] = resolution_list[np.argmax(metric_subs - metric_mean)]
#   return max_reso_subs

# for each community partition individually
def get_max_dH_resolution_eachtrial(rows, cols, resolution_list, metrics):
  Hamiltonian, subs_Hamiltonian = metrics['Hamiltonian'], metrics['subs Hamiltonian']
  max_reso_subs = np.zeros((len(rows), len(cols), Hamiltonian.shape[-1]))
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      for run in range(Hamiltonian.shape[-1]):
        metric_mean = Hamiltonian[row_ind, col_ind, :, run]
        metric_subs = subs_Hamiltonian[row_ind, col_ind, :, run]
        max_reso_subs[row_ind, col_ind, run] = resolution_list[np.argmax(metric_subs - metric_mean)]
  return max_reso_subs

rows, cols = get_rowcol(G_ccg_dict)
with open('./files/comms_dict.pkl', 'rb') as f:
  comms_dict = pickle.load(f)
with open('./files/metrics.pkl', 'rb') as f:
  metrics = pickle.load(f)

resolution_list = np.arange(0, 2.1, 0.1)
max_reso_subs = get_max_dH_resolution_eachtrial(rows, cols, resolution_list, metrics)

################# community with Hamiltonian
def get_max_pos_reso_eachtrial(G_ccg_dict, max_neg_reso):
  rows, cols = get_rowcol(G_ccg_dict)
  max_pos_reso = np.zeros((len(rows), len(cols), max_neg_reso.shape[-1]))
  for row_ind, row in enumerate(rows):
    for col_ind, col in enumerate(cols):
      G = G_ccg_dict[row][col]
      num_pos = sum([w for i,j,w in G.edges.data('weight') if w > 0])
      num_neg = sum([w for i,j,w in G.edges.data('weight') if w < 0])
      for run in range(max_neg_reso.shape[-1]):
        max_pos_reso[row_ind, col_ind, run] = abs(max_neg_reso[row_ind, col_ind, run] * num_neg / num_pos)
  return max_pos_reso

max_pos_reso_subs = get_max_pos_reso_eachtrial(G_ccg_dict, max_reso_subs)
#%%
################# max pos & neg resolution for Hamiltonian
start_time = time.time()
resolution_list = np.arange(0, 2.1, 0.1)
resolution_list = [round(reso, 2) for reso in resolution_list]
rows, cols = get_rowcol(G_ccg_dict)
with open('./files/comms_dict.pkl', 'rb') as f:
  comms_dict = pickle.load(f)
print("--- %s minutes" % ((time.time() - start_time)/60))
########################### save comms_dict with optimal pos and neg resolution
def save_best_comms_dict(G_dict, comms_dict, max_pos_neg_reso):
  rows, cols = get_rowcol(G_dict)
  best_comms_dict = {row:{} for row in rows}
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      max_reso = tuple(max_pos_neg_reso[row_ind][col_ind])
      comms_list = comms_dict[row][col][max_reso]
      best_comms_dict[row][col] = comms_list
  with open('./files/best_comms_dict.pkl', 'wb') as f:
    pickle.dump(best_comms_dict, f)
    
save_best_comms_dict(G_ccg_dict, comms_dict, max_reso_subs)
#%%
################# max pos & neg resolution for Hamiltonian
resolution_list = np.arange(0, 2.1, 0.1)
resolution_list = [round(reso, 2) for reso in resolution_list]
rows, cols = get_rowcol(G_ccg_dict)
with open('./files/best_comms_dict.pkl', 'rb') as f:
  best_comms_dict = pickle.load(f)
with open('./files/real_Hamiltonian.npy', 'rb') as f:
  real_Hamiltonian = np.load(f)
with open('./files/subs_Hamiltonian.npy', 'rb') as f:
  subs_Hamiltonian = np.load(f)
def get_max_dH_pos_neg_resolution(rows, cols, real_H, subs_H): 
  max_reso_subs = np.zeros((len(rows), len(cols), 2)) # last dim 0 for pos, 1 for neg
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      metric_mean = real_H[row_ind, col_ind].mean(-1)
      metric_subs = subs_H[row_ind, col_ind].mean(-1)
      pos_ind, neg_ind = np.unravel_index(np.argmax(metric_subs - metric_mean), np.array(metric_mean).shape)
      max_reso_subs[row_ind, col_ind] = resolution_list[pos_ind], resolution_list[neg_ind]
  return max_reso_subs

max_reso_subs = get_max_dH_pos_neg_resolution(rows, cols, real_Hamiltonian, subs_Hamiltonian)

with open('./files/signal_correlation_dict.pkl', 'rb') as f:
  signal_correlation_dict = pickle.load(f)
#%%
########################### visualize the delta_H
def plot_dH(rows, cols, real_H, subs_H, resolution_list): 
  fig, axes = plt.subplots(len(rows), len(cols), figsize=(3.5*len(cols), 3*len(rows)))
  # find minimum of minima & maximum of maxima
  minmin = (real_H.mean(-1) - subs_H.mean(-1)).min()
  maxmax = (real_H.mean(-1) - subs_H.mean(-1)).max()
  divnorm=colors.TwoSlopeNorm(vmin=minmin, vcenter=0, vmax=maxmax)
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      ax = axes[row_ind, col_ind]
      metric_mean = real_H[row_ind, col_ind].mean(-1)
      metric_subs = subs_H[row_ind, col_ind].mean(-1)
      im = ax.imshow((metric_mean - metric_subs).T, cmap="gist_heat_r") #, norm=divnorm
      plt.colorbar(im, ax=ax)
      ax.set_xticks(.5 + np.arange(0, len(resolution_list), 4))
      ax.set_xticklabels(labels=resolution_list[::4], fontsize=14)
      ax.set_yticks(.5 + np.arange(0, len(resolution_list), 4))
      ax.set_yticklabels(labels=resolution_list[::4], fontsize=14)
      if row_ind == len(rows) - 1:
        ax.set_xlabel(r'$\gamma^+$', fontsize=20,color='k') #, weight='bold'
      if col_ind == 0:
        ax.set_ylabel(r'$\gamma^-$', fontsize=20,color='k') #, weight='bold'
  # plt.show()
  plt.savefig('./plots/dH_all.jpg')
  
plot_dH(rows, cols, real_Hamiltonian, subs_Hamiltonian, resolution_list)
#%%
# Figure S4A
########################### visualize the delta_H
def plot_dH_row(rows, cols, row_ind, real_H, subs_H, resolution_list): 
  fig, axes = plt.subplots(1, len(combined_stimulus_names), figsize=(3.5*len(combined_stimulus_names), 3), tight_layout=True)
  # find minimum of minima & maximum of maxima
  minmin = (real_H.mean(-1) - subs_H.mean(-1)).min()
  maxmax = (real_H.mean(-1) - subs_H.mean(-1)).max()
  divnorm=colors.TwoSlopeNorm(vmin=minmin, vcenter=0, vmax=maxmax)
  row = rows[row_ind]
  for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
    ax = axes[cs_ind]
    col = combined_stimuli[cs_ind][0]
    col_ind = cols.index(col)
    metric_mean = real_H[row_ind, col_ind].mean(-1)
    metric_subs = subs_H[row_ind, col_ind].mean(-1)
    im = ax.imshow((metric_mean - metric_subs).T, cmap="gist_heat_r") #, norm=divnorm
    cbar = plt.colorbar(im, ax=ax,fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=14)
    ax.set_xticks(.5 + np.arange(0, len(resolution_list), 4))
    ax.set_xticklabels(labels=resolution_list[::4], fontsize=14)
    ax.set_yticks(.5 + np.arange(0, len(resolution_list), 4))
    ax.set_yticklabels(labels=resolution_list[::4], fontsize=14)
    ax.set_xlabel(r'$\gamma^+$', fontsize=20,color='k') #, weight='bold'
    if col_ind == 0:
      ax.set_ylabel(r'$\gamma^-$', fontsize=20,color='k') #, weight='bold'
  # plt.show()
  plt.savefig('./plots/dH_{}.pdf'.format(row), transparent=True)
  
plot_dH_row(rows, cols, 7, real_Hamiltonian, subs_Hamiltonian, resolution_list)
#%%
def stat_modular_structure_Hamiltonian_comms(G_dict, measure, n, resolution_list, max_pos_neg_reso=None, comms_dict=None, real_H=None, subs_H=None, max_method='none', cc=False):
  rows, cols = get_rowcol(G_dict)
  num_comm, hamiltonian, subs_Hamiltonian, num_lcomm, cov_lcomm = [np.full([len(rows), len(cols)], np.nan) for _ in range(5)]
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      origin_G = G_dict[row][col].copy()
      if cc:
        G = get_lcc(origin_G)
      else:
        G = origin_G
      max_reso = tuple(max_pos_neg_reso[row_ind][col_ind])
      comms_list = comms_dict[row][col][max_reso]
      hamiltonian[row_ind, col_ind] = real_H[row_ind, col_ind, resolution_list.index(max_reso[0]), resolution_list.index(max_reso[1])].mean()
      subs_Hamiltonian[row_ind, col_ind] = subs_H[row_ind, col_ind, resolution_list.index(max_reso[0]), resolution_list.index(max_reso[1])].mean()
      nc, nl, cl = [], [], []
      for comms in comms_list:
        lcc_comms = []
        for comm in comms:
          if set(comm) <= set(G.nodes()): # only for communities in the connected component
            lcc_comms.append(comm)
          nc.append(len(lcc_comms))
          comm_size = np.array([len(comm) for comm in lcc_comms])
          nl.append(sum(comm_size >= 4))
          cl.append(comm_size[comm_size >=4].sum() / origin_G.number_of_nodes())
      num_comm[row_ind, col_ind] = np.mean(nc)
      num_lcomm[row_ind, col_ind] = np.mean(nl)
      cov_lcomm[row_ind, col_ind] = np.mean(cl)
      
  metrics2plot = {'number of communities':num_comm, 'delta Hamiltonian':hamiltonian - subs_Hamiltonian, 
  'number of large communities':num_lcomm, 'coverage of large comm':cov_lcomm}
  num_col = 2
  num_row = int(len(metrics2plot) / num_col)
  fig = plt.figure(figsize=(5*num_col, 5*num_row))
  for i, k in enumerate(metrics2plot):
    plt.subplot(num_row, num_col, i+1)
    metric = metrics2plot[k]
    for row_ind, row in enumerate(rows):
      mean = metric[row_ind, :]
      plt.plot(cols, mean, label=row, alpha=0.6)
    plt.gca().set_title(k, fontsize=14, rotation=0)
    plt.xticks(rotation=90)
    # plt.yscale('symlog')
    if i == len(metrics2plot)-1:
      plt.legend()
    if i // num_col < num_row - 1:
      plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
  plt.suptitle(max_method, size=25)
  plt.tight_layout(rect=[0, 0.03, 1, 0.98])
  # plt.show()
  figname = './plots/stat_modular_Hamiltonian_maxreso_{}_{}_{}fold.jpg'
  plt.savefig(figname.format(max_method, measure, n))

resolution_list = np.arange(0, 2.1, 0.1)
resolution_list = [round(reso, 2) for reso in resolution_list]
stat_modular_structure_Hamiltonian_comms(G_ccg_dict, measure, n, resolution_list, max_pos_neg_reso=max_reso_subs, comms_dict=comms_dict, real_H=real_Hamiltonian, subs_H=subs_Hamiltonian, max_method='subs', cc=False)
#%%
######################### find the partition with two modules for AM during natural stimuli
def find_ind(G_dict, comms_dict, row, col):
  G = G_ccg_dict[row][col].copy()
  nx.set_node_attributes(G, active_area_dict[row], "area")
  comms_list = comms_dict[row][col]
  good_comm_inds = {}
  min_num_comm = 100
  for comm_ind in range(200):
    comms = comms_list[comm_ind]
    comms = [comm for comm in comms if len(comm)>=4] # only plot large communities
    node_to_community = comm2partition(comms)
    between_community_edges = _find_between_community_edges(G.edges(), node_to_community)
    comms2plot = get_unique_elements(between_community_edges.keys())
    dominant_areas = []
    
    for ind in comms2plot:
      comm = comms[ind]
      c_regions = [active_area_dict[row][node] for node in comm]
      # _, counts = np.unique(c_regions, return_counts=True)
      counts = np.array([c_regions.count(r) for r in visual_regions])
      dominant_areas.append(visual_regions[np.argmax(counts)])
    if 'natural' in col:
      if (dominant_areas.count('VISpm') > 1) and (dominant_areas.count('VISp') > 1):
        good_comm_inds[comm_ind] = len(comms2plot)
    elif 'grating' in col:
      if (dominant_areas.count('VISpm') <= 1) and (dominant_areas.count('VISp') <= 1):
        good_comm_inds[comm_ind] = len(comms2plot)
        
    # if len(comms2plot) < min_num_comm:
    #   good_comm_inds.append(comm_ind)
    #   min_num_comm = len(comms2plot)
  return good_comm_inds

def find_comminds_plot(G_dict, comms_dict):
  good_comm_inds = []
  rows, cols = get_rowcol(G_dict)
  row = rows[-1]
  dgrat_good_comm_inds = find_ind(G_dict, comms_dict, row, cols[3])
  sgrat_good_comm_inds = find_ind(G_dict, comms_dict, row, cols[4])
  nscene_good_comm_inds = find_ind(G_dict, comms_dict, row, cols[5])
  nmovie_good_comm_inds = find_ind(G_dict, comms_dict, row, cols[7])
  return dgrat_good_comm_inds, sgrat_good_comm_inds, nscene_good_comm_inds, nmovie_good_comm_inds
      # break
dgrat_good_comm_inds, sgrat_good_comm_inds, nscene_good_comm_inds, nmovie_good_comm_inds = find_comminds_plot(G_ccg_dict, best_comms_dict)
#%%
# Figure 2A
def plot_graph_community(G_dict, row_ind, best_comms_dict):
  rows, cols = get_rowcol(G_dict)
  row = rows[row_ind]
  fig, axes = plt.subplots(1, len(combined_stimuli), figsize=(12*len(combined_stimuli), 12))
  comm_inds = [49, 5, 102, 5, 13, 157]
  for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
    comm_ind = comm_inds[cs_ind]
    col = combined_stimuli[cs_ind][-1]
    ax = axes[cs_ind]
    col_ind = cols.index(col)
    # ax.set_title(combined_stimulus_name, fontsize=25)
    print(col)
    G = G_dict[row][col]
    nx.set_node_attributes(G, active_area_dict[row], "area")
    # max_reso = max_pos_neg_reso[row_ind][col_ind][comm_ind]
    comms_list = best_comms_dict[row][col]
    comms = comms_list[comm_ind]
    if col_ind >= 4:
      th = 6
    else:
      th = 6
    comms = [comm for comm in comms if len(comm)>=th] # only plot large communities
    node_to_community = comm2partition(comms)
    between_community_edges = _find_between_community_edges(G.edges(), node_to_community)
    comms2plot = get_unique_elements(between_community_edges.keys())
    # comms2plot = range(len(comms))
    nodes2plot = [node for node in node_to_community if node_to_community[node] in comms2plot]
    node_color = {node:region_colors[visual_regions.index(G.nodes[node]['area'])] for node in nodes2plot}
    print('Number of communities {}, number of nodes: {}'.format(len(comms2plot), len(nodes2plot)))
    if len(nodes2plot):
      edge_colors = [transparent_rgb(colors.to_rgb('#2166ac'), [1,1,1], alpha=.4), transparent_rgb(colors.to_rgb('#b2182b'), [1,1,1], alpha=.3)] # blue and red
      # return G.subgraph(nodes2plot)
      G2plot = G.subgraph(nodes2plot).copy()
      for n1, n2, d in G2plot.edges(data=True):
        for att in ['confidence']:
          d.pop(att, None)
      # binary weight
      for edge in G2plot.edges():
        if G2plot[edge[0]][edge[1]]['weight'] > 0:
          G2plot[edge[0]][edge[1]]['weight'] = 1
        else:
          G2plot[edge[0]][edge[1]]['weight'] = -1
      # node_labels = {node:node_to_community[node] for node in nodes2plot}
      if cs_ind in [0]:
        origin, scale, node_origin, node_scale=(-1, -1), (.9, .9), np.array([-1., -1.]), np.array([2., 2.])
      elif cs_ind in [1]:
        origin, scale, node_origin, node_scale=(-1, -1), (.8, .8), np.array([.5, .5]), np.array([3.5, 3.5])
      elif cs_ind in [2]:
        origin, scale, node_origin, node_scale=(-1, -1), (.9, .9), np.array([-1., -1.]), np.array([2., 2.])
      elif cs_ind in [3, 4, 5]:
        origin, scale, node_origin, node_scale=(-1, -1), (.9, .9), np.array([-1., -1.]), np.array([2., 2.])
      Graph(G2plot, nodes=nodes2plot, edge_cmap=colors.LinearSegmentedColormap.from_list("", edge_colors),
            node_color=node_color, node_edge_width=0.5, node_alpha=1., edge_alpha=0.4,
            node_layout='community', node_layout_kwargs=dict(node_to_community={node: comm for node, comm in node_to_community.items() if node in nodes2plot}),
            edge_layout='straight', edge_layout_kwargs=dict(k=1), node_edge_color='k',
            origin=origin, scale=scale, node_origin=node_origin, node_scale=node_scale, ax=ax) # bundled, node_labels=node_labels
  plt.tight_layout()
  plt.savefig('./plots/all_graph_topology_{}.pdf'.format(row), transparent=True)
  # plt.savefig('./plots/graph_topology/all_graph_topology_{}.jpg'.format(row))
  # plt.show()

# to_plots = [(7, 5)]
# to_plots = [(5, 2), (6, 2), (7, 49)]
# row_ind, comm_ind = to_plots[2]
# 4
plot_graph_community(G_ccg_dict, 7, best_comms_dict)
#%%
############ save region legend for topology plots
def save_region_legend():
  fig, ax = plt.subplots(1,1, figsize=(.4*(len(combined_stimulus_names)-1), .5))
  df = pd.DataFrame([[0,0,0],[0,0,1],[0,0,2],[0,0,3],[0,0,4],[0,0,5]], columns=['x', 'y', 'type'])
  barplot = sns.scatterplot(x='x', y='y', hue="type", palette=region_colors, data=df, ax=ax)
  handles, labels = ax.get_legend_handles_labels()
  for handle in handles:
    handle.set_edgecolor("k")
    handle.set_linewidth(1.5)
  legend = ax.legend(handles, region_labels, title='', bbox_to_anchor=(0.1, -1.), handletextpad=0.3, loc='upper left', fontsize=25, frameon=False)
  for handle in legend.legendHandles:
    handle.set_sizes([200.0])
  plt.axis('off')
  export_legend(legend, './plots/region_legend.pdf')
  plt.tight_layout()
  plt.show()

save_region_legend()
#%%
# Figure 2A
######################### Plot Z score of Hamiltonian as a measure of modular or not
def arrowed_spines(
        ax,
        x_width_fraction=0.2,
        x_height_fraction=0.02,
        lw=None,
        ohg=0.2,
        locations=('bottom right', 'left up'),
        **arrow_kwargs
):
    # set/override some default plotting parameters if required
    arrow_kwargs.setdefault('overhang', ohg)
    arrow_kwargs.setdefault('clip_on', False)
    arrow_kwargs.update({'length_includes_head': True})

    # axis line width
    if lw is None:
        # FIXME: does this still work if the left spine has been deleted?
        lw = ax.spines['left'].get_linewidth()
    annots = {}
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    # get width and height of axes object to compute
    # matching arrowhead length and width
    fig = ax.get_figure()
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height
    # manual arrowhead width and length
    hw = x_width_fraction * (ymax-ymin)
    hl = x_height_fraction * (xmax-xmin)
    # compute matching arrowhead length and width
    yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width
    yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height
    # draw x and y axis
    for loc_str in locations:
        side, direction = loc_str.split(' ')
        assert side in {'top', 'bottom', 'left', 'right'}, "Unsupported side"
        assert direction in {'up', 'down', 'left', 'right'}, "Unsupported direction"

        if side in {'bottom', 'top'}:
            if direction in {'up', 'down'}:
                raise ValueError("Only left/right arrows supported on the bottom and top")
            dy = 0
            head_width = hw
            head_length = hl
            y = ymin if side == 'bottom' else ymax

            if direction == 'right':
                x = xmin
                dx = xmax - xmin
            else:
                x = xmax
                dx = xmin - xmax
        else:
            if direction in {'left', 'right'}:
                raise ValueError("Only up/downarrows supported on the left and right")
            dx = 0
            head_width = yhw
            head_length = yhl
            x = xmin if side == 'left' else xmax
            if direction == 'up':
                y = ymin
                dy = ymax - ymin
            else:
                y = ymax
                dy = ymin - ymax
        annots[loc_str] = ax.arrow(x, y, dx, dy, fc='k', ec='k', lw = lw,
                 head_width=head_width, head_length=head_length, **arrow_kwargs)

    return annots

def plot_zscore_Hamiltonian(G_dict, resolution_list, max_pos_neg_reso, real_H, subs_H, max_method='none', cc=False):
  rows, cols = get_rowcol(G_dict)
  zscore_Hamiltonian, ps = [[] for _ in range(len(combined_stimuli))], [[] for _ in range(len(combined_stimuli))]
  runs = real_Hamiltonian.shape[-1]
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      h, rh = [], []
      for run in range(runs):
        max_reso = max_pos_neg_reso[row_ind, col_ind]
        h.append(real_H[row_ind, col_ind, resolution_list.index(max_reso[0]), resolution_list.index(max_reso[1]), run])
        rh.append(subs_H[row_ind, col_ind, resolution_list.index(max_reso[0]), resolution_list.index(max_reso[1]), run])
      # return h, rh
      # zscore_Hamiltonian[combine_stimulus(col)[0]] += ztest(h, rh)[0]
      zscore_Hamiltonian[combine_stimulus(col)[0]].append(ztest(h, rh)[0])
      ps[combine_stimulus(col)[0]].append(ztest(h, rh)[1])
  fig, ax = plt.subplots(1, 1, figsize=(6*len(combined_stimuli), 1.2))
  ax.bar(range(len(combined_stimulus_names)), [np.mean(dH) for dH in zscore_Hamiltonian], width=.07, align='center', alpha=1., linewidth=5, facecolor='w', edgecolor='black', capsize=10) #
  
  # arrowed_spines(fig, ax)
  ax.set_xlim(-.4, 5.5)
  ax.set_ylim(0., -70)

  annots = arrowed_spines(ax, locations=['bottom right'], lw=4.)

  ax.set_xticks([])
  # ax.set_xticklabels(combined_stimulus_names, fontsize=25)
  for axis in ['top', 'left']:
    ax.spines[axis].set_linewidth(3.)
    ax.spines[axis].set_color('k')
  ax.xaxis.tick_top()
  ax.spines['right'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  ax.yaxis.set_tick_params(labelsize=30)
  ax.tick_params(width=3)
  # ax.set_ylabel(r'$Z_{H}$', size=60)
  ax.invert_yaxis()
  plt.tight_layout()
  # plt.show()
  plt.savefig('./plots/zscore_Hamiltonian_{}.pdf'.format(max_method), transparent=True)

plot_zscore_Hamiltonian(G_ccg_dict, resolution_list, max_pos_neg_reso=max_reso_subs, real_H=real_Hamiltonian, subs_H=subs_Hamiltonian, max_method='subs', cc=False)
#%%
def get_table_histogram(session_id, stimulus_name, regions, resolution):
  data_directory = './data/ecephys_cache_dir'
  manifest_path = os.path.join(data_directory, "manifest.json")
  cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
  session = cache.get_session_data(int(session_id),
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
    if (stimulus_name=='flash_light') or (stimulus_name=='flash_dark'):
      stim_table = session.get_stimulus_table(['flashes'])
    else:
      stim_table = session.get_stimulus_table([stimulus_name])
    stim_table=stim_table.rename(columns={"start_time": "Start", "stop_time": "End"})
    if stimulus_name=='flash_light':
      stim_table = stim_table[stim_table['color']==1]
    elif stimulus_name=='flash_dark':
      stim_table = stim_table[stim_table['color']==-1]
    if 'natural_movie' in stimulus_name:
        frame_times = stim_table.End-stim_table.Start
        print('frame rate:', 1/np.mean(frame_times), 'Hz', np.mean(frame_times))
        # stim_table.to_csv(output_path+'stim_table_'+stimulus_name+'.csv')
        # chunch each movie clip
        # stim_table = stim_table[stim_table.frame==0]
        # stim_table = stim_table.drop(['End'], axis=1)
        duration = np.mean(remove_outlier(np.diff(stim_table.Start.values))[:10]) - 1e-4
        stimulus_presentation_ids = stim_table.index.values
    elif stimulus_name=='spontaneous':
        index = np.where(stim_table.duration>=20)[0]
        if len(index): # only keep the longest spontaneous; has to be longer than 20 sec
            duration=20
            stimulus_presentation_ids = stim_table.index[index]
    else:
        # ISI = np.mean(session.get_inter_presentation_intervals_for_stimulus([stimulus_name]).interval.values)
        duration = round(np.mean(stim_table.duration.values), 2)
    # each trial should have at least 250 ms
    try: stimulus_presentation_ids
    except NameError: stimulus_presentation_ids = stim_table.index[stim_table.duration >= 0.25].values
    time_bin_edges = np.linspace(0, duration, int(duration / resolution)+1)
    cortical_units_ids = np.array([idx for idx, ccf in enumerate(df.ccf.values) if ccf in regions])
    print('Number of units is {}, duration is {}'.format(len(cortical_units_ids), duration))
    # get binarized tensor
    df_cortex = df.iloc[cortical_units_ids]
    histograms = session.presentationwise_spike_counts(
        bin_edges=time_bin_edges,
        stimulus_presentation_ids=stimulus_presentation_ids,
        unit_ids=df_cortex.unit_id.values
    )
    return stim_table.loc[stimulus_presentation_ids], histograms

################# compare functional connection probability VS within comm and cross comm
def get_signal_correlation(G_dict, resolution):
  all_regions = ['VISp', 'VISl', 'VISrl', 'VISal', 'VISpm', 'VISam', 'LGd', 'LP'] # calculate signal correlation matrix for all 8 regions, but select only cortical regions later
  rows, cols = get_rowcol(G_dict)
  signal_correlation_dict = {}
  for row in rows:
    signal_correlation_dict[row] = {}
    for combined_stimulus in combined_stimuli[1:]:
      print(row, combined_stimulus_names[combined_stimuli.index(combined_stimulus)].replace('\n', ' '))
      FR_condition = []
      for col in combined_stimulus:
        # condition, time, neuron
        stim_table, histograms = get_table_histogram(row, col, all_regions, resolution)
        if 'natural_movie' in col:
          sequences = np.moveaxis(histograms.values, -1, 0) # neuron, condition, time
          frame_ids = np.unique(stim_table.frame)
          num_neuron, trial, time = sequences[:,stim_table.frame==0,:].shape
          # new_sequences = np.zeros((num_neuron, len(frame_ids), trial*time))
          for ind, frame_id in enumerate(frame_ids):
            # new_sequences[:,ind,:] = sequences[:,stim_table.frame==frame_id,:].reshape(num_neuron, -1)
            sequence = sequences[:,stim_table.frame==frame_id,:].reshape(num_neuron, -1)
            FR_condition.append(1000 * sequence.sum(1) / sequence.shape[1]) # firing rate in Hz
        else:
          unique_condition_ids = stim_table['stimulus_condition_id'].unique()
          for condition_id in unique_condition_ids:
            inds = stim_table['stimulus_condition_id']==condition_id
          # unique_sblock = stim_table[['stimulus_block', 'stimulus_condition_id']].drop_duplicates()
          # for i in range(unique_sblock.shape[0]):
            # inds = (stim_table['stimulus_block']==unique_sblock.iloc[i]['stimulus_block']) & (stim_table['stimulus_condition_id']==unique_sblock.iloc[i]['stimulus_condition_id'])
            sequences = np.moveaxis(histograms.values[inds], -1, 0) # neuron, condition, time
            # FR_condition[:, c_ind] = 1000 * sequences.mean(1).sum(1) / sequences.shape[2] # firing rate in Hz
            FR_condition.append(1000 * sequences.mean(1).sum(1) / sequences.shape[2]) # firing rate in Hz
      FR_condition = np.vstack(FR_condition).T # transpose to (neuron, block)
      signal_correlation = np.corrcoef(FR_condition)
      np.fill_diagonal(signal_correlation, 0)
      signal_correlation_dict[row][combine_stimulus(col)[1]] = signal_correlation
  return signal_correlation_dict

start_time = time.time()
resolution = 0.001
signal_correlation_dict = get_signal_correlation(G_ccg_dict, resolution)
a_file = open('./files/signal_correlation_dict.pkl', 'wb')
pickle.dump(signal_correlation_dict, a_file)
a_file.close()
print("--- %s minutes" % ((time.time() - start_time)/60))
# resolution = 0.001
# all_regions = ['VISp', 'VISl', 'VISrl', 'VISal', 'VISpm', 'VISam', 'LGd', 'LP'] # calculate signal correlation matrix for all 8 regions, but select only cortical regions later
# rows, cols = get_rowcol(G_ccg_dict)
# row = rows[-1]
# FR_condition = []
# col = cols[7]
# stim_table, histograms = get_table_histogram(row, col, all_regions, resolution)
# sequences = np.moveaxis(histograms.values, -1, 0) # neuron, condition, time
# frame_ids = np.unique(stim_table.frame)
# num_neuron, trial, time = sequences[:,stim_table.frame==0,:].shape
# new_sequences = np.zeros((num_neuron, len(frame_ids), trial*time))
# for ind, frame_id in enumerate(frame_ids):
#   # sequence = sequences[:,stim_table.frame==frame_id,:].reshape(num_neuron, -1)
#   # FR_condition.append(1000 * sequence.sum(1) / sequence.shape[1]) # firing rate in Hz
#   sequence = sequences[:,stim_table.frame==frame_id,:]
#   FR_condition.append(1000 * sequence.mean(1).sum(1) / sequence.shape[2]) # firing rate in Hz
# FR_condition = np.vstack(FR_condition).T # transpose to (neuron, block)
# signal_correlation = np.corrcoef(FR_condition)
# np.fill_diagonal(signal_correlation, np.nan)
#%%
# Figure 3A
########################################## get signal correlation for within/cross community VS stimulus
def get_within_cross_comm(G_dict, signal_correlation_dict, comms_dict, pair_type='all'):
  rows = list(signal_correlation_dict.keys())
  within_comm_dict, cross_comm_dict = {}, {}
  runs = len(comms_dict[rows[0]][cols[0]])
  for row_ind, row in enumerate(rows):
    print(row)
    active_area = active_area_dict[row]
    node_idx = sorted(active_area.keys())
    within_comm_dict[row], cross_comm_dict[row] = {}, {}
    for combined_stimulus_name in combined_stimulus_names[1:]:
      within_comm_dict[row][combined_stimulus_name], cross_comm_dict[row][combined_stimulus_name] = [], []
      signal_correlation = signal_correlation_dict[row][combined_stimulus_name]
      for col in combined_stimuli[combined_stimulus_names.index(combined_stimulus_name)]:
        col_ind = stimulus_names.index(col)
        G = G_dict[row][col]
        comms_list = comms_dict[row][col]
        for run in range(runs):
          comms = comms_list[run]
          within_comm, cross_comm = [], []
          node_to_community = comm2partition(comms)
          if pair_type == 'all': # all neuron pairs
            neuron_pairs = list(itertools.combinations(node_idx, 2))
          elif pair_type == 'connected': # limited to connected pairs only
            neuron_pairs = G.to_undirected().edges()
          for nodei, nodej in neuron_pairs: # for all neurons
            scorr = signal_correlation[nodei, nodej] # abs(signal_correlation[nodei, nodej])
            if node_to_community[nodei] == node_to_community[nodej]:
              within_comm.append(scorr)
            else:
              cross_comm.append(scorr)
          within_comm_dict[row][combined_stimulus_name].append(np.nanmean(within_comm))
          cross_comm_dict[row][combined_stimulus_name].append(np.nanmean(cross_comm))
  df = pd.DataFrame()
  for combined_stimulus_name in combined_stimulus_names[1:]:
    print(combined_stimulus_name)
    for row in session2keep:
      within_comm, cross_comm = within_comm_dict[row][combined_stimulus_name], cross_comm_dict[row][combined_stimulus_name]
      within_comm, cross_comm = [e for e in within_comm if not np.isnan(e)], [e for e in cross_comm if not np.isnan(e)] # remove nan values
      df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(within_comm)[:,None], np.array(['within community'] * len(within_comm))[:,None], np.array([combined_stimulus_name] * len(within_comm))[:,None], np.array([row] * len(within_comm))[:,None]), 1), columns=['signal correlation', 'type', 'stimulus', 'session'])], ignore_index=True)
      df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(cross_comm)[:,None], np.array(['cross community'] * len(cross_comm))[:,None], np.array([combined_stimulus_name] * len(cross_comm))[:,None], np.array([row] * len(cross_comm))[:,None]), 1), columns=['signal correlation', 'type', 'stimulus', 'session'])], ignore_index=True)
  df['signal correlation'] = pd.to_numeric(df['signal correlation'])
  return df

# pair_type = 'all'
pair_type = 'connected'
start_time = time.time()
signalcorr_within_cross_comm_df = get_within_cross_comm(G_ccg_dict, signal_correlation_dict, best_comms_dict, pair_type=pair_type)
print("--- %s minutes" % ((time.time() - start_time)/60))
#%%
def save_within_cross_module_legend():
  fig, ax = plt.subplots(1,1, figsize=(.4*(len(combined_stimulus_names)-1), .5))
  palette = ['k','.6']
  df = pd.DataFrame([[0,0,0],[0,0,1]], columns=['x', 'y', 'type'])
  barplot = sns.barplot(x='x', y='y', hue="type", hue_order=[0, 1], palette=palette, ec='k', linewidth=2., data=df, ax=ax, capsize=.05, width=0.6)
  handles, labels = ax.get_legend_handles_labels()
  legend = ax.legend(handles, ['within module', 'cross module'], title='', bbox_to_anchor=(0.1, -1.), handletextpad=0.3, loc='upper left', fontsize=25, frameon=False) # change legend order to within/cross similar module then cross random
  plt.axis('off')
  export_legend(legend, './plots/within_cross_module_legend.pdf')
  plt.tight_layout()
  # plt.savefig('./plots/stimuli_markers.pdf', transparent=True)
  plt.show()

save_within_cross_module_legend()
#%%
# Figure 3A
######################### merge all pairs from different mice into one distribution
def plot_signal_correlation_within_cross_comm_significance(df, pair_type='all'):
  fig, ax = plt.subplots(1,1, sharex=True, sharey=True, figsize=(1.5*(len(combined_stimulus_names)-1), 5))
  palette = ['k','.6']
  barplot = sns.barplot(x='stimulus', y='signal correlation', hue="type", hue_order=['within community', 'cross community'], palette=palette, ec='k', linewidth=2., data=df, ax=ax, capsize=.05, width=0.6, errorbar=('ci', 95))
  # palette = [[plt.cm.tab20b(i) for i in range(4)][i] for i in [0, 3]]
  # barplot = sns.barplot(x='stimulus', y='signal correlation', hue="type", palette=palette, data=df, ax=ax, capsize=.05, width=0.5)
  ax.yaxis.set_tick_params(labelsize=30)
  plt.setp(ax.get_legend().get_title(), fontsize='0') # for legend title
  plt.xticks([], []) # use markers to represent stimuli!
  # plt.xticks(range(len(combined_stimulus_names)-1), combined_stimulus_names[1:], fontsize=30)
  ax.xaxis.set_tick_params(length=0)
  for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(1.5)
    ax.spines[axis].set_color('k')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.tick_params(width=1.5)
  ax.set_xlabel('')
  ax.set_ylabel('Signal correlation', fontsize=40)
  ax.legend([], [], fontsize=0)
  # handles, labels = ax.get_legend_handles_labels()
  # ax.legend(handles, ['within module', 'cross module'], title='', bbox_to_anchor=(0., 1.), handletextpad=0.3, loc='upper left', fontsize=35, frameon=False)
  # add significance annotation
  alpha_list = [.0001, .001, .01, .05]
  maxx = 0
  for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names[1:]):
    within_comm, cross_comm = df[(df.stimulus==combined_stimulus_name)&(df.type=='within community')]['signal correlation'].values.flatten(), df[(df.stimulus==combined_stimulus_name)&(df.type=='cross community')]['signal correlation'].values.flatten()
    within_sr, cross_sr = confidence_interval(within_comm)[1], confidence_interval(cross_comm)[1]
    maxx = max(within_sr, cross_sr) if max(within_sr, cross_sr) > maxx else maxx
  h, l = .05 * maxx, .05 * maxx
  for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names[1:]):
    within_comm, cross_comm = df[(df.stimulus==combined_stimulus_name)&(df.type=='within community')]['signal correlation'].values.flatten(), df[(df.stimulus==combined_stimulus_name)&(df.type=='cross community')]['signal correlation'].values.flatten()
    _, p = ztest(within_comm, cross_comm, value=0)
    diff_star = '*' * (len(alpha_list) - bisect(alpha_list, p)) if len(alpha_list) > bisect(alpha_list, p) else 'ns'
    within_sr, cross_sr = confidence_interval(within_comm)[1], confidence_interval(cross_comm)[1]
    within_sr = within_sr + h
    cross_sr = cross_sr + h
    annot_difference(diff_star, -.12 + cs_ind, .12 + cs_ind, max(within_sr, cross_sr), l, 2.5, 28, ax=ax)
  plt.tight_layout(rect=[.02, -.03, 1, 1])
  # plt.show()
  plt.savefig('./plots/signal_correlation_within_cross_comm_{}.pdf'.format(pair_type), transparent=True)

plot_signal_correlation_within_cross_comm_significance(signalcorr_within_cross_comm_df, pair_type=pair_type)
#%%
############################# paired t test for each mouse
def plot_signal_correlation_within_cross_comm_paired_significance(df, pair_type='all'):
  fig, axes = plt.subplots(1,len(combined_stimulus_names)-1, sharex=True, sharey=True, figsize=(2.8*(len(combined_stimulus_names)-1), 3))
  palette = ['k','.6']
  df = df.groupby(['type', 'stimulus', 'session']).mean().reset_index() # first average over individual mouse
  df = df.set_index('session')
  df = df.loc[session2keep]
  df.reset_index(inplace=True)
  df = df.set_index('stimulus')
  df = df.loc[combined_stimulus_names[1:]]
  df.reset_index(inplace=True)
  axes[0].set_ylabel('Signal correlation', fontsize=30)
  axes[0].set_xlim(-.5, 1.5)
  for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names[1:]):
    ax = axes[cs_ind]
    within_comm, cross_comm = df[(df.stimulus==combined_stimulus_name)&(df.type=='within community')]['signal correlation'].values.flatten(), df[(df.stimulus==combined_stimulus_name)&(df.type=='cross community')]['signal correlation'].values.flatten()
    ax.scatter(np.zeros(len(within_comm)), within_comm, s=50, color='k')
    ax.scatter(np.ones(len(cross_comm)), cross_comm, s=50, color='k')

    # plotting the lines
    for i in range(len(within_comm)):
      ax.plot([0,1], [within_comm[i], cross_comm[i]], linewidth=2.5, c='k')

    # ax.set_xticks([0,1], ['within community', 'cross community'])
    ax.set_xticks([-.3,1.3])
    ax.set_xticklabels(labels=['within\nmodule', 'cross\nmodule'], fontsize=26)
    p_value = stats.ttest_rel(within_comm, cross_comm).pvalue
    locx, locy = .6, 1.2
    ax.text(locx, locy, 'p={:.1e}'.format(p_value), horizontalalignment='center',
        verticalalignment='center', transform=ax.transAxes, fontsize=25)
    
    # palette = [[plt.cm.tab20b(i) for i in range(4)][i] for i in [0, 3]]
    # barplot = sns.barplot(x='stimulus', y='signal correlation', hue="type", palette=palette, data=df, ax=ax, capsize=.05, width=0.5)
    ax.yaxis.set_tick_params(labelsize=30)
    # plt.xticks([], []) # use markers to represent stimuli!
    ax.xaxis.set_tick_params(length=0)
    for axis in ['bottom', 'left']:
      ax.spines[axis].set_linewidth(2.5)
      ax.spines[axis].set_color('k')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=2.5)
    ax.set_xlabel('')
  # handles, labels = ax.get_legend_handles_labels()
  # ax.legend(handles, ['within module', 'cross module'], title='', bbox_to_anchor=(0., 1.), handletextpad=0.3, loc='upper left', fontsize=35, frameon=False)
  # add significance annotation
  alpha_list = [.0001, .001, .01, .05]
  
  plt.tight_layout(rect=[.02, -.03, 1, 1])
  # plt.show()
  plt.savefig('./plots/signal_correlation_within_cross_comm_{}_paired.pdf'.format(pair_type), transparent=True)

plot_signal_correlation_within_cross_comm_paired_significance(signalcorr_within_cross_comm_df, pair_type)
#%%
def save_stimuli_markers():
  fig, ax = plt.subplots(1,1, figsize=(.4*(len(combined_stimulus_names)-1), .5))
  marker_list = ["v", '*', 'P', 'X', 'o', 's']
  markersize_list = [40, 80, 50, 50, 42, 40]
  for cs_ind in range(len(combined_stimulus_names)-1):
    ax.scatter(cs_ind, 0, marker=marker_list[cs_ind+1], s=markersize_list[cs_ind+1], color='.1', facecolors='none')
  plt.axis('off')
  # plt.tight_layout()
  plt.savefig('./plots/stimuli_markers.pdf', transparent=True)
  # plt.show()

save_stimuli_markers()
#%%
def save_markers():
  for combined_stimulus_name in combined_stimulus_names:
    fig, ax = plt.subplots(1,1, figsize=(.5, .5))
    ax.scatter(0, 0, marker=stimulus2marker[combined_stimulus_name], s=8*marker_size_dict[stimulus2marker[combined_stimulus_name]], color='.1', facecolors='none', linewidth=.7)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('./plots/{}.pdf'.format(combined_stimulus_name.replace('\n', '_')), transparent=True)
    # plt.show()

save_markers()
#%%
# Figure S5A
########################## probability of being in the same module VS signal correlation
def plot_same_module_prob_signal_correlation(df):
  fig, axes = plt.subplots(1,len(combined_stimulus_names)-1, sharey=True, figsize=(5*(len(combined_stimulus_names)-1), 5))
  axes[0].set_ylim(top=1.2)
  # palette = [[plt.cm.tab20b(i) for i in range(4)][i] for i in [0, 3]]
  for cs_ind in range(len(axes)):
    ax = axes[cs_ind]
    data = df[df.stimulus==combined_stimulus_names[cs_ind+1]].copy()
    data.insert(0, 'probability of same module', 0)
    # data.loc[:, 'probability of same module'] = 0
    data.loc[data['type']=='within community','probability of same module'] = 1
    # ax.set_title(combined_stimulus_names[cs_ind+1].replace('\n', ' '), fontsize=25)
    x, y = data['signal correlation'].values.flatten(), data['probability of same module'].values.flatten()
    numbin = 6
    binned_x, binned_y = double_equal_binning(x, y, numbin=numbin, log=False)
    connect, disconnect = double_equal_binning_counts(x, y, numbin=numbin, log=False)
    ax.bar(binned_x, binned_y, width=np.diff(binned_x).max()/1.5, color='.7')
    for i in range(len(binned_x)):
      ax.annotate('{}/{}'.format(connect[i], (connect[i]+disconnect[i])), xy=(binned_x[i],binned_y[i]), ha='center', va='bottom', fontsize=20)   
    dff = pd.DataFrame(np.vstack((connect, disconnect)), index=['connected', 'disconnected'], columns=binned_x)
    table = sm.stats.Table(dff, shift_zeros=False)
    p_value = table.test_ordinal_association().pvalue
    locx, locy = .5, .9
    if p_value == 0:
      text = 'p<1e-310'
    else:
      text = 'p={:.1e}'.format(p_value)
    ax.text(locx, locy, text, horizontalalignment='center',
        verticalalignment='center', transform=ax.transAxes, fontsize=25)
    # ax.scatter(binned_x, binned_y)
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)
    for axis in ['bottom', 'left']:
      ax.spines[axis].set_linewidth(1.5)
      ax.spines[axis].set_color('k')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=1.5)
    # if cs_ind == 0:
    #   ax.set_ylabel('Probability of finding connected \npairs inside the same module', fontsize=28)
    # ax.set_xlabel('Signal correlation', fontsize=25)
    ax.set_xlabel('')
    ax.set_ylabel('')

  plt.tight_layout(rect=[.01, 0, 1, 1])
  # plt.show()
  plt.savefig('./plots/same_module_prob_signal_correlation.pdf', transparent=True)

plot_same_module_prob_signal_correlation(signalcorr_within_cross_comm_df)
#%%
# Figure S2A
########################################## get connection probability VS signal correlation
def get_connectionp_signalcorr(G_dict, signal_correlation_dict):
  rows = signal_correlation_dict.keys()
  connect_dict, disconnect_dict = {}, {}
  for row_ind, row in enumerate(rows):
    print(row)
    active_area = active_area_dict[row]
    node_idx = sorted(active_area.keys())
    connect_dict[row], disconnect_dict[row] = {}, {}
    for combined_stimulus_name in combined_stimulus_names[1:]:
      cs_ind = combined_stimulus_names.index(combined_stimulus_name)
      connect_dict[row][combined_stimulus_name], disconnect_dict[row][combined_stimulus_name] = [], []
      signal_correlation = signal_correlation_dict[row][combined_stimulus_name]
      for col in combined_stimuli[cs_ind]:
        G = G_dict[row][col]
        connect, disconnect = [], []
        for nodei, nodej in itertools.combinations(node_idx, 2):
          scorr = signal_correlation[nodei, nodej] # abs(signal_correlation[nodei, nodej])
          if not np.isnan(scorr):
            if (G.has_edge(nodei, nodej) or G.has_edge(nodej, nodei)):
              connect.append(scorr)
            else:
              disconnect.append(scorr)
        connect_dict[row][combined_stimulus_name] += connect
        disconnect_dict[row][combined_stimulus_name] += disconnect
        # Add bootstrapping to create more datapoints
        # connect_dict[row][combined_stimulus_name] += np.mean(np.array([np.random.choice(connect, 100, replace = True) for _ in range(50)]), axis=1).tolist()
        # disconnect_dict[row][combined_stimulus_name] += np.mean(np.array([np.random.choice(disconnect, 100, replace = True) for _ in range(50)]), axis=1).tolist()
  df = pd.DataFrame()
  for combined_stimulus_name in combined_stimulus_names[1:]:
    print(combined_stimulus_name)
    for row in session2keep:
      connect, disconnect = connect_dict[row][combined_stimulus_name], disconnect_dict[row][combined_stimulus_name]
      # within_comm, cross_comm = [e for e in within_comm if not np.isnan(e)], [e for e in cross_comm if not np.isnan(e)] # remove nan values
      df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(connect)[:,None], np.array(['connected'] * len(connect))[:,None], np.array([combined_stimulus_name] * len(connect))[:,None]), 1), columns=['signal correlation', 'type', 'stimulus'])], ignore_index=True)
      df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(disconnect)[:,None], np.array(['disconnected'] * len(disconnect))[:,None], np.array([combined_stimulus_name] * len(disconnect))[:,None]), 1), columns=['signal correlation', 'type', 'stimulus'])], ignore_index=True)
  df['signal correlation'] = pd.to_numeric(df['signal correlation'])
  return df

start_time = time.time()
connectionp_signalcorr_df = get_connectionp_signalcorr(G_ccg_dict, signal_correlation_dict)
print("--- %s minutes" % ((time.time() - start_time)/60))
#%%
# Figure S2A
def plot_connectionp_signal_correlation(df):
  fig, axes = plt.subplots(1,len(combined_stimulus_names)-1, figsize=(5*(len(combined_stimulus_names)-1), 5)) #
  # axes[0].set_ylim(top = 1.2)
  # palette = [[plt.cm.tab20b(i) for i in range(4)][i] for i in [0, 3]]
  for cs_ind in range(len(axes)):
    ax = axes[cs_ind]
    data = df[df.stimulus==combined_stimulus_names[cs_ind+1]].copy()
    # ax.set_title(combined_stimulus_names[cs_ind+1].replace('\n', ' '), fontsize=25)

    data.insert(0, 'probability of connection', 0)
    # data.loc[:, 'probability of same module'] = 0
    data.loc[data['type']=='connected','probability of connection'] = 1
    # ax.set_title(combined_stimulus_names[cs_ind+1].replace('\n', ' '), fontsize=25)
    x, y = data['signal correlation'].values.flatten(), data['probability of connection'].values.flatten()

    # x, y = data['signal correlation'].values.flatten(), data['connection probability'].values.flatten()
    numbin = 6
    binned_x, binned_y = double_equal_binning(x, y, numbin=numbin, log=False)
    connect, disconnect = double_equal_binning_counts(x, y, numbin=numbin, log=False)
    ax.bar(binned_x, binned_y, width=np.diff(binned_x).max()/1.5, color='.7')
    for i in range(len(binned_x)):
      ax.annotate(r'$\frac{{{}}}{{{}}}$'.format(connect[i], (connect[i]+disconnect[i])), xy=(binned_x[i],binned_y[i]), ha='center', va='bottom', fontsize=18)
    
    dff = pd.DataFrame(np.vstack((connect, disconnect)), index=['connected', 'disconnected'], columns=binned_x)
    table = sm.stats.Table(dff, shift_zeros=False)
    p_value = table.test_ordinal_association().pvalue
    
    # xy = np.vstack((x, y)).T
    # table = sm.stats.Table.from_data(xy[~np.isnan(xy).any(axis=1)])
    # p_value = table.test_ordinal_association().pvalue
    if p_value == 0:
      text = 'p<1e-310'
    else:
      text = 'p={:.1e}'.format(p_value)
    locx, locy = .5, .8
    ax.text(locx, locy, text, horizontalalignment='center',
        verticalalignment='center', transform=ax.transAxes, fontsize=25)
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)
    for axis in ['bottom', 'left']:
      ax.spines[axis].set_linewidth(2.5)
      ax.spines[axis].set_color('k')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=2.5)
    # if cs_ind == 0:
    #   ax.set_ylabel('Connection probability', fontsize=30)
    ax.set_xlabel('')
    ax.set_ylabel('')
    # ax.set_xlabel('Signal correlation', fontsize=25)

  plt.tight_layout()
  # plt.show()
  plt.savefig('./plots/connectionp_signal_correlation.pdf', transparent=True)

plot_connectionp_signal_correlation(connectionp_signalcorr_df)
#%%
# Figure S2B,C
########################################## regard bidirectional edges as 2
########################################## get positive and negative connection probability VS signal correlation in a different way!!!
def get_pos_neg_p_signalcorr(G_dict, signal_correlation_dict, pairtype='all'):
  rows = signal_correlation_dict.keys()
  pos_connect_dict, neg_connect_dict, dis_connect_dict, signal_corr_dict = [{row:{csn:[] for csn in combined_stimulus_names[1:]} for row in rows} for _ in range(4)]
  for row_ind, row in enumerate(rows):
    print(row)
    active_area = active_area_dict[row]
    node_idx = sorted(active_area.keys())
    for combined_stimulus_name in combined_stimulus_names[1:]:
      cs_ind = combined_stimulus_names.index(combined_stimulus_name)
      signal_correlation = signal_correlation_dict[row][combined_stimulus_name]
      pos_connect, neg_connect, dis_connect, signal_corr = [], [], [], []
      for col in combined_stimuli[cs_ind]:
        G = G_dict[row][col].copy()
        for nodei, nodej in itertools.combinations(node_idx, 2):
          scorr = signal_correlation[nodei, nodej] # abs(signal_correlation[nodei, nodej])
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

      signal_corr_dict[row][combined_stimulus_name] += signal_corr
      pos_connect_dict[row][combined_stimulus_name] += pos_connect
      neg_connect_dict[row][combined_stimulus_name] += neg_connect
      dis_connect_dict[row][combined_stimulus_name] += dis_connect

  pos_df, neg_df, dis_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
  for combined_stimulus_name in combined_stimulus_names[1:]:
    print(combined_stimulus_name)
    for row in session2keep:
      pos_connect, neg_connect, dis_connect, signal_corr = pos_connect_dict[row][combined_stimulus_name], neg_connect_dict[row][combined_stimulus_name], dis_connect_dict[row][combined_stimulus_name], signal_corr_dict[row][combined_stimulus_name]
      # within_comm, cross_comm = [e for e in within_comm if not np.isnan(e)], [e for e in cross_comm if not np.isnan(e)] # remove nan values
      pos_df = pd.concat([pos_df, pd.DataFrame(np.concatenate((np.array(signal_corr)[:,None], np.array(pos_connect)[:,None], np.array([combined_stimulus_name] * len(pos_connect))[:,None], np.array([row] * len(pos_connect))[:,None]), 1), columns=['signal correlation', 'type', 'stimulus', 'session'])], ignore_index=True)
      neg_df = pd.concat([neg_df, pd.DataFrame(np.concatenate((np.array(signal_corr)[:,None], np.array(neg_connect)[:,None], np.array([combined_stimulus_name] * len(neg_connect))[:,None], np.array([row] * len(neg_connect))[:,None]), 1), columns=['signal correlation', 'type', 'stimulus', 'session'])], ignore_index=True)
      dis_df = pd.concat([dis_df, pd.DataFrame(np.concatenate((np.array(dis_connect)[:,None], np.array([combined_stimulus_name] * len(dis_connect))[:,None], np.array([row] * len(dis_connect))[:,None]), 1), columns=['signal correlation', 'stimulus', 'session'])], ignore_index=True)
  pos_df['signal correlation'] = pd.to_numeric(pos_df['signal correlation'])
  pos_df['type'] = pd.to_numeric(pos_df['type'])
  neg_df['signal correlation'] = pd.to_numeric(neg_df['signal correlation'])
  neg_df['type'] = pd.to_numeric(neg_df['type'])
  dis_df['signal correlation'] = pd.to_numeric(dis_df['signal correlation'])
  return pos_df, neg_df, dis_df

start_time = time.time()
pairtype = 'all'
# pairtype = 'connected'
pos_connectionp_signalcorr_df, neg_connectionp_signalcorr_df, dis_connected_signalcorr_df = get_pos_neg_p_signalcorr(G_ccg_dict, signal_correlation_dict, pairtype=pairtype)
print("--- %s minutes" % ((time.time() - start_time)/60))
#%%
# Figure S2B,C
def plot_pos_neg_connectionp_signal_correlation(df, dname='pos',pairtype='connected'):
  # for row in session2keep:
  #   print(row)
    fig, axes = plt.subplots(1,len(combined_stimulus_names)-1, figsize=(5*(len(combined_stimulus_names)-1), 5)) #, sharey=True
    # axes[0].set_ylim(top = 1.2)
    # if dname == 'pos':
    #   axes[0].set_ylim(top = .2)
    # else:
    #   axes[0].set_ylim(top = .05)
    # palette = [[plt.cm.tab20b(i) for i in range(4)][i] for i in [0, 3]]
    for cs_ind in range(len(axes)):
      ax = axes[cs_ind]
      data = df[(df.stimulus==combined_stimulus_names[cs_ind+1])].copy() #  & (df.session==row)
      
      # Old version, bootstrapping for connected sigcorr and disconnected sigcorr separately
      # data.insert(0, 'probability of connection', 0)
      # data.loc[data['type']=='connected','probability of connection'] = 1
      # # ax.set_title(combined_stimulus_names[cs_ind+1].replace('\n', ' '), fontsize=25)
      # x, y = data['signal correlation'].values.flatten(), data['probability of connection'].values.flatten()

      x, y = data['signal correlation'].values.flatten(), data['type'].values.flatten()

      # x, y = data['signal correlation'].values.flatten(), data['connection probability'].values.flatten()
      numbin = 5
      binned_x, binned_y = double_equal_binning(x, y, numbin=numbin, log=False)
      connect, disconnect = double_equal_binning_counts(x, y, numbin=numbin, log=False)
      ax.bar(binned_x, binned_y, width=np.diff(binned_x).max()/1.5, color='.7')
      for i in range(len(binned_x)):
        ax.annotate(r'$\frac{{{}}}{{{}}}$'.format(connect[i], (connect[i]+disconnect[i])), xy=(binned_x[i],binned_y[i]), ha='center', va='bottom', fontsize=26)
        # ax.annotate('{}/{}'.format(connect[i], (connect[i]+disconnect[i])), xy=(binned_x[i],binned_y[i]), ha='center', va='bottom', fontsize=20)
      # No binning, directly do Cochran–Armitage test, will lead to NAN value if too many data points!!!
      # xy = np.vstack((x, y)).T
      # table = sm.stats.Table.from_data(xy[~np.isnan(xy).any(axis=1)], shift_zeros=False)
      # # p_value = table.test_nominal_association().pvalue
      # p_value = table.test_ordinal_association().pvalue
      
      # First binning, then do Cochran–Armitage test
      dff = pd.DataFrame(np.vstack((connect, disconnect)), index=['connected', 'disconnected'], columns=binned_x)
      table = sm.stats.Table(dff)
      p_value = table.test_ordinal_association().pvalue
      
      # p_value = CA_test(xy)
      if pairtype == 'all':
        locx, locy = .5, .7
      else:
        locx, locy = .5, .1
      ax.text(locx, locy, 'p={:.1e}'.format(p_value), horizontalalignment='center',
          verticalalignment='center', transform=ax.transAxes, fontsize=25)
      ax.xaxis.set_tick_params(labelsize=30)
      ax.yaxis.set_tick_params(labelsize=30)
      for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(2.5)
        ax.spines[axis].set_color('k')
      ax.spines['top'].set_visible(False)
      ax.spines['right'].set_visible(False)
      ax.tick_params(width=2.5)
      # ylabel = 'Probability of finding pairs with\nexcitatory functional connection' if dname == 'pos' else 'Probability of finding pairs with\ninhibitory functional connection'
      # if cs_ind == 0:
      #   ax.set_ylabel(ylabel, fontsize=30)
      # ax.set_xlabel('Signal correlation', fontsize=25)

    plt.tight_layout(rect=[.01, 0, 1, 1])
    # plt.show() 
    # plt.savefig('./plots/{}_p_{}_signal_corr_{}.pdf'.format(pairtype, dname, row), transparent=True)
    plt.savefig('./plots/{}_p_{}_signal_corr.pdf'.format(pairtype, dname), transparent=True)

plot_pos_neg_connectionp_signal_correlation(pos_connectionp_signalcorr_df, 'pos', pairtype=pairtype)
plot_pos_neg_connectionp_signal_correlation(neg_connectionp_signalcorr_df, 'neg', pairtype=pairtype)
#%%
# Figure 1G
########################################## plot signal correlation distribution for ex/in
def plot_pos_neg_signal_correlation_distri(pos_df, neg_df, dis_df):
  # for row in session2keep:
  #   print(row)
    fig, axes = plt.subplots(1,len(combined_stimulus_names)-1, figsize=(5*(len(combined_stimulus_names)-1), 3)) #, sharey=True
    for cs_ind in range(len(axes)):
      ax = axes[cs_ind]
      pos_data = pos_df[(pos_df.stimulus==combined_stimulus_names[cs_ind+1]) & (pos_df.type==1)].copy() #  & (df.session==row)
      neg_data = neg_df[(neg_df.stimulus==combined_stimulus_names[cs_ind+1]) & (neg_df.type==1)].copy() #  & (df.session==row)
      dis_data = dis_df[dis_df.stimulus==combined_stimulus_names[cs_ind+1]].copy()
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
      ax.set_ylabel([], fontsize=0)
      # ax.set_xlabel('Signal correlation', fontsize=25)
      # handles, labels = ax.get_legend_handles_labels()
      ax.legend().set_visible(False)

    plt.tight_layout(rect=[.01, 0, 1, 1])
    # plt.show()
    # plt.savefig('./plots/hist_signal_corr_posnegdis.pdf', transparent=True)
    plt.savefig('./plots/distri_signal_corr_posnegdis.pdf', transparent=True)

plot_pos_neg_signal_correlation_distri(pos_connectionp_signalcorr_df, neg_connectionp_signalcorr_df, dis_connected_signalcorr_df)
#%%
def save_distr_signalcorr_legend():
  fig, ax = plt.subplots(1,1, figsize=(.4*(len(combined_stimulus_names)-1), .5))
  ax.plot([0,1], [0,0],color='grey', label='disconnected', markersize=10, alpha=.8, linewidth=2)
  ax.plot([0,1], [0,0],color='b', label='inhibitory', markersize=10, alpha=.8, linewidth=2)
  ax.plot([0,1], [0,0],color='r', label='excitatory', markersize=10, alpha=.8, linewidth=2)
  handles, labels = ax.get_legend_handles_labels()
  legend = ax.legend(handles, ['disconnected', 'inhibitory', 'excitatory'], title='', bbox_to_anchor=(0.1, -1.), handletextpad=0.3, loc='upper left', fontsize=25, frameon=False, ncol=3) # change legend order to within/cross similar module then cross random
  plt.axis('off')
  export_legend(legend, './plots/distr_signalcorr_legend.pdf')
  plt.tight_layout()
  # plt.savefig('./plots/stimuli_markers.pdf', transparent=True)
  plt.show()

save_distr_signalcorr_legend()
#%%
# Results not good!!!
########################################## get signal correlation VS area distance between communities
def get_scorr_comm_distance(signal_correlation_dict, comms_dict, regions, max_neg_reso):
  rows = signal_correlation_dict.keys()
  within_comm_sc, cross_comm_dict, area_distance_dict = np.zeros((len(rows), len(combined_stimulus_names)-1)), {}, {}
  for row_ind, row in enumerate(rows):
    print(row)
    active_area = active_area_dict[row]
    node_idx = sorted(active_area.keys())
    cross_comm_dict[row], area_distance_dict[row] = {}, {}
    for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names[1:]):
      within_comm_list, cross_comm_dict[row][combined_stimulus_name], area_distance_dict[row][combined_stimulus_name] = [], [], []
      signal_correlation = signal_correlation_dict[row][combined_stimulus_name]
      for col in combined_stimuli[combined_stimulus_names.index(combined_stimulus_name)]:
        col_ind = stimulus_names.index(col)
        for run in range(metrics['Hamiltonian'].shape[-1]):
          max_reso = max_neg_reso[row_ind, col_ind, run]
          comms_list = comms_dict[row][col][max_reso]
          comms = comms_list[run]
          within_comm, cross_comm, area_distance = [], [], []
          node_to_community = comm2partition(comms)
          comm_areas = [[active_area[node] for node in comm] for comm in comms]
          area_distri =  [[comm_area.count(area) for area in regions] for comm_area in comm_areas]
          for nodei, nodej in itertools.combinations(node_idx, 2):
            scorr = signal_correlation[nodei, nodej] # abs(signal_correlation[nodei, nodej])
            if node_to_community[nodei] == node_to_community[nodej]:
              within_comm.append(scorr)
            else:
              if not np.isnan(scorr):
                cross_comm.append(scorr)
                area_distance.append(distance.jensenshannon(area_distri[node_to_community[nodei]], area_distri[node_to_community[nodej]]))
          within_comm_list.append(np.nanmean(within_comm))
          # cross_comm_dict[row][combined_stimulus_name] += cross_comm
          # area_distance_dict[row][combined_stimulus_name] += area_distance
          cross_comm_dict[row][combined_stimulus_name].append(np.mean(cross_comm))
          area_distance_dict[row][combined_stimulus_name].append(np.mean(area_distance))
      within_comm_sc[row_ind, cs_ind] = np.nanmean(within_comm_list)
  df = pd.DataFrame()
  for combined_stimulus_name in combined_stimulus_names[1:]:
    print(combined_stimulus_name)
    # print(combine_stimulus(col)[1])
    for row in session2keep:
      cross_comm, area_distance = cross_comm_dict[row][combined_stimulus_name], area_distance_dict[row][combined_stimulus_name]
      # if combine_stimulus(col)[0] >= 5:
      #   print(len(within_comm), len(cross_comm))
      # cross_comm = [e for e in cross_comm if not np.isnan(e)] # remove nan values
      df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(cross_comm)[:,None], np.array(area_distance)[:,None], np.array([combined_stimulus_name] * len(cross_comm))[:,None]), 1), columns=['signal correlation', 'area distance', 'stimulus'])], ignore_index=True)
  df['area distance'] = pd.to_numeric(df['area distance'])
  df['signal correlation'] = pd.to_numeric(df['signal correlation'])
  return within_comm_sc.mean(0), df

start_time = time.time()
within_comm_sc, signalcorr_area_distance_df = get_scorr_comm_distance(signal_correlation_dict, comms_dict, visual_regions, max_reso_subs)
print("--- %s minutes" % ((time.time() - start_time)/60))
#%%
# Results not good!!!
import statsmodels.api as sm
def plot_signal_correlation_comm_distance(within_comm_sc, df):
  fig, axes = plt.subplots(1,len(combined_stimulus_names)-1, sharey=True, figsize=(5*(len(combined_stimulus_names)-1), 6))
  # palette = [[plt.cm.tab20b(i) for i in range(4)][i] for i in [0, 3]]
  for cs_ind in range(len(axes)):
    ax = axes[cs_ind]
    data = df[df.stimulus==combined_stimulus_names[cs_ind+1]]
    ax.set_title(combined_stimulus_names[cs_ind+1].replace('\n', ' '), fontsize=25)
    x, y = data['area distance'].values.flatten(), data['signal correlation'].values.flatten()

    numbin = 6
    binned_x, binned_y = double_equal_binning(x, y, numbin=numbin, log=False)
    connect, disconnect = double_equal_binning_counts(x, y, numbin=numbin, log=False)
    ax.bar(binned_x, binned_y, width=np.diff(binned_x).max()/1.5, color='.7')
    ax.axhline(y=within_comm_sc[cs_ind], linewidth=3, color='.2', linestyle='--', alpha=.4)
    for i in range(len(binned_x)):
      ax.annotate('{}/{}'.format(connect[i], (connect[i]+disconnect[i])), xy=(binned_x[i],binned_y[i]), ha='center', va='bottom', fontsize=20)
    xy = np.vstack((x, y)).T
    table = sm.stats.Table.from_data(xy[~np.isnan(xy).any(axis=1)])
    p_value = table.test_ordinal_association().pvalue
    locx, locy = .2, .8
    ax.text(locx, locy, 'p={:.1e}'.format(p_value), horizontalalignment='center',
        verticalalignment='center', transform=ax.transAxes, fontsize=25)

    # binned_x, binned_y = double_binning(x, y, numbin=6, log=False)
    # ax.scatter(binned_x, binned_y)

    ax.yaxis.set_tick_params(labelsize=30)
    # plt.setp(ax.get_legend().get_title(), fontsize='0') # for legend title
    # plt.xticks(range(len(combined_stimulus_names)-1), combined_stimulus_names[1:], fontsize=25)
    ax.xaxis.set_tick_params(length=0)
    for axis in ['bottom', 'left']:
      ax.spines[axis].set_linewidth(1.5)
      ax.spines[axis].set_color('0.2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=1.5)
    if cs_ind == 0:
      ax.set_ylabel('Signal correlation', fontsize=30)
    ax.set_xlabel('Jensen-Shannon distance of area', fontsize=25)
    
  handles, labels = ax.get_legend_handles_labels()
  ax.legend(handles, labels, title='', fontsize=20, frameon=False)
  plt.tight_layout()
  # plt.show()
  plt.savefig('./plots/signal_correlation_comm_distance.pdf', transparent=True)

plot_signal_correlation_comm_distance(within_comm_sc, signalcorr_area_distance_df)
#%%
################################ Results not good!!!
def get_strongly_functional_module(G_dict, signal_correlation_dict, comms_dict, max_neg_reso):
  rows = signal_correlation_dict.keys()
  strong_module_dict = {}
  for row_ind, row in enumerate(rows):
    print(row)
    active_area = active_area_dict[row]
    node_idx = sorted(active_area.keys())
    strong_module_dict[row] = {}
    for combined_stimulus_name in combined_stimulus_names[1:]:
      signal_correlation = signal_correlation_dict[row][combined_stimulus_name]
      for col in combined_stimuli[combined_stimulus_names.index(combined_stimulus_name)]:
        G = G_dict[row][col]
        strong_module_dict[row][col] = {}
        col_ind = stimulus_names.index(col)
        for run in range(metrics['Hamiltonian'].shape[-1]):
          strong_module_dict[row][col][run] = []
          max_reso = max_neg_reso[row_ind, col_ind, run]
          comms_list = comms_dict[row][col][max_reso]
          comms = comms_list[run]
          node_to_community = comm2partition(comms)
          comms = [comm for comm in comms if len(comm) >= 4]
          cross_comm = []
          
          for nodei, nodej in itertools.combinations(node_idx, 2): # for all neurons
            if node_to_community[nodei] != node_to_community[nodej]:
              cross_comm.append(signal_correlation[nodei, nodej])
          sc_mean, sc_std = np.nanmean(cross_comm), np.nanstd(cross_comm)
          # for nodei, nodej in G.edges(): # limited to only edges
          for comm in comms:
            # comm_sc = np.nanmean([signal_correlation[e[0], e[1]] for e in G.subgraph(comm).edges()]) # limited to only edges
            comm_sc = np.nanmean([signal_correlation[nodei, nodej] for nodei, nodej in itertools.combinations(comm, 2)]) # for all neuron pairs

            if comm_sc > sc_mean + 1 * sc_std:
              strong_module_dict[row][col][run].append(comm)
  return strong_module_dict
#%%
################################ number of strongly functional modules
def get_mean_signal_corr(signal_correlation_dict, comms_dict, max_neg_reso):
  rows = signal_correlation_dict.keys()
  cross_comm = []
  for row_ind, row in enumerate(rows):
    print(row)
    active_area = active_area_dict[row]
    node_idx = sorted(active_area.keys())
    for combined_stimulus_name in combined_stimulus_names[1:]:
      signal_correlation = signal_correlation_dict[row][combined_stimulus_name]
      for col in combined_stimuli[combined_stimulus_names.index(combined_stimulus_name)]:
        col_ind = stimulus_names.index(col)
        for run in range(metrics['Hamiltonian'].shape[-1]):
          max_reso = max_neg_reso[row_ind, col_ind, run]
          comms_list = comms_dict[row][col][max_reso]
          comms = comms_list[run]
          node_to_community = comm2partition(comms)        
          for nodei, nodej in itertools.combinations(node_idx, 2): # for all neurons
            if node_to_community[nodei] != node_to_community[nodej]:
              cross_comm.append(signal_correlation[nodei, nodej])
  sc_mean, sc_std = np.nanmean(cross_comm), np.nanstd(cross_comm)
  return sc_mean, sc_std

sc_mean, sc_std = get_mean_signal_corr(signal_correlation_dict, comms_dict, max_reso_subs)
#%%
def get_strongly_functional_module(sc_mean, sc_std, signal_correlation_dict, comms_dict, max_neg_reso):
  rows = signal_correlation_dict.keys()
  strong_module_dict = {}
  for row_ind, row in enumerate(rows):
    print(row)
    strong_module_dict[row] = {}
    for combined_stimulus_name in combined_stimulus_names[1:]:
      signal_correlation = signal_correlation_dict[row][combined_stimulus_name]
      for col in combined_stimuli[combined_stimulus_names.index(combined_stimulus_name)]:
        strong_module_dict[row][col] = {}
        col_ind = stimulus_names.index(col)
        for run in range(metrics['Hamiltonian'].shape[-1]):
          strong_module_dict[row][col][run] = []
          max_reso = max_neg_reso[row_ind, col_ind, run]
          comms_list = comms_dict[row][col][max_reso]
          comms = comms_list[run]
          # comms = [comm for comm in comms if len(comm) >= 4]
          # for nodei, nodej in G.edges(): # limited to only edges
          for comm in comms:
            # comm_sc = np.nanmean([signal_correlation[e[0], e[1]] for e in G.subgraph(comm).edges()]) # limited to only edges
            comm_sc = np.nanmean([signal_correlation[nodei, nodej] for nodei, nodej in itertools.combinations(comm, 2)]) # for all neuron pairs
            if comm_sc > sc_mean + 1 * sc_std:
              strong_module_dict[row][col][run].append(comm)
              print(row, col, comm_sc)
  return strong_module_dict
strong_module_dict = get_strongly_functional_module(sc_mean, sc_std, signal_correlation_dict, comms_dict, max_reso_subs)
#%%
############################ Results not good!!!
################################ number of srongly functional modules
def plot_num_strongly_functional_module(strong_module_dict):
  rows, cols = get_rowcol(strong_module_dict)
  runs = len(strong_module_dict[rows[0]][cols[0]])
  df = pd.DataFrame()
  for col in cols:
    num_modules = []
    for row in rows:
      for run in range(runs):
        num_modules.append(len(strong_module_dict[row][col][run]))
    df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(num_modules)[:,None], np.array([combine_stimulus(col)[1]] * len(num_modules))[:,None]), 1), columns=['number of modules', 'combined stimulus'])], ignore_index=True)
  df['number of modules'] = pd.to_numeric(df['number of modules'])
  fig, ax = plt.subplots(1, 1, figsize=(6, 5))
  sns.barplot(df, x='combined stimulus', y='number of modules', ax=ax)
  plt.show()
  return df

df = plot_num_strongly_functional_module(strong_module_dict)
#%%
############################ Results not good!!!
################################ get module signal correlation and purity
def get_module_signal_correlation_purity(G_dict, signal_correlation_dict, comms_dict):
  rows, cols = get_rowcol(comms_dict)
  runs = len(comms_dict[rows[0]][cols[0]])
  signal_corr_dict, purity_dict = {}, {}
  for row_ind, row in enumerate(rows):
    print(row)
    active_area = active_area_dict[row]
    signal_corr_dict[row], purity_dict[row] = {}, {}
    for combined_stimulus_name in combined_stimulus_names[1:]:
      signal_correlation = signal_correlation_dict[row][combined_stimulus_name]
      for col in combined_stimuli[combined_stimulus_names.index(combined_stimulus_name)]:
        G = G_dict[row][col].copy()
        signal_corr_dict[row][col], purity_dict[row][col] = {}, {}
        col_ind = stimulus_names.index(col)
        for run in range(runs):
          signal_corr_dict[row][col][run], purity_dict[row][col][run] = [], []
          comms_list = comms_dict[row][col]
          comms = comms_list[run]
          comms = [comm for comm in comms if len(comm) >= 4]
          for comm in comms:
            # comm_sc = np.nanmean([signal_correlation[nodei, nodej] for nodei, nodej in itertools.combinations(comm, 2)]) # for all neuron pairs
            comm_sc = np.nanmean([signal_correlation[nodei, nodej] for nodei, nodej in G.subgraph(comm).edges()]) # for connected neuron pairs
            if not np.isnan(comm_sc):
              c_regions = [active_area[node] for node in comm]
              _, counts = np.unique(c_regions, return_counts=True)
              signal_corr_dict[row][col][run].append(comm_sc)
              purity_dict[row][col][run].append(counts.max() / counts.sum())
  return signal_corr_dict, purity_dict

signal_corr_dict, purity_dict = get_module_signal_correlation_purity(G_ccg_dict, signal_correlation_dict, best_comms_dict)
#%%
############################ Results not good!!!
def scatter_signal_correlation_purity(signal_corr_dict, purity_dict):
  rows, cols = get_rowcol(signal_corr_dict)
  fig, axes = plt.subplots(1, 5, figsize=(3*5, 3))
  for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names[1:]):
    x, y = [], []
    ax = axes[cs_ind]
    ax.set_title(combined_stimulus_name.replace('\n', ' '), pad=30, fontsize=25)
    for col in combined_stimuli[combined_stimulus_names.index(combined_stimulus_name)]:
      for row in rows:
        for run in range(len(signal_corr_dict[row][col])):
          # x.append(np.mean(purity_dict[row][col][run]))
          # y.append(np.mean(signal_corr_dict[row][col][run]))
          x += purity_dict[row][col][run]
          y += signal_corr_dict[row][col][run]
    x, y = np.array(x), np.array(y)
    ax.scatter(x, y)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    line = slope*x+intercept
    locx, locy = .8, .8
    text = 'r={:.2f}, p={:.0e}'.format(r_value, p_value)
    ax.text(locx, locy, text, horizontalalignment='center',
        verticalalignment='center', transform=ax.transAxes, fontsize=25)
    # ax.scatter(binned_x, binned_y)
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)
    for axis in ['bottom', 'left']:
      ax.spines[axis].set_linewidth(1.5)
      ax.spines[axis].set_color('0.2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=1.5)
    if cs_ind == 0:
      ax.set_ylabel('Signal correlation', fontsize=30)
    ax.set_xlabel('Purity', fontsize=25)
  for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2.)
    ax.spines[axis].set_color('k')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.tick_params(width=2.)
  # plt.suptitle('{} average purity VS community size'.format(max_method), size=40)
  plt.tight_layout()
  # plt.savefig('./plots/signal_correlation_purity.pdf', transparent=True)
  plt.show()

scatter_signal_correlation_purity(signal_corr_dict, purity_dict)
#%%
############################ Results not good!!!
def plot_signal_correlation_purity(signal_corr_dict, purity_dict):
  rows, cols = get_rowcol(signal_corr_dict)
  fig, axes = plt.subplots(1, 5, figsize=(3*5, 3))
  for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names[1:]):
    x, y = [], []
    ax = axes[cs_ind]
    ax.set_title(combined_stimulus_name.replace('\n', ' '), pad=30, fontsize=25)
    for col in combined_stimuli[combined_stimulus_names.index(combined_stimulus_name)]:
      for row in rows:
        for run in range(len(signal_corr_dict[row][col])):
          x += purity_dict[row][col][run]
          y += signal_corr_dict[row][col][run]
    x, y = np.array(x), np.array(y)
    numbin = 6
    binned_x, binned_y = double_equal_binning(x, y, numbin=numbin, log=False)
    x_counts, y_counts = double_equal_binning_counts(x, y, numbin=numbin, log=False)
    count = binning_count(x, numbin=numbin, log=False)
    ax.bar(binned_x, binned_y, width=np.diff(binned_x).max()/1.5, color='.7')
    for i in range(len(binned_x)):
      ax.annotate('{}'.format(count[i]), xy=(binned_x[i],binned_y[i]), ha='center', va='bottom', fontsize=13)
    
    # dff = pd.DataFrame(np.vstack((x_counts, y_counts)), index=['connected', 'disconnected'], columns=binned_x)
    # table = sm.stats.Table(dff)
    # p_value = table.test_ordinal_association().pvalue
    
    table = sm.stats.Table.from_data(np.vstack((x, y)).T)
    p_value = table.test_ordinal_association().pvalue
    locx, locy = .4, .1
    ax.text(locx, locy, 'p={:.1e}'.format(p_value), horizontalalignment='center',
        verticalalignment='center', transform=ax.transAxes, fontsize=25)
    # ax.scatter(binned_x, binned_y)
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)
    for axis in ['bottom', 'left']:
      ax.spines[axis].set_linewidth(1.5)
      ax.spines[axis].set_color('0.2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=1.5)
    if cs_ind == 0:
      ax.set_ylabel('Signal correlation', fontsize=30)
    ax.set_xlabel('Purity', fontsize=25)
  for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2.)
    ax.spines[axis].set_color('k')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.tick_params(width=2.)
  # plt.suptitle('{} average purity VS community size'.format(max_method), size=40)
  plt.tight_layout()
  # plt.savefig('./plots/signal_correlation_purity.pdf', transparent=True)
  plt.show()

plot_signal_correlation_purity(signal_corr_dict, purity_dict)
#%%
################################ get module CCG weights/confidence and purity
def get_module_weight_confidence_purity(G_dict, comms_dict, max_neg_reso):
  rows, cols = get_rowcol(G_dict)
  weight_dict, confidence_dict, purity_dict = {}, {}, {}
  for row_ind, row in enumerate(rows):
    print(row)
    active_area = active_area_dict[row]
    weight_dict[row], confidence_dict[row], purity_dict[row] = {}, {}, {}
    for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
      for col in combined_stimuli[cs_ind]:
        G = G_ccg_dict[row][col]
        weight_dict[row][col], confidence_dict[row][col], purity_dict[row][col] = {}, {}, {}
        col_ind = stimulus_names.index(col)
        for run in range(metrics['Hamiltonian'].shape[-1]):
          weight_dict[row][col][run], confidence_dict[row][col][run], purity_dict[row][col][run] = [], [], []
          max_reso = max_neg_reso[row_ind, col_ind, run]
          comms_list = comms_dict[row][col][max_reso]
          comms = comms_list[run]
          comms = [comm for comm in comms if len(comm) >= 4]
          for comm in comms:
            c_regions = [active_area[node] for node in comm]
            _, counts = np.unique(c_regions, return_counts=True)
            if not np.isnan(np.mean([w for _, _, w in G.subgraph(comm).edges(data='weight')])):
              weight_dict[row][col][run].append(np.mean([w for _, _, w in G.subgraph(comm).edges(data='weight')]))
              confidence_dict[row][col][run].append(np.mean([w for _, _, w in G.subgraph(comm).edges(data='confidence')]))
              purity_dict[row][col][run].append(counts.max() / counts.sum())
  return weight_dict, confidence_dict, purity_dict

weight_dict, confidence_dict, purity_dict = get_module_weight_confidence_purity(G_ccg_dict, comms_dict, max_reso_subs)
#%%
def double_equal_binning(x, y, numbin=20, log=False):
  if log:
    bins = np.logspace(np.log10(x.min()), np.log10(x.max()), numbin) # log binning
  else:
    bins = np.linspace(x.min(), x.max(), numbin) # linear binning
  digitized = np.digitize(x, bins) # binning based on community size
  binned_x = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
  binned_y = [y[digitized == i].mean() for i in range(1, len(bins))]
  return binned_x, binned_y

def binning_count(x, numbin=20, log=False):
  if log:
    bins = np.logspace(np.log10(x.min()), np.log10(x.max()), numbin) # log binning
  else:
    bins = np.linspace(x.min(), x.max(), numbin) # linear binning
  digitized = np.digitize(x, bins) # binning based on community size
  count = [(digitized == i).sum() for i in range(1, len(bins))]
  return count

def plot_data_purity(data_dict, purity_dict, name='weight'):
  rows, cols = get_rowcol(data_dict)
  fig, axes = plt.subplots(1, 6, figsize=(3*6, 3))
  for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
    x, y = [], []
    ax = axes[cs_ind]
    ax.set_title(combined_stimulus_name.replace('\n', ' '), pad=30, fontsize=25)
    for col in combined_stimuli[combined_stimulus_names.index(combined_stimulus_name)]:
      for row in rows:
        for run in range(len(data_dict[row][col])):
          # x.append(np.mean(purity_dict[row][col][run]))
          # y.append(np.mean(data_dict[row][col][run]))
          x += purity_dict[row][col][run]
          y += data_dict[row][col][run]
    x, y = np.array(x), np.array(y)
    numbin = 6
    binned_x, binned_y = double_equal_binning(x, y, numbin=numbin, log=False)
    count = binning_count(x, numbin=numbin, log=False)
    ax.bar(binned_x, binned_y, width=np.diff(binned_x).max()/1.5, color='.7')
    for i in range(len(binned_x)):
      ax.annotate('{}'.format(count[i]), xy=(binned_x[i],binned_y[i]), ha='center', va='bottom', fontsize=13)
    table = sm.stats.Table.from_data(np.vstack((x, y)).T)
    p_value = table.test_ordinal_association().pvalue
    locx, locy = .4, .1
    ax.text(locx, locy, 'p={:.1e}'.format(p_value), horizontalalignment='center',
        verticalalignment='center', transform=ax.transAxes, fontsize=25)
    # ax.scatter(binned_x, binned_y)
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)
    for axis in ['bottom', 'left']:
      ax.spines[axis].set_linewidth(1.5)
      ax.spines[axis].set_color('0.2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=1.5)
    if cs_ind == 0:
      ax.set_ylabel(name.capitalize(), fontsize=30)
    ax.set_xlabel('Purity', fontsize=25)
  for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2.)
    ax.spines[axis].set_color('k')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.tick_params(width=2.)
  # plt.suptitle('{} average purity VS community size'.format(max_method), size=40)
  plt.tight_layout()
  plt.savefig('./plots/{}_purity.pdf'.format(name), transparent=True)
  # plt.show()

plot_data_purity(weight_dict, purity_dict, name='weight')
plot_data_purity(confidence_dict, purity_dict, name='confidence')
#%%
def plot_data_VSpurity_threshold(data_dict, purity_dict, th_list, name='weight'):
  rows, cols = get_rowcol(data_dict)
  runs = len(data_dict[rows[0]][cols[0]])
  fig, axes = plt.subplots(1, 2, figsize=(4*2, 4), sharex=True)
  mean_data, num_module = np.zeros((len(stimulus_types), len(th_list))), np.zeros((len(stimulus_types), len(th_list)))
  for st_ind, stimulus_type in enumerate(stimulus_types):
    x, y = [], []
    for col in stimulus_by_type[stimulus_types.index(stimulus_type)]:
      for row in session2keep:
        for run in range(runs):
          # x.append(np.mean(purity_dict[row][col][run]))
          # y.append(np.mean(data_dict[row][col][run]))
          x += purity_dict[row][col][run]
          y += data_dict[row][col][run]
    x, y = np.array(x), np.array(y)
    for th_ind, th in enumerate(th_list):
      inds = x>=th
      mean_data[st_ind, th_ind] = y[inds].mean()
      num_module[st_ind, th_ind] = inds.sum() / (runs * len(session2keep) * len(stimulus_by_type[st_ind]))
  for st_ind, stimulus_type in enumerate(stimulus_types):
    axes[0].plot(th_list, mean_data[st_ind], label=stimulus_type, color=stimulus_type_color[st_ind])
    axes[1].plot(th_list, num_module[st_ind], label=stimulus_type, color=stimulus_type_color[st_ind])
  axes[0].set_ylabel(name.capitalize(), fontsize=25)
  axes[1].set_ylabel('Number of modules', fontsize=25)
  for ax in axes:
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)
    for axis in ['bottom', 'left']:
      ax.spines[axis].set_linewidth(1.5)
      ax.spines[axis].set_color('k')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=1.5)
    ax.set_xlabel('Threshold on purity', fontsize=25)
  # plt.suptitle('{} average purity VS community size'.format(max_method), size=40)
  plt.tight_layout()
  plt.savefig('./plots/{}_VSpurity_threshold.pdf'.format(name), transparent=True)
  # plt.show()

th_list = np.arange(0, 1, 0.05)
plot_data_VSpurity_threshold(weight_dict, purity_dict, th_list, name='weight')
plot_data_VSpurity_threshold(confidence_dict, purity_dict, th_list, name='confidence')
#%%
# Figure 3B
################################ get module size and purity
def get_module_size_coverage_purity(G_dict, regions, best_comms_dict):
  rows, cols = get_rowcol(G_dict)
  runs = len(best_comms_dict[rows[0]][cols[0]])
  size_dict, coverage_dict, purity_dict = [{row:{col:{run:[] for run in range(runs)} for col in cols} for row in rows} for _ in range(3)]
  for row_ind, row in enumerate(rows):
    print(row)
    active_area = active_area_dict[row]
    # size_dict[row], purity_dict[row] = {}, {}
    for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
      for col in combined_stimuli[cs_ind]:
        G = G_ccg_dict[row][col]
        all_regions = [active_area[node] for node in G.nodes()]
        region_counts = np.array([all_regions.count(r) for r in regions])
        # size_dict[row][col], purity_dict[row][col] = {}, {}
        col_ind = stimulus_names.index(col)
        for run in range(runs):
          comms_list = best_comms_dict[row][col]
          comms = comms_list[run]
          comms = [comm for comm in comms if len(comm) >= 4]
          for comm in comms:
            c_regions = [active_area[node] for node in comm]
            counts = np.array([c_regions.count(r) for r in regions])
            if G.subgraph(comm).number_of_edges():
              size_dict[row][col][run].append(len(comm) / G.number_of_nodes()) # relative size
              coverage_dict[row][col][run].append((counts / region_counts).max())
              purity_dict[row][col][run].append(counts.max() / counts.sum())
  return size_dict, coverage_dict, purity_dict

size_dict, coverage_dict, purity_dict = get_module_size_coverage_purity(G_ccg_dict, visual_regions, best_comms_dict)
#%%
# Figure 3B
########################### merge data points from all mice into one distribution
def plot_num_module_VSpurity_threshold(size_dict, coverage_dict, purity_dict, sth_list_lin, sth_list_log, cth_list_lin, cth_list_log, pth_list):
  rows, cols = get_rowcol(purity_dict)
  runs = len(purity_dict[rows[0]][cols[0]])
  fig, axes = plt.subplots(1, 3, figsize=(5*3, 4), sharex=True, sharey=True)
  num_module1_lin, num_module1_log, num_module2_lin, num_module2_log, num_module3 = np.zeros((len(combined_stimulus_names), len(sth_list_lin))), np.zeros((len(combined_stimulus_names), len(sth_list_log))), np.zeros((len(combined_stimulus_names), len(cth_list_lin))), np.zeros((len(combined_stimulus_names), len(cth_list_log))), np.zeros((len(combined_stimulus_names), len(pth_list)))
  for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
    size, coverage, purity = [], [], []
    for col in combined_stimuli[cs_ind]:
      for row in session2keep:
        for run in range(runs):
          # purity.append(np.mean(purity_dict[row][col][run]))
          size += size_dict[row][col][run]
          coverage += coverage_dict[row][col][run]
          purity += purity_dict[row][col][run]
    size, coverage, purity = np.array(size), np.array(coverage), np.array(purity)
    for sth_ind, sth in enumerate(sth_list_lin):
      inds = size>=sth
      num_module1_lin[cs_ind, sth_ind] = inds.sum() / (runs * len(session2keep) * len(combined_stimuli[cs_ind]))
    for sth_ind, sth in enumerate(sth_list_log):
      inds = size>=sth
      num_module1_log[cs_ind, sth_ind] = inds.sum() / (runs * len(session2keep) * len(combined_stimuli[cs_ind]))
    for cth_ind, cth in enumerate(cth_list_lin):
      inds = coverage>=cth
      num_module2_lin[cs_ind, cth_ind] = inds.sum() / (runs * len(session2keep) * len(combined_stimuli[cs_ind]))
    for cth_ind, cth in enumerate(cth_list_log):
      inds = coverage>=cth
      num_module2_log[cs_ind, cth_ind] = inds.sum() / (runs * len(session2keep) * len(combined_stimuli[cs_ind]))
    for pth_ind, pth in enumerate(pth_list):
      inds = purity>=pth
      num_module3[cs_ind, pth_ind] = inds.sum() / (runs * len(session2keep) * len(combined_stimuli[cs_ind]))
  # dotted line
  for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
    axes[0].plot(sth_list_lin, num_module1_lin[cs_ind], label=combined_stimulus_name, color='.1', marker=stimulus2marker[combined_stimulus_name], markersize=scatter_size_dict[stimulus2marker[combined_stimulus_name]], alpha=1., markerfacecolor='w')
  for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
    axes[1].plot(cth_list_lin, num_module2_lin[cs_ind], label=combined_stimulus_name, color='.1', marker=stimulus2marker[combined_stimulus_name], markersize=scatter_size_dict[stimulus2marker[combined_stimulus_name]], alpha=1., markerfacecolor='w')
  for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
    axes[2].plot(pth_list, num_module3[cs_ind], label=combined_stimulus_name, color='.1', marker=stimulus2marker[combined_stimulus_name], markersize=scatter_size_dict[stimulus2marker[combined_stimulus_name]], alpha=1., markerfacecolor='w')
  axes[0].set_xlim(right=1)
  axes[1].set_xlim(right=1)
  axes[2].set_xlim(right=1)
  axins0 = inset_axes(axes[0], loc='upper right', width="70%", height="70%")
  for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
    axins0.plot(sth_list_log, num_module1_log[cs_ind], label=combined_stimulus_name, color='.1', marker=stimulus2marker[combined_stimulus_name], markersize=scatter_size_dict[stimulus2marker[combined_stimulus_name]]/1.54, alpha=1., markerfacecolor='w')
  axins1 = inset_axes(axes[1], loc='upper right', width="60%", height="60%")
  for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
    axins1.plot(cth_list_log, num_module2_log[cs_ind], label=combined_stimulus_name, color='.1', marker=stimulus2marker[combined_stimulus_name], markersize=scatter_size_dict[stimulus2marker[combined_stimulus_name]]/1.54, alpha=1., markerfacecolor='w')
  fontsize = 20
  for axins in [axins0, axins1]:
    axins.set_xscale('log')
    axins.set_yscale('log')
    axins.set_xlim(right=1.1)
    axins.xaxis.set_tick_params(labelsize=fontsize)
    axins.yaxis.set_tick_params(labelsize=fontsize)
    for axis in ['bottom', 'left']:
      axins.spines[axis].set_linewidth(1.5)
      axins.spines[axis].set_color('k')
    axins.spines['top'].set_visible(False)
    axins.spines['right'].set_visible(False)
    axins.tick_params(width=1.5)
  # axins1.set_xlim(left=np.power(10, -1.4))
  axes[0].set_ylabel('Number of modules', fontsize=fontsize)
  for ax in axes:
    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)
    for axis in ['bottom', 'left']:
      ax.spines[axis].set_linewidth(1.5)
      ax.spines[axis].set_color('k')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=1.5)
  axes[0].set_xlabel('Threshold on normalized size', fontsize=fontsize)
  axes[1].set_xlabel('Threshold on coverage', fontsize=fontsize)
  axes[2].set_xlabel('Threshold on purity', fontsize=fontsize)
  # plt.suptitle('{} average purity VS community size'.format(max_method), size=40)
  plt.tight_layout()
  plt.savefig('./plots/num_moduleVSsize_coverage_purity_inset.pdf', transparent=True)
  # plt.show()

sth_list_lin, sth_list_log, cth_list_lin, cth_list_log, pth_list = np.linspace(np.power(10, -1.9), np.power(10, -0.02), 20), np.logspace(-1.9, -0.02, 20), np.linspace(np.power(10, -1.9), np.power(10, -0.02), 20), np.logspace(-1.3, -0.02, 20), np.arange(0, 1, 0.05)
plot_num_module_VSpurity_threshold(size_dict, coverage_dict, purity_dict, sth_list_lin, sth_list_log, cth_list_lin, cth_list_log, pth_list)
#%%
########################### average across trials for each mouse with errorbars
def plot_num_module_VSpurity_threshold_error(size_dict, purity_dict, sth_list, pth_list):
  rows, cols = get_rowcol(purity_dict)
  runs = len(purity_dict[rows[0]][cols[0]])
  fig, axes = plt.subplots(1, 2, figsize=(5*2, 4))
  num_module1, num_module2 = [np.zeros((len(combined_stimulus_names), len(session2keep)*runs, len(sth_list))) for _ in range(2)]
  for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
    for row_ind, row in enumerate(session2keep):
      for run in range(runs):
        size, purity = [], []
        for col in combined_stimuli[cs_ind]:
          # purity.append(np.mean(purity_dict[row][col][run]))
          size += size_dict[row][col][run]
          purity += purity_dict[row][col][run]
        size, purity = np.array(size), np.array(purity)
        for sth_ind, sth in enumerate(sth_list):
          inds = size>=sth
          num_module1[cs_ind, row_ind*runs+run, sth_ind] = inds.sum() / (len(combined_stimuli[cs_ind]))
        for pth_ind, pth in enumerate(pth_list):
          inds = purity>=pth
          num_module2[cs_ind, row_ind*runs+run, pth_ind] = inds.sum() / (len(combined_stimuli[cs_ind]))
  for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
    axes[0].errorbar(sth_list, num_module1[cs_ind].mean(0), yerr=get_confidence_interval(num_module1[cs_ind]), fmt=' ', linewidth=1.,color='.1', zorder=1)
    axes[0].plot(sth_list, num_module1[cs_ind].mean(0), marker=' ', linewidth=2.,color='.1', zorder=1)
    axes[0].scatter(sth_list, num_module1[cs_ind].mean(0), marker=stimulus2marker[combined_stimulus_name], s=10*error_size_dict[stimulus2marker[combined_stimulus_name]], linewidth=1.,color='k', facecolor='white', zorder=2)
  for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
    axes[1].errorbar(pth_list, num_module2[cs_ind].mean(0), yerr=get_confidence_interval(num_module2[cs_ind]), fmt=' ', linewidth=1.,color='.1', zorder=1)
    axes[1].plot(pth_list, num_module2[cs_ind].mean(0), marker=' ', linewidth=2.,color='.1', zorder=1)
    axes[1].scatter(pth_list, num_module2[cs_ind].mean(0), marker=stimulus2marker[combined_stimulus_name], s=10*error_size_dict[stimulus2marker[combined_stimulus_name]], linewidth=1.,color='k', facecolor='white', zorder=2)
  
  axes[0].set_xscale('log')
  axes[0].set_yscale('log')
  axes[0].set_xlim(right=1)
  axes[1].set_xlim(right=1)
  # handles, labels = axes[0].get_legend_handles_labels()
  # axes[0].legend(handles, [label.replace('\n', ' ') for label in labels], title='', fontsize=14, frameon=False)
  fontsize = 20
  for ax in axes:
    ax.set_ylabel('Number of modules', fontsize=fontsize)
    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)
    for axis in ['bottom', 'left']:
      ax.spines[axis].set_linewidth(1.5)
      ax.spines[axis].set_color('k')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=1.5)
  axes[0].set_xlabel('Threshold on relative size', fontsize=fontsize)
  axes[1].set_xlabel('Threshold on purity', fontsize=fontsize)
  # plt.suptitle('{} average purity VS community size'.format(max_method), size=40)
  plt.tight_layout()
  plt.savefig('./plots/num_moduleVSsize_purity_error.pdf', transparent=True)
  # plt.show()

sth_list, pth_list = np.logspace(-2.4, -0.187, 20), np.arange(0, 1, 0.05)
plot_num_module_VSpurity_threshold_error(size_dict, purity_dict, sth_list, pth_list)
#%%
# Figure 3C
################################ get module size and purity for all stimuli separately for area
def get_module_size_coverage_purity_areawise(G_dict, regions, comms_dict):
  rows, cols = get_rowcol(G_dict)
  runs = len(comms_dict[rows[0]][cols[0]])
  size_dict, coverage_dict, purity_dict = [{row:{col:{region:{run:[] for run in range(runs)} for region in regions} for col in cols} for row in rows} for _ in range(3)]
  for row_ind, row in enumerate(rows):
    print(row)
    active_area = active_area_dict[row]
    # size_dict[row], purity_dict[row] = {}, {}
    for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
      for col in combined_stimuli[cs_ind]:
        G = G_ccg_dict[row][col]
        all_regions = [active_area[node] for node in G.nodes()]
        region_counts = np.array([all_regions.count(r) for r in regions])
        # size_dict[row][col], purity_dict[row][col] = {}, {}
        col_ind = stimulus_names.index(col)
        for run in range(runs):
          comms_list = comms_dict[row][col]
          comms = comms_list[run]
          comms = [comm for comm in comms if len(comm) >= 4]
          for comm in comms:
            c_regions = [active_area[node] for node in comm]
            counts = np.array([c_regions.count(r) for r in regions])
            dominant_area = regions[np.argmax(counts)]
            if G.subgraph(comm).number_of_edges():
              size_dict[row][col][dominant_area][run].append(len(comm) / G.number_of_nodes()) # relative size
              coverage_dict[row][col][dominant_area][run].append((counts / region_counts).max())
              purity_dict[row][col][dominant_area][run].append(counts.max() / counts.sum())
  return size_dict, coverage_dict, purity_dict

size_dict, coverage_dict, purity_dict = get_module_size_coverage_purity_areawise(G_ccg_dict, visual_regions, best_comms_dict)
#%%
# Figure 3C
########################### plot number of modules for all stimuli areawise
def plot_num_module_VSpurity_threshold_areawise(purity_dict, pth_list, regions):
  region_colors = ['#b3de69', '#80b1d3', '#fdb462', '#c3c3c3', '#fccde5', '#cec5f2']
  rows, cols = get_rowcol(purity_dict)
  runs = len(purity_dict[rows[0]][cols[0]])
  fig, axes = plt.subplots(1, 6, figsize=(4*6, 4), sharex=True, sharey=True)
  num_module = np.zeros((len(combined_stimulus_names), len(pth_list)))
  for r_ind, region in enumerate(regions[::-1]):
    ax = axes[r_ind]
    for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
      purity = []
      for col in combined_stimuli[cs_ind]:
        for row in session2keep:
          for run in range(runs):
            purity += purity_dict[row][col][region][run]
      purity = np.array(purity)
      for pth_ind, pth in enumerate(pth_list):
        inds = purity>=pth
        num_module[cs_ind, pth_ind] = inds.sum() / (runs * len(session2keep) * len(combined_stimuli[cs_ind]))
    # dotted line
    for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
      ax.plot(pth_list, num_module[cs_ind], label=combined_stimulus_name, color=region_colors[5-r_ind], marker=stimulus2marker[combined_stimulus_name], markersize=scatter_size_dict[stimulus2marker[combined_stimulus_name]]/1.2, alpha=1., markerfacecolor='w')
    ax.set_xlim(right=1)
    ax.set_title(region_labels[5-r_ind], fontsize=24)
  
  fontsize = 20
  axes[0].set_ylabel('Number of modules', fontsize=fontsize)
  for ax in axes:
    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)
    for axis in ['bottom', 'left']:
      ax.spines[axis].set_linewidth(1.5)
      ax.spines[axis].set_color('k')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=1.5)
    ax.set_xlabel('')
    # ax.set_xlabel('Threshold on purity', fontsize=fontsize)
  # plt.suptitle('{} average purity VS community size'.format(max_method), size=40)
  plt.tight_layout()
  plt.savefig('./plots/num_moduleVSpurity_areawise.pdf', transparent=True)
  # plt.show()

pth_list = np.arange(0, 1, 0.05)
plot_num_module_VSpurity_threshold_areawise(purity_dict, pth_list, visual_regions)
#%%
# Figure 3E,F
################################ get module size and purity for each area
def get_module_size_purity_each_area(G_dict, best_comms_dict, regions, combined_stimulus_name):
  rows, cols = get_rowcol(G_dict)
  runs = len(best_comms_dict[rows[0]][cols[0]])
  size_dict, purity_dict = {}, {}
  for row_ind, row in enumerate(rows):
    print(row)
    active_area = active_area_dict[row]
    size_dict[row], purity_dict[row] = {region:{run:[] for run in range(runs)} for region in regions}, {region:{run:[] for run in range(runs)} for region in regions}
    cs_ind = combined_stimulus_names.index(combined_stimulus_name)
    for col in combined_stimuli[cs_ind]:
      G = G_ccg_dict[row][col]
      # col_ind = stimulus_names.index(col)
      comms_list = best_comms_dict[row][col]
      for run in range(runs):
        comms = comms_list[run]
        comms = [comm for comm in comms if len(comm) >= 4]
        for comm in comms:
          c_regions = [active_area[node] for node in comm]
          # _, counts = np.unique(c_regions, return_counts=True)
          counts = np.array([c_regions.count(r) for r in regions])
          dominant_area = regions[np.argmax(counts)]
          if G.subgraph(comm).number_of_edges():
            size_dict[row][dominant_area][run].append(len(comm) / G.number_of_nodes()) # relative size
            purity_dict[row][dominant_area][run].append(counts.max() / counts.sum())
  return size_dict, purity_dict

combined_stimulus_name = 'Natural\nscenes'
# combined_stimulus_name = 'Natural\nmovies'
size_dict, purity_dict = get_module_size_purity_each_area(G_ccg_dict, best_comms_dict, visual_regions, combined_stimulus_name)
#%%
# Figure 3E,F
########################### merge data points from all mice into one distribution for each area
def plot_num_module_VSpurity_threshold_each_area(purity_dict, regions, pth_list, combined_stimulus_name):
  rows = list(purity_dict.keys())
  runs = len(purity_dict[rows[0]][regions[0]])
  print(runs)
  fig, ax = plt.subplots(1, 1, figsize=(5, 4))
  num_module = np.zeros((len(regions), len(pth_list), len(session2keep) * runs))
  cs_ind = combined_stimulus_names.index(combined_stimulus_name)
  for r_ind, region in enumerate(regions):
    for row_ind, row in enumerate(session2keep):
      for run in range(runs):
        purity = np.array(purity_dict[row][region][run])
        for pth_ind, pth in enumerate(pth_list):
          inds = purity>=pth
          num_module[r_ind, pth_ind, row_ind*runs+run] = inds.sum() / len(combined_stimuli[cs_ind]) # number of natural stimuli
  for r_ind, region in enumerate(regions):
    ax.errorbar(pth_list, num_module[r_ind].mean(-1), yerr=get_confidence_interval(num_module[r_ind].T).T, fmt=' ', linewidth=2.,color=region_colors[r_ind], alpha=.8, zorder=6-r_ind)
    ax.plot(pth_list, num_module[r_ind].mean(-1), label=region, color=region_colors[r_ind], marker='.', markersize=15, alpha=1., zorder=6-r_ind)
  ax.set_xlim(right=1)
  # handles, labels = axes[0].get_legend_handles_labels()
  # axes[0].legend(handles, [label.replace('\n', ' ') for label in labels], title='', fontsize=14, frameon=False)
  fontsize = 20
  ax.set_ylabel('Number of modules', fontsize=fontsize)
  ax.xaxis.set_tick_params(labelsize=fontsize)
  ax.yaxis.set_tick_params(labelsize=fontsize)
  for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(1.5)
    ax.spines[axis].set_color('k')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.tick_params(width=1.5)
  ax.set_xlabel('Threshold on purity', fontsize=fontsize)
  # plt.suptitle('{} average purity VS community size'.format(max_method), size=40)
  plt.tight_layout()
  plt.savefig('./plots/num_moduleVSsize_purity_all_error_each_area_{}.pdf'.format(combined_stimulus_name.replace('\n', '_')), transparent=True)
  # plt.show()

pth_list = np.arange(0.5, 1, 0.025)
plot_num_module_VSpurity_threshold_each_area(purity_dict, visual_regions, pth_list, combined_stimulus_name)
# #%%
# ########################### average across trials for each mouse with errorbars
# def plot_num_module_VSpurity_threshold_error_each_area(purity_dict, regions, pth_list, combined_stimulus_name):
#   rows = list(purity_dict.keys())
#   runs = len(purity_dict[rows[0]][regions[0]])
#   print(runs)
#   fig, ax = plt.subplots(1, 1, figsize=(5, 4))
#   num_module = np.zeros((len(regions), len(session2keep), len(pth_list)))
#   cs_ind = combined_stimulus_names.index(combined_stimulus_name)
#   for r_ind, region in enumerate(regions):
#     for row_ind, row in enumerate(session2keep):
#       purity = []
#       for run in range(runs):
#         purity += purity_dict[row][region][run]
#       purity = np.array(purity)
#       for pth_ind, pth in enumerate(pth_list):
#         inds = purity>=pth
#         num_module[r_ind, row_ind, pth_ind] = inds.sum() / (runs * len(combined_stimuli[cs_ind])) # number of natural stimuli
#   for r_ind, region in enumerate(regions):
#     # ax.plot(pth_list, num_module[r_ind], label=region, color=region_colors[r_ind], marker='.', markersize=15, alpha=1.)
#     ax.errorbar(pth_list, num_module[r_ind].mean(0), yerr=get_confidence_interval(num_module[r_ind]), fmt=' ', linewidth=2.,color=region_colors[r_ind], zorder=1)
#     ax.plot(pth_list, num_module[r_ind].mean(0), color=region_colors[r_ind], marker='.', markersize=15, alpha=1., zorder=2)
#   ax.set_xlim(right=1)
#   # handles, labels = axes[0].get_legend_handles_labels()
#   # axes[0].legend(handles, [label.replace('\n', ' ') for label in labels], title='', fontsize=14, frameon=False)
#   fontsize = 20
#   ax.set_ylabel('Number of modules', fontsize=fontsize)
#   ax.xaxis.set_tick_params(labelsize=fontsize)
#   ax.yaxis.set_tick_params(labelsize=fontsize)
#   for axis in ['bottom', 'left']:
#     ax.spines[axis].set_linewidth(1.5)
#     ax.spines[axis].set_color('k')
#   ax.spines['top'].set_visible(False)
#   ax.spines['right'].set_visible(False)
#   ax.tick_params(width=1.5)
#   ax.set_xlabel('Threshold on purity', fontsize=fontsize)
  
#   # plt.suptitle('{} average purity VS community size'.format(max_method), size=40)
#   plt.tight_layout()
#   plt.savefig('./plots/num_moduleVSsize_purity_mouse_error_each_area_{}.pdf'.format(combined_stimulus_name.replace('\n', '_')), transparent=True)
#   # plt.show()

# pth_list = np.arange(0.5, 1, 0.025)
# plot_num_module_VSpurity_threshold_error_each_area(purity_dict, visual_regions, pth_list, combined_stimulus_name)
#%%
def plot_purity_hist_each_area(size_dict, purity_dict, regions):
  rows = list(purity_dict.keys())
  runs = len(purity_dict[rows[0]][regions[0]])
  fig, axes = plt.subplots(1, 2, figsize=(5*2, 4))
  size_dist, purity_dist = [[] for _ in range(len(regions))], [[] for _ in range(len(regions))]
  df1, df2 = pd.DataFrame(), pd.DataFrame()
  for r_ind, region in enumerate(regions):
    size, purity = [], []
    for row in session2keep:
      for run in range(runs):
        # purity.append(np.mean(purity_dict[row][col][run]))
        size += size_dict[row][region][run]
        purity += purity_dict[row][region][run]
    df1 = pd.concat([df1, pd.DataFrame(np.concatenate((np.array(size)[:,None], np.array([region] * len(size))[:,None]), 1), columns=['size', 'region'])], ignore_index=True)
    df2 = pd.concat([df2, pd.DataFrame(np.concatenate((np.array(purity)[:,None], np.array([region] * len(purity))[:,None]), 1), columns=['purity', 'region'])], ignore_index=True)
  df1['size'] = pd.to_numeric(df1['size'])
  df2['purity'] = pd.to_numeric(df2['purity'])

  sns.histplot(
      data=df1, x="size", hue="region", hue_order=regions,
      log_scale=True, element="step", fill=False,
      cumulative=True, stat="density", common_norm=False, ax=axes[0]
  )
  sns.histplot(
      data=df2, x="purity", hue="region", hue_order=regions,
      log_scale=True, element="step", fill=False,
      cumulative=True, stat="density", common_norm=False, ax=axes[1]
  )

  # sns.histplot(data=df1, x="size", hue="region", hue_order=regions, ax=axes[0], palette=region_colors, multiple="stack")
  # sns.histplot(data=df2, x="purity", hue="region", hue_order=regions, ax=axes[1], palette=region_colors, multiple="stack")
  axes[0].set_xscale('log')
  axes[0].set_yscale('log')
  axes[0].set_xlim(right=1)

  # fontsize = 20
  # for ax in axes:
  #   ax.set_ylabel('Number of modules', fontsize=fontsize)
  #   ax.xaxis.set_tick_params(labelsize=fontsize)
  #   ax.yaxis.set_tick_params(labelsize=fontsize)
  #   for axis in ['bottom', 'left']:
  #     ax.spines[axis].set_linewidth(1.5)
  #     ax.spines[axis].set_color('k')
  #   ax.spines['top'].set_visible(False)
  #   ax.spines['right'].set_visible(False)
  #   ax.tick_params(width=1.5)
  # axes[0].set_xlabel('Threshold on relative size', fontsize=fontsize)
  # axes[1].set_xlabel('Threshold on purity', fontsize=fontsize)
  # plt.suptitle('{} average purity VS community size'.format(max_method), size=40)
  plt.tight_layout()
  # plt.savefig('./plots/num_moduleVSsize_purity_each_area.pdf', transparent=True)
  plt.show()

sth_list, pth_list = np.logspace(-2.4, -0.187, 20), np.arange(0, 1, 0.05)
plot_purity_hist_each_area(size_dict, purity_dict, visual_regions)
#%%
############################ Results not good!!!
################################ pairwise sign similairy/difference between comms of V1 area to comms of other areas
def get_DSsign_product_comm(G_dict, comms_dict, regions, max_neg_reso, pth, nth_list, etype='pos'):
  rows, cols = get_rowcol(G_dict)
  region_discre_same, region_discre_oppo = {r:{th:[] for th in nth_list} for r in regions}, {r:{th:[] for th in nth_list} for r in regions}
  for col in stimulus_by_type[3]: # only natural stimuli
    print(col)
    col_ind = cols.index(col)
    for row_ind, row in enumerate(session2keep):
      G = G_dict[row][col]
      nx.set_node_attributes(G, active_area_dict[row], "area")
      all_regions = [active_area_dict[row][node] for node in G.nodes()]
      region_counts = np.array([all_regions.count(r) for r in regions])
      for run in range(metrics['Hamiltonian'].shape[-1]):
        max_reso = max_neg_reso[row_ind, col_ind, run]
        comms_list = comms_dict[row][col][max_reso]
        comms = comms_list[run]
        comm_areas, comm_purity = [], []
        large_comms = [comm for comm in comms if len(comm)>=4]
        for comm in large_comms:
          c_regions = [active_area_dict[row][node] for node in comm]
          counts = np.array([c_regions.count(r) for r in regions])
          dominant_area = regions[np.argmax(counts)]
          comm_areas.append(dominant_area)
          comm_purity.append(counts.max() / counts.sum())
        nodes2plot = large_comms
        comm_mat = np.zeros((len(large_comms), len(large_comms)))
        for nodes1, nodes2 in itertools.permutations(nodes2plot, 2):
          if etype == 'pos':
            comm_mat[nodes2plot.index(nodes1), nodes2plot.index(nodes2)] = sum(1 for e in nx.edge_boundary(G, nodes1, nodes2) if G[e[0]][e[1]]['weight']>0)
          elif etype == 'neg':
            comm_mat[nodes2plot.index(nodes1), nodes2plot.index(nodes2)] = sum(1 for e in nx.edge_boundary(G, nodes1, nodes2) if G[e[0]][e[1]]['weight']<0)
          else:
            comm_mat[nodes2plot.index(nodes1), nodes2plot.index(nodes2)] = sum(1 for _ in nx.edge_boundary(G, nodes1, nodes2))
        comm_ds = (comm_mat - comm_mat.T) / (comm_mat + comm_mat.T)
        comm_ds[(comm_mat + comm_mat.T)==0] = 0
        for r_ind, region in enumerate(regions):
          if comm_areas.count(region) > 1:
            indx = np.where(np.array(comm_areas)==region)[0]
            product = [0, 0]
            for ind1, ind2 in itertools.combinations(indx, 2):
              if (comm_purity[ind1] >= pth) and (comm_purity[ind1] >= pth): # add constraint on purity for both modules
                result = np.delete(comm_ds[ind1], indx) * np.delete(comm_ds[ind2], indx)
                product[0] += (result > 0).sum()
                product[1] += (result < 0).sum()
            for th in nth_list:
              if sum(product) >= th:
                if product[0] == product[1]:
                  region_discre_same[region][th].append(safe_division(product[0], sum(product)))
                  region_discre_oppo[region][th].append(safe_division(product[1], sum(product)))
                elif product[0] > product[1]:
                  region_discre_same[region][th].append(safe_division(product[0], sum(product)))
                else:
                  region_discre_oppo[region][th].append(safe_division(product[1], sum(product)))
  return region_discre_same, region_discre_oppo

# etype = 'pos'
# etype = 'neg'
etype = 'all'
pth = .9
nth_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
region_discre_same, region_discre_oppo = get_DSsign_product_comm(G_ccg_dict, comms_dict, visual_regions, max_reso_subs, pth, nth_list, etype=etype)

################################ module patterns

def plot_consistencyVSnum_neighbor_module(region_discre_same, region_discre_oppo, pth, nth_list, regions, etype='pos'):
  fig, axes = plt.subplots(1, 2, figsize=(5*2, 4), sharex=True, sharey=True)
  axes[0].set_ylim(0.45, 1.05)
  for r_ind, region in enumerate(regions):
    # axes[0].plot(th_list, [np.mean(region_discre_same[region][r]) for r in region_discre_same[region]], label=region, color=region_colors[r_ind])
    # axes[1].plot(th_list, [np.mean(region_discre_oppo[region][r]) for r in region_discre_oppo[region]], label=region, color=region_colors[r_ind])
    ymean1, ystd1 = [np.mean(region_discre_same[region][th]) for th in region_discre_same[region]], [np.std(region_discre_same[region][th]) for th in region_discre_same[region]]
    ymean2, ystd2 = [np.mean(region_discre_oppo[region][th]) for th in region_discre_oppo[region]], [np.std(region_discre_oppo[region][th]) for th in region_discre_oppo[region]]
    ci1, ci2 = [get_confidence_interval(region_discre_same[region][th], confidence=0.95) for th in region_discre_same[region]], [get_confidence_interval(region_discre_oppo[region][th], confidence=0.95) for th in region_discre_oppo[region]]
    axes[0].errorbar(nth_list, ymean1, yerr=ci1, label=region, color=region_colors[r_ind])
    axes[1].errorbar(nth_list, ymean2, yerr=ci2, label=region, color=region_colors[r_ind])
  axes[0].set_ylabel('Same fraction', fontsize=25)
  axes[1].set_ylabel('Opposite fraction', fontsize=25)
  for ax in axes:
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)
    for axis in ['bottom', 'left']:
      ax.spines[axis].set_linewidth(1.5)
      ax.spines[axis].set_color('k')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=1.5)
    ax.set_xlabel('Threshold', fontsize=25)
  handles, labels = ax.get_legend_handles_labels()
  ax.legend(handles, labels, title='', fontsize=11, frameon=False)
  title = '{} edges only'.format(etype) if etype != 'all' else '{} edges'.format(etype)
  plt.suptitle(title, size=30)
  plt.tight_layout()
  plt.savefig('./plots/{}_consistency_{}_VS_num_threshold.pdf'.format(etype, pth), transparent=True)
  # plt.show()

plot_consistencyVSnum_neighbor_module(region_discre_same, region_discre_oppo, pth, nth_list, visual_regions, etype=etype)
#%%
############################ Results not good!!!
################################ pairwise distance metrics between comms of same area to comms of other areas, unweighted!!!
def jaccard_distance(a, b):
  a, b = np.where(a)[0], np.where(b)[0]
  c = set(a).intersection(set(b))
  return 1 - float(len(c)) / (len(a) + len(b) - len(c))

def jaccard_similarity(a, b):
  a, b = np.where(a)[0], np.where(b)[0]
  c = set(a).intersection(set(b))
  return float(len(c)) / (len(a) + len(b) - len(c))

def DSC(a, b):
  a, b = np.where(a)[0], np.where(b)[0]
  c = set(a).intersection(set(b))
  return 2 * float(len(c)) / (len(a) + len(b))

def manhattan_distance(a, b):
  a[np.where(a)], b[np.where(b)] = 1,  1
  return distance.minkowski(a, b, 1)

def euclidean_distance(a, b):
  a[np.where(a)], b[np.where(b)] = 1,  1
  return distance.minkowski(a, b, 2)

def chebyshev_distance(a, b):
  a[np.where(a)], b[np.where(b)] = 1,  1
  return distance.chebyshev(a, b)

def hamming_distance(a, b):
  a[np.where(a)], b[np.where(b)] = 1,  1
  return distance.hamming(a, b)

def cosine_distance(a, b):
  a[np.where(a)], b[np.where(b)] = 1,  1
  return distance.cosine(a, b)

def get_distance(a, b, name):
  if name == 'jaccard':
    m = jaccard_distance(a, b)
  elif name == 'DSC':
    m = DSC(a, b)
  elif name == 'manhattan':
    m = manhattan_distance(a, b)
  elif name == 'euclidean':
    m = euclidean_distance(a, b)
  elif name == 'chebyshev':
    m = chebyshev_distance(a, b)
  elif name == 'hamming':
    m = hamming_distance(a, b)
  elif name == 'cosine':
    m = cosine_distance(a, b)
  return m

def get_metric_comm(G_dict, comms_dict, regions, max_neg_reso, th_list, name='jaccard', etype='pos'):
  metric, metric_random = {r:{th:[] for th in th_list} for r in regions}, []
  for col in stimulus_by_type[3]: # only natural stimuli
    print(col)
    col_ind = cols.index(col)
    for row_ind, row in enumerate(session2keep):
      G = G_dict[row][col]
      nx.set_node_attributes(G, active_area_dict[row], "area")
      all_regions = [active_area_dict[row][node] for node in G.nodes()]
      region_counts = np.array([all_regions.count(r) for r in regions])
      for run in range(metrics['Hamiltonian'].shape[-1]):
        max_reso = max_neg_reso[row_ind, col_ind, run]
        comms_list = comms_dict[row][col][max_reso]
        comms = comms_list[run]
        comm_areas, comm_purity = [], []
        large_comms = [comm for comm in comms if len(comm)>=4]
        for comm in large_comms:
          c_regions = [active_area_dict[row][node] for node in comm]
          counts = np.array([c_regions.count(r) for r in regions])
          dominant_area = regions[np.argmax(counts)]
          comm_areas.append(dominant_area)
          comm_purity.append(counts.max() / counts.sum())
        nodes2plot = large_comms
        comm_mat = np.zeros((len(large_comms), len(large_comms)))
        for nodes1, nodes2 in itertools.permutations(nodes2plot, 2):
          if etype == 'pos':
            comm_mat[nodes2plot.index(nodes1), nodes2plot.index(nodes2)] = sum(1 for e in nx.edge_boundary(G, nodes1, nodes2) if G[e[0]][e[1]]['weight']>0)
          elif etype == 'neg':
            comm_mat[nodes2plot.index(nodes1), nodes2plot.index(nodes2)] = sum(1 for e in nx.edge_boundary(G, nodes1, nodes2) if G[e[0]][e[1]]['weight']<0)
          else:
            comm_mat[nodes2plot.index(nodes1), nodes2plot.index(nodes2)] = sum(1 for _ in nx.edge_boundary(G, nodes1, nodes2))
        comm_ds = (comm_mat - comm_mat.T) / (comm_mat + comm_mat.T)
        comm_ds[(comm_mat + comm_mat.T)==0] = 0
        for region in regions:
          if comm_areas.count(region) > 1:
            indx = np.where(np.array(comm_areas)==region)[0]
            for ind1, ind2 in itertools.combinations(indx, 2):
              for th in th_list:
                if (comm_purity[ind1] >= th) and (comm_purity[ind2] >= th):
                  if len(np.where(np.delete(comm_ds[ind1], indx))[0]) and len(np.where(np.delete(comm_ds[ind2], indx))[0]): # at least one of them has a neighbor module
                    a, b = np.delete(comm_ds[ind1], indx), np.delete(comm_ds[ind2], indx)
                    metric[region][th].append(get_distance(a, b, name))
        mrs = []
        for i, j in itertools.combinations(range(len(large_comms)), 2):
          if comm_areas[i] != comm_areas[j]:
            indx = [i, j]
            if len(np.where(np.delete(comm_ds[i], indx))[0]) and len(np.where(np.delete(comm_ds[j], indx))[0]): # at least one of them has a neighbor module
              a, b = np.delete(comm_ds[i], indx), np.delete(comm_ds[j], indx)
              mrs.append(get_distance(a, b, name))
        if not np.isnan(np.nanmean(mrs)):
          metric_random.append(np.nanmean(mrs))
  return metric, metric_random

names = ['jaccard', 'DSC', 'manhattan', 'euclidean', 'chebyshev', 'hamming', 'cosine']
name = names[6]
# etype = 'pos'
# etype = 'neg'
etype = 'all'
pth_list = np.arange(.5, 1, .05)
metric, metric_random = get_metric_comm(G_ccg_dict, comms_dict, visual_regions, max_reso_subs, pth_list, name=name, etype=etype)

################################ Z score of Jaccard_distanceVSpurity
def plot_zscore_metricVSpurity(metric, metric_random, th_list, regions, name='jaccard', etype='pos'):
  fig, ax = plt.subplots(1, 1, figsize=(6, 5), sharex=True, sharey=True)
  # ax.set_ylim(0.45, 1.05)
  rm, rstd = np.nanmean(metric_random), np.nanstd(metric_random)
  print(rm, rstd)
  for r_ind, region in enumerate(regions):
    ymean = [np.mean((metric[region][th]-rm)/rstd) for th in metric[region]]
    ci = [get_confidence_interval((np.array(metric[region][th])-rm)/rstd, confidence=0.95) for th in metric[region]]
    ax.errorbar(th_list, ymean, yerr=ci, label=region, color=region_colors[r_ind])
  ax.set_ylabel('Z score of {}'.format(name), fontsize=25)
  ax.xaxis.set_tick_params(labelsize=30)
  ax.yaxis.set_tick_params(labelsize=30)
  for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(1.5)
    ax.spines[axis].set_color('k')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.tick_params(width=1.5)
  ax.set_xlabel('Threshold on purity', fontsize=25)
  handles, labels = ax.get_legend_handles_labels()
  ax.legend(handles, labels, title='', fontsize=11, frameon=False)
  title = '{} edges only'.format(etype) if etype != 'all' else '{} edges'.format(etype)
  plt.suptitle(title, size=30)
  plt.tight_layout()
  plt.savefig('./plots/{}_zscore_{}_VS_purity_threshold.pdf'.format(etype, name), transparent=True)
  # plt.show()

plot_zscore_metricVSpurity(metric, metric_random, pth_list, visual_regions, name=name, etype=etype)
#%%
############################ Results not good!!!
################################ pairwise Jensen Shannon distance metrics between comms of same area to comms of other areas, weighted!!!
def get_weighted_metric_comm(G_dict, comms_dict, regions, max_neg_reso, th_list, direct=False, etype='pos'):
  metric, metric_random = {r:{th:[] for th in th_list} for r in regions}, []
  for col in stimulus_by_type[3]: # only natural stimuli
    print(col)
    col_ind = cols.index(col)
    for row_ind, row in enumerate(session2keep):
      G = G_dict[row][col]
      nx.set_node_attributes(G, active_area_dict[row], "area")
      all_regions = [active_area_dict[row][node] for node in G.nodes()]
      region_counts = np.array([all_regions.count(r) for r in regions])
      for run in range(metrics['Hamiltonian'].shape[-1]):
        max_reso = max_neg_reso[row_ind, col_ind, run]
        comms_list = comms_dict[row][col][max_reso]
        comms = comms_list[run]
        comm_areas, comm_purity = [], []
        large_comms = [comm for comm in comms if len(comm)>=4]
        for comm in large_comms:
          c_regions = [active_area_dict[row][node] for node in comm]
          counts = np.array([c_regions.count(r) for r in regions])
          dominant_area = regions[np.argmax(counts)]
          comm_areas.append(dominant_area)
          comm_purity.append(counts.max() / counts.sum())
        nodes2plot = large_comms
        comm_mat = np.zeros((len(large_comms), len(large_comms)))
        if direct:
          for nodes1, nodes2 in itertools.permutations(nodes2plot, 2):
            if etype == 'pos':
              comm_mat[nodes2plot.index(nodes1), nodes2plot.index(nodes2)] = sum(1 for e in nx.edge_boundary(G, nodes1, nodes2) if G[e[0]][e[1]]['weight']>0)
            elif etype == 'neg':
              comm_mat[nodes2plot.index(nodes1), nodes2plot.index(nodes2)] = sum(1 for e in nx.edge_boundary(G, nodes1, nodes2) if G[e[0]][e[1]]['weight']<0)
            else:
              comm_mat[nodes2plot.index(nodes1), nodes2plot.index(nodes2)] = sum(1 for _ in nx.edge_boundary(G, nodes1, nodes2))
        else:
          for nodes1, nodes2 in itertools.permutations(nodes2plot, 2):
            if etype == 'pos':
              comm_mat[nodes2plot.index(nodes1), nodes2plot.index(nodes2)] = sum(1 for e in nx.edge_boundary(G, nodes1, nodes2) if G[e[0]][e[1]]['weight']>0) + sum(1 for e in nx.edge_boundary(G, nodes2, nodes1) if G[e[0]][e[1]]['weight']>0)
            elif etype == 'neg':
              comm_mat[nodes2plot.index(nodes1), nodes2plot.index(nodes2)] = sum(1 for e in nx.edge_boundary(G, nodes1, nodes2) if G[e[0]][e[1]]['weight']<0) + sum(1 for e in nx.edge_boundary(G, nodes2, nodes1) if G[e[0]][e[1]]['weight']<0)
            else:
              comm_mat[nodes2plot.index(nodes1), nodes2plot.index(nodes2)] = sum(1 for _ in nx.edge_boundary(G, nodes1, nodes2)) + sum(1 for _ in nx.edge_boundary(G, nodes2, nodes1))
        for region in regions:
          if comm_areas.count(region) > 1:
            indx = np.where(np.array(comm_areas)==region)[0]
            for ind1, ind2 in itertools.combinations(indx, 2):
              for th in th_list:
                if (comm_purity[ind1] >= th) and (comm_purity[ind2] >= th):
                  if np.delete(comm_mat[ind1], indx).sum() and np.delete(comm_mat[ind2], indx).sum(): # at least one of them has a neighbor module
                    a, b = np.delete(comm_mat[ind1], indx), np.delete(comm_mat[ind2], indx)
                    metric[region][th].append(distance.jensenshannon(a/a.sum(), b/b.sum()))
        mrs = []
        for i, j in itertools.combinations(range(len(large_comms)), 2):
          if comm_areas[i] != comm_areas[j]:
            indx = [i, j]
            if np.delete(comm_mat[i], indx).sum() and np.delete(comm_mat[j], indx).sum(): # at least one of them has a neighbor module
              a, b = np.delete(comm_mat[i], indx), np.delete(comm_mat[j], indx)
              mrs.append(distance.jensenshannon(a/a.sum(), b/b.sum()))
        if not np.isnan(np.nanmean(mrs)):
          metric_random.append(np.nanmean(mrs))
  return metric, metric_random

direct = False
# direct = True
etype = 'pos'
# etype = 'neg'
# etype = 'all'
pth_list = np.arange(.5, 1, .05)
metric, metric_random = get_weighted_metric_comm(G_ccg_dict, comms_dict, visual_regions, max_reso_subs, pth_list, direct=direct, etype=etype)

################################ Z score of Jaccard_distanceVSpurity
def plot_zscore_weighted_metricVSpurity(metric, metric_random, th_list, regions, direct=False, etype='pos'):
  fig, ax = plt.subplots(1, 1, figsize=(6, 5), sharex=True, sharey=True)
  # ax.set_ylim(0.45, 1.05)
  rm, rstd = np.nanmean(metric_random), np.nanstd(metric_random)
  print(rm, rstd)
  for r_ind, region in enumerate(regions):
    ymean = [np.mean((metric[region][th]-rm)/rstd) for th in metric[region]]
    ci = [get_confidence_interval((np.array(metric[region][th])-rm)/rstd, confidence=0.95) for th in metric[region]]
    ax.errorbar(th_list, ymean, yerr=ci, label=region, color=region_colors[r_ind])
  ax.set_ylabel('Z score of JS distance', fontsize=25)
  ax.xaxis.set_tick_params(labelsize=30)
  ax.yaxis.set_tick_params(labelsize=30)
  for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(1.5)
    ax.spines[axis].set_color('k')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.tick_params(width=1.5)
  ax.set_xlabel('Threshold on purity', fontsize=25)
  handles, labels = ax.get_legend_handles_labels()
  ax.legend(handles, labels, title='', fontsize=11, frameon=False)
  title = '{} edges only'.format(etype) if etype != 'all' else '{} edges'.format(etype)
  plt.suptitle(title, size=30)
  plt.tight_layout()
  fname = './plots/{}_zscore_JS_distance_VS_purity_threshold.pdf' if not direct else './plots/{}_zscore_direct_JS_distance_VS_purity_threshold.pdf'
  plt.savefig(fname.format(etype), transparent=True)
  # plt.show()

plot_zscore_weighted_metricVSpurity(metric, metric_random, pth_list, visual_regions, direct=direct, etype=etype)
#%%
################################ data VS area distance for pairwise modules
def get_data_area_distance_module(G_dict, comms_dict, regions, max_neg_reso):
  pos_density, neg_frac, area_distance = {r:[] for r in regions}, {r:[] for r in regions}, {r:[] for r in regions}
  for col in stimulus_by_type[3]: # only natural stimuli
    print(col)
    col_ind = cols.index(col)
    for row_ind, row in enumerate(session2keep):
      G = G_dict[row][col]
      nx.set_node_attributes(G, active_area_dict[row], "area")
      all_regions = [active_area_dict[row][node] for node in G.nodes()]
      region_counts = np.array([all_regions.count(r) for r in regions])
      for run in range(len(comms_dict[row][col][0])):
        max_reso = max_neg_reso[row_ind, col_ind, run]
        comms_list = comms_dict[row][col][max_reso]
        comms = comms_list[run]
        comm_areas, comm_area_distr = [], []
        large_comms = [comm for comm in comms if len(comm)>=4]
        for comm in large_comms:
          c_regions = [active_area_dict[row][node] for node in comm]
          counts = np.array([c_regions.count(r) for r in regions])
          dominant_area = regions[np.argmax(counts)]
          comm_areas.append(dominant_area)
          comm_area_distr.append(counts / counts.sum())
        nodes2plot = large_comms
        pos_density_mat, neg_frac_mat = np.zeros((len(large_comms), len(large_comms))), np.zeros((len(large_comms), len(large_comms)))
        for nodes1, nodes2 in itertools.permutations(nodes2plot, 2):
          pos_density_mat[nodes2plot.index(nodes1), nodes2plot.index(nodes2)] = (sum(1 for e in nx.edge_boundary(G, nodes1, nodes2) if G[e[0]][e[1]]['weight']>0) + sum(1 for e in nx.edge_boundary(G, nodes2, nodes1) if G[e[0]][e[1]]['weight']>0)) / (len(nodes1) * len(nodes2))
          if (sum(1 for _ in nx.edge_boundary(G, nodes1, nodes2)) + sum(1 for _ in nx.edge_boundary(G, nodes2, nodes1))):
            neg_frac_mat[nodes2plot.index(nodes1), nodes2plot.index(nodes2)] = (sum(1 for e in nx.edge_boundary(G, nodes1, nodes2) if G[e[0]][e[1]]['weight']<0) + sum(1 for e in nx.edge_boundary(G, nodes2, nodes1) if G[e[0]][e[1]]['weight']<0)) / (sum(1 for _ in nx.edge_boundary(G, nodes1, nodes2)) + sum(1 for _ in nx.edge_boundary(G, nodes2, nodes1)))
          else:
            neg_frac_mat[nodes2plot.index(nodes1), nodes2plot.index(nodes2)] = -1 # for modules with no connections
        for region in regions:
          if comm_areas.count(region) > 1:
            indx = np.where(np.array(comm_areas)==region)[0]
            for ind1, ind2 in itertools.combinations(indx, 2):
              # for th in th_list:
              #   if (comm_purity[ind1] >= th) and (comm_purity[ind2] >= th):
                  pos_density[region].append(pos_density_mat[ind1, ind2])
                  neg_frac[region].append(neg_frac_mat[ind1, ind2]) # could be -1 for modules with no connections
                  area_distance[region].append(distance.jensenshannon(comm_area_distr[ind1], comm_area_distr[ind2]))
  return pos_density, neg_frac, area_distance

pos_density, neg_frac, area_distance = get_data_area_distance_module(G_ccg_dict, comms_dict, visual_regions, max_reso_subs)
#%%
import statsmodels.api as sm
def double_equal_binning(x, y, numbin=20, log=False):
  if log:
    bins = np.logspace(np.log10(x.min()), np.log10(x.max()), numbin) # log binning
  else:
    bins = np.linspace(x.min(), x.max(), numbin) # linear binning
  digitized = np.digitize(x, bins) # binning based on community size
  binned_x = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
  binned_y = [y[digitized == i].mean() for i in range(1, len(bins))]
  return binned_x, binned_y

def binning_count(x, numbin=20, log=False):
  if log:
    bins = np.logspace(np.log10(x.min()), np.log10(x.max()), numbin) # log binning
  else:
    bins = np.linspace(x.min(), x.max(), numbin) # linear binning
  digitized = np.digitize(x, bins) # binning based on community size
  count = [(digitized == i).sum() for i in range(1, len(bins))]
  return count

def plot_data_area_distance(area_distance, data, regions, name='positive_density'):
  fig, axes = plt.subplots(1, len(regions), figsize=(3.5*6, 3.5))
  for r_ind, region in enumerate(regions):
    x, y = area_distance[region], data[region]
    ax = axes[r_ind]
    ax.set_title(region_labels[r_ind], pad=30, fontsize=25)
    x, y = np.array(x), np.array(y)
    inds = np.where(y!=-1)
    x, y = x[inds], y[inds]
    numbin = 6
    binned_x, binned_y = double_equal_binning(x, y, numbin=numbin, log=False)
    count = binning_count(x, numbin=numbin, log=False)
    ax.bar(binned_x, binned_y, width=np.diff(binned_x).max()/1.5, color='.7')
    for i in range(len(binned_x)):
      ax.annotate('{}'.format(count[i]), xy=(binned_x[i],binned_y[i]), ha='center', va='bottom', fontsize=13)
    table = sm.stats.Table.from_data(np.vstack((x, y)).T)
    p_value = table.test_ordinal_association().pvalue
    locx, locy = .4, .1
    ax.text(locx, locy, 'p={:.1e}'.format(p_value), horizontalalignment='center',
        verticalalignment='center', transform=ax.transAxes, fontsize=25)
    # ax.scatter(binned_x, binned_y)
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)
    for axis in ['bottom', 'left']:
      ax.spines[axis].set_linewidth(1.5)
      ax.spines[axis].set_color('0.2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=1.5)
    if r_ind == 0:
      ax.set_ylabel(name.capitalize(), fontsize=30)
    ax.set_xlabel('JS distance of areas', fontsize=25)
  for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2.)
    ax.spines[axis].set_color('k')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.tick_params(width=2.)
  plt.tight_layout()
  plt.savefig('./plots/{}_area_distance.pdf'.format(name), transparent=True)
  # plt.show()

plot_data_area_distance(area_distance, pos_density, visual_regions, name='positive_density')
plot_data_area_distance(area_distance, neg_frac, visual_regions, name='negative_fraction')
#%%
################################ Signal correlation of within/cross module per area
# def get_signal_corr_within_cross_comm_same_area(G_dict, signal_correlation_dict, comms_dict, regions):
#   rows = signal_correlation_dict.keys()
#   within_comm_dict, cross_comm_dict, rand_cross_comm_list = {row:{r:[] for r in regions} for row in rows}, {row:{r:[] for r in regions} for row in rows}, []
#   for row_ind, row in enumerate(rows):
#     print(row)
#     active_area = active_area_dict[row]
#     node_idx = sorted(active_area.keys())   
#     for combined_stimulus_name in ['Natural\nscenes', 'Natural\nmovies']:
#       # within_comm_dict[row][combined_stimulus_name], cross_comm_dict[row][combined_stimulus_name] = [], []
#       signal_correlation = signal_correlation_dict[row][combined_stimulus_name]
#       for col in combined_stimuli[combined_stimulus_names.index(combined_stimulus_name)]:
#         col_ind = stimulus_names.index(col)
#         G = G_dict[row][col]
#         for run in range(len(comms_dict[row][col][0])):
#           comms_list = comms_dict[row][col]
#           comms = comms_list[run]
#           large_comms = [comm for comm in comms if len(comm) >= 4]
#           within_comm, cross_comm = [], []
#           node_to_community = comm2partition(comms)
#           comm_areas = []
#           for comm in large_comms:
#             c_regions = [active_area_dict[row][node] for node in comm]
#             counts = np.array([c_regions.count(r) for r in regions])
#             dominant_area = regions[np.argmax(counts)]
#             comm_areas.append(dominant_area)
#           for region in regions:
#             if comm_areas.count(region) >= 1:
#               indx = np.where(np.array(comm_areas)==region)[0]
#               for ind in indx:
#                 inds = np.array(list(large_comms[ind]))
#                 within_comm_dict[row][region].append(np.nanmean(signal_correlation[inds[:,None], inds][np.triu_indices(len(inds), 1)].tolist()))
#             if comm_areas.count(region) > 1:
#               indx = np.where(np.array(comm_areas)==region)[0]
#               for ind1, ind2 in itertools.combinations(indx, 2):
#                 inds1, inds2 = np.array(list(large_comms[ind1])), np.array(list(large_comms[ind2]))
#                 cross_comm_dict[row][region].append(np.nanmean(signal_correlation[inds1[:,None], inds2].flatten().tolist()))

#           for nodei, nodej in itertools.combinations(node_idx, 2): # for all neurons
#             if node_to_community[nodei] != node_to_community[nodej]:
#               rand_cross_comm_list.append(signal_correlation[nodei, nodej])
#   df = pd.DataFrame()
#   for region in regions:
#     print(region)
#     for row in session2keep:
#       within_comm, cross_comm = within_comm_dict[row][region], cross_comm_dict[row][region]
#       within_comm, cross_comm = [e for e in within_comm if not np.isnan(e)], [e for e in cross_comm if not np.isnan(e)] # remove nan values
#       df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(within_comm)[:,None], np.array(['within community'] * len(within_comm))[:,None], np.array([region] * len(within_comm))[:,None]), 1), columns=['signal correlation', 'type', 'region'])], ignore_index=True)
#       df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(cross_comm)[:,None], np.array(['cross community'] * len(cross_comm))[:,None], np.array([region] * len(cross_comm))[:,None]), 1), columns=['signal correlation', 'type', 'region'])], ignore_index=True)
#   df['signal correlation'] = pd.to_numeric(df['signal correlation'])
#   return df, rand_cross_comm_list

# start_time = time.time()
# signalcorr_within_cross_comm_same_area_df, rand_cross_comm_list = get_signal_corr_within_cross_comm_same_area(G_ccg_dict, signal_correlation_dict, best_comms_dict, visual_regions)
# print("--- %s minutes" % ((time.time() - start_time)/60))

#%%
############################ check why not consistent with plot_data_within_cross_comm_same_area_significance!!!
# def plot_signal_correlation_within_cross_comm_same_area_significance(df, rand_cross_comm_list, regions):
#   fig, ax = plt.subplots(1,1, sharex=True, sharey=True, figsize=(1.5*(len(combined_stimulus_names)-1), 6))
#   # palette = [[plt.cm.tab20b(i) for i in range(4)][i] for i in [0, 3]]
#   palette = ['w','k']
#   barplot = sns.barplot(x='region', y='signal correlation', hue="type", hue_order=['cross community','within community'], palette=palette, ec='k', linewidth=2., data=df, ax=ax, capsize=.05, width=0.6)
#   ax.axhline(y=np.nanmean(rand_cross_comm_list), linestyle='--', linewidth=3, color='.2', label='cross random module')
#   ax.yaxis.set_tick_params(labelsize=30)
#   plt.setp(ax.get_legend().get_title(), fontsize='0') # for legend title
#   plt.xticks(range(len(regions)), region_labels, fontsize=30)
#   ax.xaxis.set_tick_params(length=0)
#   for axis in ['bottom', 'left']:
#     ax.spines[axis].set_linewidth(2.5)
#     ax.spines[axis].set_color('k')
#   ax.spines['top'].set_visible(False)
#   ax.spines['right'].set_visible(False)
#   ax.tick_params(width=2.5)
#   ax.set_xlabel('')
#   ax.set_ylabel('')
#   # ax.set_ylabel('Signal correlation', fontsize=30)
#   handles, labels = ax.get_legend_handles_labels()
#   ax.legend(handles, [], fontsize=0)
#   # add significance annotation
#   alpha_list = [.0001, .001, .01, .05]
#   h, l = 0.008, .005
#   for r_ind, region in enumerate(regions):
#     within_comm, cross_comm = df[(df.region==region)&(df.type=='within community')]['signal correlation'].values.flatten(), df[(df.region==region)&(df.type=='cross community')]['signal correlation'].values.flatten()
#     _, p = ztest(within_comm, cross_comm, value=0)
#     diff_star = '*' * (len(alpha_list) - bisect(alpha_list, p)) if len(alpha_list) > bisect(alpha_list, p) else 'ns'
#     within_sr, cross_sr = confidence_interval(within_comm)[1], confidence_interval(cross_comm)[1]
#     # within_sr, cross_sr = within_comm.mean(), cross_comm.mean()
#     within_sr = within_sr + h
#     cross_sr = cross_sr + h
#     annot_difference(diff_star, -.15 + r_ind, .15 + r_ind, max(within_sr, cross_sr), l, 2.5, 28, ax=ax)
#   ax.invert_xaxis()
#   plt.tight_layout(rect=[0.03, 0., 1., 1])
#   # plt.show()
#   plt.savefig('./plots/signal_correlation_within_cross_comm_same_area.pdf', transparent=True)

# plot_signal_correlation_within_cross_comm_same_area_significance(signalcorr_within_cross_comm_same_area_df, rand_cross_comm_list, visual_regions)
#%%
# Figure 3E,F
def save_within_cross_random_legend():
  fig, ax = plt.subplots(1,1, figsize=(.4*(len(combined_stimulus_names)-1), .5))
  palette = ['k','w']
  df = pd.DataFrame([[0,0,0],[0,0,1]], columns=['x', 'y', 'type'])
  barplot = sns.barplot(x='x', y='y', hue="type", hue_order=[0, 1], palette=palette, ec='k', linewidth=2., data=df, ax=ax, capsize=.05, width=0.6)
  ax.axhline(y=1, linestyle='--', linewidth=3, color='.2', label='cross random module')
  handles, labels = ax.get_legend_handles_labels()
  legend = ax.legend(handles[1:] + [handles[0]], ['within module', 'cross module (within area)', 'cross module (cross area)'], title='', bbox_to_anchor=(0.1, -1.), handletextpad=0.3, loc='upper left', fontsize=25, frameon=False, ncol=3) # change legend order to within/cross similar module then cross random
  plt.axis('off')
  export_legend(legend, './plots/within_cross_random_module_legend.pdf')
  plt.tight_layout()
  # plt.savefig('./plots/stimuli_markers.pdf', transparent=True)
  plt.show()

save_within_cross_random_legend()
#%%
# Figure 3E,F
################################ positive density and negative fraction of within/cross module per area
def get_data_module_per_area(G_dict, comms_dict, regions, combined_stimulus_name):
  within_comm_dict, cross_comm_dict, within_pos_density, cross_pos_density, within_pos_frac, cross_pos_frac, within_neg_frac, cross_neg_frac = {r:[] for r in regions}, {r:[] for r in regions}, {r:[] for r in regions}, {r:[] for r in regions}, {r:[] for r in regions}, {r:[] for r in regions}, {r:[] for r in regions}, {r:[] for r in regions}
  rand_cross_comm_list, rand_cross_pos_density_list, rand_cross_pos_frac_list, rand_cross_neg_frac_list = [], [], [], []
  cs_ind = combined_stimulus_names.index(combined_stimulus_name)
  for col in combined_stimuli[cs_ind]:
  # for col in stimulus_by_type[3]: # only natural stimuli
    print(col)
    col_ind = cols.index(col)
    for row_ind, row in enumerate(session2keep):
      signal_correlation = signal_correlation_dict[row][combine_stimulus(col)[1]]
      active_area = active_area_dict[row]
      node_idx = sorted(active_area.keys())
      G = G_dict[row][col].copy()
      nx.set_node_attributes(G, active_area_dict[row], "area")
      for run in range(len(comms_dict[row][col][0])):
        comms_list = comms_dict[row][col]
        comms = comms_list[run]
        comm_areas = []
        large_comms = [comm for comm in comms if len(comm)>=4]
        node_to_community = comm2partition(comms)
        for comm in large_comms:
          c_regions = [active_area_dict[row][node] for node in comm]
          counts = np.array([c_regions.count(r) for r in regions])
          dominant_area = regions[np.argmax(counts)]
          comm_areas.append(dominant_area)
        nodes2plot = large_comms
        pos_density_mat, pos_frac_mat, neg_frac_mat = np.zeros((len(large_comms), len(large_comms))), np.zeros((len(large_comms), len(large_comms))), np.zeros((len(large_comms), len(large_comms)))
        for nodes1, nodes2 in itertools.permutations(nodes2plot, 2):
          # pos_density_mat[nodes2plot.index(nodes1), nodes2plot.index(nodes2)] = (sum(1 for e in nx.edge_boundary(G, nodes1, nodes2) if G[e[0]][e[1]]['weight']>0) + sum(1 for e in nx.edge_boundary(G, nodes2, nodes1) if G[e[0]][e[1]]['weight']>0)) / (len(nodes1) * len(nodes2)) # only positive edges
          pos_density_mat[nodes2plot.index(nodes1), nodes2plot.index(nodes2)] = (sum(1 for e in nx.edge_boundary(G, nodes1, nodes2)) + sum(1 for e in nx.edge_boundary(G, nodes2, nodes1) if G[e[0]][e[1]]['weight']>0)) / (len(nodes1) * len(nodes2)) # all edges
          if (sum(1 for _ in nx.edge_boundary(G, nodes1, nodes2)) + sum(1 for _ in nx.edge_boundary(G, nodes2, nodes1))):
            pos_frac_mat[nodes2plot.index(nodes1), nodes2plot.index(nodes2)] = (sum(1 for e in nx.edge_boundary(G, nodes1, nodes2) if G[e[0]][e[1]]['weight']>0) + sum(1 for e in nx.edge_boundary(G, nodes2, nodes1) if G[e[0]][e[1]]['weight']>0)) / (sum(1 for _ in nx.edge_boundary(G, nodes1, nodes2)) + sum(1 for _ in nx.edge_boundary(G, nodes2, nodes1)))
            neg_frac_mat[nodes2plot.index(nodes1), nodes2plot.index(nodes2)] = (sum(1 for e in nx.edge_boundary(G, nodes1, nodes2) if G[e[0]][e[1]]['weight']<0) + sum(1 for e in nx.edge_boundary(G, nodes2, nodes1) if G[e[0]][e[1]]['weight']<0)) / (sum(1 for _ in nx.edge_boundary(G, nodes1, nodes2)) + sum(1 for _ in nx.edge_boundary(G, nodes2, nodes1)))
          else:
            pos_frac_mat[nodes2plot.index(nodes1), nodes2plot.index(nodes2)] = -1 # for modules with no connections
            neg_frac_mat[nodes2plot.index(nodes1), nodes2plot.index(nodes2)] = -1 # for modules with no connections
        for region in regions:
          if comm_areas.count(region) >= 1:
            indx = np.where(np.array(comm_areas)==region)[0]
            for ind in indx:
              inds = np.array(list(large_comms[ind]))
              within_comm_dict[region].append(np.nanmean(signal_correlation[inds[:,None], inds][np.triu_indices(len(inds), 1)].tolist()))
              # within_pos_density[region].append(sum(1 for _, _, w in G.subgraph(large_comms[ind]).edges(data='weight') if w>0) / (len(large_comms[ind]) * (len(large_comms[ind])-1))) # only positive edges
              within_pos_density[region].append(G.subgraph(large_comms[ind]).number_of_edges() / (len(large_comms[ind]) * (len(large_comms[ind])-1))) # all edges
              # if sum(1 for _ in G.subgraph(large_comms[ind]).edges()):
              within_pos_frac[region].append(sum(1 for _, _, w in G.subgraph(large_comms[ind]).edges(data='weight') if w>0) / sum(1 for _ in G.subgraph(large_comms[ind]).edges()))
              within_neg_frac[region].append(sum(1 for _, _, w in G.subgraph(large_comms[ind]).edges(data='weight') if w<0) / sum(1 for _ in G.subgraph(large_comms[ind]).edges()))
              # else:
              #   within_neg_frac[region].append(-1)
          if comm_areas.count(region) > 1:
            indx = np.where(np.array(comm_areas)==region)[0]
            for ind1, ind2 in itertools.combinations(indx, 2):
              # for th in th_list:
              #   if (comm_purity[ind1] >= th) and (comm_purity[ind2] >= th):
                  inds1, inds2 = np.array(list(large_comms[ind1])), np.array(list(large_comms[ind2]))
                  cross_comm_dict[region].append(np.nanmean(signal_correlation[inds1[:,None], inds2].flatten().tolist()))
                  cross_pos_density[region].append(pos_density_mat[ind1, ind2])
                  cross_pos_frac[region].append(pos_frac_mat[ind1, ind2])
                  cross_neg_frac[region].append(neg_frac_mat[ind1, ind2]) # could be -1 for modules with no connections
        for nodei, nodej in itertools.combinations(node_idx, 2): # for all neurons
          if node_to_community[nodei] != node_to_community[nodej]:
            rand_cross_comm_list.append(signal_correlation[nodei, nodej])
        for commi, commj in itertools.combinations(large_comms, 2): # for all neurons
          rand_cross_pos_density_list.append((sum(1 for e in nx.edge_boundary(G, commi, commj) if G[e[0]][e[1]]['weight']>0) + sum(1 for e in nx.edge_boundary(G, commj, commi) if G[e[0]][e[1]]['weight']>0)) / (len(commi) * len(commj)))
          summ = (sum(1 for _ in nx.edge_boundary(G, commi, commj)) + sum(1 for _ in nx.edge_boundary(G, commj, commi)))
          if summ:
            rand_cross_pos_frac_list.append((sum(1 for e in nx.edge_boundary(G, commi, commj) if G[e[0]][e[1]]['weight']>0) + sum(1 for e in nx.edge_boundary(G, commj, commi) if G[e[0]][e[1]]['weight']>0)) / summ)
            rand_cross_neg_frac_list.append((sum(1 for e in nx.edge_boundary(G, commi, commj) if G[e[0]][e[1]]['weight']<0) + sum(1 for e in nx.edge_boundary(G, commj, commi) if G[e[0]][e[1]]['weight']<0)) / summ)
          else:
            rand_cross_pos_frac_list.append(-1)
            rand_cross_neg_frac_list.append(-1) # for modules with no connections
  df = pd.DataFrame()
  for region in regions:
    print(region)
    within_comm, cross_comm, within_posD, cross_posD, within_posF, cross_posF, within_negF, cross_negF =  within_comm_dict[region], cross_comm_dict[region], within_pos_density[region], cross_pos_density[region], within_pos_frac[region], cross_pos_frac[region], within_neg_frac[region], cross_neg_frac[region]
    within_comm, cross_comm = [e for e in within_comm if not np.isnan(e)], [e for e in cross_comm if not np.isnan(e)] # remove nan values
    cross_posF, cross_negF = [e for e in cross_posF if e != -1], [e for e in cross_negF if e != -1] # remove nan values
    df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(within_comm)[:,None], np.array(['signal correlation'] * len(within_comm))[:,None], np.array(['within community'] * len(within_comm))[:,None], np.array([region] * len(within_comm))[:,None]), 1), columns=['data', 'dtype', 'type', 'region'])], ignore_index=True)
    df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(cross_comm)[:,None], np.array(['signal correlation'] * len(cross_comm))[:,None], np.array(['cross community'] * len(cross_comm))[:,None], np.array([region] * len(cross_comm))[:,None]), 1), columns=['data', 'dtype', 'type', 'region'])], ignore_index=True)
    df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(within_posD)[:,None], np.array(['positive density'] * len(within_posD))[:,None], np.array(['within community'] * len(within_posD))[:,None], np.array([region] * len(within_posD))[:,None]), 1), columns=['data', 'dtype', 'type', 'region'])], ignore_index=True)
    df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(cross_posD)[:,None], np.array(['positive density'] * len(cross_posD))[:,None], np.array(['cross community'] * len(cross_posD))[:,None], np.array([region] * len(cross_posD))[:,None]), 1), columns=['data', 'dtype', 'type', 'region'])], ignore_index=True)
    df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(within_posF)[:,None], np.array(['positive fraction'] * len(within_posF))[:,None], np.array(['within community'] * len(within_posF))[:,None], np.array([region] * len(within_posF))[:,None]), 1), columns=['data', 'dtype', 'type', 'region'])], ignore_index=True)
    df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(cross_posF)[:,None], np.array(['positive fraction'] * len(cross_posF))[:,None], np.array(['cross community'] * len(cross_posF))[:,None], np.array([region] * len(cross_posF))[:,None]), 1), columns=['data', 'dtype', 'type', 'region'])], ignore_index=True)
    df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(within_negF)[:,None], np.array(['negative fraction'] * len(within_negF))[:,None], np.array(['within community'] * len(within_negF))[:,None], np.array([region] * len(within_negF))[:,None]), 1), columns=['data', 'dtype', 'type', 'region'])], ignore_index=True)
    df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(cross_negF)[:,None], np.array(['negative fraction'] * len(cross_negF))[:,None], np.array(['cross community'] * len(cross_negF))[:,None], np.array([region] * len(cross_negF))[:,None]), 1), columns=['data', 'dtype', 'type', 'region'])], ignore_index=True)
  df['data'] = pd.to_numeric(df['data'])
  return df, rand_cross_comm_list, rand_cross_pos_density_list, rand_cross_pos_frac_list, rand_cross_neg_frac_list

start_time = time.time()
# combined_stimulus_name = 'Natural\nscenes'
combined_stimulus_name = 'Natural\nmovies'
data_within_cross_comm_same_area_df, rand_cross_comm_list, rand_cross_pos_density_list, rand_cross_pos_frac_list, rand_cross_neg_frac_list = get_data_module_per_area(G_ccg_dict, best_comms_dict, visual_regions, combined_stimulus_name)
print("--- %s minutes" % ((time.time() - start_time)/60))
#%%
# Figure 3E,F
def plot_data_within_cross_comm_same_area_significance(df, data_list, regions, name, combined_stimulus_name):
  fig, ax = plt.subplots(1,1, sharex=True, sharey=True, figsize=(1.3*(len(combined_stimulus_names)-1), 4))
  # palette = [[plt.cm.tab20b(i) for i in range(4)][i] for i in [0, 3]]
  palette = ['w','k']
  data = df[df['dtype']==name]
  barplot = sns.barplot(x='region', y='data', hue="type", hue_order=['cross community','within community'], palette=palette, ec='k', linewidth=2., data=data, ax=ax, capsize=.05, width=0.6)
  if 'fraction' in name:
    ax.axhline(y=np.nanmean([e for e in data_list if e != -1]), linestyle='--', linewidth=3, color='.2')
  else:
    ax.axhline(y=np.nanmean(data_list), linestyle='--', linewidth=3, color='.2')
  ax.yaxis.set_tick_params(labelsize=30)
  plt.setp(ax.get_legend().get_title(), fontsize='0') # for legend title
  plt.xticks(range(len(regions)), region_labels, fontsize=30)
  ax.xaxis.set_tick_params(length=0)
  for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2.5)
    ax.spines[axis].set_color('k')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.tick_params(width=2.5)
  ax.set_xlabel('')
  ax.set_ylabel('')
  handles, labels = ax.get_legend_handles_labels()
  ax.legend(handles, [], fontsize=0)
  if name == 'signal correlation':
    ylabel = name.capitalize()
  elif name == 'positive density':
    ylabel = 'Connection probability'
  elif name == 'positive fraction':
    ylabel = 'Excitatory fraction'
  elif name == 'negative fraction':
    ylabel = 'Inhibitory fraction'
  ax.set_ylabel(ylabel, fontsize=30)
  # add significance annotation
  alpha_list = [.0001, .001, .01, .05]
  maxx = 0
  for r_ind, region in enumerate(regions):
    within_comm, cross_comm = data[(data.region==region)&(data.type=='within community')]['data'].values.flatten(), data[(data.region==region)&(data.type=='cross community')]['data'].values.flatten()
    within_sr, cross_sr = confidence_interval(within_comm)[1], confidence_interval(cross_comm)[1]
    maxx = max(within_sr, cross_sr) if max(within_sr, cross_sr) > maxx else maxx
  h, l = .05 * maxx, .05 * maxx
  for r_ind, region in enumerate(regions):
    within_comm, cross_comm = data[(data.region==region)&(data.type=='within community')]['data'].values.flatten(), data[(data.region==region)&(data.type=='cross community')]['data'].values.flatten()
    _, p = ztest(within_comm, cross_comm, value=0)
    diff_star = '*' * (len(alpha_list) - bisect(alpha_list, p)) if len(alpha_list) > bisect(alpha_list, p) else 'ns'
    within_sr, cross_sr = confidence_interval(within_comm)[1], confidence_interval(cross_comm)[1]
    within_sr = within_sr + h
    cross_sr = cross_sr + h
    annot_difference(diff_star, -.15 + r_ind, .15 + r_ind, max(within_sr, cross_sr), l, 2.5, 28, ax=ax)
  ax.invert_xaxis()
  plt.tight_layout()
  # plt.show()
  plt.savefig('./plots/{}_within_cross_comm_same_area_{}.pdf'.format(name.replace(' ', '_'), combined_stimulus_name.replace('\n', '_')), transparent=True)

plot_data_within_cross_comm_same_area_significance(data_within_cross_comm_same_area_df, rand_cross_comm_list, visual_regions, name='signal correlation', combined_stimulus_name=combined_stimulus_name)
plot_data_within_cross_comm_same_area_significance(data_within_cross_comm_same_area_df, rand_cross_pos_density_list, visual_regions, name='positive density', combined_stimulus_name=combined_stimulus_name)
plot_data_within_cross_comm_same_area_significance(data_within_cross_comm_same_area_df, rand_cross_pos_frac_list, visual_regions, name='positive fraction', combined_stimulus_name=combined_stimulus_name)
plot_data_within_cross_comm_same_area_significance(data_within_cross_comm_same_area_df, rand_cross_neg_frac_list, visual_regions, name='negative fraction', combined_stimulus_name=combined_stimulus_name)
#%%
################################ directionality score between communities
def plot_directionality_score_comm(G_dict, row_ind, comms_dict, max_neg_reso, etype='pos'):
  rows, cols = get_rowcol(G_dict)
  row = rows[row_ind]
  fig, axes = plt.subplots(2, 4, figsize=(6*4, 5*2))
  for i, j in [(f,s) for f in range(axes.shape[0]) for s in range(axes.shape[1])]:
    col_ind = i * axes.shape[1] + j
    col = cols[col_ind]
    ax = axes[i, j]
  # fig, axes = plt.subplots(1, 8, figsize=(6*8, 5))
  # for col_ind, col in enumerate(cols):
  #   ax = axes[col_ind]
    ax.set_title(col, fontsize=30)
    G = G_dict[row][col]
    nx.set_node_attributes(G, active_area_dict[row], "area")
    max_reso = max_neg_reso[row_ind][col_ind]
    comms_list = comms_dict[row][col][max_reso]
    comms = comms_list[0]
    node_to_community = comm2partition([comm for comm in comms if len(comm)>=6])
    between_community_edges = _find_between_community_edges(G.edges(), node_to_community)
    comms2plot = get_unique_elements(between_community_edges.keys())
    # print(comms2plot)
    nodes2plot = [[node for node in node_to_community if node_to_community[node] == comm] for comm in comms2plot]
    comm_mat = np.zeros((len(comms2plot), len(comms2plot)))
    for nodes1, nodes2 in itertools.permutations(nodes2plot, 2):
      # print(sum(1 for _ in nx.edge_boundary(G, nodes1, nodes2)))
      # if sum(1 for _ in nx.edge_boundary(G, nodes1, nodes2)) > max_e:
      #   edge_set = nx.edge_boundary(G, nodes1, nodes2)
      #   max_e = sum(1 for _ in nx.edge_boundary(G, nodes1, nodes2))
      if etype == 'pos':
        comm_mat[nodes2plot.index(nodes1), nodes2plot.index(nodes2)] = sum(1 for e in nx.edge_boundary(G, nodes1, nodes2) if G[e[0]][e[1]]['weight']>0)
      elif etype == 'neg':
        comm_mat[nodes2plot.index(nodes1), nodes2plot.index(nodes2)] = sum(1 for e in nx.edge_boundary(G, nodes1, nodes2) if G[e[0]][e[1]]['weight']<0)
      else:
        comm_mat[nodes2plot.index(nodes1), nodes2plot.index(nodes2)] = sum(1 for _ in nx.edge_boundary(G, nodes1, nodes2))
    comm_ds = (comm_mat - comm_mat.T) / (comm_mat + comm_mat.T)
    comm_ds[(comm_mat + comm_mat.T)==0] = 0
    ticklabels = comms2plot
    sns.heatmap(comm_ds.T, cmap='seismic', center=0, ax=ax, xticklabels=ticklabels, yticklabels=ticklabels)
    ax.xaxis.set_tick_params(labelsize=25)
    ax.yaxis.set_tick_params(labelsize=25)
    ax.invert_yaxis()
  plt.savefig('./plots/ds_comm_{}.pdf'.format(etype), transparent=True)

row_ind = 7
# col_ind = 6
start_time = time.time()
plot_directionality_score_comm(G_ccg_dict, row_ind, comms_dict, max_reso_config, 'pos')
plot_directionality_score_comm(G_ccg_dict, row_ind, comms_dict, max_reso_config, 'neg')
plot_directionality_score_comm(G_ccg_dict, row_ind, comms_dict, max_reso_config, 'all')
print("--- %s minutes" % ((time.time() - start_time)/60))
#%%
################################ directionality score between areas
def plot_directionality_score_area(G_dict, active_area_dict, regions, etype='pos'):
  rows, cols = get_rowcol(G_dict)
  fig, axes = plt.subplots(2, 3, figsize=(6*3, 5*2))
  for i, j in [(f,s) for f in range(axes.shape[0]) for s in range(axes.shape[1])]:
    cs_ind = i * axes.shape[1] + j
    ax = axes[i, j]
    ax.set_title(combined_stimulus_names[cs_ind].replace('\n', ' '), fontsize=30)
    area_ds = np.zeros((len(regions), len(regions)))
    for col in combined_stimuli[cs_ind]:
      for row in session2keep:
        ind_area_mat = np.zeros((len(regions), len(regions)))
        active_area = active_area_dict[row]
        G = G_dict[row][col]
        nx.set_node_attributes(G, active_area, "area")
        nodes2plot = [[node for node in active_area if active_area[node] == area] for area in regions]
        for nodes1, nodes2 in itertools.permutations(nodes2plot, 2):
          # print(sum(1 for _ in nx.edge_boundary(G, nodes1, nodes2)))
          # if sum(1 for _ in nx.edge_boundary(G, nodes1, nodes2)) > max_e:
          #   edge_set = nx.edge_boundary(G, nodes1, nodes2)
          #   max_e = sum(1 for _ in nx.edge_boundary(G, nodes1, nodes2))
          if etype == 'pos':
            num = sum(1 for e in nx.edge_boundary(G, nodes1, nodes2) if G[e[0]][e[1]]['weight']>0)
          elif etype == 'neg':
            num = sum(1 for e in nx.edge_boundary(G, nodes1, nodes2) if G[e[0]][e[1]]['weight']<0)
          else:
            num = sum(1 for _ in nx.edge_boundary(G, nodes1, nodes2))
          ind_area_mat[nodes2plot.index(nodes1), nodes2plot.index(nodes2)] += num
        ind_area_ds = (ind_area_mat - ind_area_mat.T) / (ind_area_mat + ind_area_mat.T)
        ind_area_ds[(ind_area_mat + ind_area_mat.T)==0] = 0
        area_ds += ind_area_ds
    ticklabels = region_labels
    sns.heatmap((area_ds / (len(session2keep) * len(combined_stimuli[cs_ind]))).T, cmap='seismic', center=0, ax=ax, xticklabels=ticklabels, yticklabels=ticklabels)
    ax.xaxis.set_tick_params(labelsize=25)
    ax.yaxis.set_tick_params(labelsize=25)
    ax.invert_xaxis()
  plt.savefig('./plots/ds_area_{}.pdf'.format(etype))

start_time = time.time()
plot_directionality_score_area(G_ccg_dict, active_area_dict, visual_regions, 'pos')
plot_directionality_score_area(G_ccg_dict, active_area_dict, visual_regions, 'neg')
plot_directionality_score_area(G_ccg_dict, active_area_dict, visual_regions, 'all')
print("--- %s minutes" % ((time.time() - start_time)/60))
#%%
################################ pos DS VS neg DS for low to high area pairs
def plot_posDS_negDS(G_dict, active_area_dict, regions):
  fig, axes = plt.subplots(2, 3, figsize=(6*3, 5*2))
  for cs_ind, combined_stimulus in enumerate(combined_stimuli):
    i, j = cs_ind // axes.shape[1], cs_ind % axes.shape[1]
    ax = axes[i, j]
    ax.set_title(combined_stimulus_names[cs_ind].replace('\n', ' '), fontsize=30)
  
    label = combined_stimulus_names[cs_ind].replace('\n', ' ')
    pos_area_ds, neg_area_ds = np.zeros((len(regions), len(regions))), np.zeros((len(regions), len(regions)))
    X, Y = [], []
    all_pos_ds, all_neg_ds = [], []
    for col in combined_stimulus:
      for row in session2keep:
        pos_ind_area_mat, neg_ind_area_mat = np.zeros((len(regions), len(regions))), np.zeros((len(regions), len(regions)))
        active_area = active_area_dict[row]
        G = G_dict[row][col]
        nx.set_node_attributes(G, active_area, "area")
        nodes2plot = [[node for node in active_area if active_area[node] == area] for area in regions]
        for nodes1, nodes2 in itertools.permutations(nodes2plot, 2):
          pos_num = sum(1 for e in nx.edge_boundary(G, nodes1, nodes2) if G[e[0]][e[1]]['weight']>0)
          neg_num = sum(1 for e in nx.edge_boundary(G, nodes1, nodes2) if G[e[0]][e[1]]['weight']<0)
          pos_ind_area_mat[nodes2plot.index(nodes1), nodes2plot.index(nodes2)] += pos_num
          neg_ind_area_mat[nodes2plot.index(nodes1), nodes2plot.index(nodes2)] += neg_num
        pos_ind_area_ds = (pos_ind_area_mat - pos_ind_area_mat.T) / (pos_ind_area_mat + pos_ind_area_mat.T)
        pos_ind_area_ds[(pos_ind_area_mat + pos_ind_area_mat.T)==0] = 0
        neg_ind_area_ds = (neg_ind_area_mat - neg_ind_area_mat.T) / (neg_ind_area_mat + neg_ind_area_mat.T)
        neg_ind_area_ds[(neg_ind_area_mat + neg_ind_area_mat.T)==0] = 0
        
        pos_ind_area_ds[(pos_ind_area_mat + pos_ind_area_mat.T)<2] = np.nan
        neg_ind_area_ds[(neg_ind_area_mat + neg_ind_area_mat.T)<1] = np.nan
        pos_data = pos_ind_area_ds[np.tril_indices(len(regions), -1)]
        neg_data = neg_ind_area_ds[np.tril_indices(len(regions), -1)]
        all_pos_ds.append(pos_data)
        all_neg_ds.append(neg_data)
    # print(np.vstack(all_neg_ds))
    X, Y = np.nanmean(np.vstack(all_pos_ds), 0), np.nanmean(np.vstack(all_neg_ds), 0)
    inds = (~np.isnan(X) & ~np.isnan(Y))
    X, Y = X[inds], Y[inds]
    #     pos_area_ds += pos_ind_area_ds
    #     neg_area_ds += neg_ind_area_ds
    # X = pos_area_ds[np.triu_indices(len(regions), 1)] / (len(session2keep) * len(combined_stimuli[cs_ind]))
    # Y = neg_area_ds[np.triu_indices(len(regions), 1)] / (len(session2keep) * len(combined_stimuli[cs_ind]))
    ax.scatter(X, Y, label=label, color=combined_stimulus_colors[cs_ind], alpha=.6)
    ax.axvline(x=0, linewidth=3, color='.2', linestyle='--', alpha=.4)
    ax.axhline(y=0, linewidth=3, color='.2', linestyle='--', alpha=.4)
    X, Y = (list(t) for t in zip(*sorted(zip(X, Y))))
    X, Y = np.array(X), np.array(Y)
    slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
    line = slope*X+intercept
    text = 'r={:.2f}, p={:.1e}'.format(r_value, p_value)
    ax.plot(X, line, color=combined_stimulus_colors[cs_ind], linestyle='-', linewidth=4, alpha=.8, label=label) # (5,(10,3))
    locx, locy = .7, .5
    ax.text(locx, locy, text, horizontalalignment='center',
        verticalalignment='center', transform=ax.transAxes, fontsize=20)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    if i == 1:
      ax.set_xlabel('Excitatory directionality score', fontsize=20,color='k') #, weight='bold'
    if j == 0:
      ax.set_ylabel('Inhibitory directionality score', fontsize=20,color='k') #, weight='bold'

  plt.savefig('./plots/posDS_negDS.pdf')
  # plt.legend()
  # plt.show()

start_time = time.time()
plot_posDS_negDS(G_ccg_dict, active_area_dict, visual_regions)
print("--- %s minutes" % ((time.time() - start_time)/60))
#%%
regions = visual_regions
G_dict = G_ccg_dict
combined_stimulus = ['natural_scenes']
X, Y = [], []
all_pos_ds, all_neg_ds = [], []
for col in combined_stimulus:
  for row in session2keep:
    pos_ind_area_mat, neg_ind_area_mat = np.zeros((len(regions), len(regions))), np.zeros((len(regions), len(regions)))
    active_area = active_area_dict[row]
    G = G_dict[row][col]
    nx.set_node_attributes(G, active_area, "area")
    nodes2plot = [[node for node in active_area if active_area[node] == area] for area in regions]
    for nodes1, nodes2 in itertools.permutations(nodes2plot, 2):
      pos_num = sum(1 for e in nx.edge_boundary(G, nodes1, nodes2) if G[e[0]][e[1]]['weight']>0)
      neg_num = sum(1 for e in nx.edge_boundary(G, nodes1, nodes2) if G[e[0]][e[1]]['weight']<0)
      pos_ind_area_mat[nodes2plot.index(nodes1), nodes2plot.index(nodes2)] += pos_num
      neg_ind_area_mat[nodes2plot.index(nodes1), nodes2plot.index(nodes2)] += neg_num
    pos_ind_area_ds = (pos_ind_area_mat - pos_ind_area_mat.T) / (pos_ind_area_mat + pos_ind_area_mat.T)
    pos_ind_area_ds[(pos_ind_area_mat + pos_ind_area_mat.T)==0] = 0
    neg_ind_area_ds = (neg_ind_area_mat - neg_ind_area_mat.T) / (neg_ind_area_mat + neg_ind_area_mat.T)
    neg_ind_area_ds[(neg_ind_area_mat + neg_ind_area_mat.T)==0] = 0
    
    pos_ind_area_ds[(pos_ind_area_mat + pos_ind_area_mat.T)<2] = np.nan
    neg_ind_area_ds[(neg_ind_area_mat + neg_ind_area_mat.T)<2] = np.nan
    pos_data = pos_ind_area_ds[np.tril_indices(len(regions), -1)]
    neg_data = neg_ind_area_ds[np.tril_indices(len(regions), -1)]
    all_pos_ds.append(pos_data)
    all_neg_ds.append(neg_data)
#%%
################################ directionality score between areas sum instead of average
def plot_directionality_score_area_sum(G_dict, active_area_dict, regions, etype='pos'):
  rows, cols = get_rowcol(G_dict)
  fig, axes = plt.subplots(2, 3, figsize=(6*3, 5*2))
  for i, j in [(f,s) for f in range(axes.shape[0]) for s in range(axes.shape[1])]:
    cs_ind = i * axes.shape[1] + j
    ax = axes[i, j]
    ax.set_title(combined_stimulus_names[cs_ind].replace('\n', ' '), fontsize=30)
    area_mat = np.zeros((len(regions), len(regions)))
    for col in combined_stimuli[cs_ind]:
      for row in session2keep:
        active_area = active_area_dict[row]
        G = G_dict[row][col]
        nx.set_node_attributes(G, active_area, "area")
        nodes2plot = [[node for node in active_area if active_area[node] == area] for area in regions]
        for nodes1, nodes2 in itertools.permutations(nodes2plot, 2):
          if etype == 'pos':
            num = sum(1 for e in nx.edge_boundary(G, nodes1, nodes2) if G[e[0]][e[1]]['weight']>0)
          elif etype == 'neg':
            num = sum(1 for e in nx.edge_boundary(G, nodes1, nodes2) if G[e[0]][e[1]]['weight']<0)
          else:
            num = sum(1 for _ in nx.edge_boundary(G, nodes1, nodes2))
          area_mat[nodes2plot.index(nodes1), nodes2plot.index(nodes2)] += num
    area_ds = (area_mat - area_mat.T) / (area_mat + area_mat.T)
    area_ds[(area_mat + area_mat.T)==0] = 0
    ticklabels = region_labels
    sns.heatmap((area_ds / (len(session2keep) * len(combined_stimuli[cs_ind]))).T, cmap='seismic', center=0, ax=ax, xticklabels=ticklabels, yticklabels=ticklabels)
    ax.xaxis.set_tick_params(labelsize=25)
    ax.yaxis.set_tick_params(labelsize=25)
    ax.invert_xaxis()
  plt.savefig('./plots/ds_area_sum_{}.pdf'.format(etype))

start_time = time.time()
plot_directionality_score_area_sum(G_ccg_dict, active_area_dict, visual_regions, 'pos')
plot_directionality_score_area_sum(G_ccg_dict, active_area_dict, visual_regions, 'neg')
plot_directionality_score_area_sum(G_ccg_dict, active_area_dict, visual_regions, 'all')
print("--- %s minutes" % ((time.time() - start_time)/60))
# %%
# Figure S4B
def get_purity_coverage_comm_size(G_dict, comms_dict, area_dict, regions):
  rows, cols = get_rowcol(comms_dict)
  df = pd.DataFrame()
  for col_ind, col in enumerate(cols):
    print(col)
    size_col, purity_col, coverage_col = [], [], []
    for row_ind, row in enumerate(session2keep):
      comms_list = comms_dict[row][col]
      G = G_dict[row][col]
      all_regions = [area_dict[row][node] for node in G.nodes()]
      region_counts = np.array([all_regions.count(r) for r in regions])
      lr_size = region_counts.max()
      for comms in comms_list: # 100 repeats
        sizes = [len(comm) for comm in comms]
        data = []
        for comm, size in zip(comms, sizes):
          c_regions = [area_dict[row][node] for node in comm]
          # _, counts = np.unique(c_regions, return_counts=True)
          counts = np.array([c_regions.count(r) for r in regions])
          assert len(c_regions) == size == counts.sum()
          purity = counts.max() / size
          coverage = (counts / region_counts).max()
          data.append((size, purity, coverage))
        size_col += [s/lr_size for s,p,c in data if s>=4]
        purity_col += [p for s,p,c in data if s>=4]
        coverage_col += [c for s,p,c in data if s>=4]
    df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(size_col)[:,None], np.array(purity_col)[:,None], np.array(coverage_col)[:,None], np.array([stimulus2stype(col)[1]] * len(size_col))[:,None]), 1), columns=['community size', 'purity', 'coverage', 'stimulus type'])], ignore_index=True)
  df['community size'] = pd.to_numeric(df['community size'])
  df['purity'] = pd.to_numeric(df['purity'])
  df['coverage'] = pd.to_numeric(df['coverage'])
  return df

df = get_purity_coverage_comm_size(G_ccg_dict, best_comms_dict, area_dict, visual_regions)
# plt.savefig(image_name)
#%%
# Figure S4B
####################### jointplot of purity VS community size
def jointplot_purity_coverage_comm_size(df, name='purity'):
  g = sns.JointGrid()
  ax = g.ax_joint
  palette = {s:c for s, c in zip(stimulus_types, stimulus_type_color)}
  for st_ind, stimulus_type in enumerate(stimulus_types[::-1]):
    data = df[df['stimulus type']==stimulus_type]
    x, y = data['community size'], data[name]
    ax.scatter(x, y, facecolors=stimulus_type_color[len(stimulus_type_color)-1-st_ind], linewidth=.5, s=70, edgecolors='white', alpha=.6)
  ax.axvline(x=1, linewidth=3, color='.2', linestyle='--', alpha=.4)
  sns.kdeplot(data=df, x='community size', hue='stimulus type', palette=stimulus_type_color, fill=False, linewidth=2, ax=g.ax_marg_x, legend=False, common_norm=False)
  sns.kdeplot(data=df, y=name, hue='stimulus type', palette=stimulus_type_color, linewidth=2, ax=g.ax_marg_y, legend=False, common_norm=False)
  ax.set_xscale('log')
  print(ax.get_xlim())
  # ax.set_yscale('log')
  ax.set_xlabel('') #, weight='bold'
  ax.set_ylabel('') #, weight='bold'
  ax.xaxis.set_tick_params(labelsize=20)
  ax.yaxis.set_tick_params(labelsize=20)
  # change dots to empty circles for seaborn scatter
  kws = {"s": 90, "linewidth": 1.5}
  handles, labels = zip(*[
      (plt.scatter([], [], fc=color, ec='white', **kws), key) for key, color in palette.items()
  ]) #, ec=color
  # ax.legend(handles, labels, title="cat")
  # handles, labels = ax.get_legend_handles_labels()
  # if name == 'purity':
  #   ax.legend(handles, labels, title='', bbox_to_anchor=(.6, 1.), handletextpad=-0.1, loc='upper left', fontsize=15, frameon=False)
  for ax in [g.ax_joint, g.ax_marg_x, g.ax_marg_y]:
    for axis in ['bottom', 'left', 'top', 'right']:
      ax.spines[axis].set_linewidth(2.)
      ax.spines[axis].set_color('k')
    ax.tick_params(width=2.)
  plt.tight_layout()
  # plt.show()
  plt.savefig('./plots/joint_{}_comm_size.pdf'.format(name), transparent=True)

jointplot_purity_coverage_comm_size(df, name='purity')
jointplot_purity_coverage_comm_size(df, name='coverage')
#%%
# Figure S4B
def save_stimulus_type_legend():
  fig, ax = plt.subplots(1,1, figsize=(.4*(len(combined_stimulus_names)-1), .5))
  for st_ind, st in enumerate(stimulus_types):
    ax.scatter([0], [1], facecolors=stimulus_type_color[st_ind], linewidth=.5, s=200, edgecolors='white', alpha=.6, label=st)
  handles, labels = ax.get_legend_handles_labels()
  # legend = ax.legend(handles, labels, title='', fontsize=15, frameon=False)
  
  legend = ax.legend(handles, labels, title='', bbox_to_anchor=(0.1, -1.), handletextpad=0.3, loc='upper left', fontsize=25, frameon=False) # change legend order to within/cross similar module then cross random
  plt.axis('off')
  export_legend(legend, './plots/stimulus_type_legend.pdf')
  plt.tight_layout()
  # plt.savefig('./plots/stimuli_markers.pdf', transparent=True)
  plt.show()

save_stimulus_type_legend()
#%%
# Figure S4B
def get_mean_purity_coverage_Hcommsize_col(G_dict, comms_dict, area_dict, regions):
  rows, cols = get_rowcol(comms_dict)
  purity_dict, coverage_dict = {}, {}
  for col_ind, col in enumerate(cols):
    print(col)
    purity_data, coverage_data = [], []
    for row_ind, row in enumerate(session2keep):
      G = G_dict[row][col]
      all_regions = [area_dict[row][node] for node in G.nodes()]
      region_counts = np.array([all_regions.count(r) for r in regions])
      lr_size = region_counts.max()
      comms_list = comms_dict[row][col]
      for comms in comms_list: # 100 repeats
        sizes = [len(comm) for comm in comms]
        for comm, size in zip(comms, sizes):
          if size >= 4:
            c_regions = [area_dict[row][node] for node in comm]
            # _, counts = np.unique(c_regions, return_counts=True)
            counts = np.array([c_regions.count(r) for r in regions])
            assert len(c_regions) == size == counts.sum()
            purity = counts.max() / size
            coverage = (counts / region_counts).max()
            purity_data.append((size / lr_size, purity))
            coverage_data.append((size / lr_size, coverage))
    purity_dict[stimulus2stype(col)[1]] = purity_data
    coverage_dict[stimulus2stype(col)[1]] = coverage_data
  return purity_dict, coverage_dict

purity_dict, coverage_dict = get_mean_purity_coverage_Hcommsize_col(G_ccg_dict, best_comms_dict, area_dict, visual_regions)
#%%
# Figure S4B
def plot_scatter_mean_purity_coverage_Hcommsize_col(purity_dict, coverage_dict, name='purity'):
  fig, ax = plt.subplots(1, 1, figsize=(5, 2.5))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  if name == 'purity':
    data = purity_dict
  else:
    data = coverage_dict
  for col_ind, col in enumerate(data):
    if max([i for i, j in data[col]]) <= 1:
      x, y = np.array([s for s, d in data[col]]), np.array([d for s, d in data[col]])
      ox, oy = None, None
      numbin = 20
    else:
      x, y = np.array([s for s, d in data[col] if s <= 1]), np.array([d for s, d in data[col] if s <= 1])
      ox, oy = np.array([s for s, d in data[col] if s > 1]), np.array([d for s, d in data[col] if s > 1])
      numbin = 16
    binned_x, binned_y = double_binning(x, y, numbin=numbin, log=True)
    xy = np.array([binned_x, binned_y])
    xy = xy[:, ~np.isnan(xy).any(axis=0)]
    binned_x, binned_y = xy[0], xy[1]
    # x = [(a + b) / 2 for a, b in zip(bins[:-1], bins[1:])]
    # print(binned_x, binned_y)
    ax.scatter(binned_x, binned_y, color=stimulus_type_color[col_ind], label=col, marker='^', s=100, alpha=1)
    if (ox is not None) and (oy is not None):
      binned_ox, binned_oy = double_binning(ox, oy, numbin=5, log=True)
      oxy = np.array([binned_ox, binned_oy])
      oxy = oxy[:, ~np.isnan(oxy).any(axis=0)]
      binned_ox, binned_oy = oxy[0], oxy[1]
      ax.scatter(binned_ox, binned_oy, color=stimulus_type_color[col_ind], label=col, marker='^', s=100, alpha=.2)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(binned_x),binned_y)
    line = slope*np.log10(binned_x)+intercept
    locx, locy = .8, .3
    text = 'r={:.2f}, p={:.1e}'.format(r_value, p_value)
    ax.plot(binned_x, line, color=stimulus_type_color[col_ind], linestyle='-', linewidth=4, alpha=.8) # (5,(10,3))
    # ax.text(locx, locy, text, horizontalalignment='center',
    #   verticalalignment='center', transform=ax.transAxes, fontsize=18)
    
    # coef = np.polyfit(np.log10(binned_x), binned_y, 1)
    # poly1d_fn = np.poly1d(coef) 
    # ax.plot(binned_x, poly1d_fn(binned_x), '-', color=stimulus_type_color[col_ind], alpha=.8, linewidth=3) #'--k'=black dashed line
  ax.axvline(x=1, linewidth=3, color='.2', linestyle='--', alpha=.4)
  ax.set_xlim(0.0603483134041287, 5.478847527784104)
  plt.xscale('log')
  plt.xlabel('')
  plt.ylabel('')
  plt.xticks(fontsize=20)
  plt.yticks(fontsize=20)
  for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2.)
    ax.spines[axis].set_color('k')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.tick_params(width=2.)
  # plt.suptitle('{} average purity VS community size'.format(max_method), size=40)
  plt.tight_layout()
  image_name = './plots/mean_{}_comm_size_col.pdf'.format(name)
  plt.savefig(image_name, transparent=True)
  # plt.show()

plot_scatter_mean_purity_coverage_Hcommsize_col(purity_dict, coverage_dict, name='purity')
plot_scatter_mean_purity_coverage_Hcommsize_col(purity_dict, coverage_dict, name='coverage')
#%%
# Figure 2C
def get_purity_coverage_ri(G_dict, area_dict, regions, best_comms_dict):
  rows, cols = get_rowcol(G_dict)
  df = pd.DataFrame()
  for col_ind, col in enumerate(cols):
    print(col)
    w_purity_col, w_coverage_col, ri_list = [], [], []
    for row_ind, row in enumerate(session2keep):
      data = {}
      G = G_dict[row][col].copy()
      node_area = area_dict[row]
      all_regions = [node_area[node] for node in sorted(G.nodes())]
      region_counts = np.array([all_regions.count(r) for r in regions])
      for run in range(len(best_comms_dict[row][col])):
        comms_list = best_comms_dict[row][col]
        comms = comms_list[run]
        large_comms = [comm for comm in comms if len(comm)>=4]
        for comm in large_comms:
          c_regions = [active_area_dict[row][node] for node in comm]
          counts = np.array([c_regions.count(r) for r in regions])
          size = counts.sum()
          purity = counts.max() / size
          coverage = (counts / region_counts).max()
          if size in data:
            data[size][0].append(purity)
            data[size][1].append(coverage)
          else:
            data[size] = [[purity], [coverage]]
        c_size, c_purity, c_coverage = [k for k,v in data.items()], [v[0] for k,v in data.items()], [v[1] for k,v in data.items()]
        all_size = np.repeat(c_size, [len(p) for p in c_purity])
        c_size = all_size / sum(all_size)
        w_purity_col.append(sum([cs * cp for cs, cp in zip(c_size, [p for ps in c_purity for p in ps])]))
        w_coverage_col.append(sum([cs * cc for cs, cc in zip(c_size, [c for css in c_coverage for c in css])]))
        # w_purity_col.append(sum([cs * np.sum(cp) for cs, cp in zip(c_size, c_purity)]))
        # w_coverage_col.append(sum([cs * np.sum(cp) for cs, cp in zip(c_size, c_coverage)]))

        assert len(all_regions) == len(comm2label(comms)), '{}, {}'.format(len(all_regions), len(comm2label(comms)))
        # print(len(regions), len(comm2label(comms)))
        ri_list.append(adjusted_rand_score(all_regions, comm2label(comms)))
    df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(w_purity_col)[:,None], np.array(['weighted purity'] * len(w_purity_col))[:,None], np.array([combine_stimulus(col)[1]] * len(w_purity_col))[:,None]), 1), columns=['data', 'type', 'combined stimulus'])], ignore_index=True)
    df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(w_coverage_col)[:,None], np.array(['weighted coverage'] * len(w_coverage_col))[:,None], np.array([combine_stimulus(col)[1]] * len(w_coverage_col))[:,None]), 1), columns=['data', 'type', 'combined stimulus'])], ignore_index=True)
    df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(ri_list)[:,None], np.array(['rand index'] * len(ri_list))[:,None], np.array([combine_stimulus(col)[1]] * len(ri_list))[:,None]), 1), columns=['data', 'type', 'combined stimulus'])], ignore_index=True)
  df['data'] = pd.to_numeric(df['data'])
  return df

purity_coverage_ri_df = get_purity_coverage_ri(G_ccg_dict, area_dict, visual_regions, best_comms_dict)
#%%
# Figure 2C
################# plot weighted purity and rand index across stimuli with different markers
def plot_weighted_coverage_purity_rand_index_markers(df, dname):
  fig, ax = plt.subplots(1, 1, figsize=(3, 2.5))
  data = df[df['type']==dname].groupby('combined stimulus')
  x = np.arange(len(combined_stimulus_names))
  y = data.mean(numeric_only=True).loc[combined_stimulus_names].values.flatten()
  err = data.std(numeric_only=True).loc[combined_stimulus_names].values.flatten()
  # y = data.mean(numeric_only=True).loc[combined_stimulus_names].values.flatten()
  # err = data.std(numeric_only=True).loc[combined_stimulus_names].values.flatten()
  # err = 1.96 * data.std(numeric_only=True).loc[combined_stimulus_names].values.flatten() / data.size().loc[combined_stimulus_names].pow(1./2).values.flatten() # 95% confidence interval
  # err = stats.t.ppf((1 + 0.95) / 2., data.size().loc[combined_stimulus_names]-1) * data.sem().loc[combined_stimulus_names].values.flatten()
  
  for ind, (xi, yi, erri) in enumerate(zip(x, y, err)):
    ax.errorbar(xi, yi, yerr=erri, fmt=' ', linewidth=2.,color='.1', zorder=1)
    ax.scatter(xi, yi, marker=stimulus2marker[combined_stimulus_names[ind]], s=15*error_size_dict[stimulus2marker[combined_stimulus_names[ind]]], linewidth=1.,color='k', facecolor='white', zorder=2)
  ax.set(xlabel=None)
  ax.xaxis.set_tick_params(length=0)
  ax.set_xlim(-.8, len(combined_stimulus_names)-.2)
  # ax.invert_yaxis()
  ax.set_xticks([])
  ax.yaxis.set_tick_params(labelsize=20)
  ax.set_xlabel('')
  for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(1.5)
    ax.spines[axis].set_color('k')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.tick_params(width=1.5)
  plt.tight_layout()
  plt.savefig('./plots/{}_markers.pdf'.format(dname.replace(' ', '_')), transparent=True)
  # plt.show()

plot_weighted_coverage_purity_rand_index_markers(purity_coverage_ri_df, 'weighted coverage')
plot_weighted_coverage_purity_rand_index_markers(purity_coverage_ri_df, 'weighted purity')
plot_weighted_coverage_purity_rand_index_markers(purity_coverage_ri_df, 'rand index')
#%%
# Figure S5
# Z score between signal correlation distributions of pos, neg and disconnected neuron pairs
# correlation coefficient between w_purity and Z score is neg_dis=0.72 > pos_dis=0.52 > pos_neg=0.29, unsigned connected_dis = 0.53
def get_signalcorr_wpurity(G_dict, area_dict, regions, best_comms_dict, signal_correlation_dict, pairtype):
  rows, cols = get_rowcol(G_dict)
  df = pd.DataFrame()
  z_sc = np.zeros((len(session2keep), len(combined_stimulus_names)-1))
  for row_ind, row in enumerate(session2keep):
    for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names[1:]):
      sc = signal_correlation_dict[row][combined_stimulus_name]
      pos_edges, neg_edges, dis_pairs = [], [], []
      for col in combined_stimuli[cs_ind+1]:
        G = G_dict[row][col].copy()
        pos_edges += [(i,j) for i,j,w in G.edges(data='weight') if w > 0]
        neg_edges += [(i,j) for i,j,w in G.edges(data='weight') if w < 0]
        dis_pairs += [(i,j) for i,j in list(itertools.combinations(G.nodes(), 2)) if (not G.has_edge(i,j)) and (not G.has_edge(j,i))]
      pos_sigcorr = [sc[i, j] for i, j in pos_edges if not np.isnan(sc[i, j])]
      neg_sigcorr = [sc[i, j] for i, j in neg_edges if not np.isnan(sc[i, j])]
      dis_sigcorr = [sc[i, j] for i, j in dis_pairs if not np.isnan(sc[i, j])]
      if pairtype == 'pos_neg':
        z, p = ztest(pos_sigcorr, neg_sigcorr, value=0) # 0.29
      elif pairtype == 'pos_dis':
        z, p = ztest(pos_sigcorr, dis_sigcorr, value=0) # 0.52
      elif pairtype == 'neg_dis':
        z, p = ztest(neg_sigcorr, dis_sigcorr, value=0) # 0.72
      z_sc[row_ind, cs_ind] = z
      # m_sc[row_ind, cs_ind] = np.nanmean([sc[i, j] for i, j in set(edges)])
  for row_ind, row in enumerate(session2keep):
    print(row)
    for col_ind, col in enumerate(cols):
      w_purity_col, signal_corr = [], []
      data = {}
      G = G_dict[row][col].copy()
      node_area = area_dict[row]
      all_regions = [node_area[node] for node in sorted(G.nodes())]
      region_counts = np.array([all_regions.count(r) for r in regions])
      for run in range(len(best_comms_dict[row][col])):
        comms_list = best_comms_dict[row][col]
        comms = comms_list[run]
        large_comms = [comm for comm in comms if len(comm)>=4]
        for comm in large_comms:
          c_regions = [active_area_dict[row][node] for node in comm]
          counts = np.array([c_regions.count(r) for r in regions])
          size = counts.sum()
          purity = counts.max() / size
          coverage = (counts / region_counts).max()
          if size in data:
            data[size][0].append(purity)
            data[size][1].append(coverage)
          else:
            data[size] = [[purity], [coverage]]
        c_size, c_purity, c_coverage = [k for k,v in data.items()], [v[0] for k,v in data.items()], [v[1] for k,v in data.items()]
        all_size = np.repeat(c_size, [len(p) for p in c_purity])
        c_size = all_size / sum(all_size)
        w_purity_col.append(sum([cs * cp for cs, cp in zip(c_size, [p for ps in c_purity for p in ps])]))
        # w_coverage_col.append(sum([cs * cc for cs, cc in zip(c_size, [c for css in c_coverage for c in css])]))
        # w_purity_col.append(sum([cs * np.sum(cp) for cs, cp in zip(c_size, c_purity)]))
        # w_coverage_col.append(sum([cs * np.sum(cp) for cs, cp in zip(c_size, c_coverage)]))

        assert len(all_regions) == len(comm2label(comms)), '{}, {}'.format(len(all_regions), len(comm2label(comms)))
        # print(len(regions), len(comm2label(comms)))
        # ri_list.append(adjusted_rand_score(all_regions, comm2label(comms)))
      df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(w_purity_col)[:,None], np.array([combine_stimulus(col)[1]] * len(w_purity_col))[:,None], np.array([row] * len(w_purity_col))[:,None]), 1), columns=['weighted purity', 'combined stimulus', 'session'])], ignore_index=True)
  df['weighted purity'] = pd.to_numeric(df['weighted purity'])
  return z_sc, df

# pairtype = 'pos_neg'
# pairtype = 'pos_dis'
pairtype='neg_dis'
z_sc, wpurity_df = get_signalcorr_wpurity(G_ccg_dict, area_dict, visual_regions, best_comms_dict, signal_correlation_dict, pairtype=pairtype)
#%%
def plot_signalcorrVSwpurity_markers(wpurity_df, z_sc, pairtype='neg_dis'):
  fig, ax = plt.subplots(1, 1, figsize=(4, 4))
  X, Y = [], []
  for s_ind, session in enumerate(session2keep):
    data = wpurity_df[wpurity_df['session']==session].groupby('combined stimulus')
    x = data.mean(numeric_only=True).loc[combined_stimulus_names].values.flatten()[1:]
    y = z_sc[s_ind]
    # err = data.std(numeric_only=True).loc[combined_stimulus_names].values.flatten()[1:]
    # y = data.mean(numeric_only=True).loc[combined_stimulus_names].values.flatten()
    # err = data.std(numeric_only=True).loc[combined_stimulus_names].values.flatten()
    # err = 1.96 * data.std(numeric_only=True).loc[combined_stimulus_names].values.flatten() / data.size().loc[combined_stimulus_names].pow(1./2).values.flatten() # 95% confidence interval
    # err = stats.t.ppf((1 + 0.95) / 2., data.size().loc[combined_stimulus_names]-1) * data.sem().loc[combined_stimulus_names].values.flatten()
    X += list(x)
    Y += list(y)
    for ind, (xi, yi) in enumerate(zip(x, y)):
      ax.scatter(xi, yi, marker=stimulus2marker[combined_stimulus_names[ind+1]], s=10*error_size_dict[stimulus2marker[combined_stimulus_names[ind+1]]], linewidth=1.,color='k', facecolor='white', zorder=2)
    # for ind, (xi, yi, erri) in enumerate(zip(x, y, err)):
    #   ax.errorbar(xi, yi, yerr=erri, fmt=' ', linewidth=2.,color='.1', zorder=1)
    #   ax.scatter(xi, yi, marker=stimulus2marker[combined_stimulus_names[ind+1]], s=10*error_size_dict[stimulus2marker[combined_stimulus_names[ind+1]]], linewidth=1.,color='k', facecolor='white', zorder=2)
  X, Y = (list(t) for t in zip(*sorted(zip(X, Y))))
  X, Y = np.array(X), np.array(Y)
  slope, intercept, r_value, p_value, std_err = stats.linregress(X,Y)
  line = slope*X+intercept
  locx, locy = .4, .83
  text = 'r={:.2f}, p={:.1e}'.format(r_value, p_value)
  ax.plot(X, line, color='.2', linestyle=(5,(10,3)), alpha=.5)
  ax.text(locx, locy, text, horizontalalignment='center',
    verticalalignment='center', transform=ax.transAxes, fontsize=22)
  ax.xaxis.set_tick_params(labelsize=20)
  ax.yaxis.set_tick_params(labelsize=20)
  ax.set_xlabel('Weighted purity', fontsize=22)
  ax.set_ylabel(r'$Z_{sc}$', fontsize=22)
  for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(1.5)
    ax.spines[axis].set_color('k')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.tick_params(width=1.5)
  plt.tight_layout()
  plt.savefig('./plots/signalcorrVSwpurity_markers_{}.pdf'.format(pairtype), transparent=True)
  # plt.show()

plot_signalcorrVSwpurity_markers(wpurity_df, z_sc, pairtype)
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
  plt.xticks(list(range(1, len(all_ri_list)+1)), stimulus_labels)
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
  plt.xticks(list(range(1, len(all_ri_list)+1)), stimulus_labels)
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
  plt.xticks(list(range(1, len(all_ri_list)+1)), stimulus_labels)
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
          df = pd.concat([df, pd.DataFrame(np.concatenate((np.array([combine_stimulus(col)[1]] * len(num_comm_list))[:,None], np.array([region] * len(num_comm_list))[:,None], np.array(num_comm_list)[:,None], np.array(comm_size_list)[:,None], np.array(occupancy_list)[:,None], np.array(leadership_list)[:,None], np.array(purity_list)[:,None], np.array(entropy_list)[:,None]), 1), columns=['combined stimulus', 'region', 'number of communities', 'community size', 'occupancy', 'leadership', 'purity', 'entropy'])], ignore_index=True)

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
  # names = ['community size', 'occupancy', 'leadership', 'purity', 'entropy']
  names = ['occupancy', 'leadership', 'purity', 'entropy']
  # names = ['purity', 'entropy']
  for name1, name2 in itertools.combinations(names, 2):
    if (name1 != 'community size') and (name2 != 'community size'):
      sharex, sharey = False, False
    fig, axes = plt.subplots(1, 6, figsize=(24.5, 4), sharex=sharex, sharey=sharey)
    axes[0].set_ylabel(name2.capitalize())
    axes[0].set_ylabel(axes[0].get_ylabel(), fontsize=20,color='k') #, weight='bold'
    for s_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
      ax = axes[s_ind]
      for r_ind, region in enumerate(regions):
        mat = df[(df['combined stimulus']==combined_stimulus_name) & (df['region']==region)][[name1, name2]].values
        if mat.size:
          x, y = mat[:,0], mat[:,1]
          ax.scatter(x, y, facecolors='none', linewidth=.2, edgecolors=region_colors[r_ind], alpha=.7)
          ax.scatter(np.mean(x), np.mean(y), facecolors='none', linewidth=3., marker='^', s=200, edgecolors=region_colors[r_ind])
      ax.set_title(combined_stimulus_name.replace('\n', ' '), fontsize=20)
      ax.set_xlabel(name1.capitalize())
      
      ax.xaxis.set_tick_params(labelsize=16)
      ax.yaxis.set_tick_params(labelsize=16)
      ax.set_xlabel(ax.get_xlabel(), fontsize=20,color='k') #, weight='bold'
      for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_color('0.2')
      ax.spines['top'].set_visible(False)
      ax.spines['right'].set_visible(False)
      ax.tick_params(width=1.5)
      if (name1 != 'community size') and (name2 != 'community size'):
        ax.set_xlim(-.05, 1.05)
        ax.set_ylim(-.05, 1.05)
      ax.plot(np.arange(*ax.get_xlim(),.1), np.arange(*ax.get_ylim(),.1), linewidth=3, color='.2', linestyle='--', alpha=.4)
    for r_ind, region in enumerate(regions):
      ax.scatter([], [], facecolors='none', linewidth=2, edgecolors=region_colors[r_ind], label=region)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='', bbox_to_anchor=(.9, 1.), handletextpad=-0.1, loc='upper left', fontsize=16, frameon=False)
    plt.tight_layout()
    plt.savefig('./plots/{}_{}.pdf'.format(name1.replace(' ', '_'), name2.replace(' ', '_')), transparent=True)
    # plt.show()

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
    ax.set_title(stimulus_labels[col_ind].replace('\n', ' '))
  fig.tight_layout(rect=[0, 0, .9, 1])
  plt.savefig('./plots/heatmap_region_community.pdf', transparent=True)
  plt.show()

plot_heatmap_region_community(comms_dict, area_dict, visual_regions, max_neg_reso=max_reso_config)
# %%
####################### Figure 4 #######################
########################################################
#%%
# Figure 4A
################## relative count of signed node pairs
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
  rows, cols = get_rowcol(G_dict)
  fig, axes = plt.subplots(len(combined_stimulus_names), 1, figsize=(4, 1.*len(combined_stimulus_names)), sharex=True, sharey=True)
  for s_ind, stimulus_name in enumerate(combined_stimulus_names):
    ax = axes[s_ind]
    ax.set_title(stimulus_name.replace('\n', ' '), fontsize=15, rotation=0)
    all_pair_count = defaultdict(lambda: [])
    for col in combined_stimuli[s_ind]:
      for row in rows:
        G = G_dict[row][col].copy() if col in G_dict[row] else nx.DiGraph()
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
    # for median in box_plot['medians']:
    #   median.set_color('black')
    #   median.set_linewidth(3)
    for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
      plt.setp(box_plot[item], color=box_color)
    for patch in box_plot['boxes']:
      patch.set_facecolor('none')
      ax.set_xticks([])
    left, right = plt.xlim()
    ax.hlines(1, xmin=left, xmax=right, color='.5', alpha=.6, linestyles='--', linewidth=2)
    # plt.hlines(1, color='r', linestyles='--')
    if log:
      ax.set_yscale('log')
    # ax.xaxis.set_tick_params(labelsize=18)
    ax.yaxis.set_tick_params(labelsize=18)
    # ax.set_xlabel(ax.get_xlabel(), fontsize=14,color='k') #, weight='bold'
    ax.set_ylabel('', fontsize=20,color='k') #, weight='bold'
    for axis in ['bottom', 'left']:
      ax.spines[axis].set_linewidth(1.)
      ax.spines[axis].set_color('k')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=1.)
  fig.subplots_adjust(hspace=.7) #wspace=0.2
  # fig=axes[0].figure
  # fig.text(0.02,0.5, "Relative count", fontsize=20, ha="center", va="center", rotation=90)
  # plt.tight_layout(rect=[0.05, 0., .99, 1.])
  image_name = './plots/relative_count_signed_pair.pdf'
  # plt.show()
  plt.savefig(image_name, transparent=True)

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
#%%
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
#%%
# Figure S7D
################## plot weight/confidence difference between within motif/other neurons
def get_data_within_cross_motif(G_dict, signed_motif_types, weight='confidence'):
  rows, cols = get_rowcol(G_dict)
  within_motif_dict, cross_motif_dict = {}, {}
  motif_types = []
  motif_edges, motif_sms = {}, {}
  for signed_motif_type in signed_motif_types:
    motif_types.append(signed_motif_type.replace('+', '').replace('-', ''))
  for motif_type in motif_types:
    motif_edges[motif_type], motif_sms[motif_type] = get_edges_sms(motif_type, weight='confidence')
  for row_ind, row in enumerate(rows):
    print(row)
    within_motif_dict[row], cross_motif_dict[row] = {}, {}
    for col_ind, col in enumerate(cols):
      print(col)
      within_motif_dict[row][col], cross_motif_dict[row][col] = [], []
      G = G_dict[row][col]
      motifs_by_type = find_triads(G) # faster
      other_edges = set(list(G.edges()))
      for signed_motif_type in signed_motif_types:
        motif_type = signed_motif_type.replace('+', '').replace('-', '')
        motifs = motifs_by_type[motif_type]
        for motif in motifs:
          smotif_type = motif_type + get_motif_sign_new(motif, motif_edges[motif_type], motif_sms[motif_type], weight='confidence')
          # smotif_type = motif_type + get_motif_sign_new(motif, motif_type, weight=weight)
          if smotif_type == signed_motif_type:
            within_motif_dict[row][col].append(np.mean([abs(G[e[0]][e[1]][weight]) for e in motif.edges()]))
            other_edges -= set(list(motif.edges()))
      for e in other_edges:
        cross_motif_dict[row][col].append(abs(G[e[0]][e[1]][weight]))
  df = pd.DataFrame()
  for col in cols:
    for row in session2keep:
      within_motif, cross_motif = within_motif_dict[row][col], cross_motif_dict[row][col]
      within_motif, cross_motif = [e for e in within_motif if not np.isnan(e)], [e for e in cross_motif if not np.isnan(e)] # remove nan values
      df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(within_motif)[:,None], np.array(['within motif'] * len(within_motif))[:,None], np.array([combine_stimulus(col)[1]] * len(within_motif))[:,None]), 1), columns=['CCG', 'type', 'stimulus'])], ignore_index=True)
      df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(cross_motif)[:,None], np.array(['otherwise'] * len(cross_motif))[:,None], np.array([combine_stimulus(col)[1]] * len(cross_motif))[:,None]), 1), columns=['CCG', 'type', 'stimulus'])], ignore_index=True)
  df['CCG'] = pd.to_numeric(df['CCG'])
  if weight == 'confidence':
    df=df.rename(columns = {'CCG':'CCG Z score'})
  return df

sig_motif_types = ['030T+++', '120D++++', '120U++++', '120C++++', '210+++++', '300++++++']
confidence_within_cross_motif_df = get_data_within_cross_motif(G_ccg_dict, sig_motif_types, weight='confidence')
weight_within_cross_motif_df = get_data_within_cross_motif(G_ccg_dict, sig_motif_types, weight='weight')
#%%
# Figure S7D
def save_within_motif_otherwise_legend():
  fig, ax = plt.subplots(1,1, figsize=(.4*(len(combined_stimulus_names)-1), .5))
  palette = ['k','w']
  df = pd.DataFrame([[0,0,0],[0,0,1]], columns=['x', 'y', 'type'])
  barplot = sns.barplot(x='x', y='y', hue="type", hue_order=[0, 1], palette=palette, ec='k', linewidth=2., data=df, ax=ax, capsize=.05, width=0.6)
  handles, labels = ax.get_legend_handles_labels()
  legend = ax.legend(handles, ['within motif', 'otherwise'], title='', bbox_to_anchor=(0.1, -1.), handletextpad=0.3, loc='upper left', fontsize=25, frameon=False, ncol=2) # change legend order to within/cross similar module then cross random
  plt.axis('off')
  export_legend(legend, './plots/within_motif_otherwise_legend_2cols.pdf')
  plt.tight_layout()
  # plt.savefig('./plots/stimuli_markers.pdf', transparent=True)
  plt.show()

save_within_motif_otherwise_legend()
#%%
# Figure S7D
def plot_data_within_cross_motif_significance(df, weight='confidence'):
  fig, ax = plt.subplots(1,1, figsize=(1.5*(len(combined_stimulus_names)-1), 5))
  palette = ['k','w']
  y = 'CCG' if weight=='weight' else 'CCG Z score'
  barplot = sns.barplot(x='stimulus', y=y, hue="type", palette=palette, ec='k', linewidth=2., data=df, ax=ax, capsize=.05, width=0.6)
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
  if weight == 'confidence':
    ylabel = r'$Z_{CCG}$'
  elif weight == 'weight':
    ylabel = 'Connection strength'
  ax.set_ylabel(ylabel, fontsize=40) #'Absolute ' + 
  handles, labels = ax.get_legend_handles_labels()
  ax.legend([], [], fontsize=0)
  # add significance annotation
  alpha_list = [.0001, .001, .01, .05]

  maxx = 0
  for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
    within_motif, cross_motif = df[(df.stimulus==combined_stimulus_name)&(df.type=='within motif')][y].values.flatten(), df[(df.stimulus==combined_stimulus_name)&(df.type=='otherwise')][y].values.flatten()
    within_sr, cross_sr = confidence_interval(within_motif)[1], confidence_interval(cross_motif)[1]
    maxx = max(within_sr, cross_sr) if max(within_sr, cross_sr) > maxx else maxx
  if weight == 'confidence':
    h, l = .01 * maxx, .01 * maxx
  elif weight == 'weight':
    h, l = .05 * maxx, .05 * maxx
  for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
    within_motif, cross_motif = df[(df.stimulus==combined_stimulus_name)&(df.type=='within motif')][y].values.flatten(), df[(df.stimulus==combined_stimulus_name)&(df.type=='otherwise')][y].values.flatten()
    _, p = ztest(within_motif, cross_motif, value=0)
    diff_star = '*' * (len(alpha_list) - bisect(alpha_list, p)) if len(alpha_list) > bisect(alpha_list, p) else 'ns'
    within_sr, cross_sr = confidence_interval(within_motif)[1], confidence_interval(cross_motif)[1]
    within_sr = within_sr + h
    cross_sr = cross_sr + h
    annot_difference(diff_star, -.15 + cs_ind, .15 + cs_ind, max(within_sr, cross_sr), l, 2.5, 28, ax=ax)
  if weight == 'confidence':
    ax.set_ylim(bottom=4)
  plt.tight_layout(rect=[.02, -.03, 1, 1])

  # plt.tight_layout()
  # plt.show()
  plt.savefig('./plots/{}_within_cross_motif.pdf'.format(weight), transparent=True)

plot_data_within_cross_motif_significance(confidence_within_cross_motif_df, weight='confidence')
# plot_data_within_cross_motif_significance(weight_within_cross_motif_df, weight='weight')
#%%
# Figure 4F
################## plot signal correlation difference between within motif/other neurons
def get_signalcorr_within_cross_motif(G_dict, signed_motif_types, signal_correlation_dict, pair_type='all'):
  rows, cols = get_rowcol(G_dict)
  within_motif_dict, cross_motif_dict = {}, {}
  motif_types = []
  motif_edges, motif_sms = {}, {}
  for signed_motif_type in signed_motif_types:
    motif_types.append(signed_motif_type.replace('+', '').replace('-', ''))
  for motif_type in motif_types:
    motif_edges[motif_type], motif_sms[motif_type] = get_edges_sms(motif_type, weight='confidence')
  for row_ind, row in enumerate(rows):
    print(row)
    active_area = active_area_dict[row]
    node_idx = sorted(active_area.keys())
    within_motif_dict[row], cross_motif_dict[row] = {}, {}
    for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names[1:]):
      for col in combined_stimuli[combined_stimulus_names.index(combined_stimulus_name)]:
        print(col)
        within_motif_dict[row][col], cross_motif_dict[row][col] = [], []
        G = G_dict[row][col]
        signal_corr = signal_correlation_dict[row][combined_stimulus_name]
        motifs_by_type = find_triads(G) # faster
        if pair_type == 'all': # all neuron pairs
          neuron_pairs = list(itertools.combinations(node_idx, 2))
        elif pair_type == 'connected': # limited to connected pairs only
          neuron_pairs = list(G.to_undirected().edges())
        other_edges = set(neuron_pairs)
        for signed_motif_type in signed_motif_types:
          motif_type = signed_motif_type.replace('+', '').replace('-', '')
          motifs = motifs_by_type[motif_type]
          for motif in motifs:
            smotif_type = motif_type + get_motif_sign_new(motif, motif_edges[motif_type], motif_sms[motif_type], weight='confidence')
            # smotif_type = motif_type + get_motif_sign(motif, motif_type, weight='weight')
            if smotif_type == signed_motif_type:
              if pair_type == 'all': # all neuron pairs
                motif_pairs = list(itertools.combinations(motif.nodes(), 2))
              elif pair_type == 'connected': # limited to connected pairs only
                motif_pairs = list(motif.to_undirected().edges())
              within_signal_corr = [signal_corr[e] for e in motif_pairs if not np.isnan(signal_corr[e])]
              if len(within_signal_corr):
                within_motif_dict[row][col] += within_signal_corr
              other_edges -= set(motif_pairs)
        for e in other_edges:
          if not np.isnan(signal_corr[e]):
            cross_motif_dict[row][col].append(signal_corr[e])
  df = pd.DataFrame()
  for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names[1:]):
    for col in combined_stimuli[combined_stimulus_names.index(combined_stimulus_name)]:
      for row in session2keep:
        within_motif, cross_motif = within_motif_dict[row][col], cross_motif_dict[row][col]
        # within_motif, cross_motif = [e for e in within_motif if not np.isnan(e)], [e for e in cross_motif if not np.isnan(e)] # remove nan values
        df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(within_motif)[:,None], np.array(['within motif'] * len(within_motif))[:,None], np.array([combined_stimulus_name] * len(within_motif))[:,None], np.array([row] * len(within_motif))[:,None]), 1), columns=['signal_corr', 'type', 'stimulus', 'session'])], ignore_index=True)
        df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(cross_motif)[:,None], np.array(['otherwise'] * len(cross_motif))[:,None], np.array([combined_stimulus_name] * len(cross_motif))[:,None], np.array([row] * len(cross_motif))[:,None]), 1), columns=['signal_corr', 'type', 'stimulus', 'session'])], ignore_index=True)
  df['signal_corr'] = pd.to_numeric(df['signal_corr'])
  return df

# pair_type = 'all'
pair_type = 'connected'
sig_motif_types = ['030T+++', '120D++++', '120U++++', '120C++++', '210+++++', '300++++++']
signal_corr_within_cross_motif_df = get_signalcorr_within_cross_motif(G_ccg_dict, sig_motif_types, signal_correlation_dict, pair_type=pair_type)
#%%
# Figure 4F
def plot_signalcorr_within_cross_motif_significance(df, pair_type='all'):
  fig, ax = plt.subplots(1,1, figsize=(1.5*(len(combined_stimulus_names)-1), 5))
  df = df.set_index('stimulus')
  df = df.loc[combined_stimulus_names[1:]]
  df.reset_index(inplace=True)
  palette = ['k','w']
  y = 'signal_corr'
  barplot = sns.barplot(x='stimulus', y=y, hue="type", palette=palette, ec='k', linewidth=2., data=df, ax=ax, capsize=.05, width=0.6)
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
  for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names[1:]):
    within_motif, cross_motif = df[(df.stimulus==combined_stimulus_name)&(df.type=='within motif')][y].values.flatten(), df[(df.stimulus==combined_stimulus_name)&(df.type=='otherwise')][y].values.flatten()
    within_sr, cross_sr = confidence_interval(within_motif)[1], confidence_interval(cross_motif)[1]
    maxx = max(within_sr, cross_sr) if max(within_sr, cross_sr) > maxx else maxx
  h, l = .05 * maxx, .05 * maxx
  for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names[1:]):
    within_motif, cross_motif = df[(df.stimulus==combined_stimulus_name)&(df.type=='within motif')][y].values.flatten(), df[(df.stimulus==combined_stimulus_name)&(df.type=='otherwise')][y].values.flatten()
    _, p = ztest(within_motif, cross_motif, value=0)
    diff_star = '*' * (len(alpha_list) - bisect(alpha_list, p)) if len(alpha_list) > bisect(alpha_list, p) else 'ns'
    within_sr, cross_sr = confidence_interval(within_motif)[1], confidence_interval(cross_motif)[1]
    within_sr = within_sr + h
    cross_sr = cross_sr + h
    annot_difference(diff_star, -.15 + cs_ind, .15 + cs_ind, max(within_sr, cross_sr), l, 2.5, 28, ax=ax)
  plt.tight_layout(rect=[.02, -.03, 1, 1])

  # plt.tight_layout()
  # plt.show()
  plt.savefig('./plots/signalcorr_within_cross_motif_{}.pdf'.format(pair_type), transparent=True)

plot_signalcorr_within_cross_motif_significance(signal_corr_within_cross_motif_df, pair_type=pair_type)
#%%
############################# paired t test for each mouse
def plot_signal_correlation_within_cross_motif_paired_significance(df, pair_type='all'):
  fig, axes = plt.subplots(1,len(combined_stimulus_names)-1, sharex=True, sharey=True, figsize=(2.8*(len(combined_stimulus_names)-1), 3))
  palette = ['k','.6']
  df = df.groupby(['type', 'stimulus', 'session']).mean().reset_index() # first average over individual mouse
  df = df.set_index('session')
  df = df.loc[session2keep]
  df.reset_index(inplace=True)
  df = df.set_index('stimulus')
  df = df.loc[combined_stimulus_names[1:]]
  df.reset_index(inplace=True)
  axes[0].set_ylabel('Signal correlation', fontsize=30)
  axes[0].set_xlim(-.5, 1.5)
  for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names[1:]):
    ax = axes[cs_ind]
    within_comm, cross_comm = df[(df.stimulus==combined_stimulus_name)&(df.type=='within motif')]['signal_corr'].values.flatten(), df[(df.stimulus==combined_stimulus_name)&(df.type=='otherwise')]['signal_corr'].values.flatten()
    ax.scatter(np.zeros(len(within_comm)), within_comm, s=50, color='k')
    ax.scatter(np.ones(len(cross_comm)), cross_comm, s=50, color='k')

    # plotting the lines
    for i in range(len(within_comm)):
      ax.plot([0,1], [within_comm[i], cross_comm[i]], linewidth=2.5, c='k')

    # ax.set_xticks([0,1], ['within community', 'cross community'])
    ax.set_xticks([-.3,1.3])
    ax.set_xticklabels(labels=['within\nmotif', 'otherwise'], fontsize=26)
    p_value = stats.ttest_rel(within_comm, cross_comm).pvalue
    locx, locy = .6, 1.2
    ax.text(locx, locy, 'p={:.1e}'.format(p_value), horizontalalignment='center',
        verticalalignment='center', transform=ax.transAxes, fontsize=25)
    
    # palette = [[plt.cm.tab20b(i) for i in range(4)][i] for i in [0, 3]]
    # barplot = sns.barplot(x='stimulus', y='signal correlation', hue="type", palette=palette, data=df, ax=ax, capsize=.05, width=0.5)
    ax.yaxis.set_tick_params(labelsize=30)
    # plt.xticks([], []) # use markers to represent stimuli!
    ax.xaxis.set_tick_params(length=0)
    for axis in ['bottom', 'left']:
      ax.spines[axis].set_linewidth(2.5)
      ax.spines[axis].set_color('k')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=2.5)
    ax.set_xlabel('')
  # handles, labels = ax.get_legend_handles_labels()
  # ax.legend(handles, ['within module', 'cross module'], title='', bbox_to_anchor=(0., 1.), handletextpad=0.3, loc='upper left', fontsize=35, frameon=False)
  # add significance annotation
  alpha_list = [.0001, .001, .01, .05]
  
  plt.tight_layout(rect=[.02, -.03, 1, 1])
  # plt.show()
  plt.savefig('./plots/signal_correlation_within_cross_motif_{}_paired.pdf'.format(pair_type), transparent=True)

plot_signal_correlation_within_cross_motif_paired_significance(signal_corr_within_cross_motif_df, pair_type)
#%%
################## plot weight/confidence difference between within motif/other neurons
def get_data_inside_outside_motif_comm(G_dict, comms_dict, max_neg_reso, signed_motif_types, weight='confidence'):
  rows, cols = get_rowcol(G_dict)
  motif_dict = {}
  for row_ind, row in enumerate(rows):
    print(row)
    motif_dict[row] = {}
    for col_ind, col in enumerate(cols):
      print(col)
      motif_dict[row][col] = [[] for _ in range(4)] # inside/outside motif within community, inside/outside motif between communities
      G = G_dict[row][col]
      motifs_by_type = find_triads(G) # faster
      motif_edges, other_edges = [], set(list(G.edges()))
      for signed_motif_type in signed_motif_types:
        motif_type = signed_motif_type.replace('+', '').replace('-', '')
        motifs = motifs_by_type[motif_type]
        for motif in motifs:
          smotif_type = motif_type + get_motif_sign(motif, motif_type, weight=weight)
          if smotif_type == signed_motif_type:
            motif_edges += list(motif.edges())
            other_edges -= set(list(motif.edges()))
      for run in range(metrics['Hamiltonian'].shape[-1]):
        max_reso = max_neg_reso[row_ind, col_ind, run]
        comms_list = comms_dict[row][col][max_reso]
        comms = comms_list[run]
        node_to_community = comm2partition(comms)
        for e in motif_edges:
          if node_to_community[e[0]] == node_to_community[e[1]]:
            motif_dict[row][col][0].append(abs(G[e[0]][e[1]][weight]))
          else:
            motif_dict[row][col][2].append(abs(G[e[0]][e[1]][weight]))
        for e in other_edges:
          if node_to_community[e[0]] == node_to_community[e[1]]:
            motif_dict[row][col][1].append(abs(G[e[0]][e[1]][weight]))
          else:
            motif_dict[row][col][3].append(abs(G[e[0]][e[1]][weight]))
  df = pd.DataFrame()
  for col in cols:
    for row in session2keep:
      within_motif_within_comm, cross_motif_within_comm, within_motif_cross_comm, cross_motif_cross_comm = motif_dict[row][col]
      within_motif_within_comm, cross_motif_within_comm, within_motif_cross_comm, cross_motif_cross_comm = [e for e in within_motif_within_comm if not np.isnan(e)], [e for e in cross_motif_within_comm if not np.isnan(e)], [e for e in within_motif_cross_comm if not np.isnan(e)], [e for e in cross_motif_cross_comm if not np.isnan(e)] # remove nan values
      df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(within_motif_within_comm)[:,None], np.array(['within motif within comm'] * len(within_motif_within_comm))[:,None], np.array([combine_stimulus(col)[1]] * len(within_motif_within_comm))[:,None]), 1), columns=['CCG', 'type', 'stimulus'])], ignore_index=True)
      df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(cross_motif_within_comm)[:,None], np.array(['outside motif within comm'] * len(cross_motif_within_comm))[:,None], np.array([combine_stimulus(col)[1]] * len(cross_motif_within_comm))[:,None]), 1), columns=['CCG', 'type', 'stimulus'])], ignore_index=True)
      df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(within_motif_cross_comm)[:,None], np.array(['within motif cross comm'] * len(within_motif_cross_comm))[:,None], np.array([combine_stimulus(col)[1]] * len(within_motif_cross_comm))[:,None]), 1), columns=['CCG', 'type', 'stimulus'])], ignore_index=True)
      df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(cross_motif_cross_comm)[:,None], np.array(['outside motif cross comm'] * len(cross_motif_cross_comm))[:,None], np.array([combine_stimulus(col)[1]] * len(cross_motif_cross_comm))[:,None]), 1), columns=['CCG', 'type', 'stimulus'])], ignore_index=True)
  df['CCG'] = pd.to_numeric(df['CCG'])
  if weight == 'confidence':
    df=df.rename(columns = {'CCG':'CCG Z score'})
  return df

sig_motif_types = ['030T+++', '120D++++', '120U++++', '120C++++', '210+++++', '300++++++']
confidence_within_cross_motif_comm_df = get_data_inside_outside_motif_comm(G_ccg_dict, comms_dict, max_reso_subs, sig_motif_types, weight='confidence')
weight_within_cross_motif_comm_df = get_data_inside_outside_motif_comm(G_ccg_dict, comms_dict, max_reso_subs, sig_motif_types, weight='weight')
#%%
def plot_data_within_cross_motif_comm_significance(df, weight='confidence'):
  fig, ax = plt.subplots(1,1, figsize=(2*(len(combined_stimulus_names)-1), 6))
  palette = [[plt.cm.tab20b(i) for i in range(4)][i] for i in [0, 3]]
  y = 'CCG' if weight=='weight' else 'CCG Z score'
  barplot = sns.barplot(x='stimulus', y=y, hue="type", palette=palette, data=df, ax=ax, capsize=.05, width=0.6)
  ax.yaxis.set_tick_params(labelsize=30)
  plt.setp(ax.get_legend().get_title(), fontsize='0') # for legend title
  plt.xticks(range(len(combined_stimulus_names)), combined_stimulus_names, fontsize=20)
  ax.xaxis.set_tick_params(length=0)
  for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(1.5)
    ax.spines[axis].set_color('0.2')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.tick_params(width=1.5)
  ax.set_xlabel('')
  ax.set_ylabel('Absolute ' + ax.get_ylabel(), fontsize=30)
  handles, labels = ax.get_legend_handles_labels()
  ax.legend(handles, ['within motif within comm', 'outside motif within comm', 'within motif cross comm', 'outside motif cross comm'], title='', bbox_to_anchor=(.3, 1.2), handletextpad=0.3, loc='upper left', fontsize=25, frameon=False)
  # add significance annotation
  alpha_list = [.0001, .001, .01, .05]
  if weight == 'confidence':
    ax.set_ylim(top=8)
    h, l = .1, .1
  else:
    h, l = 0.002, .0005
  for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
    within_motif_within_comm, cross_motif_within_comm, within_motif_cross_comm, cross_motif_cross_comm = df[(df.stimulus==combined_stimulus_name)&(df.type=='within motif within comm')][y].values.flatten(), df[(df.stimulus==combined_stimulus_name)&(df.type=='outside motif within comm')][y].values.flatten(), df[(df.stimulus==combined_stimulus_name)&(df.type=='within motif cross comm')][y].values.flatten(), df[(df.stimulus==combined_stimulus_name)&(df.type=='outside motif cross comm')][y].values.flatten()
    _, p = ztest(within_motif_within_comm, cross_motif_within_comm, value=0)
    diff_star = '*' * (len(alpha_list) - bisect(alpha_list, p)) if len(alpha_list) > bisect(alpha_list, p) else 'ns'
    within_sr, cross_sr = confidence_interval(within_motif_within_comm)[1], confidence_interval(cross_motif_within_comm)[1]
    within_sr = within_sr + h
    cross_sr = cross_sr + h
    annot_difference(diff_star, -.22 + cs_ind, -.08 + cs_ind, max(within_sr, cross_sr), l, 2.5, ax=ax)

    _, p = ztest(within_motif_cross_comm, cross_motif_cross_comm, value=0)
    diff_star = '*' * (len(alpha_list) - bisect(alpha_list, p)) if len(alpha_list) > bisect(alpha_list, p) else 'ns'
    within_sr, cross_sr = confidence_interval(within_motif_cross_comm)[1], confidence_interval(cross_motif_cross_comm)[1]
    within_sr = within_sr + h
    cross_sr = cross_sr + h
    annot_difference(diff_star, .08 + cs_ind, .22 + cs_ind, max(within_sr, cross_sr), l, 2.5, ax=ax)
  plt.tight_layout()
  # plt.show()
  plt.savefig('./plots/{}_within_cross_motif_comm.pdf'.format(weight), transparent=True)

# plot_data_within_cross_motif_comm_significance(confidence_within_cross_motif_comm_df, weight='confidence')
plot_data_within_cross_motif_comm_significance(weight_within_cross_motif_comm_df, weight='weight')
#%%
################## remove outlier from mean_df (outside 2 std)
mean_df = remove_outlier_meandf(whole_df, mean_df, signed_motif_types)
#%%
############### Z score distribution for all signed motifs
plot_zscore_distribution(whole_df, measure, n)
#%%
# def plot_zscore_all_motif(df, model_name):
#   stimulus_order = [s for s in combined_stimulus_names if df.stimulus.str.contains(s).sum()]
#   fig, axes = plt.subplots(len(stimulus_order),1, sharex=True, sharey=True, figsize=(100, 6*len(stimulus_order)))
#   TRIAD_NAMES = ('003', '012', '102', '021D', '021U', '021C', '111D', '111U', '030T', '030C', '201', '120D', '120U', '120C', '210', '300')
#   sorted_types = [sorted([smotif for smotif in df['signed motif type'].unique() if mt in smotif]) for mt in TRIAD_NAMES]
#   sorted_types = [item for sublist in sorted_types for item in sublist]
#   for s_ind, stimulus in enumerate(stimulus_order):
#     print(stimulus)
#     data = df[df.stimulus==stimulus]
#     ax = axes[s_ind]
#     ax.set_title(combined_stimulus_names[s_ind].replace('\n', ' '), fontsize=50, rotation=0)
#     barplot = sns.barplot(data=data, x="signed motif type", y="intensity z score", order=sorted_types, ax=ax)
#     ax.xaxis.set_tick_params(labelsize=35, rotation=90)
#     ax.yaxis.set_tick_params(labelsize=35)
#     for axis in ['bottom', 'left']:
#       ax.spines[axis].set_linewidth(1.5)
#       ax.spines[axis].set_color('0.2')
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.tick_params(width=2.5)
#     ax.xaxis.set_tick_params(length=0)
#     ax.set_ylabel('Z score of intensity', fontsize=40)
#     # ax.set_ylim(top=60)
#     # plt.yscale('symlog')
#     barplot.set(xlabel=None)
#     ax.set_ylim(-10, 20)
#   plt.tight_layout()
#   figname = './plots/zscore_all_motifs_{}.pdf'.format(model_name.replace(' ', '_'))
#   # figname = './plots/zscore_all_motifs_log_{}_{}fold.jpg'.format(measure, n)
#   plt.savefig(figname, transparent=True)
# ################## box plot of z score for all signed motifs
# dfs = [whole_df1, whole_df2, whole_df3, whole_df4]
# for df_ind, df in enumerate(dfs):
#   plot_zscore_all_motif(df, model_names[df_ind])
#%%
# Figure 4B
def plot_zscore_allmotif_lollipop(df, model_name):
  # stimulus_order = [s for s in combined_stimulus_names if df.stimulus.str.contains(s).sum()]
  fig, axes = plt.subplots(len(combined_stimulus_names),1, sharex=True, sharey=True, figsize=(50, 3*len(combined_stimulus_names)))
  sorted_types = [sorted([smotif for smotif in df['signed motif type'].unique() if mt in smotif]) for mt in TRIAD_NAMES]
  sorted_types = [item for sublist in sorted_types for item in sublist]
  motif_types = TRIAD_NAMES[3:]
  motif_loc = [np.mean([i for i in range(len(sorted_types)) if mt in sorted_types[i]]) for mt in motif_types]
  # palette = [plt.cm.tab20(i) for i in range(13)]
  palette = [[plt.cm.tab20b(i) for i in range(20)][i] for i in [0,2,3,4,6,8,10,12,16,18,19]] + [[plt.cm.tab20c(i) for i in range(20)][i] for i in [4,16]]
  for s_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
    print(combined_stimulus_name)
    data = df[df.apply(lambda x: combine_stimulus(x['stimulus'])[1], axis=1)==combined_stimulus_name]
    data = data.groupby('signed motif type').mean()
    ax = axes[s_ind]
    # ax.set_title(combined_stimulus_names[s_ind].replace('\n', ' '), fontsize=35, rotation=0)
    for t, y in zip(sorted_types, data.loc[sorted_types, "intensity z score"]):
      color = palette[motif_types.index(t.replace('+', '').replace('\N{MINUS SIGN}', ''))]
      ax.plot([t,t], [0,y], color=color, marker="o", linewidth=7, markersize=20, markevery=(1,2))
    ax.set_xlim(-.5,len(sorted_types)+.5)
    ax.set_xticks([])
    # ax.set_xticks(motif_loc)
    # ax.set_xticklabels(labels=motif_types)
    # ax.xaxis.set_tick_params(labelsize=35, rotation=90)
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
  figname = './plots/zscore_all_motifs_lollipop_{}.pdf'.format(model_name.replace(' ', '_'))
  plt.savefig(figname, transparent=True)
  # plt.show()

# plot_zscore_allmotif_lollipop(whole_df)
dfs = [whole_df1, whole_df2, whole_df3, whole_df4]
# for df_ind, df in enumerate(dfs):
df_ind = 3
plot_zscore_allmotif_lollipop(dfs[df_ind], model_names[df_ind])
#%%
# Figure 4C
def scatter_ZscoreVSdensity(origin_df, G_dict):
  df = origin_df.copy()
  df['density'] = 0
  TRIAD_NAMES = ('003', '012', '102', '021D', '021U', '021C', '111D', '111U', '030T', '030C', '201', '120D', '120U', '120C', '210', '300')
  sorted_types = [sorted([smotif for smotif in df['signed motif type'].unique() if mt in smotif]) for mt in TRIAD_NAMES]
  sorted_types = [item for sublist in sorted_types for item in sublist]
  motif_types = TRIAD_NAMES[3:]
  motif_loc = [np.mean([i for i in range(len(sorted_types)) if mt in sorted_types[i]]) for mt in motif_types]
  
  color = '.1' 
  # palette = [plt.cm.tab20(i) for i in range(13)]
  palette = [[plt.cm.tab20b(i) for i in range(20)][i] for i in [0,2,3,4,6,8,10,12,16,18,19]] + [[plt.cm.tab20c(i) for i in range(20)][i] for i in [4,16]]
  rows, cols = get_rowcol(G_dict)
  fig, ax = plt.subplots(figsize=(5, 5))
  for row_ind, row in enumerate(rows):
    for col in cols:
      G = G_dict[row][col]
      df.loc[(df['session']==row) & (df['stimulus']==col), 'density'] = nx.density(G)
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
    # ax.scatter(x, y, facecolors='none', edgecolors=stimulus_type_color[st_ind], label=stimulus_types[st_ind], alpha=.9, linewidths=2)
    ax.scatter(x, y, ec='.1', fc='none', marker=stimulus2marker[combined_stimulus_name], s=10*marker_size_dict[stimulus2marker[combined_stimulus_name]], alpha=.9, linewidths=1.5)
    # ax.scatter(x, y, ec='.1', fc='none', marker=marker_list[cs_ind], s=6*markersize_list[cs_ind], alpha=.9, linewidths=2)
  
  X, Y = (list(t) for t in zip(*sorted(zip(X, Y))))
  X, Y = np.array(X), np.array(Y)
  slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(X),Y)
  line = slope*np.log10(X)+intercept
  locx, locy = .8, .1
  text = 'r={:.2f}, p={:.1e}'.format(r_value, p_value)
  ax.plot(X, line, color='.2', linestyle=(5,(10,3)), alpha=.5)
  # ax.scatter(X, Y, facecolors='none', edgecolors='.2', alpha=.6)
  ax.text(locx, locy, text, horizontalalignment='center',
     verticalalignment='center', transform=ax.transAxes, fontsize=22)
  # plt.legend(loc='upper left', fontsize=14, frameon=False)
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
  # plt.show()
  plt.savefig(f'./plots/mean_Zscore_density.pdf', transparent=True)

scatter_ZscoreVSdensity(whole_df4, G_ccg_dict)
#%%
######################## Spearman's r of Z score
def plot_zscoreVSrank(df):
  fig, axes = plt.subplots(1,len(combined_stimulus_names)-1, sharex=True, sharey=True, figsize=(5*(len(combined_stimulus_names)-1), 5))
  sorted_types = [sorted([smotif for smotif in df['signed motif type'].unique() if mt in smotif]) for mt in TRIAD_NAMES]
  sorted_types = [item for sublist in sorted_types for item in sublist]
  motif_types = TRIAD_NAMES[3:]
  motif_loc = [np.mean([i for i in range(len(sorted_types)) if mt in sorted_types[i]]) for mt in motif_types]
  # palette = [plt.cm.tab20(i) for i in range(13)]
  palette = [[plt.cm.tab20b(i) for i in range(20)][i] for i in [0,2,3,4,6,8,10,12,16,18,19]] + [[plt.cm.tab20c(i) for i in range(20)][i] for i in [4,16]]
  data_dict = {}
  for s_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
    print(combined_stimulus_name)
    data = df[df.apply(lambda x: combine_stimulus(x['stimulus'])[1], axis=1)==combined_stimulus_name]
    data = data.groupby('signed motif type').mean()
    
    data_dict[combined_stimulus_name] = data.loc[sorted_types, "intensity z score"].values.flatten()
  X = data_dict['Drifting\ngratings']
  ind = 0
  for s_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
    if combined_stimulus_name != 'Drifting\ngratings':
      ax = axes[ind]
      ind += 1
      Y = data_dict[combined_stimulus_name]
      ax.scatter(X, Y)
      r_value, p_value = spearmanr(X, Y)
      locx, locy = .5, .8
      text = 'r={:.2f}, p={:.1e}'.format(r_value, p_value)
      ax.text(locx, locy, text, horizontalalignment='center',
        verticalalignment='center', transform=ax.transAxes, fontsize=30)
      ax.xaxis.set_tick_params(labelsize=25)
      ax.yaxis.set_tick_params(labelsize=25)
      for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_color('k')
      ax.spines['top'].set_visible(False)
      ax.spines['right'].set_visible(False)
      ax.tick_params(width=1.5)
      ax.xaxis.set_tick_params(length=0)
      ax.set_ylabel('')
  plt.tight_layout()
  figname = './plots/zscore_rank.pdf'
  plt.savefig(figname, transparent=True)
  # plt.show()

plot_zscoreVSrank(whole_df4)
#%%
from scipy.stats import kendalltau
######################## Heatmap of Spearman's r of Z score
def plot_heatmap_spearman_zscore(df, name='spearmanr'):
  fig, ax = plt.subplots(1,1, figsize=(5.6,5))
  sorted_types = [sorted([smotif for smotif in df['signed motif type'].unique() if mt in smotif]) for mt in TRIAD_NAMES]
  sorted_types = [item for sublist in sorted_types for item in sublist]
  motif_types = TRIAD_NAMES[3:]
  motif_loc = [np.mean([i for i in range(len(sorted_types)) if mt in sorted_types[i]]) for mt in motif_types]
  # palette = [plt.cm.tab20(i) for i in range(13)]
  palette = [[plt.cm.tab20b(i) for i in range(20)][i] for i in [0,2,3,4,6,8,10,12,16,18,19]] + [[plt.cm.tab20c(i) for i in range(20)][i] for i in [4,16]]
  data_dict = {}
  for s_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
    print(combined_stimulus_name)
    data = df[df.apply(lambda x: combine_stimulus(x['stimulus'])[1], axis=1)==combined_stimulus_name]
    data = data.groupby('signed motif type').mean()
    data_dict[combined_stimulus_name] = data.loc[sorted_types, "intensity z score"].values.flatten()
  hm_z = np.zeros((len(combined_stimulus_names), len(combined_stimulus_names)))
  hm_z[:] = np.nan
  for cs1, cs2 in itertools.permutations(range(len(combined_stimulus_names)),2):
    if name == 'spearmanr':
      hm_z[cs1, cs2] = spearmanr(data_dict[combined_stimulus_names[cs1]], data_dict[combined_stimulus_names[cs2]])[0]
    elif name == 'kendalltau':
      hm_z[cs1, cs2] = kendalltau(data_dict[combined_stimulus_names[cs1]], data_dict[combined_stimulus_names[cs2]])[0]
  # mask = np.triu(hm_z)
  hm = sns.heatmap(hm_z, ax=ax, cmap='Greys_r', cbar=True, annot=True, annot_kws={'fontsize':12})#, mask=mask
  ax.set_xticks(0.5 + np.arange(len(combined_stimulus_names)))
  ax.set_xticklabels(labels=combined_stimulus_names, fontsize=12)
  ax.set_yticks(0.5 + np.arange(len(combined_stimulus_names)))
  ax.set_yticklabels(labels=combined_stimulus_names, fontsize=12)
  ax.xaxis.tick_top()
  # hm.set(yticklabels=[])
  # hm.set(ylabel=None)
  hm.tick_params(left=False)  # remove the ticks
  hm.tick_params(bottom=False)
  hm.tick_params(top=False)
  fig.tight_layout()
  plt.savefig('./plots/heatmap_{}_zscore.pdf'.format(name), transparent=True)
  # plt.show()

# plot_heatmap_spearman_zscore(whole_df4, name='spearmanr')
plot_heatmap_spearman_zscore(whole_df4, name='kendalltau')
#%%
# Figure 4D
from matplotlib.colors import LinearSegmentedColormap
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
  # hm_z = np.zeros((len(combined_stimulus_names), len(combined_stimulus_names)))
  # hm_z[:] = np.nan
  # for cs1, cs2 in itertools.permutations(range(len(combined_stimulus_names)),2):
  #   hm_z[cs1, cs2] = spearmanr(data_dict[combined_stimulus_names[cs1]], data_dict[combined_stimulus_names[cs2]])[0]
  # mask = np.triu(hm_z)
  hm_z = np.corrcoef(data_mat)
  np.fill_diagonal(hm_z, np.nan)

  # cmap = plt.get_cmap('hot')
  # new_cmap = truncate_colormap(cmap, 0., 0.3)
  
  colors = ['w', '.3'] # first color is black, last is red
  cm = LinearSegmentedColormap.from_list(
        "Custom", colors, N=20)

  hm = sns.heatmap(hm_z, ax=ax, cmap=cm, vmin=0, vmax=1, cbar=True, annot=True, annot_kws={'fontsize':14})#, mask=mask
  cbar = hm.collections[0].colorbar
  cbar.ax.tick_params(labelsize=20)

  # ax.set_xticks(0.5 + np.arange(len(combined_stimulus_names)))
  # ax.set_xticklabels(labels=combined_stimulus_names, fontsize=12)
  # ax.set_yticks(0.5 + np.arange(len(combined_stimulus_names)))
  # ax.set_yticklabels(labels=combined_stimulus_names, fontsize=12)
  # ax.xaxis.tick_top()

  ax.set_xticks([])
  ax.set_yticks([])
  ax.invert_yaxis()
  hm.tick_params(left=False)  # remove the ticks
  hm.tick_params(bottom=False)
  hm.tick_params(top=False)
  fig.tight_layout()
  plt.savefig('./plots/heatmap_black_correlation_zscore.pdf', transparent=True)
  # plt.show()

plot_heatmap_correlation_zscore(whole_df4)
#%%
# Figure S7E
######################## Number of significant motifs against threshold
from matplotlib.ticker import FuncFormatter # set log scale for only negative numbers
def plot_signed_sig_motif_threshold(df, threshold_list):
  # stimulus_order = [s for s in combined_stimulus_names if df.stimulus.str.contains(s).sum()]
  # stimulus_order = [s for s in stimulus_types if df[df.apply(lambda x: stimulus2stype(x['stimulus'])[1], axis=1)==stimulus_type].stimulus.str.contains(s).sum()]
  num_pos, num_neg = np.zeros((len(combined_stimulus_names), len(threshold_list))), np.zeros((len(combined_stimulus_names), len(threshold_list)))
  fig, axes = plt.subplots(1, 2, figsize=[5*2,4])
  linewidth=2.5
  for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
    data = df[df.apply(lambda x: combine_stimulus(x['stimulus'])[1], axis=1)==combined_stimulus_name]
    for t_ind, threshold in enumerate(threshold_list):
      num_pos[cs_ind, t_ind] = len(data[data['intensity z score']>=threshold])
      num_neg[cs_ind, t_ind] = len(data[data['intensity z score']<=-threshold])
    axes[0].plot(threshold_list, num_pos[cs_ind], label=combined_stimulus_name, color='.1', marker=stimulus2marker[combined_stimulus_name], markersize=scatter_size_dict[stimulus2marker[combined_stimulus_name]], alpha=1., markerfacecolor='w')
    axes[1].plot(threshold_list, num_neg[cs_ind], label=combined_stimulus_name, color='.1', marker=stimulus2marker[combined_stimulus_name], markersize=scatter_size_dict[stimulus2marker[combined_stimulus_name]], alpha=1., markerfacecolor='w')
  fontsize = 20
  for ax in axes:
    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles, [sn.replace('\n', ' ') for sn in stimulus_types], title='', loc='lower left', fontsize=7, frameon=False)#, bbox_to_anchor=(.9, 1.)
    for axis in ['bottom', 'left']:
      ax.spines[axis].set_linewidth(1.5)
      ax.spines[axis].set_color('k')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=1.5)
    ax.set_xlabel('Threshold on Z score', fontsize=25)
  # axes[0].set_ylabel('Number of significant motif patterns', fontsize=fontsize)
  ax.set_xticks([0.1,1,10])
  ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos: '$-\\mathdefault{10^{'+f'{int(np.log10(x))}'+'}}$')) # set log scale for only negative numbers
  # ax.set_xticks(threshold_list)
  # ax.set_xticklabels(x)
  # ax.set_xticks(threshold_list, labels=-threshold_list)
  # xticks = ax.get_xticks() # .astype(float)
  # print(xticks)
  # axes[1].set_xticks(xticks)
  # axes[1].set_xticklabels(labels=-xticks[::-1], fontsize=25)
  plt.tight_layout()
  plt.savefig('./plots/signed_sig_motif_threshold.pdf', transparent=True)
  # plt.show()
################## number of significant motif VS threshold
threshold_list = np.logspace(-1, np.log10(21), 20)
plot_signed_sig_motif_threshold(mean_df4, threshold_list)
#%%
# Figure 4E
################## regional distribution of significant motifs 030T+,120D+,120U+,120C+,210+,300+
sig_motif_types = ['030T+++', '120D++++', '120U++++', '120C++++', '210+++++', '300++++++']
region_count_dict = get_motif_region_census(G_ccg_dict, area_dict, sig_motif_types)
# #%%
# ################## plot regional distribution of significant motifs
# plot_sig_motif_region(region_count_dict, sig_motif_types)
#%%
# Figure 4E
################## plot errorplot for motif region
def plot_motif_region_error(whole_df, region_count_dict, signed_motif_types, mtype='all_V1'):
  fig, ax = plt.subplots(1,1, sharex=True, sharey=True, figsize=(6, 4))
  df = pd.DataFrame()
  palette = [[plt.cm.tab20b(i) for i in range(20)][i] for i in [8,16,18,19]] + [[plt.cm.tab20c(i) for i in range(20)][i] for i in [4,16]]
  for mt_ind, signed_motif_type in enumerate(signed_motif_types):
    print(signed_motif_type)
    for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
      for col in combined_stimuli[cs_ind]:
        for row_ind, row in enumerate(session2keep):
          region_com = {}
          VISp_data, rest_data = [], []
          region_count = region_count_dict[row][col]
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
          if (summ >= 5) and (whole_df[(whole_df.session==row)&(whole_df['signed motif type']==signed_motif_type)&(whole_df.stimulus==col)]['intensity z score'].item() > 1.96): # othewise flashes will disappear
            VISp_data = [sum(VISp_data)/summ]
            rest_data = [sum(rest_data)/summ]
            # VISp_data = [d/summ for d in VISp_data]
            # rest_data = [d/summ for d in rest_data]
            df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(VISp_data)[:,None], np.array([signed_motif_type] * len(VISp_data))[:,None], np.array([combined_stimulus_name] * len(VISp_data))[:,None]), 1), columns=['probability', 'type', 'stimulus'])], ignore_index=True)
  for mt_ind, signed_motif_type in enumerate(signed_motif_types):
    for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
      if not df[(df.type==signed_motif_type) & (df.stimulus==combined_stimulus_name)].shape[0]:
        df = pd.concat([df, pd.DataFrame(np.array([[0, signed_motif_type, combined_stimulus_name]]), columns=['probability', 'type', 'stimulus'])], ignore_index=True)
  df['probability'] = pd.to_numeric(df['probability'])
  for mt_ind, signed_motif_type in enumerate(signed_motif_types):
    data = df[df.type==signed_motif_type].groupby('stimulus')
    x = np.arange(len(combined_stimulus_names))
    # y = data.mean().loc[combined_stimulus_names].values.flatten()
    # err = data.std().loc[combined_stimulus_names].values.flatten()
    y = data.mean(numeric_only=True).loc[combined_stimulus_names].values.flatten()
    err = data.std(numeric_only=True).loc[combined_stimulus_names].values.flatten()
    # err = 1.96 * data.std(numeric_only=True).loc[combined_stimulus_names].values.flatten() / data.size().loc[combined_stimulus_names].pow(1./2).values.flatten() # 95% confidence interval
    # err = stats.t.ppf((1 + 0.95) / 2., data.size().loc[combined_stimulus_names]-1) * data.sem().loc[combined_stimulus_names].values.flatten()
    for ind, (xi, yi, erri) in enumerate(zip(x, y, err)):
      if yi:
        # ax.errorbar(xi + .13 * mt_ind, yi, yerr=erri, fmt=marker_list[ind], ms=markersize_list[ind], linewidth=2.,color=palette[mt_ind])
        ax.errorbar(xi + .13 * mt_ind, yi, yerr=erri, fmt=' ', linewidth=2.,color=palette[mt_ind], zorder=1)
        ax.scatter(xi + .13 * mt_ind, yi, marker=stimulus2marker[combined_stimulus_names[ind]], s=10*error_size_dict[stimulus2marker[combined_stimulus_names[ind]]], linewidth=1.,color=palette[mt_ind], zorder=2)
  
  ax.set(xlabel=None)
  ax.xaxis.set_tick_params(length=0)
  # ax.set_xlim(-.8, len(combined_stimulus_names)-.2)
  # ax.invert_yaxis()
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
  # ylabel = 'Fraction of three V1 neurons' if mtype=='all_V1' else: 'Fraction of at least one V1 neuron'
  # ax.set_ylabel('Fraction of {} V1 neurons'.format(mtype.replace('_V1', '')), fontsize=30)
  plt.tight_layout(rect=[.02, -.03, 1, 1])
  plt.savefig('./plots/error_motif_region_{}.pdf'.format(mtype), transparent=True)
  # plt.show()

plot_motif_region_error(whole_df4, region_count_dict, sig_motif_types, mtype='all_V1')
plot_motif_region_error(whole_df4, region_count_dict, sig_motif_types, mtype='one_V1')
#%%
################## community distribution of significant motifs
def get_motif_comm_census(df, G_dict, area_dict, signed_motif_types, comms_dict, max_neg_reso):
  rows, cols = get_rowcol(G_dict)
  comm_count_dict = {}
  for row_ind, row in enumerate(rows):
    print(row)
    node_area = area_dict[row]
    comm_count_dict[row] = {}
    for col_ind, col in enumerate(cols):
      # print(col)
      comm_count_dict[row][col] = {}
      G = G_dict[row][col]
      nx.set_node_attributes(G, active_area_dict[row], "area")

      for run in range(max_neg_reso.shape[-1]):
        max_reso = max_neg_reso[row_ind, col_ind, run]
        comms_list = comms_dict[row][col][max_reso]
        comms = comms_list[run]

        node_to_community = comm2partition(comms)
        comm_area = [[active_area_dict[row][node] for node in comm] for comm in comms]
        motifs_by_type = find_triads(G) # faster
        for signed_motif_type in signed_motif_types:
          # if abs(df[(df['session']==row) & (df['stimulus']==col) & (df['signed motif type']==signed_motif_type)]['intensity z score'].tolist()[0]) > 1.96: # 95% confidence level
            motif_type = signed_motif_type.replace('+', '').replace('-', '')
            motifs = motifs_by_type[motif_type]
            for motif in motifs:
              smotif_type = motif_type + get_motif_sign(motif, motif_type, weight='confidence')
              if smotif_type == signed_motif_type:
                motif_comm_type = 'different_comm'
                motif_comm = [node_to_community[node] for node in motif.nodes()]

                # region = get_motif_region(motif, node_area, motif_type)
                # motif_comm_type = 'no_V1_comm'
                # motif_comm = [node_to_community[node] for node in motif.nodes()]
                # if ('VISp' in comm_area[motif_comm[0]]) and ('VISp' in comm_area[motif_comm[1]]) and ('VISp' in comm_area[motif_comm[2]]):
                #   motif_comm_type = 'V1_comm' # all 3 from communities with at least one V1 neuron
                
                if len(set(motif_comm)) == 1:
                  motif_comm_type = 'same_comm' # within same community and at least one V1 neuron
                comm_count_dict[row][col][smotif_type+motif_comm_type] = comm_count_dict[row][col].get(smotif_type+motif_comm_type, 0) + 1
  return comm_count_dict

sig_motif_types = ['030T+++', '120D++++', '120U++++', '120C++++', '210+++++', '300++++++']
comm_count_dict = get_motif_comm_census(whole_df4, G_ccg_dict, area_dict, sig_motif_types, comms_dict, max_reso_subs)
#%%
def adjust_box_widths(g, fac):
  # iterating through Axes instances
  for ax in g.axes:
    # iterating through axes artists:
    for c in ax.get_children():
      # searching for PathPatches
      if isinstance(c, PathPatch):
        # getting current width of box:
        p = c.get_path()
        verts = p.vertices
        verts_sub = verts[:-1]
        xmin = np.min(verts_sub[:, 0])
        xmax = np.max(verts_sub[:, 0])
        xmid = 0.5*(xmin+xmax)
        xhalf = 0.5*(xmax - xmin)
        # setting new width of box
        xmin_new = xmid-fac*xhalf
        xmax_new = xmid+fac*xhalf
        verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
        verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new
        # setting new width of median line
        for l in ax.lines:
          if np.all(l.get_xdata() == [xmin, xmax]):
            l.set_xdata([xmin_new, xmax_new])

def plot_motif_comm_box(comm_count_dict, signed_motif_types):
  rows, cols = get_rowcol(comm_count_dict)
  fig, ax = plt.subplots(1,1, sharex=True, sharey=True, figsize=(3*len(combined_stimulus_names), 6))
  df = pd.DataFrame()
  palette = [[plt.cm.tab20b(i) for i in range(20)][i] for i in [8,16,18,19]] + [[plt.cm.tab20c(i) for i in range(20)][i] for i in [4,16]]
  for mt_ind, signed_motif_type in enumerate(signed_motif_types):
    print(signed_motif_type)
    for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
      for col in combined_stimuli[cs_ind]:
        for row_ind, row in enumerate(rows):
          region_com = {}
          VISp_data, rest_data = [], []
          comm_count = comm_count_dict[row][col]
          for k in comm_count:
            if signed_motif_type in k:
              rs = k.replace(signed_motif_type, '')
              region_com[rs] = region_com.get(rs, 0) + comm_count[k]
          VISp_data.append(region_com.get('same_comm', 0))
          rest_data.append(region_com.get('different_comm', 0))
          # VISp_data.append(region_com.get('V1_comm', 0))
          # rest_data.append(region_com.get('no_V1_comm', 0))
            # if sum(region_com.values()):
            #   VISp_data.append(safe_division(region_com.get('VISp_VISp_VISp', 0), sum(region_com.values())))
            #   rest_data.append(safe_division(sum([region_com[k] for k in region_com if k!= 'VISp_VISp_VISp']), sum(region_com.values())))
          # if (col == 'natural_movie_one') and (signed_motif_type=='300++++++'):
          #   print(VISp_data)
          #   print(rest_data)
          summ = sum(VISp_data) + sum(rest_data)
          if summ >= 100: # othewise flashes will disappear
            VISp_data = [sum(VISp_data)/summ]
            rest_data = [sum(rest_data)/summ]
            # VISp_data = [d/summ for d in VISp_data]
            # rest_data = [d/summ for d in rest_data]
            df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(VISp_data)[:,None], np.array([signed_motif_type] * len(VISp_data))[:,None], np.array([combined_stimulus_name] * len(VISp_data))[:,None]), 1), columns=['probability', 'type', 'stimulus'])], ignore_index=True)
            df['probability'] = pd.to_numeric(df['probability'])
  # ax.set_title(signed_motif_type, fontsize=30, rotation=0)
  # ax = sns.violinplot(x='stimulus', y='number of connections', hue="type", data=df, palette="muted", split=False)
  medianprops = dict(linestyle='-', linewidth=4)
  boxplot = sns.boxplot(x='stimulus', y='probability', hue="type", data=df, palette=palette, ax=ax, medianprops=medianprops)

  adjust_box_widths(fig, 0.7)
  boxplot.set(xlabel=None)
  ax.yaxis.set_tick_params(labelsize=30)
  # plt.setp(ax.get_legend().get_texts(), fontsize='25') # for legend text
  # plt.setp(ax.get_legend().get_title(), fontsize='0') # for legend title
  handles, labels = ax.get_legend_handles_labels()
  # ax.legend(handles, ['' for _ in range(len(handles))], title='', loc='lower left', fontsize=22, frameon=False)#, bbox_to_anchor=(.9, 1.)
  ax.legend(handles, ['' for _ in range(len(handles))], title='', loc='upper center',
   bbox_to_anchor=(0.5, 1.2), ncol=6, columnspacing=4., fontsize=22, frameon=False)
  plt.xticks(range(len(combined_stimulus_names)), combined_stimulus_names, fontsize=30)
  ax.xaxis.set_tick_params(length=0)
  box_patches = [patch for patch in ax.patches if type(patch) == mpl.patches.PathPatch]
  if len(box_patches) == 0:  # in matplotlib older than 3.5, the boxes are stored in ax.artists
    box_patches = ax.artists
  num_patches = len(box_patches)
  lines_per_boxplot = len(ax.lines) // num_patches
  for i, patch in enumerate(box_patches):
    # Set the linecolor on the patch to the facecolor, and set the facecolor to None
    col = patch.get_facecolor()
    patch.set_edgecolor(col)
    patch.set_facecolor('None')
    patch.set_linewidth(4)
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
    legpatch.set_linewidth(3.5)

  for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2.5)
    ax.spines[axis].set_color('k')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)                                                                                                                                           
  ax.tick_params(width=2.5)
  # ax.set_ylabel('Fraction of all V1 comms', fontsize=30)
  ax.set_ylabel('Fraction of same community', fontsize=30)
  plt.tight_layout()
  plt.savefig('./plots/box_motif_same_comm.pdf', transparent=True)
  # plt.savefig('./plots/box_motif_region_all_V1comm.pdf', transparent=True)
  # plt.savefig('./plots/box_motif_region_comm.pdf', transparent=True)
  # plt.show()

plot_motif_comm_box(comm_count_dict, sig_motif_types)
#%%
def plot_motif_region_bar(region_count_dict, signed_motif_types):
  rows, cols = get_rowcol(region_count_dict)
  fig, ax = plt.subplots(1,1, sharex=True, sharey=True, figsize=(4*len(cols), 6))
  df = pd.DataFrame()
  palette = [[plt.cm.tab20b(i) for i in range(20)][i] for i in [8,16,18,19]] + [[plt.cm.tab20c(i) for i in range(20)][i] for i in [4,16]]
  for mt_ind, signed_motif_type in enumerate(signed_motif_types):
    print(signed_motif_type)
    for col_ind, col in enumerate(cols):
      region_com = {}
      VISp_data, rest_data = [], []
      for row_ind, row in enumerate(rows):
        # region_com = {}
        region_count = region_count_dict[row][col]
        for k in region_count:
          if signed_motif_type in k:
            rs = k.replace(signed_motif_type, '')
            region_com[rs] = region_com.get(rs, 0) + region_count[k]
      VISp_data.append(region_com.get('VISp_VISp_VISp', 0))
      rest_data.append(sum([region_com[k] for k in region_com if k!= 'VISp_VISp_VISp']))
        # if sum(region_com.values()):
        #   VISp_data.append(safe_division(region_com.get('VISp_VISp_VISp', 0), sum(region_com.values())))
        #   rest_data.append(safe_division(sum([region_com[k] for k in region_com if k!= 'VISp_VISp_VISp']), sum(region_com.values())))
      # if (col == 'natural_movie_one') and (signed_motif_type=='300++++++'):
      #   print(VISp_data)
      #   print(rest_data)
      summ = sum(VISp_data) + sum(rest_data)
      if summ <= 3: # othewise flashes will disappear
        VISp_data, rest_data = [0], [0]
      else: # sum across all mice
        VISp_data = [sum(VISp_data)/summ]
        rest_data = [sum(rest_data)/summ]
        # VISp_data = [d/summ for d in VISp_data]
        # rest_data = [d/summ for d in rest_data]
      df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(VISp_data)[:,None], np.array([signed_motif_type] * len(VISp_data))[:,None], np.array([col] * len(VISp_data))[:,None]), 1), columns=['probability', 'type', 'stimulus'])], ignore_index=True)
      df['probability'] = pd.to_numeric(df['probability'])
  # ax.set_title(signed_motif_type, fontsize=30, rotation=0)
  # ax = sns.violinplot(x='stimulus', y='number of connections', hue="type", data=df, palette="muted", split=False)
  barplot = sns.barplot(x='stimulus', y='probability', hue="type", data=df, palette=palette, ax=ax, alpha=.4)
  barplot.set(xlabel=None)
  ax.yaxis.set_tick_params(labelsize=30)
  plt.setp(ax.get_legend().get_texts(), fontsize='25') # for legend text
  plt.setp(ax.get_legend().get_title(), fontsize='0') # for legend title
  plt.xticks(range(len(stimulus_labels)), stimulus_labels, fontsize=30)
  for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(1.5)
    ax.spines[axis].set_color('0.2')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.tick_params(width=1.5)
  ax.set_ylabel('Fraction of all V1 neurons', fontsize=30)
  plt.tight_layout()
  plt.savefig('./plots/bar_motif_region.pdf', transparent=True)

plot_motif_region_bar(region_count_dict, sig_motif_types)
# %%
def adjust_box_widths(g, fac):
  # iterating through Axes instances
  for ax in g.axes:
    # iterating through axes artists:
    for c in ax.get_children():
      # searching for PathPatches
      if isinstance(c, PathPatch):
        # getting current width of box:
        p = c.get_path()
        verts = p.vertices
        verts_sub = verts[:-1]
        xmin = np.min(verts_sub[:, 0])
        xmax = np.max(verts_sub[:, 0])
        xmid = 0.5*(xmin+xmax)
        xhalf = 0.5*(xmax - xmin)
        # setting new width of box
        xmin_new = xmid-fac*xhalf
        xmax_new = xmid+fac*xhalf
        verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
        verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new
        # setting new width of median line
        for l in ax.lines:
          if np.all(l.get_xdata() == [xmin, xmax]):
            l.set_xdata([xmin_new, xmax_new])

def plot_motif_region_box(region_count_dict, signed_motif_types, mtype='all_V1'):
  rows, cols = get_rowcol(region_count_dict)
  fig, ax = plt.subplots(1,1, sharex=True, sharey=True, figsize=(3*len(combined_stimulus_names), 6))
  df = pd.DataFrame()
  palette = [[plt.cm.tab20b(i) for i in range(20)][i] for i in [8,16,18,19]] + [[plt.cm.tab20c(i) for i in range(20)][i] for i in [4,16]]
  for mt_ind, signed_motif_type in enumerate(signed_motif_types):
    print(signed_motif_type)
    for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
      for col in combined_stimuli[cs_ind]:
        for row_ind, row in enumerate(session2keep):
          region_com = {}
          VISp_data, rest_data = [], []
          region_count = region_count_dict[row][col]
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
            # if sum(region_com.values()):
            #   VISp_data.append(safe_division(region_com.get('VISp_VISp_VISp', 0), sum(region_com.values())))
            #   rest_data.append(safe_division(sum([region_com[k] for k in region_com if k!= 'VISp_VISp_VISp']), sum(region_com.values())))
          # if (col == 'natural_movie_one') and (signed_motif_type=='300++++++'):
          #   print(VISp_data)
          #   print(rest_data)
          summ = sum(VISp_data) + sum(rest_data)

          if summ >= 5: # othewise flashes will disappear
            VISp_data = [sum(VISp_data)/summ]
            rest_data = [sum(rest_data)/summ]

            # VISp_data = [d/summ for d in VISp_data]
            # rest_data = [d/summ for d in rest_data]
            df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(VISp_data)[:,None], np.array([signed_motif_type] * len(VISp_data))[:,None], np.array([combined_stimulus_name] * len(VISp_data))[:,None]), 1), columns=['probability', 'type', 'stimulus'])], ignore_index=True)
            df['probability'] = pd.to_numeric(df['probability'])
  # ax.set_title(signed_motif_type, fontsize=30, rotation=0)
  # ax = sns.violinplot(x='stimulus', y='number of connections', hue="type", data=df, palette="muted", split=False)
  medianprops = dict(linestyle='-', linewidth=4)
  boxplot = sns.boxplot(x='stimulus', y='probability', hue="type", data=df, palette=palette, ax=ax, medianprops=medianprops)

  adjust_box_widths(fig, 0.7)
  boxplot.set(xlabel=None)
  ax.yaxis.set_tick_params(labelsize=30)
  # plt.setp(ax.get_legend().get_texts(), fontsize='25') # for legend text
  # plt.setp(ax.get_legend().get_title(), fontsize='0') # for legend title
  handles, labels = ax.get_legend_handles_labels()
  # ax.legend(handles, ['' for _ in range(len(handles))], title='', loc='lower left', fontsize=22, frameon=False)#, bbox_to_anchor=(.9, 1.)
  ax.legend(handles, ['' for _ in range(len(handles))], title='', loc='upper center',
   bbox_to_anchor=(0.5, 1.2), ncol=6, columnspacing=4., fontsize=22, frameon=False)
  plt.xticks(range(len(combined_stimulus_names)), combined_stimulus_names, fontsize=30)
  ax.xaxis.set_tick_params(length=0)
  box_patches = [patch for patch in ax.patches if type(patch) == mpl.patches.PathPatch]
  if len(box_patches) == 0:  # in matplotlib older than 3.5, the boxes are stored in ax.artists
    box_patches = ax.artists
  num_patches = len(box_patches)
  lines_per_boxplot = len(ax.lines) // num_patches
  for i, patch in enumerate(box_patches):
    # Set the linecolor on the patch to the facecolor, and set the facecolor to None
    col = patch.get_facecolor()
    patch.set_edgecolor(col)
    patch.set_facecolor('None')
    patch.set_linewidth(4)
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
    legpatch.set_linewidth(3.5)

  for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2.5)
    ax.spines[axis].set_color('k')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.tick_params(width=2.5)
  ax.set_ylabel('Fraction of {} V1 neurons'.format(mtype.replace('_V1', '')), fontsize=30)
  plt.tight_layout()
  plt.savefig('./plots/box_motif_region_{}.pdf'.format(mtype), transparent=True)
  # plt.show()

plot_motif_region_box(region_count_dict, sig_motif_types, mtype='all_V1')
plot_motif_region_box(region_count_dict, sig_motif_types, mtype='one_V1')
# %%
def adjust_box_widths(g, fac):
  # iterating through Axes instances
  for ax in g.axes:
    # iterating through axes artists:
    for c in ax.get_children():
      # searching for PathPatches
      if isinstance(c, mpl.patches.PathPatch):
        # getting current width of box:
        p = c.get_path()
        verts = p.vertices
        verts_sub = verts[:-1]
        xmin = np.min(verts_sub[:, 0])
        xmax = np.max(verts_sub[:, 0])
        xmid = 0.5*(xmin+xmax)
        xhalf = 0.5*(xmax - xmin)
        # setting new width of box
        xmin_new = xmid-fac*xhalf
        xmax_new = xmid+fac*xhalf
        verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
        verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new
        # setting new width of median line
        for l in ax.lines:
          if np.all(l.get_xdata() == [xmin, xmax]):
            l.set_xdata([xmin_new, xmax_new])

def plot_motif_role(region_count_dict, signed_motif_types, ntype='driver'):
  rows, cols = get_rowcol(region_count_dict)
  fig, ax = plt.subplots(1,1, sharex=True, sharey=True, figsize=(3*len(combined_stimulus_names), 6))
  df = pd.DataFrame()
  palette = [[plt.cm.tab20b(i) for i in range(20)][i] for i in [8,16,18,19]] + [[plt.cm.tab20c(i) for i in range(20)][i] for i in [4,16]]
  for mt_ind, signed_motif_type in enumerate(signed_motif_types):
    print(signed_motif_type)
    for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
      for col in combined_stimuli[cs_ind]:
        for row_ind, row in enumerate(rows):
          region_com = {}
          VISp_data, rest_data = [], []
          region_count = region_count_dict[row][col]
          for k in region_count:
            if signed_motif_type in k:
              rs = k.replace(signed_motif_type, '')
              region_com[rs] = region_com.get(rs, 0) + region_count[k]
          # print(region_com)
          if ntype == 'driver':
            for motif_area in region_com:
              if signed_motif_type.replace('+', '').replace('-', '') in ['030T', '120D', '120C']:
                if motif_area.startswith('VISp'):
                  VISp_data.append(region_com[motif_area])
                else:
                  rest_data.append(region_com[motif_area])
              elif signed_motif_type.replace('+', '').replace('-', '') in ['120U']:
                if (motif_area.split('_')[0] == 'VISp') or (motif_area.split('_')[1] == 'VISp'):
                  VISp_data.append(region_com[motif_area])
                else:
                  rest_data.append(region_com[motif_area])
              elif signed_motif_type.replace('+', '').replace('-', '') in ['210']:
                if (motif_area.split('_')[0] == 'VISp') or (motif_area.split('_')[2] == 'VISp'):
                  VISp_data.append(region_com[motif_area])
                else:
                  rest_data.append(region_com[motif_area])
              elif signed_motif_type.replace('+', '').replace('-', '') in ['300']:
                if 'VISp' in motif_area:
                  VISp_data.append(region_com[motif_area])
                else:
                  rest_data.append(region_com[motif_area])
          elif ntype == 'driven':
            for motif_area in region_com:
              if signed_motif_type.replace('+', '').replace('-', '') in ['030T', '120D', '120C']:
                if motif_area.endswith('VISp'):
                  VISp_data.append(region_com[motif_area])
                else:
                  rest_data.append(region_com[motif_area])
              elif signed_motif_type.replace('+', '').replace('-', '') in ['120U']:
                if motif_area.split('_')[2] == 'VISp':
                  VISp_data.append(region_com[motif_area])
                else:
                  rest_data.append(region_com[motif_area])
              elif signed_motif_type.replace('+', '').replace('-', '') in ['210']:
                if motif_area.split('_')[1] == 'VISp':
                  VISp_data.append(region_com[motif_area])
                else:
                  rest_data.append(region_com[motif_area])
              elif signed_motif_type.replace('+', '').replace('-', '') in ['300']:
                if 'VISp' in motif_area:
                  VISp_data.append(region_com[motif_area])
                else:
                  rest_data.append(region_com[motif_area])
          summ = sum(VISp_data) + sum(rest_data)
          if summ >= 5: # othewise flashes will disappear
            VISp_data = [sum(VISp_data)/summ]
            rest_data = [sum(rest_data)/summ]
            # VISp_data = [d/summ for d in VISp_data]
            # rest_data = [d/summ for d in rest_data]
            df = pd.concat([df, pd.DataFrame(np.concatenate((np.array(VISp_data)[:,None], np.array([signed_motif_type] * len(VISp_data))[:,None], np.array([combined_stimulus_name] * len(VISp_data))[:,None]), 1), columns=['probability', 'type', 'stimulus'])], ignore_index=True)
            df['probability'] = pd.to_numeric(df['probability'])
  # ax.set_title(signed_motif_type, fontsize=30, rotation=0)
  # ax = sns.violinplot(x='stimulus', y='number of connections', hue="type", data=df, palette="muted", split=False)
  medianprops = dict(linestyle='-', linewidth=4)
  boxplot = sns.boxplot(x='stimulus', y='probability', hue="type", data=df, palette=palette, ax=ax, medianprops=medianprops)

  adjust_box_widths(fig, 0.7)
  boxplot.set(xlabel=None)
  ax.yaxis.set_tick_params(labelsize=30)
  # plt.setp(ax.get_legend().get_texts(), fontsize='25') # for legend text
  # plt.setp(ax.get_legend().get_title(), fontsize='0') # for legend title
  handles, labels = ax.get_legend_handles_labels()
  # ax.legend(handles, ['' for _ in range(len(handles))], title='', loc='lower left', fontsize=22, frameon=False)#, bbox_to_anchor=(.9, 1.)
  ax.legend(handles, ['' for _ in range(len(handles))], title='', loc='upper center',
   bbox_to_anchor=(0.5, 1.2), ncol=6, columnspacing=4., fontsize=22, frameon=False)
  plt.xticks(range(len(combined_stimulus_names)), combined_stimulus_names, fontsize=30)
  ax.xaxis.set_tick_params(length=0)
  box_patches = [patch for patch in ax.patches if type(patch) == mpl.patches.PathPatch]
  if len(box_patches) == 0:  # in matplotlib older than 3.5, the boxes are stored in ax.artists
    box_patches = ax.artists
  num_patches = len(box_patches)
  lines_per_boxplot = len(ax.lines) // num_patches
  for i, patch in enumerate(box_patches):
    # Set the linecolor on the patch to the facecolor, and set the facecolor to None
    col = patch.get_facecolor()
    patch.set_edgecolor(col)
    patch.set_facecolor('None')
    patch.set_linewidth(4)
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
    legpatch.set_linewidth(3.5)

  for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2.5)
    ax.spines[axis].set_color('k')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.tick_params(width=2.5)
  ax.set_ylabel('Fraction of {} V1 neurons'.format(ntype), fontsize=30)
  plt.tight_layout()
  plt.savefig('./plots/box_motif_{}.pdf'.format(ntype), transparent=True)
  # plt.show()

plot_motif_role(region_count_dict, sig_motif_types, 'driver')
plot_motif_role(region_count_dict, sig_motif_types, 'driven')
#%%
###################### distribution of delay on each edge
def get_motif_data(motif, motif_type, weight='offset'):
  edges = list(motif.edges())
  nodes = [node for sub in edges for node in sub]
  triplets = list(set(nodes))
  if motif_type == '021D':
    node_P = most_common([i for i,j in edges])
    node_X, node_O = [j for i,j in edges]
    edge_order = [(node_P, node_X), (node_P, node_O)]
  elif motif_type == '021U':
    node_P = most_common([j for i,j in edges])
    node_X, node_O = [i for i,j in edges]
    edge_order = [(node_X, node_P), (node_O, node_P)]
  elif motif_type == '021C':
    node_X = most_common(nodes)
    triplets.remove(node_X)
    if (triplets[0], node_X) in edges:
      node_P, node_O = triplets
    else:
      node_O, node_P = triplets
    edge_order = [(node_P, node_X), (node_X, node_O)]
  elif motif_type == '111D':
    node_X = most_common([j for i,j in edges])
    node_P = [j for i,j in edges if i == node_X][0]
    triplets.remove(node_X)
    triplets.remove(node_P)
    node_O = triplets[0]
    edge_order = [(node_P, node_X), (node_X, node_P), (node_O, node_X)]
  elif motif_type == '111U':
    node_X = most_common([i for i,j in edges])
    node_P = [i for i,j in edges if j == node_X][0]
    triplets.remove(node_X)
    triplets.remove(node_P)
    node_O = triplets[0]
    edge_order = [(node_P, node_X), (node_X, node_P), (node_X, node_O)]
  elif motif_type == '030T':
    node_P = most_common([i for i,j in edges])
    node_X = most_common([j for i,j in edges])
    triplets.remove(node_P)
    triplets.remove(node_X)
    node_O = triplets[0]
    edge_order = [(node_P, node_X), (node_P, node_O), (node_O, node_X)]
  elif motif_type == '030C':
    es = edges.copy()
    np.random.shuffle(es)
    node_P, node_O = es[0]
    triplets.remove(node_P)
    triplets.remove(node_O)
    node_X = triplets[0]
    edge_order = [(node_P, node_O), (node_O, node_X), (node_X, node_P)]
  elif motif_type == '201':
    node_P = most_common([i for i,j in edges])
    triplets.remove(node_P)
    np.random.shuffle(triplets)
    node_X, node_O = triplets
    edge_order = [(node_P, node_O), (node_O, node_P), (node_P, node_X), (node_X, node_P)]
  elif motif_type == '120D' or motif_type == '120U':
    if motif_type == '120D':
      node_X = most_common([i for i,j in edges])
    else:
      node_X = most_common([j for i,j in edges])
    triplets.remove(node_X)
    np.random.shuffle(triplets)
    node_P, node_O = triplets
    if motif_type == '120D':
      edge_order = [(node_X, node_P), (node_P, node_O), (node_X, node_O), (node_O, node_P)]
    else:
      edge_order = [(node_P, node_X), (node_P, node_O), (node_O, node_X), (node_O, node_P)]
  elif motif_type == '120C':
    node_P = most_common([i for i,j in edges])
    node_X = most_common([j for i,j in edges])
    triplets.remove(node_P)
    triplets.remove(node_X)
    node_O = triplets[0]
    edge_order = [(node_P, node_X), (node_X, node_P), (node_P, node_O), (node_O, node_X)]
  elif motif_type == '210':
    node_O = most_common([node for sub in edges for node in sub])
    triplets.remove(node_O)
    if tuple(triplets) in edges:
      node_P, node_X = triplets
    else:
      node_X, node_P = triplets
    edge_order = [(node_P, node_O), (node_O, node_P), (node_O, node_X), (node_X, node_O), (node_P, node_X)]
  elif motif_type == '300':
    np.random.shuffle(triplets)
    node_P, node_X, node_O = triplets
    edge_order = [(node_X, node_P), (node_P, node_X), (node_X, node_O), (node_O, node_X), (node_P, node_O), (node_O, node_P)]
  motif_data = {edge:motif[edge[0]][edge[1]][weight] for edge in edges}
  data = [motif_data[edge] for edge in edge_order]
  return data

def get_motif_delay(G_dict, area_dict, signed_motif_types, name='offset'):
  rows, cols = get_rowcol(G_dict)
  delay_dict = {}
  for row_ind, row in enumerate(rows):
    print(row)
    node_area = area_dict[row]
    delay_dict[row] = {}
    for col_ind, col in enumerate(cols):
      print(col)
      delay_dict[row][col] = {}
      G = G_dict[row][col]
      motifs_by_type = find_triads(G) # faster
      for signed_motif_type in signed_motif_types:
        motif_type = signed_motif_type.replace('+', '').replace('-', '')
        motifs = motifs_by_type[motif_type]
        for motif in motifs:
          smotif_type = motif_type + get_motif_sign(motif, motif_type, weight='sign')
          if smotif_type == signed_motif_type:
            delay = get_motif_data(motif, motif_type, 'offset')
            # print(smotif_type, region)
            if smotif_type not in delay_dict[row][col]:
              delay_dict[row][col][smotif_type] = [[] for _ in range(len(delay))]
            for i in range(len(delay)):
              delay_dict[row][col][smotif_type][i].append(delay[i])
  return delay_dict

name = 'offset'
name = 'duration'
name = 'delay'
sig_motif_types = ['030T+++', '120D++++', '120U++++', '120C++++', '210+++++', '300++++++']
delay_dict = get_motif_delay(S_ccg_dict, area_dict, sig_motif_types, name=name)
#%%
def plot_motif_data_box(data_dict, signed_motif_types, name='offset'):
  rows, cols = get_rowcol(data_dict)
  fig, axes = plt.subplots(len(signed_motif_types),1, sharex=True, sharey=True, figsize=(2*len(cols), 6*len(signed_motif_types)))
  df = pd.DataFrame()
  palette = [[plt.cm.tab20b(i) for i in range(20)][i] for i in [8,16,18,19]] + [[plt.cm.tab20c(i) for i in range(20)][i] for i in [4,16]]
  for mt_ind, signed_motif_type in enumerate(signed_motif_types):
    print(signed_motif_type)
    ax = axes[mt_ind]
    for col_ind, col in enumerate(cols):
      col_data = None
      for row_ind, row in enumerate(rows):
        if signed_motif_type in data_dict[row][col]:
          data = data_dict[row][col][signed_motif_type]
          col_data = np.concatenate((data, np.array(data)), 1) if col_data is not None else np.array(data)
      if col_data is not None:
        for i in range(col_data.shape[0]):
          df = pd.concat([df, pd.DataFrame(np.concatenate((col_data[i][:,None], np.array(['edge {}'.format(i)] * col_data.shape[1])[:,None], np.array([col] * col_data.shape[1])[:,None]), 1), columns=['data', 'edge', 'stimulus'])], ignore_index=True)
    df['data'] = pd.to_numeric(df['data'])
    # ax.set_title(signed_motif_type, fontsize=30, rotation=0)
    # ax = sns.violinplot(x='stimulus', y='number of connections', hue="type", data=df, palette="muted", split=False)
    boxplot = sns.boxplot(x='stimulus', y='data', hue="edge", data=df, palette=palette, ax=ax, showfliers=False)
    boxplot.set(xlabel=None)
    ax.yaxis.set_tick_params(labelsize=30)
    plt.setp(ax.get_legend().get_texts(), fontsize='25') # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize='0') # for legend title
    plt.xticks(range(len(stimulus_labels)), stimulus_labels, fontsize=30)

    box_patches = [patch for patch in ax.patches if type(patch) == mpl.patches.PathPatch]
    if len(box_patches) == 0:  # in matplotlib older than 3.5, the boxes are stored in ax.artists
      box_patches = ax.artists
    num_patches = len(box_patches)
    lines_per_boxplot = len(ax.lines) // num_patches
    for i, patch in enumerate(box_patches):
      # Set the linecolor on the patch to the facecolor, and set the facecolor to None
      col = patch.get_facecolor()
      patch.set_edgecolor(col)
      patch.set_facecolor('None')
      patch.set_linewidth(4)
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
      legpatch.set_linewidth(3)

    for axis in ['bottom', 'left']:
      ax.spines[axis].set_linewidth(1.5)
      ax.spines[axis].set_color('0.2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=1.5)
    ax.set_ylabel('{} distribution'.format(name), fontsize=30)
  plt.tight_layout()
  plt.savefig('./plots/box_motif_{}.pdf'.format(name), transparent=True)
  # plt.show()

plot_motif_data_box(delay_dict, sig_motif_types, name=name)
# %%
###################### test colors in scatter
# stimulus_type_color[1] = '#fee391'
colors = stimulus_type_color + region_colors
labels = stimulus_types + visual_regions
plt.figure()
for i in range(10):
  plt.scatter(np.arange(i, i+2), i * np.arange(i, i+2), facecolors='none', edgecolors=colors[i], label=labels[i])
plt.legend()
plt.show()
#%%
def scatter_logpolar(ax, theta, r_, bullseye=0.3, **kwargs):
    min10 = np.log10(np.min(r_))
    max10 = np.log10(np.max(r_))
    r = np.log10(r_) - min10 + bullseye
    ax.scatter(theta, r, **kwargs)
    l = np.arange(np.floor(min10), max10)
    ax.set_rticks(l - min10 + bullseye) 
    ax.set_yticklabels(["1e%d" % x for x in l])
    ax.set_rlim(0, max10 - min10 + bullseye)
    ax.set_title('log-polar manual')
    return ax

def circular_lollipop(df, signed_motif_types):
  sorted_types = [sorted([smotif for smotif in signed_motif_types if mt in smotif]) for mt in TRIAD_NAMES]
  sorted_types = [item for sublist in sorted_types for item in sublist]
  zscores = []
  stimulus_name = 'natural_movie_three'
  data = df[df.stimulus==stimulus_name]
  data = data.groupby('signed motif type').mean()
  zscores += data.loc[sorted_types, 'intensity z score'].values.tolist()
  # zscores = []
  # for stimulus_name in stimulus_names:
  #   data = df[df.stimulus==stimulus_name]
  #   data = data.groupby('signed motif type').mean()
  #   zscores += data['intensity z score'].values.tolist()

  # Values for the x axis
  ANGLES = np.linspace(0, 2 * np.pi, len(zscores), endpoint=False)
  # Heights of the lines and y-position of the dot are given by the times.
  HEIGHTS = np.array(zscores)

  # Category values for the colors
  # CATEGORY_CODES = pd.Categorical(df_pw["category"]).codes
  PLUS = 0
  # Initialize layout in polar coordinates
  fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})

  # Set background color to white, both axis and figure.
  fig.patch.set_facecolor("white")
  ax.set_facecolor("white")
  # Use logarithmic scale for the radial axis
  # ax.set_rscale('symlog')
  # Angular axis starts at 90 degrees, not at 0
  ax.set_theta_offset(np.pi / 2)

  # Reverse the direction to go counter-clockwise.
  ax.set_theta_direction(-1)

  motif_types = TRIAD_NAMES[3:]
  COLORS = []
  for t in sorted_types:
    COLORS.append(motif_palette[motif_types.index(t.replace('+', '').replace('\N{MINUS SIGN}', ''))])

  # Add lines
  ax.vlines(ANGLES, 0 + PLUS, HEIGHTS + PLUS, color=COLORS, lw=0.9)
  # Add dots
  ax.scatter(ANGLES, HEIGHTS + PLUS, color=COLORS, s=scale_to_interval(HEIGHTS)) #
  # Start by removing spines for both axes
  ax.spines["start"].set_color("none")
  ax.spines["polar"].set_color("none")

  # Remove grid lines, ticks, and tick labels.
  ax.grid(False)
  ax.set_xticks([])
  ax.set_yticklabels([])
  # Add our custom grid lines for the radial axis.
  # These lines indicate one day, one week, one month and one year.
  HANGLES = np.linspace(0, 2 * np.pi, 200)
  ax.plot(HANGLES, np.repeat(-10 + PLUS, 200), color= GREY, lw=0.7)
  ax.plot(HANGLES, np.repeat(0 + PLUS, 200), color= GREY, lw=0.7)
  ax.plot(HANGLES, np.repeat(10 + PLUS, 200), color= GREY, lw=0.7)
  ax.text(np.pi/2, 1 + PLUS, '0', color= GREY, verticalalignment='center', fontsize=20)
  ax.text(np.pi/2, 11 + PLUS, '10', color= GREY, verticalalignment='center', fontsize=20)
  ax.text(np.pi/2, -9 + PLUS, '-10', color= GREY, verticalalignment='center', fontsize=20)
  
  # Highlight above threshold motifs
  hl_idx = HEIGHTS >= 10
  hl_label = np.array(sorted_types)[hl_idx]
  hl_x = ANGLES[hl_idx]
  hl_y = HEIGHTS[hl_idx] + 4
  hl_colors = np.array(COLORS)[hl_idx]
  for i in range(len(hl_label)):
    ax.text(
        x=hl_x[i], y=hl_y[i], s=hl_label[i], color=hl_colors[i],
        ha="center", va="center", ma="center", size=10,
        weight="bold"
    )
  plt.show()

GREY = '.6'
circular_lollipop(whole_df4, signed_motif_types4)
#%%
####################### make sure all dfs have same signed motif types
def add_missing_motif_type(df, mtype, signed_motif_types):
  if len(mtype) < len(signed_motif_types):
    mtype2add = [t for t in signed_motif_types if t not in mtype]
    for mt in mtype2add:
      mtype.append(mt)
      for session_id in session_ids:
        for stimulus_name in stimulus_names:
          df = pd.concat([df, pd.DataFrame([[mt, session_id, stimulus_name] + [0] * (df.shape[1]-3)], columns=df.columns)], ignore_index=True)
    df['intensity z score'] = pd.to_numeric(df['intensity z score'])
  return df, mtype

signed_motif_types = np.unique(signed_motif_types1+signed_motif_types2+signed_motif_types3+signed_motif_types4).tolist()
whole_df1, signed_motif_types1 = add_missing_motif_type(whole_df1, signed_motif_types1, signed_motif_types)
whole_df2, signed_motif_types2 = add_missing_motif_type(whole_df2, signed_motif_types2, signed_motif_types)
whole_df3, signed_motif_types3 = add_missing_motif_type(whole_df3, signed_motif_types3, signed_motif_types)
whole_df4, signed_motif_types4 = add_missing_motif_type(whole_df4, signed_motif_types4, signed_motif_types)
# %%
# Figure 4G
GREY = '.5'
motif_palette = [[plt.cm.tab20b(i) for i in range(20)][i] for i in [0,2,3,4,6,8,10,12,16,18,19]] + [[plt.cm.tab20c(i) for i in range(20)][i] for i in [4,16]]

def scale_to_interval(origin_x, low=1, high=100, logscale=False):
  if not logscale:
    x = np.abs(origin_x.copy())
  else:
    x = np.power(2, np.abs(origin_x.copy()))
    x = np.nan_to_num(x, neginf=0, posinf=0)
  y = ((x - x.min()) / (x.max() - x.min())) * (high - low) + low
  return y

def single_circular_lollipop(data, ind, sorted_types, COLORS, ax, lwv=.9, lw0=.7, neggrid=-5, posgrid=10, low=1, high=100, logscale=False):
  ANGLES = np.linspace(0, 2 * np.pi, len(data), endpoint=False)
  HEIGHTS = np.array(data)
  PLUS = 0
  ax.set_facecolor("white")
  ax.set_theta_offset(np.pi / 2)
  ax.set_theta_direction(-1)
  ax.vlines(ANGLES, 0 + PLUS, HEIGHTS + PLUS, color=COLORS, lw=lwv)
  ax.scatter(ANGLES, HEIGHTS + PLUS, color=COLORS, s=scale_to_interval(HEIGHTS, low=low, high=high, logscale=logscale)) #
  ax.spines["start"].set_color("none")
  ax.spines["polar"].set_color("none")
  ax.grid(False)
  ax.set_xticks([])
  ax.set_yticklabels([])
  HANGLES = np.linspace(0, 2 * np.pi, 200)
  ls = (0, (5, 5))
  ax.plot(HANGLES, np.repeat(neggrid + PLUS, 200), color= GREY, lw=lw0, linestyle=(0, (5, 1)))
  ax.plot(HANGLES, np.repeat(0 + PLUS, 200), color= GREY, lw=lw0, linestyle=ls) # needs to be denser
  ax.plot(HANGLES, np.repeat(posgrid + PLUS, 200), color= GREY, lw=lw0, linestyle=ls)
  # add text to scale circle
  # linoffset, logoffset = 5, 2.
  # if not logscale:
  #   ax.text(np.pi, linoffset + 2 + PLUS, '0', color= GREY, horizontalalignment='center', verticalalignment='center', fontsize=25)
  #   ax.text(np.pi, posgrid + linoffset + PLUS, str(posgrid), color= GREY, horizontalalignment='center', verticalalignment='center', fontsize=25)
  #   ax.text(np.pi, neggrid + linoffset + PLUS, str(neggrid), color= GREY, horizontalalignment='center', verticalalignment='center', fontsize=25)
  # else:
  #   ax.text(np.pi/2, logoffset + PLUS, '0', color= GREY, horizontalalignment='center', verticalalignment='center', fontsize=25)
  #   ax.text(np.pi/2, posgrid + logoffset + PLUS, str(round(np.power(2, posgrid))), color= GREY, horizontalalignment='center', verticalalignment='center', fontsize=25)
  #   if ind == 0:
  #     theta = np.pi
  #   else:
  #     theta = np.pi/2
  #   ax.text(theta, neggrid + logoffset - .5 + PLUS, str(round(-np.power(2, abs(neggrid)))), color= GREY, horizontalalignment='center', verticalalignment='center', fontsize=25)
  print(ax.get_ylim())
  if ind == 0:
    ax.set_ylim([-5, 11]) # increase 
  elif ind == 1:
    ax.set_ylim(top = 8.5)
  elif ind == 2:
    ax.set_ylim([-7, 20])
  elif ind == 3:
    ax.set_ylim([-7, 21])
  # Highlight above threshold motifs
  # hl_idx = HEIGHTS >= posgrid
  # hl_label = np.array(sorted_types)[hl_idx]
  # hl_x = ANGLES[hl_idx]
  # if not logscale:
  #   hl_y = HEIGHTS[hl_idx] + 4
  # else:
  #   hl_y = HEIGHTS[hl_idx] + 2
  # hl_colors = np.array(COLORS)[hl_idx]
  # for i in range(len(hl_label)):
  #   ax.text(
  #       x=hl_x[i], y=hl_y[i], s=hl_label[i], color=hl_colors[i],
  #       ha="center", va="center", ma="center", size=10,
  #       weight="bold")

def multi_circular_lollipop(df1, df2, df3, df4, stimulus_name='natural_movie_three'):
  fig, axes = plt.subplots(1, 4, figsize=(20, 7), subplot_kw={"projection": "polar"})
  fig.patch.set_facecolor("white")
  sorted_types = [sorted([smotif for smotif in df1['signed motif type'].unique() if mt in smotif]) for mt in TRIAD_NAMES]
  sorted_types = [item for sublist in sorted_types for item in sublist]
  motif_types = TRIAD_NAMES[3:]
  COLORS = []
  for t in sorted_types:
    COLORS.append(motif_palette[motif_types.index(t.replace('+', '').replace('\N{MINUS SIGN}', '').replace('-', ''))])
  dfs = [df1, df2, df3, df4]
  for df_ind, df in enumerate(dfs):
    print(df_ind)
    # i, j = df_ind // 2, df_ind % 2
    # ax = axes[i, j]
    ax = axes[df_ind]
    data = df[df.stimulus==stimulus_name]
    data = data.groupby('signed motif type').mean()
    zscores = data.loc[sorted_types, "intensity z score"].values.tolist()
    neggrid, posgrid, low, high, logscale = -5, 10, 0, 200, False
    if df_ind == 0: # manual logscale
      signs = np.array([1 if zs >= 0 else -1 for zs in zscores])
      zscores = np.log2(np.abs(zscores)) * signs
      neggrid = - np.log2(10)
      posgrid = np.log2(100)
      # low=0
      high=600
      logscale = True
    elif df_ind == 1:
      signs = np.array([1 if zs >= 0 else -1 for zs in zscores])
      zscores = np.log2(np.abs(zscores)) * signs
      neggrid = - np.log2(10)
      posgrid = np.log2(20)
      # low=0
      high=300
      logscale = True
    # print(zscores)
    single_circular_lollipop(zscores, df_ind, sorted_types, COLORS, ax, lwv=2.5, lw0=3., neggrid=neggrid, posgrid=posgrid, low=low, high=high, logscale=logscale)
    ax.set_title(model_names[df_ind], fontsize=25)
  fig.subplots_adjust(wspace=0.) #
  plt.tight_layout()
  # plt.show()
  plt.savefig('./plots/circular_lollipop_multimodel.pdf', transparent=True)

multi_circular_lollipop(whole_df1, whole_df2, whole_df3, whole_df4, stimulus_name='natural_movie_three')
# %%
# Figure S1B
def plot_directed_degree_distributions(G_dict, row_ind, sign, measure, n, weight=None, cc=False):
  rows, cols = get_rowcol(G_dict)
  row = rows[row_ind]
  fig, axes = plt.subplots(1, len(combined_stimuli), figsize=(4*len(combined_stimuli), 3))
  for cs_ind, combined_stimulus in enumerate(combined_stimuli):
    ax = axes[cs_ind]
    col = combined_stimulus[0]
    # ax.set_title(combined_stimulus_names[cs_ind].replace('\n', ' '), fontsize=25, rotation=0)
    G = G_dict[row][col].copy()
    if G.number_of_nodes() > 2 and G.number_of_edges() > 0:
      if cc:
        if nx.is_directed(G):
          Gcc = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
          G = G.subgraph(Gcc[0])
        else:
          Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
          G = G.subgraph(Gcc[0])
      in_degrees, in_degree_freq = degree_histogram_directed(G, type='in_degree', weight=weight)
      out_degrees, out_degree_freq = degree_histogram_directed(G, type='out_degree', weight=weight)
      if weight == None:
        in_degrees, in_degree_freq = in_degrees[1:], in_degree_freq[1:]
        out_degrees, out_degree_freq = out_degrees[1:], out_degree_freq[1:]
      ax.plot(in_degrees, np.array(in_degree_freq) / sum(in_degree_freq),color='orange',marker='o', markerfacecolor='white', label='in-degree', markersize=10, alpha=1.)
      ax.plot(out_degrees, np.array(out_degree_freq) / sum(out_degree_freq),color='limegreen',marker='o', markerfacecolor='white', label='out-degree', markersize=10, alpha=1.)
      # ax.legend(loc='upper right', fontsize=7)
      # xlabel = 'Weighted Degree' if weight is not None else 'Degree'
      # ax.set_xlabel(xlabel, fontsize=22)
      # ax.set_ylabel('Frequency', fontsize=22)
      ax.set_xlabel('')
      ax.set_ylabel('')
      ax.set_xscale('symlog')
      ax.set_yscale('log')
      for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(2.)
        ax.spines[axis].set_color('k')
      ax.spines['top'].set_visible(False)
      ax.spines['right'].set_visible(False)
      ax.xaxis.set_tick_params(labelsize=20)
      ax.yaxis.set_tick_params(labelsize=20)
      # if cs_ind == 0:
      #   handles, labels = ax.get_legend_handles_labels()
      #   ax.legend(handles, labels, title='', bbox_to_anchor=(.7, .8), loc='upper left', fontsize=16, frameon=False)
      
  plt.tight_layout()
  image_name = './plots/directed_degree_distribution_weighted_{}_{}_{}_{}fold.pdf'.format(row_ind, sign, measure, n) if weight is not None else './plots/directed_degree_distribution_unweighted_{}_{}_{}_{}fold.pdf'.format(row_ind, sign, measure, n)
  # plt.show()
  plt.savefig(image_name, transparent=True)
  # plt.savefig(image_name.replace('jpg', 'pdf'), transparent=True)

plot_directed_degree_distributions(G_ccg_dict, 7, 'all', measure, n, weight=None, cc=False)
# %%
def save_in_out_degree_legend():
  fig, ax = plt.subplots(1,1, figsize=(.4*(len(combined_stimulus_names)-1), .5))
  ax.plot([0,1], [0,0],color='orange',marker='o', markerfacecolor='white', label='in-degree', markersize=10, alpha=1.)
  ax.plot([0,1], [0,0],color='limegreen',marker='o', markerfacecolor='white', label='out-degree', markersize=10, alpha=1.)
  handles, labels = ax.get_legend_handles_labels()
  legend = ax.legend(handles, ['in degree', 'out degree'], title='', bbox_to_anchor=(0.1, -1.), handletextpad=0.3, loc='upper left', fontsize=25, frameon=False) # change legend order to within/cross similar module then cross random
  plt.axis('off')
  export_legend(legend, './plots/in_out_degree_legend.pdf')
  plt.tight_layout()
  # plt.savefig('./plots/stimuli_markers.pdf', transparent=True)
  plt.show()

save_in_out_degree_legend()
# %%
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
#%%
# Figure S7A
def plot_pair_distributions(G_dict, row_ind):
  rows, cols = get_rowcol(G_dict)
  row = rows[row_ind]
  fig, axes = plt.subplots(1, len(combined_stimuli), figsize=(6*len(combined_stimuli), 3))
  for cs_ind, combined_stimulus in enumerate(combined_stimuli):
    ax = axes[cs_ind]
    col = combined_stimulus[0]
    # ax.set_title(combined_stimulus_names[cs_ind].replace('\n', ' '), fontsize=25, rotation=0)
    G = G_dict[row][col].copy()
    num0, num1, num2 = 0, 0, 0
    nodes = list(G.nodes())
    for node_i, node_j in itertools.combinations(nodes, 2):
      if G.has_edge(node_i, node_j) and G.has_edge(node_j, node_i):
        num2 += 1
      elif G.has_edge(node_i, node_j) or G.has_edge(node_j, node_i):
        num1 += 1
      else:
        num0 += 1
    # print(num0, num1, num2)
    total_num = num0+num1+num2
    assert total_num == len(nodes) * (len(nodes) - 1) / 2, '{}, {}'.format(total_num, len(nodes) * (len(nodes) - 1) / 2)
    assert num1+num2*2 == G.number_of_edges()
    p0, p1, p2 = num0 / total_num, num1 / total_num, num2 / total_num
    
    # ax.plot(range(3), [p0, p1, p2],'ko-', markerfacecolor='white', markersize=10, alpha=1.)
    ax.bar(range(3), [p0, p1, p2], width=0.3, color='.7')
    # ax.legend(loc='upper right', fontsize=7)
    # xlabel = 'Weighted Degree' if weight is not None else 'Degree'
    # ax.set_xlabel(xlabel, fontsize=22)
    if cs_ind == 0:
      ax.set_ylabel('Frequency', fontsize=22)
    # ax.set_xscale('symlog')
    ax.set_yscale('log')
    for axis in ['bottom', 'left']:
      ax.spines[axis].set_linewidth(2.)
      ax.spines[axis].set_color('k')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([])
    # ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    # if cs_ind == 0:
    #   handles, labels = ax.get_legend_handles_labels()
    #   ax.legend(handles, labels, title='', bbox_to_anchor=(.7, .8), loc='upper left', fontsize=16, frameon=False)
      
  plt.tight_layout()
  image_name = './plots/pair_distribution_{}.pdf'.format(row_ind)
  # plt.show()
  plt.savefig(image_name, transparent=True)
  # plt.savefig(image_name.replace('jpg', 'pdf'), transparent=True)

plot_pair_distributions(G_ccg_dict, 7)
# %%
# Figure S7B
def plot_signed_pair_distributions(G_dict, row_ind):
  rows, cols = get_rowcol(G_dict)
  row = rows[row_ind]
  fig, axes = plt.subplots(1, len(combined_stimuli), figsize=(6*len(combined_stimuli), 3))
  for cs_ind, combined_stimulus in enumerate(combined_stimuli):
    ax = axes[cs_ind]
    col = combined_stimulus[0]
    # ax.set_title(combined_stimulus_names[cs_ind].replace('\n', ' '), fontsize=25, rotation=0)
    G = G_dict[row][col].copy()
    p0, p1, p2, p3, p4, p5 = count_signed_triplet_connection_p(G)
    # ax.plot(range(6), [p0, p1, p2, p3, p4, p5],'ko-', markerfacecolor='white', markersize=10, alpha=1.)
    ax.bar(range(6), [p0, p1, p2, p3, p4, p5], width=0.6, color='.7')
    # ax.legend(loc='upper right', fontsize=7)
    # xlabel = 'Weighted Degree' if weight is not None else 'Degree'
    # ax.set_xlabel(xlabel, fontsize=22)
    if cs_ind == 0:
      ax.set_ylabel('Frequency', fontsize=22)
    # ax.set_xscale('symlog')
    ax.set_yscale('log')
    for axis in ['bottom', 'left']:
      ax.spines[axis].set_linewidth(2.)
      ax.spines[axis].set_color('k')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([])
    # ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    # if cs_ind == 0:
    #   handles, labels = ax.get_legend_handles_labels()
    #   ax.legend(handles, labels, title='', bbox_to_anchor=(.7, .8), loc='upper left', fontsize=16, frameon=False)
      
  plt.tight_layout()
  image_name = './plots/signed_pair_distribution_{}.pdf'.format(row_ind)
  # plt.show()
  plt.savefig(image_name, transparent=True)
  # plt.savefig(image_name.replace('jpg', 'pdf'), transparent=True)

plot_signed_pair_distributions(S_ccg_dict, 7)
# %%
def plot_signalcorr_distributions(signal_correlation_dict):
  rows = list(signal_correlation_dict.keys())
  fig, axes = plt.subplots(len(rows), len(combined_stimuli)-1, figsize=(3*len(combined_stimuli), 3*len(rows)))
  for row_ind, row in enumerate(rows):
    active_inds = np.array(sorted(list(active_area_dict[row].keys())))
    for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names[1:]):
      ax = axes[row_ind, cs_ind]
      sc = signal_correlation_dict[row][combined_stimulus_name].copy()
      sc = sc[active_inds[:,None], active_inds] # only the active neurons with at least 2 Hz firing rate
      np.fill_diagonal(sc, np.nan)
      sc = sc.flatten()
      ax.set_title(combined_stimulus_name.replace('\n', ' '), fontsize=25, rotation=0)
      ax.hist(sc)
      ax.axvline(x=np.nanmean(sc), linewidth=3, color='r', linestyle='--', alpha=.4)
      # ax.legend(loc='upper right', fontsize=7)
      xlabel = 'signal correlation'
      ax.set_xlabel(xlabel, fontsize=22)
      ax.set_ylabel('Frequency', fontsize=22)
      # ax.set_xscale('symlog')
      # ax.set_yscale('log')
      for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(2.)
        ax.spines[axis].set_color('k')
      ax.spines['top'].set_visible(False)
      ax.spines['right'].set_visible(False)
      ax.xaxis.set_tick_params(labelsize=20)
      ax.yaxis.set_tick_params(labelsize=20)
      
  plt.tight_layout()
  # plt.show()
  plt.savefig('./plots/distribution_signal_corr.pdf', transparent=True)

plot_signalcorr_distributions(signal_correlation_dict)
# # %%
# # Figure S7
# def plot_distri_weight_confidence(G_dict):
#   rows,cols = get_rowcol(G_dict)
#   fig, axes = plt.subplots(1, 2, figsize=(7*2, 4))
#   targets = ['weight', 'confidence']
#   for ax, target in zip(axes, targets):
#     df = pd.DataFrame()
#     conf, p_conf, n_conf = [], [], []
#     for row in session2keep:
#       for col in cols:
#         G = G_dict[row][col]
#         conf += [w for _, _, w in G.edges(data=target)]
#     p_conf = np.array([c for c in conf if c > 0])
#     n_conf = np.array([abs(c) for c in conf if c < 0])
#     df = pd.concat([df, pd.DataFrame(np.concatenate((p_conf[:,None], np.array(['excitatory connection'] * len(p_conf))[:,None]), 1), columns=['data', 'type'])], ignore_index=True)
#     df = pd.concat([df, pd.DataFrame(np.concatenate((n_conf[:,None], np.array(['inhibitory connection'] * len(n_conf))[:,None]), 1), columns=['data', 'type'])], ignore_index=True)
#     df['data'] = pd.to_numeric(df['data'])
#     # h = sns.histplot(data=df, x='data', linewidth=0, stat='probability', hue='type', common_norm=False, palette=['r', 'b'], alpha=.3, ax=ax)
#     h = sns.kdeplot(data=df, x='data', hue='type', common_norm=False, ax=ax, palette=['r', 'b'], alpha=.8)
#     # ph = sns.histplot(data=p_conf, linewidth=0, stat='probability', color='r', alpha=.3, label='excitatory connection', ax=ax)
#     # nh = sns.histplot(data=n_conf, linewidth=0, stat='probability', color='b', alpha=.3, label='inhibitory link', ax=ax)
    
#     ax.axvline(x=np.nanmean(p_conf), color='r', linestyle='--', alpha=.3)
#     ax.axvline(x=np.nanmean(n_conf), color='b', linestyle='--', alpha=.3)
#     ax.set_xscale('log')
#     ax.set_yscale('log')
#     ax.tick_params(axis='x', which='both', labelsize=20) # 'both' includes major (10^n) and minor ticks (m x 10^n) for log scale
#     # ax.xaxis.set_tick_params(labelsize=20)
#     ax.yaxis.set_tick_params(labelsize=20)
#     xlabel = 'Absolute CCG weight' if target == 'weight' else 'Absolute CCG Z score'
#     ax.set_xlabel(xlabel, fontsize=20)
#     ax.set_ylabel('')
#     m_p_conf, m_n_conf = np.nanmean(p_conf), np.nanmean(n_conf)
#     ax.text(max(m_p_conf, m_n_conf), .03, "percentage difference {:2.2f} %".format(100 * abs(m_p_conf-m_n_conf)/((m_p_conf+m_n_conf)/2)), verticalalignment='center', fontsize=18)
#     for axis in ['bottom', 'left']:
#       ax.spines[axis].set_linewidth(1.5)
#       ax.spines[axis].set_color('k')
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.tick_params(width=1.5)
#     # ax.tick_params(axis='x', labelsize=20)
#     # h.set_xticklabels(h.get_xticks(), size=20)
#     ax.legend([], [], fontsize=0)
    
#   axes[0].set_ylabel('Probability', fontsize=20)
#   plt.savefig('./plots//distri_pos_neg_weight_confidence_fold.pdf', transparent=True)
#   # plt.show()
# plot_distri_weight_confidence(G_ccg_dict)
# %%
