#%%
from codecs import namereplace_errors
from sys import breakpointhook
from library import *
#%%
start_time = time.time()
measure = 'ccg'
min_spike = 50
n = 4
max_duration = 11
maxlag = 12
directory = './data/ecephys_cache_dir/sessions/adj_mat_{}_corrected/'.format(measure)
save_ccg_corrected_highland_new(directory, measure, min_spike=min_spike, max_duration=max_duration, maxlag=maxlag, n=n)
print("--- %s minutes in total" % ((time.time() - start_time)/60))
#%%
min_spike = 50
n = 4
max_duration = 11
maxlag = 12
measure = 'ccg'
directory = './data/ecephys_cache_dir/sessions/adj_mat_{}_corrected/'.format(measure)
path = directory.replace(measure, measure+'_highland')
if not os.path.exists(path):
  os.makedirs(path)
file = '719161530_flash_dark.npz'
try: 
  ccg = load_npz_3d(os.path.join(directory, file))
except:
  ccg = load_sparse_npz(os.path.join(directory, file))
try:
  ccg_jittered = load_npz_3d(os.path.join(directory, file.replace('.npz', '_bl.npz')))
except:
  ccg_jittered = load_sparse_npz(os.path.join(directory, file.replace('.npz', '_bl.npz')))
num_nodes = ccg.shape[0]
significant_ccg,significant_confidence,significant_offset,significant_duration=np.zeros((num_nodes,num_nodes)),np.zeros((num_nodes,num_nodes)),np.zeros((num_nodes,num_nodes)),np.zeros((num_nodes,num_nodes))
significant_ccg[:] = np.nan
significant_confidence[:] = np.nan
significant_offset[:] = np.nan
significant_duration[:] = np.nan
ccg_corrected = ccg - ccg_jittered
# corr = (ccg_corrected - ccg_corrected.mean(-1)[:, :, None])
for duration in np.arange(max_duration, -1, -1): # reverse order, so that sharp peaks can override highland
  print('duration {}'.format(duration))
  highland_ccg, confidence_level, offset, indx = find_highland_new(ccg_corrected, min_spike, duration, maxlag, n)
  # highland_ccg, confidence_level, offset, indx = find_highland(corr, min_spike, duration, maxlag, n)
  if np.sum(indx):
    significant_ccg[indx] = highland_ccg[indx]
    significant_confidence[indx] = confidence_level[indx]
    significant_offset[indx] = offset[indx]
    significant_duration[indx] = duration
double_0 = (significant_offset==0) & (significant_offset.T==0) & (~np.isnan(significant_ccg)) & (~np.isnan(significant_ccg.T))
print('Number of cross duration double-count edges: {}'.format(np.sum(double_0)))
if np.sum(double_0):
  remove_0 = (significant_duration >= significant_duration.T) & double_0
  significant_ccg[remove_0], significant_confidence[remove_0], significant_offset[remove_0], significant_duration[remove_0] = np.nan, np.nan, np.nan, np.nan
  for duration in np.arange(max_duration, -1, -1): # reverse order, so that sharp peaks can override highland
    highland_ccg_2nd, confidence_level_2nd, offset_2nd, indx_2nd =  find_2nd_highland(ccg_corrected[remove_0], duration, maxlag, n)
    if np.sum(indx_2nd):
      significant_ccg[remove_0][indx_2nd] = highland_ccg_2nd[indx_2nd]
      significant_confidence[remove_0][indx_2nd] = confidence_level_2nd[indx_2nd]
      significant_offset[remove_0][indx_2nd] = offset_2nd[indx_2nd]
      significant_duration[remove_0][indx_2nd] = duration
# %%
################# save active neuron inds
min_FR = 0.002 # 2 Hz
stimulus_names = ['spontaneous', 'flash_light', 'flash_dark',
        'drifting_gratings', 'static_gratings',
          'natural_scenes', 'natural_movie_one', 'natural_movie_three']
save_active_inds(min_FR, session_ids, stimulus_names)
#%%
################# save active area dict
area_dict = load_area_dict(session_ids)
save_active_area_dict(area_dict)
# %%
########### check double count 0 time lag edges
directory = './data/ecephys_cache_dir/sessions/adj_mat_ccg_highland_corrected/'
files = os.listdir(directory)
files.sort(key=lambda x:int(x[:9]))
for file in files:
  if file.endswith(".npz") and ('gabors' not in file) and ('_offset' not in file) and ('_duration' not in file) and ('_bl' not in file) and ('confidence' not in file):
    print(file)
    # break
    adj_mat = load_npz_3d(os.path.join(directory, file))
    offset_mat = load_npz_3d(os.path.join(directory, file.replace('.npz', '_offset.npz')))
    duration_mat = load_npz_3d(os.path.join(directory, file.replace('.npz', '_duration.npz')))
    confidence_level = load_npz_3d(os.path.join(directory, file.replace('.npz', '_confidence.npz')))
    print('{}, number of double count 0 edges is {}'.format(file, ((offset_mat==0) & (offset_mat.T==0)).sum()))
# %%
# start_time = time.time()
# measure = 'ccg'
# min_spike = 50
# n = 4
# max_duration = 12
# directory = './data/ecephys_cache_dir/sessions/adj_mat_{}_corrected/'.format(measure)
# save_ccg_corrected_highland(directory, measure, min_spike=min_spike, max_duration=max_duration, maxlag=12, n=n)
# print("--- %s minutes in total" % ((time.time() - start_time)/60))
#%%
area_dict, active_area_dict, mean_speed_df = load_other_data(session_ids)
measure = 'ccg'
directory = './data/ecephys_cache_dir/sessions/adj_mat_{}_highland_corrected/'.format(measure)
G_ccg_dict, offset_dict, duration_dict = load_highland_xcorr(directory, active_area_dict, weight=True)
#%%
double0_edge_heatmap(offset_dict, maxlag, 'bidirectional offset', measure, n)
double0_edge_heatmap(offset_dict, maxlag, 'all offset', measure, n)
#%%
double0_edge_heatmap(duration_dict, max_duration, 'bidirectional duration', measure, n)
double0_edge_heatmap(duration_dict, max_duration, 'all duration', measure, n)
# %%
############### find the correlation window (percentage) for each significant edge
############### opposite padding now, need mofification !!!!!!!!!!!!!!!!
def get_correlation_window(directory, offset_dict, duration_dict):
  inds_path = './data/ecephys_cache_dir/sessions/active_inds/'
  rows, cols = get_rowcol(offset_dict)
  # cols.remove('flash_dark')
  # cols.remove('flash_light')
  c_window = {}
  for row in rows:
    c_window[row] = {}
    for col in cols:
      file = row + '_' + col + '.npz'
      print(file)
      adj_mat = load_npz_3d(os.path.join('./data/ecephys_cache_dir/sessions/adj_mat_ccg_highland_corrected/', file))
      c_window[row][col] = []
      offset_mat, duration_mat = offset_dict[row][col], duration_dict[row][col]   
      sequences = load_npz_3d(os.path.join(directory, file))
      active_neuron_inds = np.load(os.path.join(inds_path, row+'.npy'))
      sequences = sequences[active_neuron_inds]
      num_neuron, num_trial, T = sequences.shape
      assert adj_mat.shape == offset_mat.shape == duration_mat.shape == (num_neuron, num_neuron)
      assert np.where(~np.isnan(adj_mat))[0].shape[0] == np.where(~np.isnan(offset_mat))[0].shape[0] == np.where(~np.isnan(duration_mat))[0].shape[0]
      for row_a, row_b in zip(*np.where(~np.isnan(adj_mat))): # each significant edge
        if adj_mat[row_a, row_b] > 0: # only for positive correlation
          offset, duration = int(offset_mat[row_a, row_b]), int(duration_mat[row_a, row_b])
          tfa, tla, tfa_ast, tla_ast, tfb, tlb, tfb_ast, tlb_ast = [[] for _ in range(8)]
          for m in range(num_trial):
            # print('Trial {} / {}'.format(m+1, num_trial))
            matrix = sequences[:,m,:]
            fr_rowa = np.count_nonzero(matrix[row_a]) / (matrix.shape[1]/1000) # Hz instead of kHz
            fr_rowb = np.count_nonzero(matrix[row_b]) / (matrix.shape[1]/1000)
            if fr_rowa * fr_rowb > 0: # there could be no spike in a certain trial
              for pad_len in range(offset, offset+duration+1): # min duration is 0
                s_a = np.pad(matrix[row_a], (pad_len, 0), 'constant', constant_values=(0,0)) # false for incorrect CCG, should be opposite padding
                s_b = np.pad(matrix[row_b], (0, pad_len), 'constant', constant_values=(0,0))
                prod = s_a * s_b
                perioda, periodb, periodp = np.where(s_a>0)[0], np.where(s_b>0)[0], np.where(prod>0)[0]
                if len(periodp):
                  tfa.append(perioda[0])
                  tla.append(perioda[-1])
                  tfa_ast.append(periodp[0])
                  tla_ast.append(periodp[-1])
                  tfb.append(periodb[0])
                  tlb.append(periodb[-1])
                  tfb_ast.append(periodp[0]-pad_len)
                  tlb_ast.append(periodp[-1]-pad_len)

          # if len(tla) and len(tfa) and len(tlb) and len(tfb) and len(tla_ast) and len(tfa_ast) and len(tlb_ast) and len(tfb_ast):
          Ta, Tb, Ta_ast, Tb_ast = max(tla)-min(tfa)+1, max(tlb)-min(tfb)+1, max(tla_ast)-min(tfa_ast)+1, max(tlb_ast)-min(tfb_ast)+1 # count start and end
          c_window[row][col].append(min(Ta_ast/Ta, Tb_ast/Tb))
        # print(min(Ta_ast/Ta, Tb_ast/Tb))
  return c_window

start_time = time.time()
directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
c_window = get_correlation_window(directory, offset_dict, duration_dict)
print("--- %s minutes" % ((time.time() - start_time)/60))
#%%
with open('c_window.pkl', 'wb') as f:
  pickle.dump(c_window, f)
# %%
def plot_multi_correlation_window(c_window, measure, n):
  ind = 1
  rows, cols = get_rowcol(c_window)
  fig = plt.figure(figsize=(9*len(cols), 6*len(rows)))
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
      sns.histplot(data=c_window[row][col], stat='probability', kde=True, linewidth=0)
      plt.axvline(x=np.nanmean(c_window[row][col]), color='r', linestyle='--')
  plt.suptitle('correlation window')
  plt.tight_layout(rect=[0, 0.03, 1, 0.98])
  plt.savefig('./plots/multi_correlation_window{}_{}fold.jpg'.format(measure, n))

with open('c_window.pkl', 'rb') as f:
  c_window_l = pickle.load(f)
plot_multi_correlation_window(c_window_l, measure, n)
# %%
# row, col = '755434585', 'natural_movie_one'
# file = row+'_'+col+'.npz'
# adj_mat = load_npz_3d(os.path.join('./data/ecephys_cache_dir/sessions/adj_mat_ccg_highland_corrected/', file))
# c_window[row][col] = []
# offset_mat, duration_mat = offset_dict[row][col], duration_dict[row][col]   
# sequences = load_npz_3d(os.path.join('./data/ecephys_cache_dir/sessions/spiking_sequence/', file))
# active_neuron_inds = np.load(os.path.join(inds_path, row+'.npy'))
# sequences = sequences[active_neuron_inds]
# num_neuron, num_trial, T = sequences.shape
# assert adj_mat.shape == offset_mat.shape == duration_mat.shape == (num_neuron, num_neuron)
# assert np.where(~np.isnan(adj_mat))[0].shape[0] == np.where(~np.isnan(offset_mat))[0].shape[0] == np.where(~np.isnan(duration_mat))[0].shape[0]
# for row_a, row_b in zip(*np.where(~np.isnan(adj_mat))): # each significant edge
#   if adj_mat[row_a, row_b] > 0: # only for positive correlation
#     offset, duration = int(offset_mat[row_a, row_b]), int(duration_mat[row_a, row_b])
#     tfa, tla, tfa_ast, tla_ast, tfb, tlb, tfb_ast, tlb_ast = [[] for _ in range(8)]
#     for m in range(num_trial):
#       # print('Trial {} / {}'.format(m+1, num_trial))
#       matrix = sequences[:,m,:]
#       fr_rowa = np.count_nonzero(matrix[row_a]) / (matrix.shape[1]/1000) # Hz instead of kHz
#       fr_rowb = np.count_nonzero(matrix[row_b]) / (matrix.shape[1]/1000)
#       if fr_rowa * fr_rowb > 0: # there could be no spike in a certain trial
#         for pad_len in range(offset, offset+duration+1): # min duration is 0
#           s_a = np.pad(matrix[row_a], (pad_len, 0), 'constant', constant_values=(0,0)) # false for incorrect CCG, should be opposite padding
#           s_b = np.pad(matrix[row_b], (0, pad_len), 'constant', constant_values=(0,0))
#           prod = s_a * s_b
#           perioda, periodb, periodp = np.where(s_a>0)[0], np.where(s_b>0)[0], np.where(prod>0)[0]
#           if len(periodp):
#             tfa.append(perioda[0])
#             tla.append(perioda[-1])
#             tfa_ast.append(periodp[0])
#             tla_ast.append(periodp[-1])
#             tfb.append(periodb[0])
#             tlb.append(periodb[-1])
#             tfb_ast.append(periodp[0]-pad_len)
#             tlb_ast.append(periodp[-1]-pad_len)

#     # if len(tla) and len(tfa) and len(tlb) and len(tfb) and len(tla_ast) and len(tfa_ast) and len(tlb_ast) and len(tfb_ast):
#     Ta, Tb, Ta_ast, Tb_ast = max(tla)-min(tfa)+1, max(tlb)-min(tfb)+1, max(tla_ast)-min(tfa_ast)+1, max(tlb_ast)-min(tfb_ast)+1 # count start and end
#     c_window[row][col].append(min(Ta_ast/Ta, Tb_ast/Tb))
# sns.histplot(data=c_window[row][col], stat='probability', kde=True, linewidth=0)
# plt.axvline(x=np.nanmean(c_window[row][col]), color='r', linestyle='--')
# %%
################# portion after threshold
directory = './data/ecephys_cache_dir/sessions/adj_mat_ccg_highland_corrected/'
threshold = 0.9
df_n, df_p = test_portion_above_threshold(directory, threshold)
#%%
directory = './data/ecephys_cache_dir/sessions/adj_mat_ccg_highland_corrected/'
threshold = 0.9
n = 4
plot_uniq_comparison(directory, threshold, n)
#%%
directory = './data/ecephys_cache_dir/sessions/adj_mat_ccg_highland_corrected/'
distri = get_normalized_entropy_distri(directory)
#%%
measure = 'ccg'
n = 4
plot_normalized_entropy(distri, measure, n)
#%%
##################### keep edges above threshold
directory = './data/ecephys_cache_dir/sessions/adj_mat_ccg_highland_corrected/'
threshold = 0.9
keep_edges_above_threshold(directory, threshold)
# %%
# test different significance levels
def save_ccg_corrected_sig_levels(directory, measure, min_spike=50, max_duration=6, maxlag=12, n=3):
  path = directory.replace(measure, measure+'_{}'.format(n))
  if not os.path.exists(path):
    os.makedirs(path)
  files = os.listdir(directory)
  files.sort(key=lambda x:int(x[:9]))
  for file in files:
    if '_bl' not in file:
      print(file)
      try: 
        ccg = load_npz_3d(os.path.join(directory, file))
      except:
        ccg = load_sparse_npz(os.path.join(directory, file))
      try:
        ccg_jittered = load_npz_3d(os.path.join(directory, file.replace('.npz', '_bl.npz')))
      except:
        ccg_jittered = load_sparse_npz(os.path.join(directory, file.replace('.npz', '_bl.npz')))
      num_nodes = ccg.shape[0]
      significant_ccg,significant_confidence,significant_offset,significant_duration=np.zeros((num_nodes,num_nodes)),np.zeros((num_nodes,num_nodes)),np.zeros((num_nodes,num_nodes)),np.zeros((num_nodes,num_nodes))
      significant_ccg[:] = np.nan
      significant_confidence[:] = np.nan
      significant_offset[:] = np.nan
      significant_duration[:] = np.nan
      ccg_corrected = ccg - ccg_jittered
      # corr = (ccg_corrected - ccg_corrected.mean(-1)[:, :, None])
      for duration in np.arange(max_duration, -1, -1): # reverse order, so that sharp peaks can override highland
        # print('duration {}'.format(duration))
        highland_ccg, confidence_level, offset, indx = find_highland_new(ccg_corrected, min_spike, duration, maxlag, n)
        # highland_ccg, confidence_level, offset, indx = find_highland(corr, min_spike, duration, maxlag, n)
        if np.sum(indx):
          significant_ccg[indx] = highland_ccg[indx]
          significant_confidence[indx] = confidence_level[indx]
          significant_offset[indx] = offset[indx]
          significant_duration[indx] = duration
      double_0 = (significant_offset==0) & (significant_offset.T==0) & (~np.isnan(significant_ccg)) & (~np.isnan(significant_ccg.T))
      # print('Number of cross duration double-count edges: {}'.format(np.sum(double_0)))
      if np.sum(double_0):
        remove_0 = (significant_duration >= significant_duration.T) & double_0
        significant_ccg[remove_0], significant_confidence[remove_0], significant_offset[remove_0], significant_duration[remove_0] = np.nan, np.nan, np.nan, np.nan
        for duration in np.arange(max_duration, -1, -1): # reverse order, so that sharp peaks can override highland
          highland_ccg_2nd, confidence_level_2nd, offset_2nd, indx_2nd =  find_2nd_highland(ccg_corrected[remove_0], duration, maxlag, n)
          if np.sum(indx_2nd):
            significant_ccg[remove_0][indx_2nd] = highland_ccg_2nd[indx_2nd]
            significant_confidence[remove_0][indx_2nd] = confidence_level_2nd[indx_2nd]
            significant_offset[remove_0][indx_2nd] = offset_2nd[indx_2nd]
            significant_duration[remove_0][indx_2nd] = duration

      # print('{} significant edges'.format(np.sum(~np.isnan(significant_ccg))))
      save_npz(significant_ccg, os.path.join(path, file))
      save_npz(significant_confidence, os.path.join(path, file.replace('.npz', '_confidence.npz')))
      save_npz(significant_offset, os.path.join(path, file.replace('.npz', '_offset.npz')))
      save_npz(significant_duration, os.path.join(path, file.replace('.npz', '_duration.npz')))

for n in range(1, 8):
  print('{} standard deviation from mean'.format(n))
  start_time = time.time()
  measure = 'ccg'
  min_spike = 50
  max_duration = 11
  maxlag = 12
  directory = './data/ecephys_cache_dir/sessions/adj_mat_{}_corrected/'.format(measure)
  save_ccg_corrected_sig_levels(directory, measure, min_spike=min_spike, max_duration=max_duration, maxlag=maxlag, n=n)
  print("--- %s minutes" % ((time.time() - start_time)/60))
#%%
##################### keep edges above threshold
def keep_edges_above_threshold_n(directory, threshold, n):
  files = os.listdir(directory)
  files.sort(key=lambda x:int(x[:9]))
  for file in files:
    if file.endswith(".npz") and ('gabors' not in file) and ('flashes' not in file) and ('_offset' not in file) and ('_duration' not in file) and ('_bl' not in file) and ('confidence' not in file):
      print(file)
      significant_ccg = load_npz_3d(os.path.join(directory, file))
      significant_duration = load_npz_3d(os.path.join(directory, file.replace('.npz', '_duration.npz')))
      significant_offset = load_npz_3d(os.path.join(directory, file.replace('.npz', '_offset.npz')))
      significant_confidence = load_npz_3d(os.path.join(directory, file.replace('.npz', '_confidence.npz')))
      ccg = load_sparse_npz(os.path.join(directory.replace('adj_mat_ccg_{}_corrected'.format(n), 'adj_mat_ccg_corrected'), file))
      ccg_jittered = load_sparse_npz(os.path.join(directory.replace('adj_mat_ccg_{}_corrected'.format(n), 'adj_mat_ccg_corrected'), file.replace('.npz', '_bl.npz')))
      ccg_corrected = ccg - ccg_jittered
      significant_inds = list(zip(*np.where(~np.isnan(significant_ccg))))
      row_as, row_bs = [], []
      for row_a, row_b in significant_inds:
        filter = np.array([1]).repeat(significant_duration[row_a,row_b]+1) # sum instead of mean
        sig = signal.convolve(ccg_corrected[row_a, row_b], filter, mode='valid', method='fft')
        # uniq = unique_with_tolerance(sig, 1e-7)
        # if len(uniq)/len(sig) >= threshold:
        entropy = normalized_entropy_with_tolerance(sig, TOL=1e-7)
        if entropy >= threshold:
          row_as.append(row_a)
          row_bs.append(row_b)
      print(len(row_as)/len(significant_inds))
      if len(row_as): # there is no edge for some significance level
        inds = (np.array(row_as), np.array(row_bs))
        filtered_ccg,filtered_confidence,filtered_offset,filtered_duration= [np.zeros_like(significant_ccg), np.zeros_like(significant_ccg), np.zeros_like(significant_ccg), np.zeros_like(significant_ccg)]
        filtered_ccg[:] = np.nan
        filtered_confidence[:] = np.nan
        filtered_offset[:] = np.nan
        filtered_duration[:] = np.nan
        filtered_ccg[inds] = significant_ccg[inds]
        filtered_confidence[inds] = significant_confidence[inds]
        filtered_offset[inds] = significant_offset[inds]
        filtered_duration[inds] = significant_duration[inds]
        print(len(np.where(~np.isnan(significant_ccg))[0]), len(np.where(~np.isnan(significant_confidence))[0]), len(np.where(~np.isnan(significant_offset))[0]), len(np.where(~np.isnan(significant_duration))[0]))
        print(len(np.where(~np.isnan(filtered_ccg))[0]), len(np.where(~np.isnan(filtered_confidence))[0]), len(np.where(~np.isnan(filtered_offset))[0]), len(np.where(~np.isnan(filtered_duration))[0]))
        save_npz(filtered_ccg, os.path.join(directory, file))
        save_npz(filtered_confidence, os.path.join(directory, file.replace('.npz', '_confidence.npz')))
        save_npz(filtered_offset, os.path.join(directory, file.replace('.npz', '_offset.npz')))
        save_npz(filtered_duration, os.path.join(directory, file.replace('.npz', '_duration.npz')))

for n in range(1, 8):
  print('{} standard deviation from mean'.format(n))
  start_time = time.time()
  measure = 'ccg'
  min_spike = 50
  max_duration = 11
  maxlag = 12
  directory = './data/ecephys_cache_dir/sessions/adj_mat_{}_{}_corrected/'.format(measure, n)
  threshold = 0.9
  keep_edges_above_threshold_n(directory, threshold, n)
  print("--- %s minutes" % ((time.time() - start_time)/60))
# %%
def load_graph_sig_level(directory, active_area_dict):
  G_dict = {}
  files = os.listdir(directory)
  files.sort(key=lambda x:int(x[:9]))
  for file in files:
    if file.endswith(".npz") and ('gabors' not in file) and ('flashes' not in file) and ('_offset' not in file) and ('_duration' not in file) and ('_bl' not in file) and ('confidence' not in file):
      adj_mat = load_npz_3d(os.path.join(directory, file))
      mouseID = file.split('_')[0]
      stimulus_name = file.replace('.npz', '').replace(mouseID + '_', '')
      if not mouseID in G_dict:
        G_dict[mouseID] = {}
      G_dict[mouseID][stimulus_name] = generate_graph(adj_mat=np.nan_to_num(adj_mat), confidence_level=[np.nan], active_area=active_area_dict[mouseID], cc=False, weight=False)
  return G_dict

seven_session_ids = ['719161530', '750749662', '754312389', '755434585', '756029989', '791319847', '797828357']
area_dict, active_area_dict, mean_speed_df = load_other_data(session_ids)
num_nodes, num_edges, dens = np.zeros((8, 7, len(combined_stimulus_names))), np.zeros((8, 7, len(combined_stimulus_names))), np.zeros((8, 7, len(combined_stimulus_names)))
n_list = list(range(1, 8))
for ind, n in enumerate(n_list):
  print('n = {}'.format(n))
  directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
  path = directory.replace('spiking_sequence', 'adj_mat_ccg_{}_corrected').format(n)
  G_ccg_dict = load_graph_sig_level(path, active_area_dict)
  G_ccg_dict = remove_gabor(G_ccg_dict)
  ######### removed neurons from thalamic region
  G_ccg_dict = remove_thalamic(G_ccg_dict, area_dict, visual_regions)
  for s_ind, session_id in enumerate(seven_session_ids):
    for c_ind, combined_stimulus in enumerate(combined_stimuli):
      node_l, edge_l, den_l = [], [], []
      for stimulus in combined_stimulus:
        G = G_ccg_dict[session_id][stimulus]
        node_l.append(nx.number_of_nodes(G))
        edge_l.append(nx.number_of_edges(G))
        den_l.append(nx.density(G))
      num_nodes[ind, s_ind, c_ind], num_edges[ind, s_ind, c_ind], dens[ind, s_ind, c_ind] = np.mean(node_l), np.mean(edge_l), np.mean(den_l)
# %%
# three metrics
color_list = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
fig, axes = plt.subplots(1, 3, figsize=(4*3, 4))
for ind, n in enumerate(n_list):
  m, std = num_nodes[ind].mean(0), num_nodes[ind].std(0)
  axes[0].plot(range(6), m, color=color_list[ind], label=str(n))
  axes[0].fill_between(range(6), m-std, m+std, alpha=.2, color=color_list[ind])
axes[0].legend()
for ind, n in enumerate(n_list):
  m, std = num_edges[ind].mean(0), num_edges[ind].std(0)
  axes[1].plot(range(6), m, color=color_list[ind], label=str(n))
  axes[1].fill_between(range(6), m-std, m+std, alpha=.2, color=color_list[ind])
axes[1].legend()
for ind, n in enumerate(n_list):
  m, std = dens[ind].mean(0), dens[ind].std(0)
  axes[2].plot(range(6), m, color=color_list[ind], label=str(n))
  axes[2].fill_between(range(6), m-std, m+std, alpha=.2, color=color_list[ind])
axes[2].legend()
axes[2].set_yscale('log')
plt.show()
#%%
# only density
color_list = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
for ind, n in enumerate(n_list):
  m, std = dens[ind].mean(0), dens[ind].std(0)
  ax.plot(range(6), m, color=color_list[ind], label=str(n))
  ax.fill_between(range(6), m-std, m+std, alpha=.2, color=color_list[ind])
ax.legend()
ax.set_xticks(range(6))
ax.set_xticklabels(combined_stimulus_names, fontsize=12)
ax.set_yscale('log')
ax.set_ylabel('density', fontsize=20)
plt.savefig('./plots/density_significance_level_line.pdf', transparent=True)
# plt.show()
#%%
stimulus2marker = {'Resting\nstate':'s', 'Flashes':'*', 'Drifting\ngratings':'X', 'Static\ngratings':'P', 'Natural\nscenes':r'$\clubsuit$', 'Natural\nmovies':'>'}
marker_size_dict = {'v':10, '*':22, 'P':13, 'X':13, 'o':11, 's':9.5, 'D':9, 'p':12, '>':10, r'$\clubsuit$':20}
scatter_size_dict = {'v':10, '*':17, 'P':13, 'X':13, 'o':11, 's':10, 'D':9, 'p':13, '>':12, r'$\clubsuit$':16}
error_size_dict = {'v':10, '*':24, 'P':16, 'X':16, 'o':11, 's':9., 'D':9, 'p':12, '>':13, r'$\clubsuit$':22}
# boxplot, only density
def plot_density_siglevel(dens, n_list):
  palette = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
  fig, ax = plt.subplots(1,1, sharex=True, sharey=True, figsize=(8, 4))
  x = np.arange(len(combined_stimulus_names))
  for n_ind, n in enumerate(n_list):
    y, err = dens[n_ind].mean(0), dens[n_ind].std(0)
    # err = 1.96 * data.std(numeric_only=True).loc[combined_stimulus_names].values.flatten() / data.size().loc[combined_stimulus_names].pow(1./2).values.flatten() # 95% confidence interval
    # err = stats.t.ppf((1 + 0.95) / 2., data.size().loc[combined_stimulus_names]-1) * data.sem().loc[combined_stimulus_names].values.flatten()
    for ind, (xi, yi, erri) in enumerate(zip(x, y, err)):
      if yi:
        # ax.errorbar(xi + .13 * mt_ind, yi, yerr=erri, fmt=marker_list[ind], ms=markersize_list[ind], linewidth=2.,color=palette[mt_ind])
        ax.errorbar(xi + .13 * n_ind, yi, yerr=erri, fmt=' ', linewidth=2.,color=palette[n_ind], zorder=1)
        ax.scatter(xi + .13 * n_ind, yi, marker=stimulus2marker[combined_stimulus_names[ind]], s=10*error_size_dict[stimulus2marker[combined_stimulus_names[ind]]], linewidth=1.,color=palette[n_ind], zorder=2)
  
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
  # ax.set_ylim(bottom=0)
  ax.set_yscale('log')
  # ylabel = 'Fraction of three V1 neurons' if mtype=='all_V1' else: 'Fraction of at least one V1 neuron'
  # ax.set_ylabel('Density', fontsize=30)
  plt.tight_layout(rect=[.02, -.03, 1, 1])
  plt.savefig('./plots/density_significance_level.pdf', transparent=True)
  # plt.show()

plot_density_siglevel(dens, n_list)
# %%
