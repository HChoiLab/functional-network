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
