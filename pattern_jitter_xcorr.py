# %%
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
#%%
################# get optimal resolution that maximizes delta Q
rows, cols = get_rowcol(G_ccg_dict)
with open('metrics.pkl', 'rb') as f:
  metrics = pickle.load(f)
resolution_list = np.arange(0, 2.1, 0.1)
max_reso_gnm, max_reso_swap = get_max_resolution(rows, cols, resolution_list, metrics)

def n_cross_correlation6(matrix, maxlag, disable): ### fastest, only causal correlation (A>B, only positive time lag on B), largest deviation from time window average
  N, M =matrix.shape
  xcorr=np.zeros((N,N))
  norm_mata = np.nan_to_num((matrix-np.mean(matrix, axis=1).reshape(-1, 1))/(np.std(matrix, axis=1).reshape(-1, 1)*np.sqrt(M)))
  norm_matb = np.nan_to_num((matrix-np.mean(matrix, axis=1).reshape(-1, 1))/(np.std(matrix, axis=1).reshape(-1, 1)*np.sqrt(M)))
  #### padding
  norm_mata = np.concatenate((norm_mata.conj(), np.zeros((N, maxlag))), axis=1)
  norm_matb = np.concatenate((np.zeros((N, maxlag)), norm_matb.conj(), np.zeros((N, maxlag))), axis=1) # must concat zeros to the left, why???????????
  total_len = len(list(itertools.permutations(range(N), 2)))
  for row_a, row_b in tqdm(itertools.permutations(range(N), 2), total=total_len , miniters=int(total_len/100), disable=disable): # , miniters=int(total_len/100)
  # for ind, (row_a, row_b) in enumerate(itertools.permutations(range(N), 2)): # , miniters=int(total_len/100)
    # faulthandler.enable()
    px, py = norm_mata[row_a, :], norm_matb[row_b, :]
    T = as_strided(py[maxlag:], shape=(maxlag+1, M + maxlag),
                    strides=(-py.strides[0], py.strides[0])) # must be py[maxlag:], why???????????
    # corr = np.dot(T, px)
    corr = T @ px
    corr = corr - uniform_filter1d(corr, maxlag, mode='mirror') # mirror for the boundary values
    xcorr[row_a, row_b] = corr[np.argmax(np.abs(corr))]
  return xcorr

def n_cross_correlation7(matrix, maxlag, disable): ### original correlation
  N, M =matrix.shape
  xcorr=np.zeros((N,N))
  norm_mata = np.nan_to_num((matrix-np.mean(matrix, axis=1).reshape(-1, 1))/(np.std(matrix, axis=1).reshape(-1, 1)*np.sqrt(M)))
  norm_matb = np.nan_to_num((matrix-np.mean(matrix, axis=1).reshape(-1, 1))/(np.std(matrix, axis=1).reshape(-1, 1)*np.sqrt(M)))
  #### padding
  norm_mata = np.concatenate((norm_mata.conj(), np.zeros((N, maxlag))), axis=1)
  norm_matb = np.concatenate((np.zeros((N, maxlag)), norm_matb.conj(), np.zeros((N, maxlag))), axis=1) # must concat zeros to the left, why???????????
  total_len = len(list(itertools.permutations(range(N), 2)))
  for row_a, row_b in tqdm(itertools.permutations(range(N), 2), total=total_len , miniters=int(total_len/100), disable=disable): # , miniters=int(total_len/100)
  # for ind, (row_a, row_b) in enumerate(itertools.permutations(range(N), 2)): # , miniters=int(total_len/100)
    # faulthandler.enable()
    px, py = norm_mata[row_a, :], norm_matb[row_b, :]
    T = as_strided(py[maxlag:], shape=(maxlag+1, M + maxlag),
                    strides=(-py.strides[0], py.strides[0])) # must be py[maxlag:], why???????????
    # corr = np.dot(T, px)
    corr = T @ px
    xcorr[row_a, row_b] = corr[np.argmax(np.abs(corr))]
  return xcorr

def n_cross_correlation8(matrix, maxlag=12, window=100, disable=True): ### fastest, only causal correlation (A>B, only positive time lag on B), largest deviation from flank
  N, M =matrix.shape
  # xcorr, peak_offset=np.zeros((N,N)), np.zeros((N,N))
  xcorr, peak_offset=np.empty((N,N)), np.empty((N,N))
  xcorr[:] = np.nan
  peak_offset[:] = np.nan
  norm_mata = np.nan_to_num((matrix-np.mean(matrix, axis=1).reshape(-1, 1))/(np.std(matrix, axis=1).reshape(-1, 1)*np.sqrt(M)))
  norm_matb = np.nan_to_num((matrix-np.mean(matrix, axis=1).reshape(-1, 1))/(np.std(matrix, axis=1).reshape(-1, 1)*np.sqrt(M)))
  #### padding
  norm_mata = np.concatenate((norm_mata.conj(), np.zeros((N, window))), axis=1)
  norm_matb = np.concatenate((np.zeros((N, window)), norm_matb.conj(), np.zeros((N, window))), axis=1) # must concat zeros to the left, why???????????
  total_len = len(list(itertools.permutations(range(N), 2)))
  for row_a, row_b in tqdm(itertools.permutations(range(N), 2), total=total_len , miniters=int(total_len/100), disable=disable): # , miniters=int(total_len/100)
  # for ind, (row_a, row_b) in enumerate(itertools.permutations(range(N), 2)): # , miniters=int(total_len/100)
    # faulthandler.enable()
    px, py = norm_mata[row_a, :], norm_matb[row_b, :]
    T = as_strided(py[window:], shape=(window+1, M + window),
                    strides=(-py.strides[0], py.strides[0])) # must be py[window:], why???????????
    # corr = np.dot(T, px)
    corr = T @ px
    corr = (corr - corr.mean())[:maxlag]
    max_offset = np.argmax(np.abs(corr))
    xcorr[row_a, row_b] = corr[max_offset]
    peak_offset[row_a, row_b] = max_offset
  return xcorr, peak_offset

def ccg_mat(matrix, maxlag=12, window=100, disable=True): ### fastest, only causal correlation (A>B, only positive time lag on B), largest deviation from flank
  N, M =matrix.shape
  # xcorr, peak_offset=np.zeros((N,N)), np.zeros((N,N))
  ccg=np.empty((N,N))
  ccg[:] = np.nan
  firing_rates = np.count_nonzero(matrix, axis=1) / (matrix.shape[1]/1000) # Hz instead of kHz
  #### padding
  norm_mata = np.concatenate((matrix.conj(), np.zeros((N, window))), axis=1)
  norm_matb = np.concatenate((np.zeros((N, window)), matrix.conj(), np.zeros((N, window))), axis=1) # must concat zeros to the left, why???????????
  total_len = len(list(itertools.permutations(range(N), 2)))
  for row_a, row_b in tqdm(itertools.permutations(range(N), 2), total=total_len , miniters=int(total_len/100), disable=disable): # , miniters=int(total_len/100)
    px, py = norm_mata[row_a, :], norm_matb[row_b, :]
    T = as_strided(py[window:], shape=(window+1, M + window),
                    strides=(-py.strides[0], py.strides[0])) # must be py[window:], why???????????
    corr = (T @ px) / ((M-np.arange(window+1))/1000 * np.sqrt(firing_rates[row_a] * firing_rates[row_b]))
    corr = (corr - corr.mean())[:maxlag]
    max_offset = np.argmax(corr)
    ccg[row_a, row_b] = corr[max_offset]
  return ccg

def ccg_2mat(matrix_a, matrix_b, maxlag=12, window=100, disable=True): ### CCG-mean of flank
  if len(matrix_a.shape) >= 2:
    N, M = matrix_a.shape
    if len(matrix_b.shape) < 2:
      matrix_b = np.tile(matrix_b, (N, 1))
  else:
    N, M = matrix_b.shape
    matrix_a = np.tile(matrix_a, (N, 1))
  firing_rates_a = np.count_nonzero(matrix_a, axis=1) / (matrix_a.shape[1]/1000) # Hz instead of kHz
  firing_rates_b = np.count_nonzero(matrix_b, axis=1) / (matrix_b.shape[1]/1000)
  ccg=np.zeros((2,2,N))
  #### padding
  mata_0 = np.concatenate((matrix_a.conj(), np.zeros((N, window))), axis=1)
  mata_1 = np.concatenate((np.zeros((N, window)), matrix_a.conj(), np.zeros((N, window))), axis=1)
  matb_0 = np.concatenate((matrix_b.conj(), np.zeros((N, window))), axis=1)
  matb_1 = np.concatenate((np.zeros((N, window)), matrix_b.conj(), np.zeros((N, window))), axis=1) # must concat zeros to the left, why???????????
  for row in tqdm(range(N), total=N, miniters=int(N/100), disable=disable): # , miniters=int(total_len/100)
    # faulthandler.enable()
    px, py = mata_0[row, :], matb_1[row, :]
    T = as_strided(py[window:], shape=(window+1, M + window),
                    strides=(-py.strides[0], py.strides[0])) # must be py[window:], why???????????
    corr = (T @ px) / ((M-np.arange(window+1))/1000 * np.sqrt(firing_rates_a[row] * firing_rates_b[row]))
    corr = (corr - corr.mean())[:maxlag]
    ccg[0, 1, row] = corr[np.argmax(np.abs(corr))]
    px, py = matb_0[row, :], mata_1[row, :]
    T = as_strided(py[window:], shape=(window+1, M + window),
                    strides=(-py.strides[0], py.strides[0])) # must be py[window:], why???????????
    # corr = np.dot(T, px)
    corr = T @ px
    corr = (corr - corr.mean())[:maxlag]
    ccg[1, 0, row] = corr[np.argmax(np.abs(corr))]
  return ccg

def n_cross_correlation8_2mat(matrix_a, matrix_b, maxlag=12, window=100, disable=True): ### CCG-mean of flank
  if len(matrix_a.shape) >= 2:
    N, M = matrix_a.shape
    if len(matrix_b.shape) < 2:
      matrix_b = np.tile(matrix_b, (N, 1))
  else:
    N, M = matrix_b.shape
    matrix_a = np.tile(matrix_a, (N, 1))
  xcorr=np.zeros((2,2,N))
  norm_mata = np.nan_to_num((matrix_a-np.mean(matrix_a, axis=1).reshape(-1, 1))/(np.std(matrix_a, axis=1).reshape(-1, 1)*np.sqrt(M)))
  norm_matb = np.nan_to_num((matrix_b-np.mean(matrix_b, axis=1).reshape(-1, 1))/(np.std(matrix_b, axis=1).reshape(-1, 1)*np.sqrt(M)))
  #### padding
  norm_mata_0 = np.concatenate((norm_mata.conj(), np.zeros((N, window))), axis=1)
  norm_mata_1 = np.concatenate((np.zeros((N, window)), norm_mata.conj(), np.zeros((N, window))), axis=1)
  norm_matb_0 = np.concatenate((norm_matb.conj(), np.zeros((N, window))), axis=1)
  norm_matb_1 = np.concatenate((np.zeros((N, window)), norm_matb.conj(), np.zeros((N, window))), axis=1) # must concat zeros to the left, why???????????
  for row in tqdm(range(N), total=N, miniters=int(N/100), disable=disable): # , miniters=int(total_len/100)
    # faulthandler.enable()
    px, py = norm_mata_0[row, :], norm_matb_1[row, :]
    T = as_strided(py[window:], shape=(window+1, M + window),
                    strides=(-py.strides[0], py.strides[0])) # must be py[window:], why???????????
    # corr = np.dot(T, px)
    corr = T @ px
    corr = (corr - corr.mean())[:maxlag]
    xcorr[0, 1, row] = corr[np.argmax(np.abs(corr))]
    px, py = norm_matb_0[row, :], norm_mata_1[row, :]
    T = as_strided(py[window:], shape=(window+1, M + window),
                    strides=(-py.strides[0], py.strides[0])) # must be py[window:], why???????????
    # corr = np.dot(T, px)
    corr = T @ px
    corr = (corr - corr.mean())[:maxlag]
    xcorr[1, 0, row] = corr[np.argmax(np.abs(corr))]
  return xcorr

def n_cross_correlation_2mat(matrix_a, matrix_b, maxlag, disable): ### fastest, only causal correlation (A>B, only positive time lag on B)
  if len(matrix_a.shape) >= 2:
    N, M = matrix_a.shape
    if len(matrix_b.shape) < 2:
      matrix_b = np.tile(matrix_b, (N, 1))
  else:
    N, M = matrix_b.shape
    matrix_a = np.tile(matrix_a, (N, 1))
  xcorr=np.zeros((2,2,N))
  norm_mata = np.nan_to_num((matrix_a-np.mean(matrix_a, axis=1).reshape(-1, 1))/(np.std(matrix_a, axis=1).reshape(-1, 1)*np.sqrt(M)))
  norm_matb = np.nan_to_num((matrix_b-np.mean(matrix_b, axis=1).reshape(-1, 1))/(np.std(matrix_b, axis=1).reshape(-1, 1)*np.sqrt(M)))
  #### padding
  norm_mata_0 = np.concatenate((norm_mata.conj(), np.zeros((N, maxlag))), axis=1)
  norm_mata_1 = np.concatenate((np.zeros((N, maxlag)), norm_mata.conj(), np.zeros((N, maxlag))), axis=1)
  norm_matb_0 = np.concatenate((norm_matb.conj(), np.zeros((N, maxlag))), axis=1)
  norm_matb_1 = np.concatenate((np.zeros((N, maxlag)), norm_matb.conj(), np.zeros((N, maxlag))), axis=1) # must concat zeros to the left, why???????????
  for row in tqdm(range(N), total=N, miniters=int(N/100), disable=disable): # , miniters=int(total_len/100)
    # faulthandler.enable()
    px, py = norm_mata_0[row, :], norm_matb_1[row, :]
    T = as_strided(py[maxlag:], shape=(maxlag+1, M + maxlag),
                    strides=(-py.strides[0], py.strides[0])) # must be py[maxlag:], why???????????
    # corr = np.dot(T, px)
    corr = T @ px
    corr = corr - uniform_filter1d(corr, maxlag, mode='mirror') # mirror for the boundary values
    xcorr[0, 1, row] = corr[np.argmax(np.abs(corr))]
    px, py = norm_matb_0[row, :], norm_mata_1[row, :]
    T = as_strided(py[maxlag:], shape=(maxlag+1, M + maxlag),
                    strides=(-py.strides[0], py.strides[0])) # must be py[maxlag:], why???????????
    # corr = np.dot(T, px)
    corr = T @ px
    corr = corr - uniform_filter1d(corr, maxlag, mode='mirror') # mirror for the boundary values
    xcorr[1, 0, row] = corr[np.argmax(np.abs(corr))]
  return xcorr

def cross_correlation_delta(matrix, maxlag, disable): ### fastest, only causal correlation (A>B, only positive time lag on B), largest deviation from time window average
  N, M =matrix.shape
  xcorr=np.zeros((N,N))
  #### padding
  norm_mata = np.concatenate((matrix.conj(), np.zeros((N, maxlag))), axis=1)
  norm_matb = np.concatenate((np.zeros((N, maxlag)), matrix.conj(), np.zeros((N, maxlag))), axis=1) # must concat zeros to the left, why???????????
  total_len = len(list(itertools.permutations(range(N), 2)))
  for row_a, row_b in tqdm(itertools.permutations(range(N), 2), total=total_len , miniters=int(total_len/100), disable=disable): # , miniters=int(total_len/100)
  # for ind, (row_a, row_b) in enumerate(itertools.permutations(range(N), 2)): # , miniters=int(total_len/100)
    # faulthandler.enable()
    px, py = norm_mata[row_a, :], norm_matb[row_b, :]
    T = as_strided(py[maxlag:], shape=(maxlag+1, M + maxlag),
                    strides=(-py.strides[0], py.strides[0])) # must be py[maxlag:], why???????????
    # corr = np.dot(T, px)
    corr = T @ px
    corr = corr - uniform_filter1d(corr, maxlag, mode='mirror') # mirror for the boundary values
    xcorr[row_a, row_b] = corr[np.argmax(np.abs(corr))]
  return xcorr

def cross_correlation(matrix, maxlag, disable): ### fastest, only causal correlation (A>B, only positive time lag on B), largest
  N, M =matrix.shape
  xcorr=np.zeros((N,N))
  #### padding
  norm_mata = np.concatenate((matrix.conj(), np.zeros((N, maxlag))), axis=1)
  norm_matb = np.concatenate((np.zeros((N, maxlag)), matrix.conj(), np.zeros((N, maxlag))), axis=1) # must concat zeros to the left, why???????????
  total_len = len(list(itertools.permutations(range(N), 2)))
  for row_a, row_b in tqdm(itertools.permutations(range(N), 2), total=total_len , miniters=int(total_len/100), disable=disable): # , miniters=int(total_len/100)
  # for ind, (row_a, row_b) in enumerate(itertools.permutations(range(N), 2)): # , miniters=int(total_len/100)
    # faulthandler.enable()
    px, py = norm_mata[row_a, :], norm_matb[row_b, :]
    T = as_strided(py[maxlag:], shape=(maxlag+1, M + maxlag),
                    strides=(-py.strides[0], py.strides[0])) # must be py[maxlag:], why???????????
    # corr = np.dot(T, px)
    corr = T @ px
    xcorr[row_a, row_b] = corr[np.argmax(np.abs(corr))]
  return xcorr


def plot_heatmap_xcorr_FR(corr, bins):
  divnorm=colors.TwoSlopeNorm(vcenter=0.)
  # pcolormesh(your_data, cmap="coolwarm", norm=divnorm)
  fig, ax = plt.subplots(figsize=(7, 6))
  im = ax.imshow(corr, norm=divnorm, cmap="RdBu_r")
  ax.set_xticks(ticks=np.arange(len(bins)))
  ax.set_xticklabels(bins)
  ax.set_yticks(ticks=np.arange(len(bins)))
  ax.set_yticklabels(bins)
  fmt = lambda x, position: '{:.1f}e-3'.format(bins[x]*1e3)
  ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
  ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
  for index, label in enumerate(ax.get_xticklabels()):
    if index % 15 == 0:
        label.set_visible(True)
    else:
        label.set_visible(False)
  for index, label in enumerate(ax.get_yticklabels()):
    if index % 15 == 0:
        label.set_visible(True)
    else:
        label.set_visible(False)
  fig.colorbar(im, ax=ax)
  plt.xlabel('firing rate of source neuron', size=15)
  plt.ylabel('firing rate of target neuron', size=15)
  plt.title('cross correlation VS firing rate', size=15)
  plt.savefig('./plots/xcorr_FR_heatmap.jpg')

def plot_multi_heatmap_xcorr_FR(session_ids, stimulus_names, xcorr_dict, bin_dict):
  ind = 1
  rows, cols = session_ids, stimulus_names
  divnorm=colors.TwoSlopeNorm(vcenter=0.)
  fig = plt.figure(figsize=(5*len(cols), 5*len(rows)))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      ax = plt.subplot(len(rows), len(cols), ind)
      if row_ind == 0:
        plt.gca().set_title(cols[col_ind], fontsize=30, rotation=0)
      if col_ind == 0:
        plt.gca().text(0, 0.5 * (bottom + top), rows[row_ind],
        horizontalalignment='right',
        verticalalignment='center',
        # rotation='vertical',
        transform=plt.gca().transAxes, fontsize=30, rotation=90)
      plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
      ind += 1
      corr, bins = xcorr_dict[row][col], bin_dict[row][col]
      # pcolormesh(your_data, cmap="coolwarm", norm=divnorm)
      im = ax.imshow(corr, norm=divnorm, cmap="RdBu_r")
      ax.set_xticks(ticks=np.arange(len(bins)))
      ax.set_xticklabels(bins)
      ax.set_yticks(ticks=np.arange(len(bins)))
      ax.set_yticklabels(bins)
      fmt = lambda x, position: '{:.1f}e-3'.format(bins[x]*1e3)
      ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
      ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
      for index, label in enumerate(ax.get_xticklabels()):
        if index % 15 == 0:
            label.set_visible(True)
        else:
            label.set_visible(False)
      for index, label in enumerate(ax.get_yticklabels()):
        if index % 15 == 0:
            label.set_visible(True)
        else:
            label.set_visible(False)
      if col_ind == 7:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
      # plt.xlabel('firing rate of source neuron', size=15)
      # plt.ylabel('firing rate of target neuron', size=15)
      # plt.title('cross correlation VS firing rate', size=15)
  plt.suptitle('cross correlation VS firing rate', size=40)
  plt.tight_layout()
  plt.savefig('./plots/xcorr_FR_multi_heatmap.jpg')

def all_xcorr(matrix, window=100, disable=True): ### fastest, only causal correlation (A>B, only positive time lag on B), largest deviation from flank
  N, M =matrix.shape
  # xcorr, peak_offset=np.zeros((N,N)), np.zeros((N,N))
  xcorr=np.empty((N,N,window+1))
  xcorr[:] = np.nan
  firing_rates = np.count_nonzero(matrix, axis=1) / matrix.shape[1]
  #### padding
  norm_mata = np.concatenate((matrix.conj(), np.zeros((N, window))), axis=1)
  norm_matb = np.concatenate((np.zeros((N, window)), matrix.conj(), np.zeros((N, window))), axis=1) # must concat zeros to the left, why???????????
  total_len = len(list(itertools.permutations(range(N), 2)))
  for row_a, row_b in tqdm(itertools.permutations(range(N), 2), total=total_len , miniters=int(total_len/100), disable=disable): # , miniters=int(total_len/100)
    px, py = norm_mata[row_a, :], norm_matb[row_b, :]
    T = as_strided(py[window:], shape=(window+1, M + window),
                    strides=(-py.strides[0], py.strides[0])) # must be py[window:], why???????????
    xcorr[row_a, row_b, :] = (T @ px) / ((M-np.arange(window+1))/1000 * np.sqrt(firing_rates[row_a] * firing_rates[row_b]))
  return xcorr

def save_ccg_corrected(sequences, fname, num_jitter=10, L=25, window=100, disable=True): ### fastest, only causal correlation (A>B, only positive time lag on B), largest deviation from flank
  xcorr = all_xcorr(sequences, window, disable=disable) # N x N x window
  save_sparse_npz(xcorr, fname)
  N = sequences.shape[0]
  # jitter
  xcorr_jittered = np.zeros((N, N, window+1)) # , num_jitter, to save memory
  pj = pattern_jitter(num_sample=num_jitter, sequences=sequences, L=L, memory=False)
  sampled_matrix = pj.jitter() # num_sample x N x T
  for i in range(num_jitter):
    print(i)
    xcorr_jittered += all_xcorr(sampled_matrix[i, :, :], window, disable=disable)
  xcorr_jittered = xcorr_jittered / num_jitter
  save_sparse_npz(xcorr_jittered, fname.replace('.npz', '_bl.npz'))

def all_n_cross_correlation8(matrix, window=100, disable=True): ### fastest, only causal correlation (A>B, only positive time lag on B), largest deviation from flank
  N, M =matrix.shape
  # xcorr, peak_offset=np.zeros((N,N)), np.zeros((N,N))
  xcorr=np.empty((N,N,window+1))
  xcorr[:] = np.nan
  norm_mata = np.nan_to_num((matrix-np.mean(matrix, axis=1).reshape(-1, 1))/(np.std(matrix, axis=1).reshape(-1, 1)*np.sqrt(M)))
  norm_matb = np.nan_to_num((matrix-np.mean(matrix, axis=1).reshape(-1, 1))/(np.std(matrix, axis=1).reshape(-1, 1)*np.sqrt(M)))
  #### padding
  norm_mata = np.concatenate((norm_mata.conj(), np.zeros((N, window))), axis=1)
  norm_matb = np.concatenate((np.zeros((N, window)), norm_matb.conj(), np.zeros((N, window))), axis=1) # must concat zeros to the left, why???????????
  total_len = len(list(itertools.permutations(range(N), 2)))
  for row_a, row_b in tqdm(itertools.permutations(range(N), 2), total=total_len , miniters=int(total_len/100), disable=disable): # , miniters=int(total_len/100)
  # for ind, (row_a, row_b) in enumerate(itertools.permutations(range(N), 2)): # , miniters=int(total_len/100)
    # faulthandler.enable()
    px, py = norm_mata[row_a, :], norm_matb[row_b, :]
    T = as_strided(py[window:], shape=(window+1, M + window),
                    strides=(-py.strides[0], py.strides[0])) # must be py[window:], why???????????
    xcorr[row_a, row_b, :] = T @ px
  return xcorr
#%%
################### effect of pattern jitter on cross correlation
####### turn off warnings
np.seterr(divide='ignore', invalid='ignore')
############# save correlation matrices #################
# min_len, min_num = (260000, 739)
min_len, min_num = (10000, 29)
min_spikes = min_len * 0.002 # 2 Hz
# measure = 'pearson'
# measure = 'cosine'
# measure = 'correlation'
# measure = 'MI'
measure = 'xcorr'
# measure = 'causality'
directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
files = os.listdir(directory)
files = [f for f in files if f.endswith('.npz')]
files.sort(key=lambda x:int(x[:9]))
path = os.path.join(directory.replace('spiking_sequence', 'adj_mat_{}'.format(measure)))
if not os.path.exists(path):
  os.makedirs(path)
num_sample = 10
num_baseline = 1
# Ls = list(np.arange(2, 101))
Ls = list(np.arange(3, 51, 2)) # L should be larger than 1 and odd
# Ls = list(np.arange(3, 101, 2)) # L should be larger than 1 and odd
Rs = [1, 100, 200, 300, 400, 500]
file = files[0] # 0, 2, 7
start_time_mouse = time.time()
print(file)
mouseID = file.replace('.npz', '').split('_')[0]
stimulus = file.replace('.npz', '').replace(mouseID + '_', '')
sequences = load_npz(os.path.join(directory, file))
sequences = sequences[np.count_nonzero(sequences[:, :min_len], axis=1) > min_spikes, :min_len]
# sequences = sequences[np.count_nonzero(sequences[:, :min_len], axis=1) > 80, :min_len]
# active_inds = np.argpartition(np.count_nonzero(sequences, axis=1), -2)[-2:] # top 2 most active neurons
active_inds = np.argpartition(np.count_nonzero(sequences, axis=1), 2)[:2] # top 2 most inactive neurons
num_nodes = 2
############## Effect of pattern jitter on cross correlation
origin_adj_mat = np.zeros((2, 2))
origin_peak_off = np.zeros((2, 2))
origin_adj_mat_bl = np.zeros((2, 2, num_baseline))
all_adj_mat_A = np.zeros((2, 2, num_sample, len(Ls), len(Rs)))
all_peak_off_A = np.zeros((2, 2, num_sample, len(Ls), len(Rs)))
adj_mat_bl_A = np.zeros((2, 2, num_baseline, len(Ls), len(Rs)))
all_adj_mat_B = np.zeros((2, 2, num_sample, len(Ls), len(Rs)))
all_peak_off_B = np.zeros((2, 2, num_sample, len(Ls), len(Rs)))
adj_mat_bl_B = np.zeros((2, 2, num_baseline, len(Ls), len(Rs)))
all_adj_mat = np.zeros((2, 2, num_sample, len(Ls), len(Rs)))
all_peak_off = np.zeros((2, 2, num_sample, len(Ls), len(Rs)))
adj_mat_bl = np.zeros((2, 2, num_baseline, len(Ls), len(Rs)))
# complete_adj_mat = corr_mat(sequences, measure, maxlag=12)
T = min_len
origin_adj_mat, origin_peak_off = corr_mat(sequences[active_inds], measure, maxlag=12)
seq = sequences[active_inds].copy()
# for b in range(num_baseline):
#   for n in range(num_nodes):
#     np.random.shuffle(seq[n,:])
#   adj_mat = corr_mat(seq, measure)
#   origin_adj_mat_bl[:, :, b] = adj_mat
#%%
############################## effect of pattern jitter on CCG
np.seterr(divide='ignore', invalid='ignore')
############# save correlation matrices #################
# min_len, min_num = (260000, 739)
min_len, min_num = (10000, 29)
min_spikes = min_len * 0.002 # 2 Hz
# measure = 'pearson'
# measure = 'cosine'
# measure = 'correlation'
# measure = 'MI'
measure = 'ccg'
# measure = 'causality'
directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
files = os.listdir(directory)
files = [f for f in files if f.endswith('.npz')]
files.sort(key=lambda x:int(x[:9]))
num_sample = 1000
num_baseline = 1
# Ls = list(np.arange(2, 101))
Ls = list(np.arange(1, 51, 2)) # L should be larger than 1 and odd
# Ls = list(np.arange(3, 101, 2)) # L should be larger than 1 and odd
Rs = [1, 100, 200, 300, 400, 500]
file_order = int(sys.argv[1])
file = files[file_order]
start_time_mouse = time.time()
print(file)
mouseID = file.replace('.npz', '').split('_')[0]
stimulus = file.replace('.npz', '').replace(mouseID + '_', '')
sequences = load_npz(os.path.join(directory, file))
sequences = sequences[np.count_nonzero(sequences[:, :min_len], axis=1) > min_spikes, :min_len]
num_nodes = 2

def ccg_significant_inds(directory, mouseID, stimulus, maxlag=12, n=7, disable=False):
  file = '{}_{}.npz'.format(mouseID, stimulus)
  try: 
    ccg = load_npz_3d(os.path.join(directory, file))
  except:
    ccg = load_sparse_npz(os.path.join(directory, file))
  try:
    ccg_jittered = load_npz_3d(os.path.join(directory, file.replace('.npz', '_bl.npz')))
  except:
    ccg_jittered = load_sparse_npz(os.path.join(directory, file.replace('.npz', '_bl.npz')))
  num_nodes = ccg.shape[0]
  deviation_dict = {}
  total_len = len(list(itertools.permutations(range(num_nodes), 2)))
  for row_a, row_b in tqdm(itertools.permutations(range(num_nodes), 2), total=total_len , miniters=int(total_len/100), disable=disable): # , miniters=int(total_len/100)
    ccg_corrected = ccg[row_a, row_b, :] - ccg_jittered[row_a, row_b, :]
    if ccg_corrected[:maxlag].max() > ccg_corrected.mean() + n * ccg_corrected.std():
      deviation_dict[ccg_corrected[:maxlag].max() - (ccg_corrected.mean() + n * ccg_corrected.std())] = [row_a, row_b]
  deviation_dict = sorted(deviation_dict.items(), key = lambda kv:kv[0], reverse=True)
  return deviation_dict

measure = 'ccg'
n = 5
directory = './data/ecephys_cache_dir/sessions/adj_mat_{}_corrected/'.format(measure)
deviation_dict  =ccg_significant_inds(directory, mouseID, stimulus, maxlag=12, n=n, disable=True)
active_inds = deviation_dict[0][1]
origin_adj_mat = ccg_mat(sequences[active_inds], maxlag=12, window=100)
#%%
def ccg_pattern_jitter(sequences, active_inds, Ls, Rs, num_sample, jitter_type='A'):
  all_adj_mat = np.zeros((2, 2, num_sample, len(Ls), len(Rs)))
  if jitter_type == 'A':
    seq_2jitter = sequences[active_inds[0]]
  elif jitter_type == 'B':
    seq_2jitter = sequences[active_inds[1]]
  else:
    seq_2jitter = sequences[active_inds]
  pj = pattern_jitter(num_sample=num_sample, sequences=seq_2jitter, L=1, R=1, memory=True)
  for L_ind, L in enumerate(Ls):
    print(L)
    pj.L = L
    for R_ind, R in enumerate(Rs):
      pj.R = R
      if pj.L == 1:
        if jitter_type == 'A' or jitter_type == 'B':
          jittered_seq = np.tile(seq_2jitter, (num_sample, 1))
        else:
          jittered_seq = np.tile(seq_2jitter, (num_sample, 1, 1))
      else:
        jittered_seq = pj.jitter()
      if jitter_type == 'A':
        mat_a, mat_b = jittered_seq, sequences[active_inds[1]]
      elif jitter_type == 'B':
        mat_a, mat_b = sequences[active_inds[0]], jittered_seq
      else:
        mat_a, mat_b = jittered_seq[:, 0, :], jittered_seq[:, 1, :]
      all_adj_mat[:, :, :, L_ind, R_ind] = ccg_2mat(mat_a, mat_b)
  return all_adj_mat

start_time = time.time()
start_time_A = start_time
print('Sampling neuron A...')
all_adj_mat_A = ccg_pattern_jitter(sequences, active_inds, Ls, Rs, num_sample, jitter_type='A')
print("--- %s minutes" % ((time.time() - start_time_A)/60))
start_time_B = time.time()
print('Sampling neuron B...')
all_adj_mat_B = ccg_pattern_jitter(sequences, active_inds, Ls, Rs, num_sample, jitter_type='B')
print("--- %s minutes" % ((time.time() - start_time_B)/60))
start_time_both = time.time()
print('Sampling neurons A and B...')
all_adj_mat = ccg_pattern_jitter(sequences, active_inds, Ls, Rs, num_sample, jitter_type='both')
print("--- %s minutes" % ((time.time() - start_time_both)/60))
print("--- %s minutes in total" % ((time.time() - start_time)/60))
#%%
############## one pair of neurons, significant xcorr vs L and R
def plot_ccg_LR(origin_adj_mat, all_adj_mat_A, all_adj_mat_B, all_adj_mat, Ls, Rs, measure, mouseID, stimulus):
  plt.figure(figsize=(20, 12))
  all_mat = [all_adj_mat_A, all_adj_mat_B, all_adj_mat]
  titles = ['Pattern jittering neuron A', 'Pattern jittering neuron B', 'Pattern jittering neurons A and B']
  for ind, R in enumerate(Rs):
    plt.subplot(2, 3, ind + 1)
    for i in range(3):
      mean = np.nanmean(all_mat[i][0, 1, :, :, Rs.index(R)], axis=0)
      std = np.nanstd(all_mat[i][0, 1, :, :, Rs.index(R)], axis=0)
      plt.plot(Ls, mean, '-o', alpha=0.6, label=titles[i])
      plt.fill_between(Ls, (mean - std), (mean + std), alpha=0.2)

    plt.plot(Ls, len(Ls) * [origin_adj_mat[0, 1]], 'k--', alpha=0.6, label='original ccg')
    plt.gca().set_title('memory length R={}'.format(R), fontsize=20, rotation=0)
    plt.xscale('log')
    plt.xticks(rotation=90)
    plt.xlabel('jitter window size L', fontsize=15)
    plt.ylabel(measure, fontsize=15)
    plt.legend()
  plt.suptitle(mouseID + ', ' + stimulus, size=30)
  plt.tight_layout()
  # plt.show()
  figname = './plots/{}_vs_LR_{}_{}.jpg'.format(measure, mouseID, stimulus)
  plt.savefig(figname)

plot_ccg_LR(origin_adj_mat, all_adj_mat_A, all_adj_mat_B, all_adj_mat, Ls, Rs, measure, mouseID, stimulus)
# %%
############## one pair of neurons, significant xcorr vs L and R
def plot_corr_LR(origin_adj_mat, all_adj_mat_A, all_adj_mat_B, all_adj_mat, Ls, R, Rs, measure):
  plt.figure(figsize=(20, 6))
  all_mat = [all_adj_mat_A, all_adj_mat_B, all_adj_mat]
  titles = ['Pattern jittering neuron A', 'Pattern jittering neuron B', 'Pattern jittering neurons A and B']
  for i in range(3):
    plt.subplot(1, 3, i + 1)
    mean = np.nanmean(all_mat[i][0, 1, :, :, Rs.index(R)], axis=0)
    std = np.nanstd(all_mat[i][0, 1, :, :, Rs.index(R)], axis=0)
    plt.plot(Ls, mean, alpha=0.6, label='A->B')
    plt.fill_between(Ls, (mean - std), (mean + std), alpha=0.2)
    mean = np.nanmean(all_mat[i][1, 0, :, :, Rs.index(R)], axis=0)
    std = np.nanstd(all_mat[i][1, 0, :, :, Rs.index(R)], axis=0)
    plt.plot(Ls, mean, alpha=0.6, label='B->A')
    plt.fill_between(Ls, (mean - std), (mean + std), alpha=0.2)
    plt.plot(Ls, len(Ls) * [origin_adj_mat[0, 1]], 'b--', alpha=0.6, label='original A->B')
    plt.plot(Ls, len(Ls) * [origin_adj_mat[1, 0]], 'r--', alpha=0.6, label='original B->A')
    plt.gca().set_title(titles[i] + ', R={}'.format(R), fontsize=20, rotation=0)
    plt.xscale('log')
    plt.xticks(rotation=90)
    plt.xlabel('Bin size L', fontsize=15)
    plt.ylabel(measure, fontsize=15)
    plt.legend()
  plt.tight_layout()
  # plt.show()
  figname = './plots/{}_vs_L_R_{}_{}.jpg'.format(measure, R, measure)
  plt.savefig(figname)

for R in Rs:
  plot_corr_LR(origin_adj_mat, all_adj_mat_A, all_adj_mat_B, all_adj_mat, Ls, R, Rs, measure)

# %%
################ is cross correlation affected by firing rate?
# adj_mat = corr_mat(sequences, measure, maxlag=12, noprogressbar=False)
# np.fill_diagonal(adj_mat, np.nan)
# # adj_mat_flat = adj_mat[~np.eye(adj_mat.shape[0],dtype=bool)]
# firing_rates = np.count_nonzero(sequences, axis=1) / sequences.shape[1]
#%%
# source_FR = np.repeat(firing_rates[:, None], len(firing_rates), axis=1)
# source_FR_flat = source_FR[~np.eye(source_FR.shape[0],dtype=bool)]
# np.fill_diagonal(source_FR, np.nan)
# plt.figure()
# plt.scatter(source_FR_flat, adj_mat_flat, alpha=0.1)
# plt.scatter(np.nanmean(source_FR, axis=1), np.nanmean(adj_mat, axis=1), color='b', alpha = 0.5)
# plt.xlabel('FR of source neuron')
# plt.ylabel('cross correlation')
# plt.tight_layout()
# plt.savefig('./plots/xcorr_FR_source.jpg')
# # plt.show()
# # %%
# target_FR = np.repeat(firing_rates[None, :], len(firing_rates), axis=0)
# target_FR_flat = target_FR[~np.eye(target_FR.shape[0],dtype=bool)]
# np.fill_diagonal(target_FR, np.nan)
# plt.figure()
# plt.scatter(target_FR_flat, adj_mat_flat, alpha=0.1)
# plt.scatter(np.nanmean(target_FR, axis=0), np.nanmean(adj_mat, axis=0), color='b', alpha = 0.5)
# plt.xlabel('FR of target neuron')
# plt.ylabel('cross correlation')
# plt.tight_layout()
# plt.savefig('./plots/xcorr_FR_target.jpg')
# # plt.show()
# # %%
# avg_FR = ((firing_rates[None, :] + firing_rates[:, None]) / 2)
# avg_FR = avg_FR[~np.eye(avg_FR.shape[0],dtype=bool)]
# plt.figure()
# plt.scatter(avg_FR, adj_mat_flat, alpha=0.1)
# uniq_FR = np.unique(avg_FR)
# corr = np.zeros_like(uniq_FR)
# for i in range(len(uniq_FR)):
#   corr[i] = np.mean(adj_mat_flat[np.where(avg_FR==uniq_FR[i])])
# plt.scatter(uniq_FR, corr, color='b', alpha = 0.5)
# plt.xlabel('average FR of source and target neurons')
# plt.ylabel('cross correlation')
# plt.tight_layout()
# plt.savefig('./plots/xcorr_FR_avg.jpg')
#%%
# bins=np.logspace(start=np.log10(firing_rates.min()), stop=np.log10(firing_rates.max()+0.0001), num=50)
# bin_num = np.digitize(firing_rates, bins)
# # %%
# corr = np.zeros((len(bins), len(bins)))
# for i in range(1, len(bins)):
#   for j in range(1, len(bins)):
#     corr[i, j] = np.nanmean(adj_mat[np.where(bin_num==i)[0][:, None], np.where(bin_num==j)[0][None, :]])
# plt.xscale('log')
# %%
################### heatmap of xcrorr vs FR
# start_time = time.time()
# min_len, min_num = (10000, 29)
# min_spikes = min_len * 0.002 # 2 Hz
# # measure = 'pearson'
# # measure = 'cosine'
# # measure = 'correlation'
# # measure = 'MI'
# measure = 'xcorr'
# # measure = 'causality'
# directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
# stimulus_names = ['spontaneous', 'flashes', 
#         'drifting_gratings', 'static_gratings',
#           'natural_scenes', 'natural_movie_one', 'natural_movie_three']
# session_ids = [719161530, 750749662, 755434585, 756029989, 791319847]
# xcorr_dict, bin_dict = {}, {}
# for session_id in session_ids:
#   print(session_id)
#   xcorr_dict[session_id], bin_dict[session_id] = {}, {}
#   for stimulus_name in stimulus_names:
#     print(stimulus_name)
#     sequences = load_npz(os.path.join(directory, str(session_id) + '_' + stimulus_name + '.npz'))
#     sequences = sequences[np.count_nonzero(sequences[:, :min_len], axis=1) > min_spikes, :min_len]
#     adj_mat = corr_mat(sequences, measure, maxlag=12, noprogressbar=False)
#     np.fill_diagonal(adj_mat, np.nan)
#     # adj_mat_flat = adj_mat[~np.eye(adj_mat.shape[0],dtype=bool)]
#     firing_rates = np.count_nonzero(sequences, axis=1) / sequences.shape[1]
#     bins=np.logspace(start=np.log10(firing_rates.min()), stop=np.log10(firing_rates.max()+0.0001), num=50)
#     bin_num = np.digitize(firing_rates, bins)
#     xcorr = np.zeros((len(bins), len(bins)))
#     for i in range(1, len(bins)):
#       for j in range(1, len(bins)):
#         xcorr[i, j] = np.nanmean(adj_mat[np.where(bin_num==i)[0][:, None], np.where(bin_num==j)[0][None, :]])
#     xcorr_dict[session_id][stimulus_name] = xcorr
#     bin_dict[session_id][stimulus_name] = bins
# print("--- %s minutes in total" % ((time.time() - start_time)/60))
#%%
# plot_multi_heatmap_xcorr_FR(session_ids, stimulus_names, xcorr_dict, bin_dict)
#%%
# min_len, min_num = (10000, 29)
# min_spikes = min_len * 0.002 # 2 Hz
# # measure = 'pearson'
# # measure = 'cosine'
# # measure = 'correlation'
# # measure = 'MI'
# measure = 'xcorr'
# # measure = 'causality'
# directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
# stimulus_names = ['spontaneous', 'flashes', 
#         'drifting_gratings', 'static_gratings',
#           'natural_scenes', 'natural_movie_one', 'natural_movie_three']
# session_ids = [719161530, 750749662, 755434585, 756029989, 791319847]
# FR_dict = {}
# for session_id in session_ids:
#   print(session_id)
#   FR_dict[session_id] = {}
#   for stimulus_name in stimulus_names:
#     print(stimulus_name)
#     sequences = load_npz(os.path.join(directory, str(session_id) + '_' + stimulus_name + '.npz'))
#     sequences = sequences[np.count_nonzero(sequences[:, :min_len], axis=1) > min_spikes, :min_len]
#     firing_rates = np.count_nonzero(sequences, axis=1) / sequences.shape[1]
#     FR_dict[session_id][stimulus_name] = firing_rates
# %%
def func_powerlaw(x, m, c):
  return x**m * c
def plot_firing_rate_distributions(FR_dict, measure):
  alphas = pd.DataFrame(index=session_ids, columns=stimulus_names)
  xmins = pd.DataFrame(index=session_ids, columns=stimulus_names)
  loglikelihoods = pd.DataFrame(index=session_ids, columns=stimulus_names)
  proportions = pd.DataFrame(index=session_ids, columns=stimulus_names)
  ind = 1
  rows, cols = get_rowcol(FR_dict)
  fig = plt.figure(figsize=(4*len(cols), 3*len(rows)))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      plt.subplot(len(rows), len(cols), ind)
      if row_ind == 0:
        plt.gca().set_title(cols[col_ind], fontsize=20, rotation=0)
      if col_ind == 0:
        plt.gca().text(0, 0.5 * (bottom + top), rows[row_ind],
        horizontalalignment='left',
        verticalalignment='center',
        # rotation='vertical',
        transform=plt.gca().transAxes, fontsize=20, rotation=90)
      plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
      ind += 1
      firing_rates = FR_dict[row][col].tolist()
      hist, bin_edges = np.histogram(firing_rates, bins=50)
      bins = (bin_edges[:-1] + bin_edges[1:]) / 2
      plt.plot(bins, np.array(hist) / sum(hist),'go-')
      [alpha, xmin, L] = plfit(firing_rates, 'finite')
      proportion = np.sum(np.array(firing_rates)>=xmin)/len(firing_rates)
      alphas.loc[int(row)][col], xmins.loc[int(row)][col], loglikelihoods.loc[int(row)][col], proportions.loc[int(row)][col] = alpha, xmin, L, proportion
      C = (np.array(hist) / sum(hist))[bins>=xmin].sum() / np.power(bins[bins>=xmin], -alpha).sum()
      plt.scatter([],[], label='alpha={:.1f}'.format(alpha), s=20)
      plt.scatter([],[], label='xmin={}'.format(xmin), s=20)
      plt.scatter([],[], label='loglikelihood={:.1f}'.format(L), s=20)
      plt.plot(bins[bins>=xmin], func_powerlaw(bins[bins>=xmin], *np.array([-alpha, C])), linestyle='--', linewidth=2, color='black')
      
      plt.legend(loc='upper right', fontsize=7)
      plt.xlabel('firing rate')
      plt.ylabel('Frequency')
      plt.xscale('log')
      plt.yscale('log')
      
  plt.tight_layout()
  image_name = './plots/FR_distribution.jpg'
  # plt.show()
  plt.savefig(image_name)

# plot_firing_rate_distributions(FR_dict, measure)
# %%
# np.seterr(divide='ignore', invalid='ignore')
# ############# save correlation matrices #################
# # min_len, min_num = (260000, 739)
# min_len, min_num = (10000, 29)
# min_spikes = min_len * 0.002 # 2 Hz
# # measure = 'pearson'
# # measure = 'cosine'
# # measure = 'correlation'
# # measure = 'MI'
# measure = 'xcorr'
# # measure = 'causality'
# directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
# files = os.listdir(directory)
# files = [f for f in files if f.endswith('.npz')]
# files.sort(key=lambda x:int(x[:9]))
# path = os.path.join(directory.replace('spiking_sequence', 'adj_mat_{}'.format(measure)))
# if not os.path.exists(path):
#   os.makedirs(path)
# num_sample = 1000
# num_baseline = 2
# num = 10
# L = 10
# R_list = [1, 20, 50, 100, 500, 1000]
# T = min_len
# file = files[0]
# start_time_mouse = time.time()
# print(file)
# mouseID = file.replace('.npz', '').split('_')[0]
# stimulus = file.replace('.npz', '').replace(mouseID + '_', '')
# sequences = load_npz(os.path.join(directory, file))
# sequences = sequences[np.count_nonzero(sequences[:, :min_len], axis=1) > min_spikes, :min_len]
# # sequences = sequences[np.count_nonzero(sequences[:, :min_len], axis=1) > 80, :min_len]
# # active_inds = np.argpartition(np.count_nonzero(sequences, axis=1), -1)[-1:] # top 1 most active neurons
# active_inds = np.argpartition(np.count_nonzero(sequences, axis=1), 1)[:1] # top 1 most inactive neurons
# spikeTrain = getSpikeTrain(sequences[active_inds, :].squeeze())
# N = len(spikeTrain)
# initDist = getInitDist(L)
# tDistMatrices = getTransitionMatrices(L, N)
# Palette = np.array(['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

# colors = np.concatenate((np.array(['r']), np.repeat(Palette[:len(R_list)], num)))
# all_spiketrain = spikeTrain[None, :]
# for R in R_list:
#     print(R)
#     sampled_spiketrain = sample_spiketrain(L, R, T, spikeTrain, initDist, tDistMatrices, num)
#     all_spiketrain = np.concatenate((all_spiketrain, sampled_spiketrain[:num, :]), axis=0)
# ################ raster plot
# #%%
# text_pos = np.arange(8, 68, 10)
# fig = plt.figure(figsize=(10, 7))
# # plt.eventplot(spikeTrain, colors='b', lineoffsets=1, linewidths=1, linelengths=1)
# plt.eventplot(all_spiketrain, colors=colors, lineoffsets=1, linewidths=0.5, linelengths=0.4)
# for ind, t_pos in enumerate(text_pos):
#   plt.text(-700, t_pos, 'R={}'.format(R_list[ind]), size=10, color=Palette[ind], weight='bold')
# plt.axis('off')
# plt.gca().invert_yaxis()
# Gamma = getGamma(L, R, T, spikeTrain)
# # plt.vlines(np.concatenate((np.min(Gamma, axis=1), np.max(Gamma, axis=1))), ymin=0, ymax=num+1, colors='k', linewidth=0.2, linestyles='dashed')
# plt.tight_layout()
# plt.show()
# plt.savefig('../plots/sampled_spiketrain_L{}.jpg'.format(L))

#%%
# %%
########## whether pattern jitter changes number of spikes
# L = 5
# R = 300
# spikeTrain = getSpikeTrain(sequences[active_inds[0], :])
# N = len(spikeTrain)
# initDist = getInitDist(L)
# tDistMatrices = getTransitionMatrices(L, N)
# sampled_spiketrain1 = sample_spiketrain(L, R, T, spikeTrain, initDist, tDistMatrices, num_sample)
# spikeTrain = getSpikeTrain(sequences[active_inds[1], :])
# N = len(spikeTrain)
# initDist = getInitDist(L)
# tDistMatrices = getTransitionMatrices(L, N)
# sampled_spiketrain2 = sample_spiketrain(L, R, T, spikeTrain, initDist, tDistMatrices, num_sample)

# #%%
# st1 = spike_timing2train(min_len, sampled_spiketrain1)
# st2 = spike_timing2train(min_len, sampled_spiketrain2)
# print(np.count_nonzero(sequences[active_inds], axis=1))
# print(np.unique(np.count_nonzero(st1, axis=1)))
# print(np.unique(np.count_nonzero(st2, axis=1)))
# # %%
# adj = n_cross_correlation_2mat(spike_timing2train(min_len, sampled_spiketrain1), spike_timing2train(min_len, sampled_spiketrain2), maxlag=12, disable=True)
# adj.mean(-1)
# %%
################# normal test for xcorrs with adjacent Ls
# np.seterr(divide='ignore', invalid='ignore')
# min_len, min_num = (10000, 29)
# min_spikes = min_len * 0.002 # 2 Hz
# measure = 'xcorr'
# directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
# files = os.listdir(directory)
# files = [f for f in files if f.endswith('.npz')]
# files.sort(key=lambda x:int(x[:9]))
# path = os.path.join(directory.replace('spiking_sequence', 'adj_mat_{}'.format(measure)))
# if not os.path.exists(path):
#   os.makedirs(path)
# num_sample = 1000
# file = files[0] # 0, 2, 7
# print(file)
# sequences = load_npz(os.path.join(directory, file))
# sequences = sequences[np.count_nonzero(sequences[:, :min_len], axis=1) > min_spikes, :min_len]
# # sequences = sequences[np.count_nonzero(sequences[:, :min_len], axis=1) > 80, :min_len]
# # active_inds = np.argpartition(np.count_nonzero(sequences, axis=1), -2)[-2:] # top 2 most active neurons
# active_inds = np.argpartition(np.count_nonzero(sequences, axis=1), 2)[:2] # top 2 most inactive neurons
# print('Sampling neurons A and B...')
# start_time = time.time()
# all_xcorr = np.zeros((2, len(Ls), num_sample))
# R = 200
# for L_ind, L in enumerate(Ls):
#   spikeTrain = getSpikeTrain(sequences[active_inds[0], :])
#   N = len(spikeTrain)
#   initDist = getInitDist(L)
#   tDistMatrices = getTransitionMatrices(L, N)
#   sampled_spiketrain1 = sample_spiketrain(L, R, T, spikeTrain, initDist, tDistMatrices, num_sample)
#   spikeTrain = getSpikeTrain(sequences[active_inds[1], :])
#   N = len(spikeTrain)
#   initDist = getInitDist(L)
#   tDistMatrices = getTransitionMatrices(L, N)
#   sampled_spiketrain2 = sample_spiketrain(L, R, T, spikeTrain, initDist, tDistMatrices, num_sample)
#   adj = n_cross_correlation_2mat(spike_timing2train(min_len, sampled_spiketrain1), spike_timing2train(min_len, sampled_spiketrain2), maxlag=12, disable=True)
#   all_xcorr[0, L_ind, :] = adj[0, 1, :]
#   all_xcorr[1, L_ind, :] = adj[1, 0, :]
# print("--- %s minutes in total" % ((time.time() - start_time)/60))
# alpha = 0.05
# SW_p_A = []
# DA_p_A = []
# SW_p_B = []
# DA_p_B = []
# print('Shapiro-Wilk Test and D’Agostino’s K^2 Test for xcorr A->B...')
# for L_ind in range(len(Ls)):
#     _, p = shapiro(all_xcorr[0, L_ind, :])
#     SW_p_A.append(p)
#     _, p = normaltest(all_xcorr[0, L_ind, :])
#     DA_p_A.append(p)
# print('Shapiro-Wilk Test and D’Agostino’s K^2 Test for xcorr B->A...')
# for L_ind in range(len(Ls)):
#     _, p = shapiro(all_xcorr[1, L_ind, :])
#     SW_p_B.append(p)
#     _, p = normaltest(all_xcorr[1, L_ind, :])
#     DA_p_B.append(p)

# # %%
# ##################### plot percentage of links that follow normal distribution
# # %%
# # alpha = 0.05
# # SW = np.zeros(len(Ls))
# # DA = np.zeros(len(Ls))
# # for L_ind in range(len(Ls)):
# #   SW.loc[int(mouseID)][stimulus] = (np.array(SW_p_A) > alpha).sum() / len(SW_p)
# #   SW_bl.loc[int(mouseID)][stimulus] = (np.array(SW_p_bl) > alpha).sum() / len(SW_p_bl)
# #   DA.loc[int(mouseID)][stimulus] = (np.array(DA_p) > alpha).sum() / len(DA_p)
# #   DA_bl.loc[int(mouseID)][stimulus] = (np.array(DA_p_bl) > alpha).sum() / len(DA_p_bl)
# # %%
# plt.figure(figsize=(7, 6))
# plt.plot(Ls, SW_p_A, label='A->B', alpha=0.5)
# plt.plot(Ls, SW_p_B, label='B->A', alpha=0.5)
# plt.gca().set_title('p value of Shapiro-Wilk Test', fontsize=20, rotation=0)
# plt.xticks(rotation=90)
# plt.legend()
# plt.tight_layout()
# plt.show()
# # plt.savefig('./plots/SW_p.jpg')
# plt.figure(figsize=(7, 6))
# plt.plot(Ls, DA_p_A, label='A->B', alpha=0.5)
# plt.plot(Ls, DA_p_B, label='B->A', alpha=0.5)
# plt.gca().set_title('p value of D’Agostino’s K^2 Test', fontsize=20, rotation=0)
# plt.xticks(rotation=90)
# plt.legend()
# plt.tight_layout()
# plt.show()
# # plt.savefig('./plots/DA_p.jpg')

#%%
# #################### z test between adjacent xcorr
# pvals_A = []
# pvals_B = []
# print('Z test...')
# for L_ind in range(len(Ls) - 1):
#   xcorr_0 = ws.DescrStatsW(all_xcorr[0, L_ind, :])
#   xcorr_1 = ws.DescrStatsW(all_xcorr[0, L_ind + 1, :])
#   cm_obj = ws.CompareMeans(xcorr_0, xcorr_1)
#   zstat, z_pval = cm_obj.ztest_ind(alternative='two-sided', usevar='unequal', value=0)
#   pvals_A.append(z_pval)
#   xcorr_0 = ws.DescrStatsW(all_xcorr[1, L_ind, :])
#   xcorr_1 = ws.DescrStatsW(all_xcorr[1, L_ind + 1, :])
#   cm_obj = ws.CompareMeans(xcorr_0, xcorr_1)
#   zstat, z_pval = cm_obj.ztest_ind(alternative='two-sided', usevar='unequal', value=0)
#   pvals_B.append(z_pval)
# # %%
# alpha = 0.05
# plt.figure(figsize=(7, 6))
# plt.plot((np.array(Ls[1:]) + np.array(Ls[:-1])) / 2, pvals_A, label='A->B', alpha=0.5)
# plt.plot((np.array(Ls[1:]) + np.array(Ls[:-1])) / 2, pvals_B, label='B->A', alpha=0.5)
# plt.plot((np.array(Ls[1:]) + np.array(Ls[:-1])) / 2, [alpha]*(len(Ls)-1), 'k--', label='95%confidence level', alpha=0.5)
# plt.gca().set_title('Z test of adjacent cross correlations', fontsize=20, rotation=0)
# plt.xticks(rotation=90)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Bin size L', size=15)
# plt.ylabel('p value', size=15)
# plt.legend()
# plt.tight_layout()
# # plt.show()
# plt.savefig('./plots/z_test_adjacent_xcorr_L.jpg')
# # %%

# plt.figure(figsize=(7, 6))
# plt.plot((np.array(Ls[1:]) + np.array(Ls[:-1])) / 2, np.abs(all_xcorr[0, 1:, :].mean(-1)-all_xcorr[0, :-1, :].mean(-1)), label='A->B', alpha=0.5)
# plt.plot((np.array(Ls[1:]) + np.array(Ls[:-1])) / 2, np.abs(all_xcorr[1, 1:, :].mean(-1)-all_xcorr[1, :-1, :].mean(-1)), label='B->A', alpha=0.5)
# plt.gca().set_title('difference between adjacent cross correlations', fontsize=20, rotation=0)
# plt.xticks(rotation=90)
# plt.xscale('log')
# # plt.yscale('log')
# plt.xlabel('Bin size L', size=15)
# plt.ylabel('absolute difference', size=15)
# plt.legend()
# plt.tight_layout()
# # plt.show()
# plt.savefig('./plots/difference_adjacent_xcorr_L.jpg')
# %%
################ is cross correlation or correlation affected by firing rate?
# adj_mat = corr_mat(sequences, measure, maxlag=12, noprogressbar=False)
# np.fill_diagonal(adj_mat, np.nan)
# # adj_mat_flat = adj_mat[~np.eye(adj_mat.shape[0],dtype=bool)]
# firing_rates = np.count_nonzero(sequences, axis=1) / sequences.shape[1]
# bins=np.logspace(start=np.log10(firing_rates.min()), stop=np.log10(firing_rates.max()+0.0001), num=50)
# bin_num = np.digitize(firing_rates, bins)
# # %%
# corr = np.zeros((len(bins), len(bins)))
# for i in range(1, len(bins)):
#   for j in range(1, len(bins)):
#     corr[i, j] = np.nanmean(adj_mat[np.where(bin_num==i)[0][:, None], np.where(bin_num==j)[0][None, :]])
# plt.xscale('log')
#%%
################## heatmap of xcorr and pcrorr vs FR
start_time = time.time()
min_len, min_num = (10000, 29)
min_spikes = min_len * 0.002 # 2 Hz
# measure = 'pearson'
# measure = 'cosine'
# measure = 'correlation'
# measure = 'MI'
measure = 'xcorr'
# measure = 'causality'
directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
stimulus_names = ['spontaneous', 'flashes', 
        'drifting_gratings', 'static_gratings',
          'natural_scenes', 'natural_movie_one', 'natural_movie_three']
session_ids = [719161530, 750749662, 755434585, 756029989, 791319847]
xcorr_dict, pcorr_dict, peak_dict, bin_dict = {}, {}, {}, {}
for session_id in session_ids:
  print(session_id)
  xcorr_dict[session_id], pcorr_dict[session_id], peak_dict[session_id], bin_dict[session_id] = {}, {}, {}, {}
  for stimulus_name in stimulus_names:
    print(stimulus_name)
    sequences = load_npz(os.path.join(directory, str(session_id) + '_' + stimulus_name + '.npz'))
    sequences = sequences[np.count_nonzero(sequences[:, :min_len], axis=1) > min_spikes, :min_len]
    adj_mat, peak_off = n_cross_correlation8(sequences, disable=False)
    np.fill_diagonal(adj_mat, np.nan)
    np.fill_diagonal(peak_off, np.nan)
    # p_adj_mat = corr_mat(sequences, measure='pearson', maxlag=12, noprogressbar=False)
    # np.fill_diagonal(p_adj_mat, np.nan)
    # adj_mat_flat = adj_mat[~np.eye(adj_mat.shape[0],dtype=bool)]
    firing_rates = np.count_nonzero(sequences, axis=1) / sequences.shape[1]
    bins=np.logspace(start=np.log10(firing_rates.min()), stop=np.log10(firing_rates.max()+0.0001), num=20)
    bin_num = np.digitize(firing_rates, bins)
    xcorr = np.zeros((len(bins), len(bins)))
    corr = np.zeros((len(bins), len(bins)))
    peaks = np.zeros((len(bins), len(bins)))
    for i in range(len(bins)):
      for j in range(len(bins)):
        xcorr[i, j] = np.nanmean(adj_mat[np.where(bin_num==i)[0][:, None], np.where(bin_num==j)[0][None, :]])
        # corr[i, j] = np.nanmean(p_adj_mat[np.where(bin_num==i)[0][:, None], np.where(bin_num==j)[0][None, :]])
        peaks[i, j] = np.nanmean(peak_off[np.where(bin_num==i)[0][:, None], np.where(bin_num==j)[0][None, :]])
    xcorr_dict[session_id][stimulus_name] = xcorr
    # pcorr_dict[session_id][stimulus_name] = corr
    peak_dict[session_id][stimulus_name] = peaks
    bin_dict[session_id][stimulus_name] = bins
print("--- %s minutes in total" % ((time.time() - start_time)/60))
#%%
def plot_multi_heatmap_corr_FR(session_ids, stimulus_names, corr_dict, bin_dict, name):
  ind = 1
  rows, cols = session_ids, stimulus_names
  divnorm=colors.TwoSlopeNorm(vcenter=0.)
  fig = plt.figure(figsize=(5*len(cols), 5*len(rows)))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      ax = plt.subplot(len(rows), len(cols), ind)
      if row_ind == 0:
        plt.gca().set_title(cols[col_ind], fontsize=30, rotation=0)
      if col_ind == 0:
        plt.gca().text(0, 0.5 * (bottom + top), rows[row_ind],
        horizontalalignment='right',
        verticalalignment='center',
        # rotation='vertical',
        transform=plt.gca().transAxes, fontsize=30, rotation=90)
      plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
      ind += 1
      corr, bins = corr_dict[row][col], bin_dict[row][col]
      # pcolormesh(your_data, cmap="coolwarm", norm=divnorm)
      im = ax.imshow(corr, norm=divnorm, cmap="RdBu_r")
      ax.set_xticks(ticks=np.arange(len(bins)))
      ax.set_xticklabels(bins)
      ax.set_yticks(ticks=np.arange(len(bins)))
      ax.set_yticklabels(bins)
      fmt = lambda x, position: '{:.1f}e-3'.format(bins[x]*1e3)
      ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
      ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
      for index, label in enumerate(ax.get_xticklabels()):
        if index % 15 == 0:
            label.set_visible(True)
        else:
            label.set_visible(False)
      for index, label in enumerate(ax.get_yticklabels()):
        if index % 15 == 0:
            label.set_visible(True)
        else:
            label.set_visible(False)
      if col_ind == 7:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
      # plt.xlabel('firing rate of source neuron', size=15)
      # plt.ylabel('firing rate of target neuron', size=15)
      # plt.title('cross correlation VS firing rate', size=15)
  plt.suptitle('{} correlation VS firing rate'.format(name), size=40)
  plt.tight_layout()
  # plt.show()
  plt.savefig('./plots/{}_FR_multi_heatmap.jpg'.format(name))
# plot_multi_heatmap_corr_FR(session_ids, stimulus_names, pcorr_dict, bin_dict, 'pearson')
plot_multi_heatmap_corr_FR(session_ids, stimulus_names, xcorr_dict, bin_dict, 'cross')
plot_multi_heatmap_corr_FR(session_ids, stimulus_names, peak_dict, bin_dict, 'peakoffset')
# %%
def plot_multi_corr_FR(session_ids, stimulus_names, corr_dict, bin_dict, name):
  ind = 1
  rows, cols = session_ids, stimulus_names
  divnorm=colors.TwoSlopeNorm(vcenter=0.)
  fig = plt.figure(figsize=(5*len(cols), 5*len(rows)))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      ax = plt.subplot(len(rows), len(cols), ind)
      if row_ind == 0:
        plt.gca().set_title(cols[col_ind], fontsize=30, rotation=0)
      if col_ind == 0:
        plt.gca().text(0, 0.5 * (bottom + top), rows[row_ind],
        horizontalalignment='right',
        verticalalignment='center',
        # rotation='vertical',
        transform=plt.gca().transAxes, fontsize=30, rotation=90)
      plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
      ind += 1
      corr, bins = corr_dict[row][col], bin_dict[row][col]
      gmean_FR = np.zeros(int(len(bins)/2))
      close_corr = np.zeros(int(len(bins)/2))
      for i in range(0, len(bins), 2):
        close_corr[int(i/2)] = corr[i, i+1]
        gmean_FR[int(i/2)] = np.sqrt(bins[i] * bins[i+1])
      ax.plot(gmean_FR, close_corr, 'o-')
      r = ma.corrcoef(ma.masked_invalid(gmean_FR), ma.masked_invalid(close_corr))
      ax.text(0.1, 0.9, 'r={:.2f}'.format(r[0, 1]), fontsize=15, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
      plt.xscale('log')
  plt.suptitle('{} correlation VS firing rate'.format(name), size=40)
  plt.tight_layout()
  # plt.show()
  plt.savefig('./plots/{}_FR_multi.jpg'.format(name))
# plot_multi_corr_FR(session_ids, stimulus_names, pcorr_dict, bin_dict, 'pearson')
plot_multi_corr_FR(session_ids, stimulus_names, xcorr_dict, bin_dict, 'cross')
plot_multi_corr_FR(session_ids, stimulus_names, peak_dict, bin_dict, 'peakoffset')
# %%
# %%
####### if cross correlation at 0 time lag == pearson correlation
# a = np.random.random((5, 10))
# pcorr = np.corrcoef(a)
# dxcorr = n_cross_correlation6(a, 2, disable=True)
# xcorr = n_cross_correlation7(a, 0, disable=True)
# print(pcorr)
# print(xcorr)
# print(dxcorr)
# %%
################### heatmap of shuffled xcrorr vs FR
# start_time = time.time()
# min_len, min_num = (10000, 29)
# min_spikes = min_len * 0.002 # 2 Hz
# # measure = 'pearson'
# # measure = 'cosine'
# # measure = 'correlation'
# # measure = 'MI'
# measure = 'xcorr'
# # measure = 'causality'
# directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
# stimulus_names = ['spontaneous', 'flashes', 
#         'drifting_gratings', 'static_gratings',
#           'natural_scenes', 'natural_movie_one', 'natural_movie_three']
# session_ids = [719161530, 750749662, 755434585, 756029989, 791319847]
# xcorr_dict, bin_dict = {}, {}
# for session_id in session_ids:
#   print(session_id)
#   xcorr_dict[session_id], bin_dict[session_id] = {}, {}
#   for stimulus_name in stimulus_names:
#     print(stimulus_name)
#     sequences = load_npz(os.path.join(directory, str(session_id) + '_' + stimulus_name + '.npz'))
#     sequences = sequences[np.count_nonzero(sequences[:, :min_len], axis=1) > min_spikes, :min_len]
#     for n in range(sequences.shape[0]):
#       np.random.shuffle(sequences[n,:])
#     adj_mat = corr_mat(sequences, measure, maxlag=12, noprogressbar=False)
#     np.fill_diagonal(adj_mat, np.nan)
#     # adj_mat_flat = adj_mat[~np.eye(adj_mat.shape[0],dtype=bool)]
#     firing_rates = np.count_nonzero(sequences, axis=1) / sequences.shape[1]
#     bins=np.logspace(start=np.log10(firing_rates.min()), stop=np.log10(firing_rates.max()+0.0001), num=50)
#     bin_num = np.digitize(firing_rates, bins)
#     xcorr = np.zeros((len(bins), len(bins)))
#     for i in range(1, len(bins)):
#       for j in range(1, len(bins)):
#         xcorr[i, j] = np.nanmean(adj_mat[np.where(bin_num==i)[0][:, None], np.where(bin_num==j)[0][None, :]])
#     xcorr_dict[session_id][stimulus_name] = xcorr
#     bin_dict[session_id][stimulus_name] = bins
# print("--- %s minutes in total" % ((time.time() - start_time)/60))
# #%%
# def plot_multi_heatmap_sxcorr_FR(session_ids, stimulus_names, xcorr_dict, bin_dict):
#   ind = 1
#   rows, cols = session_ids, stimulus_names
#   divnorm=colors.TwoSlopeNorm(vcenter=0.)
#   fig = plt.figure(figsize=(5*len(cols), 5*len(rows)))
#   left, width = .25, .5
#   bottom, height = .25, .5
#   right = left + width
#   top = bottom + height
#   for row_ind, row in enumerate(rows):
#     print(row)
#     for col_ind, col in enumerate(cols):
#       ax = plt.subplot(len(rows), len(cols), ind)
#       if row_ind == 0:
#         plt.gca().set_title(cols[col_ind], fontsize=30, rotation=0)
#       if col_ind == 0:
#         plt.gca().text(0, 0.5 * (bottom + top), rows[row_ind],
#         horizontalalignment='right',
#         verticalalignment='center',
#         # rotation='vertical',
#         transform=plt.gca().transAxes, fontsize=30, rotation=90)
#       plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
#       ind += 1
#       corr, bins = xcorr_dict[row][col], bin_dict[row][col]
#       # pcolormesh(your_data, cmap="coolwarm", norm=divnorm)
#       im = ax.imshow(corr, norm=divnorm, cmap="RdBu_r")
#       ax.set_xticks(ticks=np.arange(len(bins)))
#       ax.set_xticklabels(bins)
#       ax.set_yticks(ticks=np.arange(len(bins)))
#       ax.set_yticklabels(bins)
#       fmt = lambda x, position: '{:.1f}e-3'.format(bins[x]*1e3)
#       ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
#       ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
#       for index, label in enumerate(ax.get_xticklabels()):
#         if index % 15 == 0:
#             label.set_visible(True)
#         else:
#             label.set_visible(False)
#       for index, label in enumerate(ax.get_yticklabels()):
#         if index % 15 == 0:
#             label.set_visible(True)
#         else:
#             label.set_visible(False)
#       if col_ind == 7:
#         fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#       # plt.xlabel('firing rate of source neuron', size=15)
#       # plt.ylabel('firing rate of target neuron', size=15)
#       # plt.title('cross correlation VS firing rate', size=15)
#   plt.suptitle('shuffled cross correlation VS firing rate', size=40)
#   plt.tight_layout()
#   plt.savefig('./plots/xcorr_FR_shuffled_multi_heatmap.jpg')

# plot_multi_heatmap_sxcorr_FR(session_ids, stimulus_names, xcorr_dict, bin_dict)
# #%%
# def plot_multi_scorr_FR(session_ids, stimulus_names, corr_dict, bin_dict, name):
#   ind = 1
#   rows, cols = session_ids, stimulus_names
#   divnorm=colors.TwoSlopeNorm(vcenter=0.)
#   fig = plt.figure(figsize=(5*len(cols), 5*len(rows)))
#   left, width = .25, .5
#   bottom, height = .25, .5
#   right = left + width
#   top = bottom + height
#   for row_ind, row in enumerate(rows):
#     print(row)
#     for col_ind, col in enumerate(cols):
#       ax = plt.subplot(len(rows), len(cols), ind)
#       if row_ind == 0:
#         plt.gca().set_title(cols[col_ind], fontsize=30, rotation=0)
#       if col_ind == 0:
#         plt.gca().text(0, 0.5 * (bottom + top), rows[row_ind],
#         horizontalalignment='right',
#         verticalalignment='center',
#         # rotation='vertical',
#         transform=plt.gca().transAxes, fontsize=30, rotation=90)
#       plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
#       ind += 1
#       corr, bins = corr_dict[row][col], bin_dict[row][col]
#       gmean_FR = np.zeros(int(len(bins)/2))
#       close_corr = np.zeros(int(len(bins)/2))
#       for i in range(0, len(bins), 2):
#         close_corr[int(i/2)] = corr[i, i+1]
#         gmean_FR[int(i/2)] = np.sqrt(bins[i] * bins[i+1])
#       ax.plot(gmean_FR, close_corr, 'o-')
#       r = ma.corrcoef(ma.masked_invalid(gmean_FR), ma.masked_invalid(close_corr))
#       ax.text(0.1, 0.9, 'r={:.2f}'.format(r[0, 1]), fontsize=15, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
#       plt.xscale('log')
#   plt.suptitle('shuffled {} correlation VS firing rate'.format(name), size=40)
#   plt.tight_layout()
#   # plt.show()
#   plt.savefig('./plots/{}_corr_FR_shuffled_multi.jpg'.format(name))
# # plot_multi_corr_FR(session_ids, stimulus_names, pcorr_dict, bin_dict, 'pearson')
# plot_multi_scorr_FR(session_ids, stimulus_names, xcorr_dict, bin_dict, 'cross')
#%%
start_time = time.time()
start_time_A = start_time
for L_ind, L in enumerate(Ls):
  for R_ind, R in enumerate(Rs):
    print(L, R)
    for b in range(num_baseline):
      sample_seq = sequences[active_inds, :]
      # print('Baseline {} out of {}'.format(b+1, num_baseline))
      for n in range(num_nodes):
        np.random.shuffle(sample_seq[n,:])
      adj_mat = corr_mat(sample_seq, measure, maxlag=Ls[-1])
      adj_mat_bl_A[:, :, b, L_ind, R_ind] = adj_mat
print("--- %s minutes" % ((time.time() - start_time_A)/60))
# %%
############## one pair of neurons, significant xcorr vs L and R
def plot_sxcorr_LR(origin_adj_mat, adj_mat_bl_A, Ls, R, Rs, edge_type='active'):
  plt.figure(figsize=(6, 6))
  mean = np.nanmean(adj_mat_bl_A[0, 1, :, :, Rs.index(R)], axis=0)
  std = np.nanstd(adj_mat_bl_A[0, 1, :, :, Rs.index(R)], axis=0)
  plt.plot(Ls, mean, alpha=0.6, label='A->B')
  plt.fill_between(Ls, (mean - std), (mean + std), alpha=0.2)
  mean = np.nanmean(adj_mat_bl_A[1, 0, :, :, Rs.index(R)], axis=0)
  std = np.nanstd(adj_mat_bl_A[1, 0, :, :, Rs.index(R)], axis=0)
  plt.plot(Ls, mean, alpha=0.6, label='B->A')
  plt.fill_between(Ls, (mean - std), (mean + std), alpha=0.2)
  plt.plot(Ls, len(Ls) * [origin_adj_mat[0, 1]], 'b--', alpha=0.6, label='original A->B')
  plt.plot(Ls, len(Ls) * [origin_adj_mat[1, 0]], 'r--', alpha=0.6, label='original B->A')
  plt.gca().set_title('shuffled correlation, R={}'.format(R), fontsize=20, rotation=0)
  plt.xscale('log')
  plt.xticks(rotation=90)
  plt.xlabel('Bin size L', fontsize=15)
  plt.ylabel('cross correlation', fontsize=15)
  plt.legend()
  plt.tight_layout()
  # plt.show()
  figname = './plots/xcorr_shuffled_vs_L_R_{}_{}_{}.jpg'.format(R, edge_type, measure)
  plt.savefig(figname)

for R in Rs:
  plot_sxcorr_LR(origin_adj_mat, adj_mat_bl_A, Ls, R, Rs, 'inactive')
# %%
################### effect of maxlag on cross correlation
####### turn off warnings
np.seterr(divide='ignore', invalid='ignore')
############# save correlation matrices #################
# min_len, min_num = (260000, 739)
min_len, min_num = (10000, 29)
min_spikes = min_len * 0.002 # 2 Hz
# measure = 'pearson'
# measure = 'cosine'
# measure = 'correlation'
# measure = 'MI'
measure = 'xcorr'
# measure = 'causality'
directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
files = os.listdir(directory)
files = [f for f in files if f.endswith('.npz')]
files.sort(key=lambda x:int(x[:9]))
path = os.path.join(directory.replace('spiking_sequence', 'adj_mat_{}'.format(measure)))
if not os.path.exists(path):
  os.makedirs(path)
num_sample = 1
num_baseline = 500
lags = [1, 10, 100, 500, 1000, 5000, 7000, 9999]
file = files[0] # 0, 2, 7
start_time_mouse = time.time()
print(file)
mouseID = file.replace('.npz', '').split('_')[0]
stimulus = file.replace('.npz', '').replace(mouseID + '_', '')
sequences = load_npz(os.path.join(directory, file))
sequences = sequences[np.count_nonzero(sequences[:, :min_len], axis=1) > min_spikes, :min_len]
# sequences = sequences[np.count_nonzero(sequences[:, :min_len], axis=1) > 80, :min_len]
active_inds = np.argpartition(np.count_nonzero(sequences, axis=1), -2)[-2:] # top 2 most active neurons
inactive_inds = np.argpartition(np.count_nonzero(sequences, axis=1), 2)[:2] # top 2 most inactive neurons
num_nodes = 2
active_adj_mat_bl = np.zeros((2, 2, num_baseline, len(lags)))
inactive_adj_mat_bl = np.zeros((2, 2, num_baseline, len(lags)))
#%%
start_time = time.time()
for b in range(num_baseline):
  print(b)
  active_sample_seq = sequences[active_inds, :]
  for n in range(num_nodes):
    np.random.shuffle(active_sample_seq[n,:])
  inactive_sample_seq = sequences[inactive_inds, :]
  for n in range(num_nodes):
    np.random.shuffle(inactive_sample_seq[n,:])
  for ind, lag in enumerate(lags):
    adj_mat = cross_correlation_delta(active_sample_seq, maxlag=lag, disable=True)
    active_adj_mat_bl[:, :, b, ind] = adj_mat
    adj_mat = cross_correlation_delta(inactive_sample_seq, maxlag=lag, disable=True)
    inactive_adj_mat_bl[:, :, b, ind] = adj_mat
print("--- %s minutes" % ((time.time() - start_time)/60))
# %%
def plot_sxcorr_lag(adj_mat_bl, lags, edge_type='active'):
  plt.figure(figsize=(6, 6))
  mean = np.nanmean(adj_mat_bl[0, 1, :, :], axis=0)
  std = np.nanstd(adj_mat_bl[0, 1, :, :], axis=0)
  plt.plot(lags, mean, alpha=0.6, label='A->B')
  plt.fill_between(lags, (mean - std), (mean + std), alpha=0.2)
  mean = np.nanmean(adj_mat_bl[1, 0, :, :], axis=0)
  std = np.nanstd(adj_mat_bl[1, 0, :, :], axis=0)
  plt.plot(lags, mean, alpha=0.6, label='B->A')
  plt.fill_between(lags, (mean - std), (mean + std), alpha=0.2)
  # plt.gca().set_title('shuffled correlation', fontsize=20, rotation=0)
  plt.gca().set_title(r'$\Delta$ shuffled correlation', fontsize=20, rotation=0)
  plt.xscale('log')
  plt.xticks(rotation=90)
  plt.xlabel('maximal lag', fontsize=15)
  plt.ylabel('cross correlation', fontsize=15)
  plt.legend()
  plt.tight_layout()
  # plt.show()
  # figname = './plots/xcorr_shuffled_vs_lag_{}.jpg'.format(edge_type)
  figname = './plots/delta_xcorr_shuffled_vs_lag_{}.jpg'.format(edge_type)
  plt.savefig(figname)

plot_sxcorr_lag(active_adj_mat_bl, lags, 'active')
plot_sxcorr_lag(inactive_adj_mat_bl, lags, 'inactive')
# %%
################ CCG-mean of flank of 100ms
adj_mat, origin_peak_off = n_cross_correlation8(sequences, disable=False)
# adj_mat = np.nan_to_num(adj_mat)
# np.fill_diagonal(adj_mat, 0)
# np.fill_diagonal(origin_peak_off, np.nan)
firing_rates = np.count_nonzero(sequences, axis=1) / sequences.shape[1]
#%%
############## peak with binning
firing_rates = np.count_nonzero(sequences, axis=1) / sequences.shape[1]
bins=np.logspace(start=np.log10(firing_rates.min()), stop=np.log10(firing_rates.max()+0.0001), num=50)
bin_num = np.digitize(firing_rates, bins)
peaks = np.zeros((len(bins), len(bins)))
for i in range(1, len(bins)):
  for j in range(1, len(bins)):
    peaks[i, j] = np.nanmean(origin_peak_off[np.where(bin_num==i)[0][:, None], np.where(bin_num==j)[0][None, :]])
# %%
divnorm=colors.TwoSlopeNorm(vcenter=0.)
fig, ax = plt.subplots()
im = ax.imshow(peaks, norm=divnorm, cmap="RdBu_r")
ax.set_xticks(ticks=np.arange(len(bins)))
ax.set_xticklabels(bins)
ax.set_yticks(ticks=np.arange(len(bins)))
ax.set_yticklabels(bins)
fmt = lambda x, position: '{:.1f}e-3'.format(bins[x]*1e3)
ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
for index, label in enumerate(ax.get_xticklabels()):
  if index % 15 == 0:
      label.set_visible(True)
  else:
      label.set_visible(False)
for index, label in enumerate(ax.get_yticklabels()):
  if index % 15 == 0:
      label.set_visible(True)
  else:
      label.set_visible(False)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.title('Peak correlation offset (ms)')
plt.xlabel('firing rate of source neuron')
plt.ylabel('firing rate of target neuron')
# plt.show()
plt.savefig('./plots/peakoffset_FR.jpg')
# %%
fig = plt.figure()
plt.hist(origin_peak_off.flatten(), bins=12, density=True)
plt.xlabel('peak correlation offset (ms)')
plt.ylabel('probability')
# plt.show()
plt.savefig('./plots/peakoffset_dist.jpg')
# %%
#################### save correlation matrices
def save_ccg_corrected(sequences, fname, num_jitter=10, L=25, window=100, disable=True): ### fastest, only causal correlation (A>B, only positive time lag on B), largest deviation from flank
  xcorr = all_xcorr_ccg(sequences, window, disable=disable) # N x N x window
  save_npz(xcorr, fname)
  N = sequences.shape[0]
  # jitter
  xcorr_jittered = np.zeros((N, N, window+1, num_jitter))
  pj = pattern_jitter(num_sample=num_jitter, sequences=sequences, L=L, memory=False)
  sampled_matrix = pj.jitter() # num_sample x N x T
  for i in range(num_jitter):
    print(i)
    xcorr_jittered[:, :, :, i] = all_xcorr_ccg(sampled_matrix[i, :, :], window, disable=disable)
  save_npz(xcorr_jittered, fname.replace('.npz', '_bl.npz'))

def save_xcorr_shuffled(sequences, fname, num_baseline=10, disable=True):
  N = sequences.shape[0]
  xcorr = np.zeros((N, N))
  xcorr_bl = np.zeros((N, N, num_baseline))
  adj_mat, peaks = n_cross_correlation8(sequences, disable=disable)
  xcorr = adj_mat
  save_npz(xcorr, fname)
  save_npz(peaks, fname.replace('.npz', '_peak.npz'))
  for b in range(num_baseline):
    print(b)
    sample_seq = sequences.copy()
    np.random.shuffle(sample_seq) # rowwise for 2d array
    adj_mat_bl, peaks_bl = n_cross_correlation8(sample_seq, disable=disable)
    xcorr_bl[:, :, b] = adj_mat_bl
  save_npz(xcorr_bl, fname.replace('.npz', '_bl.npz'))
#%%
#################### significant correlation
def significant_xcorr(sequences, num_baseline, alpha=0.05, sign='all'):
  N = sequences.shape[0]
  xcorr = np.zeros((N, N))
  xcorr_bl = np.zeros((N, N, num_baseline))
  adj_mat, peaks = n_cross_correlation8(sequences, disable=False)
  xcorr = adj_mat
  for b in range(num_baseline):
    print(b)
    sample_seq = sequences.copy()
    np.random.shuffle(sample_seq) # rowwise for 2d array
    # for n in range(sample_seq.shape[0]):
    #   np.random.shuffle(sample_seq[n,:])
    adj_mat_bl, peaks_bl = n_cross_correlation8(sample_seq, disable=False)
    xcorr_bl[:, :, b] = adj_mat_bl
  k = int(num_baseline * alpha) + 1 # allow int(N * alpha) random correlations larger
  significant_adj_mat, significant_peaks=np.zeros_like(xcorr), np.zeros_like(peaks)
  significant_adj_mat[:] = np.nan
  significant_peaks[:] = np.nan
  if sign == 'pos':
    indx = xcorr > np.clip(np.partition(xcorr_bl, -k, axis=-1)[:, :, -k], a_min=0, a_max=None)
  elif sign == 'neg':
    indx = xcorr < np.clip(np.partition(xcorr_bl, k-1, axis=-1)[:, :, k-1], a_min=None, a_max=0)
  elif sign == 'all':
    pos = xcorr > np.clip(np.partition(xcorr_bl, -k, axis=-1)[:, :, -k], a_min=0, a_max=None)
    neg = xcorr < np.clip(np.partition(xcorr_bl, k-1, axis=-1)[:, :, k-1], a_min=None, a_max=0)
    indx = np.logical_or(pos, neg)
  if np.sum(indx):
    significant_adj_mat[indx] = xcorr[indx]
    significant_peaks[indx] = peaks[indx]
  return significant_adj_mat, significant_peaks

def all_xcorr_ccg(matrix, window=100, disable=True): ### fastest, only causal correlation (A>B, only positive time lag on B), largest deviation from flank
  N, M =matrix.shape
  # xcorr, peak_offset=np.zeros((N,N)), np.zeros((N,N))
  xcorr=np.empty((N,N,window+1))
  xcorr[:] = np.nan
  firing_rates = np.count_nonzero(matrix, axis=1) / matrix.shape[1]
  #### padding
  norm_mata = np.concatenate((matrix.conj(), np.zeros((N, window))), axis=1)
  norm_matb = np.concatenate((np.zeros((N, window)), matrix.conj(), np.zeros((N, window))), axis=1) # must concat zeros to the left, why???????????
  total_len = len(list(itertools.permutations(range(N), 2)))
  for row_a, row_b in tqdm(itertools.permutations(range(N), 2), total=total_len , miniters=int(total_len/100), disable=disable): # , miniters=int(total_len/100)
    px, py = norm_mata[row_a, :], norm_matb[row_b, :]
    T = as_strided(py[window:], shape=(window+1, M + window),
                    strides=(-py.strides[0], py.strides[0])) # must be py[window:], why???????????
    xcorr[row_a, row_b, :] = (T @ px) / ((M-np.arange(window+1))/1000 * np.sqrt(firing_rates[row_a] * firing_rates[row_b]))
  return xcorr

def all_xcorr_xcorr(matrix, window=100, disable=True): ### fastest, only causal correlation (A>B, only positive time lag on B), largest deviation from flank
  N, M =matrix.shape
  # xcorr, peak_offset=np.zeros((N,N)), np.zeros((N,N))
  xcorr=np.empty((N,N,window+1))
  xcorr[:] = np.nan
  norm_mata = np.nan_to_num((matrix-np.mean(matrix, axis=1).reshape(-1, 1))/(np.std(matrix, axis=1).reshape(-1, 1)*np.sqrt(M)))
  norm_matb = np.nan_to_num((matrix-np.mean(matrix, axis=1).reshape(-1, 1))/(np.std(matrix, axis=1).reshape(-1, 1)*np.sqrt(M)))
  #### padding
  norm_mata = np.concatenate((norm_mata.conj(), np.zeros((N, window))), axis=1)
  norm_matb = np.concatenate((np.zeros((N, window)), norm_matb.conj(), np.zeros((N, window))), axis=1) # must concat zeros to the left, why???????????
  total_len = len(list(itertools.permutations(range(N), 2)))
  for row_a, row_b in tqdm(itertools.permutations(range(N), 2), total=total_len , miniters=int(total_len/100), disable=disable): # , miniters=int(total_len/100)
  # for ind, (row_a, row_b) in enumerate(itertools.permutations(range(N), 2)): # , miniters=int(total_len/100)
    # faulthandler.enable()
    px, py = norm_mata[row_a, :], norm_matb[row_b, :]
    T = as_strided(py[window:], shape=(window+1, M + window),
                    strides=(-py.strides[0], py.strides[0])) # must be py[window:], why???????????
    # corr = np.dot(T, px)
    corr = T @ px
    xcorr[row_a, row_b, :] = corr = corr - corr.mean()
  return xcorr

def pattern_jitter(sequences, L, R, num_sample):
  if len(sequences.shape) > 1:
    N, T = sequences.shape
    jittered_seq = np.zeros((N, T, num_sample))
    for n in range(N):
      spikeTrain = getSpikeTrain(sequences[n, :])
      ns = spikeTrain.size
      if ns:
        initDist = getInitDist(L)
        tDistMatrices = getTransitionMatrices(L, ns)
        sampled_spiketrain = sample_spiketrain(L, R, T, spikeTrain, initDist, tDistMatrices, num_sample)
        jittered_seq[n, :, :] = spike_timing2train(T, sampled_spiketrain).T
      else:
        jittered_seq[n, :, :] = np.zeros((T, num_sample))
  else:
    T = len(sequences)
    spikeTrain = getSpikeTrain(sequences)
    ns = spikeTrain.size
    initDist = getInitDist(L)
    tDistMatrices = getTransitionMatrices(L, ns)
    sampled_spiketrain = sample_spiketrain(L, R, T, spikeTrain, initDist, tDistMatrices, num_sample)
    jittered_seq = spike_timing2train(T, sampled_spiketrain).T
  return jittered_seq

def xcorr_n_fold(matrix, n=7, num_jitter=10, L=25, R=1, maxlag=12, window=100, disable=True): ### fastest, only causal correlation (A>B, only positive time lag on B), largest deviation from flank
  xcorr = all_xcorr_ccg(matrix, window, disable=disable) # N x N x window
  N = matrix.shape[0]
  significant_ccg, peak_offset=np.empty((N,N)), np.empty((N,N))
  significant_ccg[:] = np.nan
  peak_offset[:] = np.nan
  # jitter
  xcorr_jittered = np.zeros((N, N, window+1, num_jitter))
  sampled_matrix = pattern_jitter(matrix, L, R, num_jitter) # N, T, num_jitter
  for i in range(num_jitter):
    print(i)
    xcorr_jittered[:, :, :, i] = all_xcorr_ccg(sampled_matrix[:, :, i], window, disable=disable)
  total_len = len(list(itertools.permutations(range(N), 2)))
  for row_a, row_b in tqdm(itertools.permutations(range(N), 2), total=total_len , miniters=int(total_len/100), disable=disable): # , miniters=int(total_len/100)
    ccg_corrected = (xcorr[row_a, row_b, :, None] - xcorr_jittered[row_a, row_b, :, :]).mean(-1)
    if ccg_corrected[:maxlag].max() > ccg_corrected.mean() + n * ccg_corrected.std():
    # if np.max(np.abs(corr))
      max_offset = np.argmax(ccg_corrected[:maxlag])
      significant_ccg[row_a, row_b] = ccg_corrected[:maxlag][max_offset]
      peak_offset[row_a, row_b] = max_offset
  return significant_ccg, peak_offset

np.seterr(divide='ignore', invalid='ignore')
############# save correlation matrices #################
# min_len, min_num = (260000, 739)
min_len, min_num = (10000, 29)
min_spikes = min_len * 0.002 # 2 Hz
# measure = 'pearson'
# measure = 'cosine'
# measure = 'correlation'
# measure = 'MI'
measure = 'xcorr'
# measure = 'causality'
directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
files = os.listdir(directory)
files = [f for f in files if f.endswith('.npz')]
files.sort(key=lambda x:int(x[:9]))
path = os.path.join(directory.replace('spiking_sequence', 'adj_mat_{}'.format(measure)))
if not os.path.exists(path):
  os.makedirs(path)
num_sample = 200
num_baseline = 1
# Ls = list(np.arange(2, 101))
Ls = list(np.arange(3, 51, 2)) # L should be larger than 1 and odd
# Ls = list(np.arange(3, 101, 2)) # L should be larger than 1 and odd
Rs = [1, 100, 200, 300, 400, 500]
file = files[0] # 0, 2, 7
start_time_mouse = time.time()
print(file)
mouseID = file.replace('.npz', '').split('_')[0]
stimulus = file.replace('.npz', '').replace(mouseID + '_', '')
sequences = load_npz(os.path.join(directory, file))
sequences = sequences[np.count_nonzero(sequences[:, :min_len], axis=1) > min_spikes, :min_len]
#%%
start_time = time.time()
significant_adj_mat, significant_peaks = significant_xcorr(sequences, num_baseline=2, alpha=0.01, sign='all')
print("--- %s minutes" % ((time.time() - start_time)/60))
#%%
start_time = time.time()
significant_ccg, peak_offset = xcorr_n_fold(sequences, n=7, num_jitter=2, L=25, R=1, maxlag=12, window=100, disable=False)
print("--- %s minutes" % ((time.time() - start_time)/60))
# %%
np.sum(~np.isnan(significant_adj_mat))
np.sum(~np.isnan(significant_ccg))

#%%
# %%
######### plot example significant sharp peaks
matrix = sequences
L=25; R=1; maxlag=12
window = 100
num_jitter = 10
disable = False
xcorr = all_xcorr_ccg(matrix, window, disable=disable) # N x N x window
N = matrix.shape[0]
significant_ccg, peak_offset=np.empty((N,N)), np.empty((N,N))
significant_ccg[:] = np.nan
peak_offset[:] = np.nan
# jitter
xcorr_jittered = np.zeros((N, N, window+1, num_jitter))
sampled_matrix = pattern_jitter(matrix, L, R, num_jitter) # N, T, num_jitter
for i in range(num_jitter):
  print(i)
  xcorr_jittered[:, :, :, i] = all_xcorr_ccg(sampled_matrix[:, :, i], window, disable=disable)
total_len = len(list(itertools.permutations(range(N), 2)))
for row_a, row_b in tqdm(itertools.permutations(range(N), 2), total=total_len , miniters=int(total_len/100), disable=disable): # , miniters=int(total_len/100)
  ccg_corrected = (xcorr[row_a, row_b, :, None] - xcorr_jittered[row_a, row_b, :, :]).mean(-1)
  if ccg_corrected[:maxlag].max() > ccg_corrected.mean() + 7 * ccg_corrected.std():
  # if np.max(np.abs(corr))
    max_offset = np.argmax(ccg_corrected[:maxlag])
    significant_ccg[row_a, row_b] = ccg_corrected[:maxlag][max_offset]
    peak_offset[row_a, row_b] = max_offset
#%%
cnt = 0
for row_a, row_b in list(zip(*np.where(~np.isnan(significant_ccg))))[:5]:
  ccg_corrected = (xcorr[row_a, row_b, :, None] - xcorr_jittered[row_a, row_b, :, :]).mean(-1)
  fig = plt.figure()
  plt.plot(np.arange(window+1), ccg_corrected)
  plt.xlabel('time lag (ms)')
  plt.ylabel('signigicant CCG corrected')
  plt.savefig('./plots/sample_significant_ccg_{}.jpg'.format(cnt))
  # plt.show()
  cnt += 1
#%%
xcorr = all_xcorr_xcorr(matrix, window=100, disable=False)
#%%
row = '719161530'
col = 'spontaneous'
pos_G, neg_G, peak = pos_G_dict[row][col], neg_G_dict[row][col], peak_dict[row][col]
pos_A = nx.adjacency_matrix(pos_G)
neg_A = nx.adjacency_matrix(neg_G)
print(pos_A.todense())
#%%
cnt = 0
for row_a, row_b in list(zip(*np.where(pos_A.todense())))[:5]:
  print(row_a, row_b)
  fig = plt.figure()
  plt.plot(np.arange(window+1), xcorr[row_a, row_b])
  plt.axvline(x=peak[row_a, row_b], color='r', alpha=0.2)
  plt.xlabel('time lag (ms)')
  plt.ylabel('signigicant cross correlation')
  plt.savefig('./plots/sample_pos_significant_xcorr_{}.jpg'.format(cnt))
  # plt.show()
  cnt += 1
#%%
cnt = 0
for row_a, row_b in list(zip(*np.where(~np.isnan(significant_ccg))))[:5]:
  ccg_corrected = (xcorr[row_a, row_b, :, None] - xcorr_jittered[row_a, row_b, :, :]).mean(-1)
  fig = plt.figure()
  plt.plot(np.arange(window+1), ccg_corrected)
  plt.xlabel('time lag (ms)')
  plt.ylabel('signigicant CCG corrected')
  plt.savefig('./plots/sample_significant_ccg_{}.jpg'.format(cnt))
  # plt.show()
  cnt += 1
# %%
#################### save correlation matrices
#%%
np.seterr(divide='ignore', invalid='ignore')
############# save correlation matrices #################
# min_len, min_num = (260000, 739)
min_len, min_num = (10000, 29)
min_spikes = min_len * 0.002 # 2 Hz
measure = 'xcorr'
directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
num_baseline = 100
file_order = int(sys.argv[1])
files = os.listdir(directory)
files = [f for f in files if f.endswith('.npz')]
files.sort(key=lambda x:int(x[:9]))
# path = os.path.join(directory.replace('spiking_sequence', 'adj_mat_ccg_corrected'))
path = os.path.join(directory.replace('spiking_sequence', 'adj_mat_xcorr_shuffled'))
if not os.path.exists(path):
  os.makedirs(path)
file = files[file_order] # 0, 2, 7 spontaneous, gabors, natural_movie_three
print(file)
sequences = load_npz(os.path.join(directory, file))
sequences = sequences[np.count_nonzero(sequences[:, :min_len], axis=1) > min_spikes, :min_len]
fname = os.path.join(path, file)
#%%
# start_time = time.time()
# save_ccg_corrected(sequences=sequences, fname=fname, num_jitter=num_baseline, L=25, window=100, disable=False)
# print("--- %s minutes" % ((time.time() - start_time)/60))
# %%
start_time = time.time()
save_xcorr_shuffled(sequences, fname, num_baseline=num_baseline, disable=False)
print("--- %s minutes" % ((time.time() - start_time)/60))
#%%
measure = 'xcorr'
maxlag = 12
alpha = 0.01
n = 4
# sign = 'pos'
# sign = 'neg'
sign = 'all'
directory = './data/ecephys_cache_dir/sessions/adj_mat_{}_shuffled/'.format(measure)
save_xcorr_sharp_peak(directory, sign, measure, maxlag=maxlag, alpha=alpha, n=n)
#%%
area_dict, active_area_dict, mean_speed_df = load_other_data(session_ids)
directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
path = directory.replace('spiking_sequence', 'adj_mat_{}_xcorr_larger_shuffled'.format(sign))
if not os.path.exists(path):
  os.makedirs(path)
G_shuffle_dict, peak_dict = load_significant_xcorr(path, weight=True)
measure = 'xcorr'
# %%
######### split G_dict into pos and neg
pos_G_dict, neg_G_dict = split_pos_neg(G_shuffle_dict, measure=measure)
# %%
############# keep largest connected components for pos and neg G_dict
pos_G_dict = get_lcc(pos_G_dict)
neg_G_dict = get_lcc(neg_G_dict)
# %%
print_stat(pos_G_dict)
print_stat(neg_G_dict)
# %%
plot_stat(pos_G_dict, n, neg_G_dict, measure=measure)
# %%
region_connection_heatmap(pos_G_dict, 'pos', area_dict, visual_regions, measure, n)
region_connection_heatmap(neg_G_dict, 'neg', area_dict, visual_regions, measure, n)
# %%
weight = False
region_connection_delta_heatmap(pos_G_dict, 'pos', area_dict, visual_regions, measure, n, weight)
region_connection_delta_heatmap(neg_G_dict, 'neg', area_dict, visual_regions, measure, n, weight)
# %%
weight = True
region_connection_delta_heatmap(pos_G_dict, 'pos', area_dict, visual_regions, measure, n, weight)
region_connection_delta_heatmap(neg_G_dict, 'neg', area_dict, visual_regions, measure, n, weight)
# %%
############# plot all graphs with community layout and color as region #################
cc = True
plot_multi_graphs_color(pos_G_dict, 'pos', area_dict, measure, n, cc=cc)
plot_multi_graphs_color(neg_G_dict, 'neg', area_dict, measure, n, cc=cc)
# cc = True
# %%
plot_directed_multi_degree_distributions(pos_G_dict, 'pos', measure, n, weight=None, cc=False)
plot_directed_multi_degree_distributions(neg_G_dict, 'neg', measure, n, weight=None, cc=False)
# %%
plot_directed_multi_degree_distributions(pos_G_dict, 'pos', measure, n, weight='weight', cc=False)
plot_directed_multi_degree_distributions(neg_G_dict, 'neg', measure, n, weight='weight', cc=False)
#%%
measure = 'ccg'
n = 2
directory = './data/ecephys_cache_dir/sessions/adj_mat_{}_corrected/'.format(measure)
# save_ccg_corrected_sharp_peak(directory, measure, maxlag=12, n=n)
save_ccg_corrected_sharp_integral(directory, measure, maxlag=12, n=n)
#%%
start_time = time.time()
measure = 'ccg'
min_spike = 50
n = 4
max_duration = 12
directory = './data/ecephys_cache_dir/sessions/adj_mat_{}_corrected/'.format(measure)
save_ccg_corrected_highland(directory, measure, min_spike=min_spike, max_duration=max_duration, maxlag=12, n=n)
print("--- %s minutes in total" % ((time.time() - start_time)/60))
#%%
area_dict, active_area_dict, mean_speed_df = load_other_data(session_ids)
directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
path = directory.replace('spiking_sequence', 'adj_mat_ccg_sharp_peak_corrected')
if not os.path.exists(path):
  os.makedirs(path)
G_ccg_dict, peak_dict = load_sharp_peak_xcorr(path, weight=True)
measure = 'ccg'
#%%
area_dict, active_area_dict, mean_speed_df = load_other_data(session_ids)
directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
path = directory.replace('spiking_sequence', 'adj_mat_ccg_sharp_integral_corrected')
if not os.path.exists(path):
  os.makedirs(path)
G_ccg_dict = load_sharp_integral_xcorr(path, weight=True)
measure = 'ccg'
#%%
################# save area dict
mouseIDs = ['719161530','750332458','750749662','754312389','755434585','756029989','791319847','797828357']
all_visual_regions = ['VISp', 'VISl', 'VISrl', 'VISal', 'VISpm', 'VISam', 'LGd', 'LP']
save_area_speed(mouseIDs, stimulus_names, all_visual_regions)
#%%
################# save active neuron inds
min_FR = 0.002 # 2 Hz
stimulus_names = ['spontaneous', 'flashes',
        'drifting_gratings', 'static_gratings',
          'natural_scenes', 'natural_movie_one', 'natural_movie_three']
save_active_inds(min_FR, session_ids, stimulus_names)
#%%
################# save active area dict
area_dict = load_area_dict(session_ids)
save_active_area_dict(area_dict)
#%%
# active_areas = get_all_active_areas(G_ccg_dict, area_dict)
# print(active_areas)
#%%
print_stat(G_ccg_dict)
#%%
plot_stat(G_ccg_dict, n, measure=measure)
# %%
################## save community partitions and modularity/Hamiltonian VS resolution
start_time = time.time()
num_rewire = 10
resolution_list = np.arange(0, 2.1, 0.1)
# comms_dict, metrics = comms_modularity_resolution(G_ccg_dict, resolution_list, num_rewire)
comms_dict, metrics = comms_Hamiltonian_resolution(G_ccg_dict, resolution_list, num_rewire)
with open('comms_dict.pkl', 'wb') as f:
  pickle.dump(comms_dict, f)
with open('metrics.pkl', 'wb') as f:
  pickle.dump(metrics, f)
print("--- %s minutes" % ((time.time() - start_time)/60))
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
rows, cols = get_rowcol(G_ccg_dict)
plot_Hamiltonian_resolution(rows, cols, resolution_list, metrics, measure, n)
#%%
############### load community
stat_modular_structure_Hamiltonian_comms(G_ccg_dict, measure, n, resolution_list, max_neg_reso=max_reso_gnm, comms_dict=comms_dict, metrics=metrics, max_method='gnm')
stat_modular_structure_Hamiltonian_comms(G_ccg_dict, measure, n, resolution_list, max_neg_reso=max_reso_config, comms_dict=comms_dict, metrics=metrics, max_method='config')
#%%
stat_modular_structure_Hamiltonian(G_ccg_dict, measure, n, max_pos_reso=max_pos_reso_gnm, max_neg_reso=max_reso_gnm, max_method='gnm')
stat_modular_structure_Hamiltonian(G_ccg_dict, measure, n, max_pos_reso=max_pos_reso_config, max_neg_reso=max_reso_config, max_method='config')
#%%
############### purity of community with Hamiltonian
plot_Hcomm_size_purity(comms_dict, area_dict, measure, n, max_neg_reso=max_reso_gnm, max_method='gnm')
plot_Hcomm_size_purity(comms_dict, area_dict, measure, n, max_neg_reso=max_reso_config, max_method='config')
#%%
top_purity_gnm = plot_top_Hcomm_purity(G_ccg_dict, 5, area_dict, measure, n, max_pos_reso=max_pos_reso_gnm, max_neg_reso=max_reso_gnm, max_method='gnm')
top_purity_config = plot_top_Hcomm_purity(G_ccg_dict, 5, area_dict, measure, n, max_pos_reso=max_pos_reso_config, max_neg_reso=max_reso_config, max_method='config')
top_purity_config = plot_top_Hcomm_purity(G_ccg_dict, 10, area_dict, measure, n, max_pos_reso=max_pos_reso_config, max_neg_reso=max_reso_config, max_method='config')
#%%
all_purity_gnm = plot_all_Hcomm_purity(G_ccg_dict, area_dict, measure, n, max_pos_reso=max_pos_reso_gnm, max_neg_reso=max_reso_gnm, max_method='gnm')
all_purity_config = plot_all_Hcomm_purity(G_ccg_dict, area_dict, measure, n, max_pos_reso=max_pos_reso_config, max_neg_reso=max_reso_config, max_method='config')
#%%
############### weighted purity by community size
weighted_purity_gnm = plot_weighted_Hcomm_purity(G_ccg_dict, area_dict, measure, n, max_pos_reso=max_pos_reso_gnm, max_neg_reso=max_reso_gnm, max_method='gnm')
weighted_purity_config = plot_weighted_Hcomm_purity(G_ccg_dict, area_dict, measure, n, max_pos_reso=max_pos_reso_config, max_neg_reso=max_reso_config, max_method='config')
#%%
############### community structure
stat_modular_structure(G_ccg_dict, measure, n, max_reso=max_reso_gnm, max_method='gnm')
stat_modular_structure(G_ccg_dict, measure, n, max_reso=max_reso_config, max_method='config')
# abs_neg_G_dict = get_abs_weight(neg_G_dict)
# stat_modular_structure(pos_G_dict, measure, n, abs_neg_G_dict)
#%%
size_of_each_community(G_ccg_dict, 'total', measure, n, max_reso=max_reso_gnm, max_method='gnm')
size_of_each_community(G_ccg_dict, 'total', measure, n, max_reso=max_reso_config, max_method='config')
# size_of_each_community(pos_G_dict, 'positive', measure, n)
# size_of_each_community(neg_G_dict, 'negative', measure, n)
#%%
################ distribution of community size
distribution_community_size(comms_dict, measure, n, max_neg_reso=max_reso_gnm, max_method='gnm')
distribution_community_size(comms_dict, measure, n, max_neg_reso=max_reso_config, max_method='config')
# distribution_community_size(pos_G_dict, 'positive', measure, n)
# distribution_community_size(neg_G_dict, 'negative', measure, n)
#%%
distribution_community_size_mean(comms_dict, measure, n, max_neg_reso=max_reso_gnm, max_method='gnm')
distribution_community_size_mean(comms_dict, measure, n, max_neg_reso=max_reso_config, max_method='config')
#%%
# start_time = time.time()
# num_rewire = 10
# resolution_list = np.arange(0, 2.1, 0.1)
# total_metrics, total_random_metrics = modular_resolution(G_ccg_dict, resolution_list, num_rewire)
# with open('total_metrics.pkl', 'wb') as f:
#     pickle.dump(total_metrics, f)
# with open('total_random_metrics.pkl', 'wb') as f:
#     pickle.dump(total_random_metrics, f)
# abs_neg_G_dict = get_abs_weight(neg_G_dict)
# pos_neg_metrics, pos_neg_random_metrics = modular_resolution(pos_G_dict, resolution_list, num_rewire, abs_neg_G_dict)
# with open('pos_neg_metrics.pkl', 'wb') as f:
#     pickle.dump(pos_neg_metrics, f)
# with open('pos_neg_random_metrics.pkl', 'wb') as f:
#     pickle.dump(pos_neg_random_metrics, f)
# print("--- %s minutes" % ((time.time() - start_time)/60))
#%%
# rows, cols = get_rowcol(G_ccg_dict)
# with open('total_metrics.pkl', 'rb') as f:
#     total_metrics = pickle.load(f)
# with open('total_random_metrics.pkl', 'rb') as f:
#     total_random_metrics = pickle.load(f)
# with open('pos_neg_metrics.pkl', 'rb') as f:
#     pos_neg_metrics = pickle.load(f)
# with open('pos_neg_random_metrics.pkl', 'rb') as f:
#     pos_neg_random_metrics = pickle.load(f)
# plot_modularity_resolution_rand(rows, cols, resolution_list, total_metrics, total_random_metrics, measure, n)
# plot_modularity_resolution_rand(rows, cols, resolution_list, pos_neg_metrics, pos_neg_random_metrics, measure, n)
#%%
rows, cols = get_rowcol(G_ccg_dict)
with open('comms_dict.pkl', 'rb') as f:
    comms_dict = pickle.load(f)
with open('metrics.pkl', 'rb') as f:
    metrics = pickle.load(f)
plot_modularity_resolution(rows, cols, resolution_list, metrics, measure, n)
plot_comm_diff_resolution(rows, cols, resolution_list, comms_dict, metrics, measure, n)
#%%
# plot_region_size(G_ccg_dict, area_dict, visual_regions, measure, n, 'total')
# plot_region_size(pos_G_dict, area_dict, visual_regions, measure, n, 'pos')
# plot_region_size(neg_G_dict, area_dict, visual_regions, measure, n, 'neg')
#%%
################# boxplot of region size
plot_region_size_box(G_ccg_dict, area_dict, visual_regions, measure, n, 'total')
plot_region_size_box(pos_G_dict, area_dict, visual_regions, measure, n, 'pos')
plot_region_size_box(neg_G_dict, area_dict, visual_regions, measure, n, 'neg')
#%%
################# boxplot of region degree
plot_region_degree(G_ccg_dict, area_dict, visual_regions, measure, n, 'total')
plot_region_degree(pos_G_dict, area_dict, visual_regions, measure, n, 'pos')
plot_region_degree(neg_G_dict, area_dict, visual_regions, measure, n, 'neg')
#%%
############### percentage of region in large communities
# region_large_comm(G_ccg_dict, area_dict, visual_regions, measure, n, max_reso=max_reso_gnm, max_method='gnm')
# region_large_comm(G_ccg_dict, area_dict, visual_regions, measure, n, max_reso=max_reso_swap, max_method='swap')
# abs_neg_G_dict = get_abs_weight(neg_G_dict)
# region_large_comm(pos_G_dict, area_dict, visual_regions, measure, n, abs_neg_G_dict)
#%%
############### boxplot percentage of region in large communities
region_larg_comm_box(G_ccg_dict, area_dict, visual_regions, measure, n, 'total', weight=False, max_reso=max_reso_gnm, max_method='gnm')
region_larg_comm_box(G_ccg_dict, area_dict, visual_regions, measure, n, 'total', weight=False, max_reso=max_reso_swap, max_method='swap')
region_larg_comm_box(pos_G_dict, area_dict, visual_regions, measure, n, 'pos', weight=False)
region_larg_comm_box(pos_G_dict, area_dict, visual_regions, measure, n, 'pos', weight=True)
abs_neg_G_dict = get_abs_weight(neg_G_dict)
region_larg_comm_box(abs_neg_G_dict, area_dict, visual_regions, measure, n, 'neg', weight=False)
region_larg_comm_box(abs_neg_G_dict, area_dict, visual_regions, measure, n, 'neg', weight=True)
#%%
################# purity VS community size
plot_comm_size_purity(G_ccg_dict, area_dict, measure, n, 'total', weight=False, max_reso=max_reso_gnm, max_method='gnm')
plot_comm_size_purity(G_ccg_dict, area_dict, measure, n, 'total', weight=False, max_reso=max_reso_swap, max_method='swap')
plot_comm_size_purity(pos_G_dict, area_dict, measure, n, 'pos', weight=False)
plot_comm_size_purity(pos_G_dict, area_dict, measure, n, 'pos', weight=True)
abs_neg_G_dict = get_abs_weight(neg_G_dict)
plot_comm_size_purity(abs_neg_G_dict, area_dict, measure, n, 'neg', weight=False)
plot_comm_size_purity(abs_neg_G_dict, area_dict, measure, n, 'neg', weight=True)
#%%
plot_top_comm_purity(G_ccg_dict, 1, area_dict, measure, n, 'total', weight=False, max_reso=max_reso_gnm, max_method='gnm')
plot_top_comm_purity(G_ccg_dict, 1, area_dict, measure, n, 'total', weight=False, max_reso=max_reso_swap, max_method='swap')
plot_top_comm_purity(G_ccg_dict, 3, area_dict, measure, n, 'total', weight=False, max_reso=max_reso_gnm, max_method='gnm')
plot_top_comm_purity(G_ccg_dict, 3, area_dict, measure, n, 'total', weight=False, max_reso=max_reso_swap, max_method='swap')
plot_top_comm_purity(G_ccg_dict, 5, area_dict, measure, n, 'total', weight=False, max_reso=max_reso_gnm, max_method='gnm')
plot_top_comm_purity(G_ccg_dict, 5, area_dict, measure, n, 'total', weight=False, max_reso=max_reso_swap, max_method='swap')
plot_top_comm_purity(G_ccg_dict, 10, area_dict, measure, n, 'total', weight=False, max_reso=max_reso_gnm, max_method='gnm')
plot_top_comm_purity(G_ccg_dict, 10, area_dict, measure, n, 'total', weight=False, max_reso=max_reso_swap, max_method='swap')
#%%
plot_all_comm_purity(G_ccg_dict, area_dict, measure, n, 'total', weight=False, max_reso=max_reso_gnm, max_method='gnm')
plot_all_comm_purity(G_ccg_dict, area_dict, measure, n, 'total', weight=False, max_reso=max_reso_swap, max_method='swap')
#%%
################# two-sample Kolmogorov-Smirnov test for purity vs pairwise stimuli
plot_top_comm_purity_kstest(G_ccg_dict, 1, area_dict, measure, n, 'total', weight=False, max_reso=max_reso_gnm, max_method='gnm')
plot_top_comm_purity_kstest(G_ccg_dict, 1, area_dict, measure, n, 'total', weight=False, max_reso=max_reso_swap, max_method='swap')
plot_top_comm_purity_kstest(G_ccg_dict, 3, area_dict, measure, n, 'total', weight=False, max_reso=max_reso_gnm, max_method='gnm')
plot_top_comm_purity_kstest(G_ccg_dict, 3, area_dict, measure, n, 'total', weight=False, max_reso=max_reso_swap, max_method='swap')
plot_top_comm_purity_kstest(G_ccg_dict, 5, area_dict, measure, n, 'total', weight=False, max_reso=max_reso_gnm, max_method='gnm')
plot_top_comm_purity_kstest(G_ccg_dict, 5, area_dict, measure, n, 'total', weight=False, max_reso=max_reso_swap, max_method='swap')
plot_top_comm_purity_kstest(G_ccg_dict, 10, area_dict, measure, n, 'total', weight=False, max_reso=max_reso_gnm, max_method='gnm')
plot_top_comm_purity_kstest(G_ccg_dict, 10, area_dict, measure, n, 'total', weight=False, max_reso=max_reso_swap, max_method='swap')
#%%
################ plot purity of communities with similar size vs stimulus
plot_similar_purity(G_ccg_dict, area_dict, 5, measure, n, 'total', weight=False, max_reso=max_reso_gnm, max_method='gnm')
plot_similar_purity(G_ccg_dict, area_dict, 10, measure, n, 'total', weight=False, max_reso=max_reso_gnm, max_method='gnm')
plot_similar_purity(G_ccg_dict, area_dict, 5, measure, n, 'total', weight=False, max_reso=max_reso_swap, max_method='swap')
plot_similar_purity(G_ccg_dict, area_dict, 10, measure, n, 'total', weight=False, max_reso=max_reso_swap, max_method='swap')
#%%w
G_ccg_lcc_dict = get_lcc(G_ccg_dict)
plot_size_lcc(G_ccg_dict, G_ccg_lcc_dict)
#%%
weight = False
region_connection_seperate_diagonal(G_ccg_dict, 'total', area_dict, visual_regions, measure, n, weight)
#%%
weight = True
region_connection_seperate_diagonal(G_ccg_dict, 'total', area_dict, visual_regions, measure, n, weight)
#%%
############# plot degree distribution with neurons from all mice
plot_directed_degree_distributions(G_ccg_dict, measure, n, weight=None, cc=False)
#%%
############# comparison between top n in degree and out degree
def top_in_out_degree(G_dict, measure, n, topn, weight=None, cc=False):
  rows, cols = get_rowcol(G_dict)
  fig, ax = plt.subplots()
  for col_ind, col in enumerate(cols):
    print(col)
    in_degree_seq, out_degree_seq = [], []
    for row_ind, row in enumerate(rows):
      G = G_dict[row][col] if col in G_dict[row] else nx.DiGraph()
      if cc:
        if nx.is_directed(G):
          Gcc = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
          G = G.subgraph(Gcc[0])
        else:
          Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
          G = G.subgraph(Gcc[0])
      nodes = G.nodes()
      in_degree = dict(G.in_degree(weight=weight))
      in_degseq=[in_degree.get(k,0) for k in nodes]
      out_degree = dict(G.out_degree(weight=weight))
      out_degseq=[out_degree.get(k,0) for k in nodes]
      in_degree_seq += np.partition(in_degseq, -topn)[-topn:].tolist()
      out_degree_seq += np.partition(out_degseq, -topn)[-topn:].tolist()
    # in_degrees, in_degree_freq = seq2histogram(in_degree_seq, weight=weight)
    # out_degrees, out_degree_freq = seq2histogram(out_degree_seq, weight=weight)
    # if weight == None:
    #   in_degrees, in_degree_freq = in_degrees[1:], in_degree_freq[1:]
    #   out_degrees, out_degree_freq = out_degrees[1:], out_degree_freq[1:]
    # print(in_degree_seq, out_degree_seq)
    plt.scatter(in_degree_seq, out_degree_seq, label=col, alpha=0.8)
  plt.legend(fontsize=8)
  plt.plot(np.arange(*ax.get_ylim()), np.arange(*ax.get_ylim()), 'k--')
  xlabel = 'Weighted Degree' if weight is not None else 'Degree'
  plt.xlabel('in'+xlabel)
  plt.ylabel('out'+xlabel)
  # plt.xscale('symlog')
  # plt.yscale('log')
  
  plt.tight_layout()
  image_name = './plots/in_out_weighted_degree_top{}_{}_{}fold.jpg' if weight is not None else './plots//in_out_degree_top{}_{}_{}fold.jpg'
  # plt.show()
  plt.savefig(image_name.format(topn, measure, n), dpi=300)

top_in_out_degree(G_ccg_dict, measure, n, 2, weight=None, cc=False)
#%%
############# comparison between in degree and out degree of total hub neurons
def hub_in_out_degree(G_dict, measure, n, significance, weight=None, cc=False):
  rows, cols = get_rowcol(G_dict)
  fig, ax = plt.subplots()
  for col_ind, col in enumerate(cols):
    print(col)
    in_degree_seq, out_degree_seq = [], []
    for row_ind, row in enumerate(rows):
      G = G_dict[row][col] if col in G_dict[row] else nx.DiGraph()
      if cc:
        if nx.is_directed(G):
          Gcc = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
          G = G.subgraph(Gcc[0])
        else:
          Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
          G = G.subgraph(Gcc[0])
      degree = dict(G.degree(weight=weight))
      degree = dict(sorted(degree.items(), key=lambda item: item[1], reverse=True))
      hub_threshold = np.mean(list(degree.values())) + significance * np.std(list(degree.values()))
      hubs = [k for k, v in degree.items() if v > hub_threshold]
      in_degree = dict(G.in_degree(weight=weight))
      in_degseq=[in_degree.get(k,0) for k in hubs]
      out_degree = dict(G.out_degree(weight=weight))
      out_degseq=[out_degree.get(k,0) for k in hubs]
      in_degree_seq += in_degseq
      out_degree_seq += out_degseq    
    plt.scatter(in_degree_seq, out_degree_seq, label=col, alpha=0.8)
  plt.legend(fontsize=8)
  plt.plot(np.arange(*ax.get_ylim()), np.arange(*ax.get_ylim()), 'k--')
  xlabel = 'Weighted Degree' if weight is not None else 'Degree'
  plt.xlabel('in'+xlabel)
  plt.ylabel('out'+xlabel)
  # plt.xscale('symlog')
  # plt.yscale('log')
      
  plt.tight_layout()
  image_name = './plots/hub_in_out_weighted_degree_top{}_{}_{}fold.jpg' if weight is not None else './plots//hub_in_out_degree_top{}_{}_{}fold.jpg'
  # plt.show()
  plt.savefig(image_name.format(significance, measure, n), dpi=300)

hub_in_out_degree(G_ccg_dict, measure, n, 5, weight=None, cc=False)
#%%
############# comparison between in degree and out degree of in/out hub neurons
def in_out_hub_degree(G_dict, measure, n, significance, weight=None, cc=False):
  rows, cols = get_rowcol(G_dict)
  fig, ax = plt.subplots()
  in_degree_seq, out_degree_seq = [], []
  df = pd.DataFrame(columns=['stimulus', 'degree', 'type'])
  for col_ind, col in enumerate(cols):
    print(col)
    in_degree_seq.append([])
    out_degree_seq.append([])
    for row_ind, row in enumerate(rows):
      G = G_dict[row][col] if col in G_dict[row] else nx.DiGraph()
      if cc:
        if nx.is_directed(G):
          Gcc = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
          G = G.subgraph(Gcc[0])
        else:
          Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
          G = G.subgraph(Gcc[0])
      in_degree = dict(G.in_degree(weight=weight))
      out_degree = dict(G.out_degree(weight=weight))
      inhub_threshold = np.mean(list(in_degree.values())) + significance * np.std(list(in_degree.values()))
      inhubs = [k for k, v in in_degree.items() if v > inhub_threshold]
      outhub_threshold = np.mean(list(out_degree.values())) + significance * np.std(list(out_degree.values()))
      outhubs = [k for k, v in out_degree.items() if v > outhub_threshold]
      in_degseq=[in_degree.get(k,0) for k in inhubs]
      out_degseq=[out_degree.get(k,0) for k in outhubs]
      # in_degree_seq[-1] += in_degseq
      # out_degree_seq[-1] += out_degseq
      df = pd.concat([df, pd.DataFrame(np.concatenate((np.array([col] * len(in_degseq))[:,None], np.array(in_degseq)[:,None], np.array(['in degree of hub neurons'] * len(in_degseq))[:,None]), 1), columns=['stimulus', 'degree', 'type']), 
                      pd.DataFrame(np.concatenate((np.array([col] * len(out_degseq))[:,None], np.array(out_degseq)[:,None], np.array(['out degree of hub neurons'] * len(out_degseq))[:,None]), 1), columns=['stimulus', 'degree', 'type'])], ignore_index=True)
  df['degree'] = pd.to_numeric(df['degree'])
  
  ax = sns.violinplot(x="stimulus", y='degree', hue="type", data=df, palette="Set3", cut=0)
  ax.set(xlabel=None)
  plt.xticks(rotation=90)
  # plt.xscale('symlog')
  # plt.yscale('log')
      
  plt.tight_layout()
  image_name = './plots/in_out_hub_weighted_degree_top{}_{}_{}fold.jpg' if weight is not None else './plots//in_out_hub_degree_top{}_{}_{}fold.jpg'
  # plt.show()
  plt.savefig(image_name.format(significance, measure, n), dpi=300)

in_out_hub_degree(G_ccg_dict, measure, n, 3, weight=None, cc=False)
#%%
############# communication efficiency
def multi_comm_efficiency(G_dict, measure, n, significance, cc=False):
  ind = 1
  rows, cols = get_rowcol(G_dict)
  G_sample = G_dict[rows[0]][cols[0]]
  dire = True if nx.is_directed(G_sample) else False
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
      G = G_dict[row][col].copy() if col in G_dict[row] else nx.DiGraph()
      if cc:
        if nx.is_directed(G):
          Gcc = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
          G = G.subgraph(Gcc[0])
        else:
          Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
          G = G.subgraph(Gcc[0])
      # node_idx = sorted(active_area_dict[row].keys())
      # reverse_mapping = {node_idx[i]:i for i in range(len(node_idx))}
      # G = nx.relabel_nodes(G, reverse_mapping)
      degree = dict(G.degree(weight=weight))
      hub_threshold = np.mean(list(degree.values())) + significance * np.std(list(degree.values()))
      hubs = [k for k, v in degree.items() if v > hub_threshold]
      distri = []
      for hub in hubs:
        p = nx.shortest_path_length(G, source=hub, weight='delay')
        distri += [v for k, v in p.items() if v > 0]
      sns.histplot(data=distri, kde=True, linewidth=0)
  plt.suptitle('hub significance level {} fold'.format(significance), size=50)
  plt.tight_layout(rect=[0, 0.03, 1, 0.98])
  plt.savefig('./plots/multi_comm_efficiency_{}_{}_{}fold.jpg'.format(significance, measure, n))

multi_comm_efficiency(S_ccg_dict, measure, n, 3, cc=False)
#%%
from scipy.optimize import minimize
from scipy.special import factorial
from scipy.optimize import curve_fit
from scipy import stats


def poisson(k, lamb):
    """poisson pdf, parameter lamb is the fit parameter"""
    return (lamb**k/factorial(k)) * np.exp(-lamb)

def negative_log_likelihood(params, data):
    """
    The negative log-Likelihood-Function
    """
    lnl = - np.sum(np.log(poisson(data, params[0])))
    return lnl

def negative_log_likelihood(params, data):
    ''' better alternative using scipy '''
    return -stats.poisson.logpmf(data, params[0]).sum()

def gauss(x, mu, sigma, A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)

def bimodal(x, loc, scale, mu2, sigma2, A2):
    return stats.expon.pdf(x, loc, scale)+gauss(x,mu2,sigma2,A2)

def fit_distribution(data, distribution='gamma'):
  if distribution=='gamma':
    xx = np.linspace(1,max(data),int(max(data)))
    ag,bg,cg = stats.gamma.fit(data)  
    pdf = stats.gamma.pdf(xx, ag, bg,cg)  
  elif distribution=='poisson':
    result = minimize(negative_log_likelihood,  # function to minimize
                      x0=np.ones(1),            # start value
                      args=(data,),             # additional arguments for function
                      method='Powell',          # minimization method, see docs
                      )
    x_plot = np.linspace(1,max(data),100)
    pdf = stats.poisson.pmf(x_plot, result.x)
  elif distribution=='exponential':
    P = stats.expon.fit(data)
    xx = np.linspace(1,max(data),100)
    pdf = stats.expon.pdf(xx, *P)
    print(P)
  elif distribution=='powerlaw':
    P = stats.powerlaw.fit(data)
    xx = np.linspace(1,max(data),100)
    pdf = stats.powerlaw.pdf(xx, *P)
  elif distribution=='lognormal':
    P = stats.lognorm.fit(data)
    xx = np.linspace(1,max(data),100)
    pdf = stats.lognorm.pdf(xx, *P)
  elif distribution=='mixed':
    y, x=np.histogram(data, 100)
    x=(x[1:]+x[:-1])/2 # for len(x)==len(y)
    expected = (1.0, 5.0, 10, 3, .05)
    params, cov = curve_fit(bimodal, x, y, expected)
    xx = np.linspace(x.min(), x.max(), 500)
    pdf = bimodal(xx, *params)
  return xx, pdf

def get_distri(G, weight, node_type, significance):
  degree = dict(G.degree(weight=weight))
  if node_type=='hub':
    hub_threshold = np.mean(list(degree.values())) + significance * np.std(list(degree.values()))
    nodes = [k for k, v in degree.items() if v > hub_threshold]
  elif node_type=='all':
    nodes = G.nodes()
  temp_distri = []
  for node in nodes:
    p = nx.shortest_path_length(G, source=node, weight='delay')
    temp_distri += [v for k, v in p.items() if v > 0]
  return temp_distri

############# communication efficiency
def get_comm_efficiency(G_dict, weight, node_type='hub', significance=3, algorithm=None, cc=False):
  rows, cols = get_rowcol(G_dict)
  distri = {}
  for col_ind, col in enumerate(cols):
    print(col)
    distri[col] = []
    for row_ind, row in enumerate(rows):
      G = G_dict[row][col].copy() if col in G_dict[row] else nx.DiGraph()
      if cc:
        if nx.is_directed(G):
          Gcc = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
          G = G.subgraph(Gcc[0])
        else:
          Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
          G = G.subgraph(Gcc[0])
      if algorithm is None:
        distri[col] += get_distri(G, weight, node_type, significance)
      else:
        G_rands = random_graph_generator(G, 10, algorithm, weight='delay', cc=False, Q=100)
        for G_rand in G_rands:
          distri[col] += get_distri(G_rand, weight, node_type, significance)
  return distri

# comm_efficiency(S_ccg_dict, measure, n, None, 'hub', 3, cc=False)
# comm_efficiency(S_ccg_dict, measure, n, None, 'all', cc=False)
distri = get_comm_efficiency(S_ccg_dict, None, 'all', 3, algorithm=None, cc=False)
# distri_gnm = get_comm_efficiency(S_ccg_dict, None, 'all', 3, algorithm='Gnm', cc=False)
# distri_swap = get_comm_efficiency(S_ccg_dict, None, 'all', 3, algorithm='directed_double_edge_swap', cc=False)
#%%
def plot_comm_efficiency(distri, measure, n, node_type, significance, algorithm=None):
  cols = list(distri.keys())
  fig = plt.figure(figsize=(9*len(cols)/2, 6*2))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  ind = 1
  for col in distri:
    print(col)
    ax=plt.subplot(2, int(np.ceil(len(cols)/2)), ind)
    plt.gca().set_title(col, fontsize=30, rotation=0)
    plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ind += 1
    sns.histplot(data=distri[col], linewidth=0, stat='probability', binwidth=1)
    plt.axvline(x=np.nanmean(distri[col]), color='r', linestyle='--')
    # sns.kdeplot(data=distri)
    if col in ['spontaneous']:
      distribution = 'gamma'# gamma for G and Gnm, exponential for dswap
    if col in ['flash_dark', 'flash_light', 'natural_movie_one', 'natural_movie_three']:
      distribution = 'lognormal'
    elif col in ['drifting_gratings', 'static_gratings', 'natural_scenes']:
      distribution = 'exponential'
    xx, pdf = fit_distribution(distri[col], distribution)    
    plt.plot(xx,pdf,'r-', lw=2, alpha=0.6, label='gamma pdf')
    plt.xlabel('communication delay (ms)', size=20)
    plt.ylabel('probability', size=20)
    ax.text(0.5, 0.5, distribution, size=12, ha="center", 
         transform=ax.transAxes)
    # plt.text(2, 0.65, distribution)
  suptitle = '{} neurons'.format(node_type)
  if algorithm is not None:
    suptitle = suptitle.replace(node_type, '{} '.format(algorithm)+node_type)
  if node_type=='hub':
    suptitle += ' significance level {} fold'.format(significance)
  plt.suptitle(suptitle, size=50)
  plt.tight_layout(rect=[0, 0.03, 1, 0.98])
  figname = './plots/comm_efficiency_{}_{}_{}fold.jpg'.format(node_type, measure, n)
  if algorithm is not None:
    figname = figname.replace('efficiency', 'efficiency{}'.format(algorithm))
  if node_type=='hub':
    figname = figname.replace('hub', 'hub_{}'.format(significance))
  plt.savefig(figname)

# plot_comm_efficiency(distri, measure, n, 'all', 3, algorithm=None)
# plot_comm_efficiency(distri_gnm, measure, n, 'all', 3, algorithm='Gnm')
plot_comm_efficiency(distri_swap, measure, n, 'all', 3, algorithm='directed_double_edge_swap')
#%%
############# compare average of communication delay of G, Gnm and Gswap
def comm_efficiency_comparison(distri, distri_gnm, distri_swap, measure, n):
  df = pd.DataFrame(columns=['stimulus', 'mean communication delay (ms)', 'network'])
  for col in distri:
    df = pd.concat([df, pd.DataFrame(np.concatenate((np.array([col])[:,None], np.array([np.mean(distri[col])])[:,None], np.array(['real network'])[:,None]), 1), columns=['stimulus', 'mean communication delay (ms)', 'network'])], ignore_index=True)
    df = pd.concat([df, pd.DataFrame(np.concatenate((np.array([col])[:,None], np.array([np.mean(distri_gnm[col])])[:,None], np.array(['Gnm'])[:,None]), 1), columns=['stimulus', 'mean communication delay (ms)', 'network'])], ignore_index=True)
    df = pd.concat([df, pd.DataFrame(np.concatenate((np.array([col])[:,None], np.array([np.mean(distri_swap[col])])[:,None], np.array(['swap'])[:,None]), 1), columns=['stimulus', 'mean communication delay (ms)', 'network'])], ignore_index=True)
  df['mean communication delay (ms)'] = pd.to_numeric(df['mean communication delay (ms)'])
  ax = sns.barplot(x='stimulus', y='mean communication delay (ms)', hue='network', data=df)
  ax.set(xlabel=None)
  plt.xticks(rotation=90)
  plt.tight_layout()
  plt.savefig('./plots/comm_efficiency_comparison_{}_{}fold.jpg'.format(measure, n))
comm_efficiency_comparison(distri, distri_gnm, distri_swap, measure, n)
#%%
#%%
############# compare average of communication delay of G, Gnm and Gswap
def comm_efficiency_scatter(distri, distri_gnm, distri_swap, measure, n):
  color_list = ['tab:blue', 'darkorange', 'bisque', 'limegreen', 'darkgreen', 'maroon', 'indianred', 'mistyrose']
  for col in distri:
    plt.scatter(np.mean(distri[col])/np.mean(distri_gnm[col]), np.mean(distri[col])/np.mean(distri_swap[col]), label=col, color=color_list[cols.index(col)])
  plt.legend()
  plt.xlabel('mean delay ratio real network/Gnm')
  plt.ylabel('mean delay ratio real network/Gswap')
  # plt.plot(np.arange(*plt.gca().get_ylim()), np.arange(*plt.gca().get_ylim()), 'k--')
  plt.tight_layout()
  plt.savefig('./plots/comm_efficiency_scatter_{}_{}fold.jpg'.format(measure, n))
comm_efficiency_scatter(distri, distri_gnm, distri_swap, measure, n)
#%%
plot_directed_multi_degree_distributions(G_ccg_dict, 'total', measure, n, weight='weight', cc=False)
# plot_directed_multi_degree_distributions(pos_G_dict, 'pos', measure, n, weight='weight', cc=False)
# plot_directed_multi_degree_distributions(neg_G_dict, 'neg', measure, n, weight='weight', cc=False)
# %%
############# plot all graphs with community layout and color as region #################
cc = True
plot_multi_graphs_color(G_ccg_dict, max_reso_gnm, offset_dict, 'total_offset_gnm', area_dict, active_area_dict, measure, n, cc=cc)
plot_multi_graphs_color(G_ccg_dict, max_reso_swap, offset_dict, 'total_offset_swap', area_dict, active_area_dict, measure, n, cc=cc)
plot_multi_graphs_color(G_ccg_dict, max_reso_gnm, duration_dict, 'total_duration_gnm', area_dict, active_area_dict, measure, n, cc=cc)
plot_multi_graphs_color(G_ccg_dict, max_reso_swap, duration_dict, 'total_duration_swap', area_dict, active_area_dict, measure, n, cc=cc)
#%%
############ plot distribution of intra/inter offset/duration
plot_intra_inter_data(offset_dict, G_ccg_dict, 'total', 'offset', True, active_area_dict, measure, n)
plot_intra_inter_data(offset_dict, G_ccg_dict, 'total', 'offset', False, active_area_dict, measure, n)
plot_intra_inter_data(duration_dict, G_ccg_dict, 'total', 'duration', True, active_area_dict, measure, n)
plot_intra_inter_data(duration_dict, G_ccg_dict, 'total', 'duration', False, active_area_dict, measure, n)
#%%
plot_intra_inter_data_violin(offset_dict, G_ccg_dict, 'total', 'offset', True, active_area_dict, measure, n)
plot_intra_inter_data_violin(offset_dict, G_ccg_dict, 'total', 'offset', False, active_area_dict, measure, n)
plot_intra_inter_data_violin(duration_dict, G_ccg_dict, 'total', 'duration', True, active_area_dict, measure, n)
plot_intra_inter_data_violin(duration_dict, G_ccg_dict, 'total', 'duration', False, active_area_dict, measure, n)
#%%
region_data_heatmap(offset_dict, G_ccg_dict, 'total', 'offset', active_area_dict, visual_regions, measure, n)
region_data_heatmap(duration_dict, G_ccg_dict, 'total', 'duration', active_area_dict, visual_regions, measure, n)
#%%
########### LSCC distribution
lscc_region_counts, lscc_size = get_lscc_region_count(G_ccg_dict, area_dict, visual_regions)
#%%
plot_hub_pie_chart(lscc_region_counts, 'total', 'LSCC', visual_regions)
plot_total_group_size_stimulus(lscc_size, 'total_LSCC', measure, n)
#%%
########### maximum clique distribution

clique_region_counts, max_cliq_size = get_max_clique_region_count(G_ccg_dict, area_dict, visual_regions)
#%%
plot_hub_pie_chart(clique_region_counts, 'total', 'max_clique', visual_regions)
#%%
plot_total_group_size_stimulus(max_cliq_size, 'max_clique', measure, n)
#%%
total_metric = metric_stimulus_individual(G_ccg_dict, 'total', measure, n, weight='weight', cc=False)
# %%
############# keep largest connected components for pos and neg G_dict
pos_G_dict = get_lcc(pos_G_dict)
neg_G_dict = get_lcc(neg_G_dict)
# %%
print_stat(pos_G_dict)
print_stat(neg_G_dict)
# %%
plot_stat(pos_G_dict, n, neg_G_dict, measure=measure)
# %%
weight = False
region_connection_seperate_diagonal(pos_G_dict, 'pos', area_dict, visual_regions, measure, n, weight)
region_connection_seperate_diagonal(neg_G_dict, 'neg', area_dict, visual_regions, measure, n, weight)
# %%
region_connection_heatmap(pos_G_dict, 'pos', area_dict, visual_regions, measure, n)
region_connection_heatmap(neg_G_dict, 'neg', area_dict, visual_regions, measure, n)
# %%
weight = False
region_connection_delta_heatmap(pos_G_dict, 'pos', area_dict, visual_regions, measure, n, weight)
region_connection_delta_heatmap(neg_G_dict, 'neg', area_dict, visual_regions, measure, n, weight)
# %%
weight = True
region_connection_delta_heatmap(pos_G_dict, 'pos', area_dict, visual_regions, measure, n, weight)
region_connection_delta_heatmap(neg_G_dict, 'neg', area_dict, visual_regions, measure, n, weight)
#%%
abs_neg_G_dict = get_abs_weight(neg_G_dict)
# %%
############# plot all graphs with community layout and color as region #################
# cc = True
cc = False
plot_multi_graphs_color(pos_G_dict, 'pos', area_dict, measure, n, cc=cc)
plot_multi_graphs_color(abs_neg_G_dict, 'neg', area_dict, measure, n, cc=cc)

#%%

# cc = True
# %%
plot_directed_multi_degree_distributions(pos_G_dict, 'pos', measure, n, weight=None, cc=False)
plot_directed_multi_degree_distributions(neg_G_dict, 'neg', measure, n, weight=None, cc=False)
# %%
plot_directed_multi_degree_distributions(pos_G_dict, 'pos', measure, n, weight='weight', cc=False)
plot_directed_multi_degree_distributions(neg_G_dict, 'neg', measure, n, weight='weight', cc=False)
#%%
pos_metric = metric_stimulus_individual(pos_G_dict, 'pos', measure, n, weight='weight', cc=False)
neg_metric = metric_stimulus_individual(neg_G_dict, 'neg', measure, n, weight='weight', cc=False)
#%%
############### metric_stimulus_by_region
def metric_stimulus_by_region(G_dict, sign, measure, n, weight, cc):
  rows, cols = get_rowcol(G_dict)
  metric_names = ['in_degree', 'out_degree', 'in_strength', 'out_strength', 'betweenness', 'in_closeness', 'out_closeness']
  plots_shape = (4, 2)
  metric = np.empty((len(rows), len(cols), len(metric_names)))
  metric[:] = np.nan
  fig = plt.figure(figsize=(5*plots_shape[1], 13))
  # fig = plt.figure(figsize=(20, 10))
  for metric_ind, metric_name in enumerate(metric_names):
    print(metric_name)
    for row_ind, row in enumerate(rows):
      print(row)
      for col_ind, col in enumerate(cols):
        G = G_dict[row][col] if col in G_dict[row] else nx.DiGraph()
        if G.number_of_nodes() > 2 and G.number_of_edges() > 0:
          if weight:
            m = calculate_weighted_metric(G, metric_name, cc)
          else:
            m = calculate_metric(G, metric_name, cc)
        metric[row_ind, col_ind, metric_ind] = m
    plt.subplot(*plots_shape, metric_ind + 1)
    for row_ind, row in enumerate(rows):
      plt.plot(cols, metric[row_ind, :, metric_ind], label=row, alpha=1)
    plt.gca().set_title(metric_name, fontsize=30, rotation=0)
    plt.xticks(rotation=90)
    if metric_ind // 2 < 2:
      plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
  plt.legend()
  plt.tight_layout()
  figname = './plots/metric_stimulus_individual_weighted_{}_{}_{}_fold.jpg'.format(sign, measure, n) if weight else './plots/metric_stimulus_individual_{}_{}_{}_fold.jpg'.format(sign, measure, n)
  plt.savefig(figname)
  return metric
# %%
############## undirected
weight = None
# weight = 'weight'
pos_region_counts = get_hub_region_count(pos_G_dict, area_dict, visual_regions, weight=weight)
neg_region_counts = get_hub_region_count(neg_G_dict, area_dict, visual_regions, weight=weight)
plot_hub_pie_chart(pos_region_counts, 'pos', 'hub_node', visual_regions)
plot_hub_pie_chart(neg_region_counts, 'neg', 'hub_node', visual_regions)
#%%
############## directed
# weight = None
weight = 'weight'
pos_source_region_counts, pos_target_region_counts = get_directed_hub_region_count(pos_G_dict, visual_regions, weight=weight)
neg_source_region_counts, neg_target_region_counts = get_directed_hub_region_count(neg_G_dict, visual_regions, weight=weight)
plot_directed_hub_pie_chart(pos_source_region_counts, 'pos', 'source', visual_regions, weight)
plot_directed_hub_pie_chart(pos_target_region_counts, 'pos', 'target', visual_regions, weight)
plot_directed_hub_pie_chart(neg_source_region_counts, 'neg', 'source', visual_regions, weight)
plot_directed_hub_pie_chart(neg_target_region_counts, 'neg', 'target', visual_regions, weight)
# %%
########### maximum clique distribution
pos_clique_region_counts, pos_max_cliq_size = get_max_clique_region_count(pos_G_dict, area_dict, visual_regions)
neg_clique_region_counts, neg_max_cliq_size = get_max_clique_region_count(neg_G_dict, area_dict, visual_regions)
plot_hub_pie_chart(pos_clique_region_counts, 'pos', 'max_clique', visual_regions)
plot_hub_pie_chart(neg_clique_region_counts, 'neg', 'max_clique', visual_regions)
#%%
plot_group_size_stimulus(pos_max_cliq_size, neg_max_cliq_size, 'max_clique', measure, n)
#%%
pos_lscc_region_counts, pos_lscc_size = get_lscc_region_count(pos_G_dict, area_dict, visual_regions)
neg_lscc_region_counts, neg_lscc_size = get_lscc_region_count(neg_G_dict, area_dict, visual_regions)
#%%
plot_hub_pie_chart(pos_lscc_region_counts, 'pos', 'LSCC', visual_regions)
plot_hub_pie_chart(neg_lscc_region_counts, 'neg', 'LSCC', visual_regions)
#%%
plot_group_size_stimulus(pos_lscc_size, neg_lscc_size, 'LSCC', measure, n)
#%%
plot_intra_inter_density(G_ccg_dict, 'whole', area_dict, visual_regions, measure)
plot_intra_inter_density(pos_G_dict, 'positive', area_dict, visual_regions, measure)
plot_intra_inter_density(neg_G_dict, 'negative', area_dict, visual_regions, measure)
#%%
plot_intra_inter_connection(G_ccg_dict, 'whole', area_dict, visual_regions, measure)
plot_intra_inter_connection(pos_G_dict, 'positive', area_dict, visual_regions, measure)
plot_intra_inter_connection(neg_G_dict, 'negative', area_dict, visual_regions, measure)
#%%
plot_intra_inter_ratio(G_ccg_dict, 'whole', area_dict, visual_regions, measure)
plot_intra_inter_ratio(pos_G_dict, 'positive', area_dict, visual_regions, measure)
plot_intra_inter_ratio(neg_G_dict, 'negative', area_dict, visual_regions, measure)
# %%
# G_ccg_dict = get_lcc(G_ccg_dict)
# # %%
# print_stat(G_ccg_dict)
# # %%
# plot_stat(G_ccg_dict, n=n, measure=measure)
# # %%
# region_connection_heatmap(G_ccg_dict, 'pos', area_dict, visual_regions, measure, n)
# # %%
# weight = False
# region_connection_delta_heatmap(G_ccg_dict, 'pos', area_dict, visual_regions, measure, n, weight)
# # %%
# weight = True
# region_connection_delta_heatmap(G_ccg_dict, 'pos', area_dict, visual_regions, measure, n, weight)
# # %%
# ############# plot all graphs with community layout and color as region #################
# cc = False
# plot_multi_graphs_color(G_ccg_dict, 'pos', area_dict, measure, n, cc=cc)
# # %%
# ############# plot all graphs with community layout and color as region #################
# cc = True
# plot_multi_graphs_color(G_ccg_dict, 'pos', area_dict, measure, n, cc=cc)
# # %%
# plot_directed_multi_degree_distributions(G_ccg_dict, 'pos', measure, n, weight=None, cc=False)
# # %%
# plot_directed_multi_degree_distributions(G_ccg_dict, 'pos', measure, n, weight='weight', cc=False)
#%%
################ plot example significant ccg for sharp peak
def plot_example_ccg_sharp_peak(directory, measure, maxlag=12, n=7, window=100, disable=False):
  path = directory.replace(measure, measure+'_sharp_peak')
  if not os.path.exists(path):
    os.makedirs(path)
  files = os.listdir(directory)
  files.sort(key=lambda x:int(x[:9]))
  for file in files:
    if '_bl' not in file: #   and '719161530' in file and ('static_gratings' in file or 'gabors' in file) or 'flashes' in file
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
      corr = (ccg_corrected - ccg_corrected.mean(-1)[:, :, None])[:, :, :maxlag]
      max_offset = np.argmax(np.abs(corr), -1)
      ccg_mat = np.choose(max_offset, np.moveaxis(corr, -1, 0))
      num_nodes = ccg.shape[0]
      significant_ccg, significant_peaks=np.zeros((num_nodes,num_nodes)), np.zeros((num_nodes,num_nodes))
      significant_ccg[:] = np.nan
      significant_peaks[:] = np.nan
      pos_fold = ccg_corrected[:, :, :maxlag].max(-1) > ccg_corrected.mean(-1) + n * ccg_corrected.std(-1)
      neg_fold = ccg_corrected[:, :, :maxlag].max(-1) < ccg_corrected.mean(-1) - n * ccg_corrected.std(-1)
      indx = np.logical_or(pos_fold, neg_fold)
      if np.sum(indx):
        significant_ccg[indx] = ccg_mat[indx]
        significant_peaks[indx] = max_offset[indx]
    
      significant_inds = list(zip(*np.where(~np.isnan(significant_ccg))))
      print('Number of significant links: {}, density {}'.format(len(significant_inds), len(significant_inds)/(num_nodes*(num_nodes-1))))
      np.random.shuffle(significant_inds)
      fig = plt.figure(figsize=(5*3, 5*3))
      for ind, (row_a, row_b) in enumerate(significant_inds[:9]):
        ax = plt.subplot(3, 3, ind+1)
        plt.axvline(x=significant_peaks[row_a, row_b], color='r', linestyle='--', alpha=0.9)
        plt.plot(np.arange(window+1), ccg_corrected[row_a, row_b])
        if ind % 3 == 0:
          plt.ylabel('signigicant CCG corrected', size=20)
        if ind // 3 == 3 - 1:
          plt.xlabel('time lag (ms)', size=20)
      plt.suptitle('{} fold\n{}, {}'.format(n, mouseID, stimulus_name), size=25)
      plt.savefig('./plots/sample_significant_ccg_{}fold_{}_{}.jpg'.format(n, mouseID, stimulus_name))

np.seterr(divide='ignore', invalid='ignore')
measure = 'ccg'
maxlag = 12
n = 4
directory = './data/ecephys_cache_dir/sessions/adj_mat_{}_corrected/'.format(measure)
plot_example_ccg_sharp_peak(directory, measure, maxlag=maxlag, n=n, disable=True)
#%%
################ plot example significant ccg for sharp intergral
def plot_example_ccg_sharp_integral(directory, measure, maxlag=12, n=7, window=100, disable=False):
  path = directory.replace(measure, measure+'_sharp_integral')
  if not os.path.exists(path):
    os.makedirs(path)
  files = os.listdir(directory)
  files.sort(key=lambda x:int(x[:9]))
  for file in files:
    if '_bl' not in file and 'gabors' not in file: #   and '719161530' in file and ('static_gratings' in file or 'gabors' in file) or 'flashes' in file
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
      num_nodes = ccg.shape[0]
      significant_ccg=np.zeros((num_nodes,num_nodes))
      significant_ccg[:] = np.nan
      ccg_corrected = ccg - ccg_jittered
      corr = (ccg_corrected - ccg_corrected.mean(-1)[:, :, None])
      filter = np.array([[[1/maxlag]]]).repeat(maxlag, axis=2)
      corr_integral = signal.convolve(corr, filter, mode='valid')
      ccg_mat = corr_integral[:, :, 0] # average of first maxlag window
      num_nodes = ccg.shape[0]
      pos_fold = ccg_mat > corr_integral.mean(-1) + n * corr_integral.std(-1)
      neg_fold = ccg_mat < corr_integral.mean(-1) - n * corr_integral.std(-1)
      indx = np.logical_or(pos_fold, neg_fold)
      if np.sum(indx):
        significant_ccg[indx] = ccg_mat[indx]
    
      significant_inds = list(zip(*np.where(~np.isnan(significant_ccg))))
      print('Number of significant links: {}, density {}'.format(len(significant_inds), len(significant_inds)/(num_nodes*(num_nodes-1))))
      np.random.shuffle(significant_inds)
      fig = plt.figure(figsize=(5*3, 5*3))
      for ind, (row_a, row_b) in enumerate(significant_inds[:9]):
        ax = plt.subplot(3, 3, ind+1)
        plt.axvline(x=maxlag, color='r', linestyle='--', alpha=0.9)
        plt.plot(np.arange(window+1), ccg_corrected[row_a, row_b])
        if ind % 3 == 0:
          plt.ylabel('signigicant CCG corrected', size=20)
        if ind // 3 == 3 - 1:
          plt.xlabel('time lag (ms)', size=20)
        title = 'positive edge' if pos_fold[row_a, row_b] else 'negative edge'
        plt.title(title, size=20)
      plt.suptitle('{} fold\n{}, {}'.format(n, mouseID, stimulus_name), size=25)
      plt.savefig('./plots/sample_significant_ccg_{}fold_sharp_integral_{}_{}.jpg'.format(n, mouseID, stimulus_name))

np.seterr(divide='ignore', invalid='ignore')
measure = 'ccg'
maxlag = 12
n = 4
directory = './data/ecephys_cache_dir/sessions/adj_mat_{}_corrected/'.format(measure)
plot_example_ccg_sharp_integral(directory, measure, maxlag=maxlag, n=n, disable=True)
# %%
################ plot example significant xcorr for sharp peak
def plot_example_xcorr_n_fold(directory, measure, maxlag=12, alpha=0.01, sign='all', n=7, window=100):
  files = os.listdir(directory)
  files.sort(key=lambda x:int(x[:9]))
  adj_bl_temp = load_npz_3d(os.path.join(directory, [f for f in files if '_bl' in f][0]))
  num_baseline = adj_bl_temp.shape[2] # number of shuffles
  k = int(num_baseline * alpha) + 1 # allow int(N * alpha) random correlations larger
  for file in files:
    if '_bl' not in file:
      print(file)
      mouseID = file.split('_')[0]
      stimulus_name = file.replace('.npz', '').replace(mouseID + '_', '')
      all_xcorr = load_npz_3d(os.path.join(directory, file))
      # import pdb;pdb.set_trace()
      corr = (all_xcorr - all_xcorr.mean(-1)[:, :, None])[:, :, :maxlag]
      max_offset = np.argmax(np.abs(corr), -1)
      xcorr = np.choose(max_offset, np.moveaxis(corr, -1, 0))
      xcorr_bl = load_npz_3d(os.path.join(directory, file.replace('.npz', '_bl.npz')))
      all_xcorr = all_xcorr - xcorr_bl
      significant_adj_mat, significant_peaks=np.zeros_like(xcorr), np.zeros_like(xcorr)
      significant_adj_mat[:] = np.nan
      significant_peaks[:] = np.nan
      if sign == 'pos':
        # indx = xcorr > np.clip(np.partition(xcorr_bl, -k, axis=-1)[:, :, -k], a_min=0, a_max=None)
        fold = all_xcorr[:, :, :maxlag].max(-1) > all_xcorr.mean(-1) + n * all_xcorr.std(-1)
      elif sign == 'neg':
        # indx = xcorr < np.clip(np.partition(xcorr_bl, k-1, axis=-1)[:, :, k-1], a_min=None, a_max=0)
        fold = all_xcorr[:, :, :maxlag].max(-1) < all_xcorr.mean(-1) - n * all_xcorr.std(-1)
      elif sign == 'all':
        # pos = xcorr > np.clip(np.partition(xcorr_bl, -k, axis=-1)[:, :, -k], a_min=0, a_max=None)
        # neg = xcorr < np.clip(np.partition(xcorr_bl, k-1, axis=-1)[:, :, k-1], a_min=None, a_max=0)
        # indx = np.logical_or(pos, neg)
        pos_fold = all_xcorr[:, :, :maxlag].max(-1) > all_xcorr.mean(-1) + n * all_xcorr.std(-1)
        neg_fold = all_xcorr[:, :, :maxlag].max(-1) < all_xcorr.mean(-1) - n * all_xcorr.std(-1)
        fold = np.logical_or(pos_fold, neg_fold)
      # indx = np.logical_and(indx, fold)
      indx = fold
      if np.sum(indx):
        significant_adj_mat[indx] = xcorr[indx]
        significant_peaks[indx] = max_offset[indx]
      significant_inds = list(zip(*np.where(~np.isnan(significant_adj_mat))))
      np.random.shuffle(significant_inds)
      fig = plt.figure(figsize=(5*3, 5*3))
      for ind, (row_a, row_b) in enumerate(significant_inds[:9]):
        ax = plt.subplot(3, 3, ind+1)
        plt.axvline(x=significant_peaks[row_a, row_b], color='r', linestyle='--', alpha=0.9)
        plt.plot(np.arange(window+1), all_xcorr[row_a, row_b, :])
        if ind % 3 == 0:
          plt.ylabel('signigicant xcorr', size=20)
        if ind // 3 == 3 - 1:
          plt.xlabel('time lag (ms)', size=20)
      plt.suptitle('{} fold\n{}, {}'.format(n, mouseID, stimulus_name), size=25)
      plt.savefig('./plots/sample_significant_{}_{}_{}fold_{}_{}.jpg'.format(measure, sign, n, mouseID, stimulus_name))

measure = 'xcorr'
maxlag = 12
alpha=0.01
n = 3
# sign = 'pos'
sign = 'neg'
# sign = 'all'
window=100
directory = './data/ecephys_cache_dir/sessions/adj_mat_{}_shuffled/'.format(measure)
plot_example_xcorr_n_fold(directory, measure, maxlag=maxlag, alpha=alpha, sign=sign, n=n, window=window)
#%%
#%%
################ plot whole connectivity matrix

def load_ccg_connectivity(directory, mouseIDs, stimulus_names, maxlag=12):
  ccg_mat_dict = {}
  for row_ind, mouseID in enumerate(mouseIDs):
    print(mouseID)
    ccg_mat_dict[mouseID] = {}
    for col_ind, stimulus_name in enumerate(stimulus_names):
      try: 
        ccg = load_npz_3d(os.path.join(directory, str(mouseID) + '_' + stimulus_name + '.npz'))
      except:
        ccg = load_sparse_npz(os.path.join(directory, str(mouseID) + '_' + stimulus_name + '.npz'))
      try:
        ccg_jittered = load_npz_3d(os.path.join(directory, str(mouseID) + '_' + stimulus_name + '_bl.npz'))
      except:
        ccg_jittered = load_sparse_npz(os.path.join(directory, str(mouseID) + '_' + stimulus_name + '_bl.npz'))
      ccg_corrected = ccg - ccg_jittered
      corr = (ccg_corrected - ccg_corrected.mean(-1)[:, :, None])[:, :, :maxlag]
      max_offset = np.argmax(np.abs(corr), -1)
      ccg_mat = np.choose(max_offset, np.moveaxis(corr, -1, 0))
      ccg_mat_dict[mouseID][stimulus_name] = ccg_mat
  return ccg_mat_dict

def plot_ccg_connectivity(ccg_mat_dict, mouseIDs, stimulus_names, n, measure):
  ind = 1
  fig = plt.figure(figsize=(4*len(stimulus_names), 3*len(mouseIDs)))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  for row_ind, mouseID in enumerate(mouseIDs):
    print(mouseID)
    for col_ind, stimulus_name in enumerate(stimulus_names):
      plt.subplot(len(mouseIDs), len(stimulus_names), ind)
      if row_ind == 0:
        plt.gca().set_title(stimulus_names[col_ind], fontsize=20, rotation=0)
      if col_ind == 0:
        plt.gca().text(0, 0.5 * (bottom + top), mouseIDs[row_ind],
        horizontalalignment='left',
        verticalalignment='center',
        # rotation='vertical',
        transform=plt.gca().transAxes, fontsize=20, rotation=90)
      plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
      ind += 1
      matrix = ccg_mat_dict[mouseID][stimulus_name].astype(float)
      vmax, vmin = np.nanmean(matrix) + np.nanstd(3*matrix), np.nanmean(matrix) - 3*np.nanstd(matrix)
      sns_plot = sns.heatmap(matrix,vmax=vmax,vmin=vmin,center=0,cmap="RdBu_r")# cmap="YlGnBu"
      # sns_plot = sns.heatmap(ccg_mat_dict[mouseID][stimulus_name].astype(float),norm=colors.LogNorm(),cmap="RdBu_r")# cmap="YlGnBu"
      # sns_plot = sns.heatmap(region_connection.astype(float), vmin=0, cmap="YlGnBu")
      # sns_plot.set_xticks(np.arange(len(regions))+0.5)
      # sns_plot.set_xticklabels(regions, rotation=90)
      # sns_plot.set_yticks(np.arange(len(regions))+0.5)
      # sns_plot.set_yticklabels(regions, rotation=0)
      sns_plot.invert_yaxis()
  plt.suptitle('{} fold'.format(n), size=30)
  plt.tight_layout()
  plt.savefig('./plots/connectivity_matrix_{}_{}fold.jpg'.format(measure, n))
  # plt.show()
#%%
stimulus_names = ['spontaneous', 'flashes', 
        'drifting_gratings', 'static_gratings',
          'natural_scenes', 'natural_movie_one', 'natural_movie_three']
# session_ids = sessions[sessions.session_type=='brain_observatory_1.1'].index.values # another one is functional_connectivity
mouseIDs = [719161530, 750749662, 755434585, 756029989, 791319847]
measure = 'ccg'
maxlag = 12
directory = './data/ecephys_cache_dir/sessions/adj_mat_{}_corrected/'.format(measure)
ccg_mat_dict = load_ccg_connectivity(directory, mouseIDs, stimulus_names, maxlag=12)
#%%
n = 4
plot_ccg_connectivity(ccg_mat_dict, mouseIDs, stimulus_names, n, measure)
# %%
def plot_corr_peak_stimulus(peak_dict, n, measure):
  ind = 1
  rows, cols = get_rowcol(peak_dict)
  fig = plt.figure(figsize=(6, 6))
  
  for row in rows:
    peak_mean, peak_std = np.zeros(len(cols)), np.zeros(len(cols))
    for col_ind, col in enumerate(cols):
      peak_mean[col_ind] = np.nanmean(peak_dict[row][col])
      peak_std[col_ind] = np.nanstd(peak_dict[row][col])
    plt.plot(cols, peak_mean, alpha=0.6, label=row)
    plt.fill_between(cols, peak_mean - peak_std, peak_mean + peak_std, alpha=0.2)
  plt.legend()
  plt.xticks(rotation=90)
  plt.tight_layout()
  plt.savefig('./plots/peak_stimulus_{}_{}fold'.format(measure, n))
  plt.show()

plot_corr_peak_stimulus(peak_dict, n, measure)
# %%
def plot_multi_peak_dist(peak_dict, n, measure):
  ind = 1
  rows, cols = get_rowcol(peak_dict)
  fig = plt.figure(figsize=(4*len(cols), 3*len(rows)))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      plt.subplot(len(rows), len(cols), ind)
      if row_ind == 0:
        plt.gca().set_title(cols[col_ind], fontsize=20, rotation=0)
      if col_ind == 0:
        plt.gca().text(0, 0.5 * (bottom + top), rows[row_ind],
        horizontalalignment='left',
        verticalalignment='center',
        # rotation='vertical',
        transform=plt.gca().transAxes, fontsize=20, rotation=90)
      plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
      ind += 1
      peaks = peak_dict[row][col]
      plt.hist(peaks.flatten(), bins=12, density=True)
      plt.axvline(x=np.nanmean(peaks), color='r', linestyle='--')
      # plt.text(peaks.mean(), peaks.max()/2, "mean={}".format(peaks.mean()), rotation=0, verticalalignment='center')
      plt.xlabel('peak correlation offset (ms)')
      plt.ylabel('Probability')
      
  plt.tight_layout()
  image_name = './plots/peak_distribution_{}_{}fold.jpg'.format(measure, n)
  # plt.show()
  plt.savefig(image_name)
# plot_multi_corr_FR(session_ids, stimulus_names, pcorr_dict, bin_dict, 'pearson')
# plot_multi_peak_dist(peak_dict, measure)

plot_multi_peak_dist(peak_dict, n, measure)
#%%
############### get average reponse peak offset
directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
stimulus_names = ['spontaneous', 'flashes', 
        'drifting_gratings', 'static_gratings',
          'natural_scenes', 'natural_movie_one', 'natural_movie_three']
# session_ids = sessions[sessions.session_type=='brain_observatory_1.1'].index.values # another one is functional_connectivity
mouseIDs = session_ids
response_dict = {}
for row_ind, mouseID in enumerate(mouseIDs):
  print(mouseID)
  response_dict[mouseID] = {}
  for col_ind, stimulus_name in enumerate(stimulus_names):
    print(stimulus_name)
    matrix = load_npz_3d(os.path.join(directory, str(mouseID) + '_' + stimulus_name + '.npz'))
    matrix = np.moveaxis(matrix, -1, 0) # time, neuron, condition
    mean_matrix = matrix.mean(axis=-1).mean(axis=-1)
    response_dict[mouseID][stimulus_name] = mean_matrix
#%%
def plot_average_response_smoothed(response_dict, p=0.1):
  peak_response_dict = {}
  rows, cols = get_rowcol(response_dict)
  fig = plt.figure(figsize=(4*len(cols), 3*len(rows)))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  ind = 1
  for row_ind, row in enumerate(rows):
    print(row)
    peak_response_dict[row] = {}
    for col_ind, col in enumerate(cols):
      plt.subplot(len(rows), len(cols), ind)
      if row_ind == 0:
        plt.gca().set_title(col, fontsize=20, rotation=0)
      if col_ind == 0:
        plt.gca().text(0, 0.5 * (bottom + top), row,
        horizontalalignment='left',
        verticalalignment='center',
        # rotation='vertical',
        transform=plt.gca().transAxes, fontsize=20, rotation=90)
      plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
      ind += 1
      n = int(p * response_dict[row][col].shape[0])
      smoothed_response = moving_average(response_dict[row][col], n)
      peak_response_dict[row][col] = np.argmax(smoothed_response) / len(smoothed_response)
      plt.axvline(x=np.argmax(smoothed_response), color='r', linestyle='--', alpha=0.9)
      plt.plot(smoothed_response, alpha=1)
      plt.xlabel('time after stimulus onset (ms)')
      plt.ylabel('average number of spikes')
  plt.suptitle('{} ms moving average'.format(n), size=30)
  plt.tight_layout()
  plt.savefig('./plots/average_response_smoothed_{}.jpg'.format(p))
  return peak_response_dict

# n = 25
p = 0.1
peak_response_dict = plot_average_response_smoothed(response_dict, p=p)
#%%
def plot_response_peak_stimulus(peak_response_dict, p):
  ind = 1
  rows, cols = get_rowcol(peak_response_dict)
  fig = plt.figure(figsize=(6, 6))
  for row in rows:
    peaks = np.zeros(len(cols))
    for col_ind, col in enumerate(cols):
      peaks[col_ind] = peak_response_dict[row][col]
    plt.plot(cols, peaks, alpha=0.6, label=row)
  plt.legend()
  plt.xticks(rotation=90)
  plt.ylabel('fraction of presentation duration')
  plt.title('peak response time after onset (fraction)')
  plt.tight_layout()
  plt.savefig('./plots/average_response_peakoffset_stimulus_{}.jpg'.format(p))
  plt.show()

plot_response_peak_stimulus(peak_response_dict, p)
# %%
############## calculate ccg and save
np.seterr(divide='ignore', invalid='ignore')
############# save correlation matrices #################
# min_len, min_num = (260000, 739)
min_len, min_num = (10000, 29)
min_duration = 250
min_FR = 0.002 # 2 Hz
# min_spikes = min_duration * min_FR
# min_spikes = min_len * min_FR
measure = 'ccg'
directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
num_baseline = 1
files = os.listdir(directory)
files = [f for f in files if f.endswith('.npz')]
files.sort(key=lambda x:int(x[:9]))
path = os.path.join(directory.replace('spiking_sequence', 'adj_mat_ccg_corrected'))
# path = os.path.join(directory.replace('spiking_sequence', 'adj_mat_xcorr_shuffled'))
if not os.path.exists(path):
  os.makedirs(path)
file_order = int(sys.argv[1])
# file_order = 1
file = files[file_order] # 0, 2, 7 spontaneous, gabors, natural_movie_three
print(file)
# (num_neuron, num_trial, T)
sequences = load_npz_3d(os.path.join(directory, file))
# sequences = sequences[:, :, :min_duration]
active_neuron_inds = sequences.mean(1).sum(1) > sequences.shape[2] * min_FR
sequences = sequences[active_neuron_inds]
print('Spike train shape: {}'.format(sequences.shape))
# sequences = concatenate_trial(sequences, min_duration, min_len)
# sequences = sequences[np.count_nonzero(sequences[:, :min_len], axis=1) > min_spikes, :min_len]
fname = os.path.join(path, file)
start_time = time.time()
save_mean_ccg_corrected(sequences=sequences, fname=fname, num_jitter=num_baseline, L=25, window=100, disable=False)
print("--- %s minutes" % ((time.time() - start_time)/60))
#%%
############## calculate xcorr and save
start_time = time.time()
np.seterr(divide='ignore', invalid='ignore')
############# save correlation matrices #################
# min_len, min_num = (260000, 739)
min_len, min_num = (10000, 29)
min_duration = 250
min_fre = 0.002 # 2 Hz
min_spikes = min_len * min_fre
measure = 'xcorr'
directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
files = os.listdir(directory)
files = [f for f in files if f.endswith('.npz')]
files.sort(key=lambda x:int(x[:9]))
num_baseline = 100
# path = os.path.join(directory.replace('spiking_sequence', 'adj_mat_ccg_corrected'))
path = os.path.join(directory.replace('spiking_sequence', 'adj_mat_xcorr_shuffled'))
if not os.path.exists(path):
  os.makedirs(path)
file_order = int(sys.argv[1])
file = files[file_order] # 0, 2, 7 spontaneous, gabors, natural_movie_three
print(file)
# sequences = load_npz(os.path.join(directory, file))
sequences = load_npz_3d(os.path.join(directory, file))
sequences = concatenate_trial(sequences, min_duration, min_len)
sequences = sequences[np.count_nonzero(sequences[:, :min_len], axis=1) > min_spikes, :min_len]
fname = os.path.join(path, file)
start_time = time.time()
save_xcorr_shuffled(sequences=sequences, fname=fname, window=100, num_baseline=num_baseline, disable=False)
print("--- %s minutes" % ((time.time() - start_time)/60))
#%%
##################### plot spike trains
min_len, min_num = (10000, 29)
min_duration = 250
min_fre = 0.002 # 2 Hz
min_spikes = min_len * min_fre
measure = 'ccg'
directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
num_baseline = 100
files = os.listdir(directory)
files = [f for f in files if f.endswith('.npz')]
files.sort(key=lambda x:int(x[:9]))
path = os.path.join(directory.replace('spiking_sequence', 'adj_mat_ccg_corrected'))
# path = os.path.join(directory.replace('spiking_sequence', 'adj_mat_xcorr_shuffled'))
if not os.path.exists(path):
  os.makedirs(path)
file_order = 3
file = files[file_order] # 0, 2, 7 spontaneous, gabors, natural_movie_three
print(file)
# sequences = load_npz(os.path.join(directory, file))
sequences = load_npz_3d(os.path.join(directory, file))
sequences = concatenate_trial(sequences, min_duration, min_len)
sequences = sequences[:, :min_len]
# %%
def unique(l):
  u, ind = np.unique(l, return_index=True)
  return list(u[np.argsort(ind)])

stimulus_names = ['spontaneous', 'flashes', 
        'drifting_gratings', 'static_gratings',
          'natural_scenes', 'natural_movie_one', 'natural_movie_three']
# session_ids = [719161530, 750749662, 754312389, 755434585, 756029989, 791319847]
mouse_ind, stimulus_ind = 0, 3
matrix = load_npz_3d(os.path.join(directory, str(session_ids[mouse_ind]) + '_' + stimulus_names[stimulus_ind] + '.npz'))
matrix = np.moveaxis(matrix, -1, 0) # time, neuron, condition
# print(matrix.shape)
# mean_matrix = matrix.mean(axis=-1).T # neuron, time
trial_matrix = matrix[:,:,0].T # neuron, time
a_file = open('./data/ecephys_cache_dir/sessions/area_dict.pkl', 'rb')
area_dict = pickle.load(a_file)
# change the keys of area_dict from int to string
int_2_str = dict((session_id, str(session_id)) for session_id in session_ids)
area_dict = dict((int_2_str[key], value) for (key, value) in area_dict.items())
a_file.close()
areas = list(area_dict[str(session_ids[mouse_ind])].values())
areas_uniq = unique(areas)
areas_num = [(np.array(areas)==a).sum() for a in areas_uniq]
areas_start_pos = list(np.insert(np.cumsum(areas_num)[:-1], 0, 0))
sequence_by_area = {a:[name for name, age in area_dict[str(session_ids[mouse_ind])].items() if age == a] for a in areas_uniq}
# %%
duration = 500
sorted_sample_seq = np.vstack([trial_matrix[sequence_by_area[a], :duration] for a in areas_uniq])
spike_pos = [np.nonzero(t)[0] for t in sorted_sample_seq[:, :duration]] # divided by 1000 cuz bin size is 1 ms
colors1 = [customPalette[i] for i in sum([[areas_uniq.index(a)] * areas_num[areas_uniq.index(a)] for a in areas_uniq], [])]
uniq_colors = unique(colors1)
text_pos = [s + (areas_num[areas_start_pos.index(s)] - 1) / 2 for s in areas_start_pos]
colors2 = 'black'
lineoffsets2 = 1
linelengths2 = 2
# create a horizontal plot
fig = plt.figure(figsize=(10, 16))
plt.eventplot(spike_pos, colors=colors1, lineoffsets=lineoffsets2,
                    linewidths=2, linelengths=2)
for ind, t_pos in enumerate(text_pos):
  plt.text(-70, t_pos, areas_uniq[ind], size=20, color=uniq_colors[ind], weight='bold')
  # plt.text(-1.2, t_pos, areas_uniq[ind], size=20, color=uniq_colors[ind], weight='bold')
# plt.axis('off')
plt.gca().get_yaxis().set_visible(False)
plt.xlabel('time after onset (ms)')
plt.gca().invert_yaxis()
plt.tight_layout()
# plt.show()
plt.savefig('./plots/raster_{}_{}.jpg'.format(session_ids[mouse_ind], stimulus_names[stimulus_ind]))
#%%
# %%
############### plot average response for spontaneous and natural movies
directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
stimulus_name = 'spontaneous'
# stimulus_name = 'natural_movie_one'
# stimulus_name = 'natural_movie_three'
# session_ids = sessions[sessions.session_type=='brain_observatory_1.1'].index.values # another one is functional_connectivity
mouseIDs = [719161530, 750749662, 754312389, 755434585, 756029989, 791319847]
fig = plt.figure(figsize=(4*8, 3*len(mouseIDs)))
left, width = .25, .5
bottom, height = .25, .5
right = left + width
top = bottom + height
for row_ind, mouseID in enumerate(mouseIDs):
  print(mouseID)
  print(stimulus_name)
  # adj_mat_ds = np.load(os.path.join(directory, file))
  # adj_mat_bl = np.load(os.path.join(directory, file.replace('.npy', '_bl.npy')))
  matrix = load_npz_3d(os.path.join(directory, str(mouseID) + '_' + stimulus_name + '.npz'))
  matrix = np.moveaxis(matrix, -1, 0) # time, neuron, condition
  # print(matrix.shape)
  mean_matrix = matrix.mean(axis=-1).mean(axis=-1)
  print(mean_matrix.shape)
  plt.subplot(len(mouseIDs), 1, row_ind+1)
  # if row_ind == 0:
  #   plt.gca().set_title(stimulus_name, fontsize=20, rotation=0)
  plt.gca().text(0, 0.5 * (bottom + top), mouseIDs[row_ind],
  horizontalalignment='left',
  verticalalignment='center',
  # rotation='vertical',
  transform=plt.gca().transAxes, fontsize=20, rotation=90)
  plt.tick_params(left=True, bottom=False, labelleft=True, labelbottom=False, labelright=True)
  plt.plot(mean_matrix[:5000], alpha=1)
  if row_ind == len(mouseIDs)-1:
    plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True, labelright=True)
    plt.xticks(fontsize=20)
    plt.xlabel('time after stimulus onset (ms)', size=25)
  plt.ylabel('average number of spikes')
plt.suptitle(stimulus_name, size=30)
plt.tight_layout()
plt.savefig('./plots/average_response_{}_5000.jpg'.format(stimulus_name))
#%%
############### plot average response
directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
stimulus_names = ['spontaneous', 'flashes', 
        'drifting_gratings', 'static_gratings',
          'natural_scenes', 'natural_movie_one', 'natural_movie_three']
# session_ids = sessions[sessions.session_type=='brain_observatory_1.1'].index.values # another one is functional_connectivity
mouseIDs = session_ids
fig = plt.figure(figsize=(4*len(stimulus_names), 3*len(mouseIDs)))
left, width = .25, .5
bottom, height = .25, .5
right = left + width
top = bottom + height
ind = 1
for row_ind, mouseID in enumerate(mouseIDs):
  print(mouseID)
  for col_ind, stimulus_name in enumerate(stimulus_names):
    print(stimulus_name)
    # adj_mat_ds = np.load(os.path.join(directory, file))
    # adj_mat_bl = np.load(os.path.join(directory, file.replace('.npy', '_bl.npy')))
    matrix = load_npz_3d(os.path.join(directory, str(mouseID) + '_' + stimulus_name + '.npz'))
    matrix = np.moveaxis(matrix, -1, 0) # time, neuron, condition
    # print(matrix.shape)
    mean_matrix = matrix.mean(axis=-1).mean(axis=-1)
    print(mean_matrix.shape)
    plt.subplot(len(mouseIDs), len(stimulus_names), ind)
    if row_ind == 0:
      plt.gca().set_title(stimulus_names[col_ind], fontsize=20, rotation=0)
    if col_ind == 0:
      plt.gca().text(0, 0.5 * (bottom + top), mouseIDs[row_ind],
      horizontalalignment='left',
      verticalalignment='center',
      # rotation='vertical',
      transform=plt.gca().transAxes, fontsize=20, rotation=90)
      plt.ylabel('average number of spikes')
    plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ind += 1
    plt.plot(mean_matrix, alpha=1)
    if row_ind == len(mouseIDs) - 1:
      plt.xlabel('time after stimulus onset (ms)')
plt.tight_layout()
plt.savefig('./plots/average_response.jpg')
#%%
duration = 250
stimulus_names = ['spontaneous', 'flashes', 
        'drifting_gratings', 'static_gratings',
          'natural_scenes', 'natural_movie_one', 'natural_movie_three']
# session_ids = sessions[sessions.session_type=='brain_observatory_1.1'].index.values # another one is functional_connectivity
mouseIDs = session_ids
areas_uniq = visual_regions
fig = plt.figure(figsize=(4*len(stimulus_names), 3*len(mouseIDs)))
left, width = .25, .5
bottom, height = .25, .5
right = left + width
top = bottom + height
ind = 1
for row_ind, mouseID in enumerate(mouseIDs):
  print(mouseID)
  for col_ind, stimulus_name in enumerate(stimulus_names):
    # adj_mat_ds = np.load(os.path.join(directory, file))
    # adj_mat_bl = np.load(os.path.join(directory, file.replace('.npy', '_bl.npy')))
    matrix = load_npz_3d(os.path.join(directory, str(mouseID) + '_' + stimulus_name + '.npz'))
    matrix = np.moveaxis(matrix, -1, 0) # time, neuron, condition
    response_area = np.zeros((duration, len(areas_uniq)))
    for area_ind, a in enumerate(areas_uniq):
      response_area[:, area_ind] = np.mean(matrix[ :duration, sequence_by_area[a], :], axis=-1).mean(-1)
    # sorted_sample_seq = np.vstack([matrix[ :duration, sequence_by_area[a]] for a in areas_uniq])
    # print(matrix.shape)
    # mean_matrix = matrix.mean(axis=-1).mean(axis=-1)
    # print(mean_matrix.shape)
    plt.subplot(len(mouseIDs), len(stimulus_names), ind)
    if row_ind == 0:
      plt.gca().set_title(stimulus_names[col_ind], fontsize=20, rotation=0)
    if col_ind == 0:
      plt.gca().text(0, 0.5 * (bottom + top), mouseIDs[row_ind],
      horizontalalignment='left',
      verticalalignment='center',
      # rotation='vertical',
      transform=plt.gca().transAxes, fontsize=20, rotation=90)
      plt.ylabel('average number of spikes')
    plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ind += 1
    # plt.plot(mean_matrix[:duration], alpha=1)
    plt.plot(response_area, label=areas_uniq, alpha = 0.6)
    if row_ind == len(mouseIDs) - 1:
      plt.xlabel('time after stimulus onset (ms)')
plt.legend()
plt.tight_layout()
# plt.show()
plt.savefig('./plots/average_response_area.jpg')
 # %%
stimulus_names = ['spontaneous', 'flashes', 
        'drifting_gratings', 'static_gratings',
          'natural_scenes', 'natural_movie_one', 'natural_movie_three']
# session_ids = sessions[sessions.session_type=='brain_observatory_1.1'].index.values # another one is functional_connectivity
mouseIDs = [719161530, 750749662, 755434585, 756029989, 791319847]
fig = plt.figure(figsize=(4*len(stimulus_names), 3*len(mouseIDs)))
left, width = .25, .5
bottom, height = .25, .5
right = left + width
top = bottom + height
ind = 1
for row_ind, mouseID in enumerate(mouseIDs):
  print(mouseID)
  for col_ind, stimulus_name in enumerate(stimulus_names):
    print(stimulus_name)
    # adj_mat_ds = np.load(os.path.join(directory, file))
    # adj_mat_bl = np.load(os.path.join(directory, file.replace('.npy', '_bl.npy')))
    matrix = load_npz_3d(os.path.join(directory, str(mouseID) + '_' + stimulus_name + '.npz'))
    matrix = np.moveaxis(matrix, -1, 0) # time, neuron, condition
    # print(matrix.shape)
    mean_matrix = matrix.mean(axis=-1).mean(axis=-1)
    print(mean_matrix.shape)
    plt.subplot(len(mouseIDs), len(stimulus_names), ind)
    if row_ind == 0:
      plt.gca().set_title(stimulus_names[col_ind], fontsize=20, rotation=0)
    if col_ind == 0:
      plt.gca().text(0, 0.5 * (bottom + top), mouseIDs[row_ind],
      horizontalalignment='left',LowerBoundsion=90)
    plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ind += 1
    plt.plot(mean_matrix[:300], alpha=1)
    plt.ylim(0.0015, 0.013)
    plt.xlabel('time after stimulus onset (ms)')
    plt.ylabel('average number of spikes')
plt.tight_layout()
plt.savefig('./plots/average_response_scale.jpg')
# %%
np.seterr(divide='ignore', invalid='ignore')
############# plot num neurons vs minFR #################
# min_len, min_num = (260000, 739)
min_len, min_num = (10000, 29)
min_duration = 250
min_FR = 0.004 # 4 Hz
# min_spikes = min_duration * min_FR
# min_spikes = min_len * min_FR
measure = 'ccg'
directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
num_baseline = 1
files = os.listdir(directory)
files = [f for f in files if f.endswith('.npz')]
files.sort(key=lambda x:int(x[:9]))
path = os.path.join(directory.replace('spiking_sequence', 'adj_mat_ccg_corrected'))
# path = os.path.join(directory.replace('spiking_sequence', 'adj_mat_xcorr_shuffled'))
stimulus_names = ['spontaneous', 'flashes', 
        'drifting_gratings', 'static_gratings',
          'natural_scenes', 'natural_movie_one', 'natural_movie_three']
session_ids = [719161530, 750749662, 755434585, 756029989, 791319847]
num_nodes = np.full([len(session_ids), len(stimulus_names)], np.nan)
if not os.path.exists(path):
  os.makedirs(path)
# file_order = int(sys.argv[1])
# file_order = 1
for file_order in range(len(files)):
  file = files[file_order] # 0, 2, 7 spontaneous, gabors, natural_movie_three
  mouseID = file.split('_')[0]
  stimulus_name = file.replace('.npz', '').replace(mouseID + '_', '')
  # sequences = load_npz(os.path.join(directory, file))
  # (num_neuron, num_trial, T)
  sequences = load_npz_3d(os.path.join(directory, file))
  sequences = sequences[:, :, :min_duration]
  active_neuron_inds = sequences.mean(1).sum(1) > sequences.shape[2] * min_FR
  sequences = sequences[active_neuron_inds]
  num_nodes[session_ids.index(int(mouseID)), stimulus_names.index(stimulus_name)] = np.sum(active_neuron_inds)
  print('{}, spike train shape: {}'.format(file, sequences.shape))
#%%
plt.figure(figsize=(7,6))
for row_ind, row in enumerate(session_ids):
  plt.plot(stimulus_names, num_nodes[row_ind, :], label=row, alpha=0.6)
plt.gca().set_title('min firing rate {} Hz'.format(min_FR*1000), fontsize=20, rotation=0)
plt.xticks(rotation=90)
plt.ylabel('number of nodes')
plt.legend()
plt.tight_layout()
plt.savefig('./plots/numnodes_minFR{}.jpg'.format(min_FR*1000))
# plt.show()
# %%
area_dict, active_area_dict, mean_speed_df = load_other_data(session_ids)
# %%
############# plot min num neurons per area across mice vs minFR #################
# min_len, min_num = (260000, 739)
min_len, min_num = (10000, 29)
min_duration = 250
min_FR = 0.004 # 4 Hz
# min_spikes = min_duration * min_FR
# min_spikes = min_len * min_FR
measure = 'ccg'
directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
num_baseline = 1
files = os.listdir(directory)
files = [f for f in files if f.endswith('.npz')]
files.sort(key=lambda x:int(x[:9]))
path = os.path.join(directory.replace('spiking_sequence', 'adj_mat_ccg_corrected'))
# path = os.path.join(directory.replace('spiking_sequence', 'adj_mat_xcorr_shuffled'))
stimulus_names = ['spontaneous', 'flashes', 
        'drifting_gratings', 'static_gratings',
          'natural_scenes', 'natural_movie_one', 'natural_movie_three']
session_ids = [719161530, 750749662, 755434585, 756029989, 791319847]
num_nodes_area = np.full([len(session_ids), len(stimulus_names), len(visual_regions)], np.nan)
if not os.path.exists(path):
  os.makedirs(path)
# file_order = int(sys.argv[1])
# file_order = 1
for file_order in range(len(files)):
  file = files[file_order] # 0, 2, 7 spontaneous, gabors, natural_movie_three
  mouseID = file.split('_')[0]
  stimulus_name = file.replace('.npz', '').replace(mouseID + '_', '')
  # sequences = load_npz(os.path.join(directory, file))
  # (num_neuron, num_trial, T)
  sequences = load_npz_3d(os.path.join(directory, file))
  sequences = sequences[:, :, :min_duration]
  active_neuron_inds = sequences.mean(1).sum(1) > sequences.shape[2] * min_FR
  areas = np.array([area_dict[mouseID][id] for id in np.where(active_neuron_inds)[0]])
  for ind, region in enumerate(visual_regions):
    num_nodes_area[session_ids.index(int(mouseID)), stimulus_names.index(stimulus_name), ind] = np.count_nonzero(areas == region)
  print('{}, min num of neurons per area: {}'.format(file, num_nodes_area[session_ids.index(int(mouseID)), stimulus_names.index(stimulus_name)].min()))
#%%
plt.figure(figsize=(7,6))
for r_ind, region in enumerate(visual_regions):
  plt.plot(stimulus_names, num_nodes_area[:, :, r_ind].min(0), label=region, alpha=0.6)
plt.gca().set_title('min firing rate {} Hz'.format(min_FR*1000), fontsize=20, rotation=0)
plt.xticks(rotation=90)
plt.ylabel('min number of nodes across mice')
plt.legend()
plt.tight_layout()
plt.savefig('./plots/numnodes_area_minFR{}.jpg'.format(min_FR*1000))

#%%
start_time = time.time()
measure = 'ccg'
n = 3
directory = './data/ecephys_cache_dir/sessions/adj_mat_{}_corrected/'.format(measure)
save_ccg_corrected_sharp_integral(directory, measure, maxlag=12, n=n)
print("--- %s minutes in total" % ((time.time() - start_time)/60))
#%%
from scipy import signal
a = np.arange(0, 27).reshape(3,3,3)
maxlag = 2
filter = np.array([[[1/maxlag]]]).repeat(maxlag, axis=2)
b = signal.convolve(a, filter, mode='valid')
b
# %%
def get_overlap_ccg_sharp_peak_interval(directory, rows, cols, maxlag=12, n=7):
  num_peak, num_interval, percents = np.zeros((len(rows), len(cols))), np.zeros((len(rows), len(cols))), np.zeros((len(rows), len(cols)))
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      print(col)
      file = str(row) + '_' + col + '.npz'
      try: 
        ccg = load_npz_3d(os.path.join(directory, file))
      except:
        ccg = load_sparse_npz(os.path.join(directory, file))
      try:
        ccg_jittered = load_npz_3d(os.path.join(directory, file.replace('.npz', '_bl.npz')))
      except:
        ccg_jittered = load_sparse_npz(os.path.join(directory, file.replace('.npz', '_bl.npz')))
      ccg_corrected = ccg - ccg_jittered
      corr = (ccg_corrected - ccg_corrected.mean(-1)[:, :, None])[:, :, :maxlag]
      max_offset = np.argmax(np.abs(corr), -1)
      ccg_mat = np.choose(max_offset, np.moveaxis(corr, -1, 0))
      pos_fold = ccg_corrected[:, :, :maxlag].max(-1) > ccg_corrected.mean(-1) + n * ccg_corrected.std(-1)
      neg_fold = ccg_corrected[:, :, :maxlag].max(-1) < ccg_corrected.mean(-1) - n * ccg_corrected.std(-1)
      sharp_peak_indx = np.logical_or(pos_fold, neg_fold)
      corr = (ccg_corrected - ccg_corrected.mean(-1)[:, :, None])
      filter = np.array([[[1/maxlag]]]).repeat(maxlag, axis=2)
      corr_integral = signal.convolve(corr, filter, mode='valid')
      ccg_mat = corr_integral[:, :, 0] # average of first maxlag window
      pos_fold = ccg_mat > corr_integral.mean(-1) + n * corr_integral.std(-1)
      neg_fold = ccg_mat < corr_integral.mean(-1) - n * corr_integral.std(-1)
      sharp_interval_indx = np.logical_or(pos_fold, neg_fold)
      peak_interval_indx = np.logical_and(sharp_peak_indx, sharp_interval_indx)
      num_peak[row_ind, col_ind] = len(np.where(sharp_peak_indx)[0])
      num_interval[row_ind, col_ind] = len(np.where(sharp_interval_indx)[0])
      percents[row_ind, col_ind] = len(np.where(peak_interval_indx)[0]) / num_peak[row_ind, col_ind]
  return num_peak, num_interval, percents


def plot_overlap_ccg_sharp_peak_interval(rows, cols, num_peak, num_interval, percents, n):
  fig = plt.figure(figsize=(6*len(cols), 6*len(rows)))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  metric_names = ['number of sharp peaks', 'number of sharp intervals', 'portion of sharp peaks in sharp interval']
  plots_shape = (1, 3)
  fig = plt.figure(figsize=(24, 8))
  plt.subplot(*plots_shape, 1)
  for row_ind, row in enumerate(rows):
    plt.plot(cols, num_peak[row_ind, :], label=row, alpha=1)
  plt.gca().set_title(metric_names[0], fontsize=30, rotation=0)
  plt.xticks(rotation=90)
  plt.legend()

  plt.subplot(*plots_shape, 2)
  for row_ind, row in enumerate(rows):
    plt.plot(cols, num_interval[row_ind, :], label=row, alpha=1)
  plt.gca().set_title(metric_names[1], fontsize=30, rotation=0)
  plt.xticks(rotation=90)
  plt.legend()

  plt.subplot(*plots_shape, 3)
  for row_ind, row in enumerate(rows):
    plt.plot(cols, percents[row_ind, :], label=row, alpha=1)
  plt.gca().set_title(metric_names[2], fontsize=30, rotation=0)
  plt.xticks(rotation=90)
  plt.legend()

  plt.suptitle('{} fold'.format(n), size=30)
  plt.tight_layout()
  plt.savefig('./plots/overlap_ccg_sharp_peak_interval_{}fold.jpg'.format(n))
  
#%%

stimulus_names = ['spontaneous', 'flashes', 
        'drifting_gratings', 'static_gratings',
          'natural_scenes', 'natural_movie_one', 'natural_movie_three']
session_ids = [719161530, 750749662, 755434585, 756029989, 791319847]
measure = 'ccg'
maxlag = 12
n = 6
directory = './data/ecephys_cache_dir/sessions/adj_mat_{}_corrected/'.format(measure)
num_peak, num_interval, percents = get_overlap_ccg_sharp_peak_interval(directory, session_ids, stimulus_names, maxlag=maxlag, n=n)
# %%
plot_overlap_ccg_sharp_peak_interval(session_ids, stimulus_names, num_peak, num_interval, percents, n)

#%%
start_time = time.time()
measure = 'ccg'
min_spike = 50
n = 4
max_duration = 6
directory = './data/ecephys_cache_dir/sessions/adj_mat_{}_corrected/'.format(measure)
save_ccg_corrected_highland(directory, measure, min_spike=min_spike, max_duration=max_duration, maxlag=12, n=n)
print("--- %s minutes in total" % ((time.time() - start_time)/60))
# %%
#%%
################ plot example significant ccg for highland
def plot_example_ccg_highland(directory, measure, min_spike=50, max_duration=6, maxlag=12, n=7, window=100):
  files = os.listdir(directory)
  files.sort(key=lambda x:int(x[:9]))
  for file in files:
    if '_bl' not in file and 'gabors' not in file and '719161530' in file: #   and '719161530' in file and ('static_gratings' in file or 'gabors' in file) or 'flashes' in file
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
      num_nodes = ccg.shape[0]
      significant_ccg,significant_offset,significant_duration=np.zeros((num_nodes,num_nodes)),np.zeros((num_nodes,num_nodes)),np.zeros((num_nodes,num_nodes))
      significant_ccg[:] = np.nan
      significant_offset[:] = np.nan
      significant_duration[:] = np.nan
      ccg_corrected = ccg - ccg_jittered
      # corr = (ccg_corrected - ccg_corrected.mean(-1)[:, :, None])
      inds_2plot = []
      for duration in np.arange(max_duration, -1, -1): # reverse order, so that sharp peaks can override highland
        print('duration {}'.format(duration))
        highland_ccg, confidence_level, offset, indx = find_highland(ccg_corrected, min_spike, duration, maxlag, n)
        if np.sum(indx):
          significant_ccg[indx] = highland_ccg[indx]
          significant_offset[indx] = offset[indx]
          significant_duration[indx] = duration
      
      significant_inds = list(zip(*np.where(~np.isnan(significant_ccg))))
      print('Number of significant links: {}, density {}'.format(len(significant_inds), len(significant_inds)/(num_nodes*(num_nodes-1))))
      np.random.shuffle(significant_inds)

      for duration in np.arange(0,9,1):
        duration_inds = np.where(significant_duration==duration)
        if len(duration_inds[0]):
          indexes = np.arange(len(duration_inds[0]))
          np.random.shuffle(indexes)
          inds_2plot.append([duration_inds[0][indexes[0]], duration_inds[1][indexes[0]]])

      
      fig = plt.figure(figsize=(5*3, 5*3))
      for ind, (row_a, row_b) in enumerate(inds_2plot):
        ax = plt.subplot(3, 3, ind+1)
        highland_lag = range(int(significant_offset[row_a,row_b]), int(significant_offset[row_a,row_b]+significant_duration[row_a,row_b]+1))
        plt.plot(np.arange(window+1), ccg_corrected[row_a, row_b])
        plt.plot(highland_lag, ccg_corrected[row_a, row_b, highland_lag], 'r.--', markersize=12, alpha=0.6)
        if ind % 3 == 0:
          plt.ylabel('signigicant CCG corrected', size=20)
        if ind // 3 == 3 - 1:
          plt.xlabel('time lag (ms)', size=20)
      plt.suptitle('{} fold\n{}, {}'.format(n, mouseID, stimulus_name), size=25)
      plt.savefig('./plots/sample_significant_ccg_{}fold_highland_{}_{}.jpg'.format(n, mouseID, stimulus_name))

np.seterr(divide='ignore', invalid='ignore')
measure = 'ccg'
min_spike = 50
max_duration = 12
maxlag = 12
n = 4
directory = './data/ecephys_cache_dir/sessions/adj_mat_{}_corrected/'.format(measure)
plot_example_ccg_highland(directory, measure, min_spike=50, max_duration=max_duration, maxlag=maxlag, n=n)
#%%
################ plot example significant ccg for highland smoothed
def plot_example_ccg_highland_smoothed(directory, measure, min_spike=50, max_duration=6, maxlag=12, n=7, window=100):
  files = os.listdir(directory)
  files.sort(key=lambda x:int(x[:9]))
  for file in files:
    if '_bl' not in file and 'gabors' not in file and '719161530' in file: #   and '719161530' in file and ('static_gratings' in file or 'gabors' in file) or 'flashes' in file
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
      num_nodes = ccg.shape[0]
      significant_ccg,significant_offset,significant_duration=np.zeros((num_nodes,num_nodes)),np.zeros((num_nodes,num_nodes)),np.zeros((num_nodes,num_nodes))
      significant_ccg[:] = np.nan
      significant_offset[:] = np.nan
      significant_duration[:] = np.nan
      ccg_corrected = ccg - ccg_jittered
      # corr = (ccg_corrected - ccg_corrected.mean(-1)[:, :, None])
      inds_2plot = []
      for duration in np.arange(max_duration, -1, -1): # reverse order, so that sharp peaks can override highland
        print('duration {}'.format(duration))
        highland_ccg, confidence_level, offset, indx = find_highland(ccg_corrected, min_spike, duration, maxlag, n)
        if np.sum(indx):
          significant_ccg[indx] = highland_ccg[indx]
          significant_offset[indx] = offset[indx]
          significant_duration[indx] = duration
      
      significant_inds = list(zip(*np.where(~np.isnan(significant_ccg))))
      print('Number of significant links: {}, density {}'.format(len(significant_inds), len(significant_inds)/(num_nodes*(num_nodes-1))))
      np.random.shuffle(significant_inds)

      for duration in range(max_duration+1):
        duration_inds = np.where(significant_duration==duration)
        if len(duration_inds[0]):
          indexes = np.arange(len(duration_inds[0]))
          np.random.shuffle(indexes)
          inds_2plot.append([duration_inds[0][indexes[0]], duration_inds[1][indexes[0]]])

      
      fig = plt.figure(figsize=(5*3, 5*3))
      for ind, (row_a, row_b) in enumerate(inds_2plot[:9]):
        ax = plt.subplot(3, 3, ind+1)
        filter = np.array([1]).repeat(significant_duration[row_a,row_b]+1) # sum instead of mean
        ccg_plot = signal.convolve(ccg_corrected[row_a, row_b], filter, mode='valid', method='fft')
        highland_lag = np.array([int(significant_offset[row_a,row_b])])
        plt.plot(np.arange(len(ccg_plot)), ccg_plot)
        plt.plot(highland_lag, ccg_plot[highland_lag], 'r.--', markersize=12, alpha=0.6)
        if ind % 3 == 0:
          plt.ylabel('signigicant CCG corrected', size=20)
        if ind // 3 == 3 - 1:
          plt.xlabel('time lag (ms)', size=20)
      plt.suptitle('{} fold\n{}, {}'.format(n, mouseID, stimulus_name), size=25)
      plt.savefig('./plots/sample_significant_ccg_{}fold_highland_smoothed_{}_{}.jpg'.format(n, mouseID, stimulus_name))

np.seterr(divide='ignore', invalid='ignore')
measure = 'ccg'
min_spike = 50
max_duration = 12
maxlag = 12
n = 4
directory = './data/ecephys_cache_dir/sessions/adj_mat_{}_corrected/'.format(measure)
plot_example_ccg_highland_smoothed(directory, measure, min_spike=50, max_duration=max_duration, maxlag=maxlag, n=n)
# %%
def plot_corr_data_stimulus(data_dict, measure, n, name):
  rows, cols = get_rowcol(data_dict)
  fig = plt.figure(figsize=(6, 6))
  
  for row in rows:
    data_mean, data_std = np.zeros(len(cols)), np.zeros(len(cols))
    for col_ind, col in enumerate(cols):
      data_mean[col_ind] = np.nanmean(data_dict[row][col])
      data_std[col_ind] = np.nanstd(data_dict[row][col])
    plt.plot(cols, data_mean, alpha=0.6, label=row)
    plt.fill_between(cols, data_mean - data_std, data_mean + data_std, alpha=0.2)
  plt.legend()
  plt.xticks(rotation=90)
  plt.tight_layout()
  plt.savefig('./plots/{}_stimulus_{}_{}fold'.format(name, measure, n))
  plt.show()

plot_corr_data_stimulus(offset_dict, measure, n, 'offset')
plot_corr_data_stimulus(duration_dict, measure, n, 'duration')
#%%
def plot_multi_data_dist(data_dict, measure, n, name):
  ind = 1
  rows, cols = get_rowcol(data_dict)
  fig = plt.figure(figsize=(4*len(cols), 3*len(rows)))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      plt.subplot(len(rows), len(cols), ind)
      if row_ind == 0:
        plt.gca().set_title(cols[col_ind], fontsize=20, rotation=0)
      if col_ind == 0:
        plt.gca().text(0, 0.5 * (bottom + top), rows[row_ind],
        horizontalalignment='left',
        verticalalignment='center',
        # rotation='vertical',
        transform=plt.gca().transAxes, fontsize=20, rotation=90)
      plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
      ind += 1
      data = data_dict[row][col]
      plt.hist(data.flatten(), bins=12, density=True)
      plt.axvline(x=np.nanmean(data), color='r', linestyle='--')
      plt.xlabel('{} (ms)'.format(name))
      plt.ylabel('Probability')
      
  plt.tight_layout()
  image_name = './plots/{}_distribution_{}_{}fold.jpg'.format(name, measure, n)
  # plt.show()
  plt.savefig(image_name)

plot_multi_data_dist(offset_dict, measure, n, 'offset')
plot_multi_data_dist(duration_dict, measure, n, 'duration')
# %%
region_connection_heatmap(pos_G_dict, 'pos', active_area_dict, visual_regions, measure, n)
# %%
adj_mat = np.zeros((4,4))
adj_mat[0,1]=2
adj_mat[1,2]=3
G=nx.from_numpy_array(adj_mat)
node_idx = [3, 6,9,12]
mapping = {i:node_idx[i] for i in range(len(node_idx))}
G = nx.relabel_nodes(G, mapping)
Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
lcc_G = G.subgraph(Gcc[0])
#%%
############## plot count of areas in active neurons
def plot_pie_chart(region_counts, regions):
  ind = 1
  rows, cols = get_rowcol(region_counts)
  fig = plt.figure(figsize=(4*len(cols), 3*len(rows)))
  # fig.patch.set_facecolor('black')
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      ax = plt.subplot(len(rows), len(cols), ind)
      if row_ind == 0:
        plt.gca().set_title(cols[col_ind], fontsize=20, rotation=0)
      if col_ind == 0:
        plt.gca().text(0, 0.5 * (bottom + top), rows[row_ind],
        horizontalalignment='left',
        verticalalignment='center',
        # rotation='vertical',
        transform=plt.gca().transAxes, fontsize=20, rotation=90)
      plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
      ind += 1
      labels = region_counts[row][col].keys()
      sizes = region_counts[row][col].values()
      explode = np.zeros(len(labels))  # only "explode" the 2nd slice (i.e. 'Hogs')
      # areas_uniq = ['VISp', 'VISl', 'VISrl', 'VISal', 'VISpm', 'VISam', 'LGd', 'LP']
      colors = [customPalette[regions.index(l)] for l in labels]
      patches, texts, pcts = plt.pie(sizes, radius=sum(sizes), explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
              shadow=True, startangle=90, wedgeprops={'linewidth': 3.0, 'edgecolor': 'white'})
      for i, patch in enumerate(patches):
        texts[i].set_color(patch.get_facecolor())
      # for i in range(len(p[0])):
      #   p[0][i].set_alpha(0.6)
      ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
  plt.suptitle('Active neuron distribution', size=30)
  plt.tight_layout()
  plt.savefig('./plots/active_neuron_distri.jpg')
  # plt.show()

area_dict, mean_speed_df = load_other_data(session_ids)
min_FR = 0.002 # 2 Hz
directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
region_counts = {}
files = os.listdir(directory)
files = [f for f in files if f.endswith('.npz')]
files.sort(key=lambda x:int(x[:9]))
for file_order in range(len(files)):
  file = files[file_order] # 0, 2, 7 spontaneous, gabors, natural_movie_three
  print(file)
  mouseID = file.split('_')[0]
  stimulus_name = file.replace('.npz', '').replace(mouseID + '_', '')
  sequences = load_npz_3d(os.path.join(directory, file))
  active_neuron_inds = np.where(sequences.mean(1).sum(1) > sequences.shape[2] * min_FR)[0]
  areas = [area_dict[mouseID][key] for key in active_neuron_inds]
  if mouseID not in region_counts:
    region_counts[mouseID] = {}
  region_counts[mouseID][stimulus_name] = {r:areas.count(r) for r in visual_regions}
  # region_counts[mouseID][stimulus_name] = [areas.count(r) for r in visual_regions]
plot_pie_chart(region_counts, visual_regions)
#%%
area_dict, active_area_dict, mean_speed_df = load_other_data(session_ids)
directory = directory.replace('spiking_sequence', 'adj_mat_ccg_highland_corrected')
files = os.listdir(directory)
files.sort(key=lambda x:int(x[:9]))
for file in files:
  if file.endswith(".npz") and ('_offset' not in file) and ('_duration' not in file) and ('_bl' not in file):
    print(file)
    adj_mat = load_npz_3d(os.path.join(directory, file))
    # adj_mat = np.load(os.path.join(directory, file))
    mouseID = file.split('_')[0]
    stimulus_name = file.replace('.npz', '').replace(mouseID + '_', '')
    if adj_mat.shape[0] != len(active_area_dict[mouseID][stimulus_name].keys()):
      print('!!!!!!!!!!! Not matching!!!!!!!!!!! {} {}'.format(adj_mat.shape[0], len(active_area_dict[mouseID][stimulus_name].keys())))
#%%

# fig = plt.figure(figsize=(20,15))
# num_row, num_col = 2, 2
# for ind in range(4):
#   # plt.subplot(2, 2, ind)
#   ax1 = plt.subplot2grid((22,22), (ind//num_col*11,ind%num_col*11), colspan=10, rowspan=9)
#   ax2 = plt.subplot2grid((22,22), (ind//num_col*11+9,ind%num_col*11), colspan=10, rowspan=1)
#   pv = np.random.random((10, 10))
#   mask = np.zeros_like(pv)
#   mask[np.diag_indices_from(mask)] = True

#   sns.heatmap(pv, ax=ax1, annot=True, cmap="YlGnBu",mask=mask, linecolor='b', cbar = False)
#   ax1.xaxis.tick_top()
#   ax1.set_xticklabels(range(10),rotation=40)
#   # sns.heatmap((pd.DataFrame(pv.sum(axis=0))).transpose(), ax=ax2,  annot=True, cmap="YlGnBu", cbar=False, xticklabels=False, yticklabels=False)
#   sns.barplot(x=list(range(10)), y=pv.diagonal(), ax=ax2)
# plt.show()
# # sns.heatmap(pd.DataFrame(pv.sum(axis=1)), ax=ax3,  annot=True, cmap="YlGnBu", cbar=False, xticklabels=False, yticklabels=False)
# %%
def corr_pos_neg_weight(pos_G_dict, neg_G_dict):
  rows, cols = get_rowcol(pos_G_dict)
  pos_mean_weight, neg_mean_weight = [np.full([len(rows), len(cols)], np.nan) for _ in range(2)]
  fig = plt.figure()
  for row_ind, row in enumerate(rows):
    print(row)
    for col_ind, col in enumerate(cols):
      pos_G = pos_G_dict[row][col] if col in pos_G_dict[row] else nx.DiGraph()
      pos_mean_weight[row_ind, col_ind] = np.mean(list(nx.get_edge_attributes(pos_G, "weight").values()))
      neg_G = neg_G_dict[row][col] if col in neg_G_dict[row] else nx.DiGraph()
      neg_mean_weight[row_ind, col_ind] = np.mean(list(nx.get_edge_attributes(neg_G, "weight").values()))
  metrics = {'positive average weights':pos_mean_weight, 'negative average weights':neg_mean_weight}
  x = metrics['positive average weights'].flatten()
  y = np.abs(metrics['negative average weights'].flatten())
  plt.scatter(x, y)
  plt.xlabel('positive average weights')
  plt.ylabel('negative average weights')
  plt.xticks(rotation=90)
  # plt.yscale('symlog')
  plt.legend()
  plt.tight_layout()
  plt.show()
  print(scipy.stats.pearsonr(x, y)) # (0.9494023458038405, 7.839675225810582e-29)
  # figname = './plots/stats_{}_{}fold.jpg'.format(measure, n)
  # plt.savefig(figname)

corr_pos_neg_weight(pos_G_dict, neg_G_dict)
# %%
# import dimod
# sampler = dimod.ExactSolver()
# #%%
# # %%
# import dwave_networkx as dnx
# frustrated_edges, colors = dnx.structural_imbalance(S, sampler)
# G = G_ccg_dict['719161530']['spontaneous']
# S = S_ccg_dict['719161530']['spontaneous']
# from dwave_networkx.algorithms.social import structural_imbalance_ising
# h, J = structural_imbalance_ising(S)
# frustrated_edges, colors = dnx.structural_imbalance(S)
# %%

#%%
all_triads = get_all_signed_triads(S_ccg_dict)
triad_count = triadic_census(S_ccg_dict)
#%%
################ 003 is always dominant, meaningless
triad_types = ['003', '012', '102', '021D', '021U', '021C', '111D', '111U', '030T', '030C', '201', '120D', '120U', '120C', '210', '300']
colormaps = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                      'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                      'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn']
triad_colormap = {k:v for k,v in zip(triad_types, colormaps)}
plot_multi_pie_chart_census(triad_count, triad_types, triad_colormap, measure, n, sign=False)
#%%
########### plot time offset/duration distribution of 300 triads
scale_offset, offset_mat = triad300_bidirectional_edge_census(offset_dict, S_ccg_dict, active_area_dict, 12)
scale_duration, duration_mat = triad300_bidirectional_edge_census(duration_dict, S_ccg_dict, active_area_dict, 11)
#%%
rows, cols = get_rowcol(S_ccg_dict)
plot_data_heatmap(scale_offset, offset_mat, 12, rows, cols, 'offset', measure, n)
plot_data_heatmap(scale_duration, duration_mat, 11, rows, cols, 'duration', measure, n)
#%%
# result = p_triad_func['003'](p0, p1, p2)
# def plot_triad_relative_count():
#   for col in cols:
#     for row in rows:
#       S = S_ccg_dict[row][col]
#       p0, p1, p2 = count_triplet_connection_p(S)
p_pair_func = {
  '0': lambda p: (1 - p)**2,
  '1': lambda p: 2 * (p * (1-p)),
  '2': lambda p: p**2,
}
plot_pair_relative_count(S_ccg_dict, p_pair_func, measure, n, scale=False)
plot_pair_relative_count(S_ccg_dict, p_pair_func, measure, n, scale=True)
#%%
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

plot_triad_relative_count(S_ccg_dict, p_triad_func, measure, n)
#%%
p_pair_func = {
  '0': lambda p: (1 - p)**2,
  '1': lambda p: 2 * (p * (1-p)),
  '2': lambda p: p**2,
}
for region in visual_regions:
  print(region)
  plot_singleregion_pair_relative_count(S_ccg_dict, area_dict, region, p_pair_func, measure, n, scale=False)
  plot_singleregion_pair_relative_count(S_ccg_dict, area_dict, region, p_pair_func, measure, n, scale=True)
#%%
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

for region in visual_regions:
  print(region)
  plot_singleregion_triad_relative_count(S_ccg_dict, area_dict, region, p_triad_func, measure, n)


# triad_count_df = pd.concat([pd.DataFrame(np.concatenate((triad_count, np.array([triad_type] * len(triad_count))), 1), columns=['relative count', 'type']) for triad_type, triad_count in all_triad_count.items()], ignore_index=True)
# ax = sns.violinplot(x='type', y='relative count', data=triad_count_df, color=sns.color_palette("Set2")[0])
# ax.set(xlabel=None)
# ax.set(ylabel=None)
#%%
# def plot_singleregion_triad_relative_count_violin(G_dict, area_dict, area, p_triad_func, measure, n):
#   ind = 1
#   rows, cols = get_rowcol(G_dict)
#   fig = plt.figure(figsize=(23, 15))
#   left, width = .25, .5
#   bottom, height = .25, .5
#   right = left + width
#   top = bottom + height
#   for col in cols:
#     print(col)
#     plt.subplot(4, 2, ind)
#     plt.gca().set_title(col, fontsize=20, rotation=0)
#     plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
#     ind += 1
#     all_triad_count = defaultdict(lambda: [])
#     for row in rows:
#       G = G_dict[row][col].copy() if col in G_dict[row] else nx.DiGraph()
#       area_nodes = [k  for k, v in area_dict[row].items() if v == area]
#       G = nx.subgraph(G, area_nodes)
#       G_triad_count = nx.triads.triadic_census(G)
#       num_triplet = sum(G_triad_count.values())
#       p0, p1, p2 = count_triplet_connection_p(G)
#       for triad_type in G_triad_count:
#         relative_c = safe_division(G_triad_count[triad_type], num_triplet * p_triad_func[triad_type](p0, p1, p2))
#         all_triad_count[triad_type].append(relative_c)
    
#     triad_count_df = pd.concat([pd.DataFrame(np.concatenate((np.array(triad_count)[:, None], np.array([triad_type] * len(triad_count))[:, None]), 1), columns=['relative count', 'type']) for triad_type, triad_count in all_triad_count.items()], ignore_index=True)
#     triad_count_df['relative count'] = pd.to_numeric(triad_count_df['relative count'])
#     ax = sns.violinplot(x='type', y='relative count', data=triad_count_df, color=sns.color_palette("Set2")[0])
#     ax.set(xlabel=None)
#     ax.set(ylabel=None)
    
#     # triad_types, triad_counts = [k for k,v in all_triad_count.items()], [v for k,v in all_triad_count.items()]
#     # plt.boxplot(triad_counts, showfliers=False)
#     # plt.xticks(list(range(1, len(triad_counts)+1)), triad_types, rotation=0)
#     left, right = plt.xlim()
#     plt.hlines(1, xmin=left, xmax=right, color='r', linestyles='--', linewidth=0.5, alpha=0.6)
#     # plt.hist(data.flatten(), bins=12, density=True)
#     # plt.axvline(x=np.nanmean(data), color='r', linestyle='--')
#     # plt.xlabel('region')
#     # plt.xlabel('size')
#     # plt.yscale('log')
#     plt.ylabel('relative count')
#   plt.suptitle('Relative count of all triads in {}'.format(area), size=30)
#   plt.tight_layout(rect=[0, 0.03, 1, 0.98])
#   image_name = './plots/relative_count_violin_{}_alltriad_{}_{}fold.jpg'.format(area, measure, n)
#   # plt.show()
#   plt.savefig(image_name)

# for region in visual_regions:
#   print(region)
#   plot_singleregion_triad_relative_count_violin(S_ccg_dict, area_dict, region, p_triad_func, measure, n)
#%%
# plot_multi_bar_census(signed_triad_count, measure, n)
################## plot census of signed triads for each mouse given each stimulus
all_triads = get_all_signed_transitive_triads(S_ccg_dict)
triad_count = triad_census(all_triads)
summice_triad_count = summice_triad_census(all_triads)
meanmice_triad_percent = meanmice_triad_census(all_triads)
signed_triad_count = signed_triad_census(all_triads)
summice_signed_triad_count = summice_signed_triad_census(all_triads)
meanmice_signed_triad_percent = meanmice_signed_triad_census(all_triads)
#%%
################## num of transitive triads VS stimulus
tran_triad_types = ['030T', '120D', '120U', '300']
triad_color = {'030T':'Green', '120D':'Blue', '120U':'Red', '300':'Grey'}
triad_stimulus_error_region(triad_count, tran_triad_types, triad_color, measure, n)
#%%
################## num of signed 030T triads VS stimulus
tran_triad_type = '030T'
signs = ['+++', '++-', '+-+', '-++', '+--', '-+-', '--+', '---']
signed_030T_triad_types = [tran_triad_type+y for y in signs] # reverse order so that more positive signs have darker value
triad_color = {i:customPalette[signed_030T_triad_types.index(i)] for i in signed_030T_triad_types}
triad_stimulus_error_region(signed_triad_count, signed_030T_triad_types, triad_color, measure, n)
#%%
tran_triad_types = ['030T', '120D', '120U', '300']
triad_colormap = {'030T':'Greens', '120D':'Blues', '120U':'Reds', '300':'Purples'}
plot_multi_pie_chart_census(triad_count, tran_triad_types, triad_colormap, measure, n, False)
plot_multi_pie_chart_census(summice_triad_count, tran_triad_types, triad_colormap, measure, n, False)
plot_multi_pie_chart_census(meanmice_triad_percent, tran_triad_types, triad_colormap, measure, n, False)
#%%
signs = [
  ['+++', '++-', '+-+', '+--', '-++', '-+-', '--+', '---'],
  ['++++', '+++-', '++--', '+-++', '+-+-', '+---', '--++', '--+-', '----'],
  ['++++', '+++-', '++--', '+-++', '+-+-', '+---', '--++', '--+-', '----'],
  ['++++++', '+++++-', '++++--', '+++---', '++----', '+-----', '------']
]
signed_tran_triad_types = [x+y for x in tran_triad_types for y in signs[tran_triad_types.index(x)][::-1]] # reverse order so that more positive signs have darker value
plot_multi_pie_chart_census(signed_triad_count, signed_tran_triad_types, triad_colormap, measure, n, True)
plot_multi_pie_chart_census(summice_signed_triad_count, signed_tran_triad_types, triad_colormap, measure, n, True)
plot_multi_pie_chart_census(meanmice_signed_triad_percent, signed_tran_triad_types, triad_colormap, measure, n, True)
#%%
################ sign distribution for 030T triad
tran_triad_type = '030T'
signs = ['+++', '++-', '+-+', '+--', '-++', '-+-', '--+', '---']
signed_030T_triad_types = [tran_triad_type+y for y in signs]
# plot_multi_pie_chart_census_030T(meanmice_signed_triad_percent, signed_030T_triad_types, measure, n, False)
# plot_multi_pie_chart_census_030T(meanmice_signed_triad_percent, signed_030T_triad_types, measure, n, True)
plot_multi_pie_chart_census_030T(summice_signed_triad_count, signed_030T_triad_types, measure, n, False)
plot_multi_pie_chart_census_030T(summice_signed_triad_count, signed_030T_triad_types, measure, n, True)
#%%
######################## ONLY SIGNED FFL, NO OTHER TRAN TRIADS!!!!!!!!!!!!!!!!!
######################## break other transitive triads into ffl
signed_tran2ffl_count = signed_tran2ffl_census(S_ccg_dict, all_triads)
#%%
######################## ignore other transitive triads, only record ffl
signed_ffl_count = signed_single_motif_census(all_triads, '030T')
#%%
summice_signed_tran2ffl_count = summice(signed_tran2ffl_count)
tran_triad_type = '030T'
signs = ['+++', '++-', '+-+', '-++', '+--', '-+-', '--+', '---']
signed_030T_triad_types = [tran_triad_type+y for y in signs]
plot_multi_pie_chart_census_030T(summice_signed_tran2ffl_count, signed_030T_triad_types, measure, n, False, False)
plot_multi_pie_chart_census_030T(summice_signed_tran2ffl_count, signed_030T_triad_types, measure, n, True, False)
#%%
summice_signed_ffl_count = summice(signed_ffl_count)
tran_triad_type = '030T'
signs = ['+++', '++-', '+-+', '-++', '+--', '-+-', '--+', '---']
signed_030T_triad_types = [tran_triad_type+y for y in signs]
plot_multi_pie_chart_census_030T(summice_signed_ffl_count, signed_030T_triad_types, measure, n, False, False)
plot_multi_pie_chart_census_030T(summice_signed_ffl_count, signed_030T_triad_types, measure, n, True, False)
#%%
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

def plot_signed_pair_relative_count(G_dict, p_signed_pair_func, measure, n, log=False, scale=True):
  ind = 1
  rows, cols = get_rowcol(G_dict)
  fig = plt.figure(figsize=(23, 4))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  if scale:
    ylim = 0
    for col in cols:
      for row in rows:
        G = G_dict[row][col].copy() if col in G_dict[row] else nx.DiGraph()
        signs = list(nx.get_edge_attributes(G, "sign").values())
        p_pos, p_neg = signs.count(1)/(G.number_of_nodes()*(G.number_of_nodes()-1)), signs.count(-1)/(G.number_of_nodes()*(G.number_of_nodes()-1))
        p0, p1, p2, p3, p4, p5 = count_signed_triplet_connection_p(G)
        ylim = max(ylim, p0 / p_signed_pair_func['0'](p_pos, p_neg), p1 / p_signed_pair_func['1'](p_pos, p_neg), p2 / p_signed_pair_func['2'](p_pos, p_neg), p3 / p_signed_pair_func['3'](p_pos, p_neg), p4 / p_signed_pair_func['4'](p_pos, p_neg), p5 / p_signed_pair_func['5'](p_pos, p_neg))
  for col in cols:
    print(col)
    plt.subplot(1, 7, ind)
    plt.gca().set_title(col, fontsize=20, rotation=0)
    plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ind += 1
    all_pair_count = defaultdict(lambda: [])
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
    
    triad_types, triad_counts = [k for k,v in all_pair_count.items()], [v for k,v in all_pair_count.items()]
    plt.boxplot(triad_counts, showfliers=False)
    plt.xticks(list(range(1, len(triad_counts)+1)), triad_types, rotation=0)
    left, right = plt.xlim()
    plt.hlines(1, xmin=left, xmax=right, color='r', linestyles='--', linewidth=0.5)
    # plt.hlines(1, color='r', linestyles='--')
    if scale:
      if not log:
        plt.ylim(top=ylim)
      else:
        plt.yscale('log')
        plt.ylim(top=ylim)
    # plt.hist(data.flatten(), bins=12, density=True)
    # plt.axvline(x=np.nanmean(data), color='r', linestyle='--')
    # plt.xlabel('region')
    # plt.xlabel('size')
    plt.ylabel('relative count')
  plt.suptitle('Relative count of signed pairs', size=30)
  plt.tight_layout(rect=[0, 0.03, 1, 0.98])
  image_name = './plots/relative_count_signed_pair_scale_{}_{}fold.jpg' if scale else './plots/relative_count_signed_pair_{}_{}fold.jpg'
  # plt.show()
  plt.savefig(image_name.format(measure, n))

p_signed_pair_func = {
  '0': lambda p_pos, p_neg: (1 - p_pos - p_neg)**2,
  '1': lambda p_pos, p_neg: 2 * p_pos * (1 - p_pos - p_neg),
  '2': lambda p_pos, p_neg: 2 * p_neg * (1 - p_pos - p_neg),
  '3': lambda p_pos, p_neg: p_pos ** 2,
  '4': lambda p_pos, p_neg: 2 * p_pos * p_neg,
  '5': lambda p_pos, p_neg: p_neg ** 2,
}
# plot_signed_pair_relative_count(S_ccg_dict, p_signed_pair_func, measure, n, scale=False)
plot_signed_pair_relative_count(S_ccg_dict, p_signed_pair_func, measure, n, log=True, scale=True)
#%%
################## relative count of signed ffl
p_signed_ffl_func = {
  '030T+++': lambda p_pos, p_neg: p_pos**3,
  '030T++-': lambda p_pos, p_neg: p_pos**2 * p_neg,
  '030T+-+': lambda p_pos, p_neg: p_pos**2 * p_neg,
  '030T+--': lambda p_pos, p_neg: p_pos * p_neg**2,
  '030T-++': lambda p_pos, p_neg: p_pos**2 * p_neg,
  '030T-+-': lambda p_pos, p_neg: p_pos * p_neg**2,
  '030T--+': lambda p_pos, p_neg: p_pos * p_neg**2,
  '030T---': lambda p_pos, p_neg: p_neg**3
}

tran_triad_type = '030T'
signs = ['+++', '++-', '+-+', '-++', '+--', '-+-', '--+', '---']
signed_030T_triad_types = [tran_triad_type+y for y in signs]
plot_030T_relative_count(S_ccg_dict, signed_tran2ffl_count, signed_030T_triad_types, p_signed_ffl_func, measure, n)
#%%
plot_030T_relative_count(S_ccg_dict, signed_ffl_count, signed_030T_triad_types, p_signed_ffl_func, measure, n)
#%%
################## get signed ffl relative count Gnm and configuration
signed_ffl_count_gnm, signed_ffl_count_config = get_signed_motif_count_baseline(S_ccg_dict, num_rewire=100, motif_type='030T')
#%%
################## get signed tran2ffl relative count Gnm and configuration
signed_tran2ffl_count_gnm, signed_tran2ffl_count_config = get_signed_tran2ffl_count_baseline(S_ccg_dict, num_rewire=100)
#%%
# plot_signed_ffl_relative_count_baseline(signed_tran2ffl_count, signed_tran2ffl_count_gnm, signed_tran2ffl_count_config, signed_030T_triad_types, measure, n)
plot_signed_motif_relative_count_baseline(signed_ffl_count, signed_ffl_count_gnm, signed_ffl_count_config, signed_030T_triad_types, '030T', measure, n)
#%%
signed_120D_count = signed_single_motif_census(all_triads, '120D')
################## get signed 120D relative count Gnm and configuration
signed_120D_count_gnm, signed_120D_count_config = get_signed_motif_count_baseline(S_ccg_dict, num_rewire=100, motif_type='120D')
#%%
tran_triad_type = '120D'
signs = ['++++', '+++-', '++--', '+-++', '+-+-', '+---', '--++', '--+-', '----']
signed_120D_triad_types = [tran_triad_type+y for y in signs]
plot_signed_motif_relative_count_baseline(signed_120D_count, signed_120D_count_gnm, signed_120D_count_config, signed_120D_triad_types, '120D', measure, n)
#%%
signed_120U_count = signed_single_motif_census(all_triads, '120U')
################## get signed 120D relative count Gnm and configuration
signed_120U_count_gnm, signed_120U_count_config = get_signed_motif_count_baseline(S_ccg_dict, num_rewire=100, motif_type='120U')
#%%
tran_triad_type = '120U'
signs = ['++++', '+++-', '++--', '+-++', '+-+-', '+---', '--++', '--+-', '----']
signed_120U_triad_types = [tran_triad_type+y for y in signs]
plot_signed_motif_relative_count_baseline(signed_120U_count, signed_120U_count_gnm, signed_120U_count_config, signed_120U_triad_types, '120U', measure, n)
#%%
signed_300_count = signed_single_motif_census(all_triads, '300')
#%%
################## get signed 120D relative count Gnm and configuration
signed_300_count_gnm, signed_300_count_config = get_signed_motif_count_baseline(S_ccg_dict, num_rewire=100, motif_type='300')
#%%
tran_triad_type = '300'
signs = ['++++++', '+++++-', '+++-+-', '+-+-+-', '++++--', '++----', '------', '+-+---', '+-----']
signed_300_triad_types = [tran_triad_type+y for y in signs]
plot_signed_motif_relative_count_baseline(signed_300_count, signed_300_count_gnm, signed_300_count_config, signed_300_triad_types, '300', measure, n)
#%%
################## add time lag limitation on 030T
signed_temporal_030T_count = signed_temporal_030T_census(S_ccg_dict, all_triads)
tran_triad_type = '030T'
signs = ['+++', '++-', '+-+', '-++', '+--', '-+-', '--+', '---']
signed_030T_triad_types = [tran_triad_type+y for y in signs]
triad_color = {i:customPalette[signed_030T_triad_types.index(i)] for i in signed_030T_triad_types}
triad_stimulus_error_region(signed_temporal_030T_count, signed_030T_triad_types, triad_color, measure, n, True)
#%%
summice_signed_temporal_030T_count = summice(signed_temporal_030T_count)
tran_triad_type = '030T'
signs = ['+++', '++-', '+-+', '-++', '+--', '-+-', '--+', '---']
signed_030T_triad_types = [tran_triad_type+y for y in signs]
plot_multi_pie_chart_census_030T(summice_signed_temporal_030T_count, signed_030T_triad_types, measure, n, False, True)
plot_multi_pie_chart_census_030T(summice_signed_temporal_030T_count, signed_030T_triad_types, measure, n, True, True)
#%%
################ region distribution for signed 030T triad
signed_triad_region_count = signed_triad_region_census(all_triads, area_dict)
#%%
################ region distribution for signed temporal 030T triad
signed_temporal_030T_count = signed_temporal_030T_region_census(S_ccg_dict, all_triads, area_dict)
#%%
tran_triad_type = '030T'
signs = ['+++', '++-', '+-+', '+--', '-++', '-+-', '--+', '---']
signed_030T_triad_types = [tran_triad_type+y for y in signs]
region_types = ['_'.join([i,j,k]) for i in visual_regions for j in visual_regions for k in visual_regions]
plot_pie_chart_region_census_030T(signed_temporal_030T_count, signed_030T_triad_types, region_types, measure, n, True)
#%%
tran_triad_type = '030T'
signs = ['+++', '++-', '+-+', '+--', '-++', '-+-', '--+', '---']
signed_030T_triad_types = [tran_triad_type+y for y in signs]
region_types = ['_'.join([i,j,k]) for i in visual_regions for j in visual_regions for k in visual_regions]
plot_pie_chart_region_census_030T(signed_triad_region_count, signed_030T_triad_types, region_types, measure, n)
#%%
plot_stimulus_pie_chart_region_census_030T(signed_triad_region_count, signed_030T_triad_types, region_types, measure, n)
#%%
################ sign relative count for 030T triad
p_sign_func = {
  '030T+++': lambda p_pos, p_neg: p_pos**3,
  '030T++-': lambda p_pos, p_neg: p_pos**2 * p_neg,
  '030T+-+': lambda p_pos, p_neg: p_pos**2 * p_neg,
  '030T+--': lambda p_pos, p_neg: p_pos * p_neg**2,
  '030T-++': lambda p_pos, p_neg: p_pos**2 * p_neg,
  '030T-+-': lambda p_pos, p_neg: p_pos * p_neg**2,
  '030T--+': lambda p_pos, p_neg: p_pos * p_neg**2,
  '030T---': lambda p_pos, p_neg: p_neg**3
}

tran_triad_type = '030T'
signs = ['+++', '++-', '+-+', '-++', '+--', '-+-', '--+', '---']
signed_030T_triad_types = [tran_triad_type+y for y in signs]
plot_030T_relative_count(S_ccg_dict, signed_triad_count, signed_030T_triad_types, p_sign_func, measure, n)
#%%
rows, cols = get_rowcol(S_ccg_dict)
t_balance, num_balance, num_imbalance = np.zeros((len(rows), len(cols))), np.zeros((len(rows), len(cols))), np.zeros((len(rows), len(cols)))
balance_triads, imbalance_triads, balance_t_counts, imbalance_t_counts = {}, {}, {}, {}
# row = '719161530'
for row_ind, row in enumerate(rows):
  print(row)
  balance_triads[row], imbalance_triads[row], balance_t_counts[row], imbalance_t_counts[row] = {}, {}, {}, {}
  for col_ind, col in enumerate(cols):
    S = S_ccg_dict[row][col]
    print(col)
    balance_triads[row][col], imbalance_triads[row][col], t_balance[row_ind, col_ind], num_balance[row_ind, col_ind], num_imbalance[row_ind, col_ind], balance_t_counts[row][col], imbalance_t_counts[row][col] = calculate_triad_balance(S) 
# %%
plot_balance_stat(rows, cols, t_balance, num_balance, num_imbalance, n, measure)
# %%
tran_triad_types = ['030T', '120D', '120U', '300']
plot_balance_pie_chart(balance_t_counts, 'balance', tran_triad_types)
plot_balance_pie_chart(imbalance_t_counts, 'imbalance', tran_triad_types)
# %%
for triad_type in tran_triad_types:
  triad_region_census(balance_triads, triad_type, area_dict, visual_regions, measure, n, 'balance')
  triad_region_census(imbalance_triads, triad_type, area_dict, visual_regions, measure, n, 'imbalance')
# %%
import time
import numpy as np
import multiprocessing
from gurobipy import *
objectivevalue=[]
solveTime=[]

# G = G_ccg_dict['719161530']['spontaneous']
# S = S_ccg_dict['719161530']['spontaneous']
# G = G_ccg_dict['719161530']['flashes']
# S = S_ccg_dict['719161530']['flashes']
G = G_ccg_dict['719161530']['natural_movie_one']
# S = S_ccg_dict['719161530']['natural_movie_one']
old_nodes = sorted(list(S.nodes()))
mapping = {n:old_nodes.index(n) for n in old_nodes}
S = nx.relabel_nodes(S, mapping)
undirected_S = S.to_undirected(as_view=True)

neighbors={}
Degree=[]
for u in sorted(undirected_S.nodes()):
    neighbors[u] = list(undirected_S[u])
    Degree.append(len(neighbors[u]))
# Note that reciprocated edges are counted as one and only contribute one to the degree of each endpoint

#Finding the node with the highest unsigned degree
maximum_degree = max(Degree)
[node_to_fix]=[([i for i, j in enumerate(Degree) if j == maximum_degree]).pop()]


# Model parameters
model = Model("Continuous model for lower-bounding frustration index")
# Do you want details of branching to be reported? (0=No, 1=Yes)
model.setParam(GRB.param.OutputFlag, 0) 
# There are different methods for solving optimization models:
# (-1=automatic, 0=primal simplex, 1=dual simplex, 2=barrier, 3=concurrent, 4=deterministic concurrent)
# For problems with a large number of contstraints, barrier method is more suitable
model.setParam(GRB.param.Method, 2)
# Solving the problems without crossover takes a substantially shorter time
# in cases where there are a large number of constraints. (0=without, 1=with)
model.setParam(GRB.Param.Crossover, 0)
# What is the time limit in second?
# Here, it is set to 10 hours
model.setParam('TimeLimit', 10*3600)


# How many threads to be used for exploring the feasible space in parallel?
# Here, the minimum of 32 and the availbale CPUs is used
model.setParam(GRB.Param.Threads, min(32,multiprocessing.cpu_count()))

#This chunk of code lists the graph triangles
GraphTriangles=[]
for n1 in sorted(undirected_S.nodes()):
    neighbors1 = set(undirected_S[n1])
    for n2 in filter(lambda x: x>n1, neighbors1):
        neighbors2 = set(undirected_S[n2])
        common = neighbors1 & neighbors2
        for n3 in filter(lambda x: x>n2, common):
            GraphTriangles.append([n1,n2,n3])
#print("--- %Listed",len(GraphTriangles),"triangles for the graph")


# Create decision variables and update model to integrate new variables
# Note that the variables are defined as CONTINUOUS within the unit interval
x=[]
for i in range(0,S.number_of_nodes()):
    x.append(model.addVar(lb=0.0, ub=1, vtype=GRB.CONTINUOUS, name='x'+str(i)))
index = 0
weighted_edges=nx.get_edge_attributes(S, 'sign')
sorted_weighted_edges=[]
sorted_weighted_edges.appeplt.cm.Greensnd({})
for (u,v) in weighted_edges:
    (sorted_weighted_edges[index])[(u,v)] = weighted_edges[(u,v)]
z={}    
for (i,j) in (sorted_weighted_edges[index]):
    z[(i,j)]=model.addVar(lb=0.0, ub=1, vtype=GRB.CONTINUOUS, name='z'+str(i)+','+str(j))    

model.update()

# Set the objective function
OFV=0
for (i,j) in (sorted_weighted_edges[index]):
    OFV = OFV + (1-(sorted_weighted_edges[index])[(i,j)])/2 + ((sorted_weighted_edges[index])[(i,j)])*(x[i]+x[j]-2*z[(i,j)])          
model.setObjective(OFV, GRB.MINIMIZE)

# Add constraints to the model and update model to integrate new constraints

## ADD CORE CONSTRAINTS ##

for (i,j) in (sorted_weighted_edges[index]):
        if (sorted_weighted_edges[index])[(i,j)]==1:
            model.addConstr(z[(i,j)] <= (x[i]+x[j])/2 , 'Edge positive'+str(i)+','+str(j))
        if (sorted_weighted_edges[index])[(i,j)]==-1:
            model.addConstr(z[(i,j)] >= x[i] + x[j] -1 , 'Edge negative'+str(i)+','+str(j))            

for triangle in GraphTriangles:
    [i,j,k]=triangle
    b_ij=(i,j) in sorted_weighted_edges[index] 
    b_ik=(i,k) in sorted_weighted_edges[index]
    b_jk=(j,k) in sorted_weighted_edges[index]
    if b_ij:
        if b_ik:
            if b_jk:
                model.addConstr(x[j] + z[(i,k)] >= z[(i,j)] + z[(j,k)] , 'triangle1'+','+str(i)+','+str(j)+','+str(k))
                model.addConstr(x[i] + z[(j,k)] >= z[(i,j)] + z[(i,k)] , 'triangle2'+','+str(i)+','+str(j)+','+str(k))       
                model.addConstr(x[k] + z[(i,j)] >= z[(i,k)] + z[(j,k)] , 'triangle3'+','+str(i)+','+str(j)+','+str(k))
                model.addConstr( 1 + z[(i,j)] + z[(i,k)] + z[(j,k)] >= x[i] + x[j] + x[k] , 'triangle4'+','+str(i)+','+str(j)+','+str(k))           
            else:
                model.addConstr(x[j] + z[(i,k)] >= z[(i,j)] + z[(k,j)] , 'triangle1'+','+str(i)+','+str(j)+','+str(k))
                model.addConstr(x[i] + z[(k,j)] >= z[(i,j)] + z[(i,k)] , 'triangle2'+','+str(i)+','+str(j)+','+str(k))       
                model.addConstr(x[k] + z[(i,j)] >= z[(i,k)] + z[(k,j)] , 'triangle3'+','+str(i)+','+str(j)+','+str(k))
                model.addConstr( 1 + z[(i,j)] + z[(i,k)] + z[(k,j)] >= x[i] + x[j] + x[k] , 'triangle4'+','+str(i)+','+str(j)+','+str(k))           
        elif b_jk:
            model.addConstr(x[j] + z[(k,i)] >= z[(i,j)] + z[(j,k)] , 'triangle1'+','+str(i)+','+str(j)+','+str(k))
            model.addConstr(x[i] + z[(j,k)] >= z[(i,j)] + z[(k,i)] , 'triangle2'+','+str(i)+','+str(j)+','+str(k))       
            model.addConstr(x[k] + z[(i,j)] >= z[(k,i)] + z[(j,k)] , 'triangle3'+','+str(i)+','+str(j)+','+str(k))
            model.addConstr( 1 + z[(i,j)] + z[(k,i)] + z[(j,k)] >= x[i] + x[j] + x[k] , 'triangle4'+','+str(i)+','+str(j)+','+str(k))           
        else:
            model.addConstr(x[j] + z[(k,i)] >= z[(i,j)] + z[(k,j)] , 'triangle1'+','+str(i)+','+str(j)+','+str(k))
            model.addConstr(x[i] + z[(k,j)] >= z[(i,j)] + z[(k,i)] , 'triangle2'+','+str(i)+','+str(j)+','+str(k))       
            model.addConstr(x[k] + z[(i,j)] >= z[(k,i)] + z[(k,j)] , 'triangle3'+','+str(i)+','+str(j)+','+str(k))
            model.addConstr( 1 + z[(i,j)] + z[(k,i)] + z[(k,j)] >= x[i] + x[j] + x[k] , 'triangle4'+','+str(i)+','+str(j)+','+str(k))     
    else:
        if b_ik:
            if b_jk:
                model.addConstr(x[j] + z[(i,k)] >= z[(j,i)] + z[(j,k)] , 'triangle1'+','+str(i)+','+str(j)+','+str(k))
                model.addConstr(x[i] + z[(j,k)] >= z[(j,i)] + z[(i,k)] , 'triangle2'+','+str(i)+','+str(j)+','+str(k))       
                model.addConstr(x[k] + z[(j,i)] >= z[(i,k)] + z[(j,k)] , 'triangle3'+','+str(i)+','+str(j)+','+str(k))
                model.addConstr( 1 + z[(j,i)] + z[(i,k)] + z[(j,k)] >= x[i] + x[j] + x[k] , 'triangle4'+','+str(i)+','+str(j)+','+str(k))
            else:
                model.addConstr(x[j] + z[(i,k)] >= z[(j,i)] + z[(k,j)] , 'triangle1'+','+str(i)+','+str(j)+','+str(k))
                model.addConstr(x[i] + z[(k,j)] >= z[(j,i)] + z[(i,k)] , 'triangle2'+','+str(i)+','+str(j)+','+str(k))       
                model.addConstr(x[k] + z[(j,i)] >= z[(i,k)] + z[(k,j)] , 'triangle3'+','+str(i)+','+str(j)+','+str(k))
                model.addConstr( 1 + z[(j,i)] + z[(i,k)] + z[(k,j)] >= x[i] + x[j] + x[k] , 'triangle4'+','+str(i)+','+str(j)+','+str(k))
        elif b_jk:
            model.addConstr(x[j] + z[(k,i)] >= z[(j,i)] + z[(j,k)] , 'triangle1'+','+str(i)+','+str(j)+','+str(k))
            model.addConstr(x[i] + z[(j,k)] >= z[(j,i)] + z[(k,i)] , 'triangle2'+','+str(i)+','+str(j)+','+str(k))       
            model.addConstr(x[k] + z[(j,i)] >= z[(k,i)] + z[(j,k)] , 'triangle3'+','+str(i)+','+str(j)+','+str(k))
            model.addConstr( 1 + z[(j,i)] + z[(k,i)] + z[(j,k)] >= x[i] + x[j] + x[k] , 'triangle4'+','+str(i)+','+str(j)+','+str(k))
        else:
            model.addConstr(x[j] + z[(k,i)] >= z[(j,i)] + z[(k,j)] , 'triangle1'+','+str(i)+','+str(j)+','+str(k))
            model.addConstr(x[i] + z[(k,j)] >= z[(j,i)] + z[(k,i)] , 'triangle2'+','+str(i)+','+str(j)+','+str(k))       
            model.addConstr(x[k] + z[(j,i)] >= z[(k,i)] + z[(k,j)] , 'triangle3'+','+str(i)+','+str(j)+','+str(k))
            model.addConstr( 1 + z[(j,i)] + z[(k,i)] + z[(k,j)] >= x[i] + x[j] + x[k] , 'triangle4'+','+str(i)+','+str(j)+','+str(k))
model.update()

## ADD ADDITIONAL CONSTRAINTS (speed-ups) ##

# Colour the node with the highest degree as 1
model.addConstr(x[node_to_fix]==1 , '1stnodecolour')   
model.update()


# Solve
start_time = time.time()
model.optimize()
solveTime.append(time.time() - start_time) 

# Save optimal objective function values
obj = model.getObjective()
objectivevalue.append((obj.getValue()))
    
# Report the optimal objective function value for each instance
print('Instance', index,' solution equals',np.around(objectivevalue[index])) 
print("-"*92)
    
    # Printing the solution (optional)
    #print("Optimal values of the decision variables")
    #for v in model.getVars():
    #    print (v.varName, v.x)
    #print()    

# Save the lower bounds as a list for the next step (computing the frutsration index)
LowerBounds=np.around(objectivevalue)  

print("-"*32,"***  EXPERIMENT STATS  ***","-"*32)
print("-"*92)
print("Lower bounds on frustration index:",LowerBounds)
print()
print("Solve times:",np.around(solveTime, decimals=2))
print("Average solve time",np.mean(solveTime))
#print("Solve time Standard Deviation",np.std(solveTime))
# %%
S_ccg_dict = add_sign(G_ccg_dict)
# S = S_ccg_dict['719161530']['natural_movie_one']
# S = S_ccg_dict['719161530']['spontaneous']
# S = S_ccg_dict['719161530']['flashes']
# S = S_ccg_dict['719161530']['natural_movie_one']
rows, cols = get_rowcol(S_ccg_dict)
ng_dict, (c_mat, d_mat, fi_mat, F_mat) = {}, [np.full([len(rows), len(cols)], np.nan) for _ in range(4)]
for row_ind, row in enumerate(rows):
  print(row)
  ng_dict[row] =  {}
  for col_ind, col in enumerate(cols):
    print(col)
    S = S_ccg_dict[row][col].copy()
    print('Number of nodes: {}, number of edges: {}'.format(S.number_of_nodes(), S.number_of_edges()))
    node_group, cohesiveness, divisiveness, frustration_index, F = get_meso_macro_balance(S)
    ng_dict[row][col], c_mat[row_ind, col_ind], d_mat[row_ind, col_ind], fi_mat[row_ind, col_ind], F_mat[row_ind, col_ind] = node_group, cohesiveness, divisiveness, frustration_index, F
# %%
def plot_meso_macro_balance(c_mat, d_mat, fi_mat, F_mat, rows, cols, measure, n):
  metrics = {'Cohesiveness':c_mat, 'Divisiveness':d_mat, 'Frustration index':fi_mat, 'F(G)':F_mat}
  num_row, num_col = 2, 2
  fig = plt.figure(figsize=(5*num_col, 5*num_row))
  for i, k in enumerate(metrics):
    plt.subplot(num_row, num_col, i+1)
    metric = metrics[k]
    for row_ind, row in enumerate(rows):
      mean = metric[row_ind, :]
      plt.plot(cols, mean, '.-', label=row, alpha=0.6)
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
  # plt.show()
  figname = './plots/stat_meso_macro_balance_{}_{}fold.jpg'
  plt.savefig(figname.format(measure, n))

plot_meso_macro_balance(c_mat, d_mat, fi_mat, F_mat, rows, cols, measure, n)
# %%
def plot_bipartisan(ng_dict, rows, cols, area_dict, regions, measure, n):
  num_row, num_col = len(rows), len(cols)
  fig = plt.figure(figsize=(3*num_col, 4*num_row))
  scale_max = np.zeros(len(rows))
  for row_ind, row in enumerate(rows):
    for col_ind , col in enumerate(cols):
      plt.subplot(num_row, num_col, row_ind * num_col + col_ind+1)
      region_partisan_0 = [area_dict[row][node] for node, g in ng_dict[row][col].items() if g == 0]
      region_partisan_1 = [area_dict[row][node] for node, g in ng_dict[row][col].items() if g == 1]
      A = np.array([region_partisan_0.count(regions[0]), region_partisan_1.count(regions[0])])
      B = np.array([region_partisan_0.count(regions[1]), region_partisan_1.count(regions[1])])
      C = np.array([region_partisan_0.count(regions[2]), region_partisan_1.count(regions[2])])
      D = np.array([region_partisan_0.count(regions[3]), region_partisan_1.count(regions[3])])
      E = np.array([region_partisan_0.count(regions[4]), region_partisan_1.count(regions[4])])
      F = np.array([region_partisan_0.count(regions[5]), region_partisan_1.count(regions[5])])
      # Plot stacked bar chart
          
      plt.bar(['partisan 0', 'partisan 1'], A, label=regions[0]) #, color='cyan',
      plt.bar(['partisan 0', 'partisan 1'], B, bottom=A, label=regions[1]) #, color='green'
      plt.bar(['partisan 0', 'partisan 1'], C, bottom=A+B, label=regions[2]) #, color='red'
      plt.bar(['partisan 0', 'partisan 1'], D, bottom=A+B+C, label=regions[3]) #, color='yellow'
      plt.bar(['partisan 0', 'partisan 1'], E, bottom=A+B+C+D, label=regions[4]) #, color='yellow'
      plt.bar(['partisan 0', 'partisan 1'], F, bottom=A+B+C+D+E, label=regions[5]) #, color='yellow'
      if (A+B+C+D+E+F).max() * 1.05 > scale_max[row_ind]:
        scale_max[row_ind] = (A+B+C+D+E+F).max() * 1.05
      plt.xticks(rotation=0)
      plt.ylabel('number of neurons')
      # plt.xticks(rotation=90)
      # plt.yscale('symlog')
      if row_ind == 0:
        plt.title(col)
        if col_ind == 0:
          plt.legend(fontsize=13)
      if row_ind < num_row - 1:
        plt.tick_params(
          axis='x',          # changes apply to the x-axis
          which='both',      # both major and minor ticks are affected
          bottom=False,      # ticks along the bottom edge are off
          top=False,         # ticks along the top edge are off
          labelbottom=False) # labels along the bottom edge are off
  
  for row_ind, row in enumerate(rows):
    for col_ind , col in enumerate(cols):
      plt.subplot(num_row, num_col, row_ind * num_col + col_ind+1)
      plt.ylim(0, scale_max[row_ind])
  
  plt.tight_layout()
  # plt.suptitle(k, fontsize=14, rotation=0)
  # plt.show()
  figname = './plots/region_bipartisan_{}_{}fold.jpg'
  plt.savefig(figname.format(measure, n))

plot_bipartisan(ng_dict, rows, cols, area_dict, visual_regions, measure, n)

#%%
################# get correlation window fraction
directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
start_time = time.time()
c_window = get_correlation_window(directory, offset_dict, duration_dict)
print("--- %s minutes" % ((time.time() - start_time)/60))
with open('c_window.pkl', 'wb') as f:
  pickle.dump(c_window, f)
#%%
with open('c_window.pkl', 'rb') as f:
  c_window_l = pickle.load(f)
#%%
plot_multi_correlation_window(c_window_l, measure, n)
#%%
plot_correlation_window_box(c_window_l, measure, n)
#%%
################# time to return to source neuron
def get_return_time(G_dict):
  rows, cols = get_rowcol(G_dict)
  time_dict = {}
  for row in rows:
    print(row)
    time_dict[row] = {}
    for col in cols:
      time_dict[row][col] = []
      G = G_dict[row][col].copy()
      nodes = G.nodes()
      for node in nodes:
        shortest_time = []
        successors = G.successors(node)
        for s in successors:
          if nx.has_path(G, s, node):
            shortest_time.append(nx.shortest_path_length(G, source=node, target=s)+nx.shortest_path_length(G, source=s, target=node))
            # shortest_time.append(nx.shortest_path_length(G, source=node, target=s, weight='delay')+nx.shortest_path_length(G, source=s, target=node, weight='delay'))
        if len(shortest_time):
          time_dict[row][col].append(min(shortest_time))
        # else:
        #   time_dict[row][col].append(0)
  return time_dict

time_dict = get_return_time(S_ccg_dict)
# %%
def plot_return_time(time_dict, measure, n):
  ind = 1
  rows, cols = get_rowcol(time_dict)
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
      sns.histplot(data=time_dict[row][col], stat='probability', kde=True, linewidth=0)
      plt.axvline(x=np.nanmean(time_dict[row][col]), color='r', linestyle='--')
      plt.xlabel('return hops')
      # plt.xlabel('return time ms')
  plt.suptitle('return hops', size=50)
  # plt.suptitle('return time (ms)', size=50)
  plt.tight_layout(rect=[0, 0.03, 1, 0.98])
  plt.savefig('./plots/multi_return_hop{}_{}fold.jpg'.format(measure, n))

plot_return_time(time_dict, measure, n)
# %%
def plot_time_degree(G_dict, measure, n):
  ind = 1
  rows, cols = get_rowcol(G_dict)
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
      G = G_dict[row][col]
      out_degree, time, duration, delay = [], [], [], []
      for node in G.nodes():
        if G.out_degree(node):
          out_degree.append(G.out_degree(node))
          temp1, temp2, temp3 = [], [], []
          for neighbor in G.successors(node):
            temp1.append(G[node][neighbor]['offset'])
            temp2.append(G[node][neighbor]['duration'])
            temp3.append(G[node][neighbor]['delay'])
          time.append(np.mean(temp1))
          duration.append(np.mean(temp2))
          delay.append(np.mean(temp3))

      plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
      ind += 1
      plt.scatter(out_degree, time, label='time lag', color='r', alpha=0.4)
      plt.scatter(out_degree, duration, label='duration', color='b', alpha=0.4)
      plt.scatter(out_degree, delay, label='delay', color='g', alpha=0.4)
      plt.xlabel('out degree')
      plt.ylabel('t (ms)')
      plt.legend()
      # plt.xlabel('return time ms')
  # plt.suptitle('return time (ms)', size=50)
  plt.tight_layout(rect=[0, 0.03, 1, 0.98])
  plt.savefig('./plots/multi_time_degree{}_{}fold.jpg'.format(measure, n))

plot_time_degree(S_ccg_dict, measure, n)
# %%
def plot_time_degree_edge(G_dict, measure, n):
  rows, cols = get_rowcol(G_dict)
  fig = plt.figure(figsize=(9*len(cols)/2, 6*2))
  left, width = .25, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  ind = 1
  for col in cols:
    print(col)
    degree, offset, duration, delay = [], [], [], []
    for row in rows:
      G = G_dict[row][col]
      for edge in G.edges():
        offset.append(G[edge[0]][edge[1]]['offset'])
        duration.append(G[edge[0]][edge[1]]['duration'])
        delay.append(G[edge[0]][edge[1]]['delay'])
        degree.append(sum([G.out_degree(edge[0]), G.in_degree(edge[1])])/2)
    uniq_degree = sorted(set(degree))
    degree, offset, duration, delay = np.array(degree), np.array(offset), np.array(duration), np.array(delay)
    offset_mean, offset_std, duration_mean, duration_std, delay_mean, delay_std = [[] for _ in range(6)]
    for d in uniq_degree:
      idx = np.where(degree==d)[0]
      offset_mean.append(offset[idx].mean())
      offset_std.append(offset[idx].std())
      duration_mean.append(duration[idx].mean())
      duration_std.append(duration[idx].std())
      delay_mean.append(delay[idx].mean())
      delay_std.append(delay[idx].std())

    offset_mean, offset_std, duration_mean, duration_std, delay_mean, delay_std = np.array(offset_mean), np.array(offset_std), np.array(duration_mean), np.array(duration_std), np.array(delay_mean), np.array(delay_std)
    ax=plt.subplot(2, int(np.ceil(len(cols)/2)), ind)
    plt.gca().set_title(col, fontsize=30, rotation=0)
    plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ind += 1
    plt.plot(uniq_degree, offset_mean, label='time lag', color='r', alpha=0.4)
    plt.fill_between(uniq_degree, offset_mean - offset_std, offset_mean + offset_std, color='r', alpha=0.2)
    plt.plot(uniq_degree, duration_mean, label='duration lag', color='b', alpha=0.4)
    plt.fill_between(uniq_degree, duration_mean - duration_std, duration_mean + duration_std, color='b', alpha=0.2)
    plt.plot(uniq_degree, delay_mean, label='delay', color='g', alpha=0.4)
    plt.fill_between(uniq_degree, delay_mean - delay_std, delay_mean + delay_std, color='g', alpha=0.2)
    plt.xlabel('mean degree')
    plt.ylabel('t (ms)')
    plt.legend()
  plt.tight_layout()
  figname = './plots/time_degree_edge_{}_{}fold.jpg'.format(measure, n)
  plt.savefig(figname)

plot_time_degree_edge(S_ccg_dict, measure, n)
#%%
################## initilize each neuron with 1-d state (color) and propagate
# plot_state(G_ccg_dict, 7, 4, active_area_dict, measure, n, timesteps=50)
epsilon_list = [0, 5, 10, 20, 50, 100]
# epsilon = 5
for row_ind in range(2, 8):
  print(row_ind)
  for epsilon in epsilon_list:
    plot_state(G_ccg_dict, row_ind, epsilon, active_area_dict, measure, n, timesteps=50)
#%%
##################### plot state change with one hot encoding
# plot_state_onehot(G_ccg_dict, 7, 3, 4, active_area_dict, measure, n, timesteps=200)
for row_ind in range(0, 8):
  print(row_ind)
  if row_ind != 1:
    for col_ind in range(0, 8):
      plot_state_onehot(G_ccg_dict, row_ind, col_ind, 4, active_area_dict, measure, n, timesteps=200)
# %%
# plot_state_jsdistance(G_ccg_dict, 7, 4, active_area_dict, measure, n, timesteps=200)
for row_ind in range(0, 8):
  print(row_ind)
  if row_ind != 1:
    plot_state_jsdistance(G_ccg_dict, row_ind, 4, active_area_dict, measure, n, timesteps=200)
#%%
################# original region fraction for each region
plot_state_region_fraction(G_ccg_dict, 4, active_area_dict, measure, n, timesteps=200)
#%%
################# bar plot of steady state for each region and stimulus
plot_steady_distribution(G_ccg_dict, 4, active_area_dict, measure, n, timesteps=200)
#%%
################# dominance score
plot_dominance_score(G_ccg_dict, 4, active_area_dict, measure, n, timesteps=200)
#%%
################# steps to convergence
epsilon_list = np.arange(0, 202, 2)
plot_step2convergence(G_ccg_dict, epsilon_list, active_area_dict, measure, n, step2confirm=5, maxsteps=2000)
#%%
################# stable region fraction against epsilon
epsilon_list = np.arange(0, 202, 2)
# epsilon_list = np.arange(0, 51, 1)
plot_region_frac_epsilon(G_ccg_dict, epsilon_list, active_area_dict, measure, n, step2confirm=5, maxsteps=2000)
#%%
################# rank region based on size
for row in rows:
  uniq_area, counts = np.unique(list(active_area_dict[row].values()),return_counts=True)
  print(row, [x for _, x in sorted(zip(counts, uniq_area), reverse=True)])
#%%
################# rank region based on out degree
df = pd.DataFrame(index=rows, columns=cols)
for row in rows:
  for col_ind in range(1, 5):
    col = cols[col_ind]
    G = G_ccg_dict[row][col]
    uniq_area = np.unique(list(active_area_dict[row].values()))
    area_count = []
    for area in uniq_area:
      nodes = [node for node in active_area_dict[row] if active_area_dict[row][node]==area]
      area_count.append(sum([G.in_degree(node) for node in nodes]))
    df.loc[row][col] = uniq_area[np.argmax(area_count)]
df.drop(['spontaneous', 'natural_scenes', 'natural_movie_one', 'natural_movie_three'], axis=1)
#%%

# %%
