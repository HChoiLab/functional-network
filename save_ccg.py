#%%
from library import *
#%%
############## calculate ccg and save
np.seterr(divide='ignore', invalid='ignore')
############# save correlation matrices #################
# min_len, min_num = (260000, 739)
# min_len, min_num = (10000, 29)
# min_FR = 0.002 # 2 Hz
# min_spikes = min_duration * min_FR
# min_spikes = min_len * min_FR
measure = 'ccg'
directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
num_baseline = 1
# files = os.listdir(directory)
# files = [f for f in files if f.endswith('.npz')]
# files.sort(key=lambda x:int(x[:9]))

# stimulus_names = ['flash_light', 'flash_dark']
# session_ids = [719161530, 750332458, 750749662, 754312389, 755434585, 756029989, 791319847, 797828357]

stimulus_names = ['spontaneous',
        'drifting_gratings', 'static_gratings',
          'natural_scenes', 'natural_movie_one', 'natural_movie_three']
# stimulus_names = ['flash_light', 'flash_dark']
session_ids = [719161530, 750749662, 754312389, 755434585, 756029989, 791319847]
# session_ids = [750332458, 797828357]

path = os.path.join(directory.replace('spiking_sequence', 'adj_mat_ccg_corrected'))
inds_path = './data/ecephys_cache_dir/sessions/active_inds/'
# path = os.path.join(directory.replace('spiking_sequence', 'adj_mat_xcorr_shuffled'))
if not os.path.exists(path):
  os.makedirs(path)
combination = list(itertools.product(session_ids, stimulus_names))
file_order = int(sys.argv[1])
# file_order = 1

session_id, stimulus_name = combination[file_order]
file = '{}_{}.npz'.format(session_id, stimulus_name)
# (num_neuron, num_trial, T)
sequences = load_npz_3d(os.path.join(directory, file))
active_neuron_inds = np.load(os.path.join(inds_path, str(session_id)+'.npy'))
# active_neuron_inds = sequences.mean(1).sum(1) > sequences.shape[2] * min_FR
sequences = sequences[active_neuron_inds]
print('{} spike train shape: {}'.format(file, sequences.shape))
# sequences = concatenate_trial(sequences, min_duration, min_len)
# sequences = sequences[np.count_nonzero(sequences[:, :min_len], axis=1) > min_spikes, :min_len]
fname = os.path.join(path, file)
start_time = time.time()
save_mean_ccg_corrected(sequences=sequences, fname=fname, num_jitter=num_baseline, L=25, window=100, disable=False)
print("--- %s minutes" % ((time.time() - start_time)/60))
