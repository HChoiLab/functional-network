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
