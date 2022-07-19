#%%
from library import *
#%%
start_time = time.time()
measure = 'ccg'
min_spike = 50
n = 4
max_duration = 11
directory = './data/ecephys_cache_dir/sessions/adj_mat_{}_corrected/'.format(measure)
save_ccg_corrected_highland_new(directory, measure, min_spike=min_spike, max_duration=max_duration, maxlag=12, n=n)
print("--- %s minutes in total" % ((time.time() - start_time)/60))
# %%
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
# %%
