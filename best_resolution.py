#%%
from library import *
################## save community partitions and Hamiltonian VS resolution
start_time = time.time()
area_dict, active_area_dict, mean_speed_df = load_other_data(session_ids)
directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
path = directory.replace('spiking_sequence', 'adj_mat_ccg_highland_corrected')
if not os.path.exists(path):
  os.makedirs(path)
G_ccg_dict, offset_dict, duration_dict = load_highland_xcorr(path, active_area_dict, weight=True)
measure = 'ccg'
G_ccg_dict = remove_gabor(G_ccg_dict)
G_ccg_dict = remove_thalamic(G_ccg_dict, area_dict, visual_regions)
n = 4
#%%
num_repeat, num_rewire = 50, 50
resolution_list = np.arange(0, 2.1, 0.1)
comms_dict, metrics = comms_Hamiltonian_resolution(G_ccg_dict, resolution_list, num_repeat, num_rewire, False)
with open('comms_dict.pkl', 'wb') as f:
  pickle.dump(comms_dict, f)
with open('metrics.pkl', 'wb') as f:
  pickle.dump(metrics, f)
print("--- %s minutes" % ((time.time() - start_time)/60))
#%%
