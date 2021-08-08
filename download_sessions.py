#%%
import os
# import shutil
import numpy as np
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
data_directory = './data/ecephys_cache_dir'
manifest_path = os.path.join(data_directory, "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
sessions = cache.get_session_table()
# filtered_sessions = sessions[(sessions.sex == 'M') & \
#                              (sessions.full_genotype.str.find('Sst') > -1) & \
#                              (sessions.session_type == 'brain_observatory_1.1') & \
#                              (['VISl' in acronyms for acronyms in 
#                                sessions.ecephys_structure_acronyms])]
probes = cache.get_probes()
channels = cache.get_channels()
units = cache.get_units()
num_sessions = len(sessions)
num_neurons = len(units)
num_probes = len(units['name'].unique())

def get_spiking_sequence(session_id, stimulus_name, structure_acronym):
  session = cache.get_session_data(session_id)
  print(session.metadata)
  print(session.structurewise_unit_counts)
  presentations = session.get_stimulus_table(stimulus_name)
  units = session.units[session.units["ecephys_structure_acronym"]==structure_acronym]
  time_step = 0.01
  time_bins = np.arange(0, 1.5 + time_step, time_step)
  histograms = session.presentationwise_spike_counts(
      stimulus_presentation_ids=presentations.index.values,  
      bin_edges=time_bins,
      unit_ids=units.index.values)
  return histograms

def get_whole_spiking_sequence(session_id, stimulus_name):
  session = cache.get_session_data(session_id)
  presentations = session.get_stimulus_table(stimulus_name)
  units = session.units
  time_step = 0.01
  time_bins = np.arange(0, 1.5 + time_step, time_step)
  histograms = session.presentationwise_spike_counts(
      stimulus_presentation_ids=presentations.index.values,  
      bin_edges=time_bins,
      unit_ids=units.index.values)
  return histograms
# %%
# # mean histogram is a matrix
# mean_histograms = histograms.mean(dim="stimulus_presentation_id")
# fig, ax = plt.subplots(figsize=(8, 8))
# ax.pcolormesh(
#     mean_histograms["time_relative_to_stimulus_onset"], 
#     np.arange(mean_histograms["unit_id"].size),
#     mean_histograms.T, 
#     vmin=0,
#     vmax=1)
# ax.set_ylabel("unit", fontsize=24)
# ax.set_xlabel("time relative to stimulus onset (s)", fontsize=24)
# ax.set_title("peristimulus time histograms for VISp units on flash presentations", fontsize=24)
# plt.show()
# %%
stimulus_names = ['spontaneous', 'flashes', 'gabors',
       'static_gratings', 'drifting_gratings', 'drifting_gratings_contrast',
        'natural_movie_one', 'natural_movie_three', 'natural_scenes']
session_ids = sessions[sessions.session_type=='brain_observatory_1.1'].index.values # another one is functional_connectivity
session_ids = [719161530, 750749662, 755434585, 756029989, 791319847]
directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
if not os.path.isdir(directory):
  os.mkdir(directory)
ind = 1
all_num = len(session_ids)*len(stimulus_names)
for session_id in session_ids:
  for stimulus_name in stimulus_names:
      histograms = get_whole_spiking_sequence(session_id, stimulus_name)
      mean_histograms = histograms.mean(dim="stimulus_presentation_id")
      mean_histograms.to_netcdf((directory + '{}_{}.nc').format(session_id, stimulus_name))
      print('finished {} / {}'.format(ind, all_num))
      ind += 1
# %%
