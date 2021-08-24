#%%
import os
import time
import pandas as pd
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

def remove_outlier(array):
  mean = np.mean(array)
  standard_deviation = np.std(array)
  distance_from_mean = abs(array - mean)
  max_deviations = 2
  not_outlier = distance_from_mean < max_deviations * standard_deviation
  return array[not_outlier]

def get_region_spiking_sequence(session_id, stimulus_name, structure_acronym):
  session = cache.get_session_data(session_id)
  print(session.metadata)
  print(session.structurewise_unit_counts)
  presentations = session.get_stimulus_table(stimulus_name)
  units = session.units[session.units["ecephys_structure_acronym"]==structure_acronym]
  time_step = 0.001
  time_bins = np.arange(0, 2.0 + time_step, time_step)
  histograms = session.presentationwise_spike_counts(
      stimulus_presentation_ids=presentations.index.values,  
      bin_edges=time_bins,
      unit_ids=units.index.values)
  return histograms

def get_whole_spiking_sequence(session_id, stimulus_name):
  time_stimulus_dict = {'spontaneous':2, 'flashes':0.25, 'gabors':0.2, 'drifting_gratings':2, 'static_gratings':0.25, 'natural_scenes':0.25, 'natural_movie_one':0.03, 'natural_movie_three':0.03}
  session = cache.get_session_data(session_id)
  presentations = session.get_stimulus_table(stimulus_name)
  units = session.units
  time_step = 0.001
  time_bins = np.arange(0, time_stimulus_dict[stimulus_name] + time_step, time_step)
  histograms = session.presentationwise_spike_counts(
      stimulus_presentation_ids=presentations.index.values,  
      bin_edges=time_bins,
      unit_ids=units.index.values)
  return histograms

def get_regions_spiking_sequence(session_id, stimulus_name, regions):
  session = cache.get_session_data(session_id,
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
    stim_table = session.get_stimulus_table([stimulus_name])
    stim_table=stim_table.rename(columns={"start_time": "Start", "stop_time": "End"})
    if 'natural_movie' in stimulus_name:
        frame_times = stim_table.End-stim_table.Start
        print('frame rate:', 1/np.mean(frame_times), 'Hz', np.mean(frame_times))
        # stim_table.to_csv(output_path+'stim_table_'+stimulus_name+'.csv')
        # chunch each movie clip
        stim_table = stim_table[stim_table.frame==0]
        stim_table = stim_table.drop(['End'], axis=1)
        duration = np.mean(remove_outlier(np.diff(stim_table.Start.values))[:10]) - 1e-4
    elif stimulus_name=='spontaneous':
        index = np.where(stim_table.duration>=20)[0]
        if len(index): # only keep the longest spontaneous; has to be longer than 20 sec
            duration=20
            stimulus_presentation_ids = stim_table.index[index]
    else:
        ISI = np.mean(session.get_inter_presentation_intervals_for_stimulus([stimulus_name]).interval.values)
        duration = round(np.mean(stim_table.duration.values), 2)+ISI
    if stimulus_name == 'gabors':
      duration -= 0.02
    try: stimulus_presentation_ids
    except NameError: stimulus_presentation_ids = stim_table.index.values
    #binarize tensor
    # binarize with 1 second bins
    time_bin_edges = np.linspace(0, duration, int(duration*1000)+1)

    # and get a set of units with only decent snr
    #decent_snr_unit_ids = session.units[
    #    session.units['snr'] >= 1.5
    #].index.values
    cortical_units_ids = np.array([idx for idx, ccf in enumerate(df.ccf.values) if ccf in regions])
    print('Number of units is {}, duration is {}'.format(len(cortical_units_ids), duration))
    # get binarized tensor
    df_cortex = df.iloc[cortical_units_ids]
    histograms = session.presentationwise_spike_counts(
        bin_edges=time_bin_edges,
        stimulus_presentation_ids=stimulus_presentation_ids,
        unit_ids=df_cortex.unit_id.values
    )
    return histograms
    # df_cortex.ccf is the area dict
# %%
start_time = time.time()
stimulus_names = ['spontaneous', 'flashes', 'gabors',
       'static_gratings', 'drifting_gratings',
        'natural_movie_one', 'natural_movie_three', 'natural_scenes']
# session_ids = sessions[sessions.session_type=='brain_observatory_1.1'].index.values # another one is functional_connectivity
session_ids = [719161530, 750749662, 755434585, 756029989, 791319847]
visual_regions = ['VISp', 'VISl', 'VISrl', 'VISal', 'VISpm', 'VISam', 'LGd', 'LP']
directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
if not os.path.isdir(directory):
  os.mkdir(directory)
ind = 1
all_num = len(session_ids)*len(stimulus_names)
for session_id in session_ids:
  for stimulus_name in stimulus_names:
      histograms = get_regions_spiking_sequence(session_id, stimulus_name, visual_regions)
      mean_histograms = histograms.mean(dim="stimulus_presentation_id")
      mean_histograms.to_netcdf((directory + '{}_{}.nc').format(session_id, stimulus_name))
      print('finished {}, {},  {} / {}'.format(session_id, stimulus_name, ind, all_num))
      ind += 1
print("--- %s minutes ---" % ((time.time() - start_time) / 60))
# %%
# session_id = 756029989
# stimulus_name = 'spontaneous'
# regions = ['VISp', 'VISl', 'VISrl', 'VISal', 'VISpm', 'VISam', 'LGd', 'LP']

# %%
