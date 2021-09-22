# %%
import os
import sys
import xarray as xr
# import shutil
import numpy as np
import pandas as pd
import time
from matplotlib import pyplot as plt
import networkx as nx
# from requests.sessions import session
from scipy.spatial.distance import pdist, squareform, cosine
from sklearn.metrics.pairwise import cosine_similarity
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
data_directory = './data/ecephys_cache_dir'
manifest_path = os.path.join(data_directory, "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

def compute_adj(path, measure):
  for file in os.listdir(path):
    print(file)
    if file.endswith(".npy"):
        sequences = np.load(os.path.join(path, file)).T
        sequences = np.nan_to_num(sequences)
        if measure == 'pearson':
          adj_mat = np.corrcoef(sequences)
        elif measure == 'cosine':
          adj_mat = cosine_similarity(sequences)
        elif measure == 'correlation':
          adj_mat = squareform(pdist(sequences, 'correlation'))
        else:
          sys.exit('Unrecognized measure value!!! Choices: pearson, cosin, correlation.')
        adj_mat = np.nan_to_num(adj_mat)
        np.fill_diagonal(adj_mat, 0)
        np.save(os.path.join(path.replace('spiking_sequence', '{}_matrix'.format(measure)), file.replace('.nc', '.npy')), adj_mat)

def corr_mat(sequences, measure, threshold=0.5, percentile=90):
  if measure == 'pearson':
    adj_mat = np.corrcoef(sequences)
  elif measure == 'cosine':
    adj_mat = cosine_similarity(sequences)
  elif measure == 'correlation':
    adj_mat = squareform(pdist(sequences, 'correlation'))
  else:
    sys.exit('Unrecognized measure value!!! Choices: pearson, cosin, correlation.')
  adj_mat = np.nan_to_num(adj_mat)
  np.fill_diagonal(adj_mat, 0)
  if measure == 'pearson':
    adj_mat[adj_mat < threshold] = 0
  else:
    adj_mat[np.where(adj_mat<np.nanpercentile(np.abs(adj_mat), percentile))] = 0
  return adj_mat

def generate_graph(sequences, measure, lcc=True, weight=False, threshold=0.5, percentile=90):
  adj_mat = corr_mat(sequences, measure, threshold, percentile)
  if not weight:
    adj_mat[adj_mat.nonzero()] = 1
  G = nx.from_numpy_array(adj_mat) # same as from_numpy_matrix
  if lcc: # extract the largest (strongly) connected components
    if np.allclose(adj_mat, adj_mat.T, rtol=1e-05, atol=1e-08): # if the matrix is symmetric
      largest_cc = max(nx.connected_components(G), key=len)
    else:
      largest_cc = max(nx.strongly_connected_components(G), key=len)
    G = nx.subgraph(G, largest_cc)
  return G

def generate_region_graph(sequences, indices, measure, lcc=True, weight=False, threshold=0.3, percentile=90):
  sequences = sequences[indices, :]
  # sequences = np.nan_to_num(sequences)
  adj_mat = corr_mat(sequences, measure, threshold, percentile)
  if not weight:
    adj_mat[adj_mat.nonzero()] = 1
  G = nx.from_numpy_array(adj_mat) # same as from_numpy_matrix
  if lcc: # extract the largest (strongly) connected components
    if np.allclose(adj_mat, adj_mat.T, rtol=1e-05, atol=1e-08): # if the matrix is symmetric
      largest_cc = max(nx.connected_components(G), key=len)
    else:
      largest_cc = max(nx.strongly_connected_components(G), key=len)
    G = nx.subgraph(G, largest_cc)
  return G

def remove_outlier(array):
  mean = np.mean(array)
  standard_deviation = np.std(array)
  distance_from_mean = abs(array - mean)
  max_deviations = 2
  not_outlier = distance_from_mean < max_deviations * standard_deviation
  return array[not_outlier]


def bootstrap(histograms, num_presentations):
  pop = histograms.data
  axis = histograms.dims.index('stimulus_presentation_id')
  sample_data = None
  if axis == 0 and pop.shape[0] < num_presentations:
    sample_data = pop[np.random.randint(pop.shape[0], size=(num_presentations - pop.shape[0], pop.shape[1], pop.shape[2])), np.arange(pop.shape[1]).reshape(1, -1, 1), np.arange(pop.shape[2]).reshape(1, 1, -1)]
  elif axis == 1 and pop.shape[1] < num_presentations:
    sample_data = pop[np.arange(pop.shape[0]).reshape(-1, 1, 1), np.random.randint(pop.shape[1], size=(pop.shape[0], num_presentations - pop.shape[1], pop.shape[2])), np.arange(pop.shape[2]).reshape(1, 1, -1)]
  elif axis == 2 and pop.shape[2] < num_presentations:
    sample_data = pop[np.arange(pop.shape[0]).reshape(-1, 1, 1), np.arange(pop.shape[1]).reshape(1, -1, 1), np.random.randint(pop.shape[2], size=(pop.shape[0], pop.shape[1], num_presentations - pop.shape[2]))]
  if sample_data is not None:
    # pop = np.concatenate((pop, sample_data), axis=axis)
    samples = xr.DataArray(data=sample_data, dims=histograms.dims, coords={"stimulus_presentation_id":np.arange(sample_data.shape[axis]), 'time_relative_to_stimulus_onset':histograms['time_relative_to_stimulus_onset'].data, 'unit_id':histograms['unit_id'].data})
    histograms = xr.concat((histograms, samples),dim="stimulus_presentation_id")
  return histograms

def get_regions_spiking_sequence(session_id, stimulus_name, regions, resolution):
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
    duration = min(duration, 3)
    # binarize with 1 second bins
    time_bin_edges = np.linspace(0, duration, int(duration / resolution)+1)

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

def load_npy_regions_as_graph_whole(directory, regions, weight, measure, threshold, percentile):
  G_dict = {}
  area_dict = {}
  files = os.listdir(directory)
  files.sort(key=lambda x:int(x[:9]))
  for file in files:
    print(file)
    if file.endswith(".npy"):
      sequences = np.load(os.path.join(directory, file)).T
      # data = xr.open_dataset(os.path.join(directory, file)).to_array()
      # if len(data.data.shape) > 2:
      #   sequences = data.data.squeeze().T
      # else:
      #   sequences = data.data.T
      mouseID = file.split('_')[0]
      if not mouseID in G_dict:
        session = cache.get_session_data(int(mouseID),
                                amplitude_cutoff_maximum=np.inf,
                                presence_ratio_minimum=-np.inf,
                                isi_violations_maximum=np.inf)
        df = session.units
        df = df.rename(columns={"channel_local_index": "channel_id", 
                                "ecephys_structure_acronym": "ccf", 
                                "probe_id":"probe_global_id", 
                                "probe_description":"probe_id",
                                'probe_vertical_position': "ypos"})
        cortical_units_ids = np.array([idx for idx, ccf in enumerate(df.ccf.values) if ccf in regions])
        df_cortex = df.iloc[cortical_units_ids]
        instruction = df_cortex.ccf
        if set(instruction.unique()) == set(regions): # if the mouse has all regions recorded
          G_dict[mouseID] = {}
        instruction = instruction.reset_index()
      stimulus_name = file.replace('.npy', '').replace(mouseID + '_', '')
      G_dict[mouseID][stimulus_name] = generate_graph(sequences, measure=measure, lcc=True, weight=weight, threshold=threshold, percentile=percentile)
      if not mouseID in area_dict:
        area_dict[mouseID] = {}
      for i in range(instruction.shape[0]):
        area_dict[mouseID][i] = instruction.ccf.iloc[i]
  return G_dict, area_dict

def plot_all_spikes(directory, session_ids, stimulus_names, resolution):
    fig = plt.figure(figsize=(4*len(stimulus_names), 3*len(session_ids)))
    left, width = .25, .5
    bottom, height = .25, .5
    right = left + width
    top = bottom + height
    ind = 1
    for session_ind, session_id in enumerate(session_ids):
        print(session_id)
        for stimulus_ind, stimulus_name in enumerate(stimulus_names):
            plt.subplot(len(session_ids), len(stimulus_names), ind)
            if session_ind == 0:
                plt.gca().set_title(stimulus_name, fontsize=20, rotation=0)
            if stimulus_ind == 0:
                plt.gca().text(0, 0.5 * (bottom + top), session_id,
            horizontalalignment='left',
            verticalalignment='center',
            # rotation='vertical',
            transform=plt.gca().transAxes, fontsize=20, rotation=90)
            plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
            ind += 1
            sequences = np.load(os.path.join(directory, '{}_{}.npy'.format(session_id, stimulus_name)))
            plt.plot(sequences[-100:, :10], alpha=0.2)
            plt.xlabel('step')
            plt.ylabel('average number of spikes')
    plt.suptitle('resolution={}s'.format(resolution), fontsize=30)
    plt.tight_layout()
    plt.savefig('./plots/all_spikes_{}.jpg'.format(resolution))
# %%
# measure = 'pearson'
# threshold = 0.5
# percentile = 99
# weight = False # unweighted network
# visual_regions = ['VISp', 'VISl', 'VISrl', 'VISal', 'VISpm', 'VISam', 'LGd', 'LP']
# directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
# G_dict, area_dict = load_npy_regions_as_graph_whole(directory, visual_regions, weight, measure, threshold, percentile)

# %%
############# save npy files #############
start_time = time.time()
stimulus_names = ['spontaneous', 'flashes', 'gabors',
        'drifting_gratings', 'static_gratings',
          'natural_scenes', 'natural_movie_one', 'natural_movie_three']
session_ids = [719161530, 750749662, 755434585, 756029989, 791319847]
visual_regions = ['VISp', 'VISl', 'VISrl', 'VISal', 'VISpm', 'VISam', 'LGd', 'LP']
resolution_dict = {'spontaneous':0.08, 'flashes':0.008, 'gabors':0.0004, 'drifting_gratings':0.0016, 'static_gratings':0.00012, 'natural_scenes':0.00012, 'natural_movie_one':0.04, 'natural_movie_three':0.08}
resolution = 0.002
directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
if not os.path.isdir(directory):
  os.mkdir(directory)
ind = 1
all_num = len(session_ids)*len(stimulus_names)
for session_id in session_ids:
  for stimulus_name in stimulus_names:
    # resolution = resolution_dict[stimulus_name]
    print('resolution {} for {}'.format(resolution, stimulus_name))
    histograms = get_regions_spiking_sequence(session_id, stimulus_name, visual_regions, resolution)
    histograms = bootstrap(histograms, 6000)
    mean_histograms = histograms.mean(dim="stimulus_presentation_id")
    np.save((directory + '{}_{}.npy').format(session_id, stimulus_name), mean_histograms.data)
    print('finished {}, {},  {} / {}'.format(session_id, stimulus_name, ind, all_num))
    ind += 1
print("--- %s minutes in total" % ((time.time() - start_time)/60))
# %%
############# calculate mean spikes from cache #############
# start_time = time.time()
# stimulus_names = ['spontaneous', 'flashes', 'gabors',
#         'drifting_gratings', 'static_gratings',
#           'natural_scenes', 'natural_movie_one', 'natural_movie_three']
# session_ids = [719161530, 750749662, 755434585, 756029989, 791319847]
# visual_regions = ['VISp', 'VISl', 'VISrl', 'VISal', 'VISpm', 'VISam', 'LGd', 'LP']
# resolution_dict = {'spontaneous':0.2, 'flashes':0.02, 'gabors':0.001, 'drifting_gratings':0.004, 'static_gratings':0.0003, 'natural_scenes':0.0003, 'natural_movie_one':0.1, 'natural_movie_three':0.2}
# directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
# if not os.path.isdir(directory):
#   os.mkdir(directory)
# ind = 1
# all_num = len(session_ids)*len(stimulus_names)
# mean_spikes = {'spontaneous':0, 'flashes':0, 'gabors':0, 'drifting_gratings':0, 'static_gratings':0, 'natural_scenes':0, 'natural_movie_one':0, 'natural_movie_three':0}
# for session_id in session_ids:
#   for stimulus_name in stimulus_names:
#     resolution = resolution_dict[stimulus_name]
#     histograms = get_regions_spiking_sequence(session_id, stimulus_name, visual_regions, resolution)
#     sum_histograms = histograms.sum(dim="stimulus_presentation_id")
#     mean_spikes[stimulus_name] += sum_histograms.data.mean()
#     print('finished {}, {},  {} / {}'.format(session_id, stimulus_name, ind, all_num))
#     ind += 1
# print(mean_spikes)
# print("--- %s minutes in total" % ((time.time() - start_time)/60))

# %%
############# calculate mean spikes for one session from cache #############
# start_time = time.time()
# stimulus_names = ['spontaneous', 'flashes', 'gabors',
#         'drifting_gratings', 'static_gratings',
#           'natural_scenes', 'natural_movie_one', 'natural_movie_three']
# session_ids = [719161530, 750749662, 755434585, 756029989, 791319847]
# visual_regions = ['VISp', 'VISl', 'VISrl', 'VISal', 'VISpm', 'VISam', 'LGd', 'LP']
# resolution_dict = {'spontaneous':0.2, 'flashes':0.02, 'gabors':0.001, 'drifting_gratings':0.004, 'static_gratings':0.0003, 'natural_scenes':0.0003, 'natural_movie_one':0.1, 'natural_movie_three':0.2}
# directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
# if not os.path.isdir(directory):
#   os.mkdir(directory)
# ind = 1
# all_num = len(stimulus_names)
# mean_spikes = {'spontaneous':0, 'flashes':0, 'gabors':0, 'drifting_gratings':0, 'static_gratings':0, 'natural_scenes':0, 'natural_movie_one':0, 'natural_movie_three':0}
# session_id = session_ids[0]
# for stimulus_name in stimulus_names:
#     resolution = resolution_dict[stimulus_name]
#     histograms = get_regions_spiking_sequence(session_id, stimulus_name, visual_regions, resolution)
#     sum_histograms = histograms.sum(dim="stimulus_presentation_id")
#     mean_spikes[stimulus_name] += sum_histograms.data.mean()
#     print('finished {}, {},  {} / {}'.format(session_id, stimulus_name, ind, all_num))
#     ind += 1
# for i in mean_spikes:
#     mean_spikes[i] /= all_num
# print(mean_spikes)
# print("--- %s minutes in total" % ((time.time() - start_time)/60))
# %%
# directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
# stimulus_names = ['spontaneous', 'flashes', 'gabors',
#         'drifting_gratings', 'static_gratings',
#           'natural_scenes', 'natural_movie_one', 'natural_movie_three']
# session_ids = [719161530, 750749662, 755434585, 756029989, 791319847]
# plot_all_spikes(directory, session_ids, stimulus_names, 'heterogeneous')
# %%
