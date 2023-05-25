#%%
from library import *
combined_stimuli = [['spontaneous'], ['flash_dark', 'flash_light'], ['drifting_gratings'], ['static_gratings'], ['natural_scenes'], ['natural_movie_one', 'natural_movie_three']]
combined_stimulus_names = ['Resting\nstate', 'Flashes', 'Drifting\ngratings', 'Static\ngratings', 'Natural\nscenes', 'Natural\nmovies']
combined_stimulus_colors = ['#8dd3c7', '#fee391', '#bc80bd', '#bc80bd', '#fb8072', '#fb8072']
# plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
session2keep = ['719161530','750749662','754312389','755434585','756029989','791319847','797828357']
stimulus_by_type = [['spontaneous'], ['flash_dark', 'flash_light'], ['drifting_gratings', 'static_gratings'], ['natural_scenes', 'natural_movie_one', 'natural_movie_three']]
stimulus_types = ['Resting state', 'Flashes', 'Gratings', 'Natural stimuli']
# stimulus_type_color = ['tab:blue', 'darkorange', 'darkgreen', 'maroon']
stimulus_type_color = ['#8dd3c7', '#fee391', '#bc80bd', '#fb8072']
stimulus_labels = ['Resting\nstate', 'Dark\nflash', 'Light\nflash', 'Drifting\ngrating', 
              'Static\ngrating', 'Natural\nscenes', 'Natural\nmovie 1', 'Natural\nmovie 3']
region_labels = ['AM', 'PM', 'AL', 'RL', 'LM', 'V1']
# region_colors = ['#b3de69', '#80b1d3', '#fdb462', '#d9d9d9', '#fccde5', '#bebada']
region_colors = ['#d9e9b5', '#c0d8e9', '#fed3a1', '#c3c3c3', '#fad3e4', '#cec5f2']
TRIAD_NAMES = ('003', '012', '102', '021D', '021U', '021C', '111D', '111U', '030T', '030C', '201', '120D', '120U', '120C', '210', '300')
model_names = [u'Erdős-Rényi model', 'Degree-preserving model', 'Pair-preserving model', 'Signed-pair-preserving model']

area_dict, active_area_dict, mean_speed_df = load_other_data(session_ids)
cortical_inds = get_cortical_inds(active_area_dict, visual_regions)
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
pos_G_dict, neg_G_dict = split_pos_neg(G_ccg_dict)
# %%
old_combined_stimuli = [['spontaneous'], ['flashes'], ['drifting_gratings'], ['static_gratings'], ['natural_scenes'], ['natural_movie_one', 'natural_movie_three']]
combined_mspeed_df = pd.DataFrame()
for cs_ind, combined_stimulus in enumerate(old_combined_stimuli):
  combined_mspeed_df[combined_stimulus_names[cs_ind]] = mean_speed_df[combined_stimulus].mean(axis=1)
# combined_mspeed_df
# %%
def get_time_varying_speed(session_ids):
  data_directory = './data/ecephys_cache_dir'
  manifest_path = os.path.join(data_directory, "manifest.json")
  cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
  speed_dict = {}
  for mouseID in session_ids:
    print(mouseID)
    session = cache.get_session_data(int(mouseID),
                                amplitude_cutoff_maximum=np.inf,
                                presence_ratio_minimum=-np.inf,
                                isi_violations_maximum=np.inf)
    if not mouseID in speed_dict:
      speed_dict[mouseID] = {}
    for cs_ind, combined_stimulus in enumerate(combined_stimuli):
      stimulus_speed = []
      for stimulus_name in combined_stimulus:
        print(stimulus_name)
        if (stimulus_name=='flash_light') or (stimulus_name=='flash_dark'):
          stim_table = session.get_stimulus_table(['flashes'])
        else:
          stim_table = session.get_stimulus_table([stimulus_name])
        stim_table=stim_table.rename(columns={"start_time": "Start", "stop_time": "End"})
        if stimulus_name=='flash_light':
          stim_table = stim_table[stim_table['color']==1]
        elif stimulus_name=='flash_dark':
          stim_table = stim_table[stim_table['color']==-1]
        if 'natural_movie' in stimulus_name:
          frame_times = stim_table.End-stim_table.Start
          print('frame rate:', 1/np.mean(frame_times), 'Hz', np.mean(frame_times))
          # stim_table.to_csv(output_path+'stim_table_'+stimulus_name+'.csv')
          # chunch each movie clip
        elif stimulus_name=='spontaneous':
          index = np.where(stim_table.duration>=20)[0]
          if len(index): # only keep the longest spontaneous; has to be longer than 20 sec
            stim_table = stim_table.iloc[index]
        # all_speed = []
        if not 'natural_movie' in stimulus_name:
          for i in stim_table.index: # each row is a presentation
            speed = session.running_speed[(session.running_speed['start_time']>=stim_table.loc[i, 'Start']) & (session.running_speed['end_time']<=stim_table.loc[i, 'End'])]
            # all_speed.append(speed.velocity.values)
            stimulus_speed.append(speed.velocity.values)
        else:
          blocks = stim_table.stimulus_block.unique()
          for b in blocks:
            speed = session.running_speed[(session.running_speed['start_time']>=stim_table[stim_table.stimulus_block==b]['Start'].min()) & (session.running_speed['end_time']<=stim_table[stim_table.stimulus_block==b]['End'].max())]
            # all_speed.append(speed.velocity.values)
            stimulus_speed.append(speed.velocity.values)
        # min_len = min(len(lst) for lst in all_speed)
        # speed_trial = np.array([speed[:min_len] for speed in all_speed]).mean(0)
        # stimulus_speed.append(speed_trial)
      min_len = min(len(lst) for lst in stimulus_speed)
      stimulus_speed = np.array([speed[:min_len] for speed in stimulus_speed])
      speed_dict[mouseID][combined_stimulus_names[cs_ind]] = stimulus_speed # average across different conditions
  return speed_dict

speed_dict = get_time_varying_speed(session2keep)
#%%
# data_directory = './data/ecephys_cache_dir'
# manifest_path = os.path.join(data_directory, "manifest.json")
# cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
# speed_dict = {}
# mouseID = 754312389
# session = cache.get_session_data(int(mouseID),
#                             amplitude_cutoff_maximum=np.inf,
#                             presence_ratio_minimum=-np.inf,
#                             isi_violations_maximum=np.inf)
# if not mouseID in speed_dict:
#   speed_dict[mouseID] = {}
# combined_stimulus = ['natural_movie_one', 'natural_movie_three']
# stimulus_speed = []
# for stimulus_name in combined_stimulus:
#   print(stimulus_name)
#   if (stimulus_name=='flash_light') or (stimulus_name=='flash_dark'):
#     stim_table = session.get_stimulus_table(['flashes'])
#   else:
#     stim_table = session.get_stimulus_table([stimulus_name])
#   stim_table=stim_table.rename(columns={"start_time": "Start", "stop_time": "End"})
#   if stimulus_name=='flash_light':
#     stim_table = stim_table[stim_table['color']==1]
#   elif stimulus_name=='flash_dark':
#     stim_table = stim_table[stim_table['color']==-1]
#   if 'natural_movie' in stimulus_name:
#     frame_times = stim_table.End-stim_table.Start
#     print('frame rate:', 1/np.mean(frame_times), 'Hz', np.mean(frame_times))
#     # stim_table.to_csv(output_path+'stim_table_'+stimulus_name+'.csv')
#     # chunch each movie clip
    
#   all_speed = []
#   if not 'natural_movie' in stimulus_name:
#     for i in stim_table.index: # each row is a presentation
#       speed = session.running_speed[(session.running_speed['start_time']>=stim_table.loc[i, 'Start']) & (session.running_speed['end_time']<=stim_table.loc[i, 'End'])]
#       all_speed.append(speed.velocity.values)
#   else:
#     blocks = stim_table.stimulus_block.unique()
#     for b in blocks:
#       speed = session.running_speed[(session.running_speed['start_time']>=stim_table[stim_table.stimulus_block==b]['Start'].min()) & (session.running_speed['end_time']<=stim_table[stim_table.stimulus_block==b]['End'].max())]
#       all_speed.append(speed.velocity.values)
#   min_len = min(len(lst) for lst in all_speed)
#   speed_trial = np.array([speed[:min_len] for speed in all_speed]).mean(0)
#   stimulus_speed.append(speed_trial)
# min_len = min(len(lst) for lst in stimulus_speed)
# stimulus_speed = np.array([speed[:min_len] for speed in stimulus_speed])
# np.mean(stimulus_speed, 0) # average across different conditions
#%%
# plot speed within each presentation averaged across trials with shaded area
def plot_speed_eachpre(speed_dict):
  fig, axes = plt.subplots(7, 6, figsize=(8*4, 8*2))
  for i, j in [(f,s) for f in range(axes.shape[0]) for s in range(axes.shape[1])]:
    ax = axes[i, j]
    speed_trial = speed_dict[session2keep[i]][combined_stimulus_names[j]]
    m, std = np.nanmean(speed_trial, 0), np.nanstd(speed_trial, 0)
    ax.plot(range(speed_trial.shape[1]), m, color=u'#1f77b4')
    ax.fill_between(range(speed_trial.shape[1]), m-std,m+std, color=u'#1f77b4', alpha=0.2)
    if i == axes.shape[0] - 1:
      ax.set_xlabel('time bin since onset')
    if j == 0:
      ax.set_ylabel('running speed (cm/s)')
  # plt.show()
  plt.savefig('./plots/speed_eachpre.pdf', transparent=True)
  
plot_speed_eachpre(speed_dict)
#%%
# plot speed within each presentation for each trial
def plot_speed_eachpre_eachtrial(speed_dict):
  fig, axes = plt.subplots(7, 6, figsize=(8*4, 8*2))
  for i, j in [(f,s) for f in range(axes.shape[0]) for s in range(axes.shape[1])]:
    ax = axes[i, j]
    speed_trial = speed_dict[session2keep[i]][combined_stimulus_names[j]]
    ax.plot(speed_trial.T, color='k', alpha=0.1)
    if i == axes.shape[0] - 1:
      ax.set_xlabel('time bin since onset')
    if j == 0:
      ax.set_ylabel('running speed (cm/s)')
  # plt.show()
  plt.savefig('./plots/speed_eachpre_eachtrial.pdf', transparent=True)
  
plot_speed_eachpre_eachtrial(speed_dict)
#%%
# plot mean speed against stimulus for each mouse
def plot_mean_speed(speed_dict):
  color_list = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
  fig, ax = plt.subplots(1, 1, figsize=(5, 4))
  mean_speed = np.zeros((7, 6))
  for m_id, mouseID in enumerate(session2keep):
    for s_id, combined_stimulus_name in enumerate(combined_stimulus_names):
      speed_trial = speed_dict[mouseID][combined_stimulus_name]
      mean_speed[m_id, s_id] = speed_trial.mean()
    ax.plot(range(6), mean_speed[m_id], label='mouse {}'.format(m_id+1))
  ax.set_xticks(range(6))
  ax.set_xticklabels(combined_stimulus_names, fontsize=12)
  ax.set_ylabel('running speed (cm/s)')
  ax.legend()
  # plt.show()
  plt.savefig('./plots/mean_speed.pdf', transparent=True)
  
plot_mean_speed(speed_dict)
#%%
# plot mean speed scatter against stimulus for each mouse
stimulus2marker = {'Resting\nstate':'s', 'Flashes':'*', 'Drifting\ngratings':'X', 'Static\ngratings':'P', 'Natural\nscenes':r'$\clubsuit$', 'Natural\nmovies':'>'}
marker_size_dict = {'v':10, '*':22, 'P':13, 'X':13, 'o':11, 's':9.5, 'D':9, 'p':12, '>':10, r'$\clubsuit$':20}
scatter_size_dict = {'v':10, '*':17, 'P':13, 'X':13, 'o':11, 's':10, 'D':9, 'p':13, '>':12, r'$\clubsuit$':16}
error_size_dict = {'v':10, '*':24, 'P':16, 'X':16, 'o':11, 's':9., 'D':9, 'p':12, '>':13, r'$\clubsuit$':22}
def plot_mean_speed_error(speed_dict):
  palette = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
  fig, ax = plt.subplots(1,1, sharex=True, sharey=True, figsize=(8, 4))
  for m_id, mouseID in enumerate(session2keep):
    for s_id, combined_stimulus_name in enumerate(combined_stimulus_names):
      speed_trial = speed_dict[mouseID][combined_stimulus_name]
      xi, yi, erri = s_id, speed_trial.mean(), speed_trial.std()
      ax.errorbar(xi + .13 * m_id, yi, yerr=erri, fmt=' ', linewidth=2.,color=palette[m_id], zorder=1)
      ax.scatter(xi + .13 * m_id, yi, marker=stimulus2marker[combined_stimulus_name], s=10*error_size_dict[stimulus2marker[combined_stimulus_name]], linewidth=1.,color=palette[m_id], zorder=2)
  
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
  # ax.set_yscale('log')
  # ylabel = 'Fraction of three V1 neurons' if mtype=='all_V1' else: 'Fraction of at least one V1 neuron'
  # ax.set_ylabel('Density', fontsize=30)
  plt.tight_layout(rect=[.02, -.03, 1, 1])
  plt.savefig('./plots/mean_speed_error.pdf', transparent=True)
  # plt.show()

plot_mean_speed_error(speed_dict)
#%%
# distribution of average speed for each stimulus presentation
def plot_mean_speed_distribution(speed_dict):
  fig, axes = plt.subplots(7, 6, figsize=(4*4, 4*2))
  for m_id, mouseID in enumerate(session2keep):
    for s_id, combined_stimulus_name in enumerate(combined_stimulus_names):
      ax = axes[m_id, s_id]
      speed_trial = speed_dict[mouseID][combined_stimulus_name]
      ax.hist(speed_trial.mean(1))
      if m_id < len(session2keep) - 1:
        ax.set_xlabel('')
      else:
        ax.set_xlabel('mean running speed (cm/s)')
      if s_id == 0:
        ax.set_ylabel('count')
      ax.xaxis.set_tick_params(length=0)
      ax.set_xlim(-.5, 75)
      # ax.invert_yaxis()
      ax.yaxis.set_tick_params(labelsize=10)
      for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(1.)
        ax.spines[axis].set_color('k')
      ax.spines['top'].set_visible(False)
      ax.spines['right'].set_visible(False)
      ax.tick_params(width=2.)
  # ax.set_ylim(bottom=0)
  # ax.set_yscale('log')
  # ylabel = 'Fraction of three V1 neurons' if mtype=='all_V1' else: 'Fraction of at least one V1 neuron'
  # ax.set_ylabel('Density', fontsize=30)
  plt.tight_layout() # rect=[.02, -.03, 1, 1]
  plt.savefig('./plots/mean_speed_distribution.pdf', transparent=True)
  # plt.show()

plot_mean_speed_distribution(speed_dict)
#%%
# plot number of stationary and running trials against threshold on speed
def plot_stationary_running_trials(speed_dict, mouseID, combined_stimulus_name):
  speed_trial = speed_dict[mouseID][combined_stimulus_name]
  threshold_list = list(range(75))
  num_sta, num_run = np.zeros(len(threshold_list)), np.zeros(len(threshold_list))
  for ind, threshold in enumerate(threshold_list):
    num_sta[ind], num_run[ind] = (speed_trial.mean(1) <= threshold).sum(), (speed_trial.mean(1) > threshold).sum()
  fig, ax = plt.subplots(1, 1, figsize=(4, 4))
  ax.plot(threshold_list, num_sta, label='stationary')
  ax.plot(threshold_list, num_run, label='running')
  ax.axvline(threshold_list[np.argmax(np.diff(num_sta))+1], linestyle='--', color='k')
  print('Optimal threshold = {} cm/s'.format(threshold_list[np.argmax(np.diff(num_sta))+1]))
  ax.set_xlabel('threshold on speed (cm/s)')
  ax.set_ylabel('number of trials')
  ax.legend()
  # ax.set_xscale('log')
  plt.tight_layout()
  plt.savefig('./plots/speed_stat.pdf', transparent=True)
  # plt.savefig('./plots/speed_stat_log.pdf', transparent=True)
  # plt.show()
  
plot_stationary_running_trials(speed_dict, mouseID='719161530', combined_stimulus_name='Drifting\ngratings')
#%%
# plot number of stationary and running trials against threshold on speed
def get_stationary_running_trials(speed_dict, mouseID, combined_stimulus_name):
  speed_trial = speed_dict[mouseID][combined_stimulus_name]
  threshold = 1
  sta_ind, run_ind = np.where(speed_trial.mean(1) <= threshold)[0], np.where(speed_trial.mean(1) > threshold)[0]
  return sta_ind, run_ind
  
sta_ind, run_ind = get_stationary_running_trials(speed_dict, mouseID='719161530', combined_stimulus_name='Drifting\ngratings')
#%%
directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
inds_path = './data/ecephys_cache_dir/sessions/active_inds/'
session_id, stimulus_name = '719161530', 'drifting_gratings'
file = '{}_{}.npz'.format(session_id, stimulus_name)
sequences = load_npz_3d(os.path.join(directory, file))
active_neuron_inds = np.load(os.path.join(inds_path, str(session_id)+'.npy'))
# active_neuron_inds = sequences.mean(1).sum(1) > sequences.shape[2] * min_FR
sequences = sequences[active_neuron_inds]
print('{} spike train shape: {}'.format(file, sequences.shape))
#%%
def save_mean_ccg_corrected_stationary_running(o_sequences, inds, fname, num_jitter=10, L=25, window=100, disable=True): ### fastest, only causal correlation (A>B, only positive time lag on B), largest deviation from flank
  sequences = o_sequences[:,inds,:]
  num_neuron, num_trial, T = sequences.shape
  # num_trial = min(num_trial, 1000) # at most 1000 trials
  ccg, ccg_jittered = np.zeros((num_neuron, num_neuron, window + 1)), np.zeros((num_neuron, num_neuron, window + 1))
  pj = pattern_jitter(num_sample=num_jitter, sequences=sequences[:,0,:], L=L, memory=False)
  for m in range(num_trial):
    print('Trial {} / {}'.format(m+1, num_trial))
    ccg += get_all_ccg(sequences[:,m,:], window, disable=disable) # N x N x window
    pj.sequences = sequences[:,m,:]
    sampled_matrix = pj.jitter() # num_sample x N x T
    for i in range(num_jitter):
      ccg_jittered += get_all_ccg(sampled_matrix[i, :, :], window, disable=disable)
  ccg = ccg / num_trial
  ccg_jittered = ccg_jittered / (num_jitter * num_trial)
  save_sparse_npz(ccg, fname)
  save_sparse_npz(ccg_jittered, fname.replace('.npz', '_bl.npz'))
  
num_baseline = 1
path = os.path.join(directory.replace('spiking_sequence', 'adj_mat_ccg_corrected_stationary'))
fname = os.path.join(path, file)
start_time = time.time()
save_mean_ccg_corrected_stationary_running(o_sequences=sequences, inds=sta_ind, fname=fname, num_jitter=num_baseline, L=25, window=100, disable=False)
print("--- %s minutes" % ((time.time() - start_time)/60))
path = os.path.join(directory.replace('spiking_sequence', 'adj_mat_ccg_corrected_running'))
fname = os.path.join(path, file)
start_time = time.time()
save_mean_ccg_corrected_stationary_running(o_sequences=sequences, inds=run_ind, fname=fname, num_jitter=num_baseline, L=25, window=100, disable=False)
print("--- %s minutes" % ((time.time() - start_time)/60))
#%%
mouseID = '719161530'
stimulus_name = 'drifting_gratings'
data_directory = './data/ecephys_cache_dir'
manifest_path = os.path.join(data_directory, "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
session = cache.get_session_data(int(mouseID),
                                amplitude_cutoff_maximum=np.inf,
                                presence_ratio_minimum=-np.inf,
                                isi_violations_maximum=np.inf)
stim_table = session.get_stimulus_table([stimulus_name])
stim_table=stim_table.rename(columns={"start_time": "Start", "stop_time": "End"})
# block_ids = stim_table['stimulus_block'].unique()
all_speed = []
for i in stim_table.index: # each row is a presentation
  speed = session.running_speed[(session.running_speed['start_time']>=stim_table.loc[i, 'Start']) & (session.running_speed['end_time']<=stim_table.loc[i, 'End'])]
  all_speed.append(speed.velocity.values)
min_len = min(len(lst) for lst in all_speed)
# Create the 2D NumPy array
speed_trial = np.array([speed[:min_len] for speed in all_speed])
plt.figure()
m, std = np.nanmean(speed_trial, 0), np.nanstd(speed_trial, 0)
plt.plot(range(speed_trial.shape[1]), m, color='blue')
plt.fill_between(range(speed_trial.shape[1]), m-std,m+std, color='blue', alpha=0.2)
plt.xlabel('time bin since onset')
plt.ylabel('running speed (cm/s)')
plt.show()
# %%
# TODO: separate session into still and running
def load_area_speed(session_ids, stimulus_names, regions):
  data_directory = './data/ecephys_cache_dir'
  manifest_path = os.path.join(data_directory, "manifest.json")
  cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
  area_dict = {}
  speed_dict = {}
  for mouseID in session_ids:
    print(mouseID)
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
    # if set(instruction.unique()) == set(regions): # if the mouse has all regions recorded
    #   speed_dict[mouseID] = {}
    instruction = instruction.reset_index()
    if not mouseID in area_dict:
      area_dict[mouseID] = {}
      speed_dict[mouseID] = {}
    for i in range(instruction.shape[0]):
      area_dict[mouseID][i] = instruction.ccf.iloc[i]
    for stimulus_name in stimulus_names:
      print(stimulus_name)
      stim_table = session.get_stimulus_table([stimulus_name])
      stim_table=stim_table.rename(columns={"start_time": "Start", "stop_time": "End"})
      if 'natural_movie' in stimulus_name:
        frame_times = stim_table.End-stim_table.Start
        print('frame rate:', 1/np.mean(frame_times), 'Hz', np.mean(frame_times))
        # stim_table.to_csv(output_path+'stim_table_'+stimulus_name+'.csv')
        # chunch each movie clip
        stim_table = stim_table[stim_table.frame==0]
      speed = session.running_speed[(session.running_speed['start_time']>=stim_table['Start'].min()) & (session.running_speed['end_time']<=stim_table['End'].max())]
      speed_dict[mouseID][stimulus_name] = speed['velocity'].mean()
  # switch speed_dict to dataframe
  mouseIDs = list(speed_dict.keys())
  stimuli = list(speed_dict[list(speed_dict.keys())[0]].keys())
  mean_speed_df = pd.DataFrame(columns=stimuli, index=mouseIDs)
  for k in speed_dict:
    for v in speed_dict[k]:
      mean_speed_df.loc[k][v] = speed_dict[k][v]
  return area_dict, mean_speed_df
