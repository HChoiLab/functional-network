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
# %%
# save significant CCG for stationary and running trials
start_time = time.time()
measure = 'ccg'
min_spike = 50
n = 4
max_duration = 11
maxlag = 12
directory = './data/ecephys_cache_dir/sessions/adj_mat_{}_corrected_stationary/'.format(measure)
save_ccg_corrected_highland_new(directory, measure, min_spike=min_spike, max_duration=max_duration, maxlag=maxlag, n=n)
directory = './data/ecephys_cache_dir/sessions/adj_mat_{}_corrected_running/'.format(measure)
save_ccg_corrected_highland_new(directory, measure, min_spike=min_spike, max_duration=max_duration, maxlag=maxlag, n=n)
print("--- %s minutes in total" % ((time.time() - start_time)/60))
# %%
##################### keep edges above threshold
threshold = 0.9
directory = './data/ecephys_cache_dir/sessions/adj_mat_ccg_highland_corrected_stationary/'
keep_edges_above_threshold(directory, threshold)
directory = './data/ecephys_cache_dir/sessions/adj_mat_ccg_highland_corrected_running/'
keep_edges_above_threshold(directory, threshold)
# %%
# load stationary and running graphs
area_dict, active_area_dict, mean_speed_df = load_other_data(session_ids)
cortical_inds = get_cortical_inds(active_area_dict, visual_regions)
directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
path = directory.replace('spiking_sequence', 'adj_mat_ccg_highland_corrected_stationary')
G_stationary_dict, offset_stationary_dict, duration_stationary_dict = load_highland_xcorr(path, active_area_dict, weight=True)
measure = 'ccg'
G_stationary_dict = remove_gabor(G_stationary_dict)
######### removed neurons from thalamic region
G_stationary_dict = remove_thalamic(G_stationary_dict, area_dict, visual_regions)
path = directory.replace('spiking_sequence', 'adj_mat_ccg_highland_corrected_running')
G_running_dict, offset_running_dict, duration_running_dict = load_highland_xcorr(path, active_area_dict, weight=True)
measure = 'ccg'
G_running_dict = remove_gabor(G_running_dict)
######### removed neurons from thalamic region
G_running_dict = remove_thalamic(G_running_dict, area_dict, visual_regions)
S_stationary_dict = add_sign(G_stationary_dict)
S_stationary_dict = add_offset(S_stationary_dict, offset_stationary_dict)
S_stationary_dict = add_duration(S_stationary_dict, duration_stationary_dict)
S_stationary_dict = add_delay(S_stationary_dict)
S_running_dict = add_sign(G_running_dict)
S_running_dict = add_offset(S_running_dict, offset_running_dict)
S_running_dict = add_duration(S_running_dict, duration_running_dict)
S_running_dict = add_delay(S_running_dict)
G_stationary = G_stationary_dict['719161530']['drifting_gratings']
G_running = G_running_dict['719161530']['drifting_gratings']
# %%
# plot density comparison
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.bar(range(2), [nx.density(G_stationary), nx.density(G_running)], width=0.3, color='.2')
ax.set_xticks([0, 1])
ax.set_xticklabels(['stationary', 'running'])
ax.set_ylabel('network density')
plt.tight_layout()
plt.savefig('./plots/density_stationary_running.pdf', transparent=True)
# %%
# within area/excitatory/clustering coefficient
stimulus2marker = {'Resting\nstate':'s', 'Flashes':'*', 'Drifting\ngratings':'X', 'Static\ngratings':'P', 'Natural\nscenes':r'$\clubsuit$', 'Natural\nmovies':'>'}
marker_size_dict = {'v':10, '*':22, 'P':13, 'X':13, 'o':11, 's':9.5, 'D':9, 'p':12, '>':10, r'$\clubsuit$':20}
scatter_size_dict = {'v':10, '*':17, 'P':13, 'X':13, 'o':11, 's':10, 'D':9, 'p':13, '>':12, r'$\clubsuit$':16}
error_size_dict = {'v':10, '*':24, 'P':16, 'X':16, 'o':11, 's':9., 'D':9, 'p':12, '>':13, r'$\clubsuit$':22}

def plot_bar_whole_stationary_running(G_whole, G_stationary, G_running, area_dict, regions):
  df = pd.DataFrame()
  region_connection = np.zeros((len(regions), len(regions)))
  metric_names = ['density', 'ratio of intra-region connections', 'ratio of excitatory connections', 'cluster']
  intra_data, density_data, ex_data, in_data, cluster_data = [], [], [], [], []
  for G in [G_whole, G_stationary, G_running]:
    nodes = list(G.nodes())
    node_area = {key: area_dict['719161530'][key] for key in nodes}
    A = nx.to_numpy_array(G)
    A[A.nonzero()] = 1
    for region_ind_i, region_i in enumerate(regions):
      for region_ind_j, region_j in enumerate(regions):
        region_indices_i = np.array([k for k, v in node_area.items() if v==region_i])
        region_indices_j = np.array([k for k, v in node_area.items() if v==region_j])
        region_indices_i = np.array([nodes.index(i) for i in list(set(region_indices_i) & set(nodes))]) # some nodes not in cc are removed 
        region_indices_j = np.array([nodes.index(i) for i in list(set(region_indices_j) & set(nodes))])
        if len(region_indices_i) and len(region_indices_j):
          region_connection[region_ind_i, region_ind_j] = np.sum(A[region_indices_i[:, None], region_indices_j])
          assert np.sum(A[region_indices_i[:, None], region_indices_j]) == len(A[region_indices_i[:, None], region_indices_j].nonzero()[0])
    diag_indx = np.eye(len(regions),dtype=bool)
    intra_data.append(np.sum(region_connection[diag_indx])/np.sum(region_connection))
    # inter_data.append(np.sum(region_connection[~diag_indx])/np.sum(region_connection))
    density_data.append(nx.density(G))
    signs = list(nx.get_edge_attributes(G, "sign").values())
    ex_data.append(signs.count(1) / len(signs))
    in_data.append(signs.count(-1) / len(signs))
    cluster_data.append(calculate_directed_metric(G, 'clustering'))
  df = pd.concat([df, pd.DataFrame(np.concatenate((np.repeat(np.array(metric_names), 3)[:,None], np.array(density_data + intra_data + ex_data + cluster_data)[:,None], np.array(['whole', 'stationary', 'running'] * 4)[:,None]), 1), columns=['metric', 'data', 'behavior type'])], ignore_index=True)
  df['data'] = pd.to_numeric(df['data'])
  fig, axes = plt.subplots(2, 2, figsize=(10, 8))
  for i, j in [(f,s) for f in range(axes.shape[0]) for s in range(axes.shape[1])]:
    m_ind = i * axes.shape[1] + j
    ax = axes[i, j]
    x = ['whole', 'stationary', 'running']
    y = df[df['metric']==metric_names[m_ind]]['data'].values
    ax.bar(x, y)
    ax.set_ylabel(metric_names[m_ind], fontsize=15)
    ax.xaxis.set_tick_params(labelsize=15)
    ax.yaxis.set_tick_params(labelsize=15)
    for axis in ['bottom', 'left']:
      ax.spines[axis].set_linewidth(1.5)
      ax.spines[axis].set_color('0.2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=1.5)
  plt.tight_layout()
  # plt.show()
  plt.savefig('./plots/bar_whole_stationary_running.pdf', transparent=True)

S_whole = S_ccg_dict['719161530']['drifting_gratings']
S_stationary = S_stationary_dict['719161530']['drifting_gratings']
S_running = S_running_dict['719161530']['drifting_gratings']
plot_bar_whole_stationary_running(S_whole, S_stationary, S_running, area_dict, visual_regions)
# %%
######################## signed motif detection
with open('./files/intensity_dict.pkl', 'rb') as f:
  intensity_dict = pickle.load(f)
with open('./files/coherence_dict.pkl', 'rb') as f:
  coherence_dict = pickle.load(f)
with open('./files/sunibi_baseline_intensity_dict.pkl', 'rb') as f:
  sunibi_baseline_intensity_dict = pickle.load(f)
with open('./files/sunibi_baseline_coherence_dict.pkl', 'rb') as f:
  sunibi_baseline_coherence_dict = pickle.load(f)
with open('./files/intensity_stationary_dict.pkl', 'rb') as f:
  intensity_stationary_dict = pickle.load(f)
with open('./files/coherence_stationary_dict.pkl', 'rb') as f:
  coherence_stationary_dict = pickle.load(f)
with open('./files/sunibi_baseline_intensity_stationary_dict.pkl', 'rb') as f:
  sunibi_baseline_intensity_stationary_dict = pickle.load(f)
with open('./files/sunibi_baseline_coherence_stationary_dict.pkl', 'rb') as f:
  sunibi_baseline_coherence_stationary_dict = pickle.load(f)
with open('./files/intensity_running_dict.pkl', 'rb') as f:
  intensity_running_dict = pickle.load(f)
with open('./files/coherence_running_dict.pkl', 'rb') as f:
  coherence_running_dict = pickle.load(f)
with open('./files/sunibi_baseline_intensity_running_dict.pkl', 'rb') as f:
  sunibi_baseline_intensity_running_dict = pickle.load(f)
with open('./files/sunibi_baseline_coherence_running_dict.pkl', 'rb') as f:
  sunibi_baseline_coherence_running_dict = pickle.load(f)
################## average intensity across session
################## first Z score, then average
num_baseline = 200
whole_df1, mean_df1, signed_motif_types1 = get_intensity_zscore(intensity_dict, coherence_dict, sunibi_baseline_intensity_dict, sunibi_baseline_coherence_dict, num_baseline=num_baseline) # signed uni bi edge preserved
whole_df2, mean_df2, signed_motif_types2 = get_intensity_zscore(intensity_stationary_dict, coherence_stationary_dict, sunibi_baseline_intensity_stationary_dict, sunibi_baseline_coherence_stationary_dict, num_baseline=num_baseline) # signed uni bi edge preserved
whole_df3, mean_df3, signed_motif_types3 = get_intensity_zscore(intensity_running_dict, coherence_running_dict, sunibi_baseline_intensity_running_dict, sunibi_baseline_coherence_running_dict, num_baseline=num_baseline) # signed uni bi edge preserved
whole_df1['signed motif type'] = whole_df1['signed motif type'].str.replace('-', '\N{MINUS SIGN}') # change minus sign to match width of plus
whole_df2['signed motif type'] = whole_df2['signed motif type'].str.replace('-', '\N{MINUS SIGN}') # change minus sign to match width of plus
whole_df3['signed motif type'] = whole_df3['signed motif type'].str.replace('-', '\N{MINUS SIGN}') # change minus sign to match width of plus
signed_motif_types1 = [mt.replace('-', '\N{MINUS SIGN}') for mt in signed_motif_types1]
signed_motif_types2 = [mt.replace('-', '\N{MINUS SIGN}') for mt in signed_motif_types2]
signed_motif_types3 = [mt.replace('-', '\N{MINUS SIGN}') for mt in signed_motif_types3]
#%%
def add_missing_motif_type(df, mtype, signed_motif_types):
  if len(mtype) < len(signed_motif_types):
    mtype2add = [t for t in signed_motif_types if t not in mtype]
    for mt in mtype2add:
      mtype.append(mt)
      for session_id in session_ids:
        for stimulus_name in stimulus_names:
          df = pd.concat([df, pd.DataFrame([[mt, session_id, stimulus_name] + [0] * (df.shape[1]-3)], columns=df.columns)], ignore_index=True)
    df['intensity z score'] = pd.to_numeric(df['intensity z score'])
  return df, mtype

signed_motif_types = np.unique(signed_motif_types1+signed_motif_types2+signed_motif_types3).tolist()
whole_df1, signed_motif_types1 = add_missing_motif_type(whole_df1, signed_motif_types1, signed_motif_types)
whole_df2, signed_motif_types2 = add_missing_motif_type(whole_df2, signed_motif_types2, signed_motif_types)
whole_df3, signed_motif_types3 = add_missing_motif_type(whole_df3, signed_motif_types3, signed_motif_types)
TRIAD_NAMES = ('003', '012', '102', '021D', '021U', '021C', '111D', '111U', '030T', '030C', '201', '120D', '120U', '120C', '210', '300')
# %%
def plot_zscore_allmotif_lollipop_whole_stationary_running(df_whole, df_stationary, df_running):
  # stimulus_order = [s for s in combined_stimulus_names if df.stimulus.str.contains(s).sum()]
  fig, axes = plt.subplots(3,1, sharex=True, sharey=True, figsize=(50, 3*3))
  all_smotif_types = set(list(df_whole['signed motif type'].values) + list(df_stationary['signed motif type'].values) + list(df_running['signed motif type'].values))
  sorted_types = [sorted([smotif for smotif in all_smotif_types if mt in smotif]) for mt in TRIAD_NAMES]
  sorted_types = [item for sublist in sorted_types for item in sublist]
  motif_types = TRIAD_NAMES[3:]
  motif_loc = [np.mean([i for i in range(len(sorted_types)) if mt in sorted_types[i]]) for mt in motif_types]
  # palette = [plt.cm.tab20(i) for i in range(13)]
  palette = [[plt.cm.tab20b(i) for i in range(20)][i] for i in [0,2,3,4,6,8,10,12,16,18,19]] + [[plt.cm.tab20c(i) for i in range(20)][i] for i in [4,16]]
  for s_ind, df in enumerate([df_whole, df_stationary, df_running]):
    data = df
    data = data.groupby('signed motif type').mean()
    ax = axes[s_ind] # spontaneous in the bottom
    # ax.set_title(combined_stimulus_names[s_ind].replace('\n', ' '), fontsize=35, rotation=0)
    for t, y in zip(sorted_types, data.loc[sorted_types, "intensity z score"]):
      color = palette[motif_types.index(t.replace('+', '').replace('\N{MINUS SIGN}', ''))]
      ax.plot([t,t], [0,y], color=color, marker="o", linewidth=7, markersize=20, markevery=(1,2))
    ax.set_xlim(-.5,len(sorted_types)+.5)
    ax.set_xticks([])
    # ax.set_xticks(motif_loc)
    # ax.set_xticklabels(labels=motif_types)
    # ax.xaxis.set_tick_params(labelsize=35, rotation=90)
    ax.yaxis.set_tick_params(labelsize=45)
    for axis in ['bottom', 'left']:
      ax.spines[axis].set_linewidth(4.5)
      ax.spines[axis].set_color('k')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=4.5)
    ax.xaxis.set_tick_params(length=0)
    ax.set_ylabel('')
    # if model_names.index(model_name) <= 1:
    #   ax.set_yscale('symlog')
    # else:
    ax.set_ylim(-13, 21)
  plt.tight_layout()
  figname = './plots/zscore_all_motifs_lollipop_whole_stationary_running.pdf'
  plt.savefig(figname, transparent=True)
  # plt.show()

df_whole = whole_df1[(whole_df1['session']=='719161530')&(whole_df1['stimulus']=='drifting_gratings')]
df_stationary = whole_df2
df_running = whole_df3
plot_zscore_allmotif_lollipop_whole_stationary_running(df_whole, df_stationary, df_running)
# %%
