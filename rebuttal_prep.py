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
#%%
##################### plot best CCG sequence for all types of connections (whole concatenated CCG with negative time lags)
def largest_indices(array, M):
  if len(array) <= M:
    indices = np.arange(len(array))
  else:
    indices = np.argpartition(array, -M)[-M:]
  return indices
  
def smallest_indices(array, M):
  if len(array) <= M:
    indices = np.arange(len(array))
  else:
    indices = np.argpartition(array, M)[:M]
  return indices

def get_best_ccg_edge_type(directory, edgetype, M=20):
  files = os.listdir(directory)
  files.sort(key=lambda x:int(x[:9]))
  file2plot, inds_2plot, conf_2plot = [], [], []
  for file in files:
    if ('_bl' not in file) and ('gabors' not in file) and ('flashes' not in file): #   and ('drifting_gratings' in file) and ('719161530' in file) and '719161530' in file and ('static_gratings' in file or 'gabors' in file) or 'flashes' in file
      print(file)
      sig_dir = './data/ecephys_cache_dir/sessions/adj_mat_ccg_highland_corrected/'
      significant_ccg = load_npz_3d(os.path.join(sig_dir, file))
      confidence_level = load_npz_3d(os.path.join(sig_dir, file.replace('.npz', '_confidence.npz')))
      significant_offset = load_npz_3d(os.path.join(sig_dir, file.replace('.npz', '_offset.npz')))
      significant_duration = load_npz_3d(os.path.join(sig_dir, file.replace('.npz', '_duration.npz')))
      significant_inds = list(zip(*np.where(~np.isnan(significant_ccg))))
      np.random.shuffle(significant_inds)
      if 'uni' in edgetype:
        poslag_inds = significant_offset > 1
      else:
        poslag_inds = (significant_offset > 1) & ((significant_offset.T > 1))
      if edgetype == 'uni_pos':
        edgetype_inds = (confidence_level>0) & np.isnan(confidence_level.T)
      elif edgetype == 'uni_neg':
        edgetype_inds = (confidence_level<0) & np.isnan(confidence_level.T)
      elif edgetype == 'bi_pos':
        edgetype_inds = (confidence_level>0) & (confidence_level.T>0)
      elif edgetype == 'bi_both':
        edgetype_inds = (confidence_level>0) & (confidence_level.T<0)
      elif edgetype == 'bi_neg':
        edgetype_inds = (confidence_level<0) & (confidence_level.T<0)
      valid_inds = np.where(poslag_inds & edgetype_inds)
      if 'both' in edgetype:
        conf_valid = abs(confidence_level[valid_inds]) + abs(confidence_level[valid_inds].T)
      elif 'uni' in edgetype:
        conf_valid = confidence_level[valid_inds]
      elif 'bi' in edgetype:
        conf_valid = confidence_level[valid_inds] + confidence_level[valid_inds].T
      if len(valid_inds[0]):
        if ('pos' in edgetype) or ('both' in edgetype):
          indices = largest_indices(conf_valid, M)
        elif 'neg' in edgetype:
          indices = smallest_indices(conf_valid, M)
        file2plot += [file]*len(indices)
        inds_2plot += [(valid_inds[0][ind], valid_inds[1][ind]) for ind in indices]
        conf_2plot += abs(conf_valid[indices]).tolist()
          
  sorted_lists = sorted(zip(file2plot, inds_2plot, conf_2plot), key=lambda x: x[2], reverse=True)
  file2plot, inds_2plot, conf_2plot = zip(*sorted_lists)
  return inds_2plot, file2plot

directory = './data/ecephys_cache_dir/sessions/adj_mat_ccg_corrected/'
uni_pos_indx, uni_pos_file = get_best_ccg_edge_type(directory, edgetype='uni_pos')
uni_neg_indx, uni_neg_file = get_best_ccg_edge_type(directory, edgetype='uni_neg')
bi_pos_indx, bi_pos_file = get_best_ccg_edge_type(directory, edgetype='bi_pos')
bi_both_indx, bi_botj_file = get_best_ccg_edge_type(directory, edgetype='bi_both')
bi_neg_indx, bi_neg_file = get_best_ccg_edge_type(directory, edgetype='bi_neg')
#%%
#################### plot concatenated CCG for each type of connections
def plot_concat_ccg_edgetype(indx, files, etype, ind, window=100, scalebar=False):
  file = files[ind]
  linewidth = 4.
  inds_path = './data/ecephys_cache_dir/sessions/active_inds/'
  # all_active_inds = list(np.load(os.path.join(inds_path, str(797828357)+'.npy'))) # including thalamus
  try: 
    ccg = load_npz_3d(os.path.join(directory, file))
  except:
    ccg = load_sparse_npz(os.path.join(directory, file))
  try:
    ccg_jittered = load_npz_3d(os.path.join(directory, file.replace('.npz', '_bl.npz')))
  except:
    ccg_jittered = load_sparse_npz(os.path.join(directory, file.replace('.npz', '_bl.npz')))
  ccg_corrected = ccg - ccg_jittered
  sig_dir = './data/ecephys_cache_dir/sessions/adj_mat_ccg_highland_corrected/'
  significant_ccg = load_npz_3d(os.path.join(sig_dir, file))
  significant_offset = load_npz_3d(os.path.join(sig_dir, file.replace('.npz', '_offset.npz')))
  significant_duration = load_npz_3d(os.path.join(sig_dir, file.replace('.npz', '_duration.npz')))
  row_a, row_b = indx[ind]
  fig, ax = plt.subplots(1, 1, figsize=(8*1, 4))
  # row_a, row_b = all_active_inds.index(row_a), all_active_inds.index(row_b)
  ccg_con = np.concatenate((np.flip(ccg_corrected[row_b, row_a, 1:]), ccg_corrected[row_a, row_b]))
  ax.plot(np.arange(-window, window+1), ccg_con, linewidth=linewidth, color='k')
  ax.axvline(x=0, linestyle='--', color='k', linewidth=3)
  color = 'firebrick' if significant_ccg[row_a, row_b] > 0 else 'blue'
  highland_lag = np.arange(int(significant_offset[row_a,row_b]), int(significant_offset[row_a,row_b]+significant_duration[row_a,row_b]+1))
  ax.plot(highland_lag, ccg_corrected[row_a, row_b, highland_lag], color=color, marker='o', linewidth=linewidth, markersize=15, alpha=0.8)
  if 'bi' in etype:
    color = 'firebrick' if significant_ccg[row_b, row_a] > 0 else 'blue'
    highland_lag = np.arange(int(significant_offset[row_b,row_a]), int(significant_offset[row_b,row_a]+significant_duration[row_b,row_a]+1))
    ax.plot(-highland_lag, ccg_corrected[row_b, row_a, highland_lag], color=color, marker='o', linewidth=linewidth, markersize=15, alpha=0.8)
  ax.set_yticks([])
  ax.set_xlim([-window/2, window/2])
  if scalebar:
    fontprops = fm.FontProperties(size=40)
    size_v = (ccg_corrected[row_a, row_b].max()-ccg_corrected[row_a, row_b].min())/30
    scalebar = AnchoredSizeBar(ax.transData,
                              100, '100 ms', 'lower center',
                              borderpad=0,
                              pad=-1.4,
                              sep=5,
                              color='k',
                              frameon=False,
                              size_vertical=size_v,
                              fontproperties=fontprops)

    ax.add_artist(scalebar)
  ax.set_axis_off()
  plt.tight_layout()
  plt.subplots_adjust(left=0.,
                    bottom=0.,
                    right=1.,
                    top=1.,
                    wspace=1.2)
  plt.savefig('./plots/concat_{}_example_ccg_{}.pdf'.format(etype, ind), transparent=True)
  # plt.show()
for ind in [0, 1, 2, 5]:
  plot_concat_ccg_edgetype(uni_pos_indx, uni_pos_file, 'uni_pos', ind, window=100, scalebar=False) # 0, 1, 2, 5
for ind in [0, 3, 7]:
  plot_concat_ccg_edgetype(uni_neg_indx, uni_neg_file, 'uni_neg', ind, window=100, scalebar=False) # 0, 3, 7
for ind in [7, 9, 15, 26, 30, 39, 40, 48]:
  plot_concat_ccg_edgetype(bi_pos_indx, bi_pos_file, 'bi_pos', ind, window=100, scalebar=False) # 7, 9, 15, 26, 30, 39!, 40, 48
for ind in [3, 10, 32, 34]:
  plot_concat_ccg_edgetype(bi_both_indx, bi_botj_file, 'bi_both', ind, window=100, scalebar=False) # 3, 10, 32, 34
for ind in [2, 3]:
  plot_concat_ccg_edgetype(bi_neg_indx, bi_neg_file, 'bi_neg', ind, window=100, scalebar=False) # 2, 5, 9
#%%
# plot example ccg for traditional method
def plot_concat_ccg_edgetype_traditional(indx, files, etype, ind, window=100, scalebar=False):
  file = files[ind]
  linewidth = 4.
  inds_path = './data/ecephys_cache_dir/sessions/active_inds/'
  # all_active_inds = list(np.load(os.path.join(inds_path, str(797828357)+'.npy'))) # including thalamus
  try: 
    ccg = load_npz_3d(os.path.join(directory, file))
  except:
    ccg = load_sparse_npz(os.path.join(directory, file))
  try:
    ccg_jittered = load_npz_3d(os.path.join(directory, file.replace('.npz', '_bl.npz')))
  except:
    ccg_jittered = load_sparse_npz(os.path.join(directory, file.replace('.npz', '_bl.npz')))
  ccg_corrected = ccg - ccg_jittered
  sig_dir = './data/ecephys_cache_dir/sessions/adj_mat_ccg_highland_corrected/'
  significant_ccg = load_npz_3d(os.path.join(sig_dir, file))
  significant_offset = load_npz_3d(os.path.join(sig_dir, file.replace('.npz', '_offset.npz')))
  significant_duration = load_npz_3d(os.path.join(sig_dir, file.replace('.npz', '_duration.npz')))
  row_a, row_b = indx[ind]
  fig, ax = plt.subplots(1, 1, figsize=(8*1, 4))
  # row_a, row_b = all_active_inds.index(row_a), all_active_inds.index(row_b)
  ccg_con = np.concatenate((np.flip(ccg_corrected[row_b, row_a, 1:]), ccg_corrected[row_a, row_b]))
  ax.plot(np.arange(-window, window+1), ccg_con, linewidth=linewidth, color='k')
  ax.axvline(x=0, linestyle='--', color='k', linewidth=3)
  # color = 'firebrick' if significant_ccg[row_a, row_b] > 0 else 'blue'
  # highland_lag = np.arange(int(significant_offset[row_a,row_b]), int(significant_offset[row_a,row_b]+significant_duration[row_a,row_b]+1))
  # ax.plot(highland_lag, ccg_corrected[row_a, row_b, highland_lag], color=color, marker='o', linewidth=linewidth, markersize=15, alpha=0.8)
  if 'bi' in etype:
    color = 'firebrick' if significant_ccg[row_b, row_a] > 0 else 'blue'
    highland_lag = np.arange(int(significant_offset[row_b,row_a]), int(significant_offset[row_b,row_a]+significant_duration[row_b,row_a]+1))
    ax.plot(-highland_lag, ccg_corrected[row_b, row_a, highland_lag], color=color, marker='o', linewidth=linewidth, markersize=15, alpha=0.8)
  ax.set_yticks([])
  ax.set_xlim([-window/2, window/2])
  if scalebar:
    fontprops = fm.FontProperties(size=40)
    size_v = (ccg_corrected[row_a, row_b].max()-ccg_corrected[row_a, row_b].min())/30
    scalebar = AnchoredSizeBar(ax.transData,
                              100, '100 ms', 'lower center',
                              borderpad=0,
                              pad=-1.4,
                              sep=5,
                              color='k',
                              frameon=False,
                              size_vertical=size_v,
                              fontproperties=fontprops)

    ax.add_artist(scalebar)
  ax.set_axis_off()
  plt.tight_layout()
  plt.subplots_adjust(left=0.,
                    bottom=0.,
                    right=1.,
                    top=1.,
                    wspace=1.2)
  # plt.savefig('./plots/concat_{}_example_ccg_{}.pdf'.format(etype, ind), transparent=True)
  plt.show()
for ind in [200]:
  plot_concat_ccg_edgetype_traditional(uni_pos_indx, uni_pos_file, 'uni_pos', ind, window=100, scalebar=False)
#%%
# find excitatory/inhibitory polysynaptic connections
def get_best_ccg_poly(directory, sign, M=20):
  files = os.listdir(directory)
  files.sort(key=lambda x:int(x[:9]))
  file2plot, inds_2plot, conf_2plot = [], [], []
  for file in files:
    if ('_bl' not in file) and ('gabors' not in file) and ('flashes' not in file): #   and ('drifting_gratings' in file) and ('719161530' in file) and '719161530' in file and ('static_gratings' in file or 'gabors' in file) or 'flashes' in file
      print(file)
      sig_dir = './data/ecephys_cache_dir/sessions/adj_mat_ccg_highland_corrected/'
      significant_ccg = load_npz_3d(os.path.join(sig_dir, file))
      confidence_level = load_npz_3d(os.path.join(sig_dir, file.replace('.npz', '_confidence.npz')))
      significant_offset = load_npz_3d(os.path.join(sig_dir, file.replace('.npz', '_offset.npz')))
      significant_duration = load_npz_3d(os.path.join(sig_dir, file.replace('.npz', '_duration.npz')))
      significant_inds = list(zip(*np.where(~np.isnan(significant_ccg))))
      np.random.shuffle(significant_inds)
      poly_inds = significant_duration > 2
      poslag_inds = significant_offset > 1
      if sign == 'pos':
        edgetype_inds = (confidence_level>0) & np.isnan(confidence_level.T)
      elif sign == 'neg':
        edgetype_inds = (confidence_level<0) & np.isnan(confidence_level.T)
      valid_inds = np.where(poslag_inds & edgetype_inds & poly_inds)
      conf_valid = confidence_level[valid_inds]
      if len(valid_inds[0]):
        if sign == 'pos':
          indices = largest_indices(conf_valid, M)
        elif sign == 'neg':
          indices = smallest_indices(conf_valid, M)
        file2plot += [file]*len(indices)
        inds_2plot += [(valid_inds[0][ind], valid_inds[1][ind]) for ind in indices]
        conf_2plot += abs(conf_valid[indices]).tolist()
          
  sorted_lists = sorted(zip(file2plot, inds_2plot, conf_2plot), key=lambda x: x[2], reverse=True)
  file2plot, inds_2plot, conf_2plot = zip(*sorted_lists)
  return inds_2plot, file2plot

directory = './data/ecephys_cache_dir/sessions/adj_mat_ccg_corrected/'
poly_pos_indx, poly_pos_file = get_best_ccg_poly(directory, sign='pos')
poly_neg_indx, poly_neg_file = get_best_ccg_poly(directory, sign='neg')
#%%
#################### plot concatenated CCG for polysynaptic connections
def plot_concat_ccg_poly(indx, files, sign, ind, window=100, scalebar=False):
  file = files[ind]
  linewidth = 4.
  inds_path = './data/ecephys_cache_dir/sessions/active_inds/'
  # all_active_inds = list(np.load(os.path.join(inds_path, str(797828357)+'.npy'))) # including thalamus
  try: 
    ccg = load_npz_3d(os.path.join(directory, file))
  except:
    ccg = load_sparse_npz(os.path.join(directory, file))
  try:
    ccg_jittered = load_npz_3d(os.path.join(directory, file.replace('.npz', '_bl.npz')))
  except:
    ccg_jittered = load_sparse_npz(os.path.join(directory, file.replace('.npz', '_bl.npz')))
  ccg_corrected = ccg - ccg_jittered
  sig_dir = './data/ecephys_cache_dir/sessions/adj_mat_ccg_highland_corrected/'
  significant_ccg = load_npz_3d(os.path.join(sig_dir, file))
  significant_offset = load_npz_3d(os.path.join(sig_dir, file.replace('.npz', '_offset.npz')))
  significant_duration = load_npz_3d(os.path.join(sig_dir, file.replace('.npz', '_duration.npz')))
  row_a, row_b = indx[ind]
  fig, ax = plt.subplots(1, 1, figsize=(8*1, 4))
  # row_a, row_b = all_active_inds.index(row_a), all_active_inds.index(row_b)
  ccg_con = np.concatenate((np.flip(ccg_corrected[row_b, row_a, 1:]), ccg_corrected[row_a, row_b]))
  ax.plot(np.arange(-window, window+1), ccg_con, linewidth=linewidth, color='k')
  ax.axvline(x=0, linestyle='--', color='k', linewidth=3)
  color = 'firebrick' if significant_ccg[row_a, row_b] > 0 else 'blue'
  highland_lag = np.arange(int(significant_offset[row_a,row_b]), int(significant_offset[row_a,row_b]+significant_duration[row_a,row_b]+1))
  ax.plot(highland_lag, ccg_corrected[row_a, row_b, highland_lag], color=color, marker='o', linewidth=linewidth, markersize=15, alpha=0.8)
  if 'bi' in sign:
    color = 'firebrick' if significant_ccg[row_b, row_a] > 0 else 'blue'
    highland_lag = np.arange(int(significant_offset[row_b,row_a]), int(significant_offset[row_b,row_a]+significant_duration[row_b,row_a]+1))
    ax.plot(-highland_lag, ccg_corrected[row_b, row_a, highland_lag], color=color, marker='o', linewidth=linewidth, markersize=15, alpha=0.8)
  ax.set_yticks([])
  ax.set_xlim([-window/2, window/2])
  if scalebar:
    fontprops = fm.FontProperties(size=40)
    size_v = (ccg_corrected[row_a, row_b].max()-ccg_corrected[row_a, row_b].min())/30
    scalebar = AnchoredSizeBar(ax.transData,
                              100, '100 ms', 'lower center',
                              borderpad=0,
                              pad=-1.4,
                              sep=5,
                              color='k',
                              frameon=False,
                              size_vertical=size_v,
                              fontproperties=fontprops)

    ax.add_artist(scalebar)
  ax.set_axis_off()
  plt.tight_layout()
  plt.subplots_adjust(left=0.,
                    bottom=0.,
                    right=1.,
                    top=1.,
                    wspace=1.2)
  plt.savefig('./plots/concat_poly_{}_example_ccg_{}.pdf'.format(sign, ind), transparent=True)
  # plt.show()
  
# plot_concat_ccg_poly(poly_pos_indx, poly_pos_file, 'pos', 49, window=100, scalebar=False)
plot_concat_ccg_poly(poly_neg_indx, poly_neg_file, 'neg', 33, window=100, scalebar=False)
# for ind in range(50, 60):
#   print(ind)
#   plot_concat_ccg_poly(poly_pos_indx, poly_pos_file, 'pos', ind, window=100, scalebar=False) # 49
#   plot_concat_ccg_poly(poly_neg_indx, poly_neg_file, 'neg', ind, window=100, scalebar=False) # 21, 33
#%%
with open('./files/intensity_dict.pkl', 'rb') as f:
  intensity_dict = pickle.load(f)
with open('./files/coherence_dict.pkl', 'rb') as f:
  coherence_dict = pickle.load(f)
with open('./files/sunibi_baseline_intensity_dict.pkl', 'rb') as f:
  sunibi_baseline_intensity_dict = pickle.load(f)
with open('./files/sunibi_baseline_coherence_dict.pkl', 'rb') as f:
  sunibi_baseline_coherence_dict = pickle.load(f)
################## average intensity across session
################## first Z score, then average
num_baseline = 200
whole_df4, mean_df4, signed_motif_types4 = get_intensity_zscore(intensity_dict, coherence_dict, sunibi_baseline_intensity_dict, sunibi_baseline_coherence_dict, num_baseline=num_baseline) # signed uni bi edge preserved
whole_df4['signed motif type'] = whole_df4['signed motif type'].str.replace('-', '\N{MINUS SIGN}') # change minus sign to match width of plus
signed_motif_types4 = [mt.replace('-', '\N{MINUS SIGN}') for mt in signed_motif_types4]
#%%
def plot_zscore_allmotif_lollipop(df, model_name):
  # stimulus_order = [s for s in combined_stimulus_names if df.stimulus.str.contains(s).sum()]
  fig, axes = plt.subplots(len(combined_stimulus_names),1, sharex=True, sharey=True, figsize=(50, 3*len(combined_stimulus_names)))
  sorted_types = [sorted([smotif for smotif in df['signed motif type'].unique() if mt in smotif]) for mt in TRIAD_NAMES]
  sorted_types = [item for sublist in sorted_types for item in sublist]
  motif_types = TRIAD_NAMES[3:]
  motif_loc = [np.mean([i for i in range(len(sorted_types)) if mt in sorted_types[i]]) for mt in motif_types]
  # palette = [plt.cm.tab20(i) for i in range(13)]
  palette = [[plt.cm.tab20b(i) for i in range(20)][i] for i in [0,2,3,4,6,8,10,12,16,18,19]] + [[plt.cm.tab20c(i) for i in range(20)][i] for i in [4,16]]
  for s_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
    print(combined_stimulus_name)
    data = df[df.apply(lambda x: combine_stimulus(x['stimulus'])[1], axis=1)==combined_stimulus_name]
    data = data.groupby('signed motif type').mean()
    ax = axes[len(axes)-1-s_ind] # spontaneous in the bottom
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
    # ax.set_ylabel('Z score', fontsize=40)
    if model_names.index(model_name) <= 1:
      ax.set_yscale('symlog')
    else:
      ax.set_ylim(-20, 30)
  plt.tight_layout()
  # figname = './plots/zscore_all_motifs_lollipop_{}.pdf'.format(model_name.replace(' ', '_'))
  # plt.savefig(figname, transparent=True)
  plt.show()

# plot_zscore_allmotif_lollipop(whole_df)
# dfs = [whole_df1, whole_df2, whole_df3, whole_df4]
# for df_ind, df in enumerate(dfs):
df_ind = 3
plot_zscore_allmotif_lollipop(whole_df4[whole_df4['session']=='797828357'], model_names[df_ind])
#%%
# get all motif edges, select the best
def largest_indices(array, M):
  if len(array) <= M:
    indices = np.arange(len(array))
  else:
    indices = np.argpartition(array, -M)[-M:]
  return indices
  
def smallest_indices(array, M):
  if len(array) <= M:
    indices = np.arange(len(array))
  else:
    indices = np.argpartition(array, M)[:M]
  return indices

def get_all_motifs(G_dict, S_dict, signed_motif_type, weight='confidence'):
  signed_motif_types = ['030T+++', '120D++++', '120U++++', '120C++++', '210+++++', '300++++++']
  motif_types = []
  motif_edges_, motif_sms = {}, {}
  for sm_type in signed_motif_types:
    motif_types.append(sm_type.replace('+', '').replace('-', ''))
  for motif_type in motif_types:
    motif_edges_[motif_type], motif_sms[motif_type] = get_edges_sms(motif_type, weight=weight)
  motif_type = signed_motif_type.replace('+', '').replace('-', '')
  motif_edges, motif_file = [], []
  for session_id in session2keep:
    print(session_id)
    for stimulus_name in stimulus_names:
      G = G_dict[session_id][stimulus_name]
      S = S_dict[session_id][stimulus_name]
      motifs_by_type = find_triads(G) # faster
      motifs = motifs_by_type[motif_type]
      for motif in motifs:
        smotif_type = motif_type + get_motif_sign_new(motif, motif_edges_[motif_type], motif_sms[motif_type], weight=weight)
        # smotif_type = motif_type + get_motif_sign(motif, motif_type, weight=weight)
        if smotif_type == signed_motif_type:
          em = iso.numerical_edge_match(weight, 1)
          motif_weightone = motif.copy()
          for u, v, w in motif_weightone.edges(data=weight):
            motif_weightone[u][v][weight] = 1 if w > 0 else -1
          for unique_sm in motif_sms[motif_type]:
            if nx.is_isomorphic(motif_weightone, unique_sm, edge_match=em):  # match weight
              unique_form = unique_sm
              break
          GM = isomorphism.GraphMatcher(unique_form, motif_weightone)
          assert GM.is_isomorphic(), 'Not isomorphic!!!'
          node_mapping = GM.mapping
          edges = [(node_mapping[edge[0]], node_mapping[edge[1]]) for edge in motif_edges_[motif_type]]
          min_offset = min([S[edge[0]][edge[1]]['offset'] for edge in edges])
          if min_offset > 1:
            motif_edges.append(edges + [np.mean([G[edge[0]][edge[1]]['confidence'] for edge in edges])])
            motif_file.append('{}/{}'.format(session_id, stimulus_name))
  sorted_lists = sorted(zip(motif_edges, motif_file), key=lambda x: x[0][-1], reverse=True)
  motif_edges, motif_file = zip(*sorted_lists)
  return motif_edges, motif_file

motif_edges, motif_file = get_all_motifs(G_ccg_dict, S_ccg_dict, signed_motif_type='030T+++', weight='confidence')
#%%
# plot best ccg edges
def plot_multi_best_ccg_motif(motif_edges, motif_file, M=10, window=100, length=100):
  directory = './data/ecephys_cache_dir/sessions/adj_mat_ccg_corrected/'
  shape = M, 3
  fig = plt.figure(figsize=(8*shape[1], 3*shape[0]))
  for motif_ind in range(shape[0]):
    file = motif_file[motif_ind].replace('/', '_') + '.npz'
    inds_path = './data/ecephys_cache_dir/sessions/active_inds/'
    all_active_inds = list(np.load(os.path.join(inds_path, file.split('_')[0]+'.npy'))) # including thalamus
    try: 
      ccg = load_npz_3d(os.path.join(directory, file))
    except:
      ccg = load_sparse_npz(os.path.join(directory, file))
    try:
      ccg_jittered = load_npz_3d(os.path.join(directory, file.replace('.npz', '_bl.npz')))
    except:
      ccg_jittered = load_sparse_npz(os.path.join(directory, file.replace('.npz', '_bl.npz')))
    ccg_corrected = ccg - ccg_jittered
    sig_dir = './data/ecephys_cache_dir/sessions/adj_mat_ccg_highland_corrected/'
    significant_ccg = load_npz_3d(os.path.join(sig_dir, file))
    significant_offset = load_npz_3d(os.path.join(sig_dir, file.replace('.npz', '_offset.npz')))
    significant_duration = load_npz_3d(os.path.join(sig_dir, file.replace('.npz', '_duration.npz')))
    for edge_ind in range(shape[1]):
      row_a, row_b = motif_edges[motif_ind][edge_ind]
      row_a, row_b = all_active_inds.index(row_a), all_active_inds.index(row_b)
      highland_lag = range(int(significant_offset[row_a,row_b]), int(significant_offset[row_a,row_b]+significant_duration[row_a,row_b]+1))
      ax = plt.subplot(shape[0], shape[1], motif_ind*shape[1]+edge_ind+1)
      
      plt.plot(np.arange(window+1)[:length], ccg_corrected[row_a, row_b][:length], linewidth=3, color='k')
      plt.plot(highland_lag, ccg_corrected[row_a, row_b, highland_lag], color='firebrick', marker='o', linewidth=3, markersize=10, alpha=0.8)
      if edge_ind == 0:
        plt.ylabel('CCG corrected', size=25)
      if motif_ind == shape[0] - 1:
        plt.xlabel('time lag (ms)', size=25)
      plt.xticks(fontsize=22) #, weight='bold'
      plt.yticks(fontsize=22) # , weight='bold'
      for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_color('0.2')
      ax.spines['top'].set_visible(False)
      ax.spines['right'].set_visible(False)
      ax.tick_params(width=1.5)
  plt.tight_layout()
  plt.savefig('./plots/best_ccg_motif_{}.jpg'.format('030T+++'))
  # plt.show()

plot_multi_best_ccg_motif(motif_edges, motif_file, M=10, window=100, length=100)
#%%
def plot_best_ccg(motif_edges, motif_file, ind, window=100, scalebar=False):
  directory = './data/ecephys_cache_dir/sessions/adj_mat_ccg_corrected/'
  pcolor = 'firebrick'
  linewidth = 6.
  for edge_ind, (row_a, row_b) in enumerate(motif_edges[ind][:-1]):
    file = motif_file[ind].replace('/', '_') + '.npz'
    inds_path = './data/ecephys_cache_dir/sessions/active_inds/'
    all_active_inds = list(np.load(os.path.join(inds_path, file.split('_')[0]+'.npy'))) # including thalamus
    try: 
      ccg = load_npz_3d(os.path.join(directory, file))
    except:
      ccg = load_sparse_npz(os.path.join(directory, file))
    try:
      ccg_jittered = load_npz_3d(os.path.join(directory, file.replace('.npz', '_bl.npz')))
    except:
      ccg_jittered = load_sparse_npz(os.path.join(directory, file.replace('.npz', '_bl.npz')))
    ccg_corrected = ccg - ccg_jittered
    sig_dir = './data/ecephys_cache_dir/sessions/adj_mat_ccg_highland_corrected/'
    significant_offset = load_npz_3d(os.path.join(sig_dir, file.replace('.npz', '_offset.npz')))
    significant_duration = load_npz_3d(os.path.join(sig_dir, file.replace('.npz', '_duration.npz')))
    fig, ax = plt.subplots(1, 1, figsize=(8*1, 4))
    row_a, row_b = all_active_inds.index(row_a), all_active_inds.index(row_b)
    highland_lag = range(int(significant_offset[row_a,row_b]), int(significant_offset[row_a,row_b]+significant_duration[row_a,row_b]+1))
    ax.plot(np.arange(window+1), ccg_corrected[row_a, row_b], linewidth=linewidth, color='k')
    ax.plot(highland_lag, ccg_corrected[row_a, row_b, highland_lag], color=pcolor, marker='o', linewidth=linewidth, markersize=15, alpha=0.8)
    ax.set_yticks([])
    ax.set_xlim([-1, 50])
    if scalebar:
      fontprops = fm.FontProperties(size=40)
      size_v = (ccg_corrected[row_a, row_b].max()-ccg_corrected[row_a, row_b].min())/30
      scalebar = AnchoredSizeBar(ax.transData,
                                100, '100 ms', 'lower center',
                                borderpad=0,
                                pad=-1.4,
                                sep=5,
                                color='k',
                                frameon=False,
                                size_vertical=size_v,
                                fontproperties=fontprops)

      ax.add_artist(scalebar)
    ax.set_axis_off()
    plt.tight_layout()
    plt.subplots_adjust(left=0.,
                      bottom=0.,
                      right=1.,
                      top=1.,
                      wspace=1.2)
    plt.savefig('./plots/eFFLb_example_ccg_{}_{}.pdf'.format(ind, edge_ind), transparent=True)
    # plt.show()
plot_best_ccg(motif_edges, motif_file, 5, window=100, scalebar=False)
# %%
def get_example_spike_trains(motif_edges, motif_file, file_ind):
  directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
  mouse_id, stimulus_name = motif_file[file_ind].split('/')
  sequences = load_npz_3d(os.path.join(directory, motif_file[file_ind].replace('/', '_') + '.npz'))
  motif_nodes = np.array([motif_edges[file_ind][:-1][0][0], motif_edges[file_ind][:-1][0][1], motif_edges[file_ind][:-1][1][1]])
  print(motif_nodes)
  motif_spikes = sequences[motif_nodes]
  return motif_spikes
  
file_ind = 5 # 5
motif_spikes = get_example_spike_trains(motif_edges, motif_file, file_ind)
# %%
def plot_raster(motif_spikes, motif_file, file_ind):
  mouse_id, stimulus_name = motif_file[file_ind].split('/')
  palatte = ['#b3cde3', '#ccebc5', '#ffffcc']
  trial_indices = np.arange(0, 75)
  sorted_sample_seq = np.vstack([a[trial_indices, :250] for a in motif_spikes])
  spike_pos = [np.nonzero(t)[0] / 1000 for t in sorted_sample_seq[:, :]] # divided by 1000 cuz bin size is 1 ms
  # colors1 = [palatte[0]] * 100 + [palatte[1]] * 100 + [palatte[2]] * 100
  lineoffsets2 = 1
  linelengths2 = 2.
  # create a horizontal plot
  fig = plt.figure(figsize=(4, 6))
  plt.eventplot(spike_pos, colors='k', lineoffsets=lineoffsets2,
                      linelengths=linelengths2) # colors=colors1
  plt.axis('off')
  plt.gca().invert_yaxis()
  #### add horizontal band
  band_pos = np.arange(0, 3*len(trial_indices)+1, len(trial_indices))
  xgmin, xgmax=.045, .955
  alpha_list = [.4, .4, .4, .6, .5, .5]
  for ind, (loc1, loc2) in enumerate(zip(band_pos[:-1], band_pos[1:])):
    plt.gca().axhspan(loc1, loc2 , xmin=0.03, xmax=0.97, facecolor=palatte[ind], alpha=0.8) #
  #### add box
  # for stimulus_ind in range(len(stimulus2plot)):
  #   plt.gca().add_patch(Rectangle((s_loc1[stimulus_ind]/1000,0),(s_lengths[stimulus_ind]-1)/1000,sorted_sample_seq.shape[0],linewidth=4,edgecolor='k',alpha=.3,facecolor='none')) # region_colors[region_ind]
  plt.tight_layout()
  plt.savefig('./plots/raster_motif_{}.pdf'.format(file_ind), transparent=True)
  # plt.show()

plot_raster(motif_spikes, motif_file, file_ind)
# %%
def plot_PSTH(motif_spikes, file_ind):
  spikes2plot = motif_spikes[:,:,:300].copy()
  palatte = ['#b3cde3', '#ccebc5', '#ffffcc']
  fig, axes = plt.subplots(3, 1, figsize=(4, 6)) # , sharex=True, sharey=True
  for ind, ax in enumerate(axes):
    ax.plot(range(spikes2plot.shape[2]), spikes2plot[ind].mean(0)*1000, color='k')
    ax.set_ylabel('firing rate (Hz)')
    for axis in ['bottom', 'left']:
      ax.spines[axis].set_linewidth(1.5)
      ax.spines[axis].set_color('0.2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
  axes[2].set_xlabel('time since stimulus onset (ms)')  
  plt.tight_layout()
  plt.savefig('./plots/PSTH_motif_{}.pdf'.format(file_ind), transparent=True)
  # plt.show()

plot_PSTH(motif_spikes, file_ind)
# %%
# subsampling a subnetwork
G = G_ccg_dict['797828357']['natural_movie_one'].copy()
sorted_nodes = sorted(dict(G.degree()).items(), key=lambda x:x[1], reverse=True)[::2]
subgraph_nodes = [item[0] for item in sorted_nodes[:10]]
# subgraph_nodes
subgraph = G.subgraph(subgraph_nodes)
subgraph.number_of_edges(), subgraph.number_of_nodes()
#%%
# sample a subgraph with positive and negative edges
G = G_ccg_dict['797828357']['drifting_gratings'].copy()
sorted_nodes = sorted(dict(G.degree()).items(), key=lambda x:x[1], reverse=True)[::2]
hub_node = sorted_nodes[0][0]
sub_nodes = [512, 525, 886, 49, 411, 466, 468, 409, 766, 80]
subgraph = G.subgraph(sub_nodes)
def node_size(degree):
  return 100 * degree  # Adjust the scaling factor as desired

def plot_example_graph(subgraph, seed, k, name, pos=None):
  edge_weights = nx.get_edge_attributes(subgraph, 'weight')

  # Define a colormap for positive and negative edges
  cmap = plt.cm.RdBu

  # Create a list of edge colors based on the sign of edge weights
  edge_colors = ['#fb8072' if weight >= 0 else '#80b1d3' for weight in edge_weights.values()]

  # Plot the graph with edge colors
  node_degrees = subgraph.degree()
  sizes = [node_size(degree) for _, degree in node_degrees]
  if pos == None:
    pos = nx.spring_layout(subgraph, seed=seed, k=k) # , seed=12, k=0.8
  plt.figure()
  nx.draw_networkx(subgraph, pos=pos , node_size=sizes, edge_color=edge_colors, width=4.0, node_color='#737373', connectionstyle='arc3, rad = 0.1', with_labels=False) #
  plt.axis('off')
  plt.tight_layout()
  plt.savefig('./plots/example_graph_{}.pdf'.format(name), transparent=True)
  # plt.show()
  return pos

pos = plot_example_graph(subgraph, seed=14, k=0.8, name='real', pos=None)
# %%
num_rewire = 1
algorithms = ['Gnm', 'directed_double_edge_swap', 'uni_bi_swap', 'signed_uni_bi_swap']
random_Gs = []
for algorithm in algorithms:
  random_Gs.append(random_graph_generator(subgraph, num_rewire, algorithm=algorithm)[0])
  print(algorithm)
  plot_example_graph(random_Gs[-1])
# %%
for G_ind, rand_G in enumerate(random_Gs):
  print(algorithms[G_ind])
  plot_example_graph(rand_G, seed=1, k=0.4, name=algorithms[G_ind])
# %%
# regenerate surrogate networks
# random_Gs[0] = random_graph_generator(subgraph, num_rewire, algorithm='Gnm')[0]
plot_example_graph(random_Gs[0], seed=1, k=0.4, name=algorithms[0], pos=pos)
# %%
# random_Gs[1] = random_graph_generator(subgraph, num_rewire, algorithm='directed_double_edge_swap')[0]
plot_example_graph(random_Gs[1], seed=1, k=0.4, name=algorithms[1], pos=pos)
# %%
# random_Gs[2] = random_graph_generator(subgraph, num_rewire, algorithm='uni_bi_swap')[0]
plot_example_graph(random_Gs[2], seed=2, k=0.4, name=algorithms[2], pos=pos)
# %%
plot_example_graph(random_Gs[3], seed=5, k=0.8, name=algorithms[3], pos=pos)
# %%
# randomly split trials into halves and construct graphs
measure = 'ccg'
num_baseline = 1
directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
inds_path = './data/ecephys_cache_dir/sessions/active_inds/'
session_id, stimulus_name = '797828357', 'natural_scenes'
file = '{}_{}.npz'.format(session_id, stimulus_name)
sequences = load_npz_3d(os.path.join(directory, file))
active_neuron_inds = np.load(os.path.join(inds_path, str(session_id)+'.npy'))
# active_neuron_inds = sequences.mean(1).sum(1) > sequences.shape[2] * min_FR
sequences = sequences[active_neuron_inds]
print('{} spike train shape: {}'.format(file, sequences.shape))
#%%
def save_mean_ccg_corrected_halves(o_sequences, fname, inds, num_jitter=10, L=25, window=100, disable=True): ### fastest, only causal correlation (A>B, only positive time lag on B), largest deviation from flank
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
path = os.path.join(directory.replace('spiking_sequence', 'adj_mat_ccg_corrected_halves'))
start_time = time.time()
attempt = int(sys.argv[1])
np.random.seed(attempt)

# # randomly select all trials
# half_ind1 = np.random.choice(np.arange(sequences.shape[1]), sequences.shape[1]//2, replace=False)
# half_ind2 = np.setdiff1d(np.arange(sequences.shape[1]), half_ind1, assume_unique=True)

# randomly select trials, but make sure each image is in each trial for the same presentations
session_id = '797828357'
stimulus_name = 'natural_scenes'
data_directory = './data/ecephys_cache_dir'
manifest_path = os.path.join(data_directory, "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
session = cache.get_session_data(int(session_id),
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
stim_table = session.get_stimulus_table([stimulus_name])
stim_table=stim_table.rename(columns={"start_time": "Start", "stop_time": "End"})
stimulus_presentation_ids = stim_table.index[stim_table.duration >= 0.25].values
image_ids = stim_table['stimulus_condition_id'].unique()
all_stimulus_ids = stim_table.loc[stimulus_presentation_ids]['stimulus_condition_id'].values
np.random.shuffle(image_ids)
half_ind1 = []
for image_id in image_ids:
  locs = np.where(all_stimulus_ids==image_id)[0]
  half_ind1 += np.random.choice(locs, len(locs)//2, replace=False).tolist()
half_ind1 = np.array(sorted(half_ind1))
half_ind2 = np.setdiff1d(range(len(all_stimulus_ids)), half_ind1, assume_unique=True)
print('{}th attempt, '.format(attempt), half_ind1.shape, half_ind2.shape)
fname1 = os.path.join(path, file.replace('.npz', '_{}_{}.npz'.format(attempt, 1)))
fname2 = os.path.join(path, file.replace('.npz', '_{}_{}.npz'.format(attempt, 2)))
save_mean_ccg_corrected_halves(o_sequences=sequences, inds=half_ind1, fname=fname1, num_jitter=num_baseline, L=25, window=100, disable=True)
save_mean_ccg_corrected_halves(o_sequences=sequences, inds=half_ind2, fname=fname2, num_jitter=num_baseline, L=25, window=100, disable=True)
print("--- %s minutes" % ((time.time() - start_time)/60))
#%%
# save significant CCG for halves
start_time = time.time()
measure = 'ccg'
min_spike = 50
n = 4
max_duration = 11
maxlag = 12
directory = './data/ecephys_cache_dir/sessions/adj_mat_{}_corrected_halves/'.format(measure)
save_ccg_corrected_highland_new(directory, measure, min_spike=min_spike, max_duration=max_duration, maxlag=maxlag, n=n)
print("--- %s minutes in total" % ((time.time() - start_time)/60))
#%%
threshold = 0.9
directory = './data/ecephys_cache_dir/sessions/adj_mat_ccg_highland_corrected_halves/'
keep_edges_above_threshold(directory, threshold)
# %%
# load stationary and running graphs
def load_highland_halves(directory, file_name, active_area_dict, weight):
  G_dict, offset_dict, duration_dict = {}, {}, {}
  files = os.listdir(directory)
  files.sort(key=lambda x:x)
  for file in files:
    if file.endswith(".npz") and ('gabors' not in file) and ('flashes' not in file) and ('_offset' not in file) and ('_duration' not in file) and ('_bl' not in file) and ('confidence' not in file):
      print(file)
      adj_mat = load_npz_3d(os.path.join(directory, file))
      confidence_level = load_npz_3d(os.path.join(directory, file.replace('.npz', '_confidence.npz')))
      # adj_mat = np.load(os.path.join(directory, file))
      repeat, half = file.replace(file_name, '').replace('.npz', '').split('_')
      if not repeat in G_dict:
        G_dict[repeat], offset_dict[repeat], duration_dict[repeat] = {}, {}, {}
      G_dict[repeat][half] = generate_graph(adj_mat=np.nan_to_num(adj_mat), confidence_level=confidence_level, active_area=active_area_dict[file_name.split('_')[0]], cc=False, weight=weight)
      offset_dict[repeat][half] = load_npz_3d(os.path.join(directory, file.replace('.npz', '_offset.npz')))
      duration_dict[repeat][half] = load_npz_3d(os.path.join(directory, file.replace('.npz', '_duration.npz')))
  return G_dict, offset_dict, duration_dict


session_id, stimulus_name = '797828357', 'natural_scenes'
file_name = session_id + '_' + stimulus_name + '_'
area_dict, active_area_dict, mean_speed_df = load_other_data(session_ids)
cortical_inds = get_cortical_inds(active_area_dict, visual_regions)
directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
path = directory.replace('spiking_sequence', 'adj_mat_ccg_highland_corrected_halves')
G_halves_dict, offset_halves_dict, duration_halves_dict = load_highland_halves(path, file_name, active_area_dict, weight=True)
measure = 'ccg'
def remove_thalamic_halves(G_dict, area_dict, regions):
  rows, cols = get_rowcol(G_dict)
  for row in rows:
    for col in cols:
      G = G_dict[row][col].copy()
      nodes = [n for n in G.nodes() if area_dict['797828357'][n] in regions]
      G_dict[row][col] = G.subgraph(nodes)
  return G_dict

G_halves_dict = remove_thalamic_halves(G_halves_dict, area_dict, visual_regions)
S_halves_dict = add_sign(G_halves_dict)
S_halves_dict = add_offset(S_halves_dict, offset_halves_dict)
S_halves_dict = add_duration(S_halves_dict, duration_halves_dict)
S_halves_dict = add_delay(S_halves_dict)
# %%
# plot density, within-fraction, ex-fraction, clustering for halves graph
stimulus2marker = {'Resting\nstate':'s', 'Flashes':'*', 'Drifting\ngratings':'X', 'Static\ngratings':'P', 'Natural\nscenes':r'$\clubsuit$', 'Natural\nmovies':'>'}
marker_size_dict = {'v':10, '*':22, 'P':13, 'X':13, 'o':11, 's':9.5, 'D':9, 'p':12, '>':10, r'$\clubsuit$':20}
scatter_size_dict = {'v':10, '*':17, 'P':13, 'X':13, 'o':11, 's':10, 'D':9, 'p':13, '>':12, r'$\clubsuit$':16}
error_size_dict = {'v':10, '*':24, 'P':16, 'X':16, 'o':11, 's':9., 'D':9, 'p':12, '>':13, r'$\clubsuit$':22}

def plot_bar_halves(G_whole, G_halves_dict, area_dict, regions):
  df = pd.DataFrame()
  region_connection = np.zeros((len(regions), len(regions)))
  metric_names = ['density', 'ratio of intra-region connections', 'ratio of excitatory connections', 'clustering coefficient']
  G = G_whole.copy()
  nodes = list(G.nodes())
  node_area = {key: area_dict['797828357'][key] for key in nodes}
  A = nx.to_numpy_array(G, nodelist=nodes)
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
  intra_data = np.sum(region_connection[diag_indx])/np.sum(region_connection)
  # inter_data.append(np.sum(region_connection[~diag_indx])/np.sum(region_connection))
  density_data = nx.density(G)
  signs = list(nx.get_edge_attributes(G, "sign").values())
  ex_data = signs.count(1) / len(signs)
  in_data = signs.count(-1) / len(signs)
  cluster_data = calculate_directed_metric(G, 'clustering')
  data_whole = [density_data, intra_data, ex_data, cluster_data]
  for repeat in sorted(list(G_halves_dict.keys())):
    for half in G_halves_dict[repeat]:
      G = G_halves_dict[repeat][half].copy()
      # nodes = list(G.nodes())
      # node_area = {key: area_dict['797828357'][key] for key in nodes}
      A = nx.to_numpy_array(G, nodelist=nodes)
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
      intra_data = np.sum(region_connection[diag_indx])/np.sum(region_connection)
      # inter_data.append(np.sum(region_connection[~diag_indx])/np.sum(region_connection))
      density_data = nx.density(G)
      signs = list(nx.get_edge_attributes(G, "sign").values())
      ex_data = signs.count(1) / len(signs)
      in_data = signs.count(-1) / len(signs)
      cluster_data = calculate_directed_metric(G, 'clustering')
      df = pd.concat([df, pd.DataFrame([['ratio of intra-region connections', intra_data, repeat, half]], columns=['metric', 'data', 'repeat', 'half'])], ignore_index=True)
      df = pd.concat([df, pd.DataFrame([['density', density_data, repeat, half]], columns=['metric', 'data', 'repeat', 'half'])], ignore_index=True)
      df = pd.concat([df, pd.DataFrame([['ratio of excitatory connections', ex_data, repeat, half]], columns=['metric', 'data', 'repeat', 'half'])], ignore_index=True)
      df = pd.concat([df, pd.DataFrame([['clustering coefficient', cluster_data, repeat, half]], columns=['metric', 'data', 'repeat', 'half'])], ignore_index=True)
  # df = pd.concat([df, pd.DataFrame(np.concatenate((np.repeat(np.array(metric_names), 3)[:,None], np.array(density_data + intra_data + ex_data + cluster_data)[:,None], np.array(['whole', 'stationary', 'running'] * 4)[:,None]), 1), columns=['metric', 'data', 'behavior type'])], ignore_index=True)
  df['data'] = pd.to_numeric(df['data'])
  fig, axes = plt.subplots(2, 2, figsize=(10, 8))
  for i, j in [(f,s) for f in range(axes.shape[0]) for s in range(axes.shape[1])]:
    m_ind = i * axes.shape[1] + j
    ax = axes[i, j]
    # x = sorted(list(G_halves_dict.keys()))
    # y = df[df['metric']==metric_names[m_ind]]['data'].values
    sns.barplot(df[df['metric']==metric_names[m_ind]], x='repeat', y='data', hue='half', ax=ax)
    ax.axhline(y=data_whole[m_ind], linestyle='--', color='k')
    # ax.bar(x, y)
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
  plt.savefig('./plots/bar_whole_halves.pdf', transparent=True)

S_whole = S_ccg_dict['797828357']['natural_scenes']
plot_bar_halves(S_whole, S_halves_dict, area_dict, visual_regions)
#%%
# def load_1(directory, active_area_dict, weight):
#   G_dict, offset_dict, duration_dict = {}, {}, {}
#   files = os.listdir(directory)
#   files.sort(key=lambda x:int(x[:9]))
#   file = '797828357_natural_scenes.npz'
#   confidence_level = load_npz_3d(os.path.join(directory, file.replace('.npz', '_confidence.npz')))
#   adj = confidence_level.copy()
#   adj[~np.isnan(adj)] = 1
#   return adj
# directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
# path = directory.replace('spiking_sequence', 'adj_mat_ccg_highland_corrected')
# if not os.path.exists(path):
#   os.makedirs(path)
# adj_whole = load_1(path, active_area_dict, weight=True)
# %%
# plot the similarity between adjacency matrices
def overlap_coefficient(set1, set2):
  set1, set2 = set(set1), set(set2)
  intersection = len(set1.intersection(set2))
  min_size = min(len(set1), len(set2))
  if min_size == 0:
    return 0.0
  else:
    return intersection / min_size

def jaccard_similarity(array1, array2):
    set1 = set(array1)
    set2 = set(array2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    similarity = intersection / union
    return similarity

def plot_halves_adj_corr(G_whole, G_halves_dict):
  rows, cols = get_rowcol(G_halves_dict)
  nodes = sorted(list(G_whole.nodes()))
  adj = nx.to_numpy_array(G_whole, nodelist=nodes)
  adj[adj.nonzero()] = 1
  adj_list = [adj]
  num = len(rows) * len(cols) + 1
  corr = np.zeros((num, num))
  for r_ind, row in enumerate(rows):
    for c_ind, col in enumerate(cols):
      G = G_halves_dict[row][col]
      assert nodes == sorted(list(G.nodes())), '{}, {}'.format(G_whole.number_of_nodes(), G.number_of_nodes())
      adj = nx.to_numpy_array(G, nodelist=nodes)
      adj[adj.nonzero()] = 1
      adj_list.append(adj)
  for i in range(len(adj_list)):
    for j in range(len(adj_list)):
      if i != j:
        # corr[i, j] = (adj_list[i] * adj_list[j]).sum() / np.prod(adj_list[0].shape)
        # corr[i, j] = overlap_coefficient(adj_list[i].flatten().nonzero()[0], adj_list[j].flatten().nonzero()[0])
        corr[i, j] = jaccard_similarity(adj_list[i].flatten().nonzero()[0], adj_list[j].flatten().nonzero()[0])
        
  np.fill_diagonal(corr, np.nan)
  print(np.nanmin(corr), np.nanmax(corr))
  ticklabels = ['whole'] + [str(ind) for ind in range(1, 11)]
  sns.heatmap(corr, xticklabels=ticklabels, yticklabels=ticklabels)
  plt.tight_layout()
  # plt.savefig('./plots/overlap_whole_halves.pdf', transparent=True)
  plt.savefig('./plots/jaccard_whole_halves.pdf', transparent=True)
  # plt.show()
      
G_whole = G_ccg_dict['797828357']['natural_scenes']
plot_halves_adj_corr(G_whole, G_halves_dict)
# %%
# plot the similarity between adjacency matrices
def plot_stimulus_adj_corr(G_whole, G_dict):
  rows, cols = get_rowcol(G_dict)
  nodes = sorted(list(G_whole.nodes()))
  # adj = nx.to_numpy_array(G_whole, nodelist=nodes)
  # adj[adj.nonzero()] = 1
  adj_list = []
  num = len(cols)
  corr = np.zeros((num, num))
  for c_ind, col in enumerate(cols):
    G = G_dict['797828357'][col]
    assert nodes == sorted(list(G.nodes())), '{}, {}'.format(G_whole.number_of_nodes(), G.number_of_nodes())
    adj = nx.to_numpy_array(G, nodelist=nodes)
    adj[adj.nonzero()] = 1
    adj_list.append(adj)
  for i in range(len(adj_list)):
    for j in range(len(adj_list)):
      if i != j:
        # corr[i, j] = (adj_list[i] * adj_list[j]).sum() / np.prod(adj_list[0].shape)
        # corr[i, j] = overlap_coefficient(adj_list[i].flatten().nonzero()[0], adj_list[j].flatten().nonzero()[0])
        corr[i, j] = jaccard_similarity(adj_list[i].flatten().nonzero()[0], adj_list[j].flatten().nonzero()[0])
  np.fill_diagonal(corr, np.nan)
  print(np.nanmin(corr), np.nanmax(corr))
  ticklabels = cols
  sns.heatmap(corr, xticklabels=ticklabels, yticklabels=ticklabels)
  plt.tight_layout()
  # plt.savefig('./plots/overlap_naturalscene_stimulus.pdf', transparent=True)
  plt.savefig('./plots/jaccard_naturalscene_stimulus.pdf', transparent=True)
  # plt.show()
      
plot_stimulus_adj_corr(G_whole, G_ccg_dict)
# %%
######################## signed motif detection for halves graph
with open('./files/intensity_dict.pkl', 'rb') as f:
  intensity_dict = pickle.load(f)
with open('./files/coherence_dict.pkl', 'rb') as f:
  coherence_dict = pickle.load(f)
with open('./files/sunibi_baseline_intensity_dict.pkl', 'rb') as f:
  sunibi_baseline_intensity_dict = pickle.load(f)
with open('./files/sunibi_baseline_coherence_dict.pkl', 'rb') as f:
  sunibi_baseline_coherence_dict = pickle.load(f)
with open('./files/intensity_halves_dict.pkl', 'rb') as f:
  intensity_halves_dict = pickle.load(f)
with open('./files/coherence_halves_dict.pkl', 'rb') as f:
  coherence_halves_dict = pickle.load(f)
with open('./files/sunibi_baseline_intensity_halves_dict.pkl', 'rb') as f:
  sunibi_baseline_intensity_halves_dict = pickle.load(f)
with open('./files/sunibi_baseline_coherence_halves_dict.pkl', 'rb') as f:
  sunibi_baseline_coherence_halves_dict = pickle.load(f)
################## average intensity across session
################## first Z score, then average
num_baseline = 200
whole_df1, mean_df1, signed_motif_types1 = get_intensity_zscore(intensity_dict, coherence_dict, sunibi_baseline_intensity_dict, sunibi_baseline_coherence_dict, num_baseline=num_baseline) # signed uni bi edge preserved
whole_df2, mean_df2, signed_motif_types2 = get_intensity_zscore(intensity_halves_dict, coherence_halves_dict, sunibi_baseline_intensity_halves_dict, sunibi_baseline_coherence_halves_dict, num_baseline=num_baseline) # signed uni bi edge preserved
whole_df1['signed motif type'] = whole_df1['signed motif type'].str.replace('-', '\N{MINUS SIGN}') # change minus sign to match width of plus
whole_df2['signed motif type'] = whole_df2['signed motif type'].str.replace('-', '\N{MINUS SIGN}') # change minus sign to match width of plus
signed_motif_types1 = [mt.replace('-', '\N{MINUS SIGN}') for mt in signed_motif_types1]
signed_motif_types2 = [mt.replace('-', '\N{MINUS SIGN}') for mt in signed_motif_types2]
#%%
def add_missing_motif_type_halves(df, mtype, signed_motif_types):
  if len(mtype) < len(signed_motif_types):
    mtype2add = [t for t in signed_motif_types if t not in mtype]
    for mt in mtype2add:
      mtype.append(mt)
      for session_id in df['session'].unique():
        for stimulus_name in df['stimulus'].unique():
          df = pd.concat([df, pd.DataFrame([[mt, session_id, stimulus_name] + [0] * (df.shape[1]-3)], columns=df.columns)], ignore_index=True)
    df['intensity z score'] = pd.to_numeric(df['intensity z score'])
  return df, mtype

whole_df1 = whole_df1[(whole_df1['session']=='797828357')&(whole_df1['stimulus']=='natural_scenes')]
signed_motif_types1 = list(whole_df1['signed motif type'].unique())
signed_motif_types = np.unique(signed_motif_types1+signed_motif_types2).tolist()
whole_df1, signed_motif_types1 = add_missing_motif_type_halves(whole_df1, signed_motif_types1, signed_motif_types)
whole_df2, signed_motif_types2 = add_missing_motif_type_halves(whole_df2, signed_motif_types2, signed_motif_types)
TRIAD_NAMES = ('003', '012', '102', '021D', '021U', '021C', '111D', '111U', '030T', '030C', '201', '120D', '120U', '120C', '210', '300')
# %%
# plot zscores of all motifs using lollipop plot for halves graph
def plot_zscore_allmotif_lollipop_halves(df):
  # stimulus_order = [s for s in combined_stimulus_names if df.stimulus.str.contains(s).sum()]
  num_repeats = len(df.session.unique())
  fig, axes = plt.subplots(num_repeats,2, sharex=True, sharey=True, figsize=(50*2, 3*num_repeats))
  all_smotif_types = list(set(list(df['signed motif type'].values)))
  sorted_types = [sorted([smotif for smotif in all_smotif_types if mt in smotif]) for mt in TRIAD_NAMES]
  sorted_types = [item for sublist in sorted_types for item in sublist]
  motif_types = TRIAD_NAMES[3:]
  motif_loc = [np.mean([i for i in range(len(sorted_types)) if mt in sorted_types[i]]) for mt in motif_types]
  # palette = [plt.cm.tab20(i) for i in range(13)]
  palette = [[plt.cm.tab20b(i) for i in range(20)][i] for i in [0,2,3,4,6,8,10,12,16,18,19]] + [[plt.cm.tab20c(i) for i in range(20)][i] for i in [4,16]]
  for r_ind in range(num_repeats):
    for half in range(1, 3):
      ax = axes[r_ind, half-1] # spontaneous in the bottom
      data = df[(df['session']==str(r_ind)) & (df['stimulus']==str(half))]
      data = data.groupby('signed motif type').mean()
      # ax.set_title(combined_stimulus_names[s_ind].replace('\n', ' '), fontsize=35, rotation=0)
      for t, y in zip(sorted_types, data.loc[sorted_types, "intensity z score"]):
        color = palette[motif_types.index(t.replace('+', '').replace('\N{MINUS SIGN}', ''))]
        ax.plot([t,t], [0,y], color=color, marker="o", linewidth=7, markersize=20, markevery=(1,2))
      ax.set_xlim(-.5,len(sorted_types)+.5)
      ax.set_xticks([])
      ax.yaxis.set_tick_params(labelsize=45)
      for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(4.5)
        ax.spines[axis].set_color('k')
      ax.spines['top'].set_visible(False)
      ax.spines['right'].set_visible(False)
      ax.tick_params(width=4.5)
      ax.xaxis.set_tick_params(length=0)
      ax.set_ylabel('')
      ax.set_ylim(-18, 38)
  plt.tight_layout()
  figname = './plots/zscore_all_motifs_lollipop_halves.pdf'
  plt.savefig(figname, transparent=True)
  # plt.show()

df_halves = whole_df2
plot_zscore_allmotif_lollipop_halves(df_halves)
# %%
# plot the whole graph
def plot_zscore_allmotif_lollipop_whole(df):
  # stimulus_order = [s for s in combined_stimulus_names if df.stimulus.str.contains(s).sum()]
  fig, ax = plt.subplots(1,1, sharex=True, sharey=True, figsize=(50, 3))
  all_smotif_types = list(set(list(df['signed motif type'].values)))
  sorted_types = [sorted([smotif for smotif in all_smotif_types if mt in smotif]) for mt in TRIAD_NAMES]
  sorted_types = [item for sublist in sorted_types for item in sublist]
  motif_types = TRIAD_NAMES[3:]
  motif_loc = [np.mean([i for i in range(len(sorted_types)) if mt in sorted_types[i]]) for mt in motif_types]
  # palette = [plt.cm.tab20(i) for i in range(13)]
  palette = [[plt.cm.tab20b(i) for i in range(20)][i] for i in [0,2,3,4,6,8,10,12,16,18,19]] + [[plt.cm.tab20c(i) for i in range(20)][i] for i in [4,16]]
  data = df
  data = data.groupby('signed motif type').mean()
  # ax.set_title(combined_stimulus_names[s_ind].replace('\n', ' '), fontsize=35, rotation=0)
  for t, y in zip(sorted_types, data.loc[sorted_types, "intensity z score"]):
    color = palette[motif_types.index(t.replace('+', '').replace('\N{MINUS SIGN}', ''))]
    ax.plot([t,t], [0,y], color=color, marker="o", linewidth=7, markersize=20, markevery=(1,2))
  ax.set_xlim(-.5,len(sorted_types)+.5)
  ax.set_xticks([])
  ax.yaxis.set_tick_params(labelsize=45)
  for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(4.5)
    ax.spines[axis].set_color('k')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.tick_params(width=4.5)
  ax.xaxis.set_tick_params(length=0)
  ax.set_ylabel('')
  ax.set_ylim(-18, 38)
  plt.tight_layout()
  figname = './plots/zscore_all_motifs_lollipop_whole.pdf'
  plt.savefig(figname, transparent=True)
  # plt.show()

df_whole = whole_df1
plot_zscore_allmotif_lollipop_whole(df_whole)
# %%
# get motif stimulus list dictionary for each mouse and each stimulus and each motif type
def get_motif_IDs(G_dict, signed_motif_types):
  rows, cols = get_rowcol(G_dict)
  motif_id_dict = {}
  motif_types = []
  motif_edges, motif_sms = {}, {}
  for signed_motif_type in signed_motif_types:
    motif_types.append(signed_motif_type.replace('+', '').replace('-', ''))
  for motif_type in motif_types:
    motif_edges[motif_type], motif_sms[motif_type] = get_edges_sms(motif_type, weight='confidence')
  for row_ind, row in enumerate(rows):
    print(row)
    motif_id_dict[row] = {}
    for cs_ind, combined_stimulus_name in enumerate(combined_stimulus_names):
      print(combined_stimulus_name)
      motif_id_dict[row][combined_stimulus_name] = {s_type:[] for s_type in signed_motif_types}
      for col in combined_stimuli[cs_ind]:
        G = G_dict[row][col]
        motifs_by_type = find_triads(G) # faster
        for signed_motif_type in signed_motif_types:
          motif_type = signed_motif_type.replace('+', '').replace('-', '')
          motifs = motifs_by_type[motif_type]
          for motif in motifs:
            smotif_type = motif_type + get_motif_sign_new(motif, motif_edges[motif_type], motif_sms[motif_type], weight='confidence')
            if (smotif_type == signed_motif_type) and (not tuple(list(motif.nodes())) in motif_id_dict[row][combined_stimulus_name][smotif_type]):
              motif_id_dict[row][combined_stimulus_name][smotif_type].append(tuple(list(motif.nodes())))
  return motif_id_dict

# get motif IDs for halves graph
def get_halves_motif_IDs(G_dict, signed_motif_types):
  rows, cols = get_rowcol(G_dict)
  motif_id_dict = {}
  motif_types = []
  motif_edges, motif_sms = {}, {}
  for signed_motif_type in signed_motif_types:
    motif_types.append(signed_motif_type.replace('+', '').replace('-', ''))
  for motif_type in motif_types:
    motif_edges[motif_type], motif_sms[motif_type] = get_edges_sms(motif_type, weight='confidence')
  for row_ind, row in enumerate(rows):
    print(row)
    motif_id_dict[row] = {}
    for col in cols:
      motif_id_dict[row][col] = {s_type:[] for s_type in signed_motif_types}
      G = G_dict[row][col]
      motifs_by_type = find_triads(G) # faster
      for signed_motif_type in signed_motif_types:
        motif_type = signed_motif_type.replace('+', '').replace('-', '')
        motifs = motifs_by_type[motif_type]
        for motif in motifs:
          smotif_type = motif_type + get_motif_sign_new(motif, motif_edges[motif_type], motif_sms[motif_type], weight='confidence')
          if (smotif_type == signed_motif_type) and (not tuple(list(motif.nodes())) in motif_id_dict[row][col][smotif_type]):
            motif_id_dict[row][col][smotif_type].append(tuple(list(motif.nodes())))
  return motif_id_dict

sig_motif_types = ['030T+++', '120D++++', '120U++++', '120C++++', '210+++++', '300++++++']
motif_id_dict = get_motif_IDs(G_ccg_dict, sig_motif_types)
motif_id_halves_dict = get_halves_motif_IDs(G_halves_dict, sig_motif_types)
# %%
# unique motifs for halves graph
motif_halves_dict = {sig_motif_type:{} for sig_motif_type in sig_motif_types}
for sig_motif_type in sig_motif_types:
  motif_halves_dict[sig_motif_type]['whole'] = motif_id_dict['797828357']['Natural\nscenes'][sig_motif_type]
for sig_motif_type in sig_motif_types:
  for repeat in motif_id_halves_dict:
    for half in motif_id_halves_dict[repeat]:
      motif_halves_dict[sig_motif_type][repeat + '_' + half] = motif_id_halves_dict[repeat][half][sig_motif_type]
# %%
# make upset plot for each motif type, combine across mouse
def plot_multi_upsetplot_combined(motif_halves_dict, sig_motif_types):
  for sig_motif_type in sig_motif_types:
    motif_stim = from_contents(motif_halves_dict[sig_motif_type])
    # print(motif_stim.index)
    # motif_stim.index.rename(['',' ','  ','   ','    ','     '], inplace=True)
    fig = plt.figure(figsize=(10, 6))
    overlapping_arr = np.array(list(motif_stim.index))
    print((overlapping_arr.sum(1)>1).sum()/overlapping_arr.shape[0]) # fraction of motifs that appear on at least two networks
    plot(motif_stim, sort_categories_by='input', sort_by='cardinality', fig=fig) # , sort_by='cardinality'
    # plt.suptitle('Ordered by cardinality')
    fig.savefig('./plots/upset_plot_whole_halves_{}.pdf'.format(sig_motif_type.replace('+', '').replace('-', '')), transparent=True)

plot_multi_upsetplot_combined(motif_halves_dict, sig_motif_types)
# %%
# unique motif for different stimuli for this mouse
def get_motif_stim_list_dict_onesession(motif_id_dict, session_id, signed_motif_types):
  motif_stim_dict = {smt:{} for smt in signed_motif_types}
  for smt_ind, signed_motif_type in enumerate(signed_motif_types):
    motif_stim_dict[signed_motif_type] = {}
    for s_ind, session_id in enumerate(session2keep):
      for combined_stimulus_name in combined_stimulus_names:
        motif_stim_dict[signed_motif_type][combined_stimulus_name] = motif_id_dict[session_id][combined_stimulus_name][signed_motif_type]
  return motif_stim_dict
      
motif_stim_list_dict = get_motif_stim_list_dict_onesession(motif_id_dict, session_id, sig_motif_types)
# %%
def get_motif_overlap(motif_stim_list_dict, motif_halves_dict, sig_motif_types):
  metric_names = ['sharing fraction', 'overlap fraction']
  df = pd.DataFrame()
  for sig_motif_type in sig_motif_types:
    motif_stim = from_contents(motif_halves_dict[sig_motif_type])
    overlapping_arr = np.array(list(motif_stim.index))
    sharing = (overlapping_arr.sum(1)>1).sum()/overlapping_arr.shape[0] # fraction of motifs that appear on at least two networks
    overlaps = []
    for i in range(overlapping_arr.shape[1]):
      count_rows_with_condition = np.sum((overlapping_arr[:, i] == True) & (np.sum(overlapping_arr, axis=1) == 1))
      count_rows_with_i_true = np.sum(overlapping_arr[:, i] == True)
      ratio = count_rows_with_condition / count_rows_with_i_true
      # mask = (overlapping_arr[:, i] == True) & (np.sum(overlapping_arr, axis=1) == 1)
      if ~np.isnan(ratio):
        overlaps.append(1 - ratio)
    df = pd.concat([df, pd.DataFrame([['sharing fraction', sharing, sig_motif_type, 'within stimulus']], columns=['metric', 'data', 'motif', 'type'])], ignore_index=True)
    df = pd.concat([df, pd.DataFrame(np.array([['overlap fraction']*len(overlaps), overlaps, [sig_motif_type]*len(overlaps), ['within stimulus']*len(overlaps)]).T, columns=['metric', 'data', 'motif', 'type'])], ignore_index=True)
    
    motif_stim = from_contents(motif_stim_list_dict[sig_motif_type])
    overlapping_arr = np.array(list(motif_stim.index))
    sharing = (overlapping_arr.sum(1)>1).sum()/overlapping_arr.shape[0] # fraction of motifs that appear on at least two networks
    overlaps = []
    for i in range(overlapping_arr.shape[1]):
      count_rows_with_condition = np.sum((overlapping_arr[:, i] == True) & (np.sum(overlapping_arr, axis=1) == 1))
      count_rows_with_i_true = np.sum(overlapping_arr[:, i] == True)
      ratio = count_rows_with_condition / count_rows_with_i_true
      # mask = (overlapping_arr[:, i] == True) & (np.sum(overlapping_arr, axis=1) == 1)
      if ~np.isnan(ratio):
        overlaps.append(1 - ratio)
    df = pd.concat([df, pd.DataFrame([['sharing fraction', sharing, sig_motif_type, 'across stimuli']], columns=['metric', 'data', 'motif', 'type'])], ignore_index=True)
    df = pd.concat([df, pd.DataFrame(np.array([['overlap fraction']*len(overlaps), overlaps, [sig_motif_type]*len(overlaps), ['across stimuli']*len(overlaps)]).T, columns=['metric', 'data', 'motif', 'type'])], ignore_index=True)
  df['data'] = pd.to_numeric(df['data'])
  palette = ['k','w']
  fig, axes = plt.subplots(1, 2, figsize=(10, 4))
  for i in range(2):
    ax = axes[i]
    barplot = sns.barplot(x='motif', y='data', hue='type', hue_order=['within stimulus', 'across stimuli'], palette=palette, ec='k', linewidth=2., data=df[df['metric']==metric_names[i]], ax=ax, capsize=.05, width=0.6, errorbar=('ci', 95))
    # sns.barplot(df[df['metric']==metric_names[i]], x='motif', y='data', hue='type', ax=ax)
    ax.set_ylabel(metric_names[i], fontsize=15)
    ax.set_xlabel('')
    if i == 1:
      ax.legend([], frameon=False)
    ax.xaxis.set_tick_params(labelsize=15, rotation=90)
    ax.yaxis.set_tick_params(labelsize=15)
    for axis in ['bottom', 'left']:
      ax.spines[axis].set_linewidth(1.5)
      ax.spines[axis].set_color('0.2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=1.5)
  plt.tight_layout()
  # plt.show()
  plt.savefig('./plots/bar_motif_overlap_whole_halves.pdf', transparent=True)

get_motif_overlap(motif_stim_list_dict, motif_halves_dict, sig_motif_types)
# %%
