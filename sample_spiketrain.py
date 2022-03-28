#%%
import numpy as np
import matplotlib.pyplot as plt
# from Data import *
from tqdm import tqdm

def getSpikeData(T, fireRate):
    spikeData = np.zeros(T)
    binprob = (1./T)*fireRate
    rands = np.random.uniform(size=T)
    spikeData[rands <= binprob] = 1
    return spikeData

def getSpikeTrain(spikeData):
    spikeTrain = np.squeeze(np.where(spikeData>0))
    return spikeTrain

def getInitDist(L):
    initDist = np.random.rand(L)
    return initDist/initDist.sum()

def getTransitionMatrices(L, N):
    tDistMatrices = np.zeros((N - 1, L, L))
    for i in range(tDistMatrices.shape[0]):
        matrix = np.random.rand(L, L)
        stochMatrix = matrix/matrix.sum(axis=1)[:,None]
        tDistMatrices[i, :, :] = stochMatrix.astype('f')
    return tDistMatrices

def getX1(initDist, L, R, spikeTrain):
    # Omega = getOmega(L, obsTar)
    Gamma = getGamma(L, R, spikeTrain)
    randX = np.random.random()
    ind = np.where(randX <= np.cumsum(initDist))[0][0]
    return Gamma[0][ind]

def initializeX(initX, Prob):
    return initX + np.sum(Prob == 0)

def getGamma(L, R, spikeTrain):
    Gamma = []
    ks = [] # list of k_d
    ks.append(0)
    n = len(spikeTrain)
    temp = int(spikeTrain[ks[-1]]/L)*L
    Gamma.append(np.arange(temp, temp + L, 1))
    for i in range(1, n):
        if spikeTrain[i] - spikeTrain[i-1] > R:
            ks.append(i)
        temp = int(spikeTrain[ks[-1]]/L)*L+spikeTrain[i]-spikeTrain[ks[-1]]
        Gamma.append(np.arange(temp, temp + L, 1))
    return Gamma

def getSurrogate(spikeTrain, L, R, initDist, tDistMatrices):
    surrogate = []
    # Omega = getOmega(L, spikeTrain)
    Gamma = getGamma(L, R, spikeTrain)
    givenX = getX1(initDist, L, R, spikeTrain)
    surrogate.append(givenX)
    for i, row in enumerate(tDistMatrices):
        if spikeTrain[i+1] - spikeTrain[i] <= R:
            givenX = surrogate[-1] + spikeTrain[i+1] - spikeTrain[i]
        else:
            index = np.where(np.array(Gamma[i]) == givenX)[0]
            p_i = np.squeeze(np.array(row[index]))
            initX = initializeX(Gamma[i + 1][0], p_i)
            randX = np.random.random()
            # safe way to find the ind
            larger = np.where(randX <= np.cumsum(p_i))[0]
            if larger.shape[0]:
                ind = larger[0]
            else:
                ind = len(p_i) - 1
            givenX = initX + np.sum(p_i[:ind]!=0)
        surrogate.append(givenX)
    return surrogate

def sample_spiketrain(L, R, spikeTrain, initDist, tDistMatrices, sample_size):
    spikeTrainMat = np.zeros((sample_size, len(spikeTrain)))
    for i in tqdm(range(sample_size)):
        surrogate = getSurrogate(spikeTrain, L, R, initDist, tDistMatrices)
        spikeTrainMat[i, :] = surrogate
    return spikeTrainMat

#%%
L = 50
R = 10
fRate = 10#40
Size = 1000#100
spikeData = getSpikeData(Size, fRate)
spikeTrain = getSpikeTrain(spikeData)
N = len(spikeTrain)
initDist = getInitDist(L)
tDistMatrices = getTransitionMatrices(L, N)
sampled_spiketrain = sample_spiketrain(L, R, spikeTrain, initDist, tDistMatrices, 1000)
# %%
################ raster plot
num = 50
colors = ['r'] + [u'#1f77b4'] * num
fig = plt.figure(figsize=(6, 4))
# plt.eventplot(spikeTrain, colors='b', lineoffsets=1, linewidths=1, linelengths=1)
plt.eventplot(np.concatenate((spikeTrain[None, :], sampled_spiketrain[:num, :]), axis=0), colors=colors, lineoffsets=1, linewidths=1, linelengths=1)
plt.axis('off')
plt.gca().invert_yaxis()
Gamma = getGamma(L, R, spikeTrain)
# plt.vlines(np.concatenate((np.min(Gamma, axis=1), np.max(Gamma, axis=1))), ymin=0, ymax=num+1, colors='k', linewidth=0.2, linestyles='dashed')
plt.tight_layout()
plt.show()
# plt.savefig('./plots/raster.jpg')
# plt.savefig('./plots/raster.pdf', transparent=True)

# %%
Palette = np.array(['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
num = 10
L = 100
R_list = [1, 20, 50, 100, 500, 1000]
N = len(spikeTrain)
initDist = getInitDist(L)
tDistMatrices = getTransitionMatrices(L, N)
colors = np.concatenate((np.array(['r']), np.repeat(Palette[:len(R_list)], num)))
all_spiketrain = spikeTrain[None, :]
for R in R_list:
    print(R)
    sampled_spiketrain = sample_spiketrain(L, R, spikeTrain, initDist, tDistMatrices, 1000)
    all_spiketrain = np.concatenate((all_spiketrain, sampled_spiketrain[:num, :]), axis=0)
################ raster plot
#%%
text_pos = np.arange(8, 68, 10)
fig = plt.figure(figsize=(10, 7))
# plt.eventplot(spikeTrain, colors='b', lineoffsets=1, linewidths=1, linelengths=1)
plt.eventplot(all_spiketrain, colors=colors, lineoffsets=1, linewidths=1, linelengths=1)
for ind, t_pos in enumerate(text_pos):
  plt.text(-80, t_pos, 'R={}'.format(R_list[ind]), size=10, color=Palette[ind], weight='bold')
plt.axis('off')
plt.gca().invert_yaxis()
Gamma = getGamma(L, R, spikeTrain)
# plt.vlines(np.concatenate((np.min(Gamma, axis=1), np.max(Gamma, axis=1))), ymin=0, ymax=num+1, colors='k', linewidth=0.2, linestyles='dashed')
plt.tight_layout()
plt.savefig('../plots/sampled_spiketrain_L{}.jpg'.format(L))
# plt.show()
# %%
def load_npz(filename):
    """
    load npz files with sparse matrix and dimension
    output dense matrix with the correct dim
    """
    npzfile = np.load(filename, allow_pickle=True) 
    sparse_matrix = npzfile['arr_0'][0]
    ndim=npzfile['arr_0'][1]

    new_matrix_2d = np.array(sparse_matrix.todense())
    new_matrix = new_matrix_2d
    # new_matrix = new_matrix_2d.reshape(ndim)
    return new_matrix
import os
import time
min_len, min_num = (10000, 29)
min_spikes = min_len * 0.002 # 2 Hz
# measure = 'pearson'
# measure = 'cosine'
# measure = 'correlation'
# measure = 'MI'
measure = 'xcorr'
# measure = 'causality'
directory = '../data/ecephys_cache_dir/sessions/spiking_sequence/'
files = os.listdir(directory)
files.sort(key=lambda x:int(x[:9]))
path = os.path.join(directory.replace('spiking_sequence', 'adj_mat_{}'.format(measure)))
if not os.path.exists(path):
  os.makedirs(path)
num_sample = 20
portions = np.arange(0.05, 1.05, 0.05)
# %%
start_time = time.time()
for file in files:
  if file.endswith(".npz"):
    start_time_mouse = time.time()
    print(file)
    mouseID = file.replace('.npz', '').split('_')[0]
    stimulus = file.replace('.npz', '').replace(mouseID + '_', '')
    break
sequences = load_npz(os.path.join(directory, file))
sequences = sequences[np.count_nonzero(sequences[:, :min_len], axis=1) > 80, :min_len]
num_nodes = sequences.shape[0]

# %%
spikeTrain = getSpikeTrain(sequences[0, :])
N = len(spikeTrain)
initDist = getInitDist(L)
tDistMatrices = getTransitionMatrices(L, N)
sample = sample_spiketrain(L, R, spikeTrain, initDist, tDistMatrices, sample_size=1000)
# %%
