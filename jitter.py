#%%
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class pattern_jitter():
    def __init__(self, num_sample, sequences, L, R=None, memory=True):
        super(pattern_jitter,self).__init__()
        self.num_sample = num_sample
        self.sequences = np.array(sequences)
        if len(self.sequences.shape) > 1:
            self.N, self.T = self.sequences.shape
        else:
            self.T = len(self.sequences)
            self.N = None
        self.L = L
        self.memory = memory
        if self.memory:
            assert R is not None, 'R needs to be given if memory is True!'
            self.R = R
        else:
            self.R = None

    def spike_timing2train(self, spikeTrain):
        if len(spikeTrain.shape) == 1:
            spikeData = np.zeros(self.T)
            spikeData[spikeTrain.astype(int)] = 1
        else:
            spikeData = np.zeros((spikeTrain.shape[0], self.T))
            spikeData[np.repeat(np.arange(spikeTrain.shape[0]), spikeTrain.shape[1]), spikeTrain.ravel().astype(int)] = 1
        return spikeData

    def getSpikeTrain(self, spikeData):
        if len(spikeData.shape) == 1:
            spikeTrain = np.squeeze(np.where(spikeData>0)).ravel()
        else:
            spikeTrain = np.zeros((spikeData.shape[0], len(np.where(spikeData[0, :]>0)[0])))
            for i in range(spikeData.shape[0]):
                spikeTrain[i, :] = np.squeeze(np.where(spikeData[i, :]>0)).ravel()
        return spikeTrain

    def getInitDist(self):
        initDist = np.random.rand(self.L)
        return initDist/initDist.sum()

    def getTransitionMatrices(self, num_spike):
        tDistMatrices = np.zeros((num_spike - 1, self.L, self.L))
        for i in range(tDistMatrices.shape[0]):
            matrix = np.random.rand(self.L, self.L)
            stochMatrix = matrix/matrix.sum(axis=1)[:,None]
            tDistMatrices[i, :, :] = stochMatrix.astype('f')
        return tDistMatrices

    def getX1(self, jitter_window, initDist):
        
        randX = np.random.random()
        ind = np.where(randX <= np.cumsum(initDist))[0][0]
        return jitter_window[0][ind]

    def initializeX(self, initX, Prob):
        return initX + np.sum(Prob == 0)

    def getOmega(self, spikeTrain):
        Omega = []
        n = spikeTrain.size
        for i in range(n):
            temp = spikeTrain[i] - np.ceil(self.L/2) + 1
            temp = max(0, temp)
            temp = min(temp, self.T - self.L)
            Omega.append(np.arange(temp, temp + self.L, 1))
        return Omega

    def getGamma(self, spikeTrain):
        Gamma = []
        ks = [] # list of k_d
        ks.append(0)
        n = spikeTrain.size
        temp = int(spikeTrain[ks[-1]]/self.L)*self.L
        temp = max(0, temp)
        temp = min(temp, self.T - self.L)
        Gamma.append(np.arange(temp, temp + self.L, 1))
        for i in range(1, n):
            if spikeTrain[i] - spikeTrain[i-1] > self.R:
                ks.append(i)
            temp = int(spikeTrain[ks[-1]]/self.L)*self.L+spikeTrain[i]-spikeTrain[ks[-1]]
            temp = max(0, temp)
            temp = min(temp, self.T - self.L)
            Gamma.append(np.arange(temp, temp + self.L, 1))
        return Gamma

    def getSurrogate(self, spikeTrain, initDist, tDistMatrices):
        surrogate = []
        if self.memory:
            jitter_window = self.getGamma(spikeTrain)
        else:
            jitter_window = self.getOmega(spikeTrain)
        givenX = self.getX1(jitter_window, initDist)
        surrogate.append(givenX)
        for i, row in enumerate(tDistMatrices):
            if self.memory and spikeTrain[i+1] - spikeTrain[i] <= self.R:
                givenX = surrogate[-1] + spikeTrain[i+1] - spikeTrain[i]
            else:
                index = np.where(np.array(jitter_window[i]) == givenX)[0]
                p_i = np.squeeze(np.array(row[index]))
                initX = self.initializeX(jitter_window[i + 1][0], p_i)
                randX = np.random.random()
                # safe way to find the ind
                larger = np.where(randX <= np.cumsum(p_i))[0]
                if larger.shape[0]:
                    ind = larger[0]
                else:
                    ind = len(p_i) - 1
                givenX = initX + np.sum(p_i[:ind]!=0)
            givenX = min(self.T - 1, givenX) # possible same location
            if givenX in surrogate:
                locs = jitter_window[i + 1]
                available_locs = [loc for loc in locs if loc not in surrogate]
                givenX = np.random.choice(available_locs)
            surrogate.append(givenX)
        return surrogate

    def sample_spiketrain(self, spikeTrain, initDist, tDistMatrices):
        spikeTrainMat = np.zeros((self.num_sample, spikeTrain.size))
        for i in tqdm(range(self.num_sample), disable=True):
            surrogate = self.getSurrogate(spikeTrain, initDist, tDistMatrices)
            spikeTrainMat[i, :] = surrogate
        return spikeTrainMat

    def jitter(self):
        # num_sample x N x T
        if self.N is not None:
            jittered_seq = np.zeros((self.num_sample, self.N, self.T))
            for n in range(self.N):
                spikeTrain = self.getSpikeTrain(self.sequences[n, :])
                num_spike = spikeTrain.size
                if num_spike:
                    initDist = self.getInitDist()
                    tDistMatrices = self.getTransitionMatrices(num_spike)
                    sampled_spiketrain = self.sample_spiketrain(spikeTrain, initDist, tDistMatrices)
                    jittered_seq[:, n, :] = self.spike_timing2train(sampled_spiketrain)
                else:
                    jittered_seq[:, n, :] = np.zeros((self.T, self.num_sample))
        else:
            spikeTrain = self.getSpikeTrain(self.sequences)
            num_spike = spikeTrain.size
            initDist = self.getInitDist()
            tDistMatrices = self.getTransitionMatrices(num_spike)
            sampled_spiketrain = self.sample_spiketrain(spikeTrain, initDist, tDistMatrices)
            jittered_seq = self.spike_timing2train(sampled_spiketrain).squeeze()
        return jittered_seq

def getSpikeData(T, fireRate):
    spikeData = np.zeros(T)
    binprob = (1./T)*fireRate
    rands = np.random.uniform(size=T)
    spikeData[rands <= binprob] = 1
    return spikeData

def getSpikeTrain(spikeData):
    spikeTrain = np.squeeze(np.where(spikeData>0))
    return spikeTrain

def spike_timing2train(T, spikeTrain):
    if len(spikeTrain.shape) == 1:
        spikeData = np.zeros(T)
        spikeData[spikeTrain.astype(int)] = 1
    else:
        spikeData = np.zeros((spikeTrain.shape[0], T))
        spikeData[np.repeat(np.arange(spikeTrain.shape[0]), spikeTrain.shape[1]), spikeTrain.ravel().astype(int)] = 1
    return spikeData

L = 50
R = 10
fRate = 10#40
Size = 1000#100
spikeData = getSpikeData(Size, fRate)
spikeTrain = getSpikeTrain(spikeData)
sequences = spike_timing2train(Size, spikeTrain)
pj = pattern_jitter(num_sample=10, sequences=sequences, L=L, R=R, memory=False)
jittered_seq = pj.jitter()
# %%
Palette = np.array(['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
num = 10
L = 50
R_list = [1, 20, 50, 100, 500, 1000]
colors = np.concatenate((np.array(['r']), np.repeat(Palette[:len(R_list)], num)))
all_spiketrain = spikeTrain[None, :]
pj = pattern_jitter(num_sample=num, sequences=sequences, L=L, R=1, memory=True)
jittered_seq = pj.jitter()
for R in R_list:
    print(R)
    pj.R = R
    all_spiketrain = np.concatenate((all_spiketrain, pj.getSpikeTrain(pj.jitter())[:num, :]), axis=0)
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
plt.tight_layout()
# plt.savefig('../plots/sampled_spiketrain_L{}.jpg'.format(L))
plt.show()
# %%
Palette = np.array(['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
num = 10
L_list = [2, 10, 20, 30, 40, 50, 60, 70, 80]
colors = np.concatenate((np.array(['r']), np.repeat(Palette[:len(L_list)], num)))
all_spiketrain = spikeTrain[None, :]
pj = pattern_jitter(num_sample=num, sequences=sequences, L=L, memory=False)
jittered_seq = pj.jitter()
for L in L_list:
    print(L)
    pj.L = L
    all_spiketrain = np.concatenate((all_spiketrain, pj.getSpikeTrain(pj.jitter())[:num, :]), axis=0)
text_pos = np.arange(8, 98, 10)
fig = plt.figure(figsize=(10, 7))
# plt.eventplot(spikeTrain, colors='b', lineoffsets=1, linewidths=1, linelengths=1)
plt.eventplot(all_spiketrain, colors=colors, lineoffsets=1, linewidths=1, linelengths=1)
for ind, t_pos in enumerate(text_pos):
  plt.text(-80, t_pos, 'L={}'.format(L_list[ind]), size=10, color=Palette[ind], weight='bold')
plt.axis('off')
plt.gca().invert_yaxis()
plt.tight_layout()
# plt.savefig('../plots/sampled_spiketrain_L{}.jpg'.format(L))
plt.show()
