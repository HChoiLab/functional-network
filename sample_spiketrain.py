#%%
import numpy as np
import matplotlib.pyplot as plt

from ClosedSynchrony import *
from PatternJitter import *
from TransitMatrix import *
from Data import *
from tqdm import tqdm

# x_tilde = [10, 15, 22, 29, 34, 40, 45, 51]
# L = 5
# R = 4
Ref = [8, 13, 19, 28, 34, 42, 44, 49]
Ref_02 = [10, 14, 17]
Ref_03 = [10, 14, 17, 20]


def getX1(Dist, L, R, ObsX):
     # in [0,1)
    # print('Transition Dist.:', dist)
    # print('Random X:', randX)
    up = 0
    sum = 0
    length = 0
    hisLen = 0

    dist = []
    obsTar = []

    length = L
    hisLen = R
    dist = Dist
    obsTar = ObsX

    # Omega = getOmega(length, obsTar)
    Omega = getGamma(R, L, obsTar)
    result = Omega[0][up]
    randX = np.random.random()

    for k in dist:
        # print('P[',i,']:', k)
        sum += k
        if randX <= sum:
            # print('Sum:', sum)
            return result
        up += 1
        result = Omega[0][up]

def initializeX(initX, Prob):

    init_x = initX
    prob = Prob
    m = len(prob)
    for k in range(m):
        if prob[k] == 0:
            init_x += 1

    return init_x

# getSpikeTrain(ObsTar, length, hisLen, initD, tDistMat)

def getSurrogate(spikeTrain, L, R, initDist, tDistMatrices):

    # print('////**** Simulation is starting. ****////')

    chain = 1
    surrogate = []

    length = L
    hisLen = R
    # Omega = getOmega(length, spikeTrain)
    Omega = getGamma(R, length, spikeTrain)
    ks = [] # list of kd
    ks.append(0)
    x1 = getX1(initDist, length, hisLen, spikeTrain)
    givenX = x1
    surrogate.append(x1)

    for i, row in enumerate(tDistMatrices):
        if spikeTrain[i+1] - spikeTrain[i] <= R:
            X = surrogate[-1] + spikeTrain[i+1] - spikeTrain[i]
            surrogate.append(X)
            givenX = X
            chain += 1
        else:
            sum = 0
            randX = 0
            index = np.where(np.array(Omega[i]) == givenX)[0]
            p_i = np.squeeze(np.array(row[index]))
            initX = initializeX(Omega[chain][0], p_i)
            randX = np.random.random()
            m = len(p_i)
            for j in range(m):
                if p_i[j] != 0:
                    sum += p_i[j]
                    if randX <= sum:
                        surrogate.append(initX)
                        givenX = initX
                        chain += 1
                        break
                    initX += 1
                else:
                    j += 1
        
    # print('////**** Simulation is done. ****////', '\n')
    return surrogate

def getSpikeTrainMat(L, R, spikeTrain, initDist, tDistMatrices, N):

    length = L
    hisLen = R
    spikeTrainMat = []
    for i in tqdm(range(N)):
        # print('[[[[[[[Spike Train Index: ', i,']]]]]]]')
        surrogate = getSurrogate(spikeTrain, length, hisLen, initDist, tDistMatrices)
        spikeTrainMat.append(surrogate)
    Tmat = np.array(spikeTrainMat)
    return Tmat

def getAmountSync(Reference, Target):
    s = 0
    S = []
    Ref = []
    Tmat = []
    ref = Reference
    Tmat = Target
    for j, Tj in enumerate(Tmat):
        # Check how many elements are equal in two arrays (R, T)
        # print('Tj: ', Tj)
        s = np.sum(ref == np.array(Tj))
        # print('Coincidence: ', s)
        S.append(s)
        # print('# Sync: ', s)
    return S

L = 50
R = 10
fRate = 20#40
Size = 1000#100
spikeData = getSpikeData(Size, fRate)
spikeTrain = getSpikeTrain(spikeData)

# print('Spike Data: ')
# print(spikeData)

N = len(spikeTrain)
initDist = getInitDist(L)
tDistMatrices = getTransitionMatrices(L, N)
ref = getReference(Size, L, N)

# print('Initial Distribution: ')
# print(initDist)
# print('Transition Matrices: ')
# print(tDistMatrices)

################################################################
# Compute the Synchrony Distribution by Monte Carlo resamples
################################################################
Tmat = getSpikeTrainMat(L, R, spikeTrain, initDist, tDistMatrices, 1000)
# print('Spike Trains: ')
# print(Tmat)

# %%
################ raster plot
num = 50
colors = ['r'] + [u'#1f77b4'] * num
fig = plt.figure(figsize=(6, 4))
# plt.eventplot(spikeTrain, colors='b', lineoffsets=1, linewidths=1, linelengths=1)
plt.eventplot(np.concatenate((spikeTrain[None, :], Tmat[:num, :]), axis=0), colors=colors, lineoffsets=1, linewidths=1, linelengths=1)
plt.axis('off')
plt.gca().invert_yaxis()
Omega = getGamma(R, L, spikeTrain)
# plt.vlines(np.concatenate((np.min(Omega, axis=1), np.max(Omega, axis=1))), ymin=0, ymax=num+1, colors='k', linewidth=0.2, linestyles='dashed')
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
    Tmat = getSpikeTrainMat(L, R, spikeTrain, initDist, tDistMatrices, 1000)
    all_spiketrain = np.concatenate((all_spiketrain, Tmat[:num, :]), axis=0)
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
Omega = getGamma(R, L, spikeTrain)
# plt.vlines(np.concatenate((np.min(Omega, axis=1), np.max(Omega, axis=1))), ymin=0, ymax=num+1, colors='k', linewidth=0.2, linestyles='dashed')
plt.tight_layout()
plt.savefig('../plots/sampled_spiketrain_L{}.jpg'.format(L))
# plt.show()
# %%
