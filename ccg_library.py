#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ccg.py --- Python library for jitter-corrected ccg method with "sharp peak/interval" detection.
# Author : Disheng Tang
# Date : 2024-08-20
# Homepage : https://dishengtang.github.io/

import itertools
import numpy as np
from scipy import signal
from tqdm import tqdm
from numpy.lib.stride_tricks import as_strided

######################################## jitter class ########################################
class pattern_jitter():
	"""
	Pattern jitter algorithm for generating synthetic spike trains by preserving recent spiking history of all spikes.
	Harrison, M. T., & Geman, S. (2009). A rate and history-preserving resampling algorithm for neural spike trains. Neural Computation, 21(5), 1244-1258.
	"""
	def __init__(self, num_sample, spikeTrain, L, R=None, memory=True):
		"""
		Initializes the pattern_jitter class with given parameters.

		Args:
			num_sample (int): Number of synthetic spike train samples to generate.
			spikeTrain (array-like): Input spike train spikeTrain, should be 1d (T,) or 2d (N,T) array.
			L (int): Length of the jitter window.
			R (int, optional): Memory parameter indicating the maximum allowable interval to consider recent history. Required if memory=True.
			memory (bool): If True, uses the history-preserving method; otherwise, uses the simple spike jitter method.
		"""
		super(pattern_jitter,self).__init__()
		self.num_sample = num_sample
		self.spikeTrain = np.array(spikeTrain)
		if len(self.spikeTrain.shape) > 1:
			self.N, self.T = self.spikeTrain.shape
		else:
			self.T = len(self.spikeTrain)
			self.N = None
		self.L = L
		self.memory = memory
		if self.memory:
			assert R is not None, 'R needs to be given if memory is True!'
			self.R = R
		else:
			self.R = None

	def spike_time2train(self, spikeTime):
		"""
		Converts spike time information to a binary spike train format.

		Args:
			spikeTime (array-like): Spike time information, should be 1d (T,) or 2d (N,T) array.

		Returns:
			np.ndarray: Binary spike train.
		"""
		if len(spikeTime.shape) == 1:
			spikeTrain = np.zeros(self.T)
			spikeTrain[spikeTime.astype(int)] = 1
		else:
			spikeTrain = np.zeros((spikeTime.shape[0], self.T))
			spikeTrain[np.repeat(np.arange(spikeTime.shape[0]), spikeTime.shape[1]), spikeTime.ravel().astype(int)] = 1
		return spikeTrain

	def spike_train2time(self, spikeTrain):
		"""
		Converts binary spike train data back to spike time information.

		Args:
			spikeData (array-like): Binary spike train.

		Returns:
			np.ndarray: Spike time information.
		"""
		if len(spikeTrain.shape) == 1:
			spikeTime = np.squeeze(np.where(spikeTrain>0)).ravel()
		else:
			spikeTime = np.zeros((spikeTrain.shape[0], len(np.where(spikeTrain[0, :]>0)[0])))
			for i in range(spikeTrain.shape[0]):
				spikeTime[i, :] = np.squeeze(np.where(spikeTrain[i, :]>0)).ravel()
		return spikeTime

	def getInitDist(self):
		"""
		Generates an initial distribution for the jitter algorithm.

		Returns:
			np.ndarray: Normalized initial distribution.
		"""
		initDist = np.random.rand(self.L)
		return initDist/initDist.sum()

	def getTransitionMatrices(self, num_spike):
		"""
		Generates transition matrices for the jitter algorithm.

		Args:
			num_spike (int): Number of spikes in the train.

		Returns:
			np.ndarray: Transition matrices.
		"""
		tDistMatrices = np.zeros((num_spike - 1, self.L, self.L))
		for i in range(tDistMatrices.shape[0]):
			matrix = np.random.rand(self.L, self.L)
			stochMatrix = matrix/matrix.sum(axis=1)[:,None]
			tDistMatrices[i, :, :] = stochMatrix.astype('f')
		return tDistMatrices

	def getX1(self, jitter_window, initDist):
		"""
		Selects the initial spike location based on the initial distribution.

		Args:
			jitter_window (array-like): The jitter window.
			initDist (array-like): Initial distribution.

		Returns:
			int: Initial spike location.
		"""
		randX = np.random.random()
		ind = np.where(randX <= np.cumsum(initDist))[0][0]
		return jitter_window[0][ind]

	def initializeX(self, initX, Prob):
		"""
		Initializes the spike location with given probabilities.

		Args:
			initX (int): Initial spike location.
			Prob (array-like): Probability distribution.

		Returns:
			int: Initialized spike location.
		"""
		return initX + np.sum(Prob == 0)

	def getOmega(self, spikeTime):
		"""
		Generates the Omega set for the jitter algorithm without memory.

		Args:
			spikeTime (array-like): Spike time information.

		Returns:
			list: Omega set.
		"""
		Omega = []
		n = spikeTime.size
		for i in range(n):
			temp = spikeTime[i] - np.ceil(self.L/2) + 1
			temp = max(0, temp)
			temp = min(temp, self.T - self.L)
			Omega.append(np.arange(temp, temp + self.L, 1))
		return Omega

	def getGamma(self, spikeTime):
		"""
		Generates the Gamma set for the jitter algorithm with memory.

		Args:
			spikeTime (array-like): Spike time information.

		Returns:
			list: Gamma set.
		"""
		Gamma = []
		ks = [] # list of k_d
		ks.append(0)
		n = spikeTime.size
		temp = int(spikeTime[ks[-1]]/self.L)*self.L
		temp = max(0, temp)
		temp = min(temp, self.T - self.L)
		Gamma.append(np.arange(temp, temp + self.L, 1))
		for i in range(1, n):
			if spikeTime[i] - spikeTime[i-1] > self.R:
				ks.append(i)
			temp = int(spikeTime[ks[-1]]/self.L)*self.L+spikeTime[i]-spikeTime[ks[-1]]
			temp = max(0, temp)
			temp = min(temp, self.T - self.L)
			Gamma.append(np.arange(temp, temp + self.L, 1))
		return Gamma

	def getSurrogate(self, spikeTime, initDist, tDistMatrices):
		"""
		Generates a surrogate spike train based on the jitter algorithm.

		Args:
			spikeTime (array-like): Original spike time.
			initDist (array-like): Initial distribution.
			tDistMatrices (array-like): Transition matrices.

		Returns:
			list: Surrogate spike train.
		"""
		surrogate = []
		if self.memory:
			jitter_window = self.getGamma(spikeTime)
		else:
			jitter_window = self.getOmega(spikeTime)
		givenX = self.getX1(jitter_window, initDist)
		surrogate.append(givenX)
		for i, row in enumerate(tDistMatrices):
			if self.memory and spikeTime[i+1] - spikeTime[i] <= self.R:
					givenX = surrogate[-1] + spikeTime[i+1] - spikeTime[i]
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

	def sample_spiketime(self, spikeTime, initDist, tDistMatrices):
		"""
		Generates multiple samples of surrogate spike times.

		Args:
			spikeTime (array-like): Original spike time.
			initDist (array-like): Initial distribution.
			tDistMatrices (array-like): Transition matrices.

		Returns:
			np.ndarray: Matrix of sampled surrogate spike times.
		"""
		sampled_spiketimes = np.zeros((self.num_sample, spikeTime.size))
		for i in tqdm(range(self.num_sample), disable=True):
			surrogate = self.getSurrogate(spikeTime, initDist, tDistMatrices)
			sampled_spiketimes[i, :] = surrogate
		return sampled_spiketimes

	def jitter(self):
		"""
		Main method to perform the jitter operation on the input sequences.

		Returns:
			np.ndarray: Jittered sequences.
		"""
		if self.N is not None:
			jittered_spikeTrain = np.zeros((self.num_sample, self.N, self.T))
			for n in range(self.N):
				spikeTime = self.spike_train2time(self.spikeTrain[n, :])
				num_spike = spikeTime.size
				if num_spike:
					initDist = self.getInitDist()
					tDistMatrices = self.getTransitionMatrices(num_spike)
					sampled_spiketimes = self.sample_spiketime(spikeTime, initDist, tDistMatrices)
					jittered_spikeTrain[:, n, :] = self.spike_time2train(sampled_spiketimes)
				else:
					jittered_spikeTrain[:, n, :] = np.zeros((self.num_sample, self.T))
		else:
			spikeTime = self.spike_train2time(self.spikeTrain)
			num_spike = spikeTime.size
			initDist = self.getInitDist()
			tDistMatrices = self.getTransitionMatrices(num_spike)
			sampled_spiketimes = self.sample_spiketime(spikeTime, initDist, tDistMatrices)
			jittered_spikeTrain = self.spike_time2train(sampled_spiketimes).squeeze()
		return jittered_spikeTrain

######################################## CCG class ########################################
class CCG:
	def __init__(self, num_jitter=100, L=25, window=100, memory=False, use_parallel=True, num_cores=14):
		"""
		Initializes the CCG class with options for parallel computing.

		Args:
			num_jitter (int): Number of jittered samples to generate. Default is 100.
			L (int): Window size for the pattern jitter method. Default is 25.
			window (int): The size of the window (in time bins) for calculating the CCG.
			memory (bool): If True, uses the pattern jitter method; otherwise, uses the simple spike jitter method.
			use_parallel (bool): Whether to use parallel computing. Default is True.
			num_cores (int): Number of CPU cores to use for parallel computing. Default is 14.
		"""
		self.num_jitter = num_jitter
		self.L = L
		self.window = window
		self.memory = memory
		self.use_parallel = use_parallel
		self.num_cores = num_cores
		if self.use_parallel:
			from joblib import Parallel, delayed
			self.Parallel = Parallel
			self.delayed = delayed

	def calculate_ccg_pair_single_trial(self, padded_st1, padded_st2, firing_rates, ind_A, ind_B, T):
		"""
		Calculates the cross-correlogram (CCG) between neuron A and neuron B over a specified time window.
		
		Args:
			padded_st1 (np.ndarray): The first matrix of spike trains, padded on both sides with zeros.
			padded_st2 (np.ndarray): The second matrix of spike trains, padded only on the right with zeros.
			firing_rates (np.ndarray): The firing rates of each neuron (spikes per second).
			ind_A (int): The index of the neuron A.
			ind_B (int): The index of the neuron B.
			T (int): The number of time bins in the original (unpadded) spike trains.
				
		Returns:
			np.ndarray: The CCG for the specified pair of neurons over the window size.
		"""
		px, py = padded_st1[ind_A, :], padded_st2[ind_B, :]
		shifted = as_strided(px[self.window:], shape=(self.window + 1, T + self.window),
										strides=(-px.strides[0], px.strides[0]))
		return (shifted @ py) / ((T - np.arange(self.window + 1)) / 1000 * np.sqrt(firing_rates[ind_A] * firing_rates[ind_B]))

	def calculate_all_ccgs_single_trial(self, spikeTrain_single_trial, disable=True):
		"""
		Calculates CCGs for all pairs of neurons for a single trial, with optional parallel computing for accelearation.
		
		Args:
			spikeTrain_single_trial (np.ndarray): The spike train matrix for a single trial (N x T) where N is the number of neurons and T is the number of time bins.
			disable (bool): Disable the progress bar. Default is True.
				
		Returns:
			ccgs (np.ndarray): A matrix (N x N x window+1) containing the CCGs for all pairs of neurons.
		"""
		N, T = spikeTrain_single_trial.shape
		ccgs = np.zeros((N, N, self.window + 1))
		# Make the diagonal elements of the matrix NaN
		mask = np.eye(N, dtype=bool)[:, :, None]
		mask = np.broadcast_to(mask, ccgs.shape)
		ccgs[mask] = np.nan
		# ccgs[:] = np.nan
		firing_rates = np.count_nonzero(spikeTrain_single_trial, axis=1) / (spikeTrain_single_trial.shape[1] / 1000)  # in Hz,, default time bin is 1ms, change this based on time bin size
		# Pad the matrices for CCG calculation
		padded_st1 = np.concatenate((np.zeros((N, self.window)), spikeTrain_single_trial.conj(), np.zeros((N, self.window))), axis=1)
		padded_st2 = np.concatenate((spikeTrain_single_trial.conj(), np.zeros((N, self.window))), axis=1)
		# Ensure both neurons have non-zero firing rates
		valid_inds = np.where(firing_rates > 0)[0]
		total_list = list(itertools.permutations(valid_inds, 2))
		for ind_A, ind_B in tqdm(total_list, total=len(total_list), disable=disable):
			ccgs[ind_A, ind_B, :] = self.calculate_ccg_pair_single_trial(padded_st1, padded_st2, firing_rates, ind_A, ind_B, T)
		return ccgs

	def process_trial(self, trial_ind, spikeTrain, pj):
		spikeTrain_single_trial = spikeTrain[:, trial_ind, :]
		ccg_trial = self.calculate_all_ccgs_single_trial(spikeTrain_single_trial, disable=True)
		pj.spikeTrain = spikeTrain_single_trial
		sampled_matrix = pj.jitter()  # num_jitter x N x T
		ccg_jittered_trial = np.zeros_like(ccg_trial)
		for jitter_ind in range(self.num_jitter):
			ccg_jittered_trial += self.calculate_all_ccgs_single_trial(sampled_matrix[jitter_ind, :, :], disable=True)
		return ccg_trial, ccg_jittered_trial / self.num_jitter

	def calculate_mean_ccg_corrected(self, spikeTrain, disable=True):
		"""
		Calculates the mean CCG with jitter correction using the pattern jitter method, considering only causal correlations.

		Args:
			spikeTrain (np.ndarray): The spike train tensor (num_neuron x num_trial x T) where T is the number of time bins.
			disable (bool): Disable the progress bar. Default is True.

		Returns:
			np.ndarray: The corrected CCG (N x N x window+1) after subtracting the jittered CCGs.
		"""
		num_neuron, num_trial, T = spikeTrain.shape
		assert T > self.window, "Please reset the CCG window size to be smaller than the number of bins in the spike train."
		ccgs = np.zeros((num_neuron, num_neuron, self.window + 1))
		ccg_jittered = np.zeros((num_neuron, num_neuron, self.window + 1))
		# Initialize the pattern jitter method
		pj = pattern_jitter(num_sample=self.num_jitter, spikeTrain=spikeTrain[:, 0, :], L=self.L, memory=self.memory)
		if self.use_parallel:
			# with parallel_backend('multiprocessing'):
			result = self.Parallel(n_jobs=self.num_cores)(
				self.delayed(self.process_trial)(trial_ind, spikeTrain, pj)
				for trial_ind in tqdm(range(num_trial), disable=disable)
			)
		else:
			result = [self.process_trial(trial_ind, spikeTrain, pj) for trial_ind in tqdm(range(num_trial), disable=disable)]
		for ccg_trial, ccg_jittered_trial in result:
			ccgs += ccg_trial
			ccg_jittered += ccg_jittered_trial
		ccgs = ccgs / num_trial
		ccg_jittered = ccg_jittered / num_trial
		ccg_jitter_corrected = ccgs - ccg_jittered
		return ccg_jitter_corrected

######################################## model class ########################################
# Izhikevich neuron model
# Use variable delay for different connections
def generate_spikes_Izhikevich_variable_delay(ground_truth, delay_matrix, n_trial=40, T=250, current_value=112):
	'''
	ground_truth is the ground truth connections with values being connection strenths
	n_trial is the number of trials
	T is the trial length (ms)
	variable delay for different connections (ms)
	the shape generated is neuron * trial * bin
	'''
	sources, targets = np.nonzero(delay_matrix)
	n_neuron = ground_truth.shape[0]
	# Input parameters
	C = 100
	vr = -60 # resting membrane potential
	vt = -40 #  instantaneous threshold potential
	k = 0.7  # Parameters used for RS
	a = 0.03
	# 《Dynamical Systems in Neuroscience》by Eugene M. Izhikevich
	# when b = 0, quadratic integrate-and-fire neuron with adaptation
	# when b < 0, quadratic integrate-and-fire neuron with a passive dendritic compartment
	# when b > 0, a novel class of spiking models
	b = -2
	c = -50 # reset membrane potential v = c if v >= v_peak
	d = 100  # Neocortical pyramidal neurons u = u + d if v >= v_peak
	vpeak = 35  # Spike cutoff
	tau = 1  # Time span and step (ms)
	n_bin = int(T / tau)  # Number of simulation steps
	firings = []  # Spike timings
	#
	noise_amp = 2 # 0.25
	for tr in range(n_trial):
		v = np.full((n_neuron, n_bin), vr, dtype=float)
		u = np.zeros((n_neuron, n_bin))
		I = np.zeros((n_neuron, n_bin))
		# current_value = 110 # for 20 neurons
		# I[0, :] = 130 * np.random.rand(n_bin)
		for neuron in range(0,n_neuron):
			I[neuron, :] = current_value * np.random.rand(n_bin)
		R = np.zeros((n_neuron, n_bin))
		for t in range(0, n_bin - 1):
			dv = tau * (
			k * (v[:, t] - vr) * (v[:, t] - vt) - u[:, t] + I[:, t] + R[:, t]
			) / C + noise_amp * np.random.randn(1)
			du = tau * a * (b * (v[:, t] - vr) - u[:, t])
			v[:, t + 1] = v[:, t] + dv
			u[:, t + 1] = u[:, t] + du
			fired = np.where(v[:, t + 1] >= vpeak)[0]
			if fired.size > 0:
				firings.extend([(t + 1, f, tr) for f in fired])
				v[fired, t] = vpeak
				v[fired, t + 1] = c
				u[fired, t + 1] = u[fired, t + 1] + d
			# Neurotransmitter only lasts for one ms
			# Variable delay for different connections
			fire_mask = np.where(sources==fired)[0]
			fired_srcs, fired_tgts = sources[fire_mask], targets[fire_mask]
			fired_delays = t + 1 + delay_matrix[fired_srcs, fired_tgts]
			time_mask = np.where(fired_delays<n_bin)[0]
			R[fired_tgts[time_mask], fired_delays[time_mask]] += ground_truth[fired_srcs, fired_tgts][time_mask]
		
	all_spiketrains = np.zeros((n_neuron, n_trial, n_bin))
	for neuron in range(n_neuron):
		for firing in firings:
			if firing[1] == neuron:
				all_spiketrains[firing[1], firing[2], firing[0]] = 1
	return all_spiketrains

######################################## test class ########################################
class SharpPeakIntervalDetection:
	"""
	Class to detect sharp peak intervals in cross-correlograms (CCGs), handling the removal
	of double-counted peaks at zero time lag.
	"""

	def __init__(self, max_duration=6, maxlag=12, n=4):
		"""
		Initializes the SharpPeakIntervalDetection class.

		Args:
			ccg_corrected (np.ndarray): The jitter-corrected cross-correlogram data.
			max_duration (int): The maximum duration to consider for sharp peak/interval detection.
			maxlag (int): The maximum lag to consider.
			n (int): The threshold of Z-score for significance detection.
		"""
		self.max_duration = max_duration
		self.maxlag = maxlag
		self.n = n

	def find_sharp_peak_interval(self, ccg_corrected, duration=6):
		"""
		Detects sharp peak/interval in the given cross-correlogram matrix.

		Args:
			ccg_corrected (np.ndarray): Cross-correlogram matrix.
			duration (int): Duration of the peak/interval detection window, duration=0 means a sharp peak.

		Returns:
			np.ndarray: Matrix of highland CCG values.
			np.ndarray: Confidence level matrix.
			np.ndarray: Offset matrix.
			np.ndarray: Index matrix indicating where significant peaks/intervals were found.
		"""
		import warnings
		warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")
		num_nodes = ccg_corrected.shape[0]
		highland_ccg = np.full((num_nodes, num_nodes), np.nan)
		offset = np.full((num_nodes, num_nodes), np.nan)
		confidence_level = np.zeros((num_nodes, num_nodes))
		
		filter = np.array([[[1]]]).repeat(duration + 1, axis=2)
		ccg_integral = signal.convolve(ccg_corrected, filter, mode='valid', method='fft')
		mu, sigma = np.nanmean(ccg_integral, -1), np.nanstd(ccg_integral, -1)
		abs_deviation = np.abs(ccg_integral[:, :, :self.maxlag - duration + 1] - mu[:, :, None])
		extreme_offset = np.argmax(abs_deviation, -1)
		ccg_mat_extreme = np.choose(extreme_offset, np.moveaxis(ccg_integral[:, :, :self.maxlag - duration + 1], -1, 0))
		pos_fold = ccg_mat_extreme > mu + self.n * sigma
		neg_fold = ccg_mat_extreme < mu - self.n * sigma
		c_level = (ccg_mat_extreme - mu) / sigma
		indx = np.logical_or(pos_fold, neg_fold)
		
		highland_ccg[indx] = ccg_mat_extreme[indx]
		confidence_level[indx] = c_level[indx]
		offset[indx] = extreme_offset[indx]
		
		# Handle double-counting of peaks at zero time lag
		pos_double_0 = (extreme_offset == 0) & (extreme_offset.T == 0) & (pos_fold == pos_fold.T) & pos_fold
		neg_double_0 = (extreme_offset == 0) & (extreme_offset.T == 0) & (neg_fold == neg_fold.T) & neg_fold
		double_0 = np.logical_or(pos_double_0, neg_double_0)
		
		if np.sum(double_0):
			extreme_offset_2nd = np.argpartition(abs_deviation, -2, axis=-1)[:, :, -2]
			ccg_mat_extreme_2nd = np.choose(extreme_offset_2nd, np.moveaxis(ccg_integral[:, :, :self.maxlag - duration + 1], -1, 0))
			c_level_2nd = (ccg_mat_extreme_2nd - mu) / sigma
			pos_remove_0 = np.logical_and(ccg_mat_extreme_2nd >= ccg_mat_extreme_2nd.T, pos_double_0)
			neg_remove_0 = np.logical_and(ccg_mat_extreme_2nd <= ccg_mat_extreme_2nd.T, neg_double_0)
			remove_0 = np.logical_or(pos_remove_0, neg_remove_0)
			highland_ccg[remove_0], confidence_level[remove_0], offset[remove_0], indx[remove_0] = np.nan, 0, np.nan, False
			pos_fold_2nd = np.logical_and(ccg_mat_extreme_2nd > mu + self.n * sigma, pos_remove_0)
			neg_fold_2nd = np.logical_and(ccg_mat_extreme_2nd < mu - self.n * sigma, neg_remove_0)
			indx_2nd = np.logical_or(pos_fold_2nd, neg_fold_2nd)
			indx_2nd = np.logical_and(indx_2nd, remove_0)
			highland_ccg[indx_2nd], confidence_level[indx_2nd], offset[indx_2nd] = ccg_mat_extreme_2nd[indx_2nd], c_level_2nd[indx_2nd], extreme_offset_2nd[indx_2nd]
			indx = np.logical_or(indx, indx_2nd)
		
		return highland_ccg, confidence_level, offset, indx

	def find_2nd_sharp_peak_interval(self, ccg_corrected, duration=6):
		"""
		Finds the second-largest sharp peak/interval for double-counted edges.

		Args:
			ccg_corrected (np.ndarray): Cross-correlogram matrix.
			duration (int): Duration of the peak/interval detection window, duration=0 means a sharp peak.

		Returns:
			np.ndarray: Matrix of the second-largest highland CCG values.
			np.ndarray: Confidence level matrix.
			np.ndarray: Offset matrix.
			np.ndarray: Index matrix indicating where the second-largest peaks/intervals were found.
		"""
		import warnings
		warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")
		num_pairs = ccg_corrected.shape[0]
		highland_ccg_2nd = np.full(num_pairs, np.nan)
		offset_2nd = np.full(num_pairs, np.nan)
		confidence_level_2nd = np.zeros(num_pairs)
		
		filter = np.array([[1]]).repeat(duration + 1, axis=1)
		ccg_integral = signal.convolve(ccg_corrected, filter, mode='valid', method='fft')
		mu, sigma = np.nanmean(ccg_integral, -1), np.nanstd(ccg_integral, -1)
		abs_deviation = np.abs(ccg_integral[:, :self.maxlag - duration + 1] - mu[:, None])
		extreme_offset_2nd = np.argpartition(abs_deviation, -2, axis=-1)[:, -2]
		ccg_mat_extreme_2nd = np.choose(extreme_offset_2nd, np.moveaxis(ccg_integral[:, :self.maxlag - duration + 1], -1, 0))
		c_level_2nd = (ccg_mat_extreme_2nd - mu) / sigma
		pos_fold_2nd = ccg_mat_extreme_2nd > mu + self.n * sigma
		neg_fold_2nd = ccg_mat_extreme_2nd < mu - self.n * sigma
		indx_2nd = np.logical_or(pos_fold_2nd, neg_fold_2nd)
		
		highland_ccg_2nd[indx_2nd], confidence_level_2nd[indx_2nd], offset_2nd[indx_2nd] = ccg_mat_extreme_2nd[indx_2nd], c_level_2nd[indx_2nd], extreme_offset_2nd[indx_2nd]

		return highland_ccg_2nd, confidence_level_2nd, offset_2nd, indx_2nd

	def get_significant_ccg(self, ccg_corrected):
		"""
		Identifies significant cross-correlograms and removes double-counted peaks at zero time lag.

		Returns:
			np.ndarray: Matrix of significant CCG values.
			np.ndarray: Confidence level matrix for significant connections.
			np.ndarray: Offset matrix for significant connections.
			np.ndarray: Duration matrix for significant connections.
		"""
		import warnings
		warnings.filterwarnings("ignore", category=RuntimeWarning, message="Use of fft convolution on input with NAN or inf results in NAN or inf output")
		warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
		warnings.filterwarnings("ignore", category=RuntimeWarning, message="Degrees of freedom <= 0 for slice")
		num_nodes = ccg_corrected.shape[0]
		significant_ccg = np.full((num_nodes, num_nodes), np.nan)
		significant_confidence = np.full((num_nodes, num_nodes), np.nan)
		significant_offset = np.full((num_nodes, num_nodes), np.nan)
		significant_duration = np.full((num_nodes, num_nodes), np.nan)
		for duration in np.arange(self.max_duration, -1, -1):
			highland_ccg, confidence_level, offset, indx = self.find_sharp_peak_interval(ccg_corrected, duration)
			mask = indx & (np.abs(np.nan_to_num(confidence_level)) > np.abs(np.nan_to_num(significant_confidence)))
			
			if np.sum(mask):
				significant_ccg[mask] = highland_ccg[mask]
				significant_confidence[mask] = confidence_level[mask]
				significant_offset[mask] = offset[mask]
				significant_duration[mask] = duration

		# Handle double-counting of peaks at zero time lag
		double_0 = (significant_offset == 0) & (significant_offset.T == 0) & \
				   (~np.isnan(significant_ccg)) & (~np.isnan(significant_ccg.T))
		
		if np.sum(double_0):
			remove_0 = (significant_duration >= significant_duration.T) & double_0
			significant_ccg[remove_0] = np.nan
			significant_confidence[remove_0] = np.nan
			significant_offset[remove_0] = np.nan
			significant_duration[remove_0] = np.nan
			
			for duration in np.arange(self.max_duration, -1, -1):
				highland_ccg_2nd, confidence_level_2nd, offset_2nd, indx_2nd = self.find_2nd_sharp_peak_interval(
					ccg_corrected[remove_0], duration)
				
				if np.sum(indx_2nd):
					significant_ccg[remove_0][indx_2nd] = highland_ccg_2nd[indx_2nd]
					significant_confidence[remove_0][indx_2nd] = confidence_level_2nd[indx_2nd]
					significant_offset[remove_0][indx_2nd] = offset_2nd[indx_2nd]
					significant_duration[remove_0][indx_2nd] = duration

		return significant_ccg, significant_confidence, significant_offset, significant_duration
