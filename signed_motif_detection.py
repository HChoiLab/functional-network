#!/usr/bin/env python
# -*- coding: utf-8 -*-

# signed_motif_detection.py --- Python library for identifying signed network motifs based on various reference models.
# Author : Disheng Tang
# Date : 2024-05-10
# Homepage : https://dishengtang.github.io/

import os
import numpy as np
import networkx as nx
import pandas as pd
import pickle
import itertools
from tqdm import tqdm
import networkx.algorithms.isomorphism as iso
import matplotlib.pyplot as plt
from joblib import Parallel, delayed # keep this line if use parallel generation of random graphs, otherwise comment this line

def safe_division(n, d):
	return n / d if d else 0
	
def get_lcc(G):
	if nx.is_directed(G):
		Gcc = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
	else:
		Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
	return G.subgraph(Gcc[0])

# Functions for loading example graphs
def load_npz_3d(filename):
	"""
	load npz files with sparse matrix and dimension
	output dense matrix with the correct dim
	"""
	npzfile = np.load(filename, allow_pickle=True) 
	sparse_matrix = npzfile['arr_0'][0]
	ndim=npzfile['arr_0'][1]

	new_matrix_2d = np.array(sparse_matrix.todense())
	new_matrix = new_matrix_2d.reshape(ndim)
	return new_matrix

# build a graph from an adjacency matrix without changing the IDs of the nodes
def simple_mat2graph(adj_mat, confidence_level, cc=False, weight=True):
	if not weight:
		adj_mat[adj_mat.nonzero()] = 1
	G = nx.from_numpy_array(adj_mat, create_using=nx.DiGraph) # same as from_numpy_matrix
	nodes = sorted(G.nodes())
	cl = {(nodes[i],nodes[j]):confidence_level[i,j] for i,j in zip(*np.where(~np.isnan(confidence_level)))}
	nx.set_edge_attributes(G, cl, 'confidence')
	if cc: # extract the largest (strongly) connected components
		if np.allclose(adj_mat, adj_mat.T, rtol=1e-05, atol=1e-08): # if the matrix is symmetric, which means undirected graph
			largest_cc = max(nx.connected_components(G), key=len)
		else:
			largest_cc = max(nx.strongly_connected_components(G), key=len)
		G = nx.subgraph(G, largest_cc)
	return G

def load_graphs(directory, session, stimulus, cc=False, weight=True):
	file = f'{session}_{stimulus}.npz'
	adj_mat = load_npz_3d(os.path.join(directory, file))
	confidence_level = load_npz_3d(os.path.join(directory, file.replace('.npz', '_confidence.npz')))
	G = simple_mat2graph(adj_mat=np.nan_to_num(adj_mat), confidence_level=confidence_level, cc=cc, weight=weight)
	return G

################### an efficient way of finding motifs, adapted from networkx.algorithms.triads
def _tricode(G, v, u, w):
	"""Returns the integer code of the given triad.

	This is some fancy magic that comes from Batagelj and Mrvar's paper. It
	treats each edge joining a pair of `v`, `u`, and `w` as a bit in
	the binary representation of an integer.

	"""
	combos = ((v, u, 1), (u, v, 2), (v, w, 4), (w, v, 8), (u, w, 16),
						(w, u, 32))
	return sum(x for u, v, x in combos if v in G[u])

def find_triads(G):
	#: The integer codes representing each type of triad.
	#: Triads that are the same up to symmetry have the same code.
	# a total of 2^6 = 64 connectivities, but many of them are isomorphic
	TRICODES = (1, 2, 2, 3, 2, 4, 6, 8, 2, 6, 5, 7, 3, 8, 7, 11, 2, 6, 4, 8, 5, 9,
							9, 13, 6, 10, 9, 14, 7, 14, 12, 15, 2, 5, 6, 7, 6, 9, 10, 14, 4, 9,
							9, 12, 8, 13, 14, 15, 3, 7, 8, 11, 7, 12, 14, 15, 8, 14, 13, 15,
							11, 15, 15, 16)
	#: The names of each type of triad. The order of the elements is
	#: important: it corresponds to the tricodes given in :data:`TRICODES`.
	TRIAD_NAMES = ('003', '012', '102', '021D', '021U', '021C', '111D', '111U',
								'030T', '030C', '201', '120D', '120U', '120C', '210', '300')
	#: A dictionary mapping triad code to triad name.
	TRICODE_TO_NAME = {i: TRIAD_NAMES[code - 1] for i, code in enumerate(TRICODES)}
	triad_nodes = {name: set([]) for name in TRIAD_NAMES}
	m = {v: i for i, v in enumerate(G)}
	for v in G:
		vnbrs = set(G.pred[v]) | set(G.succ[v])
		for u in vnbrs:
			if m[u] > m[v]:
				unbrs = set(G.pred[u]) | set(G.succ[u])
				neighbors = (vnbrs | unbrs) - {u, v}
				for w in neighbors:
					if m[u] < m[w] or (m[v] < m[w] < m[u] and
														v not in G.pred[w] and
														v not in G.succ[w]):
							code = _tricode(G, v, u, w)
							triad_nodes[TRICODE_TO_NAME[code]].add(
									tuple(sorted([u, v, w])))
	for triad_type in triad_nodes:
		if len(triad_nodes[triad_type]):
			G_list = []
			for triad in triad_nodes[triad_type]:
				G_list.append(G.subgraph(triad))
			triad_nodes[triad_type] = G_list
	return triad_nodes
	
# get the standard edges of a motif type
def find_all_unique_smotifs(edges, weight='weight'):
	em = iso.numerical_edge_match(weight, 1)
	G0 = nx.DiGraph()
	all_ws = list(itertools.product([1, -1], repeat=len(edges))) # either 1 or -1, all possible combinations
	edge2add = [(*edge, w) for edge, w in zip(edges, all_ws[0])]
	G0.add_weighted_edges_from((edge2add), weight=weight)
	unique_sms = [G0]
	for all_w in tqdm(all_ws[1:], total=len(all_ws)-1, disable=True):
		G = nx.DiGraph()
		edge2add = [(*edge, w) for edge, w in zip(edges, all_w)]
		G.add_weighted_edges_from((edge2add), weight=weight)
		is_unique = True
		for ex_G in unique_sms:
			if nx.is_isomorphic(G, ex_G, edge_match=em):  # match weight
				is_unique = False
				break
		if is_unique:
			unique_sms.append(G)
	return unique_sms

def get_edges_sms(motif_type, weight='weight'):
	if motif_type == '021D':
		edges = [(0, 1), (0, 2)]
	elif motif_type == '021U':
		edges = [(0, 1), (2, 1)]
	elif motif_type == '021C':
		edges = [(0, 1), (1, 2)]
	elif motif_type == '111D':
		edges = [(0, 1), (1, 2), (2, 1)]
	elif motif_type == '111U':
		edges = [(0, 1), (0, 2), (2, 0)]
	elif motif_type == '030T':
		edges = [(0, 1), (1, 2), (0, 2)]
	elif motif_type == '030C':
		edges = [(0, 1), (1, 2), (2, 0)]
	elif motif_type == '201':
		edges = [(0, 1), (1, 0), (0, 2), (2, 0)]
	elif motif_type == '120D':
		edges = [(0, 1), (0, 2), (1, 2), (2, 1)]
	elif motif_type == '120U':
		edges = [(0, 2), (1, 2), (0, 1), (1, 0)]
	elif motif_type == '120C':
		edges = [(0, 1), (1, 2), (0, 2), (2, 0)]
	elif motif_type == '210':
		edges = [(0, 1), (1, 2), (2, 1), (0, 2), (2, 0)]
	elif motif_type == '300':
		edges = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 0), (0, 2)]
	unique_sms = find_all_unique_smotifs(edges, weight=weight)
	return edges, unique_sms

# get sign of the motif
def to_unique_form(signed_motif, unique_sms, weight='weight'):
	em = iso.numerical_edge_match(weight, 1)
	for unique_sm in unique_sms:
		if nx.is_isomorphic(signed_motif, unique_sm, edge_match=em):  # match weight
			unique_form = unique_sm
			break
	return unique_form

def get_motif_intensity_coherence(motif, weight='weight'):
	edges = motif.edges()
	w_list = []
	for edge in edges:
		w_list.append(abs(motif[edge[0]][edge[1]][weight]))
	I = np.prod(w_list)**(1.0/len(w_list))
	intensity = I
	coherence = I / np.mean(w_list)
	return intensity, coherence

def get_motif_sign(origin_motif, edges, unique_sms, weight='weight'):
	motif = origin_motif.copy()
	for u, v, w in motif.edges(data=weight):
		motif[u][v][weight] = 1 if w > 0 else -1
	unique_form = to_unique_form(motif, unique_sms, weight=weight)
	signs = [unique_form[edge[0]][edge[1]][weight] for edge in edges]
	signs = ''.join(['+' if sign > 0 else '-' for sign in signs])
	return signs

def add_missing_motif_type(df, mtype, signed_motif_types):
	if len(mtype) < len(signed_motif_types):
		mtype2add = [t for t in signed_motif_types if t not in mtype]
		for mt in mtype2add:
			mtype.append(mt)
			df = pd.concat([df, pd.DataFrame([[mt] + [0] * (df.shape[1]-1)], columns=df.columns)], ignore_index=True)
		df['intensity'] = pd.to_numeric(df['intensity'])
		df['coherence'] = pd.to_numeric(df['coherence'])
	return df, mtype

def motif_census_one_graph(G, all_signed_motif_types, motif_types, motif_edges, motif_sms, weight='weight'):
	intensity_df = pd.DataFrame(columns=['signed_motif_type', 'intensity', 'coherence'])
	motifs_by_type = find_triads(G)
	for motif_type in motif_types:
		motifs = motifs_by_type[motif_type]
		for motif in motifs:
			intensity, coherence = get_motif_intensity_coherence(motif, weight=weight)
			signed_motif_type = motif_type + get_motif_sign(motif, motif_edges[motif_type], motif_sms[motif_type], weight=weight)
			intensity_df = pd.concat([intensity_df, pd.DataFrame({'signed_motif_type': [signed_motif_type], 'intensity': [intensity], 'coherence': [coherence]})])
	intensity_df = intensity_df.groupby(['signed_motif_type']).sum().reset_index()
	signed_motif_types = intensity_df['signed_motif_type'].unique().tolist()
	intensity_df, _ = add_missing_motif_type(intensity_df, signed_motif_types, all_signed_motif_types)
	return intensity_df.set_index('signed_motif_type').loc[all_signed_motif_types]

def motif_census(G, random_graphs, all_signed_motif_types, motif_types, motif_edges, motif_sms, weight='weight', parallel=False, num_cores=23):
	num_rewire = len(random_graphs)
	print('Motif census for the real graph...')
	intensity_df = motif_census_one_graph(G, all_signed_motif_types, motif_types, motif_edges, motif_sms, weight=weight)
	print('Motif census for the random graphs...')
	if parallel:
		result = Parallel(n_jobs=num_cores)(delayed(motif_census_one_graph)(random_graphs[rep], all_signed_motif_types, motif_types, motif_edges, motif_sms, weight=weight) for rep in tqdm(range(num_rewire), disable=False))
		all_intensity_random_df = pd.concat([intensity_random_df['intensity'] for intensity_random_df in result], axis=1)
	else:
		all_intensity_random_df = pd.DataFrame()
		for random_graph in tqdm(random_graphs, total=num_rewire):
			intensity_random_df = motif_census_one_graph(random_graph, all_signed_motif_types, motif_types, motif_edges, motif_sms, weight=weight)
			all_intensity_random_df = pd.concat([all_intensity_random_df, intensity_random_df['intensity']], axis=1)
	m, std = all_intensity_random_df.mean(axis=1), all_intensity_random_df.std(axis=1)
	# Use a pseudo standard deviation value to avoid NaN Z score for motifs not present in the 200 realizations of the reference model
	# suppose it appeared once in one of the 200 realizations, and all connections have a minimum Z_CCG score of 4 (according to 4-fold significance level)
	pseudo_intensity_random = [4] + [0] * (num_rewire-1)
	pseudo_std_value = np.std(pseudo_intensity_random)
	pseudo_std = std.copy()
	pseudo_std[(m==0) & (std==0)] = pseudo_std_value
	intensity_df['intensity_random_mean'] = m
	intensity_df['intensity_random_std'] = pseudo_std
	intensity_z_score = (intensity_df['intensity'] - m) / pseudo_std
	# Note that the small values of pseudo std could lead to extremely large values for motifs present in the real graph but not in any realizations of reference model,
	# in practice use as many realizations as possible and remove outliers of Z scores from multiple animals (e.g., 2 std across animals) to eliminate noise.
	intensity_df['intensity_z_score'] = intensity_z_score
	return intensity_df

def count_pair_connection_p(G):
	# 0, 1, 2 edges
	num0, num1, num2 = 0, 0, 0
	nodes = list(G.nodes())
	for node_i in range(len(nodes)):
		for node_j in range(len(nodes)):
			if node_i != node_j:
				if G.has_edge(nodes[node_i], nodes[node_j]) or G.has_edge(nodes[node_j], nodes[node_i]):
					if G.has_edge(nodes[node_i], nodes[node_j]) and G.has_edge(nodes[node_j], nodes[node_i]):
						num2 += 1
					else:
						num1 += 1
				else:
					num0 += 1
	assert num0 + num1 + num2 == len(nodes) * (len(nodes) - 1)
	assert num1 / 2 + num2 == G.number_of_edges()
	p0, p1, p2 = safe_division(num0, num0 + num1 + num2), safe_division(num1, num0 + num1 + num2), safe_division(num2, num0 + num1 + num2)
	return p0, p1, p2

def count_signed_pair_connection_p(G, weight):
	# 0, 1pos, 1neg, 2pos, 1pos+1neg, 2neg
	num0, num1, num2, num3, num4, num5 = 0, 0, 0, 0, 0, 0
	nodes = list(G.nodes())
	for node_i in range(len(nodes)):
		for node_j in range(node_i+1, len(nodes)):
			i2j, j2i = G.has_edge(nodes[node_i], nodes[node_j]), G.has_edge(nodes[node_j], nodes[node_i])
			if i2j and (not j2i):
				i2jw = G[nodes[node_i]][nodes[node_j]][weight]
				if i2jw > 0:
					num1 += 1
				else:
					num2 += 1
			elif (not i2j) and j2i:
				j2iw = G[nodes[node_j]][nodes[node_i]][weight]
				if j2iw > 0:
					num1 += 1
				else:
					num2 += 1
			elif i2j and j2i:
				i2jw, j2iw = G[nodes[node_i]][nodes[node_j]][weight], G[nodes[node_j]][nodes[node_i]][weight]
				if (i2jw>0) and (j2iw>0):
					num3 += 1
				elif (i2jw<0) and (j2iw<0):
					num5 += 1
				else:
					num4 += 1
			else:
				num0 += 1
	assert num0 + num1 + num2 + num3 + num4 + num5 == len(nodes) * (len(nodes) - 1) / 2
	assert (num1 + num2) + 2 * (num3 + num4 + num5) == G.number_of_edges()
	pos = [e[2][weight] for e in G.edges(data=True) if e[2][weight]>0]
	neg = [e[2][weight] for e in G.edges(data=True) if e[2][weight]<0]
	assert (num1 + 2 * num3 + num4) == len(pos), f'{num1 + 2 * num3 + num4}, {len(pos)}'
	assert (num2 + num4 + 2 * num5) == len(neg), f'{num2 + num4 + 2 * num5}, {len(neg)}'
	summ = num0 + num1 + num2 + num3 + num4 + num5
	p0, p1, p2, p3, p4, p5 = safe_division(num0, summ), safe_division(num1, summ), safe_division(num2, summ), safe_division(num3, summ), safe_division(num4, summ), safe_division(num5, summ)
	return p0, p1, p2, p3, p4, p5

####################### Surrogate network generator via different reference models
def remove_common(a, b):
	return list(set(a).difference(b)), list(set(b).difference(a))

def generate_one_random_graph(input_G, model, weight='weight', cc=False, Q=100, max_tries=1e75):
	"""
		Generate one random graph based on the input graph and the reference model.

		Args:
			input_G (nx.DiGraph): The input directed, weighted graph.
			model (str): The reference model to generate the random graph, including 'erdos_renyi', 'degree_preserving', 'pair_preserving', and'signed_pair_preserving'.
			weight (str): The edge weight attribute.
			cc (bool): Whether to focus on the largest connected component of the input graph.
			Q (int): The number of swaps for each edge to perform.
			max_tries (int): The maximum number of attempts to perform edge swaps.

		Returns:
			nx.DiGraph: The generated random graph.
	"""
	origin_G = input_G.copy()
	if cc: # get the largest connected component of the graph
		origin_G = get_lcc(origin_G)
	weight_dict = nx.get_edge_attributes(origin_G, weight)
	weights = list(weight_dict.values())
	if model in ['degree_preserving', 'pair_preserving', 'signed_pair_preserving']:
		# preserve degree distribution based on double egdge swap method when generating random graphs, adapted from networkx.algorithms.swap
		nswap = Q*origin_G.number_of_edges()
		keys, out_degrees = zip(*origin_G.out_degree())  # keys, degree
		cdf = nx.utils.cumulative_distribution(out_degrees)  # cdf of degree
		discrete_sequence = nx.utils.discrete_sequence
	if model == 'erdos_renyi':
		node_idx = sorted(origin_G.nodes())
		mapping = {i:node_idx[i] for i in range(len(node_idx))}
		n, m = origin_G.number_of_nodes(), origin_G.number_of_edges()
		G = nx.gnm_random_graph(n, m, seed=None, directed=True)
		G = nx.relabel_nodes(G, mapping)
	elif model == 'degree_preserving':
		# swap u->v, x->y to u->y, x->v
		G = origin_G.copy()
		n_tries = 0
		swapcount = 0
		while swapcount < nswap:
			# pick two random edges without creating edge list
			# choose source node indices from discrete distribution
			(ui, xi) = discrete_sequence(2, cdistribution=cdf, seed=None)
			if ui == xi:
				continue  # same source, skip
			u = keys[ui]  # convert index to label
			x = keys[xi]
			# choose target uniformly from neighbors
			v = np.random.choice(list(G[u]))
			y = np.random.choice(list(G[x]))
			if (v == y) or (u == y) or (x == v):
				continue  # same target or self loop, skip
			if (y not in G[u]) and (v not in G[x]):  # don't create existing edges
				G.add_edge(u, y)
				G.add_edge(x, v)
				G.remove_edge(u, v)
				G.remove_edge(x, y)
				swapcount += 1
			if n_tries >= max_tries:
				e = (
					f"Maximum number of swap attempts ({n_tries}) exceeded "
					f"before desired swaps achieved ({nswap})."
				)
				raise nx.NetworkXAlgorithmError(e)
			n_tries += 1
	elif model == 'pair_preserving':
		# swap u->v, x->y to u->y, x->v, u<->v, x<->y to u<->y, x<->v
		G = origin_G.copy()
		n_tries = 0
		swapcount = 0
		while swapcount < nswap:
			(ui, xi) = discrete_sequence(2, cdistribution=cdf, seed=None)
			if ui == xi:
				continue  # same source, skip
			u = keys[ui]  # convert index to label
			x = keys[xi]
			# choose target uniformly from neighbors
			u_neigh, x_neigh = list(G[u]), list(G[x])
			if x in u_neigh:
				u_neigh.remove(x) # avoid self loop
			if u in x_neigh:
				x_neigh.remove(u) # avoid self loop
			u_neigh, x_neigh = remove_common(u_neigh, x_neigh) # avoid existing edges u->y and x->v
			if (not len(u_neigh)) or (not len(x_neigh)):
				continue
			ns = list(itertools.product(u_neigh, x_neigh))
			np.random.shuffle(ns)
			for v, y in ns:
				# for uni edges, they do not form new bidirectional edges
				# for bi edges, they can be switched
				if (u not in G[y]) and (x not in G[v]): # avoid existing edge or switch from unidirectional to bidirectional
					if (u not in G[v]) and (x not in G[y]):
						# unidirectional edges
						edge2add = [(u, y), (x, v)]
						edge2remove = [(u, v), (x, y)]
						G.add_edges_from(edge2add)
						G.remove_edges_from(edge2remove)
						swapcount += 1
						break
					elif (u in G[v]) and (x in G[y]):
						# bidirectional edges
						edge2add = [(u, y), (y, u), (x, v), (v, x)]
						edge2remove = [(u, v), (v, u), (x, y), (y, x)]
						G.add_edges_from(edge2add)
						G.remove_edges_from(edge2remove)
						swapcount += 2
						break
					else:
						continue
				else:
					continue
			if n_tries >= max_tries:
				e = (
					f"Maximum number of swap attempts ({n_tries}) exceeded "
					f"before desired swaps achieved ({nswap})."
				)
				raise nx.NetworkXAlgorithmError(e)
			n_tries += 1
	elif model == 'signed_pair_preserving':
		# swap u->v, x->y to u->y, x->v, u<->v, x<->y to u<->y, x<->v, with signed edge distri preserved
		G = origin_G.copy()
		n_tries = 0
		swapcount = 0
		while swapcount < nswap:
			(ui, xi) = discrete_sequence(2, cdistribution=cdf, seed=None)
			if ui == xi:
				continue  # same source, skip
			u = keys[ui]  # convert index to label
			x = keys[xi]
			# choose target uniformly from neighbors
			u_neigh, x_neigh = list(G[u]), list(G[x])
			if x in u_neigh:
				u_neigh.remove(x) # avoid self loop
			if u in x_neigh:
				x_neigh.remove(u) # avoid self loop
			u_neigh, x_neigh = remove_common(u_neigh, x_neigh) # avoid existing edges u->y and x->v
			if (not len(u_neigh)) or (not len(x_neigh)):
				continue
			ns = list(itertools.product(u_neigh, x_neigh))
			np.random.shuffle(ns)
			for v, y in ns:
				# for uni edges, they do not form new bidirectional edges
				# for bi edges, they can be switched
				if (u not in G[y]) and (x not in G[v]): # avoid existing edge or switch from unidirectional to bidirectional
					if (u not in G[v]) and (x not in G[y]):
						# unidirectional edges
						edge2remove = [(u, v), (x, y)]
						# unidirectional edge weights can stick either with u & x (source node) or v & y (target node)
						ews = [G.get_edge_data(*e)[weight] for e in edge2remove]
						np.random.shuffle(ews)
						edge2add = [(u, y, ews[0]), (x, v, ews[1])]
						G.add_weighted_edges_from((edge2add), weight=weight)
						G.remove_edges_from(edge2remove)
						swapcount += 1 # count as 1 swap
						break
					elif (u in G[v]) and (x in G[y]):
						# bidirectional edges
						edge2remove = [(u, v), (v, u), (x, y), (y, x)]
						# bidirectional edge weights can stick either with u & x or v & y
						ews = [(G[u][v][weight],G[v][u][weight]), (G[x][y][weight], G[y][x][weight])]
						np.random.shuffle(ews)
						edge2add = [(u, y, ews[0][0]), (y, u, ews[0][1]), (x, v, ews[1][0]), (v, x, ews[1][1])]
						G.add_weighted_edges_from((edge2add), weight=weight)
						G.remove_edges_from(edge2remove)
						swapcount += 2 # count as 2 swaps
						break
					else:
						continue
				else:
					continue
			if n_tries >= max_tries:
				e = (
					f"Maximum number of swap attempts ({n_tries}) exceeded "
					f"before desired swaps achieved ({nswap})."
				)
				raise nx.NetworkXAlgorithmError(e)
			n_tries += 1
	# add back the original link weights
	if model != 'signed_pair_preserving':
		np.random.shuffle(weights)
		if len(weights):
			for ind, e in enumerate(G.edges()):
				G[e[0]][e[1]][weight] = weights[ind]
	return G

def random_graph_generator(input_G, num_rewire, model, weight='weight', cc=False, Q=100, parallel=False, num_cores=23, disable=False):
	if parallel:
		result = Parallel(n_jobs=num_cores)(delayed(generate_one_random_graph)(input_G, model, weight=weight, cc=cc, Q=Q) for rep in tqdm(range(num_rewire), disable=disable))
		random_graphs = list(result)
	else:
		random_graphs = []
		for _ in tqdm(range(num_rewire), disable=disable):
			random_graphs.append(generate_one_random_graph(input_G, model, weight=weight, cc=cc, Q=Q))
	return random_graphs

def verify_random_graphs(origin_G, random_graphs, model, weight='weight'):
	print(f'******Property verification for the generated random graphs using {model} model******')
	# test if the number of nodes and edges and density are the same
	print('Number of nodes/edges and density test...')
	for random_G in tqdm(random_graphs):
		# assert nx.density(origin_G)==nx.density(random_G), 'The density of the original graph and the random graph should be the same!'
		assert (origin_G.number_of_nodes()==random_G.number_of_nodes()) and (origin_G.number_of_edges()==random_G.number_of_edges()), 'The number of nodes and edges should be the same!'
	# test if sum of weights is the same
	print('Sum of edge weights test...')
	for random_G in tqdm(random_graphs):
		assert np.isclose(random_G.size(weight=weight), origin_G.size(weight=weight)), 'Sum of edge weights should be the same!'
	if model in ['degree_preserving', 'pair_preserving', 'signed_pair_preserving']:
		# test if degree distribution is the same
		print('Degree distribution test...')
		nodes = sorted(list(origin_G.nodes()))
		indegree_seq, outdegree_seq = [origin_G.in_degree(node) for node in nodes], [origin_G.out_degree(node) for node in nodes]
		for random_G in tqdm(random_graphs):
			assert (indegree_seq==[random_G.in_degree(node) for node in nodes]) and (outdegree_seq==[random_G.out_degree(node) for node in nodes]), 'The degree distribution should be the same!'
		if model in ['pair_preserving', 'signed_pair_preserving']:
			# test if pair distribution is the same
			print('Pair distribution test...')
			for random_G in tqdm(random_graphs):
				assert np.isclose(count_pair_connection_p(random_G), count_pair_connection_p(origin_G)).all(), 'The pair distribution should be the same!'
			if model == 'signed_pair_preserving':
				# test if signed pair distribution is the same
				print('Signed pair distribution test...')
				for random_G in tqdm(random_graphs):
					assert np.isclose(count_signed_pair_connection_p(random_G, weight=weight), count_signed_pair_connection_p(origin_G, weight=weight)).all(), 'The signed pair distribution should be the same!'
	print('******All tests are passed!******')

def plot_motif_intensity_z_scores(intensity_df):
	palette = [[plt.cm.tab20b(i) for i in range(20)][i] for i in [0,2,3,4,6,8,10,12,16,18,19]] + [[plt.cm.tab20c(i) for i in range(20)][i] for i in [4,16]]
	TRIAD_NAMES = ('003', '012', '102', '021D', '021U', '021C', '111D', '111U',
									'030T', '030C', '201', '120D', '120U', '120C', '210', '300')
	# sorted_types is in the same order as Fig. 2D and Fig. S6
	sorted_types = [sorted([smotif for smotif in intensity_df.index if mt in smotif]) for mt in TRIAD_NAMES]
	sorted_types = [item for sublist in sorted_types for item in sublist]
	motif_types = TRIAD_NAMES[3:]
	fig, ax = plt.subplots(1,1, sharex=True, sharey=True, figsize=(50, 3*1))
	for t, y in zip(sorted_types, intensity_df.loc[sorted_types, "intensity_z_score"]):
		color = palette[motif_types.index(t.replace('+', '').replace('-', ''))]
		ax.plot([t,t], [0,y], color=color, marker="o", linewidth=7, markersize=20, markevery=(1,2))
	ax.set_xlim(-.5,len(sorted_types)+.5)
	ax.set_xticks([])
	ax.yaxis.set_tick_params(labelsize=45)
	for axis in ['bottom', 'left']:
		ax.spines[axis].set_linewidth(3)
		ax.spines[axis].set_color('k')
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.tick_params(width=3)
	ax.xaxis.set_tick_params(length=0)
	ax.set_ylabel('')
	ax.set_ylim(-10, 32)
	plt.tight_layout()
	plt.show()

all_signed_motif_types = ['021D++', '021D+-', '021D--', '021U++', '021U+-', '021U--',
			 '021C++', '021C+-', '021C-+', '021C--', '111D+++', '111D++-',
			 '111D+-+', '111D+--', '111D-++', '111D-+-', '111D--+', '111D---',
			 '111U+++', '111U++-', '111U+-+', '111U+--', '111U-++', '111U-+-',
			 '111U--+', '111U---', '030T+++', '030T++-', '030T+-+', '030T+--',
			 '030T-++', '030T-+-', '030T--+', '030T---', '030C+++', '030C++-',
			 '030C+--', '030C---', '201++++', '201+++-', '201++-+', '201++--',
			 '201+-+-', '201+--+', '201+---', '201-+-+', '201-+--', '201----',
			 '120D++++', '120D+++-', '120D++--', '120D+-++', '120D+-+-',
			 '120D+--+', '120D+---', '120D--++', '120D--+-', '120D----',
			 '120U++++', '120U+++-', '120U++--', '120U+-++', '120U+-+-',
			 '120U+--+', '120U+---', '120U--++', '120U--+-', '120U----',
			 '120C++++', '120C+++-', '120C++-+', '120C++--', '120C+-++',
			 '120C+-+-', '120C+--+', '120C+---', '120C-+++', '120C-++-',
			 '120C-+-+', '120C-+--', '120C--++', '120C--+-', '120C---+',
			 '120C----', '210+++++', '210++++-', '210+++-+', '210+++--',
			 '210++-++', '210++-+-', '210++--+', '210++---', '210+-+++',
			 '210+-++-', '210+-+-+', '210+-+--', '210+--++', '210+--+-',
			 '210+---+', '210+----', '210-++++', '210-+++-', '210-++-+',
			 '210-++--', '210-+-++', '210-+-+-', '210-+--+', '210-+---',
			 '210--+++', '210--++-', '210--+-+', '210--+--', '210---++',
			 '210---+-', '210----+', '210-----', '300++++++', '300+++++-',
			 '300++++--', '300+++-+-', '300+++--+', '300+++---', '300++-++-',
			 '300++-+--', '300++----', '300+-+-+-', '300+-+--+', '300+-+---',
			 '300+--+--', '300+----+', '300+-----', '300------']
