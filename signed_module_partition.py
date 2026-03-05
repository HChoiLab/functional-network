import os
import pickle
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from netgraph import Graph
import networkx as nx
from networkx import NetworkXError
from networkx.algorithms.community.community_utils import is_partition
from networkx.utils import py_random_state
from collections import defaultdict, deque
from tqdm import tqdm
from joblib import Parallel, delayed # keep this line if use parallel generation of random graphs, otherwise comment this line
visual_regions = ['VISam', 'VISpm', 'VISal', 'VISrl', 'VISl', 'VISp']
region_labels = ['AM', 'PM', 'AL', 'RL', 'LM', 'V1']
region_colors = ['#d9e9b5', '#c0d8e9', '#fed3a1', '#c3c3c3', '#fad3e4', '#cec5f2']

def transparent_rgb(rgb, bg_rgb, alpha):
	return [alpha * c1 + (1 - alpha) * c2 for (c1, c2) in zip(rgb, bg_rgb)]

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
 
#############################################################################
# signed louvain algorithm with Signed Modularity for community detection in signed graphs
class NotAPartition(NetworkXError):
		"""Raised if a given collection is not a partition."""

		def __init__(self, G, collection):
				msg = f"{G} is not a valid partition of the graph {collection}"
				super().__init__(msg)

def signed_degree(G, sign_type='pos', dir_type='out', weight='weight'):
	# change neg to absolute
	degree = {}
	for node in G.nodes():
		if dir_type == 'out':
			if sign_type == 'pos':
				weights = [G[node][neib][weight] for neib in G.successors(node) if G[node][neib][weight] > 0]
			elif sign_type == 'neg':
				weights = [abs(G[node][neib][weight]) for neib in G.successors(node) if G[node][neib][weight] < 0]
			elif sign_type == 'abs':
				weights = [abs(G[node][neib][weight]) for neib in G.successors(node)]
		elif dir_type == 'in':
			if sign_type == 'pos':
				weights = [G[neib][node][weight] for neib in G.predecessors(node) if G[neib][node][weight] > 0]
			elif sign_type == 'neg':
				weights = [abs(G[neib][node][weight]) for neib in G.predecessors(node) if G[neib][node][weight] < 0]
			elif sign_type == 'abs':
				weights = [abs(G[neib][node][weight]) for neib in G.predecessors(node)]
		elif dir_type == 'undirected':
			if sign_type == 'pos':
				weights = [G[node][neib][weight] for neib in G.neighbors(node) if G[neib][node][weight] > 0]
			elif sign_type == 'neg':
				weights = [abs(G[node][neib][weight]) for neib in G.neighbors(node) if G[neib][node][weight] < 0]
			elif sign_type == 'abs':
				weights = [abs(G[node][neib][weight]) for neib in G.neighbors(node)]
		degree[node] = sum(weights)
	return degree

def signed_modularity(G, communities, weight="weight", pos_resolution=1, neg_resolution=1):
		if not isinstance(communities, list):
				communities = list(communities)
		if not is_partition(G, communities):
				raise NotAPartition(G, communities)

		directed = G.is_directed()
		if directed:
		#   out_degree = signed_degree(G, 'abs', 'out', weight)
			out_degree_pos = signed_degree(G, 'pos', 'out', weight)
			out_degree_neg = signed_degree(G, 'neg', 'out', weight)
			in_degree_pos = signed_degree(G, 'pos', 'in', weight)
			in_degree_neg = signed_degree(G, 'neg', 'in', weight)
			# m = sum(abs(out_degree.values()))
		else:
				out_degree_pos = in_degree_pos = signed_degree(G, 'pos', 'undirected', weight)
				out_degree_neg = in_degree_neg = signed_degree(G, 'neg', 'undirected', weight)
		pos_norm = safe_division(1, sum(out_degree_pos.values()))
		neg_norm = safe_division(1, sum(out_degree_neg.values()))
		norm = safe_division(1, sum(out_degree_pos.values()) + sum(out_degree_neg.values()))
		def signed_community_contribution(community):
				comm = set(community)
				L_c = sum(wt for u, v, wt in G.edges(comm, data=weight, default=1) if v in comm)
				out_degree_pos_sum = sum(out_degree_pos[u] for u in comm)
				in_degree_pos_sum = sum(in_degree_pos[u] for u in comm) if directed else out_degree_pos_sum
				out_degree_neg_sum = sum(out_degree_neg[u] for u in comm)
				in_degree_neg_sum = sum(in_degree_neg[u] for u in comm) if directed else out_degree_neg_sum

				return norm * (L_c - pos_resolution * out_degree_pos_sum * in_degree_pos_sum * pos_norm + neg_resolution * out_degree_neg_sum * in_degree_neg_sum * neg_norm)

		return sum(map(signed_community_contribution, communities))

@py_random_state("seed")
def signed_louvain_communities(
		G, weight="weight", pos_resolution=1, neg_resolution=1, threshold=0.0000001, seed=None
):
		d = signed_louvain_partitions(G, weight, pos_resolution, neg_resolution, threshold, seed)
		q = deque(d, maxlen=1)
		return q.pop()


@py_random_state("seed")
def signed_louvain_partitions(
		G, weight="weight", pos_resolution=1, neg_resolution=1, threshold=0.0000001, seed=None
):
		partition = [{u} for u in G.nodes()]
		Q = signed_modularity(G, partition, weight, pos_resolution, neg_resolution)
		is_directed = G.is_directed()
		if G.is_multigraph():
				graph = _convert_multigraph(G, weight, is_directed)
		else:
				graph = G.__class__()
				graph.add_nodes_from(G)
				graph.add_weighted_edges_from(G.edges(data=weight, default=0.0), weight=weight)
		
		out_degree_pos = signed_degree(G, 'pos', 'out', weight)
		out_degree_neg = signed_degree(G, 'neg', 'out', weight)
		pos_norm = safe_division(1, sum(out_degree_pos.values()))
		neg_norm = safe_division(1, sum(out_degree_neg.values()))
		# pos_norm = 1 / sum(out_degree_pos.values())
		# neg_norm = 1 / sum(out_degree_neg.values())
		partition, inner_partition, improvement = _one_level(
				graph, pos_norm, neg_norm, partition, weight, pos_resolution, neg_resolution, is_directed, seed
		)
		improvement = True
		while improvement:
				yield partition
				new_Q = signed_modularity(graph, inner_partition, weight, pos_resolution, neg_resolution)
				if new_Q - Q  <= threshold:
						return
				Q = new_Q
				graph = _gen_graph(graph, inner_partition, weight)
				partition, inner_partition, improvement = _one_level(
						graph, pos_norm, neg_norm, partition, weight, pos_resolution, neg_resolution, is_directed, seed
				)

def _one_level(G, pos_norm, neg_norm, partition, weight='weight', pos_resolution=1, neg_resolution=1, is_directed=False, seed=None):
		node2com = {u: i for i, u in enumerate(G.nodes())}
		inner_partition = [{u} for u in G.nodes()]
		if is_directed:
			out_degrees_pos = signed_degree(G, 'pos', 'out', weight)
			out_degrees_neg = signed_degree(G, 'neg', 'out', weight)
			in_degrees_pos = signed_degree(G, 'pos', 'in', weight)
			in_degrees_neg = signed_degree(G, 'neg', 'in', weight)
			Stot_in_pos = [deg for deg in in_degrees_pos.values()]
			Stot_in_neg = [deg for deg in in_degrees_neg.values()]
			Stot_out_pos = [deg for deg in out_degrees_pos.values()]
			Stot_out_neg = [deg for deg in out_degrees_neg.values()]
			# Calculate weights for both in and out neighbours
			nbrs = {} # key is each node and its in and out neighbors, value is their weight
			for u in G:
				nbrs[u] = defaultdict(float)
				for _, n, wt in G.out_edges(u, data=weight):
					nbrs[u][n] += wt
				for n, _, wt in G.in_edges(u, data=weight):
					nbrs[u][n] += wt
		else:
				pos_degrees = signed_degree(G, 'pos', 'undirected', weight)
				neg_degrees = signed_degree(G, 'neg', 'undirected', weight)
				Stot_pos = [deg for deg in pos_degrees.values()]
				Stot_neg = [deg for deg in neg_degrees.values()]
				nbrs = {u: {v: data[weight] for v, data in G[u].items() if v != u} for u in G}
		rand_nodes = list(G.nodes)
		seed.shuffle(rand_nodes)
		nb_moves = 1
		improvement = False
		while nb_moves > 0:
				nb_moves = 0
				for u in rand_nodes:
						best_gain = 0
						best_com = node2com[u]
						weights2com = _neighbor_weights(nbrs[u], node2com) # summed weights of in and out neighbors of u, key is their current community
						if is_directed:
								in_degree_pos = in_degrees_pos[u]
								out_degree_pos = out_degrees_pos[u]
								in_degree_neg = in_degrees_neg[u]
								out_degree_neg = out_degrees_neg[u]
								Stot_in_pos[best_com] -= in_degree_pos # pos in degree of other nodes in the same community
								Stot_out_pos[best_com] -= out_degree_pos
								Stot_in_neg[best_com] -= in_degree_neg
								Stot_out_neg[best_com] -= out_degree_neg
								remove_cost = (
										+ weights2com[best_com]
										- pos_resolution * pos_norm * (out_degree_pos * Stot_in_pos[best_com] + in_degree_pos * Stot_out_pos[best_com])
										+ neg_resolution * neg_norm * (out_degree_neg * Stot_in_neg[best_com] + in_degree_neg * Stot_out_neg[best_com])
								)
						else:
								pos_degree = pos_degrees[u]
								neg_degree = neg_degrees[u]
								Stot_pos[best_com] -= pos_degree
								Stot_neg[best_com] -= neg_degree
								remove_cost = +weights2com[best_com] - pos_resolution * (
										Stot_pos[best_com] * pos_degree) * 2 * pos_norm \
										+ neg_resolution * (
										Stot_neg[best_com] * neg_degree) * 2 * neg_norm
						for nbr_com, wt in weights2com.items(): # compare every other community nbr_com node u can move to
								if wt > 0: # only move to a neighbor's community if weight is positive, unless will cause node u to consistently jump between negative neighbor's community
										if is_directed:
												gain = (
														remove_cost
														- wt
														+ pos_resolution * pos_norm * (out_degree_pos * Stot_in_pos[nbr_com] + in_degree_pos * Stot_out_pos[nbr_com])
														- neg_resolution * neg_norm * (out_degree_neg * Stot_in_neg[nbr_com] + in_degree_neg * Stot_out_neg[nbr_com])
												)
										else:
												gain = (
														remove_cost
														- wt
														+ pos_resolution * (Stot_pos[nbr_com] * pos_degree) * 2 * pos_norm
														- neg_resolution * (Stot_neg[nbr_com] * neg_degree) * 2 * neg_norm
												)
										if gain < best_gain:
												best_gain = gain
												best_com = nbr_com
						if is_directed:
								Stot_in_pos[best_com] += in_degree_pos
								Stot_out_pos[best_com] += out_degree_pos
								Stot_in_neg[best_com] += in_degree_neg
								Stot_out_neg[best_com] += out_degree_neg
						else:
								Stot_pos[best_com] += pos_degree
								Stot_neg[best_com] += neg_degree
						if best_com != node2com[u]:
								com = G.nodes[u].get("nodes", {u})
								partition[node2com[u]].difference_update(com)
								inner_partition[node2com[u]].remove(u)
								partition[best_com].update(com)
								inner_partition[best_com].add(u)
								improvement = True
								nb_moves += 1
								node2com[u] = best_com
	
		partition = list(filter(len, partition))
		inner_partition = list(filter(len, inner_partition))
		return partition, inner_partition, improvement


def _neighbor_weights(nbrs, node2com):
		weights = defaultdict(float)
		for nbr, wt in nbrs.items():
				weights[node2com[nbr]] += wt
		return weights


def _gen_graph(G, partition, weight):
		# generate a graph whose node is community, edge weight is the summation of weights between two communities
		H = G.__class__()
		node2com = {}
		for i, part in enumerate(partition):
				nodes = set()
				for node in part:
						node2com[node] = i
						nodes.update(G.nodes[node].get("nodes", {node}))
				H.add_node(i, nodes=nodes)

		for node1, node2, wt in G.edges(data=True):
				wt = wt[weight]
				com1 = node2com[node1]
				com2 = node2com[node2]
				temp = H.get_edge_data(com1, com2, {weight: 0})[weight]
				H.add_edge(com1, com2, **{weight: wt + temp})
		return H

def _convert_multigraph(G, weight, is_directed):
		if is_directed:
				H = nx.DiGraph()
		else:
				H = nx.Graph()
		H.add_nodes_from(G)
		for u, v, wt in G.edges(data=weight, default=1):
				if H.has_edge(u, v):
						H[u][v][weight] += wt
				else:
						H.add_edge(u, v, weight=wt)
		return H

def get_comms_sModularity(input_G, resolution_list, num_repeat, weight='weight', cc=False, disable_tqdm=False):
	resolution_list = np.round(resolution_list, 2)
	G = input_G.copy()
	all_comms = {pos_resolution: {neg_resolution: [] for neg_resolution in resolution_list} for pos_resolution in resolution_list}
	sModularity = np.full([len(resolution_list), len(resolution_list), num_repeat], np.nan)
	if cc:
		G = get_lcc(G)
	all_combinations = [(pres_ind, pos_resolution, nres_ind, neg_resolution, repeat) for pres_ind, pos_resolution in enumerate(resolution_list) for nres_ind, neg_resolution in enumerate(resolution_list) for repeat in range(num_repeat)]
	for pres_ind, pos_resolution, nres_ind, neg_resolution, repeat in tqdm(all_combinations, disable=disable_tqdm):
		comms = signed_louvain_communities(G.copy(), pos_resolution=pos_resolution, neg_resolution=neg_resolution, weight=weight)
		all_comms[pos_resolution][neg_resolution].append(comms)
		sModularity[pres_ind, nres_ind, repeat] = signed_modularity(G.copy(), communities=comms, weight=weight, pos_resolution=pos_resolution, neg_resolution=neg_resolution)
	return all_comms, sModularity

def get_comms_sModularity_null(input_Gs, resolution_list, num_repeat, weight='weight', cc=False, parallel=False, num_cores=23, disable=False):
	if parallel:
		result = Parallel(n_jobs=num_cores)(delayed(get_comms_sModularity)(input_G, resolution_list, num_repeat, weight=weight, cc=cc, disable_tqdm=True) for input_G in tqdm(input_Gs, disable=disable))
		all_comms_null, sModularity_null = [r[0] for r in result], [r[1] for r in result]
	else:
		all_comms_null, sModularity_null = [], []
		for input_G in tqdm(input_Gs, disable=disable):	
			result = get_comms_sModularity(input_G, resolution_list, num_repeat, weight=weight, cc=cc, disable_tqdm=True)
			all_comms_null.append(result[0])
			sModularity_null.append(result[1])
		
	return all_comms_null, np.asarray(sModularity_null)

# Combine clustering results based on voting-based method: keep clusters that appear the most during all runs
# Find eligible and ineligible nodes (not partitioned into single-node cluster during any stimuli)
def combine_clustering_results(comms_list):
  node_ids = np.unique([node for comms in comms_list for comm in comms for node in comm])
  cluster_dict = {}
  # Loop over each run of the clustering algorithm
  for comms in comms_list:
      # Loop over each cluster in the current run
      for cluster in comms:
          # Update the cluster assignment count for each node in the cluster
          for node in cluster:
              if node in cluster_dict:
                  cluster_dict[node].add(frozenset(cluster))
              else:
                  cluster_dict[node] = {frozenset(cluster)}
  # Initialize a list of unassigned nodes
  assigned_nodes = []
  unassigned_nodes = node_ids.tolist()
  # Initialize an empty list to store the final clustering result
  combined_comms = []
  # Loop over the unassigned nodes and assign them to a cluster
  while len(unassigned_nodes) > 0:
      # Initialize a dictionary to keep track of the cluster assignment counts for the current node
      node_cluster_counts = {}
      # Loop over the unassigned nodes
      for node in unassigned_nodes:
          # Check if the current node has any cluster assignments
          if node in cluster_dict:
              # Loop over the cluster assignment counts for the current node
              for cluster in cluster_dict[node]:
                  if cluster in node_cluster_counts:
                      node_cluster_counts[cluster] += 1
                  else:
                      node_cluster_counts[cluster] = 1
      # Assign the current node to the cluster with the highest vote count
      best_cluster = max(node_cluster_counts, key=node_cluster_counts.get)
      # Each node can only be assigned to one cluster
      best_cluster = [node for node in best_cluster if node not in assigned_nodes]
      combined_comms.append(sorted(list(best_cluster)))
      # Remove the current node and its assigned cluster from the unassigned nodes and cluster assignments
      unassigned_nodes = np.setdiff1d(unassigned_nodes, list(best_cluster))
      assigned_nodes += list(best_cluster)
      for node in best_cluster:
          if node in cluster_dict:
              cluster_dict.pop(node)
  return combined_comms