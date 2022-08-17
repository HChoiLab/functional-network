#%%
import networkx as nx
from networkx import NetworkXError
from networkx.algorithms.community.community_utils import is_partition

def signed_degree(G, sign_type='pos', dir_type='out', weight='weight'):
  degree = {}
  for node in G.nodes():
    if dir_type == 'out':
      if sign_type == 'pos':
        weights = [G[node][n][weight] for n in G.successors(node) if G[node][n][weight] > 0]
      elif sign_type == 'neg':
        weights = [G[node][n][weight] for n in G.successors(node) if G[node][n][weight] < 0]
      elif sign_type == 'abs':
        weights = [abs(G[node][n][weight]) for n in G.successors(node)]
    else:
      if sign_type == 'pos':
        weights = [G[n][node][weight] for n in G.predecessors(node) if G[n][node][weight] > 0]
      elif sign_type == 'neg':
        weights = [G[n][node][weight] for n in G.predecessors(node) if G[n][node][weight] < 0]
      elif sign_type == 'abs':
        weights = [abs(G[node][n][weight]) for n in G.predecessors(node)]
    degree[node] = sum(weights)
  return degree

class NotAPartition(NetworkXError):
    """Raised if a given collection is not a partition."""

    def __init__(self, G, collection):
        msg = f"{G} is not a valid partition of the graph {collection}"
        super().__init__(msg)


def signed_modularity(G, communities, weight="weight", pos_resolution=1, neg_resolution=1):
    if not isinstance(communities, list):
        communities = list(communities)
    if not is_partition(G, communities):
        raise NotAPartition(G, communities)

    directed = G.is_directed()
    if directed:
      out_degree = signed_degree(G, 'abs', 'out', weight)
      out_degree_pos = signed_degree(G, 'pos', 'out', weight)
      out_degree_neg = signed_degree(G, 'neg', 'out', weight)
      in_degree_pos = signed_degree(G, 'pos', 'in', weight)
      in_degree_neg = signed_degree(G, 'neg', 'in', weight)
      pos_norm = 1 / sum(out_degree_pos.values())
      neg_norm = 1 / sum(out_degree_neg.values())
      # m = sum(abs(out_degree.values()))
    else:
        out_degree = in_degree = dict(G.degree(weight=weight))
        deg_sum = sum(out_degree.values())
        m = deg_sum / 2
        norm = 1 / deg_sum**2

    def signed_community_contribution(community):
        comm = set(community)
        L_c = sum(wt for u, v, wt in G.edges(comm, data=weight, default=1) if v in comm)
        out_degree_pos_sum = sum(out_degree_pos[u] for u in comm)
        in_degree_pos_sum = sum(in_degree_pos[u] for u in comm) if directed else out_degree_pos_sum
        out_degree_neg_sum = sum(out_degree_neg[u] for u in comm)
        in_degree_neg_sum = sum(in_degree_neg[u] for u in comm) if directed else out_degree_neg_sum

        return - L_c + pos_resolution * out_degree_pos_sum * in_degree_pos_sum * pos_norm - neg_resolution * out_degree_neg_sum * in_degree_neg_sum * neg_norm

    return sum(map(signed_community_contribution, communities))
  

# simulated annealing search of a one-dimensional objective function
from numpy import asarray
from numpy import exp
from numpy.random import randn
from numpy.random import rand
from numpy.random import seed
 
# objective function
def objective(x):
	return x[0]**2.0
 
# simulated annealing algorithm
def simulated_annealing(objective, bounds, n_iterations, step_size, temp):
	# generate an initial point
	best = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
	# evaluate the initial point
	best_eval = objective(best)
	# current working solution
	curr, curr_eval = best, best_eval
	# run the algorithm
	for i in range(n_iterations):
		# take a step
		candidate = curr + randn(len(bounds)) * step_size
		# evaluate candidate point
		candidate_eval = objective(candidate)
		# check for new best solution
		if candidate_eval < best_eval:
			# store new best point
			best, best_eval = candidate, candidate_eval
			# report progress
			print('>%d f(%s) = %.5f' % (i, best, best_eval))
		# difference between candidate and current point evaluation
		diff = candidate_eval - curr_eval
		# calculate temperature for current epoch
		t = temp / float(i + 1)
		# calculate metropolis acceptance criterion
		metropolis = exp(-diff / t)
		# check if we should keep the new point
		if diff < 0 or rand() < metropolis:
			# store the new current point
			curr, curr_eval = candidate, candidate_eval
	return [best, best_eval]
 
def greedy_signed_modularity_communities(
    G, weight=None, resolution=1, cutoff=1, best_n=None):
    if (cutoff < 1) or (cutoff > G.number_of_nodes()):
        raise ValueError(f"cutoff must be between 1 and {len(G)}. Got {cutoff}.")
    if best_n is not None:
        if (best_n < 1) or (best_n > G.number_of_nodes()):
            raise ValueError(f"best_n must be between 1 and {len(G)}. Got {best_n}.")
        if best_n < cutoff:
            raise ValueError(f"Must have best_n >= cutoff. Got {best_n} < {cutoff}")
        if best_n == 1:
            return [set(G)]
    else:
        best_n = G.number_of_nodes()
    # retrieve generator object to construct output
    community_gen = _greedy_modularity_communities_generator(
        G, weight=weight, resolution=resolution
    )

    # construct the first best community
    communities = next(community_gen)

    # continue merging communities until one of the breaking criteria is satisfied
    while len(communities) > cutoff:
        try:
            dq = next(community_gen)
        # StopIteration occurs when communities are the connected components
        except StopIteration:
            communities = sorted(communities, key=len, reverse=True)
            # if best_n requires more merging, merge big sets for highest modularity
            while len(communities) > best_n:
                comm1, comm2, *rest = communities
                communities = [comm1 ^ comm2]
                communities.extend(rest)
            return communities

        # keep going unless max_mod is reached or best_n says to merge more
        if dq < 0 and len(communities) <= best_n:
            break
        communities = next(community_gen)

    return sorted(communities, key=len, reverse=True)

# seed the pseudorandom number generator
seed(1)
# define range for input
bounds = asarray([[-5.0, 5.0]])
# define the total iterations
n_iterations = 1000
# define the maximum step size
step_size = 0.1
# initial temperature
temp = 10
# perform the simulated annealing search
best, score = simulated_annealing(objective, bounds, n_iterations, step_size, temp)
print('Done!')
print('f(%s) = %f' % (best, score))
#%%
from library import *
area_dict, active_area_dict, mean_speed_df = load_other_data(session_ids)
directory = './data/ecephys_cache_dir/sessions/spiking_sequence/'
path = directory.replace('spiking_sequence', 'adj_mat_ccg_highland_corrected')
if not os.path.exists(path):
  os.makedirs(path)
G_ccg_dict, offset_dict, duration_dict = load_highland_xcorr(path, active_area_dict, weight=True)
measure = 'ccg'

G_ccg_dict = remove_gabor(G_ccg_dict)
G_ccg_dict = remove_thalamic(G_ccg_dict, area_dict, visual_regions)
offset_dict = remove_thalamic_mat(offset_dict, active_area_dict, visual_regions)
duration_dict = remove_thalamic_mat(duration_dict, active_area_dict, visual_regions)
active_area_dict = remove_thalamic_area(active_area_dict, visual_regions)
n = 4
S_ccg_dict = add_sign(G_ccg_dict)
S_ccg_dict = add_offset(S_ccg_dict, offset_dict)
# %%
rows, cols = get_rowcol(G_ccg_dict)
df = pd.DataFrame(index=rows, columns=cols)
for row in rows:
  for col in cols:
    G = G_ccg_dict[row][col]
    comms = nx_comm.louvain_communities(G, weight='weight', resolution=1)
    df.loc[row][col] = signed_modularity(G, comms, weight="weight", pos_resolution=1, neg_resolution=1)
    
# %%
df.T.plot()
plt.xticks(rotation=90)
# %%
from collections import defaultdict, deque

import networkx as nx
from networkx.utils import py_random_state

__all__ = ["signed_louvain_communities", "signed_louvain_partitions"]

def signed_modularity(G, communities, weight="weight", pos_resolution=1, neg_resolution=1):
    if not isinstance(communities, list):
        communities = list(communities)
    if not is_partition(G, communities):
        raise NotAPartition(G, communities)

    directed = G.is_directed()
    if directed:
      out_degree = signed_degree(G, 'abs', 'out', weight)
      out_degree_pos = signed_degree(G, 'pos', 'out', weight)
      out_degree_neg = signed_degree(G, 'neg', 'out', weight)
      in_degree_pos = signed_degree(G, 'pos', 'in', weight)
      in_degree_neg = signed_degree(G, 'neg', 'in', weight)
      pos_norm = 1 / sum(out_degree_pos.values())
      neg_norm = 1 / sum(out_degree_neg.values())
      # m = sum(abs(out_degree.values()))
    else:
        out_degree = in_degree = dict(G.degree(weight=weight))
        deg_sum = sum(out_degree.values())
        m = deg_sum / 2
        norm = 1 / deg_sum**2

    def signed_community_contribution(community):
        comm = set(community)
        L_c = sum(wt for u, v, wt in G.edges(comm, data=weight, default=1) if v in comm)
        out_degree_pos_sum = sum(out_degree_pos[u] for u in comm)
        in_degree_pos_sum = sum(in_degree_pos[u] for u in comm) if directed else out_degree_pos_sum
        out_degree_neg_sum = sum(out_degree_neg[u] for u in comm)
        in_degree_neg_sum = sum(in_degree_neg[u] for u in comm) if directed else out_degree_neg_sum

        return - L_c + pos_resolution * out_degree_pos_sum * in_degree_pos_sum * pos_norm - neg_resolution * out_degree_neg_sum * in_degree_neg_sum * neg_norm

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
    mod = signed_modularity(G, partition, weight, pos_resolution, neg_resolution)
    is_directed = G.is_directed()
    if G.is_multigraph():
        graph = _convert_multigraph(G, weight, is_directed)
    else:
        graph = G.__class__()
        graph.add_nodes_from(G)
        graph.add_weighted_edges_from(G.edges(data=weight, default=1))

    m = graph.size(weight="weight")
    partition, inner_partition, improvement = _one_level(
        graph, m, partition, weight, pos_resolution, neg_resolution, is_directed, seed
    )
    improvement = True
    while improvement:
        yield partition
        new_mod = signed_modularity(graph, inner_partition, weight, pos_resolution, neg_resolution)
        if new_mod - mod <= threshold:
            return
        mod = new_mod
        graph = _gen_graph(graph, inner_partition)
        partition, inner_partition, improvement = _one_level(
            graph, m, partition, weight, pos_resolution, neg_resolution, is_directed, seed
        )

def _one_level(G, m, partition, weight='weight', pos_resolution=1, neg_resolution=1, is_directed=False, seed=None):
    node2com = {u: i for i, u in enumerate(G.nodes())}
    inner_partition = [{u} for u in G.nodes()]
    if is_directed:
      out_degree_pos = signed_degree(G, 'pos', 'out', weight)
      out_degree_neg = signed_degree(G, 'neg', 'out', weight)
      in_degree_pos = signed_degree(G, 'pos', 'in', weight)
      in_degree_neg = signed_degree(G, 'neg', 'in', weight)
      Stot_in_pos = [deg for deg in in_degree_pos.values()]
      Stot_in_neg = [deg for deg in in_degree_neg.values()]
      Stot_out_pos = [deg for deg in out_degree_pos.values()]
      Stot_out_neg = [deg for deg in out_degree_neg.values()]
      # Calculate weights for both in and out neighbours
      nbrs = {}
      for u in G:
        nbrs[u] = defaultdict(float)
        for _, n, wt in G.out_edges(u, data="weight"):
          nbrs[u][n] += wt
        for n, _, wt in G.in_edges(u, data="weight"):
          nbrs[u][n] += wt
    else:
        degrees = dict(G.degree(weight="weight"))
        Stot = [deg for deg in degrees.values()]
        nbrs = {u: {v: data["weight"] for v, data in G[u].items() if v != u} for u in G}
    rand_nodes = list(G.nodes)
    seed.shuffle(rand_nodes)
    nb_moves = 1
    improvement = False
    while nb_moves > 0:
        nb_moves = 0
        for u in rand_nodes:
            best_mod = 0
            best_com = node2com[u]
            weights2com = _neighbor_weights(nbrs[u], node2com)
            if is_directed:
                in_degree_pos = in_degree_pos[u]
                out_degree_pos = out_degree_pos[u]
                in_degree_neg = in_degree_neg[u]
                out_degree_neg = out_degree_neg[u]
                Stot_in_pos[best_com] -= in_degree_pos
                Stot_out_pos[best_com] -= out_degree_pos
                remove_cost = (
                    -weights2com[best_com] / m
                    + resolution
                    * (out_degree * Stot_in[best_com] + in_degree * Stot_out[best_com])
                    / m**2
                )
            else:
                degree = degrees[u]
                Stot[best_com] -= degree
                remove_cost = -weights2com[best_com] / m + resolution * (
                    Stot[best_com] * degree
                ) / (2 * m**2)
            for nbr_com, wt in weights2com.items():
                if is_directed:
                    gain = (
                        remove_cost
                        + wt / m
                        - resolution
                        * (
                            out_degree * Stot_in[nbr_com]
                            + in_degree * Stot_out[nbr_com]
                        )
                        / m**2
                    )
                else:
                    gain = (
                        remove_cost
                        + wt / m
                        - resolution * (Stot[nbr_com] * degree) / (2 * m**2)
                    )
                if gain > best_mod:
                    best_mod = gain
                    best_com = nbr_com
            if is_directed:
                Stot_in[best_com] += in_degree
                Stot_out[best_com] += out_degree
            else:
                Stot[best_com] += degree
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
    """Calculate weights between node and its neighbor communities.

    Parameters
    ----------
    nbrs : dictionary
           Dictionary with nodes' neighbours as keys and their edge weight as value.
    node2com : dictionary
           Dictionary with all graph's nodes as keys and their community index as value.

    """
    weights = defaultdict(float)
    for nbr, wt in nbrs.items():
        weights[node2com[nbr]] += wt
    return weights


def _gen_graph(G, partition):
    """Generate a new graph based on the partitions of a given graph"""
    H = G.__class__()
    node2com = {}
    for i, part in enumerate(partition):
        nodes = set()
        for node in part:
            node2com[node] = i
            nodes.update(G.nodes[node].get("nodes", {node}))
        H.add_node(i, nodes=nodes)

    for node1, node2, wt in G.edges(data=True):
        wt = wt["weight"]
        com1 = node2com[node1]
        com2 = node2com[node2]
        temp = H.get_edge_data(com1, com2, {"weight": 0})["weight"]
        H.add_edge(com1, com2, **{"weight": wt + temp})
    return H


def _convert_multigraph(G, weight, is_directed):
    """Convert a Multigraph to normal Graph"""
    if is_directed:
        H = nx.DiGraph()
    else:
        H = nx.Graph()
    H.add_nodes_from(G)
    for u, v, wt in G.edges(data=weight, default=1):
        if H.has_edge(u, v):
            H[u][v]["weight"] += wt
        else:
            H.add_edge(u, v, weight=wt)
    return H
