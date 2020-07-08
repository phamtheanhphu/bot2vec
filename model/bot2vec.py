import numpy as np
import networkx as nx
import random
import community as community_louvain
import sys
from gensim.models import Word2Vec
import time


class Bot2Vec():

    def __init__(self, osn_input_file, emb_output_file, p=1, q=1, r=1, w=20, l=30, neg=5, d=128, window_size=5,
                 num_workers=1, max_iter=5):

        self.osn_input_file = osn_input_file
        self.emb_output_file = emb_output_file

        self.osn = None
        self.nodes = None
        self.edges = None
        self.total_nodes = 0
        self.total_edges = 0

        # dictionary structure of node_id - community_id
        self.node_community_dict = {}

        # dictionary structure of community_id - node_id
        self.community_node_dict = {}

        # number of walk per node (w)
        self.w = w

        # walk length (l)
        self.l = l

        # negative sampling batch size (neg)
        self.neg = neg

        # embedding dimensions (d)
        self.d = d

        # window size for Skip-gram model
        self.window_size = window_size

        # number of used threads/workers & iterations for training representation model
        self.num_workers = num_workers
        self.max_iter = max_iter

        # model's parameters for the intra-community oriented random walk
        self.p = p  # return parameter
        self.q = q  # in-out parameter
        self.r = r  # out-community parameter

        # data for alias method sampling (inspired from Node2Vec)
        self.alias_nodes = {}
        self.alias_edges = {}

        # generated contextual nodes for each node as walks
        self.walks = []

    def train(self):

        # read the OSN as the NetworkX graph structure format
        self.__read_osn(self.osn_input_file)

        # extracting community structure of given osn
        self.__extract_community_structure()

        # training steps
        self.__init()
        self.__generate_walks()
        self.__learn_representations()

    def __read_osn(self, osn_input_file):
        print('Reading the OSN data...')
        self.osn = nx.read_edgelist(osn_input_file, nodetype=int, create_using=nx.DiGraph())
        for edge in self.osn.edges():
            self.osn[edge[0]][edge[1]]['weight'] = 1

        self.nodes = self.osn.nodes
        self.total_nodes = len(self.nodes)
        self.edges = self.osn.edges
        self.total_edges = len(self.edges)
        print(' -> Done, reading total [{:d}] nodes and [{:d}] edges'.format(self.total_nodes, self.total_edges))

    def __extract_community_structure(self):
        print('Discovering existing communities in given OSN...')
        self.node_community_dict = community_louvain.best_partition(self.osn.to_undirected())
        self.community_node_dict = {}
        for node_id in self.node_community_dict.keys():
            community_idx = self.node_community_dict[node_id]
            if community_idx not in self.community_node_dict.keys():
                self.community_node_dict.update({community_idx: [node_id]})
            else:
                self.community_node_dict[community_idx].append(node_id)
        print(' -> Done, extracting total [{:d}] communities'.format(len(self.community_node_dict.keys())))

    def __get_community_idx(self, node_id):
        return self.node_community_dict[node_id]

    def __get_total_community_nodes(self, community_idx):
        return len(self.community_node_dict[community_idx])

    def __bot2vec_random_walk(self, start_node):

        # begin the walk with the given start node
        walk = [start_node]

        while len(walk) < self.l:
            current_node = walk[-1]
            current_node_neighbors = sorted(self.osn.neighbors(current_node))
            if len(current_node_neighbors) > 0:
                # sampling next user node from set of current user node's neighbors
                if len(walk) == 1:  # walk is started at start_node, no previous node
                    next_node = current_node_neighbors[
                        self.__alias_draw(self.alias_nodes[current_node][0], self.alias_nodes[current_node][1])]
                    walk.append(next_node)
                else:
                    # sampling next node from v(i-1) user node -> v(i+1) user node
                    previous_node = walk[-2]
                    next_node = current_node_neighbors[
                        self.__alias_draw(self.alias_edges[(previous_node, current_node)][0],
                                          self.alias_edges[(previous_node, current_node)][1])]
                    walk.append(next_node)
            else:
                break

        return walk

    def __generate_walks(self):
        print('Generating walks for each user node...')
        nodes = list(self.nodes)
        for walk_iter in range(self.w):
            random.shuffle(nodes)
            for node in nodes:
                self.walks.append(self.__bot2vec_random_walk(start_node=node))
        self.walks = [[str(step) for step in walk] for walk in self.walks]
        print(' -> Done, generating total [{:d}] walks'.format(len(self.walks)))

    def __learn_representations(self):
        print('Learning the representations of given OSN...')
        if (len(self.walks) > 0):
            # applying word2vec model for learning the representations of nodes in given OSN
            start_time = time.time()
            word2vec_model = Word2Vec(self.walks,
                                      size=self.d,
                                      window=self.window_size,
                                      min_count=0,
                                      sg=1,
                                      workers=self.num_workers,
                                      iter=self.max_iter)

            word2vec_model.wv.save_word2vec_format(self.emb_output_file)
            print(' -> Done, finish to learn the representation of given OSN in [{:.4f}] seconds'.format(
                time.time() - start_time))
            return
        else:
            print('ERROR: there is no generated walk from given OSN, this process is terminated !')

    def __init(self):
        for node in self.nodes:
            # computing the direct neighborhood transitional probabilities
            unnormalized_probs = [self.osn[node][nbr]['weight'] for nbr in sorted(self.osn.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
            self.alias_nodes[node] = self.__alias_setup(probs=normalized_probs)

        for edge in self.edges:
            # computing the transitional probabilities for both two directions
            self.alias_edges[edge] = self.__compute_trans_probs(edge[0], edge[1])
            self.alias_edges[(edge[1], edge[0])] = self.__compute_trans_probs(edge[1], edge[0])

    def __compute_trans_probs(self, src, dst):

        unnormalized_probs = []

        src_community_idx = self.__get_community_idx(src)

        for dst_nbr in sorted(self.osn.neighbors(dst)):
            dst_community_idx = self.__get_community_idx(src)
            if dst_nbr == src:  # case 1: return to back to source node v(i) -> v(i-1) (spd=0)
                alpha_prob = self.osn[dst][dst_nbr]['weight'] / self.p
                if src_community_idx != dst_community_idx:
                    # v(i-1) and v(i+1) user nodes are in different communities
                    beta_prob = alpha_prob / (self.__get_total_community_nodes(src_community_idx) * self.r)
                    final_trans_prob = beta_prob
                else:
                    final_trans_prob = alpha_prob
            elif self.osn.has_edge(dst_nbr, src):  # case 2: v(i-1) and v(i+1) has a relation (spd=1)
                alpha_prob = trans_prob = self.osn[dst][dst_nbr]['weight']
                if src_community_idx != dst_community_idx:
                    # v(i-1) and v(i+1) user nodes are in different communities
                    beta_prob = alpha_prob / (self.__get_total_community_nodes(src_community_idx) * self.r)
                    final_trans_prob = beta_prob
                else:
                    final_trans_prob = alpha_prob

            else:  # otherwise, case 3: v(i-1) and v(i+1) has no relation (spd=2)
                alpha_prob = trans_prob = self.osn[dst][dst_nbr]['weight'] / self.q
                if src_community_idx != dst_community_idx:
                    # v(i-1) and v(i+1) user nodes are in different communities
                    beta_prob = alpha_prob / (self.__get_total_community_nodes(src_community_idx) * self.r)
                    final_trans_prob = beta_prob
                else:
                    final_trans_prob = alpha_prob

            unnormalized_probs.append(final_trans_prob)

        # computing global normalized constant
        lamda = sum(unnormalized_probs)

        # normalizing all transitional probabilities
        normalized_probs = [float(u_prob) / lamda for u_prob in unnormalized_probs]

        return self.__alias_setup(probs=normalized_probs)

    def __alias_setup(self, probs):
        # https://lips.cs.princeton.edu/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
        K = len(probs)
        q = np.zeros(K)
        J = np.zeros(K, dtype=np.int)

        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            q[kk] = K * prob
            if q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            J[small] = large
            q[large] = q[large] + q[small] - 1.0
            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        return J, q

    def __alias_draw(self, J, q):
        '''
        Draw sample from a non-uniform discrete distribution using alias sampling.
        '''
        K = len(J)

        kk = int(np.floor(np.random.rand() * K))
        if np.random.rand() < q[kk]:
            return kk
        else:
            return J[kk]


def main():
    osn_input_file = '../data/inputs/karate-club/karate.edgelist.txt'
    emb_output_file = '../data/outputs/karate-club/karate-club.emb'
    bot2vec = Bot2Vec(osn_input_file, emb_output_file)
    bot2vec.train()


if __name__ == "__main__":
    sys.exit(main())
