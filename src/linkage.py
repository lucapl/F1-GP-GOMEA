import numpy as np
import random
import math
import itertools

from collections import Counter
from scipy.stats import spearmanr, pearsonr, entropy
import scipy.cluster.hierarchy as scipy_hier

from src.utils.fpcontrol import restore_fenv
from src.stats import spearman_cor, pearson_cor, calculate_entropy_list, calc_entropy


class LinkageModel():
    '''
    Abstract of how a linkage model should be implemented to work with this implementation of GOMEA
    '''
    def __init__(self, population):
        self.population = population
    def __len__(self):
        ...
    def __getitem__(self, donor):
        ...
    def get_stats(self):
        '''
        these stats will be added to the logbook
        '''
        return {}


class CommonLinkageModel(LinkageModel):
    def _shuffle(self):
        self.random_pert = np.random.permutation(len(self))
        self.i = 0

    def __len__(self):
        return len(self.linkage_model)

    def __getitem__(self, donor):
        '''
        here donor is only passed to work with other linkage models, the positional linkage is random and the same for the whole pop
        '''
        selected = self.linkage_model[self.random_pert[self.i]]

        self.i += 1
        if self.i % len(self) == 0:
            self._shuffle()

        return selected


def univariate_linkage(population: list):
    max_ = max(map(lambda ind: len(ind), population))
    return [(i,) for i in range(max_)]


class UnivariateLinkage(CommonLinkageModel):
    '''
    Naive, returns all single letter positions in genotype
    '''
    def __init__(self, population):
        super().__init__(population)
        self.linkage_model = univariate_linkage(population)
        super()._shuffle()


class LinkageTree(CommonLinkageModel):
    def __init__(self, population):
        super().__init__(population)
        self.linkage_model = self._create_linkage_tree()
        self._shuffle()

    def _calculate_dependencies(self):
        ...  # should calculate self.similarities

    def _get_condensed_distance_matrix(self):
        simil_matrix = np.zeros((self.n, self.n))
        for (i, j), value in self.similarities.items():
            simil_matrix[i, j] = value
            simil_matrix[j, i] = value
        simil_matrix = np.nan_to_num(simil_matrix)
        max_sim = np.max(simil_matrix)
        if max_sim == 0:
            # basically means all genotypes are the same, no point in creating a linkage model
            return None
        dissim_matrix = 1 - (simil_matrix / max_sim)
        return dissim_matrix[np.triu_indices_from(dissim_matrix, k=1)]

    def _convert_linkage_to_fos(self, Z):
        n_points = len(Z) + 1
        clusters = self._create_univariate(n_points)

        for cluster_i, cluster_j, distance, observations in Z:
            cluster_i = clusters[int(cluster_i)]
            cluster_j = clusters[int(cluster_j)]

            clusters.append(cluster_i + cluster_j)

        return clusters

    def _hierarchihal_clustering(self):
        condensed_distance_matrix = self._get_condensed_distance_matrix()
        if condensed_distance_matrix is None:
            self.Z = np.array([])
            return self._create_univariate(self.n)
        self.Z = scipy_hier.linkage(condensed_distance_matrix, method='average')
        return self._convert_linkage_to_fos(self.Z)[:-1]

    def _create_univariate(self, n):
        return [(i,) for i in range(n)]

    def _create_linkage_tree(self):
        self._calculate_dependencies()
        return self._hierarchihal_clustering()


class LinkageTreeFramsF1(LinkageTree):
    def __init__(self, population: list, original_control_word=None):
        self.original_control_word = original_control_word
        if self.original_control_word != None:
            restore_fenv(original_control_word)
        super().__init__(population)

    def _hierarchihal_clustering(self):
        longest_geno = max(self.population, key=lambda ind: len(ind))
        geno_len = len(longest_geno)
        self.n = geno_len
        return super()._hierarchihal_clustering()

    def _calculate_dependencies(self, pad_token="_"):
        #longest_len = len(max(self.population, key=lambda ind: len(ind)))
        mapped = [list(map(lambda s: s.name, ind)) for ind in self.population]
        padded = list(zip(*itertools.zip_longest(*mapped, fillvalue=pad_token)))#[list(ind[0].ljust(longest_len, '_')) for ind in self.population]
        genos_array = np.array(padded)
        total = len(self.population)
        single_counts = {i: Counter(genos_array[:, i][genos_array[:, i]!=pad_token]) for i in range(genos_array.shape[1])}
        pair_counts = {
            (i, j): Counter(zip(genos_array[:, i][genos_array[:, i]!=pad_token], genos_array[:, j][genos_array[:, j]!=pad_token]))
            for i in range(genos_array.shape[1]) for j in range(i + 1, genos_array.shape[1])
        }
        self.positional_entropies = {i: calculate_entropy_list(list(count.values()), total) for i, count in single_counts.items()}
        self.joint_entropies = {(i,j): calculate_entropy_list(list(count.values()), total) for (i,j), count in pair_counts.items()}

        self.similarities = {
            (i,j): self.positional_entropies[i] + self.positional_entropies[j] - self.joint_entropies[(i,j)]
            for (i,j), pair_count in pair_counts.items()
        }

    def get_stats(self):
        return {'linkage_tree': self.Z.tolist()}
