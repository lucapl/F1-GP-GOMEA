from collections import Counter

import numpy as np
from scipy.stats import entropy, pearsonr, spearmanr

from src.utils.fpcontrol import restore_fenv


def correlation(population, char_order, cor_func, original_control_word=None):
    if original_control_word is not None:
        restore_fenv(
            original_control_word
        )  # framslib changes control word, which causes crashes
    evals = [sum(ind.fitness.values) for ind in population]
    letter_counters = [Counter(ind[0]) for ind in population]
    character_matrix = np.array(
        [[counter.get(c, 0) for c in char_order] for counter in letter_counters]
    )

    return [cor_func(letter, evals).statistic for letter in character_matrix.T]


def spearman_cor(population, char_order, original_control_word=None):
    return correlation(population, char_order, spearmanr, original_control_word)


def pearson_cor(population, char_order, original_control_word=None):
    return correlation(population, char_order, pearsonr, original_control_word)


def calc_entropy(population: list[str]):
    counts = Counter(population)
    probs = [v / len(population) for k, v in counts.items()]
    return entropy(probs)


def calc_gene_diversity(
    population: list[str], prefix_len: int, control_word, base: "float | None" = None
):
    if control_word is not None:
        restore_fenv(control_word)
    longest_len = len(max(population, key=lambda ind: len(ind))) - prefix_len
    padded = [list(ind[prefix_len:].ljust(longest_len, "_")) for ind in population]
    genos_arrays = np.array(padded)
    total = len(population)
    single_counts = {
        i: Counter(genos_arrays[:, i]) for i in range(genos_arrays.shape[1])
    }
    return np.mean(
        [
            calculate_entropy_list(list(count.values()), total, base)
            for i, count in single_counts.items()
        ]
    )


def calc_uniqueness(population: list[str]):
    return len(set(population)) / len(population)


def IoU(pop1: list[str], pop2: list[str]):
    A = set(pop1)
    B = set(pop2)
    intersection = A.intersection(B)
    union = A.union(B)
    return len(intersection) / len(union) if union else 0


def calculate_entropy_list(_list: list, total: "int | None" = None, base=None):
    if total is None:
        total = sum(_list)
    probs = [count / total for count in _list]
    value = entropy(probs, base=base)
    return value if not np.isnan(value) else 0
