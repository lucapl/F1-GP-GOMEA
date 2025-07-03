import argparse
import os
import random
from copy import deepcopy
from functools import partial

import numpy as np
from deap import base, creator, gp, tools
from FramsticksLib import FramsticksLib  # type checking
from FramsticksLibCompetition import FramsticksLibCompetition

from src.gomea import forced_improvement, gom, override_nodes
from src.gpf1 import create_f1_pset, parse
from src.linkage import LinkageTreeFramsF1
from src.utils.elitism import SaveBest
from src.utils.fpcontrol import print_fenv_state, restore_fenv, fpenv_context_restore
from src.utils.stopping import EarlyStopper, earlyStoppingOrMaxIter

# creator.create("FitnessMax", base.Fitness, weights=[1])
# creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

solution_cache: dict[str, float] = {}
"avoids wasting precious eval count"


class BasicToolbox:
    def __init__(self):
        # from DEAP.base.Toolbox.__init__:
        self.clone = deepcopy
        self.map = map

    def clone(self, individual):
        return tools.clone(individual)


class OurToolbox(BasicToolbox):
    "Replacement for DEAP `base.Toolbox`"

    framsLib: FramsticksLib
    flib: FramsticksLib

    def __init__(
        self,
        framsLib: FramsticksLib,
        args: argparse.Namespace,
        pset: gp.PrimitiveSetTyped,
        if_forced_improv_save_best: bool,
        invalid_fitness=-999_999.0,
    ):
        super().__init__()
        self.args = args
        self.framsLib = framsLib
        self.flib = framsLib  # alias
        self.pset = pset

        # for evaluate:
        self.eval_criteria = args.criteria
        self.mock_test = args.MOCK_EVALS
        self.invalid_fitness = invalid_fitness
        self.save_best = SaveBest() if if_forced_improv_save_best else None

        # self.should_stop = partial(
        self.early_stopper = EarlyStopper(self.args.early_stop, toolbox=self)

        # evaluation statistics
        self.solution_cache_hits = 0
        "how many evalutions were cached?"
        self.solution_cache_misses = 0

        self.n_mock_evals = 0

    def should_stop(self, population, gen):
        return earlyStoppingOrMaxIter(
            population=population,
            gen=gen,
            max_gen=self.args.ngen,
            early_stopper=self.early_stopper,
        )

    def generate_random(
        self,
        parts: tuple[int, int],
        neurons: tuple[int, int],
        iters: int,
        geno_format="1",
    ):  
        with fpenv_context_restore(verbose=False):
            geno = self.flib.getRandomGenotype(
                self.flib.getSimplest(geno_format),
                *parts,
                *neurons,
                iters,
                return_even_if_failed=True,
            )
        return geno

    def create_subtree(self, low=0, high=100, type_=None):
        # the same thing as random_ind???
        n = np.random.randint(low, high)
        return parse(self.generate_random(self.flib, n).replace(" ", ""), self.pset)

    def random_individual(self, geno_format="1"):
        return parse(
            self.generate_random(
                iters=self.args.initial_geno_mutations,
                parts=self.args.parts,
                neurons=self.args.neurons,
                geno_format=geno_format,
            ).replace(" ", ""),
            self.pset,
        )
        # iters=args.initial_geno_mutations, parts=args.parts, neurons=args.neurons)
        # return self._create_ind()

    def _create_ind(
        self,
        parts: tuple[int, int],
        neurons: tuple[int, int],
        iters: int,
        geno_format="1",
    ):
        return parse(
            self.generate_random(self.flib, parts, neurons, iters, geno_format).replace(
                " ", ""
            ),
            self.pset,
        )

    # toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.random_individual)
    # toolbox.register("population", tools.initRepeat, list, toolbox.individual, args.popsize)
    def individual(self):
        return tools.initIterate(creator.Individual, self.random_individual)

    def population(self, n: int | None = None) -> "list[creator.Individual]":
        size = n if n is not None else self.args.popsize
        return tools.initRepeat(list, self.individual, size)

    # toolbox.register("mutate", mutate, pset=pset, pmut=args.pmut, toolbox=toolbox, framsLib=framsLib)
    def mutate(self, individual):
        if np.random.random() >= self.args.pmut:
            return individual
        mutated = [str(gp.compile(individual, self.pset))]
        with fpenv_context_restore(verbose=False):
            mutated = self.framsLib.mutate(mutated)
        mutated = parse(mutated[0].replace(" ", ""), self.pset)

        ind = creator.Individual(mutated)
        # ind.fitness = toolbox.clone(individual.fitness)
        return ind

    def get_evaluations(self) -> int:
        if self.mock_test:
            return self.n_mock_evals
        if self.args.count_nevals:
            # if isinstance(self.framsLib, FramsticksLibCompetition):
            return self.framsLib.get_evals()
        return 0

    # toolbox.register("evaluate", evaluate, pset=pset, flib=framsLib, invalid_fitness=-999999.0,
    # criteria=args.criteria, mock_test=args.MOCK_EVALS)
    def evaluate(self, ptree):
        try:
            geno = str(gp.compile(ptree, self.pset))
        except Exception:
            return (self.invalid_fitness,)

        # if geno in solution_cache:
        #     return (solution_cache[geno],)
        # Lookup dictionary only once:
        cached_f: float | None = solution_cache.get(geno)
        if cached_f is not None:
            self.solution_cache_hits += 1
            return (cached_f,)
        self.solution_cache_misses += 1

        geno = [geno]
        try:
            with fpenv_context_restore(verbose=False):
                valid = self.flib.isValidCreature(geno)[0]
        except Exception:
            print(geno)
            raise Exception
        if not valid:
            return (self.invalid_fitness,)
        # before running a creature through a simulation we ensure the genotype is valid
        if self.mock_test:
            value = random.expovariate()
            self.n_mock_evals += 1
        else:
            # value = self.flib.evaluate(geno)[0]["evaluations"][""][criteria]
            with fpenv_context_restore(verbose=False):
                value = self.flib.evaluate(geno)[0]["evaluations"][""][self.eval_criteria]
        solution_cache[geno[0]] = value
        return (value,)

    # toolbox.register("genepool_optimal_mixing", gom, toolbox=toolbox, forcedImprov=isForcedImprov)
    # toolbox.register("forced_improvement", forced_improvement, toolbox=toolbox)
    # toolbox.register("build_linkage_model", LinkageTreeFramsF1, original_control_word=None)
    # toolbox.register("override_nodes", override_nodes, fillvalue="_", toolbox=toolbox)

    def genepool_optimal_mixing(self, *args, **kwargs):
        # forcedImprov=self.is_forced_improvement
        return gom(
            *args, toolbox=self, forcedImprov=not self.args.no_forced_improv, **kwargs
        )

    def forced_improvement(self, *args, **kwargs):
        return forced_improvement(*args, toolbox=self, **kwargs)

    def build_linkage_model(self, *args, **kwargs):
        return LinkageTreeFramsF1(*args, original_control_word=None, **kwargs)

    def override_nodes(self, *args, **kwargs):
        return override_nodes(*args, fillvalue="_", toolbox=self, **kwargs)
    
    # toolbox.register("get_best", save_best.get_best if args.forced_improv_global else default_get_best, toolbox=toolbox)
    def get_best(self, population):
        if self.save_best is not None:
            return self.save_best.get_best()
        return tools.selBest(population, 1)[0]


###########
#old toolbox definitions
###########
    # toolbox = base.Toolbox()
    # # early_stopper = EarlyStopper(args.early_stop, toolbox)
    # # basic operators
    # toolbox.register("random_individual", create_ind, flib=framsLib, pset=pset, iters=args.initial_geno_mutations, parts=args.parts, neurons=args.neurons)
    # toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.random_individual)
    # toolbox.register("population", tools.initRepeat, list, toolbox.individual, args.popsize)
    # # evaluation for testing
    # max_len = args.initial_geno_mutations
    # # gomea operators
    # isForcedImprov = not args.no_forced_improv
    # print("Is forced improvent on?: ", isForcedImprov)
    # toolbox.register("genepool_optimal_mixing", gom, toolbox=toolbox, forcedImprov=isForcedImprov)
    # toolbox.register("forced_improvement", forced_improvement, toolbox=toolbox)
    # toolbox.register("build_linkage_model", LinkageTreeFramsF1, original_control_word=None)
    # toolbox.register("override_nodes", override_nodes, fillvalue="_", toolbox=toolbox)
    # toolbox.register("mutate", mutate, pset=pset, pmut=args.pmut, toolbox=toolbox, framsLib=framsLib)
    # toolbox.register("get_evaluations", framsLib.get_evals if args.count_nevals else lambda: 0)
    # save_best = SaveBest(toolbox) if args.forced_improv_global else None
    # toolbox.register("get_best", save_best.get_best if args.forced_improv_global else default_get_best, toolbox=toolbox)
    # toolbox.register("evaluate", 
    #     evaluate, 
    #     pset=pset, 
    #     flib=framsLib, 
    #     invalid_fitness=-999999.0, 
    #     criteria=args.criteria, 
    #     mock_test=args.MOCK_EVALS, 
    #     save_best=save_best)