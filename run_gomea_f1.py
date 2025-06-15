import os, sys
import json
import math
import argparse
from functools import partial
from pathlib import Path

from framspy.src.FramsticksLibCompetition import FramsticksLibCompetition
from framspy.src.FramsticksEvolution import save_genotypes

import numpy as np
from deap import creator, base, tools, gp

from src.gomea import load_checkpoint, eaGOMEA, gom, forced_improvement, override_nodes
from src.linkage import LinkageTreeFramsF1
from src.gomea import eaGOMEA, forced_improvement, gom, migrate_subpops, override_nodes
from src.gpf1 import create_f1_pset, parse

from src.stats import calc_uniqueness, calc_gene_diversity, calc_entropy

from src.utils.stopping import earlyStoppingOrMaxIter, EarlyStopper
from src.utils.fpcontrol import *


def prepare_gomea_parser(parser):
    parser.add_argument('-n', '--ngen', default=15, type=int)
    parser.add_argument('-p', '--popsize', default=20, type=int)
    parser.add_argument('-e', '--early_stop', default=10, type=int)
    parser.add_argument('-g', '--initial_geno_mutations', default=100, type=int)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--sims',
                        nargs='+',
                        default=['eval-allcriteria.sim', 'recording-body-coords.sim'],
                        help='List of simulation files to use.')
    parser.add_argument('--no_forced_improv', action='store_true')
    parser.add_argument('--framspy', help="Specifies location of framspy/simfiles.", default="./framspy")
    parser.add_argument('--framslib', help="Specifies location of framstick engine.", default="./Framsticks52")
    parser.add_argument("--nmix", default=5, type=int)

    return parser

#original_control_word = save_fenv() # this is important to not cause floating point exceptions and others
#toolbox = LinkageToolbox('build_linkage_model', 'override_nodes')
#toolbox.define_default_linkages(CHARS, PREFIX, original_control_word)


def evaluate(ptree, pset, flib, invalid_fitness, criteria):
    try:
        geno = [str(gp.compile(ptree, pset))]
    except:
        return (invalid_fitness,)
    try:
        valid = flib.isValidCreature(geno)[0]
    except:
        print(geno)
        raise Exception
    if not valid:
        return (invalid_fitness,)
    # before running a creature through a simulation we ensure the genotype is valid
    value = flib.evaluate(geno)[0]#["evaluations"][''][criteria]
    return (value,) 


def generate_random(flib, n=100, geno_format="1"):
    geno = [flib.getSimplest(geno_format)]
    for i in range(n):
        geno = flib.mutate(geno)
    return geno


def create_ind(flib, pset, n=100):
    return parse(generate_random(flib, n)[0].replace(" ",""), pset)


if __name__ == '__main__':
    # prepare arguments
    parser = argparse.ArgumentParser(
                    prog='GOMEA experiment',
                    description='runs a gomea experiment on f1 framsticks population',
                    epilog='glhf')
    parser = prepare_gomea_parser(parser)
    args = parser.parse_args()

    ##########################
    # load Framsticks library
    ##########################
    # sim files
    frasmpy_path = Path(args.framspy)
    sim_formatted = ';'.join([
        (frasmpy_path/sim_file).absolute().as_posix()
        for sim_file in args.sims
    ])

    # engine
    # print_fenv_state("Before loading framsticks")
    framsLib = FramsticksLibCompetition(args.framslib, None, sim_formatted)
    # print_fenv_state("After loading framsticks")
    # restore_fenv(original_control_word)

    #####################
    # deap definitions
    #####################
    pset = create_f1_pset()
    # framsticks classes
    creator.create("FitnessMax", base.Fitness, weights=[1])
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    early_stopper = EarlyStopper(args.early_stop, toolbox)
    # basic operators
    toolbox.register("random_individual", create_ind, flib=framsLib, pset=pset, n=args.initial_geno_mutations)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.random_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual, args.popsize)

    # evaluation for testing
    max_len = args.initial_geno_mutations
    toolbox.register("evaluate", evaluate, pset=pset, flib=framsLib, invalid_fitness=-999999.0, criteria="distance")

    # early stopper or max itera
    toolbox.register("should_stop", partial(earlyStoppingOrMaxIter, max_gen=args.ngen, early_stopper=EarlyStopper(args.early_stop, toolbox)))

    # gomea operators
    isForcedImprov = not args.no_forced_improv
    print("Is forced improvent on?: ", isForcedImprov)
    toolbox.register("genepool_optimal_mixing", gom, toolbox=toolbox, forcedImprov=isForcedImprov)
    toolbox.register("forced_improvement", forced_improvement, toolbox=toolbox)
    toolbox.register("build_linkage_model", LinkageTreeFramsF1, original_control_word=None)
    toolbox.register("override_nodes", override_nodes, fillvalue="_", toolbox=toolbox)

    toolbox.register("get_evaluations", framsLib.get_evals)

    ####################
    # stats logging
    ####################
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_fit.register("avg", np.mean)
    stats_fit.register("std", np.std)
    stats_fit.register("min", np.min)
    stats_fit.register("max", np.max)

    # stats_geno = tools.Statistics(lambda ind: ind[0])
    # stats_geno.register("entropy", calc_entropy)
    # stats_geno.register("gene_diversity", calc_gene_diversity, prefix_len=PREFIX, control_word=original_control_word)

    stats_geno_len = tools.Statistics(lambda ind: len(ind))
    stats_geno_len.register("avg", np.mean)
    stats_geno_len.register("std", np.std)
    stats_geno_len.register("min", np.min)
    stats_geno_len.register("max", np.max)

    mstats = tools.MultiStatistics(fitness=stats_fit, genotype_length=stats_geno_len)

    #####################
    # population initialization
    #####################
    hof = None
    logbook = None
    start_gen = 0

    n_subpops = 4
    subpops = [toolbox.population() for _ in range(n_subpops)]
    logbooks = [None] * n_subpops
    start_gens = [0] * n_subpops

    ########################
    # MAIN ALGORITHM
    ########################
    MIGRATION_INTERVAL = 5
    MIGRATION_COUNT = 2
    for mix in range(args.nmix):
        print(f"--- Mix {mix}/{args.nmix} ---")
        for i in range(n_subpops):
            print(f"- Subpopulation {i}/{n_subpops}")
            subpops[i], logbooks[i] = eaGOMEA(
                subpops[i],
                toolbox,
                start_gen=start_gens[i],
                logbook=logbooks[i],
                stats=mstats,
                halloffame=None,
                checkpoint_freq=None,
                verbose=args.verbose,
            )
            start_gens[i] += 1

        if mix % MIGRATION_INTERVAL == 0:
            migrate_subpops(toolbox, subpops, k=MIGRATION_COUNT, replace=True)

    final_population = [ind for pop in subpops for ind in pop]
    best = tools.selBest(final_population, 1)[0]
    print("Best final individual:", best)
    print("Fitness:", best.fitness.values[0])

    #######################
    # saving outputs
    #######################
    framsLib.end()
